from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from ablation_model import IndependentOrganAgeOutput


REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_MODEL_DIR = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "01_model"
if str(CANONICAL_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CANONICAL_MODEL_DIR))

from ehr_age_losses import (  # noqa: E402
    AgePreservingLossConfig,
    cause_specific_cox_ph_loss,
    piecewise_exponential_nll,
)


@dataclass(frozen=True)
class IndependentHeadsLossConfig:
    huber_beta_years: float = 2.0
    independent_age_huber_weight: float = 1.0
    strict_direct_gap_weight: float = 0.75
    survival_weight: float = 0.5
    evaluator_gap_cox_weight: float = 0.25
    evaluator_gap_cox_scale_years: float = 2.0
    consistency_weight: float = 0.1

    def validate(self) -> None:
        if self.huber_beta_years <= 0 or self.evaluator_gap_cox_scale_years <= 0:
            raise ValueError("loss scales must be positive")
        weights = (
            self.independent_age_huber_weight,
            self.strict_direct_gap_weight,
            self.survival_weight,
            self.evaluator_gap_cox_weight,
            self.consistency_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("loss weights must be non-negative")


def effective_standard_loss_config(
    variant_id: str,
    base: AgePreservingLossConfig | None = None,
) -> AgePreservingLossConfig:
    config = base or AgePreservingLossConfig(
        base_huber_weight=1.0,
        base_mse_weight=0.0,
        base_correlation_weight=0.0,
        final_all_huber_weight=0.0,
        strict_final_huber_weight=0.0,
        strict_offset_weight=0.75,
        strict_center_weight=0.0,
        strict_slope_weight=0.0,
        survival_weight=0.5,
        evaluator_gap_cox_weight=0.25,
        consistency_weight=0.1,
    )
    if variant_id == "fig04_04_no_prognostic_supervision":
        return replace(config, survival_weight=0.0, evaluator_gap_cox_weight=0.0)
    if variant_id == "fig04_05_no_strict_offset_shrinkage":
        return replace(config, strict_offset_weight=0.0)
    if variant_id == "fig04_06_no_event_thinning_consistency":
        return replace(config, consistency_weight=0.0)
    return config


def effective_event_dropout_probability(variant_id: str) -> float:
    return 0.0 if variant_id == "fig04_06_no_event_thinning_consistency" else 0.1


def independent_heads_joint_loss(
    output: IndependentOrganAgeOutput,
    *,
    chronological_age: torch.Tensor,
    strict_healthy_reference: torch.Tensor,
    duration_months: torch.Tensor,
    event: torch.Tensor,
    at_risk: torch.Tensor,
    interval_edges_months: torch.Tensor,
    config: IndependentHeadsLossConfig | None = None,
    survival_multiplier: float = 1.0,
    augmented_output: IndependentOrganAgeOutput | None = None,
) -> dict[str, torch.Tensor]:
    """Closed loss for direct organ-age heads; no scalar base-age decomposition."""

    cfg = config or IndependentHeadsLossConfig()
    cfg.validate()
    if not 0.0 <= survival_multiplier <= 1.0:
        raise ValueError("survival_multiplier must lie in [0,1]")
    age = chronological_age.float()
    if output.predicted_age.ndim != 2 or output.predicted_age.shape[0] != len(age):
        raise ValueError("predicted_age must have shape [batch, organ]")
    if output.age_residual.shape != output.predicted_age.shape:
        raise ValueError("age_residual shape changed")
    expected_age = output.age_anchor + output.age_residual
    if not torch.equal(output.predicted_age, expected_age):
        raise ValueError("independent-head prediction identity failed")
    if output.base_age is not None:
        raise ValueError("independent heads cannot expose a shared scalar base_age")
    target = age.unsqueeze(1).expand_as(output.predicted_age)
    independent_age_huber = F.smooth_l1_loss(
        output.predicted_age, target, beta=cfg.huber_beta_years
    )
    strict = strict_healthy_reference.bool()
    if strict.shape != age.shape:
        raise ValueError("strict-reference mask and chronological age must align")
    if strict.any():
        strict_direct_gap = F.smooth_l1_loss(
            output.predicted_age[strict],
            target[strict],
            beta=1.0,
        )
    else:
        strict_direct_gap = output.predicted_age.sum() * 0.0
    survival = piecewise_exponential_nll(
        output.interval_log_rate,
        duration_months.float(),
        event,
        at_risk,
        interval_edges_months=interval_edges_months,
    )
    direct_raw_gap = output.predicted_age - target
    evaluator_gap_cox = cause_specific_cox_ph_loss(
        direct_raw_gap / cfg.evaluator_gap_cox_scale_years,
        duration_months.float(),
        event,
        at_risk,
    )
    if augmented_output is None:
        consistency = output.predicted_age.sum() * 0.0
    else:
        consistency = F.smooth_l1_loss(
            augmented_output.predicted_age,
            output.predicted_age.detach(),
            beta=cfg.huber_beta_years,
        )
        consistency = consistency + F.smooth_l1_loss(
            augmented_output.risk_age_offset,
            output.risk_age_offset.detach(),
            beta=1.0,
        )
    weighted_survival = cfg.survival_weight * float(survival_multiplier) * survival
    weighted_evaluator_gap_cox = (
        cfg.evaluator_gap_cox_weight
        * float(survival_multiplier)
        * evaluator_gap_cox
    )
    total = (
        cfg.independent_age_huber_weight * independent_age_huber
        + cfg.strict_direct_gap_weight * strict_direct_gap
        + weighted_survival
        + weighted_evaluator_gap_cox
        + cfg.consistency_weight * consistency
    )
    return {
        "total": total,
        "independent_age_huber": independent_age_huber,
        "strict_direct_gap": strict_direct_gap,
        "survival": survival,
        "weighted_survival": weighted_survival,
        "evaluator_gap_cox": evaluator_gap_cox,
        "weighted_evaluator_gap_cox": weighted_evaluator_gap_cox,
        "consistency": consistency,
    }

