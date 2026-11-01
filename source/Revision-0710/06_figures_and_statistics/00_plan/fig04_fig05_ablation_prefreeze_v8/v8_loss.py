from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

import v8_contract
import v8_inputs


REPO_ROOT = Path(__file__).resolve().parents[4]
V1_ROOT = (
    REPO_ROOT
    / "Revision-0710"
    / "06_figures_and_statistics"
    / "00_plan"
    / "fig04_fig05_ablation_prefreeze_v1"
)
CANONICAL_MODEL_DIR = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "01_model"
for path in (V1_ROOT, CANONICAL_MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Verify all three immutable V1 runtime files before either V1 helper is loaded.
V1_RUNTIME_PREFLIGHT = v8_contract.preflight_v1_runtime_bindings(Path(__file__).resolve().parent)

from ehr_age_losses import AgePreservingLossConfig, ehr_age_joint_loss  # noqa: E402


def _load_v1_loss_module():
    name = "revision0710_fig0405_v1_loss_for_v8"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V1_ROOT / "ablation_loss.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable V1 loss helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V1_LOSS = _load_v1_loss_module()


def _known(variant_id: str) -> str:
    value = str(variant_id)
    if value not in v8_inputs.KNOWN_VARIANTS:
        raise ValueError(f"unknown variant: {value!r}")
    return value


def variant_loss_policy(variant_id: str) -> dict[str, Any]:
    value = _known(variant_id)
    policy: dict[str, Any] = {
        "loss_route": "canonical_ehr_age_joint_loss",
        "base_huber_weight": 1.0,
        "base_mse_weight": 0.0,
        "base_correlation_weight": 0.0,
        "final_all_huber_weight": 0.0,
        "strict_final_huber_weight": 0.0,
        "strict_offset_weight": 0.75,
        "strict_center_weight": 0.0,
        "strict_slope_weight": 0.0,
        "survival_weight": 0.5,
        "evaluator_gap_cox_weight": 0.25,
        "evaluator_gap_cox_scale_years": 2.0,
        "consistency_weight": 0.1,
        "event_dropout_probability": 0.1,
    }
    if value == "fig04_04_no_survival_supervision":
        policy["survival_weight"] = 0.0
    elif value == "fig04_05_no_strict_offset_shrinkage":
        policy["strict_offset_weight"] = 0.0
    elif value == "fig04_06_no_consistency_objective":
        policy["consistency_weight"] = 0.0
    elif value == "fig04_07_seven_independent_organ_age_heads":
        policy["loss_route"] = "independent_heads_joint_loss"
    return policy


def _standard_config(policy: dict[str, Any]) -> AgePreservingLossConfig:
    keys = set(AgePreservingLossConfig.__dataclass_fields__)
    return AgePreservingLossConfig(
        **{key: policy[key] for key in keys if key in policy}
    )


def compute_variant_loss(
    variant_id: str,
    output,
    *,
    labels: dict[str, torch.Tensor],
    interval_edges_months: torch.Tensor,
    survival_multiplier: float,
    augmented_output=None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    value = _known(variant_id)
    expected_label_keys = {
        "chronological_age",
        "strict_healthy_reference",
        "duration_months",
        "event",
        "at_risk",
    }
    if set(labels) != expected_label_keys:
        raise ValueError("loss labels do not match the closed schema")
    policy = variant_loss_policy(value)
    if policy["loss_route"] == "independent_heads_joint_loss":
        config = _V1_LOSS.IndependentHeadsLossConfig(
            huber_beta_years=2.0,
            independent_age_huber_weight=1.0,
            strict_direct_gap_weight=float(policy["strict_offset_weight"]),
            survival_weight=float(policy["survival_weight"]),
            evaluator_gap_cox_weight=float(policy["evaluator_gap_cox_weight"]),
            evaluator_gap_cox_scale_years=float(policy["evaluator_gap_cox_scale_years"]),
            consistency_weight=float(policy["consistency_weight"]),
        )
        losses = _V1_LOSS.independent_heads_joint_loss(
            output,
            **labels,
            interval_edges_months=interval_edges_months,
            config=config,
            survival_multiplier=survival_multiplier,
            augmented_output=augmented_output,
        )
    else:
        losses = ehr_age_joint_loss(
            output,
            **labels,
            interval_edges_months=interval_edges_months,
            config=_standard_config(policy),
            survival_multiplier=survival_multiplier,
            augmented_output=augmented_output,
        )
    wiring = {
        "variant_id": value,
        "loss_route": policy["loss_route"],
        "loss_policy": policy,
        "augmented_forward_used": augmented_output is not None,
    }
    return losses, wiring
