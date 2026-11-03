from __future__ import annotations

from .types import EHRAGEOutput

from dataclasses import dataclass

import torch

import torch.nn.functional as F

@dataclass(frozen=True)
class AgePreservingLossConfig:
    huber_beta_years: float = 2.0
    age_scale_years: float = 10.0
    base_huber_weight: float = 1.0
    base_view_huber_weight: float = 0.0
    base_mse_weight: float = 0.25
    base_correlation_weight: float = 0.05
    final_all_huber_weight: float = 0.1
    strict_final_huber_weight: float = 2.0
    strict_offset_weight: float = 0.75
    strict_center_weight: float = 0.2
    strict_slope_weight: float = 0.5
    survival_weight: float = 0.5
    evaluator_gap_cox_weight: float = 0.25
    evaluator_gap_cox_scale_years: float = 2.0
    consistency_weight: float = 0.1

    def validate(self) -> None:
        if (
            self.huber_beta_years <= 0.0
            or self.age_scale_years <= 0.0
            or self.evaluator_gap_cox_scale_years <= 0.0
        ):
            raise ValueError("Huber beta and all loss scales must be positive")
        weights = (
            self.base_huber_weight,
            self.base_view_huber_weight,
            self.base_mse_weight,
            self.base_correlation_weight,
            self.final_all_huber_weight,
            self.strict_final_huber_weight,
            self.strict_offset_weight,
            self.strict_center_weight,
            self.strict_slope_weight,
            self.survival_weight,
            self.evaluator_gap_cox_weight,
            self.consistency_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("loss weights must be non-negative")

def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0

def _correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if len(prediction) < 2:
        return _zero(prediction)
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = prediction.square().mean().sqrt() * target.square().mean().sqrt()
    # sqrt has an infinite derivative at zero. A tensor `where` is insufficient
    # because autograd still evaluates the undefined constant-prediction branch.
    if not torch.isfinite(denominator.detach()) or float(denominator.detach()) <= 1e-8:
        return _zero(prediction)
    return 1.0 - (prediction * target).mean() / denominator

def _strict_calibration_terms(
    predicted_age: torch.Tensor,
    target_age: torch.Tensor,
    strict_mask: torch.Tensor,
    *,
    age_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = strict_mask.bool()
    if int(selected.sum()) < 2:
        zero = _zero(predicted_age)
        return zero, zero
    error = predicted_age[selected] - target_age[selected].unsqueeze(1)
    age = target_age[selected]
    center = (error.mean(dim=0) / age_scale).square().mean()
    centered_age = age - age.mean()
    centered_error = error - error.mean(dim=0, keepdim=True)
    variance = centered_age.square().mean().clamp_min(1e-8)
    slope = (centered_error * centered_age.unsqueeze(1)).mean(dim=0) / variance
    return center, slope.square().mean()

def piecewise_exponential_nll(
    interval_log_rate: torch.Tensor,
    duration_months: torch.Tensor,
    event: torch.Tensor,
    at_risk: torch.Tensor,
    *,
    interval_edges_months: torch.Tensor,
) -> torch.Tensor:
    """Cause-specific piecewise-exponential NLL, macro-averaged by organ."""

    if interval_log_rate.ndim != 3:
        raise ValueError("interval_log_rate must have shape [batch,organ,interval]")
    batch, organs, intervals = interval_log_rate.shape
    expected = (batch, organs)
    if duration_months.shape != expected or event.shape != expected or at_risk.shape != expected:
        raise ValueError("survival labels must align with [batch,organ]")
    edges = interval_edges_months.to(interval_log_rate.device, interval_log_rate.dtype)
    if edges.ndim != 1 or len(edges) != intervals + 1:
        raise ValueError("interval edges must contain K+1 values")
    if not torch.isfinite(interval_log_rate).all():
        raise ValueError("interval_log_rate must be finite")
    if not torch.isfinite(duration_months).all():
        raise ValueError("duration_months must be finite")
    if not torch.isfinite(edges).all():
        raise ValueError("interval edges must be finite")
    if not torch.isclose(edges[0], edges.new_zeros(()), atol=1e-7, rtol=0.0):
        raise ValueError("interval edges must start at zero")
    if not torch.all(edges[1:] > edges[:-1]):
        raise ValueError("interval edges must be strictly increasing")
    for name, values in (("event", event), ("at_risk", at_risk)):
        if not torch.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} must contain only 0/1")
    risk_mask = at_risk.bool()
    if (event.bool() & ~risk_mask).any():
        raise ValueError("event=1 is invalid outside the at-risk set")
    duration = torch.where(risk_mask, duration_months, torch.zeros_like(duration_months))
    if (duration[risk_mask] <= 0).any() or (duration[risk_mask] > edges[-1] + 1e-4).any():
        raise ValueError("at-risk durations must lie in (0,last interval edge]")

    starts = edges[:-1].view(1, 1, -1)
    ends = edges[1:].view(1, 1, -1)
    exposure = (torch.minimum(duration.unsqueeze(-1), ends) - starts).clamp_min(0.0)
    exposure = torch.minimum(exposure, ends - starts)
    event_bin = torch.bucketize(duration.contiguous(), edges[1:-1], right=False)
    event_indicator = F.one_hot(event_bin.long(), num_classes=intervals).to(interval_log_rate.dtype)
    event_indicator = event_indicator * event.to(interval_log_rate.dtype).unsqueeze(-1)
    bounded_log_rate = interval_log_rate.clamp(min=-20.0, max=5.0)
    rate = torch.exp(bounded_log_rate)
    row_nll = (rate * exposure - event_indicator * bounded_log_rate).sum(dim=-1)

    organ_losses = []
    for organ_index in range(organs):
        selected = risk_mask[:, organ_index]
        if selected.any():
            organ_losses.append(row_nll[selected, organ_index].mean())
    if not organ_losses:
        return _zero(interval_log_rate)
    return torch.stack(organ_losses).mean()

def evaluator_aligned_raw_age_gap(
    output: EHRAGEOutput,
    chronological_age: torch.Tensor,
    *,
    stop_base_age_gradient: bool = True,
) -> torch.Tensor:
    """Return the raw gap exported to the evaluator, optionally isolating the base head.

    The formal evaluator first computes ``predicted_age - actual_age`` and then
    subtracts a validation-reference linear age trend. This function preserves
    exactly that pre-correction value. Detaching the base term prevents the risk
    objective from directly moving the chronological-age head; gradients still
    train the organ offset and its shared representation.
    """

    age = chronological_age.float()
    if output.base_age.shape != age.shape:
        raise ValueError("base age and chronological age must align")
    if output.risk_age_offset.ndim != 2 or output.risk_age_offset.shape[0] != len(age):
        raise ValueError("organ offset must have shape [batch,organ]")
    if output.predicted_age.shape != output.risk_age_offset.shape:
        raise ValueError("predicted age and organ offset must align")
    expected_prediction = output.base_age.unsqueeze(1) + output.risk_age_offset
    if not torch.allclose(output.predicted_age, expected_prediction, atol=1e-5, rtol=1e-6):
        raise ValueError("predicted_age must equal base_age + risk_age_offset")
    base = output.base_age.detach() if stop_base_age_gradient else output.base_age
    return base.unsqueeze(1) - age.unsqueeze(1) + output.risk_age_offset

def cause_specific_cox_ph_loss(
    score: torch.Tensor,
    duration_months: torch.Tensor,
    event: torch.Tensor,
    at_risk: torch.Tensor,
) -> torch.Tensor:
    """Breslow cause-specific Cox loss, macro-averaged over organs.

    Higher ``score`` means higher hazard. Death-censored rows remain in risk
    sets until their supplied duration; rows with ``at_risk=0`` are excluded.
    With minibatch training, risk sets are minibatch approximations and must be
    paired with the full piecewise-exponential likelihood for hazard calibration.
    """

    if score.ndim != 2:
        raise ValueError("score must have shape [batch,organ]")
    if duration_months.shape != score.shape or event.shape != score.shape or at_risk.shape != score.shape:
        raise ValueError("Cox labels must align with score [batch,organ]")
    if not torch.isfinite(score).all() or not torch.isfinite(duration_months).all():
        raise ValueError("Cox score and duration must be finite")
    for name, values in (("event", event), ("at_risk", at_risk)):
        if not torch.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} must contain only 0/1")
    # CUDA and CPU do not implement logcumsumexp backward for float16.
    # Keep this numerically sensitive risk-set calculation in float32 under AMP.
    working_score = score.float()
    working_duration = duration_months.float()
    risk_mask = at_risk.bool()
    event_mask = event.bool()
    if (event_mask & ~risk_mask).any():
        raise ValueError("event=1 is invalid outside the at-risk set")
    if (duration_months[risk_mask] <= 0).any():
        raise ValueError("at-risk durations must be strictly positive")

    organ_losses = []
    for organ_index in range(score.shape[1]):
        selected = risk_mask[:, organ_index]
        selected_events = event_mask[selected, organ_index]
        if int(selected.sum()) < 2 or not selected_events.any():
            continue
        organ_score = working_score[selected, organ_index]
        organ_duration = working_duration[selected, organ_index]
        order = torch.argsort(organ_duration, descending=True, stable=True)
        sorted_score = organ_score[order]
        sorted_duration = organ_duration[order]
        sorted_event = selected_events[order]

        _, tie_counts = torch.unique_consecutive(sorted_duration, return_counts=True)
        tie_ends = torch.cumsum(tie_counts, dim=0) - 1
        cumulative_log_risk = torch.logcumsumexp(sorted_score, dim=0)
        tie_log_risk = cumulative_log_risk.index_select(0, tie_ends)
        log_risk_set = torch.repeat_interleave(tie_log_risk, tie_counts)
        organ_losses.append(-(sorted_score[sorted_event] - log_risk_set[sorted_event]).mean())
    if not organ_losses:
        return _zero(working_score)
    return torch.stack(organ_losses).mean()

def ehr_age_joint_loss(
    output: EHRAGEOutput,
    *,
    chronological_age: torch.Tensor,
    strict_healthy_reference: torch.Tensor,
    duration_months: torch.Tensor,
    event: torch.Tensor,
    at_risk: torch.Tensor,
    interval_edges_months: torch.Tensor,
    config: AgePreservingLossConfig | None = None,
    survival_multiplier: float = 1.0,
    augmented_output: EHRAGEOutput | None = None,
) -> dict[str, torch.Tensor]:
    """Compute joint losses; future labels are loss-only and never model inputs."""

    cfg = config or AgePreservingLossConfig()
    cfg.validate()
    if not 0.0 <= survival_multiplier <= 1.0:
        raise ValueError("survival_multiplier must lie in [0,1]")
    age = chronological_age.float()
    if output.base_age.shape != age.shape:
        raise ValueError("base age and chronological age must align")
    if not torch.isfinite(age).all():
        raise ValueError("chronological_age must be finite")
    if output.predicted_age.shape != output.risk_age_offset.shape:
        raise ValueError("predicted age and organ offset must align")
    if output.predicted_age.shape[0] != len(age):
        raise ValueError("organ predictions and chronological age must align")

    expanded_age = age.unsqueeze(1).expand_as(output.predicted_age)
    base_huber = F.smooth_l1_loss(output.base_age, age, beta=cfg.huber_beta_years)
    if output.base_view_ages is None:
        base_view_huber = _zero(output.base_age)
    else:
        if (
            output.base_view_ages.ndim != 2
            or output.base_view_ages.shape[0] != len(age)
        ):
            raise ValueError("base view ages must have shape [batch,view]")
        view_targets = age.unsqueeze(1).expand_as(output.base_view_ages)
        base_view_huber = F.smooth_l1_loss(
            output.base_view_ages,
            view_targets,
            beta=cfg.huber_beta_years,
        )
    base_mse = ((output.base_age - age) / cfg.age_scale_years).square().mean()
    base_correlation = _correlation_loss(output.base_age, age)
    final_all_huber = F.smooth_l1_loss(
        output.predicted_age, expanded_age, beta=cfg.huber_beta_years
    )

    if strict_healthy_reference.shape != age.shape:
        raise ValueError("strict_healthy_reference and chronological age must align")
    if not torch.all(
        (strict_healthy_reference == 0) | (strict_healthy_reference == 1)
    ):
        raise ValueError("strict_healthy_reference must contain only 0/1")
    strict = strict_healthy_reference.bool()
    if strict.any():
        strict_final_huber = F.smooth_l1_loss(
            output.predicted_age[strict],
            expanded_age[strict],
            beta=cfg.huber_beta_years,
        )
        strict_offset = F.smooth_l1_loss(
            output.risk_age_offset[strict],
            torch.zeros_like(output.risk_age_offset[strict]),
            beta=1.0,
        )
    else:
        strict_final_huber = _zero(output.predicted_age)
        strict_offset = _zero(output.risk_age_offset)
    strict_center, strict_slope = _strict_calibration_terms(
        output.predicted_age,
        age,
        strict,
        age_scale=cfg.age_scale_years,
    )
    survival = piecewise_exponential_nll(
        output.interval_log_rate,
        duration_months.float(),
        event,
        at_risk,
        interval_edges_months=interval_edges_months,
    )
    evaluator_gap = evaluator_aligned_raw_age_gap(
        output,
        age,
        stop_base_age_gradient=True,
    )
    evaluator_gap_cox = cause_specific_cox_ph_loss(
        evaluator_gap / cfg.evaluator_gap_cox_scale_years,
        duration_months.float(),
        event,
        at_risk,
    )
    if augmented_output is None:
        consistency = _zero(output.predicted_age)
    else:
        consistency = F.smooth_l1_loss(
            augmented_output.base_age,
            output.base_age.detach(),
            beta=cfg.huber_beta_years,
        )
        consistency = consistency + F.smooth_l1_loss(
            augmented_output.risk_age_offset,
            output.risk_age_offset.detach(),
            beta=1.0,
        )

    weighted_survival = cfg.survival_weight * float(survival_multiplier) * survival
    weighted_evaluator_gap_cox = (
        cfg.evaluator_gap_cox_weight * float(survival_multiplier) * evaluator_gap_cox
    )
    total = (
        cfg.base_huber_weight * base_huber
        + cfg.base_view_huber_weight * base_view_huber
        + cfg.base_mse_weight * base_mse
        + cfg.base_correlation_weight * base_correlation
        + cfg.final_all_huber_weight * final_all_huber
        + cfg.strict_final_huber_weight * strict_final_huber
        + cfg.strict_offset_weight * strict_offset
        + cfg.strict_center_weight * strict_center
        + cfg.strict_slope_weight * strict_slope
        + weighted_survival
        + weighted_evaluator_gap_cox
        + cfg.consistency_weight * consistency
    )
    return {
        "total": total,
        "base_huber": base_huber,
        "base_view_huber": base_view_huber,
        "base_mse": base_mse,
        "base_correlation": base_correlation,
        "final_all_huber": final_all_huber,
        "strict_final_huber": strict_final_huber,
        "strict_offset": strict_offset,
        "strict_center": strict_center,
        "strict_slope": strict_slope,
        "survival": survival,
        "weighted_survival": weighted_survival,
        "evaluator_gap_cox": evaluator_gap_cox,
        "weighted_evaluator_gap_cox": weighted_evaluator_gap_cox,
        "consistency": consistency,
    }
