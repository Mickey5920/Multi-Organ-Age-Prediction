from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AgeExpertLossConfig:
    huber_beta_years: float = 2.0
    age_scale_years: float = 10.0
    all_huber_weight: float = 1.0
    all_mse_weight: float = 0.25
    all_correlation_weight: float = 0.05
    strict_huber_weight: float = 2.0
    strict_center_weight: float = 0.2
    strict_slope_weight: float = 0.5
    consistency_weight: float = 0.1

    def validate(self) -> None:
        if self.huber_beta_years <= 0.0 or self.age_scale_years <= 0.0:
            raise ValueError("Huber beta and age scale must be positive")
        weights = (
            self.all_huber_weight,
            self.all_mse_weight,
            self.all_correlation_weight,
            self.strict_huber_weight,
            self.strict_center_weight,
            self.strict_slope_weight,
            self.consistency_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("age-expert loss weights must be non-negative")


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def flattened_correlation_loss(
    prediction: torch.Tensor,
    expanded_target: torch.Tensor,
) -> torch.Tensor:
    """Return 1-Pearson over canonical participant-organ rows in float32."""

    if prediction.shape != expanded_target.shape or prediction.ndim != 2:
        raise ValueError("correlation inputs must be aligned [participant,organ] tensors")
    raw_score = prediction.float().reshape(-1)
    raw_target = expanded_target.float().reshape(-1)
    if len(raw_score) < 2:
        return _zero(raw_score)

    # A direct mean of a large, non-power-of-two constant tensor can differ from
    # that constant by one float32 reduction ULP. That pseudo-variance makes an
    # undefined Pearson term look valid and gives it an unbounded AMP-scaled
    # gradient. Range is invariant to this reduction error. The tolerance also
    # treats variations below float32 resolution at the tensor's scale as
    # numerically degenerate, matching the frozen zero-term contract.
    epsilon = torch.finfo(raw_score.dtype).eps
    score_range = raw_score.amax() - raw_score.amin()
    target_range = raw_target.amax() - raw_target.amin()
    score_scale = raw_score.abs().amax().clamp_min(1.0)
    target_scale = raw_target.abs().amax().clamp_min(1.0)
    score_degenerate = score_range.detach() <= 8.0 * epsilon * score_scale.detach()
    target_degenerate = target_range.detach() <= 8.0 * epsilon * target_scale.detach()
    if bool(score_degenerate) or bool(target_degenerate):
        return _zero(raw_score)

    # Subtract a represented sample before the reduction. In exact arithmetic
    # this is the same centering as x-mean(x), but it avoids cancellation for
    # predictions with a large level and a comparatively small spread.
    score = raw_score - raw_score[0]
    target = raw_target - raw_target[0]
    score = score - score.mean()
    target = target - target.mean()
    denominator = score.square().mean().sqrt() * target.square().mean().sqrt()
    detached = denominator.detach()
    if not torch.isfinite(detached) or float(detached) <= 1e-8:
        return _zero(score)
    return 1.0 - (score * target).mean() / denominator


def strict_calibration_terms(
    prediction: torch.Tensor,
    chronological_age: torch.Tensor,
    strict_mask: torch.Tensor,
    *,
    age_scale_years: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return squared strict-reference error center and age slope."""

    selected = strict_mask.bool()
    working_prediction = prediction.float()
    working_age = chronological_age.float()
    if int(selected.sum()) < 2:
        zero = _zero(working_prediction)
        return zero, zero
    error = working_prediction[selected] - working_age[selected].unsqueeze(1)
    age = working_age[selected]
    center = (error.mean(dim=0) / float(age_scale_years)).square().mean()
    centered_age = age - age.mean()
    centered_error = error - error.mean(dim=0, keepdim=True)
    variance = centered_age.square().mean()
    if not torch.isfinite(variance.detach()) or float(variance.detach()) <= 1e-8:
        return center, _zero(working_prediction)
    slope = (centered_error * centered_age.unsqueeze(1)).mean(dim=0) / variance
    return center, slope.square().mean()


def age_expert_loss(
    prediction: torch.Tensor,
    *,
    chronological_age: torch.Tensor,
    strict_healthy_reference: torch.Tensor,
    config: AgeExpertLossConfig | None = None,
    augmented_prediction: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the preregistered D06-0 age-only objective."""

    cfg = config or AgeExpertLossConfig()
    cfg.validate()
    if prediction.ndim != 2 or prediction.shape[1] != 7:
        raise ValueError("canonical age prediction must have shape [participant,7]")
    age = chronological_age.float()
    if age.shape != (len(prediction),) or not torch.isfinite(age).all():
        raise ValueError("chronological age must be a finite [participant] tensor")
    if strict_healthy_reference.shape != age.shape:
        raise ValueError("strict-reference mask must align with chronological age")
    if not torch.all(
        (strict_healthy_reference == 0) | (strict_healthy_reference == 1)
    ):
        raise ValueError("strict-reference mask must contain only 0/1")

    score = prediction.float()
    if not torch.isfinite(score).all():
        raise ValueError("age prediction must be finite")
    target = age.unsqueeze(1).expand_as(score)
    all_huber = F.smooth_l1_loss(score, target, beta=cfg.huber_beta_years)
    all_mse = ((score - target) / cfg.age_scale_years).square().mean()
    all_correlation = flattened_correlation_loss(score, target)

    strict = strict_healthy_reference.bool()
    if strict.any():
        strict_huber = F.smooth_l1_loss(
            score[strict], target[strict], beta=cfg.huber_beta_years
        )
    else:
        strict_huber = _zero(score)
    strict_center, strict_slope = strict_calibration_terms(
        score,
        age,
        strict,
        age_scale_years=cfg.age_scale_years,
    )

    if augmented_prediction is None:
        consistency = _zero(score)
    else:
        if augmented_prediction.shape != prediction.shape:
            raise ValueError("dropout prediction must align with clean prediction")
        augmented = augmented_prediction.float()
        if not torch.isfinite(augmented).all():
            raise ValueError("dropout prediction must be finite")
        consistency = F.smooth_l1_loss(
            augmented,
            score.detach(),
            beta=cfg.huber_beta_years,
        )

    total = (
        cfg.all_huber_weight * all_huber
        + cfg.all_mse_weight * all_mse
        + cfg.all_correlation_weight * all_correlation
        + cfg.strict_huber_weight * strict_huber
        + cfg.strict_center_weight * strict_center
        + cfg.strict_slope_weight * strict_slope
        + cfg.consistency_weight * consistency
    )
    return {
        "total": total,
        "all_huber": all_huber,
        "all_mse": all_mse,
        "all_correlation": all_correlation,
        "strict_huber": strict_huber,
        "strict_center": strict_center,
        "strict_slope": strict_slope,
        "consistency": consistency,
    }
