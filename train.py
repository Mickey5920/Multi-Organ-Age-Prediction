from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Dataloader import MODEL_INPUT_KEYS, TARGET_KEYS
from model import EHRAGEConfig, EHRAGEModel, EHRAGEOutput, ORGAN_SYSTEMS


@dataclass(frozen=True)
class LossConfig:
    huber_beta_years: float = 1.0
    base_view_huber_weight: float = 2.0
    strict_offset_weight: float = 0.75
    survival_weight: float = 0.5
    evaluator_gap_cox_weight: float = 0.25
    evaluator_gap_cox_scale_years: float = 2.0
    consistency_weight: float = 0.1

    def validate(self) -> None:
        if self.huber_beta_years <= 0.0 or self.evaluator_gap_cox_scale_years <= 0.0:
            raise ValueError("loss scales must be positive")
        weights = (
            self.base_view_huber_weight,
            self.strict_offset_weight,
            self.survival_weight,
            self.evaluator_gap_cox_weight,
            self.consistency_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("loss weights must be nonnegative")


def configure_reproducibility(seed: int = 7210) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


def piecewise_exponential_nll(
    interval_log_rate: torch.Tensor,
    duration_months: torch.Tensor,
    event: torch.Tensor,
    at_risk: torch.Tensor,
    interval_edges_months: torch.Tensor,
) -> torch.Tensor:
    if interval_log_rate.ndim != 3:
        raise ValueError("interval_log_rate must have shape [batch,organ,interval]")
    batch, organs, intervals = interval_log_rate.shape
    expected = (batch, organs)
    if duration_months.shape != expected or event.shape != expected or at_risk.shape != expected:
        raise ValueError("survival labels must align with [batch,organ]")
    edges = interval_edges_months.to(interval_log_rate.device, interval_log_rate.dtype)
    if edges.ndim != 1 or len(edges) != intervals + 1:
        raise ValueError("interval edges must contain K+1 values")
    if not torch.isfinite(interval_log_rate).all() or not torch.isfinite(duration_months).all():
        raise ValueError("survival inputs must be finite")
    if not torch.isclose(edges[0], edges.new_zeros(()), atol=1e-7, rtol=0.0):
        raise ValueError("interval edges must start at zero")
    if not torch.all(edges[1:] > edges[:-1]):
        raise ValueError("interval edges must be strictly increasing")
    risk_mask = at_risk.bool()
    event_mask = event.bool()
    if (event_mask & ~risk_mask).any():
        raise ValueError("event=1 is invalid outside the at-risk set")
    duration = torch.where(risk_mask, duration_months, torch.zeros_like(duration_months))
    if (duration[risk_mask] <= 0).any() or (duration[risk_mask] > edges[-1] + 1e-4).any():
        raise ValueError("at-risk durations must lie in the interval range")
    starts = edges[:-1].view(1, 1, -1)
    ends = edges[1:].view(1, 1, -1)
    exposure = (torch.minimum(duration.unsqueeze(-1), ends) - starts).clamp_min(0.0)
    exposure = torch.minimum(exposure, ends - starts)
    event_bin = torch.bucketize(duration.contiguous(), edges[1:-1], right=False)
    event_indicator = F.one_hot(event_bin.long(), num_classes=intervals).to(interval_log_rate.dtype)
    event_indicator = event_indicator * event_mask.to(interval_log_rate.dtype).unsqueeze(-1)
    bounded_log_rate = interval_log_rate.clamp(min=-20.0, max=5.0)
    row_nll = (torch.exp(bounded_log_rate) * exposure - event_indicator * bounded_log_rate).sum(dim=-1)
    losses = []
    for organ_index in range(organs):
        selected = risk_mask[:, organ_index]
        if selected.any():
            losses.append(row_nll[selected, organ_index].mean())
    if not losses:
        return interval_log_rate.sum() * 0.0
    return torch.stack(losses).mean()


def cause_specific_cox_loss(
    score: torch.Tensor,
    duration_months: torch.Tensor,
    event: torch.Tensor,
    at_risk: torch.Tensor,
) -> torch.Tensor:
    if score.ndim != 2:
        raise ValueError("score must have shape [batch,organ]")
    if duration_months.shape != score.shape or event.shape != score.shape or at_risk.shape != score.shape:
        raise ValueError("Cox labels must align with score")
    working_score = score.float()
    working_duration = duration_months.float()
    risk_mask = at_risk.bool()
    event_mask = event.bool()
    if (event_mask & ~risk_mask).any():
        raise ValueError("event=1 is invalid outside the at-risk set")
    losses = []
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
        losses.append(-(sorted_score[sorted_event] - log_risk_set[sorted_event]).mean())
    if not losses:
        return working_score.sum() * 0.0
    return torch.stack(losses).mean()


def compact_event_dropout(
    inputs: Mapping[str, torch.Tensor], probability: float
) -> dict[str, torch.Tensor]:
    if not 0.0 <= probability < 1.0:
        raise ValueError("event dropout probability must lie in [0,1)")
    result = {key: value for key, value in inputs.items()}
    if probability == 0.0:
        return result
    valid = inputs["event_valid_mask"].bool()
    keep = valid & torch.rand(valid.shape, device=valid.device).ge(probability)
    destination = keep.long().cumsum(dim=1) - 1
    rows, columns = torch.nonzero(keep, as_tuple=True)
    target_columns = destination[rows, columns]
    for key in (
        "event_code_id",
        "event_text_id",
        "event_relative_days",
        "event_type_id",
        "event_valid_mask",
    ):
        source = inputs[key]
        compacted = torch.zeros_like(source)
        compacted[rows, target_columns] = source[rows, columns]
        result[key] = compacted
    return result


def survival_multiplier(epoch: int, ramp_start: float = 0.1, ramp_epochs: int = 15) -> float:
    if epoch < 1 or not 0.0 <= ramp_start <= 1.0 or ramp_epochs <= 0:
        raise ValueError("invalid survival ramp configuration")
    progress = min(max(epoch - 1, 0) / max(ramp_epochs - 1, 1), 1.0)
    return ramp_start + (1.0 - ramp_start) * progress


def ehr_age_loss(
    output: EHRAGEOutput,
    chronological_age: torch.Tensor,
    strict_healthy_reference: torch.Tensor,
    duration_months: torch.Tensor,
    event: torch.Tensor,
    at_risk: torch.Tensor,
    interval_edges_months: torch.Tensor,
    config: LossConfig | None = None,
    survival_weight_multiplier: float = 1.0,
    augmented_output: EHRAGEOutput | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or LossConfig()
    cfg.validate()
    if not 0.0 <= survival_weight_multiplier <= 1.0:
        raise ValueError("survival_weight_multiplier must lie in [0,1]")
    age = chronological_age.float()
    if output.base_view_ages.ndim != 2 or output.base_view_ages.shape[0] != len(age):
        raise ValueError("base_view_ages must have shape [batch,view]")
    view_target = age.unsqueeze(1).expand_as(output.base_view_ages)
    base_view_huber = F.smooth_l1_loss(
        output.base_view_ages, view_target, beta=cfg.huber_beta_years
    )
    strict = strict_healthy_reference.bool()
    if strict.any():
        strict_offset = F.smooth_l1_loss(
            output.risk_age_offset[strict],
            torch.zeros_like(output.risk_age_offset[strict]),
            beta=1.0,
        )
    else:
        strict_offset = output.risk_age_offset.sum() * 0.0
    survival = piecewise_exponential_nll(
        output.interval_log_rate,
        duration_months.float(),
        event,
        at_risk,
        interval_edges_months,
    )
    raw_gap = output.base_age.detach().unsqueeze(1) - age.unsqueeze(1) + output.risk_age_offset
    evaluator_gap_cox = cause_specific_cox_loss(
        raw_gap / cfg.evaluator_gap_cox_scale_years,
        duration_months.float(),
        event,
        at_risk,
    )
    if augmented_output is None:
        consistency = output.predicted_age.sum() * 0.0
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
    weighted_survival = cfg.survival_weight * survival_weight_multiplier * survival
    weighted_gap_cox = cfg.evaluator_gap_cox_weight * survival_weight_multiplier * evaluator_gap_cox
    total = (
        cfg.base_view_huber_weight * base_view_huber
        + cfg.strict_offset_weight * strict_offset
        + weighted_survival
        + weighted_gap_cox
        + cfg.consistency_weight * consistency
    )
    return {
        "total": total,
        "base_view_huber": base_view_huber,
        "strict_offset": strict_offset,
        "survival": survival,
        "weighted_survival": weighted_survival,
        "evaluator_gap_cox": evaluator_gap_cox,
        "weighted_evaluator_gap_cox": weighted_gap_cox,
        "consistency": consistency,
    }


def split_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = {
        key: batch[key].to(device, non_blocking=True) for key in MODEL_INPUT_KEYS
    }
    targets = {
        key: batch[key].to(device, non_blocking=True) for key in TARGET_KEYS
    }
    return inputs, targets


def train_one_epoch(
    model: EHRAGEModel,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    loss_config: LossConfig,
    interval_edges: torch.Tensor,
    event_dropout_probability: float = 0.1,
    gradient_clip_norm: float = 1.0,
    amp: bool = True,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    observed = 0
    multiplier = survival_multiplier(epoch)
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = split_batch(batch, device)
        thinned = compact_event_dropout(inputs, event_dropout_probability)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(**inputs)
            augmented = model(**thinned) if event_dropout_probability > 0.0 else None
            losses = ehr_age_loss(
                output,
                **targets,
                interval_edges_months=interval_edges,
                config=loss_config,
                survival_weight_multiplier=multiplier,
                augmented_output=augmented,
            )
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(targets["chronological_age"])
        observed += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
    if observed == 0:
        raise RuntimeError("training loader produced no batches")
    return {key: value / observed for key, value in totals.items()}


@torch.no_grad()
def predict(
    model: EHRAGEModel,
    loader,
    device: torch.device,
    amp: bool = True,
) -> dict[str, torch.Tensor]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        "predicted_age": [],
        "base_age": [],
        "risk_score": [],
        "risk_age_offset": [],
        "base_view_ages": [],
        "chronological_age": [],
        "strict_healthy_reference": [],
    }
    for batch in loader:
        inputs, targets = split_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(**inputs)
        for key in (
            "predicted_age",
            "base_age",
            "risk_score",
            "risk_age_offset",
            "base_view_ages",
        ):
            collected[key].append(getattr(output, key).float().cpu())
        collected["chronological_age"].append(targets["chronological_age"].float().cpu())
        collected["strict_healthy_reference"].append(
            targets["strict_healthy_reference"].bool().cpu()
        )
    return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def strict_age_metrics(prediction: Mapping[str, torch.Tensor]) -> dict[str, float]:
    strict = prediction["strict_healthy_reference"].bool()
    predicted = prediction["predicted_age"][strict].reshape(-1).double()
    target = prediction["chronological_age"][strict].repeat_interleave(
        len(ORGAN_SYSTEMS)
    ).double()
    if len(predicted) < 2:
        return {"mae": math.nan, "pearson": math.nan, "r2": math.nan}
    error = predicted - target
    mae = float(error.abs().mean())
    centered_prediction = predicted - predicted.mean()
    centered_target = target - target.mean()
    denominator = centered_prediction.square().sum().sqrt() * centered_target.square().sum().sqrt()
    pearson = float((centered_prediction * centered_target).sum() / denominator.clamp_min(1e-12))
    r2 = float(1.0 - error.square().sum() / centered_target.square().sum().clamp_min(1e-12))
    return {"mae": mae, "pearson": pearson, "r2": r2}


def fit_osag_correction(prediction: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    strict = prediction["strict_healthy_reference"].bool()
    age = prediction["chronological_age"][strict].double()
    predicted = prediction["predicted_age"][strict].double()
    if len(age) < 3:
        raise ValueError("at least three strict-reference participants are required")
    gap = predicted - age.unsqueeze(1)
    design = torch.stack((torch.ones_like(age), age), dim=1)
    coefficients = torch.linalg.lstsq(design, gap).solution
    return {
        "intercept": coefficients[0].float(),
        "slope": coefficients[1].float(),
    }


def apply_osag_correction(
    predicted_age: torch.Tensor,
    chronological_age: torch.Tensor,
    correction: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    age = chronological_age.float().unsqueeze(1)
    raw_gap = predicted_age.float() - age
    intercept = correction["intercept"].to(age.device, age.dtype).unsqueeze(0)
    slope = correction["slope"].to(age.device, age.dtype).unsqueeze(0)
    expected_gap = intercept + age * slope
    return raw_gap - expected_gap


def save_checkpoint(
    path: str | Path,
    model: EHRAGEModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: EHRAGEConfig,
    loss_config: LossConfig,
    standardization: Mapping[str, torch.Tensor],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "loss_config": asdict(loss_config),
            "standardization": dict(standardization),
            "organ_systems": ORGAN_SYSTEMS,
        },
        Path(path),
    )
