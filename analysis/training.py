from __future__ import annotations

from torch.nn import Module as EHRAGEModel

import os

import random

from dataclasses import dataclass

from typing import Any, Mapping

import numpy as np

import torch

from torch import nn

from .data import JointBatch, JointVectorizedBatcher, compact_event_dropout

from .losses import AgePreservingLossConfig, ehr_age_joint_loss

@dataclass(frozen=True)
class PredictionBundle:
    participant_indices: np.ndarray
    chronological_age: np.ndarray
    strict_healthy: np.ndarray
    duration_months: np.ndarray
    event: np.ndarray
    at_risk: np.ndarray
    sex: np.ndarray
    predicted_age: np.ndarray
    risk_score: np.ndarray

def configure_reproducibility(seed: int, *, deterministic: bool, tf32: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))

def _move_inputs(inputs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}

def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)

def _move_labels(batch: JointBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "chronological_age": batch.chronological_age.to(device, non_blocking=True),
        "strict_healthy_reference": batch.strict_healthy.to(device, non_blocking=True),
        "duration_months": batch.duration_months.to(device, non_blocking=True),
        "event": batch.event.to(device, non_blocking=True),
        "at_risk": batch.at_risk.to(device, non_blocking=True),
    }

def _train_epoch(
    *,
    model: EHRAGEModel,
    batcher: JointVectorizedBatcher,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    interval_edges: torch.Tensor,
    loss_config: AgePreservingLossConfig,
    survival_multiplier: float,
    event_dropout_probability: float,
    gradient_clip_norm: float,
    amp: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    observed_n = 0
    optimizer_steps = 0
    amp_skipped_steps = 0
    for batch_index, batch in enumerate(batcher):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        inputs = _move_inputs(batch.inputs, device)
        labels = _move_labels(batch, device)
        thinned = compact_event_dropout(inputs, probability=event_dropout_probability)
        with _autocast(device, amp and device.type == "cuda"):
            output = model(**inputs)
            augmented = model(**thinned) if event_dropout_probability > 0.0 else None
            losses = ehr_age_joint_loss(
                output,
                **labels,
                interval_edges_months=interval_edges,
                config=loss_config,
                survival_multiplier=survival_multiplier,
                augmented_output=augmented,
            )
        scale_before = float(scaler.get_scale())
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        step_skipped = scaler.is_enabled() and float(scaler.get_scale()) < scale_before
        amp_skipped_steps += int(step_skipped)
        optimizer_steps += int(not step_skipped)
        batch_n = len(batch.chronological_age)
        observed_n += batch_n
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_n
    if observed_n == 0:
        raise RuntimeError("training batcher produced no batches")
    result = {key: value / observed_n for key, value in totals.items()}
    result["optimizer_steps"] = float(optimizer_steps)
    result["amp_skipped_steps"] = float(amp_skipped_steps)
    return result

def _predict(
    *,
    model: EHRAGEModel,
    batcher: JointVectorizedBatcher,
    device: torch.device,
    amp: bool,
    max_batches: int | None,
) -> PredictionBundle:
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "participant_indices",
            "chronological_age",
            "strict_healthy",
            "duration_months",
            "event",
            "at_risk",
            "sex",
            "predicted_age",
            "risk_score",
        )
    }
    with torch.inference_mode():
        for batch_index, batch in enumerate(batcher):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = _move_inputs(batch.inputs, device)
            with _autocast(device, amp and device.type == "cuda"):
                output = model(**inputs)
            chunks["participant_indices"].append(batch.participant_indices.numpy())
            chunks["chronological_age"].append(batch.chronological_age.numpy())
            chunks["strict_healthy"].append(batch.strict_healthy.numpy())
            chunks["duration_months"].append(batch.duration_months.numpy())
            chunks["event"].append(batch.event.numpy())
            chunks["at_risk"].append(batch.at_risk.numpy())
            chunks["sex"].append(batch.inputs["sex"].numpy())
            chunks["predicted_age"].append(output.predicted_age.float().cpu().numpy())
            chunks["risk_score"].append(output.risk_score.float().cpu().numpy())
    if not chunks["participant_indices"]:
        raise RuntimeError("prediction batcher produced no batches")
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    return PredictionBundle(**arrays)

def _loss_config(payload: Mapping[str, Any]) -> AgePreservingLossConfig:
    allowed = set(AgePreservingLossConfig.__dataclass_fields__)
    return AgePreservingLossConfig(**{key: payload[key] for key in allowed if key in payload})

def _survival_multiplier(epoch: int, loss_payload: Mapping[str, Any]) -> float:
    start = float(loss_payload.get("survival_ramp_start_fraction", 0.1))
    ramp_epochs = int(loss_payload.get("survival_ramp_epochs", 15))
    if not 0.0 <= start <= 1.0 or ramp_epochs <= 0:
        raise ValueError("invalid survival ramp configuration")
    progress = min(max(epoch - 1, 0) / max(ramp_epochs - 1, 1), 1.0)
    return start + (1.0 - start) * progress
