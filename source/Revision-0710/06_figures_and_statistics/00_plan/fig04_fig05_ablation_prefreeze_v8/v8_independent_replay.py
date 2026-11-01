from __future__ import annotations

import hashlib
import math
import os
import random
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

import v8_inputs
import v8_loss
import v8_model


INTERVAL_EDGES = torch.tensor([0.0, 12.0, 24.0, 36.0, 48.0, 60.0])


class IndependentReplayError(RuntimeError):
    pass


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, dict):
        digest.update(b"D\0")
        keys: list[tuple[bytes, Any]] = []
        for key in value:
            key_digest = hashlib.sha256()
            _update_digest(key_digest, key)
            keys.append((key_digest.digest(), key))
        for encoded, key in sorted(keys, key=lambda item: item[0]):
            digest.update(encoded)
            _update_digest(digest, value[key])
        digest.update(b"d\0")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"L\0" if isinstance(value, list) else b"Q\0")
        for item in value:
            _update_digest(digest, item)
        digest.update(b"l\0")
        return
    if value is None:
        digest.update(b"N\0")
        return
    if isinstance(value, bool):
        digest.update(b"B1\0" if value else b"B0\0")
        return
    if isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii") + b"\0")
        return
    if isinstance(value, float):
        digest.update(b"F" + struct.pack(">d", float(value)))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded)
        return
    raise IndependentReplayError(f"unsupported canonical value: {type(value)!r}")


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _seed(seed: int, device: torch.device) -> None:
    configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if configured != ":4096:8":
        raise IndependentReplayError("deterministic CUBLAS workspace changed")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _freeze(model: nn.Module, variant_id: str) -> None:
    freeze_exact: set[str] = set()
    freeze_fragments: tuple[str, ...] = ()
    if variant_id == "fig04_04_no_survival_supervision":
        freeze_exact.add("model.baseline_log_rate")
    elif variant_id == "fig05_02_no_hospital_icd":
        freeze_fragments = (
            "event_encoder.code_projection",
            "backbone.token_encoder.code_projection",
        )
    elif variant_id == "fig05_04_no_event_text":
        freeze_fragments = (
            "event_encoder.text_projection",
            "backbone.token_encoder.text_projection",
        )
    elif variant_id == "fig05_05_no_longitudinal_events":
        freeze_fragments = ("model.event_encoder.", "model.private_retain_age_tower.backbone.")
    for name, parameter in model.named_parameters():
        if name in freeze_exact or any(fragment in name for fragment in freeze_fragments):
            parameter.requires_grad_(False)


def _named_trainable_sha256(named: list[tuple[str, nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _lr_multiplier(epoch: int, warmup_epochs: int, horizon_epochs: int) -> float:
    scheduler_index = int(epoch) - 1
    if scheduler_index < warmup_epochs:
        return float((scheduler_index + 1) / max(1, warmup_epochs))
    progress = (scheduler_index - warmup_epochs) / (horizon_epochs - warmup_epochs)
    return float(0.5 * (1.0 + math.cos(math.pi * progress)))


def _survival_multiplier(epoch: int, ramp_epochs: int = 15) -> float:
    progress = min(max((int(epoch) - 1) / (ramp_epochs - 1), 0.0), 1.0)
    return float(0.1 + 0.9 * progress)


def _iter_batches(bundle: Any, batch_size: int, seed: int, iteration: int):
    n = len(bundle.participant_ids)
    generator = torch.Generator().manual_seed(int(seed) + int(iteration))
    positions = torch.arange(n)[torch.randperm(n, generator=generator)]
    for start in range(0, n, int(batch_size)):
        selected = positions[start : start + int(batch_size)]
        yield (
            tuple(bundle.participant_ids[index] for index in selected.tolist()),
            {key: value.index_select(0, selected) for key, value in bundle.inputs.items()},
            {key: value.index_select(0, selected) for key, value in bundle.labels.items()},
        )


@dataclass(frozen=True)
class ReplayResult:
    initial_full_model_sha256: str
    initial_trainable_sha256: str
    initial_optimizer_sha256: str
    initial_scaler_sha256: str
    capture_model_state_sha256: str
    capture_trajectory_sha256: str
    final_model_state_sha256: str
    final_optimizer_state_sha256: str
    final_scaler_state_sha256: str
    final_trajectory_sha256: str
    epochs: tuple[dict[str, Any], ...]
    capture_state_dict: dict[str, torch.Tensor]


def replay_training(
    model: nn.Module,
    *,
    variant_id: str,
    train_bundle: Any,
    protocol: dict[str, Any],
    seed: int,
) -> ReplayResult:
    """Independent training implementation used only by the audit verifier."""

    device = torch.device(str(protocol["device"]))
    model.to(device)
    _freeze(model, variant_id)
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        [value for _, value in named],
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(protocol["amp"] and device.type == "cuda")
    )
    initial_full = state_sha256(model.state_dict())
    initial_trainable = _named_trainable_sha256(named)
    initial_optimizer = state_sha256(optimizer.state_dict())
    initial_scaler = state_sha256(scaler.state_dict())
    trajectory = state_sha256(
        {
            "schema_version": "revision0710.fig04_fig05.training_trajectory_seed.v8",
            "variant_id": variant_id,
            "seed": int(seed),
            "initial_full_model_sha256": initial_full,
            "initial_trainable_sha256": initial_trainable,
            "initial_optimizer_sha256": initial_optimizer,
            "initial_scaler_sha256": initial_scaler,
        }
    )
    _seed(seed, device)
    policy = v8_loss.variant_loss_policy(variant_id)
    rows: list[dict[str, Any]] = []
    capture_state: dict[str, torch.Tensor] | None = None
    capture_model_sha: str | None = None
    capture_trajectory_sha: str | None = None
    epochs = int(protocol["train_through_epoch"])
    for epoch in range(1, epochs + 1):
        model.train()
        current_lr = float(protocol["learning_rate"]) * _lr_multiplier(
            epoch,
            int(protocol["warmup_epochs"]),
            int(protocol["scheduler_horizon_epochs"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        step_chain = state_sha256(
            {
                "schema_version": "revision0710.fig04_fig05.training_epoch_step_seed.v8",
                "previous_trajectory_sha256": trajectory,
                "epoch": int(epoch),
            }
        )
        steps = 0
        for batch_index, (participant_ids, inputs_cpu, labels_cpu) in enumerate(
            _iter_batches(train_bundle, int(protocol["batch_size"]), seed, epoch - 1)
        ):
            optimizer.zero_grad(set_to_none=True)
            base_inputs = v8_inputs.apply_variant_inputs(inputs_cpu, variant_id)
            inputs = {key: value.to(device) for key, value in base_inputs.items()}
            labels = {key: value.to(device) for key, value in labels_cpu.items()}
            generator = torch.Generator(device=device.type).manual_seed(
                int(seed + epoch * 1009 + batch_index)
            )
            thinned = v8_inputs.compact_event_dropout(
                inputs,
                probability=float(policy["event_dropout_probability"]),
                generator=generator,
            )
            dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=bool(protocol["amp"])):
                output = model(**inputs)
                augmented = (
                    model(**thinned)
                    if float(policy["event_dropout_probability"]) > 0.0
                    else None
                )
                losses, _ = v8_loss.compute_variant_loss(
                    variant_id,
                    output,
                    labels=labels,
                    interval_edges_months=INTERVAL_EDGES.to(device),
                    survival_multiplier=_survival_multiplier(epoch),
                    augmented_output=augmented,
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), float(protocol["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            loss_value = float(losses["total"].detach())
            step_chain = state_sha256(
                {
                    "previous_step_sha256": step_chain,
                    "epoch": int(epoch),
                    "batch_index": int(batch_index),
                    "participant_ids": tuple(str(value) for value in participant_ids),
                    "loss_hex": float(loss_value).hex(),
                    "preclip_gradient_norm_hex": float(gradient_norm).hex(),
                    "learning_rate_hex": float(current_lr).hex(),
                    "scaler_scale_hex": float(scaler.get_scale()).hex(),
                }
            )
            steps += 1
        model_sha = state_sha256(model.state_dict())
        optimizer_sha = state_sha256(optimizer.state_dict())
        scaler_sha = state_sha256(scaler.state_dict())
        trajectory = state_sha256(
            {
                "previous_trajectory_sha256": trajectory,
                "epoch": int(epoch),
                "optimizer_steps": int(steps),
                "step_chain_sha256": step_chain,
                "model_state_sha256": model_sha,
                "optimizer_state_sha256": optimizer_sha,
                "scaler_state_sha256": scaler_sha,
            }
        )
        rows.append(
            {
                "epoch": int(epoch),
                "optimizer_steps": int(steps),
                "step_chain_sha256": step_chain,
                "model_state_sha256": model_sha,
                "optimizer_state_sha256": optimizer_sha,
                "scaler_state_sha256": scaler_sha,
                "trajectory_sha256": trajectory,
            }
        )
        if epoch == int(protocol["capture_epoch"]):
            capture_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            capture_model_sha = model_sha
            capture_trajectory_sha = trajectory
    if capture_state is None or capture_model_sha is None or capture_trajectory_sha is None:
        raise IndependentReplayError("capture epoch was not replayed")
    return ReplayResult(
        initial_full_model_sha256=initial_full,
        initial_trainable_sha256=initial_trainable,
        initial_optimizer_sha256=initial_optimizer,
        initial_scaler_sha256=initial_scaler,
        capture_model_state_sha256=capture_model_sha,
        capture_trajectory_sha256=capture_trajectory_sha,
        final_model_state_sha256=rows[-1]["model_state_sha256"],
        final_optimizer_state_sha256=rows[-1]["optimizer_state_sha256"],
        final_scaler_state_sha256=rows[-1]["scaler_state_sha256"],
        final_trajectory_sha256=rows[-1]["trajectory_sha256"],
        epochs=tuple(rows),
        capture_state_dict=capture_state,
    )


def replay_outputs(
    model: nn.Module,
    bundle: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent row-order inference; production prediction_frame is not called."""

    model.to(device)
    model.eval()
    ages: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(bundle.participant_ids), int(batch_size)):
            stop = min(start + int(batch_size), len(bundle.participant_ids))
            inputs = {
                key: value[start:stop].to(device) for key, value in bundle.inputs.items()
            }
            output = model(**inputs)
            ages.append(output.predicted_age.float().cpu().numpy())
            risks.append(output.risk_score.float().cpu().numpy())
    return np.concatenate(ages, axis=0).reshape(-1), np.concatenate(risks, axis=0).reshape(-1)
