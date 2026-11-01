from __future__ import annotations

import hashlib
import io
import math
import os
import random
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

import v8_inputs
import v8_loss
import v8_model


INTERVAL_EDGES = torch.tensor([0.0, 12.0, 24.0, 36.0, 48.0, 60.0])


@dataclass(frozen=True)
class TrainingBatch:
    participant_ids: tuple[str, ...]
    inputs: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]


def configure_reproducibility(seed: int, *, deterministic: bool, tf32: bool) -> None:
    if deterministic:
        configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if configured != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with the frozen deterministic setting")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)


def synthetic_batch(batch_size: int = 16, seed: int = 7210) -> TrainingBatch:
    if batch_size < 8:
        raise ValueError("synthetic training batch requires at least eight participants")
    generator = torch.Generator().manual_seed(int(seed))
    sequence = 6
    phenotype = torch.randn((batch_size, 78), generator=generator)
    missing = torch.zeros((batch_size, 78), dtype=torch.bool)
    missing[:, ::11] = True
    sex = torch.arange(batch_size, dtype=torch.int64) % 2
    event_type = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.int64).repeat(batch_size, 1)
    code = torch.zeros((batch_size, sequence), dtype=torch.int64)
    text = torch.zeros((batch_size, sequence), dtype=torch.int64)
    for column in range(sequence):
        if column % 2 == 0:
            code[:, column] = 1 + ((torch.arange(batch_size) + column) % 31)
            text[:, column] = 1 + ((torch.arange(batch_size) * 3 + column) % 47)
        else:
            text[:, column] = 1 + ((torch.arange(batch_size) * 5 + column) % 47)
    relative = -torch.tensor([720, 600, 480, 360, 240, 120], dtype=torch.int64).repeat(batch_size, 1)
    valid = torch.ones((batch_size, sequence), dtype=torch.bool)
    age = torch.linspace(45.0, 72.0, batch_size)
    strict = (torch.arange(batch_size) % 3 == 0).to(torch.int64)
    organs = 7
    duration = torch.empty((batch_size, organs), dtype=torch.float32)
    event = torch.zeros((batch_size, organs), dtype=torch.bool)
    for organ in range(organs):
        duration[:, organ] = 12.0 + ((torch.arange(batch_size) * 7 + organ * 3) % 45).float()
        event[:, organ] = (torch.arange(batch_size) + organ) % 4 == 0
    at_risk = torch.ones((batch_size, organs), dtype=torch.bool)
    return TrainingBatch(
        participant_ids=tuple(f"syn_{seed}_{index:04d}" for index in range(batch_size)),
        inputs={
            "phenotype": phenotype,
            "phenotype_missing_mask": missing,
            "sex": sex,
            "event_code_id": code,
            "event_text_id": text,
            "event_relative_days": relative,
            "event_type_id": event_type,
            "event_valid_mask": valid,
        },
        labels={
            "chronological_age": age,
            "strict_healthy_reference": strict,
            "duration_months": duration,
            "event": event,
            "at_risk": at_risk,
        },
    )


def _tensor_state_sha(named_parameters: Iterable[tuple[str, nn.Parameter]]) -> str:
    buffer = io.BytesIO()
    for name, parameter in named_parameters:
        value = parameter.detach().cpu().contiguous()
        buffer.write(name.encode("utf-8") + b"\0")
        buffer.write(str(tuple(value.shape)).encode("ascii") + b"\0")
        buffer.write(str(value.dtype).encode("ascii") + b"\0")
        buffer.write(value.numpy().tobytes(order="C"))
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _update_canonical_digest(digest: Any, value: Any) -> None:
    """Hash nested torch state without relying on pickle/zip serialization bytes."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, dict):
        digest.update(b"D\0")
        encoded_keys = []
        for key in value:
            key_digest = hashlib.sha256()
            _update_canonical_digest(key_digest, key)
            encoded_keys.append((key_digest.digest(), key))
        for key_bytes, key in sorted(encoded_keys, key=lambda item: item[0]):
            digest.update(key_bytes)
            _update_canonical_digest(digest, value[key])
        digest.update(b"d\0")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"L\0" if isinstance(value, list) else b"Q\0")
        for item in value:
            _update_canonical_digest(digest, item)
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
    raise TypeError(f"unsupported canonical state value: {type(value)!r}")


def canonical_state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_canonical_digest(digest, value)
    return digest.hexdigest()


def canonical_attestation_sha256(value: Any) -> str:
    return canonical_state_sha256(value)


def _freeze_by_variant(model: nn.Module, variant_id: str) -> list[str]:
    value = v8_model.variant_model_policy(variant_id)
    del value
    freeze_exact: set[str] = set()
    freeze_fragments: tuple[str, ...] = ()
    if variant_id == "fig04_04_no_survival_supervision":
        freeze_exact.add("model.baseline_log_rate")
    elif variant_id == "fig05_02_no_hospital_icd":
        freeze_fragments = ("event_encoder.code_projection", "backbone.token_encoder.code_projection")
    elif variant_id == "fig05_04_no_event_text":
        freeze_fragments = ("event_encoder.text_projection", "backbone.token_encoder.text_projection")
    elif variant_id == "fig05_05_no_longitudinal_events":
        freeze_fragments = ("model.event_encoder.", "model.private_retain_age_tower.backbone.")
    for name, parameter in model.named_parameters():
        if name in freeze_exact or any(fragment in name for fragment in freeze_fragments):
            parameter.requires_grad_(False)
    return sorted(
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    )


def build_optimizer_manifest(model: nn.Module, variant_id: str) -> dict[str, Any]:
    if getattr(model, "variant_id", None) != variant_id:
        raise ValueError("optimizer variant does not match model variant")
    frozen = _freeze_by_variant(model, variant_id)
    named = list(model.named_parameters())
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    names = [name for name, _ in trainable]
    return {
        "variant_id": variant_id,
        "model_parameter_names": [name for name, _ in named],
        "model_parameter_count": int(sum(parameter.numel() for _, parameter in named)),
        "trainable_parameter_names": names,
        "trainable_parameter_count": int(sum(parameter.numel() for _, parameter in trainable)),
        "optimizer_parameter_names": list(names),
        "optimizer_parameter_count": int(sum(parameter.numel() for _, parameter in trainable)),
        "frozen_or_unused_parameter_names": frozen,
        "initial_trainable_sha256": _tensor_state_sha(trainable),
        "optimizer_name_set_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
    }


def build_adamw(
    model: nn.Module,
    manifest: dict[str, Any],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    by_name = dict(model.named_parameters())
    names = list(manifest["optimizer_parameter_names"])
    if len(names) != len(set(names)) or any(name not in by_name for name in names):
        raise ValueError("optimizer manifest contains invalid names")
    parameters = [by_name[name] for name in names]
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("optimizer manifest includes frozen parameters")
    return torch.optim.AdamW(parameters, lr=float(lr), weight_decay=float(weight_decay))


def validate_optimizer_exact(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    manifest: dict[str, Any],
) -> None:
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    observed = [
        by_id[id(parameter)]
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected = list(manifest["optimizer_parameter_names"])
    if observed != expected or len(observed) != len(set(observed)):
        raise ValueError("optimizer parameter set/order differs from manifest")
    frozen = set(manifest["frozen_or_unused_parameter_names"])
    if frozen.intersection(observed):
        raise ValueError("optimizer contains frozen or structurally unused parameters")


def learning_rate_multiplier(epoch: int, *, warmup_epochs: int, horizon_epochs: int) -> float:
    epoch = int(epoch)
    if epoch < 1 or horizon_epochs < 1 or warmup_epochs < 0 or warmup_epochs >= horizon_epochs:
        raise ValueError("invalid scheduler epoch or horizon")
    # Canonical training constructs LambdaLR before E1 and calls scheduler.step()
    # after each epoch. Therefore epoch E uses lambda index E-1.
    scheduler_index = epoch - 1
    if scheduler_index < warmup_epochs:
        return float((scheduler_index + 1) / max(1, warmup_epochs))
    progress = (scheduler_index - warmup_epochs) / (horizon_epochs - warmup_epochs)
    return float(0.5 * (1.0 + math.cos(math.pi * progress)))


def _to_device(values: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in values.items()}


def _survival_multiplier(epoch: int, *, ramp_epochs: int = 15) -> float:
    start = 0.1
    if ramp_epochs <= 1:
        return 1.0
    progress = min(max((epoch - 1) / (ramp_epochs - 1), 0.0), 1.0)
    return float(start + (1.0 - start) * progress)


def train_fresh(
    model: nn.Module,
    variant_id: str,
    batches: Iterable[TrainingBatch],
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    capture_epoch: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    warmup_epochs: int,
    scheduler_horizon_epochs: int,
    amp: bool,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if not 1 <= capture_epoch <= epochs <= scheduler_horizon_epochs:
        raise ValueError("invalid training/capture horizon")
    configure_reproducibility(seed, deterministic=True, tf32=True)
    model.to(device)
    manifest = build_optimizer_manifest(model, variant_id)
    optimizer = build_adamw(
        model, manifest, lr=learning_rate, weight_decay=weight_decay
    )
    validate_optimizer_exact(model, optimizer, manifest)
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(amp and device.type == "cuda")
    )
    initial_full_model_sha = canonical_state_sha256(model.state_dict())
    initial_optimizer_sha = canonical_state_sha256(optimizer.state_dict())
    initial_scaler_sha = canonical_state_sha256(scaler.state_dict())
    trajectory_sha = canonical_attestation_sha256(
        {
            "schema_version": "revision0710.fig04_fig05.training_trajectory_seed.v8",
            "variant_id": variant_id,
            "seed": int(seed),
            "initial_full_model_sha256": initial_full_model_sha,
            "initial_trainable_sha256": manifest["initial_trainable_sha256"],
            "initial_optimizer_sha256": initial_optimizer_sha,
            "initial_scaler_sha256": initial_scaler_sha,
        }
    )
    policy = v8_loss.variant_loss_policy(variant_id)
    history: list[dict[str, Any]] = []
    capture_sha: str | None = None
    capture_model_sha: str | None = None
    capture_trajectory_sha: str | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        multiplier = learning_rate_multiplier(
            epoch,
            warmup_epochs=warmup_epochs,
            horizon_epochs=scheduler_horizon_epochs,
        )
        current_lr = float(learning_rate * multiplier)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        totals = 0.0
        steps = 0
        max_gradient_norm = 0.0
        step_sha = canonical_attestation_sha256(
            {
                "schema_version": "revision0710.fig04_fig05.training_epoch_step_seed.v8",
                "previous_trajectory_sha256": trajectory_sha,
                "epoch": int(epoch),
            }
        )
        for batch_index, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            base_inputs = v8_inputs.apply_variant_inputs(batch.inputs, variant_id)
            inputs = _to_device(base_inputs, device)
            labels = _to_device(batch.labels, device)
            dropout_probability = float(policy["event_dropout_probability"])
            dropout_generator = torch.Generator(device=device.type).manual_seed(
                int(seed + epoch * 1009 + batch_index)
            )
            thinned = v8_inputs.compact_event_dropout(
                inputs,
                probability=dropout_probability,
                generator=dropout_generator,
            )
            autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=bool(amp),
            ):
                output = model(**inputs)
                augmented = model(**thinned) if dropout_probability > 0.0 else None
                losses, wiring = v8_loss.compute_variant_loss(
                    variant_id,
                    output,
                    labels=labels,
                    interval_edges_months=INTERVAL_EDGES.to(device),
                    survival_multiplier=_survival_multiplier(epoch),
                    augmented_output=augmented,
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite training loss")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            scaler.step(optimizer)
            scaler.update()
            detached_loss = float(losses["total"].detach())
            detached_gradient = float(gradient_norm)
            totals += detached_loss
            steps += 1
            step_sha = canonical_attestation_sha256(
                {
                    "previous_step_sha256": step_sha,
                    "epoch": int(epoch),
                    "batch_index": int(batch_index),
                    "participant_ids": tuple(str(value) for value in batch.participant_ids),
                    "loss_hex": float(detached_loss).hex(),
                    "preclip_gradient_norm_hex": float(detached_gradient).hex(),
                    "learning_rate_hex": float(current_lr).hex(),
                    "scaler_scale_hex": float(scaler.get_scale()).hex(),
                }
            )
        end_model_sha = canonical_state_sha256(model.state_dict())
        end_optimizer_sha = canonical_state_sha256(optimizer.state_dict())
        end_scaler_sha = canonical_state_sha256(scaler.state_dict())
        trajectory_sha = canonical_attestation_sha256(
            {
                "previous_trajectory_sha256": trajectory_sha,
                "epoch": int(epoch),
                "optimizer_steps": int(steps),
                "step_chain_sha256": step_sha,
                "model_state_sha256": end_model_sha,
                "optimizer_state_sha256": end_optimizer_sha,
                "scaler_state_sha256": end_scaler_sha,
            }
        )
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": steps,
                "mean_total_loss": totals / steps,
                "learning_rate": current_lr,
                "max_preclip_gradient_norm": max_gradient_norm,
                "amp_autocast_entered": bool(amp),
                "loss_route": wiring["loss_route"],
                "event_dropout_probability": dropout_probability,
                "step_chain_sha256": step_sha,
                "model_state_sha256": end_model_sha,
                "optimizer_state_sha256": end_optimizer_sha,
                "scaler_state_sha256": end_scaler_sha,
                "trajectory_sha256": trajectory_sha,
            }
        )
        if epoch == capture_epoch:
            if checkpoint_path.exists():
                raise FileExistsError(f"capture checkpoint already exists: {checkpoint_path}")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": "revision0710.fig04_fig05.ablation_checkpoint.v8",
                "status": "CAPTURED_FIXED_EPOCH",
                "variant_id": variant_id,
                "epoch": epoch,
                "seed": seed,
                "model_config": asdict(model.model.config),
                "age_anchor": float(model.model.age_anchor.detach().cpu()),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_parameter_names": manifest["optimizer_parameter_names"],
                "initial_trainable_sha256": manifest["initial_trainable_sha256"],
                "initial_full_model_sha256": initial_full_model_sha,
                "initial_optimizer_sha256": initial_optimizer_sha,
                "training_trajectory_sha256_at_capture": trajectory_sha,
            }
            torch.save(payload, checkpoint_path)
            capture_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            capture_model_sha = end_model_sha
            capture_trajectory_sha = trajectory_sha
    if capture_sha is None or capture_model_sha is None or capture_trajectory_sha is None:
        raise RuntimeError("fixed capture checkpoint was not written")
    trajectory_attestation = {
        "schema_version": "revision0710.fig04_fig05.training_trajectory_attestation.v8",
        "status": "COMPLETE_INDEPENDENT_FULL_REPLAY_REQUIRED",
        "variant_id": variant_id,
        "seed": int(seed),
        "initial_full_model_sha256": initial_full_model_sha,
        "initial_trainable_sha256": manifest["initial_trainable_sha256"],
        "initial_optimizer_sha256": initial_optimizer_sha,
        "initial_scaler_sha256": initial_scaler_sha,
        "completed_epochs": int(epochs),
        "capture_epoch": int(capture_epoch),
        "capture_checkpoint_sha256": capture_sha,
        "capture_model_state_sha256": capture_model_sha,
        "capture_trajectory_sha256": capture_trajectory_sha,
        "final_model_state_sha256": history[-1]["model_state_sha256"],
        "final_optimizer_state_sha256": history[-1]["optimizer_state_sha256"],
        "final_scaler_state_sha256": history[-1]["scaler_state_sha256"],
        "final_trajectory_sha256": trajectory_sha,
        "epochs": [
            {
                key: row[key]
                for key in (
                    "epoch",
                    "optimizer_steps",
                    "step_chain_sha256",
                    "model_state_sha256",
                    "optimizer_state_sha256",
                    "scaler_state_sha256",
                    "trajectory_sha256",
                )
            }
            for row in history
        ],
    }
    return {
        "schema_version": "revision0710.fig04_fig05.ablation_training_result.v8",
        "status": "TRAINING_COMPLETE_CAPTURE_SEALED",
        "variant_id": variant_id,
        "seed": seed,
        "completed_epochs": epochs,
        "capture_epoch": capture_epoch,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": capture_sha,
        "optimizer_manifest": manifest,
        "history": history,
        "trajectory_attestation": trajectory_attestation,
        "forward_called": True,
        "backward_called": True,
        "gradient_clip_called": True,
        "scheduler_called": True,
        "amp_path_entered": bool(amp),
        "fresh_initialization": True,
        "resume_used": False,
    }
