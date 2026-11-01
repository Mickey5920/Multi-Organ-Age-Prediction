from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


BASELINE_DIR = Path(__file__).resolve().parents[1]
COMMON_DIR = BASELINE_DIR / "common"
if str(COMMON_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(COMMON_DIR))

from baseline_models import MODEL_NAMES, ORGAN_SYSTEMS, build_age_model, model_inputs_from_batch
from compact_cache_dataset import CompactCacheBundle, load_compact_cache
from masked_backbones import EHRFORMER_REFERENCE_COMMIT, EHRFORMER_REFERENCE_REPOSITORY


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    outer_fold: int
    cache_dir: str = str(BASELINE_DIR.parent / "01_data_rebuild" / "04_cache_v1")
    output_root: str = str(BASELINE_DIR / "runs")
    batch_size: int = 4096
    max_epochs: int = 100
    patience: int = 12
    seed_base: int = 7100
    seed: int | None = None
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    smooth_l1_beta: float = 2.0
    num_workers: int = 0
    deterministic: bool = True
    tf32: bool = True
    amp: bool = True
    freeze_event_embeddings: bool = True
    selection_metric: str = "validation_all_population_macro_mae"
    max_batches: int | None = None

    @property
    def resolved_seed(self) -> int:
        return self.seed_base + self.outer_fold if self.seed is None else self.seed

    def validate(self) -> None:
        assert_no_prohibited_training_keys(asdict(self))
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {MODEL_NAMES}")
        if self.outer_fold not in range(5):
            raise ValueError("outer_fold must be in {0,1,2,3,4}")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("training counts must be positive")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive when supplied")
        if self.selection_metric != "validation_all_population_macro_mae":
            raise ValueError("checkpoint selection metric is frozen by contract")


def fold_roles(outer_fold: int) -> dict[str, tuple[int, ...]]:
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be in {0,1,2,3,4}")
    validation = (outer_fold + 1) % 5
    train = tuple(fold for fold in range(5) if fold not in {outer_fold, validation})
    return {"train": train, "validation": (validation,), "test": (outer_fold,)}


def assert_no_prohibited_training_keys(payload: dict[str, Any]) -> None:
    prohibited = ("warm_start", "teacher", "distill", "resume_from", "pretrained_checkpoint")

    def visit(value: Any, prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if any(token in normalized for token in prohibited):
                raise ValueError(f"prohibited training source key: {prefix}{key}")
            visit(child, f"{prefix}{key}.")

    visit(payload)


def _pearson(target: np.ndarray, predicted: np.ndarray) -> float | None:
    centered_target = target - target.mean()
    centered_prediction = predicted - predicted.mean()
    denominator = float(np.sqrt(np.sum(centered_target**2) * np.sum(centered_prediction**2)))
    if denominator == 0:
        return None
    return float(np.sum(centered_target * centered_prediction) / denominator)


def age_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or prediction.shape != (len(target), len(ORGAN_SYSTEMS)):
        raise ValueError("age metric arrays must be [n] and [n,7]")
    denominator = float(np.sum((target - target.mean()) ** 2))
    organs: dict[str, dict[str, float | None]] = {}
    for index, organ in enumerate(ORGAN_SYSTEMS):
        error = prediction[:, index] - target
        organs[organ] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "pearson": _pearson(target, prediction[:, index]),
            "r2": None if denominator == 0 else float(1.0 - np.sum(error**2) / denominator),
        }
    return {
        "n": int(len(target)),
        "macro_mae": float(np.mean([item["mae"] for item in organs.values()])),
        "macro_rmse": float(np.mean([item["rmse"] for item in organs.values()])),
        "macro_pearson": _mean_optional([item["pearson"] for item in organs.values()]),
        "macro_r2": _mean_optional([item["r2"] for item in organs.values()]),
        "organs": organs,
    }


def _mean_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def prediction_frame(
    participant_ids: tuple[str, ...] | list[str],
    *,
    fold_id: int,
    target: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    if len(participant_ids) != len(target) or prediction.shape != (
        len(participant_ids),
        len(ORGAN_SYSTEMS),
    ):
        raise ValueError("prediction export arrays are not participant-aligned")
    payload: dict[str, Any] = {
        "participant_id": list(participant_ids),
        "fold_id": np.full(len(participant_ids), int(fold_id), dtype=np.int8),
        "chronological_age": np.asarray(target, dtype=np.float32),
    }
    for index, organ in enumerate(ORGAN_SYSTEMS):
        payload[f"predicted_age_{organ}"] = np.asarray(prediction[:, index], dtype=np.float32)
    return pd.DataFrame(payload)


def long_prediction_frame(
    participant_ids: tuple[str, ...] | list[str],
    *,
    outer_fold: int,
    split_role: str,
    target: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    if split_role not in {"validation", "test"}:
        raise ValueError("split_role must be validation or test")
    wide = prediction_frame(
        participant_ids,
        fold_id=outer_fold,
        target=target,
        prediction=prediction,
    )
    long = wide.melt(
        id_vars=["participant_id", "chronological_age"],
        value_vars=[f"predicted_age_{organ}" for organ in ORGAN_SYSTEMS],
        var_name="organ_system",
        value_name="predicted_age",
    )
    long["organ_system"] = long["organ_system"].str.removeprefix("predicted_age_")
    long = long.rename(columns={"chronological_age": "actual_age"})
    long["outer_fold"] = np.int8(outer_fold)
    long["split_role"] = split_role
    return long[
        [
            "participant_id",
            "organ_system",
            "actual_age",
            "predicted_age",
            "outer_fold",
            "split_role",
        ]
    ]


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure_reproducibility(seed: int, *, deterministic: bool, tf32: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _verify_cache_gates(cache_dir: Path) -> dict[str, str]:
    gate_paths = {
        "cache_hard_gate": cache_dir / "cache_hard_gate.json",
        "leakage_gate": cache_dir.parent / "05_leakage_audit" / "leakage_gate.json",
    }
    hashes: dict[str, str] = {}
    for name, path in gate_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"{name} is not PASS: {path}")
        hashes[name] = sha256_file(path)
    hashes["batch_manifest"] = sha256_file(cache_dir / "batch_manifest.json")
    hashes["embedding_tables"] = sha256_file(cache_dir / "embedding_tables.pt")
    return hashes


def _indices_for_folds(bundle: CompactCacheBundle, folds: tuple[int, ...]) -> list[int]:
    wanted = torch.zeros_like(bundle.dataset.fold_id, dtype=torch.bool)
    for fold in folds:
        wanted |= bundle.dataset.fold_id.eq(fold)
    return torch.nonzero(wanted, as_tuple=False).flatten().tolist()


class VectorizedBatcher:
    """Batch aligned tensors with one vectorized index operation per field."""

    def __init__(
        self,
        bundle: CompactCacheBundle,
        indices: list[int],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = bundle.dataset
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.iteration = 0

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.iteration)
            order = torch.randperm(len(self.indices), generator=generator)
            positions = self.indices.index_select(0, order)
        else:
            positions = self.indices
        self.iteration += 1
        for start in range(0, len(positions), self.batch_size):
            selected = positions[start : start + self.batch_size]
            inputs = {
                key: value.index_select(0, selected)
                for key, value in self.dataset.inputs.items()
            }
            target = self.dataset.chronological_age.index_select(0, selected)
            yield inputs, target


def _loader(
    bundle: CompactCacheBundle,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> VectorizedBatcher:
    if num_workers != 0:
        raise ValueError("vectorized in-memory batching requires num_workers=0")
    return VectorizedBatcher(
        bundle,
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def _move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    selected = model_inputs_from_batch(inputs)
    return {key: value.to(device, non_blocking=True) for key, value in selected.items()}


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _train_epoch(
    model: nn.Module,
    loader: VectorizedBatcher,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch_index, (inputs, target) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp and device.type == "cuda"):
            prediction = model(**_move_inputs(inputs, device))
            expanded_target = target.unsqueeze(1).expand_as(prediction)
            loss = criterion(prediction, expanded_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(target)
        total_n += len(target)
    if total_n == 0:
        raise RuntimeError("training loader produced no batches")
    return total_loss / total_n


def _predict(
    model: nn.Module,
    loader: VectorizedBatcher,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    target_chunks: list[np.ndarray] = []
    prediction_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, (inputs, target) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            with _autocast(device, amp and device.type == "cuda"):
                prediction = model(**_move_inputs(inputs, device))
            target_chunks.append(target.numpy())
            prediction_chunks.append(prediction.float().cpu().numpy())
    if not target_chunks:
        raise RuntimeError("evaluation loader produced no batches")
    return np.concatenate(target_chunks), np.concatenate(prediction_chunks)


def _evaluated_indices(indices: list[int], observed_n: int) -> list[int]:
    if observed_n > len(indices):
        raise ValueError("evaluation produced more rows than the split contains")
    return indices[:observed_n]


def run_training(config: TrainingConfig) -> Path:
    run_started = time.perf_counter()
    config.validate()
    seed = config.resolved_seed
    configure_reproducibility(seed, deterministic=config.deterministic, tf32=config.tf32)
    cache_dir = Path(config.cache_dir).resolve()
    source_hashes = _verify_cache_gates(cache_dir)
    code_paths = {
        "baseline_training": Path(__file__).resolve(),
        "baseline_models": COMMON_DIR / "baseline_models.py",
        "masked_backbones": COMMON_DIR / "masked_backbones.py",
        "compact_cache_dataset": COMMON_DIR / "compact_cache_dataset.py",
    }
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    roles = fold_roles(config.outer_fold)
    bundle = load_compact_cache(cache_dir, outer_fold=config.outer_fold, fold_ids=range(5))
    indices = {role: _indices_for_folds(bundle, folds) for role, folds in roles.items()}
    train_age = bundle.dataset.chronological_age[indices["train"]]
    age_anchor = float(train_age.mean())

    protocol: dict[str, Any] = {
        "checkpoint_selection": "validation all-population macro MAE only",
        "test_touched_during_training": False,
        "warm_start": False,
        "teacher": False,
        "distillation": False,
    }
    if config.model_name.startswith("ehrformer_style"):
        protocol["architecture_reference"] = {
            "repository": EHRFORMER_REFERENCE_REPOSITORY,
            "commit": EHRFORMER_REFERENCE_COMMIT,
            "scope": "architecture only; common EHR-AGE inputs, outputs, objective, folds, and evaluation",
        }
    materialized = asdict(config) | {
        "cache_dir": str(cache_dir),
        "output_root": str(Path(config.output_root).resolve()),
        "fold_roles": {key: list(value) for key, value in roles.items()},
        "age_anchor_train_only": age_anchor,
        "resolved_seed": seed,
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "protocol": protocol,
    }
    config_hash = canonical_hash(materialized)
    run_dir = (
        Path(config.output_root).resolve()
        / config.model_name
        / f"fold{config.outer_fold}"
        / config_hash[:12]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(materialized, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_age_model(
        config.model_name,
        bundle.code_embedding,
        bundle.text_embedding,
        age_anchor=age_anchor,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        freeze_embeddings=config.freeze_event_embeddings,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"START model={config.model_name} fold={config.outer_fold} seed={seed} "
        f"device={device} train={len(indices['train']):,} val={len(indices['validation']):,} "
        f"test={len(indices['test']):,} params={parameter_count:,} trainable={trainable_parameter_count:,}",
        flush=True,
    )
    loaders = {
        role: _loader(
            bundle,
            role_indices,
            batch_size=config.batch_size,
            shuffle=role == "train",
            seed=seed,
            num_workers=config.num_workers,
            device=device,
        )
        for role, role_indices in indices.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.SmoothL1Loss(beta=config.smooth_l1_beta)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    checkpoint_path = run_dir / "best_checkpoint.pt"
    best_mae = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            criterion,
            device,
            amp=config.amp,
            max_batches=config.max_batches,
        )
        validation_target, validation_prediction = _predict(
            model,
            loaders["validation"],
            device,
            amp=config.amp,
            max_batches=config.max_batches,
        )
        validation_metrics = age_metrics(validation_target, validation_prediction)
        selection_value = float(validation_metrics["macro_mae"])
        improved = selection_value < best_mae
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_macro_mae": selection_value,
                "checkpoint_improved": improved,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        if improved:
            best_mae = selection_value
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config.model_name,
                    "outer_fold": config.outer_fold,
                    "epoch": epoch,
                    "validation_macro_mae": selection_value,
                    "config_hash": config_hash,
                    "age_anchor_train_only": age_anchor,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        )
        print(
            f"EPOCH model={config.model_name} fold={config.outer_fold} epoch={epoch} "
            f"train_loss={train_loss:.6f} val_macro_mae={selection_value:.6f} "
            f"best={best_mae:.6f}@{best_epoch} stale={stale_epochs}/{config.patience} "
            f"seconds={history[-1]['epoch_seconds']:.1f} peak_allocated_mb={peak_allocated_mb:.0f}",
            flush=True,
        )
        if stale_epochs >= config.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for role in ("validation", "test"):
        target, prediction = _predict(
            model,
            loaders[role],
            device,
            amp=config.amp,
            max_batches=config.max_batches,
        )
        evaluated = _evaluated_indices(indices[role], len(target))
        participant_ids = [bundle.dataset.participant_ids[index] for index in evaluated]
        frame = prediction_frame(
            participant_ids,
            fold_id=roles[role][0],
            target=target,
            prediction=prediction,
        )
        parquet_path = run_dir / f"{role}_predictions.parquet"
        csv_path = run_dir / f"{role}_predictions.csv"
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(csv_path, index=False)
        artifacts[f"{role}_predictions_parquet"] = str(parquet_path)
        artifacts[f"{role}_predictions_csv"] = str(csv_path)
        long_frame = long_prediction_frame(
            participant_ids,
            outer_fold=config.outer_fold,
            split_role=role,
            target=target,
            prediction=prediction,
        )
        long_parquet_path = run_dir / f"{role}_predictions_long.parquet"
        long_csv_path = run_dir / f"{role}_predictions_long.csv.gz"
        long_frame.to_parquet(long_parquet_path, index=False)
        long_frame.to_csv(long_csv_path, index=False, compression="gzip")
        artifacts[f"{role}_predictions_long_parquet"] = str(long_parquet_path)
        artifacts[f"{role}_predictions_long_csv_gz"] = str(long_csv_path)
        split_metrics[role] = age_metrics(target, prediction) | {
            "fold_id": roles[role][0],
            "complete_split": len(target) == len(indices[role]),
            "expected_n": len(indices[role]),
        }
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    metrics_path = run_dir / "metrics.json"
    metrics = {
        "status": "SMOKE_ONLY" if config.max_batches is not None else "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": config.model_name,
        "outer_fold": config.outer_fold,
        "selection_metric": config.selection_metric,
        "best_epoch": best_epoch,
        "best_validation_macro_mae": best_mae,
        "age_anchor_train_only": age_anchor,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "elapsed_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "peak_cuda_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "split_counts": {key: len(value) for key, value in indices.items()},
        "metrics": split_metrics,
        "artifacts": artifacts,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    hash_paths = {
        "run_config": config_path,
        "best_checkpoint": checkpoint_path,
        "metrics": metrics_path,
        "training_history": run_dir / "training_history.csv",
        **{key: Path(value) for key, value in artifacts.items()},
    }
    artifact_hashes = {key: sha256_file(path) for key, path in hash_paths.items()}
    artifact_hashes["canonical_config_hash"] = config_hash
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2), encoding="utf-8"
    )
    print(
        f"COMPLETE model={config.model_name} fold={config.outer_fold} "
        f"best_val_macro_mae={best_mae:.6f} best_epoch={best_epoch} run_dir={run_dir}",
        flush=True,
    )
    return run_dir
