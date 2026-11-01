from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
BASELINE_DIR = EXPERIMENT_DIR.parent
REVISION_ROOT = BASELINE_DIR.parent
COMMON_DIR = BASELINE_DIR / "common"
BASELINE_SCRIPT_DIR = BASELINE_DIR / "scripts"
EHR_MODEL_DIR = REVISION_ROOT / "03_ehr_age" / "01_model"
EHR_RUNNER_DIR = REVISION_ROOT / "03_ehr_age" / "02_runner"
for path in (HERE, COMMON_DIR, BASELINE_SCRIPT_DIR, EHR_MODEL_DIR, EHR_RUNNER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_training import (
    _indices_for_folds,
    _loader,
    _predict,
    canonical_hash,
    configure_reproducibility,
    fold_roles,
    sha256_file,
)
from compact_cache_dataset import load_compact_cache
from ehr_age_model import ORGAN_SYSTEMS
from ehr_age_training import pooled_age_metrics
from experiment_models import GENERIC_NEURAL_MODELS, GenericOrganAgeModel, build_generic_backbone


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "experiment.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "02_generic_age_only"


def _long_predictions(
    participant_ids: list[str],
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    outer_fold: int,
    split_role: str,
) -> pd.DataFrame:
    n = len(participant_ids)
    return pd.DataFrame(
        {
            "participant_id": np.repeat(np.asarray(participant_ids, dtype=object), len(ORGAN_SYSTEMS)),
            "organ_system": np.tile(np.asarray(ORGAN_SYSTEMS, dtype=object), n),
            "actual_age": np.repeat(target.astype(np.float64), len(ORGAN_SYSTEMS)),
            "predicted_age": prediction.astype(np.float32).reshape(-1),
            "outer_fold": np.full(n * len(ORGAN_SYSTEMS), outer_fold, dtype=np.int8),
            "split_role": split_role,
        }
    )


def _train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp: bool,
    max_batches: int | None,
) -> float:
    model.train()
    total = 0.0
    observed = 0
    for batch_index, (inputs, target) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        target = target.to(device, non_blocking=True)
        moved = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            prediction = model(**moved)
            loss = criterion(prediction, target.unsqueeze(1).expand_as(prediction))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach()) * len(target)
        observed += len(target)
    if observed == 0:
        raise RuntimeError("training loader produced no batches")
    return total / observed


def run(args: argparse.Namespace) -> Path:
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    specification = raw["generic_age_only"]
    shared = raw["shared"]
    seed = int(shared["seed_base"]) + int(args.outer_fold)
    configure_reproducibility(
        seed,
        deterministic=bool(shared["deterministic"]),
        tf32=bool(shared["tf32"]),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("generic neural baseline training requires CUDA")

    cache_dir = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
    bundle = load_compact_cache(
        cache_dir, outer_fold=args.outer_fold, fold_ids=range(5)
    )
    roles = fold_roles(args.outer_fold)
    indices = {
        role: _indices_for_folds(bundle, folds) for role, folds in roles.items()
    }
    age_anchor = float(bundle.dataset.chronological_age[indices["train"]].mean())
    maximum_events = int(bundle.dataset.inputs["event_valid_mask"].shape[1])
    backbone = build_generic_backbone(
        args.model,
        bundle.code_embedding,
        bundle.text_embedding,
        hidden_dim=int(specification["hidden_dim"]),
        num_heads=int(specification["num_heads"]),
        num_layers=int(specification["num_layers"]),
        dropout=float(specification["dropout"]),
        maximum_events=maximum_events,
        freeze_embeddings=True,
    )
    model = GenericOrganAgeModel(
        backbone,
        hidden_dim=int(specification["hidden_dim"]),
        age_anchor=age_anchor,
        dropout=float(specification["dropout"]),
    )
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)

    materialized = {
        "schema_version": "revision0710.generic_age_baseline.run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model,
        "outer_fold": int(args.outer_fold),
        "fold_roles": {key: list(value) for key, value in roles.items()},
        "participant_counts": {key: len(value) for key, value in indices.items()},
        "age_anchor_train_only": age_anchor,
        "maximum_events": maximum_events,
        "seed": seed,
        "max_batches": args.max_batches,
        "specification": specification,
        "shared": shared,
        "source_hashes": {
            "experiment_config": sha256_file(args.config),
            "experiment_models": sha256_file(HERE / "experiment_models.py"),
            "training_script": sha256_file(Path(__file__).resolve()),
            "cache_hard_gate": sha256_file(cache_dir / "cache_hard_gate.json"),
            "batch_manifest": sha256_file(cache_dir / "batch_manifest.json"),
        },
        "protocol": {
            "from_scratch": True,
            "chronological_age_target_only": True,
            "test_used_for_training_or_selection": False,
            "checkpoint_selection": "validation all-participant macro MAE",
        },
    }
    config_hash = canonical_hash(materialized)
    run_dir = (
        args.output_root.resolve()
        / args.model
        / f"fold{args.outer_fold}"
        / config_hash[:12]
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(materialized, indent=2), encoding="utf-8")

    batch_size = int(specification["batch_size"])
    loaders = {
        role: _loader(
            bundle,
            role_indices,
            batch_size=batch_size,
            shuffle=role == "train",
            seed=seed,
            num_workers=0,
            device=device,
        )
        for role, role_indices in indices.items()
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(specification["learning_rate"]),
        weight_decay=float(specification["weight_decay"]),
    )
    criterion = nn.SmoothL1Loss(beta=float(specification["smooth_l1_beta"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(shared["amp"]))
    maximum_epochs = int(args.max_epochs or specification["maximum_epochs"])
    patience = int(specification["early_stopping_patience"])
    best_mae = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "best_checkpoint.pt"
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            criterion,
            device,
            amp=bool(shared["amp"]),
            max_batches=args.max_batches,
        )
        target, prediction = _predict(
            model,
            loaders["validation"],
            device,
            amp=bool(shared["amp"]),
            max_batches=args.max_batches,
        )
        validation_mae = float(
            np.abs(prediction - target[:, None]).mean()
        )
        improved = validation_mae < best_mae
        if improved:
            best_mae = validation_mae
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "outer_fold": int(args.outer_fold),
                    "epoch": epoch,
                    "validation_macro_mae": validation_mae,
                    "age_anchor_train_only": age_anchor,
                    "config_hash": config_hash,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_mae": validation_mae,
            "checkpoint_improved": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"EPOCH model={args.model} fold={args.outer_fold} epoch={epoch} "
            f"train_loss={train_loss:.6f} validation_mae={validation_mae:.6f} "
            f"best={best_mae:.6f}@{best_epoch} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    split_metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for role in ("validation", "test"):
        target, prediction = _predict(
            model,
            loaders[role],
            device,
            amp=bool(shared["amp"]),
            max_batches=args.max_batches,
        )
        evaluated_indices = indices[role][: len(target)]
        participant_ids = [
            str(bundle.dataset.participant_ids[index]) for index in evaluated_indices
        ]
        frame = _long_predictions(
            participant_ids,
            target,
            prediction,
            outer_fold=args.outer_fold,
            split_role=role,
        )
        output_path = run_dir / f"{role}_predictions_long.parquet"
        frame.to_parquet(output_path, index=False)
        artifacts[f"{role}_predictions_long_parquet"] = str(output_path)
        split_metrics[role] = pooled_age_metrics(target, prediction) | {
            "complete_split": args.max_batches is None,
            "expected_participants": len(indices[role]),
        }

    metrics_payload = {
        "status": "SMOKE_ONLY" if args.max_batches is not None else "COMPLETE",
        "model_name": args.model,
        "outer_fold": int(args.outer_fold),
        "best_epoch": best_epoch,
        "best_validation_macro_mae": best_mae,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        "metrics": split_metrics,
        "artifacts": artifacts,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    hash_paths = {
        "run_config": config_path,
        "best_checkpoint": checkpoint_path,
        "training_history": run_dir / "training_history.csv",
        "metrics": metrics_path,
        **{key: Path(value) for key, value in artifacts.items()},
    }
    hashes = {key: sha256_file(path) for key, path in hash_paths.items()}
    hashes["canonical_config_hash"] = config_hash
    (run_dir / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE run_dir={run_dir}", flush=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=GENERIC_NEURAL_MODELS)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-epochs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
