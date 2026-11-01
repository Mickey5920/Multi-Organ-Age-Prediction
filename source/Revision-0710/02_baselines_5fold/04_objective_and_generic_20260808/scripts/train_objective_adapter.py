from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
BASELINE_DIR = EXPERIMENT_DIR.parent
REVISION_ROOT = BASELINE_DIR.parent
EHR_RUNNER_DIR = REVISION_ROOT / "03_ehr_age" / "02_runner"
EHR_MODEL_DIR = REVISION_ROOT / "03_ehr_age" / "01_model"
BASELINE_SCRIPT_DIR = BASELINE_DIR / "scripts"
for path in (HERE, EHR_RUNNER_DIR, EHR_MODEL_DIR, BASELINE_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_training import canonical_hash, sha256_file
from ehr_age_data import (
    Fold0DevelopmentData,
    JointLabels,
    JointVectorizedBatcher,
    load_restricted_compact_cache,
)
from ehr_age_model import ORGAN_SYSTEMS
from ehr_age_training import (
    _loss_config,
    _predict,
    _survival_multiplier,
    _train_epoch,
    configure_reproducibility,
    pooled_age_metrics,
    trainable_parameter_hash,
)
from experiment_models import CONTEMPORARY_MODELS, ProspectiveBaselineAdapter, build_contemporary_backbone
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "experiment.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "01_objective_matched"
DEFAULT_CACHE = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
OUTCOMES_PATH = (
    REVISION_ROOT / "01_data_rebuild" / "09_v2_labels" / "organ_outcomes_60m.parquet"
)
ROLE_AGE_ROOT = (
    REVISION_ROOT / "03_ehr_age" / "05_corrected_primary_v1" / "00_role_artifacts"
)
ROLE_DIRECTORIES = {
    "train": "01_train",
    "calibration": "02_calibration",
    "outer_test": "03_outer_test",
}
OUTCOME_COLUMNS = (
    "participant_id",
    "organ_system",
    "duration_months",
    "event",
    "at_risk",
    "strict_healthy_reference_60m",
)


def role_fold_ids(outer_fold: int) -> dict[str, tuple[int, ...]]:
    calibration = (outer_fold + 1) % 5
    train = tuple(
        fold for fold in range(5) if fold not in {outer_fold, calibration}
    )
    return {
        "train": train,
        "calibration": (calibration,),
        "outer_test": (outer_fold,),
    }


def _align_labels(
    frame: pd.DataFrame,
    participant_ids: Sequence[str],
    *,
    expected_role: str,
    expected_folds: Sequence[int],
) -> JointLabels:
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(frame["fold_id"].astype(int).unique()) != set(expected_folds):
        raise RuntimeError(f"{expected_role} labels contain unexpected folds")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError(f"{expected_role} labels contain duplicate rows")
    expected_index = pd.MultiIndex.from_product(
        [pd.Index([str(value) for value in participant_ids]), list(ORGAN_SYSTEMS)],
        names=["participant_id", "organ_system"],
    )
    indexed = frame.set_index(["participant_id", "organ_system"]).reindex(
        expected_index
    )
    if indexed[["event", "at_risk"]].isna().any().any():
        raise RuntimeError(f"{expected_role} labels are incomplete after alignment")
    missing_duration = indexed["duration_months"].isna()
    if (missing_duration & indexed["at_risk"].ne(0)).any():
        raise RuntimeError(f"{expected_role} has missing at-risk duration")
    if (missing_duration & indexed["event"].ne(0)).any():
        raise RuntimeError(f"{expected_role} has an event with missing duration")
    indexed["duration_months"] = indexed["duration_months"].fillna(0.0)
    n = len(participant_ids)
    strict = indexed["strict_healthy_reference_60m"].to_numpy(
        dtype=np.int8
    ).reshape(n, -1)
    if not np.all(strict == strict[:, :1]):
        raise RuntimeError(f"{expected_role} strict Health differs across organs")
    labels = JointLabels(
        strict_healthy=torch.from_numpy(strict[:, 0].astype(bool)),
        duration_months=torch.from_numpy(
            indexed["duration_months"].to_numpy(dtype=np.float32).reshape(n, -1)
        ),
        event=torch.from_numpy(
            indexed["event"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)
        ),
        at_risk=torch.from_numpy(
            indexed["at_risk"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)
        ),
        inner_role=tuple(expected_role for _ in range(n)),
    )
    labels.validate(n)
    return labels


def load_role_data(
    outcomes: pd.DataFrame, *, outer_fold: int, role: str
) -> Fold0DevelopmentData:
    expected_folds = role_fold_ids(outer_fold)[role]
    age_path = (
        ROLE_AGE_ROOT
        / f"outer_fold_{outer_fold}"
        / ROLE_DIRECTORIES[role]
        / "age_targets.parquet"
    )
    ages = pd.read_parquet(
        age_path, columns=["participant_id", "fold_id", "chronological_age"]
    )
    ages["participant_id"] = ages["participant_id"].astype(str)
    if ages["participant_id"].duplicated().any():
        raise RuntimeError(f"{role} age targets contain duplicate participants")
    if set(ages["fold_id"].astype(int).unique()) != set(expected_folds):
        raise RuntimeError(f"{role} age targets contain unexpected folds")
    cache = load_restricted_compact_cache(
        cache_dir=DEFAULT_CACHE,
        outer_fold=outer_fold,
        fold_ids=expected_folds,
        age_target_cache_path=age_path,
    )
    labels_frame = ages[["participant_id", "fold_id"]].merge(
        outcomes,
        on="participant_id",
        how="left",
        validate="one_to_many",
    )
    if len(labels_frame) != len(ages) * len(ORGAN_SYSTEMS):
        raise RuntimeError(f"{role} did not receive seven organ labels per participant")
    labels = _align_labels(
        labels_frame,
        cache.dataset.participant_ids,
        expected_role=role,
        expected_folds=expected_folds,
    )
    indices = torch.arange(len(cache.dataset), dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(
        cache=cache,
        labels=labels,
        train_indices=indices if role == "train" else empty,
        calibration_indices=indices if role == "calibration" else empty,
        screen_indices=indices if role == "outer_test" else empty,
        label_manifest={
            "status": "PASS",
            "outer_fold": outer_fold,
            "role": role,
            "fold_ids": list(expected_folds),
            "age_targets_path": str(age_path.resolve()),
            "age_targets_sha256": sha256_file(age_path),
            "outcomes_path": str(OUTCOMES_PATH.resolve()),
            "outcomes_sha256": sha256_file(OUTCOMES_PATH),
        },
    )


def _long_predictions(
    participant_ids: list[str],
    prediction: Any,
    *,
    outer_fold: int,
    split_role: str,
) -> pd.DataFrame:
    n = len(participant_ids)
    return pd.DataFrame(
        {
            "participant_id": np.repeat(np.asarray(participant_ids, dtype=object), len(ORGAN_SYSTEMS)),
            "organ_system": np.tile(np.asarray(ORGAN_SYSTEMS, dtype=object), n),
            "actual_age": np.repeat(prediction.chronological_age.astype(np.float64), len(ORGAN_SYSTEMS)),
            "predicted_age": prediction.predicted_age.astype(np.float32).reshape(-1),
            "outer_fold": np.full(n * len(ORGAN_SYSTEMS), outer_fold, dtype=np.int8),
            "split_role": split_role,
        }
    )


def _risk_predictions(
    participant_ids: list[str], prediction: Any, *, outer_fold: int, split_role: str
) -> pd.DataFrame:
    payload: dict[str, Any] = {
        "participant_id": participant_ids,
        "outer_fold": np.full(len(participant_ids), outer_fold, dtype=np.int8),
        "split_role": split_role,
    }
    for index, organ in enumerate(ORGAN_SYSTEMS):
        payload[f"risk_score_{organ}"] = prediction.risk_score[:, index].astype(np.float32)
    return pd.DataFrame(payload)


def _prediction_ids(data: Any, prediction: Any) -> list[str]:
    return [
        str(data.cache.dataset.participant_ids[int(index)])
        for index in prediction.participant_indices
    ]


def _scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_epochs: int, horizon_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(index: int) -> float:
        if index < warmup_epochs:
            return float(index + 1) / max(warmup_epochs, 1)
        progress = (index - warmup_epochs) / max(horizon_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def run(args: argparse.Namespace) -> Path:
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    specification = raw["objective_matched"]
    shared = raw["shared"]
    epochs = int(args.max_epochs or specification["fixed_epochs"])
    seed = int(shared["seed_base"]) + int(args.outer_fold)
    configure_reproducibility(
        seed,
        deterministic=bool(shared["deterministic"]),
        tf32=bool(shared["tf32"]),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("objective-matched training requires CUDA")

    outcomes = pd.read_parquet(OUTCOMES_PATH, columns=list(OUTCOME_COLUMNS))
    outcomes["participant_id"] = outcomes["participant_id"].astype(str)
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("organ outcomes contain duplicate participant-organ rows")
    data = {
        role: load_role_data(outcomes, outer_fold=args.outer_fold, role=role)
        for role in ("train", "calibration", "outer_test")
    }
    del outcomes
    participant_sets = {
        role: set(bundle.cache.dataset.participant_ids) for role, bundle in data.items()
    }
    if (
        participant_sets["train"] & participant_sets["calibration"]
        or participant_sets["train"] & participant_sets["outer_test"]
        or participant_sets["calibration"] & participant_sets["outer_test"]
    ):
        raise RuntimeError("training, validation, and test participants overlap")

    age_anchor = float(data["train"].cache.dataset.chronological_age.mean())
    backbone = build_contemporary_backbone(
        args.model,
        data["train"].cache.code_embedding,
        data["train"].cache.text_embedding,
        hidden_dim=int(specification["hidden_dim"]),
        num_heads=int(specification["num_heads"]),
        num_layers=int(specification["num_layers"]),
        dropout=float(specification["dropout"]),
        freeze_embeddings=True,
    )
    edges = specification["survival_interval_edges_months"]
    model = ProspectiveBaselineAdapter(
        backbone,
        hidden_dim=int(specification["hidden_dim"]),
        age_anchor=age_anchor,
        dropout=float(specification["dropout"]),
        risk_offset_cap_years=float(specification["risk_offset_cap_years"]),
        risk_temperature=float(specification["risk_temperature"]),
        survival_intervals=len(edges) - 1,
        baseline_log_rate_init=float(specification["baseline_log_rate_init"]),
    )
    initial_parameter_hash = trainable_parameter_hash(model)
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)

    method_name = f"{args.model}_prospective"
    materialized = {
        "schema_version": "revision0710.objective_matched_baseline.run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": method_name,
        "outer_fold": int(args.outer_fold),
        "fold_roles": {
            key: list(value) for key, value in role_fold_ids(args.outer_fold).items()
        },
        "participant_counts": {
            role: len(bundle.cache.dataset) for role, bundle in data.items()
        },
        "age_anchor_train_only": age_anchor,
        "seed": seed,
        "epochs": epochs,
        "max_batches": args.max_batches,
        "specification": specification,
        "shared": shared,
        "initial_trainable_parameter_sha256": initial_parameter_hash,
        "source_hashes": {
            "experiment_config": sha256_file(args.config),
            "experiment_models": sha256_file(HERE / "experiment_models.py"),
            "training_script": sha256_file(Path(__file__).resolve()),
            "outcomes": sha256_file(OUTCOMES_PATH),
        },
        "protocol": {
            "from_scratch": True,
            "future_outcomes_are_loss_only": True,
            "test_used_for_training_or_selection": False,
            "checkpoint_selection": f"prespecified fixed epoch {epochs}",
            "organ_query_cross_attention": False,
        },
    }
    config_hash = canonical_hash(materialized)
    output_root = args.output_root.resolve()
    run_dir = output_root / args.model / f"fold{args.outer_fold}" / config_hash[:12]
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "run_config.json").write_text(
        json.dumps(materialized, indent=2), encoding="utf-8"
    )

    batch_size = int(specification["batch_size"])
    batchers = {
        "train": JointVectorizedBatcher(
            data["train"],
            data["train"].train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        ),
        "validation": JointVectorizedBatcher(
            data["calibration"],
            data["calibration"].calibration_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
        "test": JointVectorizedBatcher(
            data["outer_test"],
            data["outer_test"].screen_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(specification["learning_rate"]),
        weight_decay=float(specification["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer,
        warmup_epochs=int(specification["warmup_epochs"]),
        horizon_epochs=int(specification["scheduler_horizon_epochs"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(shared["amp"])
    )
    interval_edges = torch.tensor(edges, dtype=torch.float32, device=device)
    loss_config = _loss_config(specification["loss"])
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        losses = _train_epoch(
            model=model,
            batcher=batchers["train"],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            interval_edges=interval_edges,
            loss_config=loss_config,
            survival_multiplier=_survival_multiplier(epoch, specification["loss"]),
            event_dropout_probability=float(specification["event_dropout_probability"]),
            gradient_clip_norm=float(specification["gradient_clip_norm"]),
            amp=bool(shared["amp"]),
            max_batches=args.max_batches,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - epoch_started,
            "losses": losses,
        }
        history.append(row)
        print(
            f"EPOCH model={args.model} fold={args.outer_fold} epoch={epoch}/{epochs} "
            f"loss={losses['total']:.6f} seconds={row['seconds']:.1f}",
            flush=True,
        )

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": method_name,
            "outer_fold": int(args.outer_fold),
            "epoch": epochs,
            "age_anchor_train_only": age_anchor,
            "config_hash": config_hash,
        },
        checkpoint_path,
    )
    (run_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    role_data = {"validation": data["calibration"], "test": data["outer_test"]}
    for role in ("validation", "test"):
        prediction = _predict(
            model=model,
            batcher=batchers[role],
            device=device,
            amp=bool(shared["amp"]),
            max_batches=args.max_batches,
        )
        participant_ids = _prediction_ids(role_data[role], prediction)
        long_frame = _long_predictions(
            participant_ids,
            prediction,
            outer_fold=args.outer_fold,
            split_role=role,
        )
        long_path = run_dir / f"{role}_predictions_long.parquet"
        long_frame.to_parquet(long_path, index=False)
        risk_path = run_dir / f"{role}_risk_scores.parquet"
        _risk_predictions(
            participant_ids,
            prediction,
            outer_fold=args.outer_fold,
            split_role=role,
        ).to_parquet(risk_path, index=False)
        artifacts[f"{role}_predictions_long_parquet"] = str(long_path)
        artifacts[f"{role}_risk_scores_parquet"] = str(risk_path)
        metrics[role] = {
            "all_participants": pooled_age_metrics(
                prediction.chronological_age, prediction.predicted_age
            ),
            "strict_healthy_reference_60m": pooled_age_metrics(
                prediction.chronological_age,
                prediction.predicted_age,
                participant_mask=prediction.strict_healthy,
            ),
            "complete_split": args.max_batches is None,
        }

    metrics_payload = {
        "status": "SMOKE_ONLY" if args.max_batches is not None else "COMPLETE",
        "model_name": method_name,
        "outer_fold": int(args.outer_fold),
        "fixed_epoch": epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        "metrics": metrics,
        "artifacts": artifacts,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    hash_paths = {
        "run_config": run_dir / "run_config.json",
        "checkpoint": checkpoint_path,
        "training_history": run_dir / "training_history.json",
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
    parser.add_argument("--model", required=True, choices=CONTEMPORARY_MODELS)
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-epochs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
