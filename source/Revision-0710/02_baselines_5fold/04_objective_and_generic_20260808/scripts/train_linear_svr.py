from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import SGDRegressor


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
BASELINE_DIR = EXPERIMENT_DIR.parent
REVISION_ROOT = BASELINE_DIR.parent
COMMON_DIR = BASELINE_DIR / "common"
BASELINE_SCRIPT_DIR = BASELINE_DIR / "scripts"
EHR_MODEL_DIR = REVISION_ROOT / "03_ehr_age" / "01_model"
EHR_RUNNER_DIR = REVISION_ROOT / "03_ehr_age" / "02_runner"
for path in (COMMON_DIR, BASELINE_SCRIPT_DIR, EHR_MODEL_DIR, EHR_RUNNER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_training import canonical_hash, fold_roles, sha256_file
from compact_cache_dataset import CompactCacheBundle, load_compact_cache
from ehr_age_model import ORGAN_SYSTEMS
from ehr_age_training import pooled_age_metrics


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "experiment.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "02_generic_age_only"


def _indices_for_folds(bundle: CompactCacheBundle, folds: tuple[int, ...]) -> np.ndarray:
    fold_id = bundle.dataset.fold_id.numpy()
    return np.flatnonzero(np.isin(fold_id, np.asarray(folds, dtype=np.int64)))


def _batches(
    indices: np.ndarray, batch_size: int, *, shuffle: bool, seed: int
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def _feature_matrix(
    bundle: CompactCacheBundle, indices: np.ndarray, bins: int
) -> sp.csr_matrix:
    selected = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    inputs = bundle.dataset.inputs
    phenotype = inputs["phenotype"].index_select(0, selected).numpy().astype(np.float32)
    missing = (
        inputs["phenotype_missing_mask"]
        .index_select(0, selected)
        .numpy()
        .astype(np.float32)
    )
    sex = inputs["sex"].index_select(0, selected).numpy().astype(np.float32)[:, None]
    valid = inputs["event_valid_mask"].index_select(0, selected).numpy().astype(bool)
    event_type = inputs["event_type_id"].index_select(0, selected).numpy().astype(np.int64)
    relative_days = (
        inputs["event_relative_days"].index_select(0, selected).numpy().astype(np.float32)
    )
    code_id = inputs["event_code_id"].index_select(0, selected).numpy().astype(np.int64)
    text_id = inputs["event_text_id"].index_select(0, selected).numpy().astype(np.int64)

    count = valid.sum(axis=1).astype(np.float32)
    denominator = np.maximum(count, 1.0)
    days_before = np.maximum(-relative_days, 0.0) * valid
    mean_days = days_before.sum(axis=1) / denominator
    centered = (days_before - mean_days[:, None]) * valid
    std_days = np.sqrt(np.square(centered).sum(axis=1) / denominator)
    maximum_days = np.where(valid, days_before, 0.0).max(axis=1)
    minimum_days = np.where(valid, days_before, np.inf).min(axis=1)
    minimum_days[~np.isfinite(minimum_days)] = 0.0
    type_one = ((event_type == 1) & valid).sum(axis=1).astype(np.float32) / denominator
    type_two = ((event_type == 2) & valid).sum(axis=1).astype(np.float32) / denominator
    stats = np.column_stack(
        (
            count / valid.shape[1],
            type_one,
            type_two,
            mean_days / 36525.0,
            std_days / 36525.0,
            minimum_days / 36525.0,
            maximum_days / 36525.0,
        )
    ).astype(np.float32)
    dense = sp.csr_matrix(np.concatenate((phenotype, missing, sex, stats), axis=1))

    row_grid = np.broadcast_to(np.arange(len(indices))[:, None], valid.shape)
    normalizer = 1.0 / np.sqrt(denominator)
    code_mask = valid & (code_id != 0)
    text_mask = valid & (text_id != 0)
    code_rows = row_grid[code_mask]
    text_rows = row_grid[text_mask]
    code_columns = np.mod(code_id[code_mask] * 2654435761, bins).astype(np.int64)
    text_columns = np.mod(text_id[text_mask] * 2246822519, bins).astype(np.int64)
    code_values = normalizer[code_rows].astype(np.float32)
    text_values = normalizer[text_rows].astype(np.float32)
    code = sp.csr_matrix(
        (code_values, (code_rows, code_columns)), shape=(len(indices), bins)
    )
    text = sp.csr_matrix(
        (text_values, (text_rows, text_columns)), shape=(len(indices), bins)
    )
    return sp.hstack((dense, code, text), format="csr", dtype=np.float32)


def _predict(
    model: SGDRegressor,
    bundle: CompactCacheBundle,
    indices: np.ndarray,
    *,
    bins: int,
    batch_size: int,
    target_mean: float,
    target_std: float,
    maximum_batches: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    evaluated: list[np.ndarray] = []
    for batch_index, batch in enumerate(
        _batches(indices, batch_size, shuffle=False, seed=0)
    ):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        features = _feature_matrix(bundle, batch, bins)
        prediction = model.predict(features).astype(np.float32) * target_std + target_mean
        predictions.append(prediction)
        targets.append(bundle.dataset.chronological_age[batch].numpy())
        evaluated.append(batch)
    if not predictions:
        raise RuntimeError("linear SVR prediction produced no batches")
    return np.concatenate(targets), np.concatenate(predictions), np.concatenate(evaluated)


def _long_predictions(
    participant_ids: list[str],
    target: np.ndarray,
    scalar_prediction: np.ndarray,
    *,
    outer_fold: int,
    split_role: str,
) -> pd.DataFrame:
    prediction = np.repeat(scalar_prediction[:, None], len(ORGAN_SYSTEMS), axis=1)
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


def run(args: argparse.Namespace) -> Path:
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    specification = raw["generic_age_only"]
    shared = raw["shared"]
    seed = int(shared["seed_base"]) + int(args.outer_fold)
    np.random.seed(seed)
    cache_dir = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
    bundle = load_compact_cache(
        cache_dir, outer_fold=args.outer_fold, fold_ids=range(5)
    )
    roles = fold_roles(args.outer_fold)
    indices = {
        role: _indices_for_folds(bundle, folds) for role, folds in roles.items()
    }
    train_target = bundle.dataset.chronological_age[indices["train"]].numpy()
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    if not math.isfinite(target_std) or target_std <= 0.0:
        raise RuntimeError("training chronological age has invalid variance")

    bins = int(specification["linear_svr_hash_bins_per_channel"])
    batch_size = int(specification["linear_svr_batch_size"])
    maximum_epochs = int(args.max_epochs or specification["linear_svr_epochs"])
    patience = int(specification["linear_svr_patience"])
    materialized = {
        "schema_version": "revision0710.linear_svr_age_baseline.run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "linear_svr",
        "outer_fold": int(args.outer_fold),
        "fold_roles": {key: list(value) for key, value in roles.items()},
        "participant_counts": {key: len(value) for key, value in indices.items()},
        "seed": seed,
        "target_mean_train_only": target_mean,
        "target_std_train_only": target_std,
        "maximum_epochs": maximum_epochs,
        "max_batches": args.max_batches,
        "feature_contract": {
            "phenotype_and_missingness": 156,
            "sex": 1,
            "event_summary": 7,
            "hashed_code_bins": bins,
            "hashed_text_bins": bins,
            "hashing": "deterministic multiplicative modulo hashing",
        },
        "estimator": {
            "class": "sklearn.linear_model.SGDRegressor",
            "loss": "epsilon_insensitive",
            "epsilon": 0.1,
            "penalty": "l2",
            "alpha": float(specification["linear_svr_alpha"]),
            "learning_rate": "invscaling",
            "eta0": float(specification["linear_svr_eta0"]),
            "average": True,
        },
        "source_hashes": {
            "experiment_config": sha256_file(args.config),
            "training_script": sha256_file(Path(__file__).resolve()),
            "cache_hard_gate": sha256_file(cache_dir / "cache_hard_gate.json"),
            "batch_manifest": sha256_file(cache_dir / "batch_manifest.json"),
        },
        "protocol": {
            "from_scratch": True,
            "chronological_age_target_only": True,
            "test_used_for_training_or_selection": False,
            "checkpoint_selection": "validation MAE",
            "seven_organ_predictions": "shared scalar age estimate repeated across organs",
        },
    }
    config_hash = canonical_hash(materialized)
    run_dir = (
        args.output_root.resolve()
        / "linear_svr"
        / f"fold{args.outer_fold}"
        / config_hash[:12]
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(materialized, indent=2), encoding="utf-8")

    model = SGDRegressor(
        loss="epsilon_insensitive",
        epsilon=0.1,
        penalty="l2",
        alpha=float(specification["linear_svr_alpha"]),
        learning_rate="invscaling",
        eta0=float(specification["linear_svr_eta0"]),
        random_state=seed,
        max_iter=1,
        tol=None,
        average=True,
    )
    best_model: SGDRegressor | None = None
    best_mae = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        epoch_started = time.perf_counter()
        train_errors: list[float] = []
        for batch_index, batch in enumerate(
            _batches(
                indices["train"],
                batch_size,
                shuffle=True,
                seed=seed + 1000 * epoch,
            )
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            features = _feature_matrix(bundle, batch, bins)
            target = (
                bundle.dataset.chronological_age[batch].numpy() - target_mean
            ) / target_std
            model.partial_fit(features, target)
            fitted = model.predict(features).astype(np.float32) * target_std + target_mean
            train_errors.append(
                float(
                    np.abs(fitted - bundle.dataset.chronological_age[batch].numpy()).mean()
                )
            )
        validation_target, validation_prediction, _ = _predict(
            model,
            bundle,
            indices["validation"],
            bins=bins,
            batch_size=batch_size,
            target_mean=target_mean,
            target_std=target_std,
            maximum_batches=args.max_batches,
        )
        validation_mae = float(np.abs(validation_prediction - validation_target).mean())
        improved = validation_mae < best_mae
        if improved:
            best_mae = validation_mae
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "train_batch_mae": float(np.mean(train_errors)),
            "validation_mae": validation_mae,
            "checkpoint_improved": improved,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"EPOCH model=linear_svr fold={args.outer_fold} epoch={epoch} "
            f"train_mae={row['train_batch_mae']:.6f} validation_mae={validation_mae:.6f} "
            f"best={best_mae:.6f}@{best_epoch} stale={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            break
    if best_model is None:
        raise RuntimeError("linear SVR did not produce a valid checkpoint")
    model = best_model
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    coefficient_path = run_dir / "model_coefficients.npz"
    np.savez_compressed(
        coefficient_path,
        coef=np.asarray(model.coef_, dtype=np.float32),
        intercept=np.asarray(model.intercept_, dtype=np.float32),
        target_mean=np.asarray([target_mean], dtype=np.float32),
        target_std=np.asarray([target_std], dtype=np.float32),
    )

    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for role in ("validation", "test"):
        target, scalar_prediction, evaluated = _predict(
            model,
            bundle,
            indices[role],
            bins=bins,
            batch_size=batch_size,
            target_mean=target_mean,
            target_std=target_std,
            maximum_batches=args.max_batches,
        )
        participant_ids = [
            str(bundle.dataset.participant_ids[index]) for index in evaluated
        ]
        frame = _long_predictions(
            participant_ids,
            target,
            scalar_prediction,
            outer_fold=args.outer_fold,
            split_role=role,
        )
        output_path = run_dir / f"{role}_predictions_long.parquet"
        frame.to_parquet(output_path, index=False)
        artifacts[f"{role}_predictions_long_parquet"] = str(output_path)
        repeated = np.repeat(
            scalar_prediction[:, None], len(ORGAN_SYSTEMS), axis=1
        )
        metrics[role] = pooled_age_metrics(target, repeated) | {
            "complete_split": args.max_batches is None,
            "expected_participants": len(indices[role]),
        }

    metrics_payload = {
        "status": "SMOKE_ONLY" if args.max_batches is not None else "COMPLETE",
        "model_name": "linear_svr",
        "outer_fold": int(args.outer_fold),
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    hash_paths = {
        "run_config": config_path,
        "model_coefficients": coefficient_path,
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
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(5))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-epochs", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
