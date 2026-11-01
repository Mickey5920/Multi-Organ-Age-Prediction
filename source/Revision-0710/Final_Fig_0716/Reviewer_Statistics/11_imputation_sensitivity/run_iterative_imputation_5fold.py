from __future__ import annotations

"""Five-fold IterativeImputer sensitivity under the frozen Fig4 control contract."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_ROOT = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[4]
BASE_RUNNER_DIR = (
    REPO_ROOT
    / "Revision-0710"
    / "04_ablation_runs"
    / "fig04_fig05_corrected_primary_v1"
    / "00_runner"
)
V8_ROOT = (
    REPO_ROOT
    / "Revision-0710"
    / "06_figures_and_statistics"
    / "00_plan"
    / "fig04_fig05_ablation_prefreeze_v8"
)
CANONICAL_RUNNER_DIR = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "02_runner"
for location in (BASE_RUNNER_DIR, V8_ROOT, CANONICAL_RUNNER_DIR):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import run_direct_5fold as base  # noqa: E402
import v8_contract  # noqa: E402
import v8_data  # noqa: E402
import v8_model  # noqa: E402
import v8_training  # noqa: E402
from ehr_age_data import _load_selected_shards  # noqa: E402, SLF001


CACHE_DIR = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "04_cache_v1"
CONTROL_TASK_ROOT = (
    REPO_ROOT
    / "Revision-0710"
    / "04_ablation_runs"
    / "fig04_fig05_corrected_primary_v1"
    / "01_tasks"
)
TASK_ROOT = OUTPUT_ROOT / "01_tasks"
SOURCE_ROOT = OUTPUT_ROOT / "Source_Data"
FIGURE_ROOT = OUTPUT_ROOT / "Figures"
STATUS_PATH = OUTPUT_ROOT / "RUN_STATUS.json"
VARIANT_ID = "fig04_01_canonical_control"
SEED = 7210
IMPUTER_MAX_ITER = 5
IMPUTER_NEAREST_FEATURES = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # The frozen Fig4 runner records Infinity when the pre-clip gradient norm
    # overflows under AMP, while finite losses and parameters are retained.
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def role_folds(outer_fold: int) -> dict[str, tuple[int, ...]]:
    calibration = (outer_fold + 1) % 5
    train = tuple(fold for fold in range(5) if fold not in {outer_fold, calibration})
    return {
        "train": train,
        "calibration": (calibration,),
        "outer_test": (outer_fold,),
    }


def task_paths(outer_fold: int) -> dict[str, Path]:
    root = TASK_ROOT / f"outer_fold_{outer_fold}"
    return {
        "root": root,
        "train": root / "01_train",
        "calibration": root / "02_calibration",
        "outer": root / "03_outer_test",
        "imputer": root / "00_imputer",
        "complete": root / "TASK_COMPLETE.json",
        "failure": root / "FAILED.json",
    }


def load_raw_phenotypes(
    *, fold_ids: Iterable[int], expected_participant_ids: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load_selected_shards(CACHE_DIR, {int(value) for value in fold_ids})
    participant_ids = tuple(arrays["participant_id"].astype(str).tolist())
    if participant_ids != expected_participant_ids:
        raise RuntimeError("raw phenotype rows do not match the sealed role order")
    raw = np.asarray(arrays["phenotype_raw_zero"], dtype=np.float32)
    missing = np.asarray(arrays["phenotype_missing_mask"], dtype=bool)
    if raw.shape != (len(participant_ids), 78) or missing.shape != raw.shape:
        raise RuntimeError("raw phenotype shape changed")
    if not np.isfinite(raw).all():
        raise RuntimeError("raw zero-filled phenotype cache contains non-finite values")
    return raw, missing


class FoldIterativeTransform:
    def __init__(self, *, max_iter: int, n_nearest_features: int, seed: int) -> None:
        self.max_iter = int(max_iter)
        self.n_nearest_features = int(n_nearest_features)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.imputer_: IterativeImputer | None = None

    def _standardized_nan_matrix(
        self, raw: np.ndarray, missing: np.ndarray
    ) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("phenotype scaler has not been fit")
        matrix = raw.astype(np.float64, copy=True)
        matrix[missing] = np.nan
        matrix = (matrix - self.mean_[None, :]) / self.std_[None, :]
        return matrix

    def fit_transform(self, raw: np.ndarray, missing: np.ndarray) -> np.ndarray:
        observed = raw.astype(np.float64, copy=True)
        observed[missing] = np.nan
        mean = np.nanmean(observed, axis=0)
        std = np.nanstd(observed, axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise RuntimeError("every phenotype feature needs observed training values")
        std = np.where(std < 1e-6, 1.0, std)
        self.mean_ = mean
        self.std_ = std
        standardized = self._standardized_nan_matrix(raw, missing)
        lower = np.nanmin(standardized, axis=0)
        upper = np.nanmax(standardized, axis=0)
        if np.any(lower >= upper):
            raise RuntimeError("a phenotype feature lacks usable training variation")
        self.imputer_ = IterativeImputer(
            estimator=BayesianRidge(),
            missing_values=np.nan,
            sample_posterior=False,
            max_iter=self.max_iter,
            tol=1e-3,
            n_nearest_features=self.n_nearest_features,
            initial_strategy="median",
            imputation_order="ascending",
            skip_complete=True,
            min_value=lower,
            max_value=upper,
            verbose=1,
            random_state=self.seed,
        )
        result = self.imputer_.fit_transform(standardized)
        return self._validate_output(result, expected_rows=len(raw))

    def transform(self, raw: np.ndarray, missing: np.ndarray) -> np.ndarray:
        if self.imputer_ is None:
            raise RuntimeError("IterativeImputer has not been fit")
        result = self.imputer_.transform(self._standardized_nan_matrix(raw, missing))
        return self._validate_output(result, expected_rows=len(raw))

    @staticmethod
    def _validate_output(result: np.ndarray, *, expected_rows: int) -> np.ndarray:
        result = np.asarray(result, dtype=np.float32)
        if result.shape != (expected_rows, 78) or not np.isfinite(result).all():
            raise RuntimeError("IterativeImputer produced invalid phenotype values")
        return np.ascontiguousarray(result)

    def manifest(self, *, fit_seconds: float, missing: np.ndarray) -> dict[str, Any]:
        if self.imputer_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("cannot summarize an unfitted imputer")
        return {
            "schema_version": "revision0710.iterative_imputation_sensitivity.v1",
            "status": "FIT_ON_TRAIN_ROLE_ONLY",
            "method": "IterativeImputer(BayesianRidge)",
            "initial_strategy": "median",
            "max_iter": self.max_iter,
            "effective_n_iter": int(self.imputer_.n_iter_),
            "tol": 1e-3,
            "n_nearest_features": self.n_nearest_features,
            "sample_posterior": False,
            "imputation_order": "ascending",
            "skip_complete": True,
            "bounded_to_training_observed_range_after_standardization": True,
            "random_seed": self.seed,
            "fit_seconds": float(fit_seconds),
            "train_missing_cells": int(missing.sum()),
            "train_total_cells": int(missing.size),
            "train_missing_fraction": float(missing.mean()),
            "missingness_indicators_retained_as_model_inputs": True,
            "scaling": "training-observed mean/std before chained equations",
            "sklearn_version": sklearn.__version__,
        }


def replace_phenotype(
    bundle: v8_data.RoleBundle, phenotype: np.ndarray
) -> v8_data.RoleBundle:
    inputs = dict(bundle.inputs)
    inputs["phenotype"] = torch.from_numpy(phenotype)
    if inputs["phenotype_missing_mask"].shape != inputs["phenotype"].shape:
        raise RuntimeError("retained missingness indicators do not align with imputed inputs")
    return v8_data.RoleBundle(
        role=bundle.role,
        outer_fold=bundle.outer_fold,
        participant_ids=bundle.participant_ids,
        inputs=inputs,
        labels=bundle.labels,
    )


def environment_record() -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "UNAVAILABLE"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def load_role_with_raw(
    contract: dict[str, Any], *, outer_fold: int, role: str
) -> tuple[v8_data.RoleBundle, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    bundle, code, text = base.load_role(contract, outer_fold=outer_fold, role=role)
    raw, missing = load_raw_phenotypes(
        fold_ids=role_folds(outer_fold)[role],
        expected_participant_ids=bundle.participant_ids,
    )
    if not np.array_equal(
        missing, bundle.inputs["phenotype_missing_mask"].numpy().astype(bool)
    ):
        raise RuntimeError("raw missingness mask differs from the sealed role input")
    return bundle, code, text, raw, missing


def run_one(outer_fold: int, contract: dict[str, Any]) -> None:
    paths = task_paths(outer_fold)
    if paths["complete"].is_file():
        print(f"outer={outer_fold} already complete", flush=True)
        return
    checkpoint_path = paths["train"] / "fixed_epoch_0055.pt"
    recover_after_training = (
        paths["failure"].is_file()
        and checkpoint_path.is_file()
        and (paths["imputer"] / "iterative_imputer.joblib").is_file()
        and not (paths["calibration"] / "age_correction.json").exists()
    )
    if paths["root"].exists() and not recover_after_training:
        raise RuntimeError(
            f"incomplete task directory requires inspection before rerun: {paths['root']}"
        )
    if not recover_after_training:
        for key in ("train", "calibration", "outer", "imputer"):
            paths[key].mkdir(parents=True, exist_ok=False)
        write_json_atomic(
            paths["root"] / "TASK_SPEC.json",
            {
                "schema_version": "revision0710.iterative_imputation_task.v1",
                "started_at": utc_now(),
                "outer_fold": outer_fold,
                "roles": {key: list(value) for key, value in role_folds(outer_fold).items()},
                "variant_id": VARIANT_ID,
                "seed": SEED,
                "epochs": 60,
                "capture_epoch": 55,
                "batch_size": 2048,
                "learning_rate": 0.0002,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
                "warmup_epochs": 3,
                "scheduler_horizon_epochs": 60,
                "base_corrected_primary_contract": str(base.CONTRACT_PATH.resolve()),
                "base_corrected_primary_contract_sha256": sha256_file(base.CONTRACT_PATH),
                "base_runner": str((BASE_RUNNER_DIR / "run_direct_5fold.py").resolve()),
                "base_runner_sha256": sha256_file(BASE_RUNNER_DIR / "run_direct_5fold.py"),
                "script_sha256": sha256_file(SCRIPT_PATH),
                "environment": environment_record(),
            },
        )
    try:
        if recover_after_training:
            print(f"outer={outer_fold} recovering after completed training", flush=True)
            transform = joblib.load(paths["imputer"] / "iterative_imputer.joblib")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            tensors = checkpoint.get("model_state_dict", {})
            if (
                int(checkpoint.get("epoch", -1)) != 55
                or checkpoint.get("variant_id") != VARIANT_ID
                or not tensors
                or any(not torch.isfinite(value).all() for value in tensors.values())
            ):
                raise RuntimeError("recovery checkpoint failed the finite E55 validation")
            resolved_failure = paths["root"] / "RESOLVED_FAILURE_AFTER_TRAINING.json"
            if not resolved_failure.exists():
                os.replace(paths["failure"], resolved_failure)
            write_json_atomic(
                paths["train"] / "TRAINING_COMPLETE_RECOVERED.json",
                {
                    "schema_version": "revision0710.iterative_imputation_training_recovery.v1",
                    "status": "E60_COMPLETED_E55_CHECKPOINT_FINITE",
                    "reason": "post-training JSON serialization rejected Infinity in pre-clip gradient norm",
                    "reference_control_contains_same_infinity_record": True,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_epoch": 55,
                    "finite_model_tensors": len(tensors),
                    "continued_training_repeated": False,
                },
            )
            write_json_atomic(paths["train"] / "MODEL_CONFIG.json", asdict(base.production_config()))
            del checkpoint, tensors
        else:
            print(f"outer={outer_fold} loading sealed train role and raw phenotypes", flush=True)
            train_bundle, train_code, train_text, train_raw, train_missing = load_role_with_raw(
                contract, outer_fold=outer_fold, role="train"
            )
            transform = FoldIterativeTransform(
                max_iter=IMPUTER_MAX_ITER,
                n_nearest_features=IMPUTER_NEAREST_FEATURES,
                seed=SEED,
            )
            started = time.perf_counter()
            train_phenotype = transform.fit_transform(train_raw, train_missing)
            fit_seconds = time.perf_counter() - started
            imputer_manifest = transform.manifest(
                fit_seconds=fit_seconds, missing=train_missing
            )
            np.savez_compressed(
                paths["imputer"] / "training_observed_scaler.npz",
                mean=transform.mean_.astype(np.float64),
                std=transform.std_.astype(np.float64),
            )
            joblib.dump(transform, paths["imputer"] / "iterative_imputer.joblib", compress=3)
            imputer_manifest.update(
                {
                    "outer_fold": outer_fold,
                    "train_participants": len(train_bundle.participant_ids),
                    "train_participant_ids_sha256": v8_contract.canonical_json_sha256(
                        {"participant_ids": sorted(train_bundle.participant_ids)}
                    ),
                    "scaler_sha256": sha256_file(
                        paths["imputer"] / "training_observed_scaler.npz"
                    ),
                    "imputer_sha256": sha256_file(
                        paths["imputer"] / "iterative_imputer.joblib"
                    ),
                }
            )
            write_json_atomic(paths["imputer"] / "IMPUTER_MANIFEST.json", imputer_manifest)
            train_bundle = replace_phenotype(train_bundle, train_phenotype)
            del train_raw, train_missing, train_phenotype
            gc.collect()

            print(
                f"outer={outer_fold} imputer_fit_seconds={fit_seconds:.1f} "
                f"effective_iter={imputer_manifest['effective_n_iter']}",
                flush=True,
            )
            v8_training.configure_reproducibility(SEED, deterministic=True, tf32=True)
            model = v8_model.build_production_model(
                VARIANT_ID,
                train_code,
                train_text,
                age_anchor=float(train_bundle.labels["chronological_age"].float().mean()),
                config=base.production_config(),
            )
            train_result = v8_training.train_fresh(
                model,
                VARIANT_ID,
                v8_data.RoleBatcher(train_bundle, batch_size=2048, seed=SEED, shuffle=True),
                device=torch.device("cuda"),
                seed=SEED,
                epochs=60,
                capture_epoch=55,
                learning_rate=0.0002,
                weight_decay=0.0001,
                gradient_clip_norm=1.0,
                warmup_epochs=3,
                scheduler_horizon_epochs=60,
                amp=True,
                checkpoint_path=checkpoint_path,
            )
            write_json_atomic(paths["train"] / "TRAINING_COMPLETE.json", train_result)
            write_json_atomic(paths["train"] / "MODEL_CONFIG.json", asdict(base.production_config()))
            del train_bundle, train_code, train_text, model
            gc.collect()
            torch.cuda.empty_cache()

        print(f"outer={outer_fold} calibration inference", flush=True)
        calibration_bundle, calibration_code, calibration_text, raw, missing = load_role_with_raw(
            contract, outer_fold=outer_fold, role="calibration"
        )
        calibration_bundle = replace_phenotype(
            calibration_bundle, transform.transform(raw, missing)
        )
        del raw, missing
        calibration_model = base.restore_capture_model(
            checkpoint_path,
            variant_id=VARIANT_ID,
            code=calibration_code,
            text=calibration_text,
        ).to(torch.device("cuda"))
        calibration_predictions = v8_data.prediction_frame(
            calibration_model,
            calibration_bundle,
            variant_id=VARIANT_ID,
            run_id=f"iterative_imputation__outer{outer_fold}",
            evaluation_epoch=55,
            device=torch.device("cuda"),
            inference_batch_size=2048,
        )
        calibration_predictions["split_role"] = "calibration"
        correction = v8_data.fit_age_correction(calibration_predictions)
        corrected_calibration = v8_data.apply_age_correction(
            calibration_predictions, correction
        )
        nuisance = v8_data.fit_cox_nuisance(corrected_calibration, penalizer=1e-4)
        corrected_calibration.to_parquet(
            paths["calibration"] / "calibration_predictions_corrected.parquet",
            index=False,
        )
        write_json_atomic(paths["calibration"] / "age_correction.json", correction)
        write_json_atomic(paths["calibration"] / "cox_nuisance.json", nuisance)
        del (
            calibration_bundle,
            calibration_code,
            calibration_text,
            calibration_model,
            calibration_predictions,
            corrected_calibration,
        )
        gc.collect()
        torch.cuda.empty_cache()

        print(f"outer={outer_fold} outer-test inference and scoring", flush=True)
        outer_bundle, outer_code, outer_text, raw, missing = load_role_with_raw(
            contract, outer_fold=outer_fold, role="outer_test"
        )
        outer_bundle = replace_phenotype(outer_bundle, transform.transform(raw, missing))
        del raw, missing, transform
        outer_model = base.restore_capture_model(
            checkpoint_path,
            variant_id=VARIANT_ID,
            code=outer_code,
            text=outer_text,
        ).to(torch.device("cuda"))
        outer_predictions = v8_data.prediction_frame(
            outer_model,
            outer_bundle,
            variant_id=VARIANT_ID,
            run_id=f"iterative_imputation__outer{outer_fold}",
            evaluation_epoch=55,
            device=torch.device("cuda"),
            inference_batch_size=2048,
        )
        outer_predictions["split_role"] = "outer_test"
        corrected_outer = v8_data.apply_age_correction(outer_predictions, correction)
        fold_metrics, organ_metrics = v8_data.score_outer_metrics(corrected_outer, nuisance)
        fold_record = {
            "method": "iterative_imputation",
            "outer_fold": outer_fold,
            "evaluation_epoch": 55,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            **fold_metrics,
        }
        corrected_outer.to_parquet(
            paths["outer"] / "prediction_rows.parquet", index=False
        )
        pd.DataFrame([fold_record]).to_csv(
            paths["outer"] / "fold_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(organ_metrics).assign(
            method="iterative_imputation", outer_fold=outer_fold
        ).to_csv(paths["outer"] / "organ_metrics.csv", index=False, lineterminator="\n")
        write_json_atomic(
            paths["complete"],
            {
                "schema_version": "revision0710.iterative_imputation_task_complete.v1",
                "status": "COMPLETE",
                "completed_at": utc_now(),
                "outer_fold": outer_fold,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "fold_metrics": fold_record,
                "imputer_manifest": str(
                    (paths["imputer"] / "IMPUTER_MANIFEST.json").resolve()
                ),
            },
        )
        del outer_bundle, outer_code, outer_text, outer_model, outer_predictions, corrected_outer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"outer={outer_fold} COMPLETE metrics={fold_record}", flush=True)
    except Exception as exc:
        write_json_atomic(
            paths["failure"],
            {
                "schema_version": "revision0710.iterative_imputation_failure.v1",
                "status": "FAILED",
                "failed_at": utc_now(),
                "outer_fold": outer_fold,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def control_task_for_fold(outer_fold: int) -> Path:
    matches = sorted(
        CONTROL_TASK_ROOT.glob(
            f"*_fig04_01_canonical_control__outer{outer_fold}"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one completed control task for outer {outer_fold}")
    return matches[0]


def pooled_strict_metrics(paths: list[Path]) -> dict[str, float | int]:
    values: list[pd.DataFrame] = []
    for path in paths:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(
                path,
                columns=[
                    "participant_id",
                    "actual_age",
                    "predicted_age",
                    "strict_healthy_reference_60m",
                ],
            )
        else:
            frame = pd.read_csv(
                path,
                usecols=[
                    "participant_id",
                    "actual_age",
                    "predicted_age",
                    "strict_healthy_reference_60m",
                ],
            )
        values.append(frame.loc[frame["strict_healthy_reference_60m"].eq(1)])
    strict = pd.concat(values, ignore_index=True)
    actual = strict["actual_age"].to_numpy(float)
    predicted = strict["predicted_age"].to_numpy(float)
    residual = predicted - actual
    denominator = np.square(actual - actual.mean()).sum()
    return {
        "strict_n_participants": int(strict["participant_id"].nunique()),
        "strict_n_participant_organ_rows": int(len(strict)),
        "strict_mae": float(np.abs(residual).mean()),
        "strict_pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "strict_r2": float(1.0 - np.square(residual).sum() / denominator),
    }


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, math.nan, math.nan
    half = float(st.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - half, mean + half


def render_summary(paired: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("raw_strict_mae", "Strict MAE (years)", False),
        ("raw_strict_pearson", "Strict Pearson r", True),
        ("raw_strict_r2", "Strict R²", True),
        ("age_sex_harrell_macro_delta_c", "Age+sex ΔC-index", True),
    ]
    colors = {"median_control": "#6B7280", "iterative_imputation": "#0F766E"}
    labels = {"median_control": "Median control", "iterative_imputation": "Iterative"}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0), constrained_layout=True)
    for axis, (metric, title, higher_better) in zip(axes, metrics, strict=True):
        wide = paired.pivot(index="outer_fold", columns="method", values=metric)
        for _, row in wide.iterrows():
            axis.plot([0, 1], [row["median_control"], row["iterative_imputation"]], color="#D1D5DB", lw=0.8, zorder=1)
        for x, method in enumerate(("median_control", "iterative_imputation")):
            values = wide[method].to_numpy(float)
            mean, low, high = mean_ci(values)
            axis.scatter(np.full(len(values), x), values, s=18, color=colors[method], alpha=0.75, zorder=2)
            axis.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="D", ms=5, color=colors[method], capsize=3, lw=1.2, zorder=3)
        delta = wide["iterative_imputation"] - wide["median_control"]
        direction = "higher" if higher_better else "lower"
        axis.set_title(f"{title}\nΔ={delta.mean():+.4f} ({direction} better)", fontsize=9)
        axis.set_xticks([0, 1], [labels["median_control"], labels["iterative_imputation"]], rotation=20, ha="right")
        axis.grid(axis="y", color="#E5E7EB", lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Alternative phenotype imputation sensitivity (five paired outer folds)", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=400, bbox_inches="tight")
    plt.close(fig)


def summarize() -> None:
    if not all(task_paths(fold)["complete"].is_file() for fold in range(5)):
        print("summary deferred until all five folds complete", flush=True)
        return
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    control_rows = []
    iterative_rows = []
    control_predictions = []
    iterative_predictions = []
    for fold in range(5):
        control_task = control_task_for_fold(fold)
        control = pd.read_csv(control_task / "03_outer_test" / "fold_metrics.csv").iloc[0].to_dict()
        control["method"] = "median_control"
        control_rows.append(control)
        control_predictions.append(control_task / "03_outer_test" / "prediction_rows.csv")
        iterative = pd.read_csv(task_paths(fold)["outer"] / "fold_metrics.csv").iloc[0].to_dict()
        iterative["method"] = "iterative_imputation"
        iterative_rows.append(iterative)
        iterative_predictions.append(task_paths(fold)["outer"] / "prediction_rows.parquet")
    paired = pd.DataFrame(control_rows + iterative_rows)
    metric_columns = [
        "raw_strict_mae",
        "raw_strict_pearson",
        "raw_strict_r2",
        "age_sex_harrell_macro_delta_c",
    ]
    paired[["method", "outer_fold", *metric_columns]].sort_values(
        ["outer_fold", "method"]
    ).to_csv(SOURCE_ROOT / "paired_fold_metrics.csv", index=False, lineterminator="\n")
    wide = paired.pivot(index="outer_fold", columns="method", values=metric_columns)
    paired_delta_rows = []
    for metric in metric_columns:
        values = (
            wide[metric]["iterative_imputation"]
            - wide[metric]["median_control"]
        ).to_numpy(float)
        mean, low, high = mean_ci(values)
        paired_delta_rows.append(
            {
                "metric": metric,
                "estimand": "mean_outer_fold_paired_difference",
                "comparison": "iterative_imputation_minus_median_control",
                "n_outer_folds": len(values),
                "mean_difference": mean,
                "fold_t95_low": low,
                "fold_t95_high": high,
                "direction_favorable": (
                    "negative" if metric == "raw_strict_mae" else "positive"
                ),
            }
        )
    pd.DataFrame(paired_delta_rows).to_csv(
        SOURCE_ROOT / "paired_fold_delta_summary.csv",
        index=False,
        lineterminator="\n",
    )

    aggregate_rows = []
    for method, predictions, rows in (
        ("median_control", control_predictions, control_rows),
        ("iterative_imputation", iterative_predictions, iterative_rows),
    ):
        pooled = pooled_strict_metrics(predictions)
        delta_values = np.asarray(
            [float(row["age_sex_harrell_macro_delta_c"]) for row in rows]
        )
        delta_mean, delta_low, delta_high = mean_ci(delta_values)
        aggregate_rows.append(
            {
                "method": method,
                **pooled,
                "age_sex_harrell_macro_delta_c_mean": delta_mean,
                "age_sex_harrell_macro_delta_c_fold_t95_low": delta_low,
                "age_sex_harrell_macro_delta_c_fold_t95_high": delta_high,
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(SOURCE_ROOT / "aggregate_metrics.csv", index=False, lineterminator="\n")
    by_method = aggregate.set_index("method")
    delta_row = {
        "comparison": "iterative_imputation_minus_median_control",
        "strict_mae": float(by_method.loc["iterative_imputation", "strict_mae"] - by_method.loc["median_control", "strict_mae"]),
        "strict_pearson": float(by_method.loc["iterative_imputation", "strict_pearson"] - by_method.loc["median_control", "strict_pearson"]),
        "strict_r2": float(by_method.loc["iterative_imputation", "strict_r2"] - by_method.loc["median_control", "strict_r2"]),
        "age_sex_harrell_macro_delta_c": float(by_method.loc["iterative_imputation", "age_sex_harrell_macro_delta_c_mean"] - by_method.loc["median_control", "age_sex_harrell_macro_delta_c_mean"]),
    }
    pd.DataFrame([delta_row]).to_csv(
        SOURCE_ROOT / "aggregate_delta_vs_control.csv", index=False, lineterminator="\n"
    )
    render_summary(paired, FIGURE_ROOT / "Supplementary_Imputation_Sensitivity")
    write_json_atomic(
        OUTPUT_ROOT / "QA_REPORT.json",
        {
            "schema_version": "revision0710.iterative_imputation_summary_qa.v1",
            "status": "PASS",
            "completed_folds": 5,
            "paired_control": "corrected-primary Fig4 canonical control",
            "aggregate_metrics": aggregate_rows,
            "delta_vs_control": delta_row,
            "source_data": {
                path.name: sha256_file(path)
                for path in sorted(SOURCE_ROOT.glob("*.csv"))
            },
            "figures": {
                path.name: sha256_file(path)
                for path in sorted(FIGURE_ROOT.glob("Supplementary_Imputation_Sensitivity.*"))
            },
        },
    )


def update_status(active_fold: int | None, message: str) -> None:
    states = {}
    for fold in range(5):
        paths = task_paths(fold)
        if paths["complete"].is_file():
            state = "COMPLETE"
        elif paths["failure"].is_file():
            state = "FAILED"
        elif paths["root"].exists():
            state = "IN_PROGRESS"
        else:
            state = "PENDING"
        states[str(fold)] = state
    write_json_atomic(
        STATUS_PATH,
        {
            "schema_version": "revision0710.iterative_imputation_run_status.v1",
            "updated_at": utc_now(),
            "active_fold": active_fold,
            "message": message,
            "fold_states": states,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must be in 0..4")
    if args.summarize_only:
        summarize()
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the five-fold imputation sensitivity")
    contract, _ = base.launch_preflight()
    update_status(None, "preflight passed")
    for fold in args.folds:
        update_status(fold, "imputation, training, calibration, and outer scoring")
        run_one(fold, contract)
        update_status(None, f"outer fold {fold} complete")
    summarize()
    update_status(None, "requested folds complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
