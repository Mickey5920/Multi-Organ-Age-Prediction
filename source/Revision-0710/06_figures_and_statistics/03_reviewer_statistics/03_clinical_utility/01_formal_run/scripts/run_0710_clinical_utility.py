#!/usr/bin/env python
"""Run validation-to-test clinical utility metrics for frozen Revision-0710."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)
THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
REPORT_TIMES = (12.0, 24.0, 60.0)
PENALIZER = 1e-4

A_AGE_SEX = (
    "chronological_age",
    "sex_male",
)
B_CONTEXT_PRIMARY = A_AGE_SEX + (
    "smoking_current",
    "smoking_missing",
    "education_higher",
    "education_missing",
    "townsend",
    "townsend_missing",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "employment_other",
    "employment_missing",
    "sleep_duration",
    "sleep_missing",
)
C_CONTEXT_DENSITY_PRIMARY = B_CONTEXT_PRIMARY + (
    "phenotype_missing_fraction",
    "log1p_preindex_icd_count",
    "log1p_preindex_gp_count",
    "has_preindex_icd",
    "has_preindex_gp",
    "prevalent_organ_count",
)
TIERS = {
    "A_age_sex": A_AGE_SEX,
    "B_context_primary": B_CONTEXT_PRIMARY,
    "C_context_density_primary": C_CONTEXT_DENSITY_PRIMARY,
}
DEFAULT_TIERS = ("A_age_sex", "C_context_density_primary")
AUGMENTATIONS = ("base", "target_gap", "all_gaps")
PREDICTION_USECOLS = (
    "participant_id",
    "outer_fold",
    "split_role",
    "organ_system",
    "corrected_age_gap",
    "duration_months",
    "event",
    "at_risk",
)
TASK_DIRS = {
    0: "01_fig04_01_canonical_control__outer0",
    1: "02_fig04_01_canonical_control__outer1",
    2: "03_fig04_01_canonical_control__outer2",
    3: "04_fig04_01_canonical_control__outer3",
    4: "05_fig04_01_canonical_control__outer4",
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root")


ROOT = repo_root()
METRIC_ROOT = ROOT / "Revision-0710" / "05_downstream" / "06_metric_v2"
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from common.clinical_utility_metrics import (  # noqa: E402
    fixed_bin_calibration,
    harrell_c_index,
    ipcw_binary_state,
    ipcw_decision_curve,
    ipcw_logistic_calibration,
    ipcw_reclassification,
)
from common.cox_scoring import fit_lifelines_cox_scores  # noqa: E402
from common.survival_metrics import (  # noqa: E402
    build_metric_time_grid,
    evaluate_censoring_aware_metrics,
    flatten_primary_metrics,
)


FORMAL_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PARQUET = (
    METRIC_ROOT / "01_contextual_covariates" / "participant_context_covariates.parquet"
)
CONTEXT_PICKLE = FORMAL_ROOT / "02_input_cache" / "participant_context_covariates.pkl"
TASK_ROOT = (
    ROOT
    / "Revision-0710"
    / "04_ablation_runs"
    / "fig04_fig05_corrected_primary_v1"
    / "01_tasks"
)
OUTPUT_ROOT = FORMAL_ROOT / "03_outputs"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_context(args: argparse.Namespace) -> None:
    output = args.context_pickle.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.context_parquet)
    required = {"participant_id", "fold_id", *C_CONTEXT_DENSITY_PRIMARY}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"contextual covariates missing columns: {missing}")
    frame["participant_id"] = frame["participant_id"].astype(str)
    if len(frame) != 456_773 or frame["participant_id"].duplicated().any():
        raise RuntimeError("contextual covariate row/key contract failed")
    frame.to_pickle(output)
    write_json(
        output.parent / "INPUT_CACHE_MANIFEST.json",
        {
            "schema_version": "revision0710.clinical_utility.input_cache.v1",
            "status": "PASS",
            "source": {
                "path": str(args.context_parquet.resolve()),
                "bytes": args.context_parquet.stat().st_size,
                "sha256": sha256_file(args.context_parquet),
            },
            "output": {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "rows": int(len(frame)),
                "unique_participants": int(frame["participant_id"].nunique()),
                "columns": list(frame.columns),
            },
            "legacy_revision0626_read": False,
        },
    )
    print(f"Prepared {output} with {len(frame):,} participants", flush=True)


def prediction_paths(outer_fold: int) -> tuple[Path, Path]:
    task = TASK_ROOT / TASK_DIRS[outer_fold]
    return (
        task / "02_calibration" / "calibration_predictions_corrected.csv",
        task / "03_outer_test" / "prediction_rows.csv",
    )


def load_prediction_frame(path: Path, outer_fold: int, expected_role: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(PREDICTION_USECOLS),
        dtype={"participant_id": str, "organ_system": str, "split_role": str},
        low_memory=False,
    )
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)) != {outer_fold}:
        raise RuntimeError(f"outer-fold mismatch in {path}")
    observed_role = set(frame["split_role"].astype(str))
    if observed_role != {expected_role}:
        raise RuntimeError(f"split-role mismatch in {path}: {sorted(observed_role)}")
    if set(frame["organ_system"]) != set(ORGANS):
        raise RuntimeError(f"organ coverage mismatch in {path}")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError(f"duplicate participant-organ rows in {path}")
    counts = frame.groupby("participant_id")["organ_system"].nunique()
    if not counts.eq(len(ORGANS)).all():
        raise RuntimeError(f"incomplete seven-organ predictions in {path}")
    return frame


def gap_wide(frame: pd.DataFrame) -> pd.DataFrame:
    wide = frame.pivot(
        index="participant_id", columns="organ_system", values="corrected_age_gap"
    )
    wide = wide.reindex(columns=list(ORGANS))
    wide.columns = [f"gap__{organ}" for organ in wide.columns]
    return wide.reset_index()


def organ_outcomes(frame: pd.DataFrame, organ: str) -> pd.DataFrame:
    selected = frame.loc[
        frame["organ_system"].eq(organ) & pd.to_numeric(frame["at_risk"]).eq(1),
        ["participant_id", "duration_months", "event"],
    ].copy()
    selected = selected.rename(columns={"duration_months": "duration"})
    selected["duration"] = pd.to_numeric(selected["duration"], errors="coerce")
    selected["event"] = pd.to_numeric(selected["event"], errors="coerce").astype(np.int8)
    if selected["participant_id"].duplicated().any():
        raise RuntimeError(f"duplicate at-risk rows for {organ}")
    if selected["duration"].isna().any() or selected["duration"].le(0.0).any():
        raise RuntimeError(f"invalid at-risk duration for {organ}")
    return selected


def build_cell_frames(
    prediction_fit: pd.DataFrame,
    prediction_score: pd.DataFrame,
    context: pd.DataFrame,
    fit_gaps: pd.DataFrame,
    score_gaps: pd.DataFrame,
    *,
    outer_fold: int,
    organ: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    fit = organ_outcomes(prediction_fit, organ).merge(
        fit_gaps, on="participant_id", how="left", validate="one_to_one"
    ).merge(context, on="participant_id", how="left", validate="one_to_one")
    score = organ_outcomes(prediction_score, organ).merge(
        score_gaps, on="participant_id", how="left", validate="one_to_one"
    ).merge(context, on="participant_id", how="left", validate="one_to_one")
    if fit.isna().all(axis=1).any() or score.isna().all(axis=1).any():
        raise RuntimeError(f"empty joined row in fold {outer_fold} {organ}")
    expected_fit_fold = (outer_fold + 1) % 5
    if set(pd.to_numeric(fit["fold_id"], errors="raise").astype(int)) != {expected_fit_fold}:
        raise RuntimeError(f"validation fold mismatch for outer {outer_fold} {organ}")
    if set(pd.to_numeric(score["fold_id"], errors="raise").astype(int)) != {outer_fold}:
        raise RuntimeError(f"test fold mismatch for outer {outer_fold} {organ}")
    model_columns = list(C_CONTEXT_DENSITY_PRIMARY) + [f"gap__{item}" for item in ORGANS]
    medians: dict[str, float] = {}
    for column in model_columns:
        fit_values = pd.to_numeric(fit[column], errors="coerce")
        score_values = pd.to_numeric(score[column], errors="coerce")
        median = float(fit_values.median())
        if not np.isfinite(median):
            raise RuntimeError(f"no validation median for {column}: fold {outer_fold} {organ}")
        fit[column] = fit_values.fillna(median).astype(np.float64)
        score[column] = score_values.fillna(median).astype(np.float64)
        if not np.isfinite(fit[column]).all() or not np.isfinite(score[column]).all():
            raise RuntimeError(f"non-finite model value for {column}: fold {outer_fold} {organ}")
        if fit_values.isna().any() or score_values.isna().any():
            medians[column] = median
    administrative_boundary = float(min(fit["duration"].max(), score["duration"].max()))
    if not np.isclose(administrative_boundary, 60.0, atol=0.1):
        raise RuntimeError(
            f"unexpected administrative boundary {administrative_boundary}: fold {outer_fold} {organ}"
        )
    for frame in (fit, score):
        at_boundary = np.isclose(
            frame["duration"].to_numpy(dtype=np.float64),
            administrative_boundary,
            rtol=0.0,
            atol=1e-6,
        )
        boundary_event = at_boundary & frame["event"].eq(1).to_numpy()
        frame["right_open_boundary_event_reclassified"] = boundary_event.astype(np.int8)
        frame.loc[boundary_event, "event"] = 0
    fit["split_role"] = "validation"
    score["split_role"] = "test"
    fit["outer_fold"] = outer_fold
    score["outer_fold"] = outer_fold
    return fit, score, medians


def covariates_for(tier: str, augmentation: str, organ: str) -> tuple[str, ...]:
    base = tuple(TIERS[tier])
    if augmentation == "base":
        return base
    if augmentation == "target_gap":
        return base + (f"gap__{organ}",)
    if augmentation == "all_gaps":
        return base + tuple(f"gap__{item}" for item in ORGANS)
    raise ValueError(augmentation)


def probability_at(bundle, time_grid, report_time: float, *, fit: bool) -> np.ndarray:
    effective = time_grid.effective_for(report_time)
    index = next(
        idx for idx, value in enumerate(bundle.times) if np.isclose(value, effective, atol=1e-10)
    )
    survival = bundle.fit_survival_probability if fit else bundle.survival_probability
    return 1.0 - survival[:, index]


def model_id(tier: str, augmentation: str) -> str:
    return f"{tier}__{augmentation}"


def run_fold(
    outer_fold: int,
    context: pd.DataFrame,
    tiers: Sequence[str],
    *,
    output_dir: Path,
    penalizer: float,
) -> None:
    fold_dir = output_dir / "folds" / f"outer_fold_{outer_fold}"
    complete_path = fold_dir / "COMPLETE.json"
    if complete_path.exists():
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if payload.get("tiers") == list(tiers) and float(payload.get("penalizer")) == penalizer:
            print(f"Fold {outer_fold}: already complete", flush=True)
            return
        raise RuntimeError(f"Fold {outer_fold} has an incompatible completion receipt")
    fold_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = fold_dir / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    calibration_path, test_path = prediction_paths(outer_fold)
    started = time.time()
    fit_predictions = load_prediction_frame(calibration_path, outer_fold, "calibration")
    score_predictions = load_prediction_frame(test_path, outer_fold, "outer_test")
    fit_gap_wide = gap_wide(fit_predictions)
    score_gap_wide = gap_wide(score_predictions)
    model_rows: list[dict] = []
    incremental_rows: list[dict] = []
    calibration_rows: list[dict] = []
    dca_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    imputation_rows: list[dict] = []

    for organ_index, organ in enumerate(ORGANS, start=1):
        cell_started = time.time()
        fit, score, medians = build_cell_frames(
            fit_predictions,
            score_predictions,
            context,
            fit_gap_wide,
            score_gap_wide,
            outer_fold=outer_fold,
            organ=organ,
        )
        for column, median in medians.items():
            imputation_rows.append(
                {"outer_fold": outer_fold, "organ_system": organ, "column": column, "fit_median": median}
            )
        time_grid = build_metric_time_grid(fit, score, report_times=REPORT_TIMES)
        effective_60m = time_grid.effective_for(60.0)
        binary_state = ipcw_binary_state(fit, score, horizon=effective_60m)
        predictions: dict[str, dict[str, object]] = {}
        cache_payload: dict[str, np.ndarray] = {
            "participant_id": pd.to_numeric(score["participant_id"], errors="raise").to_numpy(dtype=np.int64),
            "duration": score["duration"].to_numpy(dtype=np.float32),
            "event": score["event"].to_numpy(dtype=np.int8),
        }
        for tier in tiers:
            for augmentation in AUGMENTATIONS:
                name = model_id(tier, augmentation)
                covariates = covariates_for(tier, augmentation, organ)
                bundle = fit_lifelines_cox_scores(
                    fit,
                    score,
                    covariates=covariates,
                    time_grid=time_grid,
                    phase="formal",
                    penalizer=penalizer,
                )
                report, _ = evaluate_censoring_aware_metrics(
                    fit,
                    score,
                    risk_score=bundle.risk_score,
                    survival_probability=bundle.survival_probability,
                    time_grid=time_grid,
                    calibration_bins=10,
                )
                scalar = flatten_primary_metrics(report)
                risks: dict[float, np.ndarray] = {}
                calibration_summary: dict[str, float | int] = {}
                for report_time in REPORT_TIMES:
                    fit_risk = probability_at(bundle, time_grid, report_time, fit=True)
                    score_risk = probability_at(bundle, time_grid, report_time, fit=False)
                    risks[report_time] = score_risk
                    summary, bins = fixed_bin_calibration(
                        fit,
                        score,
                        fit_risk=fit_risk,
                        score_risk=score_risk,
                        horizon=time_grid.effective_for(report_time),
                        report_time=report_time,
                        n_bins=10,
                    )
                    prefix = f"cal_{int(report_time)}m_"
                    calibration_summary.update(
                        {prefix + key: value for key, value in summary.items() if key != "time_months"}
                    )
                    bins.insert(0, "model", name)
                    bins.insert(0, "organ_system", organ)
                    bins.insert(0, "outer_fold", outer_fold)
                    calibration_rows.extend(bins.to_dict("records"))
                dca = ipcw_decision_curve(risks[60.0], binary_state, thresholds=THRESHOLDS)
                dca.insert(0, "model", name)
                dca.insert(0, "organ_system", organ)
                dca.insert(0, "outer_fold", outer_fold)
                dca_rows.extend(dca.to_dict("records"))
                harrell = harrell_c_index(score, bundle.risk_score)
                model_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "organ_system": organ,
                        "tier": tier,
                        "augmentation": augmentation,
                        "model": name,
                        "fit_n": len(fit),
                        "fit_events": int(fit["event"].sum()),
                        "fit_right_open_boundary_events_reclassified": int(
                            fit["right_open_boundary_event_reclassified"].sum()
                        ),
                        "test_n": len(score),
                        "test_events": int(score["event"].sum()),
                        "test_right_open_boundary_events_reclassified": int(
                            score["right_open_boundary_event_reclassified"].sum()
                        ),
                        "n_covariates": len(covariates),
                        "harrell_c": harrell,
                        **scalar,
                        **calibration_summary,
                        "ipcw_g_60m": binary_state.censoring_survival_horizon,
                        "ipcw_max_weight_60m": binary_state.max_weight,
                        "ipcw_effective_sample_size_60m": binary_state.effective_sample_size,
                    }
                )
                for covariate, coefficient in bundle.coefficients.items():
                    coefficient_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "organ_system": organ,
                            "model": name,
                            "covariate": covariate,
                            "coefficient_per_validation_sd": coefficient,
                        }
                    )
                predictions[name] = {
                    "risk_score": bundle.risk_score,
                    "risks": risks,
                    "metrics": {"harrell_c": harrell, **scalar},
                    "dca": dca,
                }
                cache_key = name.replace("-", "_")
                cache_payload[f"{cache_key}__risk_score"] = bundle.risk_score.astype(np.float32)
                for report_time, risk in risks.items():
                    cache_payload[f"{cache_key}__risk_{int(report_time)}m"] = risk.astype(np.float32)

        for tier in tiers:
            baseline_name = model_id(tier, "base")
            baseline = predictions[baseline_name]
            for augmentation in ("target_gap", "all_gaps"):
                augmented_name = model_id(tier, augmentation)
                augmented = predictions[augmented_name]
                base_metrics = baseline["metrics"]
                new_metrics = augmented["metrics"]
                reclassification = ipcw_reclassification(
                    baseline["risks"][60.0], augmented["risks"][60.0], binary_state
                )
                base_dca = baseline["dca"][["threshold", "net_benefit"]].rename(
                    columns={"net_benefit": "base_net_benefit"}
                )
                new_dca = augmented["dca"][["threshold", "net_benefit"]].rename(
                    columns={"net_benefit": "augmented_net_benefit"}
                )
                delta_dca = base_dca.merge(new_dca, on="threshold", validate="one_to_one")
                delta_dca["delta_net_benefit"] = (
                    delta_dca["augmented_net_benefit"] - delta_dca["base_net_benefit"]
                )
                delta_dca.insert(0, "comparison", f"{augmented_name}_vs_{baseline_name}")
                delta_dca.insert(0, "organ_system", organ)
                delta_dca.insert(0, "outer_fold", outer_fold)
                dca_rows.extend(delta_dca.to_dict("records"))
                incremental_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "organ_system": organ,
                        "tier": tier,
                        "comparison": f"{augmented_name}_vs_{baseline_name}",
                        "baseline_model": baseline_name,
                        "augmented_model": augmented_name,
                        "test_n": len(score),
                        "test_events": int(score["event"].sum()),
                        "delta_harrell_c": new_metrics["harrell_c"] - base_metrics["harrell_c"],
                        "delta_uno_c_60m": new_metrics["uno_c_60m"] - base_metrics["uno_c_60m"],
                        "delta_auc_12m": new_metrics["auc_12m"] - base_metrics["auc_12m"],
                        "delta_auc_24m": new_metrics["auc_24m"] - base_metrics["auc_24m"],
                        "delta_auc_60m": new_metrics["auc_60m"] - base_metrics["auc_60m"],
                        "brier_improvement_12m": base_metrics["brier_12m"] - new_metrics["brier_12m"],
                        "brier_improvement_24m": base_metrics["brier_24m"] - new_metrics["brier_24m"],
                        "brier_improvement_60m": base_metrics["brier_60m"] - new_metrics["brier_60m"],
                        "ibs_improvement": base_metrics["ibs"] - new_metrics["ibs"],
                        **reclassification,
                        "mean_delta_net_benefit": float(delta_dca["delta_net_benefit"].mean()),
                        "fraction_dca_thresholds_improved": float(delta_dca["delta_net_benefit"].gt(0.0).mean()),
                        "area_under_delta_net_benefit": float(
                            np.trapz(delta_dca["delta_net_benefit"], delta_dca["threshold"])
                        ),
                    }
                )
        np.savez_compressed(cache_dir / f"{organ}.npz", **cache_payload)
        print(
            f"Fold {outer_fold} [{organ_index}/{len(ORGANS)}] {organ}: "
            f"{len(fit):,} fit / {len(score):,} test, {time.time() - cell_started:.1f}s",
            flush=True,
        )

    outputs = {
        "cell_model_metrics.csv": pd.DataFrame(model_rows),
        "cell_incremental_metrics.csv": pd.DataFrame(incremental_rows),
        "calibration_bins.csv": pd.DataFrame(calibration_rows),
        "decision_curve.csv": pd.DataFrame(dca_rows),
        "cox_coefficients.csv": pd.DataFrame(coefficient_rows),
        "fit_imputation_medians.csv": pd.DataFrame(imputation_rows),
    }
    for filename, frame in outputs.items():
        frame.to_csv(fold_dir / filename, index=False, lineterminator="\n")
    write_json(
        complete_path,
        {
            "schema_version": "revision0710.clinical_utility.fold_complete.v1",
            "status": "PASS",
            "outer_fold": outer_fold,
            "tiers": list(tiers),
            "augmentations": list(AUGMENTATIONS),
            "penalizer": penalizer,
            "organs_completed": list(ORGANS),
            "model_cells": len(model_rows),
            "incremental_cells": len(incremental_rows),
            "elapsed_seconds": time.time() - started,
            "input_files": {
                "calibration_predictions": {
                    "path": str(calibration_path),
                    "bytes": calibration_path.stat().st_size,
                    "sha256": sha256_file(calibration_path),
                },
                "outer_test_predictions": {
                    "path": str(test_path),
                    "bytes": test_path.stat().st_size,
                    "sha256": sha256_file(test_path),
                },
            },
        },
    )


def concatenate_folds(output_dir: Path, folds: Sequence[int]) -> dict[str, pd.DataFrame]:
    filenames = (
        "cell_model_metrics.csv",
        "cell_incremental_metrics.csv",
        "calibration_bins.csv",
        "decision_curve.csv",
        "cox_coefficients.csv",
        "fit_imputation_medians.csv",
    )
    combined: dict[str, pd.DataFrame] = {}
    for filename in filenames:
        frames = [pd.read_csv(output_dir / "folds" / f"outer_fold_{fold}" / filename) for fold in folds]
        combined[filename] = pd.concat(frames, ignore_index=True)
        combined[filename].to_csv(output_dir / filename, index=False, lineterminator="\n")
    return combined


def aggregate_results(output_dir: Path, combined: dict[str, pd.DataFrame]) -> None:
    model = combined["cell_model_metrics.csv"]
    incremental = combined["cell_incremental_metrics.csv"]
    model_numeric = [
        column for column in model.columns
        if column not in {"outer_fold", "organ_system", "tier", "augmentation", "model"}
        and pd.api.types.is_numeric_dtype(model[column])
    ]
    incremental_numeric = [
        column for column in incremental.columns
        if column not in {
            "outer_fold", "organ_system", "tier", "comparison", "baseline_model", "augmented_model"
        }
        and pd.api.types.is_numeric_dtype(incremental[column])
    ]
    organ_model = model.groupby(
        ["organ_system", "tier", "augmentation", "model"], as_index=False
    )[model_numeric].mean()
    macro_model = organ_model.groupby(
        ["tier", "augmentation", "model"], as_index=False
    )[model_numeric].mean()
    organ_incremental = incremental.groupby(
        ["organ_system", "tier", "comparison", "baseline_model", "augmented_model"], as_index=False
    )[incremental_numeric].mean()
    macro_incremental = organ_incremental.groupby(
        ["tier", "comparison", "baseline_model", "augmented_model"], as_index=False
    )[incremental_numeric].mean()
    organ_model.to_csv(output_dir / "organ_model_metrics.csv", index=False, lineterminator="\n")
    macro_model.to_csv(output_dir / "macro_model_metrics.csv", index=False, lineterminator="\n")
    organ_incremental.to_csv(output_dir / "organ_incremental_metrics.csv", index=False, lineterminator="\n")
    macro_incremental.to_csv(output_dir / "macro_incremental_metrics.csv", index=False, lineterminator="\n")


def environment_receipt() -> dict:
    import lifelines
    import scipy
    import sklearn
    import sksurv

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "scikit_survival": sksurv.__version__,
        "lifelines": lifelines.__version__,
    }


def repair_calibration(args: argparse.Namespace) -> None:
    for outer_fold in args.folds:
        fold_dir = args.output_dir / "folds" / f"outer_fold_{outer_fold}"
        metrics_path = fold_dir / "cell_model_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        model_metrics = pd.read_csv(metrics_path)
        calibration_path, _ = prediction_paths(outer_fold)
        fit_predictions = pd.read_csv(
            calibration_path,
            usecols=["participant_id", "organ_system", "duration_months", "event", "at_risk"],
            dtype={"participant_id": str, "organ_system": str},
            low_memory=False,
        )
        for organ in ORGANS:
            fit = organ_outcomes(fit_predictions, organ)
            with np.load(fold_dir / "prediction_cache" / f"{organ}.npz") as cache:
                score = pd.DataFrame(
                    {
                        "participant_id": cache["participant_id"].astype(str),
                        "duration": cache["duration"].astype(np.float64),
                        "event": cache["event"].astype(np.int8),
                    }
                )
                boundary = float(min(fit["duration"].max(), score["duration"].max()))
                fit_boundary_event = (
                    np.isclose(fit["duration"], boundary, rtol=0.0, atol=1e-6)
                    & fit["event"].eq(1)
                )
                fit.loc[fit_boundary_event, "event"] = 0
                for report_time in REPORT_TIMES:
                    effective = (
                        float(np.nextafter(boundary, 0.0))
                        if np.isclose(report_time, 60.0, atol=0.1)
                        else report_time
                    )
                    state = ipcw_binary_state(fit, score, horizon=effective)
                    for name in model_metrics.loc[
                        model_metrics["organ_system"].eq(organ), "model"
                    ].unique():
                        cache_key = name.replace("-", "_")
                        risk = cache[f"{cache_key}__risk_{int(report_time)}m"].astype(np.float64)
                        intercept, slope = ipcw_logistic_calibration(risk, state)
                        selected = model_metrics["organ_system"].eq(organ) & model_metrics["model"].eq(name)
                        model_metrics.loc[
                            selected, f"cal_{int(report_time)}m_ipcw_logistic_calibration_intercept"
                        ] = intercept
                        model_metrics.loc[
                            selected, f"cal_{int(report_time)}m_ipcw_logistic_calibration_slope"
                        ] = slope
        model_metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    combined = concatenate_folds(args.output_dir, args.folds)
    aggregate_results(args.output_dir, combined)
    write_json(
        args.output_dir / "CALIBRATION_SLOPE_REPAIR_RECEIPT.json",
        {
            "schema_version": "revision0710.clinical_utility.calibration_slope.v1",
            "status": "PASS",
            "method": "sklearn LogisticRegression weighted by validation-fitted IPCW",
            "folds": list(args.folds),
            "remaining_numeric_nan": int(
                combined["cell_model_metrics.csv"].select_dtypes("number").isna().sum().sum()
            ),
        },
    )


def run(args: argparse.Namespace) -> None:
    if not args.context_pickle.exists():
        raise FileNotFoundError(
            f"Prepared context pickle not found: {args.context_pickle}. Run prepare-context first."
        )
    invalid_tiers = sorted(set(args.tiers).difference(TIERS))
    if invalid_tiers:
        raise ValueError(f"unknown tiers: {invalid_tiers}")
    invalid_folds = sorted(set(args.folds).difference(range(5)))
    if invalid_folds:
        raise ValueError(f"invalid folds: {invalid_folds}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = pd.read_pickle(args.context_pickle)
    context["participant_id"] = context["participant_id"].astype(str)
    if len(context) != 456_773 or context["participant_id"].duplicated().any():
        raise RuntimeError("prepared contextual covariate contract failed")
    write_json(args.output_dir / "ENVIRONMENT_RECEIPT.json", environment_receipt())
    for outer_fold in args.folds:
        run_fold(
            outer_fold,
            context,
            args.tiers,
            output_dir=args.output_dir,
            penalizer=args.penalizer,
        )
    combined = concatenate_folds(args.output_dir, args.folds)
    aggregate_results(args.output_dir, combined)
    write_json(
        args.output_dir / "FINAL_RECEIPT.json",
        {
            "schema_version": "revision0710.clinical_utility.point_results.v1",
            "status": "CAUSE_SPECIFIC_POINT_RESULTS_COMPLETE",
            "folds": list(args.folds),
            "tiers": list(args.tiers),
            "organs": list(ORGANS),
            "augmentations": list(AUGMENTATIONS),
            "penalizer": args.penalizer,
            "horizons_months": list(REPORT_TIMES),
            "decision_curve_thresholds": list(THRESHOLDS),
            "decision_curve_interpretation": "exploratory prespecified continuity grid",
            "death_handling": "cause-specific censoring",
            "competing_risk_claim_allowed": False,
            "context_pickle": {
                "path": str(args.context_pickle.resolve()),
                "bytes": args.context_pickle.stat().st_size,
                "sha256": sha256_file(args.context_pickle),
            },
            "formal_bootstrap_complete": False,
            "legacy_revision0626_fitted_values_read": False,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-context")
    prepare.add_argument("--context-parquet", type=Path, default=CONTEXT_PARQUET)
    prepare.add_argument("--context-pickle", type=Path, default=CONTEXT_PICKLE)
    execute = subparsers.add_parser("run")
    execute.add_argument("--context-pickle", type=Path, default=CONTEXT_PICKLE)
    execute.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    execute.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    execute.add_argument("--tiers", nargs="+", default=list(DEFAULT_TIERS))
    execute.add_argument("--penalizer", type=float, default=PENALIZER)
    repair = subparsers.add_parser("repair-calibration")
    repair.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    repair.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-context":
        prepare_context(args)
    elif args.command == "run":
        run(args)
    else:
        repair_calibration(args)


if __name__ == "__main__":
    main()
