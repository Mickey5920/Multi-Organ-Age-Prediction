"""Fixed-OOF missingness and phenotype-outlier sensitivity for Revision-0710.

This script does not refit or select EHR-AGE. It derives participant-level
quality strata from baseline phenotypes using other-fold reference data, then
re-evaluates the frozen five-fold OOF predictions and downstream Cox models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[3]
REVISION = REPO / "Revision-0710"
OOF_PATH = (
    REVISION
    / "03_ehr_age/04_fixed_replication/02_published_fivefold/02_oof_hr_km_inputs"
    / "ehr_age_d07_003_e55_oof.parquet"
)
RAW_PHENOTYPES = REVISION / "01_data_rebuild/01_index_and_events/raw_phenotypes.parquet"
PRIMARY_SURVIVAL = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics/04_cox_diagnostics_and_competing_risk"
    / "outputs/primary_survival_age_sex_v1/analysis"
)
COX_CORE_DIR = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics/04_cox_diagnostics_and_competing_risk"
)
DEFAULT_OUTPUT = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/06_supplementary_closure_0716"
    / "02_missingness_outlier"
)

sys.path.insert(0, str(COX_CORE_DIR))
from survival_analysis_core import benjamini_hochberg, fit_cox_breslow  # noqa: E402


Z95 = 1.959963984540054
PRIMARY_SAMPLE = "primary_all"
SUBSET_LABELS = {
    PRIMARY_SAMPLE: "Primary (all)",
    "low_missingness_q1": "Low missingness (Q1)",
    "exclude_outlier_top1": "Exclude outlier burden top 1%",
    "low_missingness_q1_exclude_outlier_top1": "Low missingness + outlier filtered",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if actual.size < 3:
        return {"n": int(actual.size), "mae": np.nan, "pearson": np.nan, "r2": np.nan}
    residual = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(residual))),
        "pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "r2": float(1.0 - np.sum(residual**2) / denominator),
    }


def crossfit_missingness_quartiles(
    missing_fraction: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    quartile = np.zeros(len(folds), dtype=np.int8)
    rows: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds)):
        target = folds == fold
        reference = missing_fraction[~target]
        q25, q50, q75 = np.quantile(reference, [0.25, 0.50, 0.75])
        quartile[target] = np.select(
            [missing_fraction[target] <= q25, missing_fraction[target] <= q50,
             missing_fraction[target] <= q75],
            [1, 2, 3],
            default=4,
        ).astype(np.int8)
        rows.append(
            {
                "outer_fold": int(fold),
                "reference_folds": ",".join(str(int(x)) for x in sorted(np.unique(folds)) if x != fold),
                "reference_n": int(reference.size),
                "q25_missing_fraction": float(q25),
                "q50_missing_fraction": float(q50),
                "q75_missing_fraction": float(q75),
                "target_n": int(target.sum()),
                "outcome_columns_read": False,
            }
        )
    return quartile, pd.DataFrame(rows)


def _robust_scale(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(reference, axis=0)
    mad = np.nanmedian(np.abs(reference - median), axis=0)
    scale = 1.4826 * mad
    method = np.full(reference.shape[1], "MAD_x_1.4826", dtype=object)
    invalid = ~np.isfinite(scale) | (scale <= 1e-12)
    if invalid.any():
        q25, q75 = np.nanquantile(reference[:, invalid], [0.25, 0.75], axis=0)
        fallback = (q75 - q25) / 1.349
        use = invalid.copy()
        use[invalid] = np.isfinite(fallback) & (fallback > 1e-12)
        scale[use] = fallback[np.isfinite(fallback) & (fallback > 1e-12)]
        method[use] = "IQR_div_1.349"
        invalid = ~np.isfinite(scale) | (scale <= 1e-12)
    if invalid.any():
        fallback = np.nanstd(reference[:, invalid], axis=0, ddof=0)
        use = invalid.copy()
        use[invalid] = np.isfinite(fallback) & (fallback > 1e-12)
        scale[use] = fallback[np.isfinite(fallback) & (fallback > 1e-12)]
        method[use] = "SD_ddof0"
        invalid = ~np.isfinite(scale) | (scale <= 1e-12)
    if invalid.any():
        scale[invalid] = np.nan
        method[invalid] = "unusable_zero_variance"
    return median, scale, method


def _outlier_burden(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        robust_z = np.abs(values - median) / scale
        exceedance = np.maximum(robust_z - 5.0, 0.0)
        observed = np.isfinite(robust_z).sum(axis=1)
        numerator = np.nansum(np.square(exceedance), axis=1)
    return np.divide(
        numerator,
        observed,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=observed > 0,
    )


def crossfit_outlier_burden(
    values: np.ndarray, folds: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    score = np.full(len(folds), np.nan, dtype=float)
    is_top1 = np.zeros(len(folds), dtype=np.int8)
    reference_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    unique_folds = sorted(np.unique(folds))
    for fold in unique_folds:
        target = folds == fold
        reference_values = values[~target]
        median, scale, method = _robust_scale(reference_values)
        reference_score = _outlier_burden(reference_values, median, scale)
        threshold = float(np.nanquantile(reference_score, 0.99))
        target_score = _outlier_burden(values[target], median, scale)
        score[target] = target_score
        is_top1[target] = (target_score > threshold).astype(np.int8)
        threshold_rows.append(
            {
                "outer_fold": int(fold),
                "reference_folds": ",".join(str(int(x)) for x in unique_folds if x != fold),
                "reference_n": int((~target).sum()),
                "target_n": int(target.sum()),
                "outlier_burden_q99": threshold,
                "target_excluded_n": int((target_score > threshold).sum()),
                "target_excluded_fraction": float(np.mean(target_score > threshold)),
                "robust_z_threshold": 5.0,
                "burden_formula": "mean(max(abs(robust_z)-5,0)^2) over observed phenotype slots",
                "outcome_columns_read": False,
            }
        )
        for index, feature in enumerate(feature_names):
            reference_rows.append(
                {
                    "outer_fold": int(fold),
                    "feature": feature,
                    "reference_median": float(median[index]) if np.isfinite(median[index]) else np.nan,
                    "reference_scale": float(scale[index]) if np.isfinite(scale[index]) else np.nan,
                    "scale_method": str(method[index]),
                    "reference_nonmissing_n": int(np.isfinite(reference_values[:, index]).sum()),
                }
            )
    return score, is_top1, pd.DataFrame(threshold_rows), pd.DataFrame(reference_rows)


def build_participant_flags(raw: pd.DataFrame, fold_map: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if raw["participant_id"].duplicated().any():
        raise ValueError("raw phenotype table has duplicate participant_id")
    fold_counts = fold_map.groupby("participant_id", observed=True)["outer_fold"].nunique()
    if int(fold_counts.max()) != 1:
        raise ValueError("a participant occurs in multiple outer folds")
    fold_map = fold_map.drop_duplicates("participant_id").copy()
    raw = raw.copy()
    raw["participant_id"] = raw["participant_id"].astype(str)
    fold_map["participant_id"] = fold_map["participant_id"].astype(str)
    raw = raw.merge(fold_map, on="participant_id", how="inner", validate="one_to_one")
    if len(raw) != len(fold_map):
        raise ValueError(f"phenotype/fold participant mismatch: raw={len(raw)} fold={len(fold_map)} joined={len(raw)}")
    feature_names = [
        column
        for column in raw.columns
        if column not in {"participant_id", "sex", "reported_age_at_recruitment", "outer_fold"}
    ]
    if len(feature_names) != 78:
        raise ValueError(f"expected 78 phenotype slots, found {len(feature_names)}")
    values = raw[feature_names].to_numpy(dtype=np.float64)
    observed = np.isfinite(values).sum(axis=1)
    missing_fraction = 1.0 - observed / float(len(feature_names))
    folds = raw["outer_fold"].to_numpy(dtype=np.int8)
    missing_q, missing_ref = crossfit_missingness_quartiles(missing_fraction, folds)
    outlier_score, outlier_top1, outlier_threshold, outlier_ref = crossfit_outlier_burden(
        values, folds, feature_names
    )
    flags = raw[["participant_id", "outer_fold"]].copy()
    flags["phenotype_slots_total"] = len(feature_names)
    flags["phenotype_slots_observed"] = observed.astype(np.int16)
    flags["phenotype_missing_fraction"] = missing_fraction
    flags["missingness_quartile_crossfit"] = missing_q
    flags["low_missingness_q1"] = (missing_q == 1).astype(np.int8)
    flags["outlier_burden_crossfit"] = outlier_score
    flags["outlier_burden_top1_crossfit"] = outlier_top1
    flags["exclude_outlier_top1"] = (outlier_top1 == 0).astype(np.int8)
    flags["low_missingness_q1_exclude_outlier_top1"] = (
        (missing_q == 1) & (outlier_top1 == 0)
    ).astype(np.int8)
    return flags, {
        "missingness_reference": missing_ref,
        "outlier_threshold": outlier_threshold,
        "outlier_feature_reference": outlier_ref,
    }


def subset_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        PRIMARY_SAMPLE: np.ones(len(frame), dtype=bool),
        "low_missingness_q1": frame["low_missingness_q1"].eq(1).to_numpy(),
        "exclude_outlier_top1": frame["exclude_outlier_top1"].eq(1).to_numpy(),
        "low_missingness_q1_exclude_outlier_top1": frame[
            "low_missingness_q1_exclude_outlier_top1"
        ].eq(1).to_numpy(),
    }


def evaluate_strict_age(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict = oof.loc[oof["strict_healthy_reference_60m"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for organ, organ_frame in strict.groupby("organ_system", observed=True, sort=True):
        masks = subset_masks(organ_frame)
        for sample, mask in masks.items():
            metrics = regression_metrics(
                organ_frame.loc[mask, "actual_age"], organ_frame.loc[mask, "predicted_age_raw"]
            )
            rows.append({"organ_system": organ, "analysis_sample": sample,
                         "analysis_sample_label": SUBSET_LABELS[sample], **metrics})
            for fold, fold_frame in organ_frame.loc[mask].groupby("outer_fold", observed=True):
                fold_rows.append(
                    {"organ_system": organ, "analysis_sample": sample, "outer_fold": int(fold),
                     **regression_metrics(fold_frame["actual_age"], fold_frame["predicted_age_raw"])}
                )
        for quartile in (1, 2, 3, 4):
            part = organ_frame.loc[organ_frame["missingness_quartile_crossfit"].eq(quartile)]
            rows.append(
                {"organ_system": organ, "analysis_sample": f"missingness_q{quartile}",
                 "analysis_sample_label": f"Missingness Q{quartile}",
                 **regression_metrics(part["actual_age"], part["predicted_age_raw"])}
            )
    metrics = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    primary = metrics.loc[metrics["analysis_sample"].eq(PRIMARY_SAMPLE),
                          ["organ_system", "mae", "pearson", "r2"]].rename(
        columns={"mae": "primary_mae", "pearson": "primary_pearson", "r2": "primary_r2"}
    )
    delta = metrics.merge(primary, on="organ_system", how="left", validate="many_to_one")
    delta["delta_mae_vs_primary"] = delta["mae"] - delta["primary_mae"]
    delta["delta_pearson_vs_primary"] = delta["pearson"] - delta["primary_pearson"]
    delta["delta_r2_vs_primary"] = delta["r2"] - delta["primary_r2"]
    return metrics, folds, delta


def add_primary_gap_z(oof: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(PRIMARY_SURVIVAL / "01_crossfit_exposure_reference.csv")
    reference = reference[["organ_system", "outer_fold", "reference_mean", "reference_sd_ddof0"]]
    out = oof.merge(reference, on=["organ_system", "outer_fold"], how="left", validate="many_to_one")
    if out[["reference_mean", "reference_sd_ddof0"]].isna().any().any():
        raise ValueError("missing primary crossfit exposure reference")
    out["gap_z_crossfit"] = (
        out["corrected_age_gap"] - out["reference_mean"]
    ) / out["reference_sd_ddof0"]
    return out


def run_sensitivity_cox(oof: pd.DataFrame) -> pd.DataFrame:
    primary = pd.read_csv(PRIMARY_SURVIVAL / "02_cox_primary_summary.csv")
    primary_rows = primary[
        ["organ_system", "n", "events", "gap_hr_per_crossfit_sd", "gap_hr_ci95_low",
         "gap_hr_ci95_high", "gap_p_value"]
    ].copy()
    primary_rows.insert(1, "analysis_sample", PRIMARY_SAMPLE)
    primary_rows["analysis_sample_label"] = SUBSET_LABELS[PRIMARY_SAMPLE]
    primary_rows["estimation_source"] = "formal_primary_survival_reuse"
    rows = primary_rows.to_dict("records")
    risk = add_primary_gap_z(oof.loc[oof["at_risk"].eq(1)].copy())
    for organ, organ_frame in risk.groupby("organ_system", observed=True, sort=True):
        organ_frame = organ_frame.copy()
        organ_frame["age_per_10y"] = (organ_frame["actual_age"] - 60.0) / 10.0
        for sample in (
            "low_missingness_q1",
            "exclude_outlier_top1",
            "low_missingness_q1_exclude_outlier_top1",
        ):
            part = organ_frame.loc[organ_frame[sample].eq(1)]
            x = part[["age_per_10y", "sex", "gap_z_crossfit"]].to_numpy(dtype=float)
            fit = fit_cox_breslow(
                x,
                part["duration_months"].to_numpy(dtype=float),
                part["event"].to_numpy(dtype=np.int8),
                strata=part["outer_fold"].to_numpy(dtype=np.int8),
            )
            if not fit.converged:
                raise RuntimeError(f"Cox model did not converge: {organ} {sample}")
            beta = float(fit.beta[2])
            se = float(fit.standard_error[2])
            rows.append(
                {
                    "organ_system": organ,
                    "analysis_sample": sample,
                    "analysis_sample_label": SUBSET_LABELS[sample],
                    "n": int(len(part)),
                    "events": int(part["event"].sum()),
                    "gap_hr_per_crossfit_sd": float(np.exp(beta)),
                    "gap_hr_ci95_low": float(np.exp(beta - Z95 * se)),
                    "gap_hr_ci95_high": float(np.exp(beta + Z95 * se)),
                    "gap_p_value": float(fit.p_value[2]),
                    "estimation_source": "fixed_oof_subset_cox_age_sex_outer_fold_stratified",
                    "converged": bool(fit.converged),
                    "iterations": int(fit.iterations),
                    "ties": fit.ties,
                }
            )
    result = pd.DataFrame(rows)
    result["gap_q_value_bh_7_organs"] = np.nan
    for sample, index in result.groupby("analysis_sample", observed=True).groups.items():
        result.loc[index, "gap_q_value_bh_7_organs"] = benjamini_hochberg(
            result.loc[index, "gap_p_value"]
        )
    return result.sort_values(["analysis_sample", "organ_system"]).reset_index(drop=True)


def participant_summary(flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_label, frame in [("all", flags), *[(str(f), flags.loc[flags["outer_fold"].eq(f)]) for f in sorted(flags["outer_fold"].unique())]]:
        for quartile in (1, 2, 3, 4):
            part = frame.loc[frame["missingness_quartile_crossfit"].eq(quartile)]
            rows.append(
                {
                    "outer_fold": fold_label,
                    "missingness_quartile_crossfit": quartile,
                    "n": int(len(part)),
                    "missing_fraction_mean": float(part["phenotype_missing_fraction"].mean()),
                    "missing_fraction_median": float(part["phenotype_missing_fraction"].median()),
                    "missing_fraction_min": float(part["phenotype_missing_fraction"].min()),
                    "missing_fraction_max": float(part["phenotype_missing_fraction"].max()),
                    "outlier_top1_n": int(part["outlier_burden_top1_crossfit"].sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = output / "Source_Data"
    cache = output / "Analysis_Cache"
    source.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(RAW_PHENOTYPES)
    fold_map = pd.read_parquet(OOF_PATH, columns=["participant_id", "outer_fold"])
    flags, references = build_participant_flags(raw, fold_map)
    flags.to_parquet(cache / "participant_sensitivity_flags.parquet", index=False)
    write_csv(source / "missingness_crossfit_reference.csv", references["missingness_reference"])
    write_csv(source / "outlier_crossfit_thresholds.csv", references["outlier_threshold"])
    write_csv(source / "outlier_feature_crossfit_reference.csv", references["outlier_feature_reference"])
    write_csv(source / "participant_quality_group_counts.csv", participant_summary(flags))

    oof_columns = [
        "participant_id", "organ_system", "outer_fold", "actual_age", "sex",
        "strict_healthy_reference_60m", "predicted_age_raw", "corrected_age_gap",
        "at_risk", "duration_months", "event",
    ]
    oof = pd.read_parquet(OOF_PATH, columns=oof_columns)
    oof["participant_id"] = oof["participant_id"].astype(str)
    oof = oof.merge(
        flags.drop(columns=["outer_fold"]), on="participant_id", how="left", validate="many_to_one"
    )
    if oof["missingness_quartile_crossfit"].isna().any():
        raise ValueError("OOF rows missing participant sensitivity flags")

    age_metrics, age_fold_metrics, age_delta = evaluate_strict_age(oof)
    write_csv(source / "strict_age_metrics_by_quality_subset.csv", age_metrics)
    write_csv(source / "strict_age_metrics_by_quality_subset_fold.csv", age_fold_metrics)
    write_csv(source / "strict_age_metric_deltas_vs_primary.csv", age_delta)
    cox = run_sensitivity_cox(oof)
    write_csv(source / "cox_hr_by_quality_subset.csv", cox)

    manifest = {
        "schema_version": "revision0710.fixed_oof_missingness_outlier_sensitivity.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "contract": {
            "model_refit": False,
            "checkpoint_selection": False,
            "prediction_source": "frozen five-fold OOF EHR-AGE",
            "strict_age_population": "strict_healthy_reference_60m == 1",
            "strict_age_prediction_field": "predicted_age_raw",
            "missingness": "fraction missing among the 78 baseline phenotype slots",
            "missingness_quartiles": "other-four-fold empirical quartiles applied to each held-out fold",
            "low_missingness_subset": "crossfit missingness Q1",
            "outlier_score": "mean(max(abs(robust_z)-5,0)^2) across observed phenotype slots",
            "outlier_reference": "other-four-fold feature median and MAD*1.4826; IQR/1.349 then SD fallback",
            "outlier_exclusion": "above other-four-fold q99 burden threshold in held-out fold",
            "cox": "age + sex + target-organ corrected age gap; outer-fold-stratified Breslow cause-specific Cox",
            "gap_unit": "one SD from the formal primary non-target-fold exposure reference",
            "primary_hr": "reused exactly from formal primary survival output",
        },
        "row_counts": {
            "participants": int(flags["participant_id"].nunique()),
            "oof_rows": int(len(oof)),
            "strict_oof_rows": int(oof["strict_healthy_reference_60m"].sum()),
            "outlier_excluded_participants": int(flags["outlier_burden_top1_crossfit"].sum()),
        },
        "inputs": [
            {"path": OOF_PATH.as_posix(), "sha256": sha256(OOF_PATH)},
            {"path": RAW_PHENOTYPES.as_posix(), "sha256": sha256(RAW_PHENOTYPES)},
            {"path": (PRIMARY_SURVIVAL / "01_crossfit_exposure_reference.csv").as_posix(),
             "sha256": sha256(PRIMARY_SURVIVAL / "01_crossfit_exposure_reference.csv")},
            {"path": (PRIMARY_SURVIVAL / "02_cox_primary_summary.csv").as_posix(),
             "sha256": sha256(PRIMARY_SURVIVAL / "02_cox_primary_summary.csv")},
        ],
        "outputs": sorted(path.name for path in source.glob("*.csv")),
    }
    write_json(output / "MANIFEST.json", manifest)
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
