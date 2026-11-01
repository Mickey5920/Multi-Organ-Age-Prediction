from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
MIMIC_VARIANTS = (
    "mimic_primary_hospital_icd_only_explicit_unit_phenotype",
    "mimic_sensitivity_hospital_icd_only_bare_omr_unit_assumptions",
    "mimic_sensitivity_hospital_icd_plus_inpatient_prescription_proxy",
)
MIMIC_RX_VARIANT = MIMIC_VARIANTS[2]
MIMIC_BARE_VARIANT = MIMIC_VARIANTS[1]
RELEASES = (
    "P_2017_MARCH2020_PREPANDEMIC",
    "L_AUG2021_AUG2023",
)
SURVIVAL_CORE_RELATIVE_PATH = (
    "Revision-0710/06_figures_and_statistics/03_reviewer_statistics/"
    "04_cox_diagnostics_and_competing_risk/survival_analysis_core.py"
)
SURVIVAL_CORE_SHA256 = (
    "6c4dd7eef4ea690d0d88a330f117d8d0b45274d788e10862ab665e245e435a1b"
)
PH_TRANSFORMS = ("rank_lifelines_cumsum_event", "log1p_days")
DENSITY_COVARIATES = (
    "log1p_pre_crop_event_count_z",
    "log1p_history_span_days_z",
    "observed_phenotype_fraction_z",
)
RISK_MODELS = (
    "age_sex",
    "age_sex_corrected_gap",
    "age_sex_density",
    "age_sex_density_corrected_gap",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_density_stratum(pre_crop_event_count: pd.Series | np.ndarray) -> np.ndarray:
    """Apply absolute, training-independent pre-index event-count boundaries."""
    values = np.asarray(pre_crop_event_count, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("pre-crop event count cannot be negative")
    return np.select(
        [values <= 7, values <= 31],
        ["low_0_7", "medium_8_31"],
        default="high_32_plus",
    )


def _timeline_density(events: pd.DataFrame, subject_ids: Sequence[str]) -> pd.DataFrame:
    frame = events.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise")
    frame["landmark_time"] = pd.to_datetime(frame["landmark_time"], errors="raise")
    grouped = frame.groupby("participant_id", sort=False).agg(
        pre_crop_event_count=("event_time", "size"),
        first_event_time=("event_time", "min"),
        landmark_time=("landmark_time", "max"),
    )
    result = pd.DataFrame({"subject_id": pd.Series(subject_ids, dtype="string")})
    result = result.merge(
        grouped.reset_index().rename(columns={"participant_id": "subject_id"}),
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    result["pre_crop_event_count"] = (
        result["pre_crop_event_count"].fillna(0).astype("int64")
    )
    span = (
        pd.to_datetime(result["landmark_time"])
        - pd.to_datetime(result["first_event_time"])
    ).dt.total_seconds() / 86400.0
    result["history_span_days"] = span.fillna(0.0).clip(lower=0.0).astype(float)
    return result[["subject_id", "pre_crop_event_count", "history_span_days"]]


def build_label_blind_mimic_density(
    *,
    mimic_participants: pd.DataFrame,
    icd_events: pd.DataFrame,
    prescription_events: pd.DataFrame,
    phenotype_missing: np.ndarray,
    bare_phenotype_missing: np.ndarray,
    crop_variant_timeline: Callable[[pd.DataFrame, str], pd.DataFrame],
    prepare_variant_timeline: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> pd.DataFrame:
    """Build participant density without predictions, outcomes, or post-index data."""
    subject_ids = mimic_participants["subject_id"].astype(str).tolist()
    if len(subject_ids) != len(set(subject_ids)):
        raise AssertionError("MIMIC density subject universe is not unique")
    for matrix in (phenotype_missing, bare_phenotype_missing):
        if matrix.shape != (len(subject_ids), 78):
            raise AssertionError("MIMIC density phenotype mask shape mismatch")
    rows: list[pd.DataFrame] = []
    for variant in MIMIC_VARIANTS:
        source = (
            pd.concat([icd_events, prescription_events], ignore_index=True)
            if variant == MIMIC_RX_VARIANT
            else icd_events.copy()
        )
        prepared = prepare_variant_timeline(source, variant)
        pre = _timeline_density(prepared, subject_ids)
        selected = crop_variant_timeline(source, variant)
        post_counts = (
            selected.assign(participant_id=selected["participant_id"].astype(str))
            .groupby("participant_id", sort=False)
            .size()
        )
        missing = (
            bare_phenotype_missing if variant == MIMIC_BARE_VARIANT else phenotype_missing
        )
        pre["input_variant"] = variant
        pre["post_crop_event_count"] = (
            pre["subject_id"].map(post_counts).fillna(0).astype("int64")
        )
        pre["observed_phenotype_slots"] = (~missing).sum(axis=1).astype("int16")
        pre["density_stratum"] = fixed_density_stratum(pre["pre_crop_event_count"])
        pre["density_stratum_rule"] = "pre_crop_events:low=0-7;medium=8-31;high>=32"
        pre["l32_mask_policy"] = "left_aligned_active_events_then_inactive_padding_to_32"
        pre["label_or_outcome_fields_read"] = False
        rows.append(pre)
    result = pd.concat(rows, ignore_index=True)
    key = ["subject_id", "input_variant"]
    if result.duplicated(key).any() or len(result) != len(subject_ids) * 3:
        raise AssertionError("MIMIC density exact participant-variant universe mismatch")
    if (result["post_crop_event_count"] > 32).any():
        raise AssertionError("MIMIC post-crop density exceeds L32")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


NHANES_UNIT_RANGES: Mapping[str, tuple[float, float]] = {
    "%": (0.0, 100.0),
    "10^12/L": (0.0, 50.0),
    "10^9/L": (0.0, 10_000.0),
    "beats/min": (0.0, 300.0),
    "cm": (0.0, 300.0),
    "fL": (0.0, 1_000.0),
    "g/cm2": (0.0, 10.0),
    "g/dL": (0.0, 100.0),
    "g/L": (0.0, 1_000.0),
    "kg": (0.0, 500.0),
    "kg/m2": (0.0, 200.0),
    "L": (0.0, 20.0),
    "L/min": (0.0, 100.0),
    "mg/L": (0.0, 1_000_000.0),
    "mmHg": (0.0, 400.0),
    "mmol/L": (0.0, 10_000.0),
    "mmol/mol": (0.0, 1_000.0),
    "nmol/L": (0.0, 10_000_000.0),
    "pg": (0.0, 1_000.0),
    "ratio": (0.0, 10.0),
    "U/L": (0.0, 1_000_000.0),
    "umol/L": (0.0, 10_000_000.0),
}


def build_nhanes_release_slot_qc(
    *,
    participants: pd.DataFrame,
    raw: np.ndarray,
    missing: np.ndarray,
    release_slot_map: pd.DataFrame,
) -> pd.DataFrame:
    if raw.shape != missing.shape or raw.shape[1] != 78 or raw.shape[0] != len(participants):
        raise AssertionError("NHANES QC array/participant shape mismatch")
    if set(release_slot_map["target_unit"].unique()) - set(NHANES_UNIT_RANGES):
        raise AssertionError("NHANES QC has an unbounded target unit")
    rows: list[dict[str, Any]] = []
    offset = 0
    for release in RELEASES:
        n = int(participants["release_id"].eq(release).sum())
        if participants.iloc[offset : offset + n]["release_id"].ne(release).any():
            raise AssertionError("NHANES raw matrix release order mismatch")
        release_map = release_slot_map.loc[release_slot_map["release_id"].eq(release)]
        if release_map["slot_index"].astype(int).tolist() != list(range(78)):
            raise AssertionError("NHANES QC release map is not exact slots 0..77")
        for mapping in release_map.itertuples(index=False):
            slot = int(mapping.slot_index)
            observed_mask = ~missing[offset : offset + n, slot]
            values = raw[offset : offset + n, slot][observed_mask].astype(np.float64)
            finite = np.isfinite(values)
            lower, upper = NHANES_UNIT_RANGES[str(mapping.target_unit)]
            range_ok = finite & (values >= lower) & (values <= upper)
            quantiles: list[float | None]
            if finite.any():
                quantiles = [
                    float(value)
                    for value in np.quantile(
                        values[finite], [0.01, 0.25, 0.5, 0.75, 0.99], method="linear"
                    )
                ]
                minimum = float(np.min(values[finite]))
                maximum = float(np.max(values[finite]))
            else:
                quantiles = [None] * 5
                minimum = maximum = None
            nonfinite_n = int((~finite).sum())
            out_of_range_n = int((finite & ~range_ok).sum())
            status = (
                "not_measured"
                if len(values) == 0
                else "pass"
                if nonfinite_n == 0 and out_of_range_n == 0
                else "fail_abort_before_inference"
            )
            rows.append(
                {
                    "release_id": release,
                    "slot_index": slot,
                    "feature_name": str(mapping.feature_name),
                    "target_unit": str(mapping.target_unit),
                    "n_eligible": n,
                    "n_observed": len(values),
                    "n_nonfinite_observed": nonfinite_n,
                    "n_outside_fixed_range": out_of_range_n,
                    "fixed_range_min": lower,
                    "fixed_range_max": upper,
                    "observed_min": minimum,
                    "q01": quantiles[0],
                    "q25": quantiles[1],
                    "q50": quantiles[2],
                    "q75": quantiles[3],
                    "q99": quantiles[4],
                    "observed_max": maximum,
                    "qc_status": status,
                    "execution_rule": "nonfinite_or_outside_fixed_unit_range_aborts_before_inference",
                    "trimming_or_winsorization": "prohibited",
                    "quantiles_used_for_decision": False,
                }
            )
        offset += n
    result = pd.DataFrame(rows)
    if len(result) != 156 or result.duplicated(["release_id", "slot_index"]).any():
        raise AssertionError("NHANES QC exact release-slot universe mismatch")
    return result


def require_nhanes_qc_pass(qc: pd.DataFrame) -> None:
    failures = qc.loc[qc["qc_status"].eq("fail_abort_before_inference")]
    if len(failures):
        keys = failures[["release_id", "slot_index"]].astype(str).agg(":".join, axis=1)
        raise AssertionError("NHANES fixed QC failed: " + ",".join(keys.tolist()))


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return None
    return float(stats.pearsonr(x, y).statistic)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    return float(stats.spearmanr(x, y).statistic)


def build_mimic_density_summary(
    density: pd.DataFrame, ensemble: pd.DataFrame
) -> pd.DataFrame:
    prediction = ensemble.loc[ensemble["cohort"].eq("MIMIC_IV_3_1")].rename(
        columns={"participant_id": "subject_id"}
    )
    analysis = prediction.merge(
        density,
        on=["subject_id", "input_variant"],
        how="inner",
        validate="many_to_one",
    )
    analysis["absolute_age_error"] = np.abs(
        analysis["predicted_age_raw_mean5"] - analysis["chronological_age"]
    )
    density_variables = (
        "pre_crop_event_count",
        "post_crop_event_count",
        "history_span_days",
        "observed_phenotype_slots",
    )
    targets = (
        "chronological_age",
        "corrected_age_gap_mean5",
        "absolute_age_error",
    )
    rows: list[dict[str, Any]] = []
    for (variant, organ), group in analysis.groupby(
        ["input_variant", "organ_system"], sort=False
    ):
        for density_variable in density_variables:
            x = group[density_variable].to_numpy(dtype=np.float64)
            for target in targets:
                y = group[target].to_numpy(dtype=np.float64)
                for metric, estimate in (
                    ("pearson", _pearson(x, y)),
                    ("spearman", _spearman(x, y)),
                ):
                    rows.append(
                        {
                            "input_variant": variant,
                            "organ_system": organ,
                            "analysis": "density_correlation",
                            "density_measure": density_variable,
                            "density_stratum": "all",
                            "target": target,
                            "metric": metric,
                            "n": len(group),
                            "estimate": estimate,
                            "estimability_status": "estimable" if estimate is not None else "singular_design",
                            "fixed_training_independent_strata": True,
                        }
                    )
        for stratum in ("low_0_7", "medium_8_31", "high_32_plus"):
            selected = group.loc[group["density_stratum"].eq(stratum)]
            error = (
                selected["predicted_age_raw_mean5"].to_numpy(dtype=float)
                - selected["chronological_age"].to_numpy(dtype=float)
            )
            values = {
                "mae": float(np.mean(np.abs(error))) if len(error) else None,
                "rmse": float(np.sqrt(np.mean(error**2))) if len(error) else None,
                "mean_error": float(np.mean(error)) if len(error) else None,
            }
            for metric, estimate in values.items():
                rows.append(
                    {
                        "input_variant": variant,
                        "organ_system": organ,
                        "analysis": "fixed_density_stratum_age_error",
                        "density_measure": "pre_crop_event_count",
                        "density_stratum": stratum,
                        "target": "raw_mean5_age_error",
                        "metric": metric,
                        "n": len(selected),
                        "estimate": estimate,
                        "estimability_status": "estimable" if estimate is not None else "insufficient_complete_cases",
                        "fixed_training_independent_strata": True,
                    }
                )
    return pd.DataFrame(rows)


def _standardize_density(
    training: pd.DataFrame, testing: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train = training.copy()
    test = testing.copy()
    raw_columns = {
        "log1p_pre_crop_event_count_z": np.log1p(train["pre_crop_event_count"]),
        "log1p_history_span_days_z": np.log1p(train["history_span_days"]),
        "observed_phenotype_fraction_z": train["observed_phenotype_slots"] / 78.0,
    }
    test_values = {
        "log1p_pre_crop_event_count_z": np.log1p(test["pre_crop_event_count"]),
        "log1p_history_span_days_z": np.log1p(test["history_span_days"]),
        "observed_phenotype_fraction_z": test["observed_phenotype_slots"] / 78.0,
    }
    statistics: dict[str, float] = {}
    stat_prefix = {
        "log1p_pre_crop_event_count_z": "log1p_pre_crop_event_count",
        "log1p_history_span_days_z": "log1p_history_span_days",
        "observed_phenotype_fraction_z": "observed_phenotype_fraction",
    }
    for name, values in raw_columns.items():
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"density training-fold SD is zero: {name}")
        train[name] = (np.asarray(values, dtype=float) - mean) / sd
        test[name] = (np.asarray(test_values[name], dtype=float) - mean) / sd
        statistics[f"{stat_prefix[name]}_train_mean"] = mean
        statistics[f"{stat_prefix[name]}_train_sd"] = sd
    gap_mean = float(train["corrected_age_gap_mean5"].mean())
    gap_sd = float(train["corrected_age_gap_mean5"].std(ddof=1))
    if not np.isfinite(gap_sd) or gap_sd <= 0:
        raise ValueError("training-fold corrected-gap SD is zero")
    train["corrected_gap_z"] = (train["corrected_age_gap_mean5"] - gap_mean) / gap_sd
    test["corrected_gap_z"] = (test["corrected_age_gap_mean5"] - gap_mean) / gap_sd
    statistics["corrected_age_gap_train_mean"] = gap_mean
    statistics["corrected_age_gap_train_sd"] = gap_sd
    return train, test, statistics


def build_density_adjusted_risk_sensitivity(
    *,
    incident: pd.DataFrame,
    ensemble: pd.DataFrame,
    density: pd.DataFrame,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    months_to_days: Callable[[int], float],
    cause_specific_cif_predictions: Callable[..., tuple[np.ndarray, np.ndarray]],
    harrell_c: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ipcw_binary_weights: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
    weighted_auc: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    weighted_brier: Callable[..., float],
    weighted_linear_calibration: Callable[..., tuple[float, float]],
    global_subject_bootstrap_plan: Callable[..., tuple[np.ndarray, np.ndarray]],
    conditional_bootstrap_values: Callable[..., Mapping[str, tuple[float | None, float | None]]],
) -> pd.DataFrame:
    prediction = ensemble.loc[ensemble["cohort"].eq("MIMIC_IV_3_1")].rename(
        columns={"participant_id": "subject_id"}
    )
    analysis = incident.merge(
        prediction[["subject_id", "input_variant", "organ_system", "corrected_age_gap_mean5"]],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    ).merge(
        density,
        on=["subject_id", "input_variant"],
        how="inner",
        validate="many_to_one",
    )
    analysis["age_per10"] = analysis["chronological_age"].astype(float) / 10.0
    subject_ids, plan_counts = global_subject_bootstrap_plan(
        global_subject_ids=np.sort(analysis["subject_id"].astype(str).unique()),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    rows: list[dict[str, Any]] = []
    metrics = (
        "harrell_c",
        "delta_harrell_c",
        "ipcw_auc",
        "ipcw_brier",
        "ipcw_linear_calibration_intercept",
        "ipcw_linear_calibration_slope",
    )
    for (variant, organ, horizon), group in analysis.groupby(
        ["input_variant", "organ_system", "horizon_months"], sort=False
    ):
        horizon_days = months_to_days(int(horizon))
        for fold in range(5):
            train0 = group.loc[group["hash_fold"].ne(fold)].reset_index(drop=True)
            test0 = group.loc[group["hash_fold"].eq(fold)].reset_index(drop=True)
            common = {
                "input_variant": variant,
                "organ_system": organ,
                "horizon_months": int(horizon),
                "hash_fold": fold,
                "analysis_role": "density_adjusted_heldout_sensitivity",
                "n_test": len(test0),
                "n_events": int(test0["event_indicator"].sum()),
                "n_competing_deaths": int(test0["competing_death_indicator"].sum()),
                "density_covariates": "+".join(DENSITY_COVARIATES),
                "density_standardization": "training_fold_mean_sd_only",
                "fixed_training_independent_strata": True,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
                "uncertainty_scope": "conditional_on_fixed_training_fold_target_death_cox_and_censor_km",
                "clinical_utility_claim_allowed": False,
                "comparison_reference_model": "age_sex_density",
                "log1p_pre_crop_event_count_train_mean": None,
                "log1p_pre_crop_event_count_train_sd": None,
                "log1p_history_span_days_train_mean": None,
                "log1p_history_span_days_train_sd": None,
                "observed_phenotype_fraction_train_mean": None,
                "observed_phenotype_fraction_train_sd": None,
                "corrected_age_gap_train_mean": None,
                "corrected_age_gap_train_sd": None,
            }
            if len(test0) < 4 or common["n_events"] == 0:
                status = "zero_target_events" if common["n_events"] == 0 else "insufficient_complete_cases"
                for model in ("age_sex_density", "age_sex_density_corrected_gap"):
                    for metric in metrics:
                        rows.append({**common, "model": model, "metric": metric, "estimability_status": status, "estimate": None, "ci_lower": None, "ci_upper": None})
                continue
            try:
                training, testing, scaling = _standardize_density(train0, test0)
                common.update(scaling)
                base_columns = ["age_per10", "sex_male", *DENSITY_COVARIATES]
                augmented_columns = [*base_columns, "corrected_gap_z"]
                base_risk, base_probability = cause_specific_cif_predictions(
                    training, testing, base_columns, horizon_days
                )
                augmented_risk, augmented_probability = cause_specific_cif_predictions(
                    training, testing, augmented_columns, horizon_days
                )
                cause_test = (
                    testing["event_indicator"].to_numpy(dtype=np.int8)
                    + 2 * testing["competing_death_indicator"].to_numpy(dtype=np.int8)
                )
                censor_training = training["censor_reason"].isin(
                    ["subject_observation_end", "death_ascertainment_cap"]
                ).to_numpy(dtype=np.int8)
                outcome, weight, _, _ = ipcw_binary_weights(
                    testing["observed_time_days"].to_numpy(dtype=float),
                    cause_test,
                    horizon_days,
                    training["observed_time_days"].to_numpy(dtype=float),
                    censor_training,
                    return_diagnostics=True,
                )
                event = testing["event_indicator"].to_numpy(dtype=np.int8)
                time = testing["observed_time_days"].to_numpy(dtype=float)
                base_c = harrell_c(time, event, base_risk)
            except (ValueError, np.linalg.LinAlgError):
                for model in ("age_sex_density", "age_sex_density_corrected_gap"):
                    for metric in metrics:
                        rows.append({**common, "model": model, "metric": metric, "estimability_status": "singular_design", "estimate": None, "ci_lower": None, "ci_upper": None})
                continue
            for model, risk, probability in (
                ("age_sex_density", base_risk, base_probability),
                ("age_sex_density_corrected_gap", augmented_risk, augmented_probability),
            ):
                c_value = harrell_c(time, event, risk)
                intercept, slope = weighted_linear_calibration(outcome, probability, weight)
                point = {
                    "harrell_c": c_value,
                    "delta_harrell_c": c_value - base_c,
                    "ipcw_auc": weighted_auc(outcome, probability, weight),
                    "ipcw_brier": weighted_brier(outcome, probability, weight, normalization_n=len(testing)),
                    "ipcw_linear_calibration_intercept": intercept,
                    "ipcw_linear_calibration_slope": slope,
                }
                intervals = conditional_bootstrap_values(
                    testing=testing,
                    risk=risk,
                    base_risk=base_risk,
                    probability=probability,
                    outcome=outcome,
                    ipcw_weight=weight,
                    plan_subject_ids=subject_ids,
                    plan_counts=plan_counts,
                )
                for metric, estimate in point.items():
                    lower, upper = intervals[metric]
                    rows.append({**common, "model": model, "metric": metric, "estimability_status": "estimable", "estimate": estimate, "ci_lower": lower, "ci_upper": upper, "uncertainty_scope": common["uncertainty_scope"] if lower is not None else "point_estimate_only_bootstrap_insufficient"})
    return pd.DataFrame(rows)


def load_survival_core(root: Path) -> Any:
    path = root / SURVIVAL_CORE_RELATIVE_PATH
    if sha256_file(path) != SURVIVAL_CORE_SHA256:
        raise AssertionError("shared survival-analysis core bytes changed")
    spec = importlib.util.spec_from_file_location("revision0710_v8_survival_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ph_failure_rows(
    *,
    common: Mapping[str, Any],
    covariates: Sequence[str],
    status: str,
    failure_code: str,
) -> list[dict[str, Any]]:
    return [
        {
            **common,
            "covariate": covariate,
            "transform": transform,
            "estimability_status": status,
            "fit_converged": False,
            "failure_code": failure_code,
            "chi_square": None,
            "df": None,
            "p_value": None,
        }
        for covariate in [*covariates, "GLOBAL"]
        for transform in PH_TRANSFORMS
    ]


def _global_schoenfeld_ph_test(
    *,
    core: Any,
    fit: Any,
    design: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> pd.DataFrame:
    """Joint PH score tests using the same scaled residuals as the bound core."""
    strata = np.zeros(len(design), dtype=np.int8)
    prepared = core._prepare_cox_strata(design, duration, event, strata)
    residuals: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for part in prepared:
        eta = part.x @ fit.beta
        shift = float(np.max(eta))
        weight = np.exp(eta - shift)
        risk0 = np.cumsum(weight)[part.ends]
        risk1 = np.column_stack(
            [np.cumsum(weight * part.x[:, j])[part.ends] for j in range(design.shape[1])]
        )
        means = risk1 / risk0[:, None]
        for group_index in range(len(part.starts) - 1, -1, -1):
            start, end = part.starts[group_index], part.ends[group_index]
            local = np.flatnonzero(part.event[start : end + 1] == 1)
            if local.size:
                indices = start + local
                residuals.append(part.x[indices] - means[group_index])
                times.append(part.duration[indices])
    if not residuals:
        raise ValueError("no event residuals for global PH test")
    residual = np.vstack(residuals)
    event_time = np.concatenate(times)
    n_events = len(event_time)
    scaled = n_events * residual @ fit.covariance
    transforms = {
        "rank_lifelines_cumsum_event": np.arange(1, n_events + 1, dtype=float),
        "log1p_days": np.log(event_time),
    }
    rows = []
    for transform, transformed_time in transforms.items():
        centered = transformed_time - transformed_time.mean()
        time_ss = float(np.sum(centered**2))
        score = np.sum(centered[:, None] * scaled, axis=0)
        score_covariance = n_events * time_ss * np.asarray(fit.covariance, dtype=float)
        rank = int(np.linalg.matrix_rank(score_covariance, tol=1e-12))
        if rank <= 0:
            statistic = math.nan
            p_value = math.nan
        else:
            statistic = float(score @ np.linalg.pinv(score_covariance, rcond=1e-12) @ score)
            p_value = float(stats.chi2.sf(statistic, rank))
        rows.append(
            {
                "covariate": "GLOBAL",
                "time_transform": transform,
                "n_events": n_events,
                "chi_square": statistic,
                "df": rank,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _fit_ph_rows(
    *,
    core: Any,
    frame: pd.DataFrame,
    covariates: Sequence[str],
    event_column: str,
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    n_events = int(frame[event_column].sum())
    design = frame[list(covariates)].to_numpy(dtype=float)
    duration = frame["observed_time_days"].to_numpy(dtype=float)
    complete = (
        np.isfinite(design).all(axis=1)
        & np.isfinite(duration)
        & np.isfinite(frame[event_column].to_numpy(dtype=float))
    )
    bound = {
        **common,
        "n_input": len(frame),
        "n_complete": int(complete.sum()),
        "n_events": n_events,
    }
    if len(frame) <= len(covariates) + 2:
        return _ph_failure_rows(common=bound, covariates=covariates, status="insufficient_complete_cases", failure_code="n_not_greater_than_covariate_count_plus_2")
    if n_events == 0:
        status = (
            "zero_competing_deaths"
            if common.get("cause") == "death"
            else "zero_target_events"
        )
        return _ph_failure_rows(common=bound, covariates=covariates, status=status, failure_code="zero_events_for_cause")
    if not np.isfinite(design).all() or np.any(np.std(design, axis=0) <= 0):
        return _ph_failure_rows(common=bound, covariates=covariates, status="singular_design", failure_code="nonfinite_or_zero_variance_covariate")
    try:
        ph_duration = frame["observed_time_days"].to_numpy(dtype=float) + 1.0
        fit = core.fit_cox_breslow(
            design,
            ph_duration,
            frame[event_column].to_numpy(dtype=np.int8),
        )
        if not fit.converged:
            return _ph_failure_rows(common=bound, covariates=covariates, status="singular_design", failure_code="shared_core_fit_not_converged")
        diagnostic = core.schoenfeld_ph_test(
            fit,
            design,
            ph_duration,
            frame[event_column].to_numpy(dtype=np.int8),
            covariate_names=list(covariates),
        )
        global_diagnostic = _global_schoenfeld_ph_test(
            core=core,
            fit=fit,
            design=design,
            duration=ph_duration,
            event=frame[event_column].to_numpy(dtype=np.int8),
        )
        diagnostic.loc[diagnostic["time_transform"].eq("log"), "time_transform"] = "log1p_days"
        diagnostic = pd.concat([diagnostic, global_diagnostic], ignore_index=True)
    except ValueError:
        return _ph_failure_rows(common=bound, covariates=covariates, status="singular_design", failure_code="shared_core_value_error")
    except np.linalg.LinAlgError:
        return _ph_failure_rows(common=bound, covariates=covariates, status="singular_design", failure_code="shared_core_linear_algebra_error")
    except FloatingPointError:
        return _ph_failure_rows(common=bound, covariates=covariates, status="singular_design", failure_code="shared_core_floating_point_error")
    expected = {
        (covariate, transform)
        for covariate in [*covariates, "GLOBAL"]
        for transform in PH_TRANSFORMS
    }
    observed = set(zip(diagnostic["covariate"], diagnostic["time_transform"]))
    if observed != expected:
        raise AssertionError("shared-core PH diagnostic omitted a covariate/transform row")
    rows = []
    for row in diagnostic.itertuples(index=False):
        is_global_failure = (
            row.covariate == "GLOBAL"
            and (
                int(row.df) <= 0
                or not np.isfinite(float(row.chi_square))
                or not np.isfinite(float(row.p_value))
            )
        )
        rows.append(
            {
                **bound,
                "covariate": row.covariate,
                "transform": row.time_transform,
                "estimability_status": "singular_design" if is_global_failure else "estimable",
                "fit_converged": True,
                "failure_code": "global_score_covariance_rank_zero_or_nonfinite" if is_global_failure else "none",
                "chi_square": None if is_global_failure else float(row.chi_square),
                "df": None if is_global_failure else int(row.df),
                "p_value": None if is_global_failure else float(row.p_value),
            }
        )
    return rows


def build_complete_mimic_ph_diagnostics(
    *,
    root: Path,
    incident: pd.DataFrame,
    ensemble: pd.DataFrame,
    density: pd.DataFrame,
) -> pd.DataFrame:
    core = load_survival_core(root)
    prediction = ensemble.loc[ensemble["cohort"].eq("MIMIC_IV_3_1")].rename(
        columns={"participant_id": "subject_id"}
    )
    analysis = incident.merge(
        prediction[["subject_id", "input_variant", "organ_system", "corrected_age_gap_mean5"]],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    ).merge(
        density,
        on=["subject_id", "input_variant"],
        how="inner",
        validate="many_to_one",
    )
    analysis["age_per10"] = analysis["chronological_age"].astype(float) / 10.0
    rows: list[dict[str, Any]] = []
    shared = {
        "survival_core_relative_path": SURVIVAL_CORE_RELATIVE_PATH,
        "survival_core_sha256": SURVIVAL_CORE_SHA256,
        "ties": "breslow",
        "prediction_fit_backend": "statsmodels_PHReg_efron",
        "diagnostic_fit_backend": "diagnostic_only_byte_bound_shared_core_breslow_refit",
        "diagnostic_beta_or_risk_reused_for_prediction": False,
        "diagnostic_duration_rule": "observed_time_days_plus_1_for_all_rows_preserves_order_and_ties",
    }
    for (variant, organ, horizon), group0 in analysis.groupby(
        ["input_variant", "organ_system", "horizon_months"], sort=False
    ):
        group = group0.reset_index(drop=True)
        gap_sd = float(group["corrected_age_gap_mean5"].std(ddof=1))
        if np.isfinite(gap_sd) and gap_sd > 0:
            group["corrected_gap_z"] = (
                group["corrected_age_gap_mean5"] - group["corrected_age_gap_mean5"].mean()
            ) / gap_sd
        else:
            group["corrected_gap_z"] = 0.0
        common = {
            **shared,
            "analysis_role": "full_nonheldout_incident_eligible_association",
            "input_variant": variant,
            "organ_system": organ,
            "horizon_months": int(horizon),
            "hash_fold": -1,
            "model": "age_sex_corrected_gap",
            "cause": "target",
        }
        rows.extend(
            _fit_ph_rows(
                core=core,
                frame=group,
                covariates=["age_per10", "sex_male", "corrected_gap_z"],
                event_column="event_indicator",
                common=common,
            )
        )
        for fold in range(5):
            train0 = group.loc[group["hash_fold"].ne(fold)].reset_index(drop=True)
            test0 = group.loc[group["hash_fold"].eq(fold)].reset_index(drop=True)
            try:
                training, _, _ = _standardize_density(train0, test0)
                standardization_failure = None
            except (ValueError, np.linalg.LinAlgError) as exc:
                training = train0.copy()
                standardization_failure = "density_training_fold_standardization_failure"
                gap_mean = float(training["corrected_age_gap_mean5"].mean())
                gap_sd = float(training["corrected_age_gap_mean5"].std(ddof=1))
                if np.isfinite(gap_sd) and gap_sd > 0:
                    training["corrected_gap_z"] = (
                        training["corrected_age_gap_mean5"] - gap_mean
                    ) / gap_sd
                else:
                    training["corrected_gap_z"] = 0.0
            model_covariates = {
                "age_sex": ["age_per10", "sex_male"],
                "age_sex_corrected_gap": ["age_per10", "sex_male", "corrected_gap_z"],
                "age_sex_density": ["age_per10", "sex_male", *DENSITY_COVARIATES],
                "age_sex_density_corrected_gap": ["age_per10", "sex_male", *DENSITY_COVARIATES, "corrected_gap_z"],
            }
            for model, covariates in model_covariates.items():
                for cause, event_column in (
                    ("target", "event_indicator"),
                    ("death", "competing_death_indicator"),
                ):
                    common = {
                        **shared,
                        "analysis_role": "heldout_training_model_diagnostic",
                        "input_variant": variant,
                        "organ_system": organ,
                        "horizon_months": int(horizon),
                        "hash_fold": fold,
                        "model": model,
                        "cause": cause,
                    }
                    if standardization_failure and "density" in model:
                        rows.extend(
                            _ph_failure_rows(
                                common={
                                    **common,
                                    "n_input": len(training),
                                    "n_complete": len(training),
                                    "n_events": int(training[event_column].sum()),
                                },
                                covariates=covariates,
                                status="singular_design",
                                failure_code=standardization_failure,
                            )
                        )
                    else:
                        rows.extend(
                            _fit_ph_rows(
                                core=core,
                                frame=training,
                                covariates=covariates,
                                event_column=event_column,
                                common=common,
                            )
                        )
    result = pd.DataFrame(rows)
    key = [
        "analysis_role",
        "input_variant",
        "organ_system",
        "horizon_months",
        "hash_fold",
        "model",
        "cause",
        "covariate",
        "transform",
    ]
    if result.duplicated(key).any():
        raise AssertionError("complete PH diagnostic key is not unique")
    expected_model_causes = set(RISK_MODELS)
    heldout = result.loc[result["analysis_role"].eq("heldout_training_model_diagnostic")]
    if set(heldout["model"]) != expected_model_causes or set(heldout["cause"]) != {"target", "death"}:
        raise AssertionError("complete PH diagnostic model/cause universe mismatch")
    full_covariates = ["age_per10", "sex_male", "corrected_gap_z", "GLOBAL"]
    heldout_covariates = {
        "age_sex": ["age_per10", "sex_male", "GLOBAL"],
        "age_sex_corrected_gap": ["age_per10", "sex_male", "corrected_gap_z", "GLOBAL"],
        "age_sex_density": ["age_per10", "sex_male", *DENSITY_COVARIATES, "GLOBAL"],
        "age_sex_density_corrected_gap": ["age_per10", "sex_male", *DENSITY_COVARIATES, "corrected_gap_z", "GLOBAL"],
    }
    expected_keys: set[tuple[Any, ...]] = set()
    for variant in MIMIC_VARIANTS:
        for organ in ORGANS:
            for horizon in (6, 12):
                for covariate in full_covariates:
                    for transform in PH_TRANSFORMS:
                        expected_keys.add(("full_nonheldout_incident_eligible_association", variant, organ, horizon, -1, "age_sex_corrected_gap", "target", covariate, transform))
                for fold in range(5):
                    for model, covariates in heldout_covariates.items():
                        for cause in ("target", "death"):
                            for covariate in covariates:
                                for transform in PH_TRANSFORMS:
                                    expected_keys.add(("heldout_training_model_diagnostic", variant, organ, horizon, fold, model, cause, covariate, transform))
    observed_keys = set(result[key].itertuples(index=False, name=None))
    if len(result) != 17_136 or len(expected_keys) != 17_136 or observed_keys != expected_keys:
        raise AssertionError("complete PH diagnostic exact 17,136-row key universe mismatch")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)
