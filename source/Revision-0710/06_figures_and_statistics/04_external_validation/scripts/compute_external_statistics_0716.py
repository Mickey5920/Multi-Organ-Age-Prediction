from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
V8_DIR = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/00_plan/"
    "external_validation_prefreeze_v8"
)
if str(V8_DIR) not in sys.path:
    sys.path.insert(0, str(V8_DIR))

import execute_external_validation_v8_runtime as v8  # noqa: E402


WORK = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/04_external_validation/core_0716"
)
CACHE = WORK / "shared/02_label_blind_cache"
INFERENCE = WORK / "shared/04_inference_work"
OUTCOMES = WORK / "shared/05_outcomes_after_prediction_seal"
STATS = WORK / "shared/06_statistics"
FINAL = ROOT / "Revision-0710/Final_Fig_0716"
FIG13 = FINAL / "Fig13/Source_Data"
FIG14 = FINAL / "Fig14/Source_Data"
SEED = 7_102_026
ORGANS = list(v8.ORGANS)
MIMIC_PRIMARY = v8.MIMIC_PRIMARY_VARIANT
NHANES_PRIMARY = v8.NHANES_PRIMARY_VARIANT


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_inputs() -> dict[str, pd.DataFrame]:
    frames = {
        "per": pd.read_parquet(INFERENCE / "sealed_per_checkpoint_predictions.parquet"),
        "ensemble": pd.read_parquet(INFERENCE / "sealed_ensemble_predictions.parquet"),
        "mimic": pd.read_parquet(CACHE / "mimic_participants.parquet"),
        "nhanes": pd.read_parquet(CACHE / "nhanes_participants.parquet"),
        "survey": pd.read_parquet(CACHE / "nhanes_survey_design_universe.parquet"),
        "incident": pd.read_parquet(OUTCOMES / "incident_endpoints.parquet"),
        "recurrent": pd.read_parquet(OUTCOMES / "recurrent_endpoints.parquet"),
        "questionnaire": pd.read_parquet(
            OUTCOMES / "nhanes_questionnaire_labels.parquet"
        ),
        "eligibility": pd.read_parquet(
            OUTCOMES / "mimic_endpoint_subject_eligibility.parquet"
        ),
        "density": pd.read_parquet(
            CACHE / "mimic_participant_density_label_blind.parquet"
        ),
        "mimic_coverage": pd.read_parquet(CACHE / "mimic_feature_coverage.parquet"),
        "nhanes_coverage": pd.read_parquet(CACHE / "nhanes_feature_coverage.parquet"),
        "nhanes_qc": pd.read_csv(CACHE / "nhanes_release_slot_qc.csv"),
    }
    frames["per"]["participant_id"] = frames["per"]["participant_id"].astype(str)
    frames["ensemble"]["participant_id"] = frames["ensemble"][
        "participant_id"
    ].astype(str)
    frames["mimic"]["subject_id"] = frames["mimic"]["subject_id"].astype(str)
    frames["nhanes"]["participant_id"] = frames["nhanes"]["participant_id"].astype(
        str
    )
    frames["survey"]["participant_id"] = frames["survey"][
        "participant_id"
    ].astype(str)
    for name in ("incident", "recurrent", "eligibility"):
        frames[name]["subject_id"] = frames[name]["subject_id"].astype(str)
    frames["questionnaire"]["participant_id"] = frames["questionnaire"][
        "participant_id"
    ].astype(str)
    frames["density"]["subject_id"] = frames["density"]["subject_id"].astype(str)
    v8.validate_mimic_endpoint_rows(frames["incident"])
    v8.validate_mimic_endpoint_rows(frames["recurrent"])
    return frames


def checkpoint_range(age_metrics: pd.DataFrame) -> pd.DataFrame:
    primary = age_metrics.loc[
        age_metrics["population"].isin(
            [
                "mimic_primary_common_support_40_73_integer_year",
                "nhanes_primary_examined_40_73_exact",
            ]
        )
        & age_metrics["organ_system"].isin(ORGANS)
        & age_metrics["metric"].isin(
            ["mae", "rmse", "pearson", "spearman", "r2", "calibration_slope"]
        )
    ].copy()
    fold = primary.loc[primary["estimator"].str.startswith("fold")].copy()
    mean5 = primary.loc[primary["estimator"].eq("mean5")].copy()
    keys = [
        "cohort",
        "release_id",
        "input_variant",
        "population",
        "organ_system",
        "metric",
    ]
    ranges = (
        fold.groupby(keys, dropna=False, as_index=False)["estimate"]
        .agg(checkpoint_min="min", checkpoint_max="max", checkpoint_sd="std")
    )
    result = mean5.merge(ranges, on=keys, how="left", validate="one_to_one")
    result = result.rename(columns={"estimate": "mean5_estimate"})
    result["uncertainty_label"] = (
        "range across five fixed checkpoints; not a sampling confidence interval"
    )
    return result


def calibration_bins(ensemble: pd.DataFrame, cohort: str) -> pd.DataFrame:
    selected = ensemble.loc[ensemble["cohort"].eq(cohort)].copy()
    if cohort == "MIMIC_IV_3_1":
        selected = selected.loc[
            selected["chronological_age"].between(40, 73, inclusive="both")
            & selected["input_variant"].eq(MIMIC_PRIMARY)
        ].copy()
        selected["partition"] = "MIMIC_IV_3_1"
    else:
        selected = selected.loc[selected["input_variant"].eq(NHANES_PRIMARY)].copy()
        selected["partition"] = selected["release_id"].astype(str)
    rows = []
    for (partition, organ), group in selected.groupby(
        ["partition", "organ_system"], sort=False
    ):
        group = group.copy()
        group["age_bin"] = pd.cut(
            group["chronological_age"],
            bins=[39.5, 44.5, 49.5, 54.5, 59.5, 64.5, 69.5, 73.5],
            include_lowest=True,
            right=True,
        )
        for age_bin, values in group.groupby("age_bin", observed=True, sort=True):
            rows.append(
                {
                    "cohort": cohort,
                    "partition": partition,
                    "organ_system": organ,
                    "age_bin": str(age_bin),
                    "n": len(values),
                    "chronological_age_mean": values["chronological_age"].mean(),
                    "chronological_age_sd": values["chronological_age"].std(ddof=1),
                    "predicted_age_mean5_mean": values[
                        "predicted_age_raw_mean5"
                    ].mean(),
                    "predicted_age_mean5_sd": values[
                        "predicted_age_raw_mean5"
                    ].std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def mimic_hr_per_five_years(
    association: pd.DataFrame,
    incident: pd.DataFrame,
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    prediction = ensemble.loc[
        ensemble["cohort"].eq("MIMIC_IV_3_1")
        & ensemble["input_variant"].eq(MIMIC_PRIMARY)
    ].rename(columns={"participant_id": "subject_id"})
    analysis = incident.merge(
        prediction[
            ["subject_id", "input_variant", "organ_system", "corrected_age_gap_mean5"]
        ],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    )
    sd = (
        analysis.groupby(["input_variant", "organ_system", "horizon_months"])[
            "corrected_age_gap_mean5"
        ]
        .std(ddof=1)
        .rename("gap_sd_years")
        .reset_index()
    )
    result = association.merge(
        sd,
        on=["input_variant", "organ_system", "horizon_months"],
        how="left",
        validate="one_to_one",
    )
    scale = 5.0 / result["gap_sd_years"]
    for source, target in (
        ("hazard_ratio", "hr_per_5y_gap"),
        ("ci_lower", "hr_per_5y_ci_lower"),
        ("ci_upper", "hr_per_5y_ci_upper"),
    ):
        result[target] = np.exp(np.log(result[source].astype(float)) * scale)
    result["five_year_interpretation"] = (
        "cause-specific hazard ratio per +5 years corrected target-organ age gap; "
        "competing death is censored"
    )
    return result


def build_mimic_cause_specific_hr(
    incident: pd.DataFrame, ensemble: pd.DataFrame
) -> pd.DataFrame:
    from scipy.stats import norm, pearsonr

    prediction = ensemble.loc[
        ensemble["cohort"].eq("MIMIC_IV_3_1")
        & ensemble["input_variant"].eq(MIMIC_PRIMARY)
    ].rename(columns={"participant_id": "subject_id"})
    analysis = incident.merge(
        prediction[
            ["subject_id", "input_variant", "organ_system", "corrected_age_gap_mean5"]
        ],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    )
    analysis["age_per10"] = analysis["chronological_age"].astype(float) / 10.0
    analysis["gap_per5"] = analysis["corrected_age_gap_mean5"].astype(float) / 5.0
    rows = []
    for (organ, horizon), group in analysis.groupby(
        ["organ_system", "horizon_months"], sort=False
    ):
        group = group.reset_index(drop=True)
        common = {
            "input_variant": MIMIC_PRIMARY,
            "organ_system": organ,
            "horizon_months": int(horizon),
            "n": group["subject_id"].nunique(),
            "target_events": int(group["event_indicator"].sum()),
            "competing_deaths": int(group["competing_death_indicator"].sum()),
            "gap_sd_years": group["corrected_age_gap_mean5"].std(ddof=1),
            "exposure": "corrected target-organ age gap per +5 years",
            "covariates": "chronological age per 10 years + sex + corrected target-organ gap per 5 years",
            "estimand": "cause-specific target-event hazard; competing death censored",
            "ties": "efron",
            "standard_error": "model-based; one independent row per participant-organ-horizon",
            "interpretation": "association_not_causal_not_a_prediction_performance_metric",
        }
        try:
            complete, result = v8._cox_fit(
                group,
                ["age_per10", "sex_male", "gap_per5"],
                robust_groups=None,
            )
            coefficient = float(result.params[2])
            standard_error = float(result.bse[2])
            p_value = float(2.0 * norm.sf(abs(coefficient / standard_error)))
            covariance = np.asarray(result.cov_params(), dtype=float)
            residual_matrix = np.asarray(result.schoenfeld_residuals, dtype=float)
            residual = (residual_matrix @ covariance)[:, 2]
            event_mask = complete["event_indicator"].to_numpy(dtype=bool)
            residual = residual[event_mask]
            event_time = np.log(
                np.maximum(
                    complete.loc[event_mask, "observed_time_days"].to_numpy(dtype=float),
                    1e-6,
                )
            )
            valid = np.isfinite(residual) & np.isfinite(event_time)
            ph_p = (
                float(pearsonr(residual[valid], event_time[valid]).pvalue)
                if valid.sum() >= 3
                else None
            )
            rows.append(
                {
                    **common,
                    "estimability_status": "estimable",
                    "hr_per_5y_gap": float(np.exp(coefficient)),
                    "hr_per_5y_ci_lower": float(
                        np.exp(coefficient - 1.959963984540054 * standard_error)
                    ),
                    "hr_per_5y_ci_upper": float(
                        np.exp(coefficient + 1.959963984540054 * standard_error)
                    ),
                    "p_value": p_value,
                    "bh_q_value": None,
                    "ph_test_p": ph_p,
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            rows.append(
                {
                    **common,
                    "estimability_status": "singular_design",
                    "hr_per_5y_gap": None,
                    "hr_per_5y_ci_lower": None,
                    "hr_per_5y_ci_upper": None,
                    "p_value": None,
                    "bh_q_value": None,
                    "ph_test_p": None,
                }
            )
    result = pd.DataFrame(rows)
    for _, indices in result.groupby("horizon_months").groups.items():
        valid = [index for index in indices if pd.notna(result.at[index, "p_value"])]
        if valid:
            result.loc[valid, "bh_q_value"] = v8.benjamini_hochberg(
                result.loc[valid, "p_value"].to_numpy(dtype=float),
                family_size=7,
            )
    return result


def questionnaire_prevalence(
    labels: pd.DataFrame, participants: pd.DataFrame
) -> pd.DataFrame:
    merged = labels.merge(
        participants[["participant_id", "release_id", "survey_weight"]],
        on=["participant_id", "release_id"],
        how="left",
        validate="many_to_one",
    )
    rows = []
    for (release, organ), group in merged.groupby(
        ["release_id", "organ_system"], sort=False
    ):
        complete = group.loc[group["label"].notna()].copy()
        weight = complete["survey_weight"].to_numpy(dtype=float)
        label = complete["label"].to_numpy(dtype=float)
        weighted = (
            float(np.sum(weight * label) / np.sum(weight))
            if len(complete) and np.isfinite(weight).all() and np.sum(weight) > 0
            else None
        )
        rows.append(
            {
                "release_id": release,
                "organ_system": organ,
                "n_eligible": len(group),
                "n_complete": len(complete),
                "cases": int(complete["label"].sum()) if len(complete) else 0,
                "unweighted_prevalence": complete["label"].mean()
                if len(complete)
                else None,
                "survey_weighted_prevalence": weighted,
                "interpretation": "cross-sectional self-reported questionnaire label",
            }
        )
    return pd.DataFrame(rows)


def event_summary(incident: pd.DataFrame, recurrent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, frame in (("incident_primary", incident), ("recurrent_sensitivity", recurrent)):
        for (organ, horizon), group in frame.groupby(
            ["organ_system", "horizon_months"], sort=False
        ):
            rows.append(
                {
                    "analysis_role": role,
                    "organ_system": organ,
                    "horizon_months": int(horizon),
                    "n": group["subject_id"].nunique(),
                    "target_events": int(group["event_indicator"].sum()),
                    "competing_deaths": int(group["competing_death_indicator"].sum()),
                    "event_free_at_horizon": int(group["censor_reason"].eq("horizon").sum()),
                    "censored_before_horizon": int(
                        group["censor_reason"].eq("subject_observation_end").sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_basic() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    FIG13.mkdir(parents=True, exist_ok=True)
    FIG14.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    manifest = json.loads((CACHE / "MANIFEST.json").read_text(encoding="utf-8"))
    age = v8._build_zero_shot_age_metrics(
        data["per"],
        data["ensemble"],
        bootstrap_replicates=0,
        bootstrap_seed=SEED,
    )
    age_range = checkpoint_range(age)
    mimic_flow = v8._build_mimic_flow(
        manifest, data["incident"], data["recurrent"], data["eligibility"]
    )
    association = build_mimic_cause_specific_hr(
        data["incident"], data["ensemble"]
    )
    local_calibration = v8._build_nhanes_local_calibration(
        data["ensemble"], data["nhanes"]
    )
    survey_age = v8._build_nhanes_survey_age_metrics(
        data["ensemble"], data["nhanes"]
    )
    questionnaire_or = v8._build_nhanes_questionnaire_associations(
        data["ensemble"],
        data["nhanes"],
        data["questionnaire"],
        data["survey"],
    )
    event_counts = event_summary(data["incident"], data["recurrent"])
    mimic_calibration = calibration_bins(data["ensemble"], "MIMIC_IV_3_1")
    nhanes_calibration = calibration_bins(data["ensemble"], "NHANES")
    prevalence = questionnaire_prevalence(data["questionnaire"], data["survey"])
    nhanes_flow = pd.DataFrame(
        [
            {
                "release_id": release,
                "flow_step": "MEC_examined_age_40_73",
                "n": int(group["participant_id"].nunique()),
            }
            for release, group in data["nhanes"].groupby("release_id", sort=False)
        ]
    )

    outputs = {
        STATS / "zero_shot_age_metrics.csv": age,
        STATS / "zero_shot_age_metrics_checkpoint_range.csv": age_range,
        FIG13 / "Fig13A_mimic_cohort_flow.csv": mimic_flow,
        FIG13 / "Fig13B_raw_age_metrics_checkpoint_range.csv": age_range.loc[
            age_range["cohort"].eq("MIMIC_IV_3_1")
        ],
        FIG13 / "Fig13C_raw_age_calibration_bins.csv": mimic_calibration,
        FIG13 / "Fig13E_cause_specific_hr.csv": association,
        FIG13 / "MIMIC_incident_recurrent_event_counts.csv": event_counts,
        FIG13 / "MIMIC_feature_coverage.csv": data["mimic_coverage"],
        FIG14 / "Fig14A_release_cohort_flow.csv": nhanes_flow,
        FIG14 / "Fig14B_release_slot_qc.csv": data["nhanes_qc"],
        FIG14 / "Fig14C_raw_age_metrics_checkpoint_range.csv": age_range.loc[
            age_range["cohort"].eq("NHANES")
        ],
        FIG14 / "Fig14D_raw_age_calibration_bins.csv": nhanes_calibration,
        FIG14 / "Fig14E_local_crossfit_recalibration.csv": local_calibration,
        FIG14 / "Fig14F_questionnaire_or.csv": questionnaire_or,
        FIG14 / "NHANES_release_survey_age_metrics.csv": survey_age,
        FIG14 / "NHANES_questionnaire_prevalence.csv": prevalence,
        FIG14 / "NHANES_feature_coverage.csv": data["nhanes_coverage"],
    }
    for path, frame in outputs.items():
        write_csv(path, frame)
    receipt = {
        "schema_version": "revision0710.external_validation.core_0716.basic_statistics.v1",
        "status": "BASIC_EXTERNAL_STATISTICS_COMPLETE",
        "created_at": now_iso(),
        "prediction_seal_sha256": sha256_file(
            INFERENCE / "PREDICTION_SEAL_RECEIPT.json"
        ),
        "outcome_receipt_sha256": sha256_file(
            OUTCOMES / "OUTCOME_READ_RECEIPT.json"
        ),
        "outputs": {
            str(path.relative_to(ROOT)): {
                "rows": len(frame),
                "sha256": sha256_file(path),
            }
            for path, frame in outputs.items()
        },
    }
    write_json(STATS / "BASIC_STATISTICS_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def _risk_metric_bundle(frame: pd.DataFrame) -> dict[str, float]:
    time = frame["observed_time_days"].to_numpy(dtype=float)
    event = frame["event_indicator"].to_numpy(dtype=np.int8)
    outcome = frame["ipcw_outcome"].to_numpy(dtype=float)
    weight = frame["ipcw_weight"].to_numpy(dtype=float)
    base_risk = frame["base_risk"].to_numpy(dtype=float)
    augmented_risk = frame["augmented_risk"].to_numpy(dtype=float)
    base_probability = frame["base_probability"].to_numpy(dtype=float)
    augmented_probability = frame["augmented_probability"].to_numpy(dtype=float)
    base_c = v8.harrell_c(time, event, base_risk)
    augmented_c = v8.harrell_c(time, event, augmented_risk)
    base_auc = v8.weighted_auc(outcome, base_probability, weight)
    augmented_auc = v8.weighted_auc(outcome, augmented_probability, weight)
    base_brier = v8.weighted_brier(
        outcome, base_probability, weight, normalization_n=len(frame)
    )
    augmented_brier = v8.weighted_brier(
        outcome, augmented_probability, weight, normalization_n=len(frame)
    )
    return {
        "base_c": base_c,
        "augmented_c": augmented_c,
        "delta_c": augmented_c - base_c,
        "base_auc": base_auc,
        "augmented_auc": augmented_auc,
        "delta_auc": augmented_auc - base_auc,
        "base_brier": base_brier,
        "augmented_brier": augmented_brier,
        "brier_improvement": base_brier - augmented_brier,
    }


def _paired_fold_bootstrap(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    point = _risk_metric_bundle(frame)
    fold_indices = {
        int(fold): values.index.to_numpy(dtype=np.int64)
        for fold, values in frame.groupby("outer_fold", sort=True)
    }
    rng = np.random.default_rng(seed)
    valid: list[dict[str, float]] = []
    attempts = 0
    while len(valid) < replicates and attempts < replicates * 3:
        attempts += 1
        sampled = []
        for indices in fold_indices.values():
            sampled.extend(
                indices[
                    rng.integers(0, len(indices), size=len(indices), endpoint=False)
                ].tolist()
            )
        try:
            valid.append(_risk_metric_bundle(frame.loc[sampled].reset_index(drop=True)))
        except ValueError:
            continue
    if len(valid) != replicates:
        raise RuntimeError(
            f"paired bootstrap produced {len(valid)}/{replicates} valid replicates"
        )
    intervals = {
        metric: (
            float(np.quantile([row[metric] for row in valid], 0.025, method="linear")),
            float(np.quantile([row[metric] for row in valid], 0.975, method="linear")),
        )
        for metric in ("delta_c", "delta_auc", "brier_improvement")
    }
    return point, intervals, attempts


def _risk_seed(organ: str, horizon: int, comparison: str) -> int:
    payload = f"{SEED}|{organ}|{horizon}|{comparison}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def compute_risk(replicates: int) -> None:
    data = load_inputs()
    prediction = data["ensemble"].loc[
        data["ensemble"]["cohort"].eq("MIMIC_IV_3_1")
        & data["ensemble"]["input_variant"].eq(MIMIC_PRIMARY)
    ].rename(columns={"participant_id": "subject_id"})
    analysis = data["incident"].merge(
        prediction[
            ["subject_id", "input_variant", "organ_system", "corrected_age_gap_mean5"]
        ],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    )
    analysis = analysis.merge(
        data["density"][
            [
                "subject_id",
                "input_variant",
                "pre_crop_event_count",
                "history_span_days",
                "observed_phenotype_slots",
            ]
        ],
        on=["subject_id", "input_variant"],
        how="inner",
        validate="many_to_one",
    )
    analysis["age_per10"] = analysis["chronological_age"].astype(float) / 10.0
    analysis["gap_per5"] = analysis["corrected_age_gap_mean5"].astype(float) / 5.0
    analysis["log1p_preindex_events"] = np.log1p(
        analysis["pre_crop_event_count"].astype(float)
    )
    analysis["history_span_years"] = analysis["history_span_days"].astype(float) / 365.25
    analysis["observed_slots_per10"] = (
        analysis["observed_phenotype_slots"].astype(float) / 10.0
    )
    comparisons = {
        "age_sex_plus_gap": (
            ["age_per10", "sex_male"],
            ["age_per10", "sex_male", "gap_per5"],
        ),
        "age_sex_density_plus_gap": (
            [
                "age_per10",
                "sex_male",
                "log1p_preindex_events",
                "history_span_years",
                "observed_slots_per10",
            ],
            [
                "age_per10",
                "sex_male",
                "log1p_preindex_events",
                "history_span_years",
                "observed_slots_per10",
                "gap_per5",
            ],
        ),
    }
    oof_rows: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    for (organ, horizon), group in analysis.groupby(
        ["organ_system", "horizon_months"], sort=False
    ):
        horizon_days = v8.months_to_days(int(horizon))
        for fold in range(5):
            training = group.loc[group["hash_fold"].ne(fold)].reset_index(drop=True)
            testing = group.loc[group["hash_fold"].eq(fold)].reset_index(drop=True)
            cause_test = (
                testing["event_indicator"].to_numpy(dtype=np.int8)
                + 2 * testing["competing_death_indicator"].to_numpy(dtype=np.int8)
            )
            censor_training = training["censor_reason"].isin(
                ["subject_observation_end", "death_ascertainment_cap"]
            ).to_numpy(dtype=np.int8)
            try:
                outcome, weight, observed, diagnostics = v8.ipcw_binary_weights(
                    testing["observed_time_days"].to_numpy(dtype=float),
                    cause_test,
                    horizon_days,
                    training["observed_time_days"].to_numpy(dtype=float),
                    censor_training,
                    return_diagnostics=True,
                )
            except ValueError as exc:
                failure_rows.append(
                    {
                        "organ_system": organ,
                        "horizon_months": int(horizon),
                        "outer_fold": fold,
                        "comparison": "all",
                        "failure": f"ipcw:{type(exc).__name__}:{exc}",
                    }
                )
                continue
            for comparison, (base_columns, augmented_columns) in comparisons.items():
                try:
                    base_risk, base_probability = v8._cause_specific_cif_predictions(
                        training, testing, base_columns, horizon_days
                    )
                    augmented_risk, augmented_probability = (
                        v8._cause_specific_cif_predictions(
                            training, testing, augmented_columns, horizon_days
                        )
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    failure_rows.append(
                        {
                            "organ_system": organ,
                            "horizon_months": int(horizon),
                            "outer_fold": fold,
                            "comparison": comparison,
                            "failure": f"model:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
                oof_rows.append(
                    pd.DataFrame(
                        {
                            "subject_id": testing["subject_id"].astype(str),
                            "organ_system": organ,
                            "horizon_months": int(horizon),
                            "outer_fold": fold,
                            "comparison": comparison,
                            "observed_time_days": testing["observed_time_days"].to_numpy(
                                dtype=float
                            ),
                            "event_indicator": testing["event_indicator"].to_numpy(
                                dtype=np.int8
                            ),
                            "competing_death_indicator": testing[
                                "competing_death_indicator"
                            ].to_numpy(dtype=np.int8),
                            "ipcw_outcome": outcome,
                            "ipcw_weight": weight,
                            "ipcw_observed": observed,
                            "base_risk": base_risk,
                            "augmented_risk": augmented_risk,
                            "base_probability": base_probability,
                            "augmented_probability": augmented_probability,
                            "ipcw_g_min_raw": diagnostics["g_min_raw"],
                            "ipcw_weight_cap": diagnostics["weight_cap_p99_training"],
                        }
                    )
                )
    if not oof_rows:
        raise RuntimeError("no OOF risk predictions were produced")
    oof = pd.concat(oof_rows, ignore_index=True)
    expected = set(
        (organ, horizon, comparison, fold)
        for organ in ORGANS
        for horizon in (6, 12)
        for comparison in comparisons
        for fold in range(5)
    )
    observed_keys = set(
        map(
            tuple,
            oof[
                ["organ_system", "horizon_months", "comparison", "outer_fold"]
            ].drop_duplicates().itertuples(index=False, name=None),
        )
    )
    if observed_keys != expected:
        write_csv(STATS / "risk_model_failures.csv", pd.DataFrame(failure_rows))
        raise RuntimeError(
            f"OOF risk key universe incomplete: missing={len(expected - observed_keys)}"
        )
    oof_path = STATS / "mimic_oof_risk_predictions.parquet"
    STATS.mkdir(parents=True, exist_ok=True)
    temporary = oof_path.with_suffix(".parquet.tmp")
    oof.to_parquet(temporary, index=False)
    os.replace(temporary, oof_path)
    metrics_rows = []
    calibration_rows = []
    for (organ, horizon, comparison), group in oof.groupby(
        ["organ_system", "horizon_months", "comparison"], sort=False
    ):
        group = group.reset_index(drop=True)
        point, intervals, attempts = _paired_fold_bootstrap(
            group,
            replicates=replicates,
            seed=_risk_seed(str(organ), int(horizon), str(comparison)),
        )
        for metric, delta_key in (
            ("harrell_c", "delta_c"),
            ("ipcw_auc", "delta_auc"),
            ("ipcw_brier", "brier_improvement"),
        ):
            base_key = {
                "harrell_c": "base_c",
                "ipcw_auc": "base_auc",
                "ipcw_brier": "base_brier",
            }[metric]
            augmented_key = {
                "harrell_c": "augmented_c",
                "ipcw_auc": "augmented_auc",
                "ipcw_brier": "augmented_brier",
            }[metric]
            metrics_rows.append(
                {
                    "organ_system": organ,
                    "horizon_months": int(horizon),
                    "comparison": comparison,
                    "metric": metric,
                    "n": group["subject_id"].nunique(),
                    "target_events": int(group["event_indicator"].sum()),
                    "competing_deaths": int(
                        group["competing_death_indicator"].sum()
                    ),
                    "ipcw_observed": int(group["ipcw_observed"].sum()),
                    "base_estimate": point[base_key],
                    "augmented_estimate": point[augmented_key],
                    "delta_definition": (
                        "augmented-base"
                        if metric != "ipcw_brier"
                        else "base-augmented; positive is improvement"
                    ),
                    "delta": point[delta_key],
                    "delta_ci_lower": intervals[delta_key][0],
                    "delta_ci_upper": intervals[delta_key][1],
                    "bootstrap_replicates": replicates,
                    "bootstrap_attempts": attempts,
                    "bootstrap_unit": "participant resampled within fixed outer fold",
                    "bootstrap_scope": "paired conditional on fixed OOF model and censor fits",
                }
            )
        if int(horizon) == 12 and comparison == "age_sex_density_plus_gap":
            calibrated = group.loc[group["ipcw_weight"].gt(0)].copy()
            calibrated["risk_decile"] = pd.qcut(
                calibrated["augmented_probability"],
                q=10,
                labels=False,
                duplicates="drop",
            )
            for risk_decile, values in calibrated.groupby("risk_decile", sort=True):
                weight = values["ipcw_weight"].to_numpy(dtype=float)
                calibration_rows.append(
                    {
                        "organ_system": organ,
                        "horizon_months": 12,
                        "model": "age_sex_density_plus_gap",
                        "risk_decile": int(risk_decile) + 1,
                        "n": len(values),
                        "predicted_risk_mean": np.average(
                            values["augmented_probability"], weights=weight
                        ),
                        "ipcw_observed_risk": np.average(
                            values["ipcw_outcome"], weights=weight
                        ),
                        "weight_sum": weight.sum(),
                    }
                )
    metrics = pd.DataFrame(metrics_rows)
    calibration = pd.DataFrame(calibration_rows)
    aj_rows = []
    for organ, group in data["incident"].loc[
        data["incident"]["horizon_months"].eq(12)
    ].groupby("organ_system", sort=False):
        cause = (
            group["event_indicator"].to_numpy(dtype=np.int8)
            + 2 * group["competing_death_indicator"].to_numpy(dtype=np.int8)
        )
        curve = v8.aalen_johansen(
            group["observed_time_days"].to_numpy(dtype=float), cause
        )
        for day in np.linspace(0.0, v8.months_to_days(12), 25):
            eligible = curve.loc[curve["time_days"].le(day)]
            row = eligible.iloc[-1] if len(eligible) else None
            aj_rows.append(
                {
                    "organ_system": organ,
                    "horizon_months": 12,
                    "time_days": day,
                    "target_cif": 0.0 if row is None else float(row["target_cif"]),
                    "death_cif": 0.0 if row is None else float(row["death_cif"]),
                    "n": group["subject_id"].nunique(),
                }
            )
    aj = pd.DataFrame(aj_rows)
    outputs = {
        FIG13 / "Fig13D_paired_incremental_risk_metrics.csv": metrics,
        FIG13 / "Fig13F_risk_calibration_deciles.csv": calibration,
        FIG13 / "Fig13F_aalen_johansen_overall.csv": aj,
        FIG13 / "MIMIC_risk_model_failures.csv": pd.DataFrame(failure_rows),
    }
    for path, frame in outputs.items():
        write_csv(path, frame)
    receipt = {
        "schema_version": "revision0710.external_validation.core_0716.risk_statistics.v1",
        "status": "PAIRED_OOF_RISK_STATISTICS_COMPLETE",
        "created_at": now_iso(),
        "bootstrap_replicates": replicates,
        "oof_rows": len(oof),
        "oof_sha256": sha256_file(oof_path),
        "outputs": {
            str(path.relative_to(ROOT)): {
                "rows": len(frame),
                "sha256": sha256_file(path),
            }
            for path, frame in outputs.items()
        },
    }
    write_json(STATS / "RISK_STATISTICS_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("basic", "risk"))
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    args = parser.parse_args()
    if args.stage == "basic":
        compute_basic()
    else:
        compute_risk(args.bootstrap_replicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
