from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "artifacts_v8"
AUTHORIZATION = HERE / "EXTERNAL_VALIDATION_FORMAL_AUTHORIZATION_V8.json"
FORMAL_LOCK = HERE / ".EXTERNAL_VALIDATION_FORMAL_V8_ONCE.lock"
CONSUMED_AUTHORIZATION = HERE / "EXTERNAL_VALIDATION_FORMAL_AUTHORIZATION_V8.CONSUMED.json"
FORMAL_TARGET = ROOT / "Revision-0710/06_figures_and_statistics/04_external_validation/source_data_v1"
FORMAL_TAG = "source_data_v1"
FORMAL_SEED = 7_102_026
FORMAL_BOOTSTRAP_REPLICATES = 200
HASH_FOLD_SALT = "revision0710-external-v3"
MIMIC_PRIMARY_VARIANT = "mimic_primary_hospital_icd_only_explicit_unit_phenotype"
MIMIC_BARE_OMR_SENSITIVITY = "mimic_sensitivity_hospital_icd_only_bare_omr_unit_assumptions"
MIMIC_RX_SENSITIVITY = "mimic_sensitivity_hospital_icd_plus_inpatient_prescription_proxy"
NHANES_PRIMARY_VARIANT = "nhanes_primary_phenotype_only_label_blind"
MIMIC_VARIANTS = (
    MIMIC_PRIMARY_VARIANT,
    MIMIC_BARE_OMR_SENSITIVITY,
    MIMIC_RX_SENSITIVITY,
)
ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
OUTPUT_TABLE_PATHS = {
    "per_checkpoint_predictions": "shared/04_inference/per_checkpoint_predictions.parquet",
    "ensemble_predictions": "shared/04_inference/ensemble_predictions.parquet",
    "external_vocabulary_coverage": "shared/03_cache_audit/external_vocabulary_coverage.csv",
    "zero_shot_age_metrics": "shared/05_age_metrics/zero_shot_age_metrics.csv",
    "mimic_cohort_flow": "01_mimic/02_cohort_and_endpoints/cohort_flow.csv",
    "mimic_feature_coverage": "01_mimic/01_mapping/feature_coverage.csv",
    "mimic_incident_endpoints": "01_mimic/06_incident_risk/incident_endpoints.parquet",
    "mimic_recurrent_endpoints": "01_mimic/07_recurrent_sensitivity/recurrent_endpoints.parquet",
    "mimic_predictive_metrics": "01_mimic/06_incident_risk/predictive_metrics.csv",
    "mimic_full_cohort_associations": "01_mimic/06_incident_risk/full_cohort_associations.csv",
    "mimic_incident_competing_risk_curves": "01_mimic/06_incident_risk/competing_risk_curves.csv",
    "mimic_recurrent_sensitivity_metrics": "01_mimic/07_recurrent_sensitivity/recurrent_sensitivity_metrics.csv",
    "mimic_density_diagnostics": "01_mimic/08_density_sensitivity/density_diagnostics.csv",
    "mimic_density_adjusted_risk_sensitivity": "01_mimic/08_density_sensitivity/density_adjusted_risk_sensitivity.csv",
    "mimic_cox_ph_diagnostics": "01_mimic/09_cox_diagnostics/mimic_cox_ph_diagnostics.csv",
    "nhanes_feature_coverage": "02_nhanes/01_mapping/feature_coverage.csv",
    "nhanes_release_slot_qc": "02_nhanes/01_mapping/release_slot_qc.csv",
    "nhanes_questionnaire_associations": "02_nhanes/07_cross_sectional_disease/questionnaire_associations.csv",
    "nhanes_local_recalibration_sensitivity": "02_nhanes/06_local_recalibration_sensitivity/metrics.csv",
    "nhanes_release_survey_sensitivity": "02_nhanes/05_age_metrics/release_survey_sensitivity.csv",
}


class ExternalExecutionError(RuntimeError):
    pass


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalExecutionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_closed_object)
    if not isinstance(value, dict):
        raise ExternalExecutionError(f"expected JSON object: {path}")
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_hash_fold(identifier: object) -> int:
    if identifier is None or (isinstance(identifier, float) and math.isnan(identifier)):
        raise ValueError("participant identifier cannot be missing")
    clean = " ".join(str(identifier).split())
    if not clean:
        raise ValueError("participant identifier cannot be empty")
    payload = f"{HASH_FOLD_SALT}|{clean}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 5


def encode_mimic_sex(value: object) -> int:
    if value == "M":
        return 1
    if value == "F":
        return 0
    raise ValueError("MIMIC sex must be exactly M or F")


def encode_nhanes_sex(value: object) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("NHANES RIAGENDR must be numeric 1 or 2") from exc
    if not np.isfinite(numeric):
        raise ValueError("NHANES RIAGENDR cannot be missing")
    if numeric == 1.0:
        return 1
    if numeric == 2.0:
        return 0
    raise ValueError("NHANES RIAGENDR must be exactly 1 or 2")


def mimic_integer_age(anchor_age: object, anchor_year: object, index_time: object) -> int:
    age = int(anchor_age)
    year = int(anchor_year)
    if float(anchor_age) != age or float(anchor_year) != year:
        raise ValueError("MIMIC anchor age/year must have integer resolution")
    if age == 91:
        raise ValueError("MIMIC anchor_age=91 top-code is excluded")
    timestamp = pd.Timestamp(index_time)
    if pd.isna(timestamp):
        raise ValueError("MIMIC index time is missing")
    result = age + int(timestamp.year) - year
    if result < 18:
        raise ValueError("MIMIC index age is not adult")
    return result


def months_to_days(months: int) -> float:
    if months not in {6, 12}:
        raise ValueError("V8 authorizes only 6 and 12 month horizons")
    return {6: 182.5, 12: 365.0}[months]


def _stable_event_sort(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "participant_id",
        "event_time",
        "source",
        "event_type_id",
        "canonical_code",
        "canonical_description",
        "source_row_number",
        "landmark_time",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"timeline columns missing: {sorted(missing)}")
    value = frame.copy()
    value["event_time"] = pd.to_datetime(value["event_time"], errors="coerce")
    value["landmark_time"] = pd.to_datetime(value["landmark_time"], errors="coerce")
    if value[["event_time", "landmark_time"]].isna().any().any():
        raise ValueError("timeline contains missing timestamp")
    if value["canonical_code"].isna().any() or value["canonical_description"].isna().any():
        raise ValueError("timeline contains missing canonical token")
    value["code_bytes"] = value["canonical_code"].astype(str).map(lambda x: x.encode("utf-8"))
    value["description_bytes"] = value["canonical_description"].astype(str).map(lambda x: x.encode("utf-8"))
    return value.sort_values(
        [
            "participant_id",
            "event_time",
            "event_type_id",
            "code_bytes",
            "description_bytes",
            "source_row_number",
        ],
        kind="mergesort",
    )


def prepare_variant_timeline(frame: pd.DataFrame, input_variant: str) -> pd.DataFrame:
    """Apply the exact source, stable-sort, strict-preindex, and dedup rules."""
    if input_variant not in MIMIC_VARIANTS:
        raise ValueError(f"unknown MIMIC input variant: {input_variant}")
    ordered = _stable_event_sort(frame)
    if (ordered["event_time"] >= ordered["landmark_time"]).any():
        raise ValueError("active event at or after landmark")
    allowed = {"hospital_icd"}
    if input_variant == MIMIC_RX_SENSITIVITY:
        allowed.add("inpatient_prescription_proxy")
    if not set(ordered["source"]).issubset(allowed):
        raise ValueError("event source is forbidden for input variant")
    dedup_key = [
        "participant_id",
        "event_time",
        "canonical_code",
        "canonical_description",
        "source",
    ]
    return ordered.drop_duplicates(dedup_key, keep="first").reset_index(drop=True)


def crop_variant_timeline(frame: pd.DataFrame, input_variant: str, max_len: int = 32) -> pd.DataFrame:
    ordered = prepare_variant_timeline(frame, input_variant)
    dedup_key = [
        "participant_id",
        "event_time",
        "canonical_code",
        "canonical_description",
        "source",
    ]
    selected: list[pd.DataFrame] = []
    for _, group in ordered.groupby("participant_id", sort=False):
        if input_variant != MIMIC_RX_SENSITIVITY:
            keep = group.tail(max_len)
        else:
            icd = group.loc[group["source"].eq("hospital_icd")].tail(8)
            rx = group.loc[group["source"].eq("inpatient_prescription_proxy")].tail(8)
            mandatory = pd.concat([icd, rx]).drop_duplicates(dedup_key, keep="first")
            remaining = group.loc[~group.index.isin(mandatory.index)].tail(max_len - len(mandatory))
            keep = pd.concat([mandatory, remaining]).sort_values(
                ["event_time", "event_type_id", "code_bytes", "description_bytes", "source_row_number"],
                kind="mergesort",
            )
        if len(keep) > max_len:
            raise AssertionError("variant crop exceeds L32")
        selected.append(keep)
    if not selected:
        return ordered.iloc[:0].copy()
    return pd.concat(selected, ignore_index=True)


def left_align_l32(selected: pd.DataFrame, participant_order: Sequence[str]) -> dict[str, np.ndarray]:
    arrays = {
        "event_type_id": np.zeros((len(participant_order), 32), dtype=np.int8),
        "event_valid_mask": np.zeros((len(participant_order), 32), dtype=bool),
    }
    grouped = {str(key): value for key, value in selected.groupby("participant_id", sort=False)}
    for row_index, participant_id in enumerate(participant_order):
        group = grouped.get(str(participant_id))
        if group is None:
            continue
        count = len(group)
        arrays["event_type_id"][row_index, :count] = group["event_type_id"].to_numpy(dtype=np.int8)
        arrays["event_valid_mask"][row_index, :count] = True
        if arrays["event_valid_mask"][row_index, count:].any():
            raise AssertionError("L32 padding is not left aligned")
    return arrays


def validate_nhanes_padding(event_arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "event_code_id",
        "event_text_id",
        "event_relative_days",
        "event_type_id",
        "event_valid_mask",
    }
    if set(event_arrays) != required:
        raise ValueError("NHANES event array set mismatch")
    for name, value in event_arrays.items():
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[1] != 32:
            raise ValueError(f"NHANES {name} is not N x 32")
        if np.any(array):
            raise ValueError("NHANES primary L32 must contain padding only")


def require_immutable_embedding_prefix(
    original_prefix_bytes: bytes, extended_table_bytes: bytes
) -> None:
    if not isinstance(original_prefix_bytes, bytes) or not isinstance(
        extended_table_bytes, bytes
    ):
        raise TypeError("embedding prefix comparison requires bytes")
    if not extended_table_bytes.startswith(original_prefix_bytes):
        raise ValueError("original UKB embedding prefix bytes changed")


def validate_mimic_endpoint_rows(frame: pd.DataFrame) -> None:
    required = {
        "has_any_preindex_icd9",
        "endpoint_code_system",
        "horizon_months",
        "input_variant",
    }
    if not required.issubset(frame.columns):
        raise ValueError("MIMIC endpoint eligibility columns missing")
    if frame["has_any_preindex_icd9"].astype(bool).any():
        raise ValueError("pre-index ICD9 row entered a MIMIC endpoint table")
    if not frame["endpoint_code_system"].eq("ICD10").all():
        raise ValueError("MIMIC endpoint code system must be ICD10 only")
    if not set(frame["horizon_months"].astype(int)).issubset({6, 12}):
        raise ValueError("MIMIC endpoint horizon must be 6 or 12 months")
    if not set(frame["input_variant"]).issubset(set(MIMIC_VARIANTS)):
        raise ValueError("MIMIC endpoint input variant is not frozen")


def adjudicate_endpoint(
    *,
    target_days: float | None,
    death_days: float | None,
    observation_end_days: float,
    death_cap_days: float,
    horizon_months: int,
) -> dict[str, Any]:
    horizon = months_to_days(horizon_months)
    target = np.inf if target_days is None else float(target_days)
    death = np.inf if death_days is None else float(death_days)
    observation = float(observation_end_days)
    death_cap = float(death_cap_days)
    if observation < 0 or death_cap < 0:
        raise ValueError("observation/death cap precedes index")
    terminal = min(target, death, observation, death_cap, horizon)
    if death <= target and death <= observation and death <= death_cap and death <= horizon:
        return {"horizon_days": horizon, "observed_time_days": terminal, "observed_at_horizon": True, "event_indicator": 0, "competing_death_indicator": 1, "censor_reason": "death"}
    if target < death and target <= observation and target <= death_cap and target <= horizon:
        return {"horizon_days": horizon, "observed_time_days": terminal, "observed_at_horizon": True, "event_indicator": 1, "competing_death_indicator": 0, "censor_reason": "target"}
    if observation <= death_cap and observation < horizon:
        reason = "subject_observation_end"
    elif death_cap < horizon:
        reason = "death_ascertainment_cap"
    else:
        reason = "horizon"
    return {"horizon_days": horizon, "observed_time_days": terminal, "observed_at_horizon": reason == "horizon", "event_indicator": 0, "competing_death_indicator": 0, "censor_reason": reason}


def aalen_johansen(time: np.ndarray, cause: np.ndarray) -> pd.DataFrame:
    duration = np.asarray(time, dtype=np.float64)
    event = np.asarray(cause, dtype=np.int8)
    if duration.shape != event.shape or np.any(duration < 0) or not set(np.unique(event)).issubset({0, 1, 2}):
        raise ValueError("invalid AJ input")
    survival = 1.0
    cif_target = 0.0
    cif_death = 0.0
    rows = []
    sorted_duration = np.sort(duration)
    event_points = np.sort(np.unique(duration[event != 0]))
    target_points, target_counts = np.unique(
        duration[event == 1], return_counts=True
    )
    death_points, death_counts = np.unique(duration[event == 2], return_counts=True)
    target_lookup = dict(zip(target_points.tolist(), target_counts.tolist()))
    death_lookup = dict(zip(death_points.tolist(), death_counts.tolist()))
    for point in event_points:
        at_risk = int(len(duration) - np.searchsorted(sorted_duration, point, side="left"))
        d_target = int(target_lookup.get(float(point), 0))
        d_death = int(death_lookup.get(float(point), 0))
        prior = survival
        cif_target += prior * d_target / at_risk
        cif_death += prior * d_death / at_risk
        survival *= 1.0 - (d_target + d_death) / at_risk
        rows.append({"time_days": float(point), "n_at_risk": at_risk, "target_events": d_target, "death_events": d_death, "survival": survival, "target_cif": cif_target, "death_cif": cif_death})
    return pd.DataFrame(rows)


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    duration = np.asarray(time, dtype=np.float64)
    observed = np.asarray(event, dtype=np.int8)
    score = np.asarray(risk, dtype=np.float64)
    if not (duration.shape == observed.shape == score.shape):
        raise ValueError("Harrell C arrays differ in shape")
    if np.any(~np.isfinite(duration)) or np.any(~np.isfinite(score)):
        raise ValueError("Harrell C requires finite time and risk")
    unique_score = np.unique(score)
    rank = np.searchsorted(unique_score, score) + 1
    tree = np.zeros(len(unique_score) + 1, dtype=np.float64)

    def add(position: int, value: float) -> None:
        while position < len(tree):
            tree[position] += value
            position += position & -position

    def prefix(position: int) -> float:
        total = 0.0
        while position > 0:
            total += tree[position]
            position -= position & -position
        return total

    order = np.argsort(-duration, kind="mergesort")
    concordant = 0.0
    comparable = 0.0
    total_later = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        point = duration[order[start]]
        while stop < len(order) and duration[order[stop]] == point:
            stop += 1
        for index in order[start:stop]:
            if observed[index] != 1:
                continue
            lower = prefix(int(rank[index]) - 1)
            through = prefix(int(rank[index]))
            concordant += lower + 0.5 * (through - lower)
            comparable += total_later
        for index in order[start:stop]:
            add(int(rank[index]), 1.0)
            total_later += 1.0
        start = stop
    if comparable == 0:
        raise ValueError("Harrell C not estimable without comparable target events")
    return concordant / comparable


def censor_km_left_limit(time: np.ndarray, censor_event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    duration = np.asarray(time, dtype=np.float64)
    censor = np.asarray(censor_event, dtype=np.int8)
    points = np.sort(np.unique(duration[censor == 1]))
    # Store post-jump values. _left_g then selects jumps strictly before the
    # query, which is exactly G(t-) and avoids the former double left shift.
    post_jump_values = []
    survival = 1.0
    for point in points:
        at_risk = np.sum(duration >= point)
        deaths = np.sum((duration == point) & (censor == 1))
        survival *= 1.0 - deaths / at_risk
        post_jump_values.append(survival)
    return points, np.asarray(post_jump_values, dtype=np.float64)


def _left_g(points: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(points, query, side="left") - 1
    result = np.ones(len(query), dtype=np.float64)
    valid = indices >= 0
    result[valid] = values[indices[valid]]
    return np.maximum(result, 0.05)


def _left_g_raw(points: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(points, query, side="left") - 1
    result = np.ones(len(query), dtype=np.float64)
    valid = indices >= 0
    result[valid] = values[indices[valid]]
    return result


def ipcw_binary_weights(
    time: np.ndarray,
    cause: np.ndarray,
    horizon_days: float,
    training_censor_time: np.ndarray,
    training_censor_event: np.ndarray,
    *,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]
]:
    duration = np.asarray(time, dtype=np.float64)
    status = np.asarray(cause, dtype=np.int8)
    points, values = censor_km_left_limit(training_censor_time, training_censor_event)
    case = (status == 1) & (duration <= horizon_days)
    death_control = (status == 2) & (duration <= horizon_days)
    horizon_control = duration >= horizon_days
    observed = case | death_control | horizon_control
    outcome = case.astype(np.float64)
    evaluation_time = np.where(horizon_control, horizon_days, duration)
    weights = np.zeros(len(duration), dtype=np.float64)
    raw_g = _left_g_raw(points, values, evaluation_time[observed])
    weights[observed] = 1.0 / np.maximum(raw_g, 0.05)
    training_g = _left_g_raw(
        points, values, np.minimum(training_censor_time, horizon_days)
    )
    positive_training_weights = 1.0 / np.maximum(training_g, 0.05)
    cap = float(np.quantile(positive_training_weights, 0.99, method="linear"))
    uncapped = weights.copy()
    weights = np.minimum(weights, cap)
    if not return_diagnostics:
        return outcome, weights, observed
    positive = weights[weights > 0]
    diagnostics: dict[str, float | int] = {
        "g_min_raw": float(min(np.min(raw_g, initial=1.0), np.min(training_g, initial=1.0))),
        "g_floor": 0.05,
        "weight_cap_p99_training": cap,
        "n_test": int(len(duration)),
        "n_observed_at_horizon": int(observed.sum()),
        "n_censored_before_horizon": int((~observed).sum()),
        "n_weights_capped": int(np.sum(uncapped > cap)),
        "effective_sample_size": (
            float(np.square(positive.sum()) / np.square(positive).sum())
            if len(positive) and np.square(positive).sum() > 0
            else 0.0
        ),
    }
    return outcome, weights, observed, diagnostics


def weighted_auc(outcome: np.ndarray, risk: np.ndarray, weight: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=np.float64)
    score = np.asarray(risk, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(score) & np.isfinite(w) & (w > 0)
    y, score, w = y[valid], score[valid], w[valid]
    case_weight = w * (y == 1)
    control_weight = w * (y == 0)
    denominator = float(case_weight.sum() * control_weight.sum())
    if denominator <= 0:
        raise ValueError("weighted AUC has no weighted cases or controls")
    order = np.argsort(score, kind="mergesort")
    numerator = 0.0
    cumulative_controls = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        point = score[order[start]]
        while stop < len(order) and score[order[stop]] == point:
            stop += 1
        indices = order[start:stop]
        cases_at_point = float(case_weight[indices].sum())
        controls_at_point = float(control_weight[indices].sum())
        numerator += cases_at_point * (
            cumulative_controls + 0.5 * controls_at_point
        )
        cumulative_controls += controls_at_point
        start = stop
    return numerator / denominator


def weighted_brier(
    outcome: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
    *,
    normalization_n: int | None = None,
) -> float:
    y = np.asarray(outcome, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    denominator = len(y) if normalization_n is None else int(normalization_n)
    if np.any((p < 0) | (p > 1)) or w.sum() <= 0 or denominator <= 0:
        raise ValueError("invalid weighted Brier input")
    return float(np.sum(w * np.square(y - p)) / denominator)


def fit_crossfit_age_calibration(predicted_age: np.ndarray, sex_male: np.ndarray, chronological_age: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_age, dtype=np.float64)
    sex = np.asarray(sex_male, dtype=np.float64)
    age = np.asarray(chronological_age, dtype=np.float64)
    design = np.column_stack([np.ones(len(age)), predicted, sex])
    if len(age) < 4 or np.linalg.matrix_rank(design) != 3:
        raise ValueError("local age calibration design is singular/insufficient")
    return np.linalg.lstsq(design, age, rcond=None)[0]


def benjamini_hochberg(
    p_values: Sequence[float], *, family_size: int | None = None
) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("BH p-values must be finite probabilities")
    multiplicity = len(values) if family_size is None else int(family_size)
    if multiplicity < len(values) or multiplicity <= 0:
        raise ValueError("BH family size cannot be smaller than observed p-values")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * multiplicity / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.minimum(ranked, 1.0)
    return result


def _load_verifier() -> Any:
    path = HERE / "verify_external_validation_prefreeze_v8.py"
    spec = importlib.util.spec_from_file_location("external_validation_v8_independent_verifier", path)
    if spec is None or spec.loader is None:
        raise ExternalExecutionError("cannot load independent verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_formal_authorization() -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    # Every operation below validation is local to this V8 directory/repository.
    # No external source path is stat'ed, opened, or rehashed until the exact
    # authorization payload has passed and has been atomically consumed.
    resume = CONSUMED_AUTHORIZATION.is_file() and FORMAL_LOCK.is_file()
    if resume and AUTHORIZATION.exists():
        raise PermissionError(
            "resume state is ambiguous because original and consumed authorizations coexist"
        )
    if not AUTHORIZATION.is_file() and not resume:
        raise PermissionError(f"formal authorization is absent: {AUTHORIZATION}")
    lock_descriptor: int | None = None
    lock_owned = False
    consumed = False
    try:
        if not resume:
            lock_descriptor = os.open(
                FORMAL_LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            )
            lock_owned = True
            with os.fdopen(
                lock_descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                lock_descriptor = None
                handle.write("V8 formal authorization validation in progress\n")
        if FORMAL_TARGET.exists() or list(
            FORMAL_TARGET.parent.glob(f".{FORMAL_TARGET.name}.staging.*")
        ):
            raise FileExistsError("formal target or staging target already exists")
        verifier = _load_verifier()
        contract = verifier.read_json(
            ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
        )
        authorization_path = CONSUMED_AUTHORIZATION if resume else AUTHORIZATION
        authorization = verifier.read_json(authorization_path)
        verifier.validate_formal_authorization_payload(
            authorization,
            artifact_dir=ARTIFACT_DIR,
            repo_root=ROOT,
            contract=contract,
        )
        publication = authorization["formal_publication"]
        if (
            publication["mode"] != "FORMAL_ONCE_CREATE_ONLY"
            or publication["executor_name"] != "formal_executor"
        ):
            raise PermissionError("authorization mode/executor mismatch")
        if authorization["review_freeze"] != {
            "independent_review_status": "PASS",
            "contract_bytes_frozen": True,
            "all_authorized_hashes_exact": True,
        }:
            raise PermissionError("independent review/freeze gate is not closed PASS")
        from v8_execution_guard import verify_local_code_bindings

        # This is deliberately before source-registry verification or any
        # external source stat/read/rehash.
        verify_local_code_bindings(
            repo_root=ROOT,
            bindings=contract["bindings"]["code_artifacts"],
        )
        authorization_sha256 = sha256_file(authorization_path)
        consumed_path = AUTHORIZATION.with_name(
            f"{AUTHORIZATION.stem}.CONSUMED.json"
        )
        if not resume:
            if consumed_path.exists():
                raise FileExistsError("formal authorization was already consumed")
            os.rename(AUTHORIZATION, consumed_path)
            consumed = True
        else:
            consumed = True
        # External file access is authorized only after the atomic rename.
        verifier.verify_artifacts(
            ARTIFACT_DIR,
            ROOT,
            rehash_external=True,
            formal_runtime_state=True,
        )
        return contract, authorization, consumed_path, authorization_sha256
    except Exception:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        # Invalid/unfrozen payloads are repairable and consume nothing. Once a
        # valid payload is consumed, the lock deliberately remains fail-closed.
        if lock_owned and not consumed:
            FORMAL_LOCK.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class FormalContext:
    contract: dict[str, Any]
    authorization: dict[str, Any]
    stage: Path


class FormalPipeline:
    """Create-once executor for the exact V8 cache, inference, and 20-table contract."""

    def __init__(self, context: FormalContext):
        self.context = context
        self.registry = read_json(ARTIFACT_DIR / "SOURCE_REGISTRY_V8.json")
        self.schemas = read_json(ARTIFACT_DIR / "OUTPUT_SCHEMAS_V8.json")
        self.sources = {row["source_id"]: Path(row["absolute_path"]) for row in self.registry["external_sources"]}
        self.environment_lock_path = ARTIFACT_DIR / "ENVIRONMENT_LOCK_V8.json"
        self.environment_lock = read_json(self.environment_lock_path)
        self.environment_lock_sha256 = sha256_file(self.environment_lock_path)

    def run(self) -> dict[str, Any]:
        cache_sources = {
            row["source_id"]: row["absolute_path"]
            for row in self.registry["external_sources"]
            if row["role"] != "label_only"
        }
        cache = self._run_worker_phase(
            "01_cache", "statistics", {"sources": cache_sources}
        )
        predictions = self._run_worker_phase(
            "02_inference",
            "inference",
            {"cache": cache},
            worker_artifact_dir=Path(cache["inference_artifact_dir"]),
        )
        sealed = self._run_worker_phase(
            "03_prediction_seal",
            "statistics",
            {"predictions": predictions},
            worker_artifact_dir=Path(cache["inference_artifact_dir"]),
        )
        endpoints = self._run_worker_phase(
            "04_outcomes",
            "statistics",
            {
                "sources": {
                    row["source_id"]: row["absolute_path"]
                    for row in self.registry["external_sources"]
                },
                "sealed": sealed,
            },
        )
        table_bundle = self._run_worker_phase(
            "05_tables",
            "statistics",
            {
                "cache": cache,
                "sealed": sealed,
                "endpoints": endpoints,
                "bootstrap_seed": FORMAL_SEED,
                "bootstrap_replicates": FORMAL_BOOTSTRAP_REPLICATES,
                "output_paths": OUTPUT_TABLE_PATHS,
            },
        )
        table_stage = table_bundle.pop("_phase_output_root")
        independent_recompute = self._run_worker_phase(
            "06_independent_recompute",
            "statistics",
            {
                "cache": cache,
                "sealed": sealed,
                "endpoints": endpoints,
                "bootstrap_seed": FORMAL_SEED,
                "bootstrap_replicates": FORMAL_BOOTSTRAP_REPLICATES,
                "output_paths": OUTPUT_TABLE_PATHS,
                "table_stage": table_stage,
            },
        )
        independent_recompute.pop("_phase_output_root", None)
        return {
            "cache_manifest_sha256": sha256_file(cache["manifest"]),
            "prediction_seal_sha256": sha256_file(sealed["receipt"]),
            "table_count": 20,
            "_table_output_root": table_stage,
            **table_bundle,
            **independent_recompute,
        }

    def _run_worker_phase(
        self,
        stage_name: str,
        role: str,
        payload: Mapping[str, Any],
        *,
        worker_artifact_dir: Path | None = None,
    ) -> dict[str, Any]:
        from v8_execution_guard import run_or_resume_stage

        phase = self.context.stage / stage_name
        phase.mkdir(parents=True, exist_ok=True)
        phase_spec = {
            "schema_version": "revision0710.external_validation.phase_spec.v8",
            "stage_name": stage_name,
            "environment_role": role,
            "environment_lock_sha256": self.environment_lock_sha256,
            "contract_sha256": sha256_file(
                ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
            ),
            "worker_artifact_dir": str(
                (worker_artifact_dir or ARTIFACT_DIR).resolve()
            ),
            "payload": payload,
        }
        payload_sha = hashlib.sha256(
            json.dumps(
                phase_spec,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        input_bindings = [
            {
                "name": "phase_spec",
                "canonical_payload_sha256": payload_sha,
            }
        ]

        def runner(stage_root: Path) -> Sequence[Path]:
            phase_result = stage_root / "PHASE_RESULT_V8.json"
            if phase_result.exists():
                return [path for path in stage_root.rglob("*") if path.is_file()]
            attempts = stage_root / "attempts"
            attempts.mkdir(parents=True, exist_ok=True)
            attempt_number = len(
                [path for path in attempts.iterdir() if path.is_dir()]
            ) + 1
            attempt = attempts / f"attempt_{attempt_number:04d}"
            attempt.mkdir(parents=True, exist_ok=False)
            context_path = attempt / "WORKER_CONTEXT_V8.json"
            context_payload = {
                "schema_version": "revision0710.external_validation.worker_context.v8",
                "stage_name": stage_name,
                "environment_role": role,
                "environment_lock_path": str(self.environment_lock_path.resolve()),
                "environment_lock_sha256": self.environment_lock_sha256,
                "root": str(ROOT.resolve()),
                "artifact_dir": str(
                    (worker_artifact_dir or ARTIFACT_DIR).resolve()
                ),
                "contract_path": str(
                    (
                        ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
                    ).resolve()
                ),
                "stage": str(attempt.resolve()),
                "payload": payload,
            }
            write_json_exclusive(context_path, context_payload)
            executable = Path(self.environment_lock[role]["python_executable"])
            command = [
                str(executable),
                str(HERE / "execute_external_validation_v8_runtime.py"),
                "--worker-stage",
                stage_name,
                "--context-json",
                str(context_path),
            ]
            worker_environment = os.environ.copy()
            worker_environment.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "PYTHONHASHSEED": "0",
                }
            )
            subprocess.run(
                command,
                cwd=HERE,
                check=True,
                env=worker_environment,
            )
            result = read_json(attempt / "WORKER_RESULT_V8.json")
            if (
                result.get("status")
                != "WORKER_STAGE_COMPLETE_PENDING_ORCHESTRATOR_RECEIPT"
                or result.get("stage_name") != stage_name
                or result.get("environment_role") != role
                or result.get("environment_lock_sha256")
                != self.environment_lock_sha256
            ):
                raise ExternalExecutionError(f"worker result mismatch: {stage_name}")
            write_json_exclusive(
                phase_result,
                {
                    "schema_version": "revision0710.external_validation.phase_result.v8",
                    "status": "PHASE_COMPLETE_FROM_VERIFIED_ATTEMPT",
                    "stage_name": stage_name,
                    "successful_attempt": attempt.name,
                    "failed_attempts_preserved": attempt_number - 1,
                    "value": result["value"],
                },
            )
            return [path for path in stage_root.rglob("*") if path.is_file()]

        run_or_resume_stage(
            stage_name=stage_name,
            stage_root=phase,
            input_bindings=input_bindings,
            environment_sha256=self.environment_lock_sha256,
            runner=runner,
        )
        result = read_json(phase / "PHASE_RESULT_V8.json")
        value = dict(result["value"])
        value["_phase_output_root"] = str(
            (phase / "attempts" / result["successful_attempt"]).resolve()
        )
        return value

    def build_label_blind_cache(self) -> dict[str, Path]:
        from execute_external_validation_v8_runtime import build_label_blind_cache

        return build_label_blind_cache(
            root=ROOT,
            artifact_dir=ARTIFACT_DIR,
            stage=self.context.stage,
            sources=self.sources,
            contract=self.context.contract,
        )

    def run_five_checkpoint_inference(self, cache: Mapping[str, Path]) -> dict[str, Path]:
        from execute_external_validation_v8_runtime import run_five_checkpoint_inference

        return run_five_checkpoint_inference(
            root=ROOT,
            artifact_dir=ARTIFACT_DIR,
            stage=self.context.stage,
            cache=cache,
            contract=self.context.contract,
        )

    def seal_prediction_tables(self, predictions: Mapping[str, Path]) -> dict[str, Path]:
        from execute_external_validation_v8_runtime import seal_prediction_tables

        return seal_prediction_tables(
            stage=self.context.stage,
            predictions=predictions,
            contract=self.context.contract,
        )

    def build_outcomes_after_prediction_seal(self, sealed: Mapping[str, Path]) -> dict[str, Path]:
        from execute_external_validation_v8_runtime import build_outcomes_after_prediction_seal

        return build_outcomes_after_prediction_seal(
            artifact_dir=ARTIFACT_DIR,
            stage=self.context.stage,
            sources=self.sources,
            prediction_seal=sealed,
            contract=self.context.contract,
        )

    def compute_all_twenty_tables(
        self,
        cache: Mapping[str, Path],
        sealed: Mapping[str, Path],
        endpoints: Mapping[str, Path],
    ) -> dict[str, pd.DataFrame]:
        from execute_external_validation_v8_runtime import compute_all_twenty_tables

        return compute_all_twenty_tables(
            artifact_dir=ARTIFACT_DIR,
            cache=cache,
            sealed=sealed,
            endpoints=endpoints,
            contract=self.context.contract,
            bootstrap_seed=FORMAL_SEED,
            bootstrap_replicates=FORMAL_BOOTSTRAP_REPLICATES,
        )

    def validate_and_write_tables(
        self, tables: Mapping[str, pd.DataFrame]
    ) -> dict[str, Any]:
        from execute_external_validation_v8_runtime import validate_and_write_tables

        return validate_and_write_tables(
            stage=self.context.stage,
            tables=tables,
            schemas=self.schemas,
            output_paths=OUTPUT_TABLE_PATHS,
        )


def _read_published_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ExternalExecutionError(f"unsupported published table: {path}")


def validate_completed_publish_bundle(
    bundle: Path,
    *,
    work_stage: Path,
    authorization_sha256: str,
    consumed_authorization: Path,
) -> dict[str, Any]:
    """Rehash and structurally validate every byte in a resumable publish bundle."""
    if not bundle.is_dir():
        raise ExternalExecutionError(f"completed publish bundle is absent: {bundle}")
    phase_names = (
        "01_cache",
        "02_inference",
        "03_prediction_seal",
        "04_outcomes",
        "05_tables",
        "06_independent_recompute",
    )
    expected_files = {
        *OUTPUT_TABLE_PATHS.values(),
        "FORMAL_TABLE_MANIFEST_V8.json",
        "INDEPENDENT_TWENTY_TABLE_RECOMPUTE_V8.json",
        "FORMAL_EXECUTION_RECEIPT.json",
        *{f"stage_receipts/{name}.json" for name in phase_names},
    }
    observed_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise ExternalExecutionError(
            "completed publish bundle file universe mismatch: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )

    contract_path = ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"
    contract_sha256 = sha256_file(contract_path)
    executor_sha256 = sha256_file(Path(__file__).resolve())
    receipt_path = bundle / "FORMAL_EXECUTION_RECEIPT.json"
    receipt = read_json(receipt_path)
    required_receipt = {
        "status": "FORMAL_EXECUTION_COMPLETE_CREATE_ONCE",
        "authorization_sha256": authorization_sha256,
        "consumed_authorization_path": str(consumed_authorization.resolve()),
        "formal_once_lock_path": str(FORMAL_LOCK.resolve()),
        "contract_sha256": contract_sha256,
        "executor_sha256": executor_sha256,
        "tag": FORMAL_TAG,
        "table_count": 20,
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise ExternalExecutionError(f"formal receipt binding mismatch: {key}")

    schemas = read_json(ARTIFACT_DIR / "OUTPUT_SCHEMAS_V8.json")
    schema_by_id = {record["table_id"]: record for record in schemas["tables"]}
    manifest_path = bundle / "FORMAL_TABLE_MANIFEST_V8.json"
    manifest = read_json(manifest_path)
    if (
        list(manifest)
        != ["schema_version", "status", "table_count", "tables"]
        or manifest.get("schema_version")
        != "revision0710.external_validation.formal_table_manifest.v8"
        or manifest.get("status")
        != "TWENTY_TABLES_REHASHED_FORMULA_TOLERANCE_CROSSHASH_VALIDATED"
        or manifest.get("table_count") != 20
    ):
        raise ExternalExecutionError("formal table manifest header is not closed V8")
    if receipt.get("table_manifest_sha256") != sha256_file(manifest_path):
        raise ExternalExecutionError("formal receipt does not bind table manifest bytes")
    expected_table_ids = list(OUTPUT_TABLE_PATHS)
    if [record.get("table_id") for record in manifest["tables"]] != expected_table_ids:
        raise ExternalExecutionError("formal table manifest ordered table universe mismatch")
    if receipt.get("table_files") != manifest["tables"]:
        raise ExternalExecutionError("formal receipt/table manifest records differ")

    table_hashes: dict[str, str] = {}
    table_rows: dict[str, int] = {}
    for record in manifest["tables"]:
        expected_record_keys = [
            "table_id",
            "relative_path",
            "bytes",
            "sha256",
            "rows",
            "columns",
            "schema_formula_tolerance_sha256",
        ]
        if list(record) != expected_record_keys:
            raise ExternalExecutionError("formal table manifest record is open/reordered")
        table_id = str(record["table_id"])
        relative_path = OUTPUT_TABLE_PATHS[table_id]
        if record["relative_path"] != relative_path:
            raise ExternalExecutionError(f"published table path mismatch: {table_id}")
        path = bundle / relative_path
        if (
            path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ExternalExecutionError(f"published table byte/hash mismatch: {table_id}")
        frame = _read_published_table(path)
        if len(frame) != int(record["rows"]) or list(frame.columns) != record["columns"]:
            raise ExternalExecutionError(f"published table row/column mismatch: {table_id}")
        schema_contract = [
            {
                "name": column["name"],
                "formula": column["formula"],
                "tolerance": column["tolerance"],
            }
            for column in schema_by_id[table_id]["columns"]
        ]
        formula_sha256 = hashlib.sha256(
            json.dumps(
                schema_contract,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        if formula_sha256 != record["schema_formula_tolerance_sha256"]:
            raise ExternalExecutionError(f"published table formula binding mismatch: {table_id}")
        table_hashes[table_id] = str(record["sha256"])
        table_rows[table_id] = int(record["rows"])

    independent_path = bundle / "INDEPENDENT_TWENTY_TABLE_RECOMPUTE_V8.json"
    independent = read_json(independent_path)
    if (
        independent.get("status")
        != "ALL_TWENTY_TABLES_RECOMPUTED_FROM_SEALED_UPSTREAM_IN_FRESH_PROCESS"
        or independent.get("fresh_process") is not True
        or independent.get("table_count") != 20
        or [record.get("table_id") for record in independent.get("tables", [])]
        != expected_table_ids
    ):
        raise ExternalExecutionError("independent recompute receipt is incomplete")
    if receipt.get("independent_recompute_receipt_sha256") != sha256_file(
        independent_path
    ):
        raise ExternalExecutionError("formal receipt does not bind independent recompute")
    for record in independent["tables"]:
        table_id = str(record["table_id"])
        if (
            record.get("formal_table_sha256") != table_hashes[table_id]
            or int(record.get("rows", -1)) != table_rows[table_id]
            or record.get("recomputed_from_upstream") is not True
        ):
            raise ExternalExecutionError(
                f"independent recompute/table mismatch: {table_id}"
            )

    environment_sha256 = sha256_file(ARTIFACT_DIR / "ENVIRONMENT_LOCK_V8.json")
    stage_receipt_hashes: dict[str, str] = {}
    stage_output_records: dict[str, dict[str, dict[str, Any]]] = {}
    for phase_name in phase_names:
        copied = bundle / "stage_receipts" / f"{phase_name}.json"
        original = work_stage / phase_name / "STAGE_COMPLETION_V8.json"
        if not original.is_file() or sha256_file(copied) != sha256_file(original):
            raise ExternalExecutionError(f"published stage receipt mismatch: {phase_name}")
        phase_receipt = read_json(copied)
        if (
            list(phase_receipt)
            != [
                "schema_version",
                "status",
                "stage_name",
                "environment_lock_sha256",
                "input_bindings",
                "outputs",
            ]
            or phase_receipt.get("schema_version")
            != "revision0710.external_validation.stage_receipt.v8"
            or phase_receipt.get("status")
            != "STAGE_COMPLETE_CREATE_ONCE_REUSABLE"
            or phase_receipt.get("stage_name") != phase_name
            or phase_receipt.get("environment_lock_sha256") != environment_sha256
        ):
            raise ExternalExecutionError(f"published stage receipt header mismatch: {phase_name}")
        phase_root = work_stage / phase_name
        phase_root_resolved = phase_root.resolve()
        output_records: dict[str, dict[str, Any]] = {}
        for output in phase_receipt["outputs"]:
            if list(output) != ["relative_path", "bytes", "sha256"]:
                raise ExternalExecutionError(
                    f"published stage output record is open/reordered: {phase_name}"
                )
            relative_path = output["relative_path"]
            if (
                not isinstance(relative_path, str)
                or Path(relative_path).is_absolute()
                or Path(relative_path).as_posix() != relative_path
                or relative_path in output_records
            ):
                raise ExternalExecutionError(
                    f"published stage output path is unsafe/duplicated: {phase_name}"
                )
            output_path = (phase_root / relative_path).resolve()
            try:
                output_path.relative_to(phase_root_resolved)
            except ValueError as exc:
                raise ExternalExecutionError(
                    f"published stage output escapes phase root: {phase_name}"
                ) from exc
            if (
                not output_path.is_file()
                or output_path.stat().st_size != int(output["bytes"])
                or sha256_file(output_path) != output["sha256"]
            ):
                raise ExternalExecutionError(
                    f"published stage lineage output mismatch: {phase_name}"
                )
            output_records[relative_path] = output
        stage_output_records[phase_name] = output_records
        stage_receipt_hashes[phase_name] = sha256_file(copied)

    def successful_attempt(
        phase_name: str,
    ) -> tuple[Path, str, dict[str, Any], dict[str, dict[str, Any]]]:
        phase_root = (work_stage / phase_name).resolve()
        phase_result_relative = "PHASE_RESULT_V8.json"
        records = stage_output_records[phase_name]
        if phase_result_relative not in records:
            raise ExternalExecutionError(
                f"successful phase result is absent from stage receipt: {phase_name}"
            )
        phase_result = read_json(phase_root / phase_result_relative)
        if (
            list(phase_result)
            != [
                "schema_version",
                "status",
                "stage_name",
                "successful_attempt",
                "failed_attempts_preserved",
                "value",
            ]
            or phase_result.get("schema_version")
            != "revision0710.external_validation.phase_result.v8"
            or phase_result.get("status") != "PHASE_COMPLETE_FROM_VERIFIED_ATTEMPT"
            or phase_result.get("stage_name") != phase_name
        ):
            raise ExternalExecutionError(f"successful phase result mismatch: {phase_name}")
        attempt_name = phase_result.get("successful_attempt")
        attempts_root = (phase_root / "attempts").resolve()
        if not isinstance(attempt_name, str) or Path(attempt_name).name != attempt_name:
            raise ExternalExecutionError(f"unsafe successful attempt name: {phase_name}")
        attempt_root = (attempts_root / attempt_name).resolve()
        if attempt_root.parent != attempts_root or not attempt_root.is_dir():
            raise ExternalExecutionError(f"successful attempt root mismatch: {phase_name}")
        return attempt_root, attempt_name, phase_result, records

    def require_successful_attempt_output(
        phase_name: str,
        relative_within_attempt: str,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        attempt_root, attempt_name, phase_result, records = successful_attempt(
            phase_name
        )
        output_relative = (
            Path("attempts") / attempt_name / relative_within_attempt
        ).as_posix()
        record = records.get(output_relative)
        output_path = (work_stage / phase_name / output_relative).resolve()
        try:
            output_path.relative_to(attempt_root)
        except ValueError as exc:
            raise ExternalExecutionError(
                f"successful-attempt output escapes attempt root: {phase_name}"
            ) from exc
        if record is None or not output_path.is_file():
            raise ExternalExecutionError(
                f"successful-attempt output is not stage-receipt-bound: {phase_name}/{relative_within_attempt}"
            )
        return output_path, record, phase_result

    original_manifest, stage05_manifest_record, stage05_result = (
        require_successful_attempt_output("05_tables", "FORMAL_TABLE_MANIFEST_V8.json")
    )
    if (
        sha256_file(manifest_path) != stage05_manifest_record["sha256"]
        or manifest_path.stat().st_size != int(stage05_manifest_record["bytes"])
        or stage05_result.get("value", {}).get("table_manifest_relative_path")
        != "FORMAL_TABLE_MANIFEST_V8.json"
        or stage05_result.get("value", {}).get("table_manifest_sha256")
        != stage05_manifest_record["sha256"]
        or stage05_result.get("value", {}).get("table_files") != manifest["tables"]
        or sha256_file(original_manifest) != sha256_file(manifest_path)
    ):
        raise ExternalExecutionError(
            "published manifest diverges from stage05 successful-attempt original"
        )
    for table_id, relative_path in OUTPUT_TABLE_PATHS.items():
        original_table, stage05_table_record, _ = require_successful_attempt_output(
            "05_tables", relative_path
        )
        published_table = bundle / relative_path
        if (
            stage05_table_record["sha256"] != table_hashes[table_id]
            or int(stage05_table_record["bytes"]) != published_table.stat().st_size
            or sha256_file(original_table) != sha256_file(published_table)
        ):
            raise ExternalExecutionError(
                f"published table diverges from stage05 successful-attempt original: {table_id}"
            )

    original_independent, stage06_independent_record, stage06_result = (
        require_successful_attempt_output(
            "06_independent_recompute",
            "INDEPENDENT_TWENTY_TABLE_RECOMPUTE_V8.json",
        )
    )
    if (
        sha256_file(independent_path) != stage06_independent_record["sha256"]
        or independent_path.stat().st_size != int(stage06_independent_record["bytes"])
        or stage06_result.get("value", {}).get("independent_recompute_receipt")
        != str(original_independent)
        or stage06_result.get("value", {}).get(
            "independent_recompute_receipt_sha256"
        )
        != stage06_independent_record["sha256"]
        or sha256_file(original_independent) != sha256_file(independent_path)
    ):
        raise ExternalExecutionError(
            "published independent receipt diverges from stage06 successful-attempt original"
        )
    return {
        "status": "PUBLISH_BUNDLE_DEEP_REHASH_PASS",
        "file_count": len(observed_files),
        "table_count": len(table_hashes),
        "stage_receipt_sha256": stage_receipt_hashes,
        "stage05_manifest_sha256": stage05_manifest_record["sha256"],
        "stage06_independent_receipt_sha256": stage06_independent_record["sha256"],
    }


def run_formal() -> dict[str, Any]:
    from v8_execution_guard import validate_current_environment

    contract, authorization, consumed_authorization, authorization_sha256 = (
        require_formal_authorization()
    )
    environment_lock = read_json(ARTIFACT_DIR / "ENVIRONMENT_LOCK_V8.json")
    validate_current_environment(
        environment_lock["statistics"], expected_role="statistics"
    )
    stage = FORMAL_TARGET.parent / (
        f".{FORMAL_TARGET.name}.v8_work.{authorization_sha256[:16]}"
    )
    stage.mkdir(parents=True, exist_ok=True)
    try:
        result = FormalPipeline(FormalContext(contract, authorization, stage)).run()
        table_output_root = Path(result.pop("_table_output_root"))
        publish_stage = stage / "07_publish"
        bundle = publish_stage / "bundle_complete"
        publish_stage.mkdir(parents=True, exist_ok=True)
        if not bundle.exists():
            attempt = publish_stage / f"attempt_{uuid.uuid4().hex}"
            attempt.mkdir(parents=True, exist_ok=False)
            for relative_path in OUTPUT_TABLE_PATHS.values():
                source = table_output_root / relative_path
                destination = attempt / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            shutil.copy2(
                table_output_root / "FORMAL_TABLE_MANIFEST_V8.json",
                attempt / "FORMAL_TABLE_MANIFEST_V8.json",
            )
            lineage = attempt / "stage_receipts"
            lineage.mkdir(parents=True, exist_ok=False)
            for phase_name in (
                "01_cache",
                "02_inference",
                "03_prediction_seal",
                "04_outcomes",
                "05_tables",
                "06_independent_recompute",
            ):
                shutil.copy2(
                    stage / phase_name / "STAGE_COMPLETION_V8.json",
                    lineage / f"{phase_name}.json",
                )
            shutil.copy2(
                Path(result["independent_recompute_receipt"]),
                attempt / "INDEPENDENT_TWENTY_TABLE_RECOMPUTE_V8.json",
            )
            write_json_exclusive(
                attempt / "FORMAL_EXECUTION_RECEIPT.json",
                {
                "schema_version": "revision0710.external_validation.formal_execution_receipt.v8",
                "status": "FORMAL_EXECUTION_COMPLETE_CREATE_ONCE",
                "authorization_sha256": authorization_sha256,
                "consumed_authorization_path": str(consumed_authorization.resolve()),
                "formal_once_lock_path": str(FORMAL_LOCK.resolve()),
                "contract_sha256": sha256_file(ARTIFACT_DIR / "EXTERNAL_VALIDATION_CONTRACT_V8.json"),
                "executor_sha256": sha256_file(Path(__file__).resolve()),
                "tag": FORMAL_TAG,
                **result,
                },
            )
            validate_completed_publish_bundle(
                attempt,
                work_stage=stage,
                authorization_sha256=authorization_sha256,
                consumed_authorization=consumed_authorization,
            )
            os.rename(attempt, bundle)
        publish_validation = validate_completed_publish_bundle(
            bundle,
            work_stage=stage,
            authorization_sha256=authorization_sha256,
            consumed_authorization=consumed_authorization,
        )
        os.rename(bundle, FORMAL_TARGET)
        return {"status": "FORMAL_EXTERNAL_VALIDATION_CREATED_ONCE", "target": str(FORMAL_TARGET.resolve()), "publish_validation": publish_validation, **result}
    except Exception:
        # V8 intentionally retains completed create-once stage receipts and
        # partial work. A retry can reuse only fully rehashed stages.
        raise


def tiny_dry_run(destination: Path) -> dict[str, Any]:
    if (
        AUTHORIZATION.exists()
        or CONSUMED_AUTHORIZATION.exists()
        or FORMAL_LOCK.exists()
        or FORMAL_TARGET.exists()
    ):
        raise ExternalExecutionError(
            "tiny dry-run requires formal authorization/consumed/lock and target to remain absent"
        )
    if destination.exists():
        raise FileExistsError(destination)
    base = pd.Timestamp("2020-01-10")
    rows = []
    for index in range(40):
        rows.append(
            {
                "participant_id": "synthetic",
                "event_time": base - pd.Timedelta(days=40 - index),
                "landmark_time": base,
                "source": "hospital_icd",
                "event_type_id": 1,
                "canonical_code": f"I{index:02d}",
                "canonical_description": f"synthetic ICD {index}",
                "source_row_number": index,
            }
        )
    selected = crop_variant_timeline(pd.DataFrame(rows), MIMIC_PRIMARY_VARIANT)
    aligned = left_align_l32(selected, ["synthetic", "padding"])
    if int(aligned["event_valid_mask"][0].sum()) != 32 or aligned["event_valid_mask"][1].any():
        raise AssertionError("primary hospital-ICD-only L32 dry-run failed")
    nhanes = {
        "event_code_id": np.zeros((2, 32), dtype=np.int64),
        "event_text_id": np.zeros((2, 32), dtype=np.int64),
        "event_relative_days": np.zeros((2, 32), dtype=np.int32),
        "event_type_id": np.zeros((2, 32), dtype=np.int8),
        "event_valid_mask": np.zeros((2, 32), dtype=bool),
    }
    validate_nhanes_padding(nhanes)
    endpoint = adjudicate_endpoint(
        target_days=10,
        death_days=10,
        observation_end_days=200,
        death_cap_days=365,
        horizon_months=6,
    )
    if endpoint["censor_reason"] != "death":
        raise AssertionError("death did not win exact target tie")
    aj = aalen_johansen(
        np.array([5.0, 7.0, 10.0, 12.0]), np.array([1, 0, 2, 1], dtype=np.int8)
    )
    c_stat = harrell_c(
        np.array([5.0, 7.0, 10.0, 12.0]),
        np.array([1, 0, 0, 1], dtype=np.int8),
        np.array([0.9, 0.4, 0.2, 0.1]),
    )
    coefficients = fit_crossfit_age_calibration(
        np.array([40.0, 50.0, 60.0, 70.0]),
        np.array([0, 1, 0, 1]),
        np.array([41.0, 52.0, 59.0, 71.0]),
    )
    stage = destination.parent / f".{destination.name}.staging.{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": "revision0710.external_validation.executor_tiny_dry_run.v8",
        "status": "TINY_EXECUTOR_DRY_RUN_PASS_NO_EXTERNAL_LABEL_READ",
        "formal_authorization_absent": True,
        "formal_target_absent": True,
        "external_source_tables_read": [],
        "formal_metrics_created": False,
        "primary_icd_only_selected": len(selected),
        "primary_icd_only_left_aligned": True,
        "nhanes_padding_only": True,
        "death_wins_tie": True,
        "aj_rows": len(aj),
        "harrell_c": c_stat,
        "calibration_coefficients_sha256": hashlib.sha256(np.asarray(coefficients, dtype="<f8").tobytes()).hexdigest(),
        "executor_sha256": sha256_file(Path(__file__).resolve()),
    }
    try:
        write_json_exclusive(stage / "RECEIPT.json", receipt)
        os.rename(stage, destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--formal", action="store_true")
    action.add_argument("--tiny-dry-run", action="store_true")
    parser.add_argument("--destination", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.formal:
        result = run_formal()
    else:
        if args.destination is None:
            raise SystemExit("--tiny-dry-run requires --destination")
        result = tiny_dry_run(args.destination.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
