from __future__ import annotations

import hashlib

import math

from typing import Any, Mapping, Sequence

import numpy as np

import pandas as pd

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
