from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


MONTH_DAYS = 365.25 / 12.0


@dataclass(frozen=True)
class CauseSpecificOutcome:
    event: int
    duration_months: float
    competing_death: int
    at_risk: int
    censor_reason: str


def _timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _valid_after(value: pd.Timestamp, index_date: pd.Timestamp) -> bool:
    return bool(pd.notna(value) and value > index_date)


def filter_and_select_preindex_events(
    events: pd.DataFrame,
    *,
    index_dates: Mapping[str, pd.Timestamp],
    max_len: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep aligned, strictly pre-index tuples and retain the latest history."""
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    required = {"participant_id", "event_date"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {missing}")

    frame = events.copy()
    frame["participant_id"] = (
        frame["participant_id"].astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip()
    )
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    normalized_index = {str(key).removesuffix(".0"): _timestamp(value) for key, value in index_dates.items()}
    frame["index_date"] = frame["participant_id"].map(normalized_index)

    undated = frame["event_date"].isna()
    missing_index = frame["index_date"].isna()
    relative_days = (frame["event_date"] - frame["index_date"]).dt.days
    same_day = relative_days.eq(0)
    post_index = relative_days.gt(0)
    valid = ~undated & ~missing_index & relative_days.lt(0)

    selected = frame.loc[valid].copy()
    selected["relative_days"] = relative_days.loc[valid].astype(np.float32)
    tuple_cols = [
        col
        for col in ["participant_id", "event_date", "code", "description", "source"]
        if col in selected.columns
    ]
    before_dedup = len(selected)
    selected = selected.drop_duplicates(tuple_cols, keep="first")
    deduplicated = before_dedup - len(selected)

    tie_cols = [col for col in ["code", "description", "source"] if col in selected.columns]
    sort_cols = ["participant_id", "event_date", *tie_cols]
    selected = selected.sort_values(sort_cols, kind="mergesort")
    selected = selected.groupby("participant_id", sort=False, group_keys=False).tail(max_len)
    selected = selected.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    selected["attention_mask"] = np.int8(1)

    stats = {
        "rows_seen": int(len(frame)),
        "rows_retained": int(len(selected)),
        "same_day_excluded": int(same_day.fillna(False).sum()),
        "post_index_excluded": int(post_index.fillna(False).sum()),
        "undated_excluded": int(undated.sum()),
        "missing_index_excluded": int(missing_index.sum()),
        "exact_duplicates_removed": int(deduplicated),
    }
    return selected, stats


def build_cause_specific_outcome(
    *,
    index_date: object,
    first_target_event: object = None,
    death_date: object = None,
    observed_followup_end: object = None,
    horizon_months: int = 60,
) -> CauseSpecificOutcome:
    index = _timestamp(index_date)
    if pd.isna(index):
        raise ValueError("index_date is required")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive")

    target = _timestamp(first_target_event)
    death = _timestamp(death_date)
    observed_end = _timestamp(observed_followup_end)
    horizon_end = index + pd.to_timedelta(horizon_months * MONTH_DAYS, unit="D")

    followup_candidates = [horizon_end]
    if _valid_after(observed_end, index):
        followup_candidates.append(observed_end)
    elif pd.notna(observed_end):
        return CauseSpecificOutcome(0, float("nan"), 0, 0, "no_post_index_followup")
    elif not _valid_after(target, index) and not _valid_after(death, index):
        return CauseSpecificOutcome(0, float("nan"), 0, 0, "no_post_index_followup")
    censor_without_death = min(followup_candidates)

    death_is_observed = _valid_after(death, index) and death <= censor_without_death
    censor_end = min(censor_without_death, death) if death_is_observed else censor_without_death
    target_is_observed = _valid_after(target, index) and target <= censor_end and target <= horizon_end

    if target_is_observed:
        end = target
        event = 1
        competing_death = 0
        reason = "target_event"
    else:
        end = censor_end
        event = 0
        competing_death = int(death_is_observed)
        if death_is_observed:
            reason = "death_before_target"
        elif end < horizon_end:
            reason = "observed_followup_end"
        else:
            reason = "administrative_60m"

    duration = max(float((end - index).days) / MONTH_DAYS, 1e-6)
    return CauseSpecificOutcome(event, duration, competing_death, 1, reason)


def audit_preindex_events(events: pd.DataFrame) -> dict[str, object]:
    required = {"relative_days", "attention_mask"}
    missing = sorted(required - set(events.columns))
    if missing:
        return {"status": "FAIL", "missing_columns": missing}

    relative = pd.to_numeric(events["relative_days"], errors="coerce")
    mask = pd.to_numeric(events["attention_mask"], errors="coerce").fillna(0).eq(1)
    non_preindex = mask & (relative.isna() | relative.ge(0))
    inactive_nonzero = ~mask & relative.fillna(0).ne(0)
    status = "PASS" if not non_preindex.any() and not inactive_nonzero.any() else "FAIL"
    return {
        "status": status,
        "rows": int(len(events)),
        "active_rows": int(mask.sum()),
        "non_preindex_input_rows": int(non_preindex.sum()),
        "inactive_nonzero_time_rows": int(inactive_nonzero.sum()),
        "max_active_relative_days": float(relative.loc[mask].max()) if mask.any() else None,
    }


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def assign_stratified_folds(
    participants: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: str = "revision0710",
) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    required = {"participant_id", "stratum"}
    missing = sorted(required - set(participants.columns))
    if missing:
        raise ValueError(f"participants missing required columns: {missing}")

    frame = participants.copy()
    frame["participant_id"] = frame["participant_id"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    if frame["participant_id"].eq("").any() or frame["participant_id"].duplicated().any():
        raise ValueError("participant_id must be non-empty and unique")
    frame["stratum"] = frame["stratum"].astype("string").fillna("missing")
    frame["participant_hash"] = [
        _stable_int(f"{seed}|{stratum}|{pid}")
        for pid, stratum in zip(frame["participant_id"], frame["stratum"])
    ]
    frame = frame.sort_values(["stratum", "participant_hash", "participant_id"], kind="mergesort")
    offsets = {stratum: _stable_int(f"{seed}|{stratum}") % n_folds for stratum in frame["stratum"].unique()}
    within = frame.groupby("stratum", sort=False).cumcount()
    frame["fold_id"] = [
        int((idx + offsets[stratum]) % n_folds)
        for idx, stratum in zip(within, frame["stratum"])
    ]
    return frame.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
