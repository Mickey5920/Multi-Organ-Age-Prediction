from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from source_manifest_core import normalize_participant_id


EVENT_TYPE_IDS = {"hospital_icd": 1, "gp_prescription": 2}
TEXT_ROLES = {"hospital_icd": "icd_description", "gp_prescription": "gp_drug"}


def fit_phenotype_transform(
    values: np.ndarray,
    missing_mask: np.ndarray,
    *,
    train_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool)
    train = np.asarray(train_mask, dtype=bool)
    if values.ndim != 2 or missing.shape != values.shape:
        raise ValueError("values and missing_mask must be aligned 2D arrays")
    if train.shape != (len(values),) or not train.any():
        raise ValueError("train_mask must select at least one aligned row")
    train_values = values[train].astype(np.float64, copy=True)
    train_values[missing[train]] = np.nan
    median = np.nanmedian(train_values, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("every phenotype feature needs an observed training value")
    filled = np.where(np.isnan(train_values), median[None, :], train_values)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "median": median.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def apply_phenotype_transform(
    values: np.ndarray,
    missing_mask: np.ndarray,
    transform: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    missing = np.asarray(missing_mask, dtype=bool)
    if values.ndim != 2 or missing.shape != values.shape:
        raise ValueError("values and missing_mask must be aligned 2D arrays")
    median = np.asarray(transform["median"], dtype=np.float32)
    mean = np.asarray(transform["mean"], dtype=np.float32)
    std = np.asarray(transform["std"], dtype=np.float32)
    if median.shape != values.shape[1:] or mean.shape != median.shape or std.shape != median.shape:
        raise ValueError("phenotype transform dimensions do not match values")
    filled = np.where(missing, median[None, :], values)
    result = (filled - mean[None, :]) / std[None, :]
    if not np.isfinite(result).all():
        raise ValueError("phenotype transform produced non-finite values")
    return result.astype(np.float32)


def event_text_keys(events: pd.DataFrame) -> pd.Series:
    missing = sorted({"source", "description"} - set(events.columns))
    if missing:
        raise ValueError(f"events missing text-key columns: {missing}")
    source = events["source"].astype("string").fillna("").str.strip()
    description = events["description"].astype("string").fillna("").str.strip()
    role = source.map(TEXT_ROLES)
    if role.isna().any():
        raise ValueError(f"unknown event sources for text vocabulary: {sorted(source[role.isna()].unique())}")
    return role + "\x1f" + description


def select_unified_timeline(
    icd_events: pd.DataFrame,
    gp_events: pd.DataFrame,
    *,
    participant_ids: Iterable[str],
    max_len: int,
    minimum_per_source: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    minimum_per_source = dict(minimum_per_source or {})
    unknown_minimum_sources = sorted(set(minimum_per_source) - set(EVENT_TYPE_IDS))
    if unknown_minimum_sources:
        raise ValueError(f"minimum_per_source has unknown sources: {unknown_minimum_sources}")
    if any(value < 0 for value in minimum_per_source.values()):
        raise ValueError("minimum_per_source values must be non-negative")
    if sum(minimum_per_source.values()) > max_len:
        raise ValueError("minimum_per_source total cannot exceed max_len")
    required = {
        "participant_id",
        "event_date",
        "code",
        "description",
        "source",
        "source_row_number",
        "relative_days",
        "attention_mask",
    }
    for name, frame in [("ICD", icd_events), ("GP", gp_events)]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} events missing required columns: {missing}")

    allowed_ids = {str(value).removesuffix(".0").strip() for value in participant_ids}
    combined = pd.concat(
        [icd_events[list(required)].copy(), gp_events[list(required)].copy()],
        ignore_index=True,
    )
    combined["participant_id"] = normalize_participant_id(combined["participant_id"])
    unexpected_ids = set(combined["participant_id"]) - allowed_ids
    if unexpected_ids:
        raise ValueError(f"events contain {len(unexpected_ids)} participants outside the age cohort")
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce")
    combined["relative_days"] = pd.to_numeric(combined["relative_days"], errors="coerce")
    combined["attention_mask"] = pd.to_numeric(combined["attention_mask"], errors="coerce")
    combined["source"] = combined["source"].astype("string").fillna("").str.strip()
    combined["code"] = combined["code"].astype("string").fillna("").str.strip()
    combined["description"] = combined["description"].astype("string").fillna("").str.strip()
    invalid_source = ~combined["source"].isin(EVENT_TYPE_IDS)
    if invalid_source.any():
        raise ValueError(f"unknown event sources: {sorted(combined.loc[invalid_source, 'source'].unique())}")
    invalid_time = (
        combined["event_date"].isna()
        | combined["relative_days"].isna()
        | combined["relative_days"].ge(0)
        | combined["attention_mask"].ne(1)
    )
    if invalid_time.any():
        raise ValueError(f"all active cache events must be strictly pre-index; invalid rows={int(invalid_time.sum())}")
    missing_text = combined["description"].eq("")
    if missing_text.any():
        raise ValueError(f"active events require non-empty description text; rows={int(missing_text.sum())}")

    combined["event_type_id"] = combined["source"].map(EVENT_TYPE_IDS).astype(np.uint8)
    combined["source_priority"] = combined["event_type_id"]
    sort_columns = [
        "participant_id",
        "event_date",
        "source_priority",
        "code",
        "description",
        "source_row_number",
    ]
    combined = combined.sort_values(sort_columns, kind="mergesort")
    combined = combined.drop_duplicates(
        ["participant_id", "event_date", "code", "description", "source"], keep="first"
    )
    if any(minimum_per_source.values()):
        source_minimum = combined["source"].map(minimum_per_source).fillna(0).astype(np.int32)
        source_rank_from_latest = combined.groupby(
            ["participant_id", "source"], sort=False
        ).cumcount(ascending=False)
        required = source_rank_from_latest.lt(source_minimum)
        required_count = required.groupby(combined["participant_id"], sort=False).transform("sum")
        remaining_capacity = (max_len - required_count).astype(np.int32)
        optional = combined.loc[~required].copy()
        optional["optional_rank_from_latest"] = optional.groupby(
            "participant_id", sort=False
        ).cumcount(ascending=False)
        optional_capacity = remaining_capacity.loc[optional.index]
        optional_keep = optional["optional_rank_from_latest"].lt(optional_capacity)
        keep_index = combined.index[required].union(optional.index[optional_keep])
        combined = combined.loc[keep_index].copy()
    else:
        combined = combined.groupby("participant_id", sort=False, group_keys=False).tail(max_len)
    combined = combined.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return combined.drop(columns="source_priority")


def build_vocab(values: Iterable[object]) -> tuple[list[str], dict[str, int]]:
    normalized = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    }
    vocab = ["", *sorted(normalized)]
    return vocab, {value: index for index, value in enumerate(vocab)}


def pack_compact_sequences(
    selected_events: pd.DataFrame,
    *,
    participant_order: Sequence[str],
    max_len: int,
    code_to_id: Mapping[str, int],
    text_to_id: Mapping[str, int],
) -> dict[str, np.ndarray]:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    participants = [str(value).removesuffix(".0").strip() for value in participant_order]
    if len(set(participants)) != len(participants) or any(not value for value in participants):
        raise ValueError("participant_order must contain unique non-empty IDs")
    frame = selected_events.copy()
    frame["participant_id"] = normalize_participant_id(frame["participant_id"])
    counts = frame.groupby("participant_id", sort=False).size()
    if len(counts) and int(counts.max()) > max_len:
        raise ValueError(f"selected timeline exceeds max_len={max_len}")
    row_map = pd.Series(np.arange(len(participants), dtype=np.int64), index=participants)
    row_index = frame["participant_id"].map(row_map)
    if row_index.isna().any():
        raise ValueError("selected events contain participants outside participant_order")
    position = frame.groupby("participant_id", sort=False).cumcount()

    shape = (len(participants), max_len)
    code_ids = np.zeros(shape, dtype=np.int32)
    text_ids = np.zeros(shape, dtype=np.int32)
    relative_days = np.zeros(shape, dtype=np.int32)
    event_types = np.zeros(shape, dtype=np.uint8)
    valid_mask = np.zeros(shape, dtype=bool)
    event_count = np.zeros(len(participants), dtype=np.int16)
    if len(frame):
        mapped_codes = frame["code"].astype("string").fillna("").map(code_to_id)
        mapped_texts = event_text_keys(frame).map(text_to_id)
        if mapped_codes.isna().any() or mapped_texts.isna().any():
            raise ValueError("event text is missing from deterministic vocabulary")
        rows = row_index.to_numpy(dtype=np.int64)
        cols = position.to_numpy(dtype=np.int64)
        code_ids[rows, cols] = mapped_codes.to_numpy(dtype=np.int32)
        text_ids[rows, cols] = mapped_texts.to_numpy(dtype=np.int32)
        relative_days[rows, cols] = pd.to_numeric(
            frame["relative_days"], errors="raise"
        ).to_numpy(dtype=np.int32)
        event_types[rows, cols] = pd.to_numeric(
            frame["event_type_id"], errors="raise"
        ).to_numpy(dtype=np.uint8)
        valid_mask[rows, cols] = True
        count_map = counts.reindex(participants, fill_value=0)
        event_count[:] = count_map.to_numpy(dtype=np.int16)

    if np.any(relative_days[valid_mask] >= 0):
        raise ValueError("packed active event times must be strictly negative")
    inactive = ~valid_mask
    if (
        np.any(code_ids[inactive] != 0)
        or np.any(text_ids[inactive] != 0)
        or np.any(relative_days[inactive] != 0)
        or np.any(event_types[inactive] != 0)
    ):
        raise ValueError("packed padding fields must all be zero")
    return {
        "event_code_id": code_ids,
        "event_text_id": text_ids,
        "event_relative_days": relative_days,
        "event_type_id": event_types,
        "event_valid_mask": valid_mask,
        "event_count": event_count,
    }
