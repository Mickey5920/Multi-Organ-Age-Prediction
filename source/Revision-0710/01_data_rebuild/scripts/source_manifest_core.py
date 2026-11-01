from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def normalize_participant_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def _missing_mask(frame: pd.DataFrame) -> pd.DataFrame:
    text = frame.astype("string")
    return frame.isna() | text.apply(
        lambda column: column.str.strip().isin(["", "nan", "None", "<NA>"])
    )


def build_age_model_cohort(
    assessment: pd.DataFrame,
    phenotype: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    max_missing_count: int = 19,
) -> pd.DataFrame:
    """Build an age-training cohort without consulting follow-up or outcomes."""
    if max_missing_count < 0:
        raise ValueError("max_missing_count must be non-negative")
    required_assessment = {"participant_id", "index_date", "birth_date"}
    required_phenotype = {"participant_id", "sex", *feature_columns}
    missing_assessment = sorted(required_assessment - set(assessment.columns))
    missing_phenotype = sorted(required_phenotype - set(phenotype.columns))
    if missing_assessment:
        raise ValueError(f"assessment missing required columns: {missing_assessment}")
    if missing_phenotype:
        raise ValueError(f"phenotype missing required columns: {missing_phenotype}")
    if not feature_columns:
        raise ValueError("feature_columns must be explicit and non-empty")

    index = assessment[["participant_id", "index_date", "birth_date"]].copy()
    index["participant_id"] = normalize_participant_id(index["participant_id"])
    index["index_date"] = pd.to_datetime(index["index_date"], errors="coerce")
    index["birth_date"] = pd.to_datetime(index["birth_date"], errors="coerce")
    if index["participant_id"].eq("").any() or index["participant_id"].duplicated().any():
        raise ValueError("assessment participant_id must be non-empty and unique")
    index["chronological_age"] = (
        (index["index_date"] - index["birth_date"]).dt.days / 365.25
    )

    selected_columns = ["participant_id", "sex", *feature_columns]
    pheno = phenotype[selected_columns].copy()
    pheno["participant_id"] = normalize_participant_id(pheno["participant_id"])
    if pheno["participant_id"].eq("").any() or pheno["participant_id"].duplicated().any():
        raise ValueError("phenotype participant_id must be non-empty and unique")
    missing = _missing_mask(pheno[list(feature_columns)])
    pheno["phenotype_missing_count"] = missing.sum(axis=1).astype(np.int16)
    pheno["phenotype_missing_fraction"] = (
        pheno["phenotype_missing_count"] / float(len(feature_columns))
    ).astype(np.float32)

    cohort = index.merge(
        pheno[["participant_id", "sex", "phenotype_missing_count", "phenotype_missing_fraction"]],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
    eligible = (
        cohort["index_date"].notna()
        & cohort["birth_date"].notna()
        & cohort["chronological_age"].between(18.0, 100.0, inclusive="both")
        & cohort["phenotype_missing_count"].le(max_missing_count)
    )
    cohort = cohort.loc[eligible].copy()
    cohort["age_model_eligible"] = np.int8(1)
    return cohort.sort_values("participant_id", kind="mergesort").reset_index(drop=True)


def split_icd_event_roles(
    events: pd.DataFrame,
    *,
    index_dates: Mapping[str, pd.Timestamp],
) -> pd.DataFrame:
    required = {"participant_id", "event_date"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events missing required columns: {missing}")

    out = events.copy()
    out["participant_id"] = normalize_participant_id(out["participant_id"])
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    normalized_index = {
        str(key).removesuffix(".0").strip(): pd.to_datetime(value, errors="coerce")
        for key, value in index_dates.items()
    }
    out["index_date"] = out["participant_id"].map(normalized_index)
    valid = out["event_date"].notna() & out["index_date"].notna()
    out["relative_days"] = (out["event_date"] - out["index_date"]).dt.days
    out["is_model_input"] = valid & out["relative_days"].lt(0)
    out["is_prevalent"] = valid & out["relative_days"].le(0)
    out["is_incident"] = valid & out["relative_days"].gt(0)
    return out


def derive_observed_followup_end(
    frame: pd.DataFrame,
    *,
    candidate_columns: Sequence[str],
    lost_column: str = "lost_to_followup_date",
) -> pd.Series:
    missing = sorted(set(candidate_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"follow-up candidates missing columns: {missing}")
    candidates = frame[list(candidate_columns)].apply(pd.to_datetime, errors="coerce")
    observed = candidates.max(axis=1)
    if lost_column in frame.columns:
        lost = pd.to_datetime(frame[lost_column], errors="coerce")
        use_lost = lost.notna() & (observed.isna() | lost.lt(observed))
        observed = observed.where(~use_lost, lost)
    observed.name = "observed_followup_end"
    return observed
