"""Pure recoding helpers for leakage-safe Instance-0 contextual covariates."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class CovariateMaterializationError(ValueError):
    """Raised when a raw baseline covariate violates the frozen dictionary."""


NULL_TEXT = {"", "prefer not to answer", "do not know", "nan", "none", "<na>"}
SMOKING_CURRENT = {"yes, on most or all days", "only occasionally"}
SMOKING_NO = {"no"}
EDUCATION_HIGHER = {
    "college or university degree",
    "a levels/as levels or equivalent",
    "nvq or hnd or hnc or equivalent",
    "other professional qualifications eg: nursing, teaching",
}
EDUCATION_LOWER = {
    "o levels/gcses or equivalent",
    "cses or equivalent",
    "none of the above",
}
EMPLOYMENT_EMPLOYED = {
    "in paid employment or self-employed",
    "doing unpaid or voluntary work",
}
EMPLOYMENT_RETIRED = {"retired"}
EMPLOYMENT_UNEMPLOYED = {"unemployed"}
EMPLOYMENT_OTHER = {
    "unable to work because of sickness or disability",
    "looking after home and/or family",
    "none of the above",
    "full or part-time student",
}
ALCOHOL_CATEGORIES = {"current", "previous", "never"}


def normalize_participant_id(series: pd.Series) -> pd.Series:
    result = series.astype("string").fillna("").str.strip()
    result = result.str.replace(r"\.0$", "", regex=True)
    invalid = result.str.lower().isin(NULL_TEXT)
    return result.mask(invalid, "")


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()


def split_tokens(value: str) -> set[str]:
    return {token.strip() for token in value.split("|") if token.strip()}


def _unexpected_tokens(values: pd.Series, allowed: Iterable[str]) -> set[str]:
    allowed_set = set(allowed)
    unexpected: set[str] = set()
    for value in values:
        if value in NULL_TEXT:
            continue
        unexpected.update(split_tokens(value).difference(allowed_set))
    return unexpected


def encode_smoking(series: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    text = normalize_text(series)
    unexpected = set(text.unique()).difference(NULL_TEXT | SMOKING_CURRENT | SMOKING_NO)
    if unexpected:
        raise CovariateMaterializationError(f"unexpected smoking categories: {sorted(unexpected)}")
    missing = text.isin(NULL_TEXT)
    current = pd.Series(np.nan, index=series.index, dtype=np.float32)
    current.loc[text.isin(SMOKING_NO)] = 0.0
    current.loc[text.isin(SMOKING_CURRENT)] = 1.0
    result = pd.DataFrame(
        {
            "smoking_current": current,
            "smoking_missing": missing.astype(np.int8),
        },
        index=series.index,
    )
    return result, {"unexpected_tokens": [], "missing_n": int(missing.sum())}


def encode_education(series: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    text = normalize_text(series)
    allowed = EDUCATION_HIGHER | EDUCATION_LOWER
    unexpected = _unexpected_tokens(text, allowed)
    if unexpected:
        raise CovariateMaterializationError(f"unexpected education tokens: {sorted(unexpected)}")
    missing = text.isin(NULL_TEXT)
    higher = pd.Series(np.nan, index=series.index, dtype=np.float32)
    for index, value in text.loc[~missing].items():
        higher.loc[index] = float(bool(split_tokens(value).intersection(EDUCATION_HIGHER)))
    result = pd.DataFrame(
        {
            "education_higher": higher,
            "education_missing": missing.astype(np.int8),
        },
        index=series.index,
    )
    return result, {"unexpected_tokens": [], "missing_n": int(missing.sum())}


def encode_employment(series: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    text = normalize_text(series)
    allowed = EMPLOYMENT_EMPLOYED | EMPLOYMENT_RETIRED | EMPLOYMENT_UNEMPLOYED | EMPLOYMENT_OTHER
    unexpected = _unexpected_tokens(text, allowed)
    if unexpected:
        raise CovariateMaterializationError(f"unexpected employment tokens: {sorted(unexpected)}")
    missing = text.isin(NULL_TEXT)
    result = pd.DataFrame(index=series.index)
    groups = {
        "employment_employed": EMPLOYMENT_EMPLOYED,
        "employment_retired": EMPLOYMENT_RETIRED,
        "employment_unemployed": EMPLOYMENT_UNEMPLOYED,
        "employment_other": EMPLOYMENT_OTHER,
    }
    for column, tokens in groups.items():
        result[column] = text.map(
            lambda value: int(bool(split_tokens(value).intersection(tokens))) if value not in NULL_TEXT else 0
        ).astype(np.int8)
    result["employment_missing"] = missing.astype(np.int8)
    return result, {"unexpected_tokens": [], "missing_n": int(missing.sum())}


def encode_alcohol(series: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    text = normalize_text(series)
    unexpected = set(text.unique()).difference(NULL_TEXT | ALCOHOL_CATEGORIES)
    if unexpected:
        raise CovariateMaterializationError(f"unexpected alcohol categories: {sorted(unexpected)}")
    missing = text.isin(NULL_TEXT)
    result = pd.DataFrame(index=series.index)
    for category in sorted(ALCOHOL_CATEGORIES):
        result[f"sensitivity_alcohol_{category}"] = text.eq(category).astype(np.int8)
    result["sensitivity_alcohol_missing"] = missing.astype(np.int8)
    return result, {"unexpected_tokens": [], "missing_n": int(missing.sum())}


def encode_continuous(
    series: pd.Series,
    *,
    value_name: str,
    missing_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    text = normalize_text(series)
    values = pd.to_numeric(text.mask(text.isin(NULL_TEXT)), errors="coerce")
    invalid_numeric = ~text.isin(NULL_TEXT) & values.isna()
    if invalid_numeric.any():
        invalid = sorted(text.loc[invalid_numeric].unique().tolist())
        raise CovariateMaterializationError(
            f"{value_name} contains unexpected non-numeric values: {invalid}"
        )
    if minimum is not None and values.dropna().lt(minimum).any():
        raise CovariateMaterializationError(f"{value_name} contains values below {minimum}")
    if maximum is not None and values.dropna().gt(maximum).any():
        raise CovariateMaterializationError(f"{value_name} contains values above {maximum}")
    result = pd.DataFrame(
        {
            value_name: values.astype(np.float32),
            missing_name: values.isna().astype(np.int8),
        },
        index=series.index,
    )
    audit = {
        "unexpected_tokens": [],
        "missing_n": int(values.isna().sum()),
        "minimum": float(values.min()) if values.notna().any() else None,
        "maximum": float(values.max()) if values.notna().any() else None,
    }
    return result, audit
