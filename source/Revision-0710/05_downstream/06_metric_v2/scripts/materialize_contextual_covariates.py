#!/usr/bin/env python
"""Materialize the frozen Instance-0 and strict-preindex contextual covariates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
METRIC_DIR = SCRIPT_DIR.parent
REVISION_DIR = METRIC_DIR.parents[1]
ROOT = REVISION_DIR.parent
COMMON_DIR = METRIC_DIR / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from contextual_covariates import (  # noqa: E402
    CovariateMaterializationError,
    encode_alcohol,
    encode_continuous,
    encode_education,
    encode_employment,
    encode_smoking,
    normalize_participant_id,
)


DEFAULT_OUTPUT_DIR = METRIC_DIR / "01_contextual_covariates"
DICTIONARY = METRIC_DIR / "CANDIDATE_DATA_DICTIONARY.json"
AGE_COHORT = REVISION_DIR / "01_data_rebuild" / "01_index_and_events" / "age_model_cohort.parquet"
FOLDS = REVISION_DIR / "01_data_rebuild" / "03_folds" / "fold_manifest_5fold.parquet"
ICD_SUMMARY = REVISION_DIR / "01_data_rebuild" / "01_index_and_events" / "icd_participant_summary.parquet"
GP_SUMMARY = REVISION_DIR / "01_data_rebuild" / "01_index_and_events" / "gp_participant_summary.parquet"
ASSOCIATION_1 = Path("E:/MPU/Project/Dataset-0911/Association/Associations-1.csv")
ASSOCIATION_2 = Path("E:/MPU/Project/Dataset-0911/Association/Associations-2.csv")
ASSOCIATION_4 = Path("E:/MPU/Project/Dataset-0911/Association/Associations-4.csv")
ASSOCIATION_5 = Path("E:/MPU/Project/Dataset-0911/Association/Associations-5.csv")
EXPECTED_COHORT_N = 456_773
ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)

SOURCE_COLUMNS = {
    ASSOCIATION_1: (
        "Participant ID",
        "Current tobacco smoking | Instance 0",
        "Qualifications | Instance 0",
        "Townsend deprivation index at recruitment",
        "Alcohol drinker status | Instance 0",
    ),
    ASSOCIATION_2: (
        "Participant ID",
        "Current employment status | Instance 0",
        "Summed MET minutes per week for all activity | Instance 0",
    ),
    ASSOCIATION_4: (
        "Participant ID",
        "Sleep duration | Instance 0",
    ),
    ASSOCIATION_5: (
        "Participant ID",
        "Particulate matter air pollution (pm2.5); 2010",
    ),
}
PARQUET_SOURCE_COLUMNS = {
    AGE_COHORT: (
        "participant_id",
        "chronological_age",
        "sex",
        "phenotype_missing_fraction",
        "age_model_eligible",
    ),
    FOLDS: ("participant_id", "fold_id"),
    ICD_SUMMARY: (
        "participant_id",
        "model_input_icd_count",
        "has_model_input_icd",
        *[f"has_prevalent_{organ}" for organ in ORGANS],
    ),
    GP_SUMMARY: ("participant_id", "model_input_gp_count", "has_model_input_gp"),
}

PRIMARY_MODEL_COLUMNS = (
    "chronological_age",
    "sex_male",
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
    "phenotype_missing_fraction",
    "log1p_preindex_icd_count",
    "log1p_preindex_gp_count",
    "has_preindex_icd",
    "has_preindex_gp",
    "prevalent_organ_count",
)
SENSITIVITY_MODEL_COLUMNS = (
    "sensitivity_alcohol_current",
    "sensitivity_alcohol_never",
    "sensitivity_alcohol_previous",
    "sensitivity_alcohol_missing",
    "sensitivity_physical_activity_met",
    "sensitivity_log1p_physical_activity_met",
    "sensitivity_physical_activity_missing",
    "sensitivity_pm25_2010",
    "sensitivity_pm25_2010_missing",
)
AUXILIARY_COLUMNS = ("preindex_icd_count", "preindex_gp_count")
FORBIDDEN_OUTPUT_FRAGMENTS = (
    "predicted",
    "age_gap",
    "incident",
    "event_indicator",
    "time_to_event",
    "duration_months",
    "death",
    "followup",
    "outcome",
    "hazard",
    "risk_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_exact_ids(frame: pd.DataFrame, expected_ids: set[str], *, table: str) -> None:
    if frame["participant_id"].eq("").any() or frame["participant_id"].duplicated().any():
        raise CovariateMaterializationError(f"{table} has empty or duplicate participant IDs")
    observed = set(frame["participant_id"])
    if observed != expected_ids:
        raise CovariateMaterializationError(
            f"{table} participant coverage mismatch: missing={len(expected_ids - observed)}, "
            f"extra={len(observed - expected_ids)}"
        )


def read_association(path: Path, columns: Iterable[str], expected_ids: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=list(columns), dtype=str, low_memory=False)
    if len(frame) != 502_163:
        raise CovariateMaterializationError(f"{path.name} expected 502,163 rows, got {len(frame):,}")
    frame = frame.rename(columns={"Participant ID": "participant_id"})
    frame["participant_id"] = normalize_participant_id(frame["participant_id"])
    if frame["participant_id"].eq("").any() or frame["participant_id"].duplicated().any():
        raise CovariateMaterializationError(f"{path.name} has empty or duplicate participant IDs")
    frame = frame.loc[frame["participant_id"].isin(expected_ids)].copy()
    require_exact_ids(frame, expected_ids, table=path.name)
    return frame


def merge_one_to_one(left: pd.DataFrame, right: pd.DataFrame, *, table: str) -> pd.DataFrame:
    result = left.merge(right, on="participant_id", how="left", validate="one_to_one")
    if len(result) != len(left):
        raise CovariateMaterializationError(f"row count changed while joining {table}")
    return result


def numeric_audit(frame: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        role = roles.get(column, "metadata")
        if column == "participant_id":
            rows.append(
                {
                    "column": column,
                    "role": role,
                    "dtype": str(frame[column].dtype),
                    "n": int(len(frame)),
                    "missing_n": int(frame[column].isna().sum()),
                    "missing_pct": float(frame[column].isna().mean() * 100.0),
                    "unique_n": int(frame[column].nunique()),
                    "minimum": None,
                    "q25": None,
                    "median": None,
                    "q75": None,
                    "maximum": None,
                }
            )
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.dropna()
        quantiles = valid.quantile([0.25, 0.5, 0.75]) if len(valid) else pd.Series(dtype=float)
        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(frame[column].dtype),
                "n": int(len(frame)),
                "missing_n": int(values.isna().sum()),
                "missing_pct": float(values.isna().mean() * 100.0),
                "unique_n": int(frame[column].nunique(dropna=True)),
                "minimum": float(valid.min()) if len(valid) else None,
                "q25": float(quantiles.loc[0.25]) if len(valid) else None,
                "median": float(quantiles.loc[0.5]) if len(valid) else None,
                "q75": float(quantiles.loc[0.75]) if len(valid) else None,
                "maximum": float(valid.max()) if len(valid) else None,
            }
        )
    return pd.DataFrame(rows)


def timing_audit() -> pd.DataFrame:
    rows = [
        ("chronological_age", "age_model_cohort", "index assessment", True, "primary"),
        ("sex_male", "age_model_cohort", "baseline/static", True, "primary"),
        ("smoking_current", "Associations-1", "Instance 0 only", True, "primary"),
        ("education_higher", "Associations-1", "Instance 0 only", True, "primary"),
        ("townsend", "Associations-1", "at recruitment", True, "primary"),
        ("employment indicators", "Associations-2", "Instance 0 only", True, "primary"),
        ("sleep_duration", "Associations-4", "Instance 0 only", True, "primary"),
        ("phenotype_missing_fraction", "age_model_cohort", "78 Instance-0 phenotypes", True, "primary"),
        ("preindex ICD density", "icd_participant_summary", "event_date < index_date", True, "primary"),
        ("preindex GP density", "gp_participant_summary", "event_date < index_date", True, "primary"),
        ("prevalent_organ_count", "icd_participant_summary", "event_date <= index_date", True, "primary"),
        ("alcohol indicators", "Associations-1", "Instance 0 only", True, "sensitivity"),
        ("physical_activity_met", "Associations-2", "Instance 0 only", True, "sensitivity"),
        ("pm25_2010", "Associations-5", "calendar-year 2010 exposure may postdate recruitment", False, "sensitivity_only"),
    ]
    return pd.DataFrame(
        rows,
        columns=("variable", "source", "timing_semantics", "leakage_strict_primary_allowed", "role"),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "participant_context_covariates.parquet"
    manifest_path = args.output_dir / "participant_context_covariates_manifest.json"
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")

    age = pd.read_parquet(
        AGE_COHORT,
        columns=(
            "participant_id",
            "chronological_age",
            "sex",
            "phenotype_missing_fraction",
            "age_model_eligible",
        ),
    )
    age["participant_id"] = normalize_participant_id(age["participant_id"])
    if len(age) != EXPECTED_COHORT_N or not age["age_model_eligible"].astype(bool).all():
        raise CovariateMaterializationError("age cohort count or eligibility mismatch")
    expected_ids = set(age["participant_id"])
    require_exact_ids(age, expected_ids, table="age_model_cohort")
    output = age[["participant_id", "chronological_age", "phenotype_missing_fraction"]].copy()
    sex = age["sex"].astype(str).str.strip().str.lower().map({"female": 0, "male": 1})
    if sex.isna().any():
        raise CovariateMaterializationError("sex contains unsupported values")
    output["sex_male"] = sex.astype(np.int8)

    folds = pd.read_parquet(FOLDS, columns=("participant_id", "fold_id"))
    folds["participant_id"] = normalize_participant_id(folds["participant_id"])
    require_exact_ids(folds, expected_ids, table="fold_manifest")
    if not set(pd.to_numeric(folds["fold_id"], errors="coerce").unique()).issubset(set(range(5))):
        raise CovariateMaterializationError("fold_id must contain only 0..4")
    output = merge_one_to_one(output, folds, table="fold_manifest")

    icd_columns = (
        "participant_id",
        "model_input_icd_count",
        "has_model_input_icd",
        *[f"has_prevalent_{organ}" for organ in ORGANS],
    )
    icd = pd.read_parquet(ICD_SUMMARY, columns=icd_columns)
    icd["participant_id"] = normalize_participant_id(icd["participant_id"])
    require_exact_ids(icd, expected_ids, table="icd_participant_summary")
    icd_out = pd.DataFrame({"participant_id": icd["participant_id"]})
    icd_out["preindex_icd_count"] = pd.to_numeric(icd["model_input_icd_count"], errors="raise").astype(np.int32)
    if icd_out["preindex_icd_count"].lt(0).any():
        raise CovariateMaterializationError("negative preindex ICD count")
    icd_out["log1p_preindex_icd_count"] = np.log1p(icd_out["preindex_icd_count"]).astype(np.float32)
    icd_out["has_preindex_icd"] = icd["has_model_input_icd"].astype(np.int8)
    icd_out["prevalent_organ_count"] = icd[
        [f"has_prevalent_{organ}" for organ in ORGANS]
    ].astype(np.int8).sum(axis=1).astype(np.int8)
    output = merge_one_to_one(output, icd_out, table="ICD summaries")

    gp = pd.read_parquet(
        GP_SUMMARY,
        columns=("participant_id", "model_input_gp_count", "has_model_input_gp"),
    )
    gp["participant_id"] = normalize_participant_id(gp["participant_id"])
    require_exact_ids(gp, expected_ids, table="gp_participant_summary")
    gp_out = pd.DataFrame({"participant_id": gp["participant_id"]})
    gp_out["preindex_gp_count"] = pd.to_numeric(gp["model_input_gp_count"], errors="raise").astype(np.int32)
    if gp_out["preindex_gp_count"].lt(0).any():
        raise CovariateMaterializationError("negative preindex GP count")
    gp_out["log1p_preindex_gp_count"] = np.log1p(gp_out["preindex_gp_count"]).astype(np.float32)
    gp_out["has_preindex_gp"] = gp["has_model_input_gp"].astype(np.int8)
    output = merge_one_to_one(output, gp_out, table="GP summaries")

    association_frames = {
        path: read_association(path, columns, expected_ids)
        for path, columns in SOURCE_COLUMNS.items()
    }
    a1 = association_frames[ASSOCIATION_1]
    a2 = association_frames[ASSOCIATION_2]
    a4 = association_frames[ASSOCIATION_4]
    a5 = association_frames[ASSOCIATION_5]
    recode_audit: dict[str, object] = {}

    smoking, recode_audit["smoking_current"] = encode_smoking(
        a1["Current tobacco smoking | Instance 0"]
    )
    smoking.insert(0, "participant_id", a1["participant_id"].to_numpy())
    output = merge_one_to_one(output, smoking, table="smoking")

    education, recode_audit["education_higher"] = encode_education(
        a1["Qualifications | Instance 0"]
    )
    education.insert(0, "participant_id", a1["participant_id"].to_numpy())
    output = merge_one_to_one(output, education, table="education")

    townsend, recode_audit["townsend"] = encode_continuous(
        a1["Townsend deprivation index at recruitment"],
        value_name="townsend",
        missing_name="townsend_missing",
    )
    townsend.insert(0, "participant_id", a1["participant_id"].to_numpy())
    output = merge_one_to_one(output, townsend, table="Townsend")

    employment, recode_audit["employment_status"] = encode_employment(
        a2["Current employment status | Instance 0"]
    )
    employment.insert(0, "participant_id", a2["participant_id"].to_numpy())
    output = merge_one_to_one(output, employment, table="employment")

    sleep, recode_audit["sleep_duration"] = encode_continuous(
        a4["Sleep duration | Instance 0"],
        value_name="sleep_duration",
        missing_name="sleep_missing",
        minimum=0.0,
        maximum=24.0,
    )
    sleep.insert(0, "participant_id", a4["participant_id"].to_numpy())
    output = merge_one_to_one(output, sleep, table="sleep")

    alcohol, recode_audit["alcohol_status"] = encode_alcohol(
        a1["Alcohol drinker status | Instance 0"]
    )
    alcohol.insert(0, "participant_id", a1["participant_id"].to_numpy())
    output = merge_one_to_one(output, alcohol, table="alcohol sensitivity")

    activity, recode_audit["physical_activity_met"] = encode_continuous(
        a2["Summed MET minutes per week for all activity | Instance 0"],
        value_name="sensitivity_physical_activity_met",
        missing_name="sensitivity_physical_activity_missing",
        minimum=0.0,
    )
    activity["sensitivity_log1p_physical_activity_met"] = np.log1p(
        activity["sensitivity_physical_activity_met"]
    ).astype(np.float32)
    activity.insert(0, "participant_id", a2["participant_id"].to_numpy())
    output = merge_one_to_one(output, activity, table="physical activity sensitivity")

    pm25, recode_audit["pm25_2010"] = encode_continuous(
        a5["Particulate matter air pollution (pm2.5); 2010"],
        value_name="sensitivity_pm25_2010",
        missing_name="sensitivity_pm25_2010_missing",
        minimum=0.0,
    )
    pm25.insert(0, "participant_id", a5["participant_id"].to_numpy())
    output = merge_one_to_one(output, pm25, table="PM2.5 sensitivity")

    ordered_columns = (
        "participant_id",
        "fold_id",
        *PRIMARY_MODEL_COLUMNS,
        *AUXILIARY_COLUMNS,
        *SENSITIVITY_MODEL_COLUMNS,
    )
    if set(output.columns) != set(ordered_columns):
        missing = sorted(set(ordered_columns).difference(output.columns))
        extra = sorted(set(output.columns).difference(ordered_columns))
        raise CovariateMaterializationError(f"output schema mismatch: missing={missing}, extra={extra}")
    output = output.loc[:, ordered_columns].sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    require_exact_ids(output, expected_ids, table="materialized output")
    forbidden = sorted(
        column
        for column in output.columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_OUTPUT_FRAGMENTS)
    )
    if forbidden:
        raise CovariateMaterializationError(f"forbidden outcome/prediction columns: {forbidden}")
    if "sensitivity_pm25_2010" in PRIMARY_MODEL_COLUMNS:
        raise CovariateMaterializationError("PM2.5 2010 entered the primary model columns")
    # Four raw contextual values are intentionally unimputed; their paired
    # missing flags remain present for fold-local preprocessing.
    missing_primary = output.loc[:, PRIMARY_MODEL_COLUMNS].columns[
        output.loc[:, PRIMARY_MODEL_COLUMNS].isna().any()
    ].tolist()
    expected_missing = [
        "smoking_current",
        "education_higher",
        "townsend",
        "sleep_duration",
    ]
    if sorted(missing_primary) != sorted(expected_missing):
        raise CovariateMaterializationError(
            f"unexpected primary missing-value columns: {missing_primary}"
        )

    roles = {column: "primary_model" for column in PRIMARY_MODEL_COLUMNS}
    roles.update({column: "sensitivity_model" for column in SENSITIVITY_MODEL_COLUMNS})
    roles.update({column: "auxiliary_audit" for column in AUXILIARY_COLUMNS})
    roles.update({"participant_id": "identifier", "fold_id": "split_metadata"})
    missingness_path = args.output_dir / "covariate_missingness_range_audit.csv"
    timing_path = args.output_dir / "covariate_timing_source_audit.csv"
    recode_path = args.output_dir / "category_recode_audit.json"
    numeric_audit(output, roles).to_csv(missingness_path, index=False)
    timing_audit().to_csv(timing_path, index=False)
    recode_path.write_text(json.dumps(recode_audit, indent=2, ensure_ascii=True), encoding="utf-8")

    output.to_parquet(output_path, index=False, compression="zstd")
    source_paths = (AGE_COHORT, FOLDS, ICD_SUMMARY, GP_SUMMARY, *SOURCE_COLUMNS.keys())
    source_manifest = []
    for path in source_paths:
        source_manifest.append(
            {
                "path": str(path.resolve()),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "columns_read": list(
                    SOURCE_COLUMNS[path] if path in SOURCE_COLUMNS else PARQUET_SOURCE_COLUMNS[path]
                ),
            }
        )
    fold_counts = output["fold_id"].value_counts().sort_index().to_dict()
    manifest = {
        "schema_version": "revision0710_contextual_covariates_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "row_count": int(len(output)),
        "unique_participants": int(output["participant_id"].nunique()),
        "fold_counts": {str(int(key)): int(value) for key, value in fold_counts.items()},
        "output": {
            "path": str(output_path.resolve()),
            "bytes": int(output_path.stat().st_size),
            "sha256": sha256_file(output_path),
            "columns": list(output.columns),
            "dtypes": {column: str(dtype) for column, dtype in output.dtypes.items()},
        },
        "frozen_dictionary": {
            "path": str(DICTIONARY.resolve()),
            "sha256": sha256_file(DICTIONARY),
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "sources": source_manifest,
        "primary_model_columns": list(PRIMARY_MODEL_COLUMNS),
        "sensitivity_model_columns": list(SENSITIVITY_MODEL_COLUMNS),
        "auxiliary_audit_columns": list(AUXILIARY_COLUMNS),
        "requires_fit_partition_imputation": [
            "smoking_current",
            "education_higher",
            "townsend",
            "sleep_duration",
        ],
        "no_global_imputation_or_scaling_applied": True,
        "association_instance_policy": "Instance 0 only; no later-instance fallback",
        "ehr_timing_policy": "density uses event_date < index_date; prevalence uses event_date <= index_date",
        "pm25_policy": "sensitivity_only_timing_ambiguous_not_in_primary",
        "sealed_predictions_read": False,
        "legacy_cox_table_read": False,
        "outcome_columns_present": False,
        "audits": {
            "missingness_range": {
                "path": str(missingness_path.resolve()),
                "sha256": sha256_file(missingness_path),
            },
            "timing_source": {
                "path": str(timing_path.resolve()),
                "sha256": sha256_file(timing_path),
            },
            "category_recode": {
                "path": str(recode_path.resolve()),
                "sha256": sha256_file(recode_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
