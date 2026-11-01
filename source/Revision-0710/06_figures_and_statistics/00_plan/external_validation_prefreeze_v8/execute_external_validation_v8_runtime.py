from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from execute_external_validation_v8 import (
    MIMIC_BARE_OMR_SENSITIVITY,
    MIMIC_PRIMARY_VARIANT,
    MIMIC_RX_SENSITIVITY,
    NHANES_PRIMARY_VARIANT,
    ORGANS,
    aalen_johansen,
    adjudicate_endpoint,
    benjamini_hochberg,
    crop_variant_timeline,
    deterministic_hash_fold,
    encode_mimic_sex,
    encode_nhanes_sex,
    fit_crossfit_age_calibration,
    harrell_c,
    ipcw_binary_weights,
    mimic_integer_age,
    months_to_days,
    prepare_variant_timeline,
    sha256_file,
    require_immutable_embedding_prefix,
    validate_mimic_endpoint_rows,
    weighted_auc,
    weighted_brier,
    write_json_exclusive,
)
from v8_scientific_extensions import (
    SURVIVAL_CORE_RELATIVE_PATH,
    SURVIVAL_CORE_SHA256,
    build_complete_mimic_ph_diagnostics,
    build_density_adjusted_risk_sensitivity,
    build_label_blind_mimic_density,
    build_mimic_density_summary,
    build_nhanes_release_slot_qc,
    load_survival_core,
    require_nhanes_qc_pass,
)


ORGANS_LIST = list(ORGANS)
MIMIC_VARIANTS = [MIMIC_PRIMARY_VARIANT, MIMIC_BARE_OMR_SENSITIVITY, MIMIC_RX_SENSITIVITY]
RELEASES = ["P_2017_MARCH2020_PREPANDEMIC", "L_AUG2021_AUG2023"]
CHECKPOINT_AGE_ANCHORS = {
    0: ("age_anchor_train_folds_2_to_4_only", 57.03986740112305),
    1: ("age_anchor_train_role_only", 57.03695297241211),
    2: ("age_anchor_train_role_only", 57.03803253173828),
    3: ("age_anchor_train_role_only", 57.04105758666992),
    4: ("age_anchor_train_role_only", 57.04116439819336),
}

PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS = 4
CONDITIONAL_BOOTSTRAP_REPLICATES_PER_MODEL = 200
OPERATIONAL_RESERVATION_MIN_HOURS = 36.0
OPERATIONAL_RESERVATION_SAFER_HOURS = 48.0
RESOURCE_RESERVATION_COMPONENTS = [
    "label-blind cache construction",
    "CLIP vocabulary extension encoding and model restore",
    "prediction sealing and disk IO",
    "NHANES survey and age-metric aggregation",
    "stage05 and stage06 four-model conditional B=200 bootstrap statistics",
    "complete shared-core PH diagnostics for all target/death model covariates",
    "publication copy, rehash, and receipt validation",
]


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_parquet(path, index=False)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def _canonical_icd(code: object, version: object) -> str:
    if code is None or pd.isna(code):
        raise ValueError("missing ICD code")
    normalized = unicodedata.normalize("NFKC", str(code))
    value = normalized.strip().upper().replace(".", "")
    if not value or any(character.isspace() for character in value):
        raise ValueError("ICD code must not contain internal whitespace")
    if not value.isascii() or not value.isalnum():
        raise ValueError("invalid ICD code")
    version_value = int(version)
    if version_value == 10:
        if not value[0].isalpha() or not value[1:].isalnum():
            raise ValueError("invalid ICD10 code")
        return value
    if version_value == 9:
        return f"ICD9\x1f{value}"
    raise ValueError("unsupported ICD version")


def _participant_cohort(
    sources: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    patients = pd.read_csv(
        sources["mimic_patients"],
        compression="gzip",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )
    admissions = pd.read_csv(
        sources["mimic_admissions"],
        compression="gzip",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"],
    )
    flow: dict[str, int] = {
        "raw_subjects": int(patients["subject_id"].nunique()),
    }
    for column in ("admittime", "dischtime", "deathtime"):
        admissions[column] = pd.to_datetime(admissions[column], errors="coerce")
    demographics = patients.loc[
        patients[["gender", "anchor_age", "anchor_year"]].notna().all(axis=1)
    ].copy()
    flow["has_demographics"] = int(demographics["subject_id"].nunique())
    demographics = demographics.loc[demographics["gender"].isin(["M", "F"])].copy()
    flow["sex_binary_m_or_f"] = int(demographics["subject_id"].nunique())
    demographics = demographics.loc[demographics["anchor_age"].ne(91)].copy()
    flow["anchor_age_not_91_topcoded"] = int(demographics["subject_id"].nunique())
    frame = admissions.merge(
        demographics[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="inner",
        validate="many_to_one",
    )
    frame = frame.loc[
        frame["admittime"].notna()
        & frame["dischtime"].notna()
        & frame["dischtime"].gt(frame["admittime"])
    ].copy()
    if not (
        frame["anchor_age"].eq(np.floor(frame["anchor_age"]))
        & frame["anchor_year"].eq(np.floor(frame["anchor_year"]))
    ).all():
        raise AssertionError("MIMIC anchor age/year are not integer resolution")
    frame["chronological_age"] = (
        frame["anchor_age"].astype(int)
        + frame["dischtime"].dt.year.astype(int)
        - frame["anchor_year"].astype(int)
    )
    frame["sex_male"] = frame["gender"].map({"M": 1, "F": 0}).astype("int8")
    frame["hadm_numeric"] = pd.to_numeric(frame["hadm_id"], errors="raise")
    # Admission validity is defined before ordinal selection. Otherwise an
    # invalid/pediatric/in-hospital-death row can incorrectly occupy rank two.
    frame = frame.loc[frame["chronological_age"].ge(18)].copy()
    flow["adult_age_at_time_valid_discharge"] = int(frame["subject_id"].nunique())
    frame = frame.loc[
        frame["deathtime"].isna() | frame["deathtime"].gt(frame["dischtime"])
    ].copy()
    flow["alive_at_time_valid_adult_discharge"] = int(
        frame["subject_id"].nunique()
    )
    frame = frame.sort_values(
        ["subject_id", "admittime", "dischtime", "hadm_numeric"], kind="mergesort"
    )
    frame["valid_admission_order"] = (
        frame.groupby("subject_id", sort=False).cumcount() + 1
    )
    has_two = frame.groupby("subject_id", sort=False)["valid_admission_order"].max().ge(2)
    flow["has_two_valid_adult_surviving_discharge_admissions"] = int(has_two.sum())
    index = frame.loc[frame["valid_admission_order"].eq(2)].copy()
    flow["second_valid_adult_surviving_discharge_selected"] = int(
        index["subject_id"].nunique()
    )
    if index["subject_id"].duplicated().any():
        raise AssertionError("second valid admission is not unique")
    index["index_time"] = index["dischtime"]
    index["hash_fold"] = index["subject_id"].map(deterministic_hash_fold).astype("int8")
    columns = [
        "subject_id",
        "hadm_id",
        "index_time",
        "chronological_age",
        "sex_male",
        "hash_fold",
    ]
    return index[columns].reset_index(drop=True), frame, flow


def _update_observation_end(
    cohort: pd.DataFrame, sources: Mapping[str, Path]
) -> pd.Series:
    observed = cohort.set_index("subject_id")["index_time"].copy()
    subject_set = set(observed.index.astype(int))
    scans = [
        (sources["mimic_labevents"], "charttime", ["subject_id", "charttime"]),
        (sources["mimic_omr"], "chartdate", ["subject_id", "chartdate"]),
        (sources["mimic_prescriptions"], "stoptime", ["subject_id", "stoptime", "starttime"]),
    ]
    for path, primary_time, usecols in scans:
        for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=1_000_000):
            chunk = chunk.loc[chunk["subject_id"].isin(subject_set)].copy()
            if chunk.empty:
                continue
            if primary_time == "stoptime":
                stop = pd.to_datetime(chunk["stoptime"], errors="coerce")
                start = pd.to_datetime(chunk["starttime"], errors="coerce")
                chunk["event_time"] = stop.fillna(start)
            else:
                chunk["event_time"] = pd.to_datetime(chunk[primary_time], errors="coerce")
            maxima = chunk.groupby("subject_id")["event_time"].max()
            for subject_id, value in maxima.dropna().items():
                current = observed.get(subject_id)
                if pd.isna(current) or value > current:
                    observed.loc[subject_id] = value
    return observed


def _last_valid_discharge(admissions: pd.DataFrame) -> pd.Series:
    value = admissions.copy()
    value["admittime"] = pd.to_datetime(value["admittime"], errors="coerce")
    value["dischtime"] = pd.to_datetime(value["dischtime"], errors="coerce")
    value = value.loc[
        value["admittime"].notna()
        & value["dischtime"].notna()
        & value["dischtime"].gt(value["admittime"])
    ]
    return value.groupby("subject_id", sort=False)["dischtime"].max()


def _subject_followup_cap(
    death_date: pd.Series,
    last_valid_discharge: pd.Series,
    index_time: pd.Series,
) -> pd.Series:
    """Return the date-resolved DOD/last-discharge follow-up cap."""
    index = pd.to_datetime(index_time, errors="coerce")
    if index.index.has_duplicates or index.isna().any():
        raise AssertionError("follow-up cap index times are missing or duplicated")
    death = pd.to_datetime(death_date.reindex(index.index), errors="coerce")
    last_discharge = pd.to_datetime(
        last_valid_discharge.reindex(index.index), errors="coerce"
    )
    last_cap = last_discharge + pd.Timedelta(days=365)
    # patients.dod is date-only. Resolve it at the landmark clock time so a
    # death on calendar day +k is exactly k follow-up days, not k minus the
    # landmark's time-of-day fraction.
    resolved_death = index + (death.dt.normalize() - index.dt.normalize())
    aligned = pd.concat(
        [
            resolved_death.rename("resolved_death_time"),
            last_cap.rename("last_discharge_plus_365"),
        ],
        axis=1,
    )
    if aligned["last_discharge_plus_365"].isna().any():
        raise AssertionError("follow-up cap lacks a last valid discharge")
    return aligned.min(axis=1)


def _materialize_preindex_diagnosis_allowlist(
    cohort: pd.DataFrame,
    valid_admissions: pd.DataFrame,
    diagnosis_source: Path,
    destination: Path,
) -> dict[str, Any]:
    """Arrow-filter raw diagnosis batches before any conversion to pandas."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    index = cohort.set_index("subject_id")["index_time"]
    eligible = valid_admissions[
        ["subject_id", "hadm_id", "admittime", "dischtime"]
    ].copy()
    eligible["landmark_time"] = eligible["subject_id"].map(index)
    eligible = eligible.loc[
        eligible["landmark_time"].notna()
        & eligible["dischtime"].notna()
        & eligible["dischtime"].lt(eligible["landmark_time"])
    ].copy()
    if eligible["hadm_id"].duplicated().any():
        raise AssertionError("MIMIC pre-index admission allowlist is not unique")
    eligible_hadm = np.sort(
        pd.to_numeric(eligible["hadm_id"], errors="raise").to_numpy(dtype=np.int64)
    )
    allowed = pa.array(eligible_hadm, type=pa.int64())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    stream = pa.input_stream(str(diagnosis_source), compression="gzip")
    reader = pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(block_size=8 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(
            include_columns=[
                "subject_id",
                "hadm_id",
                "seq_num",
                "icd_code",
                "icd_version",
            ],
            column_types={"subject_id": pa.int64(), "hadm_id": pa.int64()},
        ),
    )
    writer: pq.ParquetWriter | None = None
    raw_rows = 0
    selected_rows = 0
    raw_batches = 0
    try:
        for batch in reader:
            raw_batches += 1
            batch_rows = batch.num_rows
            mask = pc.is_in(batch.column("hadm_id"), value_set=allowed)
            selected_index = pc.indices_nonzero(mask)
            filtered = batch.take(selected_index)
            global_row_number = pc.add(
                pc.cast(selected_index, pa.int64()), pa.scalar(raw_rows, pa.int64())
            )
            raw_rows += batch_rows
            if filtered.num_rows == 0:
                continue
            filtered = pa.RecordBatch.from_arrays(
                [*filtered.columns, global_row_number],
                names=[*filtered.schema.names, "source_row_number"],
            )
            if not pc.all(pc.is_in(filtered.column("hadm_id"), value_set=allowed)).as_py():
                raise AssertionError("post-index admission escaped Arrow predicate")
            if writer is None:
                writer = pq.ParquetWriter(destination, filtered.schema)
            writer.write_batch(filtered)
            selected_rows += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()
        stream.close()
    if writer is None:
        empty = pa.table(
            {
                "subject_id": pa.array([], type=pa.int64()),
                "hadm_id": pa.array([], type=pa.int64()),
                "seq_num": pa.array([], type=pa.float64()),
                "icd_code": pa.array([], type=pa.string()),
                "icd_version": pa.array([], type=pa.float64()),
                "source_row_number": pa.array([], type=pa.int64()),
            }
        )
        pq.write_table(empty, destination)
    receipt = {
        "schema_version": "revision0710.external_validation.preindex_diagnosis_materialization.v8",
        "status": "PREINDEX_ONLY_ARROW_ALLOWLIST_MATERIALIZED_BEFORE_PANDAS",
        "predicate": "hadm_id in completed-admission allowlist with dischtime < landmark",
        "raw_source_read_api": "pyarrow.csv.open_csv_record_batch_stream",
        "raw_rows_converted_to_pandas": 0,
        "raw_batches": raw_batches,
        "raw_rows_streamed": raw_rows,
        "preindex_rows_materialized": selected_rows,
        "allowlist_hadm_count": len(eligible_hadm),
        "output_bytes": destination.stat().st_size,
        "output_sha256": sha256_file(destination),
    }
    receipt_path = destination.with_name("PREINDEX_DIAGNOSIS_MATERIALIZATION_V8.json")
    write_json_exclusive(receipt_path, receipt)
    return {"table": destination, "receipt": receipt_path, **receipt}


def _mimic_diagnosis_events(
    cohort: pd.DataFrame,
    valid_admissions: pd.DataFrame,
    sources: Mapping[str, Path],
    preindex_diagnosis_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize only diagnoses from strictly pre-index valid admissions."""
    dictionary = pd.read_csv(
        sources["mimic_diagnosis_dictionary"], compression="gzip"
    )
    dictionary["key"] = list(
        zip(
            dictionary["icd_code"].astype(str),
            dictionary["icd_version"].astype(int),
        )
    )
    title = dict(zip(dictionary["key"], dictionary["long_title"].astype(str)))
    index = cohort.set_index("subject_id")["index_time"]
    eligible = valid_admissions[
        ["subject_id", "hadm_id", "admittime", "dischtime"]
    ].copy()
    eligible["landmark_time"] = eligible["subject_id"].map(index)
    eligible = eligible.loc[
        eligible["landmark_time"].notna()
        & eligible["dischtime"].notna()
        & eligible["dischtime"].lt(eligible["landmark_time"])
    ].copy()
    if eligible["hadm_id"].duplicated().any():
        raise AssertionError("MIMIC admission ID is not unique")
    admission_lookup = eligible.set_index("hadm_id")
    eligible_hadm = set(admission_lookup.index.tolist())
    rows: list[pd.DataFrame] = []
    invalid_rows: list[dict[str, Any]] = []
    preindex_only = pd.read_parquet(preindex_diagnosis_path)
    for raw_chunk in (preindex_only,):
        if not raw_chunk["hadm_id"].isin(eligible_hadm).all():
            raise AssertionError("pre-index materialization contains a forbidden admission")
        source_numbers = raw_chunk["source_row_number"].to_numpy(dtype=np.int64)
        chunk = raw_chunk.copy()
        if chunk.empty:
            continue
        chunk["source_row_number"] = source_numbers
        chunk["event_time"] = chunk["hadm_id"].map(admission_lookup["dischtime"])
        chunk["landmark_time"] = chunk["hadm_id"].map(
            admission_lookup["landmark_time"]
        )
        canonical_codes: list[str | None] = []
        code_systems: list[str] = []
        valid_mask: list[bool] = []
        for record in chunk.itertuples(index=False):
            try:
                version = int(record.icd_version)
            except (TypeError, ValueError):
                version = -1
            code_system = (
                "ICD9"
                if version == 9
                else "ICD10"
                if version == 10
                else "not_applicable"
            )
            try:
                canonical = _canonical_icd(record.icd_code, version)
            except (TypeError, ValueError):
                canonical = None
                invalid_rows.append(
                    {
                        "modality": "hospital_icd",
                        "code_system": code_system,
                        "reason": "invalid_or_unsupported_code",
                        "participant_id": str(record.subject_id),
                    }
                )
            canonical_codes.append(canonical)
            code_systems.append(code_system)
            valid_mask.append(canonical is not None)
        chunk["canonical_code"] = canonical_codes
        chunk["code_system"] = code_systems
        chunk = chunk.loc[np.asarray(valid_mask, dtype=bool)].copy()
        if chunk.empty:
            continue
        chunk["canonical_description"] = [
            title.get(
                (str(code), int(version)),
                f"ICD{int(version)} code {str(code)}",
            )
            for code, version in zip(chunk["icd_code"], chunk["icd_version"])
        ]
        chunk["participant_id"] = chunk["subject_id"].astype(str)
        chunk["source"] = "hospital_icd"
        chunk["event_type_id"] = 1
        rows.append(
            chunk[
                [
                    "participant_id",
                    "event_time",
                    "landmark_time",
                    "source",
                    "code_system",
                    "event_type_id",
                    "canonical_code",
                    "canonical_description",
                    "source_row_number",
                ]
            ]
        )
    columns = [
        "participant_id",
        "event_time",
        "landmark_time",
        "source",
        "code_system",
        "event_type_id",
        "canonical_code",
        "canonical_description",
        "source_row_number",
    ]
    events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    audit = pd.DataFrame(
        invalid_rows,
        columns=["modality", "code_system", "reason", "participant_id"],
    )
    return events, audit


def _mimic_prescription_events(
    cohort: pd.DataFrame, sources: Mapping[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = cohort.set_index("subject_id")["index_time"]
    rows: list[pd.DataFrame] = []
    invalid_rows: list[dict[str, Any]] = []
    row_offset = 0
    for raw_chunk in pd.read_csv(
        sources["mimic_prescriptions"],
        compression="gzip",
        usecols=["subject_id", "starttime", "drug"],
        chunksize=1_000_000,
    ):
        source_numbers = np.arange(
            row_offset, row_offset + len(raw_chunk), dtype=np.int64
        )
        row_offset += len(raw_chunk)
        raw_chunk["landmark_time"] = raw_chunk["subject_id"].map(index)
        raw_chunk["event_time"] = pd.to_datetime(
            raw_chunk["starttime"], errors="coerce"
        )
        preindex_mask = (
            raw_chunk["landmark_time"].notna()
            & raw_chunk["event_time"].notna()
            & raw_chunk["event_time"].lt(raw_chunk["landmark_time"])
        )
        chunk = raw_chunk.loc[preindex_mask].copy()
        if chunk.empty:
            continue
        chunk["source_row_number"] = source_numbers[preindex_mask.to_numpy()]
        chunk["canonical_description"] = chunk["drug"].map(
            lambda value: "" if pd.isna(value) else " ".join(str(value).split())
        )
        invalid = chunk["canonical_description"].eq("")
        invalid_rows.extend(
            {
                "modality": "inpatient_prescription_proxy",
                "code_system": "INPATIENT_RX",
                "reason": "invalid_or_missing_text",
                "participant_id": str(subject_id),
            }
            for subject_id in chunk.loc[invalid, "subject_id"]
        )
        chunk = chunk.loc[~invalid].copy()
        if chunk.empty:
            continue
        chunk["participant_id"] = chunk["subject_id"].astype(str)
        chunk["source"] = "inpatient_prescription_proxy"
        chunk["code_system"] = "INPATIENT_RX"
        chunk["event_type_id"] = 2
        chunk["canonical_code"] = ""
        rows.append(
            chunk[
                [
                    "participant_id",
                    "event_time",
                    "landmark_time",
                    "source",
                    "code_system",
                    "event_type_id",
                    "canonical_code",
                    "canonical_description",
                    "source_row_number",
                ]
            ]
        )
    columns = [
        "participant_id",
        "event_time",
        "landmark_time",
        "source",
        "code_system",
        "event_type_id",
        "canonical_code",
        "canonical_description",
        "source_row_number",
    ]
    events = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    audit = pd.DataFrame(
        invalid_rows,
        columns=["modality", "code_system", "reason", "participant_id"],
    )
    return events, audit


def _convert_formula(formula: str, value: pd.Series) -> pd.Series:
    formulas = {
        "value": 1.0,
        "value * 2.54": 2.54,
        "value * 0.45359237": 0.45359237,
        "value * 88.4": 88.4,
        "value * 10": 10.0,
        "value * 17.104": 17.104,
        "value * 0.357": 0.357,
        "value * 0.2495": 0.2495,
        "value * 0.02586": 0.02586,
        "value * 0.0555": 0.0555,
        "value * 0.3229": 0.3229,
        "value * 0.0347": 0.0347,
        "value * 0.01129": 0.01129,
        "value * 59.48": 59.48,
        "value * 2.496": 2.496,
    }
    for prefix in (
        "value;",
        "51133:value; 52769:value/1000; 53157:value",
    ):
        if formula.startswith(prefix):
            return value
    if formula == "(value - 2.15) * 10.929":
        return (value - 2.15) * 10.929
    if formula not in formulas:
        raise ValueError(f"unsupported frozen formula: {formula}")
    return value * formulas[formula]


def _value_rule_accepts(value: float, rule: str, source_unit: str) -> bool:
    if not np.isfinite(value):
        return False
    if "2<=value<=25" in rule:
        return 2.0 <= value <= 25.0
    if "percent branch 0<=value<=100" in rule:
        return 0.0 <= value <= 100.0 if source_unit == "%" else value >= 0.0
    if "0<=value<=100" in rule or source_unit == "%":
        return 0.0 <= value <= 100.0
    if "negative values rejected" in rule:
        return value >= 0.0
    raise ValueError(f"unsupported frozen primary_value_rule: {rule}")


def _exact_omr_selector_names(selector: str) -> set[str]:
    if selector.startswith("result_name == '") and selector.endswith("'"):
        return {selector[len("result_name == '") : -1]}
    prefix = "result_name in {"
    if selector.startswith(prefix) and selector.endswith("}"):
        return {name for name in selector[len(prefix) : -1].split(",") if name}
    raise ValueError(f"unsupported frozen OMR selector: {selector}")


def _lab_available_preindex_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["charttime"].notna()
        & frame["storetime"].notna()
        & frame["valuenum"].notna()
        & frame["charttime"].le(frame["index_time"])
        & frame["storetime"].le(frame["index_time"])
        & frame["charttime"].ge(frame["index_time"] - pd.Timedelta(days=365))
    )


def _omr_available_preindex_mask(frame: pd.DataFrame) -> pd.Series:
    index_date = frame["index_time"].dt.normalize()
    return (
        frame["chartdate"].notna()
        & frame["chartdate"].lt(index_date)
        & frame["chartdate"].ge(index_date - pd.Timedelta(days=365))
    )


def _build_mimic_features(
    cohort: pd.DataFrame, artifact_dir: Path, sources: Mapping[str, Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    mapping = pd.read_csv(artifact_dir / "MIMIC_78_SLOT_MAP_V8.csv", keep_default_na=False)
    registry = json.loads(
        (artifact_dir / "SOURCE_REGISTRY_V8.json").read_text(encoding="utf-8")
    )
    source_hashes = {
        row["source_id"]: row["sha256"] for row in registry["external_sources"]
    }
    subject_order = cohort["subject_id"].astype(int).tolist()
    subject_to_row = {subject_id: index for index, subject_id in enumerate(subject_order)}
    index_time = cohort.set_index("subject_id")["index_time"]
    raw = np.zeros((len(cohort), 78), dtype=np.float32)
    missing = np.ones((len(cohort), 78), dtype=bool)
    coverage = []
    lab_map: dict[tuple[int, str], list[tuple[int, str, str]]] = {}
    for row in mapping.loc[mapping["source_file_id"].eq("mimic_labevents")].itertuples(index=False):
        for branch in row.primary_source_selector.split("|"):
            itemid = int(branch.split("[", 1)[0])
            unit = branch.split("valueuom=", 1)[1].rstrip("]")
            lab_map.setdefault((itemid, unit), []).append((int(row.slot_index), row.primary_formula, row.primary_value_rule))
    latest: dict[
        tuple[int, int], tuple[pd.Timestamp, pd.Timestamp, int, float]
    ] = {}
    bare_extra: dict[
        tuple[int, int], tuple[pd.Timestamp, pd.Timestamp, int, float]
    ] = {}
    allowed_itemids = {key[0] for key in lab_map}
    for chunk in pd.read_csv(
        sources["mimic_labevents"],
        compression="gzip",
        usecols=["labevent_id", "subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"],
        chunksize=1_000_000,
    ):
        chunk = chunk.loc[chunk["subject_id"].isin(subject_to_row) & chunk["itemid"].isin(allowed_itemids)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["storetime"] = pd.to_datetime(chunk["storetime"], errors="coerce")
        chunk["index_time"] = chunk["subject_id"].map(index_time)
        chunk = chunk.loc[_lab_available_preindex_mask(chunk)]
        for record in chunk.itertuples(index=False):
            branches = lab_map.get((int(record.itemid), str(record.valueuom)), [])
            for slot, formula, value_rule in branches:
                value = float(record.valuenum)
                if not _value_rule_accepts(value, value_rule, str(record.valueuom)):
                    continue
                if slot == 24 and int(record.itemid) == 52769:
                    converted = value / 1000.0
                else:
                    converted = float(_convert_formula(formula, pd.Series([value])).iloc[0])
                key = (int(record.subject_id), slot)
                candidate = (
                    record.charttime,
                    record.storetime,
                    int(record.labevent_id),
                    converted,
                )
                if key not in latest or candidate[:3] > latest[key][:3]:
                    latest[key] = candidate
    omr_rows = mapping.loc[mapping["source_file_id"].eq("mimic_omr")]
    allowed_names = {"Height", "Weight"}
    for row in omr_rows.itertuples(index=False):
        allowed_names.update(_exact_omr_selector_names(row.primary_source_selector))
    for chunk in pd.read_csv(
        sources["mimic_omr"],
        compression="gzip",
        usecols=["subject_id", "chartdate", "seq_num", "result_name", "result_value"],
        chunksize=500_000,
    ):
        chunk = chunk.loc[chunk["subject_id"].isin(subject_to_row) & chunk["result_name"].isin(allowed_names)].copy()
        chunk["chartdate"] = pd.to_datetime(chunk["chartdate"], errors="coerce")
        chunk["index_time"] = chunk["subject_id"].map(index_time)
        chunk = chunk.loc[_omr_available_preindex_mask(chunk)]
        for record in chunk.itertuples(index=False):
            name = str(record.result_name)
            text = str(record.result_value)
            if name in {"Height", "Weight"}:
                try:
                    source_value = float(text)
                except ValueError:
                    source_value = math.nan
                slot = 3 if name == "Height" else 11
                converted = math.nan
                if name == "Height" and 40 <= source_value <= 90:
                    converted = source_value * 2.54
                elif name == "Height" and 120 <= source_value <= 230:
                    converted = source_value
                elif name == "Weight" and 50 <= source_value <= 1000:
                    converted = source_value * 0.45359237
                if np.isfinite(converted):
                    key = (int(record.subject_id), slot)
                    candidate = (
                        record.chartdate,
                        record.chartdate,
                        int(record.seq_num),
                        float(converted),
                    )
                    if key not in bare_extra or candidate[:3] > bare_extra[key][:3]:
                        bare_extra[key] = candidate
            for row in omr_rows.itertuples(index=False):
                if name not in _exact_omr_selector_names(row.primary_source_selector):
                    continue
                if int(row.slot_index) in {8, 9}:
                    parts = text.split("/")
                    if len(parts) != 2:
                        continue
                    try:
                        systolic, diastolic = map(float, parts)
                    except ValueError:
                        continue
                    if not (20 <= diastolic <= 200 and 40 <= systolic <= 300 and systolic > diastolic):
                        continue
                    value = diastolic if int(row.slot_index) == 8 else systolic
                else:
                    try:
                        source_value = float(text)
                    except ValueError:
                        continue
                    slot = int(row.slot_index)
                    if slot == 3 and not 40 <= source_value <= 90:
                        continue
                    if slot == 10 and not 10 <= source_value <= 100:
                        continue
                    if slot == 11 and not 50 <= source_value <= 1000:
                        continue
                    value = float(_convert_formula(row.primary_formula, pd.Series([source_value])).iloc[0])
                key = (int(record.subject_id), int(row.slot_index))
                candidate = (
                    record.chartdate,
                    record.chartdate,
                    int(record.seq_num),
                    value,
                )
                if key not in latest or candidate[:3] > latest[key][:3]:
                    latest[key] = candidate
    for (subject_id, slot), (*_, value) in latest.items():
        row_index = subject_to_row[subject_id]
        raw[row_index, slot] = value
        missing[row_index, slot] = False
    raw_bare = raw.copy()
    missing_bare = missing.copy()
    for key, candidate in bare_extra.items():
        primary_candidate = latest.get(key)
        if primary_candidate is not None and primary_candidate[:3] >= candidate[:3]:
            continue
        subject_id, slot = key
        row_index = subject_to_row[subject_id]
        raw_bare[row_index, slot] = candidate[3]
        missing_bare[row_index, slot] = False
    for row in mapping.itertuples(index=False):
        slot = int(row.slot_index)
        for variant in MIMIC_VARIANTS:
            is_bare = variant == MIMIC_BARE_OMR_SENSITIVITY
            variant_missing = missing_bare if is_bare else missing
            observed = int((~variant_missing[:, slot]).sum())
            coverage.append(
                {
                    "input_variant": variant,
                    "slot_index": slot,
                    "feature_name": row.feature_name,
                    "map_status": row.map_status,
                    "observed_n": observed,
                    "eligible_n": len(cohort),
                    "observed_fraction": observed / len(cohort) if len(cohort) else math.nan,
                    "source_file_sha256": (
                        source_hashes.get(str(row.source_file_id), "0" * 64)
                    ),
                    "source_selector": (
                        row.sensitivity_source_selector
                        if is_bare
                        else row.primary_source_selector
                    ),
                    "source_unit": (
                        row.sensitivity_source_unit if is_bare else row.primary_source_unit
                    ),
                    "target_unit": row.target_unit,
                    "formula": (
                        row.sensitivity_formula if is_bare else row.primary_formula
                    ),
                    "value_rule": (
                        row.sensitivity_value_rule if is_bare else row.primary_value_rule
                    ),
                    "all_78_slots_adjudicated": True,
                }
            )
    return raw, missing, raw_bare, missing_bare, pd.DataFrame(coverage)


def _build_nhanes_features(
    artifact_dir: Path, sources: Mapping[str, Path]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    mapping = pd.read_csv(artifact_dir / "NHANES_78_SLOT_RELEASE_MAP_V8.csv", keep_default_na=False)
    registry = json.loads(
        (artifact_dir / "SOURCE_REGISTRY_V8.json").read_text(encoding="utf-8")
    )
    source_hashes = {
        row["source_id"]: row["sha256"] for row in registry["external_sources"]
    }
    participants = []
    raw_parts = []
    missing_parts = []
    coverage_rows = []
    for release_id, demo_id in zip(RELEASES, ["nhanes_p_demo", "nhanes_l_demo"]):
        demo = pd.read_sas(sources[demo_id], format="xport", encoding="latin1")
        primary_age_candidate = _nhanes_primary_integer_age_guard(
            demo["RIDAGEYR"], demo["RIDSTATR"]
        )
        demo = demo.loc[
            primary_age_candidate & demo["RIAGENDR"].isin([1, 2])
        ].copy()
        demo["participant_id"] = demo["SEQN"].astype(int).astype(str)
        demo["chronological_age"] = demo["RIDAGEYR"].astype(float)
        demo["sex_male"] = demo["RIAGENDR"].map({1: 1, 2: 0}).astype("int8")
        demo["release_id"] = release_id
        demo["hash_fold"] = demo["participant_id"].map(deterministic_hash_fold).astype("int8")
        frames: dict[str, pd.DataFrame] = {}
        release_map = mapping.loc[mapping["release_id"].eq(release_id)]
        values = np.zeros((len(demo), 78), dtype=np.float32)
        masks = np.ones((len(demo), 78), dtype=bool)
        seqn_order = demo["SEQN"].to_numpy()
        for row in release_map.itertuples(index=False):
            if not row.source_file_id:
                coverage_rows.append({"release_id": release_id, "input_variant": NHANES_PRIMARY_VARIANT, "slot_index": int(row.slot_index), "feature_name": row.feature_name, "map_status": row.map_status, "observed_n": 0, "eligible_n": len(demo), "observed_fraction": 0.0, "source_file_sha256": None, "all_78_slots_adjudicated": True})
                continue
            if row.source_file_id not in frames:
                frames[row.source_file_id] = pd.read_sas(sources[row.source_file_id], format="xport", encoding="latin1").set_index("SEQN")
            source = frames[row.source_file_id]
            variables = row.source_variables.split("|")
            aligned = source.reindex(seqn_order)
            if row.formula in variables:
                result = pd.to_numeric(aligned[row.formula], errors="coerce")
            elif row.formula == "URXUCR * 88.4":
                result = pd.to_numeric(aligned["URXUCR"], errors="coerce") * 88.4
            elif row.formula == "LBXPLTSI * LBXMPSI / 10000.0":
                result = pd.to_numeric(aligned["LBXPLTSI"], errors="coerce") * pd.to_numeric(aligned["LBXMPSI"], errors="coerce") / 10000.0
            elif row.formula == "(LBXGH - 2.15) * 10.929":
                result = (pd.to_numeric(aligned["LBXGH"], errors="coerce") - 2.15) * 10.929
            elif row.formula.startswith("BMXWAIST / BMXHIP"):
                waist = pd.to_numeric(aligned["BMXWAIST"], errors="coerce")
                hip = pd.to_numeric(aligned["BMXHIP"], errors="coerce")
                result = waist / hip.where(hip > 0)
            else:
                raise ValueError(f"unsupported NHANES frozen formula: {row.formula}")
            observed = result.notna().to_numpy()
            values[observed, int(row.slot_index)] = result.loc[observed].to_numpy(dtype=np.float32)
            masks[observed, int(row.slot_index)] = False
            coverage_rows.append({"release_id": release_id, "input_variant": NHANES_PRIMARY_VARIANT, "slot_index": int(row.slot_index), "feature_name": row.feature_name, "map_status": row.map_status, "observed_n": int(observed.sum()), "eligible_n": len(demo), "observed_fraction": float(observed.mean()) if len(observed) else math.nan, "source_file_sha256": source_hashes[row.source_file_id], "all_78_slots_adjudicated": True})
        participants.append(demo[["participant_id", "release_id", "chronological_age", "sex_male", "hash_fold", "SEQN", "SDMVSTRA", "SDMVPSU", "WTMECPRP" if release_id == RELEASES[0] else "WTMEC2YR"]])
        raw_parts.append(values)
        missing_parts.append(masks)
    return pd.concat(participants, ignore_index=True), np.concatenate(raw_parts), np.concatenate(missing_parts), pd.DataFrame(coverage_rows)


def _nhanes_primary_integer_age_guard(
    age: pd.Series, ridstatr: pd.Series
) -> pd.Series:
    values = pd.to_numeric(age, errors="coerce")
    candidate = ridstatr.eq(2) & values.between(40, 73, inclusive="both")
    if (candidate & values.ne(np.floor(values))).any():
        raise AssertionError("NHANES primary age contains fractional RIDAGEYR")
    return candidate


def _build_nhanes_survey_design_universe(
    sources: Mapping[str, Path],
) -> pd.DataFrame:
    """Retain the full examined release design; 40-73 is a domain flag only."""
    rows = []
    for release_id, demo_id, weight_variable in (
        (RELEASES[0], "nhanes_p_demo", "WTMECPRP"),
        (RELEASES[1], "nhanes_l_demo", "WTMEC2YR"),
    ):
        demo = pd.read_sas(sources[demo_id], format="xport", encoding="latin1")
        required = {
            "SEQN",
            "RIDSTATR",
            "RIDAGEYR",
            "RIAGENDR",
            "SDMVSTRA",
            "SDMVPSU",
            weight_variable,
        }
        if not required.issubset(demo.columns):
            raise AssertionError(f"NHANES full design columns missing: {release_id}")
        demo = demo.loc[demo["RIDSTATR"].eq(2)].copy()
        age = pd.to_numeric(demo["RIDAGEYR"], errors="coerce")
        domain = (
            age.between(40, 73, inclusive="both")
            & age.eq(np.floor(age))
            & demo["RIAGENDR"].isin([1, 2])
        )
        frame = pd.DataFrame(
            {
                "participant_id": demo["SEQN"].astype(int).astype(str),
                "release_id": release_id,
                "chronological_age": age.astype(float),
                "sex_male": demo["RIAGENDR"].map({1: 1, 2: 0}),
                "SDMVSTRA": demo["SDMVSTRA"],
                "SDMVPSU": demo["SDMVPSU"],
                "survey_weight": pd.to_numeric(
                    demo[weight_variable], errors="coerce"
                ),
                "weight_variable": weight_variable,
                "domain_40_73": domain.astype(bool),
            }
        )
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["release_id", "participant_id"]).any():
        raise AssertionError("NHANES full survey design key is not unique")
    return result


def build_label_blind_cache(
    *,
    root: Path,
    artifact_dir: Path,
    stage: Path,
    sources: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> dict[str, Path]:
    cache_dir = stage / "shared/02_label_blind_cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    mimic, admissions, mimic_flow = _participant_cohort(sources)
    preindex_materialization = _materialize_preindex_diagnosis_allowlist(
        mimic,
        admissions,
        sources["mimic_diagnoses"],
        cache_dir / "mimic_diagnoses_preindex_allowlist.parquet",
    )
    mimic_events, mimic_drop_audit = _mimic_diagnosis_events(
        mimic,
        admissions,
        sources,
        Path(preindex_materialization["table"]),
    )
    rx_events, rx_drop_audit = _mimic_prescription_events(mimic, sources)
    vocabulary_drop_audit = pd.concat(
        [mimic_drop_audit, rx_drop_audit], ignore_index=True
    )
    mimic_raw, mimic_missing, mimic_bare_raw, mimic_bare_missing, mimic_coverage = _build_mimic_features(mimic, artifact_dir, sources)
    nhanes, nhanes_raw, nhanes_missing, nhanes_coverage = _build_nhanes_features(artifact_dir, sources)
    nhanes_design = _build_nhanes_survey_design_universe(sources)
    density = build_label_blind_mimic_density(
        mimic_participants=mimic,
        icd_events=mimic_events,
        prescription_events=rx_events,
        phenotype_missing=mimic_missing,
        bare_phenotype_missing=mimic_bare_missing,
        crop_variant_timeline=crop_variant_timeline,
        prepare_variant_timeline=prepare_variant_timeline,
    )
    nhanes_qc = build_nhanes_release_slot_qc(
        participants=nhanes,
        raw=nhanes_raw,
        missing=nhanes_missing,
        release_slot_map=pd.read_csv(
            artifact_dir / "NHANES_78_SLOT_RELEASE_MAP_V8.csv",
            keep_default_na=False,
        ),
    )
    _write_parquet(cache_dir / "mimic_participants.parquet", mimic)
    _write_parquet(cache_dir / "nhanes_participants.parquet", nhanes)
    _write_parquet(cache_dir / "nhanes_survey_design_universe.parquet", nhanes_design)
    _write_parquet(cache_dir / "mimic_icd_events_preindex.parquet", mimic_events)
    _write_parquet(cache_dir / "mimic_inpatient_prescription_proxy_preindex.parquet", rx_events)
    _write_parquet(cache_dir / "mimic_vocabulary_drop_audit.parquet", vocabulary_drop_audit)
    _write_parquet(cache_dir / "mimic_feature_coverage.parquet", mimic_coverage)
    _write_parquet(cache_dir / "nhanes_feature_coverage.parquet", nhanes_coverage)
    _write_parquet(cache_dir / "mimic_participant_density_label_blind.parquet", density)
    _write_csv(cache_dir / "nhanes_release_slot_qc.csv", nhanes_qc)
    require_nhanes_qc_pass(nhanes_qc)
    np.savez_compressed(cache_dir / "phenotypes.npz", mimic_raw=mimic_raw, mimic_missing=mimic_missing, mimic_bare_raw=mimic_bare_raw, mimic_bare_missing=mimic_bare_missing, nhanes_raw=nhanes_raw, nhanes_missing=nhanes_missing)
    bridge = cache_dir / "inference_bridge"
    bridge.mkdir(parents=True, exist_ok=False)
    _write_csv_gzip(bridge / "mimic_participants.csv.gz", mimic)
    _write_csv_gzip(bridge / "nhanes_participants.csv.gz", nhanes)
    _write_csv_gzip(bridge / "mimic_icd_events_preindex.csv.gz", mimic_events)
    _write_csv_gzip(
        bridge / "mimic_inpatient_prescription_proxy_preindex.csv.gz", rx_events
    )
    code_vocab_path = (
        root
        / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_code_vocab.parquet"
    )
    text_vocab_path = (
        root
        / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_text_vocab.parquet"
    )
    _write_csv_gzip(
        bridge / "frozen_event_code_vocab.csv.gz", pd.read_parquet(code_vocab_path)
    )
    _write_csv_gzip(
        bridge / "frozen_event_text_vocab.csv.gz", pd.read_parquet(text_vocab_path)
    )
    full_registry = json.loads(
        (artifact_dir / "SOURCE_REGISTRY_V8.json").read_text(encoding="utf-8")
    )
    inference_ids = {
        "event_code_vocab",
        "event_text_vocab",
        "embedding_tables",
        "ehr_age_model",
        "private_retain_age",
        "replication_contract",
        "clip_weight",
        *{f"checkpoint_fold_{fold}" for fold in range(5)},
        *{f"phenotype_transform_fold_{fold}" for fold in range(5)},
        *{f"correction_fold_{fold}" for fold in range(5)},
    }
    inference_records = [
        record
        for record in full_registry["locked_artifacts"]
        if record["artifact_id"] in inference_ids
    ]
    if {record["artifact_id"] for record in inference_records} != inference_ids:
        raise AssertionError("inference-only artifact registry is incomplete")
    inference_artifact_dir = cache_dir / "inference_worker_artifacts"
    inference_artifact_dir.mkdir(parents=True, exist_ok=False)
    inference_registry_path = (
        inference_artifact_dir / "INFERENCE_LOCKED_ARTIFACT_REGISTRY_V8.json"
    )
    write_json_exclusive(
        inference_registry_path,
        {
            "schema_version": "revision0710.external_validation.inference_locked_artifact_registry.v8",
            "status": "NO_EXTERNAL_SOURCE_PATHS_IN_INFERENCE_PROCESS",
            "external_sources": [],
            "locked_artifacts": inference_records,
        },
    )
    manifest = {
        "schema_version": "revision0710.external_validation.label_blind_cache.v8",
        "status": "LABEL_BLIND_CACHE_BUILT_BEFORE_OUTCOME_READ",
        "questionnaire_files_read": [],
        "postindex_diagnosis_rows_read_into_cache": 0,
        "postindex_diagnosis_rows_materialized_before_prediction_seal": 0,
        "diagnosis_source_streamed_with_preindex_hadm_filter": True,
        "diagnosis_stream_api": "pyarrow.csv.open_csv_record_batch_stream",
        "diagnosis_preindex_materialization_receipt_sha256": sha256_file(
            Path(preindex_materialization["receipt"])
        ),
        "raw_diagnosis_rows_converted_to_pandas_before_allowlist": 0,
        "diagnosis_canonicalization_after_preindex_hadm_filter_only": True,
        "postindex_observation_or_death_fields_read": [],
        "inference_worker_source_allowlist": [
            "inference_bridge/*.csv.gz",
            "phenotypes.npz",
            "locked UKB transforms/checkpoints/embedding table",
        ],
        "inference_worker_raw_external_source_paths_received": [],
        "mimic_participants": len(mimic),
        "mimic_flow_counts_before_outcome_read": mimic_flow,
        "nhanes_participants": len(nhanes),
        "nhanes_full_examined_design_rows": len(nhanes_design),
        "nhanes_40_73_is_survey_domain_not_design_filter": True,
        "mimic_density_participant_variant_rows": len(density),
        "mimic_density_label_or_outcome_fields_read": False,
        "nhanes_release_slot_qc_rows": len(nhanes_qc),
        "nhanes_external_data_driven_trimming_or_winsorization": False,
        "nhanes_qc_quantiles_used_for_decision": False,
        "input_variants": [*MIMIC_VARIANTS, NHANES_PRIMARY_VARIANT],
        "files": {},
    }
    for path in sorted(value for value in cache_dir.rglob("*") if value.is_file()):
        if path.name == "MANIFEST.json":
            continue
        relative = path.relative_to(cache_dir).as_posix()
        manifest["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    write_json_exclusive(cache_dir / "MANIFEST.json", manifest)
    return {
        "directory": cache_dir,
        "manifest": cache_dir / "MANIFEST.json",
        "inference_artifact_dir": inference_artifact_dir,
        "inference_registry": inference_registry_path,
        "mimic_density": cache_dir / "mimic_participant_density_label_blind.parquet",
        "nhanes_release_slot_qc": cache_dir / "nhanes_release_slot_qc.csv",
    }


def _encode_extension_rows(
    texts: Sequence[str], *, model: Any, tokenizer: Any
) -> np.ndarray:
    output = []
    import torch

    with torch.inference_mode():
        for start in range(0, len(texts), 4):
            tokens = tokenizer(list(texts[start : start + 4])).cpu()
            output.append(model.encode_text(tokens).float().cpu().numpy().astype("<f2"))
    if not output:
        return np.empty((0, 512), dtype="<f2")
    return np.ascontiguousarray(np.concatenate(output, axis=0), dtype="<f2")


def _extend_vocab_and_embeddings(
    root: Path,
    events: pd.DataFrame,
    registry: Mapping[str, Any],
    output_dir: Path,
    cache_dir: Path,
) -> tuple[dict[str, int], dict[tuple[str, str], int], Any, Any, pd.DataFrame]:
    import torch

    code_vocab_path = root / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_code_vocab.parquet"
    text_vocab_path = root / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_text_vocab.parquet"
    embedding_path = root / "Revision-0710/01_data_rebuild/04_cache_v1/embedding_tables.pt"
    bridge = cache_dir / "inference_bridge"
    code_vocab = pd.read_csv(bridge / "frozen_event_code_vocab.csv.gz").sort_values(
        "vocab_id", kind="mergesort"
    )
    text_vocab = pd.read_csv(bridge / "frozen_event_text_vocab.csv.gz").sort_values(
        "vocab_id", kind="mergesort"
    )
    if code_vocab["vocab_id"].tolist() != list(range(9906)) or text_vocab["vocab_id"].tolist() != list(range(44565)):
        raise AssertionError("original vocabulary row order/count mismatch")
    code_to_id = dict(zip(code_vocab["raw_text"].astype(str), code_vocab["vocab_id"].astype(int)))
    text_to_id = {
        (str(role), str(text)): int(index)
        for role, text, index in zip(text_vocab["semantic_role"], text_vocab["raw_text"], text_vocab["vocab_id"])
    }
    active_code = events.loc[events["source"].eq("hospital_icd"), "canonical_code"].astype(str)
    new_codes = sorted(set(active_code).difference(code_to_id), key=lambda x: x.encode("utf-8"))
    text_pairs = []
    for row in events.itertuples(index=False):
        role = (
            "icd_description"
            if row.source == "hospital_icd"
            else "inpatient_prescription_proxy"
        )
        text_pairs.append((role, str(row.canonical_description)))
    new_text_pairs = sorted(
        set(text_pairs).difference(text_to_id), key=lambda x: (x[0] + "\x1f" + x[1]).encode("utf-8")
    )
    for offset, token in enumerate(new_codes, start=len(code_vocab)):
        code_to_id[token] = offset
    for offset, token in enumerate(new_text_pairs, start=len(text_vocab)):
        text_to_id[token] = offset
    code_extension_path = output_dir / "event_code_vocab_extension.csv"
    text_extension_path = output_dir / "event_text_vocab_extension.csv"
    _write_csv(
        code_extension_path,
        pd.DataFrame(
            {
                "vocab_id": np.arange(
                    len(code_vocab), len(code_vocab) + len(new_codes), dtype=np.int64
                ),
                "raw_text": new_codes,
                "semantic_role": ["icd_code"] * len(new_codes),
            }
        ),
    )
    _write_csv(
        text_extension_path,
        pd.DataFrame(
            {
                "vocab_id": np.arange(
                    len(text_vocab),
                    len(text_vocab) + len(new_text_pairs),
                    dtype=np.int64,
                ),
                "semantic_role": [role for role, _ in new_text_pairs],
                "raw_text": [text for _, text in new_text_pairs],
            }
        ),
    )
    locked = {row["artifact_id"]: row for row in registry["locked_artifacts"]}
    weight_path = Path(locked["clip_weight"]["path"])
    import open_clip

    # The inference worker sets torch threads/determinism exactly once before
    # this function. Load one CLIP model and reuse it for both extension sets.
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(weight_path)
    )
    clip_model = clip_model.cpu().float().eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    code_extension = _encode_extension_rows(
        new_codes, model=clip_model, tokenizer=tokenizer
    )
    text_extension = _encode_extension_rows(
        [text for _, text in new_text_pairs], model=clip_model, tokenizer=tokenizer
    )
    original = torch.load(embedding_path, map_location="cpu", weights_only=True)
    original_code = original["code_embedding"].cpu().contiguous()
    original_text = original["text_embedding"].cpu().contiguous()
    if tuple(original_code.shape) != (9906, 512) or tuple(original_text.shape) != (44565, 512):
        raise AssertionError("original embedding shape mismatch")
    original_code_bytes = np.ascontiguousarray(original_code.numpy().astype("<f2", copy=False)).tobytes()
    original_text_bytes = np.ascontiguousarray(original_text.numpy().astype("<f2", copy=False)).tobytes()
    code_embedding = torch.cat([original_code, torch.from_numpy(code_extension.copy())], dim=0)
    text_embedding = torch.cat([original_text, torch.from_numpy(text_extension.copy())], dim=0)
    require_immutable_embedding_prefix(
        original_code_bytes,
        np.ascontiguousarray(
            code_embedding.numpy().astype("<f2", copy=False)
        ).tobytes(),
    )
    require_immutable_embedding_prefix(
        original_text_bytes,
        np.ascontiguousarray(
            text_embedding.numpy().astype("<f2", copy=False)
        ).tobytes(),
    )
    embedding_output = output_dir / "extended_embedding_tables.pt"
    torch.save({"code_embedding": code_embedding, "text_embedding": text_embedding, "padding_id": 0, "embedding_dim": 512}, embedding_output)
    extension_manifest_path = output_dir / "VOCABULARY_EXTENSION_MANIFEST.json"
    write_json_exclusive(
        extension_manifest_path,
        {
            "schema_version": "revision0710.external_validation.vocabulary_extension.v8",
            "status": "DETERMINISTIC_APPEND_ONLY_PREFIX_IMMUTABLE",
            "original_code_vocab_rows": 9906,
            "original_text_vocab_rows": 44565,
            "code_extension_rows": len(new_codes),
            "text_extension_rows": len(new_text_pairs),
            "original_code_vocab_bytes": code_vocab_path.stat().st_size,
            "original_code_vocab_sha256": sha256_file(code_vocab_path),
            "original_text_vocab_bytes": text_vocab_path.stat().st_size,
            "original_text_vocab_sha256": sha256_file(text_vocab_path),
            "original_code_embedding_prefix_bytes": len(original_code_bytes),
            "original_code_embedding_prefix_sha256": hashlib.sha256(original_code_bytes).hexdigest(),
            "original_text_embedding_prefix_bytes": len(original_text_bytes),
            "original_text_embedding_prefix_sha256": hashlib.sha256(original_text_bytes).hexdigest(),
            "code_extension_csv_sha256": sha256_file(code_extension_path),
            "text_extension_csv_sha256": sha256_file(text_extension_path),
            "extended_embedding_table_sha256": sha256_file(embedding_output),
            "token_sort": "canonical UTF-8 bytes ascending; text role then 0x1f then text",
            "label_or_endpoint_fields_read": False,
        },
    )
    coverage = pd.DataFrame(
        [
            {
                "vocabulary_kind": "code",
                "original_vocab_rows": 9906,
                "extension_rows": len(new_codes),
                "original_vocab_prefix_bytes": code_vocab_path.stat().st_size,
                "original_vocab_prefix_sha256": sha256_file(code_vocab_path),
                "original_embedding_rows": 9906,
                "original_embedding_prefix_bytes": len(original_code_bytes),
                "original_embedding_prefix_sha256": hashlib.sha256(original_code_bytes).hexdigest(),
                "extension_manifest_sha256": sha256_file(extension_manifest_path),
            },
            {
                "vocabulary_kind": "text",
                "original_vocab_rows": 44565,
                "extension_rows": len(new_text_pairs),
                "original_vocab_prefix_bytes": text_vocab_path.stat().st_size,
                "original_vocab_prefix_sha256": sha256_file(text_vocab_path),
                "original_embedding_rows": 44565,
                "original_embedding_prefix_bytes": len(original_text_bytes),
                "original_embedding_prefix_sha256": hashlib.sha256(original_text_bytes).hexdigest(),
                "extension_manifest_sha256": sha256_file(extension_manifest_path),
            },
        ]
    )
    return code_to_id, text_to_id, code_embedding, text_embedding, coverage


def _pack_events(
    events: pd.DataFrame,
    participant_order: Sequence[str],
    code_to_id: Mapping[str, int],
    text_to_id: Mapping[tuple[str, str], int],
) -> dict[str, np.ndarray]:
    n = len(participant_order)
    packed = {
        "event_code_id": np.zeros((n, 32), dtype=np.int64),
        "event_text_id": np.zeros((n, 32), dtype=np.int64),
        "event_relative_days": np.zeros((n, 32), dtype=np.int32),
        "event_type_id": np.zeros((n, 32), dtype=np.int8),
        "event_valid_mask": np.zeros((n, 32), dtype=bool),
    }
    grouped = {str(key): value for key, value in events.groupby("participant_id", sort=False)}
    for row_index, participant_id in enumerate(participant_order):
        group = grouped.get(str(participant_id))
        if group is None:
            continue
        if len(group) > 32:
            raise AssertionError("packed group exceeds L32")
        for column_index, record in enumerate(group.itertuples(index=False)):
            is_icd = record.source == "hospital_icd"
            role = (
                "icd_description" if is_icd else "inpatient_prescription_proxy"
            )
            packed["event_code_id"][row_index, column_index] = code_to_id[str(record.canonical_code)] if is_icd else 0
            packed["event_text_id"][row_index, column_index] = text_to_id[(role, str(record.canonical_description))]
            relative = math.floor((pd.Timestamp(record.event_time) - pd.Timestamp(record.landmark_time)).total_seconds() / 86400.0)
            if relative >= 0:
                raise AssertionError("packed event is not strictly pre-index")
            packed["event_relative_days"][row_index, column_index] = relative
            packed["event_type_id"][row_index, column_index] = int(record.event_type_id)
            packed["event_valid_mask"][row_index, column_index] = True
    return packed


def _restore_fixed_model(
    root: Path,
    checkpoint_path: Path,
    code_embedding: Any,
    text_embedding: Any,
    *,
    outer_fold: int,
) -> Any:
    import torch

    model_dir = root / "Revision-0710/03_ehr_age/01_model"
    runner_dir = root / "Revision-0710/03_ehr_age/02_runner"
    for path in (model_dir, runner_dir):
        if str(path) not in os.sys.path:
            os.sys.path.insert(0, str(path))
    from ehr_age_model import EHRAGEModel
    from ehr_age_training import _model_config

    parent = json.loads(
        (root / "Revision-0710/03_ehr_age/00_contract/ehr_age_d07_003_private_retain_v1.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if outer_fold not in CHECKPOINT_AGE_ANCHORS:
        raise AssertionError("checkpoint outer fold is outside 0..4")
    expected_anchor_key, expected_anchor = CHECKPOINT_AGE_ANCHORS[outer_fold]
    observed_anchor_keys = {key for key in checkpoint if "age_anchor" in key}
    if observed_anchor_keys != {expected_anchor_key}:
        raise AssertionError(
            f"fold {outer_fold} checkpoint age-anchor key mismatch: "
            f"{sorted(observed_anchor_keys)}"
        )
    age_anchor = float(checkpoint[expected_anchor_key])
    if not np.isfinite(age_anchor):
        raise AssertionError("checkpoint age anchor is nonfinite")
    if age_anchor != expected_anchor:
        raise AssertionError(f"fold {outer_fold} checkpoint age-anchor value mismatch")
    model = EHRAGEModel(
        code_embedding,
        text_embedding,
        age_anchor=age_anchor,
        config=_model_config(parent["model"]),
    )
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = checkpoint["trainable_state_dict"]
    if set(trainable) != set(state):
        raise AssertionError("fixed checkpoint trainable tensor names mismatch")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state[name].to(dtype=parameter.dtype))
        buffers = dict(model.named_buffers())
        stored = checkpoint["buffer_state_dict"]
        if set(buffers) != set(stored):
            raise AssertionError("fixed checkpoint buffer names mismatch")
        for name, buffer in buffers.items():
            buffer.copy_(stored[name].to(dtype=buffer.dtype))
    return model.cpu().eval()


def run_five_checkpoint_inference(
    *,
    root: Path,
    artifact_dir: Path,
    stage: Path,
    cache: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> dict[str, Path]:
    import torch

    cache_dir = Path(cache["directory"])
    inference_dir = stage / "shared/04_inference_work"
    inference_dir.mkdir(parents=True, exist_ok=False)
    registry = json.loads(Path(cache["inference_registry"]).read_text(encoding="utf-8"))
    if registry.get("external_sources") != []:
        raise AssertionError("inference worker received an external source path")
    bridge = cache_dir / "inference_bridge"
    mimic = pd.read_csv(bridge / "mimic_participants.csv.gz")
    nhanes = pd.read_csv(bridge / "nhanes_participants.csv.gz")
    mimic["index_time"] = pd.to_datetime(mimic["index_time"], errors="raise")
    icd = pd.read_csv(bridge / "mimic_icd_events_preindex.csv.gz")
    rx = pd.read_csv(
        bridge / "mimic_inpatient_prescription_proxy_preindex.csv.gz"
    )
    for frame in (icd, rx):
        frame["event_time"] = pd.to_datetime(frame["event_time"], errors="raise")
        frame["landmark_time"] = pd.to_datetime(
            frame["landmark_time"], errors="raise"
        )
    all_events = pd.concat([icd, rx], ignore_index=True)
    code_to_id, text_to_id, code_embedding, text_embedding, vocab_coverage = _extend_vocab_and_embeddings(
        root, all_events, registry, inference_dir, cache_dir
    )
    _write_csv(inference_dir / "vocabulary_prefix_audit.csv", vocab_coverage)
    phenotypes = np.load(cache_dir / "phenotypes.npz")
    participant_density = pd.read_parquet(
        cache_dir / "mimic_participant_density_label_blind.parquet"
    )
    locked = {row["artifact_id"]: row for row in registry["locked_artifacts"]}
    cache_manifest_sha256 = sha256_file(cache["manifest"])
    path = inference_dir / "raw_per_checkpoint_predictions.csv.gz"
    if path.exists():
        raise FileExistsError(path)
    output_buffer: list[dict[str, Any]] = []
    wrote_header = False

    def flush_predictions() -> None:
        nonlocal wrote_header
        if not output_buffer:
            return
        pd.DataFrame(output_buffer).to_csv(
            path,
            mode="a",
            header=not wrote_header,
            index=False,
            lineterminator="\n",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        wrote_header = True
        output_buffer.clear()
    for variant in MIMIC_VARIANTS:
        variant_source = icd if variant != MIMIC_RX_SENSITIVITY else pd.concat([icd, rx], ignore_index=True)
        selected = crop_variant_timeline(variant_source, variant)
        packed = _pack_events(selected, mimic["subject_id"].astype(str).tolist(), code_to_id, text_to_id)
        expected_post = (
            participant_density.loc[
                participant_density["input_variant"].eq(variant)
            ]
            .set_index("subject_id")
            .reindex(mimic["subject_id"].astype(str))["post_crop_event_count"]
            .to_numpy(dtype=np.int64)
        )
        packed_post = packed["event_valid_mask"].sum(axis=1).astype(np.int64)
        if not np.array_equal(packed_post, expected_post):
            raise AssertionError(
                "packed event_valid_mask count differs from label-blind density post-crop count"
            )
        for fold in range(5):
            transform_path = root / locked[f"phenotype_transform_fold_{fold}"]["path"]
            transform = np.load(transform_path)
            if variant == MIMIC_BARE_OMR_SENSITIVITY:
                raw = phenotypes["mimic_bare_raw"].astype(np.float32)
                missing = phenotypes["mimic_bare_missing"].astype(bool)
            else:
                raw = phenotypes["mimic_raw"].astype(np.float32)
                missing = phenotypes["mimic_missing"].astype(bool)
            filled = np.where(missing, transform["median"], raw)
            transformed = (filled - transform["mean"]) / transform["std"]
            checkpoint_record = locked[f"checkpoint_fold_{fold}"]
            checkpoint_path = root / checkpoint_record["path"]
            model = _restore_fixed_model(
                root,
                checkpoint_path,
                code_embedding,
                text_embedding,
                outer_fold=fold,
            )
            with torch.inference_mode():
                for start in range(0, len(mimic), 512):
                    stop = min(start + 512, len(mimic))
                    result = model(
                        torch.from_numpy(transformed[start:stop]),
                        torch.from_numpy(missing[start:stop]),
                        torch.from_numpy(mimic["sex_male"].to_numpy(dtype=np.float32)[start:stop]),
                        torch.from_numpy(packed["event_code_id"][start:stop]),
                        torch.from_numpy(packed["event_text_id"][start:stop]),
                        torch.from_numpy(packed["event_relative_days"][start:stop]),
                        torch.from_numpy(packed["event_type_id"][start:stop]),
                        torch.from_numpy(packed["event_valid_mask"][start:stop]),
                    )
                    for local, subject_row in enumerate(mimic.iloc[start:stop].itertuples(index=False)):
                        for organ_index, organ in enumerate(ORGANS):
                            output_buffer.append({"cohort": "MIMIC_IV_3_1", "participant_id": str(subject_row.subject_id), "release_id": None, "landmark_policy": "mimic_second_valid_adult_discharge", "input_variant": variant, "organ_system": organ, "checkpoint_outer_fold": fold, "sex_male": int(subject_row.sex_male), "chronological_age": float(subject_row.chronological_age), "predicted_age_raw": float(result.predicted_age[local, organ_index]), "base_age_raw": float(result.base_age[local]), "risk_age_offset": float(result.risk_age_offset[local, organ_index]), "risk_score": float(result.risk_score[local, organ_index]), "checkpoint_sha256": checkpoint_record["sha256"], "external_cache_manifest_sha256": cache_manifest_sha256})
                    if len(output_buffer) >= 100_000:
                        flush_predictions()
            del model
    nhanes_packed = {name: np.zeros((len(nhanes), 32), dtype=dtype) for name, dtype in (("event_code_id", np.int64), ("event_text_id", np.int64), ("event_relative_days", np.int32), ("event_type_id", np.int8), ("event_valid_mask", bool))}
    for fold in range(5):
        transform = np.load(root / locked[f"phenotype_transform_fold_{fold}"]["path"])
        raw = phenotypes["nhanes_raw"].astype(np.float32)
        missing = phenotypes["nhanes_missing"].astype(bool)
        transformed = (np.where(missing, transform["median"], raw) - transform["mean"]) / transform["std"]
        checkpoint_record = locked[f"checkpoint_fold_{fold}"]
        model = _restore_fixed_model(
            root,
            root / checkpoint_record["path"],
            code_embedding,
            text_embedding,
            outer_fold=fold,
        )
        with torch.inference_mode():
            for start in range(0, len(nhanes), 512):
                stop = min(start + 512, len(nhanes))
                result = model(torch.from_numpy(transformed[start:stop]), torch.from_numpy(missing[start:stop]), torch.from_numpy(nhanes["sex_male"].to_numpy(dtype=np.float32)[start:stop]), torch.from_numpy(nhanes_packed["event_code_id"][start:stop]), torch.from_numpy(nhanes_packed["event_text_id"][start:stop]), torch.from_numpy(nhanes_packed["event_relative_days"][start:stop]), torch.from_numpy(nhanes_packed["event_type_id"][start:stop]), torch.from_numpy(nhanes_packed["event_valid_mask"][start:stop]))
                for local, subject_row in enumerate(nhanes.iloc[start:stop].itertuples(index=False)):
                    for organ_index, organ in enumerate(ORGANS):
                        output_buffer.append({"cohort": "NHANES", "participant_id": str(subject_row.participant_id), "release_id": subject_row.release_id, "landmark_policy": "nhanes_mec_exam", "input_variant": NHANES_PRIMARY_VARIANT, "organ_system": organ, "checkpoint_outer_fold": fold, "sex_male": int(subject_row.sex_male), "chronological_age": float(subject_row.chronological_age), "predicted_age_raw": float(result.predicted_age[local, organ_index]), "base_age_raw": float(result.base_age[local]), "risk_age_offset": float(result.risk_age_offset[local, organ_index]), "risk_score": float(result.risk_score[local, organ_index]), "checkpoint_sha256": checkpoint_record["sha256"], "external_cache_manifest_sha256": cache_manifest_sha256})
                if len(output_buffer) >= 100_000:
                    flush_predictions()
        del model
    flush_predictions()
    if not path.exists() or not wrote_header:
        raise AssertionError("inference produced no prediction rows")
    return {"directory": inference_dir, "raw_predictions": path, "locked_registry": Path(cache["inference_registry"]), "vocabulary_audit": inference_dir / "vocabulary_prefix_audit.csv", "vocabulary_extension_manifest": inference_dir / "VOCABULARY_EXTENSION_MANIFEST.json"}


def seal_prediction_tables(
    *, stage: Path, predictions: Mapping[str, Path], contract: Mapping[str, Any]
) -> dict[str, Path]:
    work = Path(predictions["directory"])
    raw = pd.read_csv(predictions["raw_predictions"])
    registry = json.loads(Path(predictions["locked_registry"]).read_text(encoding="utf-8"))
    if registry.get("external_sources") != []:
        raise AssertionError("prediction seal received an external source path")
    locked = {row["artifact_id"]: row for row in registry["locked_artifacts"]}
    correction: dict[tuple[int, str], tuple[float, float]] = {}
    for fold in range(5):
        record = locked[f"correction_fold_{fold}"]
        path = Path(record["path"])
        if record["path_kind"] == "repo_relative":
            path = Path(__file__).resolve().parents[4] / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["parameters"]:
            correction[(fold, row["organ_system"])] = (float(row["intercept"]), float(row["slope"]))
    parameters = [correction[(int(fold), str(organ))] for fold, organ in zip(raw["checkpoint_outer_fold"], raw["organ_system"])]
    raw["raw_age_gap"] = raw["predicted_age_raw"] - raw["chronological_age"]
    raw["correction_intercept"] = [value[0] for value in parameters]
    raw["correction_slope"] = [value[1] for value in parameters]
    raw["corrected_age_gap"] = raw["raw_age_gap"] - (
        raw["correction_intercept"] + raw["correction_slope"] * raw["chronological_age"]
    )
    key = ["cohort", "release_id", "participant_id", "input_variant", "organ_system"]
    if raw.duplicated(key + ["checkpoint_outer_fold"]).any():
        raise AssertionError("per-checkpoint prediction key duplicate")
    fold_sets = raw.groupby(key, dropna=False)["checkpoint_outer_fold"].agg(lambda x: sorted(x.tolist()))
    if not fold_sets.map(lambda x: x == [0, 1, 2, 3, 4]).all():
        raise AssertionError("prediction rows lack exact five checkpoint set")
    per_path = work / "sealed_per_checkpoint_predictions.parquet"
    per_columns = [
        "cohort", "release_id", "participant_id", "landmark_policy", "input_variant",
        "organ_system", "checkpoint_outer_fold", "sex_male", "chronological_age",
        "predicted_age_raw", "base_age_raw", "risk_age_offset", "risk_score",
        "raw_age_gap", "correction_intercept", "correction_slope", "corrected_age_gap",
        "checkpoint_sha256", "external_cache_manifest_sha256",
    ]
    _write_parquet(per_path, raw[per_columns].sort_values(key + ["checkpoint_outer_fold"], kind="mergesort"))
    grouped = raw.groupby(key, dropna=False, sort=False)
    ensemble = grouped.agg(
        sex_male=("sex_male", "first"),
        chronological_age=("chronological_age", "first"),
        predicted_age_raw_mean5=("predicted_age_raw", "mean"),
        predicted_age_raw_sd5=("predicted_age_raw", "std"),
        raw_age_gap_mean5=("raw_age_gap", "mean"),
        corrected_age_gap_mean5=("corrected_age_gap", "mean"),
        risk_score_mean5=("risk_score", "mean"),
    ).reset_index()
    ensemble["complete_five_checkpoint_set"] = True
    ensemble_path = work / "sealed_ensemble_predictions.parquet"
    _write_parquet(ensemble_path, ensemble.sort_values(key, kind="mergesort"))
    receipt_path = work / "PREDICTION_SEAL_RECEIPT.json"
    write_json_exclusive(
        receipt_path,
        {
            "schema_version": "revision0710.external_validation.prediction_seal.v8",
            "status": "PREDICTIONS_SEALED_BEFORE_OUTCOME_READ",
            "per_checkpoint_path": str(per_path.resolve()),
            "per_checkpoint_sha256": sha256_file(per_path),
            "ensemble_path": str(ensemble_path.resolve()),
            "ensemble_sha256": sha256_file(ensemble_path),
            "checkpoint_folds": [0, 1, 2, 3, 4],
            "correction_formula": "raw_gap - (intercept + slope * chronological_age)",
            "external_outcome_rows_read_before_seal": 0,
        },
    )
    return {
        "per_checkpoint": per_path,
        "ensemble": ensemble_path,
        "receipt": receipt_path,
        "vocabulary_audit": Path(predictions["vocabulary_audit"]),
        "vocabulary_extension_manifest": Path(
            predictions["vocabulary_extension_manifest"]
        ),
    }


def _icd10_organs(code: str, mapping: pd.DataFrame) -> set[str]:
    try:
        canonical = _canonical_icd(code, 10)
    except ValueError:
        return set()
    category = canonical[:3]
    result = set()
    for row in mapping.itertuples(index=False):
        if str(row.category_start) <= category <= str(row.category_end):
            result.add(str(row.organ_system))
    return result


def _build_nhanes_questionnaire_labels(
    artifact_dir: Path,
    sources: Mapping[str, Path],
    participants: pd.DataFrame,
) -> pd.DataFrame:
    mapping = pd.read_csv(artifact_dir / "NHANES_QUESTIONNAIRE_LABEL_MAP_V8.csv", keep_default_na=False)
    output = []
    loaded: dict[str, pd.DataFrame] = {}
    for row in mapping.itertuples(index=False):
        release = str(row.release_id)
        subset = participants.loc[participants["release_id"].eq(release), ["participant_id", "SEQN"]].copy()
        if row.estimability == "not_estimable":
            for participant_id in subset["participant_id"]:
                output.append({"participant_id": str(participant_id), "release_id": release, "organ_system": row.organ_system, "label": np.nan, "label_status": "not_estimable_by_design"})
            continue
        merged = subset.set_index("SEQN")
        variables = str(row.source_variables).split("|")
        values = []
        for source_id in str(row.source_file_ids).split("|"):
            if source_id not in loaded:
                loaded[source_id] = pd.read_sas(sources[source_id], format="xport", encoding="latin1").set_index("SEQN")
            available = [variable for variable in variables if variable in loaded[source_id].columns]
            if available:
                values.append(loaded[source_id][available].reindex(merged.index))
        matrix = pd.concat(values, axis=1).loc[:, ~pd.concat(values, axis=1).columns.duplicated()]
        labels = _questionnaire_composite_labels(matrix)
        for participant_id, label in zip(merged["participant_id"], labels):
            output.append({"participant_id": str(participant_id), "release_id": release, "organ_system": row.organ_system, "label": label, "label_status": "estimable" if np.isfinite(label) else "missing_questionnaire_response"})
    return pd.DataFrame(output)


def _questionnaire_composite_labels(matrix: pd.DataFrame) -> np.ndarray:
    positive = matrix.eq(1).any(axis=1)
    # A composite is negative only when every frozen component is an
    # explicit source value 2. Any missing, refused, don't-know, or other
    # code makes the composite missing unless a component is positive.
    negative = matrix.notna().all(axis=1) & matrix.eq(2).all(axis=1)
    return np.where(positive, 1.0, np.where(negative, 0.0, np.nan))


def _completed_diagnosis_endpoint_history(
    diagnosis: pd.DataFrame,
    valid_admissions: pd.DataFrame,
    cohort: pd.DataFrame,
    icd10_map: pd.DataFrame,
) -> tuple[set[int], set[tuple[int, str]], pd.Series]:
    value = diagnosis.merge(
        valid_admissions[["subject_id", "hadm_id", "dischtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )
    index = cohort.set_index("subject_id")["index_time"]
    value["index_time"] = value["subject_id"].map(index)
    value = value.loc[
        value["index_time"].notna() & value["dischtime"].notna()
    ].copy()
    excluded_icd9 = set(
        map(
            int,
            value.loc[
                value["dischtime"].le(value["index_time"])
                & value["icd_version"].eq(9),
                "subject_id",
            ].unique(),
        )
    )
    value = value.loc[value["icd_version"].eq(10)].copy()
    value["organs"] = value["icd_code"].map(
        lambda code: _icd10_organs(str(code), icd10_map)
    )
    value = value.loc[value["organs"].map(bool)]
    expanded = []
    for row in value.itertuples(index=False):
        for organ in sorted(row.organs):
            expanded.append(
                {
                    "subject_id": int(row.subject_id),
                    "hadm_id": int(row.hadm_id),
                    "diagnosis_available_time": row.dischtime,
                    "seq_num": int(row.seq_num),
                    "organ_system": organ,
                    "pre_or_index": row.dischtime <= row.index_time,
                }
            )
    organ_diagnosis = pd.DataFrame(
        expanded,
        columns=[
            "subject_id",
            "hadm_id",
            "diagnosis_available_time",
            "seq_num",
            "organ_system",
            "pre_or_index",
        ],
    )
    if organ_diagnosis.empty:
        return excluded_icd9, set(), pd.Series(dtype="datetime64[ns]")
    baseline = set(
        zip(
            organ_diagnosis.loc[
                organ_diagnosis["pre_or_index"], "subject_id"
            ],
            organ_diagnosis.loc[
                organ_diagnosis["pre_or_index"], "organ_system"
            ],
        )
    )
    post = organ_diagnosis.loc[~organ_diagnosis["pre_or_index"]].sort_values(
        [
            "subject_id",
            "organ_system",
            "diagnosis_available_time",
            "seq_num",
            "hadm_id",
        ],
        kind="mergesort",
    )
    first_post = (
        post.drop_duplicates(["subject_id", "organ_system"], keep="first")
        .set_index(["subject_id", "organ_system"])["diagnosis_available_time"]
    )
    return excluded_icd9, baseline, first_post


def _date_only_dod_rule(
    index_time: pd.Timestamp, death_date: pd.Timestamp | None
) -> dict[str, Any]:
    index_date = pd.Timestamp(index_time).normalize()
    if death_date is None or pd.isna(death_date):
        return {
            "status": "no_recorded_dod",
            "normalized_death": pd.NaT,
            "death_days": None,
            "same_day_death": False,
        }
    normalized_death = pd.Timestamp(death_date).normalize()
    if normalized_death < index_date:
        return {
            "status": "excluded_inconsistent_dod_before_index_date",
            "normalized_death": normalized_death,
            "death_days": None,
            "same_day_death": False,
        }
    same_day = normalized_death == index_date
    return {
        "status": "endpoint_eligible",
        "normalized_death": normalized_death,
        "death_days": float((normalized_death - index_date).days),
        "same_day_death": bool(same_day),
    }


def build_outcomes_after_prediction_seal(
    *,
    artifact_dir: Path,
    stage: Path,
    sources: Mapping[str, Path],
    prediction_seal: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> dict[str, Path]:
    receipt = json.loads(Path(prediction_seal["receipt"]).read_text(encoding="utf-8"))
    if receipt.get("status") != "PREDICTIONS_SEALED_BEFORE_OUTCOME_READ":
        raise AssertionError("prediction seal is not committed")
    if sha256_file(Path(prediction_seal["per_checkpoint"])) != receipt["per_checkpoint_sha256"] or sha256_file(Path(prediction_seal["ensemble"])) != receipt["ensemble_sha256"]:
        raise AssertionError("prediction seal hash mismatch")
    cache_dir = stage / "shared/02_label_blind_cache"
    cohort = pd.read_parquet(cache_dir / "mimic_participants.parquet")
    nhanes = pd.read_parquet(cache_dir / "nhanes_participants.parquet")
    admissions = pd.read_csv(sources["mimic_admissions"], compression="gzip", usecols=["subject_id", "hadm_id", "admittime", "dischtime"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    last_discharge = _last_valid_discharge(admissions)
    valid_admissions = admissions.loc[
        admissions["admittime"].notna()
        & admissions["dischtime"].notna()
        & admissions["dischtime"].gt(admissions["admittime"])
    ].copy()
    patients = pd.read_csv(
        sources["mimic_patients"],
        compression="gzip",
        usecols=["subject_id", "dod"],
    )
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    dod = patients.set_index("subject_id")["dod"]
    cohort_index_time = cohort.set_index("subject_id")["index_time"]
    followup_cap = _subject_followup_cap(dod, last_discharge, cohort_index_time)
    observation_end = _update_observation_end(cohort, sources)
    observation_end = pd.concat(
        [observation_end.rename("event_time"), last_discharge.rename("event_time")]
    ).groupby(level=0).max()
    observation_end = pd.concat(
        [observation_end.rename("observation_end"), followup_cap.rename("followup_cap")],
        axis=1,
    ).min(axis=1)
    diagnosis = pd.read_csv(sources["mimic_diagnoses"], compression="gzip", usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    icd10_map = pd.read_csv(artifact_dir / "MIMIC_ICD10_ORGAN_MAP_V8.csv", keep_default_na=False)
    excluded_icd9, baseline, first_post = _completed_diagnosis_endpoint_history(
        diagnosis,
        valid_admissions,
        cohort,
        icd10_map,
    )
    incident_rows = []
    recurrent_rows = []
    eligibility_rows = []
    for subject in cohort.itertuples(index=False):
        subject_id = int(subject.subject_id)
        if subject_id in excluded_icd9:
            eligibility_rows.append(
                {
                    "subject_id": str(subject_id),
                    "eligibility_status": "excluded_preindex_icd9",
                    "has_any_preindex_icd9": True,
                    "same_day_death": False,
                    "inconsistent_dod_before_index_date": False,
                }
            )
            continue
        subject_observation_end = observation_end.get(subject_id, subject.index_time)
        subject_followup_cap = followup_cap.get(subject_id, pd.NaT)
        if pd.isna(subject_followup_cap):
            raise AssertionError("MIMIC subject lacks the frozen follow-up cap")
        death_date = dod.get(subject_id, pd.NaT)
        dod_rule = _date_only_dod_rule(pd.Timestamp(subject.index_time), death_date)
        same_day_death = bool(dod_rule["same_day_death"])
        if dod_rule["status"] == "excluded_inconsistent_dod_before_index_date":
            eligibility_rows.append(
                {
                    "subject_id": str(subject_id),
                    "eligibility_status": "excluded_inconsistent_dod_before_index_date",
                    "has_any_preindex_icd9": False,
                    "same_day_death": False,
                    "inconsistent_dod_before_index_date": True,
                }
            )
            continue
        if same_day_death:
            # DOD has date-only resolution. Survived-discharge eligibility
            # establishes that death followed discharge on this date, so the
            # prespecified conservative competing-event time is day 0.
            subject_followup_cap = pd.Timestamp(subject.index_time)
            subject_observation_end = pd.Timestamp(subject.index_time)
        observation_days = (pd.Timestamp(subject_observation_end) - pd.Timestamp(subject.index_time)).total_seconds() / 86400.0
        death_cap_days = (pd.Timestamp(subject_followup_cap) - pd.Timestamp(subject.index_time)).total_seconds() / 86400.0
        death_days = dod_rule["death_days"]
        eligibility_rows.append(
            {
                "subject_id": str(subject_id),
                "eligibility_status": "endpoint_eligible",
                "has_any_preindex_icd9": False,
                "same_day_death": same_day_death,
                "inconsistent_dod_before_index_date": False,
            }
        )
        for organ in ORGANS:
            target_time = first_post.get((subject_id, organ), pd.NaT)
            target_days = None if pd.isna(target_time) else (pd.Timestamp(target_time) - pd.Timestamp(subject.index_time)).total_seconds() / 86400.0
            is_recurrent = (subject_id, organ) in baseline
            for variant in MIMIC_VARIANTS:
                for horizon in (6, 12):
                    terminal = adjudicate_endpoint(target_days=target_days, death_days=death_days, observation_end_days=observation_days, death_cap_days=death_cap_days, horizon_months=horizon)
                    row = {"subject_id": str(subject_id), "input_variant": variant, "age_population": "mimic_risk_adult_nontopcoded_integer_year", "organ_system": organ, "horizon_months": horizon, "index_time": subject.index_time, "chronological_age": int(subject.chronological_age), "sex_male": int(subject.sex_male), "event_time_days": target_days, "death_time_days": death_days, "subject_observation_end_days": observation_days, "death_ascertainment_cap_days": death_cap_days, **terminal, "has_any_preindex_icd9": False, "endpoint_code_system": "ICD10", "hash_fold": int(subject.hash_fold), "estimability_status": "estimable"}
                    (recurrent_rows if is_recurrent else incident_rows).append(row)
    outcome_dir = stage / "shared/05_outcomes_after_prediction_seal"
    outcome_dir.mkdir(parents=True, exist_ok=False)
    incident_path = outcome_dir / "incident_endpoints.parquet"
    recurrent_path = outcome_dir / "recurrent_endpoints.parquet"
    _write_parquet(incident_path, pd.DataFrame(incident_rows))
    _write_parquet(recurrent_path, pd.DataFrame(recurrent_rows))
    questionnaire = _build_nhanes_questionnaire_labels(artifact_dir, sources, nhanes)
    questionnaire_path = outcome_dir / "nhanes_questionnaire_labels.parquet"
    _write_parquet(questionnaire_path, questionnaire)
    eligibility_path = outcome_dir / "mimic_endpoint_subject_eligibility.parquet"
    eligibility = pd.DataFrame(eligibility_rows).sort_values(
        "subject_id", kind="mergesort"
    )
    _write_parquet(eligibility_path, eligibility)
    outcome_receipt = outcome_dir / "OUTCOME_READ_RECEIPT.json"
    write_json_exclusive(outcome_receipt, {"schema_version": "revision0710.external_validation.outcome_read.v8", "status": "OUTCOMES_READ_ONLY_AFTER_PREDICTION_SEAL", "prediction_seal_sha256": sha256_file(Path(prediction_seal["receipt"])), "incident_sha256": sha256_file(incident_path), "recurrent_sha256": sha256_file(recurrent_path), "questionnaire_sha256": sha256_file(questionnaire_path), "eligibility_sha256": sha256_file(eligibility_path), "model_input_subjects": int(len(cohort)), "endpoint_eligible_subjects": int(eligibility["eligibility_status"].eq("endpoint_eligible").sum()), "excluded_preindex_icd9_subjects": int(eligibility["eligibility_status"].eq("excluded_preindex_icd9").sum()), "excluded_inconsistent_dod_subjects": int(eligibility["eligibility_status"].eq("excluded_inconsistent_dod_before_index_date").sum()), "same_day_death_subjects": int(eligibility["same_day_death"].sum()), "death_date_resolution_rule": "DOD before index calendar date is inconsistent/excluded; otherwise date-only DOD is resolved as index_time plus the integer calendar-day offset, including same-date day-0 competing death", "followup_cap_formula": "min(index_time + calendar_day_offset(patients.dod, index_date), last_valid_discharge + 365 days)", "postseal_only_fields": ["patients.dod", "postindex diagnoses", "subject observation end", "last valid discharge follow-up cap", "NHANES questionnaire labels"]})
    return {"incident": incident_path, "recurrent": recurrent_path, "questionnaire": questionnaire_path, "eligibility": eligibility_path, "receipt": outcome_receipt}


AGE_METRICS = (
    "mae",
    "rmse",
    "pearson",
    "spearman",
    "r2",
    "calibration_intercept",
    "calibration_slope",
)
ESTIMABILITY_STATUSES = {
    "estimable",
    "zero_target_events",
    "zero_competing_deaths",
    "single_outcome_class",
    "singular_design",
    "insufficient_complete_cases",
    "not_estimable_by_design",
}
MIMIC_FLOW_STEPS = (
    "raw_subjects",
    "has_demographics",
    "sex_binary_m_or_f",
    "anchor_age_not_91_topcoded",
    "adult_age_at_time_valid_discharge",
    "alive_at_time_valid_adult_discharge",
    "has_two_valid_adult_surviving_discharge_admissions",
    "second_valid_adult_surviving_discharge_selected",
    "has_subject_specific_observation_end",
    "has_valid_6_or_12_month_endpoint_window",
    "no_preindex_icd9_for_primary_endpoint",
    "model_input_cache_eligible",
)


def _age_metric(metric: str, observed: np.ndarray, predicted: np.ndarray) -> float:
    from scipy.stats import spearmanr

    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < 3:
        raise ValueError("insufficient complete age rows")
    residual = p - y
    if metric == "mae":
        return float(np.mean(np.abs(residual)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(residual))))
    if metric == "pearson":
        if np.std(y) == 0 or np.std(p) == 0:
            raise ValueError("Pearson is singular")
        return float(np.corrcoef(y, p)[0, 1])
    if metric == "spearman":
        value = float(spearmanr(y, p).statistic)
        if not np.isfinite(value):
            raise ValueError("Spearman is singular")
        return value
    if metric == "r2":
        denominator = float(np.sum(np.square(y - np.mean(y))))
        if denominator == 0:
            raise ValueError("R2 is singular")
        return float(1.0 - np.sum(np.square(y - p)) / denominator)
    design = np.column_stack([np.ones(len(p)), p])
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("calibration regression is singular")
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    if metric == "calibration_intercept":
        return float(coefficient[0])
    if metric == "calibration_slope":
        return float(coefficient[1])
    raise ValueError(f"unknown age metric: {metric}")


def _point_age_metrics(
    frame: pd.DataFrame,
    *,
    predicted_column: str,
    observed_column: str,
    macro_by_organ: bool = False,
) -> dict[str, tuple[float | None, str]]:
    identifiers = frame["participant_id"].astype(str).unique()
    if len(identifiers) < 4:
        return {
            metric: (None, "insufficient_complete_cases")
            for metric in AGE_METRICS
        }

    def evaluate(sample: pd.DataFrame, metric: str) -> float:
        if not macro_by_organ:
            return _age_metric(
                metric,
                sample[observed_column].to_numpy(),
                sample[predicted_column].to_numpy(),
            )
        values = []
        for _, organ_rows in sample.groupby("organ_system", sort=False):
            values.append(
                _age_metric(
                    metric,
                    organ_rows[observed_column].to_numpy(),
                    organ_rows[predicted_column].to_numpy(),
                )
            )
        if len(values) != len(ORGANS):
            raise ValueError("macro organ set incomplete")
        return float(np.mean(values))

    result: dict[str, tuple[float | None, str]] = {}
    for metric in AGE_METRICS:
        try:
            result[metric] = (evaluate(frame, metric), "estimable")
        except ValueError:
            result[metric] = (None, "singular_design")
    return result


def _km_curve(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    duration = np.asarray(time, dtype=np.float64)
    observed = np.asarray(event, dtype=np.int8)
    survival = 1.0
    rows = []
    order = np.argsort(duration, kind="mergesort")
    sorted_time = duration[order]
    sorted_event = observed[order]
    points, starts, counts = np.unique(
        sorted_time, return_index=True, return_counts=True
    )
    for point, start, count in zip(points, starts, counts):
        at_risk = int(len(duration) - start)
        failures = int(sorted_event[start : start + count].sum())
        censored = int(count - failures)
        if failures:
            survival *= 1.0 - failures / at_risk
        rows.append(
            {
                "time_days": float(point),
                "n_at_risk": at_risk,
                "events": failures,
                "censored": censored,
                "estimate": float(survival),
            }
        )
    return pd.DataFrame(rows)


def _cox_fit(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    *,
    robust_groups: np.ndarray | None = None,
    status_column: str = "event_indicator",
) -> Any:
    columns = ["observed_time_days", status_column, *covariates]
    complete = frame.loc[np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all(axis=1)]
    if len(complete) < max(20, len(covariates) + 3):
        raise ValueError("insufficient complete Cox rows")
    if int(complete[status_column].sum()) == 0:
        raise ValueError("zero target events")
    exog = complete[list(covariates)].to_numpy(dtype=np.float64)
    if np.linalg.matrix_rank(exog) != len(covariates):
        raise ValueError("singular Cox design")
    groups = None
    if robust_groups is not None:
        groups = np.asarray(robust_groups)[complete.index.to_numpy()]
    try:
        from statsmodels.duration.hazard_regression import PHReg

        model = PHReg(
            complete["observed_time_days"].to_numpy(dtype=np.float64),
            exog,
            status=complete[status_column].to_numpy(dtype=np.int8),
            ties="efron",
        )
        result = model.fit(groups=groups, disp=0)
    except ModuleNotFoundError:
        from types import SimpleNamespace

        from lifelines import CoxPHFitter

        fit = complete[["observed_time_days", status_column, *covariates]].copy()
        cluster_col = None
        if groups is not None:
            cluster_col = "_cluster_id"
            fit[cluster_col] = groups
        model = CoxPHFitter()
        model.fit(
            fit,
            duration_col="observed_time_days",
            event_col=status_column,
            cluster_col=cluster_col,
            robust=cluster_col is not None,
            show_progress=False,
        )
        covariance = (
            model.variance_matrix_
            .reindex(index=covariates, columns=covariates)
            .to_numpy(dtype=np.float64)
        )
        event_residuals = model.compute_residuals(
            fit, kind="schoenfeld"
        ).reindex(columns=covariates)
        residuals = np.full((len(complete), len(covariates)), np.nan)
        positions = pd.Series(
            np.arange(len(complete), dtype=np.int64), index=complete.index
        )
        residual_positions = positions.loc[event_residuals.index].to_numpy(
            dtype=np.int64
        )
        residuals[residual_positions] = event_residuals.to_numpy(
            dtype=np.float64
        )
        result = SimpleNamespace(
            params=model.params_.reindex(covariates).to_numpy(dtype=np.float64),
            bse=model.standard_errors_.reindex(covariates).to_numpy(
                dtype=np.float64
            ),
            cov_params=lambda: covariance,
            schoenfeld_residuals=residuals,
        )
    if np.any(~np.isfinite(result.params)) or np.any(~np.isfinite(result.bse)):
        raise ValueError("nonfinite Cox fit")
    return complete, result


def _phreg_baseline_hazard_jumps(
    complete: pd.DataFrame,
    result: Any,
    covariates: Sequence[str],
    *,
    status_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct post-jump Breslow baseline hazards from full risk sets."""
    duration = complete["observed_time_days"].to_numpy(dtype=np.float64)
    status = complete[status_column].to_numpy(dtype=np.int8)
    design = complete[list(covariates)].to_numpy(dtype=np.float64)
    linear = design @ np.asarray(result.params, dtype=np.float64)
    relative_risk = np.exp(linear)
    if (
        len(duration) == 0
        or np.any(~np.isfinite(duration))
        or np.any(~np.isfinite(relative_risk))
        or np.any(relative_risk <= 0)
        or np.any((status != 0) & (status != 1))
    ):
        raise ValueError("invalid rows for PHReg baseline-hazard reconstruction")

    order = np.argsort(duration, kind="mergesort")
    sorted_time = duration[order]
    sorted_status = status[order]
    sorted_risk = relative_risk[order]
    points, starts, counts = np.unique(
        sorted_time, return_index=True, return_counts=True
    )
    risk_denominator = np.cumsum(sorted_risk[::-1], dtype=np.float64)[::-1][starts]
    failures = np.asarray(
        [sorted_status[start : start + count].sum() for start, count in zip(starts, counts)],
        dtype=np.float64,
    )
    event_mask = failures > 0
    event_times = points[event_mask].astype(np.float64, copy=False)
    jumps = failures[event_mask] / risk_denominator[event_mask]
    if (
        len(event_times) == 0
        or np.any(~np.isfinite(jumps))
        or np.any(jumps <= 0)
        or np.any(np.diff(event_times) <= 0)
    ):
        raise ValueError("invalid PHReg baseline-hazard jumps")
    return event_times, jumps


def _integrate_cause_specific_hazard_jumps(
    target_times: np.ndarray,
    target_jumps: np.ndarray,
    target_linear: np.ndarray,
    death_times: np.ndarray,
    death_jumps: np.ndarray,
    death_linear: np.ndarray,
    horizon_days: float,
) -> np.ndarray:
    """Integrate target CIF over explicit target and competing-death jumps."""
    target_times = np.asarray(target_times, dtype=np.float64)
    target_jumps = np.asarray(target_jumps, dtype=np.float64)
    death_times = np.asarray(death_times, dtype=np.float64)
    death_jumps = np.asarray(death_jumps, dtype=np.float64)
    target_linear = np.asarray(target_linear, dtype=np.float64)
    death_linear = np.asarray(death_linear, dtype=np.float64)
    if (
        target_times.ndim != 1
        or target_jumps.shape != target_times.shape
        or death_times.ndim != 1
        or death_jumps.shape != death_times.shape
        or target_linear.ndim != 1
        or death_linear.shape != target_linear.shape
        or not np.isfinite(horizon_days)
        or horizon_days < 0
        or np.any(~np.isfinite(target_times))
        or np.any(~np.isfinite(death_times))
        or np.any(~np.isfinite(target_jumps))
        or np.any(~np.isfinite(death_jumps))
        or np.any(target_jumps <= 0)
        or np.any(death_jumps <= 0)
        or np.any(np.diff(target_times) <= 0)
        or np.any(np.diff(death_times) <= 0)
        or np.any(~np.isfinite(target_linear))
        or np.any(~np.isfinite(death_linear))
    ):
        raise ValueError("invalid cause-specific hazard-jump inputs")
    target_risk = np.exp(target_linear)
    death_risk = np.exp(death_linear)
    if np.any(~np.isfinite(target_risk)) or np.any(~np.isfinite(death_risk)):
        raise ValueError("nonfinite cause-specific relative risk")

    event_times = np.unique(
        np.concatenate(
            [
                target_times[target_times <= horizon_days],
                death_times[death_times <= horizon_days],
            ]
        )
    )
    target_lookup = dict(zip(target_times.tolist(), target_jumps.tolist()))
    death_lookup = dict(zip(death_times.tolist(), death_jumps.tolist()))
    cif = np.zeros(len(target_linear), dtype=np.float64)
    survival = np.ones(len(target_linear), dtype=np.float64)
    for point in event_times:
        increment_target = target_lookup.get(float(point), 0.0) * target_risk
        increment_death = death_lookup.get(float(point), 0.0) * death_risk
        total = increment_target + increment_death
        event_probability = -np.expm1(-total)
        target_share = np.divide(
            increment_target,
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )
        cif += survival * event_probability * target_share
        survival *= np.exp(-total)
    return np.clip(cif, 0.0, 1.0)


def _cox_predictions(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    covariates: Sequence[str],
    horizon_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    complete, result = _cox_fit(training.reset_index(drop=True), covariates)
    test_x = testing[list(covariates)].to_numpy(dtype=np.float64)
    linear = test_x @ np.asarray(result.params, dtype=np.float64)
    times, jumps = _phreg_baseline_hazard_jumps(
        complete,
        result,
        covariates,
        status_column="event_indicator",
    )
    baseline = float(jumps[times <= horizon_days].sum())
    probability = 1.0 - np.exp(-baseline * np.exp(linear))
    return linear, np.clip(probability, 0.0, 1.0)


def _cause_specific_cif_predictions(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    covariates: Sequence[str],
    horizon_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict target CIF from target/death cause-specific Cox hazards."""
    complete, target_fit = _cox_fit(
        training.reset_index(drop=True), covariates, status_column="event_indicator"
    )
    test_x = testing[list(covariates)].to_numpy(dtype=np.float64)
    target_beta = np.asarray(target_fit.params, dtype=np.float64)
    target_linear = test_x @ target_beta

    target_times, target_jumps = _phreg_baseline_hazard_jumps(
        complete,
        target_fit,
        covariates,
        status_column="event_indicator",
    )
    if int(complete["competing_death_indicator"].sum()) > 0:
        death_complete, death_fit = _cox_fit(
            complete.reset_index(drop=True),
            covariates,
            status_column="competing_death_indicator",
        )
        death_beta = np.asarray(death_fit.params, dtype=np.float64)
        death_linear = test_x @ death_beta
        death_times, death_jumps = _phreg_baseline_hazard_jumps(
            death_complete,
            death_fit,
            covariates,
            status_column="competing_death_indicator",
        )
    else:
        death_linear = np.zeros(len(testing), dtype=np.float64)
        death_times = np.empty(0, dtype=np.float64)
        death_jumps = np.empty(0, dtype=np.float64)
    cif = _integrate_cause_specific_hazard_jumps(
        target_times,
        target_jumps,
        target_linear,
        death_times,
        death_jumps,
        death_linear,
        horizon_days,
    )
    return target_linear, np.clip(cif, 0.0, 1.0)


def _weighted_linear_calibration(
    outcome: np.ndarray, probability: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    y = np.asarray(outcome, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    design = np.column_stack([np.ones(valid.sum()), p[valid]])
    if valid.sum() < 4 or np.linalg.matrix_rank(design) != 2:
        raise ValueError("weighted calibration is singular")
    root_weight = np.sqrt(w[valid])
    coefficient = np.linalg.lstsq(
        design * root_weight[:, None], y[valid] * root_weight, rcond=None
    )[0]
    return float(coefficient[0]), float(coefficient[1])


def _survey_logistic_taylor(
    y: np.ndarray,
    design: np.ndarray,
    weight: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
    *,
    full_design_strata: np.ndarray,
    full_design_psu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    from scipy.special import expit

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if len(y) < x.shape[1] + 5 or len(np.unique(y)) != 2 or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("survey logistic design is not estimable")
    beta = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(100):
        probability = expit(np.clip(x @ beta, -30, 30))
        working_weight = w * probability * (1.0 - probability)
        hessian = x.T @ (working_weight[:, None] * x)
        score = x.T @ (w * (y - probability))
        if np.linalg.matrix_rank(hessian) != x.shape[1]:
            raise ValueError("survey logistic Hessian is singular")
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise ValueError("survey logistic did not converge")
    probability = expit(np.clip(x @ beta, -30, 30))
    bread_inverse = np.linalg.inv(x.T @ ((w * probability * (1 - probability))[:, None] * x))
    individual_score = x * (w * (y - probability))[:, None]
    cluster = pd.DataFrame(
        individual_score,
        columns=[f"s{index}" for index in range(x.shape[1])],
    )
    cluster["strata"] = strata
    cluster["psu"] = psu
    aggregate = cluster.groupby(["strata", "psu"], sort=False).sum().reset_index()
    score_columns = [f"s{index}" for index in range(x.shape[1])]
    full_clusters = pd.DataFrame(
        {"strata": full_design_strata, "psu": full_design_psu}
    ).dropna().drop_duplicates()
    aggregate = full_clusters.merge(
        aggregate,
        on=["strata", "psu"],
        how="left",
        validate="one_to_one",
    )
    aggregate[score_columns] = aggregate[score_columns].fillna(0.0)
    meat = _centered_psu_meat(
        aggregate[score_columns].to_numpy(dtype=np.float64),
        aggregate["strata"].to_numpy(),
    )
    covariance = bread_inverse @ meat @ bread_inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    design_df = int(len(full_clusters) - full_clusters["strata"].nunique())
    if design_df <= 0:
        raise ValueError("NHANES survey design has nonpositive design degrees of freedom")
    return beta, standard_error, design_df


def _centered_psu_meat(scores: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Taylor PSU meat with lonely PSUs centered on the grand PSU mean."""
    score_matrix = np.asarray(scores, dtype=np.float64)
    stratum_id = np.asarray(strata)
    if score_matrix.ndim != 2 or len(score_matrix) != len(stratum_id):
        raise ValueError("PSU score/stratum dimensions mismatch")
    grand_center = score_matrix.mean(axis=0, keepdims=True)
    meat = np.zeros((score_matrix.shape[1], score_matrix.shape[1]), dtype=np.float64)
    for stratum in pd.unique(stratum_id):
        stratum_scores = score_matrix[stratum_id == stratum]
        if len(stratum_scores) == 1:
            centered = stratum_scores - grand_center
            factor = 1.0
        else:
            centered = stratum_scores - stratum_scores.mean(axis=0, keepdims=True)
            factor = len(stratum_scores) / (len(stratum_scores) - 1.0)
        meat += factor * centered.T @ centered
    return meat


def _build_zero_shot_age_metrics(
    per_checkpoint: pd.DataFrame,
    ensemble: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    estimator_frames: list[tuple[str, pd.DataFrame, str]] = []
    for fold in range(5):
        subset = per_checkpoint.loc[per_checkpoint["checkpoint_outer_fold"].eq(fold)].copy()
        subset["metric_release_id"] = subset["release_id"].fillna("MIMIC_IV_3_1")
        estimator_frames.append((f"fold{fold}", subset, "predicted_age_raw"))
    ensemble_metric = ensemble.copy()
    ensemble_metric["metric_release_id"] = ensemble_metric["release_id"].fillna(
        "MIMIC_IV_3_1"
    )
    estimator_frames.append(("mean5", ensemble_metric, "predicted_age_raw_mean5"))
    population_definitions = {
        "MIMIC_IV_3_1": (
            "mimic_primary_common_support_40_73_integer_year",
            "mimic_sensitivity_adult_nontopcoded_integer_year",
            "mimic_sensitivity_common_support_age_minus_0_5_year",
            "mimic_sensitivity_common_support_age_plus_0_5_year",
        ),
        "NHANES": ("nhanes_primary_examined_40_73_exact",),
    }
    for estimator, source, prediction_column in estimator_frames:
        for (cohort, release_id, variant), cohort_rows in source.groupby(
            ["cohort", "metric_release_id", "input_variant"], sort=False
        ):
            for population in population_definitions[str(cohort)]:
                selected = cohort_rows.copy()
                if str(cohort) == "MIMIC_IV_3_1":
                    common = selected["chronological_age"].between(40, 73, inclusive="both")
                    if population == "mimic_sensitivity_adult_nontopcoded_integer_year":
                        selected = selected.loc[selected["chronological_age"].ge(18)].copy()
                    else:
                        selected = selected.loc[common].copy()
                selected["metric_age"] = selected["chronological_age"].astype(float)
                if population.endswith("minus_0_5_year"):
                    selected["metric_age"] -= 0.5
                elif population.endswith("plus_0_5_year"):
                    selected["metric_age"] += 0.5
                summaries: list[tuple[str, pd.DataFrame, bool]] = [
                    (organ, selected.loc[selected["organ_system"].eq(organ)].copy(), False)
                    for organ in ORGANS
                ]
                summaries.extend(
                    [
                        ("pooled_rows", selected.copy(), False),
                        ("macro_organ_mean", selected.copy(), True),
                    ]
                )
                for summary, summary_rows, macro in summaries:
                    bundle = _point_age_metrics(
                        summary_rows,
                        predicted_column=prediction_column,
                        observed_column="metric_age",
                        macro_by_organ=macro,
                    )
                    for metric in AGE_METRICS:
                        estimate, status = bundle[metric]
                        primary_estimand = bool(
                            estimator == "mean5"
                            and (
                                (
                                    cohort == "MIMIC_IV_3_1"
                                    and variant == MIMIC_PRIMARY_VARIANT
                                    and population
                                    == "mimic_primary_common_support_40_73_integer_year"
                                )
                                or (
                                    cohort == "NHANES"
                                    and variant == NHANES_PRIMARY_VARIANT
                                    and population
                                    == "nhanes_primary_examined_40_73_exact"
                                )
                            )
                        )
                        rows.append(
                            {
                                "cohort": cohort,
                                "release_id": release_id,
                                "input_variant": variant,
                                "population": population,
                                "organ_system": summary,
                                "estimator": estimator,
                                "metric": metric,
                                "n_participants": int(summary_rows["participant_id"].nunique()),
                                "n_rows": int(len(summary_rows)),
                                "estimability_status": status,
                                "estimate": estimate,
                                "ci_lower": None,
                                "ci_upper": None,
                                "uncertainty_scope": "point_estimate_only_no_resampling",
                                "bootstrap_replicates": 0,
                                "bootstrap_seed": bootstrap_seed,
                                "no_external_fit": True,
                                "raw_zero_shot_primary": primary_estimand,
                            }
                        )
    return pd.DataFrame(rows)


def _build_vocabulary_coverage(
    cache_dir: Path,
    predictions: Mapping[str, Path],
) -> pd.DataFrame:
    audit = pd.read_csv(predictions["vocabulary_audit"])
    if set(audit["vocabulary_kind"]) != {"code", "text"}:
        raise AssertionError("vocabulary prefix audit lacks code/text rows")
    audit_by_kind = audit.set_index("vocabulary_kind")
    icd = pd.read_parquet(cache_dir / "mimic_icd_events_preindex.parquet")
    rx = pd.read_parquet(
        cache_dir / "mimic_inpatient_prescription_proxy_preindex.parquet"
    )
    dropped = pd.read_parquet(cache_dir / "mimic_vocabulary_drop_audit.parquet")
    mimic_participants = pd.read_parquet(cache_dir / "mimic_participants.parquet")
    nhanes_participants = pd.read_parquet(cache_dir / "nhanes_participants.parquet")
    extension_manifest_sha = sha256_file(
        Path(predictions["vocabulary_extension_manifest"])
    )
    if not audit["extension_manifest_sha256"].eq(extension_manifest_sha).all():
        raise AssertionError("vocabulary audit does not bind extension manifest")
    root = Path(__file__).resolve().parents[4]
    code_vocab = pd.read_parquet(
        root / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_code_vocab.parquet"
    )
    text_vocab = pd.read_parquet(
        root / "Revision-0710/01_data_rebuild/04_cache_v1/vocab/event_text_vocab.parquet"
    )
    frozen_codes = set(code_vocab["raw_text"].astype(str))
    frozen_text = set(
        zip(
            text_vocab["semantic_role"].astype(str),
            text_vocab["raw_text"].astype(str),
        )
    )
    rows: list[dict[str, Any]] = []

    def append_row(
        *,
        variant: str,
        kind: str,
        modality: str,
        code_system: str,
        token_class: str,
        reason: str,
        source: pd.DataFrame,
        selected: pd.DataFrame,
        token_values: pd.Series | None,
        participant_count_override: int | None = None,
        source_rows_override: int | None = None,
    ) -> None:
        summary = audit_by_kind.loc[kind]
        rows.append(
            {
                "cohort": "MIMIC_IV_3_1",
                "source_partition": "MIMIC_IV_3_1",
                "input_variant": variant,
                "vocabulary_kind": kind,
                "modality": modality,
                "code_system": code_system,
                "token_class": token_class,
                "reason": reason,
                "source_rows": (
                    int(len(source))
                    if source_rows_override is None
                    else int(source_rows_override)
                ),
                "unique_canonical_tokens": (
                    0 if token_values is None else int(token_values.nunique())
                ),
                "participants": (
                    int(source["participant_id"].astype(str).nunique())
                    if participant_count_override is None
                    and "participant_id" in source.columns
                    else 0
                    if participant_count_override is None
                    else int(participant_count_override)
                ),
                "pre_crop_active_events": (
                    0 if token_class in {"invalid_dropped", "zero_event"} else int(len(source))
                ),
                "post_crop_active_events": (
                    0 if token_class in {"invalid_dropped", "zero_event"} else int(len(selected))
                ),
                "original_vocab_rows": int(summary["original_vocab_rows"]),
                "original_vocab_prefix_bytes": int(summary["original_vocab_prefix_bytes"]),
                "original_vocab_prefix_sha256": str(summary["original_vocab_prefix_sha256"]),
                "extension_rows": int(summary["extension_rows"]),
                "original_embedding_rows": int(summary["original_embedding_rows"]),
                "original_embedding_prefix_bytes": int(summary["original_embedding_prefix_bytes"]),
                "original_embedding_prefix_sha256": str(summary["original_embedding_prefix_sha256"]),
                "extension_manifest_sha256": extension_manifest_sha,
                "original_embedding_prefix_bytes_unchanged": True,
                "label_or_endpoint_fields_read": False,
            }
        )

    for variant in MIMIC_VARIANTS:
        variant_events = (
            icd
            if variant != MIMIC_RX_SENSITIVITY
            else pd.concat([icd, rx], ignore_index=True)
        )
        selected = crop_variant_timeline(variant_events, variant)
        for code_system in ("ICD9", "ICD10"):
            active_system = icd.loc[icd["code_system"].eq(code_system)].copy()
            selected_system = selected.loc[
                selected["source"].eq("hospital_icd")
                & selected["code_system"].eq(code_system)
            ].copy()
            for kind in ("code", "text"):
                if kind == "code":
                    active_exact = active_system["canonical_code"].astype(str).isin(
                        frozen_codes
                    )
                    selected_exact = selected_system["canonical_code"].astype(str).isin(
                        frozen_codes
                    )
                    token_column = "canonical_code"
                else:
                    active_exact = active_system["canonical_description"].astype(str).map(
                        lambda value: ("icd_description", value) in frozen_text
                    )
                    selected_exact = selected_system["canonical_description"].astype(str).map(
                        lambda value: ("icd_description", value) in frozen_text
                    )
                    token_column = "canonical_description"
                for token_class, exact_flag in (
                    ("exact_frozen_vocab", True),
                    ("deterministic_extension", False),
                ):
                    source_class = active_system.loc[active_exact.eq(exact_flag)].copy()
                    selected_class = selected_system.loc[
                        selected_exact.eq(exact_flag)
                    ].copy()
                    append_row(
                        variant=variant,
                        kind=kind,
                        modality="hospital_icd",
                        code_system=code_system,
                        token_class=token_class,
                        reason="none",
                        source=source_class,
                        selected=selected_class,
                        token_values=source_class[token_column],
                    )
                invalid = dropped.loc[
                    dropped["modality"].eq("hospital_icd")
                    & dropped["code_system"].eq(code_system)
                ].copy()
                append_row(
                    variant=variant,
                    kind=kind,
                    modality="hospital_icd",
                    code_system=code_system,
                    token_class="invalid_dropped",
                    reason="invalid_or_unsupported_code",
                    source=invalid,
                    selected=invalid.iloc[0:0],
                    token_values=None,
                )
        unsupported = dropped.loc[
            dropped["modality"].eq("hospital_icd")
            & dropped["code_system"].eq("not_applicable")
        ].copy()
        for kind in ("code", "text"):
            append_row(
                variant=variant,
                kind=kind,
                modality="hospital_icd",
                code_system="not_applicable",
                token_class="invalid_dropped",
                reason="invalid_or_unsupported_code",
                source=unsupported,
                selected=unsupported.iloc[0:0],
                token_values=None,
            )
        all_participants = set(mimic_participants["subject_id"].astype(str))
        active_participants = set(selected["participant_id"].astype(str))
        zero_event_participants = len(all_participants.difference(active_participants))
        for kind in ("code", "text"):
            append_row(
                variant=variant,
                kind=kind,
                modality="padding",
                code_system="not_applicable",
                token_class="zero_event",
                reason="padding",
                source=selected.iloc[0:0],
                selected=selected.iloc[0:0],
                token_values=None,
                participant_count_override=zero_event_participants,
                source_rows_override=0,
            )
        if variant != MIMIC_RX_SENSITIVITY:
            for kind in ("code", "text"):
                append_row(
                    variant=variant,
                    kind=kind,
                    modality="inpatient_prescription_proxy",
                    code_system="INPATIENT_RX",
                    token_class="zero_event",
                    reason="source_forbidden_for_input_variant",
                    source=rx,
                    selected=rx.iloc[0:0],
                    token_values=None,
                )
        else:
            append_row(
                variant=variant,
                kind="code",
                modality="inpatient_prescription_proxy",
                code_system="INPATIENT_RX",
                token_class="zero_event",
                reason="unmapped_inpatient_prescription",
                source=rx.iloc[0:0],
                selected=rx.iloc[0:0],
                token_values=None,
            )
            rx_selected = selected.loc[
                selected["source"].eq("inpatient_prescription_proxy")
            ].copy()
            rx_exact = rx["canonical_description"].astype(str).map(
                lambda value: ("inpatient_prescription_proxy", value) in frozen_text
            )
            rx_selected_exact = rx_selected["canonical_description"].astype(str).map(
                lambda value: ("inpatient_prescription_proxy", value) in frozen_text
            )
            for token_class, exact_flag in (
                ("exact_frozen_vocab", True),
                ("deterministic_extension", False),
            ):
                source_class = rx.loc[rx_exact.eq(exact_flag)].copy()
                selected_class = rx_selected.loc[
                    rx_selected_exact.eq(exact_flag)
                ].copy()
                append_row(
                    variant=variant,
                    kind="text",
                    modality="inpatient_prescription_proxy",
                    code_system="INPATIENT_RX",
                    token_class=token_class,
                    reason="none",
                    source=source_class,
                    selected=selected_class,
                    token_values=source_class["canonical_description"],
                )
            invalid_rx = dropped.loc[
                dropped["modality"].eq("inpatient_prescription_proxy")
            ].copy()
            append_row(
                variant=variant,
                kind="text",
                modality="inpatient_prescription_proxy",
                code_system="INPATIENT_RX",
                token_class="invalid_dropped",
                reason="invalid_or_missing_text",
                source=invalid_rx,
                selected=invalid_rx.iloc[0:0],
                token_values=None,
            )
    for release in RELEASES:
        for kind in ("code", "text"):
            summary = audit_by_kind.loc[kind]
            rows.append(
                {
                    "cohort": "NHANES",
                    "source_partition": release,
                    "input_variant": NHANES_PRIMARY_VARIANT,
                    "vocabulary_kind": kind,
                    "modality": "padding",
                    "code_system": "not_applicable",
                    "token_class": "zero_event",
                    "reason": "padding",
                    "source_rows": 0,
                    "unique_canonical_tokens": 0,
                    "participants": int(
                        nhanes_participants["release_id"].eq(release).sum()
                    ),
                    "pre_crop_active_events": 0,
                    "post_crop_active_events": 0,
                    "original_vocab_rows": int(summary["original_vocab_rows"]),
                    "original_vocab_prefix_bytes": int(summary["original_vocab_prefix_bytes"]),
                    "original_vocab_prefix_sha256": str(summary["original_vocab_prefix_sha256"]),
                    "extension_rows": int(summary["extension_rows"]),
                    "original_embedding_rows": int(summary["original_embedding_rows"]),
                    "original_embedding_prefix_bytes": int(summary["original_embedding_prefix_bytes"]),
                    "original_embedding_prefix_sha256": str(summary["original_embedding_prefix_sha256"]),
                    "extension_manifest_sha256": extension_manifest_sha,
                    "original_embedding_prefix_bytes_unchanged": True,
                    "label_or_endpoint_fields_read": False,
                }
            )
    return pd.DataFrame(rows)


def _build_mimic_flow(
    manifest: Mapping[str, Any],
    incident: pd.DataFrame,
    recurrent: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    counts = dict(manifest["mimic_flow_counts_before_outcome_read"])
    endpoints = pd.concat([incident, recurrent], ignore_index=True)
    endpoint_subjects = int(endpoints["subject_id"].nunique())
    expected_endpoint_subjects = set(
        eligibility.loc[
            eligibility["eligibility_status"].eq("endpoint_eligible"),
            "subject_id",
        ].astype(str)
    )
    if set(endpoints["subject_id"].astype(str)) != expected_endpoint_subjects:
        raise AssertionError("endpoint rows do not cover the exact eligible subject set")
    observed_end_subjects = int(
        endpoints.loc[endpoints["subject_observation_end_days"].gt(0), "subject_id"]
        .astype(str)
        .nunique()
    )
    observed_horizon_subjects = int(
        endpoints.loc[endpoints["observed_at_horizon"].astype(bool), "subject_id"]
        .astype(str)
        .nunique()
    )
    counts["has_subject_specific_observation_end"] = observed_end_subjects
    counts["has_valid_6_or_12_month_endpoint_window"] = observed_horizon_subjects
    counts["no_preindex_icd9_for_primary_endpoint"] = min(
        observed_horizon_subjects, endpoint_subjects
    )
    counts["model_input_cache_eligible"] = min(
        counts["no_preindex_icd9_for_primary_endpoint"],
        int(manifest["mimic_participants"]),
    )
    rows = []
    for variant in MIMIC_VARIANTS:
        previous = None
        for order, step in enumerate(MIMIC_FLOW_STEPS, start=1):
            current = int(counts[step])
            if previous is not None and current > previous:
                raise AssertionError(f"MIMIC flow increases at {step}")
            excluded = 0 if previous is None else max(previous - current, 0)
            rows.append(
                {
                    "input_variant": variant,
                    "step_order": order,
                    "step_name": step,
                    "n_subjects": current,
                    "excluded_at_step": excluded,
                    "definition": f"frozen mechanical eligibility step: {step}",
                }
            )
            previous = current
    return pd.DataFrame(rows)


def _global_subject_bootstrap_plan(
    *,
    global_subject_ids: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the one shared global subject-cluster plan exactly once."""
    identifiers = np.sort(np.asarray(global_subject_ids).astype(str))
    if len(identifiers) == 0 or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("global bootstrap subject universe is empty or duplicated")
    counts = np.zeros((replicates, len(identifiers)), dtype=np.uint16)
    for replicate in range(replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        sampled_index = rng.integers(0, len(identifiers), size=len(identifiers))
        current = np.bincount(sampled_index, minlength=len(identifiers))
        if int(current.max(initial=0)) > np.iinfo(np.uint16).max:
            raise OverflowError("bootstrap cluster multiplicity exceeds uint16")
        counts[replicate] = current.astype(np.uint16)
    return identifiers, counts


def _conditional_bootstrap_prediction_values(
    *,
    testing: pd.DataFrame,
    risk: np.ndarray,
    base_risk: np.ndarray,
    probability: np.ndarray,
    outcome: np.ndarray,
    ipcw_weight: np.ndarray,
    plan_subject_ids: np.ndarray,
    plan_counts: np.ndarray,
) -> dict[str, tuple[float | None, float | None]]:
    """Resample held-out subjects while keeping all training-fold nuisances fixed."""
    draws: dict[str, list[float]] = {
        "harrell_c": [],
        "delta_harrell_c": [],
        "ipcw_auc": [],
        "ipcw_brier": [],
        "ipcw_linear_calibration_intercept": [],
        "ipcw_linear_calibration_slope": [],
    }
    participant_ids = testing["subject_id"].astype(str).to_numpy()
    positions = np.searchsorted(plan_subject_ids, participant_ids)
    if (
        np.any(positions >= len(plan_subject_ids))
        or not np.array_equal(plan_subject_ids[positions], participant_ids)
    ):
        raise AssertionError("held-out subject is absent from global bootstrap universe")
    if len(participant_ids) == 0:
        return {metric: (None, None) for metric in draws}
    row_index = np.arange(len(testing), dtype=np.int64)
    time = testing["observed_time_days"].to_numpy(dtype=np.float64)
    event = testing["event_indicator"].to_numpy(dtype=np.int8)
    for replicate in range(plan_counts.shape[0]):
        index = np.repeat(row_index, plan_counts[replicate, positions].astype(np.int64))
        if len(index) < 4 or int(event[index].sum()) == 0:
            continue
        try:
            current_c = harrell_c(time[index], event[index], risk[index])
            base_c = harrell_c(time[index], event[index], base_risk[index])
            draws["harrell_c"].append(current_c)
            draws["delta_harrell_c"].append(current_c - base_c)
            sampled_outcome = outcome[index]
            sampled_weight = ipcw_weight[index]
            draws["ipcw_auc"].append(
                weighted_auc(sampled_outcome, probability[index], sampled_weight)
            )
            draws["ipcw_brier"].append(
                weighted_brier(
                    sampled_outcome,
                    probability[index],
                    sampled_weight,
                    normalization_n=len(index),
                )
            )
            intercept, slope = _weighted_linear_calibration(
                sampled_outcome, probability[index], sampled_weight
            )
            draws["ipcw_linear_calibration_intercept"].append(intercept)
            draws["ipcw_linear_calibration_slope"].append(slope)
        except (ValueError, np.linalg.LinAlgError):
            pass
    result = {}
    for metric, values in draws.items():
        if len(values) < max(20, plan_counts.shape[0] // 2):
            result[metric] = (None, None)
        else:
            lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
            result[metric] = (float(lower), float(upper))
    return result


def _build_mimic_predictive_metrics(
    incident: pd.DataFrame,
    ensemble: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    prediction = ensemble.loc[ensemble["cohort"].eq("MIMIC_IV_3_1")].copy()
    prediction = prediction.rename(columns={"participant_id": "subject_id"})
    analysis = incident.merge(
        prediction[
            [
                "subject_id",
                "input_variant",
                "organ_system",
                "corrected_age_gap_mean5",
            ]
        ],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    )
    analysis["age_per10"] = analysis["chronological_age"].astype(float) / 10.0
    rows = []
    definitions = {
        "ipcw_outcome_definition": "case=target_before_death_censor_horizon; control=event_free_at_horizon_or_competing_death_before_horizon; loss_or_admin_before_horizon=unobserved",
        "ipcw_censor_definition": "G fitted on training folds; censor event=subject_observation_end_or_death_cap_before_target_death_horizon; target_and_death_censor_G_as_non-censor_terminal",
        "ipcw_weight_definition": "left_limit_G_min_0.05_training_weight_cap_p99",
        "absolute_risk_estimand": "target_cif_from_postjump_breslow_target_and_death_cause_specific_hazards",
        "ipcw_auc_score": "target_cif_probability",
        "brier_normalization": "sum_ipcw_squared_error_divided_by_full_test_n",
        "bootstrap_unit": "one_global_subject_cluster_draw_shared_across_organs_horizons_folds_variants",
        "bootstrap_refits_target_cox_death_cox_and_censor_km": False,
        "uncertainty_scope": "conditional_on_fixed_training_fold_target_death_cox_and_censor_km",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_namespace": f"PCG64-SeedSequence-{bootstrap_seed}-global-MIMIC-subject",
    }
    global_subject_ids = np.sort(analysis["subject_id"].astype(str).unique())
    plan_subject_ids, plan_counts = _global_subject_bootstrap_plan(
        global_subject_ids=global_subject_ids,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    for (variant, organ, horizon), group in analysis.groupby(
        ["input_variant", "organ_system", "horizon_months"], sort=False
    ):
        horizon_days = months_to_days(int(horizon))
        for fold in range(5):
            training = group.loc[group["hash_fold"].ne(fold)].reset_index(drop=True)
            testing = group.loc[group["hash_fold"].eq(fold)].reset_index(drop=True)
            n_events = int(testing["event_indicator"].sum())
            n_deaths = int(testing["competing_death_indicator"].sum())
            common = {
                "input_variant": variant,
                "organ_system": organ,
                "horizon_months": int(horizon),
                "analysis_role": "primary_in_system_incident_prediction",
                "hash_fold": fold,
                "n_test": len(testing),
                "n_events": n_events,
                "n_competing_deaths": n_deaths,
                "ipcw_training_fold_only": True,
                "ipcw_g_min_raw": None,
                "ipcw_g_floor": 0.05,
                "ipcw_weight_cap_p99_training": None,
                "ipcw_effective_sample_size": None,
                "ipcw_n_observed_at_horizon": None,
                "ipcw_n_censored_before_horizon": None,
                "ipcw_n_weights_capped": None,
                **definitions,
            }
            if len(testing) < 4 or n_events == 0:
                status = "zero_target_events" if n_events == 0 else "insufficient_complete_cases"
                for model in ("age_sex", "age_sex_corrected_gap"):
                    for metric in (
                        "harrell_c",
                        "delta_harrell_c",
                        "ipcw_auc",
                        "ipcw_brier",
                        "ipcw_linear_calibration_intercept",
                        "ipcw_linear_calibration_slope",
                    ):
                        rows.append(
                            {
                                **common,
                                "model": model,
                                "metric": metric,
                                "estimability_status": status,
                                "estimate": None,
                                "ci_lower": None,
                                "ci_upper": None,
                            }
                        )
                continue
            try:
                base_risk, base_probability = _cause_specific_cif_predictions(
                    training, testing, ["age_per10", "sex_male"], horizon_days
                )
                augmented_risk, augmented_probability = _cause_specific_cif_predictions(
                    training,
                    testing,
                    ["age_per10", "sex_male", "corrected_age_gap_mean5"],
                    horizon_days,
                )
                cause_test = (
                    testing["event_indicator"].to_numpy(dtype=np.int8)
                    + 2 * testing["competing_death_indicator"].to_numpy(dtype=np.int8)
                )
                censor_training = training["censor_reason"].isin(
                    ["subject_observation_end", "death_ascertainment_cap"]
                ).to_numpy(dtype=np.int8)
                outcome, weight, _, diagnostics = ipcw_binary_weights(
                    testing["observed_time_days"].to_numpy(dtype=np.float64),
                    cause_test,
                    horizon_days,
                    training["observed_time_days"].to_numpy(dtype=np.float64),
                    censor_training,
                    return_diagnostics=True,
                )
                common.update(
                    {
                        "ipcw_g_min_raw": diagnostics["g_min_raw"],
                        "ipcw_weight_cap_p99_training": diagnostics[
                            "weight_cap_p99_training"
                        ],
                        "ipcw_effective_sample_size": diagnostics[
                            "effective_sample_size"
                        ],
                        "ipcw_n_observed_at_horizon": diagnostics[
                            "n_observed_at_horizon"
                        ],
                        "ipcw_n_censored_before_horizon": diagnostics[
                            "n_censored_before_horizon"
                        ],
                        "ipcw_n_weights_capped": diagnostics["n_weights_capped"],
                    }
                )
            except (ValueError, np.linalg.LinAlgError):
                for model in ("age_sex", "age_sex_corrected_gap"):
                    for metric in (
                        "harrell_c",
                        "delta_harrell_c",
                        "ipcw_auc",
                        "ipcw_brier",
                        "ipcw_linear_calibration_intercept",
                        "ipcw_linear_calibration_slope",
                    ):
                        rows.append(
                            {
                                **common,
                                "model": model,
                                "metric": metric,
                                "estimability_status": "singular_design",
                                "estimate": None,
                                "ci_lower": None,
                                "ci_upper": None,
                            }
                        )
                continue
            event = testing["event_indicator"].to_numpy(dtype=np.int8)
            time = testing["observed_time_days"].to_numpy(dtype=np.float64)
            base_c = harrell_c(time, event, base_risk)
            for model, risk, probability in (
                ("age_sex", base_risk, base_probability),
                ("age_sex_corrected_gap", augmented_risk, augmented_probability),
            ):
                c_value = harrell_c(time, event, risk)
                intercept, slope = _weighted_linear_calibration(outcome, probability, weight)
                point = {
                    "harrell_c": c_value,
                    "delta_harrell_c": c_value - base_c,
                    "ipcw_auc": weighted_auc(outcome, probability, weight),
                    "ipcw_brier": weighted_brier(
                        outcome,
                        probability,
                        weight,
                        normalization_n=len(testing),
                    ),
                    "ipcw_linear_calibration_intercept": intercept,
                    "ipcw_linear_calibration_slope": slope,
                }
                intervals = _conditional_bootstrap_prediction_values(
                    testing=testing,
                    risk=risk,
                    base_risk=base_risk,
                    probability=probability,
                    outcome=outcome,
                    ipcw_weight=weight,
                    plan_subject_ids=plan_subject_ids,
                    plan_counts=plan_counts,
                )
                for metric, estimate in point.items():
                    lower, upper = intervals[metric]
                    rows.append(
                        {
                            **common,
                            "model": model,
                            "metric": metric,
                            "estimability_status": "estimable",
                            "estimate": estimate,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "uncertainty_scope": (
                                definitions["uncertainty_scope"]
                                if lower is not None
                                else "point_estimate_only_bootstrap_insufficient"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _build_mimic_associations(
    incident: pd.DataFrame, ensemble: pd.DataFrame
) -> pd.DataFrame:
    from scipy.stats import norm, pearsonr

    prediction = ensemble.loc[ensemble["cohort"].eq("MIMIC_IV_3_1")].rename(
        columns={"participant_id": "subject_id"}
    )
    analysis = incident.merge(
        prediction[
            ["subject_id", "input_variant", "organ_system", "corrected_age_gap_mean5"]
        ],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    )
    rows = []
    for (variant, organ, horizon), group in analysis.groupby(
        ["input_variant", "organ_system", "horizon_months"], sort=False
    ):
        group = group.reset_index(drop=True)
        gap_sd = float(group["corrected_age_gap_mean5"].std(ddof=1))
        common = {
            "input_variant": variant,
            "organ_system": organ,
            "horizon_months": int(horizon),
            "analysis_population": "full_nonheldout_incident_eligible_organ_specific_cohort",
            "n": int(group["subject_id"].nunique()),
            "baseline_exclusion_rule": "exclude_baseline_target_icd10_and_any_preindex_icd9_then_use_all_remaining_hash_folds",
            "exposure": "corrected_age_gap_mean5_per_incident_eligible_analysis_population_sd",
            "covariates": "chronological_age_integer_year + sex_male + corrected_age_gap_mean5_per_incident_eligible_analysis_population_sd",
            "ties": "efron",
            "standard_error": "robust_sandwich_subject",
            "ph_time_transform": "log_observed_event_time",
            "ph_residual_covariance": "scaled_schoenfeld_equals_raw_schoenfeld_times_PHReg_cov_params",
            "ph_n_events": int(group["event_indicator"].sum()),
            "interpretation": "nonheldout_incident_eligible_association_not_prediction_not_causal",
        }
        if not np.isfinite(gap_sd) or gap_sd <= 0 or int(group["event_indicator"].sum()) == 0:
            status = "zero_target_events" if int(group["event_indicator"].sum()) == 0 else "singular_design"
            rows.append(
                {
                    **common,
                    "estimability_status": status,
                    "hazard_ratio": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "p_value": None,
                    "bh_q_value": None,
                    "ph_test_p": None,
                }
            )
            continue
        group["gap_sd"] = group["corrected_age_gap_mean5"] / gap_sd
        try:
            complete, result = _cox_fit(
                group,
                ["chronological_age", "sex_male", "gap_sd"],
                robust_groups=group["subject_id"].to_numpy(),
            )
            coefficient = float(result.params[2])
            standard_error = float(result.bse[2])
            p_value = float(2.0 * norm.sf(abs(coefficient / standard_error)))
            residual_matrix = np.asarray(
                result.schoenfeld_residuals, dtype=np.float64
            )
            covariance = np.asarray(result.cov_params(), dtype=np.float64)
            if covariance.shape != (3, 3) or residual_matrix.shape[1] != 3:
                raise ValueError("scaled Schoenfeld dimensions mismatch")
            residual = (residual_matrix @ covariance)[:, 2]
            event_mask = complete["event_indicator"].to_numpy(dtype=bool)
            residual = residual[event_mask]
            event_time = np.log(
                complete.loc[event_mask, "observed_time_days"].to_numpy(dtype=np.float64)
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
                    "hazard_ratio": float(np.exp(coefficient)),
                    "ci_lower": float(np.exp(coefficient - 1.959963984540054 * standard_error)),
                    "ci_upper": float(np.exp(coefficient + 1.959963984540054 * standard_error)),
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
                    "hazard_ratio": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "p_value": None,
                    "bh_q_value": None,
                    "ph_test_p": None,
                }
            )
    result = pd.DataFrame(rows)
    for _, indices in result.groupby(["input_variant", "horizon_months"]).groups.items():
        valid = [index for index in indices if pd.notna(result.at[index, "p_value"])]
        if valid:
            result.loc[valid, "bh_q_value"] = benjamini_hochberg(
                result.loc[valid, "p_value"].to_numpy(dtype=np.float64),
                family_size=7,
            )
    return result


def _build_mimic_curves(
    incident: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows = []
    curves = (
        "aalen_johansen_target_cif",
        "aalen_johansen_death_cif",
        "kaplan_meier_target_free_survival_cause_specific",
    )
    for (variant, organ, horizon), group in incident.groupby(
        ["input_variant", "organ_system", "horizon_months"], sort=False
    ):
        time = group["observed_time_days"].to_numpy(dtype=np.float64)
        cause = (
            group["event_indicator"].to_numpy(dtype=np.int8)
            + 2 * group["competing_death_indicator"].to_numpy(dtype=np.int8)
        )
        evaluation = np.sort(np.unique(np.concatenate([[0.0], time])))
        sorted_time = np.sort(time)
        at_risk_vector = len(time) - np.searchsorted(
            sorted_time, evaluation, side="left"
        )
        target_points, target_counts = np.unique(
            time[cause == 1], return_counts=True
        )
        death_points, death_counts = np.unique(
            time[cause == 2], return_counts=True
        )
        censor_points, censor_counts = np.unique(
            time[cause == 0], return_counts=True
        )
        target_lookup = dict(zip(target_points.tolist(), target_counts.tolist()))
        death_lookup = dict(zip(death_points.tolist(), death_counts.tolist()))
        censor_lookup = dict(zip(censor_points.tolist(), censor_counts.tolist()))
        aj = aalen_johansen(time, cause)
        for curve in curves:
            if curve.startswith("aalen_johansen"):
                source = aj
                column = "target_cif" if curve.endswith("target_cif") else "death_cif"
                default = 0.0
                role = "aj_primary"
            else:
                event = (
                    cause == 1 if curve.endswith("cause_specific") else cause != 0
                ).astype(np.int8)
                source = _km_curve(time, event)
                column = "estimate"
                default = 1.0
                role = "km_descriptive_sensitivity"
            positions = np.searchsorted(
                source["time_days"].to_numpy(dtype=np.float64), evaluation, side="right"
            ) - 1 if len(source) else np.full(len(evaluation), -1)
            estimate = np.full(len(evaluation), default, dtype=np.float64)
            valid = positions >= 0
            if len(source):
                estimate[valid] = source[column].to_numpy(dtype=np.float64)[positions[valid]]
            for index, point in enumerate(evaluation):
                at_risk = int(at_risk_vector[index])
                target = int(target_lookup.get(float(point), 0))
                death = int(death_lookup.get(float(point), 0))
                censored = int(censor_lookup.get(float(point), 0))
                status = "estimable"
                if curve.endswith("target_cif") and int(np.sum(cause == 1)) == 0:
                    status = "zero_target_events"
                elif curve.endswith("death_cif") and int(np.sum(cause == 2)) == 0:
                    status = "zero_competing_deaths"
                rows.append(
                    {
                        "input_variant": variant,
                        "organ_system": organ,
                        "horizon_months": int(horizon),
                        "analysis_role": role,
                        "curve": curve,
                        "time_days": float(point),
                        "n_at_risk": at_risk,
                        "target_events": target,
                        "death_events": death,
                        "censored": censored,
                        "estimability_status": status,
                        "estimate": float(estimate[index]) if status == "estimable" else None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "uncertainty_scope": "point_curve_only_no_simultaneous_band",
                        "bootstrap_replicates": 0,
                        "bootstrap_seed": bootstrap_seed,
                        "interpretation": "hospital_recorded_incident_risk_not_population_incidence_not_causal",
                    }
                )
    return pd.DataFrame(rows)


def _build_recurrent_metrics(
    recurrent: pd.DataFrame,
    ensemble: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    prediction = ensemble.loc[ensemble["cohort"].eq("MIMIC_IV_3_1")].rename(
        columns={"participant_id": "subject_id"}
    )
    analysis = recurrent.merge(
        prediction[
            ["subject_id", "input_variant", "organ_system", "corrected_age_gap_mean5"]
        ],
        on=["subject_id", "input_variant", "organ_system"],
        how="inner",
        validate="many_to_one",
    )
    rows = []
    for (variant, organ, horizon), group in analysis.groupby(
        ["input_variant", "organ_system", "horizon_months"], sort=False
    ):
        group = group.reset_index(drop=True)
        target_events = int(group["event_indicator"].sum())
        death_events = int(group["competing_death_indicator"].sum())
        cause = (
            group["event_indicator"].to_numpy(dtype=np.int8)
            + 2 * group["competing_death_indicator"].to_numpy(dtype=np.int8)
        )
        aj = aalen_johansen(
            group["observed_time_days"].to_numpy(dtype=np.float64), cause
        )
        target_cif = float(aj["target_cif"].iloc[-1]) if len(aj) else None
        death_cif = float(aj["death_cif"].iloc[-1]) if len(aj) else None
        estimates: dict[str, tuple[float | None, float | None, float | None, str]] = {
            "target_cumulative_incidence": (
                target_cif if target_events else None,
                None,
                None,
                "estimable" if target_events else "zero_target_events",
            ),
            "death_cumulative_incidence": (
                death_cif if death_events else None,
                None,
                None,
                "estimable" if death_events else "zero_competing_deaths",
            ),
        }
        if target_events:
            gap_sd = float(group["corrected_age_gap_mean5"].std(ddof=1))
            group["gap_sd"] = group["corrected_age_gap_mean5"] / gap_sd
            try:
                _, result = _cox_fit(
                    group,
                    ["chronological_age", "sex_male", "gap_sd"],
                    robust_groups=group["subject_id"].to_numpy(),
                )
                coefficient = float(result.params[2])
                error = float(result.bse[2])
                estimates["cause_specific_hazard_ratio"] = (
                    float(np.exp(coefficient)),
                    float(np.exp(coefficient - 1.959963984540054 * error)),
                    float(np.exp(coefficient + 1.959963984540054 * error)),
                    "estimable",
                )
            except (ValueError, np.linalg.LinAlgError):
                estimates["cause_specific_hazard_ratio"] = (
                    None,
                    None,
                    None,
                    "singular_design",
                )
        else:
            estimates["cause_specific_hazard_ratio"] = (
                None,
                None,
                None,
                "zero_target_events",
            )
        for metric in (
            "cause_specific_hazard_ratio",
            "target_cumulative_incidence",
            "death_cumulative_incidence",
        ):
            estimate, lower, upper, status = estimates[metric]
            rows.append(
                {
                    "input_variant": variant,
                    "organ_system": organ,
                    "horizon_months": int(horizon),
                    "metric": metric,
                    "n": len(group),
                    "target_events": target_events,
                    "death_events": death_events,
                    "estimability_status": status,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "uncertainty_scope": (
                        "robust_wald_95_percent_interval"
                        if metric == "cause_specific_hazard_ratio"
                        else "point_estimate_only_no_resampling"
                    ),
                    "physically_separate_from_incident": True,
                    "interpretation": "recurrent_hospital_recorded_sensitivity_not_incident_prediction_not_causal",
                }
            )
    return pd.DataFrame(rows)


def _build_nhanes_local_calibration(
    ensemble: pd.DataFrame, participants: pd.DataFrame
) -> pd.DataFrame:
    predictions = ensemble.loc[ensemble["cohort"].eq("NHANES")].merge(
        participants[["participant_id", "release_id", "hash_fold"]],
        on=["participant_id", "release_id"],
        how="inner",
        validate="many_to_one",
    )
    rows = []
    formula = "chronological_age = beta0 + beta1*predicted_age_raw_mean5 + beta2*sex_male"
    order = "intercept,predicted_age_raw_mean5,sex_male"
    rule = "fit_on_other_four_release_specific_participant_hash_folds"
    for (release, organ), group in predictions.groupby(
        ["release_id", "organ_system"], sort=False
    ):
        for fold in range(5):
            training = group.loc[group["hash_fold"].ne(fold)].copy()
            testing = group.loc[group["hash_fold"].eq(fold)].copy()
            try:
                coefficient = fit_crossfit_age_calibration(
                    training["predicted_age_raw_mean5"].to_numpy(dtype=np.float64),
                    training["sex_male"].to_numpy(dtype=np.float64),
                    training["chronological_age"].to_numpy(dtype=np.float64),
                )
                calibrated = (
                    coefficient[0]
                    + coefficient[1]
                    * testing["predicted_age_raw_mean5"].to_numpy(dtype=np.float64)
                    + coefficient[2] * testing["sex_male"].to_numpy(dtype=np.float64)
                )
                status = "estimable"
            except (ValueError, np.linalg.LinAlgError):
                coefficient = np.full(3, np.nan)
                calibrated = np.full(len(testing), np.nan)
                status = "singular_design"
            for term, estimate in zip(
                ("intercept", "predicted_age_raw_mean5", "sex_male"), coefficient
            ):
                rows.append(
                    {
                        "release_id": release,
                        "input_variant": NHANES_PRIMARY_VARIANT,
                        "organ_system": organ,
                        "hash_fold": fold,
                        "record_type": "coefficient",
                        "term": term,
                        "metric": "not_applicable_coefficient_row",
                        "regression_formula": formula,
                        "coefficient_application_order": order,
                        "fit_fold_rule": rule,
                        "fit_n": len(training),
                        "test_n": len(testing),
                        "estimability_status": status,
                        "estimate": float(estimate) if np.isfinite(estimate) else None,
                        "zero_shot_primary": False,
                        "label": "NHANES-local cross-fitted calibration sensitivity; not zero-shot",
                    }
                )
            for metric in AGE_METRICS:
                metric_status = status
                estimate = None
                if status == "estimable":
                    try:
                        estimate = _age_metric(
                            metric,
                            testing["chronological_age"].to_numpy(dtype=np.float64),
                            calibrated,
                        )
                    except ValueError:
                        metric_status = "singular_design"
                rows.append(
                    {
                        "release_id": release,
                        "input_variant": NHANES_PRIMARY_VARIANT,
                        "organ_system": organ,
                        "hash_fold": fold,
                        "record_type": "heldout_metric",
                        "term": "not_applicable_metric_row",
                        "metric": metric,
                        "regression_formula": formula,
                        "coefficient_application_order": order,
                        "fit_fold_rule": rule,
                        "fit_n": len(training),
                        "test_n": len(testing),
                        "estimability_status": metric_status,
                        "estimate": estimate,
                        "zero_shot_primary": False,
                        "label": "NHANES-local cross-fitted calibration sensitivity; not zero-shot",
                    }
                )
    return pd.DataFrame(rows)


def _build_nhanes_questionnaire_associations(
    ensemble: pd.DataFrame,
    participants: pd.DataFrame,
    questionnaire: pd.DataFrame,
    survey_design: pd.DataFrame,
) -> pd.DataFrame:
    from scipy.stats import t

    prediction = ensemble.loc[ensemble["cohort"].eq("NHANES")].merge(
        participants,
        on=["participant_id", "release_id"],
        how="inner",
        suffixes=("", "_participant"),
        validate="many_to_one",
    )
    analysis = prediction.merge(
        questionnaire,
        on=["participant_id", "release_id", "organ_system"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for release in RELEASES:
        weight_variable = "WTMECPRP" if release == RELEASES[0] else "WTMEC2YR"
        full_release = survey_design.loc[
            survey_design["release_id"].eq(release)
            & survey_design[["SDMVSTRA", "SDMVPSU", "survey_weight"]]
            .notna()
            .all(axis=1)
            & survey_design["survey_weight"].gt(0)
        ].copy()
        full_clusters = full_release[["SDMVSTRA", "SDMVPSU"]].drop_duplicates()
        design_df = int(len(full_clusters) - full_clusters["SDMVSTRA"].nunique())
        for organ in ORGANS:
            group = analysis.loc[
                analysis["release_id"].eq(release)
                & analysis["organ_system"].eq(organ)
            ].copy()
            common = {
                "release_id": release,
                "input_variant": NHANES_PRIMARY_VARIANT,
                "organ_system": organ,
                "analysis": "survey_weighted_logistic_questionnaire_association",
                "metric": "odds_ratio_per_1sd_corrected_gap",
                "weight_variable": weight_variable,
                "strata_variable": "SDMVSTRA",
                "psu_variable": "SDMVPSU",
                "covariates": "intercept + corrected_age_gap_mean5_per_release_survey_weighted_sd + chronological_age_survey_weighted_centered_per10y + sex_male",
                "inference": "survey_domain_pseudolikelihood_logit_taylor_psu_within_strata_lonely_psu_centered_t_design_df",
                "survey_domain": "MEC_examined_integer_age_40_73_with_full_release_design_retained",
                "full_design_n": len(full_release),
                "full_design_strata": int(full_clusters["SDMVSTRA"].nunique()),
                "full_design_psus": len(full_clusters),
                "design_df": design_df,
                "reference_distribution": "t_design_df",
                "input_label_overlap_count": 0,
                "interpretation": "association_not_incident_prediction_not_causal",
            }
            if organ == "immune":
                rows.append(
                    {
                        **common,
                        "estimability_status": "not_estimable_by_design",
                        "n": 0,
                        "cases": None,
                        "estimate": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "p_value": None,
                        "bh_q_value": None,
                    }
                )
                continue
            complete = group.loc[
                group[
                    [
                        "label",
                        weight_variable,
                        "SDMVSTRA",
                        "SDMVPSU",
                        "corrected_age_gap_mean5",
                        "chronological_age",
                        "sex_male",
                    ]
                ].notna().all(axis=1)
                & group[weight_variable].gt(0)
            ].copy()
            cases = int(complete["label"].sum()) if len(complete) else 0
            if len(complete) < 20:
                status = "insufficient_complete_cases"
            elif complete["label"].nunique() != 2:
                status = "single_outcome_class"
            else:
                status = "estimable"
            estimate = lower = upper = p_value = None
            if status == "estimable":
                complete_weight = complete[weight_variable].to_numpy(
                    dtype=np.float64
                )
                gap = complete["corrected_age_gap_mean5"].to_numpy(
                    dtype=np.float64
                )
                weight_total = float(complete_weight.sum())
                gap_mean = float(np.sum(complete_weight * gap) / weight_total)
                gap_sd = float(
                    np.sqrt(
                        np.sum(complete_weight * np.square(gap - gap_mean))
                        / weight_total
                    )
                )
                if not np.isfinite(gap_sd) or gap_sd <= 0:
                    status = "singular_design"
                else:
                    complete["gap_sd"] = (
                        complete["corrected_age_gap_mean5"] - gap_mean
                    ) / gap_sd
                    age_mean = float(
                        np.sum(
                            complete_weight
                            * complete["chronological_age"].to_numpy(dtype=np.float64)
                        )
                        / weight_total
                    )
                    complete["age_centered_per10"] = (
                        complete["chronological_age"]
                        - age_mean
                    ) / 10.0
                    design = np.column_stack(
                        [
                            np.ones(len(complete)),
                            complete["gap_sd"],
                            complete["age_centered_per10"],
                            complete["sex_male"],
                        ]
                    )
                    try:
                        coefficient, error, fitted_df = _survey_logistic_taylor(
                            complete["label"].to_numpy(dtype=np.float64),
                            design,
                            complete[weight_variable].to_numpy(dtype=np.float64),
                            complete["SDMVSTRA"].to_numpy(),
                            complete["SDMVPSU"].to_numpy(),
                            full_design_strata=full_release["SDMVSTRA"].to_numpy(),
                            full_design_psu=full_release["SDMVPSU"].to_numpy(),
                        )
                        if fitted_df != design_df:
                            raise ValueError("survey design df changed inside domain fit")
                        critical = float(t.ppf(0.975, design_df))
                        estimate = float(np.exp(coefficient[1]))
                        lower = float(
                            np.exp(coefficient[1] - critical * error[1])
                        )
                        upper = float(
                            np.exp(coefficient[1] + critical * error[1])
                        )
                        p_value = float(
                            2.0 * t.sf(abs(coefficient[1] / error[1]), design_df)
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        status = "singular_design"
            rows.append(
                {
                    **common,
                    "estimability_status": status,
                    "n": len(complete),
                    "cases": cases,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": p_value,
                    "bh_q_value": None,
                }
            )
    result = pd.DataFrame(rows)
    for release, indices in result.groupby("release_id").groups.items():
        valid = [index for index in indices if pd.notna(result.at[index, "p_value"])]
        if valid:
            result.loc[valid, "bh_q_value"] = benjamini_hochberg(
                result.loc[valid, "p_value"].to_numpy(dtype=np.float64),
                family_size=6,
            )
    return result


def _weighted_age_metric(
    metric: str, observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray
) -> float:
    from scipy.stats import rankdata

    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[valid], p[valid], w[valid]
    if len(y) < 3:
        raise ValueError("insufficient weighted age rows")
    w = w / w.sum()
    residual = p - y
    if metric == "mae":
        return float(np.sum(w * np.abs(residual)))
    if metric == "rmse":
        return float(np.sqrt(np.sum(w * np.square(residual))))
    if metric == "spearman":
        y, p = rankdata(y), rankdata(p)
    if metric in {"pearson", "spearman"}:
        y_centered = y - np.sum(w * y)
        p_centered = p - np.sum(w * p)
        denominator = np.sqrt(
            np.sum(w * y_centered**2) * np.sum(w * p_centered**2)
        )
        if denominator <= 0:
            raise ValueError("weighted correlation singular")
        return float(np.sum(w * y_centered * p_centered) / denominator)
    if metric == "r2":
        denominator = float(np.sum(w * np.square(y - np.sum(w * y))))
        if denominator <= 0:
            raise ValueError("weighted R2 singular")
        return float(1.0 - np.sum(w * np.square(y - p)) / denominator)
    design = np.column_stack([np.ones(len(p)), p])
    root = np.sqrt(w)
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("weighted calibration singular")
    coefficient = np.linalg.lstsq(
        design * root[:, None], y * root, rcond=None
    )[0]
    return float(coefficient[0] if metric == "calibration_intercept" else coefficient[1])


def _build_nhanes_survey_age_metrics(
    ensemble: pd.DataFrame, participants: pd.DataFrame
) -> pd.DataFrame:
    analysis = ensemble.loc[ensemble["cohort"].eq("NHANES")].merge(
        participants,
        on=["participant_id", "release_id"],
        how="inner",
        suffixes=("", "_participant"),
        validate="many_to_one",
    )
    rows = []
    for (release, organ), group in analysis.groupby(
        ["release_id", "organ_system"], sort=False
    ):
        weight_variable = "WTMECPRP" if release == RELEASES[0] else "WTMEC2YR"
        for metric in AGE_METRICS:
            try:
                estimate = _weighted_age_metric(
                    metric,
                    group["chronological_age"].to_numpy(dtype=np.float64),
                    group["predicted_age_raw_mean5"].to_numpy(dtype=np.float64),
                    group[weight_variable].to_numpy(dtype=np.float64),
                )
                status = "estimable"
            except ValueError:
                estimate = None
                status = "singular_design"
            rows.append(
                {
                    "release_id": release,
                    "input_variant": NHANES_PRIMARY_VARIANT,
                    "organ_system": organ,
                    "metric": metric,
                    "weight_variable": weight_variable,
                    "estimability_status": status,
                    "estimate": estimate,
                    "cross_release_pooled_weight": False,
                    "sensitivity_only": True,
                    "between_release_test_performed": False,
                    "interpretation": "release_specific_descriptive_sensitivity_no_heterogeneity_test",
                }
            )
    return pd.DataFrame(rows)


def _assert_exact_key_universe(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    expected: set[tuple[Any, ...]],
    label: str,
) -> None:
    observed = set(map(tuple, frame[list(columns)].itertuples(index=False, name=None)))
    if observed != expected or len(frame) != len(expected):
        raise AssertionError(
            f"{label} exact key universe mismatch: "
            f"rows={len(frame)}, expected={len(expected)}, "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )


def _validate_prediction_output_universes(
    per: pd.DataFrame,
    ensemble: pd.DataFrame,
    *,
    mimic_participants: pd.DataFrame,
    nhanes_participants: pd.DataFrame,
) -> None:
    per_key = [
        "cohort",
        "release_id",
        "participant_id",
        "input_variant",
        "organ_system",
        "checkpoint_outer_fold",
    ]
    ensemble_key = per_key[:-1]
    if per.duplicated(per_key).any() or ensemble.duplicated(ensemble_key).any():
        raise AssertionError("prediction output universe has duplicate keys")
    if mimic_participants["subject_id"].duplicated().any() or nhanes_participants[
        ["release_id", "participant_id"]
    ].duplicated().any():
        raise AssertionError("sealed participant universe has duplicate identities")

    expected_mimic = mimic_participants[
        ["subject_id", "chronological_age", "sex_male"]
    ].copy()
    expected_mimic["participant_id"] = expected_mimic["subject_id"].astype(str)
    expected_mimic = expected_mimic.drop(columns="subject_id")
    expected_nhanes = nhanes_participants[
        ["release_id", "participant_id", "chronological_age", "sex_male"]
    ].copy()
    expected_nhanes["release_id"] = expected_nhanes["release_id"].astype(str)
    expected_nhanes["participant_id"] = expected_nhanes["participant_id"].astype(str)

    per_mimic = per.loc[per["cohort"].eq("MIMIC_IV_3_1")].copy()
    per_nhanes = per.loc[per["cohort"].eq("NHANES")].copy()
    ensemble_mimic = ensemble.loc[ensemble["cohort"].eq("MIMIC_IV_3_1")].copy()
    ensemble_nhanes = ensemble.loc[ensemble["cohort"].eq("NHANES")].copy()
    if len(per_mimic) + len(per_nhanes) != len(per) or len(
        ensemble_mimic
    ) + len(ensemble_nhanes) != len(ensemble):
        raise AssertionError("prediction output contains an unknown cohort")
    if not per_mimic["release_id"].isna().all() or not ensemble_mimic[
        "release_id"
    ].isna().all():
        raise AssertionError("MIMIC prediction release_id must be null")
    if per_nhanes["release_id"].isna().any() or ensemble_nhanes[
        "release_id"
    ].isna().any():
        raise AssertionError("NHANES prediction release_id must be nonnull")
    for frame in (per_mimic, ensemble_mimic):
        if not frame["input_variant"].isin(MIMIC_VARIANTS).all():
            raise AssertionError("MIMIC prediction contains a non-MIMIC input variant")
    for frame in (per_nhanes, ensemble_nhanes):
        if not frame["input_variant"].eq(NHANES_PRIMARY_VARIANT).all():
            raise AssertionError("NHANES prediction contains a non-primary input variant")
    for frame in (per, ensemble):
        if not frame["organ_system"].isin(ORGANS).all():
            raise AssertionError("prediction output contains an unknown organ")
    if not per["checkpoint_outer_fold"].isin(range(5)).all():
        raise AssertionError("per-checkpoint output contains an unknown fold")

    def assert_demographics(
        frame: pd.DataFrame,
        expected: pd.DataFrame,
        identity: list[str],
        label: str,
    ) -> None:
        observed = frame[
            [*identity, "chronological_age", "sex_male"]
        ].drop_duplicates()
        if observed.duplicated(identity).any():
            raise AssertionError(f"{label} prediction demographics vary within identity")
        compared = expected.merge(
            observed,
            on=identity,
            how="outer",
            validate="one_to_one",
            suffixes=("_expected", "_observed"),
            indicator=True,
        )
        if not compared["_merge"].eq("both").all():
            raise AssertionError(f"{label} prediction identity universe differs from cache")
        if not np.allclose(
            compared["chronological_age_expected"].to_numpy(dtype=np.float64),
            compared["chronological_age_observed"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        ) or not np.array_equal(
            compared["sex_male_expected"].to_numpy(dtype=np.int8),
            compared["sex_male_observed"].to_numpy(dtype=np.int8),
        ):
            raise AssertionError(f"{label} prediction demographics differ from cache")

    assert_demographics(
        per_mimic,
        expected_mimic,
        ["participant_id"],
        "MIMIC per-checkpoint",
    )
    assert_demographics(
        per_nhanes,
        expected_nhanes,
        ["release_id", "participant_id"],
        "NHANES per-checkpoint",
    )
    assert_demographics(
        ensemble_mimic,
        expected_mimic,
        ["participant_id"],
        "MIMIC ensemble",
    )
    assert_demographics(
        ensemble_nhanes,
        expected_nhanes,
        ["release_id", "participant_id"],
        "NHANES ensemble",
    )

    if len(per_mimic) != len(expected_mimic) * len(MIMIC_VARIANTS) * len(ORGANS) * 5:
        raise AssertionError("MIMIC per-checkpoint row cardinality mismatch")
    if len(per_nhanes) != len(expected_nhanes) * len(ORGANS) * 5:
        raise AssertionError("NHANES per-checkpoint row cardinality mismatch")
    for frame, identity, variants, label in (
        (per_mimic, ["participant_id"], MIMIC_VARIANTS, "MIMIC"),
        (
            per_nhanes,
            ["release_id", "participant_id"],
            [NHANES_PRIMARY_VARIANT],
            "NHANES",
        ),
    ):
        fold_groups = frame.groupby(
            [*identity, "input_variant", "organ_system"],
            dropna=False,
            sort=False,
        )["checkpoint_outer_fold"].agg(["count", "nunique", "min", "max", "sum"])
        if not (
            fold_groups["count"].eq(5).all()
            and fold_groups["nunique"].eq(5).all()
            and fold_groups["min"].eq(0).all()
            and fold_groups["max"].eq(4).all()
            and fold_groups["sum"].eq(10).all()
        ):
            raise AssertionError(f"{label} per-checkpoint fold Cartesian coverage mismatch")
        participant_counts = frame.groupby(identity, dropna=False, sort=False).size()
        if not participant_counts.eq(len(variants) * len(ORGANS) * 5).all():
            raise AssertionError(f"{label} per-checkpoint participant coverage mismatch")

    if len(ensemble_mimic) != len(expected_mimic) * len(MIMIC_VARIANTS) * len(ORGANS):
        raise AssertionError("MIMIC ensemble row cardinality mismatch")
    if len(ensemble_nhanes) != len(expected_nhanes) * len(ORGANS):
        raise AssertionError("NHANES ensemble row cardinality mismatch")
    if not ensemble_mimic.groupby("participant_id", sort=False).size().eq(
        len(MIMIC_VARIANTS) * len(ORGANS)
    ).all() or not ensemble_nhanes.groupby(
        ["release_id", "participant_id"], sort=False
    ).size().eq(len(ORGANS)).all():
        raise AssertionError("ensemble participant Cartesian coverage mismatch")


def _validate_upstream_output_universes(
    tables: Mapping[str, pd.DataFrame],
    *,
    mimic_participants: pd.DataFrame,
    nhanes_participants: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> None:
    """Bind every output-key universe to the sealed upstream participant universe."""
    if mimic_participants.empty or nhanes_participants.empty:
        raise AssertionError("sealed participant universe is empty")
    _validate_prediction_output_universes(
        tables["per_checkpoint_predictions"],
        tables["ensemble_predictions"],
        mimic_participants=mimic_participants,
        nhanes_participants=nhanes_participants,
    )

    zero_expected: set[tuple[Any, ...]] = set()
    summaries = [*ORGANS, "pooled_rows", "macro_organ_mean"]
    estimators = ["fold0", "fold1", "fold2", "fold3", "fold4", "mean5"]
    mimic_populations = [
        "mimic_primary_common_support_40_73_integer_year",
        "mimic_sensitivity_adult_nontopcoded_integer_year",
        "mimic_sensitivity_common_support_age_minus_0_5_year",
        "mimic_sensitivity_common_support_age_plus_0_5_year",
    ]
    for values in itertools.product(
        MIMIC_VARIANTS, mimic_populations, summaries, estimators, AGE_METRICS
    ):
        zero_expected.add(("MIMIC_IV_3_1", "MIMIC_IV_3_1", *values))
    for release, summary, estimator, metric in itertools.product(
        RELEASES, summaries, estimators, AGE_METRICS
    ):
        zero_expected.add(
            (
                "NHANES",
                release,
                NHANES_PRIMARY_VARIANT,
                "nhanes_primary_examined_40_73_exact",
                summary,
                estimator,
                metric,
            )
        )
    _assert_exact_key_universe(
        tables["zero_shot_age_metrics"],
        columns=[
            "cohort",
            "release_id",
            "input_variant",
            "population",
            "organ_system",
            "estimator",
            "metric",
        ],
        expected=zero_expected,
        label="zero-shot age metrics",
    )

    vocabulary_expected: set[tuple[Any, ...]] = set()
    for variant in MIMIC_VARIANTS:
        for code_system, kind, token_class in itertools.product(
            ("ICD9", "ICD10"),
            ("code", "text"),
            ("exact_frozen_vocab", "deterministic_extension"),
        ):
            vocabulary_expected.add(
                ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, kind, "hospital_icd", code_system, token_class, "none")
            )
        for code_system, kind in itertools.product(("ICD9", "ICD10"), ("code", "text")):
            vocabulary_expected.add(
                ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, kind, "hospital_icd", code_system, "invalid_dropped", "invalid_or_unsupported_code")
            )
        for kind in ("code", "text"):
            vocabulary_expected.add(
                ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, kind, "hospital_icd", "not_applicable", "invalid_dropped", "invalid_or_unsupported_code")
            )
            vocabulary_expected.add(
                ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, kind, "padding", "not_applicable", "zero_event", "padding")
            )
        if variant != MIMIC_RX_SENSITIVITY:
            for kind in ("code", "text"):
                vocabulary_expected.add(
                    ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, kind, "inpatient_prescription_proxy", "INPATIENT_RX", "zero_event", "source_forbidden_for_input_variant")
                )
        else:
            vocabulary_expected.add(
                ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, "code", "inpatient_prescription_proxy", "INPATIENT_RX", "zero_event", "unmapped_inpatient_prescription")
            )
            for token_class in ("exact_frozen_vocab", "deterministic_extension"):
                vocabulary_expected.add(
                    ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, "text", "inpatient_prescription_proxy", "INPATIENT_RX", token_class, "none")
                )
            vocabulary_expected.add(
                ("MIMIC_IV_3_1", "MIMIC_IV_3_1", variant, "text", "inpatient_prescription_proxy", "INPATIENT_RX", "invalid_dropped", "invalid_or_missing_text")
            )
    for release, kind in itertools.product(RELEASES, ("code", "text")):
        vocabulary_expected.add(
            ("NHANES", release, NHANES_PRIMARY_VARIANT, kind, "padding", "not_applicable", "zero_event", "padding")
        )
    _assert_exact_key_universe(
        tables["external_vocabulary_coverage"],
        columns=[
            "cohort",
            "source_partition",
            "input_variant",
            "vocabulary_kind",
            "modality",
            "code_system",
            "token_class",
            "reason",
        ],
        expected=vocabulary_expected,
        label="external vocabulary coverage",
    )

    flow_expected = set(itertools.product(MIMIC_VARIANTS, range(1, 13)))
    _assert_exact_key_universe(
        tables["mimic_cohort_flow"],
        columns=["input_variant", "step_order"],
        expected=flow_expected,
        label="MIMIC cohort flow",
    )
    if not all(
        group.sort_values("step_order")["step_name"].tolist() == list(MIMIC_FLOW_STEPS)
        for _, group in tables["mimic_cohort_flow"].groupby("input_variant")
    ):
        raise AssertionError("MIMIC cohort flow ordered names changed")
    _assert_exact_key_universe(
        tables["mimic_feature_coverage"],
        columns=["input_variant", "slot_index"],
        expected=set(itertools.product(MIMIC_VARIANTS, range(78))),
        label="MIMIC feature coverage",
    )
    _assert_exact_key_universe(
        tables["nhanes_feature_coverage"],
        columns=["release_id", "input_variant", "slot_index"],
        expected={
            (release, NHANES_PRIMARY_VARIANT, slot)
            for release, slot in itertools.product(RELEASES, range(78))
        },
        label="NHANES feature coverage",
    )

    incident = tables["mimic_incident_endpoints"]
    recurrent = tables["mimic_recurrent_endpoints"]
    endpoint_key = ["subject_id", "input_variant", "organ_system", "horizon_months"]
    endpoints = pd.concat([incident, recurrent], ignore_index=True)
    if endpoints.duplicated(endpoint_key).any():
        raise AssertionError("incident/recurrent union duplicates an endpoint key")
    eligible_ids = set(
        eligibility.loc[
            eligibility["eligibility_status"].eq("endpoint_eligible"), "subject_id"
        ].astype(str)
    )
    if set(endpoints["subject_id"].astype(str)) != eligible_ids:
        raise AssertionError("endpoint subject universe differs from eligibility receipt")
    if len(endpoints) != len(eligible_ids) * 3 * 7 * 2:
        raise AssertionError("endpoint union closed row cardinality mismatch")
    if not endpoints.groupby("subject_id", sort=False).size().eq(3 * 7 * 2).all():
        raise AssertionError("endpoint subject Cartesian coverage mismatch")

    product_specs = (
        (
            "mimic_predictive_metrics",
            ["input_variant", "organ_system", "horizon_months", "hash_fold", "model", "metric"],
            itertools.product(
                MIMIC_VARIANTS,
                ORGANS,
                (6, 12),
                range(5),
                ("age_sex", "age_sex_corrected_gap"),
                (
                    "harrell_c",
                    "delta_harrell_c",
                    "ipcw_auc",
                    "ipcw_brier",
                    "ipcw_linear_calibration_intercept",
                    "ipcw_linear_calibration_slope",
                ),
            ),
        ),
        (
            "mimic_full_cohort_associations",
            ["input_variant", "organ_system", "horizon_months", "exposure"],
            itertools.product(
                MIMIC_VARIANTS,
                ORGANS,
                (6, 12),
                ("corrected_age_gap_mean5_per_incident_eligible_analysis_population_sd",),
            ),
        ),
        (
            "mimic_recurrent_sensitivity_metrics",
            ["input_variant", "organ_system", "horizon_months", "metric"],
            itertools.product(
                MIMIC_VARIANTS,
                ORGANS,
                (6, 12),
                (
                    "cause_specific_hazard_ratio",
                    "target_cumulative_incidence",
                    "death_cumulative_incidence",
                ),
            ),
        ),
        (
            "nhanes_questionnaire_associations",
            ["release_id", "organ_system"],
            itertools.product(RELEASES, ORGANS),
        ),
        (
            "nhanes_release_survey_sensitivity",
            ["release_id", "organ_system", "metric"],
            itertools.product(RELEASES, ORGANS, AGE_METRICS),
        ),
    )
    for table_id, columns, values in product_specs:
        _assert_exact_key_universe(
            tables[table_id],
            columns=columns,
            expected=set(values),
            label=table_id,
        )

    local_expected: set[tuple[Any, ...]] = set()
    for release, organ, fold in itertools.product(RELEASES, ORGANS, range(5)):
        for term in ("intercept", "predicted_age_raw_mean5", "sex_male"):
            local_expected.add(
                (release, NHANES_PRIMARY_VARIANT, organ, fold, "coefficient", term, "not_applicable_coefficient_row")
            )
        for metric in AGE_METRICS:
            local_expected.add(
                (release, NHANES_PRIMARY_VARIANT, organ, fold, "heldout_metric", "not_applicable_metric_row", metric)
            )
    _assert_exact_key_universe(
        tables["nhanes_local_recalibration_sensitivity"],
        columns=[
            "release_id",
            "input_variant",
            "organ_system",
            "hash_fold",
            "record_type",
            "term",
            "metric",
        ],
        expected=local_expected,
        label="NHANES local recalibration sensitivity",
    )

    curves = tables["mimic_incident_competing_risk_curves"]
    curve_names = (
        "aalen_johansen_target_cif",
        "aalen_johansen_death_cif",
        "kaplan_meier_target_free_survival_cause_specific",
    )
    expected_curve_groups = set(itertools.product(MIMIC_VARIANTS, ORGANS, (6, 12), curve_names))
    observed_curve_groups = set(
        curves[["input_variant", "organ_system", "horizon_months", "curve"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_curve_groups != expected_curve_groups:
        raise AssertionError("competing-risk curve group universe mismatch")
    for keys, group in curves.groupby(
        ["input_variant", "organ_system", "horizon_months", "curve"], sort=False
    ):
        source = incident.loc[
            incident["input_variant"].eq(keys[0])
            & incident["organ_system"].eq(keys[1])
            & incident["horizon_months"].eq(keys[2])
        ]
        expected_times = np.unique(
            np.concatenate(
                [
                    np.array([0.0]),
                    source["observed_time_days"].to_numpy(dtype=np.float64),
                ]
            )
        )
        observed_times = np.sort(group["time_days"].to_numpy(dtype=np.float64))
        if not np.array_equal(observed_times, expected_times):
            raise AssertionError(f"curve time universe mismatch: {keys}")


def compute_all_twenty_tables(
    *,
    artifact_dir: Path,
    cache: Mapping[str, Path],
    sealed: Mapping[str, Path],
    endpoints: Mapping[str, Path],
    contract: Mapping[str, Any],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, pd.DataFrame]:
    if bootstrap_seed != 7_102_026 or bootstrap_replicates != 200:
        raise AssertionError("formal bootstrap seed/replicate mismatch")
    prediction_receipt = json.loads(Path(sealed["receipt"]).read_text(encoding="utf-8"))
    outcome_receipt = json.loads(Path(endpoints["receipt"]).read_text(encoding="utf-8"))
    if prediction_receipt.get("status") != "PREDICTIONS_SEALED_BEFORE_OUTCOME_READ":
        raise AssertionError("prediction receipt is not sealed")
    if outcome_receipt.get("status") != "OUTCOMES_READ_ONLY_AFTER_PREDICTION_SEAL":
        raise AssertionError("outcomes were not read after the prediction seal")
    if outcome_receipt["prediction_seal_sha256"] != sha256_file(Path(sealed["receipt"])):
        raise AssertionError("outcome receipt does not bind the prediction seal")
    if outcome_receipt["eligibility_sha256"] != sha256_file(
        Path(endpoints["eligibility"])
    ):
        raise AssertionError("outcome receipt does not bind subject eligibility")
    cache_dir = Path(cache["directory"])
    manifest = json.loads(Path(cache["manifest"]).read_text(encoding="utf-8"))
    if (
        manifest.get("questionnaire_files_read") != []
        or manifest.get("postindex_diagnosis_rows_read_into_cache") != 0
        or manifest.get("postindex_diagnosis_rows_materialized_before_prediction_seal") != 0
        or manifest.get("diagnosis_source_streamed_with_preindex_hadm_filter") is not True
        or manifest.get("diagnosis_canonicalization_after_preindex_hadm_filter_only") is not True
        or manifest.get("postindex_observation_or_death_fields_read") != []
    ):
        raise AssertionError("label-blind cache manifest reports outcome leakage")
    per_checkpoint = pd.read_parquet(sealed["per_checkpoint"])
    ensemble = pd.read_parquet(sealed["ensemble"])
    incident = pd.read_parquet(endpoints["incident"])
    recurrent = pd.read_parquet(endpoints["recurrent"])
    questionnaire = pd.read_parquet(endpoints["questionnaire"])
    eligibility = pd.read_parquet(endpoints["eligibility"])
    mimic_participants = pd.read_parquet(cache_dir / "mimic_participants.parquet")
    participants = pd.read_parquet(cache_dir / "nhanes_participants.parquet")
    survey_design = pd.read_parquet(
        cache_dir / "nhanes_survey_design_universe.parquet"
    )
    density = pd.read_parquet(cache_dir / "mimic_participant_density_label_blind.parquet")
    nhanes_qc = pd.read_csv(cache_dir / "nhanes_release_slot_qc.csv")
    validate_mimic_endpoint_rows(incident)
    validate_mimic_endpoint_rows(recurrent)
    tables = {
        "per_checkpoint_predictions": per_checkpoint,
        "ensemble_predictions": ensemble,
        "external_vocabulary_coverage": _build_vocabulary_coverage(cache_dir, sealed),
        "zero_shot_age_metrics": _build_zero_shot_age_metrics(
            per_checkpoint,
            ensemble,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "mimic_cohort_flow": _build_mimic_flow(
            manifest, incident, recurrent, eligibility
        ),
        "mimic_feature_coverage": pd.read_parquet(
            cache_dir / "mimic_feature_coverage.parquet"
        ),
        "mimic_incident_endpoints": incident,
        "mimic_recurrent_endpoints": recurrent,
        "mimic_predictive_metrics": _build_mimic_predictive_metrics(
            incident,
            ensemble,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "mimic_full_cohort_associations": _build_mimic_associations(
            incident, ensemble
        ),
        "mimic_incident_competing_risk_curves": _build_mimic_curves(
            incident,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "mimic_recurrent_sensitivity_metrics": _build_recurrent_metrics(
            recurrent,
            ensemble,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        ),
        "mimic_density_diagnostics": build_mimic_density_summary(
            density, ensemble
        ),
        "mimic_density_adjusted_risk_sensitivity": build_density_adjusted_risk_sensitivity(
            incident=incident,
            ensemble=ensemble,
            density=density,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            months_to_days=months_to_days,
            cause_specific_cif_predictions=_cause_specific_cif_predictions,
            harrell_c=harrell_c,
            ipcw_binary_weights=ipcw_binary_weights,
            weighted_auc=weighted_auc,
            weighted_brier=weighted_brier,
            weighted_linear_calibration=_weighted_linear_calibration,
            global_subject_bootstrap_plan=_global_subject_bootstrap_plan,
            conditional_bootstrap_values=_conditional_bootstrap_prediction_values,
        ),
        "mimic_cox_ph_diagnostics": build_complete_mimic_ph_diagnostics(
            root=Path(__file__).resolve().parents[4],
            incident=incident,
            ensemble=ensemble,
            density=density,
        ),
        "nhanes_feature_coverage": pd.read_parquet(
            cache_dir / "nhanes_feature_coverage.parquet"
        ),
        "nhanes_release_slot_qc": nhanes_qc,
        "nhanes_questionnaire_associations": _build_nhanes_questionnaire_associations(
            ensemble, participants, questionnaire, survey_design
        ),
        "nhanes_local_recalibration_sensitivity": _build_nhanes_local_calibration(
            ensemble, participants
        ),
        "nhanes_release_survey_sensitivity": _build_nhanes_survey_age_metrics(
            ensemble, participants
        ),
    }
    if list(tables) != list(OUTPUT_TABLE_PATHS):
        raise AssertionError("formal executor did not produce the exact ordered 20 tables")
    _validate_upstream_output_universes(
        tables,
        mimic_participants=mimic_participants,
        nhanes_participants=participants,
        eligibility=eligibility,
    )
    if len(tables["nhanes_release_slot_qc"]) != 156:
        raise AssertionError("NHANES release-slot QC cardinality mismatch")
    if len(tables["mimic_density_diagnostics"]) != 693:
        raise AssertionError("MIMIC density diagnostics cardinality mismatch")
    if len(tables["mimic_density_adjusted_risk_sensitivity"]) != 2520:
        raise AssertionError("MIMIC density-adjusted risk sensitivity cardinality mismatch")
    if tables["nhanes_release_slot_qc"]["qc_status"].eq(
        "fail_abort_before_inference"
    ).any():
        raise AssertionError("a failed NHANES QC row reached formal tables")
    ph = tables["mimic_cox_ph_diagnostics"]
    ph_key = [
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
    if len(ph) != 17_136 or ph.duplicated(ph_key).any() or not ph["survival_core_sha256"].eq(
        SURVIVAL_CORE_SHA256
    ).all():
        raise AssertionError("complete MIMIC PH diagnostics are not byte-bound/unique")
    return tables


def _coerce_schema_dtype(series: pd.Series, dtype: str, nullable: bool) -> pd.Series:
    if dtype == "string":
        result = series.astype("string")
    elif dtype.startswith("datetime64"):
        result = pd.to_datetime(series, errors="raise").astype("datetime64[ns]")
    elif dtype == "bool":
        if nullable:
            result = series.astype("boolean")
        else:
            if series.isna().any():
                raise AssertionError("nonnullable bool contains missing value")
            result = series.astype(bool)
    elif dtype.startswith("int"):
        if nullable:
            bits = dtype[3:]
            result = series.astype(f"Int{bits}")
        else:
            if series.isna().any():
                raise AssertionError("nonnullable integer contains missing value")
            numeric = pd.to_numeric(series, errors="raise")
            if not np.equal(numeric, np.floor(numeric)).all():
                raise AssertionError("integer schema column has fractional value")
            result = numeric.astype(dtype)
    elif dtype.startswith("float"):
        result = pd.to_numeric(series, errors="raise").astype(dtype)
    else:
        raise AssertionError(f"unsupported closed schema dtype: {dtype}")
    if not nullable and result.isna().any():
        raise AssertionError("nonnullable schema column contains missing value")
    return result


def _validate_table_against_schema(
    table_id: str, frame: pd.DataFrame, schema: Mapping[str, Any]
) -> pd.DataFrame:
    expected_columns = [column["name"] for column in schema["columns"]]
    if list(frame.columns) != expected_columns:
        missing = [name for name in expected_columns if name not in frame.columns]
        extra = [name for name in frame.columns if name not in expected_columns]
        if missing or extra:
            raise AssertionError(
                f"{table_id} columns mismatch; missing={missing}, extra={extra}"
            )
        frame = frame[expected_columns].copy()
    else:
        frame = frame.copy()
    for column in schema["columns"]:
        name = column["name"]
        frame[name] = _coerce_schema_dtype(
            frame[name], str(column["dtype"]), bool(column["nullable"])
        )
        enum = column["enum"]
        if enum is not None:
            observed = set(frame[name].dropna().tolist())
            if not observed.issubset(set(enum)):
                raise AssertionError(
                    f"{table_id}/{name} value outside closed enum: {observed-set(enum)}"
                )
    key = list(schema["key"])
    if frame.duplicated(key).any():
        raise AssertionError(f"{table_id} duplicate key")
    if "estimability_status" in frame.columns and "estimate" in frame.columns:
        invalid_status = set(frame["estimability_status"]).difference(ESTIMABILITY_STATUSES)
        if invalid_status:
            raise AssertionError(f"{table_id} invalid estimability status")
        bad_null = frame["estimability_status"].ne("estimable") & frame["estimate"].notna()
        if bad_null.any():
            raise AssertionError(f"{table_id} non-estimable row has a numeric estimate")
        missing_estimate = frame["estimability_status"].eq("estimable") & frame[
            "estimate"
        ].isna()
        if missing_estimate.any():
            raise AssertionError(f"{table_id} estimable row has a null estimate")
    if "estimability_status" in frame.columns and "hazard_ratio" in frame.columns:
        if (
            frame["estimability_status"].eq("estimable")
            & frame["hazard_ratio"].isna()
        ).any():
            raise AssertionError(f"{table_id} estimable HR row is null")
    if "estimability_status" in frame.columns and {
        "ci_lower",
        "ci_upper",
    }.issubset(frame.columns):
        point_only = pd.Series(False, index=frame.index)
        if "uncertainty_scope" in frame.columns:
            point_only = frame["uncertainty_scope"].astype(str).str.startswith(
                "point_"
            )
        missing_ci = frame["estimability_status"].eq("estimable") & ~point_only & (
            frame["ci_lower"].isna() | frame["ci_upper"].isna()
        )
        if missing_ci.any():
            raise AssertionError(f"{table_id} estimable row has null confidence interval")
    return frame.sort_values(key, kind="mergesort").reset_index(drop=True)


def _schema_numeric_tolerance(column: Mapping[str, Any]) -> float:
    tolerance = column["tolerance"]
    if tolerance == "exact":
        return 0.0
    if not isinstance(tolerance, str):
        raise AssertionError(f"numeric schema column lacks tolerance: {column['name']}")
    try:
        value = float(tolerance)
    except ValueError as exc:
        raise AssertionError(f"invalid numeric tolerance: {tolerance}") from exc
    if not np.isfinite(value) or value < 0:
        raise AssertionError(f"invalid numeric tolerance: {tolerance}")
    return value


def _assert_close(
    observed: pd.Series,
    expected: pd.Series | np.ndarray,
    *,
    tolerance: float,
    label: str,
) -> None:
    left = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=0.0, atol=tolerance, equal_nan=True
    ):
        raise AssertionError(f"formula/tolerance mismatch: {label}")


def _validate_formula_crosshash_coverage(
    validated: Mapping[str, pd.DataFrame], schema_by_id: Mapping[str, Any]
) -> None:
    for table_id, schema in schema_by_id.items():
        for column in schema["columns"]:
            if not isinstance(column["formula"], str) or not column["formula"].strip():
                raise AssertionError(f"{table_id}/{column['name']} lacks a formula")
            if str(column["dtype"]).startswith("float"):
                _schema_numeric_tolerance(column)

    per = validated["per_checkpoint_predictions"]
    per_schema = {
        column["name"]: column
        for column in schema_by_id["per_checkpoint_predictions"]["columns"]
    }
    _assert_close(
        per["raw_age_gap"],
        per["predicted_age_raw"] - per["chronological_age"],
        tolerance=_schema_numeric_tolerance(per_schema["raw_age_gap"]),
        label="per_checkpoint_predictions.raw_age_gap",
    )
    _assert_close(
        per["corrected_age_gap"],
        per["raw_age_gap"]
        - (
            per["correction_intercept"]
            + per["correction_slope"] * per["chronological_age"]
        ),
        tolerance=_schema_numeric_tolerance(per_schema["corrected_age_gap"]),
        label="per_checkpoint_predictions.corrected_age_gap",
    )
    if per["external_cache_manifest_sha256"].nunique() != 1:
        raise AssertionError("prediction rows do not bind one label-blind cache manifest")
    if per.groupby("checkpoint_outer_fold")["checkpoint_sha256"].nunique().ne(1).any():
        raise AssertionError("checkpoint fold maps to multiple checkpoint hashes")
    if per.groupby("checkpoint_outer_fold")["checkpoint_sha256"].first().nunique() != 5:
        raise AssertionError("five checkpoint folds do not bind five unique checkpoints")

    key = ["cohort", "release_id", "participant_id", "input_variant", "organ_system"]
    aggregate = per.groupby(key, dropna=False, sort=False).agg(
        predicted_age_raw_mean5=("predicted_age_raw", "mean"),
        predicted_age_raw_sd5=("predicted_age_raw", "std"),
        raw_age_gap_mean5=("raw_age_gap", "mean"),
        corrected_age_gap_mean5=("corrected_age_gap", "mean"),
        risk_score_mean5=("risk_score", "mean"),
    ).reset_index()
    ensemble = validated["ensemble_predictions"].merge(
        aggregate,
        on=key,
        how="outer",
        validate="one_to_one",
        suffixes=("", "_formula"),
        indicator=True,
    )
    if not ensemble["_merge"].eq("both").all():
        raise AssertionError("ensemble/per-checkpoint key crosshash mismatch")
    ensemble_schema = {
        column["name"]: column
        for column in schema_by_id["ensemble_predictions"]["columns"]
    }
    for column in (
        "predicted_age_raw_mean5",
        "predicted_age_raw_sd5",
        "raw_age_gap_mean5",
        "corrected_age_gap_mean5",
        "risk_score_mean5",
    ):
        _assert_close(
            ensemble[column],
            ensemble[f"{column}_formula"],
            tolerance=_schema_numeric_tolerance(ensemble_schema[column]),
            label=f"ensemble_predictions.{column}",
        )

    for table_id, grouping, family_size in (
        (
            "mimic_full_cohort_associations",
            ["input_variant", "horizon_months"],
            7,
        ),
        ("nhanes_questionnaire_associations", ["release_id"], 6),
    ):
        table = validated[table_id]
        for _, group in table.groupby(grouping, sort=False):
            valid = group["p_value"].notna()
            if valid.any():
                expected_q = benjamini_hochberg(
                    group.loc[valid, "p_value"].to_numpy(dtype=np.float64),
                    family_size=family_size,
                )
                _assert_close(
                    group.loc[valid, "bh_q_value"],
                    expected_q,
                    tolerance=1e-12,
                    label=f"{table_id}.fixed_family_bh",
                )
            if group.loc[~valid, "bh_q_value"].notna().any():
                raise AssertionError(f"{table_id} null p-value has nonnull BH q-value")

    zero = validated["zero_shot_age_metrics"]
    if not zero["no_external_fit"].all():
        raise AssertionError("an age-transfer row is not frozen zero-shot")
    expected_primary = zero["estimator"].eq("mean5") & (
        (
            zero["cohort"].eq("MIMIC_IV_3_1")
            & zero["input_variant"].eq(MIMIC_PRIMARY_VARIANT)
            & zero["population"].eq(
                "mimic_primary_common_support_40_73_integer_year"
            )
        )
        | (
            zero["cohort"].eq("NHANES")
            & zero["input_variant"].eq(NHANES_PRIMARY_VARIANT)
            & zero["population"].eq("nhanes_primary_examined_40_73_exact")
        )
    )
    if not zero["raw_zero_shot_primary"].astype(bool).eq(expected_primary).all():
        raise AssertionError("raw zero-shot primary/sensitivity flags are incorrect")
    if validated["nhanes_local_recalibration_sensitivity"]["zero_shot_primary"].any():
        raise AssertionError("NHANES external-only recalibration entered the primary path")
    if not validated["nhanes_questionnaire_associations"][
        "input_label_overlap_count"
    ].eq(0).all():
        raise AssertionError("NHANES questionnaire labels overlap model inputs")

    for table_id, group_columns in (
        ("mimic_feature_coverage", ["input_variant"]),
        ("nhanes_feature_coverage", ["release_id"]),
    ):
        for _, group in validated[table_id].groupby(group_columns, sort=False):
            if sorted(group["slot_index"].astype(int).tolist()) != list(range(78)):
                raise AssertionError(f"{table_id} lacks exact all-78 slot coverage")

    incident = validated["mimic_incident_endpoints"]
    recurrent = validated["mimic_recurrent_endpoints"]
    endpoint_key = ["subject_id", "input_variant", "organ_system", "horizon_months"]
    if set(map(tuple, incident[endpoint_key].to_numpy())).intersection(
        set(map(tuple, recurrent[endpoint_key].to_numpy()))
    ):
        raise AssertionError("incident and recurrent endpoint tables overlap")
    for table_id, table in (
        ("mimic_incident_endpoints", incident),
        ("mimic_recurrent_endpoints", recurrent),
    ):
        horizon_days = table["horizon_months"].map({6: 182.5, 12: 365.0}).astype(float)
        if (table["observed_time_days"] - horizon_days > 1e-9).any():
            raise AssertionError(f"{table_id} extends beyond an authorized horizon")
        if (table["subject_observation_end_days"] - table["death_ascertainment_cap_days"] > 1e-9).any():
            raise AssertionError(f"{table_id} observation exceeds frozen follow-up cap")
        for row in table.itertuples(index=False):
            expected = adjudicate_endpoint(
                target_days=(
                    None if pd.isna(row.event_time_days) else row.event_time_days
                ),
                death_days=(
                    None if pd.isna(row.death_time_days) else row.death_time_days
                ),
                observation_end_days=row.subject_observation_end_days,
                death_cap_days=row.death_ascertainment_cap_days,
                horizon_months=int(row.horizon_months),
            )
            for name in (
                "horizon_days",
                "observed_time_days",
                "event_indicator",
                "competing_death_indicator",
                "censor_reason",
                "observed_at_horizon",
            ):
                observed_value = getattr(row, name)
                expected_value = expected[name]
                if isinstance(expected_value, float):
                    if not np.isclose(
                        float(observed_value), expected_value, rtol=0.0, atol=1e-9
                    ):
                        raise AssertionError(
                            f"{table_id} endpoint adjudication mismatch: {name}"
                        )
                elif observed_value != expected_value:
                    raise AssertionError(
                        f"{table_id} endpoint adjudication mismatch: {name}"
                    )

    predictive = validated["mimic_predictive_metrics"]
    for metric in ("harrell_c", "ipcw_auc"):
        values = predictive.loc[
            predictive["metric"].eq(metric)
            & predictive["estimability_status"].eq("estimable"),
            "estimate",
        ]
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise AssertionError(f"predictive {metric} outside [0,1]")
    estimable_predictive = predictive["estimability_status"].eq("estimable")
    if (
        predictive.loc[estimable_predictive, "ipcw_n_observed_at_horizon"]
        + predictive.loc[estimable_predictive, "ipcw_n_censored_before_horizon"]
    ).ne(predictive.loc[estimable_predictive, "n_test"]).any():
        raise AssertionError("IPCW observed/censored diagnostic does not sum to test n")
    if predictive.loc[estimable_predictive, "ipcw_g_floor"].ne(0.05).any():
        raise AssertionError("IPCW positivity floor changed")
    if (
        predictive.loc[estimable_predictive, "ipcw_effective_sample_size"]
        > predictive.loc[estimable_predictive, "n_test"] + 1e-9
    ).any():
        raise AssertionError("IPCW effective sample size exceeds test n")

    curves = validated["mimic_incident_competing_risk_curves"]
    curve_key = ["input_variant", "organ_system", "horizon_months", "curve"]
    for keys, group in curves.groupby(curve_key, sort=False):
        group = group.sort_values("time_days", kind="mergesort")
        if not group["time_days"].is_monotonic_increasing:
            raise AssertionError(f"curve time order changed: {keys}")
        values = group["estimate"].dropna().to_numpy(dtype=np.float64)
        if np.any((values < -1e-12) | (values > 1 + 1e-12)):
            raise AssertionError(f"curve estimate outside [0,1]: {keys}")
        if str(keys[-1]).startswith("aalen_johansen") and np.any(
            np.diff(values) < -1e-12
        ):
            raise AssertionError(f"AJ CIF is not monotone: {keys}")
        if str(keys[-1]).startswith("kaplan_meier") and np.any(
            np.diff(values) > 1e-12
        ):
            raise AssertionError(f"KM survival is not monotone: {keys}")

    survey = validated["nhanes_questionnaire_associations"]
    if not survey["design_df"].eq(
        survey["full_design_psus"] - survey["full_design_strata"]
    ).all():
        raise AssertionError("NHANES design degrees of freedom mismatch")
    if not survey["reference_distribution"].eq("t_design_df").all():
        raise AssertionError("NHANES survey inference does not use design t/Wald")

    for table_id in (
        "zero_shot_age_metrics",
        "mimic_predictive_metrics",
        "mimic_incident_competing_risk_curves",
        "mimic_recurrent_sensitivity_metrics",
    ):
        frame = validated[table_id]
        if {"ci_lower", "ci_upper"}.issubset(frame.columns):
            complete_ci = frame[["ci_lower", "ci_upper"]].notna().all(axis=1)
            if (
                frame.loc[complete_ci, "ci_lower"]
                > frame.loc[complete_ci, "ci_upper"]
            ).any():
                raise AssertionError(f"{table_id} has reversed confidence bounds")


def _read_formal_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise AssertionError(f"unsupported formal table extension: {path}")


def validate_and_write_tables(
    *,
    stage: Path,
    tables: Mapping[str, pd.DataFrame],
    schemas: Mapping[str, Any],
    output_paths: Mapping[str, str],
) -> dict[str, Any]:
    if schemas.get("table_count") != 20:
        raise AssertionError("formal output schema does not bind exactly 20 tables")
    schema_by_id = {table["table_id"]: table for table in schemas["tables"]}
    if list(tables) != list(output_paths) or list(schema_by_id) != list(output_paths):
        raise AssertionError("formal table/schema/output order mismatch")
    validated: dict[str, pd.DataFrame] = {}
    for table_id, frame in tables.items():
        validated[table_id] = _validate_table_against_schema(
            table_id, frame, schema_by_id[table_id]
        )
    _validate_formula_crosshash_coverage(validated, schema_by_id)
    if len(validated["mimic_cohort_flow"]) != 3 * 12:
        raise AssertionError("MIMIC flow is not 3 variants times 12 exact steps")
    vocabulary = validated["external_vocabulary_coverage"]
    if len(vocabulary) != 60:
        raise AssertionError("external vocabulary coverage closed cardinality is not 60")
    if set(vocabulary["token_class"]) != {
        "exact_frozen_vocab",
        "deterministic_extension",
        "invalid_dropped",
        "zero_event",
    }:
        raise AssertionError("external vocabulary coverage lacks a required token class")
    if not {"ICD9", "ICD10", "INPATIENT_RX", "not_applicable"}.issubset(
        set(vocabulary["code_system"])
    ):
        raise AssertionError("external vocabulary coverage does not separate code systems")
    if len(validated["mimic_feature_coverage"]) != 3 * 78:
        raise AssertionError("MIMIC coverage is not 3 variants times 78 slots")
    if len(validated["nhanes_feature_coverage"]) != 2 * 78:
        raise AssertionError("NHANES coverage is not 2 releases times 78 slots")
    if len(validated["mimic_predictive_metrics"]) != 3 * 7 * 2 * 5 * 2 * 6:
        raise AssertionError("predictive metric closed cardinality mismatch")
    if len(validated["mimic_full_cohort_associations"]) != 3 * 7 * 2:
        raise AssertionError("MIMIC association closed cardinality mismatch")
    if len(validated["mimic_recurrent_sensitivity_metrics"]) != 3 * 7 * 2 * 3:
        raise AssertionError("MIMIC recurrent metric closed cardinality mismatch")
    if len(validated["nhanes_questionnaire_associations"]) != 2 * 7:
        raise AssertionError("NHANES questionnaire association cardinality mismatch")
    if len(validated["nhanes_local_recalibration_sensitivity"]) != 2 * 7 * 5 * 10:
        raise AssertionError("NHANES local calibration cardinality mismatch")
    if len(validated["nhanes_release_survey_sensitivity"]) != 2 * 7 * 7:
        raise AssertionError("NHANES survey age metric cardinality mismatch")
    per = validated["per_checkpoint_predictions"]
    fold_sets = per.groupby(
        ["cohort", "participant_id", "input_variant", "organ_system"], sort=False
    )["checkpoint_outer_fold"].agg(lambda value: sorted(value.tolist()))
    if not fold_sets.map(lambda value: value == [0, 1, 2, 3, 4]).all():
        raise AssertionError("per-checkpoint table lacks an exact five-checkpoint set")
    if set(validated["mimic_incident_endpoints"]["horizon_months"]) != {6, 12}:
        raise AssertionError("incident endpoint horizons are not exactly 6/12")
    if set(validated["mimic_recurrent_endpoints"]["horizon_months"]) != {6, 12}:
        raise AssertionError("recurrent endpoint horizons are not exactly 6/12")
    for table_id, frame in validated.items():
        path = stage / output_paths[table_id]
        if path.exists():
            raise FileExistsError(path)
        if path.suffix == ".parquet":
            _write_parquet(path, frame)
        elif path.suffix == ".csv":
            _write_csv(path, frame)
        else:
            raise AssertionError(f"unsupported formal table extension: {path}")
    reread_tables: dict[str, pd.DataFrame] = {}
    for table_id, relative_path in output_paths.items():
        path = stage / relative_path
        reread = _validate_table_against_schema(
            table_id, _read_formal_table(path), schema_by_id[table_id]
        )
        if len(reread) != len(validated[table_id]):
            raise AssertionError(f"{table_id} row count changed after serialization")
        reread_tables[table_id] = reread
    _validate_formula_crosshash_coverage(reread_tables, schema_by_id)
    table_records = []
    for table_id, relative_path in output_paths.items():
        path = stage / relative_path
        reread = reread_tables[table_id]
        schema_contract = [
            {
                "name": column["name"],
                "formula": column["formula"],
                "tolerance": column["tolerance"],
            }
            for column in schema_by_id[table_id]["columns"]
        ]
        table_records.append(
            {
                "table_id": table_id,
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": len(reread),
                "columns": list(reread.columns),
                "schema_formula_tolerance_sha256": hashlib.sha256(
                    json.dumps(
                        schema_contract,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest(),
            }
        )
    if len(table_records) != 20 or [row["table_id"] for row in table_records] != list(output_paths):
        raise AssertionError("formal table manifest is not the exact ordered 20-table set")
    manifest_path = stage / "FORMAL_TABLE_MANIFEST_V8.json"
    write_json_exclusive(
        manifest_path,
        {
            "schema_version": "revision0710.external_validation.formal_table_manifest.v8",
            "status": "TWENTY_TABLES_REHASHED_FORMULA_TOLERANCE_CROSSHASH_VALIDATED",
            "table_count": 20,
            "tables": table_records,
        },
    )
    return {
        "table_manifest_relative_path": "FORMAL_TABLE_MANIFEST_V8.json",
        "table_manifest_sha256": sha256_file(manifest_path),
        "table_files": table_records,
    }


def _path_mapping(value: Mapping[str, Any]) -> dict[str, Path]:
    return {str(key): Path(raw) for key, raw in value.items()}


def _jsonable_paths(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, raw in value.items():
        if isinstance(raw, Path):
            result[str(key)] = str(raw.resolve())
        else:
            result[str(key)] = raw
    return result


def run_worker_stage(stage_name: str, context_path: Path) -> dict[str, Any]:
    from v8_execution_guard import (
        sha256_file as guard_sha256_file,
        validate_current_environment,
    )

    thread_lock = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
    observed_thread_lock = {key: os.environ.get(key) for key in thread_lock}
    if observed_thread_lock != thread_lock:
        raise AssertionError(
            f"V8 worker thread/hash environment is not locked: {observed_thread_lock}"
        )
    context = json.loads(context_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "stage_name",
        "environment_role",
        "environment_lock_path",
        "environment_lock_sha256",
        "root",
        "artifact_dir",
        "contract_path",
        "stage",
        "payload",
    }
    if set(context) != required or context["stage_name"] != stage_name:
        raise AssertionError("V8 worker context is not closed or stage-bound")
    environment_lock_path = Path(context["environment_lock_path"])
    if guard_sha256_file(environment_lock_path) != context["environment_lock_sha256"]:
        raise AssertionError("V8 worker environment lock hash mismatch")
    environment_lock = json.loads(environment_lock_path.read_text(encoding="utf-8"))
    role = str(context["environment_role"])
    validate_current_environment(environment_lock[role], expected_role=role)
    if role == "inference":
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise AssertionError("V8 inference worker thread lock failed")
    root = Path(context["root"])
    artifact_dir = Path(context["artifact_dir"])
    stage = Path(context["stage"])
    payload = context["payload"]
    contract = json.loads(Path(context["contract_path"]).read_text(encoding="utf-8"))
    if stage_name == "01_cache":
        sources = _path_mapping(payload["sources"])
        value = build_label_blind_cache(
            root=root,
            artifact_dir=artifact_dir,
            stage=stage,
            sources=sources,
            contract=contract,
        )
    elif stage_name == "02_inference":
        if set(payload) != {"cache"}:
            raise AssertionError("inference worker received a forbidden source path")
        value = run_five_checkpoint_inference(
            root=root,
            artifact_dir=artifact_dir,
            stage=stage,
            cache=_path_mapping(payload["cache"]),
            contract=contract,
        )
    elif stage_name == "03_prediction_seal":
        value = seal_prediction_tables(
            stage=stage,
            predictions=_path_mapping(payload["predictions"]),
            contract=contract,
        )
    elif stage_name == "04_outcomes":
        value = build_outcomes_after_prediction_seal(
            artifact_dir=artifact_dir,
            stage=stage,
            sources=_path_mapping(payload["sources"]),
            prediction_seal=_path_mapping(payload["sealed"]),
            contract=contract,
        )
    elif stage_name == "05_tables":
        schemas = json.loads(
            (artifact_dir / "OUTPUT_SCHEMAS_V8.json").read_text(encoding="utf-8")
        )
        tables = compute_all_twenty_tables(
            artifact_dir=artifact_dir,
            cache=_path_mapping(payload["cache"]),
            sealed=_path_mapping(payload["sealed"]),
            endpoints=_path_mapping(payload["endpoints"]),
            contract=contract,
            bootstrap_seed=int(payload["bootstrap_seed"]),
            bootstrap_replicates=int(payload["bootstrap_replicates"]),
        )
        value = validate_and_write_tables(
            stage=stage,
            tables=tables,
            schemas=schemas,
            output_paths=payload["output_paths"],
        )
    elif stage_name == "06_independent_recompute":
        schemas = json.loads(
            (artifact_dir / "OUTPUT_SCHEMAS_V8.json").read_text(encoding="utf-8")
        )
        schema_by_id = {row["table_id"]: row for row in schemas["tables"]}
        recomputed = compute_all_twenty_tables(
            artifact_dir=artifact_dir,
            cache=_path_mapping(payload["cache"]),
            sealed=_path_mapping(payload["sealed"]),
            endpoints=_path_mapping(payload["endpoints"]),
            contract=contract,
            bootstrap_seed=int(payload["bootstrap_seed"]),
            bootstrap_replicates=int(payload["bootstrap_replicates"]),
        )
        table_stage = Path(payload["table_stage"])
        records = []
        for table_id, relative_path in payload["output_paths"].items():
            source_path = table_stage / relative_path
            observed = _validate_table_against_schema(
                table_id, _read_formal_table(source_path), schema_by_id[table_id]
            )
            expected = _validate_table_against_schema(
                table_id, recomputed[table_id], schema_by_id[table_id]
            )
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_dtype=True,
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
            )
            records.append(
                {
                    "table_id": table_id,
                    "formal_table_sha256": sha256_file(source_path),
                    "rows": len(expected),
                    "recomputed_from_upstream": True,
                }
            )
        independent_path = stage / "INDEPENDENT_TWENTY_TABLE_RECOMPUTE_V8.json"
        write_json_exclusive(
            independent_path,
            {
                "schema_version": "revision0710.external_validation.independent_recompute.v8",
                "status": "ALL_TWENTY_TABLES_RECOMPUTED_FROM_SEALED_UPSTREAM_IN_FRESH_PROCESS",
                "fresh_process": True,
                "age_metrics_recomputed": True,
                "endpoints_recomputed": True,
                "ipcw_and_censor_km_recomputed": True,
                "target_and_death_cox_cif_recomputed": True,
                "cox_associations_and_schoenfeld_recomputed": True,
                "aj_and_km_recomputed": True,
                "nhanes_full_design_survey_recomputed": True,
                "local_recalibration_recomputed": True,
                "table_count": 20,
                "tables": records,
            },
        )
        value = {
            "independent_recompute_receipt": independent_path,
            "independent_recompute_receipt_sha256": sha256_file(independent_path),
            "table_count": 20,
        }
    else:
        raise AssertionError(f"unknown V8 worker stage: {stage_name}")
    result_path = stage / "WORKER_RESULT_V8.json"
    result = {
        "schema_version": "revision0710.external_validation.worker_result.v8",
        "status": "WORKER_STAGE_COMPLETE_PENDING_ORCHESTRATOR_RECEIPT",
        "stage_name": stage_name,
        "environment_role": role,
        "environment_lock_sha256": context["environment_lock_sha256"],
        "thread_environment_lock": thread_lock,
        "value": _jsonable_paths(value),
    }
    write_json_exclusive(result_path, result)
    return result


def _validate_probe_environment(
    artifact_dir: Path, role: str
) -> tuple[str, dict[str, Any]]:
    from v8_execution_guard import validate_current_environment

    lock_path = artifact_dir / "ENVIRONMENT_LOCK_V8.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    observed = validate_current_environment(lock[role], expected_role=role)
    return sha256_file(lock_path), observed


def checkpoint_restore_forward_probe(
    *, root: Path, artifact_dir: Path, destination: Path
) -> dict[str, Any]:
    environment_lock_sha256, validated_environment = _validate_probe_environment(
        artifact_dir, "inference"
    )
    import torch

    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    registry = json.loads(
        (artifact_dir / "SOURCE_REGISTRY_V8.json").read_text(encoding="utf-8")
    )
    locked = {record["artifact_id"]: record for record in registry["locked_artifacts"]}
    embedding_record = locked["embedding_tables"]
    embedding_path = root / embedding_record["path"]
    if sha256_file(embedding_path) != embedding_record["sha256"]:
        raise AssertionError("checkpoint probe embedding-table hash mismatch")
    embedding = torch.load(embedding_path, map_location="cpu", weights_only=True)
    code_embedding = embedding["code_embedding"]
    text_embedding = embedding["text_embedding"]
    n = 512
    phenotype = torch.zeros((n, 78), dtype=torch.float32)
    phenotype_missing = torch.ones((n, 78), dtype=torch.bool)
    sex = torch.from_numpy((np.arange(n) % 2).astype(np.float32))
    event_code_id = torch.zeros((n, 32), dtype=torch.int64)
    event_text_id = torch.zeros((n, 32), dtype=torch.int64)
    event_relative_days = torch.zeros((n, 32), dtype=torch.int32)
    event_type_id = torch.zeros((n, 32), dtype=torch.int8)
    event_valid_mask = torch.zeros((n, 32), dtype=torch.bool)
    folds = []
    forward_seconds_by_fold: list[float] = []
    for fold in range(5):
        record = locked[f"checkpoint_fold_{fold}"]
        checkpoint_path = root / record["path"]
        if sha256_file(checkpoint_path) != record["sha256"]:
            raise AssertionError(f"checkpoint probe fold {fold} hash mismatch")
        model = _restore_fixed_model(
            root,
            checkpoint_path,
            code_embedding,
            text_embedding,
            outer_fold=fold,
        )
        repeat_seconds = []
        repeat_digests = []
        with torch.inference_mode():
            for _ in range(2):
                started = time.perf_counter()
                output = model(
                    phenotype,
                    phenotype_missing,
                    sex,
                    event_code_id,
                    event_text_id,
                    event_relative_days,
                    event_type_id,
                    event_valid_mask,
                )
                repeat_seconds.append(float(time.perf_counter() - started))
                repeat_digest = hashlib.sha256()
                for name, value in (
                    ("predicted_age", output.predicted_age),
                    ("base_age", output.base_age),
                    ("risk_age_offset", output.risk_age_offset),
                    ("risk_score", output.risk_score),
                ):
                    repeat_digest.update(name.encode("ascii"))
                    repeat_digest.update(
                        value.detach()
                        .cpu()
                        .numpy()
                        .astype("<f4", copy=False)
                        .tobytes(order="C")
                    )
                repeat_digests.append(repeat_digest.hexdigest())
        if repeat_digests[0] != repeat_digests[1]:
            raise AssertionError(
                f"checkpoint probe fold {fold} is not repeatable within process"
            )
        representative_batch_seconds = max(repeat_seconds)
        forward_seconds_by_fold.append(representative_batch_seconds)
        output_tensors = {
            "predicted_age": output.predicted_age,
            "base_age": output.base_age,
            "risk_age_offset": output.risk_age_offset,
            "risk_score": output.risk_score,
        }
        if any(not torch.isfinite(value).all() for value in output_tensors.values()):
            raise AssertionError(f"checkpoint probe fold {fold} produced nonfinite output")
        digest = hashlib.sha256()
        output_shapes = {}
        for name, value in output_tensors.items():
            array = value.detach().cpu().numpy().astype("<f4", copy=False)
            digest.update(name.encode("ascii"))
            digest.update(array.tobytes(order="C"))
            output_shapes[name] = list(array.shape)
        anchor_key, anchor_value = CHECKPOINT_AGE_ANCHORS[fold]
        folds.append(
            {
                "outer_fold": fold,
                "checkpoint_relative_path": record["path"],
                "checkpoint_sha256": record["sha256"],
                "age_anchor_key": anchor_key,
                "age_anchor_value": anchor_value,
                "state_and_buffers_exact": True,
                "n": n,
                "forward_seconds_two_repeats": repeat_seconds,
                "representative_batch_seconds_max_of_two": representative_batch_seconds,
                "output_shapes": output_shapes,
                "all_outputs_finite": True,
                "output_sha256": digest.hexdigest(),
            }
        )
        del model
    mimic_n = 95_683
    nhanes_n = 8_636
    batches_per_checkpoint = (
        math.ceil(mimic_n / n) * len(MIMIC_VARIANTS) + math.ceil(nhanes_n / n)
    )
    projected_forward_seconds = float(
        batches_per_checkpoint * sum(forward_seconds_by_fold)
    )
    payload = {
        "schema_version": "revision0710.external_validation.checkpoint_restore_forward_probe.v8",
        "status": "ALL_FIVE_FIXED_CHECKPOINTS_EXACT_RESTORE_AND_FORWARD_PASS",
        "environment_role": "inference",
        "environment_lock_sha256": environment_lock_sha256,
        "validated_environment": validated_environment,
        "torch_version": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "embedding_table_sha256": embedding_record["sha256"],
        "throughput_projection": {
            "measured_batch_size": n,
            "mimic_model_input_subjects": mimic_n,
            "mimic_input_variants": len(MIMIC_VARIANTS),
            "nhanes_model_input_subjects": nhanes_n,
            "checkpoint_count": 5,
            "batches_per_checkpoint": batches_per_checkpoint,
            "total_checkpoint_batches": batches_per_checkpoint * 5,
            "projected_formal_forward_seconds": projected_forward_seconds,
            "projected_formal_forward_hours": projected_forward_seconds / 3600.0,
            "scope": "checkpoint_forward_component_only_excludes_cache_clip_extension_model_restore_and_io",
        },
        "folds": folds,
    }
    write_json_exclusive(destination, payload)
    return payload


def _project_formal_statistics_core(
    *,
    conditional_b200_seconds_per_model: float,
    four_cause_specific_cox_seconds: float,
    shared_core_ph_fit_seconds: float,
    aalen_johansen_seconds: float,
    km_seconds: float,
) -> dict[str, Any]:
    values = np.asarray(
        [
            conditional_b200_seconds_per_model,
            four_cause_specific_cox_seconds,
            shared_core_ph_fit_seconds,
            aalen_johansen_seconds,
            km_seconds,
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("formal runtime projection requires finite nonnegative timings")
    heldout_analysis_count = len(MIMIC_VARIANTS) * len(ORGANS) * 2 * 5
    formal_recompute_passes = 2
    conditional_b200_seconds_all_models = (
        conditional_b200_seconds_per_model * PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS
    )
    cause_specific_cox_seconds_all_four_models = (
        four_cause_specific_cox_seconds * 2.0
    )
    shared_core_ph_seconds_all_eight_model_causes = (
        shared_core_ph_fit_seconds * 8.0
    )
    predictive_analysis_seconds = (
        cause_specific_cox_seconds_all_four_models
        + shared_core_ph_seconds_all_eight_model_causes
        + conditional_b200_seconds_all_models
    )
    projected_predictive_seconds = (
        predictive_analysis_seconds
        * heldout_analysis_count
        * formal_recompute_passes
    )
    association_analysis_count = len(MIMIC_VARIANTS) * len(ORGANS) * 2
    projected_association_seconds = (
        (four_cause_specific_cox_seconds / 4.0 + shared_core_ph_fit_seconds)
        * association_analysis_count
        * formal_recompute_passes
    )
    curve_analysis_count = len(MIMIC_VARIANTS) * len(ORGANS) * 2
    projected_curve_seconds = (
        (aalen_johansen_seconds * 2.0 + km_seconds)
        * curve_analysis_count
        * formal_recompute_passes
    )
    projected_statistics_core_seconds = (
        projected_predictive_seconds
        + projected_association_seconds
        + projected_curve_seconds
    )
    return {
        "heldout_analysis_count": heldout_analysis_count,
        "predictive_models_per_heldout_analysis": PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS,
        "conditional_b200_calls_per_heldout_analysis": PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS,
        "conditional_bootstrap_replicates_per_model": CONDITIONAL_BOOTSTRAP_REPLICATES_PER_MODEL,
        "conditional_b200_seconds_per_model": conditional_b200_seconds_per_model,
        "conditional_b200_seconds_all_models_per_heldout_analysis": conditional_b200_seconds_all_models,
        "cause_specific_cox_seconds_all_four_models_per_heldout_analysis": cause_specific_cox_seconds_all_four_models,
        "shared_core_ph_fit_seconds_per_model_cause": shared_core_ph_fit_seconds,
        "shared_core_ph_seconds_all_eight_model_causes_per_heldout_analysis": shared_core_ph_seconds_all_eight_model_causes,
        "association_analysis_count": association_analysis_count,
        "curve_analysis_count": curve_analysis_count,
        "formal_recompute_passes": formal_recompute_passes,
        "predictive_seconds_per_analysis_including_eight_prediction_cox_fits_eight_diagnostic_refits_and_four_model_specific_b200": predictive_analysis_seconds,
        "projected_predictive_seconds": projected_predictive_seconds,
        "projected_association_seconds": projected_association_seconds,
        "projected_curve_seconds": projected_curve_seconds,
        "projected_statistics_core_seconds": projected_statistics_core_seconds,
        "projected_statistics_core_hours": projected_statistics_core_seconds / 3600.0,
        "scope": "stage05_plus_independent_stage06_statistics_core_with_four_model_specific_B200_bootstraps_and_complete_diagnostic_refits; excludes cache_build_CLIP_restore_seal_IO_nhanes_survey_and_publication_overhead",
    }


def kernel_scalability_probe(
    *, artifact_dir: Path, destination: Path
) -> dict[str, Any]:
    environment_lock_sha256, validated_environment = _validate_probe_environment(
        artifact_dir, "statistics"
    )
    if destination.exists():
        raise FileExistsError(destination)
    n = 95_683
    rng = np.random.default_rng(7_102_026)
    duration = rng.integers(0, 366, size=n).astype(np.float64)
    cause = rng.choice(np.array([0, 1, 2], dtype=np.int8), size=n, p=[0.82, 0.12, 0.06])
    event = cause.eq(1) if isinstance(cause, pd.Series) else (cause == 1)
    event = np.asarray(event, dtype=np.int8)
    risk = rng.normal(size=n)
    binary = rng.integers(0, 2, size=n).astype(np.float64)
    weight = rng.uniform(0.2, 3.0, size=n)

    timings: dict[str, float] = {}

    def measure(name: str, callback: Any) -> Any:
        started = time.perf_counter()
        value = callback()
        timings[name] = float(time.perf_counter() - started)
        return value

    aj = measure("aalen_johansen_n95683", lambda: aalen_johansen(duration, cause))
    km = measure("km_n95683", lambda: _km_curve(duration, event))
    c_value = measure("harrell_c_n95683", lambda: harrell_c(duration, event, risk))
    auc = measure("weighted_auc_n95683", lambda: weighted_auc(binary, risk, weight))
    subject_ids = np.asarray([f"S{index:06d}" for index in range(n)])
    plan_ids, plan_counts = measure(
        "global_subject_plan_b200_n95683",
        lambda: _global_subject_bootstrap_plan(
            global_subject_ids=subject_ids,
            replicates=200,
            seed=7_102_026,
        ),
    )
    test_n = n // 5
    testing = pd.DataFrame(
        {
            "subject_id": subject_ids[:test_n],
            "observed_time_days": duration[:test_n],
            "event_indicator": event[:test_n],
        }
    )
    probability = 1.0 / (1.0 + np.exp(-risk[:test_n]))
    small_intervals = measure(
        "conditional_prediction_bootstrap_b20_test_n19136",
        lambda: _conditional_bootstrap_prediction_values(
            testing=testing,
            risk=risk[:test_n],
            base_risk=(0.75 * risk[:test_n] + 0.25 * rng.normal(size=test_n)),
            probability=probability,
            outcome=binary[:test_n],
            ipcw_weight=weight[:test_n],
            plan_subject_ids=plan_ids,
            plan_counts=plan_counts[:20],
        ),
    )
    projected_b200_seconds = timings[
        "conditional_prediction_bootstrap_b20_test_n19136"
    ] * 10.0
    train_n = n - test_n
    representative_training = pd.DataFrame(
        {
            "observed_time_days": duration[:train_n],
            "event_indicator": (cause[:train_n] == 1).astype(np.int8),
            "competing_death_indicator": (cause[:train_n] == 2).astype(np.int8),
            "age_per10": rng.uniform(1.8, 9.0, size=train_n),
            "sex_male": rng.integers(0, 2, size=train_n).astype(np.float64),
            "corrected_gap_sd": rng.normal(size=train_n),
        }
    )
    representative_testing = pd.DataFrame(
        {
            "age_per10": rng.uniform(1.8, 9.0, size=test_n),
            "sex_male": rng.integers(0, 2, size=test_n).astype(np.float64),
            "corrected_gap_sd": rng.normal(size=test_n),
        }
    )

    def representative_target_death_cox_pair() -> tuple[np.ndarray, np.ndarray]:
        _, base_probability = _cause_specific_cif_predictions(
            representative_training,
            representative_testing,
            ["age_per10", "sex_male"],
            365.0,
        )
        _, augmented_probability = _cause_specific_cif_predictions(
            representative_training,
            representative_testing,
            ["age_per10", "sex_male", "corrected_gap_sd"],
            365.0,
        )
        return base_probability, augmented_probability

    representative_probabilities = measure(
        "four_cause_specific_cox_fits_train_n76547_test_n19136",
        representative_target_death_cox_pair,
    )
    if any(np.any(~np.isfinite(value)) for value in representative_probabilities):
        raise AssertionError("representative cause-specific Cox projection is nonfinite")

    shared_core = load_survival_core(Path(__file__).resolve().parents[4])

    def representative_shared_core_ph_fit() -> pd.DataFrame:
        columns = ["age_per10", "sex_male", "corrected_gap_sd"]
        design = representative_training[columns].to_numpy(dtype=float)
        duration = representative_training["observed_time_days"].to_numpy(dtype=float) + 1.0
        event_values = representative_training["event_indicator"].to_numpy(dtype=np.int8)
        fit = shared_core.fit_cox_breslow(design, duration, event_values)
        if not fit.converged:
            raise AssertionError("representative shared-core PH fit did not converge")
        return shared_core.schoenfeld_ph_test(
            fit,
            design,
            duration,
            event_values,
            covariate_names=columns,
        )

    representative_ph = measure(
        "shared_core_ph_fit_and_two_transforms_train_n76547",
        representative_shared_core_ph_fit,
    )
    if len(representative_ph) != 6:
        raise AssertionError("representative shared-core PH row count mismatch")

    formal_statistics_projection = _project_formal_statistics_core(
        conditional_b200_seconds_per_model=projected_b200_seconds,
        four_cause_specific_cox_seconds=timings[
            "four_cause_specific_cox_fits_train_n76547_test_n19136"
        ],
        shared_core_ph_fit_seconds=timings[
            "shared_core_ph_fit_and_two_transforms_train_n76547"
        ],
        aalen_johansen_seconds=timings["aalen_johansen_n95683"],
        km_seconds=timings["km_n95683"],
    )
    if (
        len(aj) == 0
        or len(km) == 0
        or not np.isfinite(c_value)
        or not np.isfinite(auc)
        or any(bounds[0] is None for bounds in small_intervals.values())
        or max(
            timings["aalen_johansen_n95683"],
            timings["km_n95683"],
            timings["harrell_c_n95683"],
            timings["weighted_auc_n95683"],
        )
        > 15.0
        or timings["global_subject_plan_b200_n95683"] > 30.0
        or projected_b200_seconds > 600.0
        or timings["four_cause_specific_cox_fits_train_n76547_test_n19136"]
        > 120.0
        or timings["shared_core_ph_fit_and_two_transforms_train_n76547"] > 120.0
        or formal_statistics_projection["projected_statistics_core_seconds"]
        > 72.0 * 3600.0
    ):
        raise AssertionError("V8 realistic kernel scalability probe exceeded its frozen bound")
    payload = {
        "schema_version": "revision0710.external_validation.kernel_scalability_probe.v8",
        "status": "REALISTIC_N95683_FAST_KERNEL_AND_BOOTSTRAP_PLAN_PASS",
        "environment_role": "statistics",
        "environment_lock_sha256": environment_lock_sha256,
        "validated_environment": validated_environment,
        "n": n,
        "heldout_n": test_n,
        "predictive_models_per_heldout_analysis": PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS,
        "conditional_bootstrap_calls_per_heldout_analysis": PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS,
        "bootstrap_replicates_per_model": CONDITIONAL_BOOTSTRAP_REPLICATES_PER_MODEL,
        "global_plan_bytes": int(plan_counts.nbytes),
        "timings_seconds": timings,
        "conditional_prediction_bootstrap_b200_projected_seconds_per_model": projected_b200_seconds,
        "conditional_prediction_bootstrap_b200_projected_seconds_all_four_models_per_heldout_analysis": projected_b200_seconds
        * PREDICTIVE_MODELS_PER_HELDOUT_ANALYSIS,
        "formal_statistics_projection": formal_statistics_projection,
        "operational_resource_reservation": {
            "minimum_hours": OPERATIONAL_RESERVATION_MIN_HOURS,
            "safer_hours": OPERATIONAL_RESERVATION_SAFER_HOURS,
            "not_a_wall_time_upper_bound": True,
            "covers_or_reserves": RESOURCE_RESERVATION_COMPONENTS,
        },
        "bounds_seconds": {
            "each_fast_kernel": 15.0,
            "global_subject_plan_b200": 30.0,
            "conditional_prediction_bootstrap_b200_per_model": 600.0,
            "conditional_prediction_bootstrap_b200_all_four_models_per_heldout_analysis": 2400.0,
            "four_cause_specific_cox_fits_train_n76547_test_n19136": 120.0,
            "shared_core_ph_fit_and_two_transforms_train_n76547": 120.0,
            "formal_statistics_core_projection": 72.0 * 3600.0,
        },
    }
    write_json_exclusive(destination, payload)
    return payload


def _worker_main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker-stage")
    mode.add_argument("--checkpoint-probe-destination", type=Path)
    mode.add_argument("--kernel-scalability-probe-destination", type=Path)
    parser.add_argument("--context-json", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    if args.kernel_scalability_probe_destination is not None:
        if args.artifact_dir is None:
            parser.error("kernel scalability probe requires --artifact-dir")
        result = kernel_scalability_probe(
            artifact_dir=args.artifact_dir.resolve(),
            destination=args.kernel_scalability_probe_destination.resolve(),
        )
    elif args.checkpoint_probe_destination is not None:
        if args.repo_root is None or args.artifact_dir is None:
            parser.error("checkpoint probe requires --repo-root and --artifact-dir")
        result = checkpoint_restore_forward_probe(
            root=args.repo_root.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
            destination=args.checkpoint_probe_destination.resolve(),
        )
    else:
        if args.context_json is None:
            parser.error("worker stage requires --context-json")
        result = run_worker_stage(args.worker_stage, args.context_json.resolve())
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())
