from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REBUILD_DIR = ROOT / "Revision-0710" / "01_data_rebuild"
DEFAULT_OUTPUT_DIR = DEFAULT_REBUILD_DIR / "05_leakage_audit"
MONTH_DAYS = 365.25 / 12.0
SYSTEMS = [
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
]
COHORT_COLUMNS = {
    "participant_id",
    "index_date",
    "birth_date",
    "chronological_age",
    "sex",
    "phenotype_missing_count",
    "phenotype_missing_fraction",
    "age_model_eligible",
}
FOLD_COLUMNS = {
    "participant_id",
    "chronological_age",
    "sex",
    "phenotype_missing_count",
    "sex_norm",
    "age_decile",
    "missingness_bin",
    "stratum",
    "fold_id",
}
FUTURE_FIELD_TOKENS = (
    "death",
    "event",
    "incident",
    "prevalent",
    "outcome",
    "followup",
    "follow_up",
    "survival",
    "censor",
    "risk_set",
    "strict_healthy",
    "cohort_state",
)


def _normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_gp_path(index_dir: Path) -> Path:
    candidates = list(index_dir.glob("canonical_preindex_gp_l*.parquet"))
    if not candidates:
        raise FileNotFoundError("canonical_preindex_gp_l*.parquet")

    def retention_limit(path: Path) -> int:
        suffix = path.stem.rsplit("_l", maxsplit=1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    return max(candidates, key=lambda path: (retention_limit(path), path.name))


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _run_check(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = function()
        if "status" not in result:
            result["status"] = "FAIL"
            result["error"] = "check did not return a status"
        return {key: _json_scalar(value) for key, value in result.items()}
    except Exception as exc:  # Keep the full gate report even when one artifact is malformed.
        return {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "check": name,
        }


def _check_required_artifacts(rebuild_dir: Path) -> dict[str, Any]:
    required = [
        "01_index_and_events/age_model_cohort.parquet",
        "01_index_and_events/canonical_preindex_icd.parquet",
        "01_index_and_events/raw_phenotypes.parquet",
        "01_index_and_events/phenotype_missing_mask.parquet",
        "01_index_and_events/phenotype_feature_columns.json",
        "01_index_and_events/age_cohort_audit.json",
        "01_index_and_events/icd_rebuild_audit.json",
        "01_index_and_events/gp_rebuild_audit.json",
        "02_outcomes_60m/participant_followup_and_evaluation_masks.parquet",
        "02_outcomes_60m/organ_outcomes_60m.parquet",
        "02_outcomes_60m/outcome_rebuild_audit.json",
        "03_folds/fold_manifest_5fold.parquet",
        "03_folds/outer_fold_roles.parquet",
        "03_folds/fold0_validation_calibration_screen.parquet",
        "03_folds/fold_rebuild_audit.json",
    ]
    missing = [relative for relative in required if not (rebuild_dir / relative).is_file()]
    gp_candidates = sorted(
        path.name
        for path in (rebuild_dir / "01_index_and_events").glob(
            "canonical_preindex_gp_l*.parquet"
        )
    )
    if not gp_candidates:
        missing.append("01_index_and_events/canonical_preindex_gp_l*.parquet")
    return {
        "status": _status(not missing),
        "required_artifact_count": len(required) + 1,
        "missing_artifacts": missing,
        "gp_canonical_candidates": gp_candidates,
    }


def _check_age_cohort(
    rebuild_dir: Path,
    *,
    expected_cohort_n: int | None,
) -> tuple[dict[str, Any], pd.DataFrame, int]:
    index_dir = rebuild_dir / "01_index_and_events"
    lineage = _read_json(index_dir / "age_cohort_audit.json")
    cohort = pd.read_parquet(index_dir / "age_model_cohort.parquet")
    cohort["participant_id"] = _normalize_id(cohort["participant_id"])
    cohort["index_date"] = pd.to_datetime(cohort["index_date"], errors="coerce")
    cohort["birth_date"] = pd.to_datetime(cohort["birth_date"], errors="coerce")

    if expected_cohort_n is None:
        expected_cohort_n = int(lineage.get("age_model_cohort_n", len(cohort)))
    future_columns = sorted(
        column
        for column in cohort.columns
        if any(token in column.lower() for token in FUTURE_FIELD_TOKENS)
    )
    missing_columns = sorted(COHORT_COLUMNS - set(cohort.columns))
    unexpected_columns = sorted(set(cohort.columns) - COHORT_COLUMNS)
    recomputed_age = (cohort["index_date"] - cohort["birth_date"]).dt.days / 365.25
    age_error = (pd.to_numeric(cohort["chronological_age"], errors="coerce") - recomputed_age).abs()
    unique_ids = int(cohort["participant_id"].nunique())
    empty_ids = int(cohort["participant_id"].eq("").sum())
    eligible_bad = int(pd.to_numeric(cohort["age_model_eligible"], errors="coerce").ne(1).sum())
    bad_date_rows = int((cohort["index_date"].isna() | cohort["birth_date"].isna()).sum())
    missing_count = pd.to_numeric(cohort["phenotype_missing_count"], errors="coerce")
    missing_bad = int((missing_count.isna() | missing_count.lt(0) | missing_count.gt(19)).sum())
    lineage_future_fields = lineage.get("future_outcome_fields_used_for_age_eligibility", None)
    lineage_ok = bool(
        lineage.get("status") == "PASS"
        and lineage.get("age_model_cohort_future_independent") is True
        and lineage_future_fields == []
        and int(lineage.get("age_model_cohort_n", -1)) == len(cohort)
    )
    ok = bool(
        not missing_columns
        and not unexpected_columns
        and not future_columns
        and len(cohort) == expected_cohort_n
        and unique_ids == expected_cohort_n
        and empty_ids == 0
        and eligible_bad == 0
        and bad_date_rows == 0
        and missing_bad == 0
        and age_error.notna().all()
        and float(age_error.max()) <= 0.01
        and lineage_ok
    )
    result = {
        "status": _status(ok),
        "expected_participants": int(expected_cohort_n),
        "rows": int(len(cohort)),
        "unique_participants": unique_ids,
        "empty_participant_ids": empty_ids,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "future_named_columns": future_columns,
        "bad_date_rows": bad_date_rows,
        "ineligible_rows": eligible_bad,
        "bad_missing_count_rows": missing_bad,
        "max_chronological_age_recompute_error_years": (
            float(age_error.max()) if age_error.notna().any() else None
        ),
        "lineage_declares_future_independent": lineage_ok,
        "future_outcome_fields_used_for_age_eligibility": lineage_future_fields,
    }
    return result, cohort, int(expected_cohort_n)


def _audit_event_table(
    path: Path,
    *,
    cohort_index: pd.Series,
    source_name: str,
) -> dict[str, Any]:
    required = ["participant_id", "event_date", "relative_days", "attention_mask"]
    frame = pd.read_parquet(path, columns=required)
    frame["participant_id"] = _normalize_id(frame["participant_id"])
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    relative = pd.to_numeric(frame["relative_days"], errors="coerce")
    mask_numeric = pd.to_numeric(frame["attention_mask"], errors="coerce")
    mapped_index = frame["participant_id"].map(cohort_index)
    recomputed = (frame["event_date"] - mapped_index).dt.days

    unknown_participants = int(mapped_index.isna().sum())
    invalid_mask_rows = int((~mask_numeric.isin([0, 1])).sum())
    inactive_rows = int(mask_numeric.eq(0).sum())
    active_nonpreindex = int((mask_numeric.eq(1) & (relative.isna() | relative.ge(0))).sum())
    same_or_future = int((relative.isna() | relative.ge(0)).sum())
    undated_rows = int(frame["event_date"].isna().sum())
    date_nonpreindex = int((frame["event_date"].notna() & mapped_index.notna() & frame["event_date"].ge(mapped_index)).sum())
    relative_mismatch = int(
        (recomputed.notna() & relative.notna() & ~np.isclose(recomputed, relative, atol=0.0)).sum()
    )
    ok = bool(
        unknown_participants == 0
        and invalid_mask_rows == 0
        and inactive_rows == 0
        and active_nonpreindex == 0
        and same_or_future == 0
        and undated_rows == 0
        and date_nonpreindex == 0
        and relative_mismatch == 0
    )
    return {
        "source": source_name,
        "rows": int(len(frame)),
        "active_rows": int(mask_numeric.eq(1).sum()),
        "inactive_rows": inactive_rows,
        "invalid_mask_rows": invalid_mask_rows,
        "unknown_participant_rows": unknown_participants,
        "active_nonpreindex_rows": active_nonpreindex,
        "same_or_future_rows": same_or_future,
        "undated_rows": undated_rows,
        "event_date_nonpreindex_rows": date_nonpreindex,
        "relative_day_mismatch_rows": relative_mismatch,
        "max_relative_days": float(relative.max()) if relative.notna().any() else None,
        "ok": ok,
    }


def _check_preindex_events(rebuild_dir: Path, cohort: pd.DataFrame) -> dict[str, Any]:
    index_dir = rebuild_dir / "01_index_and_events"
    cohort_index = cohort.set_index("participant_id")["index_date"]
    icd = _audit_event_table(
        index_dir / "canonical_preindex_icd.parquet",
        cohort_index=cohort_index,
        source_name="icd",
    )
    gp = _audit_event_table(
        _resolve_gp_path(index_dir),
        cohort_index=cohort_index,
        source_name="gp",
    )
    icd_lineage = _read_json(index_dir / "icd_rebuild_audit.json")
    gp_lineage = _read_json(index_dir / "gp_rebuild_audit.json")

    def lineage_ok(payload: dict[str, Any], observed_rows: int) -> bool:
        gate = payload.get("leakage_gate", {})
        return bool(
            payload.get("status") == "PASS"
            and gate.get("status") == "PASS"
            and int(gate.get("rows", -1)) == observed_rows
            and int(gate.get("non_preindex_input_rows", -1)) == 0
            and int(gate.get("inactive_nonzero_time_rows", -1)) == 0
        )

    icd_lineage_ok = lineage_ok(icd_lineage, icd["rows"])
    gp_lineage_ok = lineage_ok(gp_lineage, gp["rows"])
    ok = bool(icd["ok"] and gp["ok"] and icd_lineage_ok and gp_lineage_ok)
    return {
        "status": _status(ok),
        "icd_rows": icd["rows"],
        "icd_active_rows": icd["active_rows"],
        "icd_inactive_rows": icd["inactive_rows"],
        "icd_invalid_mask_rows": icd["invalid_mask_rows"],
        "icd_unknown_participant_rows": icd["unknown_participant_rows"],
        "icd_active_nonpreindex_rows": icd["active_nonpreindex_rows"],
        "icd_same_or_future_rows": icd["same_or_future_rows"],
        "icd_event_date_nonpreindex_rows": icd["event_date_nonpreindex_rows"],
        "icd_relative_day_mismatch_rows": icd["relative_day_mismatch_rows"],
        "icd_max_relative_days": icd["max_relative_days"],
        "gp_rows": gp["rows"],
        "gp_canonical_file": _resolve_gp_path(index_dir).name,
        "gp_active_rows": gp["active_rows"],
        "gp_inactive_rows": gp["inactive_rows"],
        "gp_invalid_mask_rows": gp["invalid_mask_rows"],
        "gp_unknown_participant_rows": gp["unknown_participant_rows"],
        "gp_active_nonpreindex_rows": gp["active_nonpreindex_rows"],
        "gp_same_or_future_rows": gp["same_or_future_rows"],
        "gp_event_date_nonpreindex_rows": gp["event_date_nonpreindex_rows"],
        "gp_relative_day_mismatch_rows": gp["relative_day_mismatch_rows"],
        "gp_max_relative_days": gp["max_relative_days"],
        "icd_lineage_gate_reconciled": icd_lineage_ok,
        "gp_lineage_gate_reconciled": gp_lineage_ok,
    }


def _check_phenotype_and_mask(
    rebuild_dir: Path,
    cohort: pd.DataFrame,
    *,
    expected_n: int,
) -> dict[str, Any]:
    index_dir = rebuild_dir / "01_index_and_events"
    feature_columns = _read_json(index_dir / "phenotype_feature_columns.json")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("phenotype_feature_columns.json must be a non-empty list")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("phenotype feature names must be unique")
    raw = pd.read_parquet(
        index_dir / "raw_phenotypes.parquet", columns=["participant_id", *feature_columns]
    )
    mask = pd.read_parquet(
        index_dir / "phenotype_missing_mask.parquet", columns=["participant_id", *feature_columns]
    )
    raw["participant_id"] = _normalize_id(raw["participant_id"])
    mask["participant_id"] = _normalize_id(mask["participant_id"])
    raw = raw.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    mask = mask.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    cohort_ordered = cohort.sort_values("participant_id", kind="mergesort").reset_index(drop=True)

    raw_unique = int(raw["participant_id"].nunique())
    mask_unique = int(mask["participant_id"].nunique())
    raw_id_match = bool(raw["participant_id"].equals(cohort_ordered["participant_id"]))
    mask_id_match = bool(mask["participant_id"].equals(cohort_ordered["participant_id"]))
    invalid_mask_cells = 0
    mask_mismatch_cells = 0
    raw_missing_count = np.zeros(len(raw), dtype=np.int16)
    for column in feature_columns:
        observed_mask = pd.to_numeric(mask[column], errors="coerce")
        invalid_mask_cells += int((~observed_mask.isin([0, 1])).sum())
        text = raw[column].astype("string").str.strip()
        expected_mask = raw[column].isna() | text.isin(["", "nan", "None", "<NA>"])
        mask_mismatch_cells += int(observed_mask.ne(expected_mask.astype(np.int8)).sum())
        raw_missing_count += expected_mask.to_numpy(dtype=np.int16)
    stored_missing_count = pd.to_numeric(
        cohort_ordered["phenotype_missing_count"], errors="coerce"
    ).to_numpy()
    stored_missing_fraction = pd.to_numeric(
        cohort_ordered["phenotype_missing_fraction"], errors="coerce"
    ).to_numpy()
    missing_count_mismatch = int((stored_missing_count != raw_missing_count).sum())
    fraction_mismatch = int(
        (~np.isclose(stored_missing_fraction, raw_missing_count / len(feature_columns), atol=1e-6)).sum()
    )
    age_lineage = _read_json(index_dir / "age_cohort_audit.json")
    feature_count_lineage_ok = int(age_lineage.get("phenotype_feature_count", -1)) == len(feature_columns)
    ok = bool(
        len(raw) == expected_n
        and len(mask) == expected_n
        and raw_unique == expected_n
        and mask_unique == expected_n
        and raw_id_match
        and mask_id_match
        and invalid_mask_cells == 0
        and mask_mismatch_cells == 0
        and missing_count_mismatch == 0
        and fraction_mismatch == 0
        and feature_count_lineage_ok
    )
    result = {
        "status": _status(ok),
        "feature_count": int(len(feature_columns)),
        "raw_rows": int(len(raw)),
        "raw_unique_participants": raw_unique,
        "mask_rows": int(len(mask)),
        "mask_unique_participants": mask_unique,
        "raw_ids_equal_cohort": raw_id_match,
        "mask_ids_equal_cohort": mask_id_match,
        "invalid_mask_cells": invalid_mask_cells,
        "mask_mismatch_cells": mask_mismatch_cells,
        "participant_missing_count_mismatch_rows": missing_count_mismatch,
        "participant_missing_fraction_mismatch_rows": fraction_mismatch,
        "feature_count_lineage_reconciled": feature_count_lineage_ok,
    }
    del raw, mask
    gc.collect()
    return result


def _check_outcomes(
    rebuild_dir: Path,
    cohort: pd.DataFrame,
    *,
    expected_n: int,
) -> dict[str, Any]:
    outcome_dir = rebuild_dir / "02_outcomes_60m"
    participant = pd.read_parquet(outcome_dir / "participant_followup_and_evaluation_masks.parquet")
    participant["participant_id"] = _normalize_id(participant["participant_id"])
    participant["index_date"] = pd.to_datetime(participant["index_date"], errors="coerce")
    participant = participant.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    cohort_ordered = cohort.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    participant_id_match = bool(participant["participant_id"].equals(cohort_ordered["participant_id"]))
    baseline_mismatch = 0
    if participant_id_match:
        baseline_mismatch += int(participant["index_date"].ne(cohort_ordered["index_date"]).sum())
        baseline_mismatch += int(
            (~np.isclose(
                participant["chronological_age"],
                cohort_ordered["chronological_age"],
                atol=1e-9,
            )).sum()
        )
        baseline_mismatch += int(
            participant["phenotype_missing_count"].ne(cohort_ordered["phenotype_missing_count"]).sum()
        )

    administrative_end = participant["index_date"] + pd.to_timedelta(60 * MONTH_DAYS, unit="D")
    observed_end = pd.to_datetime(participant["observed_followup_end"], errors="coerce")
    followup_expected = observed_end.notna() & observed_end.ge(administrative_end)
    death_date = pd.to_datetime(participant["death_date"], errors="coerce")
    death_expected = (
        death_date.notna()
        & death_date.gt(participant["index_date"])
        & death_date.le(administrative_end)
    )
    first_any = pd.to_datetime(participant["first_incident_any_date"], errors="coerce")
    strict_expected = (
        ~participant["has_prevalent_any_icd"].fillna(False).astype(bool)
        & (first_any.isna() | first_any.gt(administrative_end))
        & followup_expected
        & ~death_expected
    )
    followup_mismatch = int(
        pd.to_numeric(participant["followup_ascertained_60m"], errors="coerce")
        .ne(followup_expected.astype(np.int8))
        .sum()
    )
    death_mismatch = int(
        pd.to_numeric(participant["death_within_60m"], errors="coerce")
        .ne(death_expected.astype(np.int8))
        .sum()
    )
    strict_mismatch = int(
        pd.to_numeric(participant["strict_healthy_reference_60m"], errors="coerce")
        .ne(strict_expected.astype(np.int8))
        .sum()
    )

    required_outcome_columns = [
        "participant_id",
        "event",
        "duration_months",
        "competing_death",
        "at_risk",
        "censor_reason",
        "outcome_end_date",
        "organ_system",
        "index_date",
        "first_incident_target_date",
        "has_prevalent_target",
        "strict_healthy_reference_60m",
        "full_incident_risk_set_60m",
    ]
    outcomes = pd.read_parquet(
        outcome_dir / "organ_outcomes_60m.parquet", columns=required_outcome_columns
    )
    outcomes["participant_id"] = _normalize_id(outcomes["participant_id"])
    for column in ["outcome_end_date", "index_date", "first_incident_target_date"]:
        outcomes[column] = pd.to_datetime(outcomes[column], errors="coerce")
    observed_systems = sorted(outcomes["organ_system"].dropna().unique().tolist())
    organ_counts = outcomes.groupby("organ_system", observed=True).size()
    bad_organ_counts = {
        str(system): int(count)
        for system, count in organ_counts.items()
        if int(count) != expected_n
    }
    duplicate_pairs = int(outcomes.duplicated(["participant_id", "organ_system"]).sum())
    unknown_outcome_ids = int((~outcomes["participant_id"].isin(cohort_ordered["participant_id"])).sum())
    expected_rows = expected_n * len(SYSTEMS)
    binary_columns = [
        "event",
        "competing_death",
        "at_risk",
        "has_prevalent_target",
        "strict_healthy_reference_60m",
        "full_incident_risk_set_60m",
    ]
    invalid_binary_rows = int(
        sum((~pd.to_numeric(outcomes[column], errors="coerce").isin([0, 1])).sum() for column in binary_columns)
    )
    event = outcomes["event"].eq(1)
    competing = outcomes["competing_death"].eq(1)
    at_risk = outcomes["at_risk"].eq(1)
    prevalent = outcomes["has_prevalent_target"].eq(1)
    full_risk_mismatch = int(
        outcomes["full_incident_risk_set_60m"].ne(outcomes["at_risk"]).sum()
    )
    event_logic_bad = int((event & (~at_risk | prevalent | competing)).sum())
    competing_logic_bad = int((competing & (~at_risk | event | prevalent)).sum())
    nonrisk_logic_bad = int((~at_risk & (event | competing)).sum())
    duration = pd.to_numeric(outcomes["duration_months"], errors="coerce")
    duration_presence_bad = int(
        (at_risk & (duration.isna() | duration.le(0) | duration.gt(60.000001))).sum()
        + ((~at_risk) & duration.notna()).sum()
    )
    duration_expected = (outcomes["outcome_end_date"] - outcomes["index_date"]).dt.days / MONTH_DAYS
    duration_value_bad = int(
        (at_risk & ~np.isclose(duration, duration_expected, atol=1e-6, equal_nan=False)).sum()
    )
    index_lookup = cohort.set_index("participant_id")["index_date"]
    outcome_index_mismatch = int(
        outcomes["index_date"].ne(outcomes["participant_id"].map(index_lookup)).sum()
    )
    event_date_bad = int(
        (
            event
            & (
                outcomes["first_incident_target_date"].isna()
                | outcomes["first_incident_target_date"].le(outcomes["index_date"])
                | outcomes["first_incident_target_date"].ne(outcomes["outcome_end_date"])
            )
        ).sum()
    )
    reason_bad = int(
        (event & outcomes["censor_reason"].ne("target_event")).sum()
        + (competing & outcomes["censor_reason"].ne("death_before_target")).sum()
        + (prevalent & outcomes["censor_reason"].ne("prevalent_target_excluded")).sum()
    )
    strict_lookup = participant.set_index("participant_id")["strict_healthy_reference_60m"]
    outcome_strict_mismatch = int(
        outcomes["strict_healthy_reference_60m"]
        .ne(outcomes["participant_id"].map(strict_lookup))
        .sum()
    )
    lineage = _read_json(outcome_dir / "outcome_rebuild_audit.json")
    lineage_ok = bool(
        lineage.get("status") == "PASS"
        and int(lineage.get("participant_n", -1)) == expected_n
        and int(lineage.get("organ_outcome_rows", -1)) == len(outcomes)
    )
    ok = bool(
        len(participant) == expected_n
        and participant["participant_id"].nunique() == expected_n
        and participant_id_match
        and baseline_mismatch == 0
        and followup_mismatch == 0
        and death_mismatch == 0
        and strict_mismatch == 0
        and len(outcomes) == expected_rows
        and observed_systems == sorted(SYSTEMS)
        and not bad_organ_counts
        and duplicate_pairs == 0
        and unknown_outcome_ids == 0
        and invalid_binary_rows == 0
        and full_risk_mismatch == 0
        and event_logic_bad == 0
        and competing_logic_bad == 0
        and nonrisk_logic_bad == 0
        and duration_presence_bad == 0
        and duration_value_bad == 0
        and outcome_index_mismatch == 0
        and event_date_bad == 0
        and reason_bad == 0
        and outcome_strict_mismatch == 0
        and lineage_ok
    )
    result = {
        "status": _status(ok),
        "participant_rows": int(len(participant)),
        "participant_unique_ids": int(participant["participant_id"].nunique()),
        "participant_ids_equal_cohort": participant_id_match,
        "participant_baseline_mismatch_cells": baseline_mismatch,
        "followup_60m_mismatch_rows": followup_mismatch,
        "death_60m_mismatch_rows": death_mismatch,
        "strict_healthy_mismatch_rows": strict_mismatch,
        "expected_organ_rows": int(expected_rows),
        "organ_rows": int(len(outcomes)),
        "organ_systems": observed_systems,
        "bad_organ_row_counts": bad_organ_counts,
        "duplicate_participant_organ_rows": duplicate_pairs,
        "unknown_participant_rows": unknown_outcome_ids,
        "invalid_binary_cells": invalid_binary_rows,
        "full_risk_at_risk_mismatch_rows": full_risk_mismatch,
        "bad_event_logic_rows": event_logic_bad,
        "bad_competing_death_logic_rows": competing_logic_bad,
        "bad_nonrisk_logic_rows": nonrisk_logic_bad,
        "bad_duration_presence_or_range_rows": duration_presence_bad,
        "bad_duration_value_rows": duration_value_bad,
        "outcome_index_date_mismatch_rows": outcome_index_mismatch,
        "bad_target_event_date_rows": event_date_bad,
        "bad_censor_reason_rows": reason_bad,
        "outcome_strict_mask_mismatch_rows": outcome_strict_mismatch,
        "lineage_gate_reconciled": lineage_ok,
    }
    del outcomes, participant
    gc.collect()
    return result


def _normalize_sex(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("missing")
        .str.strip()
        .str.lower()
        .replace({"f": "female", "m": "male", "": "missing"})
    )


def _check_folds(
    rebuild_dir: Path,
    cohort: pd.DataFrame,
    *,
    expected_n: int,
) -> dict[str, Any]:
    fold_dir = rebuild_dir / "03_folds"
    fold = pd.read_parquet(fold_dir / "fold_manifest_5fold.parquet")
    fold["participant_id"] = _normalize_id(fold["participant_id"])
    future_columns = sorted(
        column
        for column in fold.columns
        if any(token in column.lower() for token in FUTURE_FIELD_TOKENS)
    )
    missing_columns = sorted(FOLD_COLUMNS - set(fold.columns))
    unexpected_columns = sorted(set(fold.columns) - FOLD_COLUMNS)
    fold_ordered = fold.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    cohort_ordered = cohort.sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    id_match = bool(fold_ordered["participant_id"].equals(cohort_ordered["participant_id"]))
    baseline_mismatch = 0
    stratum_mismatch = 0
    if id_match:
        baseline_mismatch += int((~np.isclose(
            fold_ordered["chronological_age"], cohort_ordered["chronological_age"], atol=1e-9
        )).sum())
        baseline_mismatch += int(fold_ordered["sex"].ne(cohort_ordered["sex"]).sum())
        baseline_mismatch += int(
            fold_ordered["phenotype_missing_count"]
            .ne(cohort_ordered["phenotype_missing_count"])
            .sum()
        )
        expected_sex = _normalize_sex(fold_ordered["sex"])
        expected_age_decile = pd.qcut(
            fold_ordered["chronological_age"], 10, labels=False, duplicates="drop"
        ).astype(int)
        expected_missing_bin = pd.cut(
            fold_ordered["phenotype_missing_count"],
            bins=[-1, 0, 5, 10, 19],
            labels=["0", "1-5", "6-10", "11-19"],
            include_lowest=True,
        ).astype("string")
        expected_stratum = (
            expected_sex
            + "|age"
            + expected_age_decile.astype(str)
            + "|missing"
            + expected_missing_bin
        )
        stratum_mismatch += int(fold_ordered["sex_norm"].ne(expected_sex).sum())
        stratum_mismatch += int(fold_ordered["age_decile"].ne(expected_age_decile).sum())
        stratum_mismatch += int(fold_ordered["missingness_bin"].ne(expected_missing_bin).sum())
        stratum_mismatch += int(fold_ordered["stratum"].ne(expected_stratum).sum())
    fold_numeric = pd.to_numeric(fold["fold_id"], errors="coerce")
    invalid_fold_rows = int((fold_numeric.isna() | ~fold_numeric.isin(range(5))).sum())
    fold_ids = sorted(int(value) for value in fold_numeric.dropna().unique())

    roles = pd.read_parquet(fold_dir / "outer_fold_roles.parquet")
    roles["participant_id"] = _normalize_id(roles["participant_id"])
    role_fold_lookup = fold.set_index("participant_id")["fold_id"]
    role_expected_fold = roles["participant_id"].map(role_fold_lookup)
    role_fold_mismatch = int(roles["fold_id"].ne(role_expected_fold).sum())
    expected_role = np.where(
        roles["fold_id"].eq(roles["outer_fold"]),
        "test",
        np.where(
            roles["fold_id"].eq((roles["outer_fold"] + 1) % 5), "validation", "train"
        ),
    )
    outer_role_mismatch = int(roles["split_role"].ne(expected_role).sum())
    duplicate_outer_rows = int(roles.duplicated(["participant_id", "outer_fold"]).sum())
    role_counts = roles.groupby("participant_id").size()
    bad_role_count_participants = int(role_counts.ne(5).sum())
    invalid_outer_rows = int(
        (~pd.to_numeric(roles["outer_fold"], errors="coerce").isin(range(5))).sum()
    )

    inner = pd.read_parquet(fold_dir / "fold0_validation_calibration_screen.parquet")
    inner["participant_id"] = _normalize_id(inner["participant_id"])
    expected_inner = fold.loc[
        fold["fold_id"].eq(1), ["participant_id", "fold_id", "stratum"]
    ].sort_values("participant_id", kind="mergesort").reset_index(drop=True)
    actual_inner = inner[["participant_id", "fold_id", "stratum"]].sort_values(
        "participant_id", kind="mergesort"
    ).reset_index(drop=True)
    inner_membership_match = bool(actual_inner.equals(expected_inner))
    invalid_inner_roles = int((~inner["inner_role"].isin(["calibration", "screen"])).sum())
    inner_role_counts = {
        str(key): int(value) for key, value in inner["inner_role"].value_counts().items()
    }
    both_inner_roles_present = set(inner_role_counts) == {"calibration", "screen"}

    lineage = _read_json(fold_dir / "fold_rebuild_audit.json")
    allowed_assignment_fields = {
        "sex_norm",
        "chronological_age_decile",
        "phenotype_missingness_bin",
    }
    lineage_fields = set(lineage.get("assignment_fields", []))
    lineage_ok = bool(
        lineage.get("status") == "PASS"
        and int(lineage.get("participant_n", -1)) == expected_n
        and int(lineage.get("unique_participant_n", -1)) == expected_n
        and lineage.get("future_outcome_fields_used_for_assignment") == []
        and lineage.get("outer_role_gate") is True
        and lineage_fields == allowed_assignment_fields
    )
    ok = bool(
        not missing_columns
        and not unexpected_columns
        and not future_columns
        and len(fold) == expected_n
        and fold["participant_id"].nunique() == expected_n
        and id_match
        and baseline_mismatch == 0
        and stratum_mismatch == 0
        and invalid_fold_rows == 0
        and fold_ids == [0, 1, 2, 3, 4]
        and len(roles) == expected_n * 5
        and role_fold_mismatch == 0
        and outer_role_mismatch == 0
        and duplicate_outer_rows == 0
        and bad_role_count_participants == 0
        and invalid_outer_rows == 0
        and inner_membership_match
        and invalid_inner_roles == 0
        and both_inner_roles_present
        and lineage_ok
    )
    return {
        "status": _status(ok),
        "fold_rows": int(len(fold)),
        "fold_unique_participants": int(fold["participant_id"].nunique()),
        "fold_ids": fold_ids,
        "fold_sizes": {str(key): int(value) for key, value in fold.groupby("fold_id").size().items()},
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "future_named_columns": future_columns,
        "participant_ids_equal_cohort": id_match,
        "baseline_field_mismatch_cells": baseline_mismatch,
        "stratum_recompute_mismatch_cells": stratum_mismatch,
        "invalid_fold_rows": invalid_fold_rows,
        "outer_role_rows": int(len(roles)),
        "outer_role_fold_mismatch_rows": role_fold_mismatch,
        "outer_role_mismatch_rows": outer_role_mismatch,
        "duplicate_participant_outer_fold_rows": duplicate_outer_rows,
        "participants_without_five_outer_roles": bad_role_count_participants,
        "invalid_outer_fold_rows": invalid_outer_rows,
        "fold0_validation_membership_exact": inner_membership_match,
        "invalid_inner_role_rows": invalid_inner_roles,
        "fold0_validation_inner_counts": inner_role_counts,
        "both_inner_roles_present": both_inner_roles_present,
        "lineage_assignment_fields": sorted(lineage_fields),
        "future_outcome_fields_used_for_assignment": lineage.get(
            "future_outcome_fields_used_for_assignment"
        ),
        "lineage_gate_reconciled": lineage_ok,
    }


def audit_rebuild(
    rebuild_dir: Path,
    *,
    expected_cohort_n: int | None = None,
) -> dict[str, Any]:
    rebuild_dir = Path(rebuild_dir)
    checks: dict[str, dict[str, Any]] = {}
    checks["required_artifacts"] = _run_check(
        "required_artifacts", lambda: _check_required_artifacts(rebuild_dir)
    )
    if checks["required_artifacts"]["status"] != "PASS":
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "rebuild_dir": str(rebuild_dir.resolve()),
            "checks": checks,
            "failed_checks": ["required_artifacts"],
        }

    try:
        age_result, cohort, expected_n = _check_age_cohort(
            rebuild_dir, expected_cohort_n=expected_cohort_n
        )
        checks["age_cohort_and_lineage"] = age_result
    except Exception as exc:
        checks["age_cohort_and_lineage"] = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "rebuild_dir": str(rebuild_dir.resolve()),
            "checks": checks,
            "failed_checks": [
                name for name, result in checks.items() if result.get("status") != "PASS"
            ],
        }

    checks["preindex_events"] = _run_check(
        "preindex_events", lambda: _check_preindex_events(rebuild_dir, cohort)
    )
    checks["phenotype_and_missing_mask"] = _run_check(
        "phenotype_and_missing_mask",
        lambda: _check_phenotype_and_mask(rebuild_dir, cohort, expected_n=expected_n),
    )
    checks["outcome_consistency"] = _run_check(
        "outcome_consistency",
        lambda: _check_outcomes(rebuild_dir, cohort, expected_n=expected_n),
    )
    checks["fold_purity"] = _run_check(
        "fold_purity", lambda: _check_folds(rebuild_dir, cohort, expected_n=expected_n)
    )
    failed = [name for name, result in checks.items() if result.get("status") != "PASS"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failed else "FAIL",
        "rebuild_dir": str(rebuild_dir.resolve()),
        "expected_cohort_n": int(expected_n),
        "checks": checks,
        "failed_checks": failed,
    }


def _report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Revision-0710 Leakage Audit",
        "",
        f"- Status: **{result['status']}**",
        f"- Created (UTC): `{result['created_utc']}`",
        f"- Rebuild directory: `{result['rebuild_dir']}`",
        f"- Expected age cohort: `{result.get('expected_cohort_n', 'unavailable')}`",
        "",
        "## Hard Gates",
        "",
        "| Check | Status | Key evidence |",
        "|---|---:|---|",
    ]
    evidence_fields = {
        "required_artifacts": ["required_artifact_count", "missing_artifacts"],
        "age_cohort_and_lineage": ["rows", "unique_participants", "future_named_columns"],
        "preindex_events": [
            "icd_same_or_future_rows",
            "gp_same_or_future_rows",
            "icd_max_relative_days",
            "gp_max_relative_days",
        ],
        "phenotype_and_missing_mask": ["feature_count", "mask_mismatch_cells"],
        "outcome_consistency": [
            "organ_rows",
            "duplicate_participant_organ_rows",
            "bad_duration_value_rows",
            "bad_event_logic_rows",
        ],
        "fold_purity": [
            "fold_sizes",
            "outer_role_mismatch_rows",
            "future_outcome_fields_used_for_assignment",
        ],
    }
    for name, check in result["checks"].items():
        evidence = []
        for field in evidence_fields.get(name, []):
            if field in check:
                evidence.append(f"{field}={json.dumps(check[field], ensure_ascii=True)}")
        if "error" in check:
            evidence.append(f"error={check['error']}")
        lines.append(f"| `{name}` | **{check.get('status', 'FAIL')}** | {'; '.join(evidence)} |")
    if result.get("failed_checks"):
        lines.extend(
            [
                "",
                "## Failures",
                "",
                *[f"- `{name}`" for name in result["failed_checks"]],
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A PASS means that the materialized training inputs are strictly pre-index, "
            "the phenotype masks reproduce raw missingness, all seven organ outcomes are "
            "internally consistent, and outer-fold roles contain no participant overlap or "
            "future/outcome-derived assignment fields.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard leakage gates on the Revision-0710 rebuild.")
    parser.add_argument("--rebuild-dir", type=Path, default=DEFAULT_REBUILD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-cohort-n", type=int, default=456773)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Write the reports but do not return a non-zero exit code on a failed gate.",
    )
    args = parser.parse_args()
    result = audit_rebuild(args.rebuild_dir, expected_cohort_n=args.expected_cohort_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "leakage_gate.json"
    report_path = args.output_dir / "LEAKAGE_AUDIT_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    report_path.write_text(_report_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}, indent=2))
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")
    if result["status"] != "PASS" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
