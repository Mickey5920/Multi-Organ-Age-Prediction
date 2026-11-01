from __future__ import annotations

"""Rebuild ICD-derived labels with the corrected primary codebook.

The original leakage-safe cohort, follow-up sources, GP summary, and fold seed
are immutable inputs. All rebuilt files are written below this versioned
directory so an in-flight v0 experiment cannot observe a label change.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
REBUILD_ROOT = REPO_ROOT / "Revision-0710" / "01_data_rebuild"
BASE_INDEX = REBUILD_ROOT / "01_index_and_events"
BASE_OUTCOMES = REBUILD_ROOT / "02_outcomes_60m"
BASE_FOLDS = REBUILD_ROOT / "03_folds"
BASE_LEAKAGE = REBUILD_ROOT / "05_leakage_audit" / "leakage_gate.json"
CONFIG = REBUILD_ROOT / "00_config" / "source_paths.json"
MAPPING = REBUILD_ROOT / "06_icd_codebook_v1" / "02_v1_corrected_primary" / "mapping.csv"
ICD_BUILDER = REBUILD_ROOT / "scripts" / "02_build_icd_events.py"
OUTCOME_BUILDER = REBUILD_ROOT / "scripts" / "04_build_outcomes_and_folds.py"

INDEX_DIR = HERE / "01_index_and_events"
OUTCOME_DIR = HERE / "02_outcomes_60m"
FOLD_DIR = HERE / "03_folds"
AUDIT_DIR = HERE / "04_delta_audit"

STAGED_INPUTS = (
    "age_model_cohort.parquet",
    "followup_source_dates.parquet",
    "gp_participant_summary.parquet",
)
RENAL_SUMMARY_COLUMNS = (
    "prevalent_renal_count",
    "incident_renal_count",
    "first_incident_date_renal",
    "has_prevalent_renal",
)
OUTCOME_KEY = ("participant_id", "organ_system")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def equal_with_na(left: pd.Series, right: pd.Series) -> pd.Series:
    """Elementwise equality that treats paired missing values as equal."""
    return left.eq(right) | (left.isna() & right.isna())


def changed_rows(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not left.index.equals(right.index):
        raise ValueError("Cannot compare frames with different indices")
    changed = pd.Series(False, index=left.index)
    for column in columns:
        if column not in left or column not in right:
            raise ValueError(f"Missing comparison column: {column}")
        changed |= ~equal_with_na(left[column], right[column])
    return changed


def require_same_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    columns: list[str],
    label: str,
) -> None:
    mask = changed_rows(left, right, columns)
    if mask.any():
        raise ValueError(f"{label} changed in {int(mask.sum())} rows")


def stage_immutable_inputs() -> dict[str, dict[str, object]]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for name in STAGED_INPUTS:
        source = BASE_INDEX / name
        destination = INDEX_DIR / name
        if not source.exists():
            raise FileNotFoundError(source)
        source_hash = sha256_file(source)
        if destination.exists():
            if sha256_file(destination) != source_hash:
                raise ValueError(f"Staged immutable input differs from source: {destination}")
        else:
            shutil.copy2(source, destination)
        records[name] = {
            "source": file_record(source),
            "staged": file_record(destination),
            "identical": sha256_file(destination) == source_hash,
        }
    return records


def expected_outputs_exist() -> bool:
    paths = [
        INDEX_DIR / "icd_participant_summary.parquet",
        INDEX_DIR / "canonical_preindex_icd.parquet",
        OUTCOME_DIR / "organ_outcomes_60m.parquet",
        OUTCOME_DIR / "participant_followup_and_evaluation_masks.parquet",
        FOLD_DIR / "fold_manifest_5fold.parquet",
    ]
    return all(path.exists() for path in paths)


def run_builders(*, skip_build: bool) -> None:
    if skip_build:
        if not expected_outputs_exist():
            raise FileNotFoundError("--skip-build requested but corrected-primary outputs are incomplete")
        return
    generated = [
        INDEX_DIR / "icd_participant_summary.parquet",
        OUTCOME_DIR / "organ_outcomes_60m.parquet",
        FOLD_DIR / "fold_manifest_5fold.parquet",
    ]
    existing = [str(path) for path in generated if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing corrected-primary rebuild; use --skip-build to re-audit: "
            + ", ".join(existing)
        )
    subprocess.run(
        [
            sys.executable,
            str(ICD_BUILDER),
            "--config",
            str(CONFIG),
            "--index-dir",
            str(INDEX_DIR),
            "--mapping-csv",
            str(MAPPING),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(OUTCOME_BUILDER),
            "--config",
            str(CONFIG),
            "--index-dir",
            str(INDEX_DIR),
            "--outcome-dir",
            str(OUTCOME_DIR),
            "--fold-dir",
            str(FOLD_DIR),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def _indexed(path: Path, key: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.duplicated(key).any():
        raise ValueError(f"Duplicate keys in {path}: {key}")
    return frame.set_index(key).sort_index()


def audit_mapping() -> dict[str, object]:
    expected = pd.read_csv(MAPPING, dtype="string", keep_default_na=False)
    rebuilt = pd.read_csv(INDEX_DIR / "icd_mapping_export.csv", dtype="string", keep_default_na=False)
    pair_columns = ["organ_system", "icd_prefix_normalized"]
    expected_pairs = set(map(tuple, expected[pair_columns].itertuples(index=False, name=None)))
    rebuilt_pairs = set(map(tuple, rebuilt[pair_columns].itertuples(index=False, name=None)))
    if expected_pairs != rebuilt_pairs:
        raise ValueError(
            "Rebuilt mapping differs from corrected primary: "
            f"missing={len(expected_pairs - rebuilt_pairs)}, extra={len(rebuilt_pairs - expected_pairs)}"
        )
    if "N299C64" in set(rebuilt["icd_prefix_normalized"]):
        raise ValueError("Malformed N299C64 remained in rebuilt mapping")
    renal = set(
        rebuilt.loc[rebuilt["organ_system"].eq("renal"), "icd_prefix_normalized"]
    )
    if not {"N299", "C64"}.issubset(renal):
        raise ValueError("Corrected renal N299/C64 prefixes are missing")
    return {
        "corrected_primary_rows": len(expected),
        "rebuilt_rows": len(rebuilt),
        "pair_sets_identical": True,
        "malformed_N299C64_present": False,
        "renal_N299_present": True,
        "renal_C64_present": True,
    }


def audit_icd_summary() -> tuple[dict[str, object], pd.DataFrame]:
    old = _indexed(BASE_INDEX / "icd_participant_summary.parquet", ["participant_id"])
    new = _indexed(INDEX_DIR / "icd_participant_summary.parquet", ["participant_id"])
    if not old.index.equals(new.index) or set(old.columns) != set(new.columns):
        raise ValueError("ICD participant-summary schema or participant set changed")
    invariant_columns = [column for column in old.columns if column not in RENAL_SUMMARY_COLUMNS]
    require_same_frame(old, new, columns=invariant_columns, label="Non-renal ICD summary")
    renal_changed = changed_rows(old, new, list(RENAL_SUMMARY_COLUMNS))
    delta = pd.DataFrame(index=old.index[renal_changed]).reset_index()
    for column in RENAL_SUMMARY_COLUMNS:
        delta[f"old_{column}"] = old.loc[renal_changed, column].to_numpy()
        delta[f"new_{column}"] = new.loc[renal_changed, column].to_numpy()
    old_prevalent = old["has_prevalent_renal"].fillna(False).astype(bool)
    new_prevalent = new["has_prevalent_renal"].fillna(False).astype(bool)
    old_incident = old["first_incident_date_renal"].notna()
    new_incident = new["first_incident_date_renal"].notna()
    summary = {
        "participant_n": len(old),
        "nonrenal_columns_compared": len(invariant_columns),
        "nonrenal_columns_identical": True,
        "renal_changed_participants": int(renal_changed.sum()),
        "old_prevalent_renal_n": int(old_prevalent.sum()),
        "new_prevalent_renal_n": int(new_prevalent.sum()),
        "newly_prevalent_renal_n": int((~old_prevalent & new_prevalent).sum()),
        "old_first_incident_renal_n": int(old_incident.sum()),
        "new_first_incident_renal_n": int(new_incident.sum()),
        "newly_first_incident_renal_n": int((~old_incident & new_incident).sum()),
    }
    return summary, delta


def audit_outcomes() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    old = _indexed(BASE_OUTCOMES / "organ_outcomes_60m.parquet", list(OUTCOME_KEY))
    new = _indexed(OUTCOME_DIR / "organ_outcomes_60m.parquet", list(OUTCOME_KEY))
    if not old.index.equals(new.index) or set(old.columns) != set(new.columns):
        raise ValueError("Organ-outcome schema or participant-organ rows changed")

    nonrenal_index = old.index.get_level_values("organ_system") != "renal"
    old_nonrenal = old.loc[nonrenal_index]
    new_nonrenal = new.loc[nonrenal_index]
    nonrenal_survival_columns = [
        column for column in old.columns if column != "cohort_state_60m"
    ]
    require_same_frame(
        old_nonrenal,
        new_nonrenal,
        columns=nonrenal_survival_columns,
        label="Non-renal survival outcomes",
    )
    nonrenal_state_changed = changed_rows(
        old_nonrenal, new_nonrenal, ["cohort_state_60m"]
    )
    nonrenal_state_delta = pd.DataFrame(
        index=old_nonrenal.index[nonrenal_state_changed]
    ).reset_index()
    nonrenal_state_delta["old_cohort_state_60m"] = old_nonrenal.loc[
        nonrenal_state_changed, "cohort_state_60m"
    ].to_numpy()
    nonrenal_state_delta["new_cohort_state_60m"] = new_nonrenal.loc[
        nonrenal_state_changed, "cohort_state_60m"
    ].to_numpy()

    old_renal = old.xs("renal", level="organ_system", drop_level=False)
    new_renal = new.xs("renal", level="organ_system", drop_level=False)
    renal_changed = changed_rows(old_renal, new_renal, list(old.columns))
    renal_training_label_columns = ["event", "duration_months", "at_risk"]
    renal_training_label_changed = changed_rows(
        old_renal, new_renal, renal_training_label_columns
    )
    renal_first_date_changed = changed_rows(
        old_renal, new_renal, ["first_incident_target_date"]
    )
    renal_state_changed = changed_rows(
        old_renal, new_renal, ["cohort_state_60m"]
    )
    changed_ids = pd.DataFrame(
        {
            "participant_id": old_renal.index.get_level_values("participant_id")[
                renal_training_label_changed
            ].astype("string")
        }
    )
    fold_manifest = pd.read_parquet(
        FOLD_DIR / "fold_manifest_5fold.parquet",
        columns=["participant_id", "fold_id"],
    )
    fold_manifest["participant_id"] = fold_manifest["participant_id"].astype("string")
    changed_with_fold = changed_ids.merge(
        fold_manifest,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if changed_with_fold["fold_id"].isna().any():
        raise ValueError("Renal label-delta participants are missing fold assignments")
    fold_counts = (
        changed_with_fold.groupby("fold_id").size().reindex(range(5), fill_value=0)
    )
    delta = pd.DataFrame(index=old_renal.index[renal_changed]).reset_index()
    for column in old.columns:
        delta[f"old_{column}"] = old_renal.loc[renal_changed, column].to_numpy()
        delta[f"new_{column}"] = new_renal.loc[renal_changed, column].to_numpy()

    old_strict = _indexed(
        BASE_OUTCOMES / "participant_followup_and_evaluation_masks.parquet",
        ["participant_id"],
    )
    new_strict = _indexed(
        OUTCOME_DIR / "participant_followup_and_evaluation_masks.parquet",
        ["participant_id"],
    )
    require_same_frame(
        old_strict,
        new_strict,
        columns=list(old_strict.columns),
        label="Participant follow-up and strict-health masks",
    )
    summary = {
        "organ_outcome_rows": len(old),
        "nonrenal_survival_fields_identical": True,
        "nonrenal_cohort_state_changed_rows": int(nonrenal_state_changed.sum()),
        "nonrenal_cohort_state_change_reason": (
            "Adding renal C64 changes whether a participant has another included-organ "
            "baseline disease; event, duration, at-risk, censoring, and strict-health fields "
            "remain unchanged for all non-renal targets."
        ),
        "renal_any_field_changed_rows": int(renal_changed.sum()),
        "renal_training_label_changed_rows": int(renal_training_label_changed.sum()),
        "renal_training_label_columns": renal_training_label_columns,
        "renal_training_label_changed_by_fold": {
            str(int(fold)): int(count) for fold, count in fold_counts.items()
        },
        "renal_first_incident_date_changed_rows": int(renal_first_date_changed.sum()),
        "renal_cohort_state_changed_rows": int(renal_state_changed.sum()),
        "strict_health_unchanged": True,
        "strict_health_n": int(new_strict["strict_healthy_reference_60m"].sum()),
        "old_renal_prevalent_n": int(old_renal["has_prevalent_target"].sum()),
        "new_renal_prevalent_n": int(new_renal["has_prevalent_target"].sum()),
        "old_renal_risk_set_n": int(old_renal["at_risk"].sum()),
        "new_renal_risk_set_n": int(new_renal["at_risk"].sum()),
        "old_renal_event_n": int(old_renal["event"].sum()),
        "new_renal_event_n": int(new_renal["event"].sum()),
    }
    return summary, delta, nonrenal_state_delta


def audit_folds_and_leakage() -> dict[str, object]:
    fold_files = (
        "fold_manifest_5fold.parquet",
        "outer_fold_roles.parquet",
        "fold0_validation_calibration_screen.parquet",
    )
    fold_records: dict[str, object] = {}
    for name in fold_files:
        old = pd.read_parquet(BASE_FOLDS / name).sort_index(axis=1)
        new = pd.read_parquet(FOLD_DIR / name).sort_index(axis=1)
        if not old.equals(new):
            raise ValueError(f"Outcome-blind fold artifact changed: {name}")
        fold_records[name] = {
            "content_identical": True,
            "old": file_record(BASE_FOLDS / name),
            "new": file_record(FOLD_DIR / name),
        }

    canonical = pd.read_parquet(INDEX_DIR / "canonical_preindex_icd.parquet")
    relative_days = pd.to_numeric(canonical["relative_days"], errors="coerce")
    post_or_same_day = relative_days.ge(0) | relative_days.isna()
    if post_or_same_day.any():
        raise ValueError(
            f"Corrected canonical ICD contains {int(post_or_same_day.sum())} non-preindex rows"
        )
    icd_audit = json.loads((INDEX_DIR / "icd_rebuild_audit.json").read_text(encoding="utf-8"))
    outcome_audit = json.loads(
        (OUTCOME_DIR / "outcome_rebuild_audit.json").read_text(encoding="utf-8")
    )
    inherited = json.loads(BASE_LEAKAGE.read_text(encoding="utf-8"))
    if icd_audit.get("status") != "PASS" or outcome_audit.get("status") != "PASS":
        raise ValueError("Corrected ICD/outcome rebuild audit did not pass")
    if inherited.get("status") != "PASS":
        raise ValueError("Base leakage-safe rebuild did not pass its frozen gate")
    return {
        "canonical_preindex_rows": len(canonical),
        "canonical_max_relative_days": int(relative_days.max()),
        "canonical_nonpreindex_rows": 0,
        "icd_rebuild_status": "PASS",
        "outcome_rebuild_status": "PASS",
        "inherited_full_leakage_gate_status": "PASS",
        "inherited_full_leakage_gate": file_record(BASE_LEAKAGE),
        "fold_artifacts": fold_records,
    }


def write_audit(staged_inputs: dict[str, dict[str, object]]) -> dict[str, object]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    mapping_audit = audit_mapping()
    icd_summary, participant_delta = audit_icd_summary()
    outcome_summary, outcome_delta, nonrenal_state_delta = audit_outcomes()
    leakage = audit_folds_and_leakage()
    participant_delta.to_csv(
        AUDIT_DIR / "renal_participant_label_delta.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    outcome_delta.to_csv(
        AUDIT_DIR / "renal_outcome_row_delta.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    nonrenal_state_delta.to_csv(
        AUDIT_DIR / "nonrenal_cohort_state_delta.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    audit = {
        "schema_version": "revision0710.corrected_primary_rebuild.audit.v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "mapping": mapping_audit,
        "icd_participant_summary": icd_summary,
        "organ_outcomes": outcome_summary,
        "leakage_and_folds": leakage,
        "guards": {
            "base_outputs_overwritten": False,
            "running_v0_training_labels_modified": False,
            "compact_cache_rebuild_required": False,
            "strict_health_changed": False,
            "nonrenal_survival_fields_changed": False,
            "nonrenal_cohort_state_changes_recorded": True,
        },
    }
    write_json(AUDIT_DIR / "DELTA_AND_LEAKAGE_AUDIT.json", audit)

    stable_outputs = [
        INDEX_DIR / "icd_mapping_export.csv",
        INDEX_DIR / "icd_participant_summary.parquet",
        INDEX_DIR / "canonical_preindex_icd.parquet",
        INDEX_DIR / "observed_icd_code_system_matches.csv",
        INDEX_DIR / "icd_rebuild_audit.json",
        OUTCOME_DIR / "participant_followup_and_evaluation_masks.parquet",
        OUTCOME_DIR / "organ_outcomes_60m.parquet",
        OUTCOME_DIR / "organ_cohort_state_counts_60m.csv",
        OUTCOME_DIR / "outcome_rebuild_audit.json",
        FOLD_DIR / "fold_manifest_5fold.parquet",
        FOLD_DIR / "outer_fold_roles.parquet",
        FOLD_DIR / "fold0_validation_calibration_screen.parquet",
        AUDIT_DIR / "renal_participant_label_delta.csv.gz",
        AUDIT_DIR / "renal_outcome_row_delta.csv.gz",
        AUDIT_DIR / "nonrenal_cohort_state_delta.csv.gz",
        AUDIT_DIR / "DELTA_AND_LEAKAGE_AUDIT.json",
    ]
    manifest = {
        "schema_version": "revision0710.corrected_primary_rebuild.manifest.v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "mapping": file_record(MAPPING),
        "config": file_record(CONFIG),
        "builders": {
            "icd": file_record(ICD_BUILDER),
            "outcomes_and_folds": file_record(OUTCOME_BUILDER),
            "orchestrator": file_record(Path(__file__)),
        },
        "staged_immutable_inputs": staged_inputs,
        "outputs": {
            str(path.relative_to(HERE)): file_record(path) for path in stable_outputs
        },
        "audit_summary": audit,
    }
    write_json(HERE / "MANIFEST.json", manifest)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Re-run delta/leakage audit against an already complete versioned rebuild.",
    )
    args = parser.parse_args()
    write_json(
        HERE / "RUN_STATUS.json",
        {"status": "RUNNING", "updated_utc": utc_now(), "skip_build": args.skip_build},
    )
    try:
        staged = stage_immutable_inputs()
        run_builders(skip_build=args.skip_build)
        audit = write_audit(staged)
    except Exception as exc:
        write_json(
            HERE / "RUN_STATUS.json",
            {"status": "FAILED", "updated_utc": utc_now(), "error": repr(exc)},
        )
        raise
    write_json(
        HERE / "RUN_STATUS.json",
        {
            "status": "COMPLETE",
            "updated_utc": utc_now(),
            "strict_health_n": audit["organ_outcomes"]["strict_health_n"],
            "renal_changed_participants": audit["icd_participant_summary"][
                "renal_changed_participants"
            ],
            "renal_training_label_changed_rows": audit["organ_outcomes"][
                "renal_training_label_changed_rows"
            ],
        },
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
