from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
REVISION_ROOT = REPO_ROOT / "Revision-0710"
OUTPUT_ROOT = (
    REVISION_ROOT
    / "Final_Fig_0716"
    / "Reviewer_Statistics"
    / "19_r3_9_fold_age_distribution"
)
SOURCE_DATA = OUTPUT_ROOT / "Source_Data"

FOLD_MANIFEST = (
    REVISION_ROOT / "01_data_rebuild" / "03_folds" / "fold_manifest_5fold.parquet"
)
FOLD_BALANCE = (
    REVISION_ROOT / "01_data_rebuild" / "03_folds" / "fold_baseline_balance.csv"
)
FOLD_AUDIT = (
    REVISION_ROOT / "01_data_rebuild" / "03_folds" / "fold_rebuild_audit.json"
)
OUTER_ROLES = (
    REVISION_ROOT / "01_data_rebuild" / "03_folds" / "outer_fold_roles.parquet"
)
TRANSFORM_MANIFEST = (
    REVISION_ROOT / "01_data_rebuild" / "04_cache_v1" / "fold_transform_manifest.json"
)
ROLE_ROOT = (
    REVISION_ROOT
    / "03_ehr_age"
    / "05_corrected_primary_v1"
    / "00_role_artifacts"
)
FINAL_SELECTION = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "02_fivefold_frozen_huber1"
    / "FINAL_SELECTION.json"
)
FOLD0_RUN_CONFIG = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "01_runs"
    / "shared_private_independent_view_huber1_fold0"
    / "run_config.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "role": role,
        "path": path.as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def age_summary(frame: pd.DataFrame) -> dict[str, object]:
    age = frame["chronological_age"].astype(float)
    return {
        "n": int(len(frame)),
        "age_mean": float(age.mean()),
        "age_sd": float(age.std(ddof=1)),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
        "unique_participants": int(frame["participant_id"].nunique()),
    }


def main() -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)

    folds = pd.read_parquet(FOLD_MANIFEST)
    expected_columns = {
        "participant_id",
        "chronological_age",
        "sex_norm",
        "phenotype_missing_count",
        "fold_id",
    }
    missing_columns = sorted(expected_columns - set(folds.columns))
    if missing_columns:
        raise RuntimeError(f"fold manifest is missing columns: {missing_columns}")
    if len(folds) != 456_773 or folds["participant_id"].nunique() != 456_773:
        raise RuntimeError("fold manifest participant coverage is not 456,773 unique IDs")

    base_rows: list[dict[str, object]] = []
    for fold_id, frame in folds.groupby("fold_id", sort=True):
        row = {"fold_id": int(fold_id), **age_summary(frame)}
        row["phenotype_missingness_mean"] = float(
            frame["phenotype_missing_count"].astype(float).mean()
        )
        row["female_n"] = int(frame["sex_norm"].eq("female").sum())
        row["male_n"] = int(frame["sex_norm"].eq("male").sum())
        base_rows.append(row)
    base = pd.DataFrame(base_rows)

    frozen_balance = pd.read_csv(FOLD_BALANCE).sort_values("fold_id").reset_index(drop=True)
    base_check = base.sort_values("fold_id").reset_index(drop=True)
    if not base_check["n"].equals(frozen_balance["n"]):
        raise RuntimeError("recomputed fold sizes differ from frozen fold balance")
    for left, right in (
        ("age_mean", "age_mean"),
        ("age_sd", "age_sd"),
        ("age_min", "age_min"),
        ("age_max", "age_max"),
        ("phenotype_missingness_mean", "missingness_mean"),
    ):
        delta = (base_check[left] - frozen_balance[right]).abs().max()
        if float(delta) > 1e-9:
            raise RuntimeError(f"recomputed {left} differs from frozen balance: {delta}")

    role_specs = (
        ("training", "01_train"),
        ("calibration", "02_calibration"),
        ("outer_test", "03_outer_test"),
    )
    role_rows: list[dict[str, object]] = []
    role_input_records: list[dict[str, object]] = []
    for outer_fold in range(5):
        sets: dict[str, set[str]] = {}
        for role, directory in role_specs:
            path = ROLE_ROOT / f"outer_fold_{outer_fold}" / directory / "age_targets.parquet"
            frame = pd.read_parquet(path)
            if set(frame.columns) < {"participant_id", "chronological_age"}:
                raise RuntimeError(f"unexpected age-target schema: {path}")
            row = {
                "outer_rotation": outer_fold + 1,
                "role": role,
                **age_summary(frame),
            }
            role_rows.append(row)
            sets[role] = set(frame["participant_id"].astype(str))
            role_input_records.append(
                file_record(
                    path,
                    f"outer_rotation_{outer_fold + 1}:{role}:age_targets",
                    rows=len(frame),
                )
            )
        if sets["training"] & sets["calibration"]:
            raise RuntimeError(f"training/calibration overlap in rotation {outer_fold + 1}")
        if sets["training"] & sets["outer_test"]:
            raise RuntimeError(f"training/test overlap in rotation {outer_fold + 1}")
        if sets["calibration"] & sets["outer_test"]:
            raise RuntimeError(f"calibration/test overlap in rotation {outer_fold + 1}")
        if len(sets["training"] | sets["calibration"] | sets["outer_test"]) != 456_773:
            raise RuntimeError(f"rotation {outer_fold + 1} does not cover the full cohort")

    role_table = pd.DataFrame(role_rows)
    ranges = (
        role_table.groupby("role", sort=False)
        .agg(
            n_min=("n", "min"),
            n_max=("n", "max"),
            age_mean_min=("age_mean", "min"),
            age_mean_max=("age_mean", "max"),
            age_sd_min=("age_sd", "min"),
            age_sd_max=("age_sd", "max"),
            age_min=("age_min", "min"),
            age_max=("age_max", "max"),
        )
        .reset_index()
    )

    outer_roles = pd.read_parquet(OUTER_ROLES)
    role_counts = outer_roles.groupby("split_role").size().to_dict()
    expected_role_counts = {
        "train": 3 * 456_773,
        "validation": 456_773,
        "test": 456_773,
    }
    if {key: int(role_counts.get(key, -1)) for key in expected_role_counts} != expected_role_counts:
        raise RuntimeError(f"unexpected outer-role counts: {role_counts}")
    for role, expected_per_participant in (("train", 3), ("validation", 1), ("test", 1)):
        counts = (
            outer_roles.loc[outer_roles["split_role"].eq(role)]
            .groupby("participant_id")
            .size()
        )
        if len(counts) != 456_773 or not counts.eq(expected_per_participant).all():
            raise RuntimeError(f"participant rotation count failed for role={role}")

    fold_audit = json.loads(FOLD_AUDIT.read_text(encoding="utf-8"))
    transform_manifest = json.loads(TRANSFORM_MANIFEST.read_text(encoding="utf-8"))
    if fold_audit.get("future_outcome_fields_used_for_assignment") != []:
        raise RuntimeError("future outcomes were used in fold assignment")
    if len(transform_manifest) != 5:
        raise RuntimeError("fold transform manifest does not contain five rotations")
    for entry in transform_manifest:
        if len(entry["train_folds"]) != 3 or entry["train_n"] not in range(274_062, 274_068):
            raise RuntimeError(f"unexpected transform-fit role: {entry}")

    base_path = SOURCE_DATA / "base_fold_age_distribution.csv"
    role_path = SOURCE_DATA / "role_age_distribution_by_rotation.csv"
    range_path = SOURCE_DATA / "role_age_distribution_ranges.csv"
    base.to_csv(base_path, index=False, float_format="%.9f")
    role_table.to_csv(role_path, index=False, float_format="%.9f")
    ranges.to_csv(range_path, index=False, float_format="%.9f")

    summary = {
        "participant_n": 456_773,
        "age_range_years": [
            float(folds["chronological_age"].min()),
            float(folds["chronological_age"].max()),
        ],
        "base_fold_n_range": [int(base["n"].min()), int(base["n"].max())],
        "assignment_strata": fold_audit["assignment_fields"],
        "future_outcome_fields_used_for_assignment": [],
        "rotation_contract": {
            "training_folds": 3,
            "calibration_folds": 1,
            "outer_test_folds": 1,
            "participant_test_predictions": 1,
        },
        "role_ranges": ranges.to_dict(orient="records"),
        "fold0_inner_role_counts": fold_audit.get("fold0_validation_inner_counts", {}),
    }
    summary_path = SOURCE_DATA / "age_distribution_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    qa = {
        "status": "PASS",
        "participant_n": 456_773,
        "unique_participant_n": 456_773,
        "five_base_folds": len(base) == 5,
        "fifteen_rotation_role_rows": len(role_table) == 15,
        "role_disjointness_all_rotations": True,
        "full_cohort_coverage_all_rotations": True,
        "each_participant_test_once": True,
        "each_participant_calibration_once": True,
        "each_participant_training_three_times": True,
        "future_outcome_fields_used_for_assignment": False,
        "recomputed_frozen_balance_match": True,
        "transform_fit_uses_three_training_folds": True,
    }
    qa_path = OUTPUT_ROOT / "QA_REPORT.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    output_records = [
        file_record(base_path, "base_fold_age_distribution", rows=len(base)),
        file_record(role_path, "role_age_distribution_by_rotation", rows=len(role_table)),
        file_record(range_path, "role_age_distribution_ranges", rows=len(ranges)),
        file_record(summary_path, "age_distribution_summary"),
        file_record(qa_path, "qa_report"),
    ]
    input_records = [
        file_record(FOLD_MANIFEST, "participant_fold_manifest", rows=len(folds)),
        file_record(FOLD_BALANCE, "frozen_fold_balance", rows=len(frozen_balance)),
        file_record(FOLD_AUDIT, "fold_rebuild_audit"),
        file_record(OUTER_ROLES, "outer_fold_roles", rows=len(outer_roles)),
        file_record(TRANSFORM_MANIFEST, "fold_transform_manifest"),
        file_record(FINAL_SELECTION, "final_model_selection"),
        file_record(FOLD0_RUN_CONFIG, "final_model_training_config"),
        *role_input_records,
    ]
    manifest = {
        "schema_version": "r3_9.fold_age_distribution.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "reviewer_R3_9_fold_age_distribution_and_role_audit",
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "contract": {
            "participant_level_folds": 5,
            "per_rotation": "three training folds, one calibration fold, one outer-test fold",
            "future_outcomes_in_assignment": False,
            "model_refit_for_this_audit": False,
            "statistical_model_fit_for_this_audit": False,
        },
        "key_results": summary,
        "inputs": input_records,
        "script": file_record(SCRIPT_PATH, "r3_9_age_distribution_builder"),
        "outputs": output_records,
    }
    manifest_path = OUTPUT_ROOT / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: {manifest_path}")


if __name__ == "__main__":
    main()
