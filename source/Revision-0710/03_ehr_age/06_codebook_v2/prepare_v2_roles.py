from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REVISION_ROOT = WORKSPACE_ROOT / "Revision-0710"
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE / "00_roles"

OUTCOMES_PATH = (
    REVISION_ROOT / "01_data_rebuild" / "09_v2_labels" / "organ_outcomes_60m.parquet"
)
FOLDS_PATH = (
    REVISION_ROOT / "01_data_rebuild" / "03_folds" / "fold_manifest_5fold.parquet"
)
INNER_SPLIT_PATH = (
    REVISION_ROOT
    / "01_data_rebuild"
    / "03_folds"
    / "fold0_validation_calibration_screen.parquet"
)
AGE_TARGETS_PATH = (
    REVISION_ROOT
    / "01_data_rebuild"
    / "04_cache_v1"
    / "targets"
    / "chronological_age_targets.parquet"
)
CACHE_BATCH_MANIFEST = (
    REVISION_ROOT / "01_data_rebuild" / "04_cache_v1" / "batch_manifest.json"
)

DEVELOPMENT_LABELS = ROLE_ROOT / "fold0_development_labels.parquet"
DEVELOPMENT_AGES = ROLE_ROOT / "fold0_development_ages.parquet"
OPEN_LABELS = ROLE_ROOT / "fold0_open_labels.parquet"
OPEN_AGES = ROLE_ROOT / "fold0_open_ages.parquet"
SUMMARY_PATH = ROLE_ROOT / "role_summary.json"

ORGAN_SYSTEMS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
LABEL_COLUMNS = (
    "participant_id",
    "fold_id",
    "inner_role",
    "organ_system",
    "duration_months",
    "event",
    "at_risk",
    "strict_healthy_reference_60m",
)
AGE_COLUMNS = ("participant_id", "fold_id", "chronological_age")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sort_labels(frame: pd.DataFrame) -> pd.DataFrame:
    organ_order = {organ: index for index, organ in enumerate(ORGAN_SYSTEMS)}
    work = frame.assign(_organ_order=frame["organ_system"].map(organ_order))
    if work["_organ_order"].isna().any():
        raise ValueError("labels contain an unknown organ")
    return (
        work.sort_values(["fold_id", "participant_id", "_organ_order"])
        .drop(columns="_organ_order")
        .loc[:, LABEL_COLUMNS]
        .reset_index(drop=True)
    )


def validate_roles(role_root: Path = ROLE_ROOT) -> dict[str, Any]:
    paths = {
        "development_labels": role_root / DEVELOPMENT_LABELS.name,
        "development_ages": role_root / DEVELOPMENT_AGES.name,
        "open_labels": role_root / OPEN_LABELS.name,
        "open_ages": role_root / OPEN_AGES.name,
        "summary": role_root / SUMMARY_PATH.name,
    }
    if not all(path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise FileNotFoundError(f"missing v2 role artifacts: {missing}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError("v2 role summary is not PASS")
    for name in ("development_labels", "development_ages", "open_labels", "open_ages"):
        if summary["artifacts"][name]["sha256"] != sha256_file(paths[name]):
            raise ValueError(f"v2 role artifact hash changed: {name}")

    dev_labels = pd.read_parquet(paths["development_labels"])
    dev_ages = pd.read_parquet(paths["development_ages"])
    open_labels = pd.read_parquet(paths["open_labels"])
    open_ages = pd.read_parquet(paths["open_ages"])
    if tuple(dev_labels.columns) != LABEL_COLUMNS or tuple(open_labels.columns) != LABEL_COLUMNS:
        raise ValueError("v2 label schema changed")
    if tuple(dev_ages.columns) != AGE_COLUMNS or tuple(open_ages.columns) != AGE_COLUMNS:
        raise ValueError("v2 age schema changed")
    if set(dev_labels["fold_id"].astype(int)) != {1, 2, 3, 4}:
        raise ValueError("development labels do not contain exactly folds 1-4")
    if set(open_labels["fold_id"].astype(int)) != {0}:
        raise ValueError("open labels are not Fold0 only")
    if set(dev_ages["fold_id"].astype(int)) != {1, 2, 3, 4}:
        raise ValueError("development ages do not contain exactly folds 1-4")
    if set(open_ages["fold_id"].astype(int)) != {0}:
        raise ValueError("open ages are not Fold0 only")
    dev_ids = set(dev_ages["participant_id"].astype(str))
    open_ids = set(open_ages["participant_id"].astype(str))
    if dev_ids.intersection(open_ids):
        raise ValueError("development and open Fold0 participants overlap")
    if len(dev_labels) != len(dev_ids) * len(ORGAN_SYSTEMS):
        raise ValueError("development labels do not contain seven rows per participant")
    if len(open_labels) != len(open_ids) * len(ORGAN_SYSTEMS):
        raise ValueError("Fold0 labels do not contain seven rows per participant")
    if set(dev_labels["organ_system"].astype(str)) != set(ORGAN_SYSTEMS):
        raise ValueError("development labels changed the seven-organ axis")
    if set(open_labels["organ_system"].astype(str)) != set(ORGAN_SYSTEMS):
        raise ValueError("Fold0 labels changed the seven-organ axis")
    if (dev_labels["event"].eq(1) & dev_labels["at_risk"].ne(1)).any():
        raise ValueError("development event exists outside the risk set")
    if (open_labels["event"].eq(1) & open_labels["at_risk"].ne(1)).any():
        raise ValueError("Fold0 event exists outside the risk set")
    return {
        "status": "PASS",
        "development_participants": len(dev_ids),
        "open_fold0_participants": len(open_ids),
        "development_folds": [1, 2, 3, 4],
        "open_folds": [0],
        "participant_overlap": 0,
    }


def build_roles() -> dict[str, Any]:
    folds = pd.read_parquet(FOLDS_PATH, columns=["participant_id", "fold_id"])
    ages = pd.read_parquet(AGE_TARGETS_PATH, columns=list(AGE_COLUMNS))
    inner = pd.read_parquet(
        INNER_SPLIT_PATH, columns=["participant_id", "fold_id", "inner_role"]
    )
    outcomes = pd.read_parquet(
        OUTCOMES_PATH,
        columns=[
            "participant_id",
            "organ_system",
            "duration_months",
            "event",
            "at_risk",
            "strict_healthy_reference_60m",
        ],
    )
    for frame in (folds, ages, inner, outcomes):
        frame["participant_id"] = frame["participant_id"].astype(str)
    folds["fold_id"] = folds["fold_id"].astype(np.int8)
    ages["fold_id"] = ages["fold_id"].astype(np.int8)
    inner["fold_id"] = inner["fold_id"].astype(np.int8)

    if folds["participant_id"].duplicated().any() or ages["participant_id"].duplicated().any():
        raise ValueError("fold or age source contains duplicate participants")
    if inner["participant_id"].duplicated().any() or set(inner["fold_id"].astype(int)) != {1}:
        raise ValueError("inner split must contain each Fold1 participant once")
    if set(inner["inner_role"].astype(str)) != {"calibration", "screen"}:
        raise ValueError("inner split must contain calibration and screen")
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise ValueError("v2 outcomes contain duplicate participant-organ rows")
    if set(outcomes["organ_system"].astype(str)) != set(ORGAN_SYSTEMS):
        raise ValueError("v2 outcomes changed the seven-organ axis")

    fold_age = folds.merge(
        ages,
        on=["participant_id", "fold_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not fold_age["_merge"].eq("both").all():
        raise ValueError("fold and chronological-age coverage differs")
    joined = outcomes.merge(folds, on="participant_id", how="inner", validate="many_to_one")
    if len(joined) != len(folds) * len(ORGAN_SYSTEMS):
        raise ValueError("v2 outcome/fold join is incomplete")

    role_by_id = inner.set_index("participant_id")["inner_role"]
    joined["inner_role"] = joined["participant_id"].map(role_by_id)
    joined.loc[joined["fold_id"].isin([2, 3, 4]), "inner_role"] = "train"
    dev_labels = joined.loc[joined["fold_id"].isin([1, 2, 3, 4])].copy()
    if dev_labels["inner_role"].isna().any():
        raise ValueError("some Fold1 labels lack a calibration/screen role")
    dev_labels = _sort_labels(dev_labels)

    open_labels = joined.loc[joined["fold_id"].eq(0)].copy()
    open_labels["inner_role"] = "outer_test"
    open_labels = _sort_labels(open_labels)
    dev_ages = (
        ages.loc[ages["fold_id"].isin([1, 2, 3, 4]), AGE_COLUMNS]
        .sort_values(["fold_id", "participant_id"])
        .reset_index(drop=True)
    )
    open_ages = (
        ages.loc[ages["fold_id"].eq(0), AGE_COLUMNS]
        .sort_values("participant_id")
        .reset_index(drop=True)
    )

    _atomic_parquet(DEVELOPMENT_LABELS, dev_labels)
    _atomic_parquet(DEVELOPMENT_AGES, dev_ages)
    _atomic_parquet(OPEN_LABELS, open_labels)
    _atomic_parquet(OPEN_AGES, open_ages)

    fold_counts = {
        str(int(fold)): int(count)
        for fold, count in ages.groupby("fold_id", observed=True).size().items()
    }
    summary: dict[str, Any] = {
        "schema_version": "revision0710.ehr_age.codebook_v2.roles.v1",
        "status": "PASS",
        "phase": "trusted_fixed_preprocessing_before_model_development",
        "model_development_must_read_only_materialized_artifacts": True,
        "outer_test_fold_present": False,
        "outer_test_participant_rows": 0,
        "definition": {
            "train_folds": [2, 3, 4],
            "calibration_and_screen_fold": 1,
            "open_development_fold": 0,
            "model_inputs": "unchanged Revision-0710 compact L32 cache",
            "outcomes": "corrected ICD codebook v2 60-month incident endpoints",
        },
        "counts": {
            "participants_by_fold": fold_counts,
            "development_participants": int(len(dev_ages)),
            "development_participant_organ_rows": int(len(dev_labels)),
            "open_fold0_participants": int(len(open_ages)),
            "open_fold0_participant_organ_rows": int(len(open_labels)),
            "development_strict_health": int(
                dev_labels.groupby("participant_id", observed=True)[
                    "strict_healthy_reference_60m"
                ].first().sum()
            ),
            "open_fold0_strict_health": int(
                open_labels.groupby("participant_id", observed=True)[
                    "strict_healthy_reference_60m"
                ].first().sum()
            ),
        },
        "sources": {
            "organ_outcomes_60m_v2": _artifact(OUTCOMES_PATH),
            "fold_manifest_5fold": _artifact(FOLDS_PATH),
            "fold1_inner_split": _artifact(INNER_SPLIT_PATH),
            "chronological_age_targets": _artifact(AGE_TARGETS_PATH),
            "unchanged_cache_batch_manifest": _artifact(CACHE_BATCH_MANIFEST),
        },
        "artifacts": {
            "development_labels": _artifact(DEVELOPMENT_LABELS),
            "development_ages": _artifact(DEVELOPMENT_AGES),
            "open_labels": _artifact(OPEN_LABELS),
            "open_ages": _artifact(OPEN_AGES),
            # Compatibility names used by the existing leakage-safe loader.
            "joint_labels": _artifact(DEVELOPMENT_LABELS),
            "age_targets": _artifact(DEVELOPMENT_AGES),
        },
        "leakage_checks": {
            "development_contains_fold0": False,
            "development_open_participant_overlap": 0,
            "future_icd_used_as_model_input": False,
            "input_cache_rebuilt_or_copied": False,
        },
    }
    _atomic_json(SUMMARY_PATH, summary)
    validation = validate_roles(ROLE_ROOT)
    return {**validation, "summary_path": str(SUMMARY_PATH.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal codebook-v2 Fold0 role artifacts.")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_roles(ROLE_ROOT) if args.validate_only else build_roles()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
