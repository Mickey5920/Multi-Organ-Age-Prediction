#!/usr/bin/env python3
"""Run the frozen final-model clinical-utility analysis with the established metric contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd


ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)


def workspace_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "Revision-0710").is_dir():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
REVISION = ROOT / "Revision-0710"
LEGACY_RUNNER = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics/03_clinical_utility"
    / "01_formal_run/scripts/run_0710_clinical_utility.py"
)
CONTEXT_PICKLE = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics/03_clinical_utility"
    / "01_formal_run/02_input_cache/participant_context_covariates.pkl"
)
FINAL_MODEL_ROOT = (
    REVISION / "03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1"
)
FOLD0_ROOT = (
    REVISION
    / "03_ehr_age/06_codebook_v2/01_runs"
    / "shared_private_independent_view_huber1_fold0"
)
SELECTION = FINAL_MODEL_ROOT / "FINAL_SELECTION.json"
DEFAULT_OUTPUT = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/13_final_model_clinical_utility/01_point"
)
CACHE_ROOT = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/13_final_model_clinical_utility/00_input_cache"
)
PRIMARY_TIER = "C_context_density_primary"
PENALIZER = 1e-4


def load_established_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("established_clinical_utility", LEGACY_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import established runner: {LEGACY_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_paths(outer_fold: int) -> tuple[Path, Path]:
    if outer_fold == 0:
        return (
            FOLD0_ROOT / "fold1_calibration_predictions.parquet",
            FOLD0_ROOT / "fold0_predictions.parquet",
        )
    fold_root = FINAL_MODEL_ROOT / "01_outer_folds" / f"outer_fold_{outer_fold}"
    return (
        fold_root / "calibration_predictions.parquet",
        fold_root / "outer_predictions.parquet",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(outer_fold: int, role: str) -> Path:
    return CACHE_ROOT / f"outer_fold_{outer_fold}__{role}.pkl"


def validate_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    if set(frame["organ_system"].astype(str)) != set(ORGANS):
        raise RuntimeError(f"organ coverage mismatch in {path}")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError(f"duplicate participant-organ rows in {path}")
    counts = frame.groupby("participant_id", observed=True)["organ_system"].nunique()
    if not counts.eq(len(ORGANS)).all():
        raise RuntimeError(f"incomplete seven-organ predictions in {path}")


def prepare_cache() -> None:
    required = [
        "participant_id",
        "organ_system",
        "duration",
        "event",
        "at_risk",
        "corrected_age_gap",
    ]
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for outer_fold in range(5):
        calibration, outer_test = prediction_paths(outer_fold)
        for role, source in (("calibration", calibration), ("outer_test", outer_test)):
            frame = pd.read_parquet(source, columns=required)
            frame["participant_id"] = frame["participant_id"].astype(str)
            validate_prediction_frame(frame, source)
            output = cache_path(outer_fold, role)
            frame.to_pickle(output)
            manifest_rows.append(
                {
                    "outer_fold": outer_fold,
                    "role": role,
                    "source_path": source.as_posix(),
                    "source_bytes": source.stat().st_size,
                    "source_sha256": sha256_file(source),
                    "cache_path": output.as_posix(),
                    "cache_bytes": output.stat().st_size,
                    "cache_sha256": sha256_file(output),
                    "rows": len(frame),
                    "unique_participants": frame["participant_id"].nunique(),
                    "columns": list(frame.columns),
                }
            )
    payload = {
        "schema_version": "revision0710.final_model_clinical_utility.input_cache.v1",
        "status": "PASS",
        "transformation": "column-preserving parquet-to-pickle cache",
        "rows": manifest_rows,
    }
    (CACHE_ROOT / "INPUT_CACHE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_prediction_frame(path: Path, outer_fold: int, expected_role: str) -> pd.DataFrame:
    role = "calibration" if expected_role == "calibration" else "outer_test"
    cached = cache_path(outer_fold, role)
    if not cached.exists():
        raise FileNotFoundError(f"prepared prediction cache not found: {cached}")
    frame = pd.read_pickle(cached)
    frame = frame.rename(columns={"duration": "duration_months"})
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["outer_fold"] = int(outer_fold)
    frame["split_role"] = str(expected_role)
    validate_prediction_frame(frame, path)
    return frame


def configure_runner(module: ModuleType) -> None:
    module.prediction_paths = prediction_paths
    module.load_prediction_frame = load_prediction_frame


def load_context() -> pd.DataFrame:
    context = pd.read_pickle(CONTEXT_PICKLE)
    context["participant_id"] = context["participant_id"].astype(str)
    if len(context) != 456_773 or context["participant_id"].duplicated().any():
        raise RuntimeError("prepared contextual covariate contract failed")
    return context


def run_fold(module: ModuleType, outer_fold: int, output_dir: Path) -> None:
    calibration, outer_test = prediction_paths(outer_fold)
    for source in (calibration, outer_test, CONTEXT_PICKLE, SELECTION):
        if not source.exists():
            raise FileNotFoundError(source)
    module.run_fold(
        outer_fold,
        load_context(),
        [PRIMARY_TIER],
        output_dir=output_dir,
        penalizer=PENALIZER,
    )


def aggregate(module: ModuleType, output_dir: Path) -> None:
    folds = list(range(5))
    combined = module.concatenate_folds(output_dir, folds)
    module.aggregate_results(output_dir, combined)
    sources = []
    for fold in folds:
        calibration, outer_test = prediction_paths(fold)
        sources.append(
            {
                "outer_fold": fold,
                "calibration_predictions": {
                    "path": calibration.as_posix(),
                    "bytes": calibration.stat().st_size,
                    "sha256": module.sha256_file(calibration),
                },
                "outer_test_predictions": {
                    "path": outer_test.as_posix(),
                    "bytes": outer_test.stat().st_size,
                    "sha256": module.sha256_file(outer_test),
                },
            }
        )
    receipt = {
        "schema_version": "revision0710.final_model_clinical_utility.point.v1",
        "status": "COMPLETE",
        "analysis_role": "final_frozen_model_primary_clinical_density_baseline",
        "selection": {
            "path": SELECTION.as_posix(),
            "sha256": module.sha256_file(SELECTION),
        },
        "established_runner": {
            "path": LEGACY_RUNNER.as_posix(),
            "sha256": module.sha256_file(LEGACY_RUNNER),
        },
        "context_pickle": {
            "path": CONTEXT_PICKLE.as_posix(),
            "bytes": CONTEXT_PICKLE.stat().st_size,
            "sha256": module.sha256_file(CONTEXT_PICKLE),
        },
        "folds": folds,
        "tier": PRIMARY_TIER,
        "organs": list(ORGANS),
        "augmentations": list(module.AUGMENTATIONS),
        "penalizer": PENALIZER,
        "horizons_months": list(module.REPORT_TIMES),
        "decision_curve_thresholds": list(module.THRESHOLDS),
        "decision_curve_interpretation": "exploratory fixed continuity grid",
        "death_handling": "cause-specific censoring",
        "competing_risk_claim_allowed": False,
        "prediction_sources": sources,
        "environment": module.environment_receipt(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "FINAL_MODEL_POINT_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-cache", "run-fold", "aggregate"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-cache":
        prepare_cache()
        return
    module = load_established_runner()
    configure_runner(module)
    output_dir = args.output_dir.resolve()
    if args.command == "run-fold":
        if args.fold is None:
            raise ValueError("--fold is required for run-fold")
        run_fold(module, args.fold, output_dir)
    else:
        aggregate(module, output_dir)


if __name__ == "__main__":
    main()
