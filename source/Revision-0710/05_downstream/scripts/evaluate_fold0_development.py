#!/usr/bin/env python
"""Evaluate one baseline on Fold 0 validation calibration-to-screen only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = SCRIPT_DIR.parent
REVISION_DIR = DOWNSTREAM_DIR.parent
COMMON_DIR = DOWNSTREAM_DIR / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from crossfit_metrics import (  # noqa: E402
    ORGANS,
    aggregate_cox_cells,
    apply_linear_age_correction,
    apply_linear_age_correction_to_validation,
    evaluate_cox_cell,
    fit_linear_age_correction,
    raw_age_metrics,
)
from evaluate_method_5fold import (  # noqa: E402
    EXPORTED_AGE_ATOL_YEARS,
    EvaluationContractError,
    _explicit_file,
    _normalize_pid,
    _numeric,
    _require_columns,
    _sha256_file,
    load_organ_outcomes,
    load_participant_masks,
)


OUTPUT_FILES = (
    "raw_screen_strict_age_metrics.csv",
    "calibration_linear_correction_parameters.csv",
    "calibration_to_screen_cox_organ_cells.csv",
    "evaluation_summary.json",
    "evaluation_manifest.json",
)


def _validate_expected_sha256(value: str, *, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise EvaluationContractError(f"{label} must be a 64-character SHA256")
    return normalized


def _verify_sha256(path: Path, expected: str, *, label: str) -> str:
    expected = _validate_expected_sha256(expected, label=label)
    observed = _sha256_file(path)
    if observed != expected:
        raise EvaluationContractError(
            f"{label} mismatch: expected={expected}, observed={observed}, path={path}"
        )
    return observed


def load_fold0_validation_predictions(path: Path) -> tuple[pd.DataFrame, Path]:
    path = _explicit_file(path, label="validation_predictions", suffix=".parquet")
    frame = _normalize_pid(pd.read_parquet(path), table="validation_predictions")
    required = (
        "participant_id",
        "organ_system",
        "actual_age",
        "predicted_age",
        "outer_fold",
        "split_role",
    )
    _require_columns(frame, required, table="validation_predictions")
    frame = frame.loc[:, required].copy()
    if frame.empty:
        raise EvaluationContractError("validation_predictions must not be empty")
    frame["actual_age"] = _numeric(frame, "actual_age", table="validation_predictions")
    frame["predicted_age"] = _numeric(frame, "predicted_age", table="validation_predictions")
    outer_fold = _numeric(frame, "outer_fold", table="validation_predictions")
    if not outer_fold.eq(0).all():
        raise EvaluationContractError(
            "Fold0 development accepts outer_fold=0 validation predictions only; outer test is forbidden"
        )
    if not frame["split_role"].astype(str).eq("validation").all():
        raise EvaluationContractError(
            "Fold0 development accepts split_role='validation' only; outer test is forbidden"
        )
    frame["outer_fold"] = outer_fold.astype(np.int8)
    frame["split_role"] = frame["split_role"].astype(str)
    frame["organ_system"] = frame["organ_system"].astype(str)
    if set(frame["organ_system"]) != set(ORGANS):
        raise EvaluationContractError("validation_predictions must contain the seven canonical organs")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise EvaluationContractError("validation_predictions has duplicate participant-organ rows")
    return frame, path


def load_inner_roles(path: Path) -> tuple[pd.DataFrame, Path]:
    path = _explicit_file(path, label="inner_roles", suffix=".parquet")
    frame = _normalize_pid(pd.read_parquet(path), table="inner_roles")
    required = ("participant_id", "fold_id", "inner_role")
    _require_columns(frame, required, table="inner_roles")
    frame = frame.loc[:, required].copy()
    if frame.empty or frame.duplicated("participant_id").any():
        raise EvaluationContractError("inner_roles must contain unique non-empty participant rows")
    fold_id = _numeric(frame, "fold_id", table="inner_roles")
    if not fold_id.eq(1).all():
        raise EvaluationContractError("Fold0 validation inner roles must contain fold_id=1 only")
    frame["fold_id"] = fold_id.astype(np.int8)
    frame["inner_role"] = frame["inner_role"].astype(str)
    if set(frame["inner_role"]) != {"calibration", "screen"}:
        raise EvaluationContractError("inner_roles must contain calibration and screen only")
    return frame, path


def enrich_development_predictions(
    predictions: pd.DataFrame,
    participants: pd.DataFrame,
    inner_roles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_ids = set(predictions["participant_id"])
    inner_ids = set(inner_roles["participant_id"])
    if prediction_ids != inner_ids:
        raise EvaluationContractError(
            "validation prediction participant coverage must exactly equal fold0 inner-role coverage"
        )
    coverage = predictions.groupby("participant_id")["organ_system"].agg(["size", "nunique"])
    if (
        len(predictions) != len(inner_roles) * len(ORGANS)
        or not coverage["size"].eq(len(ORGANS)).all()
        or not coverage["nunique"].eq(len(ORGANS)).all()
    ):
        raise EvaluationContractError("every Fold0 validation participant must have seven organ predictions")

    participant_subset = participants.loc[participants["participant_id"].isin(inner_ids)].copy()
    if set(participant_subset["participant_id"]) != inner_ids:
        raise EvaluationContractError("participant masks do not cover every Fold0 validation participant")
    enriched = predictions.merge(
        participant_subset,
        on="participant_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_participant"),
    ).merge(
        inner_roles.loc[:, ["participant_id", "inner_role"]],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    exported_age = enriched["actual_age"].to_numpy(dtype=np.float64)
    authoritative_age = enriched["chronological_age"].to_numpy(dtype=np.float64)
    age_difference = np.abs(exported_age - authoritative_age)
    if (age_difference > EXPORTED_AGE_ATOL_YEARS).any():
        raise EvaluationContractError(
            "validation_predictions.actual_age disagrees with authoritative age beyond "
            f"atol={EXPORTED_AGE_ATOL_YEARS:g}"
        )
    enriched["exported_actual_age"] = exported_age
    enriched["actual_age"] = authoritative_age
    enriched = enriched.drop(columns="chronological_age")
    role_counts = inner_roles["inner_role"].value_counts()
    return enriched, {
        "prediction_rows": int(len(predictions)),
        "validation_participants": int(len(inner_roles)),
        "calibration_participants": int(role_counts["calibration"]),
        "screen_participants": int(role_counts["screen"]),
        "max_exported_age_abs_difference_years": float(age_difference.max()),
    }


def evaluate_development_method(
    *,
    method: str,
    predictions: pd.DataFrame,
    participants: pd.DataFrame,
    outcomes: pd.DataFrame,
    inner_roles: pd.DataFrame,
) -> dict[str, object]:
    enriched, coverage = enrich_development_predictions(predictions, participants, inner_roles)
    calibration = enriched.loc[enriched["inner_role"].eq("calibration")].copy()
    screen = enriched.loc[enriched["inner_role"].eq("screen")].copy()
    calibration_ids = set(calibration["participant_id"])
    screen_ids = set(screen["participant_id"])
    if calibration_ids & screen_ids:
        raise EvaluationContractError("calibration and screen participants must be disjoint")

    screen_strict = screen.loc[screen["strict_healthy_reference_60m"].eq(1)].copy()
    if screen_strict.empty:
        raise EvaluationContractError("screen strict healthy reference cohort is empty")
    age_metrics = raw_age_metrics(screen_strict)

    correction_model = fit_linear_age_correction(calibration)
    calibration_corrected = apply_linear_age_correction_to_validation(calibration, correction_model)
    screen_corrected = apply_linear_age_correction(screen, correction_model)
    correction_rows = [
        {"method": method, "outer_fold": 0, "fit_role": "calibration", **asdict(parameter)}
        for parameter in correction_model.parameters
    ]

    risk_outcomes = outcomes.loc[
        outcomes["at_risk"].eq(1),
        ["participant_id", "organ_system", "duration_months", "event"],
    ].rename(columns={"duration_months": "duration"})
    cox_rows: list[dict[str, object]] = []
    for organ in ORGANS:
        organ_outcomes = risk_outcomes.loc[risk_outcomes["organ_system"].eq(organ)]
        calibration_risk = calibration_corrected.loc[
            calibration_corrected["organ_system"].eq(organ),
            ["participant_id", "organ_system", "actual_age", "sex", "corrected_age_gap"],
        ].merge(
            organ_outcomes,
            on=["participant_id", "organ_system"],
            how="inner",
            validate="one_to_one",
        )
        screen_risk = screen_corrected.loc[
            screen_corrected["organ_system"].eq(organ),
            ["participant_id", "organ_system", "actual_age", "sex", "corrected_age_gap"],
        ].merge(
            organ_outcomes,
            on=["participant_id", "organ_system"],
            how="inner",
            validate="one_to_one",
        )
        cell = evaluate_cox_cell(
            calibration_risk,
            screen_risk,
            fold=0,
            organ_system=organ,
        )
        cox_rows.append({"method": method, **asdict(cell)})

    cox_cells = pd.DataFrame(cox_rows)
    cox_aggregate = aggregate_cox_cells(cox_cells, folds=(0,))
    return {
        "age_metrics": age_metrics,
        "correction_parameters": pd.DataFrame(correction_rows),
        "cox_cells": cox_cells,
        "cox_aggregate": cox_aggregate,
        "coverage": coverage,
    }


def write_outputs(
    result: dict[str, object],
    *,
    method: str,
    output_dir: Path,
    validation_predictions: Path,
    participant_masks: Path,
    organ_outcomes: Path,
    inner_roles: Path,
    source_hashes: dict[str, str],
    overwrite: bool,
) -> None:
    if any(token in str(output_dir) for token in ("*", "?", "[", "]")):
        raise EvaluationContractError("output_dir must be explicit and contain no glob tokens")
    output_dir = output_dir.expanduser().resolve()
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise EvaluationContractError(f"output files already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    age_metrics = result["age_metrics"]
    pd.DataFrame(
        [
            {
                "method": method,
                "cohort": "screen_strict_healthy_reference_60m",
                **asdict(age_metrics),
            }
        ]
    ).to_csv(output_dir / "raw_screen_strict_age_metrics.csv", index=False)
    result["correction_parameters"].to_csv(
        output_dir / "calibration_linear_correction_parameters.csv", index=False
    )
    cells_for_export = result["cox_cells"].rename(
        columns={
            "n_validation": "n_calibration",
            "n_test": "n_screen",
            "validation_events": "calibration_events",
            "test_events": "screen_events",
        }
    )
    cells_for_export.to_csv(output_dir / "calibration_to_screen_cox_organ_cells.csv", index=False)

    summary = {
        "schema_version": "revision0710_fold0_development_evaluation_summary_v1",
        "method": method,
        "scope": {
            "outer_fold": 0,
            "source_split": "validation",
            "source_fold_id": 1,
            "age_metric_population": "screen_strict_healthy_reference_60m",
            "age_correction_fit": "calibration_strict_healthy_reference_60m",
            "cox_fit": "calibration_full_incident_risk_set",
            "cox_score": "screen_full_incident_risk_set",
            "outer_test_used": False,
        },
        "age_metrics": {
            "screen_strict_healthy_reference_60m": asdict(age_metrics),
        },
        "cox": asdict(result["cox_aggregate"]),
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "revision0710_fold0_development_evaluation_manifest_v1",
        "method": method,
        "scope": summary["scope"],
        "source_files": {
            "validation_predictions": str(validation_predictions),
            "participant_masks": str(participant_masks.expanduser().resolve()),
            "organ_outcomes": str(organ_outcomes.expanduser().resolve()),
            "fold0_validation_calibration_screen": str(inner_roles),
        },
        "source_sha256": source_hashes,
        "output_sha256": {
            "evaluation_summary": _sha256_file(summary_path),
        },
        "coverage": result["coverage"],
        "contract": {
            "outer_test_used": False,
            "fit_role": "calibration",
            "score_role": "screen",
            "participant_disjoint_fit_score": True,
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--expected-predictions-sha256", required=True)
    parser.add_argument("--inner-roles", type=Path, required=True)
    parser.add_argument("--expected-inner-roles-sha256", required=True)
    parser.add_argument(
        "--participant-masks",
        type=Path,
        default=REVISION_DIR / "01_data_rebuild/02_outcomes_60m/participant_followup_and_evaluation_masks.parquet",
    )
    parser.add_argument(
        "--organ-outcomes",
        type=Path,
        default=REVISION_DIR / "01_data_rebuild/02_outcomes_60m/organ_outcomes_60m.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions, predictions_path = load_fold0_validation_predictions(args.validation_predictions)
    inner_roles, inner_roles_path = load_inner_roles(args.inner_roles)
    predictions_sha = _verify_sha256(
        predictions_path, args.expected_predictions_sha256, label="expected_predictions_sha256"
    )
    inner_roles_sha = _verify_sha256(
        inner_roles_path, args.expected_inner_roles_sha256, label="expected_inner_roles_sha256"
    )
    participant_masks_path = _explicit_file(
        args.participant_masks, label="participant_masks", suffix=".parquet"
    )
    organ_outcomes_path = _explicit_file(args.organ_outcomes, label="organ_outcomes", suffix=".parquet")
    participants = load_participant_masks(participant_masks_path)
    outcomes = load_organ_outcomes(organ_outcomes_path)
    result = evaluate_development_method(
        method=args.method,
        predictions=predictions,
        participants=participants,
        outcomes=outcomes,
        inner_roles=inner_roles,
    )
    source_hashes = {
        "validation_predictions": predictions_sha,
        "fold0_validation_calibration_screen": inner_roles_sha,
        "participant_masks": _sha256_file(participant_masks_path),
        "organ_outcomes": _sha256_file(organ_outcomes_path),
        "evaluator_code": _sha256_file(Path(__file__).resolve()),
    }
    write_outputs(
        result,
        method=args.method,
        output_dir=args.output_dir,
        validation_predictions=predictions_path,
        participant_masks=participant_masks_path,
        organ_outcomes=organ_outcomes_path,
        inner_roles=inner_roles_path,
        source_hashes=source_hashes,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
