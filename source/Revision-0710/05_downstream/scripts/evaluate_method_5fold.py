#!/usr/bin/env python
"""Evaluate one Revision-0710 method with strict five-fold cross-fitting."""

from __future__ import annotations

import argparse
import hashlib
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
    ContractError,
    aggregate_cox_cells,
    apply_linear_age_correction,
    apply_linear_age_correction_to_validation,
    evaluate_cox_cell,
    fit_linear_age_correction,
    raw_age_metrics,
)


PREDICTION_COLUMNS = (
    "participant_id",
    "organ_system",
    "actual_age",
    "predicted_age",
    "outer_fold",
    "split_role",
)
EXPORTED_AGE_ATOL_YEARS = 1e-5
OUTPUT_FILES = (
    "raw_age_metrics.csv",
    "linear_correction_parameters.csv",
    "cox_fold_organ_cells.csv",
    "cox_organ_summary.csv",
    "cox_summary.json",
    "evaluation_summary.json",
    "evaluation_manifest.json",
)


class EvaluationContractError(ContractError):
    """Raised when files or cross-fold coverage violate the evaluation contract."""


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, table: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise EvaluationContractError(f"{table} missing required columns: {missing}")


def _normalize_pid(frame: pd.DataFrame, *, table: str) -> pd.DataFrame:
    _require_columns(frame, ("participant_id",), table=table)
    result = frame.copy()
    invalid = result["participant_id"].isna() | result["participant_id"].astype(str).str.strip().eq("")
    if invalid.any():
        raise EvaluationContractError(f"{table}.participant_id contains null or empty values")
    result["participant_id"] = result["participant_id"].astype(str).str.strip()
    return result


def _numeric(frame: pd.DataFrame, column: str, *, table: str, finite: bool = True) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or (finite and not np.isfinite(values.to_numpy(dtype=np.float64)).all()):
        raise EvaluationContractError(f"{table}.{column} must contain finite numeric values")
    return values


def _explicit_file(path: Path, *, label: str, suffix: str | None = None) -> Path:
    raw = str(path)
    if any(token in raw for token in ("*", "?", "[", "]")):
        raise EvaluationContractError(f"{label} must be an explicit path without glob tokens")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationContractError(f"{label} does not exist or is not a file: {resolved}")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise EvaluationContractError(f"{label} must use {suffix}: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_prediction_files(
    *,
    method: str,
    prediction_registry: Path | None,
    predictions: Path | None,
) -> tuple[list[Path], Path | None]:
    if (prediction_registry is None) == (predictions is None):
        raise EvaluationContractError("provide exactly one of prediction_registry or predictions")
    if predictions is not None:
        return [_explicit_file(predictions, label="predictions", suffix=".parquet")], None

    registry_path = _explicit_file(prediction_registry, label="prediction_registry", suffix=".json")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"invalid prediction registry JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationContractError("prediction registry must be a JSON object")
    if payload.get("schema_version") != "revision0710_prediction_registry_v1":
        raise EvaluationContractError("unsupported prediction registry schema_version")
    if payload.get("method") != method:
        raise EvaluationContractError(
            f"prediction registry method mismatch: expected={method!r}, observed={payload.get('method')!r}"
        )
    entries = payload.get("prediction_files")
    if not isinstance(entries, list) or not entries:
        raise EvaluationContractError("prediction registry prediction_files must be a non-empty list")
    resolved_files: list[Path] = []
    for index, entry in enumerate(entries):
        expected_sha256: str | None = None
        if isinstance(entry, str):
            entry_path = entry
        elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
            entry_path = entry["path"]
            if "sha256" in entry:
                expected_sha256 = entry["sha256"]
                if (
                    not isinstance(expected_sha256, str)
                    or len(expected_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in expected_sha256.lower())
                ):
                    raise EvaluationContractError(
                        f"prediction registry entry {index} has an invalid sha256"
                    )
                expected_sha256 = expected_sha256.lower()
        else:
            raise EvaluationContractError(f"prediction registry entry {index} must contain a string path")
        path = Path(entry_path)
        if not path.is_absolute():
            path = registry_path.parent / path
        resolved = _explicit_file(path, label=f"prediction_files[{index}]", suffix=".parquet")
        if expected_sha256 is not None:
            observed_sha256 = _sha256_file(resolved)
            if observed_sha256 != expected_sha256:
                raise EvaluationContractError(
                    f"prediction_files[{index}] sha256 mismatch: "
                    f"expected={expected_sha256}, observed={observed_sha256}"
                )
        resolved_files.append(resolved)
    if len(set(resolved_files)) != len(resolved_files):
        raise EvaluationContractError("prediction registry contains duplicate file paths")
    return resolved_files, registry_path


def load_predictions(paths: Sequence[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        _require_columns(frame, PREDICTION_COLUMNS, table=f"predictions:{path.name}")
        parts.append(frame.loc[:, PREDICTION_COLUMNS].copy())
    predictions = _normalize_pid(pd.concat(parts, ignore_index=True), table="predictions")
    if predictions.empty:
        raise EvaluationContractError("predictions must not be empty")
    predictions["actual_age"] = _numeric(predictions, "actual_age", table="predictions")
    predictions["predicted_age"] = _numeric(predictions, "predicted_age", table="predictions")
    outer_fold = _numeric(predictions, "outer_fold", table="predictions")
    if not np.equal(outer_fold, np.floor(outer_fold)).all() or not outer_fold.isin(range(5)).all():
        raise EvaluationContractError("predictions.outer_fold must contain exactly fold IDs 0..4")
    predictions["outer_fold"] = outer_fold.astype(np.int8)
    predictions["organ_system"] = predictions["organ_system"].astype(str)
    predictions["split_role"] = predictions["split_role"].astype(str)
    if set(predictions["organ_system"]) != set(ORGANS):
        raise EvaluationContractError("predictions.organ_system must contain the seven canonical organs")
    if set(predictions["split_role"]) != {"validation", "test"}:
        raise EvaluationContractError("predictions.split_role must contain validation and test only")
    if predictions.duplicated(["participant_id", "organ_system", "outer_fold", "split_role"]).any():
        raise EvaluationContractError("predictions has duplicate participant-organ-fold-role rows")
    return predictions


def load_participant_masks(path: Path) -> pd.DataFrame:
    path = _explicit_file(path, label="participant_masks", suffix=".parquet")
    frame = _normalize_pid(pd.read_parquet(path), table="participant_masks")
    required = ("participant_id", "chronological_age", "sex", "strict_healthy_reference_60m")
    _require_columns(frame, required, table="participant_masks")
    if frame.duplicated("participant_id").any():
        raise EvaluationContractError("participant_masks has duplicate participant_id rows")
    frame = frame.loc[:, required].copy()
    frame["chronological_age"] = _numeric(frame, "chronological_age", table="participant_masks")
    strict = _numeric(frame, "strict_healthy_reference_60m", table="participant_masks")
    if not set(strict.unique()).issubset({0, 1}):
        raise EvaluationContractError("participant_masks.strict_healthy_reference_60m must contain 0/1")
    frame["strict_healthy_reference_60m"] = strict.astype(np.int8)
    sex_text = frame["sex"].astype(str).str.strip().str.lower()
    sex_map = {"female": 0.0, "f": 0.0, "0": 0.0, "male": 1.0, "m": 1.0, "1": 1.0}
    frame["sex"] = sex_text.map(sex_map)
    if frame["sex"].isna().any():
        bad = sorted(sex_text.loc[frame["sex"].isna()].unique().tolist())
        raise EvaluationContractError(f"participant_masks.sex contains unsupported values: {bad}")
    return frame


def load_fold_roles(path: Path) -> pd.DataFrame:
    path = _explicit_file(path, label="fold_roles", suffix=".parquet")
    frame = _normalize_pid(pd.read_parquet(path), table="fold_roles")
    required = ("participant_id", "fold_id", "outer_fold", "split_role")
    _require_columns(frame, required, table="fold_roles")
    frame = frame.loc[:, required].copy()
    for column in ("fold_id", "outer_fold"):
        values = _numeric(frame, column, table="fold_roles")
        if not np.equal(values, np.floor(values)).all() or not values.isin(range(5)).all():
            raise EvaluationContractError(f"fold_roles.{column} must contain fold IDs 0..4")
        frame[column] = values.astype(np.int8)
    frame["split_role"] = frame["split_role"].astype(str)
    if frame.duplicated(["participant_id", "outer_fold"]).any():
        raise EvaluationContractError("fold_roles has duplicate participant_id and outer_fold rows")
    if set(frame["split_role"]) != {"train", "validation", "test"}:
        raise EvaluationContractError("fold_roles.split_role must contain train, validation, and test")
    expected_role = np.where(
        frame["fold_id"].eq(frame["outer_fold"]),
        "test",
        np.where(frame["fold_id"].eq((frame["outer_fold"] + 1) % 5), "validation", "train"),
    )
    if not frame["split_role"].eq(expected_role).all():
        raise EvaluationContractError("fold_roles violates the frozen 3-train/1-validation/1-test rule")
    counts = frame.groupby("participant_id").agg(rows=("outer_fold", "size"), folds=("outer_fold", "nunique"))
    if not counts.eq(5).all(axis=None):
        raise EvaluationContractError("fold_roles must contain five unique outer-fold rows per participant")
    return frame


def load_organ_outcomes(path: Path) -> pd.DataFrame:
    path = _explicit_file(path, label="organ_outcomes", suffix=".parquet")
    frame = _normalize_pid(pd.read_parquet(path), table="organ_outcomes")
    required = ("participant_id", "organ_system", "at_risk", "duration_months", "event")
    _require_columns(frame, required, table="organ_outcomes")
    keep = list(required)
    for optional in ("full_incident_risk_set_60m", "strict_healthy_reference_60m"):
        if optional in frame.columns:
            keep.append(optional)
    frame = frame.loc[:, keep].copy()
    frame["organ_system"] = frame["organ_system"].astype(str)
    if set(frame["organ_system"]) != set(ORGANS):
        raise EvaluationContractError("organ_outcomes.organ_system must contain the seven canonical organs")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise EvaluationContractError("organ_outcomes has duplicate participant-organ rows")
    for column in ("at_risk", "event"):
        values = _numeric(frame, column, table="organ_outcomes")
        if not set(values.unique()).issubset({0, 1}):
            raise EvaluationContractError(f"organ_outcomes.{column} must contain 0/1")
        frame[column] = values.astype(np.int8)
    risk = frame["at_risk"].eq(1)
    duration = pd.to_numeric(frame["duration_months"], errors="coerce")
    if duration.loc[risk].isna().any() or not np.isfinite(duration.loc[risk].to_numpy()).all():
        raise EvaluationContractError("organ_outcomes risk-set duration_months must be finite")
    if duration.loc[risk].le(0).any():
        raise EvaluationContractError("organ_outcomes risk-set duration_months must be positive")
    if frame.loc[~risk, "event"].ne(0).any():
        raise EvaluationContractError("organ_outcomes event rows must belong to the risk set")
    frame["duration_months"] = duration
    if "full_incident_risk_set_60m" in frame:
        full = pd.to_numeric(frame["full_incident_risk_set_60m"], errors="coerce")
        if full.isna().any() or not full.eq(frame["at_risk"]).all():
            raise EvaluationContractError("organ_outcomes at_risk disagrees with full_incident_risk_set_60m")
    return frame


def validate_cross_file_coverage(
    predictions: pd.DataFrame,
    participants: pd.DataFrame,
    outcomes: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    participant_ids = set(participants["participant_id"])
    role_ids = set(roles["participant_id"])
    outcome_ids = set(outcomes["participant_id"])
    prediction_ids = set(predictions["participant_id"])
    if role_ids != participant_ids:
        raise EvaluationContractError("fold_roles participant_id coverage does not equal participant_masks")
    if outcome_ids != participant_ids:
        raise EvaluationContractError("organ_outcomes participant_id coverage does not equal participant_masks")
    if prediction_ids != participant_ids:
        raise EvaluationContractError("prediction coverage does not include every participant exactly once as test")

    outcome_coverage = outcomes.groupby("participant_id")["organ_system"].agg(["size", "nunique"])
    if not outcome_coverage["size"].eq(len(ORGANS)).all() or not outcome_coverage["nunique"].eq(len(ORGANS)).all():
        raise EvaluationContractError("organ_outcomes must cover seven unique organs for every participant")

    expected = roles.loc[roles["split_role"].isin(["validation", "test"]), [
        "participant_id",
        "outer_fold",
        "split_role",
    ]].copy()
    coverage = (
        predictions.groupby(["participant_id", "outer_fold", "split_role"], observed=True)
        .agg(rows=("organ_system", "size"), organs=("organ_system", "nunique"))
        .reset_index()
    )
    audit = expected.merge(
        coverage,
        on=["participant_id", "outer_fold", "split_role"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    complete = (
        audit["_merge"].eq("both")
        & audit["rows"].eq(len(ORGANS))
        & audit["organs"].eq(len(ORGANS))
    )
    if len(predictions) != len(expected) * len(ORGANS) or not complete.all():
        raise EvaluationContractError(
            f"prediction coverage failed: expected_rows={len(expected) * len(ORGANS)}, "
            f"observed_rows={len(predictions)}, bad_participant_fold_roles={int((~complete).sum())}"
        )

    enriched = predictions.merge(
        participants,
        on="participant_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_participant"),
    )
    exported_age = enriched["actual_age"].to_numpy(dtype=np.float64)
    authoritative_age = enriched["chronological_age"].to_numpy(dtype=np.float64)
    age_difference = np.abs(exported_age - authoritative_age)
    mismatch = age_difference > EXPORTED_AGE_ATOL_YEARS
    if mismatch.any():
        raise EvaluationContractError(
            "predictions.actual_age disagrees with authoritative participant "
            f"chronological_age beyond atol={EXPORTED_AGE_ATOL_YEARS:g} years in "
            f"{int(mismatch.sum())} rows; max_abs_difference={float(age_difference.max()):.12g}"
        )
    enriched["exported_actual_age"] = exported_age
    enriched["actual_age"] = authoritative_age
    return enriched.drop(columns="chronological_age")


def evaluate_method(
    *,
    method: str,
    predictions: pd.DataFrame,
    participants: pd.DataFrame,
    outcomes: pd.DataFrame,
    roles: pd.DataFrame,
) -> dict[str, object]:
    enriched = validate_cross_file_coverage(predictions, participants, outcomes, roles)
    test_predictions = enriched.loc[enriched["split_role"].eq("test")].copy()
    all_metrics = raw_age_metrics(test_predictions)
    strict_predictions = test_predictions.loc[test_predictions["strict_healthy_reference_60m"].eq(1)].copy()
    if strict_predictions.empty:
        raise EvaluationContractError("strict healthy test prediction cohort is empty")
    strict_metrics = raw_age_metrics(strict_predictions)
    age_rows = [
        {"method": method, "cohort": "all", **asdict(all_metrics)},
        {
            "method": method,
            "cohort": "strict_healthy_reference_60m",
            **asdict(strict_metrics),
        },
    ]

    risk_outcomes = outcomes.loc[outcomes["at_risk"].eq(1), [
        "participant_id",
        "organ_system",
        "duration_months",
        "event",
    ]].rename(columns={"duration_months": "duration"})
    correction_rows: list[dict[str, object]] = []
    cox_rows: list[dict[str, object]] = []
    for fold in range(5):
        validation = enriched.loc[
            enriched["outer_fold"].eq(fold) & enriched["split_role"].eq("validation")
        ].copy()
        test = enriched.loc[
            enriched["outer_fold"].eq(fold) & enriched["split_role"].eq("test")
        ].copy()
        correction_model = fit_linear_age_correction(validation)
        validation_corrected = apply_linear_age_correction_to_validation(validation, correction_model)
        test_corrected = apply_linear_age_correction(test, correction_model)
        for parameter in correction_model.parameters:
            correction_rows.append({"method": method, "outer_fold": fold, **asdict(parameter)})

        for organ in ORGANS:
            organ_outcomes = risk_outcomes.loc[risk_outcomes["organ_system"].eq(organ)]
            validation_risk = validation_corrected.loc[
                validation_corrected["organ_system"].eq(organ),
                ["participant_id", "organ_system", "actual_age", "sex", "corrected_age_gap"],
            ].merge(
                organ_outcomes,
                on=["participant_id", "organ_system"],
                how="inner",
                validate="one_to_one",
            )
            test_risk = test_corrected.loc[
                test_corrected["organ_system"].eq(organ),
                ["participant_id", "organ_system", "actual_age", "sex", "corrected_age_gap"],
            ].merge(
                organ_outcomes,
                on=["participant_id", "organ_system"],
                how="inner",
                validate="one_to_one",
            )
            cell = evaluate_cox_cell(
                validation_risk,
                test_risk,
                fold=fold,
                organ_system=organ,
            )
            cox_rows.append({"method": method, **asdict(cell)})

    cox_cells = pd.DataFrame(cox_rows)
    aggregate = aggregate_cox_cells(cox_cells)
    exported_age_difference = np.abs(
        enriched["exported_actual_age"].to_numpy(dtype=np.float64)
        - enriched["actual_age"].to_numpy(dtype=np.float64)
    )
    return {
        "age_metrics": pd.DataFrame(age_rows),
        "correction_parameters": pd.DataFrame(correction_rows),
        "cox_cells": cox_cells,
        "cox_aggregate": aggregate,
        "prediction_rows": int(len(predictions)),
        "participant_n": int(len(participants)),
        "outcome_rows": int(len(outcomes)),
        "actual_age_alignment": {
            "metric_source": "participant_masks.chronological_age",
            "exported_value_use": "identity_and_consistency_check_only",
            "absolute_tolerance_years": EXPORTED_AGE_ATOL_YEARS,
            "max_abs_difference_years": float(exported_age_difference.max()),
            "nonzero_difference_rows": int((exported_age_difference > 0.0).sum()),
        },
    }


def write_outputs(
    result: dict[str, object],
    *,
    method: str,
    output_dir: Path,
    prediction_files: Sequence[Path],
    prediction_registry: Path | None,
    participant_masks: Path,
    organ_outcomes: Path,
    fold_roles: Path,
    overwrite: bool,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise EvaluationContractError(f"output files already exist; pass --overwrite: {existing}")

    age_metrics = result["age_metrics"]
    correction = result["correction_parameters"]
    cells = result["cox_cells"]
    aggregate = result["cox_aggregate"]
    age_metrics.to_csv(output_dir / "raw_age_metrics.csv", index=False)
    correction.to_csv(output_dir / "linear_correction_parameters.csv", index=False)
    cells.to_csv(output_dir / "cox_fold_organ_cells.csv", index=False)
    organ_rows = [asdict(item) for item in aggregate.organs]
    pd.DataFrame(organ_rows).to_csv(output_dir / "cox_organ_summary.csv", index=False)

    cox_payload = asdict(aggregate)
    (output_dir / "cox_summary.json").write_text(
        json.dumps(cox_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    age_payload = {
        str(row["cohort"]): {
            key: row[key]
            for key in ("n_rows", "n_participants", "mae", "pearson", "r2")
        }
        for row in age_metrics.to_dict("records")
    }
    summary = {
        "schema_version": "revision0710_evaluation_summary_v1",
        "method": method,
        "age_metrics": age_payload,
        "cox": cox_payload,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "revision0710_evaluation_manifest_v1",
        "method": method,
        "prediction_registry": str(prediction_registry) if prediction_registry else None,
        "prediction_files": [str(path) for path in prediction_files],
        "participant_masks": str(participant_masks.expanduser().resolve()),
        "organ_outcomes": str(organ_outcomes.expanduser().resolve()),
        "fold_roles": str(fold_roles.expanduser().resolve()),
        "prediction_rows": result["prediction_rows"],
        "participant_n": result["participant_n"],
        "outcome_rows": result["outcome_rows"],
        "actual_age_alignment": result["actual_age_alignment"],
        "contract": {
            "correction_fit": "current outer-fold validation strict healthy reference",
            "cox_fit": "current outer-fold validation full incident risk set",
            "cox_score": "current outer-fold test full incident risk set",
            "delta_c_aggregation": "five folds within organ, then seven-organ macro",
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction-registry", type=Path)
    source.add_argument("--predictions", type=Path)
    parser.add_argument("--method", required=True)
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
    parser.add_argument(
        "--fold-roles",
        type=Path,
        default=REVISION_DIR / "01_data_rebuild/03_folds/outer_fold_roles.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction_files, registry_path = resolve_prediction_files(
        method=args.method,
        prediction_registry=args.prediction_registry,
        predictions=args.predictions,
    )
    predictions = load_predictions(prediction_files)
    participants = load_participant_masks(args.participant_masks)
    outcomes = load_organ_outcomes(args.organ_outcomes)
    roles = load_fold_roles(args.fold_roles)
    result = evaluate_method(
        method=args.method,
        predictions=predictions,
        participants=participants,
        outcomes=outcomes,
        roles=roles,
    )
    write_outputs(
        result,
        method=args.method,
        output_dir=args.output_dir,
        prediction_files=prediction_files,
        prediction_registry=registry_path,
        participant_masks=args.participant_masks,
        organ_outcomes=args.organ_outcomes,
        fold_roles=args.fold_roles,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
