#!/usr/bin/env python
"""Build the six-method baseline envelope for D07 open Fold 0 development."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DOWNSTREAM_DIR = THIS_DIR.parent
REVISION_DIR = DOWNSTREAM_DIR.parent
COMMON_DIR = DOWNSTREAM_DIR / "common"
SCRIPTS_DIR = DOWNSTREAM_DIR / "scripts"
for import_dir in (COMMON_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

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
    load_fold_roles,
    load_organ_outcomes,
    load_participant_masks,
)


METHODS = (
    "retain",
    "behrt_style",
    "medbert_style",
    "retain_organquery",
    "behrt_style_organquery",
    "medbert_style_organquery",
)
DEFAULT_REGISTRIES = {
    "retain": DOWNSTREAM_DIR / "01_prediction_registries/tier_a/retain.json",
    "behrt_style": DOWNSTREAM_DIR / "01_prediction_registries/tier_a/behrt_style.json",
    "medbert_style": DOWNSTREAM_DIR / "01_prediction_registries/tier_a/medbert_style.json",
    "retain_organquery": DOWNSTREAM_DIR
    / "01_prediction_registries/tier_b/retain_organquery.json",
    "behrt_style_organquery": DOWNSTREAM_DIR
    / "01_prediction_registries/tier_b/behrt_style_organquery.json",
    "medbert_style_organquery": DOWNSTREAM_DIR
    / "01_prediction_registries/tier_b/medbert_style_organquery.json",
}
DEFAULT_PARTICIPANT_MASKS = (
    REVISION_DIR
    / "01_data_rebuild/02_outcomes_60m/participant_followup_and_evaluation_masks.parquet"
)
DEFAULT_ORGAN_OUTCOMES = (
    REVISION_DIR / "01_data_rebuild/02_outcomes_60m/organ_outcomes_60m.parquet"
)
DEFAULT_FOLD_ROLES = REVISION_DIR / "01_data_rebuild/03_folds/outer_fold_roles.parquet"
DEFAULT_INNER_ROLES = (
    REVISION_DIR / "01_data_rebuild/03_folds/fold0_validation_calibration_screen.parquet"
)

ENVELOPE_SCHEMA = "revision0710_d07_open_fold0_baseline_envelope_v1"
MANIFEST_SCHEMA = "revision0710_d07_open_fold0_baseline_artifact_manifest_v1"
OUTPUT_FILES = (
    "baseline_method_summary.csv",
    "strict_age_metrics.csv",
    "age_correction_parameters.csv",
    "cox_organ_cells.csv",
    "fold0_baseline_envelope.json",
)
AGE_MARGINS = {"strict_mae": 0.010, "strict_pearson": 0.001, "strict_r2": 0.002}
RISK_MARGIN = 0.010


class D07Fold0BaselineError(EvaluationContractError):
    """Raised when a D07 Fold 0 baseline artifact violates its contract."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_registry_fold0_files(
    method: str, registry_path: Path
) -> tuple[Path, str, Path, str, Path]:
    registry_path = _explicit_file(registry_path, label=f"{method}.registry", suffix=".json")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D07Fold0BaselineError(f"invalid {method} registry: {exc}") from exc
    if payload.get("schema_version") != "revision0710_prediction_registry_v1":
        raise D07Fold0BaselineError(f"{method} registry has unsupported schema_version")
    if payload.get("method") != method:
        raise D07Fold0BaselineError(f"{method} registry method mismatch")
    entries = payload.get("prediction_files")
    if not isinstance(entries, list):
        raise D07Fold0BaselineError(f"{method} registry prediction_files must be a list")

    selected: dict[str, tuple[Path, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise D07Fold0BaselineError(f"{method} registry entries must be objects")
        if entry.get("outer_fold") != 0 or entry.get("split_role") not in {"validation", "test"}:
            continue
        role = str(entry["split_role"])
        if role in selected:
            raise D07Fold0BaselineError(f"{method} registry repeats Fold 0 {role}")
        expected_sha = entry.get("sha256")
        if not _is_sha256(expected_sha):
            raise D07Fold0BaselineError(f"{method} Fold 0 {role} has invalid sha256")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            raise D07Fold0BaselineError(f"{method} Fold 0 {role} path must be a string")
        prediction_path = Path(raw_path)
        if not prediction_path.is_absolute():
            prediction_path = registry_path.parent / prediction_path
        prediction_path = _explicit_file(
            prediction_path, label=f"{method}.fold0.{role}", suffix=".parquet"
        )
        observed_sha = _sha256_file(prediction_path)
        if observed_sha != expected_sha.lower():
            raise D07Fold0BaselineError(
                f"{method} Fold 0 {role} sha256 mismatch: "
                f"expected={expected_sha}, observed={observed_sha}"
            )
        selected[role] = (prediction_path, observed_sha)
    if set(selected) != {"validation", "test"}:
        raise D07Fold0BaselineError(
            f"{method} registry must provide exactly Fold 0 validation and test predictions"
        )
    return (
        selected["validation"][0],
        selected["validation"][1],
        selected["test"][0],
        selected["test"][1],
        registry_path,
    )


def _load_fold0_prediction_file(path: Path, *, split_role: str, method: str) -> pd.DataFrame:
    table = f"{method}.fold0.{split_role}"
    frame = _normalize_pid(pd.read_parquet(path), table=table)
    required = (
        "participant_id",
        "organ_system",
        "actual_age",
        "predicted_age",
        "outer_fold",
        "split_role",
    )
    _require_columns(frame, required, table=table)
    frame = frame.loc[:, required].copy()
    if frame.empty:
        raise D07Fold0BaselineError(f"{table} must not be empty")
    frame["actual_age"] = _numeric(frame, "actual_age", table=table)
    frame["predicted_age"] = _numeric(frame, "predicted_age", table=table)
    outer_fold = _numeric(frame, "outer_fold", table=table)
    if not outer_fold.eq(0).all():
        raise D07Fold0BaselineError(f"{table} must contain outer_fold=0 only")
    if not frame["split_role"].astype(str).eq(split_role).all():
        raise D07Fold0BaselineError(f"{table} split_role mismatch")
    frame["outer_fold"] = outer_fold.astype(np.int8)
    frame["split_role"] = frame["split_role"].astype(str)
    frame["organ_system"] = frame["organ_system"].astype(str)
    if set(frame["organ_system"]) != set(ORGANS):
        raise D07Fold0BaselineError(f"{table} must contain seven canonical organs")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise D07Fold0BaselineError(f"{table} has duplicate participant-organ rows")
    coverage = frame.groupby("participant_id")["organ_system"].agg(["size", "nunique"])
    if not coverage["size"].eq(len(ORGANS)).all() or not coverage["nunique"].eq(len(ORGANS)).all():
        raise D07Fold0BaselineError(f"{table} must have seven organs per participant")
    return frame


def _load_inner_calibration(path: Path) -> pd.DataFrame:
    path = _explicit_file(path, label="inner_roles", suffix=".parquet")
    frame = _normalize_pid(pd.read_parquet(path), table="inner_roles")
    _require_columns(frame, ("participant_id", "fold_id", "inner_role"), table="inner_roles")
    frame = frame.loc[:, ["participant_id", "fold_id", "inner_role"]].copy()
    if frame.empty or frame.duplicated("participant_id").any():
        raise D07Fold0BaselineError("inner_roles must contain unique non-empty participants")
    fold_id = _numeric(frame, "fold_id", table="inner_roles")
    if not fold_id.eq(1).all():
        raise D07Fold0BaselineError("inner_roles must contain Fold 1 participants only")
    frame["fold_id"] = fold_id.astype(np.int8)
    frame["inner_role"] = frame["inner_role"].astype(str)
    if set(frame["inner_role"]) != {"calibration", "screen"}:
        raise D07Fold0BaselineError("inner_roles must contain calibration and screen")
    return frame.loc[frame["inner_role"].eq("calibration")].copy()


def _enrich_predictions(
    predictions: pd.DataFrame,
    *,
    split_role: str,
    participants: pd.DataFrame,
    expected_participant_ids: set[str],
) -> tuple[pd.DataFrame, float]:
    observed_ids = set(predictions["participant_id"])
    if observed_ids != expected_participant_ids:
        raise D07Fold0BaselineError(
            f"Fold 0 {split_role} prediction coverage mismatch: "
            f"missing={len(expected_participant_ids - observed_ids)}, "
            f"extra={len(observed_ids - expected_participant_ids)}"
        )
    participant_subset = participants.loc[
        participants["participant_id"].isin(expected_participant_ids)
    ].copy()
    if set(participant_subset["participant_id"]) != expected_participant_ids:
        raise D07Fold0BaselineError(f"participant masks do not cover Fold 0 {split_role}")
    enriched = predictions.merge(
        participant_subset,
        on="participant_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_authoritative"),
    )
    exported_age = enriched["actual_age"].to_numpy(dtype=np.float64)
    authoritative_age = enriched["chronological_age"].to_numpy(dtype=np.float64)
    difference = np.abs(exported_age - authoritative_age)
    if (difference > EXPORTED_AGE_ATOL_YEARS).any():
        raise D07Fold0BaselineError(
            f"Fold 0 {split_role} exported actual_age differs from authoritative age"
        )
    enriched["exported_actual_age"] = exported_age
    enriched["actual_age"] = authoritative_age
    enriched = enriched.drop(columns="chronological_age")
    return enriched, float(difference.max(initial=0.0))


def evaluate_one_method(
    *,
    method: str,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    participants: pd.DataFrame,
    outcomes: pd.DataFrame,
    roles: pd.DataFrame,
    calibration_roles: pd.DataFrame,
) -> dict[str, object]:
    fold0_roles = roles.loc[roles["outer_fold"].eq(0)].copy()
    expected_validation_ids = set(
        fold0_roles.loc[fold0_roles["split_role"].eq("validation"), "participant_id"]
    )
    expected_test_ids = set(fold0_roles.loc[fold0_roles["split_role"].eq("test"), "participant_id"])
    if expected_validation_ids & expected_test_ids:
        raise D07Fold0BaselineError("Fold 1 validation and Fold 0 test participants overlap")
    calibration_ids = set(calibration_roles["participant_id"])
    if not calibration_ids < expected_validation_ids:
        raise D07Fold0BaselineError("calibration must be a proper subset of Fold 1 validation")

    validation, validation_age_diff = _enrich_predictions(
        validation_predictions,
        split_role="validation",
        participants=participants,
        expected_participant_ids=expected_validation_ids,
    )
    test, test_age_diff = _enrich_predictions(
        test_predictions,
        split_role="test",
        participants=participants,
        expected_participant_ids=expected_test_ids,
    )
    calibration = validation.loc[validation["participant_id"].isin(calibration_ids)].copy()
    if set(calibration["participant_id"]) != calibration_ids:
        raise D07Fold0BaselineError("validation predictions do not cover calibration participants")

    strict_test = test.loc[test["strict_healthy_reference_60m"].eq(1)].copy()
    if strict_test.empty:
        raise D07Fold0BaselineError("Fold 0 strict healthy population is empty")
    pooled_age = raw_age_metrics(strict_test)
    organ_age_rows: list[dict[str, object]] = []
    for organ in ORGANS:
        metrics = raw_age_metrics(strict_test.loc[strict_test["organ_system"].eq(organ)])
        organ_age_rows.append({"method": method, "organ_system": organ, **asdict(metrics)})

    correction = fit_linear_age_correction(calibration)
    calibration_corrected = apply_linear_age_correction_to_validation(calibration, correction)
    test_corrected = apply_linear_age_correction(test, correction)
    correction_rows = [
        {"method": method, "outer_fold": 0, "fit_fold_id": 1, **asdict(parameter)}
        for parameter in correction.parameters
    ]

    risk_outcomes = outcomes.loc[
        outcomes["at_risk"].eq(1),
        ["participant_id", "organ_system", "duration_months", "event"],
    ].rename(columns={"duration_months": "duration"})
    cox_rows: list[dict[str, object]] = []
    for organ in ORGANS:
        organ_outcomes = risk_outcomes.loc[risk_outcomes["organ_system"].eq(organ)]
        fit = calibration_corrected.loc[
            calibration_corrected["organ_system"].eq(organ),
            ["participant_id", "organ_system", "actual_age", "sex", "corrected_age_gap"],
        ].merge(
            organ_outcomes,
            on=["participant_id", "organ_system"],
            how="inner",
            validate="one_to_one",
        )
        score = test_corrected.loc[
            test_corrected["organ_system"].eq(organ),
            ["participant_id", "organ_system", "actual_age", "sex", "corrected_age_gap"],
        ].merge(
            organ_outcomes,
            on=["participant_id", "organ_system"],
            how="inner",
            validate="one_to_one",
        )
        cell = evaluate_cox_cell(fit, score, fold=0, organ_system=organ)
        cox_rows.append({"method": method, **asdict(cell)})
    cox_cells = pd.DataFrame(cox_rows)
    cox_aggregate = aggregate_cox_cells(cox_cells, folds=(0,))
    return {
        "pooled_age": pooled_age,
        "organ_age_rows": organ_age_rows,
        "correction_rows": correction_rows,
        "cox_cells": cox_cells,
        "cox_aggregate": cox_aggregate,
        "coverage": {
            "fold1_validation_participants": len(expected_validation_ids),
            "fold1_calibration_participants": len(calibration_ids),
            "fold0_test_participants": len(expected_test_ids),
            "fold0_strict_participants": int(strict_test["participant_id"].nunique()),
            "fold0_strict_rows": len(strict_test),
            "max_exported_age_abs_difference_years": max(validation_age_diff, test_age_diff),
        },
    }


def _best_metric(
    methods: list[dict[str, object]], field: str, *, direction: str
) -> tuple[float, str]:
    candidates = [(float(item["strict_age"]["pooled"][field]), str(item["method"])) for item in methods]
    if direction == "lower":
        return min(candidates)
    if direction == "higher":
        return max(candidates)
    raise AssertionError(direction)


def _build_envelope(methods: list[dict[str, object]]) -> dict[str, object]:
    mae, mae_method = _best_metric(methods, "mae", direction="lower")
    pearson, pearson_method = _best_metric(methods, "pearson", direction="higher")
    r2, r2_method = _best_metric(methods, "r2", direction="higher")
    risk_candidates = [
        (float(item["cox"]["macro"]["macro_delta_c"]), str(item["method"])) for item in methods
    ]
    risk, risk_method = max(risk_candidates)
    organ_envelope: list[dict[str, object]] = []
    for organ in ORGANS:
        candidates = []
        for item in methods:
            row = next(
                row for row in item["cox"]["organs"] if row["organ_system"] == organ
            )
            candidates.append((float(row["delta_c"]), str(item["method"])))
        value, method = max(candidates)
        organ_envelope.append(
            {
                "organ_system": organ,
                "baseline_value": value,
                "method": method,
                "direction": "higher",
            }
        )
    return {
        "strict_mae": {"baseline_value": mae, "method": mae_method, "direction": "lower"},
        "strict_pearson": {
            "baseline_value": pearson,
            "method": pearson_method,
            "direction": "higher",
        },
        "strict_r2": {"baseline_value": r2, "method": r2_method, "direction": "higher"},
        "age_sex_harrell_macro_delta_c": {
            "baseline_value": risk,
            "method": risk_method,
            "direction": "higher",
        },
        "organ_delta_c_envelope": organ_envelope,
    }


def _build_gates(envelope: dict[str, object]) -> dict[str, object]:
    gates: dict[str, object] = {}
    definitions = (
        ("strict_mae", AGE_MARGINS["strict_mae"], "<", -1.0),
        ("strict_pearson", AGE_MARGINS["strict_pearson"], ">", 1.0),
        ("strict_r2", AGE_MARGINS["strict_r2"], ">", 1.0),
        ("age_sex_harrell_macro_delta_c", RISK_MARGIN, ">", 1.0),
    )
    for metric, margin, operator, sign in definitions:
        baseline_value = float(envelope[metric]["baseline_value"])
        gates[metric] = {
            "baseline_value": baseline_value,
            "baseline_method": envelope[metric]["method"],
            "margin": margin,
            "operator": operator,
            "threshold": baseline_value + sign * margin,
        }
    gates["risk_organ_rules"] = {
        "positive_organs_required": len(ORGANS),
        "organs_beating_baseline_required": 5,
        "organ_baseline_source": "envelope.organ_delta_c_envelope",
        "organ_comparison_operator": ">",
    }
    return gates


def _method_payload(
    method: str,
    result: dict[str, object],
    *,
    registry_path: Path,
    validation_path: Path,
    validation_sha: str,
    test_path: Path,
    test_sha: str,
) -> dict[str, object]:
    aggregate = result["cox_aggregate"]
    organ_cells = result["cox_cells"]
    return {
        "method": method,
        "strict_age": {
            "pooled": asdict(result["pooled_age"]),
            "organs": [
                {key: value for key, value in row.items() if key != "method"}
                for row in result["organ_age_rows"]
            ],
        },
        "cox": {
            "macro": asdict(aggregate),
            "organs": [
                {
                    "organ_system": str(row.organ_system),
                    "n_calibration": int(row.n_validation),
                    "n_fold0": int(row.n_test),
                    "calibration_events": int(row.validation_events),
                    "fold0_events": int(row.test_events),
                    "c_base": float(row.c_base),
                    "c_plus": float(row.c_plus),
                    "delta_c": float(row.delta_c),
                }
                for row in organ_cells.itertuples(index=False)
            ],
        },
        "coverage": result["coverage"],
        "source_files": {
            "prediction_registry": str(registry_path),
            "fold1_validation_predictions": str(validation_path),
            "fold0_test_predictions": str(test_path),
        },
        "source_sha256": {
            "prediction_registry": _sha256_file(registry_path),
            "fold1_validation_predictions": validation_sha,
            "fold0_test_predictions": test_sha,
        },
    }


def build_d07_fold0_baseline(
    *,
    registry_paths: Mapping[str, Path],
    participant_masks_path: Path,
    organ_outcomes_path: Path,
    fold_roles_path: Path,
    inner_roles_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    if set(registry_paths) != set(METHODS):
        raise D07Fold0BaselineError("registry_paths must contain exactly the six canonical methods")
    output_dir = output_dir.expanduser().resolve()
    if any(token in str(output_dir) for token in ("*", "?", "[", "]")):
        raise D07Fold0BaselineError("output_dir must be explicit and contain no glob tokens")
    expected_outputs = (*OUTPUT_FILES, "artifact_manifest.json")
    existing = [name for name in expected_outputs if (output_dir / name).exists()]
    if existing and not overwrite:
        raise D07Fold0BaselineError(f"outputs already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_masks_path = _explicit_file(
        participant_masks_path, label="participant_masks", suffix=".parquet"
    )
    organ_outcomes_path = _explicit_file(
        organ_outcomes_path, label="organ_outcomes", suffix=".parquet"
    )
    fold_roles_path = _explicit_file(fold_roles_path, label="fold_roles", suffix=".parquet")
    inner_roles_path = _explicit_file(inner_roles_path, label="inner_roles", suffix=".parquet")
    participants = load_participant_masks(participant_masks_path)
    outcomes = load_organ_outcomes(organ_outcomes_path)
    roles = load_fold_roles(fold_roles_path)
    calibration_roles = _load_inner_calibration(inner_roles_path)

    method_payloads: list[dict[str, object]] = []
    age_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    cox_rows: list[pd.DataFrame] = []
    for method in METHODS:
        validation_path, validation_sha, test_path, test_sha, registry_path = (
            _load_registry_fold0_files(method, registry_paths[method])
        )
        validation_predictions = _load_fold0_prediction_file(
            validation_path, split_role="validation", method=method
        )
        test_predictions = _load_fold0_prediction_file(
            test_path, split_role="test", method=method
        )
        result = evaluate_one_method(
            method=method,
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            participants=participants,
            outcomes=outcomes,
            roles=roles,
            calibration_roles=calibration_roles,
        )
        pooled = asdict(result["pooled_age"])
        age_rows.append({"method": method, "level": "pooled", "organ_system": "all", **pooled})
        age_rows.extend(
            {"level": "organ", **row} for row in result["organ_age_rows"]
        )
        correction_rows.extend(result["correction_rows"])
        cox_rows.append(result["cox_cells"])
        method_payloads.append(
            _method_payload(
                method,
                result,
                registry_path=registry_path,
                validation_path=validation_path,
                validation_sha=validation_sha,
                test_path=test_path,
                test_sha=test_sha,
            )
        )

    envelope_values = _build_envelope(method_payloads)
    gates = _build_gates(envelope_values)
    scope = {
        "outer_fold": 0,
        "fold0_role": "open_development",
        "prediction_split_role": "test",
        "nuisance_fit": {
            "source": "outer_fold_0_validation_predictions",
            "participant_fold_id": 1,
            "inner_role": "calibration",
            "age_correction_population": "strict_healthy_reference_60m",
            "cox_population": "all_at_risk_1",
        },
        "score": {
            "source": "outer_fold_0_test_predictions",
            "participant_fold_id": 0,
            "strict_age_population": "strict_healthy_reference_60m_pooled_participant_organ",
            "harrell_population": "all_at_risk_1_by_organ_then_seven_organ_macro",
        },
        "fold0_fitted_parameters": False,
        "ehr_age_predictions_read": False,
    }
    source_files = {
        "participant_masks": str(participant_masks_path),
        "organ_outcomes": str(organ_outcomes_path),
        "fold_roles": str(fold_roles_path),
        "fold0_validation_calibration_screen": str(inner_roles_path),
        "builder_code": str(Path(__file__).resolve()),
        "metric_code": str((COMMON_DIR / "crossfit_metrics.py").resolve()),
    }
    source_sha = {key: _sha256_file(Path(path)) for key, path in source_files.items()}
    envelope_payload = {
        "schema_version": ENVELOPE_SCHEMA,
        "scope": scope,
        "method_count": len(method_payloads),
        "methods": method_payloads,
        "envelope": envelope_values,
        "gates": gates,
        "source_files": source_files,
        "source_sha256": source_sha,
    }

    age_frame = pd.DataFrame(age_rows)
    correction_frame = pd.DataFrame(correction_rows)
    cox_frame = pd.concat(cox_rows, ignore_index=True)
    summary_rows = []
    for item in method_payloads:
        pooled = item["strict_age"]["pooled"]
        macro = item["cox"]["macro"]
        summary_rows.append(
            {
                "method": item["method"],
                "strict_n_rows": pooled["n_rows"],
                "strict_n_participants": pooled["n_participants"],
                "strict_mae": pooled["mae"],
                "strict_pearson": pooled["pearson"],
                "strict_r2": pooled["r2"],
                "age_sex_harrell_macro_c_base": macro["macro_c_base"],
                "age_sex_harrell_macro_c_plus": macro["macro_c_plus"],
                "age_sex_harrell_macro_delta_c": macro["macro_delta_c"],
                "positive_organs": macro["positive_organs"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "baseline_method_summary.csv", index=False)
    age_frame.to_csv(output_dir / "strict_age_metrics.csv", index=False)
    correction_frame.to_csv(output_dir / "age_correction_parameters.csv", index=False)
    cox_frame.to_csv(output_dir / "cox_organ_cells.csv", index=False)
    _json_dump(output_dir / "fold0_baseline_envelope.json", envelope_payload)

    artifact_hashes = {name: _sha256_file(output_dir / name) for name in OUTPUT_FILES}
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "canonical_output_dir": str(output_dir),
        "envelope_path": str(output_dir / "fold0_baseline_envelope.json"),
        "envelope_sha256": artifact_hashes["fold0_baseline_envelope.json"],
        "artifacts": artifact_hashes,
        "source_sha256": source_sha,
        "contract": {
            "six_baselines_only": True,
            "fold1_calibration_only_for_nuisance_fit": True,
            "fold0_only_for_scoring": True,
            "strict_age_is_strict_only": True,
            "harrell_is_all_at_risk": True,
            "ehr_age_predictions_read": False,
        },
    }
    _json_dump(output_dir / "artifact_manifest.json", manifest)
    validate_fold0_baseline_artifacts(output_dir)
    return envelope_payload


def validate_fold0_baseline_artifacts(output_dir: Path = THIS_DIR) -> dict[str, object]:
    """Hard-validate the canonical D07 envelope and all registered output hashes."""

    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "artifact_manifest.json"
    envelope_path = output_dir / "fold0_baseline_envelope.json"
    if not manifest_path.is_file() or not envelope_path.is_file():
        raise D07Fold0BaselineError("D07 Fold 0 envelope or artifact manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise D07Fold0BaselineError(f"invalid D07 Fold 0 JSON: {exc}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise D07Fold0BaselineError("unsupported D07 artifact manifest schema")
    if envelope.get("schema_version") != ENVELOPE_SCHEMA:
        raise D07Fold0BaselineError("unsupported D07 Fold 0 envelope schema")
    if Path(manifest.get("canonical_output_dir", "")).resolve() != output_dir:
        raise D07Fold0BaselineError("artifact manifest canonical_output_dir mismatch")
    if Path(manifest.get("envelope_path", "")).resolve() != envelope_path:
        raise D07Fold0BaselineError("artifact manifest envelope_path mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(OUTPUT_FILES):
        raise D07Fold0BaselineError("artifact manifest must register exactly the canonical outputs")
    for name, expected_sha in artifacts.items():
        if not _is_sha256(expected_sha):
            raise D07Fold0BaselineError(f"invalid artifact hash for {name}")
        path = output_dir / name
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise D07Fold0BaselineError(f"artifact hash mismatch for {name}")
    if manifest.get("envelope_sha256") != artifacts["fold0_baseline_envelope.json"]:
        raise D07Fold0BaselineError("manifest envelope_sha256 mismatch")

    scope = envelope.get("scope")
    if not isinstance(scope, dict):
        raise D07Fold0BaselineError("envelope scope must be an object")
    expected_scope = {
        "outer_fold": 0,
        "fold0_role": "open_development",
        "prediction_split_role": "test",
        "fold0_fitted_parameters": False,
        "ehr_age_predictions_read": False,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise D07Fold0BaselineError(f"envelope scope.{key} mismatch")
    if scope.get("nuisance_fit", {}).get("inner_role") != "calibration":
        raise D07Fold0BaselineError("nuisance fit must use Fold 1 calibration")
    if scope.get("score", {}).get("strict_age_population") != (
        "strict_healthy_reference_60m_pooled_participant_organ"
    ):
        raise D07Fold0BaselineError("strict age population mismatch")
    if scope.get("score", {}).get("harrell_population") != (
        "all_at_risk_1_by_organ_then_seven_organ_macro"
    ):
        raise D07Fold0BaselineError("Harrell population mismatch")

    methods = envelope.get("methods")
    if not isinstance(methods, list) or [item.get("method") for item in methods] != list(METHODS):
        raise D07Fold0BaselineError("envelope must contain six canonical methods in order")
    for item in methods:
        if len(item.get("strict_age", {}).get("organs", [])) != len(ORGANS):
            raise D07Fold0BaselineError(f"{item.get('method')} lacks seven age organ rows")
        if len(item.get("cox", {}).get("organs", [])) != len(ORGANS):
            raise D07Fold0BaselineError(f"{item.get('method')} lacks seven Cox organ rows")
        source_hashes = item.get("source_sha256")
        if not isinstance(source_hashes, dict) or not all(
            _is_sha256(value) for value in source_hashes.values()
        ):
            raise D07Fold0BaselineError(f"{item.get('method')} source hashes are invalid")
    rebuilt_envelope = _build_envelope(methods)
    if envelope.get("envelope") != rebuilt_envelope:
        raise D07Fold0BaselineError("metric-wise envelope does not match method results")
    rebuilt_gates = _build_gates(rebuilt_envelope)
    if envelope.get("gates") != rebuilt_gates:
        raise D07Fold0BaselineError("D07 gates do not match the frozen margins")
    if [row["organ_system"] for row in rebuilt_envelope["organ_delta_c_envelope"]] != list(ORGANS):
        raise D07Fold0BaselineError("organ delta-C envelope order mismatch")
    if not all(_is_sha256(value) for value in envelope.get("source_sha256", {}).values()):
        raise D07Fold0BaselineError("envelope source hashes are invalid")
    return envelope


def _parse_registry_overrides(values: Sequence[str]) -> dict[str, Path]:
    registries = dict(DEFAULT_REGISTRIES)
    for value in values:
        if "=" not in value:
            raise D07Fold0BaselineError("--registry must use METHOD=PATH")
        method, raw_path = value.split("=", 1)
        if method not in METHODS:
            raise D07Fold0BaselineError(f"unknown registry method: {method}")
        registries[method] = Path(raw_path)
    return registries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--participant-masks", type=Path, default=DEFAULT_PARTICIPANT_MASKS)
    parser.add_argument("--organ-outcomes", type=Path, default=DEFAULT_ORGAN_OUTCOMES)
    parser.add_argument("--fold-roles", type=Path, default=DEFAULT_FOLD_ROLES)
    parser.add_argument("--inner-roles", type=Path, default=DEFAULT_INNER_ROLES)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        validate_fold0_baseline_artifacts(args.output_dir)
        print("VALID")
        return 0
    payload = build_d07_fold0_baseline(
        registry_paths=_parse_registry_overrides(args.registry),
        participant_masks_path=args.participant_masks,
        organ_outcomes_path=args.organ_outcomes,
        fold_roles_path=args.fold_roles,
        inner_roles_path=args.inner_roles,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload["gates"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
