#!/usr/bin/env python
"""Freeze the six-method Fold0 validation calibration-to-screen envelope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Sequence


REGISTRY_SCHEMA = "revision0710_fold0_development_evaluation_registry_v1"
SUMMARY_SCHEMA = "revision0710_fold0_development_evaluation_summary_v1"
MANIFEST_SCHEMA = "revision0710_fold0_development_evaluation_manifest_v1"
EXPECTED_METHODS = (
    ("retain", "tier_a"),
    ("behrt_style", "tier_a"),
    ("medbert_style", "tier_a"),
    ("retain_organquery", "tier_b"),
    ("behrt_style_organquery", "tier_b"),
    ("medbert_style_organquery", "tier_b"),
)
ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
OUTPUT_FILES = (
    "fold0_development_baseline_method_table.csv",
    "fold0_development_baseline_method_table.json",
    "fold0_development_baseline_envelope.csv",
    "fold0_development_baseline_envelope.json",
)
EXPECTED_SCOPE = {
    "outer_fold": 0,
    "source_split": "validation",
    "source_fold_id": 1,
    "age_metric_population": "screen_strict_healthy_reference_60m",
    "age_correction_fit": "calibration_strict_healthy_reference_60m",
    "cox_fit": "calibration_full_incident_risk_set",
    "cox_score": "screen_full_incident_risk_set",
    "outer_test_used": False,
}


class Fold0EnvelopeError(ValueError):
    """Raised when development summaries do not satisfy the frozen contract."""


def _reject_glob(path: Path, *, label: str) -> None:
    if any(token in str(path) for token in ("*", "?", "[", "]")):
        raise Fold0EnvelopeError(f"{label} must be explicit and contain no glob tokens")


def _explicit_file(path: Path, *, label: str) -> Path:
    _reject_glob(path, label=label)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise Fold0EnvelopeError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Fold0EnvelopeError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Fold0EnvelopeError(f"{label} must be a JSON object")
    return value


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Fold0EnvelopeError(f"{label} must be a JSON object")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Fold0EnvelopeError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Fold0EnvelopeError(f"{label} must be finite numeric")
    return result


def _integer(value: object, *, label: str) -> int:
    result = _finite(value, label=label)
    if not result.is_integer():
        raise Fold0EnvelopeError(f"{label} must be an integer")
    return int(result)


def _validate_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise Fold0EnvelopeError(f"{label} must be a SHA256 string")
    value = value.lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise Fold0EnvelopeError(f"{label} must be a SHA256 string")
    return value


def _resolve_entry_file(registry_path: Path, entry: dict[str, object], key: str, method: str) -> Path:
    raw = entry.get(key)
    if not isinstance(raw, str) or not raw:
        raise Fold0EnvelopeError(f"registry {key} missing for {method}")
    path = Path(raw)
    if not path.is_absolute():
        path = registry_path.parent / path
    return _explicit_file(path, label=f"{key}[{method}]")


def load_records(registry_path: Path) -> tuple[list[dict[str, object]], dict[str, object], str]:
    registry_path = _explicit_file(registry_path, label="registry")
    registry = _read_object(registry_path, label="registry")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise Fold0EnvelopeError("unsupported registry schema_version")
    entries = registry.get("methods")
    if not isinstance(entries, list):
        raise Fold0EnvelopeError("registry.methods must be a list")
    expected = dict(EXPECTED_METHODS)
    by_method: dict[str, dict[str, object]] = {}
    for index, value in enumerate(entries):
        entry = _mapping(value, label=f"registry.methods[{index}]")
        method = entry.get("method")
        if not isinstance(method, str) or method not in expected:
            raise Fold0EnvelopeError(f"registry contains unexpected method: {method!r}")
        if method in by_method:
            raise Fold0EnvelopeError(f"registry contains duplicate method: {method}")
        if entry.get("tier") != expected[method]:
            raise Fold0EnvelopeError(f"registry tier mismatch for {method}")
        by_method[method] = entry
    missing = [method for method, _ in EXPECTED_METHODS if method not in by_method]
    if missing or len(entries) != len(EXPECTED_METHODS):
        raise Fold0EnvelopeError(f"registry must contain exactly the six methods; missing={missing}")

    records: list[dict[str, object]] = []
    shared_hashes: dict[str, str] | None = None
    prediction_hashes: list[dict[str, str]] = []
    reference_counts: tuple[int, int, int] | None = None
    for method, tier in EXPECTED_METHODS:
        entry = by_method[method]
        summary_path = _resolve_entry_file(registry_path, entry, "summary_path", method)
        manifest_path = _resolve_entry_file(registry_path, entry, "manifest_path", method)
        expected_summary_sha = _validate_sha(entry.get("summary_sha256"), label=f"{method}.summary_sha256")
        expected_manifest_sha = _validate_sha(
            entry.get("manifest_sha256"), label=f"{method}.manifest_sha256"
        )
        if _sha256(summary_path) != expected_summary_sha:
            raise Fold0EnvelopeError(f"summary sha256 mismatch for {method}")
        if _sha256(manifest_path) != expected_manifest_sha:
            raise Fold0EnvelopeError(f"manifest sha256 mismatch for {method}")
        summary = _read_object(summary_path, label=f"summary[{method}]")
        manifest = _read_object(manifest_path, label=f"manifest[{method}]")
        if summary.get("schema_version") != SUMMARY_SCHEMA or manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise Fold0EnvelopeError(f"unsupported summary/manifest schema for {method}")
        if summary.get("method") != method or manifest.get("method") != method:
            raise Fold0EnvelopeError(f"method mismatch in summary/manifest for {method}")
        if summary.get("scope") != EXPECTED_SCOPE or manifest.get("scope") != EXPECTED_SCOPE:
            raise Fold0EnvelopeError(f"scope mismatch for {method}; outer test must remain unused")
        manifest_output = _mapping(manifest.get("output_sha256"), label=f"{method}.output_sha256")
        if manifest_output.get("evaluation_summary") != expected_summary_sha:
            raise Fold0EnvelopeError(f"manifest output summary hash mismatch for {method}")
        source_hashes = _mapping(manifest.get("source_sha256"), label=f"{method}.source_sha256")
        required_hash_keys = {
            "validation_predictions",
            "fold0_validation_calibration_screen",
            "participant_masks",
            "organ_outcomes",
            "evaluator_code",
        }
        if set(source_hashes) != required_hash_keys:
            raise Fold0EnvelopeError(f"source hash keys mismatch for {method}")
        normalized_hashes = {
            key: _validate_sha(value, label=f"{method}.source_sha256.{key}")
            for key, value in source_hashes.items()
        }
        method_shared = {
            key: normalized_hashes[key]
            for key in (
                "fold0_validation_calibration_screen",
                "participant_masks",
                "organ_outcomes",
                "evaluator_code",
            )
        }
        if shared_hashes is None:
            shared_hashes = method_shared
        elif method_shared != shared_hashes:
            raise Fold0EnvelopeError(f"shared Fold0 development artifact hashes differ for {method}")
        prediction_hashes.append(
            {"method": method, "validation_predictions_sha256": normalized_hashes["validation_predictions"]}
        )

        age_metrics = _mapping(summary.get("age_metrics"), label=f"{method}.age_metrics")
        strict = _mapping(
            age_metrics.get("screen_strict_healthy_reference_60m"), label=f"{method}.screen_strict"
        )
        cox = _mapping(summary.get("cox"), label=f"{method}.cox")
        counts = (
            _integer(strict.get("n_rows"), label=f"{method}.strict.n_rows"),
            _integer(strict.get("n_participants"), label=f"{method}.strict.n_participants"),
            _integer(cox.get("n_cells"), label=f"{method}.cox.n_cells"),
        )
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise Fold0EnvelopeError(f"screen population/count mismatch for {method}")
        if counts[2] != 7:
            raise Fold0EnvelopeError(f"{method} must contain seven calibration-to-screen organ cells")
        organ_rows = cox.get("organs")
        if not isinstance(organ_rows, list):
            raise Fold0EnvelopeError(f"{method}.cox.organs must be a list")
        organ_delta: dict[str, float] = {}
        for organ_value in organ_rows:
            organ_row = _mapping(organ_value, label=f"{method}.organ")
            organ = organ_row.get("organ_system")
            if not isinstance(organ, str) or organ not in ORGANS or organ in organ_delta:
                raise Fold0EnvelopeError(f"invalid or duplicate organ for {method}: {organ!r}")
            organ_delta[organ] = _finite(
                organ_row.get("mean_delta_c"), label=f"{method}.{organ}.mean_delta_c"
            )
        if set(organ_delta) != set(ORGANS):
            raise Fold0EnvelopeError(f"{method} must contain exactly seven canonical organs")
        records.append(
            {
                "tier": tier,
                "method": method,
                "screen_strict_n_rows": counts[0],
                "screen_strict_n_participants": counts[1],
                "screen_strict_mae": _finite(strict.get("mae"), label=f"{method}.strict.mae"),
                "screen_strict_pearson": _finite(
                    strict.get("pearson"), label=f"{method}.strict.pearson"
                ),
                "screen_strict_r2": _finite(strict.get("r2"), label=f"{method}.strict.r2"),
                "full_risk_macro_delta_c": _finite(
                    cox.get("macro_delta_c"), label=f"{method}.cox.macro_delta_c"
                ),
                "positive_organs": _integer(
                    cox.get("positive_organs"), label=f"{method}.cox.positive_organs"
                ),
                "organ_delta_c": organ_delta,
                "summary_sha256": expected_summary_sha,
                "manifest_sha256": expected_manifest_sha,
            }
        )
    artifacts = {
        "shared": shared_hashes,
        "baseline_validation_predictions": prediction_hashes,
        "evaluation_summaries": [
            {"method": row["method"], "sha256": row["summary_sha256"]} for row in records
        ],
        "evaluation_manifests": [
            {"method": row["method"], "sha256": row["manifest_sha256"]} for row in records
        ],
    }
    return records, artifacts, _sha256(registry_path)


def build_envelope(
    records: Sequence[dict[str, object]], source_artifacts: dict[str, object], registry_sha256: str
) -> dict[str, object]:
    specs = (
        ("screen_strict_mae", "lower", 0.01, 4.35),
        ("screen_strict_pearson", "higher", 0.001, None),
        ("screen_strict_r2", "higher", 0.002, None),
        ("full_risk_macro_delta_c", "higher", 0.01, 0.07),
    )
    metrics: dict[str, dict[str, object]] = {}
    for metric, direction, margin, absolute_gate in specs:
        winner = min(records, key=lambda row: float(row[metric])) if direction == "lower" else max(
            records, key=lambda row: float(row[metric])
        )
        baseline_value = float(winner[metric])
        baseline_decimal = Decimal(str(baseline_value))
        margin_decimal = Decimal(str(margin))
        if direction == "lower":
            threshold_decimal = baseline_decimal - margin_decimal
            if absolute_gate is not None:
                threshold_decimal = min(threshold_decimal, Decimal(str(absolute_gate)))
        else:
            threshold_decimal = baseline_decimal + margin_decimal
            if absolute_gate is not None:
                threshold_decimal = max(threshold_decimal, Decimal(str(absolute_gate)))
        metrics[metric] = {
            "direction": direction,
            "baseline_value": baseline_value,
            "source_method": winner["method"],
            "source_tier": winner["tier"],
            "required_margin": margin,
            "absolute_gate": absolute_gate,
            "ehr_age_threshold_exclusive": float(threshold_decimal),
            "ehr_age_threshold_exclusive_decimal": format(threshold_decimal, "f"),
        }
    organ_envelope = []
    for organ in ORGANS:
        winner = max(records, key=lambda row: float(_mapping(row["organ_delta_c"], label="organ")[organ]))
        organ_envelope.append(
            {
                "organ_system": organ,
                "baseline_delta_c": float(_mapping(winner["organ_delta_c"], label="organ")[organ]),
                "source_method": winner["method"],
                "source_tier": winner["tier"],
            }
        )
    return {
        "schema_version": "revision0710_fold0_development_baseline_envelope_v1",
        "scope": {
            **EXPECTED_SCOPE,
            "fit_role": "calibration",
            "score_role": "screen",
            "baseline_scope": "metric-wise best among explicit Tier A plus Tier B methods",
        },
        "source_registry_sha256": registry_sha256,
        "source_artifact_sha256": source_artifacts,
        "methods": [row["method"] for row in records],
        "metric_envelope": metrics,
        "organ_delta_c_envelope": organ_envelope,
        "ehr_age_requirements": {
            "strict_comparison": "exclusive",
            "positive_organs_required": 7,
            "organs_beating_metricwise_baseline_required": 5,
        },
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *, records: Sequence[dict[str, object]], envelope: dict[str, object], output_dir: Path, overwrite: bool
) -> None:
    _reject_glob(output_dir, label="output_dir")
    output_dir = output_dir.expanduser().resolve()
    paths = {name: output_dir / name for name in OUTPUT_FILES}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise Fold0EnvelopeError(f"outputs already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = (
        "tier",
        "method",
        "screen_strict_n_rows",
        "screen_strict_n_participants",
        "screen_strict_mae",
        "screen_strict_pearson",
        "screen_strict_r2",
        "full_risk_macro_delta_c",
        "positive_organs",
        "summary_sha256",
        "manifest_sha256",
    )
    _write_csv(
        paths["fold0_development_baseline_method_table.csv"],
        [{column: row[column] for column in columns} for row in records],
        columns,
    )
    paths["fold0_development_baseline_method_table.json"].write_text(
        json.dumps(
            {
                "schema_version": "revision0710_fold0_development_baseline_method_table_v1",
                "scope": envelope["scope"],
                "source_registry_sha256": envelope["source_registry_sha256"],
                "source_artifact_sha256": envelope["source_artifact_sha256"],
                "methods": list(records),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metric_rows = [
        {"metric": metric, **row}
        for metric, row in _mapping(envelope["metric_envelope"], label="metric_envelope").items()
    ]
    metric_columns = (
        "metric",
        "direction",
        "baseline_value",
        "source_method",
        "source_tier",
        "required_margin",
        "absolute_gate",
        "ehr_age_threshold_exclusive",
        "ehr_age_threshold_exclusive_decimal",
    )
    _write_csv(paths["fold0_development_baseline_envelope.csv"], metric_rows, metric_columns)
    paths["fold0_development_baseline_envelope.json"].write_text(
        json.dumps(envelope, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records, artifacts, registry_sha256 = load_records(args.registry)
    envelope = build_envelope(records, artifacts, registry_sha256)
    write_outputs(records=records, envelope=envelope, output_dir=args.output_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
