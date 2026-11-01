#!/usr/bin/env python
"""Build the controlled eight-method Revision-0710 baseline envelope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Sequence


SCHEMA_VERSION = "revision0710_baseline_evaluation_registry_v1"
SUMMARY_SCHEMA_VERSION = "revision0710_evaluation_summary_v1"
EXPECTED_METHODS = (
    ("retain", "tier_a"),
    ("behrt_style", "tier_a"),
    ("medbert_style", "tier_a"),
    ("ehrformer_style", "tier_a"),
    ("retain_organquery", "tier_b"),
    ("behrt_style_organquery", "tier_b"),
    ("medbert_style_organquery", "tier_b"),
    ("ehrformer_style_organquery", "tier_b"),
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
OUTPUT_FILENAMES = (
    "baseline_method_table.csv",
    "baseline_method_table.json",
    "baseline_metric_envelope.csv",
    "baseline_metric_envelope.json",
)


class BaselineSummaryError(ValueError):
    """Raised when the explicit baseline registry or a summary violates the contract."""


def _reject_glob(path: Path, *, label: str) -> None:
    if any(token in str(path) for token in ("*", "?", "[", "]")):
        raise BaselineSummaryError(f"{label} must be explicit and contain no glob tokens")


def _explicit_file(path: Path, *, label: str) -> Path:
    _reject_glob(path, label=label)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BaselineSummaryError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineSummaryError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineSummaryError(f"{label} must contain a JSON object")
    return payload


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BaselineSummaryError(f"{label} must be a JSON object")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineSummaryError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BaselineSummaryError(f"{label} must be finite numeric")
    return result


def _integer(value: object, *, label: str) -> int:
    result = _finite(value, label=label)
    if not result.is_integer():
        raise BaselineSummaryError(f"{label} must be an integer")
    return int(result)


def load_explicit_summaries(registry_path: Path) -> tuple[list[dict[str, object]], str]:
    registry_path = _explicit_file(registry_path, label="registry")
    registry = _read_object(registry_path, label="registry")
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise BaselineSummaryError("unsupported registry schema_version")
    entries = registry.get("methods")
    if not isinstance(entries, list):
        raise BaselineSummaryError("registry.methods must be a list")

    expected = dict(EXPECTED_METHODS)
    observed: dict[str, tuple[str, Path, str]] = {}
    for index, entry_value in enumerate(entries):
        entry = _mapping(entry_value, label=f"registry.methods[{index}]")
        method = entry.get("method")
        tier = entry.get("tier")
        raw_path = entry.get("summary_path")
        expected_sha = entry.get("sha256")
        if not isinstance(method, str) or not method:
            raise BaselineSummaryError(f"registry.methods[{index}].method must be non-empty")
        if method in observed:
            raise BaselineSummaryError(f"registry contains duplicate method: {method}")
        if method not in expected:
            raise BaselineSummaryError(f"registry contains unexpected method: {method}")
        if tier != expected[method]:
            raise BaselineSummaryError(
                f"registry tier mismatch for {method}: expected={expected[method]!r}, observed={tier!r}"
            )
        if not isinstance(raw_path, str) or not raw_path:
            raise BaselineSummaryError(f"registry summary_path missing for {method}")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha.lower())
        ):
            raise BaselineSummaryError(f"registry sha256 is invalid for {method}")
        summary_path = Path(raw_path)
        if not summary_path.is_absolute():
            summary_path = registry_path.parent / summary_path
        summary_path = _explicit_file(summary_path, label=f"summary_path[{method}]")
        observed_sha = _sha256(summary_path)
        if observed_sha != expected_sha.lower():
            raise BaselineSummaryError(
                f"summary sha256 mismatch for {method}: expected={expected_sha}, observed={observed_sha}"
            )
        observed[method] = (str(tier), summary_path, observed_sha)

    missing = [method for method, _ in EXPECTED_METHODS if method not in observed]
    if missing:
        raise BaselineSummaryError(f"registry is missing required methods: {missing}")
    if len(entries) != len(EXPECTED_METHODS):
        raise BaselineSummaryError("registry must contain exactly the eight controlled methods")

    records: list[dict[str, object]] = []
    reference_counts: tuple[int, int, int] | None = None
    for method, expected_tier in EXPECTED_METHODS:
        tier, summary_path, summary_sha = observed[method]
        summary = _read_object(summary_path, label=f"summary[{method}]")
        if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
            raise BaselineSummaryError(f"unsupported summary schema_version for {method}")
        if summary.get("method") != method:
            raise BaselineSummaryError(
                f"summary method mismatch: expected={method!r}, observed={summary.get('method')!r}"
            )
        age_metrics = _mapping(summary.get("age_metrics"), label=f"{method}.age_metrics")
        strict = _mapping(
            age_metrics.get("strict_healthy_reference_60m"),
            label=f"{method}.age_metrics.strict_healthy_reference_60m",
        )
        cox = _mapping(summary.get("cox"), label=f"{method}.cox")
        strict_rows = _integer(strict.get("n_rows"), label=f"{method}.strict.n_rows")
        strict_participants = _integer(
            strict.get("n_participants"), label=f"{method}.strict.n_participants"
        )
        n_cells = _integer(cox.get("n_cells"), label=f"{method}.cox.n_cells")
        counts = (strict_rows, strict_participants, n_cells)
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise BaselineSummaryError(
                f"evaluation population/count mismatch for {method}: expected={reference_counts}, observed={counts}"
            )
        if n_cells != 35:
            raise BaselineSummaryError(f"{method} must contain all 35 fold-organ Cox cells")

        organ_values = cox.get("organs")
        if not isinstance(organ_values, list):
            raise BaselineSummaryError(f"{method}.cox.organs must be a list")
        organ_delta: dict[str, float] = {}
        for organ_index, organ_value in enumerate(organ_values):
            organ_row = _mapping(organ_value, label=f"{method}.cox.organs[{organ_index}]")
            organ = organ_row.get("organ_system")
            if not isinstance(organ, str) or organ not in ORGANS:
                raise BaselineSummaryError(f"{method} contains an invalid organ_system: {organ!r}")
            if organ in organ_delta:
                raise BaselineSummaryError(f"{method} contains duplicate organ_system: {organ}")
            organ_delta[organ] = _finite(
                organ_row.get("mean_delta_c"), label=f"{method}.{organ}.mean_delta_c"
            )
        if set(organ_delta) != set(ORGANS):
            raise BaselineSummaryError(f"{method} does not contain exactly the seven canonical organs")

        records.append(
            {
                "tier": expected_tier,
                "method": method,
                "strict_n_rows": strict_rows,
                "strict_n_participants": strict_participants,
                "strict_mae": _finite(strict.get("mae"), label=f"{method}.strict.mae"),
                "strict_pearson": _finite(
                    strict.get("pearson"), label=f"{method}.strict.pearson"
                ),
                "strict_r2": _finite(strict.get("r2"), label=f"{method}.strict.r2"),
                "full_risk_macro_delta_c": _finite(
                    cox.get("macro_delta_c"), label=f"{method}.cox.macro_delta_c"
                ),
                "positive_organs": _integer(
                    cox.get("positive_organs"), label=f"{method}.cox.positive_organs"
                ),
                "organ_delta_c": organ_delta,
                "summary_sha256": summary_sha,
            }
        )
    return records, _sha256(registry_path)


def build_envelope(records: Sequence[dict[str, object]], registry_sha256: str) -> dict[str, object]:
    metric_specs = (
        ("strict_mae", "lower", 0.01, 4.35),
        ("strict_pearson", "higher", 0.001, None),
        ("strict_r2", "higher", 0.002, None),
        ("full_risk_macro_delta_c", "higher", 0.01, 0.07),
    )
    envelope: dict[str, dict[str, object]] = {}
    for metric, direction, margin, absolute_gate in metric_specs:
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
        envelope[metric] = {
            "direction": direction,
            "baseline_value": baseline_value,
            "source_method": winner["method"],
            "source_tier": winner["tier"],
            "required_margin": margin,
            "absolute_gate": absolute_gate,
            "ehr_age_threshold_exclusive": float(threshold_decimal),
            "ehr_age_threshold_exclusive_decimal": format(threshold_decimal, "f"),
        }

    organ_envelope: list[dict[str, object]] = []
    for organ in ORGANS:
        winner = max(records, key=lambda row: float(_mapping(row["organ_delta_c"], label="organ_delta_c")[organ]))
        organ_envelope.append(
            {
                "organ_system": organ,
                "baseline_mean_delta_c": float(
                    _mapping(winner["organ_delta_c"], label="organ_delta_c")[organ]
                ),
                "source_method": winner["method"],
                "source_tier": winner["tier"],
            }
        )

    return {
        "schema_version": "revision0710_baseline_metric_envelope_v1",
        "source_registry_sha256": registry_sha256,
        "scope": "Tier A plus Tier B; metric-wise best of eight explicit five-fold evaluations",
        "methods": [row["method"] for row in records],
        "metric_envelope": envelope,
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
    output_paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise BaselineSummaryError(f"outputs already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_columns = (
        "tier",
        "method",
        "strict_n_rows",
        "strict_n_participants",
        "strict_mae",
        "strict_pearson",
        "strict_r2",
        "full_risk_macro_delta_c",
        "positive_organs",
        "summary_sha256",
    )
    _write_csv(
        output_paths["baseline_method_table.csv"],
        [{column: row[column] for column in flat_columns} for row in records],
        flat_columns,
    )
    method_payload = {
        "schema_version": "revision0710_baseline_method_table_v1",
        "source_registry_sha256": envelope["source_registry_sha256"],
        "methods": list(records),
    }
    output_paths["baseline_method_table.json"].write_text(
        json.dumps(method_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    metric_rows = []
    metric_envelope = _mapping(envelope["metric_envelope"], label="metric_envelope")
    for metric in (
        "strict_mae",
        "strict_pearson",
        "strict_r2",
        "full_risk_macro_delta_c",
    ):
        row = {"metric": metric, **_mapping(metric_envelope[metric], label=metric)}
        metric_rows.append(row)
    envelope_columns = (
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
    _write_csv(output_paths["baseline_metric_envelope.csv"], metric_rows, envelope_columns)
    output_paths["baseline_metric_envelope.json"].write_text(
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
    records, registry_sha256 = load_explicit_summaries(args.registry)
    envelope = build_envelope(records, registry_sha256)
    write_outputs(records=records, envelope=envelope, output_dir=args.output_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
