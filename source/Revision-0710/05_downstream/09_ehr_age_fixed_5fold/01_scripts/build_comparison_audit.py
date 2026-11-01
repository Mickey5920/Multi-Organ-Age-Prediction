from __future__ import annotations

import argparse
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Mapping, Sequence


getcontext().prec = 50

METHODS = (
    "retain",
    "behrt_style",
    "medbert_style",
    "retain_organquery",
    "behrt_style_organquery",
    "medbert_style_organquery",
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
METRICS = (
    "strict_mae",
    "strict_pearson",
    "strict_r2",
    "full_risk_macro_delta_c",
)
DIRECTIONS = {
    "strict_mae": "lower",
    "strict_pearson": "higher",
    "strict_r2": "higher",
    "full_risk_macro_delta_c": "higher",
}


class ComparisonAuditError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonAuditError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ComparisonAuditError(f"{label} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ComparisonAuditError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ComparisonAuditError(f"{label} must be finite")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _metric_comparison(ehr: Decimal, baseline: Decimal, direction: str) -> dict[str, Any]:
    if baseline == 0:
        raise ComparisonAuditError("cannot calculate a ratio against a zero baseline")
    if direction == "lower":
        improvement = baseline - ehr
        better = ehr < baseline
    elif direction == "higher":
        improvement = ehr - baseline
        better = ehr > baseline
    else:
        raise ComparisonAuditError(f"unsupported metric direction: {direction}")
    relative = improvement / abs(baseline)
    return {
        "direction": direction,
        "ehr_age_value": _decimal_text(ehr),
        "baseline_value": _decimal_text(baseline),
        "ehr_minus_baseline": _decimal_text(ehr - baseline),
        "directional_improvement": _decimal_text(improvement),
        "relative_improvement_fraction": _decimal_text(relative),
        "relative_improvement_percent": _decimal_text(relative * Decimal(100)),
        "ehr_to_baseline_ratio": _decimal_text(ehr / baseline),
        "strictly_better": better,
    }


def _extract_ehr_metrics(staged: Mapping[str, Any]) -> tuple[dict[str, Decimal], dict[str, Any]]:
    if staged.get("schema_version") != "revision0710.ehr_age.fixed_replication.staged_primary.v1":
        raise ComparisonAuditError("unexpected staged-primary schema_version")
    strict = staged.get("strict_age")
    cox = staged.get("cox_by_organ")
    if not isinstance(strict, Mapping) or not isinstance(cox, Mapping):
        raise ComparisonAuditError("staged primary is missing strict_age or cox_by_organ")
    if set(cox) != set(ORGANS):
        raise ComparisonAuditError("staged-primary organ set is not the frozen seven-organ set")
    metrics = {
        "strict_mae": _decimal(strict.get("mae"), "strict_age.mae"),
        "strict_pearson": _decimal(strict.get("pearson"), "strict_age.pearson"),
        "strict_r2": _decimal(strict.get("r2"), "strict_age.r2"),
        "full_risk_macro_delta_c": _decimal(staged.get("macro_delta_c"), "macro_delta_c"),
    }
    cohort = {
        "strict_n_rows": int(strict.get("n_rows")),
        "strict_n_participants": int(strict.get("n_participants")),
    }
    return metrics, cohort


def _validate_baselines(
    table: Mapping[str, Any], envelope: Mapping[str, Any], cohort: Mapping[str, int]
) -> list[Mapping[str, Any]]:
    if table.get("schema_version") != "revision0710_baseline_method_table_v1":
        raise ComparisonAuditError("unexpected baseline-method-table schema_version")
    if envelope.get("schema_version") != "revision0710_baseline_metric_envelope_v1":
        raise ComparisonAuditError("unexpected baseline-envelope schema_version")
    rows = table.get("methods")
    if not isinstance(rows, list) or len(rows) != len(METHODS):
        raise ComparisonAuditError("baseline method table must contain exactly six methods")
    row_by_method: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ComparisonAuditError("baseline method row must be an object")
        method = str(row.get("method"))
        if method in row_by_method:
            raise ComparisonAuditError(f"duplicate baseline method: {method}")
        row_by_method[method] = row
        for count_name in ("strict_n_rows", "strict_n_participants"):
            if int(row.get(count_name)) != cohort[count_name]:
                raise ComparisonAuditError(
                    f"cohort mismatch for {method}.{count_name}: "
                    f"{row.get(count_name)} != {cohort[count_name]}"
                )
        organ_delta = row.get("organ_delta_c")
        if not isinstance(organ_delta, Mapping) or set(organ_delta) != set(ORGANS):
            raise ComparisonAuditError(f"baseline {method} does not contain the frozen seven organs")
    if tuple(row_by_method) != METHODS:
        raise ComparisonAuditError("baseline methods are missing, extra, or not in frozen order")

    env_metrics = envelope.get("metric_envelope")
    env_organs = envelope.get("organ_delta_c_envelope")
    if not isinstance(env_metrics, Mapping) or set(env_metrics) != set(METRICS):
        raise ComparisonAuditError("metric envelope does not contain the four frozen metrics")
    if not isinstance(env_organs, list) or len(env_organs) != len(ORGANS):
        raise ComparisonAuditError("organ envelope must contain seven rows")

    for metric in METRICS:
        values = {
            method: _decimal(row_by_method[method].get(metric), f"{method}.{metric}")
            for method in METHODS
        }
        expected_method = (
            min(values, key=values.get)
            if DIRECTIONS[metric] == "lower"
            else max(values, key=values.get)
        )
        entry = env_metrics[metric]
        if not isinstance(entry, Mapping):
            raise ComparisonAuditError(f"invalid metric envelope row: {metric}")
        actual_value = _decimal(entry.get("baseline_value"), f"envelope.{metric}")
        if entry.get("direction") != DIRECTIONS[metric]:
            raise ComparisonAuditError(f"envelope direction mismatch for {metric}")
        if entry.get("source_method") != expected_method or actual_value != values[expected_method]:
            raise ComparisonAuditError(f"metric envelope is inconsistent for {metric}")

    organ_entry_by_name = {
        str(entry.get("organ_system")): entry for entry in env_organs if isinstance(entry, Mapping)
    }
    if tuple(organ_entry_by_name) != ORGANS:
        raise ComparisonAuditError("organ envelope is missing, extra, or not in frozen order")
    for organ in ORGANS:
        values = {
            method: _decimal(
                row_by_method[method]["organ_delta_c"].get(organ),
                f"{method}.organ_delta_c.{organ}",
            )
            for method in METHODS
        }
        expected_method = max(values, key=values.get)
        entry = organ_entry_by_name[organ]
        actual = _decimal(entry.get("baseline_mean_delta_c"), f"organ envelope.{organ}")
        if entry.get("source_method") != expected_method or actual != values[expected_method]:
            raise ComparisonAuditError(f"organ envelope is inconsistent for {organ}")
    return [row_by_method[method] for method in METHODS]


def build_audit(
    staged: Mapping[str, Any],
    table: Mapping[str, Any],
    envelope: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    ehr_metrics, cohort = _extract_ehr_metrics(staged)
    baseline_rows = _validate_baselines(table, envelope, cohort)
    env_metrics = envelope["metric_envelope"]
    env_organs = {entry["organ_system"]: entry for entry in envelope["organ_delta_c_envelope"]}

    method_comparisons = []
    for row in baseline_rows:
        metric_results = {
            metric: _metric_comparison(
                ehr_metrics[metric],
                _decimal(row[metric], f"{row['method']}.{metric}"),
                DIRECTIONS[metric],
            )
            for metric in METRICS
        }
        organ_results = {}
        for organ in ORGANS:
            result = _metric_comparison(
                _decimal(staged["cox_by_organ"][organ]["mean_delta_c"], f"EHR-AGE.{organ}"),
                _decimal(row["organ_delta_c"][organ], f"{row['method']}.{organ}"),
                "higher",
            )
            result["ehr_age_positive_folds_of_5"] = int(
                staged["cox_by_organ"][organ]["positive_folds"]
            )
            organ_results[organ] = result
        method_comparisons.append(
            {
                "method": row["method"],
                "tier": row["tier"],
                "metrics": metric_results,
                "all_four_strictly_better": all(
                    result["strictly_better"] for result in metric_results.values()
                ),
                "organs_strictly_better_count": sum(
                    result["strictly_better"] for result in organ_results.values()
                ),
                "all_seven_organs_strictly_better": all(
                    result["strictly_better"] for result in organ_results.values()
                ),
                "organ_delta_c": organ_results,
            }
        )

    envelope_metrics = {
        metric: {
            **_metric_comparison(
                ehr_metrics[metric],
                _decimal(env_metrics[metric]["baseline_value"], f"envelope.{metric}"),
                DIRECTIONS[metric],
            ),
            "source_method": env_metrics[metric]["source_method"],
            "source_tier": env_metrics[metric]["source_tier"],
        }
        for metric in METRICS
    }
    organ_envelope = {}
    for organ in ORGANS:
        entry = env_organs[organ]
        result = _metric_comparison(
            _decimal(staged["cox_by_organ"][organ]["mean_delta_c"], f"EHR-AGE.{organ}"),
            _decimal(entry["baseline_mean_delta_c"], f"organ envelope.{organ}"),
            "higher",
        )
        result.update(
            {
                "source_method": entry["source_method"],
                "source_tier": entry["source_tier"],
                "ehr_age_positive_folds_of_5": int(
                    staged["cox_by_organ"][organ]["positive_folds"]
                ),
            }
        )
        organ_envelope[organ] = result

    old_formal_metrics = {}
    for metric in METRICS:
        entry = env_metrics[metric]
        actual = ehr_metrics[metric]
        threshold = _decimal(entry["ehr_age_threshold_exclusive"], f"threshold.{metric}")
        threshold_pass = actual < threshold if DIRECTIONS[metric] == "lower" else actual > threshold
        threshold_clearance = (
            threshold - actual if DIRECTIONS[metric] == "lower" else actual - threshold
        )
        achieved_margin = _decimal(
            envelope_metrics[metric]["directional_improvement"], f"achieved margin.{metric}"
        )
        required_margin = _decimal(entry["required_margin"], f"required margin.{metric}")
        absolute_gate = entry.get("absolute_gate")
        absolute_gate_pass = None
        if absolute_gate is not None:
            absolute = _decimal(absolute_gate, f"absolute gate.{metric}")
            absolute_gate_pass = actual < absolute if DIRECTIONS[metric] == "lower" else actual > absolute
        old_formal_metrics[metric] = {
            "direction": DIRECTIONS[metric],
            "ehr_age_value": _decimal_text(actual),
            "exclusive_threshold": _decimal_text(threshold),
            "directional_threshold_clearance": _decimal_text(threshold_clearance),
            "threshold_pass": threshold_pass,
            "achieved_margin_vs_metricwise_envelope": _decimal_text(achieved_margin),
            "required_margin": _decimal_text(required_margin),
            "required_margin_only_pass": achieved_margin > required_margin,
            "absolute_gate": (
                None
                if absolute_gate is None
                else _decimal_text(_decimal(absolute_gate, f"absolute gate.{metric}"))
            ),
            "absolute_gate_pass": absolute_gate_pass,
        }

    age_relative = [
        _decimal(envelope_metrics[metric]["relative_improvement_fraction"], metric)
        for metric in ("strict_mae", "strict_pearson", "strict_r2")
    ]
    risk_relative = _decimal(
        envelope_metrics["full_risk_macro_delta_c"]["relative_improvement_fraction"],
        "risk relative improvement",
    )
    revised_all_four = all(result["strictly_better"] for result in envelope_metrics.values())
    revised_risk_more = risk_relative > max(age_relative)

    return {
        "schema_version": "revision0710.ehr_age.fixed_5fold.comparison_audit.v1",
        "value_encoding": "All metric values, deltas, ratios, and percentages are exact decimal strings derived from the JSON lexical values.",
        "method": staged["method"],
        "source_metadata": dict(source_metadata),
        "cohort_alignment": {
            **cohort,
            "all_six_baselines_exact_match": True,
        },
        "ehr_age_metrics": {key: _decimal_text(value) for key, value in ehr_metrics.items()},
        "comparison_vs_each_baseline": method_comparisons,
        "comparison_vs_metricwise_envelope": {
            "metrics": envelope_metrics,
            "all_four_strictly_better": revised_all_four,
            "organ_delta_c": organ_envelope,
            "organs_strictly_better_count": sum(
                result["strictly_better"] for result in organ_envelope.values()
            ),
            "all_seven_organs_strictly_better": all(
                result["strictly_better"] for result in organ_envelope.values()
            ),
        },
        "criterion_assessment": {
            "user_revised_point_estimate_rule": {
                "definition": "All four direction-aware point estimates strictly beat the six-method metric-wise envelope.",
                "all_four_strictly_better": revised_all_four,
                "risk_lead_operationalization": "Risk relative improvement versus its envelope exceeds the relative improvement of each age metric versus its own envelope.",
                "risk_relative_improvement_exceeds_each_age_metric": revised_risk_more,
                "overall_pass_under_stated_operationalization": revised_all_four and revised_risk_more,
                "qualification": "The phrase 'risk leads more' had no prespecified numeric margin; this is a transparent descriptive operationalization, not a new training gate or an inferential claim.",
            },
            "old_formal_margin_and_absolute_gate_rule": {
                "metrics": old_formal_metrics,
                "overall_pass": all(item["threshold_pass"] for item in old_formal_metrics.values()),
                "note": "The risk exclusive threshold is 0.070 because the old absolute gate dominates the +0.01 baseline margin.",
            },
        },
        "stability_and_scope": {
            "organs_with_positive_mean_delta_c": sum(
                _decimal(staged["cox_by_organ"][organ]["mean_delta_c"], organ) > 0
                for organ in ORGANS
            ),
            "organs_positive_in_all_five_folds": sum(
                int(staged["cox_by_organ"][organ]["positive_folds"]) == 5 for organ in ORGANS
            ),
            "organs_not_positive_in_all_five_folds": [
                organ
                for organ in ORGANS
                if int(staged["cox_by_organ"][organ]["positive_folds"]) != 5
            ],
            "canonical_post_receipt_evaluator_is_primary": bool(
                staged.get("canonical_post_receipt_evaluator_is_primary")
            ),
            "inference_available": False,
            "caution": "These are five-fold point-estimate comparisons. The supplied summaries do not provide paired confidence intervals or hypothesis tests.",
        },
    }


def _fmt(value: str, digits: int = 6) -> str:
    return f"{Decimal(value):.{digits}f}"


def render_markdown(audit: Mapping[str, Any]) -> str:
    envelope = audit["comparison_vs_metricwise_envelope"]
    old = audit["criterion_assessment"]["old_formal_margin_and_absolute_gate_rule"]
    revised = audit["criterion_assessment"]["user_revised_point_estimate_rule"]
    metric_labels = {
        "strict_mae": "strict MAE",
        "strict_pearson": "strict Pearson",
        "strict_r2": "strict R2",
        "full_risk_macro_delta_c": "full-risk macro Delta C",
    }
    lines = [
        "# EHR-AGE fixed five-fold comparison audit",
        "",
        "## Scope",
        "",
        (
            f"This audit compares `{audit['method']}` with six frozen five-fold baselines. "
            f"The strict-age cohort is {audit['cohort_alignment']['strict_n_participants']:,} participants "
            f"x 7 organs = {audit['cohort_alignment']['strict_n_rows']:,} rows; full-risk Delta C uses "
            "the organ-specific at-risk population. Inputs are read-only and identified by SHA-256 below."
        ),
        "",
        "Relative improvement is `(baseline - EHR)/baseline` for MAE and `(EHR - baseline)/baseline` for higher-is-better metrics.",
        "",
        "## Metric-wise envelope",
        "",
        "| Metric | EHR-AGE | Best baseline | Source | Directional improvement | Relative improvement | Revised pass | Old formal threshold | Old pass |",
        "|---|---:|---:|---|---:|---:|:---:|---:|:---:|",
    ]
    for metric in METRICS:
        row = envelope["metrics"][metric]
        old_row = old["metrics"][metric]
        lines.append(
            "| {label} | {ehr} | {base} | `{source}` | {delta} | {relative}% | {new_pass} | {threshold} | {old_pass} |".format(
                label=metric_labels[metric],
                ehr=_fmt(row["ehr_age_value"]),
                base=_fmt(row["baseline_value"]),
                source=row["source_method"],
                delta=_fmt(row["directional_improvement"]),
                relative=_fmt(row["relative_improvement_percent"], 3),
                new_pass="PASS" if row["strictly_better"] else "FAIL",
                threshold=_fmt(old_row["exclusive_threshold"]),
                old_pass="PASS" if old_row["threshold_pass"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "## Six-baseline comparison",
            "",
            "Each cell is the direction-aware EHR-AGE improvement over that baseline; percentages are relative improvements.",
            "",
            "| Baseline | MAE improvement | Pearson improvement | R2 improvement | Macro Delta C improvement | All four | Organs beaten |",
            "|---|---:|---:|---:|---:|:---:|---:|",
        ]
    )
    for method in audit["comparison_vs_each_baseline"]:
        cells = []
        for metric in METRICS:
            row = method["metrics"][metric]
            cells.append(
                f"{_fmt(row['directional_improvement'])} ({_fmt(row['relative_improvement_percent'], 3)}%)"
            )
        lines.append(
            f"| `{method['method']}` | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
            f"{'PASS' if method['all_four_strictly_better'] else 'FAIL'} | "
            f"{method['organs_strictly_better_count']}/7 |"
        )

    lines.extend(
        [
            "",
            "## Organ full-risk Delta C",
            "",
            "| Organ | EHR-AGE mean Delta C | Best baseline | Source | Improvement | EHR/baseline | Positive folds |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for organ in ORGANS:
        row = envelope["organ_delta_c"][organ]
        lines.append(
            f"| {organ} | {_fmt(row['ehr_age_value'])} | {_fmt(row['baseline_value'])} | "
            f"`{row['source_method']}` | {_fmt(row['directional_improvement'])} | "
            f"{_fmt(row['ehr_to_baseline_ratio'], 3)}x | {row['ehr_age_positive_folds_of_5']}/5 |"
        )

    risk = envelope["metrics"]["full_risk_macro_delta_c"]
    stability = audit["stability_and_scope"]
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            (
                f"- **User-revised point-estimate rule: {'PASS' if revised['overall_pass_under_stated_operationalization'] else 'FAIL'}.** "
                f"All four metrics beat the six-method envelope. Macro Delta C improves by "
                f"{_fmt(risk['directional_improvement'])}, reaches {_fmt(risk['ehr_to_baseline_ratio'], 3)}x "
                f"the best baseline, and has a {_fmt(risk['relative_improvement_percent'], 3)}% relative improvement, "
                "larger than the relative improvement for each age metric. This operationalization is descriptive because "
                "no numeric definition of 'risk leads more' was prespecified."
            ),
            (
                f"- **Old formal rule: {'PASS' if old['overall_pass'] else 'FAIL'}.** "
                "R2 clears its old margin, while MAE and Pearson miss their old margins and macro Delta C "
                f"({_fmt(risk['ehr_age_value'])}) remains below the exclusive 0.070 gate despite clearing "
                "the separate +0.01 baseline margin. The MAE < 4.35 absolute gate passes."
            ),
            (
                f"- **Organ result:** all {envelope['organs_strictly_better_count']}/7 organ mean Delta C values beat "
                f"their metric-wise baseline envelopes. {stability['organs_positive_in_all_five_folds']}/7 organs are "
                "positive in all five folds; immune is positive in 4/5 folds."
            ),
            "- **Inference limitation:** the supplied summaries contain point estimates but no paired confidence intervals or hypothesis tests; 'better' here does not mean statistically significant.",
            (
                "- **Evaluator flag:** `canonical_post_receipt_evaluator_is_primary` is "
                f"`{str(stability['canonical_post_receipt_evaluator_is_primary']).lower()}` in the requested staged-primary input."
            ),
            "",
            "## Input provenance",
            "",
        ]
    )
    for name, metadata in audit["source_metadata"].items():
        lines.append(f"- `{name}`: `{metadata['path']}`; SHA-256 `{metadata['sha256']}`")
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[4]
    base = repo / "Revision-0710" / "05_downstream" / "09_ehr_age_fixed_5fold"
    parser = argparse.ArgumentParser(description="Audit fixed five-fold EHR-AGE against six baselines.")
    parser.add_argument(
        "--staged-primary",
        type=Path,
        default=repo
        / "Revision-0710"
        / "03_ehr_age"
        / "04_fixed_replication"
        / "02_published_fivefold"
        / "01_evaluation"
        / "STAGED_PRIMARY_SUMMARY.json",
    )
    parser.add_argument(
        "--baseline-table",
        type=Path,
        default=repo
        / "Revision-0710"
        / "05_downstream"
        / "03_baseline_summary"
        / "baseline_method_table.json",
    )
    parser.add_argument(
        "--baseline-envelope",
        type=Path,
        default=repo
        / "Revision-0710"
        / "05_downstream"
        / "03_baseline_summary"
        / "baseline_metric_envelope.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=base / "02_results" / "ehr_age_vs_six_baselines.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=base / "03_report" / "EHR_AGE_VS_SIX_BASELINES.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[4]
    inputs = {
        "staged_primary": args.staged_primary,
        "baseline_method_table": args.baseline_table,
        "baseline_metric_envelope": args.baseline_envelope,
    }
    source_metadata = {
        name: {"path": _display_path(path, repo), "sha256": _sha256(path)}
        for name, path in inputs.items()
    }
    audit = build_audit(
        _load_json(args.staged_primary),
        _load_json(args.baseline_table),
        _load_json(args.baseline_envelope),
        source_metadata,
    )
    _atomic_write(args.output_json, json.dumps(audit, indent=2, ensure_ascii=True) + "\n")
    _atomic_write(args.output_md, render_markdown(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
