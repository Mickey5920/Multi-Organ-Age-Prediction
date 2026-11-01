#!/usr/bin/env python
"""Re-evaluate the six frozen baselines against corrected-primary outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
REVISION = DOWNSTREAM.parent
EVALUATOR = DOWNSTREAM / "scripts/evaluate_method_5fold.py"
SUMMARIZER = DOWNSTREAM / "scripts/summarize_baseline_envelope.py"
CORRECTED_ROOT = REVISION / "01_data_rebuild/07_corrected_primary_v1"
PARTICIPANT_MASKS = (
    CORRECTED_ROOT / "02_outcomes_60m/participant_followup_and_evaluation_masks.parquet"
)
ORGAN_OUTCOMES = CORRECTED_ROOT / "02_outcomes_60m/organ_outcomes_60m.parquet"
FOLD_ROLES = CORRECTED_ROOT / "03_folds/outer_fold_roles.parquet"
OLD_REGISTRY = DOWNSTREAM / "03_baseline_summary/baseline_evaluation_registry.json"

METHODS = (
    ("retain", "tier_a", DOWNSTREAM / "01_prediction_registries/tier_a/retain.json"),
    ("behrt_style", "tier_a", DOWNSTREAM / "01_prediction_registries/tier_a/behrt_style.json"),
    ("medbert_style", "tier_a", DOWNSTREAM / "01_prediction_registries/tier_a/medbert_style.json"),
    (
        "retain_organquery",
        "tier_b",
        DOWNSTREAM / "01_prediction_registries/tier_b/retain_organquery.json",
    ),
    (
        "behrt_style_organquery",
        "tier_b",
        DOWNSTREAM / "01_prediction_registries/tier_b/behrt_style_organquery.json",
    ),
    (
        "medbert_style_organquery",
        "tier_b",
        DOWNSTREAM / "01_prediction_registries/tier_b/medbert_style_organquery.json",
    ),
)
STRICT_METRICS = ("n_rows", "n_participants", "mae", "pearson", "r2")
EVALUATION_OUTPUT_FILES = (
    "raw_age_metrics.csv",
    "linear_correction_parameters.csv",
    "cox_fold_organ_cells.csv",
    "cox_organ_summary.csv",
    "cox_summary.json",
    "evaluation_summary.json",
    "evaluation_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def require_files(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")


def resolve_old_summaries() -> dict[str, Path]:
    registry = load_json(OLD_REGISTRY)
    result: dict[str, Path] = {}
    for item in registry["methods"]:
        method = str(item["method"])
        path = Path(str(item["summary_path"]))
        if not path.is_absolute():
            path = OLD_REGISTRY.parent / path
        result[method] = path.resolve()
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Rebuild audits and the comprehensive manifest without re-running evaluations.",
    )
    args = parser.parse_args()

    required = (
        EVALUATOR,
        SUMMARIZER,
        PARTICIPANT_MASKS,
        ORGAN_OUTCOMES,
        FOLD_ROLES,
        OLD_REGISTRY,
        *(registry for _, _, registry in METHODS),
    )
    require_files(required)

    evaluation_root = HERE / "01_baseline_evaluation"
    summary_root = HERE / "02_baseline_summary"
    status_path = HERE / "RUN_STATUS.json"
    corrected_registry = summary_root / "baseline_evaluation_registry.json"
    completed: list[str] = [method for method, _, _ in METHODS] if args.finalize_only else []
    if args.finalize_only:
        require_files(
            (
                corrected_registry,
                *(evaluation_root / method / name for method, _, _ in METHODS for name in EVALUATION_OUTPUT_FILES),
                summary_root / "baseline_method_table.csv",
                summary_root / "baseline_method_table.json",
                summary_root / "baseline_metric_envelope.csv",
                summary_root / "baseline_metric_envelope.json",
            )
        )
    else:
        write_json(
            status_path,
            {
                "schema_version": "revision0710_corrected_baseline_run_status_v1",
                "status": "RUNNING",
                "completed_methods": [],
            },
        )
        for method, _, prediction_registry in METHODS:
            output_dir = evaluation_root / method
            command = [
                sys.executable,
                str(EVALUATOR),
                "--prediction-registry",
                str(prediction_registry),
                "--method",
                method,
                "--participant-masks",
                str(PARTICIPANT_MASKS),
                "--organ-outcomes",
                str(ORGAN_OUTCOMES),
                "--fold-roles",
                str(FOLD_ROLES),
                "--output-dir",
                str(output_dir),
            ]
            if args.overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)
            completed.append(method)
            write_json(
                status_path,
                {
                    "schema_version": "revision0710_corrected_baseline_run_status_v1",
                    "status": "RUNNING",
                    "completed_methods": completed,
                },
            )

        registry_entries = []
        for method, tier, _ in METHODS:
            summary_path = evaluation_root / method / "evaluation_summary.json"
            registry_entries.append(
                {
                    "method": method,
                    "tier": tier,
                    "summary_path": str(Path("../01_baseline_evaluation") / method / summary_path.name),
                    "sha256": sha256_file(summary_path),
                }
            )
        write_json(
            corrected_registry,
            {
                "schema_version": "revision0710_baseline_evaluation_registry_v1",
                "selection": (
                    "six explicit frozen five-fold prediction registries re-evaluated against "
                    "corrected-primary outcomes; no discovery or globbing"
                ),
                "methods": registry_entries,
            },
        )

        summarize_command = [
            sys.executable,
            str(SUMMARIZER),
            "--registry",
            str(corrected_registry),
            "--output-dir",
            str(summary_root),
        ]
        if args.overwrite:
            summarize_command.append("--overwrite")
        subprocess.run(summarize_command, check=True)

    old_summaries = resolve_old_summaries()
    audit_rows: list[dict[str, object]] = []
    strict_identical = True
    for method, _, _ in METHODS:
        old = load_json(old_summaries[method])
        new_path = evaluation_root / method / "evaluation_summary.json"
        new = load_json(new_path)
        old_strict = old["age_metrics"]["strict_healthy_reference_60m"]
        new_strict = new["age_metrics"]["strict_healthy_reference_60m"]
        strict_equal = all(old_strict[key] == new_strict[key] for key in STRICT_METRICS)
        strict_identical = strict_identical and strict_equal
        old_cox = old["cox"]
        new_cox = new["cox"]
        audit_rows.append(
            {
                "method": method,
                "strict_metrics_identical": strict_equal,
                "strict_mae": new_strict["mae"],
                "strict_pearson": new_strict["pearson"],
                "strict_r2": new_strict["r2"],
                "old_full_risk_macro_delta_c": old_cox["macro_delta_c"],
                "corrected_full_risk_macro_delta_c": new_cox["macro_delta_c"],
                "macro_delta_c_change": new_cox["macro_delta_c"] - old_cox["macro_delta_c"],
            }
        )
    if not strict_identical:
        raise RuntimeError("strict age metrics changed even though the strict cohort is invariant")

    audit_csv = HERE / "03_audit/baseline_outcome_change_audit.csv"
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    write_json(
        HERE / "03_audit/baseline_outcome_change_audit.json",
        {
            "schema_version": "revision0710_corrected_baseline_change_audit_v1",
            "strict_metrics_identical_for_all_methods": strict_identical,
            "methods": audit_rows,
        },
    )

    method_table = load_json(summary_root / "baseline_method_table.json")["methods"]
    best_mae = min(method_table, key=lambda row: float(row["strict_mae"]))
    best_pearson = max(method_table, key=lambda row: float(row["strict_pearson"]))
    best_r2 = max(method_table, key=lambda row: float(row["strict_r2"]))
    best_delta_c = max(method_table, key=lambda row: float(row["full_risk_macro_delta_c"]))
    decision_path = summary_root / "DECISION_RULE_STATUS.json"
    decision_payload = {
        "schema_version": "revision0710_corrected_baseline_decision_status_v1",
        "status": "OBSERVED_BASELINES_FROZEN; EHR_AGE_MARGIN_PENDING_CORRECTED_FOLD0_REVIEW",
        "observed_metricwise_best": {
            "strict_mae": {"direction": "lower", "method": best_mae["method"], "value": best_mae["strict_mae"]},
            "strict_pearson": {"direction": "higher", "method": best_pearson["method"], "value": best_pearson["strict_pearson"]},
            "strict_r2": {"direction": "higher", "method": best_r2["method"], "value": best_r2["strict_r2"]},
            "full_risk_macro_delta_c": {
                "direction": "higher",
                "method": best_delta_c["method"],
                "value": best_delta_c["full_risk_macro_delta_c"],
            },
        },
        "current_interpretation": {
            "strict_metrics": "EHR-AGE must beat the observed best baseline; only a small lead is required.",
            "full_risk_macro_delta_c": "EHR-AGE must beat the observed best baseline with a materially larger lead.",
            "margin_status": "No numeric lead margin is frozen before corrected-primary Fold0 review.",
        },
        "inherited_envelope_notice": (
            "baseline_metric_envelope retains the earlier absolute full-risk gate of 0.07 for provenance; "
            "it is not the current empirical baseline-only decision rule."
        ),
    }
    write_json(decision_path, decision_payload)
    decision_markdown = summary_root / "DECISION_RULE_STATUS.md"
    decision_markdown.write_text(
        "\n".join(
            [
                "# Corrected-primary baseline decision status",
                "",
                "Observed best baselines are frozen; the EHR-AGE lead margin remains pending corrected Fold0 review.",
                "",
                "| Metric | Direction | Best method | Value |",
                "|---|---|---|---:|",
                f"| strict MAE | lower | {best_mae['method']} | {best_mae['strict_mae']} |",
                f"| strict Pearson | higher | {best_pearson['method']} | {best_pearson['strict_pearson']} |",
                f"| strict R2 | higher | {best_r2['method']} | {best_r2['strict_r2']} |",
                f"| full-risk macro Delta C | higher | {best_delta_c['method']} | {best_delta_c['full_risk_macro_delta_c']} |",
                "",
                "The inherited 0.07 absolute gate is retained only for provenance and is not the current empirical baseline-only rule.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_files = [
        PARTICIPANT_MASKS,
        ORGAN_OUTCOMES,
        FOLD_ROLES,
        *(registry for _, _, registry in METHODS),
        corrected_registry,
        *(evaluation_root / method / name for method, _, _ in METHODS for name in EVALUATION_OUTPUT_FILES),
        summary_root / "baseline_method_table.csv",
        summary_root / "baseline_method_table.json",
        summary_root / "baseline_metric_envelope.csv",
        summary_root / "baseline_metric_envelope.json",
        decision_path,
        decision_markdown,
        audit_csv,
        HERE / "03_audit/baseline_outcome_change_audit.json",
    ]
    write_json(
        HERE / "MANIFEST.json",
        {
            "schema_version": "revision0710_corrected_baseline_manifest_v2",
            "python_executable": sys.executable,
            "files": [
                {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in manifest_files
            ],
        },
    )
    write_json(
        status_path,
        {
            "schema_version": "revision0710_corrected_baseline_run_status_v1",
            "status": "COMPLETE",
            "completed_methods": completed,
            "strict_metrics_identical_for_all_methods": strict_identical,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
