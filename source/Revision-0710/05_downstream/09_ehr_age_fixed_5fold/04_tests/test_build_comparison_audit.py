from __future__ import annotations

import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "01_scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_comparison_audit import (  # noqa: E402
    ComparisonAuditError,
    METHODS,
    ORGANS,
    build_audit,
    main,
)


def _fixtures() -> tuple[dict, dict, dict]:
    staged = {
        "schema_version": "revision0710.ehr_age.fixed_replication.staged_primary.v1",
        "method": "ehr_age_test",
        "strict_age": {
            "n_rows": 700,
            "n_participants": 100,
            "mae": Decimal("3.99"),
            "pearson": Decimal("0.701"),
            "r2": Decimal("0.503"),
        },
        "cox_by_organ": {
            organ: {
                "mean_delta_c": Decimal("0.05") + Decimal(index) / Decimal("1000"),
                "positive_folds": 4 if organ == "immune" else 5,
            }
            for index, organ in enumerate(ORGANS)
        },
        "macro_delta_c": Decimal("0.05"),
        "canonical_post_receipt_evaluator_is_primary": False,
    }
    rows = []
    for index, method in enumerate(METHODS):
        decrement = Decimal(index) / Decimal("1000")
        rows.append(
            {
                "tier": "tier_a" if index < 3 else "tier_b",
                "method": method,
                "strict_n_rows": 700,
                "strict_n_participants": 100,
                "strict_mae": Decimal("4.0") + Decimal(index) / Decimal("10"),
                "strict_pearson": Decimal("0.700") - decrement,
                "strict_r2": Decimal("0.500") - decrement,
                "full_risk_macro_delta_c": Decimal("0.020") - decrement,
                "positive_organs": 7,
                "organ_delta_c": {
                    organ: Decimal("0.010") - decrement + Decimal(organ_index) / Decimal("10000")
                    for organ_index, organ in enumerate(ORGANS)
                },
            }
        )
    table = {
        "schema_version": "revision0710_baseline_method_table_v1",
        "methods": rows,
    }
    envelope = {
        "schema_version": "revision0710_baseline_metric_envelope_v1",
        "metric_envelope": {
            "strict_mae": {
                "direction": "lower",
                "baseline_value": Decimal("4.0"),
                "source_method": METHODS[0],
                "source_tier": "tier_a",
                "required_margin": Decimal("0.01"),
                "absolute_gate": Decimal("4.35"),
                "ehr_age_threshold_exclusive": Decimal("3.99"),
            },
            "strict_pearson": {
                "direction": "higher",
                "baseline_value": Decimal("0.700"),
                "source_method": METHODS[0],
                "source_tier": "tier_a",
                "required_margin": Decimal("0.001"),
                "absolute_gate": None,
                "ehr_age_threshold_exclusive": Decimal("0.701"),
            },
            "strict_r2": {
                "direction": "higher",
                "baseline_value": Decimal("0.500"),
                "source_method": METHODS[0],
                "source_tier": "tier_a",
                "required_margin": Decimal("0.002"),
                "absolute_gate": None,
                "ehr_age_threshold_exclusive": Decimal("0.502"),
            },
            "full_risk_macro_delta_c": {
                "direction": "higher",
                "baseline_value": Decimal("0.020"),
                "source_method": METHODS[0],
                "source_tier": "tier_a",
                "required_margin": Decimal("0.01"),
                "absolute_gate": Decimal("0.07"),
                "ehr_age_threshold_exclusive": Decimal("0.07"),
            },
        },
        "organ_delta_c_envelope": [
            {
                "organ_system": organ,
                "baseline_mean_delta_c": Decimal("0.010")
                + Decimal(index) / Decimal("10000"),
                "source_method": METHODS[0],
                "source_tier": "tier_a",
            }
            for index, organ in enumerate(ORGANS)
        ],
    }
    return staged, table, envelope


def _json_default(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError


class ComparisonAuditTests(unittest.TestCase):
    def test_direction_aware_exact_comparisons_and_gate_split(self) -> None:
        staged, table, envelope = _fixtures()
        audit = build_audit(staged, table, envelope, {})

        comparison = audit["comparison_vs_metricwise_envelope"]
        self.assertTrue(comparison["all_four_strictly_better"])
        self.assertEqual(comparison["organs_strictly_better_count"], 7)
        self.assertEqual(
            comparison["metrics"]["strict_mae"]["directional_improvement"], "0.01"
        )
        self.assertEqual(
            comparison["metrics"]["strict_mae"]["relative_improvement_percent"],
            "0.25",
        )
        self.assertEqual(
            comparison["metrics"]["full_risk_macro_delta_c"]["ehr_to_baseline_ratio"],
            "2.5",
        )
        revised = audit["criterion_assessment"]["user_revised_point_estimate_rule"]
        self.assertTrue(revised["overall_pass_under_stated_operationalization"])
        old = audit["criterion_assessment"]["old_formal_margin_and_absolute_gate_rule"]
        self.assertFalse(old["overall_pass"])
        self.assertFalse(old["metrics"]["strict_mae"]["threshold_pass"])
        self.assertEqual(
            old["metrics"]["strict_mae"]["directional_threshold_clearance"], "0"
        )
        self.assertFalse(old["metrics"]["strict_pearson"]["threshold_pass"])
        self.assertTrue(old["metrics"]["strict_r2"]["threshold_pass"])
        self.assertFalse(old["metrics"]["full_risk_macro_delta_c"]["threshold_pass"])
        self.assertTrue(
            old["metrics"]["full_risk_macro_delta_c"]["required_margin_only_pass"]
        )

    def test_rejects_cohort_mismatch(self) -> None:
        staged, table, envelope = _fixtures()
        table["methods"][2]["strict_n_participants"] = 99
        with self.assertRaisesRegex(ComparisonAuditError, "cohort mismatch"):
            build_audit(staged, table, envelope, {})

    def test_rejects_inconsistent_metric_envelope(self) -> None:
        staged, table, envelope = _fixtures()
        envelope["metric_envelope"]["strict_mae"]["source_method"] = METHODS[1]
        with self.assertRaisesRegex(ComparisonAuditError, "metric envelope is inconsistent"):
            build_audit(staged, table, envelope, {})

    def test_cli_output_is_deterministic(self) -> None:
        staged, table, envelope = _fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_path = root / "staged.json"
            table_path = root / "table.json"
            envelope_path = root / "envelope.json"
            for path, payload in (
                (staged_path, staged),
                (table_path, table),
                (envelope_path, envelope),
            ):
                path.write_text(json.dumps(payload, default=_json_default), encoding="utf-8")
            output_json = root / "out" / "audit.json"
            output_md = root / "out" / "audit.md"
            argv = [
                "--staged-primary",
                str(staged_path),
                "--baseline-table",
                str(table_path),
                "--baseline-envelope",
                str(envelope_path),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
            self.assertEqual(main(argv), 0)
            first_json = output_json.read_bytes()
            first_md = output_md.read_bytes()
            self.assertEqual(main(argv), 0)
            self.assertEqual(output_json.read_bytes(), first_json)
            self.assertEqual(output_md.read_bytes(), first_md)


if __name__ == "__main__":
    unittest.main()
