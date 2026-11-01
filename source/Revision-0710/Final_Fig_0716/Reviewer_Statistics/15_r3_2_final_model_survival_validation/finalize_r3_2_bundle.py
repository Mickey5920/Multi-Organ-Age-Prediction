#!/usr/bin/env python3
"""Validate, summarize, and seal the independent R3.2 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def workspace_root() -> Path:
    for parent in (Path.cwd(), *Path(__file__).resolve().parents):
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
REVISION = ROOT / "Revision-0710"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Source_Data"
SURVIVAL_RECEIPT = HERE / "SURVIVAL_RUN_RECEIPT.json"
NULL_RECEIPT = HERE / "NULL_REFERENCE_RECEIPT.json"
CLINICAL = REVISION / "Final_Fig_0716/Reviewer_Statistics/13_final_model_clinical_utility"
CLINICAL_MACRO = CLINICAL / "01_point/macro_model_metrics.csv"
CLINICAL_CI = CLINICAL / "04_bootstrap_1000/macro_paired_bootstrap_ci.csv"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_record(path: Path, role: str) -> dict[str, object]:
    record: dict[str, object] = {
        "role": role,
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }
    if path.suffix == ".csv":
        table = pd.read_csv(path)
        record["rows"] = int(len(table))
        record["columns"] = list(table.columns)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        record["rows"] = int(parquet.metadata.num_rows)
        record["columns"] = parquet.schema.names
    return record


def verify_receipt(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"receipt is not COMPLETE: {path}")
    records = [*payload.get("inputs", []), *payload.get("lineage", []), *payload.get("outputs", [])]
    runner = payload.get("runner")
    if isinstance(runner, dict) and runner.get("path"):
        records.append(runner)
    failures = []
    for record in records:
        target = Path(record["path"])
        if not target.is_file():
            failures.append(f"missing:{target}")
        elif record.get("sha256") != sha256_file(target):
            failures.append(f"sha256:{target}")
    if failures:
        raise RuntimeError(json.dumps({"receipt": str(path), "failures": failures}, indent=2))
    return payload


def one(frame: pd.DataFrame, augmentation: str) -> pd.Series:
    selected = frame.loc[
        frame["tier"].eq("C_context_density_primary")
        & frame["augmentation"].eq(augmentation)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"macro clinical row missing: {augmentation}")
    return selected.iloc[0]


def ci_row(frame: pd.DataFrame, augmentation: str, metric: str) -> pd.Series:
    selected = frame.loc[frame["augmentation"].eq(augmentation) & frame["metric"].eq(metric)]
    if len(selected) != 1:
        raise RuntimeError(f"bootstrap CI row missing: {augmentation} {metric}")
    return selected.iloc[0]


def build_key_numbers() -> dict[str, object]:
    full = pd.read_csv(SOURCE / "full_adjusted_cox_gap.csv")
    ph = pd.read_csv(SOURCE / "target_gap_ph_diagnostics.csv")
    windows = pd.read_csv(SOURCE / "full_adjusted_time_window_cox.csv")
    difference = pd.read_csv(SOURCE / "q4_minus_q1_risk_difference.csv")
    absolute = pd.read_csv(SOURCE / "absolute_risk_24_60m.csv")
    null = pd.read_csv(SOURCE / "null_reference_macro_comparison.csv").iloc[0]
    macro = pd.read_csv(CLINICAL_MACRO)
    ci = pd.read_csv(CLINICAL_CI)
    base = one(macro, "base")
    all_gaps = one(macro, "all_gaps")
    c_ci = ci_row(ci, "all_gaps", "delta_harrell_c")
    brier_ci = ci_row(ci, "all_gaps", "brier_improvement_60m")

    early = windows.loc[windows["time_window"].eq("0_to_24_months")]
    late = windows.loc[windows["time_window"].eq("24_to_60_months_landmark")]
    rd24 = difference.loc[difference["requested_horizon_months"].eq(24.0)]
    rd60 = difference.loc[difference["requested_horizon_months"].eq(60.0)]
    significant_ph = ph.loc[ph["gap_q_value_bh_14_tests"].lt(0.05)]
    return {
        "scope": "R3.2 final-model Cox diagnostics, competing-risk absolute risk, and held-out null comparison",
        "row_counts": {
            "participants": 456_773,
            "joined_participant_organ": 3_197_411,
            "at_risk_participant_organ": int(full["n"].sum()),
            "target_events": int(full["events"].sum()),
            "competing_deaths": int(full["competing_deaths"].sum()),
        },
        "full_adjusted_cause_specific_cox": {
            "hr_min": float(full["gap_hr_per_crossfit_sd"].min()),
            "hr_max": float(full["gap_hr_per_crossfit_sd"].max()),
            "all_seven_bh_q_lt_0_05": bool(full["gap_q_value_bh_7_organs"].lt(0.05).all()),
            "all_converged": bool(full["converged"].all()),
            "max_absolute_beta_difference_from_prior_adjustment_table": float(
                full["reference_beta_absolute_difference"].max()
            ),
        },
        "proportional_hazards": {
            "target_gap_tests": int(len(ph)),
            "time_transforms": sorted(ph["time_transform"].unique().tolist()),
            "bh_significant_tests": int(len(significant_ph)),
            "organs_with_bh_significant_target_gap_test": sorted(
                significant_ph["organ_system"].unique().tolist()
            ),
            "organs_without_bh_significant_target_gap_test": sorted(
                set(ph["organ_system"]).difference(significant_ph["organ_system"])
            ),
            "interpretation": "pooled 60-month HR is an average association when a target-gap PH diagnostic is significant",
        },
        "time_windows": {
            "early_0_24": {
                "hr_min": float(early["gap_hr_per_crossfit_sd"].min()),
                "hr_max": float(early["gap_hr_per_crossfit_sd"].max()),
                "all_bh_q_lt_0_05": bool(
                    early["gap_q_value_bh_7_organs_within_window"].lt(0.05).all()
                ),
            },
            "late_24_60": {
                "hr_min": float(late["gap_hr_per_crossfit_sd"].min()),
                "hr_max": float(late["gap_hr_per_crossfit_sd"].max()),
                "all_bh_q_lt_0_05": bool(
                    late["gap_q_value_bh_7_organs_within_window"].lt(0.05).all()
                ),
            },
            "direction_consistent_all_14": bool(windows["gap_hr_per_crossfit_sd"].gt(1.0).all()),
        },
        "competing_risk": {
            "estimator": "Aalen-Johansen",
            "fine_gray_performed": False,
            "bootstrap_replicates": 500,
            "q4_minus_q1_target_cif_difference_24m_min": float(
                rd24["aj_target_cif_risk_difference_q4_minus_q1"].min()
            ),
            "q4_minus_q1_target_cif_difference_24m_max": float(
                rd24["aj_target_cif_risk_difference_q4_minus_q1"].max()
            ),
            "q4_minus_q1_target_cif_difference_60m_min": float(
                rd60["aj_target_cif_risk_difference_q4_minus_q1"].min()
            ),
            "q4_minus_q1_target_cif_difference_60m_max": float(
                rd60["aj_target_cif_risk_difference_q4_minus_q1"].max()
            ),
            "all_14_bootstrap_ci_low_above_zero": bool(difference["bootstrap_ci95_low"].gt(0).all()),
            "death_cif_60m_min_across_organ_quartiles": float(
                absolute.loc[absolute["requested_horizon_months"].eq(60.0), "aj_death_cif"].min()
            ),
            "death_cif_60m_max_across_organ_quartiles": float(
                absolute.loc[absolute["requested_horizon_months"].eq(60.0), "aj_death_cif"].max()
            ),
        },
        "heldout_prediction": {
            "null": {
                "definition": "calibration-fold intercept-only Kaplan-Meier risk applied to outer test",
                "harrell_c": float(null["null_harrell_c"]),
                "uno_c_60m": float(null["null_uno_c_60m"]),
                "auc_60m": float(null["null_auc_60m"]),
                "brier_60m": float(null["null_brier_60m"]),
            },
            "clinical_density_base": {
                "harrell_c": float(base["harrell_c"]),
                "uno_c_60m": float(base["uno_c_60m"]),
                "auc_60m": float(base["auc_60m"]),
                "brier_60m": float(base["brier_60m"]),
                "observed_expected_ratio_60m": float(base["cal_60m_observed_expected_ratio"]),
                "calibration_slope_60m": float(base["cal_60m_ipcw_logistic_calibration_slope"]),
                "ici_60m": float(base["cal_60m_ici"]),
            },
            "clinical_density_plus_all_seven_gaps": {
                "harrell_c": float(all_gaps["harrell_c"]),
                "uno_c_60m": float(all_gaps["uno_c_60m"]),
                "auc_60m": float(all_gaps["auc_60m"]),
                "brier_60m": float(all_gaps["brier_60m"]),
                "observed_expected_ratio_60m": float(
                    all_gaps["cal_60m_observed_expected_ratio"]
                ),
                "calibration_slope_60m": float(
                    all_gaps["cal_60m_ipcw_logistic_calibration_slope"]
                ),
                "ici_60m": float(all_gaps["cal_60m_ici"]),
            },
            "all_gaps_vs_clinical_density_base": {
                "delta_harrell_c": float(c_ci["point_estimate"]),
                "delta_harrell_c_ci95_low": float(c_ci["ci95_low"]),
                "delta_harrell_c_ci95_high": float(c_ci["ci95_high"]),
                "brier_improvement_60m": float(brier_ci["point_estimate"]),
                "brier_improvement_60m_ci95_low": float(brier_ci["ci95_low"]),
                "brier_improvement_60m_ci95_high": float(brier_ci["ci95_high"]),
                "bootstrap_replicates": int(c_ci["bootstrap_replicates"]),
                "ici_improved": bool(all_gaps["cal_60m_ici"] < base["cal_60m_ici"]),
            },
        },
        "claim_boundary": {
            "supported": "positive prognostic associations in both time windows, competing-risk-adjusted absolute-risk separation, and held-out incremental prediction beyond a strong clinical baseline",
            "not_supported": "universal proportional hazards, Fine-Gray regression, causal effects, clinical thresholds, or uniformly improved calibration",
        },
    }


def render_summary(key: dict[str, object]) -> str:
    ph = key["proportional_hazards"]
    windows = key["time_windows"]
    cr = key["competing_risk"]
    prediction = key["heldout_prediction"]
    null = prediction["null"]
    base = prediction["clinical_density_base"]
    gaps = prediction["clinical_density_plus_all_seven_gaps"]
    delta = prediction["all_gaps_vs_clinical_density_base"]
    organs = ", ".join(ph["organs_with_bh_significant_target_gap_test"])
    return f"""# R3.2 final-model survival validation: result summary

## Proportional hazards and time windows

All seven fully adjusted cause-specific Cox models converged. Of the 14
prespecified target-gap Schoenfeld tests, {ph['bh_significant_tests']} remained
significant after BH correction, involving {organs}. The pooled 60-month hazard
ratios should therefore be interpreted as average follow-up associations for
these organs, rather than constant effects.

The direction was nevertheless consistent in both prespecified windows. The
0-24 month HRs ranged from {windows['early_0_24']['hr_min']:.6f} to
{windows['early_0_24']['hr_max']:.6f}; the 24-60 month landmark HRs ranged from
{windows['late_24_60']['hr_min']:.6f} to {windows['late_24_60']['hr_max']:.6f}.
All 14 window-specific associations had BH q < 0.05.

## Competing risk

The final risk sets contained {key['row_counts']['competing_deaths']:,}
competing deaths. Aalen-Johansen analyses retained death as a competing event.
Q4-minus-Q1 target cumulative-incidence differences ranged from
{100*cr['q4_minus_q1_target_cif_difference_24m_min']:.3f} to
{100*cr['q4_minus_q1_target_cif_difference_24m_max']:.3f} percentage points at
24 months and from {100*cr['q4_minus_q1_target_cif_difference_60m_min']:.3f} to
{100*cr['q4_minus_q1_target_cif_difference_60m_max']:.3f} percentage points at
60 months; all 14 bootstrap confidence intervals excluded zero. No Fine-Gray
regression was performed or claimed.

## Held-out model fit, calibration, and null comparison

At 60 months, the intercept-only reference had Harrell C =
{null['harrell_c']:.6f} and Brier = {null['brier_60m']:.6f}. The clinical and
healthcare-density baseline had Harrell C = {base['harrell_c']:.6f} and Brier =
{base['brier_60m']:.6f}. Adding all seven gaps yielded Harrell C =
{gaps['harrell_c']:.6f} and Brier = {gaps['brier_60m']:.6f}; the paired
increment was Delta C = {delta['delta_harrell_c']:.6f} (95% CI
{delta['delta_harrell_c_ci95_low']:.6f}-{delta['delta_harrell_c_ci95_high']:.6f})
and Brier improvement = {delta['brier_improvement_60m']:.6f} (95% CI
{delta['brier_improvement_60m_ci95_low']:.6f}-{delta['brier_improvement_60m_ci95_high']:.6f}).

Calibration remained close overall (O/E {base['observed_expected_ratio_60m']:.6f}
to {gaps['observed_expected_ratio_60m']:.6f}; slope
{base['calibration_slope_60m']:.6f} to {gaps['calibration_slope_60m']:.6f}), but
ICI changed from {base['ici_60m']:.6f} to {gaps['ici_60m']:.6f} and therefore did
not improve. This supports improved discrimination and prediction error beyond
a strong baseline, while not claiming uniform calibration improvement.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def verify_manifest() -> dict[str, object]:
    path = HERE / "MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for record in [*payload["inputs"], *payload["lineage"], *payload["outputs"]]:
        target = Path(record["path"])
        if not target.is_file():
            failures.append(f"missing:{target}")
        elif sha256_file(target) != record["sha256"]:
            failures.append(f"sha256:{target}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "manifest_sha256": sha256_file(path),
        "checked_files": int(len(payload["inputs"]) + len(payload["lineage"]) + len(payload["outputs"])),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> int:
    args = parse_args()
    if args.verify_only:
        print(json.dumps(verify_manifest(), indent=2), flush=True)
        return 0
    manifest_path = HERE / "MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError("bundle already sealed; use --verify-only")
    survival = verify_receipt(SURVIVAL_RECEIPT)
    null = verify_receipt(NULL_RECEIPT)
    key = build_key_numbers()

    assertions = {
        "survival_receipt_complete": survival["status"] == "COMPLETE",
        "null_receipt_complete": null["status"] == "COMPLETE",
        "row_count_gate": key["row_counts"]
        == {
            "participants": 456_773,
            "joined_participant_organ": 3_197_411,
            "at_risk_participant_organ": 2_914_947,
            "target_events": 162_125,
            "competing_deaths": 13_972,
        },
        "all_full_cox_converged": key["full_adjusted_cause_specific_cox"]["all_converged"],
        "full_cox_parity_tolerance": key["full_adjusted_cause_specific_cox"][
            "max_absolute_beta_difference_from_prior_adjustment_table"
        ]
        < 1e-6,
        "ph_family_is_14": key["proportional_hazards"]["target_gap_tests"] == 14,
        "both_window_directions_positive": key["time_windows"]["direction_consistent_all_14"],
        "all_window_q_lt_0_05": key["time_windows"]["early_0_24"]["all_bh_q_lt_0_05"]
        and key["time_windows"]["late_24_60"]["all_bh_q_lt_0_05"],
        "all_aj_difference_ci_above_zero": key["competing_risk"][
            "all_14_bootstrap_ci_low_above_zero"
        ],
        "fine_gray_not_claimed": not key["competing_risk"]["fine_gray_performed"],
        "null_discrimination_exact_half": all(
            np.isclose(key["heldout_prediction"]["null"][metric], 0.5, atol=1e-12)
            for metric in ("harrell_c", "uno_c_60m", "auc_60m")
        ),
        "base_brier_better_than_null": key["heldout_prediction"]["clinical_density_base"][
            "brier_60m"
        ]
        < key["heldout_prediction"]["null"]["brier_60m"],
        "all_gaps_brier_better_than_base": key["heldout_prediction"][
            "clinical_density_plus_all_seven_gaps"
        ]["brier_60m"]
        < key["heldout_prediction"]["clinical_density_base"]["brier_60m"],
        "ici_nonimprovement_preserved": not key["heldout_prediction"][
            "all_gaps_vs_clinical_density_base"
        ]["ici_improved"],
    }
    if not all(assertions.values()):
        raise RuntimeError(f"QA assertions failed: {assertions}")

    key_path = SOURCE / "reviewer_key_numbers.json"
    summary_path = HERE / "RESULT_SUMMARY.md"
    qa_path = HERE / "QA_REPORT.json"
    json_write(key_path, key)
    summary_path.write_text(render_summary(key), encoding="utf-8")
    qa = {
        "schema_version": "r3_2.final_model_survival_validation.qa.v1",
        "status": "PASS",
        "created_utc": utc_now(),
        "assertions": assertions,
        "source_table_count": len(list(SOURCE.glob("*"))),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    json_write(qa_path, qa)

    source_files = sorted(path for path in SOURCE.iterdir() if path.is_file())
    inputs = [
        file_record(SURVIVAL_RECEIPT, "sealed_survival_recalculation_receipt"),
        file_record(NULL_RECEIPT, "sealed_null_reference_receipt"),
        file_record(CLINICAL_MACRO, "final_model_heldout_macro_metrics"),
        file_record(CLINICAL_CI, "final_model_heldout_paired_bootstrap_ci"),
    ]
    lineage = [
        file_record(HERE / "ANALYSIS_PLAN.md", "analysis_plan"),
        file_record(HERE / "run_survival_recalculation.py", "survival_runner"),
        file_record(HERE / "run_null_reference.py", "null_reference_runner"),
        file_record(Path(__file__).resolve(), "bundle_finalizer"),
    ]
    outputs = [file_record(path, f"source_data:{path.name}") for path in source_files]
    outputs.extend(
        [
            file_record(summary_path, "result_summary"),
            file_record(qa_path, "qa_report"),
        ]
    )
    manifest = {
        "schema_version": "r3_2.final_model_survival_validation.bundle.v1",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "analysis_role": "reviewer_R3_2_final_model_evidence_closure",
        "inputs": inputs,
        "lineage": lineage,
        "outputs": outputs,
        "key_claims": key["claim_boundary"],
    }
    json_write(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "manifest": manifest_path.resolve().as_posix(),
                "manifest_sha256": sha256_file(manifest_path),
                "assertions": assertions,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
