#!/usr/bin/env python3
"""Recalculate final-model Aalen-Johansen intervals with 1,000 bootstraps."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
REPLICATES = 1_000
SEED = 710
EXPECTED_COUNTS = {
    "participants": 456_773,
    "joined_participant_organ": 3_197_411,
    "at_risk_participant_organ": 2_914_947,
    "events": 162_125,
    "competing_deaths": 13_972,
}
EXPECTED_PREDECESSOR_MANIFEST_SHA256 = (
    "5fe9a48889a02cef8eca9790676ea6ebe1035efe1e6290d2ae2318d636204e97"
)


def workspace_root() -> Path:
    for parent in (Path.cwd(), *Path(__file__).resolve().parents):
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
REVISION = ROOT / "Revision-0710"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Source_Data"
PREDECESSOR = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/15_r3_2_final_model_survival_validation"
)
PREDECESSOR_RUNNER = PREDECESSOR / "run_survival_recalculation.py"
PREDECESSOR_MANIFEST = PREDECESSOR / "MANIFEST.json"
BURDEN_RUNNER = (
    REVISION
    / "Final_Fig_0716/Reviewer_Statistics/12_e47_disease_burden_adjustment"
    / "run_e47_disease_burden_adjustment.py"
)
PRIMARY_RUNNER = (
    REVISION
    / "06_figures_and_statistics/03_reviewer_statistics"
    / "04_cox_diagnostics_and_competing_risk/run_primary_survival.py"
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def write_table(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return {
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }


def assert_numeric_parity(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    keys: list[str],
    columns: list[str],
    label: str,
    tolerance: float = 1e-14,
) -> float:
    merged = current[keys + columns].merge(
        previous[keys + columns],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_current", "_previous"),
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"{label}: key parity failed")
    maximum = 0.0
    for column in columns:
        difference = np.abs(
            merged[f"{column}_current"].to_numpy(float)
            - merged[f"{column}_previous"].to_numpy(float)
        )
        maximum = max(maximum, float(difference.max(initial=0.0)))
    if maximum > tolerance:
        raise RuntimeError(f"{label}: point-estimate parity failed: max abs diff={maximum}")
    return maximum


def result_summary(difference: pd.DataFrame) -> str:
    at_24 = difference.loc[difference["requested_horizon_months"].eq(24.0)]
    at_60 = difference.loc[difference["requested_horizon_months"].eq(60.0)]
    return f"""# R3.2 competing-risk bootstrap successor: result summary

The final-model Aalen-Johansen analysis was repeated with 1,000 stratified
bootstrap replicates per organ. All point estimates are identical to the sealed
500-replicate predecessor; only confidence limits were re-estimated.

- 24-month Q4-minus-Q1 target-CIF differences: {100*at_24['aj_target_cif_risk_difference_q4_minus_q1'].min():.3f} to {100*at_24['aj_target_cif_risk_difference_q4_minus_q1'].max():.3f} percentage points.
- 60-month Q4-minus-Q1 target-CIF differences: {100*at_60['aj_target_cif_risk_difference_q4_minus_q1'].min():.3f} to {100*at_60['aj_target_cif_risk_difference_q4_minus_q1'].max():.3f} percentage points.
- All 14 percentile confidence intervals have lower limits above zero.

This successor replaces the 500-replicate intervals in the reviewer response.
The predecessor remains sealed for audit history and must not be presented as
the current formal uncertainty analysis.
"""


def verify_existing() -> dict[str, object]:
    manifest_path = HERE / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for record in [*payload["inputs"], *payload["lineage"], *payload["outputs"]]:
        path = Path(record["path"])
        if not path.is_file():
            failures.append(f"missing:{path}")
        elif sha256_file(path) != record["sha256"]:
            failures.append(f"sha256:{path}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "manifest_sha256": sha256_file(manifest_path),
        "checked_files": len(payload["inputs"]) + len(payload["lineage"]) + len(payload["outputs"]),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> int:
    if "--verify-only" in sys.argv:
        print(json.dumps(verify_existing(), indent=2), flush=True)
        return 0
    manifest_path = HERE / "MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError("successor bundle already sealed; use --verify-only")
    if sha256_file(PREDECESSOR_MANIFEST) != EXPECTED_PREDECESSOR_MANIFEST_SHA256:
        raise RuntimeError("predecessor manifest SHA-256 changed")
    started = time.time()
    predecessor = load_module("r3_2_predecessor", PREDECESSOR_RUNNER)
    burden = load_module("r3_2_burden", BURDEN_RUNNER)
    primary = load_module("r3_2_primary", PRIMARY_RUNNER)
    inputs = predecessor.validate_hashes()
    context, context_audit = burden.load_context(predecessor.CONTEXT, predecessor.V2_LABELS)

    audits: list[dict[str, object]] = []
    references: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    risks: list[pd.DataFrame] = []
    differences: list[pd.DataFrame] = []
    risk_tables: list[pd.DataFrame] = []
    logranks: list[pd.DataFrame] = []
    for organ_index, organ in enumerate(ORGANS):
        joined, audit = predecessor.load_joined_organ(burden, context, organ)
        risk, reference = predecessor.crossfit_exposure_transform(joined)
        reference.insert(0, "organ_system", organ)
        curve, absolute, difference, risk_table, logrank = primary._run_nonparametric(
            risk,
            organ,
            bootstrap_replicates=REPLICATES,
            seed=SEED + 1009 * organ_index,
        )
        audits.append(audit)
        references.append(reference)
        curves.append(curve)
        risks.append(absolute)
        differences.append(difference)
        risk_tables.append(risk_table)
        logranks.append(logrank)

    audit = pd.DataFrame(audits)
    observed_counts = {
        "participants": int(audit["unique_participants"].min()),
        "joined_participant_organ": int(audit["joined_rows"].sum()),
        "at_risk_participant_organ": int(audit["at_risk_n"].sum()),
        "events": int(audit["target_events"].sum()),
        "competing_deaths": int(audit["competing_deaths"].sum()),
    }
    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"row-count gate failed: {observed_counts}")

    reference = pd.concat(references, ignore_index=True)
    curve = pd.concat(curves, ignore_index=True)
    absolute = pd.concat(risks, ignore_index=True)
    difference = pd.concat(differences, ignore_index=True)
    risk_table = pd.concat(risk_tables, ignore_index=True)
    logrank = pd.concat(logranks, ignore_index=True)

    previous_absolute = pd.read_csv(PREDECESSOR / "Source_Data/absolute_risk_24_60m.csv")
    previous_difference = pd.read_csv(PREDECESSOR / "Source_Data/q4_minus_q1_risk_difference.csv")
    previous_reference = pd.read_csv(PREDECESSOR / "Source_Data/crossfit_gap_reference.csv")
    parity = {
        "absolute_risk_point_max_abs_difference": assert_numeric_parity(
            absolute,
            previous_absolute,
            ["organ_system", "gap_quartile_crossfit", "requested_horizon_months"],
            [
                "evaluated_horizon_months",
                "km_target_risk",
                "aj_target_cif",
                "aj_death_cif",
            ],
            "absolute risk",
        ),
        "q4_q1_point_max_abs_difference": assert_numeric_parity(
            difference,
            previous_difference,
            ["organ_system", "requested_horizon_months"],
            [
                "aj_target_cif_q1",
                "aj_target_cif_q4",
                "aj_target_cif_risk_difference_q4_minus_q1",
            ],
            "Q4-Q1 difference",
        ),
        "crossfit_reference_max_abs_difference": assert_numeric_parity(
            reference,
            previous_reference,
            ["organ_system", "outer_fold"],
            ["reference_n", "reference_mean", "reference_sd_ddof0", "q25", "q50", "q75"],
            "crossfit reference",
        ),
    }
    assertions = {
        "row_count_gate": observed_counts == EXPECTED_COUNTS,
        "replicates_are_1000": bool(difference["bootstrap_replicates"].eq(REPLICATES).all()),
        "all_14_ci_lower_limits_above_zero": bool(difference["bootstrap_ci95_low"].gt(0).all()),
        "point_estimates_identical_to_predecessor": all(value <= 1e-14 for value in parity.values()),
        "fine_gray_not_performed": True,
    }
    if not all(assertions.values()):
        raise RuntimeError(f"QA failed: {assertions}")

    SOURCE.mkdir(parents=True, exist_ok=True)
    output_records = [
        write_table(audit, SOURCE / "input_join_audit.csv"),
        write_table(reference, SOURCE / "crossfit_gap_reference.csv"),
        write_table(curve, SOURCE / "km_aj_curves.parquet"),
        write_table(absolute, SOURCE / "absolute_risk_24_60m.csv"),
        write_table(difference, SOURCE / "q4_minus_q1_risk_difference.csv"),
        write_table(risk_table, SOURCE / "quartile_numbers_at_risk.csv"),
        write_table(logrank, SOURCE / "logrank_tests.csv"),
    ]
    summary_path = HERE / "RESULT_SUMMARY.md"
    summary_path.write_text(result_summary(difference), encoding="utf-8")
    qa_path = HERE / "QA_REPORT.json"
    json_write(
        qa_path,
        {
            "schema_version": "r3_2.competing_risk_bootstrap_1000.qa.v1",
            "status": "PASS",
            "created_utc": utc_now(),
            "assertions": assertions,
            "point_parity": parity,
        },
    )
    output_records.extend(
        [
            {
                "path": summary_path.resolve().as_posix(),
                "sha256": sha256_file(summary_path),
                "bytes": int(summary_path.stat().st_size),
            },
            {
                "path": qa_path.resolve().as_posix(),
                "sha256": sha256_file(qa_path),
                "bytes": int(qa_path.stat().st_size),
            },
        ]
    )
    lineage = []
    for role, path in (
        ("analysis_plan", HERE / "ANALYSIS_PLAN.md"),
        ("successor_runner", Path(__file__).resolve()),
        ("predecessor_runner", PREDECESSOR_RUNNER),
        ("predecessor_manifest", PREDECESSOR_MANIFEST),
        ("aalen_johansen_support_runner", PRIMARY_RUNNER),
    ):
        lineage.append(
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        )
    manifest = {
        "schema_version": "r3_2.competing_risk_bootstrap_1000.bundle.v1",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "analysis_role": "formal_successor_to_500_replicate_competing_risk_intervals",
        "bootstrap": {
            "replicates": REPLICATES,
            "seed": SEED,
            "strata": "outer fold x cross-fitted gap quartile",
        },
        "row_counts": observed_counts,
        "context_audit": context_audit,
        "inputs": inputs,
        "lineage": lineage,
        "outputs": output_records,
        "qa": assertions,
        "elapsed_seconds": float(time.time() - started),
    }
    json_write(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "manifest_sha256": sha256_file(manifest_path),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "qa": assertions,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
