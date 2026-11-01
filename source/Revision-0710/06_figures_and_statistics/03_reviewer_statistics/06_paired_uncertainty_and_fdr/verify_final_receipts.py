#!/usr/bin/env python
"""Create once, then independently verify final paired-analysis receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_DIR = SCRIPT_DIR.parents[2]
REPO_ROOT = REVISION_DIR.parent
AGE_RECEIPT = SCRIPT_DIR / "01_age/AGE_FINAL_RECEIPT.json"
FINAL_RECEIPT = SCRIPT_DIR / "FINAL_RECEIPT.json"
METHODS = (
    "retain",
    "behrt_style",
    "medbert_style",
    "retain_organquery",
    "behrt_style_organquery",
    "medbert_style_organquery",
)


class ReceiptError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiptError(f"JSON root is not an object: {path}")
    return payload


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise ReceiptError(f"artifact does not exist: {path}")
    record: dict[str, object] = {
        "relative_path": relative(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }
    if path.suffix.lower() == ".parquet":
        record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
    return record


def write_once(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except FileExistsError as exc:
        raise ReceiptError(f"create-once receipt already exists: {path}") from exc


def create_receipts() -> dict[str, object]:
    if AGE_RECEIPT.exists() or FINAL_RECEIPT.exists():
        raise ReceiptError("final receipts already exist; use verify")
    input_manifest_path = SCRIPT_DIR / "00_input_lock/INPUT_MANIFEST.json"
    contract_path = SCRIPT_DIR / "analysis_contract.json"
    age_run_path = SCRIPT_DIR / "01_age/AGE_RUN_MANIFEST.json"
    age_audit_path = SCRIPT_DIR / "01_age/AGE_EXECUTION_AUDIT.json"
    age_audit = read_json(age_audit_path)
    age_paths = (
        SCRIPT_DIR / "01_age/strict_age_participant_sufficient_statistics.parquet",
        SCRIPT_DIR / "01_age/age_point_estimates.csv",
        SCRIPT_DIR / "01_age/age_bootstrap_replicates.parquet",
        SCRIPT_DIR / "01_age/age_paired_summary.csv",
        SCRIPT_DIR / "01_age/AGE_POINT_PARITY.json",
        age_run_path,
        age_audit_path,
    )
    age_receipt: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.age_final_receipt.v1",
        "status": "FORMAL_AGE_B1000_CREATE_ONCE_VERIFIED",
        "n_resamples": 1000,
        "seed": 7100710,
        "contract": artifact(contract_path),
        "input_manifest": artifact(input_manifest_path),
        "statistical_engine_at_age_execution": age_audit["statistical_engine_at_execution"],
        "final_statistical_engine": artifact(SCRIPT_DIR / "paired_uncertainty.py"),
        "final_execution_entrypoint": artifact(SCRIPT_DIR / "run_paired_uncertainty.py"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "outputs": [artifact(path) for path in age_paths],
        "point_parity": "PASS_AT_ABSOLUTE_TOLERANCE_1E_10",
        "immutability": "formal outputs were first writes and cannot be overwritten by the formal entrypoint",
    }
    write_once(AGE_RECEIPT, age_receipt)

    risk_input = SCRIPT_DIR / "02_risk_score_cache/00_risk_input_lock/RISK_INPUT_MANIFEST.json"
    reference = SCRIPT_DIR / "02_risk_score_cache/01_reference/REFERENCE_SCORE_MANIFEST.json"
    fdr_receipt = SCRIPT_DIR / "03_risk_bootstrap/03_all_six_fdr/FDR_COMPLETE.json"
    fdr_table = SCRIPT_DIR / "03_risk_bootstrap/03_all_six_fdr/risk_paired_summary_with_bh_fdr.csv"
    baseline_manifests = [
        SCRIPT_DIR / f"02_risk_score_cache/02_baselines/{method}/BASELINE_SCORE_MANIFEST.json"
        for method in METHODS
    ]
    formal_receipts = [
        SCRIPT_DIR / f"03_risk_bootstrap/02_formal_results/{method}/FORMAL_COMPLETE.json"
        for method in METHODS
    ]
    final_receipt: dict[str, object] = {
        "schema_version": "revision0710.reviewer_statistics.paired_uncertainty_final_receipt.v1",
        "status": "AGE_AND_ALL_SIX_RISK_FORMAL_COMPLETE",
        "age_final_receipt": artifact(AGE_RECEIPT),
        "contract": artifact(contract_path),
        "base_input_manifest": artifact(input_manifest_path),
        "risk_input_manifest": artifact(risk_input),
        "risk_reference_manifest": artifact(reference),
        "baseline_score_manifests": [artifact(path) for path in baseline_manifests],
        "formal_risk_method_receipts": [artifact(path) for path in formal_receipts],
        "fdr_receipt": artifact(fdr_receipt),
        "fdr_table": artifact(fdr_table),
        "final_statistical_engine": artifact(SCRIPT_DIR / "paired_uncertainty.py"),
        "final_execution_entrypoint": artifact(SCRIPT_DIR / "run_paired_uncertainty.py"),
        "receipt_verifier": artifact(Path(__file__)),
        "formal_settings": {
            "age_resamples": 1000,
            "risk_resamples_per_method": 1000,
            "seed": 7100710,
            "risk_macro_fdr_family_size": 6,
            "risk_organ_fdr_family_size": 42,
        },
        "risk_inference_conditioning": "test-participant bootstrap conditional on sealed/frozen calibration nuisance; nuisance-refit uncertainty is not included",
    }
    write_once(FINAL_RECEIPT, final_receipt)
    return final_receipt


def verify_artifact(record: dict[str, object]) -> None:
    path = REPO_ROOT / str(record["relative_path"])
    if not path.is_file():
        raise ReceiptError(f"receipt artifact is missing: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ReceiptError(f"receipt artifact hash mismatch: {path}")
    if int(path.stat().st_size) != int(record["bytes"]):
        raise ReceiptError(f"receipt artifact size mismatch: {path}")
    if "rows" in record and pq.ParquetFile(path).metadata.num_rows != int(record["rows"]):
        raise ReceiptError(f"receipt parquet row count mismatch: {path}")


def walk_artifacts(value: object) -> None:
    if isinstance(value, dict):
        if {"relative_path", "sha256", "bytes"}.issubset(value):
            verify_artifact(value)
        for nested in value.values():
            walk_artifacts(nested)
    elif isinstance(value, list):
        for nested in value:
            walk_artifacts(nested)


def verify_receipts() -> dict[str, object]:
    age = read_json(AGE_RECEIPT)
    final = read_json(FINAL_RECEIPT)
    if age.get("status") != "FORMAL_AGE_B1000_CREATE_ONCE_VERIFIED":
        raise ReceiptError("age final receipt status is not complete")
    if final.get("status") != "AGE_AND_ALL_SIX_RISK_FORMAL_COMPLETE":
        raise ReceiptError("overall final receipt status is not complete")
    walk_artifacts(age)
    walk_artifacts(final)
    fdr = pd.read_csv(
        SCRIPT_DIR / "03_risk_bootstrap/03_all_six_fdr/risk_paired_summary_with_bh_fdr.csv"
    )
    family_sizes = fdr.groupby("fdr_family").size().to_dict()
    expected_sizes = {"risk_macro_vs_six_baselines": 6, "risk_organ_vs_six_baselines": 42}
    if family_sizes != expected_sizes:
        raise ReceiptError(f"FDR family-size mismatch: {family_sizes}")
    return {
        "schema_version": "revision0710.reviewer_statistics.final_receipt_verification.v1",
        "status": "PASS",
        "age_final_receipt_sha256": sha256_file(AGE_RECEIPT),
        "final_receipt_sha256": sha256_file(FINAL_RECEIPT),
        "fdr_family_sizes": family_sizes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    args = parser.parse_args()
    payload = create_receipts() if args.command == "create" else verify_receipts()
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
