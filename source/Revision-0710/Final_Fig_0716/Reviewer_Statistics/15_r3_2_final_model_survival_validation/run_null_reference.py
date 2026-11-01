#!/usr/bin/env python3
"""Add an explicit intercept-only held-out survival reference for R3.2."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import (
    brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)
EXPECTED_SHA256 = {
    "cache_manifest": "5912c170c11e84e45aa666b1f41d9489c30ded2897c4325c79f832e0f536da99",
    "cell_metrics": "87ea5a1930d7b0b0158d5fa46161ea6d975c5eb1a4e9d252d743f4a57f83d62f",
    "macro_metrics": "b3553ad73b09bae3c4d1fbfca6f246efd9104665d4dd173d6121a22c1f148656",
    "bootstrap_ci": "5202e5db070e02d596a260289a0f2ea712f5ee057de02b0efed3594b30f187e3",
    "clinical_qa": "f93c569661eb1a51ae01d67c2e07958e2545e5cc3ef192d808487a89de62bcf7",
}


def workspace_root() -> Path:
    for parent in (Path.cwd(), *Path(__file__).resolve().parents):
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
REVISION = ROOT / "Revision-0710"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Source_Data"
CLINICAL = REVISION / "Final_Fig_0716/Reviewer_Statistics/13_final_model_clinical_utility"
CACHE = CLINICAL / "00_input_cache"
PATHS = {
    "cache_manifest": CACHE / "INPUT_CACHE_MANIFEST.json",
    "cell_metrics": CLINICAL / "01_point/cell_model_metrics.csv",
    "macro_metrics": CLINICAL / "01_point/macro_model_metrics.csv",
    "bootstrap_ci": CLINICAL / "04_bootstrap_1000/macro_paired_bootstrap_ci.csv",
    "clinical_qa": CLINICAL / "QA_REPORT.json",
}


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


def input_records() -> list[dict[str, object]]:
    records = []
    for role, path in PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != EXPECTED_SHA256[role]:
            raise RuntimeError(f"{role} SHA-256 mismatch: {observed}")
        records.append(
            {
                "role": role,
                "path": path.resolve().as_posix(),
                "sha256": observed,
                "bytes": int(path.stat().st_size),
            }
        )
    qa = json.loads(PATHS["clinical_qa"].read_text(encoding="utf-8"))
    if qa.get("status") != "PASS":
        raise RuntimeError("final-model clinical-utility QA is not PASS")
    return records


def validate_cache_manifest() -> list[dict[str, object]]:
    manifest = json.loads(PATHS["cache_manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or len(manifest.get("rows", [])) != 10:
        raise RuntimeError("clinical-utility input-cache manifest contract failed")
    records = []
    for row in manifest["rows"]:
        cache_path = Path(row["cache_path"])
        if not cache_path.is_file() or sha256_file(cache_path) != row["cache_sha256"]:
            raise RuntimeError(f"cache hash mismatch: {cache_path}")
        records.append(
            {
                "outer_fold": int(row["outer_fold"]),
                "role": str(row["role"]),
                "path": cache_path.resolve().as_posix(),
                "sha256": row["cache_sha256"],
                "rows": int(row["rows"]),
            }
        )
    return records


def load_cell(outer_fold: int, role: str, organ: str) -> pd.DataFrame:
    path = CACHE / f"outer_fold_{outer_fold}__{role}.pkl"
    frame = pd.read_pickle(path)
    frame = frame.loc[frame["organ_system"].astype(str).eq(organ) & frame["at_risk"].eq(1)].copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["duration"] = pd.to_numeric(frame["duration"], errors="raise").astype(np.float64)
    frame["event"] = pd.to_numeric(frame["event"], errors="raise").astype(np.int8)
    if frame["participant_id"].duplicated().any():
        raise RuntimeError(f"fold {outer_fold} {role} {organ}: duplicate participant")
    if frame["duration"].isna().any() or frame["duration"].le(0).any():
        raise RuntimeError(f"fold {outer_fold} {role} {organ}: invalid duration")
    return frame


def survival_array(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(
        event=frame["event"].to_numpy(bool),
        time=frame["duration"].to_numpy(np.float64),
    )


def km_survival_at(frame: pd.DataFrame, horizon: float) -> float:
    times, survival = kaplan_meier_estimator(
        frame["event"].to_numpy(bool),
        frame["duration"].to_numpy(np.float64),
    )
    index = np.searchsorted(times, horizon, side="right") - 1
    return float(survival[index]) if index >= 0 else 1.0


def cell_metrics(outer_fold: int, organ: str) -> dict[str, object]:
    fit = load_cell(outer_fold, "calibration", organ)
    score = load_cell(outer_fold, "outer_test", organ)
    overlap = set(fit["participant_id"]).intersection(score["participant_id"])
    if overlap:
        raise RuntimeError(f"fold {outer_fold} {organ}: calibration/test overlap")
    support_upper = float(min(fit["duration"].max(), score["duration"].max()))
    if 60.0 - support_upper > 0.1:
        raise RuntimeError(f"fold {outer_fold} {organ}: 60-month support unavailable")
    reclassified: dict[str, int] = {}
    # Match the established clinical-utility right-open support contract:
    # events recorded exactly at the common administrative boundary are
    # reclassified as administratively censored before evaluating at t-.
    for role, frame in (("fit", fit), ("test", score)):
        at_boundary = np.isclose(
            frame["duration"].to_numpy(np.float64),
            support_upper,
            rtol=0.0,
            atol=1e-6,
        )
        boundary_event = at_boundary & frame["event"].eq(1).to_numpy()
        reclassified[role] = int(boundary_event.sum())
        frame.loc[boundary_event, "event"] = 0
    horizon = float(np.nextafter(support_upper, 0.0))
    null_survival = km_survival_at(fit, horizon)
    null_risk = 1.0 - null_survival
    estimate = np.full((len(score), 1), null_survival, dtype=np.float64)
    fit_y = survival_array(fit)
    score_y = survival_array(score)
    # An intercept-only model assigns exactly the same score to every
    # participant. Harrell C, Uno C, and cumulative/dynamic AUC are therefore
    # exactly 0.5 under the standard half-credit convention for tied scores.
    # Deriving these constants avoids an unnecessary O(n^2) pair enumeration.
    harrell = uno = auc = 0.5
    brier = float(brier_score(fit_y, score_y, estimate, np.asarray([horizon]))[1][0])
    observed_risk = 1.0 - km_survival_at(score, horizon)
    return {
        "outer_fold": int(outer_fold),
        "organ_system": organ,
        "fit_n": int(len(fit)),
        "fit_events": int(fit["event"].sum()),
        "fit_right_open_boundary_events_reclassified": reclassified["fit"],
        "test_n": int(len(score)),
        "test_events": int(score["event"].sum()),
        "test_right_open_boundary_events_reclassified": reclassified["test"],
        "effective_horizon_months": horizon,
        "null_definition": "intercept-only calibration-fold Kaplan-Meier event risk",
        "null_discrimination_derivation": "constant score; 0.5 by half-credit tie convention",
        "null_predicted_risk_60m": float(null_risk),
        "test_km_observed_risk_60m": float(observed_risk),
        "null_observed_expected_ratio_60m": float(observed_risk / null_risk),
        "null_one_bin_absolute_calibration_error_60m": float(abs(observed_risk - null_risk)),
        "null_harrell_c": harrell,
        "null_uno_c_60m": uno,
        "null_auc_60m": auc,
        "null_brier_60m": brier,
    }


def main() -> int:
    receipt_path = HERE / "NULL_REFERENCE_RECEIPT.json"
    if receipt_path.exists():
        raise FileExistsError("null reference already sealed")
    started = time.time()
    inputs = input_records()
    cache_records = validate_cache_manifest()
    rows = [cell_metrics(fold, organ) for fold in range(5) for organ in ORGANS]
    null = pd.DataFrame(rows)
    if len(null) != 35:
        raise RuntimeError("expected 35 fold-organ null-reference cells")
    for metric in ("null_harrell_c", "null_uno_c_60m", "null_auc_60m"):
        if not np.allclose(null[metric], 0.5, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"constant-score discrimination parity failed: {metric}")

    base = pd.read_csv(PATHS["cell_metrics"])
    base = base.loc[
        base["tier"].eq("C_context_density_primary") & base["augmentation"].eq("base"),
        [
            "outer_fold",
            "organ_system",
            "harrell_c",
            "uno_c_60m",
            "auc_60m",
            "brier_60m",
            "cal_60m_observed_expected_ratio",
            "cal_60m_ipcw_logistic_calibration_slope",
            "cal_60m_ici",
        ],
    ].copy()
    if len(base) != 35:
        raise RuntimeError("expected 35 final clinical-baseline cells")
    comparison = null.merge(base, on=["outer_fold", "organ_system"], validate="one_to_one")
    comparison["base_minus_null_harrell_c"] = comparison["harrell_c"] - comparison["null_harrell_c"]
    comparison["base_minus_null_uno_c_60m"] = comparison["uno_c_60m"] - comparison["null_uno_c_60m"]
    comparison["base_minus_null_auc_60m"] = comparison["auc_60m"] - comparison["null_auc_60m"]
    comparison["null_minus_base_brier_60m"] = comparison["null_brier_60m"] - comparison["brier_60m"]

    numeric = [
        "null_predicted_risk_60m",
        "test_km_observed_risk_60m",
        "null_observed_expected_ratio_60m",
        "null_one_bin_absolute_calibration_error_60m",
        "null_harrell_c",
        "null_uno_c_60m",
        "null_auc_60m",
        "null_brier_60m",
        "harrell_c",
        "uno_c_60m",
        "auc_60m",
        "brier_60m",
        "cal_60m_observed_expected_ratio",
        "cal_60m_ipcw_logistic_calibration_slope",
        "cal_60m_ici",
        "base_minus_null_harrell_c",
        "base_minus_null_uno_c_60m",
        "base_minus_null_auc_60m",
        "null_minus_base_brier_60m",
    ]
    macro = {"aggregation": "unweighted mean across 5 outer folds x 7 organs", "cells": 35}
    macro.update({column: float(comparison[column].mean()) for column in numeric})
    macro_frame = pd.DataFrame([macro])

    SOURCE.mkdir(parents=True, exist_ok=True)
    cell_path = SOURCE / "null_reference_cell_metrics.csv"
    macro_path = SOURCE / "null_reference_macro_comparison.csv"
    comparison.to_csv(cell_path, index=False)
    macro_frame.to_csv(macro_path, index=False)
    outputs = [
        {
            "path": cell_path.resolve().as_posix(),
            "sha256": sha256_file(cell_path),
            "bytes": int(cell_path.stat().st_size),
            "rows": int(len(comparison)),
            "columns": list(comparison.columns),
        },
        {
            "path": macro_path.resolve().as_posix(),
            "sha256": sha256_file(macro_path),
            "bytes": int(macro_path.stat().st_size),
            "rows": 1,
            "columns": list(macro_frame.columns),
        },
    ]
    elapsed = time.time() - started
    receipt = {
        "schema_version": "r3_2.heldout_null_reference.v1",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "analysis_role": "explicit_intercept_only_reference_for_final_model_clinical_utility",
        "definition": "calibration-fold Kaplan-Meier target-event risk applied as a constant prediction to the participant-disjoint outer test fold",
        "outcome_convention": "cause-specific target event; death censored, matching the held-out clinical-utility analysis",
        "horizon_months": 60.0,
        "cache_records": cache_records,
        "inputs": inputs,
        "runner": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": outputs,
        "macro": macro,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_survival": __import__("sksurv").__version__,
        },
        "elapsed_seconds": float(elapsed),
    }
    json_write(receipt_path, receipt)
    print(json.dumps({"status": "COMPLETE", "macro": macro}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
