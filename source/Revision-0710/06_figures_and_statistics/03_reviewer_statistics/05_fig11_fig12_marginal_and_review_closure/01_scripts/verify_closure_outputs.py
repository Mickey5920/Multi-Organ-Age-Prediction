#!/usr/bin/env python3
"""Verify reviewer-closure source data before any figure rendering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORGANS = {
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
}
METRICS = {
    "delta_harrell_c",
    "delta_uno_c_60m",
    "delta_auc_60m",
    "brier_improvement_60m",
    "ipcw_idi_60m",
    "ipcw_continuous_nri_60m",
    "mean_delta_net_benefit",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    require(not frame.duplicated(keys).any(), f"{label}: duplicate key {keys}")


def require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    require(
        np.isfinite(frame[columns].to_numpy(dtype=float)).all(),
        f"{label}: non-finite values in {columns}",
    )


def verify_associations() -> dict[str, int]:
    directory = ROOT / "02_single_gap_associations"
    coefficients = pd.read_csv(directory / "Fig12/marginal_association_coefficients.csv")
    matrix = pd.read_csv(directory / "Fig11/AllGroups_Marginal_Association_Matrix.csv")
    require(len(coefficients) == 672, "association coefficient row count")
    require_unique(
        coefficients,
        ["analysis_group", "source_organ", "target_organ"],
        "associations",
    )
    require((coefficients["source_organ"] != coefficients["target_organ"]).all(), "diagonal path")
    require(set(coefficients["source_organ"]) == ORGANS, "source-organ universe")
    require(set(coefficients["target_organ"]) == ORGANS, "target-organ universe")
    require(coefficients["analysis_group"].nunique() == 16, "group universe")
    require((coefficients["other_age_gaps_in_model"] == 0).all(), "other gaps entered association")
    require_finite(
        coefficients,
        ["beta_standardized", "se_hc1", "ci95_low", "ci95_high", "p_value", "q_value_bh_42_within_group"],
        "associations",
    )
    require(matrix.shape == (42, 17), f"association matrix shape {matrix.shape}")
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "COMPLETE_SOURCE_DATA_NO_RENDERING", "association manifest status")
    return {"association_rows": len(coefficients), "association_matrix_rows": len(matrix)}


def verify_nested_cox() -> dict[str, int]:
    directory = ROOT / "03_density_burden_nested_cox"
    estimates = pd.read_csv(directory / "nested_cox_gap_estimates.csv")
    attenuation = pd.read_csv(directory / "density_burden_attenuation.csv")
    ph = pd.read_csv(directory / "m3_gap_ph_diagnostics.csv")
    require(len(estimates) == 35, "nested Cox row count")
    require(len(attenuation) == 35, "attenuation row count")
    require(len(ph) == 14, "M3 PH row count")
    require_unique(estimates, ["organ_system", "model"], "nested Cox")
    require(set(estimates["organ_system"]) == ORGANS, "nested Cox organ universe")
    require(estimates["converged"].astype(bool).all(), "nested Cox convergence")
    require((estimates["other_organ_age_gaps_in_model"] == 0).all(), "other gaps entered nested Cox")
    require_finite(
        estimates,
        ["gap_hr_per_crossfit_sd", "gap_hr_ci95_low", "gap_hr_ci95_high", "gap_p_value"],
        "nested Cox",
    )
    return {"nested_cox_rows": len(estimates), "m3_ph_rows": len(ph)}


def verify_bootstrap() -> dict[str, int]:
    directory = ROOT / "04_clinical_utility_paired_bootstrap"
    cell = pd.read_csv(directory / "cell_bootstrap_replicates.csv.gz")
    organ = pd.read_csv(directory / "organ_bootstrap_replicates.csv")
    macro = pd.read_csv(directory / "macro_bootstrap_replicates.csv")
    organ_ci = pd.read_csv(directory / "organ_paired_bootstrap_ci.csv")
    macro_ci = pd.read_csv(directory / "macro_paired_bootstrap_ci.csv")
    parity = pd.read_csv(directory / "point_metric_parity.csv")
    weight_parity = pd.read_csv(directory / "bootstrap_weight_metric_parity.csv")
    replicates = cell["bootstrap"].nunique()
    require(replicates >= 100, "bootstrap replicate count")
    require(set(cell["bootstrap"]) == set(range(replicates)), "bootstrap IDs")
    require(len(cell) == replicates * 5 * 7 * 2, "cell bootstrap row count")
    require(len(organ) == replicates * 7 * 2, "organ bootstrap row count")
    require(len(macro) == replicates * 2, "macro bootstrap row count")
    require(len(organ_ci) == 7 * 2 * 7, "organ CI row count")
    require(len(macro_ci) == 2 * 7, "macro CI row count")
    require_unique(cell, ["bootstrap", "outer_fold", "organ_system", "augmentation"], "cell bootstrap")
    require_unique(organ, ["bootstrap", "organ_system", "augmentation"], "organ bootstrap")
    require_unique(macro, ["bootstrap", "augmentation"], "macro bootstrap")
    require(set(organ_ci["metric"]) == METRICS, "organ CI metric universe")
    require(set(macro_ci["metric"]) == METRICS, "macro CI metric universe")
    require_finite(cell, sorted(METRICS), "cell bootstrap")
    require_finite(organ_ci, ["point_estimate", "ci95_low", "ci95_high"], "organ CI")
    require_finite(macro_ci, ["point_estimate", "ci95_low", "ci95_high"], "macro CI")
    require(parity["difference"].abs().max() <= 2e-5, "point parity")
    require(weight_parity["difference"].abs().max() <= 2e-8, "weighted bootstrap parity")
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "COMPLETE", "bootstrap manifest status")
    return {"bootstrap_replicates": replicates, "cell_bootstrap_rows": len(cell)}


def verify_windows() -> dict[str, int]:
    directory = ROOT / "05_m3_time_window_sensitivity"
    results = pd.read_csv(directory / "m3_time_window_gap_hr.csv")
    require(len(results) == 14, "M3 time-window row count")
    require_unique(results, ["organ_system", "time_window"], "M3 time windows")
    require(set(results["organ_system"]) == ORGANS, "M3 window organ universe")
    require(results["time_window"].nunique() == 2, "M3 window universe")
    require(results["converged"].astype(bool).all(), "M3 window convergence")
    require((results["other_organ_age_gaps_in_model"] == 0).all(), "other gaps entered M3 windows")
    require_finite(
        results,
        ["gap_hr_per_crossfit_sd", "gap_hr_ci95_low", "gap_hr_ci95_high", "gap_p_value"],
        "M3 windows",
    )
    return {"m3_time_window_rows": len(results)}


def main() -> int:
    summary = {
        **verify_associations(),
        **verify_nested_cox(),
        **verify_bootstrap(),
        **verify_windows(),
    }
    print(json.dumps({"status": "PASS", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
