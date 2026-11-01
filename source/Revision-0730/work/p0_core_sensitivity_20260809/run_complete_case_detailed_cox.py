#!/usr/bin/env python3
"""Fully adjusted fixed-model Cox sensitivity in phenotype-complete participants."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import run_baseline_only_reference_sensitivity as core


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
REFERENCE = core.ROOT / "Revision-0730/4-Data/recorded_burden_cox_gap_estimates.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    e47 = core.load_e47_module()
    context, context_audit = e47.load_context(core.DEFAULT_CONTEXT, core.DEFAULT_LABELS)
    rows: list[dict[str, object]] = []
    gap_audits: list[pd.DataFrame] = []
    covariate_audits: list[pd.DataFrame] = []
    c_frames: list[pd.DataFrame] = []
    for organ in core.ORGANS:
        frame = e47.load_organ_frame(e47.DEFAULT_OOF, context, organ)
        complete = frame.loc[frame["phenotype_missing_fraction"].eq(0)].copy()
        if complete["participant_id"].nunique() != 170_171:
            raise RuntimeError(f"{organ}: phenotype-complete participant count changed")
        # phenotype_missing_fraction is identically zero by definition in this
        # subset and is therefore omitted as a non-estimable constant. All
        # remaining covariates in the detailed-burden family are unchanged.
        original_density = e47.DENSITY_RAW
        e47.DENSITY_RAW = tuple(
            value for value in original_density if value != "phenotype_missing_fraction"
        )
        try:
            result, gap_audit, covariate_audit, c_frame = core.fit_detailed_cox(
                e47,
                complete,
                organ,
                "corrected_age_gap",
                "current_correction_complete_78_phenotypes_evaluation_subset",
            )
        finally:
            e47.DENSITY_RAW = original_density
        result["sensitivity_scope"] = (
            "fixed EHR-AGE predictions restricted to participants with all 78 phenotypes observed; "
            "no complete-case retraining; the structurally constant phenotype-missingness fraction was omitted"
        )
        rows.append(result)
        gap_audits.append(gap_audit)
        covariate_audits.append(covariate_audit)
        c_frames.append(c_frame)

    estimates = pd.DataFrame(rows)
    estimates["gap_q_value_bh_7_organs"] = e47.benjamini_hochberg(estimates["gap_p_value"])
    reference = pd.read_csv(REFERENCE)
    reference = reference.loc[reference["model"].eq("M3_detail"), [
        "organ_system",
        "n",
        "events",
        "gap_hr_per_crossfit_sd",
        "gap_hr_ci95_low",
        "gap_hr_ci95_high",
    ]].rename(columns={
        "n": "primary_n",
        "events": "primary_events",
        "gap_hr_per_crossfit_sd": "primary_hr",
        "gap_hr_ci95_low": "primary_hr_ci95_low",
        "gap_hr_ci95_high": "primary_hr_ci95_high",
    })
    estimates = estimates.merge(reference, on="organ_system", how="left", validate="one_to_one")
    estimates["relative_hr_change_vs_primary_percent"] = 100.0 * (
        estimates["gap_hr_per_crossfit_sd"] / estimates["primary_hr"] - 1.0
    )
    output_path = OUTPUT / "complete_case_detailed_burden_cox.csv"
    estimates.to_csv(output_path, index=False)
    pd.concat(gap_audits, ignore_index=True).to_csv(
        OUTPUT / "complete_case_cox_gap_transform_audit.csv", index=False
    )
    pd.concat(covariate_audits, ignore_index=True).to_csv(
        OUTPUT / "complete_case_cox_covariate_transform_audit.csv", index=False
    )
    pd.concat(c_frames, ignore_index=True).to_csv(
        OUTPUT / "complete_case_cox_foldwise_harrell_c.csv", index=False
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "fixed-model phenotype-complete fully adjusted Cox sensitivity",
        "complete_case_definition": "phenotype_missing_fraction == 0 (all 78 phenotypes observed)",
        "model_retrained": False,
        "context_audit": context_audit,
        "input_sha256": {
            "e47_oof": sha256_file(e47.DEFAULT_OOF),
            "context": sha256_file(core.DEFAULT_CONTEXT),
            "labels": sha256_file(core.DEFAULT_LABELS),
            "primary_reference": sha256_file(REFERENCE),
        },
        "output_sha256": sha256_file(output_path),
        "python": sys.version,
    }
    provenance_path = OUTPUT / "complete_case_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(estimates[[
        "organ_system",
        "n",
        "events",
        "gap_hr_per_crossfit_sd",
        "gap_hr_ci95_low",
        "gap_hr_ci95_high",
        "relative_hr_change_vs_primary_percent",
    ]].to_string(index=False))
    print(json.dumps({"status": "PASS", "provenance_sha256": sha256_file(provenance_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
