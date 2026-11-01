#!/usr/bin/env python3
"""Close the Fig. 8 BH-FDR gap without refitting any statistical model.

This script treats the existing Fig. 8 Cox result tables as immutable inputs. It
adds BH q values to the seven age-gap hypotheses within each
System_Group-by-Organ panel, separately for the primary single-gap analysis and
the simultaneous-seven-gap sensitivity analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


OUT_DIR = Path(__file__).resolve().parent
FINAL_FIG_ROOT = OUT_DIR.parents[1]
FIG8_ROOT = FINAL_FIG_ROOT / "Fig08" / "Package_0626_Style"

INPUTS = {
    "primary_single_gap": {
        "path": FIG8_ROOT / "Combined_Single_Gap_Full_Cox_Results.csv",
        "sha256": "c4b2c8eb18c2922b6649f8f600bbdffd540c41966d8fb9f43ac504eb97c2c664",
        "output_all": "fig8_primary_single_gap_full_with_bh.csv",
        "output_gaps": "fig8_primary_single_gap_age_gap_bh.csv",
        "scope": (
            "Seven prespecified organ age-gap hypotheses within one clinical "
            "group by target-organ outcome panel; each age-gap coefficient is "
            "from a separate clinical-covariates-plus-one-gap Cox model."
        ),
    },
    "all_gap_sensitivity": {
        "path": FIG8_ROOT / "Combined_Cox_Bootstrap_Results.csv",
        "sha256": "bfd80c1cf67bdbe0e1a9f38da601f6daf7645fcd8903be55f9381588562df848",
        "output_all": "fig8_all_gap_sensitivity_full_with_bh.csv",
        "output_gaps": "fig8_all_gap_sensitivity_age_gap_bh.csv",
        "scope": (
            "Seven prespecified organ age-gap coefficients within one clinical "
            "group by target-organ outcome panel; all seven age gaps are "
            "included simultaneously in the Cox model."
        ),
    },
}

GAP_ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
EXPECTED_ROWS = 1568
EXPECTED_PANELS = 112
EXPECTED_ROWS_PER_PANEL = 14
EXPECTED_GAPS_PER_PANEL = 7
ALPHA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_manual(p_values: np.ndarray) -> np.ndarray:
    """Deterministic Benjamini-Hochberg adjusted p values."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("BH input must be a non-empty one-dimensional vector")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH input contains a missing or out-of-range p value")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    m = len(ranked)
    raw_adjusted = ranked * m / np.arange(1, m + 1, dtype=float)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted_sorted = np.minimum(monotone, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def gap_name_from_covariate(value: str) -> str | None:
    prefix = "Corrected_"
    suffix = "_Age_Gap"
    if not value.startswith(prefix) or not value.endswith(suffix):
        return None
    organ = value[len(prefix) : -len(suffix)]
    return organ if organ in GAP_ORGANS else None


def target_gap_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: gap_name_from_covariate(row["covariate"]) == row["Organ"],
        axis=1,
    )


def validate_and_adjust(role: str, spec: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = Path(spec["path"])
    actual_hash = sha256(source)
    if actual_hash != spec["sha256"]:
        raise RuntimeError(
            f"Source hash mismatch for {source}: {actual_hash} != {spec['sha256']}"
        )

    # Strings preserve every source field as represented in the source CSV.
    original = pd.read_csv(source, dtype=str, keep_default_na=False)
    if len(original) != EXPECTED_ROWS:
        raise RuntimeError(f"{role}: expected {EXPECTED_ROWS} rows, observed {len(original)}")
    required = {"System_Group", "Organ", "covariate", "p"}
    missing = required.difference(original.columns)
    if missing:
        raise RuntimeError(f"{role}: missing columns {sorted(missing)}")

    panel_sizes = original.groupby(["System_Group", "Organ"], dropna=False).size()
    if len(panel_sizes) != EXPECTED_PANELS or not (panel_sizes == EXPECTED_ROWS_PER_PANEL).all():
        raise RuntimeError(
            f"{role}: expected {EXPECTED_PANELS} panels with "
            f"{EXPECTED_ROWS_PER_PANEL} rows each"
        )

    is_gap = original["covariate"].map(gap_name_from_covariate).notna()
    gaps = original.loc[is_gap].copy()
    family_sizes = gaps.groupby(["System_Group", "Organ"], dropna=False).size()
    if len(family_sizes) != EXPECTED_PANELS or not (family_sizes == EXPECTED_GAPS_PER_PANEL).all():
        raise RuntimeError(f"{role}: each panel must contain exactly seven age-gap rows")

    gaps["gap_organ"] = gaps["covariate"].map(gap_name_from_covariate)
    duplicate_keys = gaps.duplicated(["System_Group", "Organ", "gap_organ"])
    if duplicate_keys.any():
        raise RuntimeError(f"{role}: duplicate panel-by-age-gap hypothesis key")

    p_numeric = pd.to_numeric(gaps["p"], errors="coerce")
    if p_numeric.isna().any() or ((p_numeric < 0) | (p_numeric > 1)).any():
        raise RuntimeError(f"{role}: missing or out-of-range age-gap p values")

    q_by_index: dict[int, float] = {}
    parity_max_difference = 0.0
    family_rows: list[dict[str, object]] = []
    for (system_group, target_organ), family in gaps.groupby(
        ["System_Group", "Organ"], sort=True, dropna=False
    ):
        p = pd.to_numeric(family["p"], errors="raise").to_numpy(dtype=float)
        manual = bh_manual(p)
        _, reference, _, _ = multipletests(p, alpha=ALPHA, method="fdr_bh")
        max_difference = float(np.max(np.abs(manual - reference)))
        parity_max_difference = max(parity_max_difference, max_difference)
        if max_difference > 1e-15:
            raise RuntimeError(
                f"{role}/{system_group}/{target_organ}: manual BH parity failed: "
                f"{max_difference}"
            )
        for idx, q_value in zip(family.index, manual, strict=True):
            q_by_index[int(idx)] = float(q_value)

        target_mask = target_gap_mask(family)
        if int(target_mask.sum()) != 1:
            raise RuntimeError(
                f"{role}/{system_group}/{target_organ}: expected exactly one target-gap row"
            )
        family_p = pd.to_numeric(family["p"], errors="raise").to_numpy(dtype=float)
        family_q = manual
        target_pos = int(np.flatnonzero(target_mask.to_numpy())[0])
        family_rows.append(
            {
                "analysis_role": role,
                "System_Group": system_group,
                "Organ": target_organ,
                "bh_family_id": f"fig8::{role}::{system_group}::{target_organ}",
                "bh_family_size_planned": EXPECTED_GAPS_PER_PANEL,
                "bh_family_size_estimable": len(family),
                "n_nominal_p_lt_0_05": int((family_p < ALPHA).sum()),
                "n_bh_q_lt_0_05": int((family_q < ALPHA).sum()),
                "n_lost_significance_after_bh": int(
                    ((family_p < ALPHA) & (family_q >= ALPHA)).sum()
                ),
                "n_gained_significance_after_bh": int(
                    ((family_p >= ALPHA) & (family_q < ALPHA)).sum()
                ),
                "minimum_raw_p": float(np.min(family_p)),
                "minimum_bh_q": float(np.min(family_q)),
                "maximum_bh_q": float(np.max(family_q)),
                "target_gap_p": float(family_p[target_pos]),
                "target_gap_bh_q": float(family_q[target_pos]),
                "target_gap_nominal_p_lt_0_05": bool(family_p[target_pos] < ALPHA),
                "target_gap_bh_q_lt_0_05": bool(family_q[target_pos] < ALPHA),
            }
        )

    augmented = original.copy()
    augmented["analysis_role"] = role
    augmented["is_multiplicity_controlled_hypothesis"] = is_gap.map(
        {True: "True", False: "False"}
    )
    augmented["bh_family_id"] = ""
    augmented["bh_family_scope"] = ""
    augmented["bh_family_size_planned"] = ""
    augmented["bh_family_size_estimable"] = ""
    augmented["p_value_source"] = "model_based_cox_wald"
    augmented["q_value_bh_7_age_gaps_within_panel"] = ""
    augmented["significant_raw_p_lt_0_05"] = ""
    augmented["significant_bh_q_lt_0_05"] = ""
    augmented["multiplicity_status_change"] = "not_in_age_gap_family"

    for idx in gaps.index:
        system_group = original.at[idx, "System_Group"]
        target_organ = original.at[idx, "Organ"]
        p_value = float(original.at[idx, "p"])
        q_value = q_by_index[int(idx)]
        raw_sig = p_value < ALPHA
        bh_sig = q_value < ALPHA
        if raw_sig and not bh_sig:
            change = "lost_significance_after_bh"
        elif (not raw_sig) and bh_sig:
            change = "gained_significance_after_bh"
        else:
            change = "unchanged"
        augmented.at[idx, "bh_family_id"] = (
            f"fig8::{role}::{system_group}::{target_organ}"
        )
        augmented.at[idx, "bh_family_scope"] = spec["scope"]
        augmented.at[idx, "bh_family_size_planned"] = str(EXPECTED_GAPS_PER_PANEL)
        augmented.at[idx, "bh_family_size_estimable"] = str(EXPECTED_GAPS_PER_PANEL)
        augmented.at[idx, "q_value_bh_7_age_gaps_within_panel"] = format(q_value, ".17g")
        augmented.at[idx, "significant_raw_p_lt_0_05"] = str(raw_sig)
        augmented.at[idx, "significant_bh_q_lt_0_05"] = str(bh_sig)
        augmented.at[idx, "multiplicity_status_change"] = change

    # Validate that adding columns did not change any source field value.
    if not augmented.loc[:, original.columns].equals(original):
        raise RuntimeError(f"{role}: source columns changed during augmentation")

    gap_output = augmented.loc[is_gap].copy()
    gap_output.insert(
        len(original.columns),
        "gap_organ",
        gap_output["covariate"].map(gap_name_from_covariate),
    )
    gap_output["is_target_gap"] = target_gap_mask(gap_output).map(
        {True: "True", False: "False"}
    )
    family_summary = pd.DataFrame(family_rows)

    p_all = pd.to_numeric(gap_output["p"], errors="raise").to_numpy(dtype=float)
    q_all = pd.to_numeric(
        gap_output["q_value_bh_7_age_gaps_within_panel"], errors="raise"
    ).to_numpy(dtype=float)
    target_all = gap_output["is_target_gap"].eq("True").to_numpy()
    positive = p_all > 0
    inflation = np.divide(
        q_all,
        p_all,
        out=np.full_like(q_all, np.nan),
        where=positive,
    )
    summary = {
        "analysis_role": role,
        "source_path": str(source),
        "source_sha256": actual_hash,
        "source_rows": len(original),
        "panel_count": len(family_summary),
        "age_gap_hypothesis_count": len(gap_output),
        "family_size": EXPECTED_GAPS_PER_PANEL,
        "nominal_p_lt_0_05": int((p_all < ALPHA).sum()),
        "bh_q_lt_0_05": int((q_all < ALPHA).sum()),
        "lost_significance_after_bh": int(((p_all < ALPHA) & (q_all >= ALPHA)).sum()),
        "gained_significance_after_bh": int(((p_all >= ALPHA) & (q_all < ALPHA)).sum()),
        "target_gap_hypothesis_count": int(target_all.sum()),
        "target_gap_nominal_p_lt_0_05": int(((p_all < ALPHA) & target_all).sum()),
        "target_gap_bh_q_lt_0_05": int(((q_all < ALPHA) & target_all).sum()),
        "target_gap_lost_significance_after_bh": int(
            ((p_all < ALPHA) & (q_all >= ALPHA) & target_all).sum()
        ),
        "minimum_raw_p": float(np.min(p_all)),
        "minimum_bh_q": float(np.min(q_all)),
        "maximum_bh_q": float(np.max(q_all)),
        "maximum_q_to_p_ratio_for_positive_p": float(np.nanmax(inflation)),
        "manual_vs_statsmodels_max_abs_difference": parity_max_difference,
    }
    return augmented, gap_output, family_summary, summary


def build_corrected_family_table() -> pd.DataFrame:
    rows = [
        {
            "family_id": "MT01",
            "analysis": "Fig. 3 age-prediction metric comparisons",
            "family_partition": "one prespecified family",
            "hypotheses_per_family_planned": 18,
            "hypotheses_per_family_estimable": 18,
            "number_of_families_in_current_source": 1,
            "adjustment": "Benjamini-Hochberg FDR",
            "unadjusted_p_source": "paired outer-fold bootstrap comparisons",
            "reported_columns": "bootstrap p; BH q",
            "source": "Fig03/Source_Data/fig03_*_paired_comparisons.csv",
        },
        {
            "family_id": "MT02",
            "analysis": "Fig. 3 macro full-risk delta-C comparisons",
            "family_partition": "one prespecified family",
            "hypotheses_per_family_planned": 6,
            "hypotheses_per_family_estimable": 6,
            "number_of_families_in_current_source": 1,
            "adjustment": "Benjamini-Hochberg FDR",
            "unadjusted_p_source": "paired outer-fold bootstrap comparisons",
            "reported_columns": "bootstrap p; BH q",
            "source": "Fig03/Source_Data/fig03_*_paired_comparisons.csv",
        },
        {
            "family_id": "MT03",
            "analysis": "Fig. 3 organ-level full-risk delta-C comparisons",
            "family_partition": "one prespecified family across six baselines and seven organs",
            "hypotheses_per_family_planned": 42,
            "hypotheses_per_family_estimable": 42,
            "number_of_families_in_current_source": 1,
            "adjustment": "Benjamini-Hochberg FDR",
            "unadjusted_p_source": "paired outer-fold bootstrap comparisons",
            "reported_columns": "bootstrap p; BH q",
            "source": "Fig03/Source_Data/fig03_*_risk_organ_comparisons.csv",
        },
        {
            "family_id": "MT04",
            "analysis": "Fig. 8 primary single-age-gap Cox associations",
            "family_partition": "separately within each clinical group by target-organ outcome panel",
            "hypotheses_per_family_planned": 7,
            "hypotheses_per_family_estimable": 7,
            "number_of_families_in_current_source": 112,
            "adjustment": "Benjamini-Hochberg FDR within panel",
            "unadjusted_p_source": "model-based Cox Wald p",
            "reported_columns": "p; q_value_bh_7_age_gaps_within_panel",
            "source": "Reviewer_Statistics/14_multiple_testing_closure/fig8_primary_single_gap_age_gap_bh.csv",
        },
        {
            "family_id": "MT05",
            "analysis": "Fig. 8 simultaneous-seven-age-gap Cox sensitivity",
            "family_partition": "separately within each clinical group by target-organ outcome panel",
            "hypotheses_per_family_planned": 7,
            "hypotheses_per_family_estimable": 7,
            "number_of_families_in_current_source": 112,
            "adjustment": "Benjamini-Hochberg FDR within panel",
            "unadjusted_p_source": "model-based Cox Wald p",
            "reported_columns": "p; q_value_bh_7_age_gaps_within_panel",
            "source": "Reviewer_Statistics/14_multiple_testing_closure/fig8_all_gap_sensitivity_age_gap_bh.csv",
        },
        {
            "family_id": "MT06",
            "analysis": "Fig. 10 pairwise group comparisons",
            "family_partition": "separately within each target organ",
            "hypotheses_per_family_planned": 120,
            "hypotheses_per_family_estimable": 120,
            "number_of_families_in_current_source": 7,
            "adjustment": "Benjamini-Hochberg FDR within organ",
            "unadjusted_p_source": "two-sided Mann-Whitney U analytic p",
            "reported_columns": "p_value_raw; FDR_p",
            "source": "Fig10/Package_0626_Style/MWU_FDR_results/All_Pairwise_MWU_FDR_results.csv",
        },
        {
            "family_id": "MT07",
            "analysis": "Progressively adjusted target-organ Cox associations",
            "family_partition": "separately within each of five covariate adjustment sets",
            "hypotheses_per_family_planned": 7,
            "hypotheses_per_family_estimable": 7,
            "number_of_families_in_current_source": 5,
            "adjustment": "Benjamini-Hochberg FDR within adjustment set",
            "unadjusted_p_source": "model-based Cox Wald p",
            "reported_columns": "gap_p_value; gap_q_value_bh_7_organs_within_model",
            "source": "Reviewer_Statistics/03_density_burden_nested_cox/nested_cox_gap_estimates.csv",
        },
        {
            "family_id": "MT08",
            "analysis": "Fully adjusted Cox proportional-hazards diagnostics",
            "family_partition": "one family across seven organs and two prespecified time transforms",
            "hypotheses_per_family_planned": 14,
            "hypotheses_per_family_estimable": 14,
            "number_of_families_in_current_source": 1,
            "adjustment": "Benjamini-Hochberg FDR",
            "unadjusted_p_source": "scaled-Schoenfeld-residual diagnostic p",
            "reported_columns": "p_value; q_value_bh_14_gap_ph_tests",
            "source": "Reviewer_Statistics/03_density_burden_nested_cox/m3_gap_ph_diagnostics.csv",
        },
        {
            "family_id": "MT09",
            "analysis": "Fully adjusted Cox time-window sensitivity",
            "family_partition": "separately within 0-24 and 24-60 month windows",
            "hypotheses_per_family_planned": 7,
            "hypotheses_per_family_estimable": 7,
            "number_of_families_in_current_source": 2,
            "adjustment": "Benjamini-Hochberg FDR within time window",
            "unadjusted_p_source": "model-based Cox Wald p",
            "reported_columns": "gap_p_value; gap_q_value_bh_7_organs_within_window",
            "source": "Reviewer_Statistics/05_m3_time_window_sensitivity/m3_time_window_gap_hr.csv",
        },
        {
            "family_id": "MT10",
            "analysis": "Fig. 11/Fig. 12 adjusted marginal associations",
            "family_partition": "separately within each of 16 clinical groups",
            "hypotheses_per_family_planned": 42,
            "hypotheses_per_family_estimable": 42,
            "number_of_families_in_current_source": 16,
            "adjustment": "Benjamini-Hochberg FDR within clinical group",
            "unadjusted_p_source": "HC1-robust OLS t-test p",
            "reported_columns": "p_value; q_value_bh_42_within_group",
            "source": "Fig12/marginal_association_coefficients.csv",
        },
        {
            "family_id": "MT11",
            "analysis": "Full-cohort age-record-density correlations",
            "family_partition": "separately for Pearson and Spearman correlations",
            "hypotheses_per_family_planned": 10,
            "hypotheses_per_family_estimable": 10,
            "number_of_families_in_current_source": 2,
            "adjustment": "Benjamini-Hochberg FDR within scope and correlation method",
            "unadjusted_p_source": "correlation-test analytic p",
            "reported_columns": "p_value; q_value; family_id",
            "source": "Reviewer_Statistics/09_density_age_correlation/Source_Data/age_density_correlations.csv",
        },
        {
            "family_id": "MT12",
            "analysis": "Strict-reference age-record-density correlations",
            "family_partition": "separately for Pearson and Spearman correlations",
            "hypotheses_per_family_planned": 10,
            "hypotheses_per_family_estimable": 7,
            "number_of_families_in_current_source": 2,
            "adjustment": "Benjamini-Hochberg FDR over the seven estimable hypotheses",
            "unadjusted_p_source": "correlation-test analytic p; three constant variables retained as non-estimable",
            "reported_columns": "p_value; q_value; family_id; status",
            "source": "Reviewer_Statistics/09_density_age_correlation/Source_Data/age_density_correlations.csv",
        },
        {
            "family_id": "MT13",
            "analysis": "MIMIC cause-specific target-event hazard associations",
            "family_partition": "separately within each input/model version and follow-up horizon",
            "hypotheses_per_family_planned": 7,
            "hypotheses_per_family_estimable": 7,
            "number_of_families_in_current_source": 2,
            "adjustment": "Benjamini-Hochberg FDR within version and horizon",
            "unadjusted_p_source": "model-based cause-specific Cox Wald p",
            "reported_columns": "p_value; bh_q_value",
            "source": "Fig13/Source_Data/Fig13E_cause_specific_hr.csv",
        },
        {
            "family_id": "MT14",
            "analysis": "NHANES survey-weighted questionnaire associations",
            "family_partition": "separately within each input/model version and survey release",
            "hypotheses_per_family_planned": 7,
            "hypotheses_per_family_estimable": 6,
            "number_of_families_in_current_source": 2,
            "adjustment": "Benjamini-Hochberg FDR over six estimable outcomes within version and release",
            "unadjusted_p_source": "survey-design-based t/Wald p; immune outcome non-estimable by design",
            "reported_columns": "p_value; bh_q_value; estimability_status",
            "source": "Fig14/Source_Data/Fig14F_questionnaire_or.csv",
        },
    ]
    return pd.DataFrame(rows)


def verify_existing_family_sources() -> pd.DataFrame:
    """Bind every non-Fig. 8 family-table row to its physical source and BH q."""

    validation_rows: list[dict[str, object]] = []

    def load(relative_path: str) -> tuple[pd.DataFrame, Path]:
        path = FINAL_FIG_ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing multiple-testing source: {path}")
        return pd.read_csv(path), path

    def validate_groups(
        *,
        table_family_id: str,
        analysis: str,
        data: pd.DataFrame,
        group_columns: list[str],
        p_column: str,
        q_column: str,
        expected_family_count: int,
        expected_estimable_m: int,
        planned_m: int,
        source_paths: list[Path],
    ) -> None:
        missing = set(group_columns + [p_column, q_column]).difference(data.columns)
        if missing:
            raise RuntimeError(
                f"{table_family_id}: missing validation columns {sorted(missing)}"
            )
        grouped = data.groupby(group_columns, sort=True, dropna=False)
        if grouped.ngroups != expected_family_count:
            raise RuntimeError(
                f"{table_family_id}: expected {expected_family_count} families, "
                f"observed {grouped.ngroups}"
            )
        source_record = "; ".join(
            f"{path.relative_to(FINAL_FIG_ROOT).as_posix()}|sha256={sha256(path)}"
            for path in source_paths
        )
        for key, family in grouped:
            if len(family) != expected_estimable_m:
                raise RuntimeError(
                    f"{table_family_id}/{key}: expected m={expected_estimable_m}, "
                    f"observed {len(family)}"
                )
            p = pd.to_numeric(family[p_column], errors="coerce").to_numpy(dtype=float)
            q = pd.to_numeric(family[q_column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise RuntimeError(f"{table_family_id}/{key}: invalid p value")
            if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
                raise RuntimeError(f"{table_family_id}/{key}: invalid q value")
            reference = bh_manual(p)
            max_difference = float(np.max(np.abs(reference - q)))
            if max_difference > 1e-12:
                raise RuntimeError(
                    f"{table_family_id}/{key}: stored BH q mismatch; "
                    f"max abs difference={max_difference}"
                )
            if not isinstance(key, tuple):
                key = (key,)
            family_key = " | ".join(
                f"{column}={value}" for column, value in zip(group_columns, key, strict=True)
            )
            validation_rows.append(
                {
                    "table_family_id": table_family_id,
                    "analysis": analysis,
                    "family_key": family_key,
                    "planned_hypotheses": planned_m,
                    "estimable_hypotheses": len(family),
                    "p_column": p_column,
                    "q_column": q_column,
                    "maximum_absolute_bh_q_difference": max_difference,
                    "status": "PASS",
                    "physical_sources": source_record,
                }
            )

    # Fig. 3 age metrics and macro risk share two tier-specific physical files.
    paired_plain, paired_plain_path = load(
        "Fig03/Source_Data/fig03_plain_temporal_paired_comparisons.csv"
    )
    paired_organ, paired_organ_path = load(
        "Fig03/Source_Data/fig03_organ_query_paired_comparisons.csv"
    )
    paired = pd.concat([paired_plain, paired_organ], ignore_index=True)
    validate_groups(
        table_family_id="MT01",
        analysis="Fig. 3 age-prediction metric comparisons",
        data=paired.loc[paired["fdr_family"].eq("age_all_metrics_vs_six_baselines")],
        group_columns=["fdr_family"],
        p_column="p_value_two_sided_bootstrap",
        q_column="q_value_bh",
        expected_family_count=1,
        expected_estimable_m=18,
        planned_m=18,
        source_paths=[paired_plain_path, paired_organ_path],
    )
    validate_groups(
        table_family_id="MT02",
        analysis="Fig. 3 macro full-risk delta-C comparisons",
        data=paired.loc[paired["fdr_family"].eq("risk_macro_vs_six_baselines")],
        group_columns=["fdr_family"],
        p_column="p_value_two_sided_bootstrap",
        q_column="q_value_bh",
        expected_family_count=1,
        expected_estimable_m=6,
        planned_m=6,
        source_paths=[paired_plain_path, paired_organ_path],
    )

    risk_plain, risk_plain_path = load(
        "Fig03/Source_Data/fig03_plain_temporal_risk_organ_comparisons.csv"
    )
    risk_organ, risk_organ_path = load(
        "Fig03/Source_Data/fig03_organ_query_risk_organ_comparisons.csv"
    )
    risk = pd.concat([risk_plain, risk_organ], ignore_index=True)
    validate_groups(
        table_family_id="MT03",
        analysis="Fig. 3 organ-level full-risk delta-C comparisons",
        data=risk,
        group_columns=["fdr_family"],
        p_column="p_value_two_sided_bootstrap",
        q_column="q_value_bh",
        expected_family_count=1,
        expected_estimable_m=42,
        planned_m=42,
        source_paths=[risk_plain_path, risk_organ_path],
    )

    fig10, fig10_path = load(
        "Fig10/Package_0626_Style/MWU_FDR_results/All_Pairwise_MWU_FDR_results.csv"
    )
    validate_groups(
        table_family_id="MT06",
        analysis="Fig. 10 pairwise group comparisons",
        data=fig10,
        group_columns=["Organ"],
        p_column="p_value_raw",
        q_column="FDR_p",
        expected_family_count=7,
        expected_estimable_m=120,
        planned_m=120,
        source_paths=[fig10_path],
    )

    nested, nested_path = load(
        "Reviewer_Statistics/03_density_burden_nested_cox/nested_cox_gap_estimates.csv"
    )
    validate_groups(
        table_family_id="MT07",
        analysis="Progressively adjusted target-organ Cox associations",
        data=nested,
        group_columns=["model"],
        p_column="gap_p_value",
        q_column="gap_q_value_bh_7_organs_within_model",
        expected_family_count=5,
        expected_estimable_m=7,
        planned_m=7,
        source_paths=[nested_path],
    )

    ph, ph_path = load(
        "Reviewer_Statistics/03_density_burden_nested_cox/m3_gap_ph_diagnostics.csv"
    )
    ph = ph.assign(_family="fully_adjusted_ph_diagnostics")
    validate_groups(
        table_family_id="MT08",
        analysis="Fully adjusted Cox proportional-hazards diagnostics",
        data=ph,
        group_columns=["_family"],
        p_column="p_value",
        q_column="q_value_bh_14_gap_ph_tests",
        expected_family_count=1,
        expected_estimable_m=14,
        planned_m=14,
        source_paths=[ph_path],
    )

    windows, windows_path = load(
        "Reviewer_Statistics/05_m3_time_window_sensitivity/m3_time_window_gap_hr.csv"
    )
    validate_groups(
        table_family_id="MT09",
        analysis="Fully adjusted Cox time-window sensitivity",
        data=windows,
        group_columns=["time_window"],
        p_column="gap_p_value",
        q_column="gap_q_value_bh_7_organs_within_window",
        expected_family_count=2,
        expected_estimable_m=7,
        planned_m=7,
        source_paths=[windows_path],
    )

    marginal, marginal_path = load("Fig12/marginal_association_coefficients.csv")
    validate_groups(
        table_family_id="MT10",
        analysis="Fig. 11/Fig. 12 adjusted marginal associations",
        data=marginal,
        group_columns=["analysis_group"],
        p_column="p_value",
        q_column="q_value_bh_42_within_group",
        expected_family_count=16,
        expected_estimable_m=42,
        planned_m=42,
        source_paths=[marginal_path],
    )

    density, density_path = load(
        "Reviewer_Statistics/09_density_age_correlation/Source_Data/age_density_correlations.csv"
    )
    validate_groups(
        table_family_id="MT11",
        analysis="Full-cohort age-record-density correlations",
        data=density.loc[
            density["scope"].eq("full_oof") & density["status"].eq("OK")
        ],
        group_columns=["scope", "method"],
        p_column="p_value",
        q_column="q_value",
        expected_family_count=2,
        expected_estimable_m=10,
        planned_m=10,
        source_paths=[density_path],
    )
    strict_density = density.loc[density["scope"].eq("strict_healthy")]
    if len(strict_density) != 20 or int(strict_density["status"].eq("NOT_ESTIMABLE_CONSTANT").sum()) != 6:
        raise RuntimeError(
            "MT12: strict-reference density source must retain three non-estimable "
            "constant variables per correlation method"
        )
    validate_groups(
        table_family_id="MT12",
        analysis="Strict-reference age-record-density correlations",
        data=strict_density.loc[strict_density["status"].eq("OK")],
        group_columns=["scope", "method"],
        p_column="p_value",
        q_column="q_value",
        expected_family_count=2,
        expected_estimable_m=7,
        planned_m=10,
        source_paths=[density_path],
    )

    mimic, mimic_path = load("Fig13/Source_Data/Fig13E_cause_specific_hr.csv")
    validate_groups(
        table_family_id="MT13",
        analysis="MIMIC cause-specific target-event hazard associations",
        data=mimic.loc[mimic["estimability_status"].eq("estimable")],
        group_columns=["input_variant", "horizon_months"],
        p_column="p_value",
        q_column="bh_q_value",
        expected_family_count=2,
        expected_estimable_m=7,
        planned_m=7,
        source_paths=[mimic_path],
    )

    nhanes, nhanes_path = load("Fig14/Source_Data/Fig14F_questionnaire_or.csv")
    if len(nhanes) != 14 or int(nhanes["estimability_status"].eq("not_estimable_by_design").sum()) != 2:
        raise RuntimeError(
            "MT14: NHANES source must retain one non-estimable immune row per release"
        )
    validate_groups(
        table_family_id="MT14",
        analysis="NHANES survey-weighted questionnaire associations",
        data=nhanes.loc[nhanes["estimability_status"].eq("estimable")],
        group_columns=["input_variant", "release_id"],
        p_column="p_value",
        q_column="bh_q_value",
        expected_family_count=2,
        expected_estimable_m=6,
        planned_m=7,
        source_paths=[nhanes_path],
    )

    result = pd.DataFrame(validation_rows)
    if len(result) != 42:
        # Non-Fig. 8 sources contain 42 physical family instances; the two
        # Fig. 8 result sets add 224, for 266 total current family instances.
        raise RuntimeError(
            f"Expected 42 non-Fig. 8 family instances, observed {len(result)}"
        )
    return result


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_family_summaries: list[pd.DataFrame] = []
    result_summaries: list[dict[str, object]] = []
    output_paths: list[Path] = []

    for role, spec in INPUTS.items():
        augmented, gaps, family_summary, summary = validate_and_adjust(role, spec)
        all_path = OUT_DIR / str(spec["output_all"])
        gap_path = OUT_DIR / str(spec["output_gaps"])
        augmented.to_csv(all_path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        gaps.to_csv(gap_path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

        # Round-trip gate: every original field must remain identical as text.
        source_again = pd.read_csv(spec["path"], dtype=str, keep_default_na=False)
        augmented_again = pd.read_csv(all_path, dtype=str, keep_default_na=False)
        if not augmented_again.loc[:, source_again.columns].equals(source_again):
            raise RuntimeError(f"{role}: augmented CSV failed source-field round-trip parity")

        all_family_summaries.append(family_summary)
        result_summaries.append(summary)
        output_paths.extend([all_path, gap_path])

    family_summary_all = pd.concat(all_family_summaries, ignore_index=True)
    family_summary_path = OUT_DIR / "fig8_bh_family_summary.csv"
    family_summary_all.to_csv(family_summary_path, index=False, lineterminator="\n")
    output_paths.append(family_summary_path)

    significance_summary = pd.DataFrame(result_summaries)
    significance_path = OUT_DIR / "fig8_bh_significance_summary.csv"
    significance_summary.to_csv(significance_path, index=False, lineterminator="\n")
    output_paths.append(significance_path)

    status_changes = []
    for role, spec in INPUTS.items():
        gaps_path = OUT_DIR / str(spec["output_gaps"])
        gaps = pd.read_csv(gaps_path, dtype=str, keep_default_na=False)
        changed = gaps.loc[
            gaps["multiplicity_status_change"].ne("unchanged"),
            [
                "analysis_role",
                "System_Group",
                "Organ",
                "gap_organ",
                "covariate",
                "p",
                "q_value_bh_7_age_gaps_within_panel",
                "is_target_gap",
                "multiplicity_status_change",
            ],
        ].copy()
        status_changes.append(changed)
    status_changes_all = pd.concat(status_changes, ignore_index=True)
    if len(status_changes_all):
        status_changes_all["_p_sort"] = pd.to_numeric(status_changes_all["p"])
        status_changes_all = status_changes_all.sort_values(
            ["analysis_role", "_p_sort", "System_Group", "Organ", "gap_organ"],
            kind="mergesort",
        ).drop(columns="_p_sort")
    changes_path = OUT_DIR / "fig8_bh_status_changes.csv"
    status_changes_all.to_csv(changes_path, index=False, lineterminator="\n")
    output_paths.append(changes_path)

    family_table = build_corrected_family_table()
    family_table_path = OUT_DIR / "multiple_testing_families_corrected.csv"
    family_table.to_csv(family_table_path, index=False, lineterminator="\n")
    output_paths.append(family_table_path)

    family_source_validation = verify_existing_family_sources()
    family_source_validation_path = OUT_DIR / "multiple_testing_family_source_validation.csv"
    family_source_validation.to_csv(
        family_source_validation_path, index=False, lineterminator="\n"
    )
    output_paths.append(family_source_validation_path)

    changes_to_old_table = pd.DataFrame(
        [
            {
                "change": "remove_obsolete_family",
                "affected_analysis": "Former Fig. 7 prespecified state contrasts",
                "reason": "The 28-test family belongs to a superseded figure analysis and is not part of the current formal result set.",
            },
            {
                "change": "add_missing_family",
                "affected_analysis": "Fig. 8 primary single-age-gap Cox associations",
                "reason": "Seven age-gap hypotheses are now BH-adjusted within each of 112 panels.",
            },
            {
                "change": "add_missing_family",
                "affected_analysis": "Fig. 8 simultaneous-seven-age-gap Cox sensitivity",
                "reason": "Sensitivity-analysis hypotheses are adjusted separately from the primary result.",
            },
            {
                "change": "add_missing_family",
                "affected_analysis": "Fully adjusted Cox proportional-hazards diagnostics",
                "reason": "The 14 prespecified organ-by-time-transform diagnostics were absent from the old table.",
            },
            {
                "change": "correct_estimable_family_size",
                "affected_analysis": "Strict-reference age-record-density correlations",
                "reason": "Ten variables were planned, but three are constant and non-estimable; BH uses the seven estimable tests per method.",
            },
            {
                "change": "make_partition_key_explicit",
                "affected_analysis": "External cohort association families",
                "reason": "Families are version-specific and additionally partitioned by horizon or survey release.",
            },
            {
                "change": "correct_p_value_language",
                "affected_analysis": "All analyses",
                "reason": "The unadjusted p-value source now names the actual procedure instead of calling every p value raw/bootstrap.",
            },
        ]
    )
    family_changes_path = OUT_DIR / "multiple_testing_family_table_changes.csv"
    changes_to_old_table.to_csv(family_changes_path, index=False, lineterminator="\n")
    output_paths.append(family_changes_path)

    result_payload = {
        "schema_version": "r1m3.multiple_testing_closure.v1",
        "method": {
            "procedure": "Benjamini-Hochberg FDR",
            "alpha": ALPHA,
            "family_unit": "seven age-gap hypotheses within each clinical-group-by-target-organ-outcome panel",
            "primary_and_sensitivity_processed_separately": True,
            "clinical_covariates_in_family": False,
            "p_value_source": "model-based Cox Wald p",
            "model_refit_performed": False,
        },
        "input_hashes": {
            role: {
                "path": str(spec["path"]),
                "sha256": sha256(Path(spec["path"])),
            }
            for role, spec in INPUTS.items()
        },
        "results": result_summaries,
        "corrected_family_table": {
            "included_family_rows": len(family_table),
            "current_physical_family_instances": int(
                len(family_source_validation) + len(family_summary_all)
            ),
            "non_fig8_physical_family_instances_source_verified": int(
                len(family_source_validation)
            ),
            "fig8_physical_family_instances_newly_verified": int(
                len(family_summary_all)
            ),
            "maximum_existing_stored_vs_recomputed_bh_q_difference": float(
                family_source_validation[
                    "maximum_absolute_bh_q_difference"
                ].max()
            ),
            "obsolete_fig7_family_removed": True,
            "fully_adjusted_ph_family_added": True,
            "strict_reference_estimable_m": 7,
            "external_families_version_specific": True,
        },
        "validation": {
            "expected_input_rows_each": EXPECTED_ROWS,
            "expected_panels_each": EXPECTED_PANELS,
            "expected_age_gap_hypotheses_each": EXPECTED_PANELS * EXPECTED_GAPS_PER_PANEL,
            "expected_family_size": EXPECTED_GAPS_PER_PANEL,
            "duplicate_hypothesis_keys": 0,
            "missing_or_out_of_range_age_gap_p_values": 0,
            "source_fields_preserved_in_augmented_outputs": True,
            "manual_bh_matches_statsmodels": True,
        },
    }
    result_path = OUT_DIR / "RESULT_SUMMARY.json"
    write_json(result_path, result_payload)
    output_paths.append(result_path)

    manifest_records = []
    manifest_files = output_paths + [
        Path(__file__).resolve(),
        OUT_DIR / "BEST_SKILL_CARD.md",
        OUT_DIR / "README_CN.md",
    ]
    for path in sorted(manifest_files, key=lambda value: value.name.casefold()):
        if not path.is_file():
            raise RuntimeError(f"Manifest-bound file is missing: {path}")
        manifest_records.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": "r1m3.multiple_testing_closure.manifest.v1",
        "status": "PASS",
        "source_files": result_payload["input_hashes"],
        "output_files": manifest_records,
    }
    write_json(OUT_DIR / "MANIFEST.json", manifest)

    print(json.dumps(result_payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
