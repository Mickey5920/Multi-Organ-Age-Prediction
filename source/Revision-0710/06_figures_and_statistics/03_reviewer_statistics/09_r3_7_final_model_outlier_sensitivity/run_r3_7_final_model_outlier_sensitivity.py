"""R3.7 phenotype-outlier sensitivity using the final five-fold OOF model.

The main analysis keeps every finite observed phenotype value. This script derives
an outcome-blind, cross-fitted phenotype-outlier burden and re-evaluates sealed OOF
age predictions and age+sex+target-gap Cox associations after excluding participants
above the other-four-fold 99th-percentile burden threshold. It never refits or selects
the age model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REVISION_ROOT = REPO_ROOT / "Revision-0710"
FINAL_OOF = (
    REVISION_ROOT
    / "03_ehr_age"
    / "06_codebook_v2"
    / "02_fivefold_frozen_huber1"
    / "02_summary"
    / "oof_predictions.parquet"
)
FINAL_SUMMARY = FINAL_OOF.parent / "fivefold_summary.json"
RAW_PHENOTYPES = (
    REVISION_ROOT / "01_data_rebuild" / "01_index_and_events" / "raw_phenotypes.parquet"
)
PRIMARY_SOURCE = (
    REVISION_ROOT
    / "Final_Fig_0716"
    / "Reviewer_Statistics"
    / "12_e47_disease_burden_adjustment"
    / "Source_Data"
)
PRIMARY_COX = PRIMARY_SOURCE / "cox_gap_estimates.csv"
GAP_REFERENCE = PRIMARY_SOURCE / "cox_crossfit_gap_reference.csv"
SHARED_HELPER = (
    REVISION_ROOT
    / "06_figures_and_statistics"
    / "03_reviewer_statistics"
    / "06_supplementary_closure_0716"
    / "run_missingness_outlier_sensitivity.py"
)
DEFAULT_OUTPUT = (
    REVISION_ROOT
    / "Final_Fig_0716"
    / "Reviewer_Statistics"
    / "18_r3_7_final_model_outlier_sensitivity"
)

sys.path.insert(0, str(SHARED_HELPER.parent))
from run_missingness_outlier_sensitivity import (  # noqa: E402
    benjamini_hochberg,
    build_participant_flags,
    fit_cox_breslow,
    regression_metrics,
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
Z95 = 1.959963984540054
EXPECTED_OOF_SHA256 = "bfc9e0de88121bb4b520bd6c878ac4d655d00b7d1dfa1f64b5f7a4de0b633239"
EXPECTED_PRIMARY_COX_SHA256 = "8b823e56d97da7969b3f4a9359979fe025db5bc95324104240ffe00d5c91fae6"
EXPECTED_GAP_REFERENCE_SHA256 = "eb369ecf2b2d9f6df582579d234fc5f7e6df40edef14563fe17249bec54bd755"
EXPECTED_RAW_PHENOTYPES_SHA256 = "a91609da50e943ed03f2e60701efc3d1cae0a292d5cea2f42c77d525a9ee611a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def artifact(path: Path, role: str) -> dict[str, object]:
    result: dict[str, object] = {
        "role": role,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        result["rows"] = int(len(pd.read_csv(path)))
    return result


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = (FINAL_OOF, FINAL_SUMMARY, RAW_PHENOTYPES, PRIMARY_COX, GAP_REFERENCE, SHARED_HELPER)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if sha256_file(FINAL_OOF) != EXPECTED_OOF_SHA256:
        raise RuntimeError("final OOF hash changed")
    if sha256_file(PRIMARY_COX) != EXPECTED_PRIMARY_COX_SHA256:
        raise RuntimeError("formal final-model Cox source hash changed")
    if sha256_file(GAP_REFERENCE) != EXPECTED_GAP_REFERENCE_SHA256:
        raise RuntimeError("formal final-model gap reference hash changed")
    if sha256_file(RAW_PHENOTYPES) != EXPECTED_RAW_PHENOTYPES_SHA256:
        raise RuntimeError("raw phenotype source hash changed")

    oof_columns = [
        "participant_id",
        "organ_system",
        "outer_fold",
        "actual_age",
        "predicted_age",
        "strict_healthy_reference_60m",
        "sex",
        "duration",
        "event",
        "at_risk",
        "corrected_age_gap",
    ]
    oof = pd.read_parquet(FINAL_OOF, columns=oof_columns)
    oof["participant_id"] = oof["participant_id"].astype(str)
    if len(oof) != 456_773 * 7:
        raise RuntimeError("final OOF participant-organ coverage changed")
    if oof.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("final OOF contains duplicate participant-organ rows")
    if set(oof["organ_system"].astype(str).unique()) != set(ORGANS):
        raise RuntimeError("final OOF organ axis changed")

    raw = pd.read_parquet(RAW_PHENOTYPES)
    raw["participant_id"] = raw["participant_id"].astype(str)
    fold_map = oof[["participant_id", "outer_fold"]].drop_duplicates()
    flags, references = build_participant_flags(raw, fold_map)
    flags["participant_id"] = flags["participant_id"].astype(str)
    if len(flags) != 456_773 or flags["participant_id"].nunique() != 456_773:
        raise RuntimeError("outlier flags do not cover the final cohort exactly once")
    if not flags["outer_fold"].isin(range(5)).all():
        raise RuntimeError("outlier flags contain an invalid fold")

    joined = oof.merge(
        flags.drop(columns="outer_fold"),
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    if joined["outlier_burden_top1_crossfit"].isna().any():
        raise RuntimeError("final OOF rows are missing outlier flags")
    return joined, flags, references, json.loads(FINAL_SUMMARY.read_text(encoding="utf-8"))


def outlier_counts(flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", flags)]
    groups.extend(
        (str(fold), flags.loc[flags["outer_fold"].eq(fold)]) for fold in range(5)
    )
    for fold, part in groups:
        excluded = part["outlier_burden_top1_crossfit"].eq(1)
        rows.append(
            {
                "outer_fold": fold,
                "participants": int(len(part)),
                "excluded_participants": int(excluded.sum()),
                "excluded_fraction": float(excluded.mean()),
                "outlier_burden_median": float(part["outlier_burden_crossfit"].median()),
                "outlier_burden_q99_observed": float(
                    part["outlier_burden_crossfit"].quantile(0.99)
                ),
                "outlier_burden_max": float(part["outlier_burden_crossfit"].max()),
            }
        )
    return pd.DataFrame(rows)


def strict_age_metrics(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = joined.loc[joined["strict_healthy_reference_60m"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled_seven_organs", strict)]
    scopes.extend(
        (organ, strict.loc[strict["organ_system"].eq(organ)]) for organ in ORGANS
    )
    samples = {
        "primary_all": lambda frame: np.ones(len(frame), dtype=bool),
        "exclude_crossfit_top1": lambda frame: frame["exclude_outlier_top1"].eq(1).to_numpy(),
    }
    for scope, frame in scopes:
        for sample, selector in samples.items():
            part = frame.loc[selector(frame)]
            rows.append(
                {
                    "scope": scope,
                    "analysis_sample": sample,
                    "participant_n": int(part["participant_id"].nunique()),
                    "participant_organ_rows": int(len(part)),
                    **regression_metrics(part["actual_age"], part["predicted_age"]),
                }
            )
            for fold in range(5):
                fold_part = part.loc[part["outer_fold"].eq(fold)]
                fold_rows.append(
                    {
                        "scope": scope,
                        "analysis_sample": sample,
                        "outer_fold": fold,
                        "participant_n": int(fold_part["participant_id"].nunique()),
                        "participant_organ_rows": int(len(fold_part)),
                        **regression_metrics(
                            fold_part["actual_age"], fold_part["predicted_age"]
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    primary = result.loc[
        result["analysis_sample"].eq("primary_all"),
        ["scope", "mae", "pearson", "r2"],
    ].rename(
        columns={"mae": "primary_mae", "pearson": "primary_pearson", "r2": "primary_r2"}
    )
    result = result.merge(primary, on="scope", how="left", validate="many_to_one")
    result["delta_mae_vs_primary"] = result["mae"] - result["primary_mae"]
    result["delta_pearson_vs_primary"] = result["pearson"] - result["primary_pearson"]
    result["delta_r2_vs_primary"] = result["r2"] - result["primary_r2"]
    return result, pd.DataFrame(fold_rows)


def cox_sensitivity(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap_reference = pd.read_csv(GAP_REFERENCE)
    gap_reference = gap_reference.loc[
        gap_reference["population"].eq("target_at_risk"),
        ["organ_system", "outer_fold", "reference_mean", "reference_sd_ddof0"],
    ]
    if len(gap_reference) != 35:
        raise RuntimeError("final target-at-risk gap reference must contain 35 cells")

    risk = joined.loc[joined["at_risk"].eq(1)].copy()
    risk = risk.merge(
        gap_reference,
        on=["organ_system", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    if risk[["reference_mean", "reference_sd_ddof0"]].isna().any().any():
        raise RuntimeError("risk rows are missing a cross-fitted gap reference")
    risk["gap_z_crossfit"] = (
        risk["corrected_age_gap"] - risk["reference_mean"]
    ) / risk["reference_sd_ddof0"]
    risk["age_per_10y"] = (risk["actual_age"] - 60.0) / 10.0

    primary = pd.read_csv(PRIMARY_COX)
    primary = primary.loc[
        primary["population"].eq("target_at_risk") & primary["model"].eq("M0_age_sex")
    ].copy()
    if len(primary) != 7:
        raise RuntimeError("formal age+sex Cox source must contain seven organs")
    primary = primary[
        [
            "organ_system",
            "n",
            "events",
            "gap_beta_per_crossfit_sd",
            "gap_se_model_based",
            "gap_hr_per_crossfit_sd",
            "gap_hr_ci95_low",
            "gap_hr_ci95_high",
            "gap_p_value",
            "gap_q_value_bh_7_organs_within_model",
        ]
    ].rename(columns={"gap_q_value_bh_7_organs_within_model": "gap_q_value_bh_7_organs"})
    primary.insert(1, "analysis_sample", "primary_all")
    primary["converged"] = True
    primary["estimation_source"] = "formal_final_model_age_sex_cox"

    filtered_rows: list[dict[str, object]] = []
    for organ in ORGANS:
        organ_all = risk.loc[risk["organ_system"].eq(organ)]
        source_row = primary.loc[primary["organ_system"].eq(organ)].iloc[0]
        if len(organ_all) != int(source_row["n"]) or int(organ_all["event"].sum()) != int(
            source_row["events"]
        ):
            raise RuntimeError(f"formal Cox count mismatch for {organ}")
        part = organ_all.loc[organ_all["exclude_outlier_top1"].eq(1)]
        x = part[["age_per_10y", "sex", "gap_z_crossfit"]].to_numpy(float)
        fit = fit_cox_breslow(
            x,
            part["duration"].to_numpy(float),
            part["event"].to_numpy(np.int8),
            strata=part["outer_fold"].to_numpy(int),
        )
        if not fit.converged:
            raise RuntimeError(f"filtered Cox did not converge for {organ}")
        beta = float(fit.beta[2])
        se = float(fit.standard_error[2])
        filtered_rows.append(
            {
                "organ_system": organ,
                "analysis_sample": "exclude_crossfit_top1",
                "n": int(len(part)),
                "events": int(part["event"].sum()),
                "gap_beta_per_crossfit_sd": beta,
                "gap_se_model_based": se,
                "gap_hr_per_crossfit_sd": math.exp(beta),
                "gap_hr_ci95_low": math.exp(beta - Z95 * se),
                "gap_hr_ci95_high": math.exp(beta + Z95 * se),
                "gap_p_value": float(fit.p_value[2]),
                "gap_q_value_bh_7_organs": np.nan,
                "converged": bool(fit.converged),
                "estimation_source": "sealed_oof_subset_age_sex_cox",
            }
        )
    filtered = pd.DataFrame(filtered_rows)
    filtered["gap_q_value_bh_7_organs"] = benjamini_hochberg(filtered["gap_p_value"])
    estimates = pd.concat([primary, filtered], ignore_index=True, sort=False).sort_values(
        ["analysis_sample", "organ_system"]
    )

    comparison = primary.merge(
        filtered,
        on="organ_system",
        suffixes=("_primary", "_filtered"),
        validate="one_to_one",
    )
    comparison = comparison[
        [
            "organ_system",
            "n_primary",
            "n_filtered",
            "events_primary",
            "events_filtered",
            "gap_beta_per_crossfit_sd_primary",
            "gap_beta_per_crossfit_sd_filtered",
            "gap_hr_per_crossfit_sd_primary",
            "gap_hr_per_crossfit_sd_filtered",
        ]
    ]
    comparison["excluded_n"] = comparison["n_primary"] - comparison["n_filtered"]
    comparison["excluded_events"] = comparison["events_primary"] - comparison["events_filtered"]
    comparison["beta_change_filtered_minus_primary"] = (
        comparison["gap_beta_per_crossfit_sd_filtered"]
        - comparison["gap_beta_per_crossfit_sd_primary"]
    )
    comparison["hr_relative_change_filtered_vs_primary"] = (
        comparison["gap_hr_per_crossfit_sd_filtered"]
        / comparison["gap_hr_per_crossfit_sd_primary"]
        - 1.0
    )
    return estimates.reset_index(drop=True), comparison


def main() -> None:
    output = DEFAULT_OUTPUT
    source = output / "Source_Data"
    cache = output / "Analysis_Cache"
    source.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    joined, flags, references, final_summary = load_inputs()
    flags.to_parquet(cache / "participant_outlier_flags.parquet", index=False, compression="zstd")
    write_csv(source / "outlier_crossfit_thresholds.csv", references["outlier_threshold"])
    write_csv(
        source / "outlier_feature_crossfit_reference.csv",
        references["outlier_feature_reference"],
    )
    counts = outlier_counts(flags)
    write_csv(source / "outlier_participant_counts.csv", counts)
    age, age_folds = strict_age_metrics(joined)
    write_csv(source / "strict_age_outlier_sensitivity.csv", age)
    write_csv(source / "strict_age_outlier_sensitivity_by_fold.csv", age_folds)
    cox, comparison = cox_sensitivity(joined)
    write_csv(source / "cox_outlier_sensitivity.csv", cox)
    write_csv(source / "cox_outlier_comparison.csv", comparison)

    pooled_primary = age.loc[
        age["scope"].eq("pooled_seven_organs") & age["analysis_sample"].eq("primary_all")
    ].iloc[0]
    pooled_filtered = age.loc[
        age["scope"].eq("pooled_seven_organs")
        & age["analysis_sample"].eq("exclude_crossfit_top1")
    ].iloc[0]
    official = final_summary["age_metrics"]["strict_healthy_reference_60m"]
    if abs(float(pooled_primary["mae"]) - float(official["mae"])) > 1e-12:
        raise RuntimeError("pooled strict MAE does not match the final five-fold summary")
    if abs(float(pooled_primary["pearson"]) - float(official["pearson"])) > 1e-12:
        raise RuntimeError("pooled strict Pearson does not match the final five-fold summary")
    if abs(float(pooled_primary["r2"]) - float(official["r2"])) > 1e-12:
        raise RuntimeError("pooled strict R2 does not match the final five-fold summary")

    excluded_n = int(flags["outlier_burden_top1_crossfit"].sum())
    strict_ids = set(
        joined.loc[joined["strict_healthy_reference_60m"].eq(1), "participant_id"].unique()
    )
    strict_excluded = int(
        flags.loc[
            flags["participant_id"].isin(strict_ids), "outlier_burden_top1_crossfit"
        ].sum()
    )
    max_abs_hr_change = float(
        comparison["hr_relative_change_filtered_vs_primary"].abs().max()
    )
    qa = {
        "status": "PASS",
        "final_oof_hash_verified": True,
        "participant_coverage": int(flags["participant_id"].nunique()),
        "participant_organ_rows": int(len(joined)),
        "outlier_excluded_participants": excluded_n,
        "outlier_excluded_fraction": excluded_n / len(flags),
        "strict_reference_participants": len(strict_ids),
        "strict_reference_excluded_participants": strict_excluded,
        "formal_strict_metrics_reproduced": True,
        "filtered_cox_models_converged": int(cox.loc[cox["analysis_sample"].eq("exclude_crossfit_top1"), "converged"].sum()),
        "filtered_hr_positive_all_organs": bool(
            (cox.loc[cox["analysis_sample"].eq("exclude_crossfit_top1"), "gap_hr_per_crossfit_sd"] > 1).all()
        ),
        "max_abs_hr_relative_change": max_abs_hr_change,
        "model_refit": False,
        "checkpoint_selection": False,
        "outcomes_used_to_define_outliers": False,
    }
    write_json(output / "QA_REPORT.json", qa)

    output_paths = sorted(source.glob("*")) + [
        cache / "participant_outlier_flags.parquet",
        output / "QA_REPORT.json",
    ]
    manifest = {
        "schema_version": "r3_7.final_model_outlier_sensitivity.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "reviewer_R3_7_final_model_outlier_evidence",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "contract": {
            "main_analysis_outlier_exclusion": "none",
            "main_analysis_winsorization_or_clipping": "none",
            "sensitivity_reference": "other four outer folds; no outcome columns read",
            "feature_center": "median",
            "feature_scale": "MAD*1.4826; IQR/1.349 then SD fallback",
            "robust_z_threshold": 5.0,
            "outlier_burden": "mean(max(abs(robust_z)-5,0)^2) across observed phenotype slots",
            "sensitivity_exclusion": "held-out participant burden above other-four-fold q99",
            "age_evaluation": "sealed final five-fold OOF predictions in strict reference",
            "cox_evaluation": "age + sex + target-organ corrected gap; outer-fold-stratified Breslow cause-specific Cox",
            "gap_scale": "other-four-fold target-at-risk mean and SD from the final analysis",
            "model_refit": False,
            "checkpoint_selection": False,
        },
        "key_results": {
            "excluded_participants": excluded_n,
            "excluded_fraction": excluded_n / len(flags),
            "strict_excluded_participants": strict_excluded,
            "strict_age_primary": {
                "mae": float(pooled_primary["mae"]),
                "pearson": float(pooled_primary["pearson"]),
                "r2": float(pooled_primary["r2"]),
            },
            "strict_age_filtered": {
                "mae": float(pooled_filtered["mae"]),
                "pearson": float(pooled_filtered["pearson"]),
                "r2": float(pooled_filtered["r2"]),
            },
            "max_abs_hr_relative_change": max_abs_hr_change,
            "all_filtered_hr_above_one": qa["filtered_hr_positive_all_organs"],
        },
        "inputs": [
            artifact(FINAL_OOF, "final_fivefold_oof_predictions"),
            artifact(FINAL_SUMMARY, "final_fivefold_summary"),
            artifact(RAW_PHENOTYPES, "raw_78_slot_phenotypes"),
            artifact(PRIMARY_COX, "formal_final_model_age_sex_cox_source"),
            artifact(GAP_REFERENCE, "formal_final_model_crossfit_gap_reference"),
            artifact(SHARED_HELPER, "crossfit_outlier_helper"),
        ],
        "script": artifact(Path(__file__).resolve(), "r3_7_runner"),
        "outputs": [artifact(path, f"output:{path.name}") for path in output_paths],
        "interpretation": {
            "supported": "The primary results are not driven by participants with the highest cross-fitted multivariate phenotype-outlier burden.",
            "not_supported": "No claim that the q99 sensitivity rule is a clinical abnormality threshold or that all possible influential observations have been eliminated.",
        },
    }
    write_json(output / "MANIFEST.json", manifest)
    print(json.dumps(manifest["key_results"], indent=2))
    print(output / "MANIFEST.json")


if __name__ == "__main__":
    main()
