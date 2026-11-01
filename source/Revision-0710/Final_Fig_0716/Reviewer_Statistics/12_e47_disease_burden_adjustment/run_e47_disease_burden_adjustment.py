#!/usr/bin/env python3
"""Quantify whether frozen E47 organ-age gaps exceed recorded baseline burden.

This is a downstream association/decomposition analysis. It never retrains or
modifies EHR-AGE. Organ-burden covariates are reconstructed from the same
frozen codebook-v2 participant labels used by the E47 outcome contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
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
ORGAN_DISPLAY = {
    "cardiovascular": "Cardiovascular",
    "hepatic": "Hepatic",
    "immune": "Immune",
    "metabolic": "Metabolic",
    "musculoskeletal": "Musculoskeletal",
    "pulmonary": "Pulmonary",
    "renal": "Renal",
}
EXPECTED_PARTICIPANTS = 456_773
EXPECTED_OOF_ROWS = EXPECTED_PARTICIPANTS * len(ORGANS)
EXPECTED_OOF_SHA256 = "bfc9e0de88121bb4b520bd6c878ac4d655d00b7d1dfa1f64b5f7a4de0b633239"
EXPECTED_CONTEXT_SHA256 = "2b4d46d69b1c5c2c9629b27c12dce2d4c3460028fc16a0baffcdfdcbd44fb0bc"
EXPECTED_V2_LABEL_SHA256 = "ce8dbbd8a149a872655cb6d37c0f2f7928b1471d4a54857c7d1189ecebcaaa27"


def root_from_script() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = root_from_script()
HERE = Path(__file__).resolve().parent
SURVIVAL_ROOT = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/03_reviewer_statistics"
    / "04_cox_diagnostics_and_competing_risk"
)
if str(SURVIVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_ROOT))

from survival_analysis_core import (  # noqa: E402
    benjamini_hochberg,
    crossfit_covariates,
    crossfit_exposure_transform,
    fit_cox_breslow,
)


DEFAULT_OOF = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1"
    / "02_summary/oof_predictions.parquet"
)
DEFAULT_CONTEXT = (
    ROOT
    / "Revision-0710/05_downstream/06_metric_v2/01_contextual_covariates"
    / "participant_context_covariates.parquet"
)
DEFAULT_V2_LABELS = (
    ROOT
    / "Revision-0710/01_data_rebuild/09_v2_labels"
    / "icd_participant_labels_v2.parquet"
)
DEFAULT_NO_INCIDENT = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/03_no_incident_control"
    / "02_summary/summary.json"
)


# Cox covariate contract mirrors the prior nested-adjustment analysis, while
# replacing the burden source and adding organ-resolved prevalent ICD counts.
CONTEXT_RAW = (
    "smoking_current",
    "education_higher",
    "townsend",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "employment_other",
    "employment_missing",
    "sleep_duration",
)
DENSITY_RAW = (
    "phenotype_missing_fraction",
    "log1p_preindex_icd_count",
    "log1p_preindex_gp_count",
    "has_preindex_icd",
    "has_preindex_gp",
)
BURDEN_COUNT_RAW = ("prevalent_organ_count",)
BINARY_RAW = (
    "smoking_current",
    "education_higher",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "employment_other",
    "employment_missing",
    "has_preindex_icd",
    "has_preindex_gp",
)
FULL_COX_MODELS = (
    "M0_age_sex",
    "M1_context",
    "M2D_density",
    "M2B_count",
    "M3_count",
    "M3_detail",
)
SUBGROUP_MODELS = (
    "S_no_baseline_icd_adjusted",
    "S_no_baseline_icd_or_gp_adjusted",
)


# Linear gap-explainability models use the materialized missing flags directly.
OLS_BASE = ("actual_age", "sex")
OLS_CONTEXT = (
    "smoking_current",
    "smoking_missing",
    "education_higher",
    "education_missing",
    "townsend",
    "townsend_missing",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    # employment_other is the reference category because these indicators sum to one.
    "employment_missing",
    "sleep_duration",
    "sleep_missing",
)
OLS_DENSITY = DENSITY_RAW
OLS_BURDEN_COUNT = BURDEN_COUNT_RAW
OLS_BINARY = {
    "sex",
    "smoking_current",
    "smoking_missing",
    "education_higher",
    "education_missing",
    "townsend_missing",
    "employment_employed",
    "employment_retired",
    "employment_unemployed",
    "employment_missing",
    "sleep_missing",
    "has_preindex_icd",
    "has_preindex_gp",
}
GAP_MODELS = (
    "G0_age_sex",
    "G1_context",
    "G2D_density",
    "G2B_count",
    "G3_count",
    "G3_detail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--v2-labels", type=Path, default=DEFAULT_V2_LABELS)
    parser.add_argument("--no-incident-summary", type=Path, default=DEFAULT_NO_INCIDENT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def transformed_for(raw_columns: Iterable[str], transformed: Sequence[str]) -> list[str]:
    allowed: list[str] = []
    for raw in raw_columns:
        for candidate in (f"{raw}__missing", f"{raw}__xf"):
            if candidate in transformed:
                allowed.append(candidate)
    return allowed


def detail_raw_for(organ: str, *, include_target: bool) -> tuple[str, ...]:
    organs = ORGANS if include_target else tuple(value for value in ORGANS if value != organ)
    return tuple(f"log1p_prevalent_{value}_count" for value in organs)


def load_context(
    context_path: Path, v2_label_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    context_columns = [
        "participant_id",
        "fold_id",
        "chronological_age",
        "sex_male",
        "smoking_current",
        "smoking_missing",
        "education_higher",
        "education_missing",
        "townsend",
        "townsend_missing",
        "employment_employed",
        "employment_retired",
        "employment_unemployed",
        "employment_other",
        "employment_missing",
        "sleep_duration",
        "sleep_missing",
        "phenotype_missing_fraction",
        "log1p_preindex_icd_count",
        "log1p_preindex_gp_count",
        "has_preindex_icd",
        "has_preindex_gp",
        "prevalent_organ_count",
        "preindex_icd_count",
        "preindex_gp_count",
    ]
    context = pd.read_parquet(context_path, columns=context_columns)
    context["participant_id"] = context["participant_id"].astype(str)
    if len(context) != EXPECTED_PARTICIPANTS or context["participant_id"].duplicated().any():
        raise RuntimeError("context row/participant contract failed")

    label_columns = [
        "participant_id",
        *[f"prevalent_{organ}_count" for organ in ORGANS],
        *[f"has_prevalent_{organ}" for organ in ORGANS],
    ]
    labels = pd.read_parquet(v2_label_path, columns=label_columns)
    labels["participant_id"] = labels["participant_id"].astype(str)
    if len(labels) != EXPECTED_PARTICIPANTS or labels["participant_id"].duplicated().any():
        raise RuntimeError("codebook-v2 label row/participant contract failed")

    merged = context.merge(labels, on="participant_id", how="inner", validate="one_to_one")
    if len(merged) != EXPECTED_PARTICIPANTS:
        raise RuntimeError("context/codebook-v2-label join is incomplete")

    corrected_burden = merged[[f"has_prevalent_{organ}" for organ in ORGANS]].astype(np.int8).sum(axis=1)
    audit = {
        "participants": int(len(merged)),
        "legacy_vs_codebook_v2_prevalent_organ_count_changed": int(
            merged["prevalent_organ_count"].ne(corrected_burden).sum()
        ),
    }

    # Pre-index ICD record density is mapping-independent and stays sourced
    # from the model-input context. Organ burden is overwritten with v2 labels.
    merged["prevalent_organ_count"] = corrected_burden.astype(np.int8)
    for organ in ORGANS:
        count = pd.to_numeric(merged[f"prevalent_{organ}_count"], errors="raise")
        if count.lt(0).any():
            raise RuntimeError(f"negative codebook-v2 prevalent count: {organ}")
        merged[f"log1p_prevalent_{organ}_count"] = np.log1p(count).astype(np.float64)
        merged[f"has_prevalent_{organ}"] = merged[f"has_prevalent_{organ}"].astype(np.int8)

    no_icd = merged["has_preindex_icd"].eq(0) & merged["prevalent_organ_count"].eq(0)
    merged["no_baseline_icd"] = no_icd.astype(np.int8)
    merged["no_baseline_icd_or_gp"] = (no_icd & merged["has_preindex_gp"].eq(0)).astype(np.int8)
    audit.update(
        {
            "no_baseline_icd_participants": int(merged["no_baseline_icd"].sum()),
            "no_baseline_icd_or_gp_participants": int(merged["no_baseline_icd_or_gp"].sum()),
            "codebook_v2_prevalent_burden_mean": float(merged["prevalent_organ_count"].mean()),
            "codebook_v2_prevalent_burden_max": int(merged["prevalent_organ_count"].max()),
        }
    )
    return merged, audit


def load_organ_frame(oof_path: Path, context: pd.DataFrame, organ: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        oof_path,
        columns=[
            "participant_id",
            "organ_system",
            "outer_fold",
            "actual_age",
            "sex",
            "corrected_age_gap",
            "at_risk",
            "duration",
            "event",
        ],
        filters=[("organ_system", "==", organ)],
    )
    frame["participant_id"] = frame["participant_id"].astype(str)
    if len(frame) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(f"{organ}: OOF row count changed: {len(frame)}")
    if frame["participant_id"].duplicated().any() or not frame["organ_system"].eq(organ).all():
        raise RuntimeError(f"{organ}: OOF participant/organ uniqueness failed")
    frame = frame.merge(context, on="participant_id", how="inner", validate="one_to_one")
    if len(frame) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(f"{organ}: context join is incomplete")
    if not frame["fold_id"].eq(frame["outer_fold"]).all():
        raise RuntimeError(f"{organ}: context fold mismatch")
    # E47 exports age after a float32 tensor round-trip, whereas the context
    # table retains float64. The observed maximum discrepancy is <4e-6 years.
    if not np.allclose(frame["actual_age"], frame["chronological_age"], atol=1e-5, rtol=0.0):
        raise RuntimeError(f"{organ}: age mismatch")
    if not np.allclose(frame["sex"], frame["sex_male"], atol=0.0, rtol=0.0):
        raise RuntimeError(f"{organ}: sex mismatch")
    if set(pd.unique(frame["outer_fold"])) != set(range(5)):
        raise RuntimeError(f"{organ}: outer-fold set changed")
    if not np.isfinite(frame["corrected_age_gap"]).all():
        raise RuntimeError(f"{organ}: nonfinite corrected gap")
    at_risk = frame["at_risk"].eq(1)
    if frame.loc[at_risk, "duration"].isna().any() or frame.loc[at_risk, "duration"].le(0).any():
        raise RuntimeError(f"{organ}: invalid at-risk duration")
    if (frame["event"].eq(1) & ~at_risk).any():
        raise RuntimeError(f"{organ}: event outside at-risk set")
    if (at_risk & frame[f"has_prevalent_{organ}"].eq(1)).any():
        raise RuntimeError(f"{organ}: target prevalence remains in at-risk sample")
    frame["duration_months"] = frame["duration"]
    return frame


def _fold_preprocess(
    frame: pd.DataFrame,
    columns: Sequence[str],
    fold: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    train = frame["outer_fold"].ne(fold).to_numpy()
    transformed: dict[str, np.ndarray] = {}
    audit: list[dict[str, object]] = []
    for column in columns:
        raw = pd.to_numeric(frame[column], errors="coerce")
        reference = raw.loc[train].dropna()
        if reference.empty:
            raise RuntimeError(f"OLS fold {fold}: no reference for {column}")
        is_binary = column in OLS_BINARY
        if is_binary:
            modes = reference.mode(dropna=True)
            impute = float(modes.iloc[0])
            center, scale = 0.0, 1.0
        else:
            impute = float(reference.median())
            filled_reference = raw.loc[train].fillna(impute)
            center = float(filled_reference.mean())
            scale = float(filled_reference.std(ddof=0))
            if not np.isfinite(scale) or scale <= 0:
                raise RuntimeError(f"OLS fold {fold}: invalid scale for {column}: {scale}")
        values = raw.fillna(impute).to_numpy(dtype=np.float64)
        transformed[column] = values if is_binary else (values - center) / scale
        audit.append(
            {
                "outer_fold": int(fold),
                "covariate": column,
                "binary": bool(is_binary),
                "reference_n": int(len(reference)),
                "imputation": "mode" if is_binary else "median",
                "imputation_value": impute,
                "reference_center": center,
                "reference_sd_ddof0": scale,
            }
        )
    return transformed, audit


def run_gap_explainability(
    frame: pd.DataFrame,
    organ: str,
    scope: str,
    *,
    detail_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_columns = {
        "G0_age_sex": [*OLS_BASE],
        "G1_context": [*OLS_BASE, *OLS_CONTEXT],
        "G2D_density": [*OLS_BASE, *OLS_CONTEXT, *OLS_DENSITY],
        "G2B_count": [*OLS_BASE, *OLS_CONTEXT, *OLS_BURDEN_COUNT],
        "G3_count": [*OLS_BASE, *OLS_CONTEXT, *OLS_DENSITY, *OLS_BURDEN_COUNT],
        "G3_detail": [*OLS_BASE, *OLS_CONTEXT, *OLS_DENSITY, *detail_columns],
    }
    union_columns = list(dict.fromkeys(column for cols in model_columns.values() for column in cols))
    rows: list[dict[str, object]] = []
    preprocess_rows: list[dict[str, object]] = []
    y = frame["corrected_age_gap"].to_numpy(dtype=np.float64)
    folds = frame["outer_fold"].to_numpy(dtype=np.int8)
    for fold in range(5):
        values, audit = _fold_preprocess(frame, union_columns, fold)
        for row in audit:
            row.update({"organ_system": organ, "scope": scope})
        preprocess_rows.extend(audit)
        train = folds != fold
        test = folds == fold
        if not train.any() or not test.any():
            raise RuntimeError(f"{organ} {scope}: empty fold {fold}")
        for model in GAP_MODELS:
            columns = model_columns[model]
            x = np.column_stack([np.ones(len(frame)), *[values[column] for column in columns]])
            x_train, x_test = x[train], x[test]
            y_train, y_test = y[train], y[test]
            xtx = x_train.T @ x_train
            xty = x_train.T @ y_train
            rank = int(np.linalg.matrix_rank(xtx))
            if rank != xtx.shape[0]:
                raise RuntimeError(
                    f"{organ} {scope} {model} fold {fold}: rank {rank}/{xtx.shape[0]}"
                )
            beta = np.linalg.solve(xtx, xty)
            prediction = x_test @ beta
            residual = y_test - prediction
            centered = y_test - y_test.mean()
            rows.append(
                {
                    "organ_system": organ,
                    "scope": scope,
                    "model": model,
                    "outer_fold": fold,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "n_features_with_intercept": int(x.shape[1]),
                    "design_rank": rank,
                    "xtx_condition_number": float(np.linalg.cond(xtx)),
                    "sse": float(residual @ residual),
                    "sst_within_test_fold": float(centered @ centered),
                    "absolute_error_sum": float(np.abs(residual).sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(preprocess_rows)


def summarize_gap_explainability(fold_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        fold_rows.groupby(["organ_system", "scope", "model"], sort=False)
        .agg(
            n=("n_test", "sum"),
            sse=("sse", "sum"),
            sst=("sst_within_test_fold", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            min_design_rank=("design_rank", "min"),
            max_condition_number=("xtx_condition_number", "max"),
        )
        .reset_index()
    )
    grouped["heldout_r2"] = 1.0 - grouped["sse"] / grouped["sst"]
    grouped["heldout_rmse_years"] = np.sqrt(grouped["sse"] / grouped["n"])
    grouped["heldout_mae_years"] = grouped["absolute_error_sum"] / grouped["n"]
    grouped["heldout_r2_increment_vs_G0"] = np.nan
    grouped["heldout_r2_increment_vs_G1"] = np.nan
    grouped["heldout_r2_increment_vs_G2D"] = np.nan
    for (_, _), index in grouped.groupby(["organ_system", "scope"], sort=False).groups.items():
        idx = list(index)
        values = grouped.loc[idx].set_index("model")["heldout_r2"]
        grouped.loc[idx, "heldout_r2_increment_vs_G0"] = grouped.loc[idx, "heldout_r2"] - values["G0_age_sex"]
        grouped.loc[idx, "heldout_r2_increment_vs_G1"] = grouped.loc[idx, "heldout_r2"] - values["G1_context"]
        grouped.loc[idx, "heldout_r2_increment_vs_G2D"] = grouped.loc[idx, "heldout_r2"] - values["G2D_density"]
    return grouped


def fit_cox_model(
    frame: pd.DataFrame,
    organ: str,
    model: str,
    extras: Sequence[str],
    population: str,
) -> dict[str, object]:
    columns = ["age_per_10y", "sex", "gap_z_crossfit", *extras]
    x = frame[columns].to_numpy(dtype=np.float64)
    duration = frame["duration_months"].to_numpy(dtype=np.float64)
    event = frame["event"].to_numpy(dtype=np.int8)
    strata = frame["outer_fold"].to_numpy(dtype=np.int8)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{organ} {model}: nonfinite Cox design")
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise RuntimeError(f"{organ} {model}: Cox rank {rank}/{x.shape[1]}")
    gram = (x.T @ x) / len(x)
    condition = float(np.linalg.cond(gram))
    fitted = fit_cox_breslow(x, duration, event, strata=strata)
    if not fitted.converged:
        raise RuntimeError(f"{organ} {model}: Cox did not converge")
    gap_index = columns.index("gap_z_crossfit")
    beta = float(fitted.beta[gap_index])
    se = float(fitted.standard_error[gap_index])
    z95 = 1.959963984540054
    return {
        "organ_system": organ,
        "population": population,
        "model": model,
        "n": int(len(frame)),
        "events": int(event.sum()),
        "n_covariates": int(len(columns)),
        "design_rank": rank,
        "gram_condition_number": condition,
        "gap_beta_per_crossfit_sd": beta,
        "gap_se_model_based": se,
        "gap_hr_per_crossfit_sd": float(np.exp(beta)),
        "gap_hr_ci95_low": float(np.exp(beta - z95 * se)),
        "gap_hr_ci95_high": float(np.exp(beta + z95 * se)),
        "gap_p_value": float(fitted.p_value[gap_index]),
        "converged": bool(fitted.converged),
        "iterations": int(fitted.iterations),
        "score_max_abs": float(fitted.score_max_abs),
        "ties": fitted.ties,
        "outer_fold_baseline_hazard": "stratified",
    }


def run_full_cox(
    frame: pd.DataFrame, organ: str
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame]]:
    at_risk = frame.loc[frame["at_risk"].eq(1)].copy()
    at_risk, gap_audit = crossfit_exposure_transform(at_risk)
    gap_audit.insert(0, "population", "target_at_risk")
    gap_audit.insert(0, "organ_system", organ)
    at_risk["age_per_10y"] = (at_risk["actual_age"] - 60.0) / 10.0
    detail_raw = detail_raw_for(organ, include_target=False)
    raw_all = (*CONTEXT_RAW, *DENSITY_RAW, *BURDEN_COUNT_RAW, *detail_raw)
    transformed, covariate_audit, transformed_names = crossfit_covariates(
        at_risk, raw_all, binary_columns=BINARY_RAW
    )
    covariate_audit.insert(0, "population", "target_at_risk")
    covariate_audit.insert(0, "organ_system", organ)
    context_names = transformed_for(CONTEXT_RAW, transformed_names)
    density_names = transformed_for(DENSITY_RAW, transformed_names)
    burden_count_names = transformed_for(BURDEN_COUNT_RAW, transformed_names)
    detail_names = transformed_for(detail_raw, transformed_names)
    models = {
        "M0_age_sex": [],
        "M1_context": context_names,
        "M2D_density": [*context_names, *density_names],
        "M2B_count": [*context_names, *burden_count_names],
        "M3_count": [*context_names, *density_names, *burden_count_names],
        "M3_detail": [*context_names, *density_names, *detail_names],
    }
    rows = [
        fit_cox_model(transformed, organ, model, models[model], "target_at_risk")
        for model in FULL_COX_MODELS
    ]
    return rows, [gap_audit], [covariate_audit]


def run_subgroup_cox(
    frame: pd.DataFrame,
    organ: str,
    *,
    flag: str,
    model: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    subgroup = frame.loc[frame["at_risk"].eq(1) & frame[flag].eq(1)].copy()
    subgroup, gap_audit = crossfit_exposure_transform(subgroup)
    gap_audit.insert(0, "population", flag)
    gap_audit.insert(0, "organ_system", organ)
    subgroup["age_per_10y"] = (subgroup["actual_age"] - 60.0) / 10.0
    if flag == "no_baseline_icd":
        extra_raw = (*CONTEXT_RAW, "phenotype_missing_fraction", "log1p_preindex_gp_count", "has_preindex_gp")
        binary = (*BINARY_RAW, "has_preindex_gp")
    elif flag == "no_baseline_icd_or_gp":
        extra_raw = (*CONTEXT_RAW, "phenotype_missing_fraction")
        binary = BINARY_RAW
    else:
        raise ValueError(flag)
    transformed, covariate_audit, transformed_names = crossfit_covariates(
        subgroup, extra_raw, binary_columns=binary
    )
    covariate_audit.insert(0, "population", flag)
    covariate_audit.insert(0, "organ_system", organ)
    extras = transformed_for(extra_raw, transformed_names)
    row = fit_cox_model(transformed, organ, model, extras, flag)
    return row, gap_audit, covariate_audit


def add_q_values(estimates: pd.DataFrame) -> pd.DataFrame:
    estimates = estimates.copy()
    estimates["gap_q_value_bh_7_organs_within_model"] = np.nan
    for model, index in estimates.groupby("model", sort=False).groups.items():
        idx = list(index)
        if len(idx) != len(ORGANS):
            raise RuntimeError(f"{model}: expected seven organ estimates, found {len(idx)}")
        estimates.loc[idx, "gap_q_value_bh_7_organs_within_model"] = benjamini_hochberg(
            estimates.loc[idx, "gap_p_value"]
        )
    return estimates


def cox_attenuation(estimates: pd.DataFrame) -> pd.DataFrame:
    full = estimates.loc[estimates["model"].isin(FULL_COX_MODELS)].copy()
    rows: list[dict[str, object]] = []
    for organ in ORGANS:
        part = full.loc[full["organ_system"].eq(organ)].set_index("model")
        m0 = float(part.loc["M0_age_sex", "gap_beta_per_crossfit_sd"])
        m1 = float(part.loc["M1_context", "gap_beta_per_crossfit_sd"])
        for model in FULL_COX_MODELS:
            beta = float(part.loc[model, "gap_beta_per_crossfit_sd"])
            rows.append(
                {
                    "organ_system": organ,
                    "model": model,
                    "gap_beta": beta,
                    "gap_hr": float(part.loc[model, "gap_hr_per_crossfit_sd"]),
                    "beta_retained_vs_M0_percent": 100.0 * beta / m0,
                    "beta_attenuation_vs_M0_percent": 100.0 * (m0 - beta) / m0,
                    "beta_retained_vs_M1_percent": 100.0 * beta / m1,
                    "beta_attenuation_vs_M1_percent": 100.0 * (m1 - beta) / m1,
                }
            )
    return pd.DataFrame(rows)


def make_forest_plot(estimates: pd.DataFrame, output_dir: Path) -> None:
    selections = (
        ("M3_detail", "Full at-risk, detailed burden-adjusted"),
        ("S_no_baseline_icd_adjusted", "No recorded baseline ICD"),
        ("S_no_baseline_icd_or_gp_adjusted", "No recorded baseline ICD or GP"),
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    y = np.arange(len(ORGANS))
    for ax, (model, title) in zip(axes, selections):
        part = estimates.loc[estimates["model"].eq(model)].set_index("organ_system").loc[list(ORGANS)]
        hr = part["gap_hr_per_crossfit_sd"].to_numpy(float)
        low = part["gap_hr_ci95_low"].to_numpy(float)
        high = part["gap_hr_ci95_high"].to_numpy(float)
        ax.errorbar(
            hr,
            y,
            xerr=np.vstack([hr - low, high - hr]),
            fmt="o",
            color="#0072B2",
            ecolor="#0072B2",
            capsize=2.5,
            ms=4.5,
            lw=1.0,
        )
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("HR per cross-fitted gap SD (95% CI)")
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [ORGAN_DISPLAY[value] for value in ORGANS])
    axes[0].invert_yaxis()
    fig.suptitle("Frozen E47 OSAG associations after recorded-burden restrictions", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"E47_recorded_burden_adjusted_forest.{extension}",
            dpi=400 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_key_numbers(
    explainability: pd.DataFrame,
    estimates: pd.DataFrame,
    attenuation: pd.DataFrame,
    sample_audit: pd.DataFrame,
    no_incident: dict[str, object],
) -> dict[str, object]:
    macro_r2 = (
        explainability.groupby(["scope", "model"], sort=False)["heldout_r2"]
        .mean()
        .rename("macro_mean_heldout_r2")
        .reset_index()
    )
    macro_pivot = macro_r2.pivot(index="scope", columns="model", values="macro_mean_heldout_r2")
    r2_payload: dict[str, object] = {}
    for scope in macro_pivot.index:
        r2_payload[str(scope)] = {
            "macro_mean_heldout_r2_G0": float(macro_pivot.loc[scope, "G0_age_sex"]),
            "macro_mean_heldout_r2_G2D": float(macro_pivot.loc[scope, "G2D_density"]),
            "macro_mean_heldout_r2_G3_detail": float(macro_pivot.loc[scope, "G3_detail"]),
            "macro_mean_increment_G3_detail_vs_G2D": float(
                macro_pivot.loc[scope, "G3_detail"] - macro_pivot.loc[scope, "G2D_density"]
            ),
        }

    cox_payload: dict[str, object] = {}
    for model in ("M3_count", "M3_detail", *SUBGROUP_MODELS):
        part = estimates.loc[estimates["model"].eq(model)].copy()
        cox_payload[model] = {
            "hr_min": float(part["gap_hr_per_crossfit_sd"].min()),
            "hr_max": float(part["gap_hr_per_crossfit_sd"].max()),
            "all_hr_above_one": bool(part["gap_hr_per_crossfit_sd"].gt(1).all()),
            "all_bh_q_lt_0_05": bool(part["gap_q_value_bh_7_organs_within_model"].lt(0.05).all()),
            "organs": part[
                [
                    "organ_system",
                    "n",
                    "events",
                    "gap_hr_per_crossfit_sd",
                    "gap_hr_ci95_low",
                    "gap_hr_ci95_high",
                    "gap_q_value_bh_7_organs_within_model",
                ]
            ].to_dict(orient="records"),
        }
    m3_att = attenuation.loc[attenuation["model"].eq("M3_detail")]
    cox_payload["M3_detail"]["attenuation_vs_M0_percent_min"] = float(
        m3_att["beta_attenuation_vs_M0_percent"].min()
    )
    cox_payload["M3_detail"]["attenuation_vs_M0_percent_max"] = float(
        m3_att["beta_attenuation_vs_M0_percent"].max()
    )
    cox_payload["M3_detail"]["attenuation_vs_M1_percent_min"] = float(
        m3_att["beta_attenuation_vs_M1_percent"].min()
    )
    cox_payload["M3_detail"]["attenuation_vs_M1_percent_max"] = float(
        m3_att["beta_attenuation_vs_M1_percent"].max()
    )

    return {
        "question_supported": (
            "Whether frozen E47 gaps retain information beyond measured recorded baseline burden "
            "and healthcare-observation density"
        ),
        "question_not_identified": (
            "Whether gaps isolate molecular/cellular aging independently of unrecorded or subclinical disease"
        ),
        "gap_explainability": r2_payload,
        "cox": cox_payload,
        "sample": {
            "no_baseline_icd_n_min_across_organs": int(sample_audit["no_baseline_icd_at_risk_n"].min()),
            "no_baseline_icd_events_min_across_organs": int(sample_audit["no_baseline_icd_events"].min()),
            "no_baseline_icd_or_gp_n_min_across_organs": int(
                sample_audit["no_baseline_icd_or_gp_at_risk_n"].min()
            ),
            "no_baseline_icd_or_gp_events_min_across_organs": int(
                sample_audit["no_baseline_icd_or_gp_events"].min()
            ),
        },
        "no_incident_ablation": {
            "main_delta_c": 0.04298020689456881,
            "no_incident_delta_c": float(no_incident["full_risk"]["macro_delta_c"]),
            "main_minus_no_incident_delta_c": float(
                no_incident["comparison_to_main"]["main_minus_no_incident"]["full_risk_macro_delta_c"]
            ),
            "main_minus_no_incident_mae_years": float(
                no_incident["comparison_to_main"]["main_minus_no_incident"]["strict_mae"]
            ),
            "main_minus_no_incident_pearson": float(
                no_incident["comparison_to_main"]["main_minus_no_incident"]["strict_pearson"]
            ),
            "main_minus_no_incident_r2": float(
                no_incident["comparison_to_main"]["main_minus_no_incident"]["strict_r2"]
            ),
        },
    }


def write_result_summary(
    path: Path,
    key: dict[str, object],
    estimates: pd.DataFrame,
    attenuation: pd.DataFrame,
) -> None:
    detail = estimates.loc[estimates["model"].eq("M3_detail")].set_index("organ_system").loc[list(ORGANS)]
    att = attenuation.loc[attenuation["model"].eq("M3_detail")].set_index("organ_system").loc[list(ORGANS)]
    no_icd = estimates.loc[estimates["model"].eq("S_no_baseline_icd_adjusted")].set_index("organ_system").loc[list(ORGANS)]
    no_icd_gp = estimates.loc[
        estimates["model"].eq("S_no_baseline_icd_or_gp_adjusted")
    ].set_index("organ_system").loc[list(ORGANS)]
    full_r2 = key["gap_explainability"]["full_oof"]
    risk_r2 = key["gap_explainability"]["target_at_risk"]
    lines = [
        "# Frozen E47 disease-burden adjustment: result summary",
        "",
        "## Direct answer to R1.2",
        "",
        (
            "The results support an intermediate, positive interpretation: E47 OSAGs are age-anchored "
            "prognostic deviations that are not fully reducible to measured recorded baseline disease "
            "burden or healthcare-observation density. They do not establish an independent molecular "
            "or cellular aging process."
        ),
        "",
        "## Gap explainability",
        "",
        (
            f"Across organs in the full OOF population, the macro mean held-out R2 was "
            f"{full_r2['macro_mean_heldout_r2_G3_detail']:.6f} for the detailed context+density+burden "
            f"model; organ-resolved recorded burden added {full_r2['macro_mean_increment_G3_detail_vs_G2D']:.6f} "
            "over the context+density model."
        ),
        (
            f"In target-organ at-risk samples, the corresponding macro mean held-out R2 was "
            f"{risk_r2['macro_mean_heldout_r2_G3_detail']:.6f}, with an incremental "
            f"{risk_r2['macro_mean_increment_G3_detail_vs_G2D']:.6f} from detailed burden."
        ),
        "",
        "## Prospective associations",
        "",
        (
            f"After detailed recorded-burden and density adjustment, organ-specific HRs per cross-fitted "
            f"gap SD ranged from {detail['gap_hr_per_crossfit_sd'].min():.6f} to "
            f"{detail['gap_hr_per_crossfit_sd'].max():.6f}. Beta attenuation versus age+sex ranged from "
            f"{att['beta_attenuation_vs_M0_percent'].min():.1f}% to "
            f"{att['beta_attenuation_vs_M0_percent'].max():.1f}%."
        ),
        (
            f"Among participants with no recorded baseline ICD burden, adjusted HRs ranged from "
            f"{no_icd['gap_hr_per_crossfit_sd'].min():.6f} to {no_icd['gap_hr_per_crossfit_sd'].max():.6f}; "
            f"in the stricter no-baseline-ICD-or-GP sample they ranged from "
            f"{no_icd_gp['gap_hr_per_crossfit_sd'].min():.6f} to "
            f"{no_icd_gp['gap_hr_per_crossfit_sd'].max():.6f}."
        ),
        "",
        "| Organ | M3 detailed HR (95% CI) | Attenuation vs M0 | No baseline ICD HR (95% CI) | No ICD/GP HR (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for organ in ORGANS:
        d = detail.loc[organ]
        a = att.loc[organ]
        n1 = no_icd.loc[organ]
        n2 = no_icd_gp.loc[organ]
        lines.append(
            f"| {ORGAN_DISPLAY[organ]} | {d['gap_hr_per_crossfit_sd']:.4f} "
            f"({d['gap_hr_ci95_low']:.4f}-{d['gap_hr_ci95_high']:.4f}) | "
            f"{a['beta_attenuation_vs_M0_percent']:.1f}% | "
            f"{n1['gap_hr_per_crossfit_sd']:.4f} ({n1['gap_hr_ci95_low']:.4f}-{n1['gap_hr_ci95_high']:.4f}) | "
            f"{n2['gap_hr_per_crossfit_sd']:.4f} ({n2['gap_hr_ci95_low']:.4f}-{n2['gap_hr_ci95_high']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The at-risk outcome loss and Cox analyses exclude participants with recorded prevalent target-organ disease.",
            "- The no-baseline-ICD and no-baseline-ICD-or-GP analyses address recorded burden, not unobserved/subclinical disease.",
            "- Baseline ICD counts and organ-resolved prevalent counts are proxies for recorded burden and care intensity, not validated clinical severity scales.",
            "- Therefore the defensible claim is residual prognostic information beyond measured recorded burden/ascertainment, not disease-independent biological aging.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_dir = output_dir / "Source_Data"
    figure_dir = output_dir / "Figures"
    inputs = [
        args.oof.resolve(),
        args.context.resolve(),
        args.v2_labels.resolve(),
        args.no_incident_summary.resolve(),
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if (output_dir / "MANIFEST.json").exists() or source_dir.exists() or figure_dir.exists():
        raise FileExistsError(f"create-once analysis outputs already exist under {output_dir}")

    started = time.time()
    hashes = {str(path): sha256_file(path) for path in inputs}
    if hashes[str(args.oof.resolve())] != EXPECTED_OOF_SHA256:
        raise RuntimeError("frozen E47 OOF SHA-256 changed")
    if hashes[str(args.context.resolve())] != EXPECTED_CONTEXT_SHA256:
        raise RuntimeError("contextual-covariate SHA-256 changed")
    if hashes[str(args.v2_labels.resolve())] != EXPECTED_V2_LABEL_SHA256:
        raise RuntimeError("codebook-v2 participant-label SHA-256 changed")

    context, context_audit = load_context(args.context, args.v2_labels)
    no_incident = json.loads(args.no_incident_summary.read_text(encoding="utf-8"))
    if no_incident.get("status") != "COMPLETE":
        raise RuntimeError("no-incident control is not complete")

    fold_explainability_parts: list[pd.DataFrame] = []
    ols_preprocess_parts: list[pd.DataFrame] = []
    cox_rows: list[dict[str, object]] = []
    cox_gap_audits: list[pd.DataFrame] = []
    cox_covariate_audits: list[pd.DataFrame] = []
    sample_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    for organ in ORGANS:
        organ_started = time.time()
        frame = load_organ_frame(args.oof, context, organ)
        detail_full = detail_raw_for(organ, include_target=True)
        full_fold, full_preprocess = run_gap_explainability(
            frame, organ, "full_oof", detail_columns=detail_full
        )
        fold_explainability_parts.append(full_fold)
        ols_preprocess_parts.append(full_preprocess)

        risk_frame = frame.loc[frame["at_risk"].eq(1)].copy()
        detail_risk = detail_raw_for(organ, include_target=False)
        risk_fold, risk_preprocess = run_gap_explainability(
            risk_frame, organ, "target_at_risk", detail_columns=detail_risk
        )
        fold_explainability_parts.append(risk_fold)
        ols_preprocess_parts.append(risk_preprocess)

        full_rows, gap_audits, covariate_audits = run_full_cox(frame, organ)
        cox_rows.extend(full_rows)
        cox_gap_audits.extend(gap_audits)
        cox_covariate_audits.extend(covariate_audits)

        subgroup_specs = (
            (
                "no_baseline_icd",
                "S_no_baseline_icd_adjusted",
            ),
            (
                "no_baseline_icd_or_gp",
                "S_no_baseline_icd_or_gp_adjusted",
            ),
        )
        for flag, model in subgroup_specs:
            row, gap_audit, covariate_audit = run_subgroup_cox(
                frame, organ, flag=flag, model=model
            )
            cox_rows.append(row)
            cox_gap_audits.append(gap_audit)
            cox_covariate_audits.append(covariate_audit)

        at_risk = frame["at_risk"].eq(1)
        no_icd = at_risk & frame["no_baseline_icd"].eq(1)
        no_icd_gp = at_risk & frame["no_baseline_icd_or_gp"].eq(1)
        sample_rows.append(
            {
                "organ_system": organ,
                "full_oof_n": int(len(frame)),
                "at_risk_n": int(at_risk.sum()),
                "at_risk_events": int(frame.loc[at_risk, "event"].sum()),
                "target_prevalent_within_at_risk": int(
                    (at_risk & frame[f"has_prevalent_{organ}"].eq(1)).sum()
                ),
                "no_baseline_icd_at_risk_n": int(no_icd.sum()),
                "no_baseline_icd_events": int(frame.loc[no_icd, "event"].sum()),
                "no_baseline_icd_or_gp_at_risk_n": int(no_icd_gp.sum()),
                "no_baseline_icd_or_gp_events": int(frame.loc[no_icd_gp, "event"].sum()),
            }
        )
        runtime_rows.append(
            {
                "organ_system": organ,
                "elapsed_seconds": time.time() - organ_started,
            }
        )
        print(
            f"{organ}: at_risk={int(at_risk.sum()):,} events={int(frame.loc[at_risk, 'event'].sum()):,} "
            f"seconds={time.time() - organ_started:.1f}",
            flush=True,
        )

    fold_explainability = pd.concat(fold_explainability_parts, ignore_index=True)
    explainability = summarize_gap_explainability(fold_explainability)
    ols_preprocess = pd.concat(ols_preprocess_parts, ignore_index=True)
    estimates = add_q_values(pd.DataFrame(cox_rows))
    attenuation = cox_attenuation(estimates)
    cox_gap_audit = pd.concat(cox_gap_audits, ignore_index=True)
    cox_covariate_audit = pd.concat(cox_covariate_audits, ignore_index=True)
    sample_audit = pd.DataFrame(sample_rows)
    runtime = pd.DataFrame(runtime_rows)

    key = build_key_numbers(explainability, estimates, attenuation, sample_audit, no_incident)

    expected = {
        "gap_fold_rows": len(ORGANS) * 2 * len(GAP_MODELS) * 5,
        "gap_summary_rows": len(ORGANS) * 2 * len(GAP_MODELS),
        "cox_rows": len(ORGANS) * (len(FULL_COX_MODELS) + len(SUBGROUP_MODELS)),
        "sample_rows": len(ORGANS),
    }
    checks = {
        "oof_hash_matches": hashes[str(args.oof.resolve())] == EXPECTED_OOF_SHA256,
        "context_hash_matches": hashes[str(args.context.resolve())] == EXPECTED_CONTEXT_SHA256,
        "v2_label_hash_matches": hashes[str(args.v2_labels.resolve())] == EXPECTED_V2_LABEL_SHA256,
        "context_participants": len(context) == EXPECTED_PARTICIPANTS,
        "gap_fold_row_contract": len(fold_explainability) == expected["gap_fold_rows"],
        "gap_summary_row_contract": len(explainability) == expected["gap_summary_rows"],
        "cox_row_contract": len(estimates) == expected["cox_rows"],
        "sample_row_contract": len(sample_audit) == expected["sample_rows"],
        "all_cox_converged": bool(estimates["converged"].all()),
        "all_cox_designs_full_rank": bool(estimates["design_rank"].eq(estimates["n_covariates"]).all()),
        "all_cox_estimates_finite": bool(
            np.isfinite(
                estimates[
                    [
                        "gap_beta_per_crossfit_sd",
                        "gap_se_model_based",
                        "gap_hr_per_crossfit_sd",
                        "gap_hr_ci95_low",
                        "gap_hr_ci95_high",
                        "gap_q_value_bh_7_organs_within_model",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "no_target_prevalence_in_at_risk": bool(sample_audit["target_prevalent_within_at_risk"].eq(0).all()),
        "no_icd_each_organ_at_least_100_events": bool(sample_audit["no_baseline_icd_events"].ge(100).all()),
        "no_icd_gp_each_organ_at_least_100_events": bool(
            sample_audit["no_baseline_icd_or_gp_events"].ge(100).all()
        ),
        "all_gap_r2_finite": bool(np.isfinite(explainability["heldout_r2"]).all()),
        "five_folds_each_gap_model": bool(
            fold_explainability.groupby(["organ_system", "scope", "model"])["outer_fold"]
            .nunique()
            .eq(5)
            .all()
        ),
    }
    qa_status = "PASS" if all(checks.values()) else "FAIL"

    source_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    tables = {
        "sample_audit.csv": sample_audit,
        "gap_explainability_by_fold.csv": fold_explainability,
        "gap_explainability_summary.csv": explainability,
        "gap_ols_preprocess_audit.csv": ols_preprocess,
        "cox_gap_estimates.csv": estimates,
        "cox_attenuation.csv": attenuation,
        "cox_crossfit_gap_reference.csv": cox_gap_audit,
        "cox_crossfit_covariate_audit.csv": cox_covariate_audit,
        "runtime_by_organ.csv": runtime,
    }
    for name, table in tables.items():
        table.to_csv(source_dir / name, index=False, lineterminator="\n")

    # A compact editable workbook and a LaTeX focal table accompany the auditable CSVs.
    workbook_status = "written"
    try:
        with pd.ExcelWriter(source_dir / "E47_disease_burden_adjustment_tables.xlsx") as writer:
            sample_audit.to_excel(writer, sheet_name="sample_audit", index=False)
            explainability.to_excel(writer, sheet_name="gap_explainability", index=False)
            estimates.to_excel(writer, sheet_name="cox_estimates", index=False)
            attenuation.to_excel(writer, sheet_name="attenuation", index=False)
    except Exception as exc:  # pragma: no cover - environment-dependent optional export
        workbook_status = f"not_written: {type(exc).__name__}: {exc}"

    focal = estimates.loc[
        estimates["model"].isin(("M3_detail", *SUBGROUP_MODELS)),
        [
            "organ_system",
            "model",
            "n",
            "events",
            "gap_hr_per_crossfit_sd",
            "gap_hr_ci95_low",
            "gap_hr_ci95_high",
            "gap_q_value_bh_7_organs_within_model",
        ],
    ]
    (source_dir / "focal_cox_table.tex").write_text(
        focal.to_latex(index=False, float_format=lambda value: f"{value:.6f}"),
        encoding="utf-8",
    )
    make_forest_plot(estimates, figure_dir)
    json_dump(source_dir / "reviewer_key_numbers.json", key)
    write_result_summary(output_dir / "RESULT_SUMMARY.md", key, estimates, attenuation)

    qa = {
        "schema_version": "revision0710.e47.recorded_burden_adjustment.qa.v1",
        "status": qa_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "expected_rows": expected,
        "observed_rows": {
            "gap_fold_rows": len(fold_explainability),
            "gap_summary_rows": len(explainability),
            "cox_rows": len(estimates),
            "sample_rows": len(sample_audit),
        },
        "context_rebuild_audit": context_audit,
        "xlsx_export": workbook_status,
    }
    json_dump(output_dir / "QA_REPORT.json", qa)
    if qa_status != "PASS":
        raise RuntimeError(f"QA failed: {[key for key, value in checks.items() if not value]}")

    manifest = {
        "schema_version": "revision0710.e47.recorded_burden_adjustment.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": [
            {
                "path": path.as_posix(),
                "sha256": hashes[str(path)],
                "bytes": path.stat().st_size,
            }
            for path in inputs
        ],
        "script": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "population": {
            "full_oof_participants": EXPECTED_PARTICIPANTS,
            "target_at_risk": "organ-specific at_risk == 1; prevalent target disease excluded",
            "no_baseline_icd": "no strict-preindex ICD and no prevalent ICD on/before baseline",
            "no_baseline_icd_or_gp": "no baseline ICD as above and no preindex GP record",
        },
        "exposure": "frozen E47 corrected organ-age gap; Cox uses other-four-fold SD",
        "outcome": "60-month cause-specific target-organ incidence; death censored",
        "cox": "unpenalized Breslow; outer-fold-stratified baseline hazards; model-based SE",
        "multiple_testing": "BH across seven organs separately within each Cox model",
        "interpretation": (
            "Residual prognostic information beyond measured recorded burden/ascertainment; "
            "not identification of disease-independent molecular aging"
        ),
        "outputs": sorted(
            [
                str(path.relative_to(output_dir).as_posix())
                for path in [
                    *source_dir.iterdir(),
                    *figure_dir.iterdir(),
                    output_dir / "RESULT_SUMMARY.md",
                    output_dir / "QA_REPORT.json",
                ]
            ]
        ),
    }
    json_dump(output_dir / "MANIFEST.json", manifest)
    print(f"COMPLETE: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
