#!/usr/bin/env python3
"""Baseline-only calibration-reference sensitivity for frozen EHR-AGE OOF predictions.

The alternative reference is selected only from baseline information using the
frozen baseline-free group contract (`has_preindex_icd == 0` and no codebook-v2
baseline disease in the seven included organ systems). Frozen validation-role raw predictions estimate the
age-gap correction, which is applied unchanged to the participant-disjoint
held-out prediction role. EHR-AGE checkpoints and raw predictions are not
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import spearmanr


ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
EXPECTED_PARTICIPANTS = 456_773
EXPECTED_ROWS = EXPECTED_PARTICIPANTS * len(ORGANS)


def workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir() and (parent / "Revision-0730").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
HERE = Path(__file__).resolve().parent
E47_FROZEN_ROOT = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/02_fivefold_frozen_huber1"
)
E47_FOLD0_RUN = (
    ROOT
    / "Revision-0710/03_ehr_age/06_codebook_v2/01_runs"
    / "shared_private_independent_view_huber1_fold0"
)
DEFAULT_CONTEXT = (
    ROOT
    / "Revision-0710/05_downstream/06_metric_v2/01_contextual_covariates"
    / "participant_context_covariates.parquet"
)
DEFAULT_LABELS = (
    ROOT
    / "Revision-0710/01_data_rebuild/09_v2_labels/icd_participant_labels_v2.parquet"
)
DEFAULT_CANONICAL = (
    E47_FROZEN_ROOT / "02_summary/oof_predictions.parquet"
)
E47_SCRIPT = (
    ROOT
    / "Revision-0710/Final_Fig_0716/Reviewer_Statistics"
    / "12_e47_disease_burden_adjustment/run_e47_disease_burden_adjustment.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    return parser.parse_args()


def e47_prediction_entries() -> dict[tuple[int, str], dict[str, object]]:
    entries: dict[tuple[int, str], dict[str, object]] = {
        (0, "validation"): {
            "outer_fold": 0,
            "split_role": "validation",
            "path": E47_FOLD0_RUN / "fold1_calibration_predictions.parquet",
        },
        (0, "test"): {
            "outer_fold": 0,
            "split_role": "test",
            "path": E47_FOLD0_RUN / "fold0_predictions.parquet",
        },
    }
    for fold in range(1, 5):
        fold_dir = E47_FROZEN_ROOT / "01_outer_folds" / f"outer_fold_{fold}"
        entries[(fold, "validation")] = {
            "outer_fold": fold,
            "split_role": "validation",
            "path": fold_dir / "calibration_predictions.parquet",
        }
        entries[(fold, "test")] = {
            "outer_fold": fold,
            "split_role": "test",
            "path": fold_dir / "outer_predictions.parquet",
        }
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_e47_module():
    spec = importlib.util.spec_from_file_location("e47_burden", E47_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen E47 disease-burden module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_prediction_contract(frame: pd.DataFrame, *, fold: int, role: str) -> None:
    required = {
        "participant_id",
        "organ_system",
        "actual_age",
        "predicted_age",
        "outer_fold",
        "split_role",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"fold {fold} {role}: missing columns {missing}")
    if frame.empty or not frame["outer_fold"].eq(fold).all():
        raise RuntimeError(f"fold {fold} {role}: outer-fold contract failed")
    if not frame["split_role"].astype(str).eq(role).all():
        raise RuntimeError(f"fold {fold} {role}: split-role contract failed")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError(f"fold {fold} {role}: duplicate participant-organ rows")
    if set(frame["organ_system"].astype(str)) != set(ORGANS):
        raise RuntimeError(f"fold {fold} {role}: organ set changed")
    counts = frame.groupby("participant_id", sort=False)["organ_system"].nunique()
    if not counts.eq(len(ORGANS)).all():
        raise RuntimeError(f"fold {fold} {role}: incomplete participant-organ coverage")
    for column in ("actual_age", "predicted_age"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"fold {fold} {role}: nonfinite {column}")


def fit_correction(validation: pd.DataFrame, fold: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reference = validation.loc[validation["baseline_only_reference"].eq(1)].copy()
    for organ in ORGANS:
        part = reference.loc[reference["organ_system"].eq(organ)]
        if len(part) < 3 or part["actual_age"].nunique() < 2:
            raise RuntimeError(f"fold {fold} {organ}: insufficient baseline-only calibration data")
        actual = part["actual_age"].to_numpy(float)
        raw_gap = part["predicted_age"].to_numpy(float) - actual
        design = np.column_stack([np.ones(len(part)), actual])
        rank = int(np.linalg.matrix_rank(design))
        if rank != 2:
            raise RuntimeError(f"fold {fold} {organ}: rank-deficient correction design")
        coefficient, *_ = np.linalg.lstsq(design, raw_gap, rcond=None)
        if not np.isfinite(coefficient).all():
            raise RuntimeError(f"fold {fold} {organ}: nonfinite correction coefficient")
        fitted = design @ coefficient
        rows.append(
            {
                "outer_fold": fold,
                "calibration_fold": int(part["fold_id"].iloc[0]),
                "organ_system": organ,
                "reference_definition": "frozen baseline-free: no pre-index ICD and no codebook-v2 baseline disease in seven included organs",
                "n_reference_participants": int(part["participant_id"].nunique()),
                "intercept_years": float(coefficient[0]),
                "age_slope_years_per_year": float(coefficient[1]),
                "reference_raw_gap_mean_years": float(raw_gap.mean()),
                "reference_residual_gap_mean_years": float((raw_gap - fitted).mean()),
                "reference_residual_gap_age_pearson": float(np.corrcoef(actual, raw_gap - fitted)[0, 1]),
            }
        )
    return rows


def apply_correction(test: pd.DataFrame, parameters: pd.DataFrame, fold: int) -> pd.DataFrame:
    mapping_i = parameters.set_index("organ_system")["intercept_years"]
    mapping_s = parameters.set_index("organ_system")["age_slope_years_per_year"]
    intercept = test["organ_system"].map(mapping_i).to_numpy(float)
    slope = test["organ_system"].map(mapping_s).to_numpy(float)
    if not np.isfinite(intercept).all() or not np.isfinite(slope).all():
        raise RuntimeError(f"fold {fold}: incomplete correction parameter mapping")
    actual = test["actual_age"].to_numpy(float)
    predicted = test["predicted_age"].to_numpy(float)
    raw_gap = predicted - actual
    expected = intercept + slope * actual
    output = test[["participant_id", "organ_system", "outer_fold", "actual_age"]].copy()
    output["predicted_age_raw"] = predicted
    output["raw_age_gap"] = raw_gap
    output["baseline_only_expected_age_gap"] = expected
    output["baseline_only_corrected_age_gap"] = raw_gap - expected
    output["baseline_only_corrected_predicted_age"] = actual + output[
        "baseline_only_corrected_age_gap"
    ].to_numpy(float)
    return output


def age_metrics(frame: pd.DataFrame, predicted_column: str) -> dict[str, float | int]:
    actual = frame["actual_age"].to_numpy(float)
    predicted = frame[predicted_column].to_numpy(float)
    residual = predicted - actual
    denominator = float(np.square(actual - actual.mean()).sum())
    return {
        "n_participant_organ_rows": int(len(frame)),
        "n_participants": int(frame["participant_id"].nunique()),
        "mae_years": float(np.abs(residual).mean()),
        "pearson_predicted_vs_chronological_age": float(np.corrcoef(actual, predicted)[0, 1]),
        "r_squared": float(1.0 - np.square(residual).sum() / denominator),
        "pearson_age_gap_vs_chronological_age": float(np.corrcoef(actual, residual)[0, 1]),
    }


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    populations = {
        "strict_health_reference_60m": frame["strict_healthy_reference_60m"].eq(1),
        "baseline_free_no_preindex_icd_or_included_organ_disease": frame["no_baseline_icd"].eq(1),
    }
    predictions = {
        "raw_predicted_age_primary_endpoint": "predicted_age_raw",
        "current_strict_reference_corrected_age_diagnostic": "corrected_predicted_age",
        "baseline_only_reference_corrected_age_diagnostic": "baseline_only_corrected_predicted_age",
    }
    for population, mask in populations.items():
        for organ in (*ORGANS, "macro_pooled"):
            organ_mask = mask if organ == "macro_pooled" else mask & frame["organ_system"].eq(organ)
            part = frame.loc[organ_mask]
            for metric_role, column in predictions.items():
                row = {
                    "population": population,
                    "organ_system": organ,
                    "metric_role": metric_role,
                    **age_metrics(part, column),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for organ in ORGANS:
        part = frame.loc[frame["organ_system"].eq(organ)]
        current = part["corrected_age_gap"].to_numpy(float)
        alternative = part["baseline_only_corrected_age_gap"].to_numpy(float)
        difference = alternative - current
        rows.append(
            {
                "organ_system": organ,
                "n": int(len(part)),
                "pearson_current_vs_baseline_only": float(np.corrcoef(current, alternative)[0, 1]),
                "spearman_current_vs_baseline_only": float(spearmanr(current, alternative).statistic),
                "mean_difference_baseline_only_minus_current_years": float(difference.mean()),
                "sd_difference_years": float(difference.std(ddof=1)),
                "rmse_difference_years": float(np.sqrt(np.mean(np.square(difference)))),
                "current_sd_years": float(current.std(ddof=1)),
                "baseline_only_sd_years": float(alternative.std(ddof=1)),
                "sd_ratio_baseline_only_over_current": float(
                    alternative.std(ddof=1) / current.std(ddof=1)
                ),
            }
        )
    return pd.DataFrame(rows)


def assign_group(frame: pd.DataFrame, organ: str) -> pd.Series:
    baseline_free = frame["no_baseline_icd"].eq(1) & frame["at_risk"].eq(1)
    target_disease = frame[f"has_prevalent_{organ}"].eq(1)
    included_other = (
        ~target_disease
        & frame["prevalent_organ_count"].ge(1)
    )
    other_recorded = (
        frame["has_preindex_icd"].eq(1)
        & frame["prevalent_organ_count"].eq(0)
    )
    group = np.select(
        [target_disease, included_other, other_recorded, baseline_free],
        [
            "target_organ_disease",
            "target_organ_free_other_included_organ_disease",
            "other_recorded_disease_outside_seven_organs",
            "baseline_free",
        ],
        default="unclassified",
    )
    result = pd.Series(group, index=frame.index, dtype="string")
    return result


def summarize_groups(frame: pd.DataFrame, organ: str) -> pd.DataFrame:
    work = frame.copy()
    work["clinical_group"] = assign_group(work, organ)
    excluded_n = int(work["clinical_group"].eq("unclassified").sum())
    work = work.loc[~work["clinical_group"].eq("unclassified")].copy()
    rows: list[dict[str, object]] = []
    for group, part in work.groupby("clinical_group", sort=False):
        for definition, column in (
            ("current_strict_health_reference_correction", "corrected_age_gap"),
            ("baseline_only_reference_correction", "baseline_only_corrected_age_gap"),
        ):
            values = part[column].to_numpy(float)
            se = values.std(ddof=1) / math.sqrt(len(values))
            rows.append(
                {
                    "organ_system": organ,
                    "clinical_group": str(group),
                    "correction_definition": definition,
                    "n": int(len(part)),
                    "participants_outside_four_display_groups": excluded_n,
                    "mean_OSAG_years": float(values.mean()),
                    "mean_ci95_low": float(values.mean() - 1.959963984540054 * se),
                    "mean_ci95_high": float(values.mean() + 1.959963984540054 * se),
                    "sd_OSAG_years": float(values.std(ddof=1)),
                    "median_OSAG_years": float(np.median(values)),
                    "q1_OSAG_years": float(np.quantile(values, 0.25)),
                    "q3_OSAG_years": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def foldwise_c_index(frame: pd.DataFrame, design: np.ndarray, beta: np.ndarray) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for fold in range(5):
        mask = frame["outer_fold"].eq(fold).to_numpy()
        part = frame.loc[mask]
        value = float(
            concordance_index(
                part["duration_months"].to_numpy(float),
                -(design[mask] @ beta),
                part["event"].to_numpy(np.int8),
            )
        )
        rows.append(
            {
                "outer_fold": fold,
                "n": int(mask.sum()),
                "events": int(part["event"].sum()),
                "harrell_c": value,
            }
        )
    values = pd.DataFrame(rows)
    pooled = float(np.average(values["harrell_c"], weights=values["events"]))
    return pooled, rows


def fit_detailed_cox(
    e47,
    frame: pd.DataFrame,
    organ: str,
    gap_column: str,
    correction_definition: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    at_risk = frame.loc[frame["at_risk"].eq(1)].copy()
    at_risk["corrected_age_gap"] = at_risk[gap_column].to_numpy(float)
    at_risk, gap_audit = e47.crossfit_exposure_transform(at_risk)
    at_risk["age_per_10y"] = (at_risk["actual_age"] - 60.0) / 10.0
    detail_raw = e47.detail_raw_for(organ, include_target=False)
    raw_all = (*e47.CONTEXT_RAW, *e47.DENSITY_RAW, *e47.BURDEN_COUNT_RAW, *detail_raw)
    transformed, covariate_audit, transformed_names = e47.crossfit_covariates(
        at_risk, raw_all, binary_columns=e47.BINARY_RAW
    )
    context_names = e47.transformed_for(e47.CONTEXT_RAW, transformed_names)
    density_names = e47.transformed_for(e47.DENSITY_RAW, transformed_names)
    detail_names = e47.transformed_for(detail_raw, transformed_names)
    columns = ["age_per_10y", "sex", "gap_z_crossfit", *context_names, *density_names, *detail_names]
    design = transformed[columns].to_numpy(float)
    duration = transformed["duration_months"].to_numpy(float)
    event = transformed["event"].to_numpy(np.int8)
    strata = transformed["outer_fold"].to_numpy(np.int8)
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design) != design.shape[1]:
        raise RuntimeError(f"{organ} {correction_definition}: invalid Cox design")
    fitted = e47.fit_cox_breslow(design, duration, event, strata=strata)
    if not fitted.converged:
        raise RuntimeError(f"{organ} {correction_definition}: Cox did not converge")
    gap_index = columns.index("gap_z_crossfit")
    beta = float(fitted.beta[gap_index])
    se = float(fitted.standard_error[gap_index])
    c_index, c_rows = foldwise_c_index(transformed, design, fitted.beta)
    z95 = 1.959963984540054
    result = {
        "organ_system": organ,
        "correction_definition": correction_definition,
        "model": "M3_detail_exact_frozen_covariate_family",
        "n": int(len(transformed)),
        "events": int(event.sum()),
        "n_covariates": int(len(columns)),
        "gap_beta_per_crossfit_sd": beta,
        "gap_se_model_based": se,
        "gap_hr_per_crossfit_sd": float(np.exp(beta)),
        "gap_hr_ci95_low": float(np.exp(beta - z95 * se)),
        "gap_hr_ci95_high": float(np.exp(beta + z95 * se)),
        "gap_p_value": float(fitted.p_value[gap_index]),
        "event_weighted_foldwise_harrell_c": c_index,
        "converged": bool(fitted.converged),
        "iterations": int(fitted.iterations),
        "ties": fitted.ties,
        "outer_fold_baseline_hazard": "stratified",
    }
    gap_audit.insert(0, "correction_definition", correction_definition)
    gap_audit.insert(0, "organ_system", organ)
    covariate_audit.insert(0, "correction_definition", correction_definition)
    covariate_audit.insert(0, "organ_system", organ)
    c_frame = pd.DataFrame(c_rows)
    c_frame.insert(0, "correction_definition", correction_definition)
    c_frame.insert(0, "organ_system", organ)
    return result, gap_audit, covariate_audit, c_frame


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_output = args.output_dir / "baseline_only_oof_by_fold"
    fold_output.mkdir(parents=True, exist_ok=True)

    by_key = e47_prediction_entries()
    if set(by_key) != {(fold, role) for fold in range(5) for role in ("validation", "test")}:
        raise RuntimeError("prediction registry does not contain exactly five validation/test pairs")

    e47 = load_e47_module()
    full_context, burden_audit = e47.load_context(args.context, args.labels)
    reference_context = full_context[
        ["participant_id", "fold_id", "chronological_age", "no_baseline_icd"]
    ].copy()

    correction_rows: list[dict[str, object]] = []
    registry_audit: list[dict[str, object]] = []
    test_outputs: list[Path] = []
    for fold in range(5):
        validation_entry = by_key[(fold, "validation")]
        test_entry = by_key[(fold, "test")]
        validation_path = Path(validation_entry["path"])
        test_path = Path(test_entry["path"])
        for role, entry, path in (
            ("validation", validation_entry, validation_path),
            ("test", test_entry, test_path),
        ):
            observed_hash = sha256_file(path)
            registry_audit.append(
                {
                    "outer_fold": fold,
                    "split_role": role,
                    "path": str(path),
                    "sha256_observed": observed_hash,
                }
            )

        validation = pd.read_parquet(validation_path)
        validation["participant_id"] = validation["participant_id"].astype(str)
        validation["outer_fold"] = fold
        validation["split_role"] = "validation"
        require_prediction_contract(validation, fold=fold, role="validation")
        validation = validation.merge(
            reference_context,
            on="participant_id",
            how="inner",
            validate="many_to_one",
        )
        if not np.allclose(validation["actual_age"], validation["chronological_age"], atol=1e-5, rtol=0):
            raise RuntimeError(f"fold {fold} validation: chronological-age mismatch")
        calibration_folds = sorted(pd.unique(validation["fold_id"]).tolist())
        if len(calibration_folds) != 1 or calibration_folds[0] == fold:
            raise RuntimeError(f"fold {fold} validation: calibration fold is not participant-disjoint")
        validation["baseline_only_reference"] = validation["no_baseline_icd"].astype(np.int8)
        fold_parameters = pd.DataFrame(fit_correction(validation, fold))
        correction_rows.extend(fold_parameters.to_dict("records"))
        validation_ids = frozenset(validation["participant_id"])

        test = pd.read_parquet(test_path)
        test["participant_id"] = test["participant_id"].astype(str)
        test["outer_fold"] = fold
        test["split_role"] = "test"
        require_prediction_contract(test, fold=fold, role="test")
        overlap = validation_ids.intersection(test["participant_id"])
        if overlap:
            raise RuntimeError(f"fold {fold}: validation/test participant overlap ({len(overlap)})")
        test = test.merge(
            reference_context[["participant_id", "fold_id"]],
            on="participant_id",
            how="inner",
            validate="many_to_one",
        )
        if not test["fold_id"].eq(fold).all():
            raise RuntimeError(f"fold {fold} test: fold assignment mismatch")
        output = apply_correction(test, fold_parameters, fold)
        output_path = fold_output / f"outer_fold_{fold}.parquet"
        output.to_parquet(output_path, index=False, compression="zstd")
        test_outputs.append(output_path)
        del validation, test, output

    parameters = pd.DataFrame(correction_rows).sort_values(["outer_fold", "organ_system"])
    parameters.to_csv(args.output_dir / "baseline_only_correction_parameters.csv", index=False)
    pd.DataFrame(registry_audit).to_csv(args.output_dir / "prediction_registry_audit.csv", index=False)

    alternative = pd.concat([pd.read_parquet(path) for path in test_outputs], ignore_index=True)
    if len(alternative) != EXPECTED_ROWS or alternative.duplicated(["participant_id", "organ_system"]).any():
        raise RuntimeError("alternative OOF prediction coverage failed")
    if alternative["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise RuntimeError("alternative OOF participant count failed")

    analysis_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    cox_rows: list[dict[str, object]] = []
    gap_audits: list[pd.DataFrame] = []
    covariate_audits: list[pd.DataFrame] = []
    c_index_frames: list[pd.DataFrame] = []

    for organ in ORGANS:
        canonical = pd.read_parquet(
            args.canonical,
            columns=[
                "participant_id",
                "organ_system",
                "outer_fold",
                "actual_age",
                "sex",
                "strict_healthy_reference_60m",
                "predicted_age",
                "raw_age_gap",
                "expected_age_gap",
                "corrected_age_gap",
                "corrected_predicted_age",
                "at_risk",
                "duration",
                "event",
            ],
            filters=[("organ_system", "==", organ)],
        )
        canonical["participant_id"] = canonical["participant_id"].astype(str)
        canonical = canonical.rename(
            columns={"predicted_age": "predicted_age_raw", "duration": "duration_months"}
        )
        if len(canonical) != EXPECTED_PARTICIPANTS or canonical["participant_id"].duplicated().any():
            raise RuntimeError(f"{organ}: canonical held-out coverage failed")
        alt = alternative.loc[alternative["organ_system"].eq(organ)].copy()
        frame = canonical.merge(
            alt.drop(columns=["organ_system", "actual_age"]),
            on=["participant_id", "outer_fold"],
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != EXPECTED_PARTICIPANTS:
            raise RuntimeError(f"{organ}: canonical/registry held-out merge incomplete")
        if not np.allclose(frame["predicted_age_raw_x"], frame["predicted_age_raw_y"], atol=1e-6, rtol=0):
            raise RuntimeError(f"{organ}: registry/canonical raw prediction mismatch")
        frame = frame.rename(columns={"predicted_age_raw_x": "predicted_age_raw"}).drop(
            columns=["predicted_age_raw_y"]
        )
        frame = frame.merge(full_context, on="participant_id", how="inner", validate="one_to_one")
        if len(frame) != EXPECTED_PARTICIPANTS or not frame["fold_id"].eq(frame["outer_fold"]).all():
            raise RuntimeError(f"{organ}: context merge/fold contract failed")
        analysis_frames.append(
            frame[
                [
                    "participant_id",
                    "organ_system",
                    "outer_fold",
                    "actual_age",
                    "strict_healthy_reference_60m",
                    "has_preindex_icd",
                    "no_baseline_icd",
                    "predicted_age_raw",
                    "corrected_age_gap",
                    "corrected_predicted_age",
                    "baseline_only_corrected_age_gap",
                    "baseline_only_corrected_predicted_age",
                ]
            ].copy()
        )
        group_frames.append(summarize_groups(frame, organ))
        at_risk = frame["at_risk"].eq(1)
        if frame.loc[at_risk, "duration_months"].isna().any() or frame.loc[at_risk, "duration_months"].le(0).any():
            raise RuntimeError(f"{organ}: invalid E47 at-risk duration")
        if (at_risk & frame[f"has_prevalent_{organ}"].eq(1)).any():
            raise RuntimeError(f"{organ}: prevalent target remains in E47 at-risk population")
        for definition, column in (
            ("current_strict_health_reference_correction", "corrected_age_gap"),
            ("baseline_only_reference_correction", "baseline_only_corrected_age_gap"),
        ):
            result, gap_audit, covariate_audit, c_frame = fit_detailed_cox(
                e47, frame, organ, column, definition
            )
            cox_rows.append(result)
            gap_audits.append(gap_audit)
            covariate_audits.append(covariate_audit)
            c_index_frames.append(c_frame)
        del canonical, alt, frame

    analysis = pd.concat(analysis_frames, ignore_index=True)
    if len(analysis) != EXPECTED_ROWS:
        raise RuntimeError("final analysis row count failed")
    metrics = summarize_metrics(analysis)
    agreement = summarize_agreement(analysis)
    groups = pd.concat(group_frames, ignore_index=True)
    cox = pd.DataFrame(cox_rows)
    cox["gap_q_value_bh_7_organs_within_correction"] = np.nan
    for _, index in cox.groupby("correction_definition", sort=False).groups.items():
        idx = list(index)
        cox.loc[idx, "gap_q_value_bh_7_organs_within_correction"] = e47.benjamini_hochberg(
            cox.loc[idx, "gap_p_value"]
        )

    metrics.to_csv(args.output_dir / "age_and_age_gap_metrics.csv", index=False)
    agreement.to_csv(args.output_dir / "current_vs_baseline_only_OSAG_agreement.csv", index=False)
    groups.to_csv(args.output_dir / "four_group_OSAG_summary.csv", index=False)
    cox.to_csv(args.output_dir / "baseline_only_reference_detailed_burden_cox.csv", index=False)
    pd.concat(gap_audits, ignore_index=True).to_csv(
        args.output_dir / "cox_gap_transform_audit.csv", index=False
    )
    pd.concat(covariate_audits, ignore_index=True).to_csv(
        args.output_dir / "cox_covariate_transform_audit.csv", index=False
    )
    pd.concat(c_index_frames, ignore_index=True).to_csv(
        args.output_dir / "cox_foldwise_harrell_c.csv", index=False
    )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "baseline-only calibration-reference sensitivity",
        "baseline_only_reference": "has_preindex_icd == 0 and codebook-v2 prevalent_organ_count == 0; no future follow-up requirement",
        "future_outcomes_used_for_reference_selection": False,
        "model_retrained": False,
        "workspace_root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "input_sha256": {
            "e47_final_selection": sha256_file(E47_FROZEN_ROOT / "FINAL_SELECTION.json"),
            "context": sha256_file(args.context),
            "labels": sha256_file(args.labels),
            "e47_oof_predictions": sha256_file(args.canonical),
        },
        "registry_prediction_files": registry_audit,
        "burden_context_audit": burden_audit,
        "output_sha256": {},
    }
    for path in sorted(args.output_dir.glob("*.csv")):
        provenance["output_sha256"][path.name] = sha256_file(path)
    for path in sorted(fold_output.glob("*.parquet")):
        provenance["output_sha256"][str(path.relative_to(args.output_dir))] = sha256_file(path)
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(args.output_dir),
        "parameters": len(parameters),
        "analysis_rows": len(analysis),
        "cox_models": len(cox),
        "provenance_sha256": sha256_file(provenance_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
