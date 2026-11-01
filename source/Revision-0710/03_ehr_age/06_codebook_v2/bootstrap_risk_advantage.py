from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REVISION_ROOT = WORKSPACE_ROOT / "Revision-0710"
DOWNSTREAM_SCRIPTS = REVISION_ROOT / "05_downstream" / "scripts"
DOWNSTREAM_COMMON = REVISION_ROOT / "05_downstream" / "common"
for path in (DOWNSTREAM_SCRIPTS, DOWNSTREAM_COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crossfit_metrics import (  # noqa: E402
    apply_linear_age_correction,
    apply_linear_age_correction_to_validation,
    fit_linear_age_correction,
)
from evaluate_method_5fold import (  # noqa: E402
    load_fold_roles,
    load_organ_outcomes,
    load_participant_masks,
    load_predictions,
    resolve_prediction_files,
    validate_cross_file_coverage,
)
from run_fold0_current import atomic_json, sha256_file, utc_now  # noqa: E402
from run_frozen_outer_fold import ORGAN_SYSTEMS, OUTCOMES_PATH, RUN_ROOT  # noqa: E402


HERE = Path(__file__).resolve().parent
FOLD0_ROOT = HERE / "01_runs" / "shared_private_independent_view_huber1_fold0"
BASELINE_REGISTRY = (
    REVISION_ROOT
    / "05_downstream"
    / "01_prediction_registries"
    / "tier_b"
    / "retain_organquery.json"
)
PARTICIPANT_MASKS = (
    REVISION_ROOT
    / "01_data_rebuild"
    / "07_corrected_primary_v1"
    / "02_outcomes_60m"
    / "participant_followup_and_evaluation_masks.parquet"
)
FOLD_ROLES = REVISION_ROOT / "01_data_rebuild" / "03_folds" / "outer_fold_roles.parquet"
OUTPUT_ROOT = HERE / "02_fivefold_frozen_huber1" / "03_paired_bootstrap"
N_RESAMPLES = 1000
SEED = 8172026


def _harrell_c(
    event: np.ndarray, duration: np.ndarray, log_partial_hazard: np.ndarray
) -> float:
    return float(concordance_index(duration, -log_partial_hazard, event))


def _main_paths(fold: int) -> tuple[Path, Path]:
    if fold == 0:
        return (
            FOLD0_ROOT / "fold1_calibration_predictions.parquet",
            FOLD0_ROOT / "fold0_predictions.parquet",
        )
    root = RUN_ROOT / f"outer_fold_{fold}"
    return root / "calibration_predictions.parquet", root / "outer_predictions.parquet"


def _fit_risk_scores(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    covariates: tuple[str, ...],
) -> np.ndarray:
    train = validation[["duration", "event"]].copy()
    score = pd.DataFrame(index=np.arange(len(test)))
    transformed = []
    for covariate in covariates:
        mean = float(validation[covariate].mean())
        scale = float(validation[covariate].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f"Cox validation covariate has no variation: {covariate}")
        name = f"z_{covariate}"
        train[name] = (validation[covariate].to_numpy(dtype=float) - mean) / scale
        score[name] = (test[covariate].to_numpy(dtype=float) - mean) / scale
        transformed.append(name)
    model = CoxPHFitter(penalizer=1e-4)
    model.fit(
        train[["duration", "event", *transformed]],
        duration_col="duration",
        event_col="event",
        show_progress=False,
    )
    return model.predict_log_partial_hazard(score[transformed]).to_numpy(dtype=np.float64)


def _prepare_baseline_frames() -> dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame]]:
    files, _ = resolve_prediction_files(
        method="retain_organquery",
        prediction_registry=BASELINE_REGISTRY,
        predictions=None,
    )
    predictions = load_predictions(files)
    participants = load_participant_masks(PARTICIPANT_MASKS)
    outcomes = load_organ_outcomes(OUTCOMES_PATH)
    if "strict_healthy_reference_60m" not in outcomes:
        raise RuntimeError("v2 outcomes do not contain the strict-Health mask")
    strict_by_participant = outcomes.groupby("participant_id", observed=True)[
        "strict_healthy_reference_60m"
    ].agg(["first", "nunique"])
    if strict_by_participant["nunique"].ne(1).any():
        raise RuntimeError("v2 strict-Health mask varies across organ rows")
    expected_strict = participants.set_index("participant_id")[
        "strict_healthy_reference_60m"
    ].sort_index()
    observed_strict = strict_by_participant["first"].sort_index().astype(np.int8)
    if not observed_strict.index.equals(expected_strict.index) or not np.array_equal(
        observed_strict.to_numpy(), expected_strict.to_numpy(dtype=np.int8)
    ):
        raise RuntimeError(
            "legacy demographic file and codebook-v2 strict-Health masks differ"
        )
    roles = load_fold_roles(FOLD_ROLES)
    enriched = validate_cross_file_coverage(predictions, participants, outcomes, roles)
    risk_outcomes = outcomes.loc[
        outcomes["at_risk"].eq(1),
        ["participant_id", "organ_system", "duration_months", "event"],
    ].rename(columns={"duration_months": "duration"})
    result: dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in range(5):
        validation = enriched.loc[
            enriched["outer_fold"].eq(fold)
            & enriched["split_role"].eq("validation")
        ].copy()
        test = enriched.loc[
            enriched["outer_fold"].eq(fold) & enriched["split_role"].eq("test")
        ].copy()
        correction = fit_linear_age_correction(validation)
        validation = apply_linear_age_correction_to_validation(validation, correction)
        test = apply_linear_age_correction(test, correction)
        for organ in ORGAN_SYSTEMS:
            organ_outcomes = risk_outcomes.loc[
                risk_outcomes["organ_system"].astype(str).eq(organ)
            ]
            columns = [
                "participant_id",
                "organ_system",
                "actual_age",
                "sex",
                "corrected_age_gap",
            ]
            validation_organ = validation.loc[
                validation["organ_system"].astype(str).eq(organ), columns
            ].merge(
                organ_outcomes,
                on=["participant_id", "organ_system"],
                how="inner",
                validate="one_to_one",
            )
            test_organ = test.loc[
                test["organ_system"].astype(str).eq(organ), columns
            ].merge(
                organ_outcomes,
                on=["participant_id", "organ_system"],
                how="inner",
                validate="one_to_one",
            )
            result[(fold, organ)] = (validation_organ, test_organ)
    return result


def _prepare_cells() -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    baseline = _prepare_baseline_frames()
    folds: dict[int, dict[str, Any]] = {}
    point_rows: list[dict[str, Any]] = []
    for fold in range(5):
        calibration_path, outer_path = _main_paths(fold)
        calibration = pd.read_parquet(calibration_path)
        outer = pd.read_parquet(outer_path)
        calibration["participant_id"] = calibration["participant_id"].astype(str)
        outer["participant_id"] = outer["participant_id"].astype(str)
        universe = np.asarray(sorted(outer["participant_id"].unique()), dtype=object)
        universe_index = {value: index for index, value in enumerate(universe)}
        cells: dict[str, Any] = {}
        for organ in ORGAN_SYSTEMS:
            main_validation = calibration.loc[
                calibration["organ_system"].astype(str).eq(organ)
                & calibration["at_risk"].eq(1)
            ].copy()
            main_test = outer.loc[
                outer["organ_system"].astype(str).eq(organ) & outer["at_risk"].eq(1)
            ].copy()
            baseline_validation, baseline_test = baseline[(fold, organ)]
            baseline_validation["participant_id"] = baseline_validation[
                "participant_id"
            ].astype(str)
            baseline_test["participant_id"] = baseline_test["participant_id"].astype(str)
            paired = main_test[
                ["participant_id", "duration", "event"]
            ].merge(
                baseline_test[
                    ["participant_id", "duration", "event", "corrected_age_gap"]
                ],
                on="participant_id",
                suffixes=("_ehr", "_retain"),
                validate="one_to_one",
            )
            if len(paired) != len(main_test):
                raise RuntimeError(f"risk test pairing failed: fold={fold} organ={organ}")
            if not np.allclose(paired["duration_ehr"], paired["duration_retain"]):
                raise RuntimeError("paired risk durations disagree")
            if not np.array_equal(paired["event_ehr"], paired["event_retain"]):
                raise RuntimeError("paired risk events disagree")
            main_test = main_test.set_index("participant_id").loc[
                paired["participant_id"]
            ].reset_index()
            baseline_test = baseline_test.set_index("participant_id").loc[
                paired["participant_id"]
            ].reset_index()
            base_score = _fit_risk_scores(
                main_validation, main_test, ("actual_age", "sex")
            )
            ehr_score = _fit_risk_scores(
                main_validation,
                main_test,
                ("actual_age", "sex", "corrected_age_gap"),
            )
            retain_score = _fit_risk_scores(
                baseline_validation,
                baseline_test,
                ("actual_age", "sex", "corrected_age_gap"),
            )
            duration = paired["duration_ehr"].to_numpy(dtype=np.float64)
            event = paired["event_ehr"].to_numpy(dtype=np.int8).astype(bool)
            c_base = _harrell_c(event, duration, base_score)
            c_ehr = _harrell_c(event, duration, ehr_score)
            c_retain = _harrell_c(event, duration, retain_score)
            positions = np.asarray(
                [universe_index[value] for value in paired["participant_id"]],
                dtype=np.int64,
            )
            cells[organ] = {
                "duration": duration,
                "event": event,
                "base_score": base_score,
                "ehr_score": ehr_score,
                "retain_score": retain_score,
                "universe_positions": positions,
            }
            point_rows.append(
                {
                    "outer_fold": fold,
                    "organ_system": organ,
                    "c_base": c_base,
                    "c_ehr_age": c_ehr,
                    "c_retain_organquery": c_retain,
                    "ehr_age_delta_c": c_ehr - c_base,
                    "retain_organquery_delta_c": c_retain - c_base,
                    "ehr_age_advantage": c_ehr - c_retain,
                }
            )
        folds[fold] = {"universe_n": len(universe), "cells": cells}
    return folds, pd.DataFrame(point_rows)


def _bootstrap_fold(
    fold: int, payload: Mapping[str, Any], n_resamples: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + fold * 100_003)
    rows: list[dict[str, Any]] = []
    universe_n = int(payload["universe_n"])
    for replicate in range(n_resamples):
        draws = rng.integers(0, universe_n, size=universe_n)
        counts = np.bincount(draws, minlength=universe_n)
        for organ, cell in payload["cells"].items():
            multiplicity = counts[cell["universe_positions"]]
            expanded = np.repeat(np.arange(len(multiplicity)), multiplicity)
            if len(expanded) < 3:
                continue
            event = cell["event"][expanded]
            if not event.any():
                continue
            duration = cell["duration"][expanded]
            c_base = _harrell_c(event, duration, cell["base_score"][expanded])
            c_ehr = _harrell_c(event, duration, cell["ehr_score"][expanded])
            c_retain = _harrell_c(
                event, duration, cell["retain_score"][expanded]
            )
            rows.append(
                {
                    "replicate": replicate,
                    "outer_fold": fold,
                    "organ_system": organ,
                    "ehr_age_delta_c": c_ehr - c_base,
                    "retain_organquery_delta_c": c_retain - c_base,
                    "ehr_age_advantage": c_ehr - c_retain,
                }
            )
    return rows


def run_bootstrap() -> Path:
    fold_payloads, point_cells = _prepare_cells()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_bootstrap_fold, fold, payload, N_RESAMPLES, SEED): fold
            for fold, payload in fold_payloads.items()
        }
        for future in as_completed(futures):
            fold = futures[future]
            fold_rows = future.result()
            rows.extend(fold_rows)
            print(f"risk bootstrap fold {fold} complete rows={len(fold_rows)}", flush=True)
    cell_replicates = pd.DataFrame(rows)
    expected = N_RESAMPLES * 5 * len(ORGAN_SYSTEMS)
    if len(cell_replicates) < 0.95 * expected:
        raise RuntimeError("fewer than 95% valid risk bootstrap cells")
    organ_replicates = cell_replicates.groupby(
        ["replicate", "organ_system"], as_index=False
    )[
        [
            "ehr_age_delta_c",
            "retain_organquery_delta_c",
            "ehr_age_advantage",
        ]
    ].mean()
    macro_replicates = organ_replicates.groupby("replicate", as_index=False)[
        [
            "ehr_age_delta_c",
            "retain_organquery_delta_c",
            "ehr_age_advantage",
        ]
    ].mean()
    point_organs = point_cells.groupby("organ_system", as_index=False)[
        [
            "ehr_age_delta_c",
            "retain_organquery_delta_c",
            "ehr_age_advantage",
        ]
    ].mean()
    point_macro = point_organs[
        [
            "ehr_age_delta_c",
            "retain_organquery_delta_c",
            "ehr_age_advantage",
        ]
    ].mean()
    summary_rows = []
    for metric in (
        "ehr_age_delta_c",
        "retain_organquery_delta_c",
        "ehr_age_advantage",
    ):
        values = macro_replicates[metric].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "metric": metric,
                "estimate": float(point_macro[metric]),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_probability_gt_0": float(np.mean(values > 0.0)),
                "valid_resamples": int(len(values)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    point_cells.to_csv(OUTPUT_ROOT / "risk_point_cells.csv", index=False)
    point_organs.to_csv(OUTPUT_ROOT / "risk_point_organs.csv", index=False)
    cell_replicates.to_csv(
        OUTPUT_ROOT / "risk_cell_replicates.csv.gz", index=False, compression="gzip"
    )
    macro_replicates.to_csv(OUTPUT_ROOT / "risk_macro_replicates.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "risk_paired_bootstrap_ci.csv", index=False)
    atomic_json(
        OUTPUT_ROOT / "risk_bootstrap_manifest.json",
        {
            "schema_version": "revision0710.ehr_age.codebook_v2.risk_bootstrap.v1",
            "status": "COMPLETE",
            "completed_utc": utc_now(),
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "bootstrap_unit": "participant within outer fold",
            "dependence": "same resampling counts shared across organs and paired models",
            "cox_models_refit_per_resample": False,
            "comparison": "EHR-AGE versus RETAIN-organquery, the best formal full-risk baseline",
            "inputs": {
                "participant_demographics_and_strict_mask": {
                    "path": str(PARTICIPANT_MASKS.resolve()),
                    "sha256": sha256_file(PARTICIPANT_MASKS),
                    "columns_used": [
                        "participant_id",
                        "chronological_age",
                        "sex",
                        "strict_healthy_reference_60m",
                    ],
                    "outcome_columns_used": False,
                    "note": (
                        "The strict-Health mask is row-identical to codebook v2. "
                        "All at-risk, event, and duration values are loaded from v2_outcomes."
                    ),
                },
                "baseline_registry": {
                    "path": str(BASELINE_REGISTRY.resolve()),
                    "sha256": sha256_file(BASELINE_REGISTRY),
                },
                "v2_outcomes": {
                    "path": str(OUTCOMES_PATH.resolve()),
                    "sha256": sha256_file(OUTCOMES_PATH),
                },
            },
        },
    )
    print(summary.to_string(index=False))
    return OUTPUT_ROOT


if __name__ == "__main__":
    print(run_bootstrap())
