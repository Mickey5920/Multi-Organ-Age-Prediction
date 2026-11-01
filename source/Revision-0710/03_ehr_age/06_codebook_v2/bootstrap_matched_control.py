from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bootstrap_age_advantage import _participant_contributions, _weighted_metrics
from bootstrap_risk_advantage import _fit_risk_scores, _harrell_c
from run_fold0_current import atomic_json, sha256_file, utc_now
from run_frozen_outer_fold import ORGAN_SYSTEMS


HERE = Path(__file__).resolve().parent
MAIN_ROOT = HERE / "02_fivefold_frozen_huber1"
MATCHED_ROOT = HERE / "04_matched_retain_joint"
MAIN_OOF = MAIN_ROOT / "02_summary/oof_predictions.parquet"
MATCHED_OOF = MATCHED_ROOT / "02_summary/oof_predictions.parquet"
OUTPUT_ROOT = MATCHED_ROOT / "03_paired_bootstrap"
N_RESAMPLES = 1000
AGE_SEED = 9172026
RISK_SEED = 10172026


def prediction_paths(model: str, fold: int) -> tuple[Path, Path]:
    if model == "ehr_age":
        if fold == 0:
            root = HERE / "01_runs/shared_private_independent_view_huber1_fold0"
            return (
                root / "fold1_calibration_predictions.parquet",
                root / "fold0_predictions.parquet",
            )
        root = MAIN_ROOT / f"01_outer_folds/outer_fold_{fold}"
    elif model == "matched_retain":
        if fold == 0:
            root = HERE / "01_runs/current_architecture_fold0"
            return (
                root / "fold1_calibration_predictions.parquet",
                root / "fold0_predictions.parquet",
            )
        root = MATCHED_ROOT / f"01_outer_folds/outer_fold_{fold}"
    else:
        raise ValueError(model)
    return root / "calibration_predictions.parquet", root / "outer_predictions.parquet"


def age_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "participant_id",
        "organ_system",
        "outer_fold",
        "actual_age",
        "predicted_age",
        "strict_healthy_reference_60m",
    ]
    main = pd.read_parquet(MAIN_OOF, columns=columns)
    matched = pd.read_parquet(MATCHED_OOF, columns=columns)
    for frame in (main, matched):
        frame["participant_id"] = frame["participant_id"].astype(str)
        frame["organ_system"] = frame["organ_system"].astype(str)
    main = main.loc[main["strict_healthy_reference_60m"].eq(1)].copy()
    matched = matched.loc[matched["strict_healthy_reference_60m"].eq(1)].copy()
    paired = main.merge(
        matched,
        on=["participant_id", "organ_system", "outer_fold"],
        how="inner",
        suffixes=("_ehr_age", "_matched_retain"),
        validate="one_to_one",
    )
    if len(paired) != len(main) or len(paired) != len(matched):
        raise RuntimeError("matched-control strict OOF rows are not exactly paired")
    if not np.allclose(paired["actual_age_ehr_age"], paired["actual_age_matched_retain"]):
        raise RuntimeError("matched-control chronological ages disagree")
    order = {organ: index for index, organ in enumerate(ORGAN_SYSTEMS)}
    paired["_organ_order"] = paired["organ_system"].map(order)
    paired = paired.sort_values(
        ["outer_fold", "participant_id", "_organ_order"], kind="mergesort"
    )
    participants = paired.drop_duplicates("participant_id")[["participant_id", "outer_fold"]]
    n = len(participants)
    if len(paired) != n * len(ORGAN_SYSTEMS):
        raise RuntimeError("matched-control age bootstrap lost seven-organ clusters")
    target = paired["actual_age_ehr_age"].to_numpy(dtype=np.float64).reshape(n, -1)
    predictions = {
        "ehr_age": paired["predicted_age_ehr_age"].to_numpy(dtype=np.float64).reshape(n, -1),
        "matched_retain": paired["predicted_age_matched_retain"].to_numpy(dtype=np.float64).reshape(n, -1),
    }
    target_sums = {"target": target.sum(axis=1), "target2": np.square(target).sum(axis=1)}
    contributions = {
        model: _participant_contributions(target, prediction)
        for model, prediction in predictions.items()
    }
    fold_values = participants["outer_fold"].to_numpy(dtype=np.int64)
    fold_positions = {fold: np.flatnonzero(fold_values == fold) for fold in range(5)}
    row_n = n * len(ORGAN_SYSTEMS)
    point_counts = np.ones(n, dtype=np.float64)
    point = {
        model: _weighted_metrics(point_counts, target_sums, values, row_n)
        for model, values in contributions.items()
    }
    rng = np.random.default_rng(AGE_SEED)
    replicate_rows: list[dict[str, Any]] = []
    for replicate in range(N_RESAMPLES):
        counts = np.zeros(n, dtype=np.float64)
        for positions in fold_positions.values():
            draws = rng.choice(positions, size=len(positions), replace=True)
            counts += np.bincount(draws, minlength=n)
        metrics = {
            model: _weighted_metrics(counts, target_sums, values, row_n)
            for model, values in contributions.items()
        }
        for metric, direction in (("mae", "lower"), ("pearson", "higher"), ("r2", "higher")):
            raw = metrics["ehr_age"][metric] - metrics["matched_retain"][metric]
            improvement = -raw if direction == "lower" else raw
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "direction": direction,
                    "ehr_age": metrics["ehr_age"][metric],
                    "matched_retain": metrics["matched_retain"][metric],
                    "ehr_age_minus_matched_retain": raw,
                    "directional_improvement": improvement,
                }
            )
    replicates = pd.DataFrame(replicate_rows)
    summary_rows = []
    for metric, direction in (("mae", "lower"), ("pearson", "higher"), ("r2", "higher")):
        values = replicates.loc[
            replicates["metric"].eq(metric), "directional_improvement"
        ].to_numpy(dtype=np.float64)
        raw = point["ehr_age"][metric] - point["matched_retain"][metric]
        improvement = -raw if direction == "lower" else raw
        summary_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "ehr_age": point["ehr_age"][metric],
                "matched_retain": point["matched_retain"][metric],
                "ehr_age_minus_matched_retain": raw,
                "directional_improvement": improvement,
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_probability_improvement_gt_0": float(np.mean(values > 0.0)),
            }
        )
    return pd.DataFrame(summary_rows), replicates


def prepare_risk_cells() -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    folds: dict[int, dict[str, Any]] = {}
    point_rows: list[dict[str, Any]] = []
    for fold in range(5):
        main_cal_path, main_test_path = prediction_paths("ehr_age", fold)
        matched_cal_path, matched_test_path = prediction_paths("matched_retain", fold)
        main_cal = pd.read_parquet(main_cal_path)
        main_test = pd.read_parquet(main_test_path)
        matched_cal = pd.read_parquet(matched_cal_path)
        matched_test = pd.read_parquet(matched_test_path)
        for frame in (main_cal, main_test, matched_cal, matched_test):
            frame["participant_id"] = frame["participant_id"].astype(str)
            frame["organ_system"] = frame["organ_system"].astype(str)
        universe = np.asarray(sorted(main_test["participant_id"].unique()), dtype=object)
        universe_index = {value: index for index, value in enumerate(universe)}
        cells: dict[str, Any] = {}
        for organ in ORGAN_SYSTEMS:
            selected = []
            for frame in (main_cal, main_test, matched_cal, matched_test):
                selected.append(
                    frame.loc[
                        frame["organ_system"].eq(organ) & frame["at_risk"].eq(1)
                    ].copy()
                )
            main_validation, main_outer, matched_validation, matched_outer = selected
            paired = main_outer[["participant_id", "duration", "event"]].merge(
                matched_outer[["participant_id", "duration", "event"]],
                on="participant_id",
                suffixes=("_ehr", "_matched"),
                validate="one_to_one",
            )
            if len(paired) != len(main_outer) or len(paired) != len(matched_outer):
                raise RuntimeError(f"matched risk pairing failed: fold={fold} organ={organ}")
            if not np.allclose(paired["duration_ehr"], paired["duration_matched"]):
                raise RuntimeError("matched risk durations disagree")
            if not np.array_equal(paired["event_ehr"], paired["event_matched"]):
                raise RuntimeError("matched risk events disagree")
            ids = paired["participant_id"].tolist()
            main_outer = main_outer.set_index("participant_id").loc[ids].reset_index()
            matched_outer = matched_outer.set_index("participant_id").loc[ids].reset_index()
            base_score = _fit_risk_scores(main_validation, main_outer, ("actual_age", "sex"))
            ehr_score = _fit_risk_scores(
                main_validation, main_outer, ("actual_age", "sex", "corrected_age_gap")
            )
            matched_score = _fit_risk_scores(
                matched_validation,
                matched_outer,
                ("actual_age", "sex", "corrected_age_gap"),
            )
            duration = paired["duration_ehr"].to_numpy(dtype=np.float64)
            event = paired["event_ehr"].to_numpy(dtype=np.int8).astype(bool)
            c_base = _harrell_c(event, duration, base_score)
            c_ehr = _harrell_c(event, duration, ehr_score)
            c_matched = _harrell_c(event, duration, matched_score)
            cells[organ] = {
                "duration": duration,
                "event": event,
                "base_score": base_score,
                "ehr_score": ehr_score,
                "matched_score": matched_score,
                "universe_positions": np.asarray(
                    [universe_index[value] for value in ids], dtype=np.int64
                ),
            }
            point_rows.append(
                {
                    "outer_fold": fold,
                    "organ_system": organ,
                    "ehr_age_delta_c": c_ehr - c_base,
                    "matched_retain_delta_c": c_matched - c_base,
                    "ehr_age_advantage": c_ehr - c_matched,
                }
            )
        folds[fold] = {"universe_n": len(universe), "cells": cells}
    return folds, pd.DataFrame(point_rows)


def bootstrap_risk_fold(
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
            c_matched = _harrell_c(event, duration, cell["matched_score"][expanded])
            rows.append(
                {
                    "replicate": replicate,
                    "outer_fold": fold,
                    "organ_system": organ,
                    "ehr_age_delta_c": c_ehr - c_base,
                    "matched_retain_delta_c": c_matched - c_base,
                    "ehr_age_advantage": c_ehr - c_matched,
                }
            )
    return rows


def risk_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_payloads, point_cells = prepare_risk_cells()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(bootstrap_risk_fold, fold, payload, N_RESAMPLES, RISK_SEED): fold
            for fold, payload in fold_payloads.items()
        }
        for future in as_completed(futures):
            fold = futures[future]
            fold_rows = future.result()
            rows.extend(fold_rows)
            print(f"matched risk bootstrap fold {fold} complete rows={len(fold_rows)}", flush=True)
    cells = pd.DataFrame(rows)
    expected = N_RESAMPLES * 5 * len(ORGAN_SYSTEMS)
    if len(cells) < 0.95 * expected:
        raise RuntimeError("fewer than 95% valid matched risk bootstrap cells")
    organ_replicates = cells.groupby(["replicate", "organ_system"], as_index=False)[
        ["ehr_age_delta_c", "matched_retain_delta_c", "ehr_age_advantage"]
    ].mean()
    macro_replicates = organ_replicates.groupby("replicate", as_index=False)[
        ["ehr_age_delta_c", "matched_retain_delta_c", "ehr_age_advantage"]
    ].mean()
    point = point_cells.groupby("organ_system", as_index=False)[
        ["ehr_age_delta_c", "matched_retain_delta_c", "ehr_age_advantage"]
    ].mean()
    point_macro = point[
        ["ehr_age_delta_c", "matched_retain_delta_c", "ehr_age_advantage"]
    ].mean()
    summary_rows = []
    for metric in ("ehr_age_delta_c", "matched_retain_delta_c", "ehr_age_advantage"):
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
    return pd.DataFrame(summary_rows), point_cells, macro_replicates


def main() -> Path:
    age_summary, age_replicates = age_bootstrap()
    risk_summary, risk_cells, risk_replicates = risk_bootstrap()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    age_summary.to_csv(OUTPUT_ROOT / "age_paired_bootstrap_ci.csv", index=False)
    age_replicates.to_csv(OUTPUT_ROOT / "age_replicates.csv.gz", index=False, compression="gzip")
    risk_summary.to_csv(OUTPUT_ROOT / "risk_paired_bootstrap_ci.csv", index=False)
    risk_cells.to_csv(OUTPUT_ROOT / "risk_point_cells.csv", index=False)
    risk_replicates.to_csv(OUTPUT_ROOT / "risk_macro_replicates.csv", index=False)
    atomic_json(
        OUTPUT_ROOT / "manifest.json",
        {
            "schema_version": "revision0710.ehr_age.codebook_v2.matched_control_bootstrap.v1",
            "status": "COMPLETE",
            "completed_utc": utc_now(),
            "n_resamples": N_RESAMPLES,
            "bootstrap_unit": "participant within outer fold; paired models",
            "cox_models_refit_per_resample": False,
            "comparison": "EHR-AGE versus RETAIN-organquery with matched supervision and training contract",
            "inputs": {
                "main_oof": {"path": str(MAIN_OOF.resolve()), "sha256": sha256_file(MAIN_OOF)},
                "matched_oof": {"path": str(MATCHED_OOF.resolve()), "sha256": sha256_file(MATCHED_OOF)},
            },
        },
    )
    print(age_summary.to_string(index=False))
    print(risk_summary.to_string(index=False))
    return OUTPUT_ROOT


if __name__ == "__main__":
    print(main())
