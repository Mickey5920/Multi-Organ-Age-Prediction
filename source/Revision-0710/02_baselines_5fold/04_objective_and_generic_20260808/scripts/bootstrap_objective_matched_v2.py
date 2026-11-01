from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from numba import njit


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REVISION_ROOT = EXPERIMENT_ROOT.parents[1]
CODEBOOK_ROOT = REVISION_ROOT / "03_ehr_age" / "06_codebook_v2"
DOWNSTREAM_SCRIPTS = REVISION_ROOT / "05_downstream" / "scripts"
DOWNSTREAM_COMMON = REVISION_ROOT / "05_downstream" / "common"
for path in (CODEBOOK_ROOT, DOWNSTREAM_SCRIPTS, DOWNSTREAM_COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bootstrap_risk_advantage import _fit_risk_scores, _main_paths  # noqa: E402
from crossfit_metrics import (  # noqa: E402
    ORGANS,
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


OUTPUT_ROOT = EXPERIMENT_ROOT / "04_aligned_v2_evaluation_20260809" / "paired_bootstrap"
REGISTRY_ROOT = EXPERIMENT_ROOT / "03_evaluation" / "registries"
PARTICIPANT_MASKS = (
    REVISION_ROOT
    / "01_data_rebuild"
    / "07_corrected_primary_v1"
    / "02_outcomes_60m"
    / "participant_followup_and_evaluation_masks.parquet"
)
V2_OUTCOMES = (
    REVISION_ROOT / "01_data_rebuild" / "09_v2_labels" / "organ_outcomes_60m.parquet"
)
FOLD_ROLES = REVISION_ROOT / "01_data_rebuild" / "03_folds" / "outer_fold_roles.parquet"
METHODS = (
    "retain_prospective",
    "behrt_style_prospective",
    "medbert_style_prospective",
    "ehrformer_style_prospective",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harrell_c(event: np.ndarray, duration: np.ndarray, score: np.ndarray) -> float:
    return float(concordance_index(duration, -score, event))


def prepare_weighted_cindex(
    event: np.ndarray, duration: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Precompute the fixed ordering/ranks needed for an exact weighted C-index."""
    predicted_time = -np.asarray(score, dtype=np.float64)
    event = np.asarray(event, dtype=np.bool_)
    duration = np.asarray(duration, dtype=np.float64)
    died_order = np.flatnonzero(event)
    died_order = died_order[np.argsort(duration[died_order], kind="mergesort")]
    censored_order = np.flatnonzero(~event)
    censored_order = censored_order[
        np.argsort(duration[censored_order], kind="mergesort")
    ]
    unique_died_prediction = np.unique(predicted_time[died_order])
    died_left = np.searchsorted(
        unique_died_prediction, predicted_time[died_order], side="left"
    ).astype(np.int64)
    died_right = np.searchsorted(
        unique_died_prediction, predicted_time[died_order], side="right"
    ).astype(np.int64)
    censored_left = np.searchsorted(
        unique_died_prediction, predicted_time[censored_order], side="left"
    ).astype(np.int64)
    censored_right = np.searchsorted(
        unique_died_prediction, predicted_time[censored_order], side="right"
    ).astype(np.int64)
    return (
        died_order.astype(np.int64),
        duration[died_order],
        died_left,
        died_right,
        censored_order.astype(np.int64),
        duration[censored_order],
        censored_left,
        censored_right,
        np.asarray([len(unique_died_prediction)], dtype=np.int64),
    )


@njit(cache=True)
def _fenwick_prefix(tree: np.ndarray, end: int) -> int:
    total = 0
    index = end
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


@njit(cache=True)
def _fenwick_add(tree: np.ndarray, position: int, value: int) -> None:
    index = position + 1
    while index < len(tree):
        tree[index] += value
        index += index & -index


@njit(cache=True)
def weighted_concordance_from_context(
    multiplicity: np.ndarray,
    died_order: np.ndarray,
    died_time: np.ndarray,
    died_left: np.ndarray,
    died_right: np.ndarray,
    censored_order: np.ndarray,
    censored_time: np.ndarray,
    censored_left: np.ndarray,
    censored_right: np.ndarray,
    n_bins_array: np.ndarray,
) -> float:
    """Match lifelines' C-index exactly for integer observation weights."""
    tree = np.zeros(int(n_bins_array[0]) + 1, dtype=np.int64)
    died_ix = 0
    censored_ix = 0
    pool_weight = 0
    pairs = 0
    correct = 0
    tied = 0
    while died_ix < len(died_order) or censored_ix < len(censored_order):
        take_died = censored_ix >= len(censored_order) or (
            died_ix < len(died_order) and died_time[died_ix] <= censored_time[censored_ix]
        )
        if take_died:
            next_ix = died_ix + 1
            while next_ix < len(died_order) and died_time[next_ix] == died_time[died_ix]:
                next_ix += 1
            for position in range(died_ix, next_ix):
                weight = int(multiplicity[died_order[position]])
                if weight == 0:
                    continue
                lower = _fenwick_prefix(tree, int(died_left[position]))
                equal = _fenwick_prefix(tree, int(died_right[position])) - lower
                pairs += weight * pool_weight
                correct += weight * lower
                tied += weight * equal
            for position in range(died_ix, next_ix):
                weight = int(multiplicity[died_order[position]])
                if weight == 0:
                    continue
                _fenwick_add(tree, int(died_left[position]), weight)
                pool_weight += weight
            died_ix = next_ix
        else:
            next_ix = censored_ix + 1
            while (
                next_ix < len(censored_order)
                and censored_time[next_ix] == censored_time[censored_ix]
            ):
                next_ix += 1
            for position in range(censored_ix, next_ix):
                weight = int(multiplicity[censored_order[position]])
                if weight == 0:
                    continue
                lower = _fenwick_prefix(tree, int(censored_left[position]))
                equal = _fenwick_prefix(tree, int(censored_right[position])) - lower
                pairs += weight * pool_weight
                correct += weight * lower
                tied += weight * equal
            censored_ix = next_ix
    if pairs == 0:
        return np.nan
    return (correct + 0.5 * tied) / pairs


def prepare_method_frames(method: str) -> dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame]]:
    files, _ = resolve_prediction_files(
        method=method,
        prediction_registry=REGISTRY_ROOT / f"{method}.json",
        predictions=None,
    )
    predictions = load_predictions(files)
    participants = load_participant_masks(PARTICIPANT_MASKS)
    outcomes = load_organ_outcomes(V2_OUTCOMES)
    roles = load_fold_roles(FOLD_ROLES)
    enriched = validate_cross_file_coverage(predictions, participants, outcomes, roles)
    risk_outcomes = outcomes.loc[
        outcomes["at_risk"].eq(1),
        ["participant_id", "organ_system", "duration_months", "event"],
    ].rename(columns={"duration_months": "duration"})
    result: dict[tuple[int, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in range(5):
        validation = enriched.loc[
            enriched["outer_fold"].eq(fold) & enriched["split_role"].eq("validation")
        ].copy()
        test = enriched.loc[
            enriched["outer_fold"].eq(fold) & enriched["split_role"].eq("test")
        ].copy()
        correction = fit_linear_age_correction(validation)
        validation = apply_linear_age_correction_to_validation(validation, correction)
        test = apply_linear_age_correction(test, correction)
        for organ in ORGANS:
            outcome = risk_outcomes.loc[risk_outcomes["organ_system"].eq(organ)]
            columns = [
                "participant_id",
                "organ_system",
                "actual_age",
                "sex",
                "corrected_age_gap",
            ]
            validation_organ = validation.loc[
                validation["organ_system"].eq(organ), columns
            ].merge(outcome, on=["participant_id", "organ_system"], validate="one_to_one")
            test_organ = test.loc[test["organ_system"].eq(organ), columns].merge(
                outcome, on=["participant_id", "organ_system"], validate="one_to_one"
            )
            result[(fold, organ)] = (validation_organ, test_organ)
    return result


def prepare_cells() -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    folds: dict[int, dict[str, Any]] = {}
    point_rows: list[dict[str, Any]] = []
    for fold in range(5):
        calibration_path, outer_path = _main_paths(fold)
        calibration = pd.read_parquet(calibration_path)
        test = pd.read_parquet(outer_path)
        calibration["participant_id"] = calibration["participant_id"].astype(str)
        test["participant_id"] = test["participant_id"].astype(str)
        universe = np.asarray(sorted(test["participant_id"].unique()), dtype=object)
        universe_index = {value: index for index, value in enumerate(universe)}
        cells: dict[str, Any] = {}
        for organ in ORGANS:
            validation_organ = calibration.loc[
                calibration["organ_system"].eq(organ) & calibration["at_risk"].eq(1)
            ].copy()
            test_organ = test.loc[
                test["organ_system"].eq(organ) & test["at_risk"].eq(1)
            ].copy()
            base_score = _fit_risk_scores(
                validation_organ, test_organ, ("actual_age", "sex")
            )
            ehr_score = _fit_risk_scores(
                validation_organ,
                test_organ,
                ("actual_age", "sex", "corrected_age_gap"),
            )
            cells[organ] = {
                "participant_ids": test_organ["participant_id"].to_numpy(dtype=object),
                "duration": test_organ["duration"].to_numpy(dtype=np.float64),
                "event": test_organ["event"].to_numpy(dtype=np.int8).astype(bool),
                "base_score": base_score,
                "ehr_score": ehr_score,
                "method_scores": {},
                "universe_positions": np.asarray(
                    [universe_index[value] for value in test_organ["participant_id"]],
                    dtype=np.int64,
                ),
            }
        folds[fold] = {"universe_n": len(universe), "cells": cells}

    for method in METHODS:
        method_frames = prepare_method_frames(method)
        for fold in range(5):
            for organ in ORGANS:
                validation_method, test_method = method_frames[(fold, organ)]
                cell = folds[fold]["cells"][organ]
                test_method["participant_id"] = test_method["participant_id"].astype(str)
                ordered = test_method.set_index("participant_id").loc[
                    cell["participant_ids"]
                ].reset_index()
                if not np.allclose(ordered["duration"], cell["duration"]):
                    raise RuntimeError(f"duration mismatch: {method} fold={fold} organ={organ}")
                if not np.array_equal(
                    ordered["event"].to_numpy(dtype=np.int8).astype(bool), cell["event"]
                ):
                    raise RuntimeError(f"event mismatch: {method} fold={fold} organ={organ}")
                cell["method_scores"][method] = _fit_risk_scores(
                    validation_method,
                    ordered,
                    ("actual_age", "sex", "corrected_age_gap"),
                )

    for fold in range(5):
        for organ in ORGANS:
            cell = folds[fold]["cells"][organ]
            base_c = harrell_c(cell["event"], cell["duration"], cell["base_score"])
            ehr_c = harrell_c(cell["event"], cell["duration"], cell["ehr_score"])
            row: dict[str, Any] = {
                "outer_fold": fold,
                "organ_system": organ,
                "c_base": base_c,
                "c_ehr_age": ehr_c,
                "ehr_age_delta_c": ehr_c - base_c,
            }
            for method in METHODS:
                method_c = harrell_c(
                    cell["event"], cell["duration"], cell["method_scores"][method]
                )
                row[f"c_{method}"] = method_c
                row[f"delta_c_{method}"] = method_c - base_c
                row[f"advantage_vs_{method}"] = ehr_c - method_c
            point_rows.append(row)
    return folds, pd.DataFrame(point_rows)


def bootstrap_cell(
    fold: int,
    organ: str,
    cell: Mapping[str, Any],
    universe_n: int,
    n_resamples: int,
    seed: int,
    engine: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + fold * 100_003)
    rows: list[dict[str, Any]] = []
    score_arrays = {
        "base": cell["base_score"],
        "ehr": cell["ehr_score"],
        **{method: cell["method_scores"][method] for method in METHODS},
    }
    weighted_contexts = (
        {
            name: prepare_weighted_cindex(cell["event"], cell["duration"], score)
            for name, score in score_arrays.items()
        }
        if engine == "weighted"
        else {}
    )
    for replicate in range(n_resamples):
        counts = np.bincount(
            rng.integers(0, universe_n, size=universe_n), minlength=universe_n
        )
        multiplicity = counts[cell["universe_positions"]]
        if engine == "weighted":
            values = {
                name: weighted_concordance_from_context(multiplicity, *context)
                for name, context in weighted_contexts.items()
            }
            if not np.isfinite(values["base"]):
                continue
            base_c = values["base"]
            ehr_c = values["ehr"]
        else:
            expanded = np.repeat(np.arange(len(multiplicity)), multiplicity)
            if len(expanded) < 3:
                continue
            event = cell["event"][expanded]
            if not event.any():
                continue
            duration = cell["duration"][expanded]
            base_c = harrell_c(event, duration, cell["base_score"][expanded])
            ehr_c = harrell_c(event, duration, cell["ehr_score"][expanded])
        row: dict[str, Any] = {
            "replicate": replicate,
            "outer_fold": fold,
            "organ_system": organ,
            "ehr_age_delta_c": ehr_c - base_c,
        }
        for method in METHODS:
            method_c = (
                values[method]
                if engine == "weighted"
                else harrell_c(
                    event, duration, cell["method_scores"][method][expanded]
                )
            )
            row[f"delta_c_{method}"] = method_c - base_c
            row[f"advantage_vs_{method}"] = ehr_c - method_c
        rows.append(row)
    return rows


def interval_rows(point: pd.DataFrame, replicates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in [
        "ehr_age_delta_c",
        *[f"delta_c_{method}" for method in METHODS],
        *[f"advantage_vs_{method}" for method in METHODS],
    ]:
        values = replicates[metric].dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "estimate": float(point[metric].mean()),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_probability_gt_0": float(np.mean(values > 0.0)),
                "valid_resamples": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9082026)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--engine", choices=("expanded", "weighted"), default="weighted")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    folds, point_cells = prepare_cells()
    point_cells.to_csv(output_root / "risk_point_cells.csv", index=False)
    point_organs = point_cells.groupby("organ_system", as_index=False).mean(numeric_only=True)
    point_organs.to_csv(output_root / "risk_point_organs.csv", index=False)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for fold, payload in folds.items():
            for organ, cell in payload["cells"].items():
                future = pool.submit(
                    bootstrap_cell,
                    fold,
                    organ,
                    cell,
                    int(payload["universe_n"]),
                    args.n_resamples,
                    args.seed,
                    args.engine,
                )
                futures[future] = (fold, organ)
        for future in as_completed(futures):
            fold, organ = futures[future]
            cell_rows = future.result()
            rows.extend(cell_rows)
            print(
                f"risk bootstrap fold {fold} organ {organ} complete rows={len(cell_rows)}",
                flush=True,
            )
    cells = pd.DataFrame(rows)
    expected = args.n_resamples * 5 * len(ORGANS)
    if len(cells) < 0.95 * expected:
        raise RuntimeError("fewer than 95% valid bootstrap cells")
    with gzip.open(output_root / "risk_cell_replicates.csv.gz", "wt", encoding="utf-8") as handle:
        cells.to_csv(handle, index=False)
    organ_replicates = cells.groupby(["replicate", "organ_system"], as_index=False).mean(
        numeric_only=True
    )
    macro_replicates = organ_replicates.groupby("replicate", as_index=False).mean(
        numeric_only=True
    )
    macro_replicates.to_csv(output_root / "risk_macro_replicates.csv", index=False)
    point_macro = point_organs.mean(numeric_only=True).to_frame().T
    intervals = interval_rows(point_macro, macro_replicates)
    intervals.to_csv(output_root / "risk_paired_bootstrap_ci.csv", index=False)
    manifest = {
        "schema_version": "revision0710.baseline_extension.aligned_v2_paired_bootstrap.v1",
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "bootstrap_unit": "participant within outer fold",
        "shared_draws": True,
        "cox_models_refit_per_resample": False,
        "concordance_engine": args.engine,
        "methods": list(METHODS),
        "inputs": {
            "v2_outcomes": {"path": str(V2_OUTCOMES.resolve()), "sha256": sha256_file(V2_OUTCOMES)},
            "participant_masks": {
                "path": str(PARTICIPANT_MASKS.resolve()),
                "sha256": sha256_file(PARTICIPANT_MASKS),
            },
            "fold_roles": {"path": str(FOLD_ROLES.resolve()), "sha256": sha256_file(FOLD_ROLES)},
            "registries": {
                method: {
                    "path": str((REGISTRY_ROOT / f"{method}.json").resolve()),
                    "sha256": sha256_file(REGISTRY_ROOT / f"{method}.json"),
                }
                for method in METHODS
            },
        },
    }
    (output_root / "risk_bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
