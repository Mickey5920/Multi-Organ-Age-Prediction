#!/usr/bin/env python3
"""Fast exact paired bootstrap for fixed final-model clinical-utility predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)
AUGMENTATIONS = ("target_gap", "all_gaps")
METRICS = (
    "delta_harrell_c",
    "delta_uno_c_60m",
    "delta_auc_60m",
    "brier_improvement_60m",
    "ipcw_idi_60m",
    "ipcw_continuous_nri_60m",
    "mean_delta_net_benefit",
)
THRESHOLDS = np.asarray((0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30), dtype=np.float64)
TIER = "C_context_density_primary"


def workspace_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "Revision-0710").is_dir():
        return cwd
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = workspace_root()
ANALYSIS_ROOT = (
    ROOT / "Revision-0710/Final_Fig_0716/Reviewer_Statistics/13_final_model_clinical_utility"
)
INPUT_ROOT = ANALYSIS_ROOT / "02_fast_bootstrap_input"
POINT_ROOT = ANALYSIS_ROOT / "01_point"
WORK_ROOT = ANALYSIS_ROOT / "03_fast_bootstrap_work"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "04_bootstrap_1000"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@njit(cache=True)
def fenwick_add(tree: np.ndarray, index: int, value: float) -> None:
    position = index + 1
    while position < len(tree):
        tree[position] += value
        position += position & -position


@njit(cache=True)
def fenwick_prefix(tree: np.ndarray, end: int) -> float:
    total = 0.0
    position = end
    while position > 0:
        total += tree[position]
        position -= position & -position
    return total


@njit(cache=True)
def concordance_pair(
    counts: np.ndarray,
    order: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    score_rank: np.ndarray,
    score_has_equal: np.ndarray,
    uno_event_weight: np.ndarray,
    unique_score_count: int,
) -> tuple[float, float]:
    tree_harrell = np.zeros(unique_score_count + 1, dtype=np.float64)
    tree_uno = np.zeros(unique_score_count + 1, dtype=np.float64)
    pool_harrell = 0.0
    pool_uno = 0.0
    numerator_harrell = 0.0
    denominator_harrell = 0.0
    numerator_uno = 0.0
    denominator_uno = 0.0
    start = 0
    n = len(order)
    while start < n:
        stop = start + 1
        current_time = duration[order[start]]
        while stop < n and duration[order[stop]] == current_time:
            stop += 1

        for position in range(start, stop):
            index = order[position]
            if event[index] != 1:
                continue
            multiplicity = float(counts[index])
            if multiplicity <= 0.0:
                continue
            rank = int(score_rank[index])
            lower_harrell = fenwick_prefix(tree_harrell, rank)
            lower_uno = fenwick_prefix(tree_uno, rank)
            tied_harrell = 0.0
            tied_uno = 0.0
            if score_has_equal[index] == 1:
                tied_harrell = fenwick_prefix(tree_harrell, rank + 1) - lower_harrell
                tied_uno = fenwick_prefix(tree_uno, rank + 1) - lower_uno
            numerator_harrell += multiplicity * (lower_harrell + 0.5 * tied_harrell)
            denominator_harrell += multiplicity * pool_harrell
            numerator_uno += multiplicity * (lower_uno + 0.5 * tied_uno)
            denominator_uno += multiplicity * pool_uno

        for position in range(start, stop):
            index = order[position]
            if event[index] != 1:
                continue
            multiplicity = float(counts[index])
            if multiplicity <= 0.0:
                continue
            rank = int(score_rank[index])
            fenwick_add(tree_harrell, rank, multiplicity)
            pool_harrell += multiplicity
            uno_weight = multiplicity * uno_event_weight[index]
            if uno_weight > 0.0:
                fenwick_add(tree_uno, rank, uno_weight)
                pool_uno += uno_weight

        for position in range(start, stop):
            index = order[position]
            if event[index] == 1:
                continue
            multiplicity = float(counts[index])
            if multiplicity <= 0.0:
                continue
            rank = int(score_rank[index])
            lower_harrell = fenwick_prefix(tree_harrell, rank)
            lower_uno = fenwick_prefix(tree_uno, rank)
            tied_harrell = 0.0
            tied_uno = 0.0
            if score_has_equal[index] == 1:
                tied_harrell = fenwick_prefix(tree_harrell, rank + 1) - lower_harrell
                tied_uno = fenwick_prefix(tree_uno, rank + 1) - lower_uno
            numerator_harrell += multiplicity * (lower_harrell + 0.5 * tied_harrell)
            denominator_harrell += multiplicity * pool_harrell
            numerator_uno += multiplicity * (lower_uno + 0.5 * tied_uno)
            denominator_uno += multiplicity * pool_uno
        start = stop
    return numerator_harrell / denominator_harrell, numerator_uno / denominator_uno


@njit(cache=True)
def auc_metric(
    counts: np.ndarray,
    control_indices: np.ndarray,
    control_ranks: np.ndarray,
    case_indices: np.ndarray,
    case_weight: np.ndarray,
    case_left: np.ndarray,
    case_right: np.ndarray,
    unique_control_count: int,
) -> float:
    control_bins = np.zeros(unique_control_count, dtype=np.float64)
    for position in range(len(control_indices)):
        index = control_indices[position]
        control_bins[control_ranks[position]] += counts[index]
    cumulative = np.zeros(unique_control_count + 1, dtype=np.float64)
    for position in range(unique_control_count):
        cumulative[position + 1] = cumulative[position] + control_bins[position]
    total_control = cumulative[unique_control_count]
    numerator = 0.0
    total_case_weight = 0.0
    for position in range(len(case_indices)):
        index = case_indices[position]
        weight = float(counts[index]) * case_weight[position]
        if weight <= 0.0:
            continue
        lower = cumulative[case_left[position]]
        tied = cumulative[case_right[position]] - lower
        numerator += weight * (lower + 0.5 * tied)
        total_case_weight += weight
    return numerator / (total_case_weight * total_control)


@njit(cache=True)
def supportive_metrics(
    counts: np.ndarray,
    event: np.ndarray,
    base_brier_contribution: np.ndarray,
    augmented_brier_contribution: np.ndarray,
    case_weight: np.ndarray,
    control_weight: np.ndarray,
    delta_risk: np.ndarray,
    direction: np.ndarray,
    delta_net_benefit_contribution: np.ndarray,
) -> tuple[float, float, float, float, int, int]:
    total_n = 0.0
    total_events = 0
    base_brier = 0.0
    augmented_brier = 0.0
    case_total = 0.0
    control_total = 0.0
    case_delta = 0.0
    control_delta = 0.0
    nri_event = 0.0
    nri_nonevent = 0.0
    delta_net_benefit = 0.0
    for index in range(len(counts)):
        multiplicity = float(counts[index])
        if multiplicity <= 0.0:
            continue
        total_n += multiplicity
        total_events += int(counts[index]) * int(event[index])
        base_brier += multiplicity * base_brier_contribution[index]
        augmented_brier += multiplicity * augmented_brier_contribution[index]
        weighted_case = multiplicity * case_weight[index]
        weighted_control = multiplicity * control_weight[index]
        case_total += weighted_case
        control_total += weighted_control
        case_delta += weighted_case * delta_risk[index]
        control_delta += weighted_control * delta_risk[index]
        nri_event += weighted_case * direction[index]
        nri_nonevent -= weighted_control * direction[index]
        delta_net_benefit += multiplicity * delta_net_benefit_contribution[index]
    brier_improvement = (base_brier - augmented_brier) / total_n
    idi = case_delta / case_total - control_delta / control_total
    nri = nri_event / case_total + nri_nonevent / control_total
    mean_delta_net_benefit = delta_net_benefit / total_n
    return (
        brier_improvement,
        idi,
        nri,
        mean_delta_net_benefit,
        int(total_n),
        total_events,
    )


@dataclass(frozen=True)
class ModelPreparation:
    score_rank: np.ndarray
    score_has_equal: np.ndarray
    unique_score_count: int
    control_indices: np.ndarray
    control_ranks: np.ndarray
    case_indices: np.ndarray
    case_weights: np.ndarray
    case_left: np.ndarray
    case_right: np.ndarray
    unique_control_count: int


@dataclass(frozen=True)
class AugmentationPreparation:
    model: ModelPreparation
    augmented_brier_contribution: np.ndarray
    delta_risk: np.ndarray
    direction: np.ndarray
    delta_net_benefit_contribution: np.ndarray


@dataclass(frozen=True)
class CellPreparation:
    fold: int
    organ: str
    universe_size: int
    universe_positions: np.ndarray
    duration: np.ndarray
    event: np.ndarray
    uno_event_weight: np.ndarray
    case_weight: np.ndarray
    control_weight: np.ndarray
    order: np.ndarray
    base: ModelPreparation
    base_brier_contribution: np.ndarray
    augmentations: dict[str, AugmentationPreparation]


def prepare_model(
    duration: np.ndarray,
    event: np.ndarray,
    case_weight: np.ndarray,
    control_weight: np.ndarray,
    score: np.ndarray,
) -> ModelPreparation:
    survival_score = -score
    unique_event_scores = np.unique(survival_score[event == 1])
    score_rank = np.searchsorted(unique_event_scores, survival_score, side="left").astype(np.int64)
    valid = score_rank < len(unique_event_scores)
    score_has_equal = np.zeros(len(score), dtype=np.int8)
    score_has_equal[valid] = (
        unique_event_scores[score_rank[valid]] == survival_score[valid]
    ).astype(np.int8)

    control_indices = np.flatnonzero(control_weight > 0.0).astype(np.int64)
    case_indices = np.flatnonzero(case_weight > 0.0).astype(np.int64)
    unique_control_scores = np.unique(score[control_indices])
    control_ranks = np.searchsorted(
        unique_control_scores, score[control_indices], side="left"
    ).astype(np.int64)
    tolerance = 1e-8
    case_left = np.searchsorted(
        unique_control_scores, score[case_indices] - tolerance, side="left"
    ).astype(np.int64)
    case_right = np.searchsorted(
        unique_control_scores, score[case_indices] + tolerance, side="right"
    ).astype(np.int64)
    return ModelPreparation(
        score_rank=score_rank,
        score_has_equal=score_has_equal,
        unique_score_count=len(unique_event_scores),
        control_indices=control_indices,
        control_ranks=control_ranks,
        case_indices=case_indices,
        case_weights=case_weight[case_indices].astype(np.float64),
        case_left=case_left,
        case_right=case_right,
        unique_control_count=len(unique_control_scores),
    )


def prepare_augmentation(
    duration: np.ndarray,
    event: np.ndarray,
    case_weight: np.ndarray,
    control_weight: np.ndarray,
    base_risk: np.ndarray,
    augmented_score: np.ndarray,
    augmented_risk: np.ndarray,
) -> AugmentationPreparation:
    model = prepare_model(
        duration,
        event,
        case_weight,
        control_weight,
        augmented_score,
    )
    augmented_brier = (
        case_weight * np.square(1.0 - augmented_risk)
        + control_weight * np.square(augmented_risk)
    )
    delta_risk = augmented_risk - base_risk
    direction = np.zeros(len(base_risk), dtype=np.float64)
    direction[augmented_risk > base_risk] = 1.0
    direction[augmented_risk < base_risk] = -1.0
    delta_nb = np.zeros(len(base_risk), dtype=np.float64)
    for threshold in THRESHOLDS:
        odds = threshold / (1.0 - threshold)
        selection_delta = (
            (augmented_risk >= threshold).astype(np.float64)
            - (base_risk >= threshold).astype(np.float64)
        )
        delta_nb += selection_delta * (case_weight - control_weight * odds)
    delta_nb /= len(THRESHOLDS)
    return AugmentationPreparation(
        model=model,
        augmented_brier_contribution=augmented_brier.astype(np.float64),
        delta_risk=delta_risk.astype(np.float64),
        direction=direction,
        delta_net_benefit_contribution=delta_nb.astype(np.float64),
    )


def load_cell(fold: int, organ: str) -> CellPreparation:
    path = INPUT_ROOT / f"outer_fold_{fold}" / f"{organ}.npz"
    with np.load(path) as data:
        universe_size = int(data["universe_size"][0])
        universe_positions = data["universe_positions"].astype(np.int64)
        duration = data["duration"].astype(np.float64)
        event = data["event"].astype(np.int8)
        uno_event_weight = data["uno_event_weight"].astype(np.float64)
        case_weight = data["case_weight"].astype(np.float64)
        control_weight = data["control_weight"].astype(np.float64)
        base_score = data["base_score"].astype(np.float64)
        base_risk = data["base_risk"].astype(np.float64)
        target_score = data["target_gap_score"].astype(np.float64)
        target_risk = data["target_gap_risk"].astype(np.float64)
        all_score = data["all_gaps_score"].astype(np.float64)
        all_risk = data["all_gaps_risk"].astype(np.float64)
    order = np.argsort(duration, kind="mergesort").astype(np.int64)
    base = prepare_model(duration, event, case_weight, control_weight, base_score)
    base_brier = (
        case_weight * np.square(1.0 - base_risk)
        + control_weight * np.square(base_risk)
    )
    augmentations = {
        "target_gap": prepare_augmentation(
            duration,
            event,
            case_weight,
            control_weight,
            base_risk,
            target_score,
            target_risk,
        ),
        "all_gaps": prepare_augmentation(
            duration,
            event,
            case_weight,
            control_weight,
            base_risk,
            all_score,
            all_risk,
        ),
    }
    return CellPreparation(
        fold=fold,
        organ=organ,
        universe_size=universe_size,
        universe_positions=universe_positions,
        duration=duration,
        event=event,
        uno_event_weight=uno_event_weight,
        case_weight=case_weight,
        control_weight=control_weight,
        order=order,
        base=base,
        base_brier_contribution=base_brier.astype(np.float64),
        augmentations=augmentations,
    )


def model_metrics(
    cell: CellPreparation,
    model: ModelPreparation,
    counts: np.ndarray,
) -> tuple[float, float, float]:
    harrell, uno = concordance_pair(
        counts,
        cell.order,
        cell.duration,
        cell.event,
        model.score_rank,
        model.score_has_equal,
        cell.uno_event_weight,
        model.unique_score_count,
    )
    auc = auc_metric(
        counts,
        model.control_indices,
        model.control_ranks,
        model.case_indices,
        model.case_weights,
        model.case_left,
        model.case_right,
        model.unique_control_count,
    )
    return harrell, uno, auc


def incremental_metrics(
    cell: CellPreparation,
    counts: np.ndarray,
    augmentation: str,
    base_metrics: tuple[float, float, float] | None = None,
) -> dict[str, float | int]:
    if base_metrics is None:
        base_metrics = model_metrics(cell, cell.base, counts)
    augmented = cell.augmentations[augmentation]
    augmented_metrics = model_metrics(cell, augmented.model, counts)
    brier, idi, nri, delta_nb, n_resampled, events_resampled = supportive_metrics(
        counts,
        cell.event,
        cell.base_brier_contribution,
        augmented.augmented_brier_contribution,
        cell.case_weight,
        cell.control_weight,
        augmented.delta_risk,
        augmented.direction,
        augmented.delta_net_benefit_contribution,
    )
    return {
        "delta_harrell_c": augmented_metrics[0] - base_metrics[0],
        "delta_uno_c_60m": augmented_metrics[1] - base_metrics[1],
        "delta_auc_60m": augmented_metrics[2] - base_metrics[2],
        "brier_improvement_60m": brier,
        "ipcw_idi_60m": idi,
        "ipcw_continuous_nri_60m": nri,
        "mean_delta_net_benefit": delta_nb,
        "n_resampled_at_risk": n_resampled,
        "events_resampled": events_resampled,
    }


def parity_check() -> pd.DataFrame:
    manifest = json.loads((INPUT_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    rows = []
    cache: dict[tuple[int, str], CellPreparation] = {}
    for reference in manifest["parity_reference"]:
        fold = int(reference["outer_fold"])
        organ = str(reference["organ_system"])
        augmentation = str(reference["augmentation"])
        key = (fold, organ)
        if key not in cache:
            cache[key] = load_cell(fold, organ)
        counts = np.load(reference["counts_path"], allow_pickle=False).astype(np.int64)
        observed = incremental_metrics(cache[key], counts, augmentation)
        for metric in METRICS:
            expected = float(reference["metrics"][metric])
            actual = float(observed[metric])
            rows.append(
                {
                    "outer_fold": fold,
                    "organ_system": organ,
                    "augmentation": augmentation,
                    "metric": metric,
                    "established_weighted": expected,
                    "fast_exact": actual,
                    "difference": actual - expected,
                }
            )
    frame = pd.DataFrame(rows)
    maximum = float(frame["difference"].abs().max())
    if maximum > 2e-10:
        worst = frame.loc[frame["difference"].abs().idxmax()].to_dict()
        raise RuntimeError(f"fast metric parity failed: {worst}")
    return frame


def run_fold(fold: int, replicates: int, seed: int) -> pd.DataFrame:
    cells = {organ: load_cell(fold, organ) for organ in ORGANS}
    universe_size = next(iter(cells.values())).universe_size
    if {cell.universe_size for cell in cells.values()} != {universe_size}:
        raise RuntimeError(f"fold {fold}: inconsistent bootstrap universes")
    rng = np.random.default_rng(seed + 10007 * fold)
    probability = np.full(universe_size, 1.0 / universe_size, dtype=np.float64)
    rows = []
    started = time.time()
    for replicate in range(replicates):
        universe_counts = rng.multinomial(universe_size, probability)
        for organ in ORGANS:
            cell = cells[organ]
            counts = universe_counts[cell.universe_positions].astype(np.int64, copy=False)
            base_metrics = model_metrics(cell, cell.base, counts)
            for augmentation in AUGMENTATIONS:
                result = incremental_metrics(cell, counts, augmentation, base_metrics)
                rows.append(
                    {
                        "bootstrap": replicate,
                        "outer_fold": fold,
                        "organ_system": organ,
                        "augmentation": augmentation,
                        **result,
                    }
                )
        if (replicate + 1) % 50 == 0 or replicate == 0:
            print(
                f"fold {fold} bootstrap {replicate + 1}/{replicates} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_ci(
    frame: pd.DataFrame,
    group_columns: list[str],
    point_frame: pd.DataFrame,
    point_keys: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group_columns, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns, keys))
        point_match = point_frame.copy()
        for bootstrap_key, point_key in zip(group_columns, point_keys):
            point_match = point_match.loc[point_match[point_key].eq(key_map[bootstrap_key])]
        if len(point_match) != 1:
            raise RuntimeError(f"point summary key mismatch: {key_map}")
        point_row = point_match.iloc[0]
        for metric in METRICS:
            values = part[metric].to_numpy(dtype=float)
            rows.append(
                {
                    **key_map,
                    "metric": metric,
                    "point_estimate": float(point_row[metric]),
                    "bootstrap_mean": float(values.mean()),
                    "bootstrap_sd": float(values.std(ddof=1)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "probability_favorable": float(np.mean(values > 0.0)),
                    "bootstrap_replicates": len(values),
                }
            )
    return pd.DataFrame(rows)


def combine(replicates: int, seed: int, output_dir: Path) -> None:
    fold_files = [
        WORK_ROOT / f"seed_{seed}" / f"fold_{fold}__replicates_{replicates}.csv.gz"
        for fold in range(5)
    ]
    missing = [path for path in fold_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fold files: {missing}")
    if output_dir.exists():
        raise FileExistsError(f"create-once output already exists: {output_dir}")
    cell = pd.concat([pd.read_csv(path) for path in fold_files], ignore_index=True)
    expected_rows = 5 * replicates * len(ORGANS) * len(AUGMENTATIONS)
    if len(cell) != expected_rows or cell[list(METRICS)].isna().any().any():
        raise RuntimeError("bootstrap cell row/finite contract failed")
    organ = cell.groupby(
        ["bootstrap", "organ_system", "augmentation"], as_index=False
    )[list(METRICS)].mean()
    macro = organ.groupby(["bootstrap", "augmentation"], as_index=False)[list(METRICS)].mean()
    point_organ = pd.read_csv(POINT_ROOT / "organ_incremental_metrics.csv")
    point_macro = pd.read_csv(POINT_ROOT / "macro_incremental_metrics.csv")
    point_organ = point_organ.loc[point_organ["tier"].eq(TIER)].copy()
    point_organ["augmentation"] = point_organ["augmented_model"].str.replace(
        f"{TIER}__", "", regex=False
    )
    point_macro = point_macro.loc[point_macro["tier"].eq(TIER)].copy()
    point_macro["augmentation"] = point_macro["augmented_model"].str.replace(
        f"{TIER}__", "", regex=False
    )
    organ_ci = summarize_ci(
        organ,
        ["organ_system", "augmentation"],
        point_organ,
        ["organ_system", "augmentation"],
    )
    macro_ci = summarize_ci(
        macro,
        ["augmentation"],
        point_macro,
        ["augmentation"],
    )
    parity = parity_check()
    output_dir.mkdir(parents=True, exist_ok=False)
    cell.to_csv(
        output_dir / "cell_bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
        lineterminator="\n",
    )
    organ.to_csv(output_dir / "organ_bootstrap_replicates.csv", index=False, lineterminator="\n")
    macro.to_csv(output_dir / "macro_bootstrap_replicates.csv", index=False, lineterminator="\n")
    organ_ci.to_csv(output_dir / "organ_paired_bootstrap_ci.csv", index=False, lineterminator="\n")
    macro_ci.to_csv(output_dir / "macro_paired_bootstrap_ci.csv", index=False, lineterminator="\n")
    parity.to_csv(output_dir / "fast_metric_parity.csv", index=False, lineterminator="\n")
    manifest = {
        "schema_version": "revision0710.final_model_clinical_utility.fast_exact_bootstrap.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "replicates": replicates,
        "seed": seed,
        "bootstrap_unit": "participant within outer fold",
        "dependence": "same fold-level multinomial counts shared across organs and paired models",
        "model_refit": False,
        "fit_censoring_distribution_refit": False,
        "metrics": list(METRICS),
        "augmentations": list(AUGMENTATIONS),
        "death_handling": "cause-specific censoring",
        "row_counts": {"cell": len(cell), "organ": len(organ), "macro": len(macro)},
        "fast_metric_parity_max_abs_difference": float(parity["difference"].abs().max()),
        "input_manifest": {
            "path": (INPUT_ROOT / "MANIFEST.json").as_posix(),
            "sha256": sha256_file(INPUT_ROOT / "MANIFEST.json"),
        },
        "fold_inputs": [
            {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in fold_files
        ],
        "point_inputs": [
            {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                POINT_ROOT / "organ_incremental_metrics.csv",
                POINT_ROOT / "macro_incremental_metrics.csv",
            )
        ],
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("parity", "run-fold", "combine"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7110)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "parity":
        parity = parity_check()
        print(parity.to_string(index=False), flush=True)
        print(f"max_abs_difference={parity['difference'].abs().max():.12g}", flush=True)
        return
    if args.command == "run-fold":
        if args.fold is None:
            raise ValueError("--fold is required for run-fold")
        output = (
            WORK_ROOT
            / f"seed_{args.seed}"
            / f"fold_{args.fold}__replicates_{args.replicates}.csv.gz"
        )
        if output.exists():
            raise FileExistsError(f"create-once output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        result = run_fold(args.fold, args.replicates, args.seed)
        result.to_csv(output, index=False, compression="gzip", lineterminator="\n")
        return
    combine(args.replicates, args.seed, args.output_dir.resolve())


if __name__ == "__main__":
    main()
