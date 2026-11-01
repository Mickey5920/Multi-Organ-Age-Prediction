#!/usr/bin/env python3
"""Paired participant bootstrap for fixed Revision-0710 clinical predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv
from lifelines.utils.btree import _BTree


ORGANS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)
FOLDS = (0, 1, 2, 3, 4)
TIER = "C_context_density_primary"
AUGMENTATIONS = ("target_gap", "all_gaps")
THRESHOLDS = np.asarray((0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30), dtype=float)
METRICS = (
    "delta_harrell_c",
    "delta_uno_c_60m",
    "delta_auc_60m",
    "brier_improvement_60m",
    "ipcw_idi_60m",
    "ipcw_continuous_nri_60m",
    "mean_delta_net_benefit",
)
TASK_DIRS = {
    0: "01_fig04_01_canonical_control__outer0",
    1: "02_fig04_01_canonical_control__outer1",
    2: "03_fig04_01_canonical_control__outer2",
    3: "04_fig04_01_canonical_control__outer3",
    4: "05_fig04_01_canonical_control__outer4",
}


def root_from_script() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Revision-0710").is_dir():
            return parent
    raise RuntimeError("workspace root not found")


ROOT = root_from_script()
FORMAL = (
    ROOT
    / "Revision-0710/06_figures_and_statistics/03_reviewer_statistics"
    / "03_clinical_utility/01_formal_run"
)
CACHE_ROOT = FORMAL / "03_outputs/folds"
CONTEXT_PICKLE = FORMAL / "02_input_cache/participant_context_covariates.pkl"
POINT_CELL = FORMAL / "03_outputs/cell_incremental_metrics.csv"
POINT_ORGAN = FORMAL / "03_outputs/organ_incremental_metrics.csv"
POINT_MACRO = FORMAL / "03_outputs/macro_incremental_metrics.csv"
TASK_ROOT = ROOT / "Revision-0710/04_ablation_runs/fig04_fig05_corrected_primary_v1/01_tasks"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "04_clinical_utility_paired_bootstrap"


@dataclass
class Cell:
    fold: int
    organ: str
    participant_id: np.ndarray
    duration: np.ndarray
    event: np.ndarray
    fit_survival: np.ndarray
    censoring: CensoringDistributionEstimator
    effective_horizon: float
    uno_event_weight: np.ndarray
    case_weight: np.ndarray
    control_weight: np.ndarray
    base_score: np.ndarray
    base_risk: np.ndarray
    augmented_score: dict[str, np.ndarray]
    augmented_risk: dict[str, np.ndarray]
    universe_positions: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_path(fold: int) -> Path:
    return TASK_ROOT / TASK_DIRS[fold] / "02_calibration/calibration_predictions_corrected.csv"


def survival(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(event=event.astype(bool), time=duration.astype(np.float64))


def ipcw_components(
    duration: np.ndarray,
    event: np.ndarray,
    censoring: CensoringDistributionEstimator,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_by = event.astype(bool) & (duration <= horizon)
    event_free = duration > horizon
    case_weight = np.zeros(len(duration), dtype=np.float64)
    control_weight = np.zeros(len(duration), dtype=np.float64)
    if event_by.any():
        g_event = np.asarray(censoring.predict_proba(duration[event_by]), dtype=float)
        if (g_event <= 0.0).any() or not np.isfinite(g_event).all():
            raise RuntimeError("unsupported event IPCW")
        case_weight[event_by] = 1.0 / g_event
    g_horizon = float(censoring.predict_proba(np.asarray([horizon]))[0])
    if not np.isfinite(g_horizon) or g_horizon <= 0.0:
        raise RuntimeError("unsupported horizon IPCW")
    control_weight[event_free] = 1.0 / g_horizon
    return case_weight, control_weight


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    return float(np.dot(values, weights) / denominator) if denominator > 0.0 else np.nan


class WeightedBTree:
    def __init__(self, values: np.ndarray) -> None:
        self.tree = _BTree._treeify(values)
        self.weight = np.zeros_like(self.tree, dtype=np.float64)

    def insert(self, value: float, weight: float) -> None:
        if weight <= 0.0:
            return
        index = 0
        while index < len(self.tree):
            current = self.tree[index]
            self.weight[index] += weight
            if value < current:
                index = 2 * index + 1
            elif value > current:
                index = 2 * index + 2
            else:
                return
        raise RuntimeError("weighted tree value not found")

    def total(self) -> float:
        return float(self.weight[0]) if len(self.weight) else 0.0

    def rank(self, value: float) -> tuple[float, float]:
        index = 0
        rank = 0.0
        equal = 0.0
        while index < len(self.tree):
            current = self.tree[index]
            if value < current:
                index = 2 * index + 1
            elif value > current:
                rank += self.weight[index]
                right = 2 * index + 2
                if right < len(self.tree):
                    rank -= self.weight[right]
                    index = right
                else:
                    return rank, equal
            else:
                equal = self.weight[index]
                left = 2 * index + 1
                if left < len(self.tree):
                    equal -= self.weight[left]
                    rank += self.weight[left]
                    right = left + 1
                    if right < len(self.tree):
                        equal -= self.weight[right]
                return rank, equal
        return rank, equal


def weighted_concordance(
    duration: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    sample_count: np.ndarray,
    event_weight: np.ndarray,
) -> float:
    active = sample_count > 0
    duration = duration[active]
    event = event[active].astype(bool)
    survival_score = -risk_score[active]
    count = sample_count[active].astype(np.float64)
    event_weight = event_weight[active].astype(np.float64)
    died = event
    died_time = duration[died]
    died_score = survival_score[died]
    died_count = count[died]
    died_weight = event_weight[died]
    censored_time = duration[~died]
    censored_score = survival_score[~died]
    censored_count = count[~died]
    died_order = np.argsort(died_time, kind="mergesort")
    censored_order = np.argsort(censored_time, kind="mergesort")
    died_time = died_time[died_order]
    died_score = died_score[died_order]
    died_count = died_count[died_order]
    died_weight = died_weight[died_order]
    censored_time = censored_time[censored_order]
    censored_score = censored_score[censored_order]
    censored_count = censored_count[censored_order]
    insertable = died_count * died_weight > 0.0
    values = np.unique(died_score[insertable])
    if len(values) == 0:
        raise RuntimeError("no weighted events for concordance")
    tree = WeightedBTree(values)
    numerator = 0.0
    denominator = 0.0
    died_index = 0
    censored_index = 0

    def score_batch(scores: np.ndarray, counts: np.ndarray, start: int, end: int) -> tuple[float, float]:
        batch_numerator = 0.0
        batch_denominator = 0.0
        pool_total = tree.total()
        for position in range(start, end):
            multiplicity = float(counts[position])
            if multiplicity <= 0.0:
                continue
            lower, tied = tree.rank(float(scores[position]))
            batch_numerator += multiplicity * (lower + 0.5 * tied)
            batch_denominator += multiplicity * pool_total
        return batch_numerator, batch_denominator

    while died_index < len(died_time) or censored_index < len(censored_time):
        next_died = died_time[died_index] if died_index < len(died_time) else np.inf
        next_censored = censored_time[censored_index] if censored_index < len(censored_time) else np.inf
        if next_died <= next_censored:
            end = died_index + 1
            while end < len(died_time) and died_time[end] == next_died:
                end += 1
            add_num, add_den = score_batch(died_score, died_count, died_index, end)
            numerator += add_num
            denominator += add_den
            for position in range(died_index, end):
                tree.insert(
                    float(died_score[position]),
                    float(died_count[position] * died_weight[position]),
                )
            died_index = end
        else:
            end = censored_index + 1
            while end < len(censored_time) and censored_time[end] == next_censored:
                end += 1
            add_num, add_den = score_batch(censored_score, censored_count, censored_index, end)
            numerator += add_num
            denominator += add_den
            censored_index = end
    if denominator <= 0.0:
        raise RuntimeError("no comparable weighted pairs")
    return float(numerator / denominator)


def weighted_auc(
    cell: Cell,
    risk_score: np.ndarray,
    sample_count: np.ndarray,
) -> float:
    case = (cell.event == 1) & (cell.duration <= cell.effective_horizon) & (sample_count > 0)
    control = (cell.duration > cell.effective_horizon) & (sample_count > 0)
    case_weight = sample_count[case].astype(float) * cell.case_weight[case]
    control_count = sample_count[control].astype(float)
    control_risk = risk_score[control]
    order = np.argsort(control_risk, kind="mergesort")
    control_risk = control_risk[order]
    cumulative = np.concatenate([[0.0], np.cumsum(control_count[order])])
    case_risk = risk_score[case]
    tolerance = 1e-8
    left = np.searchsorted(control_risk, case_risk - tolerance, side="left")
    right = np.searchsorted(control_risk, case_risk + tolerance, side="right")
    lower_weight = cumulative[left]
    tied_weight = cumulative[right] - cumulative[left]
    numerator = float(np.dot(case_weight, lower_weight + 0.5 * tied_weight))
    denominator = float(case_weight.sum() * control_count.sum())
    if denominator <= 0.0:
        raise RuntimeError("no weighted case-control pairs for AUC")
    return numerator / denominator


def model_metrics(
    cell: Cell,
    sample_count: np.ndarray,
    score: np.ndarray,
    risk: np.ndarray,
) -> dict[str, float]:
    harrell = weighted_concordance(
        cell.duration,
        cell.event,
        score,
        sample_count,
        np.ones(len(cell.duration), dtype=np.float64),
    )
    uno = weighted_concordance(
        cell.duration,
        cell.event,
        score,
        sample_count,
        cell.uno_event_weight,
    )
    auc = weighted_auc(cell, score, sample_count)
    denominator = float(sample_count.sum())
    brier = float(
        np.dot(
            sample_count,
            cell.case_weight * np.square(1.0 - risk)
            + cell.control_weight * np.square(risk),
        )
        / denominator
    )
    return {"harrell": harrell, "uno": uno, "auc": auc, "brier": brier}


def incremental_metrics(
    cell: Cell,
    sample_count: np.ndarray,
    augmentation: str,
    *,
    base: dict[str, float] | None = None,
) -> dict[str, float]:
    if base is None:
        base = model_metrics(cell, sample_count, cell.base_score, cell.base_risk)
    augmented = model_metrics(
        cell,
        sample_count,
        cell.augmented_score[augmentation],
        cell.augmented_risk[augmentation],
    )
    old = cell.base_risk
    new = cell.augmented_risk[augmentation]
    case_weight = sample_count * cell.case_weight
    control_weight = sample_count * cell.control_weight
    old_case = weighted_mean(old, case_weight)
    new_case = weighted_mean(new, case_weight)
    old_control = weighted_mean(old, control_weight)
    new_control = weighted_mean(new, control_weight)
    up = (new > old).astype(float)
    down = (new < old).astype(float)
    idi = (new_case - old_case) - (new_control - old_control)
    nri = weighted_mean(up - down, case_weight) + weighted_mean(down - up, control_weight)
    n = float(sample_count.sum())
    delta_nb: list[float] = []
    for threshold in THRESHOLDS:
        odds = float(threshold / (1.0 - threshold))
        selected_old = old >= threshold
        selected_new = new >= threshold
        old_nb = float(case_weight[selected_old].sum() / n - control_weight[selected_old].sum() / n * odds)
        new_nb = float(case_weight[selected_new].sum() / n - control_weight[selected_new].sum() / n * odds)
        delta_nb.append(new_nb - old_nb)
    return {
        "delta_harrell_c": augmented["harrell"] - base["harrell"],
        "delta_uno_c_60m": augmented["uno"] - base["uno"],
        "delta_auc_60m": augmented["auc"] - base["auc"],
        "brier_improvement_60m": base["brier"] - augmented["brier"],
        "ipcw_idi_60m": float(idi),
        "ipcw_continuous_nri_60m": float(nri),
        "mean_delta_net_benefit": float(np.mean(delta_nb)),
    }


def explicit_model_metrics(
    cell: Cell,
    indices: np.ndarray,
    score: np.ndarray,
    risk: np.ndarray,
) -> dict[str, float]:
    duration = cell.duration[indices]
    event = cell.event[indices].astype(bool)
    score = score[indices]
    risk = risk[indices]
    score_survival = survival(duration, event)
    return {
        "harrell": float(concordance_index_censored(event, duration, score)[0]),
        "uno": float(
            concordance_index_ipcw(
                cell.fit_survival,
                score_survival,
                score,
                tau=cell.effective_horizon,
            )[0]
        ),
        "auc": float(
            cumulative_dynamic_auc(
                cell.fit_survival,
                score_survival,
                score,
                np.asarray([cell.effective_horizon]),
            )[0][0]
        ),
        "brier": float(
            brier_score(
                cell.fit_survival,
                score_survival,
                (1.0 - risk).reshape(-1, 1),
                np.asarray([cell.effective_horizon]),
            )[1][0]
        ),
    }


def explicit_incremental_metrics(
    cell: Cell,
    indices: np.ndarray,
    augmentation: str,
) -> dict[str, float]:
    base = explicit_model_metrics(cell, indices, cell.base_score, cell.base_risk)
    augmented = explicit_model_metrics(
        cell,
        indices,
        cell.augmented_score[augmentation],
        cell.augmented_risk[augmentation],
    )
    duration = cell.duration[indices]
    event = cell.event[indices]
    old = cell.base_risk[indices]
    new = cell.augmented_risk[augmentation][indices]
    case_weight, control_weight = ipcw_components(
        duration,
        event,
        cell.censoring,
        cell.effective_horizon,
    )
    idi = (
        weighted_mean(new, case_weight)
        - weighted_mean(old, case_weight)
        - weighted_mean(new, control_weight)
        + weighted_mean(old, control_weight)
    )
    up = (new > old).astype(float)
    down = (new < old).astype(float)
    nri = weighted_mean(up - down, case_weight) + weighted_mean(
        down - up,
        control_weight,
    )
    delta_nb: list[float] = []
    n = float(len(indices))
    for threshold in THRESHOLDS:
        odds = float(threshold / (1.0 - threshold))
        selected_old = old >= threshold
        selected_new = new >= threshold
        old_nb = float(
            case_weight[selected_old].sum() / n
            - control_weight[selected_old].sum() / n * odds
        )
        new_nb = float(
            case_weight[selected_new].sum() / n
            - control_weight[selected_new].sum() / n * odds
        )
        delta_nb.append(new_nb - old_nb)
    return {
        "delta_harrell_c": augmented["harrell"] - base["harrell"],
        "delta_uno_c_60m": augmented["uno"] - base["uno"],
        "delta_auc_60m": augmented["auc"] - base["auc"],
        "brier_improvement_60m": base["brier"] - augmented["brier"],
        "ipcw_idi_60m": float(idi),
        "ipcw_continuous_nri_60m": float(nri),
        "mean_delta_net_benefit": float(np.mean(delta_nb)),
    }


def bootstrap_weight_parity(
    cells: dict[tuple[int, str], Cell],
    universes: dict[int, np.ndarray],
    selected_cells: tuple[tuple[int, str], ...],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, organ in selected_cells:
        cell = cells[(fold, organ)]
        universe = universes[fold]
        rng = np.random.default_rng(seed + 10007 * fold)
        universe_counts = rng.multinomial(
            len(universe),
            np.full(len(universe), 1.0 / len(universe), dtype=float),
        )
        counts = universe_counts[cell.universe_positions]
        indices = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        weighted_base = model_metrics(cell, counts, cell.base_score, cell.base_risk)
        for augmentation in AUGMENTATIONS:
            weighted = incremental_metrics(
                cell,
                counts,
                augmentation,
                base=weighted_base,
            )
            explicit = explicit_incremental_metrics(cell, indices, augmentation)
            for metric in METRICS:
                rows.append(
                    {
                        "outer_fold": fold,
                        "organ_system": organ,
                        "augmentation": augmentation,
                        "metric": metric,
                        "weighted": weighted[metric],
                        "explicit_expansion": explicit[metric],
                        "difference": weighted[metric] - explicit[metric],
                    }
                )
    parity = pd.DataFrame(rows)
    if parity["difference"].abs().max() > 2e-8:
        worst = parity.loc[parity["difference"].abs().idxmax()].to_dict()
        raise RuntimeError(f"bootstrap weight parity failed: {worst}")
    return parity


def load_fold_cells(
    fold: int,
    context: pd.DataFrame,
) -> tuple[dict[tuple[int, str], Cell], np.ndarray]:
    ids = pd.to_numeric(
        context.loc[context["fold_id"].eq(fold), "participant_id"], errors="raise"
    ).to_numpy(dtype=np.int64)
    universe = np.sort(ids)
    cells: dict[tuple[int, str], Cell] = {}
    fit_all = pd.read_csv(
        calibration_path(fold),
        usecols=["participant_id", "organ_system", "duration_months", "event", "at_risk"],
        dtype={"participant_id": str, "organ_system": str},
        low_memory=False,
    )
    for organ in ORGANS:
        npz_path = CACHE_ROOT / f"outer_fold_{fold}/prediction_cache/{organ}.npz"
        with np.load(npz_path) as cache:
            participant_id = cache["participant_id"].astype(np.int64)
            duration = cache["duration"].astype(np.float64)
            event = cache["event"].astype(np.int8)
            base_prefix = f"{TIER}__base"
            base_score = cache[f"{base_prefix}__risk_score"].astype(np.float64)
            base_risk = cache[f"{base_prefix}__risk_60m"].astype(np.float64)
            augmented_score = {
                augmentation: cache[f"{TIER}__{augmentation}__risk_score"].astype(np.float64)
                for augmentation in AUGMENTATIONS
            }
            augmented_risk = {
                augmentation: cache[f"{TIER}__{augmentation}__risk_60m"].astype(np.float64)
                for augmentation in AUGMENTATIONS
            }
        fit = fit_all.loc[
            fit_all["organ_system"].eq(organ) & fit_all["at_risk"].eq(1),
            ["duration_months", "event"],
        ].copy()
        fit_duration = pd.to_numeric(fit["duration_months"], errors="raise").to_numpy(float)
        fit_event = pd.to_numeric(fit["event"], errors="raise").to_numpy(np.int8)
        boundary = float(min(fit_duration.max(), duration.max()))
        fit_boundary_event = np.isclose(fit_duration, boundary, atol=1e-6, rtol=0.0) & (fit_event == 1)
        fit_event[fit_boundary_event] = 0
        fit_survival = survival(fit_duration, fit_event)
        censoring = CensoringDistributionEstimator().fit(fit_survival)
        effective = float(np.nextafter(boundary, 0.0))
        score_survival = survival(duration, event)
        uno_event_weight = np.square(censoring.predict_ipcw(score_survival))
        uno_event_weight[duration >= effective] = 0.0
        case_weight, control_weight = ipcw_components(
            duration,
            event,
            censoring,
            effective,
        )
        positions = np.searchsorted(universe, participant_id)
        if np.any(positions >= len(universe)) or not np.array_equal(universe[positions], participant_id):
            raise RuntimeError(f"fold {fold} {organ}: cache IDs outside fold universe")
        cells[(fold, organ)] = Cell(
            fold=fold,
            organ=organ,
            participant_id=participant_id,
            duration=duration,
            event=event,
            fit_survival=fit_survival,
            censoring=censoring,
            effective_horizon=effective,
            uno_event_weight=uno_event_weight,
            case_weight=case_weight,
            control_weight=control_weight,
            base_score=base_score,
            base_risk=base_risk,
            augmented_score=augmented_score,
            augmented_risk=augmented_risk,
            universe_positions=positions,
        )
    return cells, universe


def run_fold_bootstrap(fold: int, replicates: int, seed: int) -> list[dict[str, object]]:
    context = pd.read_pickle(CONTEXT_PICKLE, compression="infer")
    context["participant_id"] = context["participant_id"].astype(str)
    cells, universe = load_fold_cells(fold, context)
    rng = np.random.default_rng(seed + 10007 * fold)
    probability = np.full(len(universe), 1.0 / len(universe), dtype=float)
    rows: list[dict[str, object]] = []
    started = time.time()
    for replicate in range(replicates):
        universe_counts = rng.multinomial(len(universe), probability)
        for organ in ORGANS:
            cell = cells[(fold, organ)]
            counts = universe_counts[cell.universe_positions]
            n_resampled = int(counts.sum())
            events_resampled = int(np.dot(counts, cell.event))
            if n_resampled < 100 or events_resampled < 2:
                raise RuntimeError(f"invalid bootstrap risk set: b={replicate} fold={fold} organ={organ}")
            base = model_metrics(cell, counts, cell.base_score, cell.base_risk)
            for augmentation in AUGMENTATIONS:
                result = incremental_metrics(cell, counts, augmentation, base=base)
                rows.append(
                    {
                        "bootstrap": replicate,
                        "outer_fold": fold,
                        "organ_system": organ,
                        "augmentation": augmentation,
                        "n_resampled_at_risk": n_resampled,
                        "events_resampled": events_resampled,
                        **result,
                    }
                )
        if (replicate + 1) % 10 == 0 or replicate == 0:
            print(
                f"fold {fold} bootstrap {replicate + 1}/{replicates} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return rows


def point_parity(
    cells: dict[tuple[int, str], Cell],
    point: pd.DataFrame,
    selected_cells: tuple[tuple[int, str], ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = point.loc[point["tier"].eq(TIER)].copy()
    for fold, organ in selected_cells:
        cell = cells[(fold, organ)]
        counts = np.ones(len(cell.duration), dtype=np.int64)
        base = model_metrics(cell, counts, cell.base_score, cell.base_risk)
        for augmentation in AUGMENTATIONS:
            observed = incremental_metrics(cell, counts, augmentation, base=base)
            expected_row = selected.loc[
                selected["outer_fold"].eq(fold)
                & selected["organ_system"].eq(organ)
                & selected["augmented_model"].eq(f"{TIER}__{augmentation}")
            ]
            if len(expected_row) != 1:
                raise RuntimeError(f"point row missing: fold={fold} organ={organ} aug={augmentation}")
            expected_row = expected_row.iloc[0]
            for metric in METRICS:
                difference = float(observed[metric] - expected_row[metric])
                rows.append(
                    {
                        "outer_fold": fold,
                        "organ_system": organ,
                        "augmentation": augmentation,
                        "metric": metric,
                        "recomputed": observed[metric],
                        "frozen_point": float(expected_row[metric]),
                        "difference": difference,
                    }
                )
    parity = pd.DataFrame(rows)
    if parity["difference"].abs().max() > 2e-5:
        worst = parity.loc[parity["difference"].abs().idxmax()].to_dict()
        raise RuntimeError(f"point parity failed: {worst}")
    return parity


def summarize_ci(
    frame: pd.DataFrame,
    group_columns: list[str],
    point_frame: pd.DataFrame,
    point_keys: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, part in frame.groupby(group_columns, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns, keys))
        point_match = point_frame.copy()
        for bootstrap_key, point_key in zip(group_columns, point_keys):
            point_match = point_match.loc[point_match[point_key].eq(key_map[bootstrap_key])]
        if len(point_match) != 1:
            raise RuntimeError(f"point summary key mismatch: {key_map} rows={len(point_match)}")
        point_row = point_match.iloc[0]
        for metric in METRICS:
            values = part[metric].to_numpy(float)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7110)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if args.output_dir.exists():
        raise FileExistsError(f"create-once output already exists: {args.output_dir}")
    context = pd.read_pickle(CONTEXT_PICKLE)
    context["participant_id"] = context["participant_id"].astype(str)
    point_cell = pd.read_csv(POINT_CELL)
    point_organ = pd.read_csv(POINT_ORGAN)
    point_macro = pd.read_csv(POINT_MACRO)
    if args.benchmark_only:
        from lifelines.utils import concordance_index as lifelines_concordance_index

        benchmark_cells, _ = load_fold_cells(0, context)
        cell = benchmark_cells[(0, "cardiovascular")]
        indices = np.arange(len(cell.duration), dtype=np.int64)
        duration = cell.duration[indices]
        event = cell.event[indices].astype(bool)
        score_survival = survival(duration, event)
        timings: dict[str, float] = {}
        started_metric = time.perf_counter()
        concordance_index_censored(event, duration, cell.base_score)
        timings["harrell_seconds"] = time.perf_counter() - started_metric
        started_metric = time.perf_counter()
        lifelines_concordance_index(duration, -cell.base_score, event)
        timings["harrell_lifelines_seconds"] = time.perf_counter() - started_metric
        started_metric = time.perf_counter()
        concordance_index_ipcw(
            cell.fit_survival,
            score_survival,
            cell.base_score,
            tau=cell.effective_horizon,
        )
        timings["uno_seconds"] = time.perf_counter() - started_metric
        started_metric = time.perf_counter()
        cumulative_dynamic_auc(
            cell.fit_survival,
            score_survival,
            cell.base_score,
            np.asarray([cell.effective_horizon]),
        )
        timings["auc_seconds"] = time.perf_counter() - started_metric
        started_metric = time.perf_counter()
        brier_score(
            cell.fit_survival,
            score_survival,
            (1.0 - cell.base_risk).reshape(-1, 1),
            np.asarray([cell.effective_horizon]),
        )
        timings["brier_seconds"] = time.perf_counter() - started_metric
        counts = np.ones(len(cell.duration), dtype=np.int64)
        started_metric = time.perf_counter()
        model_metrics(cell, counts, cell.base_score, cell.base_risk)
        timings["weighted_all_four_metrics_seconds"] = time.perf_counter() - started_metric
        print(json.dumps(timings, indent=2), flush=True)
        return 0
    audit_cells: dict[tuple[int, str], Cell] = {}
    audit_universes: dict[int, np.ndarray] = {}
    for fold in (0, 4):
        fold_cells, fold_universe = load_fold_cells(fold, context)
        audit_cells.update(fold_cells)
        audit_universes[fold] = fold_universe
    parity = point_parity(
        audit_cells,
        point_cell,
        selected_cells=((0, "cardiovascular"), (4, "immune")),
    )
    print(f"point parity max abs difference={parity.difference.abs().max():.3g}", flush=True)
    bootstrap_parity = bootstrap_weight_parity(
        audit_cells,
        audit_universes,
        selected_cells=((0, "cardiovascular"), (4, "immune")),
        seed=args.seed,
    )
    print(
        "bootstrap weight parity max abs difference="
        f"{bootstrap_parity.difference.abs().max():.3g}",
        flush=True,
    )
    if args.parity_only:
        print(parity.groupby("metric")["difference"].agg(["min", "max"]).to_string(), flush=True)
        print(
            bootstrap_parity.groupby("metric")["difference"]
            .agg(["min", "max"])
            .to_string(),
            flush=True,
        )
        return 0
    del audit_cells, audit_universes, context

    rows: list[dict[str, object]] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(FOLDS))) as pool:
        future_to_fold = {
            pool.submit(run_fold_bootstrap, fold, args.replicates, args.seed): fold
            for fold in FOLDS
        }
        for future in as_completed(future_to_fold):
            fold = future_to_fold[future]
            fold_rows = future.result()
            rows.extend(fold_rows)
            print(
                f"fold {fold} complete rows={len(fold_rows)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    cell_bootstrap = pd.DataFrame(rows)
    organ_bootstrap = cell_bootstrap.groupby(
        ["bootstrap", "organ_system", "augmentation"], as_index=False
    )[list(METRICS)].mean()
    macro_bootstrap = organ_bootstrap.groupby(
        ["bootstrap", "augmentation"], as_index=False
    )[list(METRICS)].mean()

    point_organ = point_organ.loc[point_organ["tier"].eq(TIER)].copy()
    point_organ["augmentation"] = point_organ["augmented_model"].str.replace(
        f"{TIER}__", "", regex=False
    )
    point_macro = point_macro.loc[point_macro["tier"].eq(TIER)].copy()
    point_macro["augmentation"] = point_macro["augmented_model"].str.replace(
        f"{TIER}__", "", regex=False
    )
    organ_ci = summarize_ci(
        organ_bootstrap,
        ["organ_system", "augmentation"],
        point_organ,
        ["organ_system", "augmentation"],
    )
    macro_ci = summarize_ci(
        macro_bootstrap,
        ["augmentation"],
        point_macro,
        ["augmentation"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    parity.to_csv(args.output_dir / "point_metric_parity.csv", index=False, lineterminator="\n")
    bootstrap_parity.to_csv(
        args.output_dir / "bootstrap_weight_metric_parity.csv",
        index=False,
        lineterminator="\n",
    )
    cell_bootstrap.to_csv(
        args.output_dir / "cell_bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
        lineterminator="\n",
    )
    organ_bootstrap.to_csv(args.output_dir / "organ_bootstrap_replicates.csv", index=False, lineterminator="\n")
    macro_bootstrap.to_csv(args.output_dir / "macro_bootstrap_replicates.csv", index=False, lineterminator="\n")
    organ_ci.to_csv(args.output_dir / "organ_paired_bootstrap_ci.csv", index=False, lineterminator="\n")
    macro_ci.to_csv(args.output_dir / "macro_paired_bootstrap_ci.csv", index=False, lineterminator="\n")
    manifest = {
        "schema_version": "revision0710.clinical_utility.paired_participant_bootstrap.v1",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "replicates": args.replicates,
        "seed": args.seed,
        "workers": min(args.workers, len(FOLDS)),
        "tier": TIER,
        "augmentations": list(AUGMENTATIONS),
        "metrics": list(METRICS),
        "bootstrap_unit": "participant within outer fold",
        "dependence": "same fold-level resampling weights shared across organs and paired models",
        "model_refit": False,
        "fit_censoring_distribution_refit": False,
        "death_handling": "cause-specific censoring",
        "point_parity_max_abs_difference": float(parity["difference"].abs().max()),
        "bootstrap_weight_parity_max_abs_difference": float(
            bootstrap_parity["difference"].abs().max()
        ),
        "elapsed_seconds": time.time() - started,
        "inputs": [
            {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in (CONTEXT_PICKLE, POINT_CELL, POINT_ORGAN, POINT_MACRO)
        ],
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
