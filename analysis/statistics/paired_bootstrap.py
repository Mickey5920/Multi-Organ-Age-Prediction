"""Deterministic paired participant-cluster bootstrap interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .survival_metrics import MetricContractError


StatisticFunction = Callable[[pd.DataFrame], Mapping[str, float]]


@dataclass(frozen=True)
class PairedBootstrapResult:
    summary: pd.DataFrame
    replicates: pd.DataFrame
    n_resamples_requested: int
    n_resamples_valid: int
    seed: int


def _aligned_frames(
    baseline: pd.DataFrame,
    augmented: pd.DataFrame,
    key_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not key_columns or len(set(key_columns)) != len(key_columns):
        raise MetricContractError("key_columns must be non-empty and unique")
    for frame, table in ((baseline, "baseline"), (augmented, "augmented")):
        missing = sorted(set(key_columns).difference(frame.columns))
        if missing:
            raise MetricContractError(f"{table} missing alignment keys: {missing}")
        if frame.duplicated(list(key_columns)).any():
            raise MetricContractError(f"{table} has duplicate alignment keys")
    left = baseline.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    right = augmented.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    if len(left) != len(right) or not left.loc[:, key_columns].equals(right.loc[:, key_columns]):
        raise MetricContractError("baseline and augmented rows are not exactly paired")
    return left, right


def _cluster_row_map(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    strata_cols: Sequence[str],
) -> dict[tuple[object, ...], tuple[np.ndarray, tuple[np.ndarray, ...]]]:
    missing = sorted({cluster_col, *strata_cols}.difference(frame.columns))
    if missing:
        raise MetricContractError(f"bootstrap frame missing columns: {missing}")
    cluster_groups = frame.groupby(cluster_col, sort=True, dropna=False).indices
    strata: dict[tuple[object, ...], list[tuple[object, np.ndarray]]] = {}
    for cluster, row_indices in cluster_groups.items():
        indices = np.asarray(row_indices, dtype=np.int64)
        if strata_cols:
            values = frame.iloc[indices][list(strata_cols)].drop_duplicates()
            if len(values) != 1:
                raise MetricContractError(
                    f"cluster {cluster!r} belongs to more than one bootstrap stratum"
                )
            stratum = tuple(values.iloc[0].tolist())
        else:
            stratum = ("__all__",)
        strata.setdefault(stratum, []).append((cluster, indices))
    result: dict[tuple[object, ...], tuple[np.ndarray, tuple[np.ndarray, ...]]] = {}
    for stratum, items in sorted(strata.items(), key=lambda pair: tuple(map(str, pair[0]))):
        items.sort(key=lambda pair: str(pair[0]))
        clusters = np.asarray([item[0] for item in items], dtype=object)
        rows = tuple(item[1] for item in items)
        result[stratum] = (clusters, rows)
    return result


def _draw_rows(
    strata: dict[tuple[object, ...], tuple[np.ndarray, tuple[np.ndarray, ...]]],
    rng: np.random.Generator,
) -> np.ndarray:
    selected_parts: list[np.ndarray] = []
    for _, rows in strata.values():
        draws = rng.integers(0, len(rows), size=len(rows))
        selected_parts.extend(rows[index] for index in draws)
    return np.concatenate(selected_parts)


def paired_participant_bootstrap(
    baseline: pd.DataFrame,
    augmented: pd.DataFrame,
    *,
    statistic: StatisticFunction,
    improvement_direction: Mapping[str, str],
    key_columns: Sequence[str] = ("participant_id", "organ_system", "outer_fold"),
    cluster_col: str = "participant_id",
    strata_cols: Sequence[str] = ("outer_fold",),
    n_resamples: int = 1000,
    seed: int = 710,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Bootstrap paired predictions, sampling a participant's rows as one cluster."""

    if n_resamples < 2:
        raise MetricContractError("n_resamples must be at least 2")
    if not 0.0 < confidence_level < 1.0:
        raise MetricContractError("confidence_level must lie in (0, 1)")
    left, right = _aligned_frames(baseline, augmented, key_columns)
    strata = _cluster_row_map(left, cluster_col=cluster_col, strata_cols=strata_cols)
    for column in strata_cols:
        if not left[column].equals(right[column]):
            raise MetricContractError(f"paired frames disagree on bootstrap stratum {column}")

    point_baseline = dict(statistic(left))
    point_augmented = dict(statistic(right))
    metric_names = sorted(set(point_baseline).intersection(point_augmented))
    if set(metric_names) != set(improvement_direction):
        raise MetricContractError(
            "improvement_direction must name exactly the metrics returned by statistic"
        )
    for metric, direction in improvement_direction.items():
        if direction not in {"higher", "lower"}:
            raise MetricContractError(f"invalid improvement direction for {metric}: {direction}")

    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, float | int]] = []
    for replicate in range(n_resamples):
        row_indices = _draw_rows(strata, rng)
        baseline_values = dict(statistic(left.iloc[row_indices].reset_index(drop=True)))
        augmented_values = dict(statistic(right.iloc[row_indices].reset_index(drop=True)))
        for metric in metric_names:
            base_value = float(baseline_values[metric])
            augmented_value = float(augmented_values[metric])
            if not np.isfinite(base_value) or not np.isfinite(augmented_value):
                continue
            raw_delta = augmented_value - base_value
            improvement = raw_delta if improvement_direction[metric] == "higher" else -raw_delta
            replicate_rows.append(
                {
                    "replicate": int(replicate),
                    "metric": metric,
                    "baseline": base_value,
                    "augmented": augmented_value,
                    "raw_augmented_minus_baseline": raw_delta,
                    "directional_improvement": improvement,
                }
            )
    replicates = pd.DataFrame(replicate_rows)
    if replicates.empty:
        raise MetricContractError("all bootstrap metric evaluations were non-finite")
    valid_per_metric = replicates.groupby("metric")["replicate"].nunique()
    if (valid_per_metric < max(2, int(np.ceil(0.9 * n_resamples)))).any():
        raise MetricContractError(
            "fewer than 90% valid paired bootstrap replicates for at least one metric"
        )

    alpha = 1.0 - confidence_level
    summaries: list[dict[str, float | int | str]] = []
    for metric in metric_names:
        subset = replicates.loc[replicates["metric"].eq(metric)]
        improvement = subset["directional_improvement"].to_numpy(dtype=np.float64)
        point_raw_delta = float(point_augmented[metric] - point_baseline[metric])
        point_improvement = (
            point_raw_delta if improvement_direction[metric] == "higher" else -point_raw_delta
        )
        summaries.append(
            {
                "metric": metric,
                "direction": improvement_direction[metric],
                "baseline": float(point_baseline[metric]),
                "augmented": float(point_augmented[metric]),
                "raw_augmented_minus_baseline": point_raw_delta,
                "directional_improvement": point_improvement,
                "ci_lower": float(np.quantile(improvement, alpha / 2.0)),
                "ci_upper": float(np.quantile(improvement, 1.0 - alpha / 2.0)),
                "bootstrap_probability_improvement_gt_0": float(np.mean(improvement > 0.0)),
                "valid_resamples": int(len(subset)),
            }
        )
    return PairedBootstrapResult(
        summary=pd.DataFrame(summaries),
        replicates=replicates,
        n_resamples_requested=int(n_resamples),
        n_resamples_valid=int(valid_per_metric.min()),
        seed=int(seed),
    )
