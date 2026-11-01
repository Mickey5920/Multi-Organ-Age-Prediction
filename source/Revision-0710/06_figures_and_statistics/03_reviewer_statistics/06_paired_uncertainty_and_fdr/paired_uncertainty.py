"""Reviewer-grade paired participant-bootstrap primitives for Revision-0710."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - formal environment locks numba.
    njit = None
    prange = range


ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)


class ContractError(ValueError):
    """Raised when an input violates the locked paired-analysis contract."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ContractError(f"{table} missing required columns: {missing}")


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str], *, table: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ContractError(f"{table}.{column} must contain finite numeric values")


def validate_paired_long_predictions(
    frame: pd.DataFrame,
    *,
    prediction_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate one paired held-out row per participant and canonical organ."""

    if not prediction_columns or len(set(prediction_columns)) != len(prediction_columns):
        raise ContractError("prediction_columns must be non-empty and unique")
    required = ("participant_id", "organ_system", "outer_fold", "actual_age", *prediction_columns)
    _require_columns(frame, required, table="paired_predictions")
    if frame.empty:
        raise ContractError("paired_predictions must not be empty")
    result = frame.loc[:, required].copy()
    invalid_pid = result["participant_id"].isna() | result["participant_id"].astype(str).str.strip().eq("")
    if invalid_pid.any():
        raise ContractError("paired_predictions.participant_id contains null or empty values")
    result["participant_id"] = result["participant_id"].astype(str).str.strip()
    result["organ_system"] = result["organ_system"].astype(str)
    if result.duplicated(["participant_id", "organ_system"]).any():
        raise ContractError("paired_predictions has duplicate participant-organ rows")
    _finite_numeric(result, ("outer_fold", "actual_age", *prediction_columns), table="paired_predictions")
    folds = pd.to_numeric(result["outer_fold"], errors="coerce")
    if not np.equal(folds, np.floor(folds)).all() or not folds.isin(range(5)).all():
        raise ContractError("paired_predictions.outer_fold must contain fold IDs 0..4")
    result["outer_fold"] = folds.astype(np.int8)
    fold_counts = result.groupby("participant_id", sort=False)["outer_fold"].nunique()
    if not fold_counts.eq(1).all():
        raise ContractError("each participant must belong to exactly one outer_fold")
    organ_sets = result.groupby("participant_id", sort=False)["organ_system"].agg(lambda x: frozenset(x))
    expected = frozenset(ORGANS)
    if not organ_sets.eq(expected).all():
        raise ContractError("each participant must have exactly the seven canonical organs")
    age_counts = result.groupby("participant_id", sort=False)["actual_age"].nunique()
    if not age_counts.eq(1).all():
        raise ContractError("actual_age must be constant across a participant's seven organs")
    return result.sort_values(["outer_fold", "participant_id", "organ_system"], kind="mergesort").reset_index(drop=True)


def build_stratified_draw_weights(
    participants: pd.DataFrame,
    *,
    replicate_ids: Sequence[int],
    seed: int,
) -> np.ndarray:
    """Draw deterministic participant multiplicities independently within outer folds."""

    _require_columns(participants, ("participant_id", "outer_fold"), table="participants")
    if participants.empty or participants["participant_id"].duplicated().any():
        raise ContractError("participants must contain unique participant_id rows")
    folds = pd.to_numeric(participants["outer_fold"], errors="coerce")
    if folds.isna().any() or not np.equal(folds, np.floor(folds)).all() or not folds.isin(range(5)).all():
        raise ContractError("participants.outer_fold must contain fold IDs 0..4")
    if len(set(int(item) for item in replicate_ids)) != len(replicate_ids):
        raise ContractError("replicate_ids must be unique")
    if any(int(item) < 0 for item in replicate_ids):
        raise ContractError("replicate_ids must be non-negative")
    fold_array = folds.to_numpy(dtype=np.int8)
    fold_indices = [np.flatnonzero(fold_array == fold) for fold in range(5)]
    if any(len(indices) == 0 for indices in fold_indices):
        raise ContractError("all five outer-fold strata must be non-empty")
    result = np.zeros((len(replicate_ids), len(participants)), dtype=np.int32)
    for row, replicate in enumerate(replicate_ids):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(replicate)]))
        for indices in fold_indices:
            local_draws = rng.integers(0, len(indices), size=len(indices))
            result[row, indices] = np.bincount(local_draws, minlength=len(indices))
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_registry_prediction_files(
    registry_path: Path,
    *,
    expected_method: str,
    verify_hash: bool = True,
) -> list[dict[str, object]]:
    """Resolve an explicit fivefold registry and optionally verify every prediction hash."""

    registry_path = Path(registry_path).expanduser().resolve()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid prediction registry: {exc}") from exc
    if payload.get("schema_version") != "revision0710_prediction_registry_v1":
        raise ContractError("unsupported prediction registry schema_version")
    if payload.get("method") != expected_method:
        raise ContractError(
            f"prediction registry method mismatch: expected={expected_method!r}, observed={payload.get('method')!r}"
        )
    entries = payload.get("prediction_files")
    if not isinstance(entries, list) or len(entries) != 10:
        raise ContractError("prediction registry must contain exactly ten validation/test files")
    resolved: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ContractError("prediction registry entries must be objects")
        fold = entry.get("outer_fold")
        role = entry.get("split_role")
        raw_path = entry.get("path")
        expected_sha = entry.get("sha256")
        rows = entry.get("rows")
        if fold not in range(5) or role not in {"validation", "test"}:
            raise ContractError("prediction registry has invalid fold or split role")
        if not isinstance(raw_path, str) or any(token in raw_path for token in ("*", "?", "[", "]")):
            raise ContractError("prediction registry paths must be explicit strings")
        path = Path(raw_path)
        if not path.is_absolute():
            path = registry_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise ContractError(f"prediction file does not exist: {path}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ContractError("prediction registry entry has invalid sha256")
        observed_sha = sha256_file(path) if verify_hash else None
        if verify_hash and observed_sha != expected_sha.lower():
            raise ContractError(
                f"prediction file sha256 mismatch: path={path}, expected={expected_sha.lower()}, observed={observed_sha}"
            )
        if not isinstance(rows, int) or rows <= 0:
            raise ContractError("prediction registry entry rows must be a positive integer")
        resolved.append(
            {
                "outer_fold": int(fold),
                "split_role": str(role),
                "rows": int(rows),
                "path": path,
                "sha256": expected_sha.lower(),
            }
        )
    resolved.sort(key=lambda item: (int(item["outer_fold"]), str(item["split_role"])))
    coverage = {(int(item["outer_fold"]), str(item["split_role"])) for item in resolved}
    expected_coverage = {(fold, role) for fold in range(5) for role in ("validation", "test")}
    if coverage != expected_coverage:
        raise ContractError("prediction registry does not cover each fold-role exactly once")
    return resolved


def participant_age_sufficient_statistics(
    frame: pd.DataFrame,
    *,
    prediction_columns: Sequence[str],
) -> pd.DataFrame:
    """Collapse seven organ rows to exact participant-level metric sufficient statistics."""

    work = validate_paired_long_predictions(frame, prediction_columns=prediction_columns)
    actual = work["actual_age"].to_numpy(dtype=np.float64)
    work["n_rows"] = 1.0
    work["sum_actual"] = actual
    work["sum_actual_sq"] = np.square(actual)
    aggregate_columns = ["n_rows", "sum_actual", "sum_actual_sq"]
    for prediction_column in prediction_columns:
        predicted = work[prediction_column].to_numpy(dtype=np.float64)
        work[f"{prediction_column}__sum_prediction"] = predicted
        work[f"{prediction_column}__sum_prediction_sq"] = np.square(predicted)
        work[f"{prediction_column}__sum_actual_prediction"] = actual * predicted
        work[f"{prediction_column}__sum_abs_error"] = np.abs(predicted - actual)
        aggregate_columns.extend(
            [
                f"{prediction_column}__sum_prediction",
                f"{prediction_column}__sum_prediction_sq",
                f"{prediction_column}__sum_actual_prediction",
                f"{prediction_column}__sum_abs_error",
            ]
        )
    participant_info = work.groupby("participant_id", sort=True).agg(outer_fold=("outer_fold", "first"))
    sums = work.groupby("participant_id", sort=True)[aggregate_columns].sum()
    result = participant_info.join(sums).reset_index()
    result.attrs["prediction_columns"] = tuple(prediction_columns)
    return result


def build_age_analysis_frame(
    ehr_oof: pd.DataFrame,
    baseline_predictions: dict[str, pd.DataFrame],
    *,
    ehr_method: str,
    exported_age_atol: float = 1e-5,
) -> pd.DataFrame:
    """Align strict held-out EHR and baseline predictions on exact participant-organ-fold keys."""

    ehr_required = (
        "participant_id",
        "organ_system",
        "outer_fold",
        "actual_age",
        "strict_healthy_reference_60m",
        "predicted_age_raw",
    )
    _require_columns(ehr_oof, ehr_required, table="ehr_oof")
    strict = pd.to_numeric(ehr_oof["strict_healthy_reference_60m"], errors="coerce")
    if strict.isna().any() or not set(strict.unique()).issubset({0, 1}):
        raise ContractError("ehr_oof.strict_healthy_reference_60m must contain only 0/1")
    if not baseline_predictions or ehr_method in baseline_predictions:
        raise ContractError("baseline_predictions must be non-empty and exclude ehr_method")
    keys = ["participant_id", "organ_system", "outer_fold"]
    work = ehr_oof.loc[strict.eq(1), [*keys, "actual_age", "predicted_age_raw"]].rename(
        columns={"predicted_age_raw": ehr_method}
    )
    work["participant_id"] = work["participant_id"].astype(str).str.strip()
    work["organ_system"] = work["organ_system"].astype(str)
    work = work.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if work.duplicated(keys).any():
        raise ContractError("ehr_oof has duplicate strict participant-organ-fold keys")
    for method, baseline in baseline_predictions.items():
        required = (*keys, "actual_age", "predicted_age")
        _require_columns(baseline, required, table=f"baseline:{method}")
        right = baseline.loc[:, required].copy()
        right["participant_id"] = right["participant_id"].astype(str).str.strip()
        right["organ_system"] = right["organ_system"].astype(str)
        right = right.sort_values(keys, kind="mergesort").reset_index(drop=True)
        if right.duplicated(keys).any():
            raise ContractError(f"baseline:{method} has duplicate participant-organ-fold keys")
        aligned = work.loc[:, keys].merge(right, on=keys, how="outer", indicator=True, validate="one_to_one")
        if len(aligned) != len(work) or not aligned["_merge"].eq("both").all():
            raise ContractError(f"baseline:{method} keys do not exactly match strict EHR keys")
        age_difference = np.abs(
            work["actual_age"].to_numpy(dtype=np.float64)
            - aligned["actual_age"].to_numpy(dtype=np.float64)
        )
        if not np.isfinite(age_difference).all() or (age_difference > exported_age_atol).any():
            raise ContractError(f"baseline:{method} actual_age disagrees with authoritative EHR age")
        predictions = pd.to_numeric(aligned["predicted_age"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(predictions).all():
            raise ContractError(f"baseline:{method}.predicted_age must be finite")
        work[method] = predictions
    return validate_paired_long_predictions(
        work,
        prediction_columns=(ehr_method, *baseline_predictions.keys()),
    )


def weighted_age_metrics(stats: pd.DataFrame, weights: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute pooled strict MAE, Pearson and R2 from participant multiplicities."""

    _require_columns(stats, ("n_rows", "sum_actual", "sum_actual_sq"), table="age_stats")
    weight = np.asarray(weights, dtype=np.float64)
    if weight.ndim != 1 or len(weight) != len(stats) or not np.isfinite(weight).all() or (weight < 0).any():
        raise ContractError("weights must be a finite non-negative vector matching age_stats")
    n_rows = float(weight @ stats["n_rows"].to_numpy(dtype=np.float64))
    if n_rows <= 1.0:
        raise ContractError("weighted age sample requires at least two rows")
    sum_actual = float(weight @ stats["sum_actual"].to_numpy(dtype=np.float64))
    sum_actual_sq = float(weight @ stats["sum_actual_sq"].to_numpy(dtype=np.float64))
    actual_ss = sum_actual_sq - sum_actual * sum_actual / n_rows
    if actual_ss <= 0.0:
        raise ContractError("weighted actual_age has no variation")
    suffix = "__sum_prediction"
    methods = sorted(column[: -len(suffix)] for column in stats.columns if column.endswith(suffix))
    if not methods:
        raise ContractError("age_stats contains no prediction sufficient statistics")
    output: dict[str, dict[str, float]] = {}
    for method in methods:
        required = (
            f"{method}__sum_prediction",
            f"{method}__sum_prediction_sq",
            f"{method}__sum_actual_prediction",
            f"{method}__sum_abs_error",
        )
        _require_columns(stats, required, table="age_stats")
        sum_prediction = float(weight @ stats[required[0]].to_numpy(dtype=np.float64))
        sum_prediction_sq = float(weight @ stats[required[1]].to_numpy(dtype=np.float64))
        sum_cross = float(weight @ stats[required[2]].to_numpy(dtype=np.float64))
        sum_abs = float(weight @ stats[required[3]].to_numpy(dtype=np.float64))
        prediction_ss = sum_prediction_sq - sum_prediction * sum_prediction / n_rows
        covariance = sum_cross - sum_actual * sum_prediction / n_rows
        if prediction_ss <= 0.0:
            raise ContractError(f"weighted {method} predictions have no variation")
        squared_error = sum_prediction_sq + sum_actual_sq - 2.0 * sum_cross
        output[method] = {
            "mae": float(sum_abs / n_rows),
            "pearson": float(covariance / np.sqrt(actual_ss * prediction_ss)),
            "r2": float(1.0 - squared_error / actual_ss),
        }
    return output


def paired_age_point_estimates(
    stats: pd.DataFrame,
    *,
    ehr_method: str,
    baseline_methods: Sequence[str],
) -> pd.DataFrame:
    """Compute paired point differences with positive values always favoring EHR-AGE."""

    metrics = weighted_age_metrics(stats, np.ones(len(stats), dtype=np.int8))
    if ehr_method not in metrics or any(method not in metrics for method in baseline_methods):
        raise ContractError("age_stats does not contain all requested methods")
    rows: list[dict[str, float | str]] = []
    for baseline_method in baseline_methods:
        for metric in ("mae", "pearson", "r2"):
            ehr_value = metrics[ehr_method][metric]
            baseline_value = metrics[baseline_method][metric]
            raw_delta = ehr_value - baseline_value
            rows.append(
                {
                    "baseline_method": baseline_method,
                    "metric": metric,
                    "direction": "lower" if metric == "mae" else "higher",
                    "ehr_value": ehr_value,
                    "baseline_value": baseline_value,
                    "raw_ehr_minus_baseline": raw_delta,
                    "directional_improvement": -raw_delta if metric == "mae" else raw_delta,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_age_from_stats(
    stats: pd.DataFrame,
    *,
    ehr_method: str,
    baseline_methods: Sequence[str],
    replicate_ids: Sequence[int],
    seed: int,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Run paired strict-age bootstrap from exact participant sufficient statistics."""

    if ehr_method in baseline_methods or len(set(baseline_methods)) != len(baseline_methods):
        raise ContractError("baseline_methods must be unique and exclude ehr_method")
    if batch_size < 1:
        raise ContractError("batch_size must be positive")
    methods = (ehr_method, *baseline_methods)
    required = ["participant_id", "outer_fold", "n_rows", "sum_actual", "sum_actual_sq"]
    for method in methods:
        required.extend(
            [
                f"{method}__sum_prediction",
                f"{method}__sum_prediction_sq",
                f"{method}__sum_actual_prediction",
                f"{method}__sum_abs_error",
            ]
        )
    _require_columns(stats, required, table="age_stats")
    participants = stats.loc[:, ["participant_id", "outer_fold"]]
    numeric_columns = required[2:]
    matrix = stats.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    column_index = {column: index for index, column in enumerate(numeric_columns)}
    ids = [int(item) for item in replicate_ids]
    rows: list[dict[str, float | int | str]] = []
    for batch_start in range(0, len(ids), batch_size):
        batch_ids = ids[batch_start : batch_start + batch_size]
        weights = build_stratified_draw_weights(participants, replicate_ids=batch_ids, seed=seed)
        weighted = weights.astype(np.float64) @ matrix
        for batch_row, replicate in enumerate(batch_ids):
            sums = weighted[batch_row]
            n_rows = sums[column_index["n_rows"]]
            sum_actual = sums[column_index["sum_actual"]]
            sum_actual_sq = sums[column_index["sum_actual_sq"]]
            actual_ss = sum_actual_sq - sum_actual * sum_actual / n_rows
            metrics: dict[str, dict[str, float]] = {}
            for method in methods:
                sum_prediction = sums[column_index[f"{method}__sum_prediction"]]
                sum_prediction_sq = sums[column_index[f"{method}__sum_prediction_sq"]]
                sum_cross = sums[column_index[f"{method}__sum_actual_prediction"]]
                prediction_ss = sum_prediction_sq - sum_prediction * sum_prediction / n_rows
                covariance = sum_cross - sum_actual * sum_prediction / n_rows
                squared_error = sum_prediction_sq + sum_actual_sq - 2.0 * sum_cross
                metrics[method] = {
                    "mae": float(sums[column_index[f"{method}__sum_abs_error"]] / n_rows),
                    "pearson": float(covariance / np.sqrt(actual_ss * prediction_ss)),
                    "r2": float(1.0 - squared_error / actual_ss),
                }
            for baseline_method in baseline_methods:
                for metric in ("mae", "pearson", "r2"):
                    ehr_value = metrics[ehr_method][metric]
                    baseline_value = metrics[baseline_method][metric]
                    raw_delta = ehr_value - baseline_value
                    improvement = -raw_delta if metric == "mae" else raw_delta
                    rows.append(
                        {
                            "replicate": int(replicate),
                            "baseline_method": baseline_method,
                            "metric": metric,
                            "ehr_value": ehr_value,
                            "baseline_value": baseline_value,
                            "raw_ehr_minus_baseline": raw_delta,
                            "directional_improvement": improvement,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_paired_replicates(
    replicates: pd.DataFrame,
    *,
    point_estimates: pd.DataFrame,
    group_columns: Sequence[str],
    family: str,
    confidence_level: float,
) -> pd.DataFrame:
    """Summarize percentile intervals and conservative empirical bootstrap p-values."""

    if not 0.0 < confidence_level < 1.0:
        raise ContractError("confidence_level must lie in (0, 1)")
    required = (*group_columns, "directional_improvement")
    _require_columns(replicates, required, table="bootstrap_replicates")
    _require_columns(point_estimates, required, table="point_estimates")
    if point_estimates.duplicated(list(group_columns)).any():
        raise ContractError("point_estimates contains duplicate groups")
    alpha = 1.0 - confidence_level
    summaries: list[dict[str, object]] = []
    point_lookup = point_estimates.set_index(list(group_columns))["directional_improvement"]
    grouped = replicates.groupby(list(group_columns), sort=True, dropna=False)
    for key, subset in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = pd.to_numeric(subset["directional_improvement"], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            raise ContractError(f"fewer than two valid bootstrap replicates for group {key_tuple}")
        try:
            point = float(point_lookup.loc[key_tuple if len(group_columns) > 1 else key_tuple[0]])
        except KeyError as exc:
            raise ContractError(f"missing point estimate for bootstrap group {key_tuple}") from exc
        lower_tail = (int(np.sum(values <= 0.0)) + 1.0) / (len(values) + 1.0)
        upper_tail = (int(np.sum(values >= 0.0)) + 1.0) / (len(values) + 1.0)
        row: dict[str, object] = {column: value for column, value in zip(group_columns, key_tuple)}
        row.update(
            {
                "family": family,
                "directional_improvement": point,
                "ci_lower_percentile": float(np.quantile(values, alpha / 2.0)),
                "ci_upper_percentile": float(np.quantile(values, 1.0 - alpha / 2.0)),
                "bootstrap_probability_improvement_gt_0": float(np.mean(values > 0.0)),
                "p_value_two_sided_bootstrap": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
                "valid_resamples": int(len(values)),
            }
        )
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    expected_groups = set(map(tuple, point_estimates.loc[:, group_columns].itertuples(index=False, name=None)))
    observed_groups = set(map(tuple, summary.loc[:, group_columns].itertuples(index=False, name=None)))
    if expected_groups != observed_groups:
        raise ContractError("bootstrap and point-estimate groups do not match exactly")
    return apply_bh_fdr(summary, p_col="p_value_two_sided_bootstrap", family_col="family")


def _fenwick_add(tree: np.ndarray, index: int, value: float) -> None:
    index += 1
    while index < len(tree):
        tree[index] += value
        index += index & -index


def _fenwick_prefix(tree: np.ndarray, stop: int) -> float:
    total = 0.0
    while stop > 0:
        total += tree[stop]
        stop -= stop & -stop
    return total


def weighted_harrell_c(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_event_time: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Harrell C for exact row expansion represented by non-negative integer weights."""

    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event)
    predicted = np.asarray(predicted_event_time, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if not (duration.ndim == event.ndim == predicted.ndim == weight.ndim == 1):
        raise ContractError("Harrell inputs must be one-dimensional")
    if not (len(duration) == len(event) == len(predicted) == len(weight)) or len(duration) < 2:
        raise ContractError("Harrell inputs must have equal length of at least two")
    if not np.isfinite(duration).all() or not np.isfinite(predicted).all() or not np.isfinite(weight).all():
        raise ContractError("Harrell inputs must be finite")
    if (duration <= 0.0).any() or (weight < 0.0).any() or not np.equal(weight, np.floor(weight)).all():
        raise ContractError("duration must be positive and weights non-negative integers")
    if not set(np.unique(event)).issubset({0, 1}):
        raise ContractError("event must contain only 0/1")
    ranks = np.searchsorted(np.unique(predicted), predicted)
    tree = np.zeros(int(ranks.max(initial=0)) + 2, dtype=np.float64)
    order = np.lexsort((-event.astype(np.int8), duration))
    pool_total = 0.0
    correct = 0.0
    tied = 0.0
    pairs = 0.0
    start = 0
    while start < len(order):
        time_value = duration[order[start]]
        stop = start + 1
        while stop < len(order) and duration[order[stop]] == time_value:
            stop += 1
        batch = order[start:stop]
        died = batch[event[batch].astype(bool)]
        censored = batch[~event[batch].astype(bool)]
        for indices, insert_after in ((died, True), (censored, False)):
            for index in indices:
                current_weight = weight[index]
                if current_weight == 0.0:
                    continue
                rank = int(ranks[index])
                less = _fenwick_prefix(tree, rank)
                equal = _fenwick_prefix(tree, rank + 1) - less
                pairs += pool_total * current_weight
                correct += less * current_weight
                tied += equal * current_weight
            if insert_after:
                for index in indices:
                    current_weight = weight[index]
                    if current_weight != 0.0:
                        _fenwick_add(tree, int(ranks[index]), current_weight)
                        pool_total += current_weight
        start = stop
    if pairs <= 0.0:
        raise ContractError("Harrell C has no admissible pairs")
    return float((correct + 0.5 * tied) / pairs)


def apply_sealed_cox_model(frame: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Return predicted event time (-log partial hazard) from a sealed Cox JSON."""

    path = Path(model_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid sealed Cox model: {exc}") from exc
    if payload.get("status") != "FITTED_AND_SEALED_BEFORE_OUTER_READ":
        raise ContractError("Cox model is not marked fitted and sealed before outer read")
    return apply_cox_payload(frame, payload)


def apply_cox_payload(frame: pd.DataFrame, payload: dict[str, object]) -> np.ndarray:
    """Return predicted event time from immutable Cox means, scales and coefficients."""

    covariates = payload.get("covariate_order")
    coefficients = np.asarray(payload.get("coefficients"), dtype=np.float64)
    means = payload.get("feature_means")
    scales = payload.get("feature_scales")
    if not isinstance(covariates, list) or not isinstance(means, dict) or not isinstance(scales, dict):
        raise ContractError("sealed Cox model has malformed feature metadata")
    if len(covariates) != len(coefficients) or not np.isfinite(coefficients).all():
        raise ContractError("sealed Cox model coefficient count mismatch")
    _require_columns(frame, covariates, table="cox_score_frame")
    design = np.empty((len(frame), len(covariates)), dtype=np.float64)
    for column_index, covariate in enumerate(covariates):
        values = pd.to_numeric(frame[covariate], errors="coerce").to_numpy(dtype=np.float64)
        mean = float(means[covariate])
        scale = float(scales[covariate])
        if not np.isfinite(values).all() or not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
            raise ContractError(f"invalid sealed Cox feature: {covariate}")
        design[:, column_index] = (values - mean) / scale
    return -(design @ coefficients)


def apply_frozen_linear_correction(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Apply published fold-organ linear-correction coefficients without fitting."""

    frame_required = ("participant_id", "outer_fold", "organ_system", "actual_age", "predicted_age")
    parameter_required = ("outer_fold", "organ_system", "intercept", "age_slope")
    _require_columns(frame, frame_required, table="predictions")
    _require_columns(parameters, parameter_required, table="correction_parameters")
    if parameters.duplicated(["outer_fold", "organ_system"]).any():
        raise ContractError("correction_parameters has duplicate fold-organ rows")
    work = frame.copy()
    work["_row_order"] = np.arange(len(work), dtype=np.int64)
    correction = parameters.loc[:, parameter_required].copy()
    result = work.merge(
        correction,
        on=["outer_fold", "organ_system"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_row_order", kind="mergesort")
    numeric_columns = ("actual_age", "predicted_age", "intercept", "age_slope")
    _finite_numeric(result, numeric_columns, table="corrected_predictions")
    actual = result["actual_age"].to_numpy(dtype=np.float64)
    raw_gap = result["predicted_age"].to_numpy(dtype=np.float64) - actual
    expected_gap = (
        result["intercept"].to_numpy(dtype=np.float64)
        + result["age_slope"].to_numpy(dtype=np.float64) * actual
    )
    result["raw_age_gap"] = raw_gap
    result["expected_age_gap"] = expected_gap
    result["corrected_age_gap"] = raw_gap - expected_gap
    return result.drop(columns=["_row_order", "intercept", "age_slope"]).reset_index(drop=True)


def fit_frozen_cox_nuisance(
    calibration: pd.DataFrame,
    *,
    covariates: Sequence[str],
    penalizer: float,
) -> dict[str, object]:
    """Fit one calibration-only Cox nuisance payload exactly once."""

    from lifelines import CoxPHFitter

    if not covariates or len(set(covariates)) != len(covariates):
        raise ContractError("covariates must be non-empty and unique")
    _require_columns(calibration, ("duration", "event", *covariates), table="cox_calibration")
    if len(calibration) < 3 or not np.isfinite(penalizer) or penalizer < 0.0:
        raise ContractError("invalid Cox calibration size or penalizer")
    _finite_numeric(calibration, ("duration", "event", *covariates), table="cox_calibration")
    duration = calibration["duration"].to_numpy(dtype=np.float64)
    event = calibration["event"].to_numpy(dtype=np.float64)
    if (duration <= 0.0).any() or not set(np.unique(event)).issubset({0.0, 1.0}) or event.sum() < 1.0:
        raise ContractError("invalid Cox calibration duration or event")
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    fit_frame = pd.DataFrame({"duration": duration, "event": event.astype(np.int8)})
    transformed: list[str] = []
    for covariate in covariates:
        values = calibration[covariate].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ContractError(f"cox_calibration.{covariate} has no variation")
        means[covariate] = mean
        scales[covariate] = scale
        transformed_name = f"z_{covariate}"
        transformed.append(transformed_name)
        fit_frame[transformed_name] = (values - mean) / scale
    model = CoxPHFitter(penalizer=float(penalizer))
    try:
        model.fit(
            fit_frame[["duration", "event", *transformed]],
            duration_col="duration",
            event_col="event",
            show_progress=False,
        )
    except Exception as exc:
        raise ContractError(f"calibration-only Cox fit failed: {exc}") from exc
    coefficients = [float(model.params_.loc[column]) for column in transformed]
    return {
        "schema_version": "revision0710.reviewer_statistics.frozen_baseline_cox_nuisance.v1",
        "status": "FITTED_ONCE_ON_FROZEN_CALIBRATION_BEFORE_BOOTSTRAP",
        "covariate_order": list(covariates),
        "feature_means": means,
        "feature_scales": scales,
        "penalizer": float(penalizer),
        "coefficients": coefficients,
        "fit_participant_count": int(len(calibration)),
        "fit_event_count": int(event.sum()),
        "log_likelihood": float(model.log_likelihood_),
    }


def _prepare_harrell_batch_arrays(
    duration: np.ndarray,
    event: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.float64)
    if duration.ndim != 1 or event.ndim != 1 or predictions.ndim != 2:
        raise ContractError("batched Harrell duration/event must be vectors and predictions a matrix")
    if len(duration) < 2 or len(duration) != len(event) or len(duration) != len(predictions):
        raise ContractError("batched Harrell inputs have incompatible lengths")
    if not np.isfinite(duration).all() or not np.isfinite(predictions).all() or (duration <= 0.0).any():
        raise ContractError("batched Harrell durations/predictions must be finite and durations positive")
    if not set(np.unique(event)).issubset({0, 1}) or not event.any():
        raise ContractError("batched Harrell event must contain 0/1 and at least one event")
    order = np.lexsort((-event, duration)).astype(np.int64)
    offsets = [0]
    batch_is_event: list[int] = []
    position = 0
    while position < len(order):
        stop = position + 1
        current_duration = duration[order[position]]
        while stop < len(order) and duration[order[stop]] == current_duration:
            stop += 1
        event_stop = position
        while event_stop < stop and event[order[event_stop]] == 1:
            event_stop += 1
        if event_stop > position:
            offsets.append(event_stop)
            batch_is_event.append(1)
        if event_stop < stop:
            offsets.append(stop)
            batch_is_event.append(0)
        position = stop
    ranks = np.empty(predictions.shape, dtype=np.int32)
    unique_counts = np.empty(predictions.shape[1], dtype=np.int32)
    for method in range(predictions.shape[1]):
        _, inverse = np.unique(predictions[:, method], return_inverse=True)
        ranks[:, method] = inverse.astype(np.int32)
        unique_counts[method] = int(inverse.max()) + 1
    return (
        order,
        np.asarray(offsets, dtype=np.int64),
        np.asarray(batch_is_event, dtype=np.int8),
        ranks,
        unique_counts,
    )


if njit is not None:

    @njit(cache=True, parallel=True)
    def _weighted_harrell_batch_numba(
        order: np.ndarray,
        offsets: np.ndarray,
        batch_is_event: np.ndarray,
        ranks: np.ndarray,
        unique_counts: np.ndarray,
        participant_index: np.ndarray,
        participant_weights: np.ndarray,
    ) -> np.ndarray:
        n_replicates = participant_weights.shape[0]
        n_methods = ranks.shape[1]
        result = np.empty((n_replicates, n_methods), dtype=np.float64)
        for job in prange(n_replicates * n_methods):
            replicate = job // n_methods
            method = job - replicate * n_methods
            tree = np.zeros(unique_counts[method] + 1, dtype=np.float64)
            pool_total = 0.0
            correct = 0.0
            tied = 0.0
            pairs = 0.0
            for batch in range(len(batch_is_event)):
                start = offsets[batch]
                stop = offsets[batch + 1]
                for position in range(start, stop):
                    row = order[position]
                    weight = participant_weights[replicate, participant_index[row]]
                    if weight == 0:
                        continue
                    rank = ranks[row, method]
                    cursor = rank
                    less = 0.0
                    while cursor > 0:
                        less += tree[cursor]
                        cursor -= cursor & -cursor
                    cursor = rank + 1
                    at_or_below = 0.0
                    while cursor > 0:
                        at_or_below += tree[cursor]
                        cursor -= cursor & -cursor
                    pairs += pool_total * weight
                    correct += less * weight
                    tied += (at_or_below - less) * weight
                if batch_is_event[batch] == 1:
                    for position in range(start, stop):
                        row = order[position]
                        weight = participant_weights[replicate, participant_index[row]]
                        if weight == 0:
                            continue
                        cursor = ranks[row, method] + 1
                        while cursor < len(tree):
                            tree[cursor] += weight
                            cursor += cursor & -cursor
                        pool_total += weight
            if pairs > 0.0:
                result[replicate, method] = (correct + 0.5 * tied) / pairs
            else:
                result[replicate, method] = np.nan
        return result


def weighted_harrell_c_batch(
    duration: np.ndarray,
    event: np.ndarray,
    predicted_event_times: np.ndarray,
    participant_index: np.ndarray,
    participant_weights: np.ndarray,
) -> np.ndarray:
    """Compute fixed-score Harrell C for multiple methods and participant draws."""

    if njit is None:
        raise ContractError("numba is required for formal batched risk bootstrap")
    participant_index = np.asarray(participant_index, dtype=np.int32)
    participant_weights = np.asarray(participant_weights, dtype=np.int32)
    if participant_index.ndim != 1 or participant_weights.ndim != 2:
        raise ContractError("participant_index must be a vector and participant_weights a matrix")
    if len(participant_index) != len(duration):
        raise ContractError("participant_index length must match risk rows")
    if (participant_index < 0).any() or participant_index.max(initial=-1) >= participant_weights.shape[1]:
        raise ContractError("participant_index falls outside participant_weights")
    if (participant_weights < 0).any():
        raise ContractError("participant_weights must be non-negative")
    prepared = _prepare_harrell_batch_arrays(duration, event, predicted_event_times)
    return _weighted_harrell_batch_numba(
        *prepared,
        participant_index,
        participant_weights,
    )


def aggregate_risk_replicates(cells: pd.DataFrame, *, ehr_method: str) -> pd.DataFrame:
    """Aggregate fold cells within organ and then organs to an unweighted macro estimand."""

    required = ("replicate", "fold", "organ_system", "method", "c_base", "c_plus")
    _require_columns(cells, required, table="risk_cells")
    if cells.duplicated(["replicate", "fold", "organ_system", "method"]).any():
        raise ContractError("risk_cells contains duplicate replicate-fold-organ-method rows")
    work = cells.copy()
    work["delta_c"] = work["c_plus"].to_numpy(dtype=np.float64) - work["c_base"].to_numpy(dtype=np.float64)
    observed_organs = set(work["organ_system"].astype(str))
    if observed_organs != set(ORGANS):
        raise ContractError("risk_cells must contain the seven canonical organs")
    organ = (
        work.groupby(["replicate", "organ_system", "method"], as_index=False, sort=True)
        .agg(delta_c=("delta_c", "mean"), fold_count=("fold", "nunique"))
    )
    if not organ["fold_count"].eq(5).all():
        raise ContractError("each replicate-organ-method must contain five folds")
    ehr_organ = organ.loc[organ["method"].eq(ehr_method), ["replicate", "organ_system", "delta_c"]].rename(
        columns={"delta_c": "ehr_delta_c"}
    )
    if len(ehr_organ) != organ["replicate"].nunique() * len(ORGANS):
        raise ContractError("risk_cells is missing the EHR method")
    organ = organ.merge(ehr_organ, on=["replicate", "organ_system"], how="left", validate="many_to_one")
    organ["ehr_minus_baseline_delta_c"] = organ["ehr_delta_c"] - organ["delta_c"]
    organ["level"] = "organ"
    macro = (
        organ.groupby(["replicate", "method"], as_index=False, sort=True)
        .agg(delta_c=("delta_c", "mean"), ehr_delta_c=("ehr_delta_c", "mean"), organ_count=("organ_system", "nunique"))
    )
    if not macro["organ_count"].eq(len(ORGANS)).all():
        raise ContractError("each replicate-method macro requires seven organs")
    macro["ehr_minus_baseline_delta_c"] = macro["ehr_delta_c"] - macro["delta_c"]
    macro["level"] = "macro"
    macro["organ_system"] = "__macro__"
    columns = (
        "replicate",
        "level",
        "organ_system",
        "method",
        "delta_c",
        "ehr_delta_c",
        "ehr_minus_baseline_delta_c",
    )
    return pd.concat([organ.loc[:, columns], macro.loc[:, columns]], ignore_index=True)


def apply_bh_fdr(frame: pd.DataFrame, *, p_col: str, family_col: str) -> pd.DataFrame:
    """Apply Benjamini-Hochberg correction independently within named families."""

    _require_columns(frame, (p_col, family_col), table="hypotheses")
    result = frame.copy()
    p_values = pd.to_numeric(result[p_col], errors="coerce")
    if p_values.isna().any() or ((p_values < 0.0) | (p_values > 1.0)).any():
        raise ContractError(f"{p_col} must lie in [0, 1]")
    result["q_value_bh"] = np.nan
    for _, indices in result.groupby(family_col, sort=True, dropna=False).groups.items():
        indices = np.asarray(list(indices), dtype=np.int64)
        values = p_values.iloc[indices].to_numpy(dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        ranked = values[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.minimum(adjusted, 1.0)
        result.iloc[indices, result.columns.get_loc("q_value_bh")] = restored
    return result
