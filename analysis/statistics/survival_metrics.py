"""Held-out censoring-aware survival metrics with explicit split contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ALLOWED_ROLE_PAIRS = {
    "screen": ("calibration", "screen"),
    "formal": ("validation", "test"),
}


class MetricContractError(ValueError):
    """Raised when an evaluation would violate the frozen metric contract."""


@dataclass(frozen=True)
class MetricTimeGrid:
    """Requested display times and right-open times used by survival libraries."""

    report_times: tuple[float, ...]
    report_effective_times: tuple[float, ...]
    ibs_times: tuple[float, ...]
    ibs_effective_times: tuple[float, ...]
    all_effective_times: tuple[float, ...]
    support_upper: float

    def effective_for(self, requested_time: float) -> float:
        for requested, effective in zip(self.report_times, self.report_effective_times):
            if np.isclose(requested, requested_time, rtol=0.0, atol=1e-10):
                return effective
        for requested, effective in zip(self.ibs_times, self.ibs_effective_times):
            if np.isclose(requested, requested_time, rtol=0.0, atol=1e-10):
                return effective
        raise MetricContractError(f"time {requested_time} is not in the metric grid")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MetricContractError(f"{table} missing required columns: {missing}")


def _finite_numeric(frame: pd.DataFrame, column: str, *, table: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise MetricContractError(f"{table}.{column} must contain finite numeric values")
    return values


def _validate_outcomes(frame: pd.DataFrame, *, table: str) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(frame, ("participant_id", "duration", "event"), table=table)
    if len(frame) < 3:
        raise MetricContractError(f"{table} requires at least three rows")
    identifiers = frame["participant_id"].astype("string")
    if identifiers.isna().any() or identifiers.str.strip().eq("").any():
        raise MetricContractError(f"{table}.participant_id contains null or empty values")
    duration = _finite_numeric(frame, "duration", table=table)
    event = _finite_numeric(frame, "event", table=table)
    if (duration <= 0.0).any():
        raise MetricContractError(f"{table}.duration must be strictly positive")
    if not set(np.unique(event)).issubset({0.0, 1.0}):
        raise MetricContractError(f"{table}.event must contain only 0/1")
    if event.sum() < 1:
        raise MetricContractError(f"{table}.event contains no events")
    return duration, event.astype(bool)


def validate_fit_score_partition(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    *,
    phase: str,
    role_col: str = "split_role",
    fold_col: str = "outer_fold",
) -> None:
    """Reject any role pairing other than calibration/screen or validation/test."""

    if phase not in ALLOWED_ROLE_PAIRS:
        raise MetricContractError(f"phase must be one of {sorted(ALLOWED_ROLE_PAIRS)}")
    fit_role, score_role = ALLOWED_ROLE_PAIRS[phase]
    for frame, expected, table in ((fit, fit_role, "fit"), (score, score_role, "score")):
        _require_columns(frame, ("participant_id", role_col), table=table)
        if frame["participant_id"].astype(str).duplicated().any():
            raise MetricContractError(f"{table} has duplicate participant_id rows")
        observed = set(frame[role_col].astype(str))
        if observed != {expected}:
            raise MetricContractError(
                f"{table}.{role_col} must contain only {expected!r} for phase={phase!r}; "
                f"observed={sorted(observed)}"
            )
        _validate_outcomes(frame, table=table)

    fit_ids = set(fit["participant_id"].astype(str))
    score_ids = set(score["participant_id"].astype(str))
    overlap = fit_ids.intersection(score_ids)
    if overlap:
        raise MetricContractError(f"fit/score participant overlap: {len(overlap)}")

    if fold_col in fit.columns or fold_col in score.columns:
        _require_columns(fit, (fold_col,), table="fit")
        _require_columns(score, (fold_col,), table="score")
        fit_fold_values = pd.to_numeric(fit[fold_col], errors="coerce")
        score_fold_values = pd.to_numeric(score[fold_col], errors="coerce")
        if fit_fold_values.isna().any() or score_fold_values.isna().any():
            raise MetricContractError(f"{fold_col} must contain finite fold IDs")
        fit_folds = set(fit_fold_values)
        score_folds = set(score_fold_values)
        if len(fit_folds) != 1 or fit_folds != score_folds:
            raise MetricContractError(
                f"fit/score must contain the same single {fold_col}; "
                f"fit={sorted(fit_folds)}, score={sorted(score_folds)}"
            )
        fold = next(iter(fit_folds))
        if not np.isfinite(fold) or int(fold) not in range(5) or float(fold) != int(fold):
            raise MetricContractError(f"{fold_col} must be an integer in 0..4")


def _effective_times(
    requested: Sequence[float],
    *,
    support_upper: float,
    support_lower: float,
    administrative_tolerance: float = 0.1,
) -> tuple[float, ...]:
    result: list[float] = []
    for raw_time in requested:
        time = float(raw_time)
        if not np.isfinite(time) or time <= 0.0:
            raise MetricContractError("metric times must be finite and positive")
        if time <= support_lower:
            raise MetricContractError(
                f"metric time {time:g} must exceed the earliest score follow-up {support_lower:g}"
            )
        if time > support_upper and not np.isclose(
            time, support_upper, rtol=0.0, atol=administrative_tolerance
        ):
            raise MetricContractError(
                f"metric time {time:g} exceeds common fit/score support {support_upper:g}"
            )
        # sksurv uses a right-open follow-up interval. Administrative 60-month
        # censoring is therefore evaluated at the representable instant before 60.
        effective = np.nextafter(support_upper, 0.0) if np.isclose(
            time, support_upper, rtol=0.0, atol=administrative_tolerance
        ) else time
        result.append(float(effective))
    if any(later <= earlier for earlier, later in zip(result, result[1:])):
        raise MetricContractError("metric times must be strictly increasing")
    return tuple(result)


def build_metric_time_grid(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    *,
    report_times: Sequence[float] = (12.0, 24.0, 60.0),
    ibs_times: Sequence[float] | None = None,
) -> MetricTimeGrid:
    """Create a time grid supported by both censoring-reference and score sets."""

    fit_duration, _ = _validate_outcomes(fit, table="fit")
    score_duration, _ = _validate_outcomes(score, table="score")
    support_upper = float(min(fit_duration.max(), score_duration.max()))
    support_lower = float(score_duration.min())
    if ibs_times is None:
        last = int(round(max(report_times)))
        ibs_times = tuple(float(value) for value in range(1, last + 1))
    report_tuple = tuple(float(value) for value in report_times)
    ibs_tuple = tuple(float(value) for value in ibs_times)
    report_effective = _effective_times(
        report_tuple,
        support_upper=support_upper,
        support_lower=support_lower,
    )
    ibs_effective = _effective_times(
        ibs_tuple,
        support_upper=support_upper,
        support_lower=support_lower,
    )
    all_effective = tuple(sorted(set(report_effective).union(ibs_effective)))
    return MetricTimeGrid(
        report_times=report_tuple,
        report_effective_times=report_effective,
        ibs_times=ibs_tuple,
        ibs_effective_times=ibs_effective,
        all_effective_times=all_effective,
        support_upper=support_upper,
    )


def _survival_array(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    from sksurv.util import Surv

    return Surv.from_arrays(event=event.astype(bool), time=duration.astype(np.float64))


def _probability_columns(
    survival_probability: np.ndarray,
    all_times: Sequence[float],
    selected_times: Sequence[float],
) -> np.ndarray:
    lookup = {float(value): index for index, value in enumerate(all_times)}
    try:
        indices = [lookup[float(value)] for value in selected_times]
    except KeyError as exc:
        raise MetricContractError(f"survival probability is missing time {exc.args[0]}") from exc
    return survival_probability[:, indices]


def _km_risk_at(
    duration: np.ndarray,
    event: np.ndarray,
    time: float,
) -> float:
    from sksurv.nonparametric import kaplan_meier_estimator

    event_times, survival = kaplan_meier_estimator(event.astype(bool), duration)
    positions = np.flatnonzero(event_times <= time)
    return float(0.0 if len(positions) == 0 else 1.0 - survival[positions[-1]])


def _calibration_at_time(
    score: pd.DataFrame,
    predicted_risk: np.ndarray,
    *,
    effective_time: float,
    report_time: float,
    n_bins: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    duration, event = _validate_outcomes(score, table="score")
    if n_bins < 2:
        raise MetricContractError("calibration n_bins must be at least 2")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(predicted_risk, quantiles))
    if len(edges) < 3:
        raise MetricContractError("predicted risk has insufficient variation for calibration bins")
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins = np.digitize(predicted_risk, edges[1:-1], right=True)
    rows: list[dict[str, float | int]] = []
    participant_errors = np.empty(len(score), dtype=np.float64)
    for bin_id in range(len(edges) - 1):
        selected = bins == bin_id
        if not selected.any():
            continue
        observed = _km_risk_at(duration[selected], event[selected], effective_time)
        mean_predicted = float(predicted_risk[selected].mean())
        absolute_error = abs(observed - mean_predicted)
        participant_errors[selected] = absolute_error
        rows.append(
            {
                "time_months": float(report_time),
                "bin": int(bin_id + 1),
                "n": int(selected.sum()),
                "events": int(event[selected].sum()),
                "mean_predicted_risk": mean_predicted,
                "km_observed_risk": observed,
                "absolute_calibration_error": float(absolute_error),
            }
        )
    overall_observed = _km_risk_at(duration, event, effective_time)
    overall_predicted = float(predicted_risk.mean())
    summary: dict[str, float | int] = {
        "time_months": float(report_time),
        "n_bins": int(len(rows)),
        "mean_predicted_risk": overall_predicted,
        "km_observed_risk": overall_observed,
        "observed_expected_ratio": float(
            overall_observed / overall_predicted if overall_predicted > 0.0 else np.nan
        ),
        "calibration_in_the_large": float(overall_observed - overall_predicted),
        "ici": float(participant_errors.mean()),
        "e50": float(np.quantile(participant_errors, 0.50)),
        "e90": float(np.quantile(participant_errors, 0.90)),
    }
    return summary, rows


def evaluate_censoring_aware_metrics(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    *,
    risk_score: Sequence[float],
    survival_probability: np.ndarray,
    time_grid: MetricTimeGrid,
    calibration_bins: int = 10,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compute Uno C, dynamic AUC, IPCW Brier/IBS, and KM calibration."""

    from sksurv.metrics import (
        brier_score,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )

    fit_duration, fit_event = _validate_outcomes(fit, table="fit")
    score_duration, score_event = _validate_outcomes(score, table="score")
    risk = np.asarray(risk_score, dtype=np.float64)
    probability = np.asarray(survival_probability, dtype=np.float64)
    expected_shape = (len(score), len(time_grid.all_effective_times))
    if risk.shape != (len(score),) or not np.isfinite(risk).all():
        raise MetricContractError(f"risk_score must have shape ({len(score)},) and be finite")
    if probability.shape != expected_shape or not np.isfinite(probability).all():
        raise MetricContractError(
            f"survival_probability must be finite with shape {expected_shape}; "
            f"observed={probability.shape}"
        )
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise MetricContractError("survival_probability must lie in [0, 1]")
    if (np.diff(probability, axis=1) > 1e-10).any():
        raise MetricContractError("survival_probability must be non-increasing over time")

    survival_fit = _survival_array(fit_duration, fit_event)
    survival_score = _survival_array(score_duration, score_event)
    horizon_effective = time_grid.report_effective_times[-1]
    uno = concordance_index_ipcw(
        survival_fit,
        survival_score,
        risk,
        tau=horizon_effective,
    )
    report_probability = _probability_columns(
        probability,
        time_grid.all_effective_times,
        time_grid.report_effective_times,
    )
    auc, mean_auc = cumulative_dynamic_auc(
        survival_fit,
        survival_score,
        risk,
        np.asarray(time_grid.report_effective_times),
    )
    _, brier = brier_score(
        survival_fit,
        survival_score,
        report_probability,
        np.asarray(time_grid.report_effective_times),
    )
    ibs_probability = _probability_columns(
        probability,
        time_grid.all_effective_times,
        time_grid.ibs_effective_times,
    )
    ibs = integrated_brier_score(
        survival_fit,
        survival_score,
        ibs_probability,
        np.asarray(time_grid.ibs_effective_times),
    )

    calibration_summaries: list[dict[str, float | int]] = []
    calibration_rows: list[dict[str, float | int]] = []
    for index, (report_time, effective_time) in enumerate(
        zip(time_grid.report_times, time_grid.report_effective_times)
    ):
        predicted_risk = 1.0 - report_probability[:, index]
        summary, rows = _calibration_at_time(
            score,
            predicted_risk,
            effective_time=effective_time,
            report_time=report_time,
            n_bins=calibration_bins,
        )
        calibration_summaries.append(summary)
        calibration_rows.extend(rows)

    report: dict[str, object] = {
        "n_fit": int(len(fit)),
        "n_score": int(len(score)),
        "fit_events": int(fit_event.sum()),
        "score_events": int(score_event.sum()),
        "uno_c_60m": float(uno[0]),
        "uno_concordant_pairs": int(uno[1]),
        "uno_discordant_pairs": int(uno[2]),
        "uno_tied_risk_pairs": int(uno[3]),
        "uno_tied_time_pairs": int(uno[4]),
        "dynamic_auc": {
            f"{time:g}m": float(value) for time, value in zip(time_grid.report_times, auc)
        },
        "mean_dynamic_auc": float(mean_auc),
        "ipcw_brier": {
            f"{time:g}m": float(value) for time, value in zip(time_grid.report_times, brier)
        },
        "ibs_interval_months": [float(time_grid.ibs_times[0]), float(time_grid.ibs_times[-1])],
        "integrated_brier_score": float(ibs),
        "calibration": calibration_summaries,
        "effective_time_note": (
            "A requested time equal to administrative follow-up is evaluated at the "
            "immediately preceding representable time to respect sksurv's right-open support."
        ),
    }
    return report, pd.DataFrame(calibration_rows)


def flatten_primary_metrics(report: Mapping[str, object]) -> dict[str, float]:
    """Return scalar metrics with unambiguous higher/lower-is-better names."""

    auc = report["dynamic_auc"]
    brier = report["ipcw_brier"]
    if not isinstance(auc, Mapping) or not isinstance(brier, Mapping):
        raise MetricContractError("invalid metric report")
    return {
        "uno_c_60m": float(report["uno_c_60m"]),
        "auc_12m": float(auc["12m"]),
        "auc_24m": float(auc["24m"]),
        "auc_60m": float(auc["60m"]),
        "brier_12m": float(brier["12m"]),
        "brier_24m": float(brier["24m"]),
        "brier_60m": float(brier["60m"]),
        "ibs": float(report["integrated_brier_score"]),
    }
