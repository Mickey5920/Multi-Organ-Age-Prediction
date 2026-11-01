"""Censoring-aware calibration, reclassification, and decision-curve metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .survival_metrics import MetricContractError


@dataclass(frozen=True)
class IPCWBinaryState:
    event_by_horizon: np.ndarray
    event_free_at_horizon: np.ndarray
    case_weight: np.ndarray
    control_weight: np.ndarray
    censoring_survival_horizon: float
    max_weight: float
    effective_sample_size: float


def _outcomes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted({"duration", "event"}.difference(frame.columns))
    if missing:
        raise MetricContractError(f"outcome frame missing columns: {missing}")
    duration = pd.to_numeric(frame["duration"], errors="coerce").to_numpy(dtype=np.float64)
    event = pd.to_numeric(frame["event"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(duration).all() or (duration <= 0.0).any():
        raise MetricContractError("duration must be finite and strictly positive")
    if not np.isfinite(event).all() or not set(np.unique(event)).issubset({0.0, 1.0}):
        raise MetricContractError("event must contain finite 0/1 values")
    return duration, event.astype(bool)


def _survival_array(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    from sksurv.util import Surv

    return Surv.from_arrays(event=event.astype(bool), time=duration.astype(np.float64))


def harrell_c_index(frame: pd.DataFrame, risk_score: Sequence[float]) -> float:
    from sksurv.metrics import concordance_index_censored

    duration, event = _outcomes(frame)
    risk = np.asarray(risk_score, dtype=np.float64)
    if risk.shape != duration.shape or not np.isfinite(risk).all():
        raise MetricContractError("risk score shape or values are invalid")
    return float(concordance_index_censored(event, duration, risk)[0])


def ipcw_binary_state(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    *,
    horizon: float,
) -> IPCWBinaryState:
    """Build validation-fitted IPCW case/control contributions at one horizon."""

    from sksurv.nonparametric import CensoringDistributionEstimator

    fit_duration, fit_event = _outcomes(fit)
    score_duration, score_event = _outcomes(score)
    censoring = CensoringDistributionEstimator().fit(
        _survival_array(fit_duration, fit_event)
    )
    event_by = score_event & (score_duration <= horizon)
    event_free = score_duration > horizon
    g_horizon = float(censoring.predict_proba(np.asarray([horizon]))[0])
    if not np.isfinite(g_horizon) or g_horizon <= 0.0:
        raise MetricContractError("validation censoring survival is unsupported at the horizon")
    case_weight = np.zeros(len(score), dtype=np.float64)
    control_weight = np.zeros(len(score), dtype=np.float64)
    if event_by.any():
        g_event = np.asarray(censoring.predict_proba(score_duration[event_by]), dtype=np.float64)
        if not np.isfinite(g_event).all() or (g_event <= 0.0).any():
            raise MetricContractError("validation censoring survival is zero at an event time")
        case_weight[event_by] = 1.0 / g_event
    control_weight[event_free] = 1.0 / g_horizon
    combined = case_weight + control_weight
    positive = combined > 0.0
    ess = float(combined.sum() ** 2 / np.square(combined).sum()) if positive.any() else 0.0
    return IPCWBinaryState(
        event_by_horizon=event_by,
        event_free_at_horizon=event_free,
        case_weight=case_weight,
        control_weight=control_weight,
        censoring_survival_horizon=g_horizon,
        max_weight=float(combined.max(initial=0.0)),
        effective_sample_size=ess,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0.0 else np.nan


def ipcw_reclassification(
    old_risk: Sequence[float],
    new_risk: Sequence[float],
    state: IPCWBinaryState,
) -> dict[str, float]:
    old = np.asarray(old_risk, dtype=np.float64)
    new = np.asarray(new_risk, dtype=np.float64)
    if old.shape != new.shape or old.shape != state.case_weight.shape:
        raise MetricContractError("reclassification arrays are not aligned")
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise MetricContractError("reclassification risks must be finite")
    old_case = _weighted_mean(old, state.case_weight)
    new_case = _weighted_mean(new, state.case_weight)
    old_control = _weighted_mean(old, state.control_weight)
    new_control = _weighted_mean(new, state.control_weight)
    up = (new > old).astype(np.float64)
    down = (new < old).astype(np.float64)
    event_component = _weighted_mean(up - down, state.case_weight)
    nonevent_component = _weighted_mean(down - up, state.control_weight)
    return {
        "ipcw_idi_60m": float((new_case - old_case) - (new_control - old_control)),
        "ipcw_continuous_nri_60m": float(event_component + nonevent_component),
        "ipcw_nri_event_component_60m": float(event_component),
        "ipcw_nri_nonevent_component_60m": float(nonevent_component),
        "mean_risk_old_events_60m": old_case,
        "mean_risk_new_events_60m": new_case,
        "mean_risk_old_nonevents_60m": old_control,
        "mean_risk_new_nonevents_60m": new_control,
    }


def ipcw_decision_curve(
    risk: Sequence[float],
    state: IPCWBinaryState,
    *,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    risk_array = np.asarray(risk, dtype=np.float64)
    if risk_array.shape != state.case_weight.shape or not np.isfinite(risk_array).all():
        raise MetricContractError("decision-curve risk is not aligned and finite")
    n = float(len(risk_array))
    prevalence = float(state.case_weight.sum() / n)
    nonevent = float(state.control_weight.sum() / n)
    rows: list[dict[str, float]] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not 0.0 < threshold < 1.0:
            raise MetricContractError("decision-curve thresholds must lie in (0, 1)")
        selected = risk_array >= threshold
        tp = float(state.case_weight[selected].sum() / n)
        fp = float(state.control_weight[selected].sum() / n)
        odds = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "net_benefit": float(tp - fp * odds),
                "treat_all_net_benefit": float(prevalence - nonevent * odds),
                "treat_none_net_benefit": 0.0,
                "ipcw_event_prevalence": prevalence,
            }
        )
    return pd.DataFrame(rows)


def _km_risk(duration: np.ndarray, event: np.ndarray, horizon: float) -> float:
    from sksurv.nonparametric import kaplan_meier_estimator

    time, survival = kaplan_meier_estimator(event.astype(bool), duration)
    position = np.flatnonzero(time <= horizon)
    return float(0.0 if len(position) == 0 else 1.0 - survival[position[-1]])


def ipcw_logistic_calibration(
    risk: np.ndarray,
    state: IPCWBinaryState,
) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    eps = 1e-8
    logit = np.log(np.clip(risk, eps, 1.0 - eps) / np.clip(1.0 - risk, eps, 1.0))
    y = state.event_by_horizon.astype(np.float64)
    weight = state.case_weight + state.control_weight
    selected = weight > 0.0
    x = logit[selected]
    y = y[selected]
    weight = weight[selected]

    if len(np.unique(y)) < 2 or np.ptp(x) <= 0.0:
        return np.nan, np.nan
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-10,
    )
    model.fit(x.reshape(-1, 1), y.astype(np.int8), sample_weight=weight)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    if not np.isfinite(intercept) or not np.isfinite(slope):
        return np.nan, np.nan
    return intercept, slope


def fixed_bin_calibration(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    *,
    fit_risk: Sequence[float],
    score_risk: Sequence[float],
    horizon: float,
    report_time: float | None = None,
    n_bins: int = 10,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate test calibration using bin boundaries frozen on validation risk."""

    fit_risk_array = np.asarray(fit_risk, dtype=np.float64)
    score_risk_array = np.asarray(score_risk, dtype=np.float64)
    if len(fit_risk_array) != len(fit) or len(score_risk_array) != len(score):
        raise MetricContractError("calibration risks are not aligned")
    if not np.isfinite(fit_risk_array).all() or not np.isfinite(score_risk_array).all():
        raise MetricContractError("calibration risks must be finite")
    if n_bins < 2:
        raise MetricContractError("calibration requires at least two bins")
    label_time = float(horizon if report_time is None else report_time)
    edges = np.unique(np.quantile(fit_risk_array, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        raise MetricContractError("validation risk has insufficient variation for calibration")
    inner_edges = edges[1:-1]
    bins = np.digitize(score_risk_array, inner_edges, right=True)
    duration, event = _outcomes(score)
    participant_error = np.full(len(score), np.nan, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for bin_id in range(len(edges) - 1):
        selected = bins == bin_id
        if not selected.any():
            continue
        observed = _km_risk(duration[selected], event[selected], horizon)
        predicted = float(score_risk_array[selected].mean())
        error = abs(observed - predicted)
        participant_error[selected] = error
        rows.append(
            {
                "time_months": label_time,
                "bin": int(bin_id + 1),
                "n": int(selected.sum()),
                "events": int(event[selected].sum()),
                "validation_risk_lower": float(edges[bin_id]),
                "validation_risk_upper": float(edges[bin_id + 1]),
                "mean_predicted_risk": predicted,
                "km_observed_risk": observed,
                "absolute_calibration_error": float(error),
            }
        )
    if np.isnan(participant_error).any():
        raise MetricContractError("at least one score row was not assigned a calibration bin")
    observed = _km_risk(duration, event, horizon)
    predicted = float(score_risk_array.mean())
    state = ipcw_binary_state(fit, score, horizon=horizon)
    intercept, slope = ipcw_logistic_calibration(score_risk_array, state)
    summary: dict[str, float | int] = {
        "time_months": label_time,
        "n_bins": int(len(rows)),
        "mean_predicted_risk": predicted,
        "km_observed_risk": observed,
        "observed_expected_ratio": float(observed / predicted) if predicted > 0.0 else np.nan,
        "calibration_in_the_large_risk_difference": float(observed - predicted),
        "ipcw_logistic_calibration_intercept": intercept,
        "ipcw_logistic_calibration_slope": slope,
        "ici": float(participant_error.mean()),
        "e50": float(np.quantile(participant_error, 0.50)),
        "e90": float(np.quantile(participant_error, 0.90)),
    }
    return summary, pd.DataFrame(rows)
