"""Validation-fitted Cox scoring for held-out censoring-aware evaluation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .survival_metrics import (
    MetricContractError,
    MetricTimeGrid,
    validate_fit_score_partition,
)


@dataclass(frozen=True)
class CoxScoreBundle:
    risk_score: np.ndarray
    survival_probability: np.ndarray
    fit_risk_score: np.ndarray
    fit_survival_probability: np.ndarray
    times: tuple[float, ...]
    coefficients: dict[str, float]
    fit_means: dict[str, float]
    fit_scales: dict[str, float]
    dropped_constant_covariates: tuple[str, ...]
    penalizer: float


def _model_ready_covariates(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    *,
    table: str,
) -> pd.DataFrame:
    missing = sorted(set(covariates).difference(frame.columns))
    if missing:
        raise MetricContractError(f"{table} missing covariates: {missing}")
    result = pd.DataFrame(index=frame.index)
    for column in covariates:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise MetricContractError(
                f"{table}.{column} is not model-ready; imputation and missing indicators "
                "must be fitted on the allowed fit partition before Cox scoring"
            )
        result[column] = values
    return result


def fit_lifelines_cox_scores(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    *,
    covariates: Sequence[str],
    time_grid: MetricTimeGrid,
    phase: str,
    penalizer: float = 0.1,
) -> CoxScoreBundle:
    """Fit standardization and Cox coefficients on fit, then score held-out rows."""

    if not covariates or len(set(covariates)) != len(covariates):
        raise MetricContractError("covariates must be a non-empty sequence without duplicates")
    if not np.isfinite(penalizer) or penalizer < 0.0:
        raise MetricContractError("penalizer must be finite and non-negative")
    validate_fit_score_partition(fit, score, phase=phase)
    x_fit = _model_ready_covariates(fit, covariates, table="fit")
    x_score = _model_ready_covariates(score, covariates, table="score")

    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    dropped: list[str] = []
    transformed_fit = pd.DataFrame(index=np.arange(len(fit)))
    transformed_score = pd.DataFrame(index=np.arange(len(score)))
    retained: list[str] = []
    for column in covariates:
        mean = float(x_fit[column].mean())
        scale = float(x_fit[column].std(ddof=0))
        means[column] = mean
        scales[column] = scale
        if not np.isfinite(scale) or scale <= 0.0:
            dropped.append(column)
            continue
        transformed = f"z__{column}"
        transformed_fit[transformed] = (x_fit[column].to_numpy() - mean) / scale
        transformed_score[transformed] = (x_score[column].to_numpy() - mean) / scale
        retained.append(transformed)
    if not retained:
        raise MetricContractError("all Cox covariates are constant in the fit partition")

    model_frame = transformed_fit.copy()
    model_frame["duration"] = pd.to_numeric(fit["duration"], errors="raise").to_numpy(dtype=np.float64)
    model_frame["event"] = pd.to_numeric(fit["event"], errors="raise").to_numpy(dtype=np.int8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from lifelines import CoxPHFitter

    model = CoxPHFitter(penalizer=float(penalizer))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(model_frame, duration_col="duration", event_col="event", show_progress=False)
        fit_risk = model.predict_log_partial_hazard(transformed_fit[retained]).to_numpy(dtype=np.float64)
        risk = model.predict_log_partial_hazard(transformed_score[retained]).to_numpy(dtype=np.float64)
        fit_survival = model.predict_survival_function(
            transformed_fit[retained],
            times=np.asarray(time_grid.all_effective_times),
        ).T.to_numpy(dtype=np.float64)
        survival = model.predict_survival_function(
            transformed_score[retained],
            times=np.asarray(time_grid.all_effective_times),
        ).T.to_numpy(dtype=np.float64)
    except Exception as exc:
        raise MetricContractError(f"held-out Cox scoring failed: {exc}") from exc
    if (
        not np.isfinite(fit_risk).all()
        or not np.isfinite(risk).all()
        or not np.isfinite(fit_survival).all()
        or not np.isfinite(survival).all()
    ):
        raise MetricContractError("held-out Cox scoring returned non-finite values")
    coefficient_map = {
        (str(name)[3:] if str(name).startswith("z__") else str(name)): float(value)
        for name, value in model.params_.items()
    }
    return CoxScoreBundle(
        risk_score=risk,
        survival_probability=survival,
        fit_risk_score=fit_risk,
        fit_survival_probability=fit_survival,
        times=time_grid.all_effective_times,
        coefficients=coefficient_map,
        fit_means=means,
        fit_scales=scales,
        dropped_constant_covariates=tuple(dropped),
        penalizer=float(penalizer),
    )
