from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import pandas as pd

def _age_metric(metric: str, observed: np.ndarray, predicted: np.ndarray) -> float:
    from scipy.stats import spearmanr

    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < 3:
        raise ValueError("insufficient complete age rows")
    residual = p - y
    if metric == "mae":
        return float(np.mean(np.abs(residual)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(residual))))
    if metric == "pearson":
        if np.std(y) == 0 or np.std(p) == 0:
            raise ValueError("Pearson is singular")
        return float(np.corrcoef(y, p)[0, 1])
    if metric == "spearman":
        value = float(spearmanr(y, p).statistic)
        if not np.isfinite(value):
            raise ValueError("Spearman is singular")
        return value
    if metric == "r2":
        denominator = float(np.sum(np.square(y - np.mean(y))))
        if denominator == 0:
            raise ValueError("R2 is singular")
        return float(1.0 - np.sum(np.square(y - p)) / denominator)
    design = np.column_stack([np.ones(len(p)), p])
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("calibration regression is singular")
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    if metric == "calibration_intercept":
        return float(coefficient[0])
    if metric == "calibration_slope":
        return float(coefficient[1])
    raise ValueError(f"unknown age metric: {metric}")

def _cox_fit(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    *,
    robust_groups: np.ndarray | None = None,
    status_column: str = "event_indicator",
) -> Any:
    columns = ["observed_time_days", status_column, *covariates]
    complete = frame.loc[np.isfinite(frame[columns].to_numpy(dtype=np.float64)).all(axis=1)]
    if len(complete) < max(20, len(covariates) + 3):
        raise ValueError("insufficient complete Cox rows")
    if int(complete[status_column].sum()) == 0:
        raise ValueError("zero target events")
    exog = complete[list(covariates)].to_numpy(dtype=np.float64)
    if np.linalg.matrix_rank(exog) != len(covariates):
        raise ValueError("singular Cox design")
    groups = None
    if robust_groups is not None:
        groups = np.asarray(robust_groups)[complete.index.to_numpy()]
    try:
        from statsmodels.duration.hazard_regression import PHReg

        model = PHReg(
            complete["observed_time_days"].to_numpy(dtype=np.float64),
            exog,
            status=complete[status_column].to_numpy(dtype=np.int8),
            ties="efron",
        )
        result = model.fit(groups=groups, disp=0)
    except ModuleNotFoundError:
        from types import SimpleNamespace

        from lifelines import CoxPHFitter

        fit = complete[["observed_time_days", status_column, *covariates]].copy()
        cluster_col = None
        if groups is not None:
            cluster_col = "_cluster_id"
            fit[cluster_col] = groups
        model = CoxPHFitter()
        model.fit(
            fit,
            duration_col="observed_time_days",
            event_col=status_column,
            cluster_col=cluster_col,
            robust=cluster_col is not None,
            show_progress=False,
        )
        covariance = (
            model.variance_matrix_
            .reindex(index=covariates, columns=covariates)
            .to_numpy(dtype=np.float64)
        )
        event_residuals = model.compute_residuals(
            fit, kind="schoenfeld"
        ).reindex(columns=covariates)
        residuals = np.full((len(complete), len(covariates)), np.nan)
        positions = pd.Series(
            np.arange(len(complete), dtype=np.int64), index=complete.index
        )
        residual_positions = positions.loc[event_residuals.index].to_numpy(
            dtype=np.int64
        )
        residuals[residual_positions] = event_residuals.to_numpy(
            dtype=np.float64
        )
        result = SimpleNamespace(
            params=model.params_.reindex(covariates).to_numpy(dtype=np.float64),
            bse=model.standard_errors_.reindex(covariates).to_numpy(
                dtype=np.float64
            ),
            cov_params=lambda: covariance,
            schoenfeld_residuals=residuals,
        )
    if np.any(~np.isfinite(result.params)) or np.any(~np.isfinite(result.bse)):
        raise ValueError("nonfinite Cox fit")
    return complete, result

def _phreg_baseline_hazard_jumps(
    complete: pd.DataFrame,
    result: Any,
    covariates: Sequence[str],
    *,
    status_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct post-jump Breslow baseline hazards from full risk sets."""
    duration = complete["observed_time_days"].to_numpy(dtype=np.float64)
    status = complete[status_column].to_numpy(dtype=np.int8)
    design = complete[list(covariates)].to_numpy(dtype=np.float64)
    linear = design @ np.asarray(result.params, dtype=np.float64)
    relative_risk = np.exp(linear)
    if (
        len(duration) == 0
        or np.any(~np.isfinite(duration))
        or np.any(~np.isfinite(relative_risk))
        or np.any(relative_risk <= 0)
        or np.any((status != 0) & (status != 1))
    ):
        raise ValueError("invalid rows for PHReg baseline-hazard reconstruction")

    order = np.argsort(duration, kind="mergesort")
    sorted_time = duration[order]
    sorted_status = status[order]
    sorted_risk = relative_risk[order]
    points, starts, counts = np.unique(
        sorted_time, return_index=True, return_counts=True
    )
    risk_denominator = np.cumsum(sorted_risk[::-1], dtype=np.float64)[::-1][starts]
    failures = np.asarray(
        [sorted_status[start : start + count].sum() for start, count in zip(starts, counts)],
        dtype=np.float64,
    )
    event_mask = failures > 0
    event_times = points[event_mask].astype(np.float64, copy=False)
    jumps = failures[event_mask] / risk_denominator[event_mask]
    if (
        len(event_times) == 0
        or np.any(~np.isfinite(jumps))
        or np.any(jumps <= 0)
        or np.any(np.diff(event_times) <= 0)
    ):
        raise ValueError("invalid PHReg baseline-hazard jumps")
    return event_times, jumps

def _integrate_cause_specific_hazard_jumps(
    target_times: np.ndarray,
    target_jumps: np.ndarray,
    target_linear: np.ndarray,
    death_times: np.ndarray,
    death_jumps: np.ndarray,
    death_linear: np.ndarray,
    horizon_days: float,
) -> np.ndarray:
    """Integrate target CIF over explicit target and competing-death jumps."""
    target_times = np.asarray(target_times, dtype=np.float64)
    target_jumps = np.asarray(target_jumps, dtype=np.float64)
    death_times = np.asarray(death_times, dtype=np.float64)
    death_jumps = np.asarray(death_jumps, dtype=np.float64)
    target_linear = np.asarray(target_linear, dtype=np.float64)
    death_linear = np.asarray(death_linear, dtype=np.float64)
    if (
        target_times.ndim != 1
        or target_jumps.shape != target_times.shape
        or death_times.ndim != 1
        or death_jumps.shape != death_times.shape
        or target_linear.ndim != 1
        or death_linear.shape != target_linear.shape
        or not np.isfinite(horizon_days)
        or horizon_days < 0
        or np.any(~np.isfinite(target_times))
        or np.any(~np.isfinite(death_times))
        or np.any(~np.isfinite(target_jumps))
        or np.any(~np.isfinite(death_jumps))
        or np.any(target_jumps <= 0)
        or np.any(death_jumps <= 0)
        or np.any(np.diff(target_times) <= 0)
        or np.any(np.diff(death_times) <= 0)
        or np.any(~np.isfinite(target_linear))
        or np.any(~np.isfinite(death_linear))
    ):
        raise ValueError("invalid cause-specific hazard-jump inputs")
    target_risk = np.exp(target_linear)
    death_risk = np.exp(death_linear)
    if np.any(~np.isfinite(target_risk)) or np.any(~np.isfinite(death_risk)):
        raise ValueError("nonfinite cause-specific relative risk")

    event_times = np.unique(
        np.concatenate(
            [
                target_times[target_times <= horizon_days],
                death_times[death_times <= horizon_days],
            ]
        )
    )
    target_lookup = dict(zip(target_times.tolist(), target_jumps.tolist()))
    death_lookup = dict(zip(death_times.tolist(), death_jumps.tolist()))
    cif = np.zeros(len(target_linear), dtype=np.float64)
    survival = np.ones(len(target_linear), dtype=np.float64)
    for point in event_times:
        increment_target = target_lookup.get(float(point), 0.0) * target_risk
        increment_death = death_lookup.get(float(point), 0.0) * death_risk
        total = increment_target + increment_death
        event_probability = -np.expm1(-total)
        target_share = np.divide(
            increment_target,
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )
        cif += survival * event_probability * target_share
        survival *= np.exp(-total)
    return np.clip(cif, 0.0, 1.0)

def _cox_predictions(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    covariates: Sequence[str],
    horizon_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    complete, result = _cox_fit(training.reset_index(drop=True), covariates)
    test_x = testing[list(covariates)].to_numpy(dtype=np.float64)
    linear = test_x @ np.asarray(result.params, dtype=np.float64)
    times, jumps = _phreg_baseline_hazard_jumps(
        complete,
        result,
        covariates,
        status_column="event_indicator",
    )
    baseline = float(jumps[times <= horizon_days].sum())
    probability = 1.0 - np.exp(-baseline * np.exp(linear))
    return linear, np.clip(probability, 0.0, 1.0)

def _cause_specific_cif_predictions(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    covariates: Sequence[str],
    horizon_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict target CIF from target/death cause-specific Cox hazards."""
    complete, target_fit = _cox_fit(
        training.reset_index(drop=True), covariates, status_column="event_indicator"
    )
    test_x = testing[list(covariates)].to_numpy(dtype=np.float64)
    target_beta = np.asarray(target_fit.params, dtype=np.float64)
    target_linear = test_x @ target_beta

    target_times, target_jumps = _phreg_baseline_hazard_jumps(
        complete,
        target_fit,
        covariates,
        status_column="event_indicator",
    )
    if int(complete["competing_death_indicator"].sum()) > 0:
        death_complete, death_fit = _cox_fit(
            complete.reset_index(drop=True),
            covariates,
            status_column="competing_death_indicator",
        )
        death_beta = np.asarray(death_fit.params, dtype=np.float64)
        death_linear = test_x @ death_beta
        death_times, death_jumps = _phreg_baseline_hazard_jumps(
            death_complete,
            death_fit,
            covariates,
            status_column="competing_death_indicator",
        )
    else:
        death_linear = np.zeros(len(testing), dtype=np.float64)
        death_times = np.empty(0, dtype=np.float64)
        death_jumps = np.empty(0, dtype=np.float64)
    cif = _integrate_cause_specific_hazard_jumps(
        target_times,
        target_jumps,
        target_linear,
        death_times,
        death_jumps,
        death_linear,
        horizon_days,
    )
    return target_linear, np.clip(cif, 0.0, 1.0)

def _weighted_linear_calibration(
    outcome: np.ndarray, probability: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    y = np.asarray(outcome, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    design = np.column_stack([np.ones(valid.sum()), p[valid]])
    if valid.sum() < 4 or np.linalg.matrix_rank(design) != 2:
        raise ValueError("weighted calibration is singular")
    root_weight = np.sqrt(w[valid])
    coefficient = np.linalg.lstsq(
        design * root_weight[:, None], y[valid] * root_weight, rcond=None
    )[0]
    return float(coefficient[0]), float(coefficient[1])

def _survey_logistic_taylor(
    y: np.ndarray,
    design: np.ndarray,
    weight: np.ndarray,
    strata: np.ndarray,
    psu: np.ndarray,
    *,
    full_design_strata: np.ndarray,
    full_design_psu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    from scipy.special import expit

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if len(y) < x.shape[1] + 5 or len(np.unique(y)) != 2 or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("survey logistic design is not estimable")
    beta = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(100):
        probability = expit(np.clip(x @ beta, -30, 30))
        working_weight = w * probability * (1.0 - probability)
        hessian = x.T @ (working_weight[:, None] * x)
        score = x.T @ (w * (y - probability))
        if np.linalg.matrix_rank(hessian) != x.shape[1]:
            raise ValueError("survey logistic Hessian is singular")
        step = np.linalg.solve(hessian, score)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        raise ValueError("survey logistic did not converge")
    probability = expit(np.clip(x @ beta, -30, 30))
    bread_inverse = np.linalg.inv(x.T @ ((w * probability * (1 - probability))[:, None] * x))
    individual_score = x * (w * (y - probability))[:, None]
    cluster = pd.DataFrame(
        individual_score,
        columns=[f"s{index}" for index in range(x.shape[1])],
    )
    cluster["strata"] = strata
    cluster["psu"] = psu
    aggregate = cluster.groupby(["strata", "psu"], sort=False).sum().reset_index()
    score_columns = [f"s{index}" for index in range(x.shape[1])]
    full_clusters = pd.DataFrame(
        {"strata": full_design_strata, "psu": full_design_psu}
    ).dropna().drop_duplicates()
    aggregate = full_clusters.merge(
        aggregate,
        on=["strata", "psu"],
        how="left",
        validate="one_to_one",
    )
    aggregate[score_columns] = aggregate[score_columns].fillna(0.0)
    meat = _centered_psu_meat(
        aggregate[score_columns].to_numpy(dtype=np.float64),
        aggregate["strata"].to_numpy(),
    )
    covariance = bread_inverse @ meat @ bread_inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    design_df = int(len(full_clusters) - full_clusters["strata"].nunique())
    if design_df <= 0:
        raise ValueError("NHANES survey design has nonpositive design degrees of freedom")
    return beta, standard_error, design_df

def _centered_psu_meat(scores: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Taylor PSU meat with lonely PSUs centered on the grand PSU mean."""
    score_matrix = np.asarray(scores, dtype=np.float64)
    stratum_id = np.asarray(strata)
    if score_matrix.ndim != 2 or len(score_matrix) != len(stratum_id):
        raise ValueError("PSU score/stratum dimensions mismatch")
    grand_center = score_matrix.mean(axis=0, keepdims=True)
    meat = np.zeros((score_matrix.shape[1], score_matrix.shape[1]), dtype=np.float64)
    for stratum in pd.unique(stratum_id):
        stratum_scores = score_matrix[stratum_id == stratum]
        if len(stratum_scores) == 1:
            centered = stratum_scores - grand_center
            factor = 1.0
        else:
            centered = stratum_scores - stratum_scores.mean(axis=0, keepdims=True)
            factor = len(stratum_scores) / (len(stratum_scores) - 1.0)
        meat += factor * centered.T @ centered
    return meat

def _weighted_age_metric(
    metric: str, observed: np.ndarray, predicted: np.ndarray, weight: np.ndarray
) -> float:
    from scipy.stats import rankdata

    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[valid], p[valid], w[valid]
    if len(y) < 3:
        raise ValueError("insufficient weighted age rows")
    w = w / w.sum()
    residual = p - y
    if metric == "mae":
        return float(np.sum(w * np.abs(residual)))
    if metric == "rmse":
        return float(np.sqrt(np.sum(w * np.square(residual))))
    if metric == "spearman":
        y, p = rankdata(y), rankdata(p)
    if metric in {"pearson", "spearman"}:
        y_centered = y - np.sum(w * y)
        p_centered = p - np.sum(w * p)
        denominator = np.sqrt(
            np.sum(w * y_centered**2) * np.sum(w * p_centered**2)
        )
        if denominator <= 0:
            raise ValueError("weighted correlation singular")
        return float(np.sum(w * y_centered * p_centered) / denominator)
    if metric == "r2":
        denominator = float(np.sum(w * np.square(y - np.sum(w * y))))
        if denominator <= 0:
            raise ValueError("weighted R2 singular")
        return float(1.0 - np.sum(w * np.square(y - p)) / denominator)
    design = np.column_stack([np.ones(len(p)), p])
    root = np.sqrt(w)
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("weighted calibration singular")
    coefficient = np.linalg.lstsq(
        design * root[:, None], y * root, rcond=None
    )[0]
    return float(coefficient[0] if metric == "calibration_intercept" else coefficient[1])
