"""Leakage-aware downstream survival utilities for Revision-0710.

The functions here estimate new association models. They never modify the
frozen EHR-AGE predictions or their sealed outer-fold evaluation metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


EPS = np.finfo(float).eps


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    """Return BH-adjusted q-values while preserving missing values."""
    p = np.asarray(list(values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    out[valid] = restored
    return out


def crossfit_exposure_transform(
    frame: pd.DataFrame,
    *,
    exposure_col: str = "corrected_age_gap",
    fold_col: str = "outer_fold",
    eligibility_col: str = "at_risk",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply non-target-fold mean/SD and quartiles to each held-out fold.

    Reference statistics use only eligible OOF rows from the other four outer
    folds. Event, duration, and competing-event columns are deliberately not
    consulted.
    """
    required = {exposure_col, fold_col, eligibility_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing transform columns: {sorted(missing)}")
    eligible = frame.loc[frame[eligibility_col].eq(1)].copy()
    if eligible.empty:
        raise ValueError("no eligible rows for cross-fitted transform")
    folds = sorted(pd.unique(eligible[fold_col]).tolist())
    if len(folds) < 2:
        raise ValueError("cross-fitted transform requires at least two folds")
    eligible["gap_z_crossfit"] = np.nan
    eligible["gap_quartile_crossfit"] = pd.Series(index=eligible.index, dtype="Int8")
    reference_rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        reference = pd.to_numeric(
            eligible.loc[eligible[fold_col].ne(fold), exposure_col], errors="coerce"
        ).dropna()
        target_index = eligible.index[eligible[fold_col].eq(fold)]
        if reference.empty or len(target_index) == 0:
            raise ValueError(f"fold {fold} lacks reference or target rows")
        mean = float(reference.mean())
        sd = float(reference.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"fold {fold} has invalid reference SD: {sd}")
        q25, q50, q75 = np.quantile(reference.to_numpy(dtype=float), [0.25, 0.5, 0.75])
        target = pd.to_numeric(eligible.loc[target_index, exposure_col], errors="coerce")
        if target.isna().any():
            raise ValueError(f"fold {fold} has missing exposure")
        eligible.loc[target_index, "gap_z_crossfit"] = (target - mean) / sd
        quartile = np.select(
            [target.le(q25), target.le(q50), target.le(q75)],
            [1, 2, 3],
            default=4,
        ).astype(np.int8)
        eligible.loc[target_index, "gap_quartile_crossfit"] = quartile
        reference_rows.append(
            {
                "outer_fold": int(fold),
                "reference_folds": ",".join(str(int(x)) for x in folds if x != fold),
                "reference_n": int(reference.shape[0]),
                "reference_mean": mean,
                "reference_sd_ddof0": sd,
                "q25": float(q25),
                "q50": float(q50),
                "q75": float(q75),
                "reference_event_columns_read": False,
            }
        )
    eligible["gap_z_crossfit"] = eligible["gap_z_crossfit"].astype(float)
    eligible["gap_quartile_crossfit"] = eligible["gap_quartile_crossfit"].astype(np.int8)
    return eligible, pd.DataFrame(reference_rows)


def crossfit_covariates(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    binary_columns: Sequence[str] = (),
    fold_col: str = "outer_fold",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fold-contained median/mode imputation and continuous scaling interface."""
    binary = set(binary_columns)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing extended covariates: {sorted(missing)}")
    out = frame.copy()
    transformed: list[str] = []
    audit: list[dict[str, object]] = []
    folds = sorted(pd.unique(out[fold_col]).tolist())
    for column in columns:
        raw = pd.to_numeric(out[column], errors="coerce")
        missing_name = f"{column}__missing"
        if raw.isna().any():
            out[missing_name] = raw.isna().astype(np.int8)
            transformed.append(missing_name)
        transformed_name = f"{column}__xf"
        out[transformed_name] = np.nan
        for fold in folds:
            reference_raw = raw.loc[out[fold_col].ne(fold)]
            reference = reference_raw.dropna()
            if reference.empty:
                raise ValueError(f"extended covariate {column} has no fold {fold} reference")
            if column in binary:
                modes = reference.mode(dropna=True)
                impute = float(modes.iloc[0])
                center, scale = 0.0, 1.0
            else:
                impute = float(reference.median())
                filled_reference = reference_raw.fillna(impute)
                center = float(filled_reference.mean())
                scale = float(filled_reference.std(ddof=0))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"extended covariate {column} has invalid fold {fold} SD")
            target_mask = out[fold_col].eq(fold)
            target = raw.loc[target_mask].fillna(impute)
            out.loc[target_mask, transformed_name] = (
                target if column in binary else (target - center) / scale
            )
            audit.append(
                {
                    "covariate": column,
                    "outer_fold": int(fold),
                    "reference_folds": ",".join(str(int(x)) for x in folds if x != fold),
                    "reference_nonmissing_n": int(reference.shape[0]),
                    "imputation": "mode" if column in binary else "median",
                    "imputation_value": impute,
                    "reference_center": center,
                    "reference_sd_ddof0": scale,
                }
            )
        transformed.append(transformed_name)
    return out, pd.DataFrame(audit), transformed


@dataclass(frozen=True)
class CoxFit:
    beta: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray
    p_value: np.ndarray
    log_likelihood: float
    iterations: int
    converged: bool
    score_max_abs: float
    ties: str = "breslow"


@dataclass(frozen=True)
class _CoxStratum:
    x: np.ndarray
    duration: np.ndarray
    event: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def _prepare_cox_strata(
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
) -> list[_CoxStratum]:
    prepared: list[_CoxStratum] = []
    for value in sorted(pd.unique(strata).tolist()):
        mask = strata == value
        order = np.argsort(-duration[mask], kind="stable")
        xs = np.asarray(x[mask][order], dtype=float)
        ts = np.asarray(duration[mask][order], dtype=float)
        es = np.asarray(event[mask][order], dtype=np.int8)
        starts = np.r_[0, np.flatnonzero(ts[1:] != ts[:-1]) + 1]
        ends = np.r_[starts[1:] - 1, len(ts) - 1]
        prepared.append(_CoxStratum(xs, ts, es, starts, ends))
    return prepared


def _cox_components(
    beta: np.ndarray, prepared: Sequence[_CoxStratum]
) -> tuple[float, np.ndarray, np.ndarray]:
    p = beta.shape[0]
    log_likelihood = 0.0
    score = np.zeros(p, dtype=float)
    information = np.zeros((p, p), dtype=float)
    for part in prepared:
        eta = part.x @ beta
        shift = float(np.max(eta))
        weight = np.exp(eta - shift)
        risk0 = np.cumsum(weight)[part.ends]
        risk1 = np.column_stack(
            [np.cumsum(weight * part.x[:, j])[part.ends] for j in range(p)]
        )
        risk2 = np.empty((len(part.ends), p, p), dtype=float)
        for j in range(p):
            for k in range(j, p):
                value = np.cumsum(weight * part.x[:, j] * part.x[:, k])[part.ends]
                risk2[:, j, k] = value
                risk2[:, k, j] = value
        deaths = np.add.reduceat(part.event, part.starts).astype(float)
        event_eta = np.add.reduceat(eta * part.event, part.starts)
        event_x = np.vstack(
            [np.add.reduceat(part.x[:, j] * part.event, part.starts) for j in range(p)]
        ).T
        active = deaths > 0
        if not active.any():
            continue
        d = deaths[active]
        s0 = risk0[active]
        mean = risk1[active] / s0[:, None]
        log_likelihood += float(np.sum(event_eta[active] - d * (np.log(s0) + shift)))
        score += np.sum(event_x[active] - d[:, None] * mean, axis=0)
        variance = risk2[active] / s0[:, None, None] - np.einsum("ni,nj->nij", mean, mean)
        information += np.sum(d[:, None, None] * variance, axis=0)
    return log_likelihood, score, information


def fit_cox_breslow(
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    max_iter: int = 40,
    tolerance: float = 1e-8,
) -> CoxFit:
    """Fit an unpenalized Cox model with Breslow ties and optional strata."""
    x = np.asarray(x, dtype=float)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=np.int8)
    if x.ndim != 2 or len(duration) != x.shape[0] or len(event) != x.shape[0]:
        raise ValueError("incompatible Cox input shapes")
    if x.shape[1] == 0 or len(x) == 0:
        raise ValueError("empty Cox design")
    if not (np.isfinite(x).all() and np.isfinite(duration).all() and (duration > 0).all()):
        raise ValueError("Cox inputs must be finite and duration must be positive")
    if not np.isin(event, [0, 1]).all() or event.sum() == 0:
        raise ValueError("Cox event must be binary with at least one event")
    if strata is None:
        strata = np.zeros(len(x), dtype=np.int8)
    else:
        strata = np.asarray(strata)
    prepared = _prepare_cox_strata(x, duration, event, strata)
    beta = np.zeros(x.shape[1], dtype=float)
    converged = False
    final_score = np.full(x.shape[1], np.nan)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        ll, score, information = _cox_components(beta, prepared)
        final_score = score
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information, rcond=1e-12) @ score
        if np.max(np.abs(step)) < tolerance:
            converged = True
            break
        step_scale = 1.0
        accepted = False
        while step_scale >= 2.0 ** -20:
            candidate = beta + step_scale * step
            candidate_ll, _, _ = _cox_components(candidate, prepared)
            if np.isfinite(candidate_ll) and candidate_ll >= ll - 1e-10:
                beta = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
    final_ll, final_score, information = _cox_components(beta, prepared)
    covariance = np.linalg.pinv(information, rcond=1e-12)
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z = np.divide(beta, standard_error, out=np.full_like(beta, np.nan), where=standard_error > 0)
    p_value = 2.0 * stats.norm.sf(np.abs(z))
    if np.max(np.abs(final_score)) < max(tolerance * 10.0, 1e-6):
        converged = True
    return CoxFit(
        beta=beta,
        covariance=covariance,
        standard_error=standard_error,
        p_value=p_value,
        log_likelihood=float(final_ll),
        iterations=iteration,
        converged=converged,
        score_max_abs=float(np.max(np.abs(final_score))),
    )


def schoenfeld_ph_test(
    fit: CoxFit,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    covariate_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Approximate Grambsch-Therneau tests using scaled Schoenfeld residuals."""
    x = np.asarray(x, dtype=float)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=np.int8)
    if strata is None:
        strata = np.zeros(len(x), dtype=np.int8)
    prepared = _prepare_cox_strata(x, duration, event, np.asarray(strata))
    residuals: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for part in prepared:
        eta = part.x @ fit.beta
        shift = float(np.max(eta))
        weight = np.exp(eta - shift)
        risk0 = np.cumsum(weight)[part.ends]
        risk1 = np.column_stack(
            [np.cumsum(weight * part.x[:, j])[part.ends] for j in range(x.shape[1])]
        )
        means = risk1 / risk0[:, None]
        # lifelines sorts by (stratum, duration ascending) before applying its
        # rank transform. The Cox risk calculations above use descending time,
        # so traverse time groups in reverse here and retain stratum blocks.
        for group_index in range(len(part.starts) - 1, -1, -1):
            start, end = part.starts[group_index], part.ends[group_index]
            local = np.flatnonzero(part.event[start : end + 1] == 1)
            if local.size:
                indices = start + local
                residuals.append(part.x[indices] - means[group_index])
                times.append(part.duration[indices])
    if not residuals:
        raise ValueError("no event residuals for PH test")
    residual = np.vstack(residuals)
    event_time = np.concatenate(times)
    n_deaths = len(event_time)
    scaled = n_deaths * residual @ fit.covariance
    names = list(covariate_names or [f"x{i}" for i in range(x.shape[1])])
    rows: list[dict[str, float | str | int]] = []
    transforms = {
        "rank_lifelines_cumsum_event": np.arange(1, n_deaths + 1, dtype=float),
        "log": np.log(event_time),
    }
    for transform_name, transformed_time in transforms.items():
        centered = transformed_time - transformed_time.mean()
        time_ss = float(np.sum(centered**2))
        for j, name in enumerate(names):
            numerator = float(np.sum(centered * scaled[:, j])) ** 2
            denominator = n_deaths * float(fit.standard_error[j] ** 2) * time_ss
            statistic = numerator / denominator if denominator > 0 else np.nan
            rows.append(
                {
                    "covariate": name,
                    "time_transform": transform_name,
                    "n_events": n_deaths,
                    "chi_square": statistic,
                    "df": 1,
                    "p_value": float(stats.chi2.sf(statistic, 1)) if np.isfinite(statistic) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fixed_effect_meta(coefficients: np.ndarray, standard_errors: np.ndarray) -> dict[str, float]:
    beta = np.asarray(coefficients, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    beta, se = beta[valid], se[valid]
    if len(beta) == 0:
        raise ValueError("no valid estimates for meta-analysis")
    weights = 1.0 / se**2
    pooled = float(np.sum(weights * beta) / np.sum(weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
    q = float(np.sum(weights * (beta - pooled) ** 2))
    df = max(len(beta) - 1, 0)
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 and df > 0 else 0.0
    return {
        "n_folds": int(len(beta)),
        "beta_fixed": pooled,
        "se_fixed": pooled_se,
        "hr_fixed": float(np.exp(pooled)),
        "hr_ci95_low_fixed": float(np.exp(pooled - 1.959963984540054 * pooled_se)),
        "hr_ci95_high_fixed": float(np.exp(pooled + 1.959963984540054 * pooled_se)),
        "p_fixed": float(2.0 * stats.norm.sf(abs(pooled / pooled_se))),
        "heterogeneity_q": q,
        "heterogeneity_df": int(df),
        "heterogeneity_p": float(stats.chi2.sf(q, df)) if df > 0 else np.nan,
        "i2_percent": float(i2),
    }


def _effective_horizon(requested: float, max_duration: float) -> float:
    if max_duration < requested and requested - max_duration <= 0.1:
        return float(max_duration)
    return float(requested)


def nonparametric_curves(
    duration: np.ndarray,
    event: np.ndarray,
    competing_death: np.ndarray,
    *,
    horizons: Sequence[float] = (24.0, 60.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic KM and Aalen-Johansen curves with exact tied-time handling."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=np.int8)
    competing_death = np.asarray(competing_death, dtype=np.int8)
    if len(duration) == 0 or not (len(duration) == len(event) == len(competing_death)):
        raise ValueError("invalid nonparametric inputs")
    if ((event + competing_death) > 1).any():
        raise ValueError("target and competing events must be mutually exclusive")
    times, inverse = np.unique(duration, return_inverse=True)
    removed = np.bincount(inverse, minlength=len(times)).astype(int)
    target = np.bincount(inverse, weights=event, minlength=len(times)).astype(int)
    competing = np.bincount(inverse, weights=competing_death, minlength=len(times)).astype(int)
    risk = len(duration)
    km_survival = 1.0
    aj_survival = 1.0
    target_cif = 0.0
    death_cif = 0.0
    greenwood = 0.0
    rows = [
        {
            "time_months": 0.0,
            "at_risk_before": int(risk),
            "target_events": 0,
            "competing_deaths": 0,
            "other_censored": 0,
            "km_survival": 1.0,
            "km_target_risk": 0.0,
            "km_target_risk_ci95_low": 0.0,
            "km_target_risk_ci95_high": 0.0,
            "aj_event_free_survival": 1.0,
            "aj_target_cif": 0.0,
            "aj_death_cif": 0.0,
        }
    ]
    z = 1.959963984540054
    for time, removed_at, d1, d2 in zip(times, removed, target, competing):
        y = risk
        if y <= 0:
            raise ValueError("risk set exhausted before final time")
        if d1:
            km_survival *= 1.0 - d1 / y
            if y > d1:
                greenwood += float(d1) / (float(y) * float(y - d1))
        target_cif += aj_survival * d1 / y
        death_cif += aj_survival * d2 / y
        aj_survival *= 1.0 - (d1 + d2) / y
        if 0.0 < km_survival < 1.0 and greenwood > 0:
            theta = np.log(-np.log(km_survival))
            se_theta = np.sqrt(greenwood) / abs(np.log(km_survival))
            survival_low = float(np.exp(-np.exp(theta + z * se_theta)))
            survival_high = float(np.exp(-np.exp(theta - z * se_theta)))
        else:
            survival_low = survival_high = float(km_survival)
        rows.append(
            {
                "time_months": float(time),
                "at_risk_before": int(y),
                "target_events": int(d1),
                "competing_deaths": int(d2),
                "other_censored": int(removed_at - d1 - d2),
                "km_survival": float(km_survival),
                "km_target_risk": float(1.0 - km_survival),
                "km_target_risk_ci95_low": float(1.0 - survival_high),
                "km_target_risk_ci95_high": float(1.0 - survival_low),
                "aj_event_free_survival": float(aj_survival),
                "aj_target_cif": float(target_cif),
                "aj_death_cif": float(death_cif),
            }
        )
        risk -= int(removed_at)
    curve = pd.DataFrame(rows)
    summaries: list[dict[str, float | int]] = []
    max_duration = float(np.max(duration))
    for requested in horizons:
        evaluated = _effective_horizon(float(requested), max_duration)
        index = int(np.searchsorted(curve["time_months"].to_numpy(), evaluated, side="right") - 1)
        row = curve.iloc[max(index, 0)]
        at_risk = int(np.sum(duration >= evaluated - 1e-12))
        summaries.append(
            {
                "requested_horizon_months": float(requested),
                "evaluated_horizon_months": evaluated,
                "at_risk_before_horizon": at_risk,
                "km_target_risk": float(row["km_target_risk"]),
                "km_target_risk_ci95_low": float(row["km_target_risk_ci95_low"]),
                "km_target_risk_ci95_high": float(row["km_target_risk_ci95_high"]),
                "aj_target_cif": float(row["aj_target_cif"]),
                "aj_death_cif": float(row["aj_death_cif"]),
            }
        )
    return curve, pd.DataFrame(summaries)


def logrank_test(
    duration: np.ndarray,
    event: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int]:
    """Standard k-sample log-rank test with censoring at competing death."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=np.int8)
    groups = np.asarray(groups)
    levels = np.sort(pd.unique(groups))
    if len(levels) < 2:
        raise ValueError("log-rank test requires at least two groups")
    group_index = np.searchsorted(levels, groups)
    risk = np.bincount(group_index, minlength=len(levels)).astype(float)
    observed_minus_expected = np.zeros(len(levels), dtype=float)
    variance = np.zeros((len(levels), len(levels)), dtype=float)
    for time in np.sort(pd.unique(duration)):
        at_time = duration == time
        events_at = at_time & event.astype(bool)
        d_group = np.bincount(group_index[events_at], minlength=len(levels)).astype(float)
        d = float(d_group.sum())
        y = float(risk.sum())
        if d > 0 and y > 1:
            probability = risk / y
            expected = d * probability
            observed_minus_expected += d_group - expected
            factor = d * (y - d) / (y - 1.0)
            variance += factor * (np.diag(probability) - np.outer(probability, probability))
        removed = np.bincount(group_index[at_time], minlength=len(levels)).astype(float)
        risk -= removed
    reduced = observed_minus_expected[:-1]
    reduced_variance = variance[:-1, :-1]
    statistic = float(reduced @ np.linalg.pinv(reduced_variance, rcond=1e-12) @ reduced)
    df = len(levels) - 1
    return {
        "n_groups": int(len(levels)),
        "chi_square": statistic,
        "df": int(df),
        "p_value": float(stats.chi2.sf(statistic, df)),
    }


def _aj_at_horizons_from_weighted_cells(
    times: np.ndarray,
    codes: np.ndarray,
    counts: np.ndarray,
    horizons: Sequence[float],
) -> np.ndarray:
    order = np.argsort(times, kind="stable")
    times, codes, counts = times[order], codes[order], counts[order]
    unique_times, inverse = np.unique(times, return_inverse=True)
    removed = np.bincount(inverse, weights=counts, minlength=len(unique_times))
    d1 = np.bincount(inverse, weights=counts * (codes == 1), minlength=len(unique_times))
    d2 = np.bincount(inverse, weights=counts * (codes == 2), minlength=len(unique_times))
    risk = float(counts.sum())
    survival = 1.0
    cif = 0.0
    values = np.zeros(len(unique_times), dtype=float)
    for i in range(len(unique_times)):
        if risk > 0:
            cif += survival * d1[i] / risk
            survival *= 1.0 - (d1[i] + d2[i]) / risk
        values[i] = cif
        risk -= removed[i]
    result = []
    max_time = float(unique_times[-1])
    for horizon in horizons:
        evaluated = _effective_horizon(float(horizon), max_time)
        index = np.searchsorted(unique_times, evaluated, side="right") - 1
        result.append(float(values[index]) if index >= 0 else 0.0)
    return np.asarray(result)


def stratified_aj_bootstrap(
    frame: pd.DataFrame,
    *,
    fold_col: str = "outer_fold",
    group_col: str = "gap_quartile_crossfit",
    duration_col: str = "duration_months",
    event_col: str = "event",
    competing_col: str = "competing_death",
    horizons: Sequence[float] = (24.0, 60.0),
    replicates: int = 500,
    seed: int = 710,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-and-quartile-stratified participant bootstrap for AJ risks."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    groups = sorted(pd.unique(frame[group_col]).tolist())
    folds = sorted(pd.unique(frame[fold_col]).tolist())
    cells: dict[tuple[object, object], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group in groups:
        for fold in folds:
            part = frame.loc[frame[group_col].eq(group) & frame[fold_col].eq(fold)]
            code = np.select(
                [part[event_col].eq(1), part[competing_col].eq(1)], [1, 2], default=0
            ).astype(np.int8)
            cell = pd.DataFrame({"time": part[duration_col].to_numpy(float), "code": code})
            aggregated = cell.groupby(["time", "code"], sort=True).size().reset_index(name="count")
            cells[(group, fold)] = (
                aggregated["time"].to_numpy(float),
                aggregated["code"].to_numpy(np.int8),
                aggregated["count"].to_numpy(int),
            )
    bootstrap = np.empty((replicates, len(groups), len(horizons)), dtype=float)
    for b in range(replicates):
        for group_index, group in enumerate(groups):
            all_times: list[np.ndarray] = []
            all_codes: list[np.ndarray] = []
            all_counts: list[np.ndarray] = []
            for fold in folds:
                times, codes, counts = cells[(group, fold)]
                sampled = rng.multinomial(int(counts.sum()), counts / counts.sum())
                all_times.append(times)
                all_codes.append(codes)
                all_counts.append(sampled)
            bootstrap[b, group_index] = _aj_at_horizons_from_weighted_cells(
                np.concatenate(all_times),
                np.concatenate(all_codes),
                np.concatenate(all_counts),
                horizons,
            )
    risk_rows: list[dict[str, float | int]] = []
    for group_index, group in enumerate(groups):
        for horizon_index, horizon in enumerate(horizons):
            values = bootstrap[:, group_index, horizon_index]
            risk_rows.append(
                {
                    "gap_quartile_crossfit": int(group),
                    "requested_horizon_months": float(horizon),
                    "bootstrap_replicates": int(replicates),
                    "bootstrap_ci95_low": float(np.quantile(values, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(values, 0.975)),
                }
            )
    difference_rows: list[dict[str, float | int]] = []
    low_index, high_index = groups.index(min(groups)), groups.index(max(groups))
    for horizon_index, horizon in enumerate(horizons):
        difference = bootstrap[:, high_index, horizon_index] - bootstrap[:, low_index, horizon_index]
        difference_rows.append(
            {
                "contrast": f"Q{int(max(groups))}-Q{int(min(groups))}",
                "requested_horizon_months": float(horizon),
                "bootstrap_replicates": int(replicates),
                "bootstrap_ci95_low": float(np.quantile(difference, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(difference, 0.975)),
            }
        )
    return pd.DataFrame(risk_rows), pd.DataFrame(difference_rows)
