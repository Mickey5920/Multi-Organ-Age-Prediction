"""Fail-closed statistical primitives for the Candidate5 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, sqrt
from typing import Iterable, Sequence

import numpy as np
from scipy.special import gammaincc


Z_975 = 1.959963984540054
MAX_CONDITION = 1.0e6


class StatisticalContractError(ValueError):
    """Raised when a fit or inference contract cannot be satisfied."""


@dataclass(frozen=True)
class MatrixDiagnostics:
    rank: int
    condition_number: float


@dataclass(frozen=True)
class RegressionResult:
    beta: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray
    diagnostics: MatrixDiagnostics
    max_leverage: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class CoxResult:
    beta: np.ndarray
    covariance: np.ndarray
    standard_error: np.ndarray
    diagnostics: MatrixDiagnostics
    score_max_abs: float
    converged: bool
    iterations: int
    baseline_hazard: tuple[tuple[object, float, float], ...]


@dataclass(frozen=True)
class MetaResult:
    beta: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    q: float
    q_df: int
    q_p_value: float
    i2_percent: float


@dataclass(frozen=True)
class PairedCResult:
    base_c: float
    augmented_c: float
    delta_c: float
    event_weight: int


def _array(value: object, *, ndim: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != ndim or out.size == 0 or not np.all(np.isfinite(out)):
        raise StatisticalContractError(f"{name} must be nonempty finite {ndim}D float64")
    return out


def _binary(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind == "b":
        raise StatisticalContractError(f"{name} must be a nonempty integer vector")
    if not np.all(np.equal(raw, np.floor(raw))) or not np.all(np.isin(raw, (0, 1))):
        raise StatisticalContractError(f"{name} must contain exact integers in {{0,1}}")
    return raw.astype(np.int8)


def matrix_diagnostics(x: object, *, max_condition: float = MAX_CONDITION) -> MatrixDiagnostics:
    matrix = _array(x, ndim=2, name="design")
    gram = matrix.T @ matrix
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1])
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * largest
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    smallest = float(eigenvalues[0])
    condition = float("inf") if smallest <= 0.0 else sqrt(largest / smallest)
    if rank != matrix.shape[1]:
        raise StatisticalContractError("rank-deficient design")
    if not np.isfinite(condition) or condition > max_condition:
        raise StatisticalContractError("design condition number exceeds contract")
    return MatrixDiagnostics(rank=rank, condition_number=condition)


def normal_two_sided_p(z: float) -> float:
    if not np.isfinite(z):
        raise StatisticalContractError("nonfinite z statistic")
    return float(1.0 - erf(abs(float(z)) / sqrt(2.0)))


def wald_summary(beta: float, standard_error: float) -> dict[str, float]:
    if not np.isfinite(beta) or not np.isfinite(standard_error) or standard_error <= 0.0:
        raise StatisticalContractError("Wald inputs must be finite with positive SE")
    low = beta - Z_975 * standard_error
    high = beta + Z_975 * standard_error
    values = {
        "beta": float(beta), "se": float(standard_error),
        "ci95_low": float(low), "ci95_high": float(high),
        "hr": float(exp(beta)), "hr_ci95_low": float(exp(low)),
        "hr_ci95_high": float(exp(high)),
        "p_value": normal_two_sided_p(beta / standard_error),
    }
    if not all(np.isfinite(v) for v in values.values()):
        raise StatisticalContractError("nonfinite Wald output")
    return values


def ols_hc3(x: object, y: object) -> RegressionResult:
    matrix = _array(x, ndim=2, name="design")
    outcome = _array(y, ndim=1, name="outcome")
    if matrix.shape[0] != outcome.size or matrix.shape[0] <= matrix.shape[1]:
        raise StatisticalContractError("OLS requires aligned n>p")
    diagnostics = matrix_diagnostics(matrix)
    gram_inverse = np.linalg.inv(matrix.T @ matrix)
    beta = gram_inverse @ matrix.T @ outcome
    residual = outcome - matrix @ beta
    leverage = np.einsum("ij,jk,ik->i", matrix, gram_inverse, matrix)
    if np.any(leverage <= 0.0) or np.any(leverage >= 1.0 - 1e-12):
        raise StatisticalContractError("invalid HC3 leverage")
    adjusted = residual / (1.0 - leverage)
    meat = matrix.T @ ((adjusted * adjusted)[:, None] * matrix)
    covariance = gram_inverse @ meat @ gram_inverse
    standard_error = np.sqrt(np.diag(covariance))
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(covariance)) and
            np.all(np.isfinite(standard_error)) and np.all(standard_error > 0.0)):
        raise StatisticalContractError("nonfinite or nonpositive OLS-HC3 inference")
    return RegressionResult(beta, covariance, standard_error, diagnostics,
                            float(np.max(leverage)), True, 1)


def logistic_hc3(
    x: object,
    y: object,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> RegressionResult:
    matrix = _array(x, ndim=2, name="design")
    outcome = _binary(y, name="outcome").astype(np.float64)
    if matrix.shape[0] != outcome.size or matrix.shape[0] <= matrix.shape[1]:
        raise StatisticalContractError("logistic regression requires aligned n>p")
    if np.unique(outcome).size != 2:
        raise StatisticalContractError("logistic outcome requires both classes")
    matrix_diagnostics(matrix)
    beta = np.zeros(matrix.shape[1], dtype=np.float64)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        linear = np.clip(matrix @ beta, -36.0, 36.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = probability * (1.0 - probability)
        if np.any(weight <= 1e-12):
            raise StatisticalContractError("complete or quasi-complete separation")
        information = matrix.T @ (weight[:, None] * matrix)
        step = np.linalg.solve(information, matrix.T @ (outcome - probability))
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            raise StatisticalContractError("nonfinite logistic coefficient")
        if float(np.max(np.abs(step))) <= tolerance:
            converged = True
            break
    if not converged:
        raise StatisticalContractError("logistic fit did not converge")
    probability = 1.0 / (1.0 + np.exp(-np.clip(matrix @ beta, -36.0, 36.0)))
    weight = probability * (1.0 - probability)
    weighted_design = np.sqrt(weight)[:, None] * matrix
    diagnostics = matrix_diagnostics(weighted_design)
    bread = np.linalg.inv(matrix.T @ (weight[:, None] * matrix))
    leverage = np.einsum("ij,jk,ik->i", weighted_design, bread, weighted_design)
    if np.any(leverage <= 0.0) or np.any(leverage >= 1.0 - 1e-12):
        raise StatisticalContractError("invalid logistic HC3 leverage")
    adjusted_score = (outcome - probability) / (1.0 - leverage)
    meat = matrix.T @ ((adjusted_score * adjusted_score)[:, None] * matrix)
    covariance = bread @ meat @ bread
    standard_error = np.sqrt(np.diag(covariance))
    if not (np.all(np.isfinite(covariance)) and np.all(np.isfinite(standard_error)) and
            np.all(standard_error > 0.0)):
        raise StatisticalContractError("nonfinite or nonpositive logistic-HC3 inference")
    return RegressionResult(beta, covariance, standard_error, diagnostics,
                            float(np.max(leverage)), True, iteration)


def benjamini_hochberg(p_values: Sequence[float], member_keys: Sequence[str]) -> np.ndarray:
    p = _array(p_values, ndim=1, name="p_values")
    keys = tuple(member_keys)
    if len(keys) != p.size or len(set(keys)) != len(keys):
        raise StatisticalContractError("BH keys must be aligned and unique")
    if np.any((p < 0.0) | (p > 1.0)):
        raise StatisticalContractError("BH p-values must be in [0,1]")
    order = sorted(range(p.size), key=lambda i: (float(p[i]), keys[i]))
    adjusted_sorted = np.empty(p.size, dtype=np.float64)
    running = 1.0
    for reverse_position in range(p.size - 1, -1, -1):
        index = order[reverse_position]
        rank = reverse_position + 1
        running = min(running, float(p[index]) * p.size / rank)
        adjusted_sorted[reverse_position] = min(1.0, running)
    output = np.empty(p.size, dtype=np.float64)
    for sorted_position, original_index in enumerate(order):
        output[original_index] = adjusted_sorted[sorted_position]
    return output


def fixed_effect_meta(beta: Sequence[float], standard_error: Sequence[float]) -> MetaResult:
    effects = _array(beta, ndim=1, name="beta")
    se = _array(standard_error, ndim=1, name="standard_error")
    if effects.size != 5 or se.size != 5 or np.any(se <= 0.0):
        raise StatisticalContractError("fixed-effect meta requires five finite positive-SE members")
    weights = 1.0 / np.square(se)
    total_weight = float(np.sum(weights))
    pooled = float(np.sum(weights * effects) / total_weight)
    pooled_se = sqrt(1.0 / total_weight)
    q = float(np.sum(weights * np.square(effects - pooled)))
    q_df = 4
    q_p = float(gammaincc(q_df / 2.0, q / 2.0))
    i2 = 0.0 if q == 0.0 else max(0.0, (q - q_df) / q) * 100.0
    values = (pooled, pooled_se, q, q_p, i2)
    if not all(np.isfinite(v) for v in values):
        raise StatisticalContractError("nonfinite fixed-effect meta output")
    return MetaResult(pooled, pooled_se, pooled - Z_975 * pooled_se,
                      pooled + Z_975 * pooled_se, q, q_df, q_p, i2)


def _cox_components(
    beta: np.ndarray,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, list[tuple[object, float, float]]]:
    log_likelihood = 0.0
    score = np.zeros(x.shape[1], dtype=np.float64)
    information = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    baseline: list[tuple[object, float, float]] = []
    linear = x @ beta
    for stratum in sorted(set(strata.tolist()), key=str):
        mask = strata == stratum
        xs = x[mask]
        ts = duration[mask]
        es = event[mask]
        ls = linear[mask]
        for time in sorted(set(ts[es == 1].tolist())):
            event_mask = (ts == time) & (es == 1)
            risk_mask = ts >= time
            d = int(np.sum(event_mask))
            risk_linear = ls[risk_mask]
            shift = float(np.max(risk_linear))
            risk_weight = np.exp(risk_linear - shift)
            risk_x = xs[risk_mask]
            s0_scaled = float(np.sum(risk_weight))
            mean = np.sum(risk_weight[:, None] * risk_x, axis=0) / s0_scaled
            second = np.einsum("i,ij,ik->jk", risk_weight, risk_x, risk_x) / s0_scaled
            log_s0 = shift + np.log(s0_scaled)
            log_likelihood += float(np.sum(ls[event_mask])) - d * log_s0
            score += np.sum(xs[event_mask], axis=0) - d * mean
            information += d * (second - np.outer(mean, mean))
            hazard = float(d * np.exp(-log_s0))
            if not np.isfinite(hazard) or hazard <= 0.0:
                raise StatisticalContractError("nonfinite or nonpositive Breslow baseline hazard")
            baseline.append((stratum, float(time), hazard))
    return log_likelihood, score, information, baseline


def cox_breslow(
    x: object,
    duration: object,
    event: object,
    *,
    strata: object | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 100,
) -> CoxResult:
    matrix = _array(x, ndim=2, name="design")
    time = _array(duration, ndim=1, name="duration")
    observed = _binary(event, name="event")
    if matrix.shape[0] != time.size or time.size != observed.size or np.any(time <= 0.0):
        raise StatisticalContractError("Cox inputs must align and duration must be positive")
    if int(np.sum(observed)) == 0:
        raise StatisticalContractError("Cox fit requires events")
    diagnostics = matrix_diagnostics(matrix)
    group = np.zeros(time.size, dtype=np.int64) if strata is None else np.asarray(strata)
    if group.ndim != 1 or group.size != time.size:
        raise StatisticalContractError("strata must be aligned 1D")
    beta = np.zeros(matrix.shape[1], dtype=np.float64)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        _, score, information, _ = _cox_components(beta, matrix, time, observed, group)
        if not np.all(np.isfinite(information)):
            raise StatisticalContractError("nonfinite Cox information")
        step = np.linalg.solve(information, score)
        beta = beta + step
        if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 50.0:
            raise StatisticalContractError("Cox separation or nonfinite coefficient")
        if float(np.max(np.abs(step))) <= tolerance:
            converged = True
            break
    if not converged:
        raise StatisticalContractError("Cox fit did not converge")
    _, score, information, baseline = _cox_components(beta, matrix, time, observed, group)
    covariance = np.linalg.inv(information)
    standard_error = np.sqrt(np.diag(covariance))
    score_max_abs = float(np.max(np.abs(score)))
    if not (np.all(np.isfinite(covariance)) and np.all(np.isfinite(standard_error)) and
            np.all(standard_error > 0.0) and np.isfinite(score_max_abs)):
        raise StatisticalContractError("invalid Cox inference")
    return CoxResult(beta, covariance, standard_error, diagnostics, score_max_abs,
                     True, iteration, tuple(baseline))


def schoenfeld_residuals(
    beta: object,
    x: object,
    duration: object,
    event: object,
    *,
    strata: object | None = None,
    participant_id: object,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = _array(beta, ndim=1, name="beta")
    matrix = _array(x, ndim=2, name="design")
    time = _array(duration, ndim=1, name="duration")
    observed = _binary(event, name="event")
    if matrix.shape != (time.size, coefficient.size) or observed.size != time.size:
        raise StatisticalContractError("Schoenfeld inputs do not align")
    group = np.zeros(time.size, dtype=np.int64) if strata is None else np.asarray(strata)
    if group.ndim != 1 or group.size != time.size:
        raise StatisticalContractError("Schoenfeld strata do not align")
    participant = np.asarray(participant_id, dtype=object)
    if (participant.ndim != 1 or participant.size != time.size or
            any(not isinstance(value, str) or not value for value in participant.tolist()) or
            len(set(participant.tolist())) != participant.size):
        raise StatisticalContractError("Schoenfeld participant IDs must be aligned unique strings")
    residual_rows: list[np.ndarray] = []
    event_times: list[float] = []
    linear = matrix @ coefficient
    for stratum in sorted(set(group.tolist()), key=str):
        mask = group == stratum
        xs, ts, es, ls, ids = matrix[mask], time[mask], observed[mask], linear[mask], participant[mask]
        for event_time in sorted(set(ts[es == 1].tolist())):
            event_indices = np.flatnonzero((ts == event_time) & (es == 1))
            event_indices = np.asarray(sorted(event_indices.tolist(), key=lambda index: ids[index]), dtype=np.int64)
            risk = ts >= event_time
            shift = float(np.max(ls[risk]))
            weights = np.exp(ls[risk] - shift)
            risk_mean = np.sum(weights[:, None] * xs[risk], axis=0) / np.sum(weights)
            for index in event_indices:
                residual_rows.append(xs[index] - risk_mean)
                event_times.append(float(event_time))
    if not residual_rows:
        raise StatisticalContractError("Schoenfeld residuals require events")
    return np.vstack(residual_rows), np.asarray(event_times, dtype=np.float64)


def schoenfeld_ph_tests(
    residuals: object,
    event_times: object,
    covariance: object,
    standard_error: object,
    *,
    transform: str,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    raw = _array(residuals, ndim=2, name="residuals")
    times = _array(event_times, ndim=1, name="event_times")
    cov = _array(covariance, ndim=2, name="covariance")
    se = _array(standard_error, ndim=1, name="standard_error")
    d, p = raw.shape
    if times.size != d or cov.shape != (p, p) or se.size != p or np.any(se <= 0.0):
        raise StatisticalContractError("PH inputs do not align")
    scaled = d * raw @ cov
    if transform == "rank_lifelines_cumsum_event":
        u = np.arange(1.0, d + 1.0)
    elif transform == "log":
        if np.any(times <= 0.0):
            raise StatisticalContractError("log PH transform requires positive event times")
        u = np.log(times)
    else:
        raise StatisticalContractError("unknown PH transform")
    centered = u - np.mean(u)
    sumsq = float(centered @ centered)
    denominator = d * np.square(se) * sumsq
    if sumsq <= 0.0 or np.any(denominator <= 0.0):
        raise StatisticalContractError("nonpositive PH denominator")
    numerator = centered @ scaled
    chi_square = np.square(numerator) / denominator
    p_values = gammaincc(0.5, chi_square / 2.0)
    if not (np.all(np.isfinite(chi_square)) and np.all(np.isfinite(p_values))):
        raise StatisticalContractError("nonfinite PH result")
    global_chi_square = float(np.sum(chi_square))
    global_df = p
    global_p = float(gammaincc(global_df / 2.0, global_chi_square / 2.0))
    return chi_square, p_values, global_chi_square, global_df, global_p


def harrell_c(duration: object, event: object, score: object, multiplicity: object | None = None) -> float:
    time = _array(duration, ndim=1, name="duration")
    observed = _binary(event, name="event")
    risk = _array(score, ndim=1, name="score")
    if time.size != observed.size or time.size != risk.size or np.any(time <= 0.0):
        raise StatisticalContractError("Harrell inputs must align with positive duration")
    if multiplicity is None:
        count = np.ones(time.size, dtype=np.int64)
    else:
        raw_count = np.asarray(multiplicity)
        if (raw_count.ndim != 1 or raw_count.size != time.size or raw_count.dtype.kind not in "iu" or
                np.any(raw_count < 0)):
            raise StatisticalContractError("multiplicity must be aligned nonnegative integers")
        count = raw_count.astype(np.int64)
    comparable = 0
    concordant = 0.0
    for i in range(time.size - 1):
        if count[i] == 0:
            continue
        for j in range(i + 1, time.size):
            weight = int(count[i]) * int(count[j])
            if weight == 0:
                continue
            earlier = None
            if time[i] < time[j] and observed[i] == 1:
                earlier = i
            elif time[j] < time[i] and observed[j] == 1:
                earlier = j
            elif time[i] == time[j] and observed[i] + observed[j] == 1:
                earlier = i if observed[i] == 1 else j
            if earlier is None:
                continue
            later = j if earlier == i else i
            comparable += weight
            if risk[earlier] > risk[later]:
                concordant += weight
            elif risk[earlier] == risk[later]:
                concordant += 0.5 * weight
    if comparable <= 0:
        raise StatisticalContractError("Harrell C has zero comparable pairs")
    result = float(concordant / comparable)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise StatisticalContractError("invalid Harrell C")
    return result


def paired_fold_weighted_c(
    fold: object,
    duration: object,
    event: object,
    base_score: object,
    augmented_score: object,
    multiplicity: object | None = None,
) -> PairedCResult:
    groups = np.asarray(fold)
    time = _array(duration, ndim=1, name="duration")
    observed = _binary(event, name="event")
    base = _array(base_score, ndim=1, name="base_score")
    augmented = _array(augmented_score, ndim=1, name="augmented_score")
    if any(value.size != time.size for value in (observed, base, augmented)) or groups.shape != time.shape:
        raise StatisticalContractError("paired C inputs do not align")
    if multiplicity is None:
        counts = np.ones(time.size, dtype=np.int64)
    else:
        raw_counts = np.asarray(multiplicity)
        if raw_counts.shape != time.shape or raw_counts.dtype.kind not in "iu" or np.any(raw_counts < 0):
            raise StatisticalContractError("paired multiplicity must be aligned nonnegative integers")
        counts = raw_counts.astype(np.int64)
    weighted_base = 0.0
    weighted_augmented = 0.0
    total_events = 0
    for group in sorted(set(groups.tolist()), key=str):
        mask = groups == group
        event_weight = int(np.sum(counts[mask] * observed[mask]))
        if event_weight <= 0:
            raise StatisticalContractError("every fold must have positive target-event weight")
        base_c = harrell_c(time[mask], observed[mask], base[mask], counts[mask])
        augmented_c = harrell_c(time[mask], observed[mask], augmented[mask], counts[mask])
        weighted_base += event_weight * base_c
        weighted_augmented += event_weight * augmented_c
        total_events += event_weight
    base_result = weighted_base / total_events
    augmented_result = weighted_augmented / total_events
    return PairedCResult(float(base_result), float(augmented_result),
                         float(augmented_result - base_result), total_events)


def paired_fixed_score_bootstrap(
    fold: object,
    duration: object,
    event: object,
    base_score: object,
    augmented_score: object,
    *,
    seed: int,
    replicates: int = 1000,
) -> tuple[np.ndarray, int]:
    groups = np.asarray(fold)
    time = _array(duration, ndim=1, name="duration")
    observed = _binary(event, name="event")
    base = _array(base_score, ndim=1, name="base_score")
    augmented = _array(augmented_score, ndim=1, name="augmented_score")
    if groups.shape != time.shape or any(v.size != time.size for v in (observed, base, augmented)):
        raise StatisticalContractError("bootstrap inputs do not align")
    if not isinstance(seed, int) or seed < 0 or replicates <= 0:
        raise StatisticalContractError("invalid bootstrap seed or replicate count")
    rng = np.random.Generator(np.random.PCG64(seed))
    ordered_masks = [np.flatnonzero(groups == group) for group in sorted(set(groups.tolist()), key=str)]
    valid: list[float] = []
    invalid = 0
    for _ in range(replicates):
        multiplicity = np.zeros(time.size, dtype=np.int64)
        for indices in ordered_masks:
            draw = rng.integers(0, indices.size, size=indices.size, endpoint=False, dtype=np.int64)
            multiplicity[indices] = np.bincount(draw, minlength=indices.size)
        try:
            valid.append(paired_fold_weighted_c(groups, time, observed, base, augmented, multiplicity).delta_c)
        except StatisticalContractError:
            invalid += 1
    output = np.asarray(valid, dtype=np.float64)
    if output.size < 2 or not np.all(np.isfinite(output)):
        raise StatisticalContractError("insufficient valid paired bootstrap replicates")
    return output, invalid
