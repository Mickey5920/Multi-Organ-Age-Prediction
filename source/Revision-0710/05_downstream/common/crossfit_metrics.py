"""Pure metric primitives implementing the Revision-0710 metric contract."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


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
    """Raised when an input violates the frozen evaluation contract."""


@dataclass(frozen=True)
class AgeMetrics:
    n_rows: int
    n_participants: int
    mae: float
    pearson: float
    r2: float


@dataclass(frozen=True)
class OrganLinearCorrection:
    organ_system: str
    intercept: float
    age_slope: float
    n_reference_rows: int


@dataclass(frozen=True)
class LinearAgeCorrectionModel:
    parameters: tuple[OrganLinearCorrection, ...]
    validation_participant_ids: frozenset[str]


@dataclass(frozen=True)
class CoxCellResult:
    fold: int
    organ_system: str
    n_validation: int
    n_test: int
    validation_events: int
    test_events: int
    c_base: float
    c_plus: float
    delta_c: float


@dataclass(frozen=True)
class OrganDeltaCSummary:
    organ_system: str
    mean_c_base: float
    mean_c_plus: float
    mean_delta_c: float
    positive_folds: int


@dataclass(frozen=True)
class CoxAggregate:
    n_cells: int
    organs: tuple[OrganDeltaCSummary, ...]
    macro_c_base: float
    macro_c_plus: float
    macro_delta_c: float
    positive_organs: int


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, table: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ContractError(f"{table} missing required columns: {missing}")


def _numeric_finite(frame: pd.DataFrame, columns: Iterable[str], *, table: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{table}.{column} must be numeric") from exc
        if not np.isfinite(values).all():
            raise ContractError(f"{table}.{column} contains null or non-finite values")
        arrays[column] = values
    return arrays


def _validate_participant_ids(frame: pd.DataFrame, *, table: str) -> None:
    ids = frame["participant_id"]
    invalid = ids.isna() | ids.astype(str).str.strip().eq("")
    if invalid.any():
        raise ContractError(f"{table}.participant_id contains null or empty values")


def _validate_prediction_rows(frame: pd.DataFrame, *, table: str) -> dict[str, np.ndarray]:
    required = ("participant_id", "organ_system", "actual_age", "predicted_age")
    _require_columns(frame, required, table=table)
    if frame.empty:
        raise ContractError(f"{table} must not be empty")
    _validate_participant_ids(frame, table=table)
    if frame["organ_system"].isna().any() or frame["organ_system"].astype(str).str.strip().eq("").any():
        raise ContractError(f"{table}.organ_system contains null or empty values")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise ContractError(f"{table} has duplicate participant_id and organ_system rows")
    return _numeric_finite(frame, ("actual_age", "predicted_age"), table=table)


def _require_exact_organs(frame: pd.DataFrame, organs: tuple[str, ...], *, table: str) -> None:
    observed = set(frame["organ_system"].astype(str))
    expected = set(organs)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ContractError(f"{table} organ_system mismatch; missing={missing}, extra={extra}")


def raw_age_metrics(frame: pd.DataFrame) -> AgeMetrics:
    """Compute primary raw age metrics on pooled held-out participant-organ rows."""

    table = "age_predictions"
    if len(frame) < 2:
        raise ContractError(f"{table} requires at least two rows")
    values = _validate_prediction_rows(frame, table=table)
    actual = values["actual_age"]
    predicted = values["predicted_age"]
    if np.var(actual) == 0.0 or np.var(predicted) == 0.0:
        raise ContractError(f"{table} requires variation in actual_age and predicted_age")

    residual = predicted - actual
    denominator = np.square(actual - actual.mean()).sum()
    return AgeMetrics(
        n_rows=int(len(frame)),
        n_participants=int(frame["participant_id"].nunique()),
        mae=float(np.abs(residual).mean()),
        pearson=float(np.corrcoef(actual, predicted)[0, 1]),
        r2=float(1.0 - np.square(residual).sum() / denominator),
    )


def fit_linear_age_correction(
    validation: pd.DataFrame,
    *,
    reference_col: str = "strict_healthy_reference_60m",
    organs: tuple[str, ...] = ORGANS,
) -> LinearAgeCorrectionModel:
    """Fit organ-specific expected age gap versus age on validation references."""

    table = "correction_validation"
    values = _validate_prediction_rows(validation, table=table)
    _require_columns(validation, (reference_col,), table=table)
    _require_exact_organs(validation, organs, table=table)
    reference = pd.to_numeric(validation[reference_col], errors="coerce")
    if reference.isna().any() or not set(reference.unique()).issubset({0, 1}):
        raise ContractError(f"{table}.{reference_col} must contain only 0/1")

    work = validation.copy()
    work["_actual_age"] = values["actual_age"]
    work["_age_gap"] = values["predicted_age"] - values["actual_age"]
    work = work.loc[reference.eq(1)].copy()
    parameters: list[OrganLinearCorrection] = []
    for organ in organs:
        subset = work.loc[work["organ_system"].astype(str).eq(organ)]
        if len(subset) < 3 or subset["_actual_age"].nunique() < 2:
            raise ContractError(f"{table} requires at least three reference rows and age variation for {organ}")
        design = np.column_stack([np.ones(len(subset)), subset["_actual_age"].to_numpy(dtype=np.float64)])
        coef, *_ = np.linalg.lstsq(design, subset["_age_gap"].to_numpy(dtype=np.float64), rcond=None)
        parameters.append(
            OrganLinearCorrection(
                organ_system=organ,
                intercept=float(coef[0]),
                age_slope=float(coef[1]),
                n_reference_rows=int(len(subset)),
            )
        )
    return LinearAgeCorrectionModel(
        parameters=tuple(parameters),
        validation_participant_ids=frozenset(validation["participant_id"].astype(str)),
    )


def _apply_linear_age_correction_unchecked(
    frame: pd.DataFrame,
    model: LinearAgeCorrectionModel,
    *,
    table: str,
) -> pd.DataFrame:
    values = _validate_prediction_rows(frame, table=table)
    expected_organs = tuple(parameter.organ_system for parameter in model.parameters)
    _require_exact_organs(frame, expected_organs, table=table)

    parameter_map = {parameter.organ_system: parameter for parameter in model.parameters}
    actual = values["actual_age"]
    predicted = values["predicted_age"]
    intercept = frame["organ_system"].astype(str).map(
        {organ: parameter.intercept for organ, parameter in parameter_map.items()}
    ).to_numpy(dtype=np.float64)
    slope = frame["organ_system"].astype(str).map(
        {organ: parameter.age_slope for organ, parameter in parameter_map.items()}
    ).to_numpy(dtype=np.float64)
    raw_gap = predicted - actual
    expected_gap = intercept + slope * actual

    result = frame.copy()
    result["raw_age_gap"] = raw_gap
    result["expected_age_gap"] = expected_gap
    result["corrected_age_gap"] = raw_gap - expected_gap
    result["corrected_predicted_age"] = actual + result["corrected_age_gap"].to_numpy(dtype=np.float64)
    return result


def apply_linear_age_correction_to_validation(
    validation: pd.DataFrame,
    model: LinearAgeCorrectionModel,
) -> pd.DataFrame:
    """Transform the same validation participants used to fit correction parameters."""

    observed_ids = frozenset(validation["participant_id"].astype(str))
    if observed_ids != model.validation_participant_ids:
        missing = len(model.validation_participant_ids - observed_ids)
        extra = len(observed_ids - model.validation_participant_ids)
        raise ContractError(f"correction validation participant_id mismatch: missing={missing}, extra={extra}")
    return _apply_linear_age_correction_unchecked(validation, model, table="correction_validation_apply")


def apply_linear_age_correction(
    test: pd.DataFrame,
    model: LinearAgeCorrectionModel,
) -> pd.DataFrame:
    """Apply a validation-fitted correction to a participant-disjoint test set."""

    overlap = model.validation_participant_ids.intersection(test["participant_id"].astype(str))
    if overlap:
        raise ContractError(f"validation/test participant_id overlap: {len(overlap)}")
    return _apply_linear_age_correction_unchecked(test, model, table="correction_test")


def _validate_survival_rows(
    frame: pd.DataFrame,
    *,
    table: str,
    organ_system: str,
) -> dict[str, np.ndarray]:
    required = (
        "participant_id",
        "organ_system",
        "actual_age",
        "sex",
        "corrected_age_gap",
        "duration",
        "event",
    )
    _require_columns(frame, required, table=table)
    if len(frame) < 3:
        raise ContractError(f"{table} requires at least three rows")
    _validate_participant_ids(frame, table=table)
    observed_organs = set(frame["organ_system"].astype(str))
    if observed_organs != {organ_system}:
        raise ContractError(f"{table}.organ_system must contain only {organ_system!r}")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise ContractError(f"{table} has duplicate participant_id and organ_system rows")

    values = _numeric_finite(
        frame,
        ("actual_age", "sex", "corrected_age_gap", "duration", "event"),
        table=table,
    )
    if (values["duration"] <= 0.0).any():
        raise ContractError(f"{table}.duration must be strictly positive")
    if not set(np.unique(values["event"])).issubset({0.0, 1.0}):
        raise ContractError(f"{table}.event must contain only 0/1")
    if not set(np.unique(values["sex"])).issubset({0.0, 1.0}):
        raise ContractError(f"{table}.sex must contain only 0/1")
    if values["event"].sum() < 1:
        raise ContractError(f"{table}.event contains no events")
    return values


def _held_out_cox_c_index(
    validation_values: dict[str, np.ndarray],
    test_values: dict[str, np.ndarray],
    covariates: tuple[str, ...],
    *,
    penalizer: float,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

    train = pd.DataFrame(
        {
            "duration": validation_values["duration"],
            "event": validation_values["event"].astype(np.int8),
        }
    )
    score = pd.DataFrame(index=np.arange(len(test_values["duration"])))
    transformed_columns: list[str] = []
    for covariate in covariates:
        mean = float(validation_values[covariate].mean())
        scale = float(validation_values[covariate].std(ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ContractError(f"cox_validation.{covariate} has no variation")
        transformed = f"z_{covariate}"
        train[transformed] = (validation_values[covariate] - mean) / scale
        score[transformed] = (test_values[covariate] - mean) / scale
        transformed_columns.append(transformed)

    model = CoxPHFitter(penalizer=penalizer)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                train[["duration", "event", *transformed_columns]],
                duration_col="duration",
                event_col="event",
                show_progress=False,
            )
        risk = model.predict_log_partial_hazard(score[transformed_columns]).to_numpy(dtype=np.float64)
        c_index = concordance_index(
            test_values["duration"],
            -risk,
            test_values["event"].astype(bool),
        )
    except Exception as exc:
        raise ContractError(f"held-out Cox evaluation failed: {exc}") from exc
    if not np.isfinite(c_index):
        raise ContractError("held-out Cox evaluation returned a non-finite C-index")
    return float(c_index)


def evaluate_cox_cell(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    fold: int,
    organ_system: str,
    penalizer: float = 1e-4,
) -> CoxCellResult:
    """Fit age+sex and age+sex+gap Cox models on validation, score on test."""

    if not isinstance(fold, (int, np.integer)) or int(fold) not in range(5):
        raise ContractError("fold must be one of 0, 1, 2, 3, 4")
    if organ_system not in ORGANS:
        raise ContractError(f"unknown organ_system: {organ_system!r}")
    if not np.isfinite(penalizer) or penalizer < 0.0:
        raise ContractError("penalizer must be finite and non-negative")

    validation_values = _validate_survival_rows(
        validation,
        table="cox_validation",
        organ_system=organ_system,
    )
    test_values = _validate_survival_rows(test, table="cox_test", organ_system=organ_system)
    validation_ids = frozenset(validation["participant_id"].astype(str))
    overlap = validation_ids.intersection(test["participant_id"].astype(str))
    if overlap:
        raise ContractError(f"validation/test participant_id overlap: {len(overlap)}")

    base_covariates = ("actual_age", "sex")
    plus_covariates = (*base_covariates, "corrected_age_gap")
    c_base = _held_out_cox_c_index(
        validation_values,
        test_values,
        base_covariates,
        penalizer=penalizer,
    )
    c_plus = _held_out_cox_c_index(
        validation_values,
        test_values,
        plus_covariates,
        penalizer=penalizer,
    )
    return CoxCellResult(
        fold=int(fold),
        organ_system=organ_system,
        n_validation=int(len(validation)),
        n_test=int(len(test)),
        validation_events=int(validation_values["event"].sum()),
        test_events=int(test_values["event"].sum()),
        c_base=c_base,
        c_plus=c_plus,
        delta_c=float(c_plus - c_base),
    )


def aggregate_cox_cells(
    cells: pd.DataFrame,
    *,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    organs: tuple[str, ...] = ORGANS,
) -> CoxAggregate:
    """Aggregate complete held-out fold-organ cells, organ first then macro."""

    table = "cox_cells"
    required = (
        "fold",
        "organ_system",
        "n_validation",
        "n_test",
        "validation_events",
        "test_events",
        "c_base",
        "c_plus",
        "delta_c",
    )
    _require_columns(cells, required, table=table)
    if cells.duplicated(["fold", "organ_system"]).any():
        raise ContractError(f"{table} has duplicate fold and organ_system cells")

    numeric = _numeric_finite(
        cells,
        (
            "fold",
            "n_validation",
            "n_test",
            "validation_events",
            "test_events",
            "c_base",
            "c_plus",
            "delta_c",
        ),
        table=table,
    )
    if not np.equal(numeric["fold"], np.floor(numeric["fold"])).all():
        raise ContractError(f"{table}.fold must contain integers")
    observed = set(zip(numeric["fold"].astype(int), cells["organ_system"].astype(str)))
    expected = {(fold, organ) for fold in folds for organ in organs}
    if len(cells) != len(expected) or observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ContractError(
            f"{table} must contain exactly {len(expected)} fold-organ cells; "
            f"missing={missing}, extra={extra}"
        )

    for count_column in ("n_validation", "n_test", "validation_events", "test_events"):
        values = numeric[count_column]
        if not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise ContractError(f"{table}.{count_column} must contain non-negative integers")
    if (numeric["n_validation"] <= 0).any() or (numeric["n_test"] <= 0).any():
        raise ContractError(f"{table} validation/test row counts must be positive")
    if (numeric["validation_events"] > numeric["n_validation"]).any():
        raise ContractError(f"{table}.validation_events exceeds n_validation")
    if (numeric["test_events"] > numeric["n_test"]).any():
        raise ContractError(f"{table}.test_events exceeds n_test")
    if ((numeric["c_base"] < 0.0) | (numeric["c_base"] > 1.0)).any():
        raise ContractError(f"{table}.c_base must lie in [0, 1]")
    if ((numeric["c_plus"] < 0.0) | (numeric["c_plus"] > 1.0)).any():
        raise ContractError(f"{table}.c_plus must lie in [0, 1]")
    if not np.allclose(
        numeric["delta_c"],
        numeric["c_plus"] - numeric["c_base"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ContractError(f"{table}.delta_c must equal c_plus - c_base")

    work = cells.copy()
    for column, values in numeric.items():
        work[column] = values
    summaries: list[OrganDeltaCSummary] = []
    for organ in organs:
        subset = work.loc[work["organ_system"].astype(str).eq(organ)]
        summaries.append(
            OrganDeltaCSummary(
                organ_system=organ,
                mean_c_base=float(subset["c_base"].mean()),
                mean_c_plus=float(subset["c_plus"].mean()),
                mean_delta_c=float(subset["delta_c"].mean()),
                positive_folds=int((subset["delta_c"] > 0.0).sum()),
            )
        )
    organ_tuple = tuple(summaries)
    return CoxAggregate(
        n_cells=int(len(cells)),
        organs=organ_tuple,
        macro_c_base=float(np.mean([item.mean_c_base for item in organ_tuple])),
        macro_c_plus=float(np.mean([item.mean_c_plus for item in organ_tuple])),
        macro_delta_c=float(np.mean([item.mean_delta_c for item in organ_tuple])),
        positive_organs=int(sum(item.mean_delta_c > 0.0 for item in organ_tuple)),
    )
