from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

import v8_contract
import v8_runtime_guard


class SchemaError(RuntimeError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_schemas(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    if value.get("schema_version") != "revision0710.fig04_fig05.executable_output_schemas.v8":
        raise SchemaError("output schema version changed")
    if not isinstance(value.get("tables"), dict):
        raise SchemaError("output schema tables missing")
    return value


def schema_for_execution(
    schemas: dict[str, Any],
    table: str,
    *,
    execution_mode: str,
    evaluation_epoch: int,
) -> dict[str, Any]:
    if table not in schemas["tables"]:
        raise SchemaError(f"unknown output table: {table}")
    if execution_mode not in {"formal", "scaled_integration"}:
        raise SchemaError("unknown execution mode")
    if execution_mode == "formal" and evaluation_epoch != 55:
        raise SchemaError("formal output schema is E55 only")
    if execution_mode == "scaled_integration" and evaluation_epoch != 1:
        raise SchemaError("scaled integration output schema is E1 only")
    schema = copy.deepcopy(schemas["tables"][table])
    if execution_mode == "scaled_integration":
        specs = schema.get("columns", schema.get("fields", {}))
        if "evaluation_epoch" in specs:
            specs["evaluation_epoch"]["const"] = 1
        if "evaluation_epoch" in schema.get("constraints", {}):
            schema["constraints"]["evaluation_epoch"]["const"] = 1
        if table == "checkpoint_manifest":
            schema["fields"]["status"]["const"] = "SEALED_SCALED_CAPTURE_ONLY"
        elif table == "cox_nuisance":
            schema["fields"]["penalizer"]["const"] = 0.1
        elif table == "fold_receipt":
            schema["fields"]["status"]["const"] = (
                "COMPLETE_SCALED_CAPTURE_ONLY_NO_SELECTION"
            )
    return schema


def read_csv_exact(
    path: Path,
    schema: dict[str, Any],
    *,
    table: str,
) -> pd.DataFrame:
    """Read a CSV without pandas numeric inference, then apply the frozen dtypes."""

    if schema.get("kind") != "table":
        raise SchemaError(f"{table} is not a table schema")
    expected_columns = list(schema["column_order"])
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        observed_columns = next(reader, None)
    if observed_columns != expected_columns:
        raise SchemaError(f"{table} CSV header is not exact")
    raw = pd.read_csv(
        path,
        dtype={column: "string" for column in expected_columns},
        keep_default_na=False,
        na_filter=False,
    )
    if list(raw.columns) != expected_columns:
        raise SchemaError(f"{table} CSV columns changed during read")
    converted: dict[str, pd.Series] = {}
    for column in expected_columns:
        spec = schema["columns"][column]
        values = raw[column]
        dtype = spec["dtype"]
        if dtype == "string":
            converted[column] = values.astype("string")
        elif dtype == "int64":
            try:
                numeric = pd.to_numeric(values, errors="raise")
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"{table}.{column} cannot be parsed as int64") from exc
            if not np.equal(numeric.to_numpy(np.float64), np.floor(numeric.to_numpy(np.float64))).all():
                raise SchemaError(f"{table}.{column} contains a non-integer token")
            converted[column] = numeric.astype(np.int64)
        elif dtype == "float64":
            try:
                converted[column] = pd.to_numeric(values, errors="raise").astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"{table}.{column} cannot be parsed as float64") from exc
        elif dtype == "bool":
            mapping = {"True": True, "False": False, "true": True, "false": False}
            if not values.isin(mapping).all():
                raise SchemaError(f"{table}.{column} contains a non-boolean token")
            converted[column] = values.map(mapping).astype(bool)
        else:
            raise SchemaError(f"unsupported CSV dtype: {dtype}")
    frame = pd.DataFrame(converted, columns=expected_columns)
    validate_table(frame, schema, table=table)
    return frame


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_series(series: pd.Series, spec: dict[str, Any], *, column: str) -> None:
    if not spec.get("nullable", False) and series.isna().any():
        raise SchemaError(f"{column} contains null values")
    nonnull = series.dropna()
    dtype = spec["dtype"]
    if dtype == "string":
        if not (ptypes.is_object_dtype(series.dtype) or ptypes.is_string_dtype(series.dtype)):
            raise SchemaError(f"{column} dtype is not string")
        if any(not isinstance(value, str) for value in nonnull):
            raise SchemaError(f"{column} contains non-string values")
    elif dtype == "int64":
        if not ptypes.is_integer_dtype(series.dtype):
            raise SchemaError(f"{column} dtype is not integer")
    elif dtype == "float64":
        if not ptypes.is_float_dtype(series.dtype):
            raise SchemaError(f"{column} dtype is not float")
        if len(nonnull) and not np.isfinite(nonnull.to_numpy(np.float64)).all():
            raise SchemaError(f"{column} contains non-finite values")
    elif dtype == "bool":
        if not ptypes.is_bool_dtype(series.dtype):
            raise SchemaError(f"{column} dtype is not bool")
    else:
        raise SchemaError(f"unsupported dtype: {dtype}")
    if "const" in spec and len(nonnull) and not nonnull.eq(spec["const"]).all():
        raise SchemaError(f"{column} violates const")
    if "enum" in spec and not set(nonnull.tolist()).issubset(set(spec["enum"])):
        raise SchemaError(f"{column} violates enum")
    if "pattern" in spec and any(re.fullmatch(spec["pattern"], value) is None for value in nonnull):
        raise SchemaError(f"{column} violates pattern")
    numeric = ptypes.is_numeric_dtype(nonnull.dtype)
    if numeric and "min" in spec and (nonnull < spec["min"]).any():
        raise SchemaError(f"{column} violates min")
    if numeric and "max" in spec and (nonnull > spec["max"]).any():
        raise SchemaError(f"{column} violates max")
    if numeric and "min_exclusive" in spec and (nonnull <= spec["min_exclusive"]).any():
        raise SchemaError(f"{column} violates exclusive minimum")


def validate_table(frame: pd.DataFrame, schema: dict[str, Any], *, table: str) -> None:
    if schema.get("kind") != "table":
        raise SchemaError(f"{table} is not a table schema")
    order = list(schema["column_order"])
    if list(frame.columns) != order or set(schema["columns"]) != set(order):
        raise SchemaError(f"{table} columns are not exact")
    for column in order:
        _validate_series(frame[column], schema["columns"][column], column=f"{table}.{column}")
    primary = list(schema.get("primary_key", []))
    if primary and frame.duplicated(primary).any():
        raise SchemaError(f"{table} primary key is duplicated")
    for formula in schema.get("formulas", []):
        op = formula["op"]
        tolerance = float(formula.get("tolerance", 0.0))
        target = frame[formula["target"]].to_numpy(np.float64)
        if op == "subtract":
            expected = (
                frame[formula["left"]].to_numpy(np.float64)
                - frame[formula["right"]].to_numpy(np.float64)
            )
            if not np.allclose(target, expected, atol=tolerance, rtol=0.0):
                raise SchemaError(f"{table} formula failed: {formula['target']}")
        elif op == "multiply_const":
            expected = frame[formula["left"]].to_numpy(np.float64) * float(
                formula["constant"]
            )
            if not np.allclose(target, expected, atol=tolerance, rtol=0.0):
                raise SchemaError(f"{table} formula failed: {formula['target']}")
        elif op == "implies":
            if (frame["event"].astype(int) > frame["at_risk"].astype(int)).any():
                raise SchemaError(f"{table} event occurs outside risk set")
        else:
            raise SchemaError(f"unsupported formula op: {op}")
    cardinality = schema.get("cardinality", {})
    if "minimum_rows" in cardinality and len(frame) < int(cardinality["minimum_rows"]):
        raise SchemaError(f"{table} minimum row count failed")
    if "rows_per_participant" in cardinality:
        groups = frame.groupby(cardinality["group_by"], dropna=False)
        counts = groups.size()
        if not counts.eq(int(cardinality["rows_per_participant"])).all():
            raise SchemaError(f"{table} participant x organ cardinality failed")
        organ_column = cardinality["organ_column"]
        expected_organs = set(cardinality["organ_values"])
        if any(set(cell[organ_column]) != expected_organs for _, cell in groups):
            raise SchemaError(f"{table} organ coverage failed")
    if "rows_per_run" in cardinality:
        counts = frame.groupby("run_id", dropna=False).size()
        if not counts.eq(int(cardinality["rows_per_run"])).all():
            raise SchemaError(f"{table} run cardinality failed")
        if "organ_values" in cardinality:
            expected_organs = set(cardinality["organ_values"])
            if any(set(cell[cardinality["organ_column"]]) != expected_organs for _, cell in frame.groupby("run_id")):
                raise SchemaError(f"{table} organ run coverage failed")


def independently_recompute_outer_metrics(
    prediction: pd.DataFrame,
    nuisance: dict[str, Any],
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    """Recompute all reported metrics from prediction rows and sealed nuisance."""

    from lifelines.utils import concordance_index

    strict = prediction.loc[prediction["strict_healthy_reference_60m"].eq(1)]
    if strict.empty:
        raise SchemaError("strict metric population is empty")
    actual = strict["actual_age"].to_numpy(np.float64)
    predicted = strict["predicted_age"].to_numpy(np.float64)
    residual = predicted - actual
    denominator = np.square(actual - actual.mean()).sum()
    if not np.isfinite(denominator) or denominator <= 0:
        raise SchemaError("strict metric age variance is zero")
    fold: dict[str, float | int] = {
        "strict_n_participants": int(strict["participant_id"].nunique()),
        "strict_n_participant_organ_rows": int(len(strict)),
        "raw_strict_mae": float(np.abs(residual).mean()),
        "raw_strict_pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "raw_strict_r2": float(1.0 - np.square(residual).sum() / denominator),
    }
    models = {
        (str(row["organ_system"]), str(row["model"])): row
        for row in nuisance["models"]
    }
    expected_cells = {
        (organ, model)
        for organ in v8_contract.ORGAN_SYSTEMS
        for model in ("base", "full")
    }
    if set(models) != expected_cells:
        raise SchemaError("sealed nuisance organ/model coverage changed")
    organ_rows: list[dict[str, float | int | str]] = []
    for organ in v8_contract.ORGAN_SYSTEMS:
        cell = prediction.loc[
            prediction["organ_system"].eq(organ) & prediction["at_risk"].eq(1)
        ]
        if cell.empty or int(cell["event"].sum()) < 1:
            raise SchemaError(f"outer risk population is insufficient: {organ}")
        scores: dict[str, float] = {}
        for model_name in ("base", "full"):
            model = models[(organ, model_name)]
            risk = np.zeros(len(cell), dtype=np.float64)
            for covariate in model["covariates"]:
                values = cell[covariate].to_numpy(np.float64)
                mean = float(model["means"][covariate])
                scale = float(model["scales"][covariate])
                coefficient = float(model["coefficients"][covariate])
                if not np.isfinite(scale) or scale <= 0:
                    raise SchemaError(f"sealed nuisance scale is invalid: {organ}/{model_name}")
                risk += (values - mean) / scale * coefficient
            scores[model_name] = float(
                concordance_index(
                    cell["duration_months"].to_numpy(np.float64),
                    -risk,
                    cell["event"].to_numpy(bool),
                )
            )
        organ_rows.append(
            {
                "organ_system": organ,
                "at_risk_n": int(len(cell)),
                "event_n": int(cell["event"].sum()),
                "age_sex_harrell_base_c": scores["base"],
                "age_sex_gap_harrell_full_c": scores["full"],
                "age_sex_harrell_delta_c": scores["full"] - scores["base"],
            }
        )
    fold["age_sex_harrell_macro_delta_c"] = float(
        np.mean([float(row["age_sex_harrell_delta_c"]) for row in organ_rows])
    )
    return fold, organ_rows


def _participant_ids_sha256(frame: pd.DataFrame) -> str:
    return v8_contract.canonical_json_sha256(
        {"participant_ids": sorted(frame["participant_id"].astype(str).unique().tolist())}
    )


def independently_refit_age_correction(
    validation: pd.DataFrame,
) -> dict[str, Any]:
    strict = validation.loc[validation["strict_healthy_reference_60m"].eq(1)]
    parameters: list[dict[str, Any]] = []
    for organ in v8_contract.ORGAN_SYSTEMS:
        cell = strict.loc[strict["organ_system"].eq(organ)]
        if len(cell) < 3 or cell["actual_age"].nunique() < 2:
            raise SchemaError(f"validation strict reference is insufficient: {organ}")
        design = np.column_stack(
            [np.ones(len(cell)), cell["actual_age"].to_numpy(np.float64)]
        )
        coefficients, *_ = np.linalg.lstsq(
            design, cell["raw_age_gap"].to_numpy(np.float64), rcond=None
        )
        parameters.append(
            {
                "organ_system": organ,
                "intercept": float(coefficients[0]),
                "age_slope": float(coefficients[1]),
                "fit_participants": int(cell["participant_id"].nunique()),
                "fit_rows": int(len(cell)),
                "fit_participant_ids_sha256": _participant_ids_sha256(cell),
            }
        )
    return {
        "participant_ids_sha256": _participant_ids_sha256(validation),
        "parameters": parameters,
    }


def independently_apply_age_correction(
    frame: pd.DataFrame,
    correction: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    parameters = {
        str(row["organ_system"]): row for row in correction["parameters"]
    }
    if set(parameters) != set(v8_contract.ORGAN_SYSTEMS):
        raise SchemaError("independent age-correction organ coverage changed")
    intercept = result["organ_system"].map(
        {organ: float(row["intercept"]) for organ, row in parameters.items()}
    ).to_numpy(np.float64)
    slope = result["organ_system"].map(
        {organ: float(row["age_slope"]) for organ, row in parameters.items()}
    ).to_numpy(np.float64)
    expected_bias = intercept + slope * result["actual_age"].to_numpy(np.float64)
    result["corrected_age_gap"] = (
        result["raw_age_gap"].to_numpy(np.float64) - expected_bias
    )
    return result


def independently_refit_cox_nuisance(
    validation: pd.DataFrame,
    *,
    penalizer: float,
) -> dict[str, Any]:
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning

    models: list[dict[str, Any]] = []
    for organ in v8_contract.ORGAN_SYSTEMS:
        cell = validation.loc[
            validation["organ_system"].eq(organ) & validation["at_risk"].eq(1)
        ].copy()
        if cell.empty or int(cell["event"].sum()) < 1:
            raise SchemaError(f"validation Cox population is insufficient: {organ}")
        for model_name, covariates in (
            ("base", ("actual_age", "sex")),
            ("full", ("actual_age", "sex", "corrected_age_gap")),
        ):
            train = cell[["duration_months", "event", *covariates]].copy()
            means = {
                name: float(train[name].to_numpy(np.float64).mean())
                for name in covariates
            }
            scales = {
                name: float(train[name].to_numpy(np.float64).std(ddof=0))
                for name in covariates
            }
            if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
                raise SchemaError(f"validation Cox variation failed: {organ}/{model_name}")
            z_names: list[str] = []
            for name in covariates:
                z_name = f"z_{name}"
                train[z_name] = (train[name] - means[name]) / scales[name]
                z_names.append(z_name)
            estimator = CoxPHFitter(penalizer=float(penalizer))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    estimator.fit(
                        train[["duration_months", "event", *z_names]],
                        duration_col="duration_months",
                        event_col="event",
                        show_progress=False,
                    )
                except ConvergenceWarning as exc:
                    raise SchemaError(
                        f"validation Cox convergence warning: {organ}/{model_name}"
                    ) from exc
            models.append(
                {
                    "organ_system": organ,
                    "model": model_name,
                    "covariates": list(covariates),
                    "means": means,
                    "scales": scales,
                    "coefficients": {
                        name: float(estimator.params_[f"z_{name}"])
                        for name in covariates
                    },
                    "fit_participants": int(cell["participant_id"].nunique()),
                    "fit_events": int(cell["event"].sum()),
                    "fit_participant_ids_sha256": _participant_ids_sha256(cell),
                    "convergence_warning": False,
                }
            )
    return {"penalizer": float(penalizer), "models": models}


def _assert_numeric_mapping_close(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
    tolerance: float,
) -> None:
    if set(observed) != set(expected):
        raise SchemaError(f"{label} keys changed")
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not np.isclose(
                float(observed_value), expected_value, atol=tolerance, rtol=0.0
            ):
                raise SchemaError(f"{label}.{key} differs from independent refit")
        elif observed_value != expected_value:
            raise SchemaError(f"{label}.{key} differs from independent refit")


def validate_independently_refitted_nuisance(
    validation: pd.DataFrame,
    stored_correction: dict[str, Any],
    stored_nuisance: dict[str, Any],
    *,
    tolerance: float = 1e-8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    independent_correction = independently_refit_age_correction(validation)
    if stored_correction["participant_ids_sha256"] != independent_correction["participant_ids_sha256"]:
        raise SchemaError("age correction participant ID hash differs from validation rows")
    observed_parameters = {
        str(row["organ_system"]): row for row in stored_correction["parameters"]
    }
    expected_parameters = {
        str(row["organ_system"]): row for row in independent_correction["parameters"]
    }
    if set(observed_parameters) != set(expected_parameters):
        raise SchemaError("age correction organ coverage differs from independent refit")
    for organ in v8_contract.ORGAN_SYSTEMS:
        _assert_numeric_mapping_close(
            observed_parameters[organ],
            expected_parameters[organ],
            label=f"age_correction.{organ}",
            tolerance=tolerance,
        )
    corrected = independently_apply_age_correction(validation, independent_correction)
    if not np.allclose(
        validation["corrected_age_gap"].to_numpy(np.float64),
        corrected["corrected_age_gap"].to_numpy(np.float64),
        atol=tolerance,
        rtol=0.0,
    ):
        raise SchemaError("validation corrected age gaps differ from independent refit")
    independent_nuisance = independently_refit_cox_nuisance(
        corrected, penalizer=float(stored_nuisance["penalizer"])
    )
    observed_models = {
        (str(row["organ_system"]), str(row["model"])): row
        for row in stored_nuisance["models"]
    }
    expected_models = {
        (str(row["organ_system"]), str(row["model"])): row
        for row in independent_nuisance["models"]
    }
    if set(observed_models) != set(expected_models):
        raise SchemaError("Cox nuisance cells differ from independent refit")
    for key in sorted(expected_models):
        observed = observed_models[key]
        expected = expected_models[key]
        for field in (
            "organ_system", "model", "covariates", "fit_participants",
            "fit_events", "fit_participant_ids_sha256", "convergence_warning",
        ):
            if observed[field] != expected[field]:
                raise SchemaError(f"Cox nuisance {key}/{field} differs from independent refit")
        for field in ("means", "scales", "coefficients"):
            _assert_numeric_mapping_close(
                observed[field],
                expected[field],
                label=f"cox_nuisance.{key}.{field}",
                tolerance=tolerance,
            )
    return independent_correction, independent_nuisance


def validate_prediction_role_alignment(
    frame: pd.DataFrame,
    bundle: Any,
    *,
    split_role: str,
    tolerance: float = 1e-6,
) -> None:
    """Compare every exported label to the frozen role bundle in row order."""

    participant_ids = np.repeat(
        np.asarray(bundle.participant_ids, dtype=object), len(v8_contract.ORGAN_SYSTEMS)
    )
    organs = np.tile(
        np.asarray(v8_contract.ORGAN_SYSTEMS, dtype=object), len(bundle.participant_ids)
    )
    if frame["participant_id"].astype(str).tolist() != participant_ids.astype(str).tolist():
        raise SchemaError(f"{split_role} participant row order differs from frozen role")
    if frame["organ_system"].astype(str).tolist() != organs.astype(str).tolist():
        raise SchemaError(f"{split_role} organ row order differs from frozen role")
    if set(frame["split_role"].astype(str)) != {split_role}:
        raise SchemaError(f"{split_role} split label changed")

    age = np.repeat(
        bundle.labels["chronological_age"].detach().cpu().numpy().astype(np.float64),
        len(v8_contract.ORGAN_SYSTEMS),
    )
    sex = np.repeat(
        bundle.inputs["sex"].detach().cpu().numpy().astype(np.int64),
        len(v8_contract.ORGAN_SYSTEMS),
    )
    strict = np.repeat(
        bundle.labels["strict_healthy_reference"].detach().cpu().numpy().astype(np.int64),
        len(v8_contract.ORGAN_SYSTEMS),
    )
    duration = bundle.labels["duration_months"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    event = bundle.labels["event"].detach().cpu().numpy().astype(np.int64).reshape(-1)
    at_risk = bundle.labels["at_risk"].detach().cpu().numpy().astype(np.int64).reshape(-1)
    for column, expected in (
        ("sex", sex),
        ("strict_healthy_reference_60m", strict),
        ("event", event),
        ("at_risk", at_risk),
    ):
        if not np.array_equal(frame[column].to_numpy(np.int64), expected):
            raise SchemaError(f"{split_role}.{column} differs from frozen role labels")


def expected_task_opened_paths(
    task: dict[str, Any],
    *,
    root: Path,
    phase_inputs: list[Path],
) -> list[str]:
    """Independently derive the exact declared input ledger for one worker phase."""

    paths = [Path(task["task_path"]), Path(task["claim_path"])]
    if task["protocol"]["execution_mode"] == "formal":
        paths.extend(
            [
                Path(task["contract_path"]),
                root / "ABLATION_QUEUE_V8.csv",
                Path(task["authorization"]["path"]),
                *v8_runtime_guard.lock_binding_paths(task["gpu_lock"]),
            ]
        )
    paths.extend(phase_inputs)
    source = task["data_source"]
    if source["format"] == "synthetic_role_bundle":
        paths.append(Path(source["role_bundle_path"]))
    else:
        contract = v8_contract.load_contract(root)
        manifest_path = Path(source["role_manifest_path"])
        manifest = v8_contract.read_json_exact(manifest_path)
        artifact = manifest["artifact"]
        cache_dir = Path(source["cache_dir"])
        bindings = contract["cache_artifact_bindings"]
        paths.extend(
            [
                manifest_path,
                Path(artifact["age_targets"]["path"]),
                Path(artifact["joint_labels"]["path"]),
                cache_dir / bindings["batch_manifest"]["path"],
                cache_dir / bindings["fold_transform_manifest"]["path"],
                cache_dir
                / bindings["fold_transforms"][int(task["outer_fold"])]["path"],
            ]
        )
        selected = {int(value) for value in artifact["fold_ids"]}
        paths.extend(
            cache_dir / row["path"]
            for row in bindings["shards"]
            if int(row["fold_id"]) in selected
        )
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = str(path.resolve())
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
    for column, expected in (("actual_age", age), ("duration_months", duration)):
        if not np.allclose(
            frame[column].to_numpy(np.float64), expected, atol=tolerance, rtol=0.0
        ):
            raise SchemaError(f"{split_role}.{column} differs from frozen role labels")


def _structural_optimizer_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "initial_trainable_sha256",
            "initial_full_model_sha256",
            "initial_optimizer_sha256",
            "initial_scaler_sha256",
        }
    }


def validate_actual_model_wiring_optimizer_history_and_replay(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    variant_id: str,
    outer_bundle: Any,
    validation_bundle: Any,
    code_embedding: Any,
    text_embedding: Any,
    prediction: pd.DataFrame,
    validation_prediction: pd.DataFrame,
    optimizer_manifest: dict[str, Any],
    wiring: dict[str, Any],
    history: dict[str, Any],
    protocol: dict[str, Any],
    train_participants: int,
    replay_device: str,
    prediction_tolerance: float = 2e-5,
) -> dict[str, Any]:
    import torch

    import v8_inputs
    import v8_loss
    import v8_model
    import v8_training
    from ehr_age_model import (
        BASE_AGE_HEAD_MODE_MLP,
        BASE_AGE_MODE_PRIVATE_RETAIN,
        EHRAGEConfig,
    )

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise SchemaError("checkpoint state_dict is empty or invalid")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise SchemaError("checkpoint state_dict contains a non-tensor")
    if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
        raise SchemaError("checkpoint state_dict contains a non-finite tensor")
    try:
        config = EHRAGEConfig(**checkpoint["model_config"])
        expected_config = EHRAGEConfig(
            hidden_dim=12 if protocol["execution_mode"] == "scaled_integration" else 192,
            num_heads=3 if protocol["execution_mode"] == "scaled_integration" else 6,
            num_transformer_layers=1 if protocol["execution_mode"] == "scaled_integration" else 3,
            num_gru_layers=1 if protocol["execution_mode"] == "scaled_integration" else 2,
            dropout=0.0 if protocol["execution_mode"] == "scaled_integration" else 0.15,
            freeze_event_embeddings=True,
            risk_offset_cap_years=8.0,
            risk_temperature=2.0,
            survival_intervals=5,
            baseline_log_rate_init=-7.5,
            base_age_mode=BASE_AGE_MODE_PRIVATE_RETAIN,
            base_age_head_mode=BASE_AGE_HEAD_MODE_MLP,
        )
        if config != expected_config:
            raise SchemaError("checkpoint model config differs from the frozen production architecture")
        model = v8_model.build_production_model(
            variant_id,
            code_embedding,
            text_embedding,
            age_anchor=float(checkpoint["age_anchor"]),
            config=config,
        )
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise SchemaError("checkpoint does not strictly restore the production model") from exc

    actual_optimizer_manifest = v8_training.build_optimizer_manifest(model, variant_id)
    if _structural_optimizer_manifest(optimizer_manifest) != _structural_optimizer_manifest(
        actual_optimizer_manifest
    ):
        raise SchemaError("optimizer manifest differs from the actual production model")
    if checkpoint["optimizer_parameter_names"] != actual_optimizer_manifest["optimizer_parameter_names"]:
        raise SchemaError("checkpoint optimizer names differ from the actual production model")
    expected_wiring = {
        "variant_id": variant_id,
        "model": v8_model.model_wiring_receipt(model),
        "input": v8_inputs.variant_input_policy(variant_id),
        "loss": v8_loss.variant_loss_policy(variant_id),
        "optimizer_parameter_names": actual_optimizer_manifest["optimizer_parameter_names"],
        "frozen_or_unused_parameter_names": actual_optimizer_manifest[
            "frozen_or_unused_parameter_names"
        ],
    }
    if wiring != expected_wiring:
        raise SchemaError("production wiring differs from the instantiated model/policies")
    optimizer = v8_training.build_adamw(
        model,
        actual_optimizer_manifest,
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    v8_training.validate_optimizer_exact(model, optimizer, actual_optimizer_manifest)
    for group in optimizer.param_groups:
        if not np.isclose(float(group["lr"]), float(protocol["learning_rate"]), atol=0.0, rtol=0.0):
            raise SchemaError("actual optimizer learning rate changed")
        if not np.isclose(float(group["weight_decay"]), float(protocol["weight_decay"]), atol=0.0, rtol=0.0):
            raise SchemaError("actual optimizer weight decay changed")

    rows = history.get("history")
    epochs = int(protocol["train_through_epoch"])
    if not isinstance(rows, list) or len(rows) != epochs:
        raise SchemaError("training history length differs from the frozen protocol")
    expected_steps = int(np.ceil(train_participants / int(protocol["batch_size"])))
    loss_policy = v8_loss.variant_loss_policy(variant_id)
    expected_route = str(loss_policy["loss_route"])
    expected_dropout = float(loss_policy["event_dropout_probability"])
    history_keys = {
        "epoch", "optimizer_steps", "mean_total_loss", "learning_rate",
        "max_preclip_gradient_norm", "amp_autocast_entered", "loss_route",
        "event_dropout_probability", "step_chain_sha256", "model_state_sha256",
        "optimizer_state_sha256", "scaler_state_sha256", "trajectory_sha256",
    }
    for epoch, row in enumerate(rows, start=1):
        if set(row) != history_keys or int(row["epoch"]) != epoch:
            raise SchemaError(f"training history E{epoch} schema/epoch changed")
        expected_lr = float(protocol["learning_rate"]) * v8_training.learning_rate_multiplier(
            epoch,
            warmup_epochs=int(protocol["warmup_epochs"]),
            horizon_epochs=int(protocol["scheduler_horizon_epochs"]),
        )
        if int(row["optimizer_steps"]) != expected_steps:
            raise SchemaError(f"training history E{epoch} optimizer steps changed")
        if not np.isclose(float(row["learning_rate"]), expected_lr, atol=1e-15, rtol=0.0):
            raise SchemaError(f"training history E{epoch} learning rate changed")
        if row["loss_route"] != expected_route:
            raise SchemaError(f"training history E{epoch} loss route changed")
        if not np.isclose(float(row["event_dropout_probability"]), expected_dropout, atol=0.0, rtol=0.0):
            raise SchemaError(f"training history E{epoch} event dropout changed")
        if row["amp_autocast_entered"] is not bool(protocol["amp"]):
            raise SchemaError(f"training history E{epoch} AMP route changed")
        values = (float(row["mean_total_loss"]), float(row["max_preclip_gradient_norm"]))
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise SchemaError(f"training history E{epoch} contains invalid loss/gradient values")
        for key in (
            "step_chain_sha256", "model_state_sha256", "optimizer_state_sha256",
            "scaler_state_sha256", "trajectory_sha256",
        ):
            if not isinstance(row[key], str) or len(row[key]) != 64:
                raise SchemaError(f"training history E{epoch} {key} changed")
    expected_model_dropout = 0.0 if protocol["execution_mode"] == "scaled_integration" else 0.15
    if not np.isclose(float(config.dropout), expected_model_dropout, atol=0.0, rtol=0.0):
        raise SchemaError("checkpoint architecture dropout differs from the frozen protocol")

    device = torch.device(replay_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SchemaError("formal checkpoint replay requires CUDA")
    v8_training.configure_reproducibility(
        int(checkpoint["seed"]), deterministic=True, tf32=True
    )
    role_receipts: dict[str, Any] = {}
    for role, bundle, stored in (
        ("validation", validation_bundle, validation_prediction),
        ("test", outer_bundle, prediction),
    ):
        replay = __import__("v8_data").prediction_frame(
            model,
            bundle,
            variant_id=variant_id,
            run_id=str(stored["run_id"].iloc[0]),
            evaluation_epoch=int(protocol["evaluate_only_epoch"]),
            device=device,
            inference_batch_size=min(512, int(protocol["batch_size"])),
        )
        replay["split_role"] = role
        validate_prediction_role_alignment(replay, bundle, split_role=role)
        for column in ("predicted_age", "risk_score", "raw_age_gap"):
            if not np.allclose(
                stored[column].to_numpy(np.float64),
                replay[column].to_numpy(np.float64),
                atol=prediction_tolerance,
                rtol=0.0,
            ):
                raise SchemaError(
                    f"{role} prediction {column} differs from production checkpoint replay"
                )
        role_receipts[role] = {
            "participant_count": len(bundle.participant_ids),
            "prediction_rows": len(replay),
        }
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "roles": role_receipts,
        "inference_batch_size": min(512, int(protocol["batch_size"])),
        "replay_device": device.type,
    }


def load_canonical_role_for_deep_validation(
    *,
    contract: dict[str, Any],
    root: Path,
    outer_fold: int,
    role: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import v8_data

    name = f"outer_fold_{outer_fold}_{role}_manifest"
    records = [
        record for record in contract["canonical_bindings"]["records"]
        if record["name"] == name
    ]
    if len(records) != 1:
        raise SchemaError(f"canonical role binding is missing: {name}")
    record = records[0]
    manifest_path = v8_contract.repo_root(root) / str(record["path"])
    manifest = v8_contract.read_json_exact(manifest_path)
    bundle, code, text = v8_data.load_revision0710_role(
        role_manifest_path=manifest_path,
        expected_manifest_sha256=str(record["sha256"]),
        cache_dir=v8_contract.repo_root(root) / v8_contract.CACHE_RELATIVE_ROOT,
        outer_fold=outer_fold,
        expected_role=role,
        cache_artifact_bindings=contract["cache_artifact_bindings"],
        expected_cache_artifact_bindings_sha256=v8_contract.cache_artifact_bindings_sha256(
            contract["cache_artifact_bindings"]
        ),
    )
    return bundle, code, text, manifest


def validate_independently_recomputed_metrics(
    prediction: pd.DataFrame,
    nuisance: dict[str, Any],
    fold_metrics: pd.DataFrame,
    organ_metrics: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> None:
    expected_fold, expected_organs = independently_recompute_outer_metrics(
        prediction, nuisance
    )
    if len(fold_metrics) != 1:
        raise SchemaError("fold metrics must contain exactly one row")
    observed_fold = fold_metrics.iloc[0]
    for key, expected in expected_fold.items():
        observed = observed_fold[key]
        if isinstance(expected, int):
            if int(observed) != expected:
                raise SchemaError(f"independent fold metric mismatch: {key}")
        elif not np.isclose(float(observed), float(expected), atol=tolerance, rtol=0.0):
            raise SchemaError(f"independent fold metric mismatch: {key}")
    observed_by_organ = {
        str(row["organ_system"]): row
        for _, row in organ_metrics.iterrows()
    }
    if set(observed_by_organ) != set(v8_contract.ORGAN_SYSTEMS):
        raise SchemaError("independent organ metric coverage mismatch")
    for expected in expected_organs:
        observed = observed_by_organ[str(expected["organ_system"])]
        for key in (
            "at_risk_n",
            "event_n",
            "age_sex_harrell_base_c",
            "age_sex_gap_harrell_full_c",
            "age_sex_harrell_delta_c",
        ):
            expected_value = expected[key]
            observed_value = observed[key]
            if isinstance(expected_value, int):
                if int(observed_value) != expected_value:
                    raise SchemaError(f"independent organ metric mismatch: {expected['organ_system']}/{key}")
            elif not np.isclose(
                float(observed_value), float(expected_value), atol=tolerance, rtol=0.0
            ):
                raise SchemaError(f"independent organ metric mismatch: {expected['organ_system']}/{key}")


def validate_training_trajectory_closure(
    attestation: dict[str, Any],
    *,
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
    optimizer_manifest: dict[str, Any],
    history: dict[str, Any],
    task: dict[str, Any],
    claim_sha256: str,
    train_source_sha256: str,
) -> None:
    import v8_training

    expected_keys = {
        "schema_version", "status", "variant_id", "seed", "run_id", "outer_fold",
        "initial_full_model_sha256", "initial_trainable_sha256",
        "initial_optimizer_sha256", "initial_scaler_sha256", "completed_epochs",
        "capture_epoch", "capture_checkpoint_sha256", "capture_model_state_sha256",
        "capture_trajectory_sha256", "final_model_state_sha256",
        "final_optimizer_state_sha256", "final_scaler_state_sha256",
        "final_trajectory_sha256", "epochs", "task_sha256", "claim_sha256",
        "train_role_source_sha256", "protocol_sha256",
    }
    if set(attestation) != expected_keys:
        raise SchemaError("training trajectory attestation fields changed")
    expected = {
        "schema_version": "revision0710.fig04_fig05.training_trajectory_attestation.v8",
        "status": "COMPLETE_INDEPENDENT_FULL_REPLAY_REQUIRED",
        "variant_id": task["variant_id"],
        "seed": int(task["seed"]),
        "run_id": task["run_id"],
        "outer_fold": int(task["outer_fold"]),
        "completed_epochs": int(task["protocol"]["train_through_epoch"]),
        "capture_epoch": int(task["protocol"]["capture_epoch"]),
        "capture_checkpoint_sha256": checkpoint_sha256,
        "task_sha256": sha256_file(Path(task["task_path"])),
        "claim_sha256": claim_sha256,
        "train_role_source_sha256": train_source_sha256,
        "protocol_sha256": v8_contract.canonical_json_sha256(task["protocol"]),
    }
    for key, value in expected.items():
        if attestation.get(key) != value:
            raise SchemaError(f"training trajectory cross-binding changed: {key}")
    source_key = (
        "role_bundle_sha256"
        if task["data_source"]["format"] == "synthetic_role_bundle"
        else "role_manifest_sha256"
    )
    if train_source_sha256 != task["data_source"][source_key]:
        raise SchemaError("training trajectory source binding changed")
    for key in (
        "initial_full_model_sha256", "initial_trainable_sha256",
        "initial_optimizer_sha256",
    ):
        if attestation[key] != checkpoint[key]:
            raise SchemaError(f"checkpoint trajectory binding changed: {key}")
    for key in (
        "initial_full_model_sha256", "initial_trainable_sha256",
        "initial_optimizer_sha256", "initial_scaler_sha256",
    ):
        if attestation[key] != optimizer_manifest[key]:
            raise SchemaError(f"optimizer trajectory binding changed: {key}")
    checkpoint_model_sha = v8_training.canonical_state_sha256(
        checkpoint["model_state_dict"]
    )
    if (
        attestation["capture_model_state_sha256"] != checkpoint_model_sha
        or attestation["capture_trajectory_sha256"]
        != checkpoint["training_trajectory_sha256_at_capture"]
    ):
        raise SchemaError("captured checkpoint differs from trajectory attestation")
    epoch_keys = {
        "epoch", "optimizer_steps", "step_chain_sha256", "model_state_sha256",
        "optimizer_state_sha256", "scaler_state_sha256", "trajectory_sha256",
    }
    epochs = attestation["epochs"]
    rows = history["history"]
    if not isinstance(epochs, list) or len(epochs) != len(rows):
        raise SchemaError("training trajectory epoch coverage changed")
    for epoch, (attested, recorded) in enumerate(zip(epochs, rows, strict=True), start=1):
        if set(attested) != epoch_keys or int(attested["epoch"]) != epoch:
            raise SchemaError(f"training trajectory E{epoch} schema changed")
        if attested != {key: recorded[key] for key in epoch_keys}:
            raise SchemaError(f"training trajectory E{epoch} differs from history")
    last = epochs[-1]
    for attestation_key, epoch_key in (
        ("final_model_state_sha256", "model_state_sha256"),
        ("final_optimizer_state_sha256", "optimizer_state_sha256"),
        ("final_scaler_state_sha256", "scaler_state_sha256"),
        ("final_trajectory_sha256", "trajectory_sha256"),
    ):
        if attestation[attestation_key] != last[epoch_key]:
            raise SchemaError(f"training trajectory final binding changed: {attestation_key}")


def order_and_validate_table(
    frame: pd.DataFrame, schema: dict[str, Any], *, table: str
) -> pd.DataFrame:
    order = list(schema.get("column_order", []))
    if set(frame.columns) != set(order):
        raise SchemaError(f"{table} columns are not closed")
    result = frame.loc[:, order].copy()
    validate_table(result, schema, table=table)
    return result


def _validate_logical_array(value: Any, logical_type: str, *, field: str) -> None:
    if not isinstance(value, list):
        raise SchemaError(f"{field} logical array changed")
    if logical_type == "array_7_organs":
        if len(value) != 7 or {row.get("organ_system") for row in value} != set(v8_contract.ORGAN_SYSTEMS):
            raise SchemaError(f"{field} organ coverage changed")
        expected = {
            "organ_system", "intercept", "age_slope", "fit_participants",
            "fit_rows", "fit_participant_ids_sha256",
        }
        for row in value:
            if not isinstance(row, dict) or set(row) != expected:
                raise SchemaError(f"{field} parameter keys changed")
            if not all(np.isfinite(float(row[key])) for key in ("intercept", "age_slope")):
                raise SchemaError(f"{field} parameter is non-finite")
            if not isinstance(row["fit_participants"], int) or row["fit_participants"] < 1:
                raise SchemaError(f"{field} participant count changed")
            if not isinstance(row["fit_rows"], int) or row["fit_rows"] < row["fit_participants"]:
                raise SchemaError(f"{field} fit row count changed")
            if re.fullmatch(r"[0-9a-f]{64}", str(row["fit_participant_ids_sha256"])) is None:
                raise SchemaError(f"{field} participant ID hash changed")
    elif logical_type == "array_14_organ_model_cells":
        expected_cells = {
            (organ, model)
            for organ in v8_contract.ORGAN_SYSTEMS
            for model in ("base", "full")
        }
        observed = {(row.get("organ_system"), row.get("model")) for row in value if isinstance(row, dict)}
        if len(value) != 14 or observed != expected_cells:
            raise SchemaError(f"{field} Cox cell coverage changed")
        expected = {
            "organ_system", "model", "covariates", "means", "scales",
            "coefficients", "fit_participants", "fit_events",
            "fit_participant_ids_sha256", "convergence_warning",
        }
        for row in value:
            if set(row) != expected:
                raise SchemaError(f"{field} Cox cell keys changed")
            covariates = ["actual_age", "sex"]
            if row["model"] == "full":
                covariates.append("corrected_age_gap")
            if row["covariates"] != covariates:
                raise SchemaError(f"{field} Cox covariates changed")
            if any(set(row[name]) != set(covariates) for name in ("means", "scales", "coefficients")):
                raise SchemaError(f"{field} Cox coefficient keys changed")
            for name in ("means", "scales", "coefficients"):
                if not all(np.isfinite(float(cell)) for cell in row[name].values()):
                    raise SchemaError(f"{field} Cox value is non-finite")
            if any(float(cell) <= 0 for cell in row["scales"].values()):
                raise SchemaError(f"{field} Cox scale is not positive")
            if not isinstance(row["fit_participants"], int) or row["fit_participants"] < 1:
                raise SchemaError(f"{field} Cox participant count changed")
            if not isinstance(row["fit_events"], int) or row["fit_events"] < 1:
                raise SchemaError(f"{field} Cox event count changed")
            if re.fullmatch(r"[0-9a-f]{64}", str(row["fit_participant_ids_sha256"])) is None:
                raise SchemaError(f"{field} Cox participant ID hash changed")
            if row["convergence_warning"] is not False:
                raise SchemaError(f"{field} Cox convergence warning is not fail-closed")
    elif logical_type == "outer_role_array":
        if value != ["outer_test"]:
            raise SchemaError(f"{field} outer role changed")
    elif logical_type == "nonempty_path_array":
        if not value or any(not isinstance(path, str) or not path for path in value):
            raise SchemaError(f"{field} paths changed")
    else:
        raise SchemaError(f"unsupported logical type: {logical_type}")


def validate_json_payload(payload: dict[str, Any], schema: dict[str, Any], *, table: str) -> None:
    if schema.get("kind") != "json" or set(payload) != set(schema["fields"]):
        raise SchemaError(f"{table} JSON fields are not exact")
    frame = pd.DataFrame({key: [value] for key, value in payload.items() if schema["fields"][key]["dtype"] != "string" or not isinstance(value, (list, dict))})
    for key, spec in schema["fields"].items():
        value = payload[key]
        if spec.get("logical_type"):
            _validate_logical_array(
                value, str(spec["logical_type"]), field=f"{table}.{key}"
            )
            continue
        _validate_series(pd.Series([value]), spec, column=f"{table}.{key}")
    if table == "attempt_receipt":
        completed = payload["status"] == "COMPLETE"
        if completed != (payload["fold_receipt_sha256"] is not None):
            raise SchemaError("attempt receipt completion hash changed")
        if completed == (payload["failure_receipt_sha256"] is not None):
            raise SchemaError("attempt receipt failure hash changed")


def synthetic_prediction_frame(*, run_id: str, participants: int) -> pd.DataFrame:
    variant = run_id.rsplit("__outer", 1)[0]
    fold = int(run_id.rsplit("__outer", 1)[1])
    rows = []
    for participant in range(participants):
        age = 50.0 + participant * 4.0
        for organ_index, organ in enumerate(v8_contract.ORGAN_SYSTEMS):
            predicted = age + 0.1 * organ_index + 0.01 * participant
            rows.append(
                {
                    "participant_id": f"p{participant:03d}",
                    "run_id": run_id,
                    "variant_id": variant,
                    "outer_fold": fold,
                    "split_role": "test",
                    "evaluation_epoch": 55,
                    "organ_system": organ,
                    "actual_age": age,
                    "predicted_age": predicted,
                    "raw_age_gap": predicted - age,
                    "corrected_age_gap": predicted - age - 0.02,
                    "risk_score": 0.05 * organ_index,
                    "sex": participant % 2,
                    "strict_healthy_reference_60m": int(participant % 2 == 0),
                    "duration_months": 20.0 + participant + organ_index,
                    "event": int((participant + organ_index) % 3 == 0),
                    "at_risk": 1,
                    "checkpoint_sha256": "a" * 64,
                    "age_correction_sha256": "b" * 64,
                    "cox_nuisance_sha256": "c" * 64,
                }
            )
    frame = pd.DataFrame(rows)
    for column in ("outer_fold", "evaluation_epoch", "sex", "strict_healthy_reference_60m", "event", "at_risk"):
        frame[column] = frame[column].astype("int64")
    for column in ("actual_age", "predicted_age", "raw_age_gap", "corrected_age_gap", "risk_score", "duration_months"):
        frame[column] = frame[column].astype("float64")
    return frame


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_synthetic_run_bundle(root: Path, run_id: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("synthetic run bundle root must be empty")
    prediction = synthetic_prediction_frame(run_id=run_id, participants=3)
    prediction_path = root / "prediction_rows.csv"
    prediction.to_csv(prediction_path, index=False, lineterminator="\n")
    artifacts: dict[str, dict[str, Any]] = {
        "prediction_rows": {"path": prediction_path.name, "sha256": sha256_file(prediction_path)}
    }
    simple_names = (
        "checkpoint_manifest",
        "age_correction",
        "cox_nuisance",
        "outer_read_ledger",
    )
    for name in simple_names:
        path = root / f"{name}.json"
        _write_json(path, {"name": name, "run_id": run_id})
        artifacts[name] = {"path": path.name, "sha256": sha256_file(path)}
    fold_metrics_path = root / "fold_metrics.csv"
    pd.DataFrame({"run_id": [run_id]}).to_csv(fold_metrics_path, index=False, lineterminator="\n")
    artifacts["fold_metrics"] = {"path": fold_metrics_path.name, "sha256": sha256_file(fold_metrics_path)}
    organ_metrics_path = root / "organ_metrics.csv"
    pd.DataFrame({"run_id": [run_id] * 7, "organ_system": list(v8_contract.ORGAN_SYSTEMS)}).to_csv(organ_metrics_path, index=False, lineterminator="\n")
    artifacts["organ_metrics"] = {"path": organ_metrics_path.name, "sha256": sha256_file(organ_metrics_path)}
    fold_receipt = {
        "checkpoint_manifest_sha256": artifacts["checkpoint_manifest"]["sha256"],
        "prediction_rows_sha256": artifacts["prediction_rows"]["sha256"],
        "age_correction_sha256": artifacts["age_correction"]["sha256"],
        "cox_nuisance_sha256": artifacts["cox_nuisance"]["sha256"],
        "outer_read_ledger_sha256": artifacts["outer_read_ledger"]["sha256"],
        "fold_metrics_sha256": artifacts["fold_metrics"]["sha256"],
        "organ_metrics_sha256": artifacts["organ_metrics"]["sha256"],
    }
    fold_receipt_path = root / "fold_receipt.json"
    _write_json(fold_receipt_path, fold_receipt)
    artifacts["fold_receipt"] = {"path": fold_receipt_path.name, "sha256": sha256_file(fold_receipt_path)}
    for name in ("attempt_receipt", "global_aggregate"):
        path = root / f"{name}.json"
        _write_json(path, {"name": name, "run_id": run_id, "fold_receipt_sha256": artifacts["fold_receipt"]["sha256"]})
        artifacts[name] = {"path": path.name, "sha256": sha256_file(path)}
    return {"artifacts": artifacts, "fold_receipt": fold_receipt}


def validate_run_bundle(root: Path, bundle: dict[str, Any]) -> None:
    expected_names = {
        "checkpoint_manifest",
        "prediction_rows",
        "age_correction",
        "cox_nuisance",
        "outer_read_ledger",
        "fold_metrics",
        "organ_metrics",
        "fold_receipt",
        "attempt_receipt",
        "global_aggregate",
    }
    artifacts = bundle.get("artifacts", {})
    if set(artifacts) != expected_names:
        raise SchemaError("run bundle artifact set changed")
    for name, record in artifacts.items():
        path = root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise SchemaError(f"run bundle hash changed: {name}")
    receipt = bundle.get("fold_receipt", {})
    links = {
        "checkpoint_manifest_sha256": "checkpoint_manifest",
        "prediction_rows_sha256": "prediction_rows",
        "age_correction_sha256": "age_correction",
        "cox_nuisance_sha256": "cox_nuisance",
        "outer_read_ledger_sha256": "outer_read_ledger",
        "fold_metrics_sha256": "fold_metrics",
        "organ_metrics_sha256": "organ_metrics",
    }
    for key, name in links.items():
        if receipt.get(key) != artifacts[name]["sha256"]:
            label = "prediction" if name == "prediction_rows" else name
            raise SchemaError(f"{label} cross-hash changed")


def validate_formal_attempt_bundle(
    attempt: Path,
    *,
    queue_row: dict[str, Any],
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Deeply revalidate the complete immutable artifact closure for one run."""

    import torch

    schemas = load_schemas(root / "OUTPUT_SCHEMAS_V8.json")
    artifact_paths = {
        "claim": attempt / "CLAIM.json",
        "train_task": attempt / "TRAIN_TASK.json",
        "calibration_task": attempt / "CALIBRATION_TASK.json",
        "outer_task": attempt / "OUTER_TASK.json",
        "checkpoint": attempt / "01_train" / "fixed_epoch_0055.pt",
        "checkpoint_manifest": attempt / "01_train" / "CHECKPOINT_MANIFEST.json",
        "training_complete": attempt / "01_train" / "TRAINING_COMPLETE.json",
        "model_optimizer_manifest": attempt / "01_train" / "MODEL_OPTIMIZER_MANIFEST.json",
        "production_wiring": attempt / "01_train" / "PRODUCTION_WIRING.json",
        "training_history": attempt / "01_train" / "TRAINING_HISTORY.json",
        "training_trajectory_attestation": attempt / "01_train" / "TRAINING_TRAJECTORY_ATTESTATION.json",
        "age_correction": attempt / "02_calibration" / "age_correction.json",
        "cox_nuisance": attempt / "02_calibration" / "cox_nuisance.json",
        "validation_predictions": attempt / "02_calibration" / "validation_predictions.csv",
        "calibration_complete": attempt / "02_calibration" / "CALIBRATION_COMPLETE.json",
        "outer_read_ledger": attempt / "OUTER_READ_LEDGER.json",
        "prediction_rows": attempt / "03_outer_test" / "prediction_rows.csv",
        "fold_metrics": attempt / "03_outer_test" / "fold_metrics.csv",
        "organ_metrics": attempt / "03_outer_test" / "organ_metrics.csv",
        "fold_receipt": attempt / "03_outer_test" / "FOLD_RECEIPT.json",
        "attempt_receipt": attempt / "ATTEMPT_RECEIPT.json",
        "complete": attempt / "COMPLETE.json",
    }
    missing = [name for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise SchemaError(f"formal attempt artifact set is incomplete: {missing}")

    json_schema_names = {
        "checkpoint_manifest",
        "age_correction",
        "cox_nuisance",
        "outer_read_ledger",
        "fold_receipt",
        "attempt_receipt",
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name in artifact_paths:
        path = artifact_paths[name]
        if path.suffix != ".json":
            continue
        payload = v8_contract.read_json_exact(path)
        if name in json_schema_names:
            schema = (
                schema_for_execution(
                    schemas, name, execution_mode="formal", evaluation_epoch=55
                )
                if name != "attempt_receipt"
                else schemas["tables"][name]
            )
            validate_json_payload(payload, schema, table=name)
        payloads[name] = payload

    prediction_schema = schema_for_execution(
        schemas, "prediction_rows", execution_mode="formal", evaluation_epoch=55
    )
    frames = {
        "validation_predictions": read_csv_exact(
            artifact_paths["validation_predictions"],
            prediction_schema,
            table="validation_predictions",
        ),
        "prediction_rows": read_csv_exact(
            artifact_paths["prediction_rows"], prediction_schema, table="prediction_rows"
        ),
        "fold_metrics": read_csv_exact(
            artifact_paths["fold_metrics"],
            schema_for_execution(
                schemas, "fold_metrics", execution_mode="formal", evaluation_epoch=55
            ),
            table="fold_metrics",
        ),
        "organ_metrics": read_csv_exact(
            artifact_paths["organ_metrics"],
            schema_for_execution(
                schemas, "organ_metrics", execution_mode="formal", evaluation_epoch=55
            ),
            table="organ_metrics",
        ),
    }

    run_id = str(queue_row["run_id"])
    variant_id = str(queue_row["variant_id"])
    outer_fold = int(queue_row["outer_fold"])
    import isolated_worker

    tasks = {
        "train": payloads["train_task"],
        "calibration": payloads["calibration_task"],
        "outer_test": payloads["outer_task"],
    }
    claim_sha = sha256_file(artifact_paths["claim"])
    for phase, task in tasks.items():
        isolated_worker.validate_task_structure(task, expected_phase=phase)
        if (
            task["run_id"] != run_id
            or task["variant_id"] != variant_id
            or int(task["outer_fold"]) != outer_fold
            or Path(task["task_path"]).resolve()
            != artifact_paths[f"{phase if phase != 'outer_test' else 'outer'}_task"].resolve()
            or Path(task["claim_path"]).resolve() != artifact_paths["claim"].resolve()
            or task["claim_sha256"] != claim_sha
        ):
            raise SchemaError(f"formal {phase} task/claim closure changed")
    checkpoint_sha = sha256_file(artifact_paths["checkpoint"])
    checkpoint_bytes = artifact_paths["checkpoint"].stat().st_size
    checkpoint_manifest = payloads["checkpoint_manifest"]
    if (
        checkpoint_manifest["run_id"] != run_id
        or checkpoint_manifest["variant_id"] != variant_id
        or int(checkpoint_manifest["outer_fold"]) != outer_fold
        or checkpoint_manifest["checkpoint_sha256"] != checkpoint_sha
        or int(checkpoint_manifest["checkpoint_bytes"]) != checkpoint_bytes
    ):
        raise SchemaError("checkpoint manifest actual file binding changed")
    checkpoint = torch.load(
        artifact_paths["checkpoint"], map_location="cpu", weights_only=False
    )
    expected_checkpoint_keys = {
        "schema_version", "status", "variant_id", "epoch", "seed",
        "model_config", "age_anchor", "model_state_dict",
        "optimizer_parameter_names", "initial_trainable_sha256",
        "initial_full_model_sha256", "initial_optimizer_sha256",
        "training_trajectory_sha256_at_capture",
    }
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != expected_checkpoint_keys
        or checkpoint["schema_version"] != "revision0710.fig04_fig05.ablation_checkpoint.v8"
        or checkpoint["status"] != "CAPTURED_FIXED_EPOCH"
        or checkpoint["variant_id"] != variant_id
        or int(checkpoint["epoch"]) != 55
        or int(checkpoint["seed"]) != 7210
        or checkpoint["initial_trainable_sha256"] != checkpoint_manifest["initial_trainable_sha256"]
    ):
        raise SchemaError("checkpoint payload identity/content changed")

    training = payloads["training_complete"]
    training_keys = {
        "schema_version", "status", "run_id", "variant_id", "outer_fold", "seed",
        "completed_epochs", "capture_epoch", "checkpoint_path", "checkpoint_sha256",
        "checkpoint_manifest_path", "checkpoint_manifest_sha256",
        "model_optimizer_manifest_sha256", "production_wiring_sha256",
        "training_history_sha256", "training_trajectory_attestation_path",
        "training_trajectory_attestation_sha256", "task_sha256", "claim_sha256",
        "forward_called", "backward_called",
        "gradient_clip_called", "scheduler_called", "amp_path_entered",
        "fresh_initialization", "resume_used", "opened_roles", "opened_paths",
        "calibration_read_count", "outer_test_read_count",
    }
    if set(training) != training_keys:
        raise SchemaError("training completion fields changed")
    expected_training_values = {
        "schema_version": "revision0710.fig04_fig05.training_complete.v8",
        "status": "TRAINING_COMPLETE_CAPTURE_SEALED",
        "run_id": run_id,
        "variant_id": variant_id,
        "outer_fold": outer_fold,
        "seed": 7210,
        "completed_epochs": 60,
        "capture_epoch": 55,
        "checkpoint_path": str(artifact_paths["checkpoint"].resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_manifest_path": str(artifact_paths["checkpoint_manifest"].resolve()),
        "checkpoint_manifest_sha256": sha256_file(artifact_paths["checkpoint_manifest"]),
        "model_optimizer_manifest_sha256": sha256_file(artifact_paths["model_optimizer_manifest"]),
        "production_wiring_sha256": sha256_file(artifact_paths["production_wiring"]),
        "training_history_sha256": sha256_file(artifact_paths["training_history"]),
        "training_trajectory_attestation_path": str(
            artifact_paths["training_trajectory_attestation"].resolve()
        ),
        "training_trajectory_attestation_sha256": sha256_file(
            artifact_paths["training_trajectory_attestation"]
        ),
        "task_sha256": sha256_file(artifact_paths["train_task"]),
        "claim_sha256": sha256_file(artifact_paths["claim"]),
        "forward_called": True,
        "backward_called": True,
        "gradient_clip_called": True,
        "scheduler_called": True,
        "amp_path_entered": True,
        "fresh_initialization": True,
        "resume_used": False,
        "opened_roles": ["train"],
        "calibration_read_count": 0,
        "outer_test_read_count": 0,
    }
    for key, expected in expected_training_values.items():
        if training.get(key) != expected:
            raise SchemaError(f"training completion cross-binding changed: {key}")
    if not isinstance(training["opened_paths"], list) or not training["opened_paths"]:
        raise SchemaError("training opened path ledger changed")

    optimizer_manifest = payloads["model_optimizer_manifest"]
    optimizer_keys = {
        "variant_id", "model_parameter_names", "model_parameter_count",
        "trainable_parameter_names", "trainable_parameter_count",
        "optimizer_parameter_names", "optimizer_parameter_count",
        "frozen_or_unused_parameter_names", "initial_trainable_sha256",
        "optimizer_name_set_sha256", "initial_full_model_sha256",
        "initial_optimizer_sha256", "initial_scaler_sha256",
    }
    if set(optimizer_manifest) != optimizer_keys or optimizer_manifest["variant_id"] != variant_id:
        raise SchemaError("model/optimizer manifest changed")
    if (
        optimizer_manifest["initial_trainable_sha256"] != checkpoint_manifest["initial_trainable_sha256"]
        or optimizer_manifest["optimizer_name_set_sha256"] != checkpoint_manifest["optimizer_name_set_sha256"]
        or int(optimizer_manifest["model_parameter_count"]) != int(checkpoint_manifest["model_parameter_count"])
        or int(optimizer_manifest["trainable_parameter_count"]) != int(checkpoint_manifest["trainable_parameter_count"])
        or optimizer_manifest["optimizer_parameter_names"] != checkpoint["optimizer_parameter_names"]
    ):
        raise SchemaError("model/optimizer/checkpoint cross-binding changed")
    wiring = payloads["production_wiring"]
    if set(wiring) != {
        "variant_id", "model", "input", "loss", "optimizer_parameter_names",
        "frozen_or_unused_parameter_names",
    } or wiring["variant_id"] != variant_id:
        raise SchemaError("production wiring fields/identity changed")
    if (
        wiring["optimizer_parameter_names"] != optimizer_manifest["optimizer_parameter_names"]
        or wiring["frozen_or_unused_parameter_names"] != optimizer_manifest["frozen_or_unused_parameter_names"]
    ):
        raise SchemaError("production wiring optimizer binding changed")
    history = payloads["training_history"]
    if set(history) != {"history"} or not isinstance(history["history"], list):
        raise SchemaError("training history envelope changed")
    if [row.get("epoch") for row in history["history"]] != list(range(1, 61)):
        raise SchemaError("training history epoch sequence changed")
    history_keys = {
        "epoch", "optimizer_steps", "mean_total_loss", "learning_rate",
        "max_preclip_gradient_norm", "amp_autocast_entered", "loss_route",
        "event_dropout_probability",
        "step_chain_sha256", "model_state_sha256", "optimizer_state_sha256",
        "scaler_state_sha256", "trajectory_sha256",
    }
    if any(set(row) != history_keys for row in history["history"]):
        raise SchemaError("training history row schema changed")
    validate_training_trajectory_closure(
        payloads["training_trajectory_attestation"],
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        optimizer_manifest=optimizer_manifest,
        history=history,
        task=tasks["train"],
        claim_sha256=claim_sha,
        train_source_sha256=tasks["train"]["data_source"]["role_manifest_sha256"],
    )

    calibration = payloads["calibration_complete"]
    calibration_keys = {
        "schema_version", "status", "run_id", "training_complete_sha256",
        "checkpoint_sha256", "age_correction_path", "age_correction_sha256",
        "cox_nuisance_path", "cox_nuisance_sha256", "validation_predictions_sha256",
        "task_sha256", "claim_sha256", "model_selection_used", "opened_roles",
        "opened_paths", "outer_test_read_count",
    }
    if set(calibration) != calibration_keys:
        raise SchemaError("calibration completion fields changed")
    expected_calibration_values = {
        "schema_version": "revision0710.fig04_fig05.calibration_complete.v8",
        "status": "CALIBRATION_NUISANCE_SEALED_NO_SELECTION",
        "run_id": run_id,
        "training_complete_sha256": sha256_file(artifact_paths["training_complete"]),
        "checkpoint_sha256": checkpoint_sha,
        "age_correction_path": str(artifact_paths["age_correction"].resolve()),
        "age_correction_sha256": sha256_file(artifact_paths["age_correction"]),
        "cox_nuisance_path": str(artifact_paths["cox_nuisance"].resolve()),
        "cox_nuisance_sha256": sha256_file(artifact_paths["cox_nuisance"]),
        "validation_predictions_sha256": sha256_file(artifact_paths["validation_predictions"]),
        "task_sha256": sha256_file(artifact_paths["calibration_task"]),
        "claim_sha256": sha256_file(artifact_paths["claim"]),
        "model_selection_used": False,
        "opened_roles": ["calibration"],
        "outer_test_read_count": 0,
    }
    for key, expected in expected_calibration_values.items():
        if calibration.get(key) != expected:
            raise SchemaError(f"calibration completion cross-binding changed: {key}")
    if not isinstance(calibration["opened_paths"], list) or not calibration["opened_paths"]:
        raise SchemaError("calibration opened path ledger changed")
    expected_train_opened = expected_task_opened_paths(
        tasks["train"], root=root, phase_inputs=[]
    )
    expected_calibration_opened = expected_task_opened_paths(
        tasks["calibration"],
        root=root,
        phase_inputs=[artifact_paths["training_complete"], artifact_paths["checkpoint"]],
    )
    if training["opened_paths"] != expected_train_opened:
        raise SchemaError("training exact opened-path closure changed")
    if calibration["opened_paths"] != expected_calibration_opened:
        raise SchemaError("calibration exact opened-path closure changed")

    for name in (
        "checkpoint_manifest", "age_correction", "cox_nuisance",
        "outer_read_ledger", "fold_receipt", "attempt_receipt",
    ):
        if payloads[name].get("run_id") != run_id:
            raise SchemaError(f"formal {name} run identity changed")
    prediction = frames["prediction_rows"]
    validation = frames["validation_predictions"]
    for name, frame, role in (
        ("prediction_rows", prediction, "test"),
        ("validation_predictions", validation, "validation"),
    ):
        if (
            set(frame["run_id"].astype(str)) != {run_id}
            or set(frame["variant_id"].astype(str)) != {variant_id}
            or set(frame["outer_fold"].astype(int)) != {outer_fold}
            or set(frame["split_role"].astype(str)) != {role}
        ):
            raise SchemaError(f"formal {name} identity/role changed")
    for name in ("fold_metrics", "organ_metrics"):
        frame = frames[name]
        if (
            set(frame["run_id"].astype(str)) != {run_id}
            or set(frame["variant_id"].astype(str)) != {variant_id}
            or set(frame["outer_fold"].astype(int)) != {outer_fold}
        ):
            raise SchemaError(f"formal {name} identity changed")

    def validate_role_coverage(frame: pd.DataFrame, role: str) -> None:
        record_name = f"outer_fold_{outer_fold}_{role}_manifest"
        records = [
            record for record in contract["canonical_bindings"]["records"]
            if record["name"] == record_name
        ]
        if len(records) != 1:
            raise SchemaError(f"formal role record is missing: {role}")
        manifest = v8_contract.read_json_exact(
            v8_contract.repo_root(root) / records[0]["path"]
        )
        artifact = manifest["artifact"]
        if len(frame) != int(artifact["participant_organ_rows"]):
            raise SchemaError(f"formal {role} prediction row count changed")
        ids = sorted(frame["participant_id"].astype(str).unique().tolist())
        if len(ids) != int(artifact["participant_count"]):
            raise SchemaError(f"formal {role} participant count changed")
        if v8_contract.canonical_json_sha256({"participant_ids": ids}) != artifact["participant_ids_canonical_sha256"]:
            raise SchemaError(f"formal {role} participant coverage changed")

    validate_role_coverage(validation, "calibration")
    validate_role_coverage(prediction, "outer_test")

    calibration_bundle, _, _, _ = load_canonical_role_for_deep_validation(
        contract=contract,
        root=root,
        outer_fold=outer_fold,
        role="calibration",
    )
    outer_bundle, outer_code, outer_text, _ = load_canonical_role_for_deep_validation(
        contract=contract,
        root=root,
        outer_fold=outer_fold,
        role="outer_test",
    )
    validate_prediction_role_alignment(
        validation, calibration_bundle, split_role="validation"
    )
    validate_prediction_role_alignment(prediction, outer_bundle, split_role="test")
    train_record = next(
        record for record in contract["canonical_bindings"]["records"]
        if record["name"] == f"outer_fold_{outer_fold}_train_manifest"
    )
    train_manifest = v8_contract.read_json_exact(
        v8_contract.repo_root(root) / str(train_record["path"])
    )
    training_protocol = contract["training_protocol"]
    formal_protocol = {
        "execution_mode": "formal",
        "batch_size": int(training_protocol["batch_size"]),
        "learning_rate": float(training_protocol["learning_rate"]),
        "weight_decay": float(training_protocol["weight_decay"]),
        "gradient_clip_norm": float(training_protocol["gradient_clip_norm"]),
        "warmup_epochs": int(training_protocol["warmup_epochs"]),
        "scheduler_horizon_epochs": int(training_protocol["scheduler_horizon_epochs"]),
        "train_through_epoch": int(training_protocol["train_through_epoch"]),
        "capture_epoch": int(training_protocol["capture_epoch"]),
        "evaluate_only_epoch": int(training_protocol["evaluate_only_epoch"]),
        "amp": bool(training_protocol["amp"]),
        "device": "cuda",
    }
    replay_receipt = validate_actual_model_wiring_optimizer_history_and_replay(
        checkpoint=checkpoint,
        checkpoint_path=artifact_paths["checkpoint"],
        variant_id=variant_id,
        outer_bundle=outer_bundle,
        validation_bundle=calibration_bundle,
        code_embedding=outer_code,
        text_embedding=outer_text,
        prediction=prediction,
        validation_prediction=validation,
        optimizer_manifest=optimizer_manifest,
        wiring=wiring,
        history=history,
        protocol=formal_protocol,
        train_participants=int(train_manifest["artifact"]["participant_count"]),
        replay_device="cuda",
    )

    correction_sha = sha256_file(artifact_paths["age_correction"])
    nuisance_sha = sha256_file(artifact_paths["cox_nuisance"])
    ledger_sha = sha256_file(artifact_paths["outer_read_ledger"])
    prediction_sha = sha256_file(artifact_paths["prediction_rows"])
    training_sha = sha256_file(artifact_paths["training_complete"])
    ledger = payloads["outer_read_ledger"]
    outer_role_record = next(
        record for record in contract["canonical_bindings"]["records"]
        if record["name"] == f"outer_fold_{outer_fold}_outer_test_manifest"
    )
    expected_ledger = {
        "schema_version": "revision0710.fig04_fig05.outer_read_ledger.v8",
        "status": "COMMITTED_BEFORE_OUTER_OPEN",
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_sha,
        "training_complete_sha256": training_sha,
        "age_correction_sha256": correction_sha,
        "cox_nuisance_sha256": nuisance_sha,
        "outer_role_manifest_sha256": outer_role_record["sha256"],
        "outer_opened_before_commit": False,
    }
    if ledger != expected_ledger:
        raise SchemaError("outer ledger exact cross-binding changed")
    for frame_name, frame in (
        ("validation_predictions", validation), ("prediction_rows", prediction)
    ):
        expected_shas = {
            "checkpoint_sha256": checkpoint_sha,
            "age_correction_sha256": correction_sha,
            "cox_nuisance_sha256": nuisance_sha,
        }
        for column, expected in expected_shas.items():
            if set(frame[column].astype(str)) != {expected}:
                raise SchemaError(f"{frame_name} cross-binding changed: {column}")

    receipt = payloads["fold_receipt"]
    expected_outer_opened = expected_task_opened_paths(
        tasks["outer_test"],
        root=root,
        phase_inputs=[
            artifact_paths["outer_read_ledger"],
            artifact_paths["training_complete"],
            artifact_paths["calibration_complete"],
            artifact_paths["checkpoint"],
            artifact_paths["age_correction"],
            artifact_paths["cox_nuisance"],
        ],
    )
    if (
        receipt.get("task_sha256") != sha256_file(artifact_paths["outer_task"])
        or receipt.get("claim_sha256") != claim_sha
        or receipt.get("opened_paths") != expected_outer_opened
    ):
        raise SchemaError("outer exact task/claim/opened-path closure changed")
    receipt_links = {
        "checkpoint_manifest_sha256": artifact_paths["checkpoint_manifest"],
        "prediction_rows_sha256": artifact_paths["prediction_rows"],
        "age_correction_sha256": artifact_paths["age_correction"],
        "cox_nuisance_sha256": artifact_paths["cox_nuisance"],
        "outer_read_ledger_sha256": artifact_paths["outer_read_ledger"],
        "fold_metrics_sha256": artifact_paths["fold_metrics"],
        "organ_metrics_sha256": artifact_paths["organ_metrics"],
    }
    for key, path in receipt_links.items():
        if receipt.get(key) != sha256_file(path):
            raise SchemaError(f"formal fold receipt cross-hash changed: {key}")
    fold_receipt_sha = sha256_file(artifact_paths["fold_receipt"])
    if payloads["attempt_receipt"].get("fold_receipt_sha256") != fold_receipt_sha:
        raise SchemaError("formal attempt receipt cross-hash changed")
    complete = payloads["complete"]
    expected_complete = {
        "schema_version": "revision0710.fig04_fig05.attempt_complete.v8",
        "status": "COMPLETE",
        "run_id": run_id,
        "seed": 7210,
        "attempt_ordinal": int(attempt.name[-4:]),
        "performance_metric_used_for_selection": False,
    }
    if complete != expected_complete:
        raise SchemaError("formal COMPLETE receipt changed")
    fold = frames["fold_metrics"].iloc[0]
    expected_fold_links = {
        "checkpoint_sha256": checkpoint_sha,
        "prediction_rows_sha256": prediction_sha,
        "age_correction_sha256": correction_sha,
        "cox_nuisance_sha256": nuisance_sha,
        "outer_read_ledger_sha256": ledger_sha,
    }
    for column, expected in expected_fold_links.items():
        if str(fold[column]) != expected:
            raise SchemaError(f"formal fold cross-binding changed: {column}")
    independent_correction, independent_nuisance = (
        validate_independently_refitted_nuisance(
            validation,
            payloads["age_correction"],
            payloads["cox_nuisance"],
        )
    )
    independently_corrected_outer = independently_apply_age_correction(
        prediction, independent_correction
    )
    if not np.allclose(
        prediction["corrected_age_gap"].to_numpy(np.float64),
        independently_corrected_outer["corrected_age_gap"].to_numpy(np.float64),
        atol=1e-8,
        rtol=0.0,
    ):
        raise SchemaError("outer corrected age gaps differ from validation-only refit")
    validate_independently_recomputed_metrics(
        independently_corrected_outer,
        independent_nuisance,
        frames["fold_metrics"],
        frames["organ_metrics"],
    )

    records = [
        {
            "name": name,
            "path": path.resolve().relative_to(attempt.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in artifact_paths.items()
    ]
    return {
        "fold_receipt_sha256": fold_receipt_sha,
        "artifact_count": len(records),
        "artifact_set_sha256": v8_contract.canonical_json_sha256(records),
        "artifacts": records,
        "production_replay": replay_receipt,
    }


def validate_scaled_attempt_bundle(
    attempt: Path,
    *,
    variant_id: str,
    root: Path,
) -> dict[str, Any]:
    """Deeply validate one complete synthetic 22-artifact attempt."""

    import torch

    import isolated_worker
    import v8_data

    run_id = f"{variant_id}__outer0"
    artifact_paths = {
        "claim": attempt / "CLAIM.json",
        "train_task": attempt / "TRAIN_TASK.json",
        "calibration_task": attempt / "CALIBRATION_TASK.json",
        "outer_task": attempt / "OUTER_TASK.json",
        "checkpoint": attempt / "01_train" / "fixed_epoch_0001.pt",
        "checkpoint_manifest": attempt / "01_train" / "CHECKPOINT_MANIFEST.json",
        "training_complete": attempt / "01_train" / "TRAINING_COMPLETE.json",
        "model_optimizer_manifest": attempt / "01_train" / "MODEL_OPTIMIZER_MANIFEST.json",
        "production_wiring": attempt / "01_train" / "PRODUCTION_WIRING.json",
        "training_history": attempt / "01_train" / "TRAINING_HISTORY.json",
        "training_trajectory_attestation": attempt / "01_train" / "TRAINING_TRAJECTORY_ATTESTATION.json",
        "age_correction": attempt / "02_calibration" / "age_correction.json",
        "cox_nuisance": attempt / "02_calibration" / "cox_nuisance.json",
        "validation_predictions": attempt / "02_calibration" / "validation_predictions.csv",
        "calibration_complete": attempt / "02_calibration" / "CALIBRATION_COMPLETE.json",
        "outer_read_ledger": attempt / "OUTER_READ_LEDGER.json",
        "prediction_rows": attempt / "03_outer_test" / "prediction_rows.csv",
        "fold_metrics": attempt / "03_outer_test" / "fold_metrics.csv",
        "organ_metrics": attempt / "03_outer_test" / "organ_metrics.csv",
        "fold_receipt": attempt / "03_outer_test" / "FOLD_RECEIPT.json",
        "attempt_receipt": attempt / "ATTEMPT_RECEIPT.json",
        "complete": attempt / "COMPLETE.json",
    }
    if len(artifact_paths) != 22:
        raise SchemaError("scaled deep artifact closure is not 22")
    missing = [name for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise SchemaError(f"scaled deep artifact closure is incomplete: {missing}")
    task_paths = {
        "train": attempt / "TRAIN_TASK.json",
        "calibration": attempt / "CALIBRATION_TASK.json",
        "outer_test": attempt / "OUTER_TASK.json",
    }
    tasks: dict[str, dict[str, Any]] = {}
    for phase, path in task_paths.items():
        task = v8_contract.read_json_exact(path)
        isolated_worker.validate_task_structure(task, expected_phase=phase)
        if task["run_id"] != run_id or task["variant_id"] != variant_id:
            raise SchemaError(f"scaled {phase} task identity changed")
        tasks[phase] = task
    claim_sha = sha256_file(artifact_paths["claim"])
    for phase, task in tasks.items():
        expected_task = artifact_paths[
            f"{phase if phase != 'outer_test' else 'outer'}_task"
        ]
        if (
            Path(task["task_path"]).resolve() != expected_task.resolve()
            or Path(task["claim_path"]).resolve() != artifact_paths["claim"].resolve()
            or task["claim_sha256"] != claim_sha
        ):
            raise SchemaError(f"scaled {phase} task/claim closure changed")

    schemas = load_schemas(root / "OUTPUT_SCHEMAS_V8.json")
    payloads = {
        name: v8_contract.read_json_exact(path)
        for name, path in artifact_paths.items()
        if path.suffix == ".json"
    }
    for name in (
        "checkpoint_manifest", "age_correction", "cox_nuisance",
        "outer_read_ledger", "fold_receipt",
    ):
        validate_json_payload(
            payloads[name],
            schema_for_execution(
                schemas, name, execution_mode="scaled_integration", evaluation_epoch=1
            ),
            table=name,
        )
    validate_json_payload(
        payloads["attempt_receipt"], schemas["tables"]["attempt_receipt"],
        table="attempt_receipt",
    )
    prediction_schema = schema_for_execution(
        schemas, "prediction_rows", execution_mode="scaled_integration", evaluation_epoch=1
    )
    frames = {
        "validation_predictions": read_csv_exact(
            artifact_paths["validation_predictions"], prediction_schema,
            table="validation_predictions",
        ),
        "prediction_rows": read_csv_exact(
            artifact_paths["prediction_rows"], prediction_schema,
            table="prediction_rows",
        ),
        "fold_metrics": read_csv_exact(
            artifact_paths["fold_metrics"],
            schema_for_execution(
                schemas, "fold_metrics", execution_mode="scaled_integration", evaluation_epoch=1
            ),
            table="fold_metrics",
        ),
        "organ_metrics": read_csv_exact(
            artifact_paths["organ_metrics"],
            schema_for_execution(
                schemas, "organ_metrics", execution_mode="scaled_integration", evaluation_epoch=1
            ),
            table="organ_metrics",
        ),
    }

    checkpoint_sha = sha256_file(artifact_paths["checkpoint"])
    checkpoint = torch.load(artifact_paths["checkpoint"], map_location="cpu", weights_only=False)
    expected_checkpoint_keys = {
        "schema_version", "status", "variant_id", "epoch", "seed",
        "model_config", "age_anchor", "model_state_dict",
        "optimizer_parameter_names", "initial_trainable_sha256",
        "initial_full_model_sha256", "initial_optimizer_sha256",
        "training_trajectory_sha256_at_capture",
    }
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != expected_checkpoint_keys
        or checkpoint["schema_version"] != "revision0710.fig04_fig05.ablation_checkpoint.v8"
        or checkpoint["status"] != "CAPTURED_FIXED_EPOCH"
        or checkpoint["variant_id"] != variant_id
        or int(checkpoint["epoch"]) != 1
        or int(checkpoint["seed"]) != 7210
    ):
        raise SchemaError("scaled checkpoint payload changed")
    checkpoint_manifest = payloads["checkpoint_manifest"]
    if (
        checkpoint_manifest["checkpoint_sha256"] != checkpoint_sha
        or int(checkpoint_manifest["checkpoint_bytes"]) != artifact_paths["checkpoint"].stat().st_size
        or checkpoint_manifest["run_id"] != run_id
        or checkpoint_manifest["variant_id"] != variant_id
        or int(checkpoint_manifest["outer_fold"]) != 0
        or checkpoint_manifest["initial_trainable_sha256"] != checkpoint["initial_trainable_sha256"]
    ):
        raise SchemaError("scaled checkpoint manifest/file binding changed")

    training = payloads["training_complete"]
    training_links = {
        "checkpoint_sha256": artifact_paths["checkpoint"],
        "checkpoint_manifest_sha256": artifact_paths["checkpoint_manifest"],
        "model_optimizer_manifest_sha256": artifact_paths["model_optimizer_manifest"],
        "production_wiring_sha256": artifact_paths["production_wiring"],
        "training_history_sha256": artifact_paths["training_history"],
        "training_trajectory_attestation_sha256": artifact_paths[
            "training_trajectory_attestation"
        ],
    }
    for key, path in training_links.items():
        if training.get(key) != sha256_file(path):
            raise SchemaError(f"scaled training cross-hash changed: {key}")
    if (
        Path(str(training.get("checkpoint_path"))).resolve() != artifact_paths["checkpoint"].resolve()
        or Path(str(training.get("checkpoint_manifest_path"))).resolve()
        != artifact_paths["checkpoint_manifest"].resolve()
    ):
        raise SchemaError("scaled training artifact paths changed")
    if (
        training.get("run_id") != run_id
        or training.get("variant_id") != variant_id
        or int(training.get("completed_epochs", -1)) != 2
        or int(training.get("capture_epoch", -1)) != 1
        or training.get("opened_roles") != ["train"]
        or int(training.get("calibration_read_count", -1)) != 0
        or int(training.get("outer_test_read_count", -1)) != 0
        or training.get("task_sha256") != sha256_file(artifact_paths["train_task"])
        or training.get("claim_sha256") != claim_sha
        or Path(str(training.get("training_trajectory_attestation_path"))).resolve()
        != artifact_paths["training_trajectory_attestation"].resolve()
    ):
        raise SchemaError("scaled training completion semantics changed")
    if training.get("opened_paths") != expected_task_opened_paths(
        tasks["train"], root=root, phase_inputs=[]
    ):
        raise SchemaError("scaled training exact opened-path closure changed")

    calibration = payloads["calibration_complete"]
    correction_sha = sha256_file(artifact_paths["age_correction"])
    nuisance_sha = sha256_file(artifact_paths["cox_nuisance"])
    if (
        calibration.get("run_id") != run_id
        or calibration.get("training_complete_sha256") != sha256_file(artifact_paths["training_complete"])
        or calibration.get("checkpoint_sha256") != checkpoint_sha
        or calibration.get("age_correction_sha256") != correction_sha
        or calibration.get("cox_nuisance_sha256") != nuisance_sha
        or calibration.get("validation_predictions_sha256") != sha256_file(artifact_paths["validation_predictions"])
        or calibration.get("opened_roles") != ["calibration"]
        or calibration.get("model_selection_used") is not False
        or int(calibration.get("outer_test_read_count", -1)) != 0
        or calibration.get("task_sha256")
        != sha256_file(artifact_paths["calibration_task"])
        or calibration.get("claim_sha256") != claim_sha
        or Path(str(calibration.get("age_correction_path"))).resolve()
        != artifact_paths["age_correction"].resolve()
        or Path(str(calibration.get("cox_nuisance_path"))).resolve()
        != artifact_paths["cox_nuisance"].resolve()
    ):
        raise SchemaError("scaled calibration completion semantics changed")
    if calibration.get("opened_paths") != expected_task_opened_paths(
        tasks["calibration"],
        root=root,
        phase_inputs=[artifact_paths["training_complete"], artifact_paths["checkpoint"]],
    ):
        raise SchemaError("scaled calibration exact opened-path closure changed")

    for frame_name, split_role in (
        ("validation_predictions", "validation"), ("prediction_rows", "test")
    ):
        frame = frames[frame_name]
        if (
            set(frame["run_id"].astype(str)) != {run_id}
            or set(frame["variant_id"].astype(str)) != {variant_id}
            or set(frame["outer_fold"].astype(int)) != {0}
            or set(frame["split_role"].astype(str)) != {split_role}
        ):
            raise SchemaError(f"scaled {frame_name} identity changed")
        for column, expected in (
            ("checkpoint_sha256", checkpoint_sha),
            ("age_correction_sha256", correction_sha),
            ("cox_nuisance_sha256", nuisance_sha),
        ):
            if set(frame[column].astype(str)) != {expected}:
                raise SchemaError(f"scaled {frame_name}.{column} binding changed")

    def load_synthetic_role(phase: str) -> tuple[Any, Any, Any]:
        source = tasks[phase]["data_source"]
        path = Path(source["role_bundle_path"])
        if sha256_file(path) != source["role_bundle_sha256"]:
            raise SchemaError(f"scaled {phase} role bundle hash changed")
        bundle = v8_data.load_role_bundle(path, expected_role=tasks[phase]["role"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71002)
            code = torch.randn(64, 8)
            text = torch.randn(64, 8)
            code[0].zero_()
            text[0].zero_()
        return bundle, code, text

    train_bundle, _, _ = load_synthetic_role("train")
    calibration_bundle, _, _ = load_synthetic_role("calibration")
    outer_bundle, outer_code, outer_text = load_synthetic_role("outer_test")
    validation = frames["validation_predictions"]
    prediction = frames["prediction_rows"]
    validate_prediction_role_alignment(
        validation, calibration_bundle, split_role="validation"
    )
    validate_prediction_role_alignment(prediction, outer_bundle, split_role="test")
    validate_training_trajectory_closure(
        payloads["training_trajectory_attestation"],
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        optimizer_manifest=payloads["model_optimizer_manifest"],
        history=payloads["training_history"],
        task=tasks["train"],
        claim_sha256=claim_sha,
        train_source_sha256=tasks["train"]["data_source"]["role_bundle_sha256"],
    )

    independent_correction, independent_nuisance = validate_independently_refitted_nuisance(
        validation, payloads["age_correction"], payloads["cox_nuisance"]
    )
    corrected_outer = independently_apply_age_correction(prediction, independent_correction)
    if not np.allclose(
        prediction["corrected_age_gap"].to_numpy(np.float64),
        corrected_outer["corrected_age_gap"].to_numpy(np.float64),
        atol=1e-8,
        rtol=0.0,
    ):
        raise SchemaError("scaled outer corrected gaps differ from independent validation refit")
    validate_independently_recomputed_metrics(
        corrected_outer,
        independent_nuisance,
        frames["fold_metrics"],
        frames["organ_metrics"],
    )

    protocol = tasks["train"]["protocol"]
    replay_receipt = validate_actual_model_wiring_optimizer_history_and_replay(
        checkpoint=checkpoint,
        checkpoint_path=artifact_paths["checkpoint"],
        variant_id=variant_id,
        outer_bundle=outer_bundle,
        validation_bundle=calibration_bundle,
        code_embedding=outer_code,
        text_embedding=outer_text,
        prediction=prediction,
        validation_prediction=validation,
        optimizer_manifest=payloads["model_optimizer_manifest"],
        wiring=payloads["production_wiring"],
        history=payloads["training_history"],
        protocol=protocol,
        train_participants=len(train_bundle.participant_ids),
        replay_device="cpu",
    )

    ledger = payloads["outer_read_ledger"]
    expected_ledger_links = {
        "checkpoint_sha256": checkpoint_sha,
        "training_complete_sha256": sha256_file(artifact_paths["training_complete"]),
        "age_correction_sha256": correction_sha,
        "cox_nuisance_sha256": nuisance_sha,
        "outer_role_manifest_sha256": tasks["outer_test"]["data_source"]["role_bundle_sha256"],
    }
    for key, expected in expected_ledger_links.items():
        if ledger.get(key) != expected:
            raise SchemaError(f"scaled outer ledger binding changed: {key}")
    receipt = payloads["fold_receipt"]
    if (
        receipt.get("task_sha256") != sha256_file(artifact_paths["outer_task"])
        or receipt.get("claim_sha256") != claim_sha
        or receipt.get("opened_paths")
        != expected_task_opened_paths(
            tasks["outer_test"],
            root=root,
            phase_inputs=[
                artifact_paths["outer_read_ledger"],
                artifact_paths["training_complete"],
                artifact_paths["calibration_complete"],
                artifact_paths["checkpoint"],
                artifact_paths["age_correction"],
                artifact_paths["cox_nuisance"],
            ],
        )
    ):
        raise SchemaError("scaled outer exact task/claim/opened-path closure changed")
    receipt_links = {
        "checkpoint_manifest_sha256": artifact_paths["checkpoint_manifest"],
        "prediction_rows_sha256": artifact_paths["prediction_rows"],
        "age_correction_sha256": artifact_paths["age_correction"],
        "cox_nuisance_sha256": artifact_paths["cox_nuisance"],
        "outer_read_ledger_sha256": artifact_paths["outer_read_ledger"],
        "fold_metrics_sha256": artifact_paths["fold_metrics"],
        "organ_metrics_sha256": artifact_paths["organ_metrics"],
    }
    for key, path in receipt_links.items():
        if receipt.get(key) != sha256_file(path):
            raise SchemaError(f"scaled fold receipt cross-hash changed: {key}")
    if payloads["attempt_receipt"].get("fold_receipt_sha256") != sha256_file(artifact_paths["fold_receipt"]):
        raise SchemaError("scaled attempt receipt cross-hash changed")
    expected_complete = {
        "schema_version": "revision0710.fig04_fig05.attempt_complete.v8",
        "status": "COMPLETE",
        "run_id": run_id,
        "seed": 7210,
        "attempt_ordinal": int(attempt.name[-4:]),
        "performance_metric_used_for_selection": False,
    }
    if payloads["complete"] != expected_complete:
        raise SchemaError("scaled COMPLETE receipt changed")

    records = [
        {
            "name": name,
            "path": path.resolve().relative_to(attempt.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in artifact_paths.items()
    ]
    return {
        "artifact_count": len(records),
        "artifact_set_sha256": v8_contract.canonical_json_sha256(records),
        "artifacts": records,
        "production_replay": replay_receipt,
    }
