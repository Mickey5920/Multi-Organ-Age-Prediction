from __future__ import annotations

import json
import importlib.util
import math
import re
import shlex
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ehr_age_data import (
    LABEL_COLUMNS,
    MODEL_DIR,
    REVISION_ROOT,
    Fold0DevelopmentData,
    JointLabels,
    JointVectorizedBatcher,
    canonical_hash,
    load_fold0_development_data,
    load_restricted_compact_cache,
    sha256_file,
    verify_pass_gate,
)
from ehr_age_training import (
    PredictionBundle,
    SelectionGates,
    _autocast,
    _evaluation_frame,
    _loss_config,
    _model_config,
    _predict,
    _survival_multiplier,
    _train_epoch,
    configure_reproducibility,
    pooled_age_metrics,
    trainable_parameter_hash,
)

from ehr_age_model import EHRAGEModel, ORGAN_SYSTEMS


RUNNER_DIR = Path(__file__).resolve().parent
EHR_AGE_ROOT = RUNNER_DIR.parent
D07_PROTOCOL_PATH = EHR_AGE_ROOT / "00_contract" / "D07_OPEN_FOLD0_DEVELOPMENT_PROTOCOL.md"
D04_CONFIG_PATH = EHR_AGE_ROOT / "00_contract" / "ehr_age_d04_base_head_mlp_v1.json"
D04_SOURCE_RUN_DIR = EHR_AGE_ROOT / "03_runs" / "fold0" / "development" / "2652eb9cc461"
DEFAULT_DEVELOPMENT_LABELS = RUNNER_DIR / "label_cache" / "fold0_development_joint_labels.parquet"
DEFAULT_DEVELOPMENT_AGES = RUNNER_DIR / "label_cache" / "fold0_development_age_targets.parquet"
DEFAULT_DEVELOPMENT_MANIFEST = (
    RUNNER_DIR / "label_cache" / "fold0_development_joint_labels.manifest.json"
)
DEFAULT_FOLD0_AGES = RUNNER_DIR / "sealed_outer_test" / "fold0_outer_test_age_targets.parquet"
DEFAULT_FOLD0_LABELS = RUNNER_DIR / "sealed_outer_test" / "fold0_outer_test_joint_labels.parquet"
DEFAULT_FOLD0_MANIFEST = (
    RUNNER_DIR / "sealed_outer_test" / "fold0_outer_test_materialization.manifest.json"
)
DEFAULT_CACHE_DIR = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
DEFAULT_D07_BASELINE_ROOT = (
    REVISION_ROOT
    / "05_downstream"
    / "08_d07_fold0_baseline"
)
DEFAULT_BASELINE_ENVELOPE = DEFAULT_D07_BASELINE_ROOT / "fold0_baseline_envelope.json"
DEFAULT_BASELINE_ARTIFACT_MANIFEST = DEFAULT_D07_BASELINE_ROOT / "artifact_manifest.json"
DEFAULT_BASELINE_VALIDATOR = DEFAULT_D07_BASELINE_ROOT / "build_d07_fold0_baseline.py"
DEFAULT_OUTPUT_ROOT = EHR_AGE_ROOT / "03_runs" / "fold0" / "open_development"

D07_PROTOCOL_SHA256 = "fef15b95f8987012c9f37ffffb0650efe9b44cd25846245a58b2860ac3abfcbe"
D04_CONFIG_SHA256 = "3b01b5858503299b778c26761c7ff462196549d8758c6066acc838cab8c8c88d"
D04_RUN_CONFIG_SHA256 = "9ab5cba3f140d568ef9d2bca8fcaef4b32202d9ab6f40da47f31d1d550d8bac6"
D04_HISTORY_SHA256 = "c71b31f4a38fcaee452fbbbdc4b5e9ddfa5455bc3a249aa5c6ad82cc0ec5e70c"
D04_CANONICAL_CONFIG_HASH = "2652eb9cc4612a9bc8e27ee7a35812e4de0dccabfed56e7247c3d836ba5c9f29"
D04_INITIAL_TRAINABLE_SHA256 = "8a44fc7c9662041558878e74edcb5215d48708c63602fdc45fae66166da00f30"
D04_TARGET_EPOCH = 12
D04_SCHEDULER_HORIZON_EPOCHS = 60
D04_SEED = 7210
TRAJECTORY_ATOL = 1e-8
TRAJECTORY_RTOL = 1e-7
ROLE_SCHEMA = "revision0710.d07.fold0_role.v1"
READ_SCHEMA = "revision0710.d07.fold0_read_ledger_entry.v1"
RECEIPT_SCHEMA = "revision0710.d07.fold0_read_receipt.v1"
CANDIDATE_SCHEMA = "revision0710.d07.open_fold0_candidate.v1"
D07_BASELINE_SCHEMA = "revision0710_d07_open_fold0_baseline_envelope_v1"
D07_BASELINE_MANIFEST_SCHEMA = "revision0710_d07_open_fold0_baseline_artifact_manifest_v1"
D07_BASELINE_METHODS = frozenset(
    {
        "retain",
        "behrt_style",
        "medbert_style",
        "retain_organquery",
        "behrt_style_organquery",
        "medbert_style_organquery",
    }
)
D07_BASELINE_SCOPE = {
    "outer_fold": 0,
    "fold0_role": "open_development",
    "prediction_split_role": "test",
    "nuisance_fit": {
        "source": "outer_fold_0_validation_predictions",
        "participant_fold_id": 1,
        "inner_role": "calibration",
        "age_correction_population": "strict_healthy_reference_60m",
        "cox_population": "all_at_risk_1",
    },
    "score": {
        "source": "outer_fold_0_test_predictions",
        "participant_fold_id": 0,
        "strict_age_population": "strict_healthy_reference_60m_pooled_participant_organ",
        "harrell_population": "all_at_risk_1_by_organ_then_seven_organ_macro",
    },
    "fold0_fitted_parameters": False,
    "ehr_age_predictions_read": False,
}
D07_GATE_SPECS = {
    "strict_mae": {"direction": "lower", "operator": "<", "margin": 0.01},
    "strict_pearson": {"direction": "higher", "operator": ">", "margin": 0.001},
    "strict_r2": {"direction": "higher", "operator": ">", "margin": 0.002},
    "age_sex_harrell_macro_delta_c": {
        "direction": "higher",
        "operator": ">",
        "margin": 0.01,
    },
}


class D07ContractError(RuntimeError):
    pass


class D04ReproductionError(D07ContractError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def write_artifact_hash_registry(
    path: Path, artifacts: Mapping[str, Path]
) -> dict[str, dict[str, str]]:
    """Write the final registry once, after every registered artifact is immutable."""

    resolved_registry = path.resolve()
    if any(artifact.resolve() == resolved_registry for artifact in artifacts.values()):
        raise D07ContractError("artifact hash registry cannot register itself")
    registry = {
        name: {"path": str(artifact.resolve()), "sha256": sha256_file(artifact)}
        for name, artifact in artifacts.items()
    }
    _write_json_exclusive(path, registry)
    return registry


def _read_receipt_identity(read_ledger_entry_path: Path) -> tuple[dict[str, Any], int, str]:
    ledger = json.loads(read_ledger_entry_path.read_text(encoding="utf-8"))
    if (
        ledger.get("schema_version") != READ_SCHEMA
        or ledger.get("status") != "READ_COMMITTED_BEFORE_FOLD0_ACCESS"
        or ledger.get("d07_protocol_sha256") != D07_PROTOCOL_SHA256
        or ledger.get("fold") != 0
        or ledger.get("fold_role") != "OPEN_DEVELOPMENT"
    ):
        raise D07ContractError("cannot issue a receipt for an invalid Fold0 read ledger")
    sequence = int(ledger.get("sequence", 0))
    candidate_id = str(ledger.get("candidate_id", ""))
    if sequence < 1 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", candidate_id) is None:
        raise D07ContractError("Fold0 read ledger has an invalid receipt identity")
    if read_ledger_entry_path.name != f"{sequence:06d}__{candidate_id}.json":
        raise D07ContractError("Fold0 read ledger filename and payload disagree")
    return ledger, sequence, candidate_id


def _write_fold0_read_receipt(
    *,
    output_root: Path,
    read_ledger_entry_path: Path,
    status: str,
    stage: str,
    artifacts: Mapping[str, Path],
    exception: BaseException | None = None,
) -> Path:
    ledger, sequence, candidate_id = _read_receipt_identity(read_ledger_entry_path)
    if status not in {"COMPLETE", "FAILED_AFTER_COMMITTED_READ"}:
        raise D07ContractError(f"unsupported Fold0 read receipt status: {status}")
    if status == "COMPLETE" and exception is not None:
        raise D07ContractError("a completion receipt cannot contain an exception")
    if status == "FAILED_AFTER_COMMITTED_READ" and exception is None:
        raise D07ContractError("a failure receipt must contain an exception")

    receipt_dir = output_root.resolve() / "FOLD0_READ_RECEIPTS"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sequence:06d}__{candidate_id}"
    existing = list(receipt_dir.glob(f"{stem}.*.json"))
    if existing:
        raise FileExistsError(
            f"Fold0 read receipt already exists for sequence {sequence} candidate {candidate_id}"
        )

    existing_artifacts = {
        name: path
        for name, path in artifacts.items()
        if path.is_file()
    }
    required_complete = {
        "fold0_open_development_predictions_long",
        "fold0_open_development_metrics",
        "candidate_summary",
        "artifact_hashes",
    }
    if status == "COMPLETE" and not required_complete.issubset(existing_artifacts):
        missing = sorted(required_complete.difference(existing_artifacts))
        raise D07ContractError(
            f"completion receipt lacks required immutable artifacts: {missing}"
        )
    bindings = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(existing_artifacts.items())
    }
    suffix = "complete" if status == "COMPLETE" else "failure"
    receipt_path = receipt_dir / f"{stem}.{suffix}.json"
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "created_utc": _utc_now(),
        "sequence": sequence,
        "candidate_id": candidate_id,
        "stage": stage,
        "d07_protocol_sha256": D07_PROTOCOL_SHA256,
        "precommit_read_ledger": {
            "path": str(read_ledger_entry_path.resolve()),
            "sha256": sha256_file(read_ledger_entry_path),
            "checkpoint_sha256": ledger["checkpoint_sha256"],
            "evaluation_manifest_sha256": ledger["evaluation_manifest_sha256"],
        },
        "artifact_bindings": bindings,
        "receipt_excluded_from_artifact_hash_registry_to_avoid_self_reference": True,
    }
    if exception is not None:
        payload["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
    _write_json_exclusive(receipt_path, payload)
    return receipt_path


def write_fold0_read_completion_receipt(
    *,
    output_root: Path,
    read_ledger_entry_path: Path,
    artifacts: Mapping[str, Path],
) -> Path:
    return _write_fold0_read_receipt(
        output_root=output_root,
        read_ledger_entry_path=read_ledger_entry_path,
        status="COMPLETE",
        stage="COMPLETE",
        artifacts=artifacts,
    )


def write_fold0_read_failure_receipt(
    *,
    output_root: Path,
    read_ledger_entry_path: Path,
    stage: str,
    exception: BaseException,
    artifacts: Mapping[str, Path],
) -> Path:
    return _write_fold0_read_receipt(
        output_root=output_root,
        read_ledger_entry_path=read_ledger_entry_path,
        status="FAILED_AFTER_COMMITTED_READ",
        stage=stage,
        artifacts=artifacts,
        exception=exception,
    )


def validate_d07_protocol(protocol_path: Path = D07_PROTOCOL_PATH) -> str:
    observed = sha256_file(protocol_path)
    if observed != D07_PROTOCOL_SHA256:
        raise D07ContractError(
            f"D07 protocol hash mismatch: expected {D07_PROTOCOL_SHA256}, observed {observed}"
        )
    return observed


def _load_d07_baseline_validator() -> Callable[[Path], Mapping[str, Any]]:
    if not DEFAULT_BASELINE_VALIDATOR.is_file():
        raise D07ContractError(
            f"D07 Fold0 baseline validator is missing: {DEFAULT_BASELINE_VALIDATOR}"
        )
    spec = importlib.util.spec_from_file_location(
        "revision0710_d07_fold0_baseline_validator", DEFAULT_BASELINE_VALIDATOR
    )
    if spec is None or spec.loader is None:
        raise D07ContractError("could not load the D07 Fold0 baseline validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validator = getattr(module, "validate_fold0_baseline_artifacts", None)
    if not callable(validator):
        raise D07ContractError(
            "D07 Fold0 baseline builder lacks validate_fold0_baseline_artifacts"
        )
    return validator


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise D07ContractError(f"D07 Fold0 baseline {field} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise D07ContractError(
            f"D07 Fold0 baseline {field} is not numeric"
        ) from exc
    if not math.isfinite(number):
        raise D07ContractError(f"D07 Fold0 baseline {field} is not finite")
    return number


def _require_exact_number(value: Any, expected: float, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-15):
        raise D07ContractError(
            f"D07 Fold0 baseline {field} changed: expected {expected}, observed {number}"
        )
    return number


def validate_d07_fold0_baseline_envelope(
    path: Path = DEFAULT_BASELINE_ENVELOPE,
) -> tuple[SelectionGates, dict[str, Any], dict[str, Any]]:
    """Load only the canonical true-Fold-0 comparator and verify its full lineage."""

    path = path.resolve()
    if path != DEFAULT_BASELINE_ENVELOPE.resolve():
        raise D07ContractError(
            "D07 requires the canonical 05_downstream/08_d07_fold0_baseline "
            "true-Fold-0 envelope; the historical Fold1 envelope is forbidden"
        )
    try:
        validation_raw = dict(
            _load_d07_baseline_validator()(DEFAULT_D07_BASELINE_ROOT.resolve())
        )
    except D07ContractError:
        raise
    except Exception as exc:
        raise D07ContractError(
            f"D07 Fold0 baseline artifact validation failed: {exc}"
        ) from exc

    payload = json.loads(path.read_text(encoding="utf-8"))
    if validation_raw != payload:
        raise D07ContractError(
            "D07 Fold0 baseline validator did not return the registered envelope"
        )
    if payload.get("schema_version") != D07_BASELINE_SCHEMA:
        raise D07ContractError("D07 Fold0 baseline envelope schema mismatch")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise D07ContractError("D07 Fold0 baseline envelope lacks scope")
    scope_mismatches = {
        key: {"expected": expected, "observed": scope.get(key)}
        for key, expected in D07_BASELINE_SCOPE.items()
        if scope.get(key) != expected
    }
    if scope_mismatches:
        raise D07ContractError(
            f"D07 Fold0 baseline envelope has invalid true-Fold-0 scope: {scope_mismatches}"
        )

    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise D07ContractError("D07 Fold0 baseline methods must be a row list")
    method_names = [str(row.get("method")) for row in methods if isinstance(row, Mapping)]
    if len(methods) != 6 or len(method_names) != 6 or set(method_names) != D07_BASELINE_METHODS:
        raise D07ContractError("D07 Fold0 envelope must contain exactly the six frozen baselines")
    if len(method_names) != len(set(method_names)):
        raise D07ContractError("D07 Fold0 baseline methods are duplicated")
    for row in methods:
        source_hashes = row.get("source_sha256", {})
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise D07ContractError(
                f"D07 Fold0 baseline {row.get('method')} lacks source SHA256 lineage"
            )

    envelope = payload.get("envelope")
    gates_payload = payload.get("gates")
    if not isinstance(envelope, Mapping) or not isinstance(gates_payload, Mapping):
        raise D07ContractError("D07 Fold0 baseline envelope or gates are missing")
    threshold_decimals: dict[str, str] = {}
    threshold_names = {
        "strict_mae": "strict_mae",
        "strict_pearson": "strict_pearson",
        "strict_r2": "strict_r2",
        "age_sex_harrell_macro_delta_c": "risk_delta_c",
    }
    for metric, spec_row in D07_GATE_SPECS.items():
        envelope_row = envelope.get(metric)
        gate_row = gates_payload.get(metric)
        if not isinstance(envelope_row, Mapping) or not isinstance(gate_row, Mapping):
            raise D07ContractError(f"D07 Fold0 baseline lacks {metric}")
        if envelope_row.get("direction") != spec_row["direction"]:
            raise D07ContractError(f"D07 Fold0 baseline {metric} direction mismatch")
        if gate_row.get("operator") != spec_row["operator"]:
            raise D07ContractError(f"D07 Fold0 baseline {metric} operator mismatch")
        baseline = _finite_number(
            envelope_row.get("baseline_value"), field=f"envelope.{metric}.baseline_value"
        )
        _require_exact_number(
            gate_row.get("baseline_value"),
            baseline,
            field=f"gates.{metric}.baseline_value",
        )
        margin = _require_exact_number(
            gate_row.get("margin"),
            float(spec_row["margin"]),
            field=f"gates.{metric}.margin",
        )
        expected_threshold = baseline - margin if spec_row["direction"] == "lower" else baseline + margin
        threshold = _require_exact_number(
            gate_row.get("threshold"),
            expected_threshold,
            field=f"gates.{metric}.threshold",
        )
        threshold_decimals[threshold_names[metric]] = str(gate_row.get("threshold"))

    organ_rows = envelope.get("organ_delta_c_envelope")
    if not isinstance(organ_rows, list):
        raise D07ContractError("D07 Fold0 baseline lacks the organ delta-C envelope")
    organ_values: dict[str, float] = {}
    for row in organ_rows:
        if not isinstance(row, Mapping):
            raise D07ContractError("D07 Fold0 organ delta-C envelope has a malformed row")
        organ = str(row.get("organ_system"))
        if organ in organ_values:
            raise D07ContractError(f"D07 Fold0 organ delta-C is duplicated for {organ}")
        if row.get("direction") != "higher":
            raise D07ContractError(f"D07 Fold0 organ delta-C direction mismatch for {organ}")
        organ_values[organ] = _finite_number(
            row.get("baseline_value"), field=f"organ_delta_c_envelope.{organ}"
        )
    if set(organ_values) != set(ORGAN_SYSTEMS):
        raise D07ContractError("D07 Fold0 envelope must contain all seven organ delta-C values")

    requirements = gates_payload.get(
        "risk_organ_rules", gates_payload.get("requirements", gates_payload)
    )
    positive_required = int(requirements.get("positive_organs_required", -1))
    beating_required = int(requirements.get("organs_beating_baseline_required", -1))
    if positive_required != 7 or beating_required != 5:
        raise D07ContractError(
            "D07 Fold0 risk requirements must be 7 positive organs and at least 5 beating baseline"
        )

    manifest = json.loads(DEFAULT_BASELINE_ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != D07_BASELINE_MANIFEST_SCHEMA:
        raise D07ContractError("D07 Fold0 baseline artifact-manifest schema mismatch")
    declared = manifest.get("artifacts", {}).get(path.name)
    declared_sha = declared.get("sha256") if isinstance(declared, Mapping) else declared
    if declared_sha != sha256_file(path):
        raise D07ContractError("D07 Fold0 envelope hash disagrees with artifact manifest")

    gates = SelectionGates(
        strict_mae_max=float(threshold_decimals["strict_mae"]),
        strict_pearson_min=float(threshold_decimals["strict_pearson"]),
        strict_r2_min=float(threshold_decimals["strict_r2"]),
        delta_c_min=float(threshold_decimals["risk_delta_c"]),
        positive_organs_min=positive_required,
        organs_beating_baseline_min=beating_required,
        baseline_organ_delta_c=organ_values,
        threshold_decimals=threshold_decimals,
        source=str(path),
        baseline_artifact_hashes=payload.get("source_sha256", {}),
        metric_contract_version="d07_open_fold0",
        envelope_schema_version=D07_BASELINE_SCHEMA,
        metric_contract_hashes={
            "baseline_envelope": sha256_file(path),
            "artifact_manifest": sha256_file(DEFAULT_BASELINE_ARTIFACT_MANIFEST),
            "baseline_validator": sha256_file(DEFAULT_BASELINE_VALIDATOR),
        },
    )
    validation = {
        "status": "VALID",
        "schema_version": payload["schema_version"],
        "outer_fold": payload["scope"]["outer_fold"],
        "method_count": len(methods),
        "organ_count": len(organ_rows),
        "ehr_age_predictions_read": payload["scope"]["ehr_age_predictions_read"],
    }
    return gates, payload, validation


def _validate_d04_lineage(
    *,
    config_path: Path,
    source_run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    expected_files = {
        config_path: D04_CONFIG_SHA256,
        source_run_dir / "run_config.json": D04_RUN_CONFIG_SHA256,
        source_run_dir / "training_history.json": D04_HISTORY_SHA256,
    }
    for path, expected in expected_files.items():
        observed = sha256_file(path)
        if observed != expected:
            raise D07ContractError(
                f"D04 lineage hash mismatch for {path}: expected {expected}, observed {observed}"
            )
    source_hashes = json.loads(
        (source_run_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    if source_hashes.get("canonical_config_hash") != D04_CANONICAL_CONFIG_HASH:
        raise D07ContractError("D04 source run canonical config hash mismatch")
    if source_hashes.get("training_history") != D04_HISTORY_SHA256:
        raise D07ContractError("D04 source artifact registry history hash mismatch")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    source_config = json.loads((source_run_dir / "run_config.json").read_text(encoding="utf-8"))
    source_history = json.loads(
        (source_run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    for key in ("model", "loss", "training", "protocol"):
        if raw.get(key) != source_config.get(key):
            raise D07ContractError(f"D04 frozen config disagrees with source run for {key}")
    training = raw["training"]
    if (
        int(training["max_epochs"]) != D04_SCHEDULER_HORIZON_EPOCHS
        or int(training["seed"]) != D04_SEED
        or training.get("deterministic") is not True
        or int(training["outer_fold"]) != 0
    ):
        raise D07ContractError("D04 seed, scheduler horizon, determinism, or fold changed")
    if len(source_history) < D04_TARGET_EPOCH:
        raise D07ContractError("D04 source history does not contain epoch 12")
    return raw, source_config, source_history[:D04_TARGET_EPOCH]


def _compare_trajectory_value(
    expected: Any,
    observed: Any,
    *,
    path: str,
    failures: list[dict[str, Any]],
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(expected) != set(observed):
            failures.append({"path": path, "expected_keys": sorted(expected), "observed": observed})
            return
        for key in expected:
            _compare_trajectory_value(
                expected[key], observed[key], path=f"{path}.{key}", failures=failures
            )
        return
    if isinstance(expected, bool) or isinstance(expected, str) or expected is None:
        if observed != expected:
            failures.append({"path": path, "expected": expected, "observed": observed})
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(
            float(expected), float(observed), rel_tol=TRAJECTORY_RTOL, abs_tol=TRAJECTORY_ATOL
        ):
            failures.append(
                {
                    "path": path,
                    "expected": expected,
                    "observed": observed,
                    "absolute_difference": abs(float(expected) - float(observed)),
                }
            )
        return
    if observed != expected:
        failures.append({"path": path, "expected": expected, "observed": observed})


def verify_training_trajectory_epoch(
    expected_history_row: Mapping[str, Any],
    observed_training_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare training-only fields without regenerating Fold 1 screen metrics."""

    expected = {
        key: expected_history_row[key]
        for key in ("epoch", "survival_multiplier", "learning_rate", "losses")
    }
    failures: list[dict[str, Any]] = []
    _compare_trajectory_value(expected, observed_training_row, path="epoch", failures=failures)
    return failures


def _current_code_hashes() -> dict[str, str]:
    paths = {
        "ehr_age_model": MODEL_DIR / "ehr_age_model.py",
        "ehr_age_losses": MODEL_DIR / "ehr_age_losses.py",
        "ehr_age_data": RUNNER_DIR / "ehr_age_data.py",
        "ehr_age_training": RUNNER_DIR / "ehr_age_training.py",
        "d07_open_fold0_development": Path(__file__).resolve(),
        "crossfit_metrics": REVISION_ROOT / "05_downstream" / "common" / "crossfit_metrics.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _source_hashes(
    *,
    cache_dir: Path,
    development_labels_path: Path,
    development_age_path: Path,
    development_manifest_path: Path,
) -> dict[str, str]:
    return {
        "cache_hard_gate": verify_pass_gate(cache_dir / "cache_hard_gate.json"),
        "leakage_gate": verify_pass_gate(
            cache_dir.parent / "05_leakage_audit" / "leakage_gate.json"
        ),
        "batch_manifest": sha256_file(cache_dir / "batch_manifest.json"),
        "embedding_tables": sha256_file(cache_dir / "embedding_tables.pt"),
        "development_labels": sha256_file(development_labels_path),
        "development_age_targets": sha256_file(development_age_path),
        "development_label_manifest": sha256_file(development_manifest_path),
    }


def _train_d04_to_epoch12(
    *,
    raw: Mapping[str, Any],
    source_history: Sequence[Mapping[str, Any]],
    data: Fold0DevelopmentData,
    run_dir: Path,
    emit: Callable[[str], None],
) -> tuple[EHRAGEModel, PredictionBundle, Path, Path]:
    training = raw["training"]
    seed = int(training["seed"])
    configure_reproducibility(
        seed,
        deterministic=bool(training["deterministic"]),
        tf32=bool(training["tf32"]),
    )
    age_anchor = float(
        data.cache.dataset.chronological_age.index_select(0, data.train_indices).mean()
    )
    model = EHRAGEModel(
        data.cache.code_embedding,
        data.cache.text_embedding,
        age_anchor=age_anchor,
        config=_model_config(raw["model"]),
    )
    initial_hash = trainable_parameter_hash(model)
    if initial_hash != D04_INITIAL_TRAINABLE_SHA256:
        raise D04ReproductionError(
            f"D04 initial trainable hash mismatch: {initial_hash}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_epochs = int(training["warmup_epochs"])

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / max(warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs) / max(
            D04_SCHEDULER_HORIZON_EPOCHS - warmup_epochs, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(training["amp"]) and device.type == "cuda"
    )
    interval_edges = torch.tensor(
        raw["model"]["survival_interval_edges_months"], dtype=torch.float32, device=device
    )
    train_batcher = JointVectorizedBatcher(
        data,
        data.train_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    observed_history: list[dict[str, Any]] = []
    parity_epochs: list[dict[str, Any]] = []
    emit(
        f"START D07-001 D04 reproduction device={device} scheduler_horizon=60 "
        f"capture_epoch={D04_TARGET_EPOCH} seed={seed}"
    )
    for epoch in range(1, D04_TARGET_EPOCH + 1):
        started = time.perf_counter()
        survival_multiplier = _survival_multiplier(epoch, raw["loss"])
        losses = _train_epoch(
            model=model,
            batcher=train_batcher,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            interval_edges=interval_edges,
            loss_config=_loss_config(raw["loss"]),
            survival_multiplier=survival_multiplier,
            event_dropout_probability=float(training["event_dropout_probability"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            amp=bool(training["amp"]),
            max_batches=None,
        )
        if losses["optimizer_steps"] < 1:
            raise D04ReproductionError("D04 reproduction epoch had no optimizer update")
        scheduler.step()
        row = {
            "epoch": epoch,
            "survival_multiplier": survival_multiplier,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "losses": losses,
        }
        failures = verify_training_trajectory_epoch(source_history[epoch - 1], row)
        parity_epochs.append(
            {
                "epoch": epoch,
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
            }
        )
        observed_history.append(row)
        emit(
            f"EPOCH {epoch} loss={losses['total']:.9f} lr={row['learning_rate']:.12g} "
            f"optimizer_steps={int(losses['optimizer_steps'])} "
            f"amp_skipped={int(losses['amp_skipped_steps'])} parity={'PASS' if not failures else 'FAIL'} "
            f"seconds={time.perf_counter() - started:.1f}"
        )
        if failures:
            failure_path = run_dir / "reproduction_failure.json"
            _write_json(
                failure_path,
                {
                    "schema_version": CANDIDATE_SCHEMA,
                    "status": "D04_REPRODUCTION_FAILED_NO_FOLD0_READ",
                    "created_utc": _utc_now(),
                    "failed_epoch": epoch,
                    "fold0_read_count": 0,
                    "failures": failures,
                },
            )
            raise D04ReproductionError(
                f"D04 training trajectory diverged at epoch {epoch}; Fold 0 remains unread"
            )

    history_path = run_dir / "training_history_e01_e12.json"
    _write_json(
        history_path,
        {
            "schema_version": "revision0710.d07.d04_training_trajectory.v1",
            "reference_fields": [
                "epoch",
                "survival_multiplier",
                "learning_rate",
                "losses",
            ],
            "excluded_reference_fields": [
                "strict_screen_metrics",
                "calibration_to_screen_cox",
                "eligible",
                "gate_failures",
                "selected",
                "seconds",
            ],
            "reason_excluded": "D07 does not regenerate Fold1 screen performance during reproduction",
            "epochs": observed_history,
        },
    )
    parity_path = run_dir / "trajectory_parity_e01_e12.json"
    _write_json(
        parity_path,
        {
            "schema_version": "revision0710.d07.d04_trajectory_parity.v1",
            "status": "PASS",
            "absolute_tolerance": TRAJECTORY_ATOL,
            "relative_tolerance": TRAJECTORY_RTOL,
            "reference_history_sha256": D04_HISTORY_SHA256,
            "checked_epochs": D04_TARGET_EPOCH,
            "fold1_screen_metrics_recomputed": False,
            "fold0_read_count_at_completion": 0,
            "epochs": parity_epochs,
        },
    )
    checkpoint_path = run_dir / "candidate_checkpoint_epoch12.pt"
    torch.save(
        {
            "schema_version": "revision0710.d07.d04_epoch12_checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": D04_TARGET_EPOCH,
            "scheduler_horizon_epochs": D04_SCHEDULER_HORIZON_EPOCHS,
            "age_anchor_train_only": age_anchor,
            "seed": seed,
            "from_random_initialization": True,
            "initial_trainable_parameter_sha256": initial_hash,
            "source_d04_config_hash": D04_CANONICAL_CONFIG_HASH,
            "source_d04_history_sha256": D04_HISTORY_SHA256,
            "trajectory_parity_sha256": sha256_file(parity_path),
            "warm_start": False,
            "fine_tune": False,
            "teacher": False,
            "distillation": False,
        },
        checkpoint_path,
    )

    calibration_batcher = JointVectorizedBatcher(
        data,
        data.calibration_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=seed,
    )
    calibration = _predict(
        model=model,
        batcher=calibration_batcher,
        device=device,
        amp=bool(training["amp"]),
        max_batches=None,
    )
    return model, calibration, checkpoint_path, parity_path


def _verify_sealed_manifest_metadata(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "revision0710_ehr_age_fold0_outer_test_materialization_v1"
        or payload.get("status") != "SEALED"
        or payload.get("outer_fold") != 0
    ):
        raise D07ContractError("Fold0 source materialization manifest is not the historical seal")
    artifacts = payload.get("artifacts", {})
    for name in ("age_targets", "joint_labels"):
        entry = artifacts.get(name, {})
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise D07ContractError(f"sealed Fold0 manifest lacks {name} hash")
    return payload


def commit_open_development_read(
    *,
    output_root: Path,
    candidate_id: str,
    sealed_manifest_path: Path,
    checkpoint_path: Path,
    evaluation_manifest_path: Path,
) -> tuple[Path, Path]:
    """Commit the irreversible role transition and one append-only read entry."""

    validate_d07_protocol()
    output_root.mkdir(parents=True, exist_ok=True)
    sealed = _verify_sealed_manifest_metadata(sealed_manifest_path)
    sealed_manifest_hash = sha256_file(sealed_manifest_path)
    evaluation = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    expected_fold0_hashes = {
        name: sealed["artifacts"][name]["sha256"]
        for name in ("age_targets", "joint_labels")
    }
    if (
        evaluation.get("schema_version")
        != "revision0710.d07.fold0_evaluation_manifest.v1"
        or evaluation.get("status") != "SEALED_BEFORE_FOLD0_READ"
        or evaluation.get("candidate_id") != candidate_id
        or evaluation.get("d07_protocol_sha256") != D07_PROTOCOL_SHA256
        or evaluation.get("checkpoint_path") != str(checkpoint_path.resolve())
        or evaluation.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        or evaluation.get("baseline_envelope_path")
        != str(DEFAULT_BASELINE_ENVELOPE.resolve())
        or evaluation.get("baseline_envelope_sha256")
        != sha256_file(DEFAULT_BASELINE_ENVELOPE)
        or evaluation.get("baseline_artifact_manifest_sha256")
        != sha256_file(DEFAULT_BASELINE_ARTIFACT_MANIFEST)
        or evaluation.get("declared_fold0_artifact_hashes") != expected_fold0_hashes
    ):
        raise D07ContractError(
            "evaluation manifest is not a valid pre-read D07 Fold0 authorization"
        )
    role_path = output_root / "FOLD0_ROLE_MANIFEST.json"
    if not role_path.exists():
        _write_json_exclusive(
            role_path,
            {
                "schema_version": ROLE_SCHEMA,
                "created_utc": _utc_now(),
                "previous_role": "SEALED_OUTER_TEST",
                "role": "OPEN_DEVELOPMENT",
                "role_change_irreversible": True,
                "confirmatory_fold0_claims_allowed": False,
                "repeated_versioned_candidate_reads_allowed": True,
                "fold1_role": "CALIBRATION_NUISANCE_ONLY_NO_NEW_SCREEN_METRICS",
                "fold2_to_4_role": "TRAINING_ONLY_NO_EHR_HELDOUT_METRICS_BEFORE_FREEZE",
                "d07_protocol_path": str(D07_PROTOCOL_PATH.resolve()),
                "d07_protocol_sha256": D07_PROTOCOL_SHA256,
                "historical_sealed_manifest_path": str(sealed_manifest_path.resolve()),
                "historical_sealed_manifest_sha256": sealed_manifest_hash,
                "declared_fold0_artifact_hashes": {
                    name: sealed["artifacts"][name]["sha256"]
                    for name in ("age_targets", "joint_labels")
                },
                "supersedes_historical_allowed_reader_for_d07": True,
            },
        )
    role = json.loads(role_path.read_text(encoding="utf-8"))
    if (
        role.get("schema_version") != ROLE_SCHEMA
        or role.get("role") != "OPEN_DEVELOPMENT"
        or role.get("role_change_irreversible") is not True
        or role.get("d07_protocol_sha256") != D07_PROTOCOL_SHA256
        or role.get("historical_sealed_manifest_sha256") != sealed_manifest_hash
    ):
        raise D07ContractError("existing Fold0 role manifest is incompatible or was modified")

    ledger_dir = output_root / "FOLD0_READ_LEDGER"
    ledger_dir.mkdir(exist_ok=True)
    if list(ledger_dir.glob(f"*__{candidate_id}.json")):
        raise FileExistsError(f"Fold0 read already committed for candidate {candidate_id}")
    existing = sorted(ledger_dir.glob("*.json"))
    sequence = len(existing) + 1
    ledger_path = ledger_dir / f"{sequence:06d}__{candidate_id}.json"
    _write_json_exclusive(
        ledger_path,
        {
            "schema_version": READ_SCHEMA,
            "status": "READ_COMMITTED_BEFORE_FOLD0_ACCESS",
            "created_utc": _utc_now(),
            "sequence": sequence,
            "candidate_id": candidate_id,
            "fold": 0,
            "fold_role": "OPEN_DEVELOPMENT",
            "role_manifest_sha256": sha256_file(role_path),
            "d07_protocol_sha256": D07_PROTOCOL_SHA256,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "evaluation_manifest_path": str(evaluation_manifest_path.resolve()),
            "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
            "declared_fold0_artifact_hashes": {
                name: sealed["artifacts"][name]["sha256"]
                for name in ("age_targets", "joint_labels")
            },
        },
    )
    return role_path, ledger_path


def _aligned_open_fold0_labels(
    frame: pd.DataFrame, participant_ids: Sequence[str]
) -> JointLabels:
    missing = set(LABEL_COLUMNS).difference(frame.columns)
    if missing:
        raise D07ContractError(f"Fold0 labels lack columns: {sorted(missing)}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(frame["fold_id"].astype(int).unique()) != {0}:
        raise D07ContractError("open Fold0 labels contain a nonzero fold")
    if set(frame["inner_role"].astype(str).unique()) != {"outer_test"}:
        raise D07ContractError("historical Fold0 labels have an unexpected role")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise D07ContractError("open Fold0 labels contain duplicate participant-organ rows")
    expected_index = pd.MultiIndex.from_product(
        [pd.Index([str(value) for value in participant_ids]), list(ORGAN_SYSTEMS)],
        names=["participant_id", "organ_system"],
    )
    indexed = frame.set_index(["participant_id", "organ_system"]).reindex(expected_index)
    if indexed[["event", "at_risk"]].isna().any().any():
        raise D07ContractError("open Fold0 labels are incomplete after alignment")
    missing_duration = indexed["duration_months"].isna()
    if (missing_duration & indexed["at_risk"].ne(0)).any():
        raise D07ContractError("at-risk Fold0 row has missing duration")
    indexed["duration_months"] = indexed["duration_months"].fillna(0.0)
    n = len(participant_ids)
    strict = indexed["strict_healthy_reference_60m"].to_numpy(dtype=np.int8).reshape(n, -1)
    if not np.all(strict == strict[:, :1]):
        raise D07ContractError("Fold0 strict-reference status disagrees across organs")
    labels = JointLabels(
        strict_healthy=torch.from_numpy(strict[:, 0].astype(bool)),
        duration_months=torch.from_numpy(
            indexed["duration_months"].to_numpy(dtype=np.float32).reshape(n, -1)
        ),
        event=torch.from_numpy(
            indexed["event"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)
        ),
        at_risk=torch.from_numpy(
            indexed["at_risk"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)
        ),
        inner_role=tuple("open_development" for _ in range(n)),
    )
    labels.validate(n)
    return labels


def load_authorized_open_fold0_data(
    *,
    cache_dir: Path,
    fold0_age_path: Path,
    fold0_labels_path: Path,
    sealed_manifest_path: Path,
    read_ledger_entry_path: Path,
    candidate_id: str,
) -> Fold0DevelopmentData:
    authorization = json.loads(read_ledger_entry_path.read_text(encoding="utf-8"))
    if (
        authorization.get("schema_version") != READ_SCHEMA
        or authorization.get("status") != "READ_COMMITTED_BEFORE_FOLD0_ACCESS"
        or authorization.get("candidate_id") != candidate_id
        or authorization.get("fold") != 0
        or authorization.get("fold_role") != "OPEN_DEVELOPMENT"
        or authorization.get("d07_protocol_sha256") != D07_PROTOCOL_SHA256
    ):
        raise D07ContractError("Fold0 read authorization is missing or invalid")
    role_path = read_ledger_entry_path.parent.parent / "FOLD0_ROLE_MANIFEST.json"
    if (
        not role_path.is_file()
        or sha256_file(role_path) != authorization.get("role_manifest_sha256")
    ):
        raise D07ContractError("Fold0 role manifest is missing or changed after authorization")
    role = json.loads(role_path.read_text(encoding="utf-8"))
    if (
        role.get("schema_version") != ROLE_SCHEMA
        or role.get("role") != "OPEN_DEVELOPMENT"
        or role.get("d07_protocol_sha256") != D07_PROTOCOL_SHA256
    ):
        raise D07ContractError("Fold0 role manifest does not authorize D07 development")

    checkpoint_path = Path(str(authorization.get("checkpoint_path", "")))
    evaluation_path = Path(str(authorization.get("evaluation_manifest_path", "")))
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != authorization.get("checkpoint_sha256")
    ):
        raise D07ContractError("authorized D07 checkpoint is missing or changed")
    if (
        not evaluation_path.is_file()
        or sha256_file(evaluation_path) != authorization.get("evaluation_manifest_sha256")
    ):
        raise D07ContractError("D07 evaluation manifest is missing or changed")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("schema_version")
        != "revision0710.d07.fold0_evaluation_manifest.v1"
        or evaluation.get("status") != "SEALED_BEFORE_FOLD0_READ"
        or evaluation.get("candidate_id") != candidate_id
        or evaluation.get("d07_protocol_sha256") != D07_PROTOCOL_SHA256
        or evaluation.get("checkpoint_sha256") != authorization.get("checkpoint_sha256")
        or evaluation.get("baseline_envelope_path")
        != str(DEFAULT_BASELINE_ENVELOPE.resolve())
        or evaluation.get("baseline_envelope_sha256")
        != sha256_file(DEFAULT_BASELINE_ENVELOPE)
        or evaluation.get("baseline_artifact_manifest_sha256")
        != sha256_file(DEFAULT_BASELINE_ARTIFACT_MANIFEST)
    ):
        raise D07ContractError("D07 evaluation manifest is incompatible with authorization")
    sealed = _verify_sealed_manifest_metadata(sealed_manifest_path)
    if evaluation.get("declared_fold0_artifact_hashes") != authorization.get(
        "declared_fold0_artifact_hashes"
    ):
        raise D07ContractError("D07 evaluation and read ledger disagree on Fold0 artifacts")
    actual_paths = {
        "age_targets": fold0_age_path,
        "joint_labels": fold0_labels_path,
    }
    for name, path in actual_paths.items():
        observed = sha256_file(path)
        declared = sealed["artifacts"][name]["sha256"]
        authorized = authorization["declared_fold0_artifact_hashes"][name]
        if observed != declared or observed != authorized:
            raise D07ContractError(f"open Fold0 {name} hash mismatch")
    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=0,
        fold_ids=(0,),
        age_target_cache_path=fold0_age_path,
    )
    if set(cache.dataset.fold_id.tolist()) != {0}:
        raise D07ContractError("authorized open-development cache is not Fold0 only")
    labels = _aligned_open_fold0_labels(
        pd.read_parquet(fold0_labels_path), cache.dataset.participant_ids
    )
    indices = torch.arange(len(cache.dataset), dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(
        cache=cache,
        labels=labels,
        train_indices=empty,
        calibration_indices=empty,
        screen_indices=indices,
        label_manifest=sealed,
    )


def calibration_to_open_fold0_cox(
    calibration: PredictionBundle,
    fold0: PredictionBundle,
    *,
    calibration_participant_ids: list[str],
    fold0_participant_ids: list[str],
    baseline_organ_delta_c: Mapping[str, float],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Any]:
    from crossfit_metrics import (
        apply_linear_age_correction,
        apply_linear_age_correction_to_validation,
        evaluate_cox_cell,
        fit_linear_age_correction,
    )

    calibration_frame = _evaluation_frame(calibration, calibration_participant_ids)
    fold0_frame = _evaluation_frame(fold0, fold0_participant_ids)
    correction = fit_linear_age_correction(calibration_frame)
    calibration_corrected = apply_linear_age_correction_to_validation(
        calibration_frame, correction
    )
    fold0_corrected = apply_linear_age_correction(fold0_frame, correction)
    organs: dict[str, dict[str, Any]] = {}
    deltas: list[float] = []
    for organ in ORGAN_SYSTEMS:
        calibration_organ = calibration_corrected.loc[
            calibration_corrected["organ_system"].astype(str).eq(organ)
            & calibration_corrected["at_risk"].eq(1)
        ].copy()
        fold0_organ = fold0_corrected.loc[
            fold0_corrected["organ_system"].astype(str).eq(organ)
            & fold0_corrected["at_risk"].eq(1)
        ].copy()
        cell = evaluate_cox_cell(
            calibration_organ,
            fold0_organ,
            fold=0,
            organ_system=organ,
        )
        organs[organ] = asdict(cell)
        deltas.append(float(cell.delta_c))
    result = {
        "definition": (
            "Fold1 calibration-fit strict-reference age correction and Cox nuisance; "
            "open Fold0 Harrell C-index on each organ's complete at_risk=1 population"
        ),
        "fit_population": "Fold1 calibration",
        "score_population": "open Fold0 organ-specific all at_risk=1",
        "strict_healthy_filter_applied_to_risk_score_population": False,
        "macro_delta_c": float(np.mean(deltas)),
        "positive_organs": int(sum(delta > 0.0 for delta in deltas)),
        "organs_beating_baseline": int(
            sum(deltas[index] > float(baseline_organ_delta_c[organ]) for index, organ in enumerate(ORGAN_SYSTEMS))
        ),
        "evaluable_organs": len(deltas),
        "organs": organs,
    }
    return result, calibration_corrected, fold0_corrected, correction


def _organ_age_metrics(
    chronological_age: np.ndarray,
    predicted_age: np.ndarray,
    strict_healthy: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for index, organ in enumerate(ORGAN_SYSTEMS):
        one_organ = np.repeat(predicted_age[:, index : index + 1], len(ORGAN_SYSTEMS), axis=1)
        metrics[organ] = pooled_age_metrics(
            chronological_age, one_organ, participant_mask=strict_healthy
        )
        metrics[organ]["row_n"] = metrics[organ]["participant_n"]
    return metrics


def _serialize_correction(correction: Any) -> dict[str, Any]:
    participant_ids = sorted(correction.validation_participant_ids)
    return {
        "fit_role": "Fold1 calibration nuisance only",
        "fit_participant_count": len(participant_ids),
        "fit_participant_ids_canonical_sha256": canonical_hash({"participant_ids": participant_ids}),
        "parameters": [asdict(parameter) for parameter in correction.parameters],
    }


def _score_and_finalize_d07_first_candidate(
    *,
    candidate_id: str,
    model: EHRAGEModel,
    calibration: PredictionBundle,
    calibration_ids: list[str],
    gates: SelectionGates,
    raw: Mapping[str, Any],
    run_dir: Path,
    output_root: Path,
    cache_dir: Path,
    fold0_age_path: Path,
    fold0_labels_path: Path,
    fold0_manifest_path: Path,
    role_path: Path,
    ledger_path: Path,
    run_config_path: Path,
    log_path: Path,
    parity_path: Path,
    checkpoint_path: Path,
    evaluation_manifest_path: Path,
    calibration_prediction_path: Path,
    started: float,
    emit: Callable[[str], None],
) -> Path:
    """Score one committed read and issue exactly one completion/failure receipt."""

    stage = "LOAD_AUTHORIZED_OPEN_FOLD0_DATA"
    artifact_hashes_path = run_dir / "artifact_hashes.json"
    artifacts: dict[str, Path] = {
        "run_config": run_config_path,
        "run_log": log_path,
        "training_history_e01_e12": run_dir / "training_history_e01_e12.json",
        "trajectory_parity_e01_e12": parity_path,
        "candidate_checkpoint_epoch12": checkpoint_path,
        "evaluation_manifest_before_fold0": evaluation_manifest_path,
        "fold1_calibration_nuisance_predictions": calibration_prediction_path,
        "fold0_role_manifest": role_path,
        "fold0_read_ledger_entry": ledger_path,
    }
    try:
        fold0_data = load_authorized_open_fold0_data(
            cache_dir=cache_dir.resolve(),
            fold0_age_path=fold0_age_path.resolve(),
            fold0_labels_path=fold0_labels_path.resolve(),
            sealed_manifest_path=fold0_manifest_path.resolve(),
            read_ledger_entry_path=ledger_path,
            candidate_id=candidate_id,
        )
        stage = "PREDICT_OPEN_FOLD0"
        training = raw["training"]
        fold0_batcher = JointVectorizedBatcher(
            fold0_data,
            fold0_data.screen_indices,
            batch_size=int(training["batch_size"]),
            shuffle=False,
            seed=D04_SEED,
        )
        device = next(model.parameters()).device
        fold0_prediction = _predict(
            model=model,
            batcher=fold0_batcher,
            device=device,
            amp=bool(training["amp"]),
            max_batches=None,
        )
        fold0_ids = [
            fold0_data.cache.dataset.participant_ids[int(index)]
            for index in fold0_prediction.participant_indices.tolist()
        ]
        if gates.baseline_organ_delta_c is None:
            raise D07ContractError("D07 Fold0 baseline envelope lacks organ delta-C values")

        stage = "EVALUATE_OPEN_FOLD0"
        risk_metrics, calibration_corrected, fold0_corrected, correction = (
            calibration_to_open_fold0_cox(
                calibration,
                fold0_prediction,
                calibration_participant_ids=calibration_ids,
                fold0_participant_ids=fold0_ids,
                baseline_organ_delta_c=gates.baseline_organ_delta_c,
            )
        )
        raw_strict = pooled_age_metrics(
            fold0_prediction.chronological_age,
            fold0_prediction.predicted_age,
            participant_mask=fold0_prediction.strict_healthy,
        )
        corrected_matrix = fold0_corrected["corrected_predicted_age"].to_numpy(
            dtype=np.float64
        ).reshape(len(fold0_ids), len(ORGAN_SYSTEMS))
        corrected_strict = pooled_age_metrics(
            fold0_prediction.chronological_age,
            corrected_matrix,
            participant_mask=fold0_prediction.strict_healthy,
        )
        eligible, gate_failures = gates.evaluate(raw_strict, risk_metrics)
        metrics = {
            "schema_version": "revision0710.d07.open_fold0_metrics.v1",
            "candidate_id": candidate_id,
            "fold": 0,
            "fold_role": "OPEN_DEVELOPMENT",
            "confirmatory_interpretation_allowed": False,
            "primary_same_epoch_row": {
                "epoch": D04_TARGET_EPOCH,
                "strict_mae": raw_strict["mae"],
                "strict_pearson": raw_strict["pearson"],
                "strict_r2": raw_strict["r2"],
                "age_sex_harrell_macro_delta_c": risk_metrics["macro_delta_c"],
            },
            "strict_age_raw_primary": {
                "population": "strict_healthy_reference_60m=1 only",
                **raw_strict,
                "organs": _organ_age_metrics(
                    fold0_prediction.chronological_age,
                    fold0_prediction.predicted_age,
                    fold0_prediction.strict_healthy,
                ),
            },
            "strict_age_fold1_corrected_diagnostic_only": {
                "population": "strict_healthy_reference_60m=1 only",
                **corrected_strict,
                "organs": _organ_age_metrics(
                    fold0_prediction.chronological_age,
                    corrected_matrix,
                    fold0_prediction.strict_healthy,
                ),
            },
            "age_sex_harrell_primary_development": risk_metrics,
            "d07_fold0_gate_eligible": eligible,
            "d07_fold0_gate_failures": gate_failures,
            "d07_fold0_thresholds": asdict(gates),
        }
        stage = "WRITE_OPEN_FOLD0_ARTIFACTS"
        metrics_path = run_dir / "fold0_open_development_metrics.json"
        _write_json(metrics_path, metrics)
        artifacts["fold0_open_development_metrics"] = metrics_path
        correction_path = run_dir / "fold1_calibration_age_correction.json"
        _write_json(correction_path, _serialize_correction(correction))
        artifacts["fold1_calibration_age_correction"] = correction_path

        fold0_corrected["model_risk_score"] = fold0_prediction.risk_score.astype(
            np.float32
        ).reshape(-1)
        fold0_corrected["fold_role"] = "OPEN_DEVELOPMENT"
        fold0_prediction_path = run_dir / "fold0_open_development_predictions_long.parquet"
        fold0_corrected.to_parquet(fold0_prediction_path, index=False)
        artifacts["fold0_open_development_predictions_long"] = fold0_prediction_path
        calibration_corrected["model_risk_score"] = calibration.risk_score.astype(
            np.float32
        ).reshape(-1)
        calibration_corrected["fold_role"] = "FOLD1_CALIBRATION_NUISANCE_ONLY"
        calibration_corrected_path = (
            run_dir / "fold1_calibration_nuisance_corrected_long.parquet"
        )
        calibration_corrected.to_parquet(calibration_corrected_path, index=False)
        artifacts["fold1_calibration_nuisance_corrected_long"] = calibration_corrected_path

        summary_path = run_dir / "candidate_summary.json"
        _write_json(
            summary_path,
            {
                "schema_version": CANDIDATE_SCHEMA,
                "status": "OPEN_FOLD0_DEVELOPMENT_SCORED",
                "created_utc": _utc_now(),
                "candidate_id": candidate_id,
                "fold0_role": "OPEN_DEVELOPMENT",
                "fold0_confirmatory_claim_allowed": False,
                "fold0_read_ledger_entry": str(ledger_path.resolve()),
                "fold0_read_count_this_candidate": 1,
                "fold1_screen_metrics_computed": False,
                "fold2_to_4_heldout_ehr_metrics_computed": False,
                "captured_epoch": D04_TARGET_EPOCH,
                "scheduler_horizon_epochs": D04_SCHEDULER_HORIZON_EPOCHS,
                "primary_same_epoch_row": metrics["primary_same_epoch_row"],
                "d07_fold0_gate_eligible": eligible,
                "d07_fold0_gate_failures": gate_failures,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        artifacts["candidate_summary"] = summary_path

        # COMPLETE is the final run-log mutation. Hash only after this append.
        stage = "FINALIZE_RUN_LOG"
        emit(
            f"COMPLETE candidate={candidate_id} strict_mae={raw_strict['mae']} "
            f"strict_r={raw_strict['pearson']} strict_r2={raw_strict['r2']} "
            f"harrell_delta_c={risk_metrics['macro_delta_c']} eligible={eligible}"
        )
        stage = "WRITE_ARTIFACT_HASH_REGISTRY"
        write_artifact_hash_registry(artifact_hashes_path, artifacts)
        artifacts["artifact_hashes"] = artifact_hashes_path
        stage = "WRITE_COMPLETION_RECEIPT"
        write_fold0_read_completion_receipt(
            output_root=output_root,
            read_ledger_entry_path=ledger_path,
            artifacts=artifacts,
        )
        return run_dir
    except BaseException as exc:
        # Once the registry exists, do not mutate run.log and invalidate its hash.
        if not artifact_hashes_path.exists():
            try:
                emit(f"FAILED_AFTER_COMMITTED_READ stage={stage} error={type(exc).__name__}: {exc}")
            except Exception:
                pass
        else:
            print(
                f"FAILED_AFTER_COMMITTED_READ stage={stage} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            artifacts["artifact_hashes"] = artifact_hashes_path
        try:
            write_fold0_read_failure_receipt(
                output_root=output_root,
                read_ledger_entry_path=ledger_path,
                stage=stage,
                exception=exc,
                artifacts=artifacts,
            )
        except BaseException as receipt_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(f"Failure receipt could not be written: {receipt_exc}")
        raise


def run_d07_first_open_fold0_candidate(
    *,
    config_path: Path = D04_CONFIG_PATH,
    source_run_dir: Path = D04_SOURCE_RUN_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    development_labels_path: Path = DEFAULT_DEVELOPMENT_LABELS,
    development_age_path: Path = DEFAULT_DEVELOPMENT_AGES,
    development_manifest_path: Path = DEFAULT_DEVELOPMENT_MANIFEST,
    fold0_age_path: Path = DEFAULT_FOLD0_AGES,
    fold0_labels_path: Path = DEFAULT_FOLD0_LABELS,
    fold0_manifest_path: Path = DEFAULT_FOLD0_MANIFEST,
    baseline_envelope_path: Path = DEFAULT_BASELINE_ENVELOPE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    command_line: Sequence[str] = (),
) -> Path:
    started = time.perf_counter()
    protocol_hash = validate_d07_protocol()
    gates, baseline_envelope, baseline_validation = (
        validate_d07_fold0_baseline_envelope(baseline_envelope_path)
    )
    raw, source_config, source_history = _validate_d04_lineage(
        config_path=config_path.resolve(), source_run_dir=source_run_dir.resolve()
    )
    candidate_id = "001_d04_epoch12"
    run_dir = output_root.resolve() / "candidates" / candidate_id
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    log_path.write_text("", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    current_sources = _source_hashes(
        cache_dir=cache_dir.resolve(),
        development_labels_path=development_labels_path.resolve(),
        development_age_path=development_age_path.resolve(),
        development_manifest_path=development_manifest_path.resolve(),
    )
    if current_sources != source_config["source_hashes"]:
        raise D07ContractError("D04 training/data source hashes changed before reproduction")
    materialized_config = {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_number": 1,
        "candidate_role": "OPEN_FOLD0_DEVELOPMENT",
        "d07_protocol_path": str(D07_PROTOCOL_PATH.resolve()),
        "d07_protocol_sha256": protocol_hash,
        "source_d04_config_path": str(config_path.resolve()),
        "source_d04_config_sha256": D04_CONFIG_SHA256,
        "source_d04_run_dir": str(source_run_dir.resolve()),
        "source_d04_run_config_sha256": D04_RUN_CONFIG_SHA256,
        "source_d04_history_sha256": D04_HISTORY_SHA256,
        "source_d04_canonical_config_hash": D04_CANONICAL_CONFIG_HASH,
        "capture_epoch": D04_TARGET_EPOCH,
        "scheduler_horizon_epochs": D04_SCHEDULER_HORIZON_EPOCHS,
        "stop_training_after_capture": True,
        "seed": D04_SEED,
        "deterministic": True,
        "fold_roles": {
            "fold0": "OPEN_DEVELOPMENT",
            "fold1": "CALIBRATION_NUISANCE_ONLY_NO_SCREEN_METRICS",
            "fold2_to_4": "TRAINING_ONLY_NO_HELDOUT_EHR_METRICS",
        },
        "primary_metric_populations": {
            "strict_mae_pearson_r2": "open Fold0 strict_healthy_reference_60m=1 only",
            "age_sex_harrell_delta_c": "open Fold0 complete organ-specific at_risk=1 population",
        },
        "fold0_artifacts": {
            "age_targets": str(fold0_age_path.resolve()),
            "joint_labels": str(fold0_labels_path.resolve()),
            "historical_manifest": str(fold0_manifest_path.resolve()),
        },
        "baseline_envelope_path": str(baseline_envelope_path.resolve()),
        "baseline_envelope_sha256": sha256_file(baseline_envelope_path),
        "baseline_envelope_schema_version": baseline_envelope["schema_version"],
        "baseline_artifact_manifest_path": str(
            DEFAULT_BASELINE_ARTIFACT_MANIFEST.resolve()
        ),
        "baseline_artifact_manifest_sha256": sha256_file(
            DEFAULT_BASELINE_ARTIFACT_MANIFEST
        ),
        "baseline_validator_path": str(DEFAULT_BASELINE_VALIDATOR.resolve()),
        "baseline_validator_sha256": sha256_file(DEFAULT_BASELINE_VALIDATOR),
        "baseline_validation": baseline_validation,
        "source_hashes": current_sources,
        "code_hashes": _current_code_hashes(),
        "evaluation_command": shlex.join(list(command_line)) if command_line else None,
        "prohibitions": {
            "warm_start": True,
            "fine_tune": True,
            "teacher": True,
            "distillation": True,
            "baseline_predictions_or_checkpoints": True,
            "epoch60_state_as_epoch12": True,
        },
    }
    run_config_path = run_dir / "run_config.json"
    _write_json_exclusive(run_config_path, materialized_config)
    data = load_fold0_development_data(
        cache_dir=cache_dir.resolve(),
        label_cache_path=development_labels_path.resolve(),
        age_target_cache_path=development_age_path.resolve(),
        label_manifest_path=development_manifest_path.resolve(),
    )
    model, calibration, checkpoint_path, parity_path = _train_d04_to_epoch12(
        raw=raw,
        source_history=source_history,
        data=data,
        run_dir=run_dir,
        emit=emit,
    )
    calibration_ids = [
        data.cache.dataset.participant_ids[int(index)]
        for index in calibration.participant_indices.tolist()
    ]
    calibration_frame = _evaluation_frame(calibration, calibration_ids)
    calibration_prediction_path = run_dir / "fold1_calibration_nuisance_predictions.parquet"
    calibration_frame.to_parquet(calibration_prediction_path, index=False)

    sealed_metadata = _verify_sealed_manifest_metadata(fold0_manifest_path.resolve())
    evaluation_manifest_path = run_dir / "EVALUATION_MANIFEST_BEFORE_FOLD0.json"
    _write_json_exclusive(
        evaluation_manifest_path,
        {
            "schema_version": "revision0710.d07.fold0_evaluation_manifest.v1",
            "status": "SEALED_BEFORE_FOLD0_READ",
            "created_utc": _utc_now(),
            "candidate_id": candidate_id,
            "d07_protocol_sha256": protocol_hash,
            "run_config_sha256": sha256_file(run_config_path),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "captured_epoch": D04_TARGET_EPOCH,
            "scheduler_horizon_epochs": D04_SCHEDULER_HORIZON_EPOCHS,
            "trajectory_parity_sha256": sha256_file(parity_path),
            "fold1_calibration_predictions_sha256": sha256_file(calibration_prediction_path),
            "declared_fold0_artifact_hashes": {
                name: sealed_metadata["artifacts"][name]["sha256"]
                for name in ("age_targets", "joint_labels")
            },
            "metric_implementation_hashes": {
                "runner": sha256_file(Path(__file__).resolve()),
                "crossfit_metrics": sha256_file(
                    REVISION_ROOT / "05_downstream" / "common" / "crossfit_metrics.py"
                ),
            },
            "metric_populations": materialized_config["primary_metric_populations"],
            "baseline_envelope_path": str(baseline_envelope_path.resolve()),
            "baseline_envelope_sha256": sha256_file(baseline_envelope_path),
            "baseline_artifact_manifest_sha256": sha256_file(
                DEFAULT_BASELINE_ARTIFACT_MANIFEST
            ),
            "evaluation_command": materialized_config["evaluation_command"],
            "candidate_fallback_allowed": False,
        },
    )
    role_path, ledger_path = commit_open_development_read(
        output_root=output_root.resolve(),
        candidate_id=candidate_id,
        sealed_manifest_path=fold0_manifest_path.resolve(),
        checkpoint_path=checkpoint_path,
        evaluation_manifest_path=evaluation_manifest_path,
    )
    return _score_and_finalize_d07_first_candidate(
        candidate_id=candidate_id,
        model=model,
        calibration=calibration,
        calibration_ids=calibration_ids,
        gates=gates,
        raw=raw,
        run_dir=run_dir,
        output_root=output_root.resolve(),
        cache_dir=cache_dir.resolve(),
        fold0_age_path=fold0_age_path.resolve(),
        fold0_labels_path=fold0_labels_path.resolve(),
        fold0_manifest_path=fold0_manifest_path.resolve(),
        role_path=role_path,
        ledger_path=ledger_path,
        run_config_path=run_config_path,
        log_path=log_path,
        parity_path=parity_path,
        checkpoint_path=checkpoint_path,
        evaluation_manifest_path=evaluation_manifest_path,
        calibration_prediction_path=calibration_prediction_path,
        started=started,
        emit=emit,
    )
