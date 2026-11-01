from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import v8_authorization
import v8_contract
import v8_data
import v8_inputs
import v8_loss
import v8_model
import v8_schema_validator
import v8_training
import v8_runtime_guard


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
CANONICAL_MODEL_DIR = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "01_model"
if str(CANONICAL_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CANONICAL_MODEL_DIR))

from ehr_age_model import (  # noqa: E402
    BASE_AGE_HEAD_MODE_MLP,
    BASE_AGE_MODE_PRIVATE_RETAIN,
    EHRAGEConfig,
)


TASK_SCHEMA = "revision0710.fig04_fig05.isolated_worker_task.v8"
COMMON_TASK_KEYS = {
    "schema_version",
    "phase",
    "role",
    "run_id",
    "variant_id",
    "outer_fold",
    "seed",
    "protocol",
    "data_source",
    "output_dir",
    "contract_path",
    "contract_sha256",
    "task_path",
    "claim_path",
    "claim_sha256",
    "authorization",
    "gpu_lock",
}
PHASE_EXTRA_KEYS = {
    "train": set(),
    "calibration": {"training_complete_path", "checkpoint_path"},
    "outer_test": {
        "training_complete_path",
        "checkpoint_path",
        "calibration_complete_path",
        "age_correction_path",
        "cox_nuisance_path",
        "outer_read_ledger_path",
    },
}


class WorkerError(RuntimeError):
    pass


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise WorkerError(f"create-only worker artifact exists: {path}") from exc


def _output_schema(task: dict[str, Any], table: str) -> dict[str, Any]:
    schemas = v8_schema_validator.load_schemas(ROOT / "OUTPUT_SCHEMAS_V8.json")
    return v8_schema_validator.schema_for_execution(
        schemas,
        table,
        execution_mode=str(task["protocol"]["execution_mode"]),
        evaluation_epoch=int(task["protocol"]["evaluate_only_epoch"]),
    )


def _validate_json_output(
    task: dict[str, Any], table: str, payload: dict[str, Any]
) -> None:
    v8_schema_validator.validate_json_payload(
        payload, _output_schema(task, table), table=table
    )


def _close_table_output(
    task: dict[str, Any], table: str, frame: pd.DataFrame
) -> pd.DataFrame:
    return v8_schema_validator.order_and_validate_table(
        frame, _output_schema(task, table), table=table
    )


def _scaled_protocol() -> dict[str, Any]:
    return {
        "execution_mode": "scaled_integration",
        "batch_size": 16,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": 0,
        "scheduler_horizon_epochs": 2,
        "train_through_epoch": 2,
        "capture_epoch": 1,
        "evaluate_only_epoch": 1,
        "amp": True,
        "device": "cpu",
    }


def _formal_protocol() -> dict[str, Any]:
    frozen = v8_contract.EXPECTED_TRAINING_PROTOCOL
    return {
        "execution_mode": "formal",
        "batch_size": frozen["batch_size"],
        "learning_rate": frozen["learning_rate"],
        "weight_decay": frozen["weight_decay"],
        "gradient_clip_norm": frozen["gradient_clip_norm"],
        "warmup_epochs": frozen["warmup_epochs"],
        "scheduler_horizon_epochs": frozen["scheduler_horizon_epochs"],
        "train_through_epoch": frozen["train_through_epoch"],
        "capture_epoch": frozen["capture_epoch"],
        "evaluate_only_epoch": frozen["evaluate_only_epoch"],
        "amp": frozen["amp"],
        "device": "cuda",
    }


def _validate_formal_task_bindings(task: dict[str, Any], expected_phase: str) -> None:
    contract_path = (ROOT / "FROZEN_ABLATION_CONTRACT_V8.json").resolve()
    if Path(task["contract_path"]).resolve() != contract_path:
        raise WorkerError("formal contract path changed")
    if task["contract_sha256"] != v8_contract.sha256_file(contract_path):
        raise WorkerError("formal contract SHA changed")

    contract = v8_contract.load_contract(ROOT)
    v8_contract.validate_contract_payload(
        contract,
        ROOT,
        verify_hashes=False,
        enforce_prefreeze_absence=False,
    )
    queue = v8_contract.read_queue(ROOT / "ABLATION_QUEUE_V8.csv")
    v8_contract.validate_queue(queue, contract)
    matches = [row for row in queue if row["run_id"] == task["run_id"]]
    if len(matches) != 1:
        raise WorkerError("formal queue run binding changed")
    row = matches[0]
    if row["variant_id"] != task["variant_id"] or int(row["outer_fold"]) != int(task["outer_fold"]):
        raise WorkerError("formal queue identity changed")

    role = "outer_test" if expected_phase == "outer_test" else expected_phase
    record_name = f"outer_fold_{int(task['outer_fold'])}_{role}_manifest"
    records = [
        record
        for record in contract["canonical_bindings"]["records"]
        if record["name"] == record_name
    ]
    if len(records) != 1:
        raise WorkerError("formal canonical role binding is missing")
    record = records[0]
    expected_source = {
        "format": "revision0710_role_manifest",
        "role": role,
        "role_manifest_path": str((v8_contract.repo_root(ROOT) / record["path"]).resolve()),
        "role_manifest_sha256": record["sha256"],
        "cache_dir": str(
            (
                v8_contract.repo_root(ROOT)
                / "Revision-0710"
                / "01_data_rebuild"
                / "04_cache_v1"
            ).resolve()
        ),
        "cache_artifact_bindings_sha256": v8_contract.cache_artifact_bindings_sha256(
            contract["cache_artifact_bindings"]
        ),
    }
    if task["data_source"] != expected_source:
        raise WorkerError("formal canonical data-source binding changed")

    output = Path(task["output_dir"]).resolve()
    attempt = output.parent
    expected_phase_dir = {
        "train": "01_train",
        "calibration": "02_calibration",
        "outer_test": "03_outer_test",
    }[expected_phase]
    if output.name != expected_phase_dir or not attempt.name.startswith("attempt_"):
        raise WorkerError("formal output phase/attempt path changed")
    try:
        ordinal = int(attempt.name.removeprefix("attempt_"))
    except ValueError as exc:
        raise WorkerError("formal attempt ordinal path changed") from exc
    if ordinal < 1 or len(attempt.name.removeprefix("attempt_")) != 4:
        raise WorkerError("formal attempt ordinal path changed")
    if attempt.parent.resolve() != Path(row["intended_run_dir"]).resolve():
        raise WorkerError("formal output run root changed")
    expected_task_path = attempt / {
        "train": "TRAIN_TASK.json",
        "calibration": "CALIBRATION_TASK.json",
        "outer_test": "OUTER_TASK.json",
    }[expected_phase]
    if Path(task["task_path"]).resolve() != expected_task_path.resolve():
        raise WorkerError("formal task file path binding changed")
    if Path(task["claim_path"]).resolve() != (attempt / "CLAIM.json").resolve():
        raise WorkerError("formal claim path binding changed")

    expected_paths: dict[str, Path] = {}
    if expected_phase in {"calibration", "outer_test"}:
        expected_paths.update(
            {
                "training_complete_path": attempt / "01_train" / "TRAINING_COMPLETE.json",
                "checkpoint_path": attempt / "01_train" / "fixed_epoch_0055.pt",
            }
        )
    if expected_phase == "outer_test":
        expected_paths.update(
            {
                "calibration_complete_path": attempt / "02_calibration" / "CALIBRATION_COMPLETE.json",
                "age_correction_path": attempt / "02_calibration" / "age_correction.json",
                "cox_nuisance_path": attempt / "02_calibration" / "cox_nuisance.json",
                "outer_read_ledger_path": attempt / "OUTER_READ_LEDGER.json",
            }
        )
    for key, expected in expected_paths.items():
        if Path(task[key]).resolve() != expected.resolve():
            raise WorkerError(f"formal task path binding changed: {key}")


def synthetic_task_template(phase: str) -> dict[str, Any]:
    if phase not in PHASE_EXTRA_KEYS:
        raise WorkerError("unknown worker phase")
    role = phase
    run_id = "fig04_01_canonical_control__outer0"
    role_name = "outer_test" if phase == "outer_test" else phase
    task: dict[str, Any] = {
        "schema_version": TASK_SCHEMA,
        "phase": phase,
        "role": role_name,
        "run_id": run_id,
        "variant_id": "fig04_01_canonical_control",
        "outer_fold": 0,
        "seed": 7210,
        "protocol": _scaled_protocol(),
        "data_source": {
            "format": "synthetic_role_bundle",
            "role": role_name,
            "role_bundle_path": str((ROOT / "smoke" / "template" / role_name / "role_bundle.pt").resolve()),
            "role_bundle_sha256": "0" * 64,
        },
        "output_dir": str((ROOT / "smoke" / "template" / phase).resolve()),
        "contract_path": str((ROOT / "FROZEN_ABLATION_CONTRACT_V8.json").resolve()),
        "contract_sha256": "0" * 64,
        "task_path": str(
            (
                ROOT
                / "smoke"
                / "template"
                / {
                    "train": "TRAIN_TASK.json",
                    "calibration": "CALIBRATION_TASK.json",
                    "outer_test": "OUTER_TASK.json",
                }[phase]
            ).resolve()
        ),
        "claim_path": str((ROOT / "smoke" / "template" / "CLAIM.json").resolve()),
        "claim_sha256": "0" * 64,
        "authorization": {"required": False, "path": None, "sha256": None},
        "gpu_lock": {
            "required": False,
            "path": None,
            "token": None,
            "owner_pid": None,
            "run_id": None,
        },
    }
    if phase in {"calibration", "outer_test"}:
        task["training_complete_path"] = str((ROOT / "smoke" / "template" / "train" / "TRAINING_COMPLETE.json").resolve())
        task["checkpoint_path"] = str((ROOT / "smoke" / "template" / "train" / "fixed_epoch_0001.pt").resolve())
    if phase == "outer_test":
        task["calibration_complete_path"] = str((ROOT / "smoke" / "template" / "calibration" / "CALIBRATION_COMPLETE.json").resolve())
        task["age_correction_path"] = str((ROOT / "smoke" / "template" / "calibration" / "age_correction.json").resolve())
        task["cox_nuisance_path"] = str((ROOT / "smoke" / "template" / "calibration" / "cox_nuisance.json").resolve())
        task["outer_read_ledger_path"] = str((ROOT / "smoke" / "template" / "OUTER_READ_LEDGER.json").resolve())
    return task


def validate_task(
    task: dict[str, Any],
    *,
    expected_phase: str,
    runtime_checks: bool = True,
) -> None:
    if expected_phase not in PHASE_EXTRA_KEYS:
        raise WorkerError("unknown expected worker phase")
    expected_keys = COMMON_TASK_KEYS | PHASE_EXTRA_KEYS[expected_phase]
    if set(task) != expected_keys:
        raise WorkerError("worker task exact keys changed")
    if task.get("schema_version") != TASK_SCHEMA or task.get("phase") != expected_phase:
        raise WorkerError("worker task schema or phase changed")
    expected_role = "outer_test" if expected_phase == "outer_test" else expected_phase
    if task.get("role") != expected_role:
        raise WorkerError("worker role does not match phase")
    variant = str(task.get("variant_id"))
    if variant not in v8_contract.VARIANT_IDS:
        raise WorkerError("unknown variant")
    outer = int(task.get("outer_fold", -1))
    if outer not in range(5) or task.get("run_id") != f"{variant}__outer{outer}":
        raise WorkerError("worker run identity changed")
    if int(task.get("seed", -1)) != 7210:
        raise WorkerError("worker seed changed")
    data_source = task.get("data_source")
    if not isinstance(data_source, dict) or data_source.get("role") != expected_role:
        raise WorkerError("worker data-source role changed")
    if data_source.get("format") == "synthetic_role_bundle":
        if set(data_source) != {"format", "role", "role_bundle_path", "role_bundle_sha256"}:
            raise WorkerError("synthetic data source keys changed")
    elif data_source.get("format") == "revision0710_role_manifest":
        if set(data_source) != {
            "format", "role", "role_manifest_path", "role_manifest_sha256",
            "cache_dir", "cache_artifact_bindings_sha256",
        }:
            raise WorkerError("canonical data source keys changed")
    else:
        raise WorkerError("unknown worker data-source format")
    protocol = task.get("protocol")
    if not isinstance(protocol, dict):
        raise WorkerError("worker protocol is not an object")
    if protocol.get("execution_mode") == "formal":
        if protocol != _formal_protocol():
            raise WorkerError("formal task protocol changed")
        authorization = task["authorization"]
        if not isinstance(authorization, dict) or set(authorization) != {"required", "path", "sha256"}:
            raise WorkerError("formal worker authorization keys changed")
        if authorization.get("required") is not True:
            raise WorkerError("formal worker authorization is not required")
        lock = task["gpu_lock"]
        if not isinstance(lock, dict) or set(lock) != v8_runtime_guard.LOCK_BINDING_KEYS:
            raise WorkerError("formal worker GPU lock keys changed")
        if runtime_checks:
            contract = v8_contract.load_contract(ROOT)
            v8_runtime_guard.validate_formal_environment(contract["environment"]["formal"])
            v8_runtime_guard.validate_global_gpu_lock(lock, task_run_id=task["run_id"])
            path = Path(str(authorization.get("path")))
            payload = v8_authorization.load_and_validate_authorization(
                path, ROOT, verify_contract_file_hashes=False
            )
            if v8_contract.sha256_file(path) != authorization.get("sha256"):
                raise WorkerError("formal worker authorization SHA changed")
            if task["run_id"] not in payload["authorized_run_ids"]:
                raise WorkerError("run is not authorized")
        # Executor/GO performed the full cache rehash. The worker now rehashes
        # code/authorization plus only the current role's selected cache files.
        _validate_formal_task_bindings(task, expected_phase)
    elif protocol != _scaled_protocol():
        raise WorkerError("scaled integration protocol changed")
    elif task["gpu_lock"] != {
        "required": False,
        "path": None,
        "token": None,
        "owner_pid": None,
        "run_id": None,
    }:
        raise WorkerError("scaled integration must not carry a GPU lock")


def validate_task_structure(task: dict[str, Any], *, expected_phase: str) -> None:
    """Exercise the real task validator without consuming auth or requiring CUDA."""

    validate_task(task, expected_phase=expected_phase, runtime_checks=False)


def _task_claim_opened_paths(task: dict[str, Any]) -> list[str]:
    task_path = Path(str(task["task_path"]))
    claim_path = Path(str(task["claim_path"]))
    if not task_path.is_file() or not claim_path.is_file():
        raise WorkerError("task/claim closure file is missing")
    stored_task = json.loads(task_path.read_text(encoding="utf-8"))
    if stored_task != task:
        raise WorkerError("executed task differs from sealed task bytes")
    if v8_contract.sha256_file(claim_path) != task["claim_sha256"]:
        raise WorkerError("claim SHA binding changed")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if (
        claim.get("run_id") != task["run_id"]
        or int(claim.get("seed", -1)) != int(task["seed"])
        or bool(claim.get("selection_uses_metrics", True))
    ):
        raise WorkerError("task claim identity changed")
    attempt = Path(task["output_dir"]).resolve().parent
    if claim_path.resolve() != (attempt / "CLAIM.json").resolve():
        raise WorkerError("claim is outside the run attempt")
    expected_task_name = {
        "train": "TRAIN_TASK.json",
        "calibration": "CALIBRATION_TASK.json",
        "outer_test": "OUTER_TASK.json",
    }[str(task["phase"])]
    if task_path.resolve() != (attempt / expected_task_name).resolve():
        raise WorkerError("task is outside the run attempt")
    opened = [str(task_path.resolve()), str(claim_path.resolve())]
    if task["protocol"]["execution_mode"] == "formal":
        opened.extend(
            [
                str(Path(task["contract_path"]).resolve()),
                str((ROOT / "ABLATION_QUEUE_V8.csv").resolve()),
                str(Path(task["authorization"]["path"]).resolve()),
                *[
                    str(path)
                    for path in v8_runtime_guard.lock_binding_paths(task["gpu_lock"])
                ],
            ]
        )
    return opened


def _closed_opened_paths(*groups: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in (item for group in groups for item in group):
        resolved = str(Path(value).resolve())
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _load_source(task: dict[str, Any]) -> tuple[v8_data.RoleBundle, torch.Tensor, torch.Tensor, list[str]]:
    source = task["data_source"]
    if source["format"] == "synthetic_role_bundle":
        path = Path(source["role_bundle_path"])
        if v8_data.sha256_file(path) != source["role_bundle_sha256"]:
            raise WorkerError("synthetic role bundle SHA changed")
        bundle = v8_data.load_role_bundle(path, expected_role=task["role"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71002)
            code = torch.randn(64, 8)
            text = torch.randn(64, 8)
            code[0].zero_()
            text[0].zero_()
        return bundle, code, text, [str(path.resolve())]
    contract = v8_contract.load_contract(ROOT)
    bundle, code, text = v8_data.load_revision0710_role(
        role_manifest_path=Path(source["role_manifest_path"]),
        expected_manifest_sha256=source["role_manifest_sha256"],
        cache_dir=Path(source["cache_dir"]),
        outer_fold=int(task["outer_fold"]),
        expected_role=task["role"],
        cache_artifact_bindings=contract["cache_artifact_bindings"],
        expected_cache_artifact_bindings_sha256=source[
            "cache_artifact_bindings_sha256"
        ],
    )
    role_manifest_path = Path(source["role_manifest_path"])
    role_manifest = v8_contract.read_json_exact(role_manifest_path)
    cache_dir = Path(source["cache_dir"])
    selected_folds = set(int(value) for value in role_manifest["artifact"]["fold_ids"])
    cache_bindings = contract["cache_artifact_bindings"]
    opened = [
        str(role_manifest_path.resolve()),
        str(Path(role_manifest["artifact"]["age_targets"]["path"]).resolve()),
        str(Path(role_manifest["artifact"]["joint_labels"]["path"]).resolve()),
        str((cache_dir / cache_bindings["batch_manifest"]["path"]).resolve()),
        str((cache_dir / cache_bindings["fold_transform_manifest"]["path"]).resolve()),
        str((cache_dir / cache_bindings["fold_transforms"][int(task["outer_fold"])]["path"]).resolve()),
    ]
    opened.extend(
        str((cache_dir / row["path"]).resolve())
        for row in cache_bindings["shards"]
        if int(row["fold_id"]) in selected_folds
    )
    return bundle, code, text, opened


def _model_config(*, tiny: bool, payload: dict[str, Any] | None = None) -> EHRAGEConfig:
    if payload is not None:
        return EHRAGEConfig(**payload)
    if tiny:
        return EHRAGEConfig(
            hidden_dim=12,
            num_heads=3,
            num_transformer_layers=1,
            num_gru_layers=1,
            dropout=0.0,
            freeze_event_embeddings=True,
            risk_offset_cap_years=8.0,
            risk_temperature=2.0,
            survival_intervals=5,
            baseline_log_rate_init=-7.5,
            base_age_mode=BASE_AGE_MODE_PRIVATE_RETAIN,
            base_age_head_mode=BASE_AGE_HEAD_MODE_MLP,
        )
    return EHRAGEConfig(
        hidden_dim=192,
        num_heads=6,
        num_transformer_layers=3,
        num_gru_layers=2,
        dropout=0.15,
        freeze_event_embeddings=True,
        risk_offset_cap_years=8.0,
        risk_temperature=2.0,
        survival_intervals=5,
        baseline_log_rate_init=-7.5,
        base_age_mode=BASE_AGE_MODE_PRIVATE_RETAIN,
        base_age_head_mode=BASE_AGE_HEAD_MODE_MLP,
    )


def _restore_model(task: dict[str, Any], code: torch.Tensor, text: torch.Tensor):
    checkpoint_path = Path(task["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant_id") != task["variant_id"] or int(checkpoint.get("epoch", -1)) != int(task["protocol"]["capture_epoch"]):
        raise WorkerError("checkpoint identity changed")
    model = v8_model.build_production_model(
        task["variant_id"],
        code,
        text,
        age_anchor=float(checkpoint["age_anchor"]),
        config=_model_config(tiny=False, payload=checkpoint["model_config"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(task["protocol"]["device"]))
    return model, checkpoint


def run_train_task(task: dict[str, Any]) -> dict[str, Any]:
    validate_task(task, expected_phase="train")
    closure_opened = _task_claim_opened_paths(task)
    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    bundle, code, text, opened = _load_source(task)
    protocol = task["protocol"]
    v8_training.configure_reproducibility(task["seed"], deterministic=True, tf32=True)
    model = v8_model.build_production_model(
        task["variant_id"],
        code,
        text,
        age_anchor=float(bundle.labels["chronological_age"].float().mean()),
        config=_model_config(tiny=protocol["execution_mode"] == "scaled_integration"),
    )
    batcher = v8_data.RoleBatcher(
        bundle,
        batch_size=int(protocol["batch_size"]),
        seed=int(task["seed"]),
        shuffle=True,
    )
    device = torch.device(protocol["device"])
    checkpoint_path = output / f"fixed_epoch_{int(protocol['capture_epoch']):04d}.pt"
    result = v8_training.train_fresh(
        model,
        task["variant_id"],
        batcher,
        device=device,
        seed=int(task["seed"]),
        epochs=int(protocol["train_through_epoch"]),
        capture_epoch=int(protocol["capture_epoch"]),
        learning_rate=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
        gradient_clip_norm=float(protocol["gradient_clip_norm"]),
        warmup_epochs=int(protocol["warmup_epochs"]),
        scheduler_horizon_epochs=int(protocol["scheduler_horizon_epochs"]),
        amp=bool(protocol["amp"]),
        checkpoint_path=checkpoint_path,
    )
    model_manifest_path = output / "MODEL_OPTIMIZER_MANIFEST.json"
    optimizer_manifest = dict(result["optimizer_manifest"])
    optimizer_manifest.update(
        {
            "initial_full_model_sha256": result["trajectory_attestation"]["initial_full_model_sha256"],
            "initial_optimizer_sha256": result["trajectory_attestation"]["initial_optimizer_sha256"],
            "initial_scaler_sha256": result["trajectory_attestation"]["initial_scaler_sha256"],
        }
    )
    _write_json_exclusive(model_manifest_path, optimizer_manifest)
    attestation = {
        **result["trajectory_attestation"],
        "run_id": task["run_id"],
        "outer_fold": int(task["outer_fold"]),
        "task_sha256": v8_contract.sha256_file(Path(task["task_path"])),
        "claim_sha256": task["claim_sha256"],
        "train_role_source_sha256": task["data_source"][
            "role_bundle_sha256"
            if task["data_source"]["format"] == "synthetic_role_bundle"
            else "role_manifest_sha256"
        ],
        "protocol_sha256": v8_contract.canonical_json_sha256(task["protocol"]),
    }
    trajectory_path = output / "TRAINING_TRAJECTORY_ATTESTATION.json"
    _write_json_exclusive(trajectory_path, attestation)
    checkpoint_manifest = {
        "schema_version": "revision0710.fig04_fig05.checkpoint_manifest.v8",
        "status": (
            "SEALED_E55_ONLY"
            if protocol["execution_mode"] == "formal"
            else "SEALED_SCALED_CAPTURE_ONLY"
        ),
        "run_id": task["run_id"],
        "variant_id": task["variant_id"],
        "outer_fold": int(task["outer_fold"]),
        "evaluation_epoch": int(protocol["capture_epoch"]),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "initial_trainable_sha256": result["optimizer_manifest"]["initial_trainable_sha256"],
        "initial_full_model_sha256": attestation["initial_full_model_sha256"],
        "initial_optimizer_sha256": attestation["initial_optimizer_sha256"],
        "capture_model_state_sha256": attestation["capture_model_state_sha256"],
        "capture_trajectory_sha256": attestation["capture_trajectory_sha256"],
        "training_trajectory_attestation_sha256": v8_contract.sha256_file(trajectory_path),
        "optimizer_name_set_sha256": result["optimizer_manifest"]["optimizer_name_set_sha256"],
        "model_parameter_count": int(result["optimizer_manifest"]["model_parameter_count"]),
        "trainable_parameter_count": int(result["optimizer_manifest"]["trainable_parameter_count"]),
        "checkpoint_bytes": int(Path(result["checkpoint_path"]).stat().st_size),
    }
    _validate_json_output(task, "checkpoint_manifest", checkpoint_manifest)
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _write_json_exclusive(checkpoint_manifest_path, checkpoint_manifest)
    wiring_path = output / "PRODUCTION_WIRING.json"
    _write_json_exclusive(
        wiring_path,
        {
            "variant_id": task["variant_id"],
            "model": v8_model.model_wiring_receipt(model),
            "input": v8_inputs.variant_input_policy(task["variant_id"]),
            "loss": v8_loss.variant_loss_policy(task["variant_id"]),
            "optimizer_parameter_names": result["optimizer_manifest"]["optimizer_parameter_names"],
            "frozen_or_unused_parameter_names": result["optimizer_manifest"]["frozen_or_unused_parameter_names"],
        },
    )
    history_path = output / "TRAINING_HISTORY.json"
    _write_json_exclusive(history_path, {"history": result["history"]})
    complete = {
        "schema_version": "revision0710.fig04_fig05.training_complete.v8",
        "status": "TRAINING_COMPLETE_CAPTURE_SEALED",
        "run_id": task["run_id"],
        "variant_id": task["variant_id"],
        "outer_fold": task["outer_fold"],
        "seed": task["seed"],
        "completed_epochs": result["completed_epochs"],
        "capture_epoch": result["capture_epoch"],
        "checkpoint_path": result["checkpoint_path"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "checkpoint_manifest_sha256": v8_contract.sha256_file(checkpoint_manifest_path),
        "model_optimizer_manifest_sha256": v8_contract.sha256_file(model_manifest_path),
        "production_wiring_sha256": v8_contract.sha256_file(wiring_path),
        "training_history_sha256": v8_contract.sha256_file(history_path),
        "training_trajectory_attestation_path": str(trajectory_path.resolve()),
        "training_trajectory_attestation_sha256": v8_contract.sha256_file(trajectory_path),
        "task_sha256": v8_contract.sha256_file(Path(task["task_path"])),
        "claim_sha256": task["claim_sha256"],
        "forward_called": result["forward_called"],
        "backward_called": result["backward_called"],
        "gradient_clip_called": result["gradient_clip_called"],
        "scheduler_called": result["scheduler_called"],
        "amp_path_entered": result["amp_path_entered"],
        "fresh_initialization": result["fresh_initialization"],
        "resume_used": result["resume_used"],
        "opened_roles": ["train"],
        "opened_paths": _closed_opened_paths(closure_opened, opened),
        "calibration_read_count": 0,
        "outer_test_read_count": 0,
    }
    complete_path = output / "TRAINING_COMPLETE.json"
    _write_json_exclusive(complete_path, complete)
    return {**complete, "training_complete_sha256": v8_contract.sha256_file(complete_path)}


def run_calibration_task(task: dict[str, Any]) -> dict[str, Any]:
    validate_task(task, expected_phase="calibration")
    closure_opened = _task_claim_opened_paths(task)
    training_path = Path(task["training_complete_path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    protocol = task["protocol"]
    if training.get("status") != "TRAINING_COMPLETE_CAPTURE_SEALED" or int(training.get("completed_epochs", -1)) != int(protocol["train_through_epoch"]):
        raise WorkerError("calibration opened before training completion")
    if training.get("checkpoint_sha256") != v8_contract.sha256_file(Path(task["checkpoint_path"])):
        raise WorkerError("calibration checkpoint binding changed")
    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    bundle, code, text, opened = _load_source(task)
    model, _ = _restore_model(task, code, text)
    frame = v8_data.prediction_frame(
        model,
        bundle,
        variant_id=task["variant_id"],
        run_id=task["run_id"],
        evaluation_epoch=int(protocol["evaluate_only_epoch"]),
        device=torch.device(protocol["device"]),
        inference_batch_size=int(protocol["batch_size"]),
    )
    frame["split_role"] = "validation"
    correction = v8_data.fit_age_correction(frame)
    correction.update(
        {
            "run_id": task["run_id"],
            "evaluation_epoch": int(protocol["evaluate_only_epoch"]),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "participant_ids_sha256": v8_contract.canonical_json_sha256(
                {"participant_ids": sorted(bundle.participant_ids)}
            ),
        }
    )
    corrected = v8_data.apply_age_correction(frame, correction)
    nuisance = v8_data.fit_cox_nuisance(
        corrected,
        penalizer=0.1 if protocol["execution_mode"] == "scaled_integration" else 1e-4,
    )
    correction["fit_role"] = "validation"
    _validate_json_output(task, "age_correction", correction)
    correction_path = output / "age_correction.json"
    _write_json_exclusive(correction_path, correction)
    nuisance.update(
        {
            "run_id": task["run_id"],
            "evaluation_epoch": int(protocol["evaluate_only_epoch"]),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "age_correction_sha256": v8_contract.sha256_file(correction_path),
        }
    )
    nuisance["fit_role"] = "validation"
    _validate_json_output(task, "cox_nuisance", nuisance)
    nuisance_path = output / "cox_nuisance.json"
    _write_json_exclusive(nuisance_path, nuisance)
    corrected["checkpoint_sha256"] = training["checkpoint_sha256"]
    corrected["age_correction_sha256"] = v8_contract.sha256_file(correction_path)
    corrected["cox_nuisance_sha256"] = v8_contract.sha256_file(nuisance_path)
    corrected = _close_table_output(task, "prediction_rows", corrected)
    prediction_path = output / "validation_predictions.csv"
    corrected.to_csv(prediction_path, index=False, lineterminator="\n")
    complete = {
        "schema_version": "revision0710.fig04_fig05.calibration_complete.v8",
        "status": "CALIBRATION_NUISANCE_SEALED_NO_SELECTION",
        "run_id": task["run_id"],
        "training_complete_sha256": v8_contract.sha256_file(training_path),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "age_correction_path": str(correction_path.resolve()),
        "age_correction_sha256": v8_contract.sha256_file(correction_path),
        "cox_nuisance_path": str(nuisance_path.resolve()),
        "cox_nuisance_sha256": v8_contract.sha256_file(nuisance_path),
        "validation_predictions_sha256": v8_contract.sha256_file(prediction_path),
        "task_sha256": v8_contract.sha256_file(Path(task["task_path"])),
        "claim_sha256": task["claim_sha256"],
        "model_selection_used": False,
        "opened_roles": ["calibration"],
        "opened_paths": _closed_opened_paths(
            closure_opened,
            [str(training_path.resolve()), str(Path(task["checkpoint_path"]).resolve())],
            opened,
        ),
        "outer_test_read_count": 0,
    }
    complete_path = output / "CALIBRATION_COMPLETE.json"
    _write_json_exclusive(complete_path, complete)
    return {**complete, "calibration_complete_sha256": v8_contract.sha256_file(complete_path)}


def run_outer_task(task: dict[str, Any]) -> dict[str, Any]:
    validate_task(task, expected_phase="outer_test")
    closure_opened = _task_claim_opened_paths(task)
    ledger_path = Path(task["outer_read_ledger_path"])
    if not ledger_path.is_file():
        raise WorkerError("outer read ledger is absent")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("status") != "COMMITTED_BEFORE_OUTER_OPEN" or ledger.get("run_id") != task["run_id"]:
        raise WorkerError("outer read ledger is invalid")
    _validate_json_output(task, "outer_read_ledger", ledger)
    training_path = Path(task["training_complete_path"])
    calibration_path = Path(task["calibration_complete_path"])
    training = json.loads(training_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    bindings = {
        "training_complete_sha256": v8_contract.sha256_file(training_path),
        "checkpoint_sha256": v8_contract.sha256_file(Path(task["checkpoint_path"])),
        "age_correction_sha256": v8_contract.sha256_file(Path(task["age_correction_path"])),
        "cox_nuisance_sha256": v8_contract.sha256_file(Path(task["cox_nuisance_path"])),
    }
    for key, value in bindings.items():
        if ledger.get(key) != value:
            raise WorkerError(f"outer ledger binding changed: {key}")
    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    # The outer role is opened only after every ledger check above succeeds.
    bundle, code, text, opened = _load_source(task)
    model, _ = _restore_model(task, code, text)
    protocol = task["protocol"]
    frame = v8_data.prediction_frame(
        model,
        bundle,
        variant_id=task["variant_id"],
        run_id=task["run_id"],
        evaluation_epoch=int(protocol["evaluate_only_epoch"]),
        device=torch.device(protocol["device"]),
        inference_batch_size=int(protocol["batch_size"]),
    )
    frame["split_role"] = "test"
    correction = json.loads(Path(task["age_correction_path"]).read_text(encoding="utf-8"))
    nuisance = json.loads(Path(task["cox_nuisance_path"]).read_text(encoding="utf-8"))
    corrected = v8_data.apply_age_correction(frame, correction)
    fold_values, organ_values = v8_data.score_outer_metrics(corrected, nuisance)
    checkpoint_sha = training["checkpoint_sha256"]
    correction_sha = calibration["age_correction_sha256"]
    nuisance_sha = calibration["cox_nuisance_sha256"]
    ledger_sha = v8_contract.sha256_file(ledger_path)
    corrected["checkpoint_sha256"] = checkpoint_sha
    corrected["age_correction_sha256"] = correction_sha
    corrected["cox_nuisance_sha256"] = nuisance_sha
    corrected = _close_table_output(task, "prediction_rows", corrected)
    prediction_path = output / "prediction_rows.csv"
    corrected.to_csv(prediction_path, index=False, lineterminator="\n")
    prediction_sha = v8_contract.sha256_file(prediction_path)
    fold_row = {
        "run_id": task["run_id"],
        "variant_id": task["variant_id"],
        "outer_fold": int(task["outer_fold"]),
        "evaluation_epoch": int(protocol["evaluate_only_epoch"]),
        **fold_values,
        "completed": True,
        "performance_selected": False,
        "checkpoint_sha256": checkpoint_sha,
        "prediction_rows_sha256": prediction_sha,
        "age_correction_sha256": correction_sha,
        "cox_nuisance_sha256": nuisance_sha,
        "outer_read_ledger_sha256": ledger_sha,
    }
    fold_path = output / "fold_metrics.csv"
    fold_frame = _close_table_output(
        task, "fold_metrics", pd.DataFrame([fold_row])
    )
    fold_frame.to_csv(fold_path, index=False, lineterminator="\n")
    organ_rows = [
        {
            "run_id": task["run_id"],
            "variant_id": task["variant_id"],
            "outer_fold": int(task["outer_fold"]),
            "evaluation_epoch": int(protocol["evaluate_only_epoch"]),
            **row,
            "completed": True,
            "performance_selected": False,
        }
        for row in organ_values
    ]
    organ_path = output / "organ_metrics.csv"
    organ_frame = _close_table_output(
        task, "organ_metrics", pd.DataFrame(organ_rows)
    )
    organ_frame.to_csv(organ_path, index=False, lineterminator="\n")
    receipt = {
        "schema_version": "revision0710.fig04_fig05.fold_receipt.v8",
        "status": "COMPLETE_E55_ONLY_NO_SELECTION" if int(protocol["evaluate_only_epoch"]) == 55 else "COMPLETE_SCALED_CAPTURE_ONLY_NO_SELECTION",
        "run_id": task["run_id"],
        "variant_id": task["variant_id"],
        "outer_fold": int(task["outer_fold"]),
        "evaluation_epoch": int(protocol["evaluate_only_epoch"]),
        "checkpoint_manifest_sha256": training["checkpoint_manifest_sha256"],
        "prediction_rows_sha256": prediction_sha,
        "age_correction_sha256": correction_sha,
        "cox_nuisance_sha256": nuisance_sha,
        "outer_read_ledger_sha256": ledger_sha,
        "fold_metrics_sha256": v8_contract.sha256_file(fold_path),
        "organ_metrics_sha256": v8_contract.sha256_file(organ_path),
        "task_sha256": v8_contract.sha256_file(Path(task["task_path"])),
        "claim_sha256": task["claim_sha256"],
        "completed": True,
        "performance_selected": False,
        "opened_roles": ["outer_test"],
        "opened_paths": _closed_opened_paths(
            closure_opened,
            [
                str(ledger_path.resolve()),
                str(training_path.resolve()),
                str(calibration_path.resolve()),
                str(Path(task["checkpoint_path"]).resolve()),
                str(Path(task["age_correction_path"]).resolve()),
                str(Path(task["cox_nuisance_path"]).resolve()),
            ],
            opened,
        ),
        "outer_refit_calls": 0,
    }
    _validate_json_output(task, "fold_receipt", receipt)
    receipt_path = output / "FOLD_RECEIPT.json"
    _write_json_exclusive(receipt_path, receipt)
    return {**receipt, "fold_receipt_sha256": v8_contract.sha256_file(receipt_path)}


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    phase = task.get("phase")
    if phase == "train":
        return run_train_task(task)
    if phase == "calibration":
        return run_calibration_task(task)
    if phase == "outer_test":
        return run_outer_task(task)
    raise WorkerError("unknown worker phase")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if task.get("protocol", {}).get("execution_mode") == "formal":
        if os.environ.get("REVISION0710_WORKER_ACCESS_GUARD_ACTIVE") != "1":
            raise WorkerError("formal worker must run inside the access-audited subprocess guard")
        expected_audit = (
            Path(task["output_dir"]).resolve().parent
            / "00_worker_access_audit"
            / f"{task['phase']}.json"
        )
        if Path(os.environ.get("REVISION0710_WORKER_ACCESS_AUDIT_PATH", "")).resolve() != expected_audit:
            raise WorkerError("formal worker access-audit path changed")
    result = run_task(task)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
