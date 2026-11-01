from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import isolated_worker
import v8_authorization
import v8_contract
import v8_schema_validator
import v8_state_machine
import v8_runtime_guard
import worker_access_guard


ROOT = Path(__file__).resolve().parent
ALLOWED_MODES = (
    "validate-only",
    "formal-run",
    "formal-aggregate",
    "scaled-integration-smoke",
)


class ExecutorError(RuntimeError):
    pass


class ReplayPersistenceError(ExecutorError):
    pass


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ExecutorError(f"create-only executor artifact exists: {path}") from exc


def _acquire_single_gpu_lock(formal_root: Path, *, run_id: str):
    return v8_runtime_guard.acquire_global_gpu_lock(formal_root, run_id=run_id)


def _release_single_gpu_lock(handle, lock_path: Path, binding: dict[str, Any]) -> None:
    v8_runtime_guard.release_global_gpu_lock(handle, lock_path, binding)


def observed_formal_environment() -> dict[str, Any]:
    return v8_runtime_guard.observed_formal_environment()


def validate_formal_environment(contract: dict[str, Any]) -> dict[str, Any]:
    return v8_runtime_guard.validate_formal_environment(contract["environment"]["formal"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=ALLOWED_MODES)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    return parser


def authorize_formal_execution(
    root: Path, *, authorization_path: Path | None
) -> dict[str, Any]:
    return v8_authorization.load_and_validate_authorization(authorization_path, root)


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


def _role_record(contract: dict[str, Any], outer: int, role: str) -> dict[str, Any]:
    name = f"outer_fold_{outer}_{role}_manifest"
    matches = [row for row in contract["canonical_bindings"]["records"] if row["name"] == name]
    if len(matches) != 1:
        raise ExecutorError(f"missing exact role binding: {name}")
    return matches[0]


def _data_source(contract: dict[str, Any], outer: int, role: str) -> dict[str, Any]:
    record = _role_record(contract, outer, role)
    return {
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


def _authorization_binding(path: Path) -> dict[str, Any]:
    return {
        "required": True,
        "path": str(path.resolve()),
        "sha256": v8_contract.sha256_file(path),
    }


def _activate_authorization(
    root: Path, authorization_path: Path, authorization: dict[str, Any]
) -> Path:
    formal_root = Path(authorization["formal_run_root"])
    receipt = {
        "schema_version": "revision0710.fig04_fig05.authorization_activation.v8",
        "status": "EXACT_60_AUTHORIZATION_ACTIVATED_BEFORE_RUNS",
        "authorization_sha256": v8_contract.sha256_file(authorization_path),
        "contract_sha256": authorization["hash_bindings"]["contract_sha256"],
        "terminal_go_sha256": authorization["independent_terminal_go"]["sha256"],
        "terminal_decision_state_sha256": authorization[
            "terminal_decision_state_sha256"
        ],
        "authorized_run_ids_sha256": authorization["authorized_run_ids_sha256"],
        "authorized_run_count": 60,
    }
    receipt_path = formal_root / "AUTHORIZATION_ACTIVATION_RECEIPT.json"
    if formal_root.exists():
        if not receipt_path.is_file() or v8_contract.read_json_exact(receipt_path) != receipt:
            raise ExecutorError("formal root activation receipt changed")
        return receipt_path
    formal_root.mkdir(parents=True, exist_ok=False)
    try:
        _write_json_exclusive(receipt_path, receipt)
    except Exception:
        formal_root.rmdir()
        raise
    return receipt_path


def _common_task(
    *,
    phase: str,
    row: dict[str, Any],
    output_dir: Path,
    data_source: dict[str, Any],
    authorization_path: Path,
    gpu_lock: dict[str, Any],
) -> dict[str, Any]:
    role = "outer_test" if phase == "outer_test" else phase
    attempt = output_dir.resolve().parent
    task_name = {
        "train": "TRAIN_TASK.json",
        "calibration": "CALIBRATION_TASK.json",
        "outer_test": "OUTER_TASK.json",
    }[phase]
    claim_path = attempt / "CLAIM.json"
    return {
        "schema_version": isolated_worker.TASK_SCHEMA,
        "phase": phase,
        "role": role,
        "run_id": row["run_id"],
        "variant_id": row["variant_id"],
        "outer_fold": row["outer_fold"],
        "seed": row["seed"],
        "protocol": _formal_protocol(),
        "data_source": data_source,
        "output_dir": str(output_dir.resolve()),
        "contract_path": str((ROOT / "FROZEN_ABLATION_CONTRACT_V8.json").resolve()),
        "contract_sha256": v8_contract.sha256_file(ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"),
        "task_path": str((attempt / task_name).resolve()),
        "claim_path": str(claim_path.resolve()),
        "claim_sha256": (
            v8_contract.sha256_file(claim_path) if claim_path.is_file() else "0" * 64
        ),
        "authorization": _authorization_binding(authorization_path),
        "gpu_lock": gpu_lock,
    }


def _worker_access_audit_path(task_path: Path) -> Path:
    task = v8_contract.read_json_exact(task_path)
    return task_path.parent / "00_worker_access_audit" / f"{task['phase']}.json"


def _validate_worker_access_audit(task_path: Path) -> dict[str, Any]:
    audit_path = _worker_access_audit_path(task_path)
    if not audit_path.is_file():
        raise ExecutorError("isolated worker access audit is absent")
    audit = v8_contract.read_json_exact(audit_path)
    task = v8_contract.read_json_exact(task_path)
    required = {
        "schema_version", "status", "run_id", "phase", "worker_pid",
        "executor_pid", "process_isolated",
        "python_audit_hook_active_before_worker_import", "task_path",
        "task_sha256", "allowed_files_sha256", "actual_sensitive_accesses",
        "actual_sensitive_access_count", "denied_sensitive_accesses",
        "worker_exit_code", "worker_error",
    }
    if set(audit) != required:
        raise ExecutorError("isolated worker access audit keys changed")
    if not (
        audit["schema_version"] == worker_access_guard.SCHEMA
        and audit["status"] == "PASS_NO_DENIED_SENSITIVE_ACCESS"
        and audit["run_id"] == task["run_id"]
        and audit["phase"] == task["phase"]
        and int(audit["executor_pid"]) == os.getpid()
        and int(audit["worker_pid"]) != os.getpid()
        and audit["process_isolated"] is True
        and audit["python_audit_hook_active_before_worker_import"] is True
        and Path(audit["task_path"]).resolve() == task_path.resolve()
        and audit["task_sha256"] == v8_contract.sha256_file(task_path)
        and int(audit["actual_sensitive_access_count"]) > 0
        and audit["denied_sensitive_accesses"] == []
        and int(audit["worker_exit_code"]) == 0
        and audit["worker_error"] is None
    ):
        raise ExecutorError("isolated worker access audit did not pass")
    return {
        "path": str(audit_path.resolve()),
        "bytes": audit_path.stat().st_size,
        "sha256": v8_contract.sha256_file(audit_path),
    }


def _invoke_worker(task_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str((ROOT / "worker_access_guard.py").resolve()),
            "--task",
            str(task_path.resolve()),
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ExecutorError(
            f"isolated worker failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutorError("isolated worker output is not JSON") from exc
    result["worker_access_audit"] = _validate_worker_access_audit(task_path)
    return result


def _invoke_independent_formal_verifier(
    attempt: Path, *, run_id: str
) -> dict[str, Any]:
    verifier = ROOT / "verify_formal_attempt_v8_independent.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(verifier.resolve()),
            "--attempt",
            str(attempt.resolve()),
            "--run-id",
            run_id,
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ExecutorError(
            "independent formal replay failed "
            f"({completed.returncode}): {completed.stderr[-2000:]}"
        )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutorError("independent formal replay output is not JSON") from exc
    if not (
        receipt.get("status")
        == "PASS_SEED7210_FULL_TRAINING_AND_VALIDATION_OUTER_REPLAY"
        and receipt.get("run_id") == run_id
        and receipt.get("production_deep_validator_imported") is False
    ):
        raise ExecutorError("independent formal replay receipt changed")
    return receipt


def _attempt_access_audit_records(attempt: Path) -> list[dict[str, Any]]:
    records = []
    for phase, task_name in (
        ("train", "TRAIN_TASK.json"),
        ("calibration", "CALIBRATION_TASK.json"),
        ("outer_test", "OUTER_TASK.json"),
    ):
        task_path = attempt / task_name
        audit_path = attempt / "00_worker_access_audit" / f"{phase}.json"
        if not task_path.is_file() or not audit_path.is_file():
            raise ExecutorError(f"worker access audit closure is absent: {phase}")
        task = v8_contract.read_json_exact(task_path)
        audit = v8_contract.read_json_exact(audit_path)
        allowed, expected_path = worker_access_guard.allowed_files_for_task(task_path, task)
        if expected_path.resolve() != audit_path.resolve():
            raise ExecutorError("worker access audit path closure changed")
        if not (
            audit.get("schema_version") == worker_access_guard.SCHEMA
            and audit.get("status") == "PASS_NO_DENIED_SENSITIVE_ACCESS"
            and audit.get("run_id") == task["run_id"]
            and audit.get("phase") == phase
            and int(audit.get("worker_pid", -1)) > 0
            and int(audit.get("executor_pid", -1)) > 0
            and int(audit["worker_pid"]) != int(audit["executor_pid"])
            and audit.get("process_isolated") is True
            and audit.get("python_audit_hook_active_before_worker_import") is True
            and audit.get("task_sha256") == v8_contract.sha256_file(task_path)
            and audit.get("allowed_files_sha256")
            == worker_access_guard._canonical_sha(sorted(str(path) for path in allowed))
            and int(audit.get("actual_sensitive_access_count", 0)) > 0
            and audit.get("denied_sensitive_accesses") == []
            and int(audit.get("worker_exit_code", -1)) == 0
            and audit.get("worker_error") is None
        ):
            raise ExecutorError(f"worker access audit is invalid: {phase}")
        records.append(
            {
                "phase": phase,
                "path": str(audit_path.resolve()),
                "bytes": audit_path.stat().st_size,
                "sha256": v8_contract.sha256_file(audit_path),
            }
        )
    return records


def _persistent_independent_replay(
    *,
    formal_root: Path,
    attempt: Path,
    run_id: str,
    validated_bundle: dict[str, Any],
) -> dict[str, Any]:
    replay_root = formal_root / "00_independent_replays"
    replay_root.mkdir(parents=False, exist_ok=True)
    path = replay_root / f"{run_id}.json"
    attempt_binding = {
        "attempt_path": str(attempt.resolve()),
        "complete_sha256": v8_contract.sha256_file(attempt / "COMPLETE.json"),
        "artifact_count": int(validated_bundle["artifact_count"]),
        "artifact_set_sha256": validated_bundle["artifact_set_sha256"],
        "worker_access_audit_set_sha256": v8_contract.canonical_json_sha256(
            _attempt_access_audit_records(attempt)
        ),
    }
    recomputed_replay = _invoke_independent_formal_verifier(attempt, run_id=run_id)
    if path.exists():
        wrapper = v8_contract.read_json_exact(path)
        if not (
            set(wrapper)
            == {
                "schema_version", "status", "run_id", "attempt_binding",
                "independent_replay", "independent_replay_sha256",
            }
            and wrapper.get("schema_version")
            == "revision0710.fig04_fig05.persistent_independent_replay.v8"
            and wrapper.get("status")
            == "PASS_CREATE_ONLY_REPLAY_PRIMARY_RECOMPUTED_ON_REUSE"
            and wrapper.get("run_id") == run_id
            and wrapper.get("attempt_binding") == attempt_binding
            and wrapper.get("independent_replay_sha256")
            == v8_contract.canonical_json_sha256(wrapper.get("independent_replay"))
        ):
            raise ReplayPersistenceError(
                f"persistent independent replay changed: {run_id}"
            )
        replay = wrapper["independent_replay"]
        if replay != recomputed_replay:
            raise ReplayPersistenceError(
                f"persistent replay differs from primary-output recomputation: {run_id}"
            )
    else:
        replay = recomputed_replay
        wrapper = {
            "schema_version": "revision0710.fig04_fig05.persistent_independent_replay.v8",
            "status": "PASS_CREATE_ONLY_REPLAY_PRIMARY_RECOMPUTED_ON_REUSE",
            "run_id": run_id,
            "attempt_binding": attempt_binding,
            "independent_replay": replay,
            "independent_replay_sha256": v8_contract.canonical_json_sha256(replay),
        }
        _write_json_exclusive(path, wrapper)
    if not (
        replay.get("status")
        == "PASS_SEED7210_FULL_TRAINING_AND_VALIDATION_OUTER_REPLAY"
        and replay.get("run_id") == run_id
        and replay.get("production_deep_validator_imported") is False
    ):
        raise ReplayPersistenceError(f"persistent replay payload changed: {run_id}")
    return {
        "run_id": run_id,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": v8_contract.sha256_file(path),
        "attempt_binding": attempt_binding,
        "independent_replay_sha256": wrapper["independent_replay_sha256"],
        "independent_replay": replay,
        "primary_outputs_recomputed_on_this_validation": True,
    }


def validate_production_shape_receipt(shape: dict[str, Any]) -> None:
    if not (
        shape.get("schema_version")
        == "revision0710.fig04_fig05.cuda_production_shape_preflight.v8"
        and shape.get("status")
        == "PASS_12_OF_12_BATCH2048_PRODUCTION_SHAPE_AMP_STEPS"
        and shape.get("contract_sha256")
        == v8_contract.sha256_file(ROOT / "FROZEN_ABLATION_CONTRACT_V8.json")
        and shape.get("variant_ids") == list(v8_contract.VARIANT_IDS)
        and shape.get("shape_contract", {}).get("batch_size") == 2048
        and shape.get("shape_contract", {}).get("sequence_length") == 32
        and shape.get("shape_contract", {}).get("phenotype_features") == 78
        and shape.get("shape_contract", {}).get("code_embedding_shape") == [9906, 512]
        and shape.get("shape_contract", {}).get("text_embedding_shape") == [44565, 512]
        and shape.get("participant_labels_opened") is False
        and shape.get("calibration_role_opened") is False
        and shape.get("outer_role_opened") is False
        and shape.get("formal_protocol_covers_same_shape") is True
        and shape.get("oom_policy") == "HARD_FAIL_NO_BATCH_OR_SEQUENCE_DOWNGRADE"
        and shape.get("oom_observed") is False
        and shape.get("formal_authorization_used") is False
        and shape.get("formal_output_used") is False
    ):
        raise ExecutorError("production-shape CUDA preflight changed")


def formal_resource_preflight(root: Path) -> dict[str, Any]:
    contract = v8_contract.load_contract(root)
    environment = validate_formal_environment(contract)
    resource = v8_contract.read_json_exact(root / "RESOURCE_BUDGET_AND_PREFLIGHT_V8.json")
    formal_root = Path(contract["execution_roots"]["formal_run_root"])
    disk = shutil.disk_usage(formal_root.anchor)
    if disk.free < int(resource["preflight_thresholds"]["minimum_free_disk_bytes"]):
        raise ExecutorError("formal preflight free disk is below threshold")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.free,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        raise ExecutorError("formal preflight nvidia-smi failed")
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ExecutorError("single-GPU policy requires exactly one visible GPU")
    name, free_mib, total_mib, temperature = [value.strip() for value in rows[0].split(",")]
    thresholds = resource["preflight_thresholds"]
    if name != resource["single_gpu_policy"]["gpu_model"]:
        raise ExecutorError("formal preflight GPU model changed")
    if int(free_mib) < int(thresholds["minimum_free_gpu_memory_mib"]):
        raise ExecutorError("formal preflight free GPU memory is below threshold")
    if int(temperature) > int(thresholds["maximum_gpu_temperature_c"]):
        raise ExecutorError("formal preflight GPU temperature is above threshold")
    shape_path = (
        root
        / "smoke"
        / "production_shape_v8_final"
        / "CUDA_PRODUCTION_SHAPE_PREFLIGHT_RECEIPT.json"
    )
    if not shape_path.is_file():
        raise ExecutorError("bound production-shape CUDA preflight is absent")
    shape = v8_contract.read_json_exact(shape_path)
    validate_production_shape_receipt(shape)
    return {
        "status": "PASS",
        "gpu_name": name,
        "gpu_free_mib": int(free_mib),
        "gpu_total_mib": int(total_mib),
        "gpu_temperature_c": int(temperature),
        "disk_free_bytes": int(disk.free),
        "max_concurrent_training_runs": 1,
        "environment": environment,
        "production_shape_preflight_sha256": v8_contract.sha256_file(shape_path),
        "production_shape_batch_size": 2048,
        "production_shape_sequence_length": 32,
    }


def execute_formal_run(
    root: Path,
    *,
    run_id: str,
    authorization_path: Path | None,
) -> dict[str, Any]:
    authorization = authorize_formal_execution(root, authorization_path=authorization_path)
    if run_id not in authorization["authorized_run_ids"]:
        raise ExecutorError("requested run is not in exact authorization")
    preflight = formal_resource_preflight(root)
    if authorization_path is None:
        raise ExecutorError("formal authorization path is required")
    _activate_authorization(root, authorization_path, authorization)
    contract = v8_contract.load_contract(root)
    rows = v8_contract.read_queue(root / "ABLATION_QUEUE_V8.csv")
    row = next(item for item in rows if item["run_id"] == run_id)
    run_root = Path(row["intended_run_dir"])
    lock_handle, lock_path, gpu_lock = _acquire_single_gpu_lock(
        Path(authorization["formal_run_root"]), run_id=run_id
    )
    attempt: Path | None = None
    try:
        attempt = v8_state_machine.claim_new_attempt(run_root, run_id=run_id, seed=7210)
        train_task = _common_task(
            phase="train",
            row=row,
            output_dir=attempt / "01_train",
            data_source=_data_source(contract, row["outer_fold"], "train"),
            authorization_path=authorization_path,
            gpu_lock=gpu_lock,
        )
        train_task_path = attempt / "TRAIN_TASK.json"
        _write_json_exclusive(train_task_path, train_task)
        train_result = _invoke_worker(train_task_path)
        training_path = attempt / "01_train" / "TRAINING_COMPLETE.json"
        checkpoint_path = Path(train_result["checkpoint_path"])

        calibration_task = _common_task(
            phase="calibration",
            row=row,
            output_dir=attempt / "02_calibration",
            data_source=_data_source(contract, row["outer_fold"], "calibration"),
            authorization_path=authorization_path,
            gpu_lock=gpu_lock,
        )
        calibration_task.update(
            {
                "training_complete_path": str(training_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
            }
        )
        calibration_task_path = attempt / "CALIBRATION_TASK.json"
        _write_json_exclusive(calibration_task_path, calibration_task)
        calibration_result = _invoke_worker(calibration_task_path)
        calibration_complete_path = attempt / "02_calibration" / "CALIBRATION_COMPLETE.json"

        outer_record = _role_record(contract, row["outer_fold"], "outer_test")
        ledger = {
            "schema_version": "revision0710.fig04_fig05.outer_read_ledger.v8",
            "status": "COMMITTED_BEFORE_OUTER_OPEN",
            "run_id": run_id,
            "checkpoint_sha256": train_result["checkpoint_sha256"],
            "training_complete_sha256": v8_contract.sha256_file(training_path),
            "age_correction_sha256": calibration_result["age_correction_sha256"],
            "cox_nuisance_sha256": calibration_result["cox_nuisance_sha256"],
            "outer_role_manifest_sha256": outer_record["sha256"],
            "outer_opened_before_commit": False,
        }
        ledger_path = attempt / "OUTER_READ_LEDGER.json"
        _write_json_exclusive(ledger_path, ledger)

        outer_task = _common_task(
            phase="outer_test",
            row=row,
            output_dir=attempt / "03_outer_test",
            data_source=_data_source(contract, row["outer_fold"], "outer_test"),
            authorization_path=authorization_path,
            gpu_lock=gpu_lock,
        )
        outer_task.update(
            {
                "training_complete_path": str(training_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "calibration_complete_path": str(calibration_complete_path.resolve()),
                "age_correction_path": calibration_result["age_correction_path"],
                "cox_nuisance_path": calibration_result["cox_nuisance_path"],
                "outer_read_ledger_path": str(ledger_path.resolve()),
            }
        )
        outer_task_path = attempt / "OUTER_TASK.json"
        _write_json_exclusive(outer_task_path, outer_task)
        outer_result = _invoke_worker(outer_task_path)
        v8_state_machine.write_completion_receipt(
            attempt,
            run_id=run_id,
            seed=7210,
            fold_receipt_path=attempt / "03_outer_test" / "FOLD_RECEIPT.json",
        )
        return {
            "status": "FORMAL_RUN_COMPLETE",
            "run_id": run_id,
            "attempt": str(attempt.resolve()),
            "fold_receipt_sha256": outer_result["fold_receipt_sha256"],
            "preflight": preflight,
        }
    except Exception as exc:
        if attempt is not None:
            v8_state_machine.write_failure_receipt(
                attempt, phase="executor", reason=f"{type(exc).__name__}: {exc}"
            )
        raise
    finally:
        _release_single_gpu_lock(lock_handle, lock_path, gpu_lock)


def _execute_formal_aggregate_under_lock(
    root: Path, *, authorization_path: Path | None
) -> dict[str, Any]:
    authorization = authorize_formal_execution(
        root, authorization_path=authorization_path
    )
    if authorization_path is None:
        raise ExecutorError("formal authorization path is required")
    _activate_authorization(root, authorization_path, authorization)
    contract = v8_contract.load_contract(root)
    rows = v8_contract.read_queue(root / "ABLATION_QUEUE_V8.csv")
    schemas = v8_schema_validator.load_schemas(root / "OUTPUT_SCHEMAS_V8.json")
    receipts: dict[str, str] = {}
    artifact_sets: dict[str, dict[str, Any]] = {}
    independent_replays: dict[str, dict[str, Any]] = {}
    formal_root = Path(contract["execution_roots"]["formal_run_root"])
    for row in rows:
        run_root = Path(row["intended_run_dir"])
        candidates = sorted(run_root.glob("attempt_[0-9][0-9][0-9][0-9]"))
        validated_bundle: dict[str, Any] | None = None
        validation_errors: list[str] = []
        for attempt in candidates:
            complete_path = attempt / "COMPLETE.json"
            if not complete_path.is_file() or (attempt / "FAILURE.json").exists():
                continue
            complete = v8_contract.read_json_exact(complete_path)
            expected_complete_keys = {
                "schema_version", "status", "run_id", "seed",
                "attempt_ordinal", "performance_metric_used_for_selection",
            }
            if (
                set(complete) != expected_complete_keys
                or complete.get("schema_version") != "revision0710.fig04_fig05.attempt_complete.v8"
                or complete.get("status") != "COMPLETE"
                or complete.get("run_id") != row["run_id"]
                or int(complete.get("seed", -1)) != 7210
                or int(complete.get("attempt_ordinal", -1)) != int(attempt.name[-4:])
                or complete.get("performance_metric_used_for_selection") is not False
            ):
                validation_errors.append(f"{attempt.name}: completion receipt invalid")
                continue
            try:
                validated_bundle = v8_schema_validator.validate_formal_attempt_bundle(
                    attempt,
                    queue_row=row,
                    contract=contract,
                    root=root,
                )
                access_audits = _attempt_access_audit_records(attempt)
                independent_replay = _persistent_independent_replay(
                    formal_root=formal_root,
                    attempt=attempt,
                    run_id=row["run_id"],
                    validated_bundle=validated_bundle,
                )
            except ReplayPersistenceError:
                raise
            except Exception as exc:
                validated_bundle = None
                validation_errors.append(f"{attempt.name}: {type(exc).__name__}: {exc}")
                continue
            break
        if validated_bundle is None:
            detail = "; ".join(validation_errors[-3:])
            raise ExecutorError(f"no deeply valid completed attempt: {row['run_id']}: {detail}")
        receipts[row["run_id"]] = validated_bundle["fold_receipt_sha256"]
        artifact_sets[row["run_id"]] = {
            "attempt": str(attempt.resolve()),
            "artifact_count": validated_bundle["artifact_count"],
            "artifact_set_sha256": validated_bundle["artifact_set_sha256"],
            "artifacts": validated_bundle["artifacts"],
            "worker_access_audits": access_audits,
            "worker_access_audit_set_sha256": v8_contract.canonical_json_sha256(
                access_audits
            ),
        }
        independent_replays[row["run_id"]] = independent_replay
    v8_state_machine.validate_complete_set_for_aggregate(receipts)
    if set(artifact_sets) != set(v8_contract.expected_run_ids()):
        raise ExecutorError("aggregate artifact set is not complete 60/60")
    artifact_index = {
        "schema_version": "revision0710.fig04_fig05.aggregate_artifact_index.v8",
        "status": "DEEP_VALIDATED_60_OF_60",
        "run_count": 60,
        "artifacts_per_run": 22,
        "artifact_count": 1320,
        "runs": [
            {"run_id": run_id, **artifact_sets[run_id]}
            for run_id in v8_contract.expected_run_ids()
        ],
    }
    if any(row["artifact_count"] != 22 for row in artifact_index["runs"]):
        raise ExecutorError("aggregate per-run artifact cardinality changed")
    artifact_index_bytes = (
        json.dumps(artifact_index, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    artifact_index_sha = hashlib.sha256(artifact_index_bytes).hexdigest()
    independent_replay_index = {
        "schema_version": "revision0710.fig04_fig05.independent_replay_index.v8",
        "status": "PASS_60_OF_60_INDEPENDENT_FULL_TRAINING_DUAL_ROLE_REPLAYS",
        "run_count": 60,
        "persistent_create_only_receipts": True,
        "resume_scope": "independent_verification_only_training_is_never_resumed",
        "runs": [independent_replays[run_id] for run_id in v8_contract.expected_run_ids()],
    }
    independent_replay_bytes = (
        json.dumps(independent_replay_index, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    independent_replay_sha = hashlib.sha256(independent_replay_bytes).hexdigest()
    aggregate = {
        "schema_version": "revision0710.fig04_fig05.global_aggregate.v8",
        "status": "COMPLETE_60_OF_60",
        "run_count": 60,
        "run_ids_sha256": v8_contract.canonical_json_sha256(
            list(v8_contract.expected_run_ids())
        ),
        "fold_receipt_set_sha256": v8_contract.canonical_json_sha256(receipts),
        "artifact_index_sha256": artifact_index_sha,
        "independent_replay_index_sha256": independent_replay_sha,
        "worker_access_audit_index_sha256": v8_contract.canonical_json_sha256(
            {
                run_id: artifact_sets[run_id]["worker_access_audit_set_sha256"]
                for run_id in v8_contract.expected_run_ids()
            }
        ),
        "artifact_set_sha256": v8_contract.canonical_json_sha256(
            {
                run_id: artifact_sets[run_id]["artifact_set_sha256"]
                for run_id in v8_contract.expected_run_ids()
            }
        ),
        "artifacts_per_run": 22,
        "artifact_count": 1320,
        "all_completed": True,
        "performance_selected": False,
        "source_gate_open": True,
        "independent_training_replays_complete": True,
        "persistent_replay_receipts_complete": True,
    }
    v8_schema_validator.validate_json_payload(
        aggregate, schemas["tables"]["global_aggregate"], table="global_aggregate"
    )
    aggregate_root = Path(contract["execution_roots"]["formal_run_root"]) / "00_aggregate"
    aggregate_root.mkdir(parents=False, exist_ok=False)
    artifact_index_path = aggregate_root / "AGGREGATE_ARTIFACT_INDEX.json"
    _write_json_exclusive(artifact_index_path, artifact_index)
    if v8_contract.sha256_file(artifact_index_path) != artifact_index_sha:
        raise ExecutorError("aggregate artifact index serialization changed")
    independent_replay_path = aggregate_root / "INDEPENDENT_REPLAY_INDEX.json"
    _write_json_exclusive(independent_replay_path, independent_replay_index)
    if v8_contract.sha256_file(independent_replay_path) != independent_replay_sha:
        raise ExecutorError("independent replay index serialization changed")
    aggregate_path = aggregate_root / "GLOBAL_AGGREGATE.json"
    _write_json_exclusive(aggregate_path, aggregate)
    return {
        "status": "FORMAL_AGGREGATE_COMPLETE_SOURCE_GATE_OPEN",
        "global_aggregate_path": str(aggregate_path.resolve()),
        "global_aggregate_sha256": v8_contract.sha256_file(aggregate_path),
        "independent_replay_index_path": str(independent_replay_path.resolve()),
        "independent_replay_index_sha256": independent_replay_sha,
        "artifact_index_path": str(artifact_index_path.resolve()),
        "artifact_index_sha256": artifact_index_sha,
        "run_count": 60,
    }


def execute_formal_aggregate(
    root: Path, *, authorization_path: Path | None
) -> dict[str, Any]:
    authorization = authorize_formal_execution(
        root, authorization_path=authorization_path
    )
    if authorization_path is None:
        raise ExecutorError("formal authorization path is required")
    _activate_authorization(root, authorization_path, authorization)
    formal_resource_preflight(root)
    handle, lock_path, binding = _acquire_single_gpu_lock(
        Path(authorization["formal_run_root"]), run_id="__aggregate__"
    )
    try:
        return _execute_formal_aggregate_under_lock(
            root, authorization_path=authorization_path
        )
    finally:
        _release_single_gpu_lock(handle, lock_path, binding)


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "validate-only":
        result = v8_contract.validate_prefreeze_bundle(ROOT, verify_hashes=True)
    elif args.mode == "formal-run":
        if not args.run_id:
            raise ExecutorError("--run-id is required")
        result = execute_formal_run(
            ROOT, run_id=args.run_id, authorization_path=args.authorization
        )
    elif args.mode == "formal-aggregate":
        result = execute_formal_aggregate(
            ROOT, authorization_path=args.authorization
        )
    else:
        if args.output_dir is None:
            raise ExecutorError("--output-dir is required")
        from run_scaled_integration_smoke import run_scaled_integration_smoke

        result = run_scaled_integration_smoke(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
