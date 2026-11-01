from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import v8_contract


class AuthorizationError(RuntimeError):
    pass


HASH_FILES = {
    "contract_sha256": "FROZEN_ABLATION_CONTRACT_V8.json",
    "variant_registry_sha256": "VARIANT_REGISTRY_V8.json",
    "queue_sha256": "ABLATION_QUEUE_V8.csv",
    "output_schemas_sha256": "OUTPUT_SCHEMAS_V8.json",
    "executor_sha256": "formal_executor.py",
    "worker_sha256": "isolated_worker.py",
    "worker_access_guard_sha256": "worker_access_guard.py",
    "model_sha256": "v8_model.py",
    "loss_sha256": "v8_loss.py",
    "input_sha256": "v8_inputs.py",
    "data_sha256": "v8_data.py",
    "schema_validator_sha256": "v8_schema_validator.py",
    "training_sha256": "v8_training.py",
    "independent_replay_sha256": "v8_independent_replay.py",
    "formal_independent_verifier_sha256": "verify_formal_attempt_v8_independent.py",
    "contract_validator_sha256": "v8_contract.py",
    "runtime_guard_sha256": "v8_runtime_guard.py",
    "v1_model_runtime_sha256": "../fig04_fig05_ablation_prefreeze_v1/ablation_model.py",
    "v1_loss_runtime_sha256": "../fig04_fig05_ablation_prefreeze_v1/ablation_loss.py",
    "v1_input_runtime_sha256": "../fig04_fig05_ablation_prefreeze_v1/ablation_inputs.py",
}

EVIDENCE_FILES = {
    "prefreeze_validation": "PREFREEZE_VALIDATION_RECEIPT_V8.json",
    "independent_mechanical_verification": "INDEPENDENT_VERIFICATION_RECEIPT_V8.json",
    "cuda_production_shape_preflight": (
        "smoke/production_shape_v8_final/"
        "CUDA_PRODUCTION_SHAPE_PREFLIGHT_RECEIPT.json"
    ),
    "resource_budget_and_preflight": "RESOURCE_BUDGET_AND_PREFLIGHT_V8.json",
    "fresh_test_result": "FRESH_TEST_RESULT_RECEIPT_V8.json",
    "scaled_integration_manifest": "smoke/scaled_integration_v8_final/MANIFEST.json",
    "smoke_run_index": "SMOKE_RUN_INDEX_V8.json",
}

GO_FILE = "INDEPENDENT_TERMINAL_REVIEW_GO_V8.json"
GO_SCHEMA_FILE = "TERMINAL_GO_SCHEMA_V8.json"
SUPERSEDED_NO_GO_FILE = "V7_TERMINAL_REVIEW_NO_GO.md"


def _tests_sha256(root: Path) -> str:
    records = []
    for path in sorted((root / "tests").glob("test_*.py")):
        records.append(
            {
                "name": path.name,
                "sha256": v8_contract.sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return v8_contract.canonical_json_sha256(records)


def _record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise AuthorizationError(f"required authorization evidence is absent: {relative}")
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": v8_contract.sha256_file(path),
    }


def expected_evidence_bindings(root: Path) -> dict[str, dict[str, Any]]:
    return {name: _record(root, relative) for name, relative in EVIDENCE_FILES.items()}


def current_v8_no_go_evidence(root: Path) -> list[dict[str, Any]]:
    """Discover every current-version NO-GO decision before accepting a GO."""

    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda value: value.name.casefold()):
        normalized = path.name.upper().replace("-", "_")
        if path.is_file() and "V8" in normalized and "NO_GO" in normalized:
            records.append(_record(root, path.name))
    return records


def require_go_only_terminal_state(root: Path) -> None:
    conflicts = current_v8_no_go_evidence(root)
    if conflicts:
        names = ", ".join(record["path"] for record in conflicts)
        raise AuthorizationError(
            f"current V8 NO-GO evidence conflicts with terminal GO: {names}"
        )


def expected_terminal_decision_state(
    root: Path, terminal_go_record: dict[str, Any]
) -> dict[str, Any]:
    require_go_only_terminal_state(root)
    state = {
        "schema_version": "revision0710.fig04_fig05.terminal_decision_state.v8",
        "status": "GO_ONLY_NO_CURRENT_V8_NO_GO",
        "terminal_go": terminal_go_record,
        "current_v8_no_go_evidence": [],
        "superseded_v7_no_go": _record(root, SUPERSEDED_NO_GO_FILE),
    }
    return state


def _parse_created_utc(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AuthorizationError("terminal GO created_utc is not canonical UTC")
    try:
        dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AuthorizationError("terminal GO created_utc is invalid") from exc


def load_and_validate_terminal_go(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require_go_only_terminal_state(root)
    go_path = root / GO_FILE
    if not go_path.is_file():
        raise AuthorizationError("exact independent terminal GO receipt is absent")
    try:
        schema = v8_contract.read_json_exact(root / GO_SCHEMA_FILE)
        go = v8_contract.read_json_exact(go_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AuthorizationError("independent terminal GO is not valid JSON") from exc
    if set(go) != set(schema["receipt_exact_keys"]):
        raise AuthorizationError("independent terminal GO exact keys changed")
    required = schema["required_values"]
    for key in ("schema_version", "status", "decision", "create_only"):
        if go.get(key) != required[key]:
            raise AuthorizationError(f"independent terminal GO mismatch: {key}")
    _parse_created_utc(go.get("created_utc"))
    reviewer = go.get("reviewer")
    if not isinstance(reviewer, dict) or set(reviewer) != set(schema["reviewer_exact_keys"]):
        raise AuthorizationError("terminal GO reviewer identity keys changed")
    if reviewer.get("independent_from_implementation") is not True:
        raise AuthorizationError("terminal GO reviewer is not independent")
    for key in ("identity", "review_tool", "signed_statement"):
        if not isinstance(reviewer.get(key), str) or not reviewer[key].strip():
            raise AuthorizationError(f"terminal GO reviewer field is empty: {key}")

    contract = v8_contract.load_contract(root)
    tests_sha = _tests_sha256(root)
    expected_evidence = expected_evidence_bindings(root)
    expected = {
        "contract_sha256": v8_contract.sha256_file(root / "FROZEN_ABLATION_CONTRACT_V8.json"),
        "artifact_bundle_sha256": v8_contract.canonical_json_sha256(
            contract["artifact_bindings"]["records"]
        ),
        "queue_sha256": v8_contract.sha256_file(root / "ABLATION_QUEUE_V8.csv"),
        "output_schemas_sha256": v8_contract.sha256_file(root / "OUTPUT_SCHEMAS_V8.json"),
        "cache_artifact_bindings_sha256": v8_contract.cache_artifact_bindings_sha256(
            contract["cache_artifact_bindings"]
        ),
        "test_bundle_sha256": tests_sha,
        "evidence_bindings": expected_evidence,
        "evidence_bindings_sha256": v8_contract.canonical_json_sha256(expected_evidence),
        "superseded_v7_no_go": _record(root, SUPERSEDED_NO_GO_FILE),
        "current_v8_no_go_evidence": [],
        "authorization_scope": {
            "authorized_run_count": 60,
            "seed": 7210,
            "train_through_epoch": 60,
            "capture_epoch": 55,
            "performance_conditioned": False,
        },
    }
    for key, value in expected.items():
        if go.get(key) != value:
            raise AuthorizationError(f"independent terminal GO binding mismatch: {key}")
    return go, _record(root, GO_FILE)


def expected_authorization_payload(root: Path) -> dict[str, Any]:
    terminal_go, terminal_go_record = load_and_validate_terminal_go(root)
    terminal_decision_state = expected_terminal_decision_state(root, terminal_go_record)
    contract = v8_contract.load_contract(root)
    run_ids = list(v8_contract.expected_run_ids())
    bindings = {
        key: v8_contract.sha256_file(root / relative)
        for key, relative in HASH_FILES.items()
    }
    bindings["test_bundle_sha256"] = _tests_sha256(root)
    cache_bindings = json.loads(json.dumps(contract["cache_artifact_bindings"]))
    bindings["cache_artifact_bindings_sha256"] = (
        v8_contract.cache_artifact_bindings_sha256(cache_bindings)
    )
    evidence = expected_evidence_bindings(root)
    return {
        "schema_version": "revision0710.fig04_fig05.formal_authorization.v8",
        "status": "FORMAL_EXECUTION_AUTHORIZED_EXACT_60",
        "authorization_granted": True,
        "created_utc": terminal_go["created_utc"],
        "seed": 7210,
        "train_through_epoch": 60,
        "capture_epoch": 55,
        "evaluate_only_epoch": 55,
        "formal_environment": contract["environment"]["formal"],
        "formal_run_root": contract["execution_roots"]["formal_run_root"],
        "hash_bindings": bindings,
        "cache_artifact_bindings": cache_bindings,
        "evidence_bindings": evidence,
        "evidence_bindings_sha256": v8_contract.canonical_json_sha256(evidence),
        "independent_terminal_go": terminal_go_record,
        "terminal_decision_state": terminal_decision_state,
        "terminal_decision_state_sha256": v8_contract.canonical_json_sha256(
            terminal_decision_state
        ),
        "authorized_run_ids": run_ids,
        "authorized_run_ids_sha256": v8_contract.canonical_json_sha256(run_ids),
        "authorized_run_count": 60,
        "all_60_runs_authorized_together": True,
        "single_gpu_policy": {
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "max_concurrent_training_runs": 1,
            "distributed_training": False,
        },
        "all_formal_targets_absent_at_authorization": True,
        "performance_conditioned_authorization": False,
        "micro_benchmark_authorization": False,
    }


def _validate_execution_targets(
    payload: dict[str, Any],
    root: Path,
    *,
    authorization_sha256: str | None,
) -> None:
    contract = v8_contract.load_contract(root)
    formal_root = Path(contract["execution_roots"]["formal_run_root"])
    activation_path = formal_root / "AUTHORIZATION_ACTIVATION_RECEIPT.json"
    if formal_root.exists():
        if authorization_sha256 is None or not activation_path.is_file():
            raise AuthorizationError("formal root exists without exact authorization activation")
        activation = v8_contract.read_json_exact(activation_path)
        expected_activation = {
            "schema_version": "revision0710.fig04_fig05.authorization_activation.v8",
            "status": "EXACT_60_AUTHORIZATION_ACTIVATED_BEFORE_RUNS",
            "authorization_sha256": authorization_sha256,
            "contract_sha256": payload["hash_bindings"]["contract_sha256"],
            "terminal_go_sha256": payload["independent_terminal_go"]["sha256"],
            "terminal_decision_state_sha256": payload[
                "terminal_decision_state_sha256"
            ],
            "authorized_run_ids_sha256": payload["authorized_run_ids_sha256"],
            "authorized_run_count": 60,
        }
        if activation != expected_activation:
            raise AuthorizationError("formal authorization activation receipt changed")
    for target in contract["publication_firewall"]["formal_targets_that_must_not_exist"]:
        path = Path(target)
        path = path if path.is_absolute() else v8_contract.repo_root(root) / path
        if path.resolve() == formal_root.resolve():
            continue
        if path.exists():
            raise AuthorizationError(f"formal publication target already exists: {path}")


def validate_authorization_payload(
    payload: dict[str, Any],
    root: Path,
    *,
    authorization_sha256: str | None = None,
    verify_contract_file_hashes: bool = True,
) -> None:
    contract = v8_contract.load_contract(root)
    v8_contract.validate_contract_payload(
        contract,
        root,
        verify_hashes=verify_contract_file_hashes,
        enforce_prefreeze_absence=False,
    )
    v8_contract.validate_queue(
        v8_contract.read_queue(root / "ABLATION_QUEUE_V8.csv"), contract
    )
    expected = expected_authorization_payload(root)
    if set(payload) != set(expected):
        raise AuthorizationError("authorization exact keys changed")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AuthorizationError(f"authorization mismatch: {key}")
    if payload["authorized_run_ids"] != list(v8_contract.expected_run_ids()):
        raise AuthorizationError("authorization run order/set changed")
    if payload["authorized_run_ids_sha256"] != v8_contract.canonical_json_sha256(
        payload["authorized_run_ids"]
    ):
        raise AuthorizationError("authorization run-id SHA changed")
    formal_root = Path(payload["formal_run_root"])
    if not formal_root.is_absolute() or formal_root.resolve() != Path(
        contract["execution_roots"]["formal_run_root"]
    ).resolve():
        raise AuthorizationError("authorization formal root changed")
    _validate_execution_targets(
        payload,
        root,
        authorization_sha256=authorization_sha256,
    )


def load_and_validate_authorization(
    path: Path | None,
    root: Path,
    *,
    verify_contract_file_hashes: bool = True,
) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise AuthorizationError("exact formal authorization is absent")
    expected_location = root / "FORMAL_AUTHORIZATION_V8.json"
    if path.resolve() != expected_location.resolve():
        raise AuthorizationError("authorization must use the frozen in-root path")
    payload = v8_contract.read_json_exact(path)
    validate_authorization_payload(
        payload,
        root,
        authorization_sha256=v8_contract.sha256_file(path),
        verify_contract_file_hashes=verify_contract_file_hashes,
    )
    return payload
