from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import formal_executor
import isolated_worker
import v8_authorization
import v8_contract
import v8_runtime_guard
import worker_access_guard
import v8_schema_validator
import verify_formal_attempt_v8_independent


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[3]


def test_v7_no_go_cannot_be_used_as_v8_terminal_go(tmp_path: Path) -> None:
    shutil.copy2(ROOT / "TERMINAL_GO_SCHEMA_V8.json", tmp_path / "TERMINAL_GO_SCHEMA_V8.json")
    shutil.copy2(
        ROOT / "V7_TERMINAL_REVIEW_NO_GO.md",
        tmp_path / "INDEPENDENT_TERMINAL_REVIEW_GO_V8.json",
    )
    with pytest.raises(v8_authorization.AuthorizationError):
        v8_authorization.load_and_validate_terminal_go(tmp_path)


def test_current_v8_no_go_and_candidate_go_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    shutil.copy2(
        ROOT / "TERMINAL_GO_SCHEMA_V8.json",
        tmp_path / "TERMINAL_GO_SCHEMA_V8.json",
    )
    schema = v8_contract.read_json_exact(tmp_path / "TERMINAL_GO_SCHEMA_V8.json")
    candidate = {key: None for key in schema["receipt_exact_keys"]}
    for key in ("schema_version", "status", "decision", "create_only"):
        candidate[key] = schema["required_values"][key]
    candidate["created_utc"] = "2026-07-11T00:00:00Z"
    candidate["reviewer"] = {
        "identity": "independent-test-reviewer",
        "review_tool": "pytest",
        "independent_from_implementation": True,
        "signed_statement": "structurally complete candidate GO",
    }
    (tmp_path / v8_authorization.GO_FILE).write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "PRIMARY_SELF_REVIEW_NO-GO_V8.md").write_text(
        "Verdict: NO-GO\n", encoding="utf-8"
    )
    with pytest.raises(v8_authorization.AuthorizationError, match="conflicts"):
        v8_authorization.load_and_validate_terminal_go(tmp_path)


def test_moved_terminal_go_is_treated_as_absent(tmp_path: Path) -> None:
    shutil.copy2(ROOT / "TERMINAL_GO_SCHEMA_V8.json", tmp_path / "TERMINAL_GO_SCHEMA_V8.json")
    (tmp_path / "MOVED_GO.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(v8_authorization.AuthorizationError, match="absent"):
        v8_authorization.load_and_validate_terminal_go(tmp_path)


def test_terminal_go_synchronized_evidence_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copy2(ROOT / "TERMINAL_GO_SCHEMA_V8.json", tmp_path / "TERMINAL_GO_SCHEMA_V8.json")
    for name in (
        "FROZEN_ABLATION_CONTRACT_V8.json",
        "ABLATION_QUEUE_V8.csv",
        "OUTPUT_SCHEMAS_V8.json",
        "V7_TERMINAL_REVIEW_NO_GO.md",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    evidence = {
        name: {"path": relative, "bytes": 1, "sha256": "b" * 64}
        for name, relative in v8_authorization.EVIDENCE_FILES.items()
    }
    go = {
        "schema_version": "revision0710.fig04_fig05.independent_terminal_go.v8",
        "status": "GO_EXACT_FROZEN_V8_FORMAL_EXECUTION",
        "decision": "GO",
        "created_utc": "2026-07-11T00:00:00Z",
        "reviewer": {
            "identity": "independent-test-reviewer",
            "review_tool": "pytest",
            "independent_from_implementation": True,
            "signed_statement": "synthetic unit-test receipt",
        },
        "contract_sha256": v8_contract.sha256_file(tmp_path / "FROZEN_ABLATION_CONTRACT_V8.json"),
        "artifact_bundle_sha256": v8_contract.canonical_json_sha256([]),
        "queue_sha256": v8_contract.sha256_file(tmp_path / "ABLATION_QUEUE_V8.csv"),
        "output_schemas_sha256": v8_contract.sha256_file(tmp_path / "OUTPUT_SCHEMAS_V8.json"),
        "cache_artifact_bindings_sha256": v8_contract.cache_artifact_bindings_sha256({}),
        "test_bundle_sha256": v8_contract.canonical_json_sha256([]),
        "evidence_bindings": {**evidence, "fresh_test_result": {**evidence["fresh_test_result"], "sha256": "0" * 64}},
        "evidence_bindings_sha256": "0" * 64,
        "superseded_v7_no_go": {
            "path": "V7_TERMINAL_REVIEW_NO_GO.md",
            "bytes": (tmp_path / "V7_TERMINAL_REVIEW_NO_GO.md").stat().st_size,
            "sha256": v8_contract.sha256_file(tmp_path / "V7_TERMINAL_REVIEW_NO_GO.md"),
        },
        "current_v8_no_go_evidence": [],
        "authorization_scope": {
            "authorized_run_count": 60,
            "seed": 7210,
            "train_through_epoch": 60,
            "capture_epoch": 55,
            "performance_conditioned": False,
        },
        "create_only": True,
    }
    (tmp_path / v8_authorization.GO_FILE).write_text(
        json.dumps(go, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        v8_authorization.v8_contract,
        "load_contract",
        lambda root: {"artifact_bindings": {"records": []}, "cache_artifact_bindings": {}},
    )
    monkeypatch.setattr(v8_authorization, "_tests_sha256", lambda root: v8_contract.canonical_json_sha256([]))
    monkeypatch.setattr(v8_authorization, "expected_evidence_bindings", lambda root: evidence)
    with pytest.raises(v8_authorization.AuthorizationError, match="evidence_bindings"):
        v8_authorization.load_and_validate_terminal_go(tmp_path)


def test_hard_kill_stale_gpu_lock_is_recovered_and_audited(tmp_path: Path) -> None:
    script = (
        "import os,sys; from pathlib import Path; "
        f"sys.path.insert(0,{str(ROOT)!r}); import v8_runtime_guard as g; "
        "g.acquire_global_gpu_lock(Path(sys.argv[1]),run_id='killed'); os._exit(0)"
    )
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert completed.returncode == 0
    handle, state_path, binding = v8_runtime_guard.acquire_global_gpu_lock(
        tmp_path, run_id="recovered"
    )
    try:
        state = v8_contract.read_json_exact(state_path)
        events = [row["event"] for row in state["events"]]
        assert "RECOVERED_STALE_ACTIVE_STATE_AFTER_OS_LOCK_BECAME_FREE" in events
        assert state["status"] == "ACTIVE"
        assert state["active"]["run_id"] == "recovered"
    finally:
        v8_runtime_guard.release_global_gpu_lock(handle, state_path, binding)
    assert v8_contract.read_json_exact(state_path)["status"] == "RELEASED"


def test_persistent_replay_is_recomputed_and_resealed_scientific_tamper_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    attempt = tmp_path / "attempt_0001"
    attempt.mkdir()
    (attempt / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    audits = [
        {"phase": phase, "path": phase, "bytes": 1, "sha256": "a" * 64}
        for phase in ("train", "calibration", "outer_test")
    ]
    monkeypatch.setattr(formal_executor, "_attempt_access_audit_records", lambda value: audits)
    calls = 0

    def replay(value: Path, *, run_id: str) -> dict:
        nonlocal calls
        calls += 1
        return {
            "schema_version": "revision0710.fig04_fig05.formal_independent_replay.v8",
            "status": "PASS_SEED7210_FULL_TRAINING_AND_VALIDATION_OUTER_REPLAY",
            "run_id": run_id,
            "checkpoint_sha256": "c" * 64,
            "roles": {
                "validation": {
                    "participants": 16,
                    "rows": 112,
                    "prediction_file_sha256": "d" * 64,
                },
                "outer": {
                    "participants": 16,
                    "rows": 112,
                    "prediction_file_sha256": "e" * 64,
                },
            },
            "production_deep_validator_imported": False,
        }

    monkeypatch.setattr(formal_executor, "_invoke_independent_formal_verifier", replay)
    bundle = {"artifact_count": 22, "artifact_set_sha256": "d" * 64}
    first = formal_executor._persistent_independent_replay(
        formal_root=formal_root, attempt=attempt, run_id="run0", validated_bundle=bundle
    )
    second = formal_executor._persistent_independent_replay(
        formal_root=formal_root, attempt=attempt, run_id="run0", validated_bundle=bundle
    )
    assert calls == 2 and first["sha256"] == second["sha256"]
    assert second["primary_outputs_recomputed_on_this_validation"] is True
    path = Path(first["path"])
    payload = v8_contract.read_json_exact(path)
    payload["independent_replay"]["roles"]["outer"]["rows"] = 999
    payload["independent_replay"]["checkpoint_sha256"] = "0" * 64
    payload["independent_replay_sha256"] = v8_contract.canonical_json_sha256(
        payload["independent_replay"]
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(
        formal_executor.ReplayPersistenceError, match="primary-output recomputation"
    ):
        formal_executor._persistent_independent_replay(
            formal_root=formal_root,
            attempt=attempt,
            run_id="run0",
            validated_bundle=bundle,
        )
    assert calls == 3


def test_actual_gpu_lock_binding_schema_is_shared_by_all_path_consumers(
    tmp_path: Path,
) -> None:
    handle, state_path, binding = v8_runtime_guard.acquire_global_gpu_lock(
        tmp_path, run_id="schema_test"
    )
    attempt = tmp_path / "attempt_0001"
    train_dir = attempt / "01_train"
    train_dir.mkdir(parents=True)
    claim_path = attempt / "CLAIM.json"
    claim = {
        "run_id": "schema_test",
        "seed": 7210,
        "selection_uses_metrics": False,
    }
    claim_path.write_text(json.dumps(claim) + "\n", encoding="utf-8")
    task_path = attempt / "TRAIN_TASK.json"
    task = {
        "task_path": str(task_path.resolve()),
        "claim_path": str(claim_path.resolve()),
        "claim_sha256": v8_contract.sha256_file(claim_path),
        "run_id": "schema_test",
        "seed": 7210,
        "phase": "train",
        "output_dir": str(train_dir.resolve()),
        "protocol": {"execution_mode": "formal"},
        "contract_path": str((ROOT / "FROZEN_ABLATION_CONTRACT_V8.json").resolve()),
        "authorization": {"required": True, "path": str((tmp_path / "AUTH.json").resolve())},
        "gpu_lock": binding,
        "data_source": {
            "format": "synthetic_role_bundle",
            "role_bundle_path": str((tmp_path / "ROLE.pt").resolve()),
        },
    }
    task_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        expected = {str(path) for path in v8_runtime_guard.lock_binding_paths(binding)}
        worker_paths = set(isolated_worker._task_claim_opened_paths(task))
        schema_paths = set(
            v8_schema_validator.expected_task_opened_paths(
                task, root=ROOT, phase_inputs=[]
            )
        )
        guarded, _ = worker_access_guard.allowed_files_for_task(task_path, task)
        guarded_paths = {str(path) for path in guarded}
        assert expected <= worker_paths
        assert expected <= schema_paths
        assert expected <= guarded_paths
        assert "lock_binding_paths" in inspect.getsource(
            verify_formal_attempt_v8_independent._expected_opened
        )
        for source in (
            inspect.getsource(isolated_worker._task_claim_opened_paths),
            inspect.getsource(v8_schema_validator.expected_task_opened_paths),
            inspect.getsource(verify_formal_attempt_v8_independent._expected_opened),
        ):
            assert '["gpu_lock"]["path"]' not in source
    finally:
        v8_runtime_guard.release_global_gpu_lock(handle, state_path, binding)


def test_direct_formal_worker_without_access_guard_is_rejected(tmp_path: Path) -> None:
    task = {
        "protocol": {"execution_mode": "formal"},
        "output_dir": str((tmp_path / "attempt" / "01_train").resolve()),
        "phase": "train",
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    env = dict(os.environ)
    env.pop("REVISION0710_WORKER_ACCESS_GUARD_ACTIVE", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "isolated_worker.py"), "--task", str(task_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "access-audited subprocess guard" in completed.stderr


def test_worker_access_guard_denies_unlisted_sensitive_path(tmp_path: Path) -> None:
    hook = worker_access_guard.AuditHook(
        allowed_files=set(),
        allowed_write_root=tmp_path / "attempt",
        audit_path=tmp_path / "attempt" / "audit.json",
    )
    forbidden = (
        REPO_ROOT
        / "Revision-0710"
        / "01_data_rebuild"
        / "04_cache_v1"
        / "targets"
        / "chronological_age_targets.parquet"
    )
    with pytest.raises(PermissionError, match="denied read"):
        hook("open", (str(forbidden), "r", os.O_RDONLY))
    assert hook.denied and hook.denied[0]["permitted"] is False


def test_formal_worker_rehash_is_role_limited_after_executor_full_rehash() -> None:
    binding_source = inspect.getsource(isolated_worker._validate_formal_task_bindings)
    task_source = inspect.getsource(isolated_worker.validate_task)
    assert "verify_hashes=False" in binding_source
    assert "verify_contract_file_hashes=False" in task_source
    assert "only the current role's selected cache files" in task_source


def test_canonical_control_estimands_and_budget_are_frozen() -> None:
    registry = v8_contract.read_json_exact(ROOT / "VARIANT_REGISTRY_V8.json")
    estimands = registry["comparison_estimands"]
    assert estimands["formal_comparator"] == "fig04_01_canonical_control"
    assert estimands["bundled_architecture_comparisons"][
        "fig04_07_seven_independent_organ_age_heads"
    ]["single_component_effect_claim_allowed"] is False
    resource = v8_contract.read_json_exact(ROOT / "RESOURCE_BUDGET_AND_PREFLIGHT_V8.json")
    raw = resource["estimate"]["raw_combined_training_and_replay_gpu_hours"]
    headroom = resource["estimate"]["evaluation_and_operational_headroom_fraction"]
    assert raw * (1.0 + headroom) == pytest.approx(89.935619359652, abs=1e-12)
    assert resource["estimate"]["estimated_total_gpu_hours"] == pytest.approx(
        89.9356193597, abs=1e-10
    )


def test_production_shape_downgrade_is_rejected() -> None:
    receipt = {
        "schema_version": "revision0710.fig04_fig05.cuda_production_shape_preflight.v8",
        "status": "PASS_12_OF_12_BATCH2048_PRODUCTION_SHAPE_AMP_STEPS",
        "variant_ids": list(v8_contract.VARIANT_IDS),
        "shape_contract": {
            "batch_size": 8,
            "sequence_length": 6,
            "phenotype_features": 78,
            "code_embedding_shape": [64, 8],
            "text_embedding_shape": [64, 8],
        },
        "participant_labels_opened": False,
        "calibration_role_opened": False,
        "outer_role_opened": False,
        "formal_protocol_covers_same_shape": True,
        "formal_authorization_used": False,
        "formal_output_used": False,
    }
    with pytest.raises(formal_executor.ExecutorError, match="production-shape"):
        formal_executor.validate_production_shape_receipt(receipt)
