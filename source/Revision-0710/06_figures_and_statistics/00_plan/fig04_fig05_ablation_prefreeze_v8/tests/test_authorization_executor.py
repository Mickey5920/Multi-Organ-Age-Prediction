from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

import formal_executor
import isolated_worker
import v8_authorization as authorization
import v8_contract as contract
import v8_runtime_guard


ROOT = Path(__file__).resolve().parents[1]


def test_formal_cli_exists_but_absent_authorization_hard_fails() -> None:
    assert "formal-run" in formal_executor.ALLOWED_MODES
    assert "scaled-integration-smoke" in formal_executor.ALLOWED_MODES
    with pytest.raises(authorization.AuthorizationError, match="absent"):
        formal_executor.authorize_formal_execution(ROOT, authorization_path=None)


def test_worker_is_not_the_canonical_fixed_replication_worker() -> None:
    source = inspect.getsource(isolated_worker)
    assert "fixed_replication_train_worker" not in source
    assert "run_fold0_reuse" not in source
    assert "restore_fixed_checkpoint" not in source


def _mock_terminal_go(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        authorization,
        "load_and_validate_terminal_go",
        lambda root: (
            {"created_utc": "2026-07-11T00:00:00Z"},
            {"path": authorization.GO_FILE, "bytes": 1, "sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(
        authorization,
        "expected_evidence_bindings",
        lambda root: {name: {"path": value, "bytes": 1, "sha256": "b" * 64}
                      for name, value in authorization.EVIDENCE_FILES.items()},
    )


def test_exact_authorization_template_covers_all_hashes_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_terminal_go(monkeypatch)
    expected = authorization.expected_authorization_payload(ROOT)
    assert expected["authorization_granted"] is True
    assert expected["authorized_run_count"] == 60
    assert expected["authorized_run_ids"] == list(contract.expected_run_ids())
    assert len(expected["authorized_run_ids_sha256"]) == 64
    required = {
        "contract_sha256",
        "variant_registry_sha256",
        "queue_sha256",
        "output_schemas_sha256",
        "executor_sha256",
        "worker_sha256",
        "worker_access_guard_sha256",
        "model_sha256",
        "loss_sha256",
        "input_sha256",
        "data_sha256",
        "schema_validator_sha256",
        "training_sha256",
        "independent_replay_sha256",
        "formal_independent_verifier_sha256",
        "contract_validator_sha256",
        "runtime_guard_sha256",
        "v1_model_runtime_sha256",
        "v1_loss_runtime_sha256",
        "v1_input_runtime_sha256",
        "cache_artifact_bindings_sha256",
        "test_bundle_sha256",
    }
    assert set(expected["hash_bindings"]) == required
    assert expected["cache_artifact_bindings"]["shards"] == contract.load_contract(ROOT)["cache_artifact_bindings"]["shards"]
    authorization.validate_authorization_payload(expected, ROOT)


@pytest.mark.parametrize(
    "mutation",
    ("seed", "order", "root", "missing_run", "extra_run", "code_hash", "gpu_policy", "target_absence"),
)
def test_authorization_mutations_are_rejected(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_terminal_go(monkeypatch)
    payload = authorization.expected_authorization_payload(ROOT)
    broken = copy.deepcopy(payload)
    if mutation == "seed":
        broken["seed"] = 7211
    elif mutation == "order":
        broken["authorized_run_ids"][0], broken["authorized_run_ids"][1] = (
            broken["authorized_run_ids"][1],
            broken["authorized_run_ids"][0],
        )
    elif mutation == "root":
        broken["formal_run_root"] = str(ROOT / "wrong")
    elif mutation == "missing_run":
        broken["authorized_run_ids"].pop()
        broken["authorized_run_count"] = 59
    elif mutation == "extra_run":
        broken["authorized_run_ids"].append("extra")
        broken["authorized_run_count"] = 61
    elif mutation == "code_hash":
        broken["hash_bindings"]["worker_sha256"] = "0" * 64
    elif mutation == "gpu_policy":
        broken["single_gpu_policy"]["max_concurrent_training_runs"] = 2
    else:
        broken["all_formal_targets_absent_at_authorization"] = False
    with pytest.raises(authorization.AuthorizationError):
        authorization.validate_authorization_payload(broken, ROOT)


def test_train_task_schema_forbids_calibration_and_outer_paths() -> None:
    task = isolated_worker.synthetic_task_template("train")
    isolated_worker.validate_task(task, expected_phase="train")
    for forbidden in ("calibration_role_path", "outer_test_role_path", "outer_read_ledger_path"):
        broken = copy.deepcopy(task)
        broken[forbidden] = "forbidden"
        with pytest.raises(isolated_worker.WorkerError, match="keys"):
            isolated_worker.validate_task(broken, expected_phase="train")


def test_phase_task_schemas_are_physically_separate() -> None:
    train = isolated_worker.synthetic_task_template("train")
    calibration = isolated_worker.synthetic_task_template("calibration")
    outer = isolated_worker.synthetic_task_template("outer_test")
    assert train["role"] == "train" and "role_bundle_path" in train["data_source"]
    assert calibration["role"] == "calibration" and "training_complete_path" in calibration
    assert outer["role"] == "outer_test" and "outer_read_ledger_path" in outer
    assert (
        train["data_source"]["role_bundle_path"]
        != calibration["data_source"]["role_bundle_path"]
        != outer["data_source"]["role_bundle_path"]
    )


def test_formal_protocol_extra_field_is_rejected_before_data_open() -> None:
    task = isolated_worker.synthetic_task_template("train")
    task["protocol"] = formal_executor._formal_protocol()
    task["protocol"]["unreviewed_extra"] = True
    with pytest.raises(isolated_worker.WorkerError, match="protocol"):
        isolated_worker.validate_task(task, expected_phase="train")


def test_global_single_gpu_lock_is_atomic_and_releasable(tmp_path: Path) -> None:
    first_handle, lock_path, first_binding = v8_runtime_guard.acquire_global_gpu_lock(
        tmp_path, run_id="run_a"
    )
    with pytest.raises(v8_runtime_guard.RuntimeGuardError, match="OS lock"):
        v8_runtime_guard.acquire_global_gpu_lock(tmp_path, run_id="run_b")
    v8_runtime_guard.release_global_gpu_lock(first_handle, lock_path, first_binding)
    second_handle, second_path, second_binding = v8_runtime_guard.acquire_global_gpu_lock(
        tmp_path, run_id="run_b"
    )
    v8_runtime_guard.release_global_gpu_lock(second_handle, second_path, second_binding)
    state = contract.read_json_exact(second_path)
    assert second_path.exists()
    assert state["status"] == "RELEASED"
    assert state["acquisition_count"] == 2
    assert [row["event"] for row in state["events"]] == [
        "ACQUIRED_OS_BACKED_LOCK",
        "RELEASED_OS_BACKED_LOCK",
        "ACQUIRED_OS_BACKED_LOCK",
        "RELEASED_OS_BACKED_LOCK",
    ]
