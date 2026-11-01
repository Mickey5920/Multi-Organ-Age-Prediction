from __future__ import annotations

"""Run multiple workers over the unchanged frozen corrected-primary queue."""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import multiworker_queue as queue_runtime
import run_direct_5fold as frozen_runner
from contract_runtime import sha256_file, stop_requested


RUNNER_ROOT = Path(__file__).resolve().parent
RUN_ROOT = RUNNER_ROOT.parent
STATE_ROOT = frozen_runner.LOG_ROOT / "multiworker_v1"
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
SESSION_PATH = STATE_ROOT / "TWO_WORKER_SESSION.json"
SHARED_GPU_LOCK_ROOT = RUN_ROOT.parent / "_single_gpu_execution_lock"


def orchestration_evidence() -> dict[str, Any]:
    return {
        "orchestrator": str(Path(__file__).resolve()),
        "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        "queue_runtime": str(Path(queue_runtime.__file__).resolve()),
        "queue_runtime_sha256": sha256_file(Path(queue_runtime.__file__).resolve()),
        "frozen_run_one_source": str(Path(frozen_runner.__file__).resolve()),
        "frozen_run_one_source_sha256": sha256_file(Path(frozen_runner.__file__).resolve()),
        "queue_sha256": sha256_file(frozen_runner.QUEUE_PATH),
        "frozen_contract_sha256": sha256_file(frozen_runner.CONTRACT_PATH),
    }


def _aggregate(
    rows: Sequence[dict[str, str]],
    worker: queue_runtime.WorkerIdentity,
    *,
    message: str,
):
    return queue_runtime.aggregate_run_status(
        rows,
        state_root=STATE_ROOT,
        task_paths=frozen_runner.task_paths,
        worker=worker,
        status_path=frozen_runner.STATUS_PATH,
        status_csv_path=frozen_runner.STATUS_CSV_PATH,
        queue_sha256=sha256_file(frozen_runner.QUEUE_PATH),
        contract_sha256=sha256_file(frozen_runner.CONTRACT_PATH),
        stop_after_current_present=stop_requested(RUN_ROOT),
        message=message,
    )


def _best_effort_aggregate(
    rows: Sequence[dict[str, str]],
    worker: queue_runtime.WorkerIdentity,
    *,
    message: str,
) -> None:
    try:
        _aggregate(rows, worker, message=message)
    except Exception as exc:
        queue_runtime.log_worker_event(
            STATE_ROOT,
            worker,
            "AGGREGATE_ERROR",
            message=message,
            error=f"{type(exc).__name__}: {exc}",
        )


def _pending(row: dict[str, str]) -> bool:
    return queue_runtime.is_task_pending(
        row,
        state_root=STATE_ROOT,
        task_paths=frozen_runner.task_paths,
    )


def _row_for_claim(
    rows: Sequence[dict[str, str]], claim: queue_runtime.TaskClaim
) -> dict[str, str]:
    matches = [row for row in rows if row["run_id"] == claim.run_id]
    if len(matches) != 1:
        raise RuntimeError(f"Claim does not map to exactly one frozen row: {claim.run_id}")
    return matches[0]


def _all_terminal_or_blocked(table) -> bool:
    return not table["run_state"].isin(["PENDING", "RUNNING", "CLAIMED"]).any()


def _validate_coordinator_session(worker_id: str) -> tuple[dict[str, Any], str]:
    session_token = os.environ.get("REV0710_DUAL_SESSION_TOKEN", "")
    coordinator_pid_raw = os.environ.get("REV0710_DUAL_COORDINATOR_PID", "")
    session_path_raw = os.environ.get("REV0710_DUAL_SESSION_PATH", "")
    if not session_token or not coordinator_pid_raw or not session_path_raw:
        raise RuntimeError("Formal workers must be launched by launch_two_worker_session.py")
    if not session_token.isascii() or len(session_token) != 64:
        raise RuntimeError("Dual-worker session token is malformed")
    session_path = Path(session_path_raw).resolve()
    if session_path != SESSION_PATH.resolve() or not session_path.is_file():
        raise RuntimeError("Dual-worker session receipt path is invalid")
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    coordinator_pid = int(coordinator_pid_raw)
    if payload.get("status") != "ACTIVE":
        raise RuntimeError("Dual-worker coordinator session is not ACTIVE")
    if payload.get("session_token") != session_token:
        raise RuntimeError("Dual-worker session token differs from coordinator receipt")
    if int(payload.get("coordinator_pid", -1)) != coordinator_pid:
        raise RuntimeError("Dual-worker coordinator PID differs from session receipt")
    if worker_id not in payload.get("worker_ids", []):
        raise RuntimeError(f"Worker is not authorized by the session receipt: {worker_id}")
    hostname = str(payload.get("hostname", ""))
    if queue_runtime.pid_is_alive(coordinator_pid, hostname) is not True:
        raise RuntimeError("Dual-worker coordinator is not alive")
    binding = payload.get("gpu_lock_binding", {})
    lock_state_path = Path(str(binding.get("state_path", ""))).resolve()
    expected_state_path = SHARED_GPU_LOCK_ROOT / "SINGLE_GPU_FORMAL_RUN_STATE.json"
    if lock_state_path != expected_state_path.resolve() or not lock_state_path.is_file():
        raise RuntimeError("Shared GPU lock state path differs from the session receipt")
    lock_state = json.loads(lock_state_path.read_text(encoding="utf-8"))
    active = lock_state.get("active", {})
    if lock_state.get("status") != "ACTIVE":
        raise RuntimeError("Shared GPU OS lock state is not ACTIVE")
    for key in ("owner_pid", "run_id", "token", "acquisition_ordinal"):
        if active.get(key) != binding.get(key):
            raise RuntimeError(f"Shared GPU lock binding changed: {key}")
    return payload, session_token


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-id",
        default=f"worker-{socket.gethostname()}-{os.getpid()}",
        help="Stable operator label; each process also receives a unique session ID.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--continuous-queue",
        action="store_true",
        help="Wait for peer-owned tasks and keep claiming until the queue is terminal.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen inputs and orchestration code without CUDA or output writes.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Refresh shared status under its lock without opening CUDA or claiming a task.",
    )
    parser.add_argument(
        "--continue-after-fold0-gate",
        action="store_true",
        help="Continue only after the completed canonical Fold0 metrics were reviewed.",
    )
    args = parser.parse_args()
    queue_runtime.validate_worker_id(args.worker_id)
    if args.max_runs is not None and args.max_runs < 0:
        raise ValueError("--max-runs must be non-negative")
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0:
        raise ValueError("poll and heartbeat intervals must be positive")

    contract, rows = frozen_runner.launch_preflight()
    evidence = orchestration_evidence()
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "queue_rows": len(rows),
                    "completed_before_launch": int(
                        sum(
                            frozen_runner.task_paths(row)["complete"].is_file()
                            for row in rows
                        )
                    ),
                    "scientific_contract_unchanged": True,
                    "run_one_unchanged": True,
                    "evidence": evidence,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    worker = queue_runtime.create_worker_identity(args.worker_id)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    queue_runtime.log_worker_event(
        STATE_ROOT,
        worker,
        "WORKER_STARTED",
        continuous_queue=bool(args.continuous_queue),
        max_runs=args.max_runs,
        evidence=evidence,
    )
    queue_runtime.write_worker_status(
        STATE_ROOT,
        worker,
        state="STARTING",
        active_run_id=None,
        message="Frozen contract preflight passed",
        evidence=evidence,
    )

    if args.aggregate_only:
        table = _aggregate(rows, worker, message="Shared queue status aggregated without training")
        queue_runtime.write_worker_status(
            STATE_ROOT,
            worker,
            state="EXITED",
            active_run_id=None,
            message="Aggregate-only process exited normally",
            evidence=evidence,
        )
        print(table["run_state"].value_counts().to_json())
        return 0
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 in the parent launcher before starting each worker"
        )
    if not frozen_runner.torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal corrected-primary Fig4/Fig5 queue")

    gate_row = next(row for row in rows if row["run_id"] == frozen_runner.FOLD0_GATE_RUN_ID)
    gate_complete_at_start = frozen_runner.task_paths(gate_row)["complete"].is_file()
    if gate_complete_at_start and not args.continue_after_fold0_gate:
        if not frozen_runner.FOLD0_GATE_JSON.is_file():
            frozen_runner.write_fold0_gate(gate_row)
        _aggregate(rows, worker, message="Canonical Fold0 complete; awaiting metric review")
        queue_runtime.write_worker_status(
            STATE_ROOT,
            worker,
            state="PAUSED_GATE",
            active_run_id=None,
            message="Relaunch with --continue-after-fold0-gate",
            evidence=evidence,
        )
        return 0

    _, session_token = _validate_coordinator_session(args.worker_id)
    slot = queue_runtime.acquire_worker_slot(
        STATE_ROOT, worker, session_token=session_token
    )

    eligible_rows: Sequence[dict[str, str]] = rows if gate_complete_at_start else [gate_row]
    completed_by_worker = 0
    exit_state = "EXITED"
    exit_message = "Worker exited normally"
    try:
        _aggregate(rows, worker, message=f"Worker {worker.worker_id} initialized")
        while True:
            _validate_coordinator_session(args.worker_id)
            if args.max_runs is not None and completed_by_worker >= args.max_runs:
                exit_message = f"Worker reached --max-runs={args.max_runs}"
                break
            if stop_requested(RUN_ROOT):
                exit_state = "STOPPED"
                exit_message = "STOP_AFTER_CURRENT observed; no new task claimed"
                break

            claim = queue_runtime.try_claim_next(
                STATE_ROOT,
                eligible_rows,
                worker,
                is_pending=_pending,
                evidence=evidence,
            )
            if claim is None:
                table = _aggregate(rows, worker, message=f"Worker {worker.worker_id} has no claim")
                if not gate_complete_at_start and frozen_runner.task_paths(gate_row)["complete"].is_file():
                    if not frozen_runner.FOLD0_GATE_JSON.is_file():
                        frozen_runner.write_fold0_gate(gate_row)
                    exit_state = "PAUSED_GATE"
                    exit_message = "Canonical Fold0 completed; automatic review pause"
                    break
                pending_or_active = table["run_state"].isin(
                    ["PENDING", "RUNNING", "CLAIMED"]
                ).any()
                if args.continuous_queue and pending_or_active:
                    queue_runtime.write_worker_status(
                        STATE_ROOT,
                        worker,
                        state="WAITING",
                        active_run_id=None,
                        message="Waiting for a peer claim to finish or a pending task",
                        evidence=evidence,
                    )
                    time.sleep(args.poll_seconds)
                    continue
                if _all_terminal_or_blocked(table):
                    exit_message = "Queue is terminal or blocked; no task is claimable"
                else:
                    exit_message = "No task was immediately claimable"
                break

            row = _row_for_claim(rows, claim)
            _aggregate(rows, worker, message=f"{worker.worker_id} claimed {claim.run_id}")
            heartbeat: queue_runtime.ClaimHeartbeat | None = None
            try:
                with queue_runtime.ClaimHeartbeat(
                    STATE_ROOT,
                    claim,
                    interval_seconds=args.heartbeat_seconds,
                    evidence=evidence,
                ) as heartbeat:
                    frozen_runner.run_one(row, contract)
            except Exception as exc:
                queue_runtime.finish_claim(
                    STATE_ROOT,
                    claim,
                    status="FAILED",
                    message=f"{type(exc).__name__}: {exc}",
                )
                exit_state = "FAILED"
                exit_message = f"Frozen run_one failed for {claim.run_id}"
                queue_runtime.request_stop_after_current(
                    RUN_ROOT, worker, message=exit_message
                )
                _best_effort_aggregate(rows, worker, message=exit_message)
                raise
            queue_runtime.finish_claim(
                STATE_ROOT,
                claim,
                status="COMPLETE",
                message=(
                    "Frozen run_one completed"
                    if heartbeat is None or heartbeat.error is None
                    else f"Frozen run_one completed; heartbeat warning: {heartbeat.error}"
                ),
            )
            completed_by_worker += 1
            _aggregate(rows, worker, message=f"Completed {claim.run_id}")
            if claim.run_id == frozen_runner.FOLD0_GATE_RUN_ID:
                frozen_runner.write_fold0_gate(row)
                exit_state = "PAUSED_GATE"
                exit_message = "Canonical Fold0 completed; automatic review pause"
                break
    except BaseException as exc:
        if exit_state != "FAILED":
            exit_state = "FAILED"
            exit_message = f"Worker aborted: {type(exc).__name__}: {exc}"
            queue_runtime.request_stop_after_current(
                RUN_ROOT, worker, message=exit_message
            )
        raise
    finally:
        queue_runtime.write_worker_status(
            STATE_ROOT,
            worker,
            state=exit_state,
            active_run_id=None,
            message=exit_message,
            evidence=evidence,
        )
        queue_runtime.log_worker_event(
            STATE_ROOT,
            worker,
            "WORKER_EXITED",
            state=exit_state,
            message=exit_message,
            completed_by_worker=completed_by_worker,
        )
        queue_runtime.finish_worker_slot(slot, worker, state=exit_state)
        _best_effort_aggregate(rows, worker, message=exit_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
