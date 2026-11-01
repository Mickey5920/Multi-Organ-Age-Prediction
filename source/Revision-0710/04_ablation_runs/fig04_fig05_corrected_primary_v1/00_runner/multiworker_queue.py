from __future__ import annotations

"""Operational locking and status helpers for the frozen ablation queue.

The scientific runner is hash-bound by the frozen contract.  This module stays
outside that contract and only coordinates calls into the unchanged ``run_one``.
"""

import json
import os
import re
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pandas as pd


CLAIM_SCHEMA = "revision0710.fig04_fig05.corrected_primary.task_claim.v1"
HEARTBEAT_SCHEMA = "revision0710.fig04_fig05.corrected_primary.claim_heartbeat.v1"
TERMINAL_SCHEMA = "revision0710.fig04_fig05.corrected_primary.claim_terminal.v1"
WORKER_SCHEMA = "revision0710.fig04_fig05.corrected_primary.worker_status.v1"
RUN_STATUS_SCHEMA = "revision0710.fig04_fig05.corrected_primary.run_status.v2"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short enough for Windows paths under deeply
    # nested claim directories.  The UUID still makes concurrent writers safe.
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:12]}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:12]}")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def validate_worker_id(value: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(
            "--worker-id must be 1-64 characters using letters, digits, '.', '_' or '-'"
        )
    return value


@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    session_id: str
    pid: int
    hostname: str
    started_at: str


@dataclass(frozen=True)
class TaskClaim:
    run_id: str
    ordinal: int
    claim_dir: Path
    worker: WorkerIdentity


def create_worker_identity(worker_id: str) -> WorkerIdentity:
    worker_id = validate_worker_id(worker_id)
    started = utc_now()
    session_id = f"{worker_id}__{socket.gethostname()}__p{os.getpid()}__{uuid.uuid4().hex[:8]}"
    return WorkerIdentity(
        worker_id=worker_id,
        session_id=session_id,
        pid=os.getpid(),
        hostname=socket.gethostname(),
        started_at=started,
    )


def claim_directory(state_root: Path, row: dict[str, str]) -> Path:
    return state_root / "claims" / f"{int(row['ordinal']):02d}_{row['run_id']}.claim"


def claim_monitor_path(state_root: Path, row: dict[str, str]) -> Path:
    return state_root / "claims" / f"{int(row['ordinal']):02d}_{row['run_id']}.json"


def worker_status_path(state_root: Path, worker: WorkerIdentity) -> Path:
    return state_root / "workers" / f"{worker.session_id}.status.json"


def worker_event_path(state_root: Path, worker: WorkerIdentity) -> Path:
    return state_root / "workers" / f"{worker.session_id}.events.jsonl"


def acquire_worker_slot(
    state_root: Path,
    worker: WorkerIdentity,
    *,
    session_token: str,
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", session_token):
        raise ValueError("dual-worker session token must be 64 lowercase hex characters")
    slot = state_root / "worker_slots" / session_token / f"{worker.worker_id}.slot"
    slot.parent.mkdir(parents=True, exist_ok=True)
    try:
        slot.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Worker slot is already occupied for this session: {worker.worker_id}"
        ) from exc
    _atomic_json(
        slot / "OWNER.json",
        {
            "schema_version": "revision0710.fig04_fig05.corrected_primary.worker_slot.v1",
            "acquired_at": utc_now(),
            "session_token": session_token,
            "worker": asdict(worker),
            "automatic_slot_stealing": False,
        },
    )
    return slot


def finish_worker_slot(slot: Path, worker: WorkerIdentity, *, state: str) -> None:
    _atomic_json(
        slot / "TERMINAL.json",
        {
            "schema_version": "revision0710.fig04_fig05.corrected_primary.worker_slot_terminal.v1",
            "terminal_at": utc_now(),
            "state": state,
            "worker": asdict(worker),
        },
    )


def write_worker_status(
    state_root: Path,
    worker: WorkerIdentity,
    *,
    state: str,
    active_run_id: str | None,
    message: str,
    evidence: dict[str, Any],
) -> None:
    _atomic_json(
        worker_status_path(state_root, worker),
        {
            "schema_version": WORKER_SCHEMA,
            "updated_at": utc_now(),
            "worker": asdict(worker),
            "state": state,
            "active_run_id": active_run_id,
            "claimed_run_ids": [] if active_run_id is None else [active_run_id],
            "worker_name": worker.worker_id,
            "worker_pid": worker.pid,
            "hostname": worker.hostname,
            "message": message,
            "evidence": evidence,
        },
    )


def log_worker_event(
    state_root: Path,
    worker: WorkerIdentity,
    event: str,
    **fields: Any,
) -> None:
    _append_jsonl(
        worker_event_path(state_root, worker),
        {
            "schema_version": "revision0710.fig04_fig05.corrected_primary.worker_event.v1",
            "at": utc_now(),
            "worker_id": worker.worker_id,
            "session_id": worker.session_id,
            "pid": worker.pid,
            "event": event,
            **fields,
        },
    )


def try_claim_task(
    state_root: Path,
    row: dict[str, str],
    worker: WorkerIdentity,
    *,
    evidence: dict[str, Any],
) -> TaskClaim | None:
    """Atomically claim a task; an existing claim is never stolen."""

    claims_root = state_root / "claims"
    claims_root.mkdir(parents=True, exist_ok=True)
    directory = claim_directory(state_root, row)
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError:
        return None

    claim = TaskClaim(
        run_id=row["run_id"],
        ordinal=int(row["ordinal"]),
        claim_dir=directory,
        worker=worker,
    )
    _atomic_json(
        directory / "CLAIM.json",
        {
            "schema_version": CLAIM_SCHEMA,
            "claimed_at": utc_now(),
            "run_id": row["run_id"],
            "ordinal": int(row["ordinal"]),
            "figure": row["figure"],
            "variant_id": row["variant_id"],
            "outer_fold": int(row["outer_fold"]),
            "worker": asdict(worker),
            "evidence": evidence,
            "policy": {
                "atomic_primitive": "mkdir",
                "automatic_claim_stealing": False,
                "task_output_identity_changed": False,
            },
        },
    )
    _atomic_json(
        claim_monitor_path(state_root, row),
        {
            "schema_version": CLAIM_SCHEMA,
            "status": "CLAIMED",
            "claimed_at": utc_now(),
            "heartbeat_at": utc_now(),
            "run_id": row["run_id"],
            "ordinal": int(row["ordinal"]),
            "worker_name": worker.worker_id,
            "owner": worker.worker_id,
            "worker_pid": worker.pid,
            "hostname": worker.hostname,
            "claim_directory": str(directory.resolve()),
        },
    )
    touch_claim(claim, phase="CLAIMED")
    log_worker_event(
        state_root,
        worker,
        "TASK_CLAIMED",
        run_id=claim.run_id,
        ordinal=claim.ordinal,
        claim_dir=str(directory.resolve()),
    )
    return claim


def try_claim_next(
    state_root: Path,
    rows: Sequence[dict[str, str]],
    worker: WorkerIdentity,
    *,
    is_pending: Callable[[dict[str, str]], bool],
    evidence: dict[str, Any],
) -> TaskClaim | None:
    """Claim the first pending row, continuing after every lost mkdir race."""

    for row in rows:
        if not is_pending(row):
            continue
        claim = try_claim_task(state_root, row, worker, evidence=evidence)
        if claim is not None:
            return claim
    return None


def touch_claim(claim: TaskClaim, *, phase: str) -> None:
    heartbeat_at = utc_now()
    _atomic_json(
        claim.claim_dir / "HEARTBEAT.json",
        {
            "schema_version": HEARTBEAT_SCHEMA,
            "heartbeat_at": heartbeat_at,
            "run_id": claim.run_id,
            "ordinal": claim.ordinal,
            "phase": phase,
            "worker": asdict(claim.worker),
        },
    )
    monitor_path = claim.claim_dir.parent / f"{claim.claim_dir.name.removesuffix('.claim')}.json"
    monitor = _read_object(monitor_path)
    if monitor is not None and str(monitor.get("status", "")).upper() == "CLAIMED":
        monitor["heartbeat_at"] = heartbeat_at
        monitor["phase"] = phase
        _atomic_json(monitor_path, monitor)


def finish_claim(
    state_root: Path,
    claim: TaskClaim,
    *,
    status: str,
    message: str,
) -> None:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError(f"Unsupported terminal claim status: {status}")
    _atomic_json(
        claim.claim_dir / "TERMINAL.json",
        {
            "schema_version": TERMINAL_SCHEMA,
            "terminal_at": utc_now(),
            "status": status,
            "message": message,
            "run_id": claim.run_id,
            "ordinal": claim.ordinal,
            "worker": asdict(claim.worker),
        },
    )
    monitor_path = claim.claim_dir.parent / f"{claim.claim_dir.name.removesuffix('.claim')}.json"
    monitor = _read_object(monitor_path) or {}
    monitor.update(
        {
            "schema_version": CLAIM_SCHEMA,
            "status": status,
            "terminal_at": utc_now(),
            "message": message,
            "run_id": claim.run_id,
            "ordinal": claim.ordinal,
            "worker_name": claim.worker.worker_id,
            "owner": claim.worker.worker_id,
            "worker_pid": claim.worker.pid,
            "hostname": claim.worker.hostname,
            "claim_directory": str(claim.claim_dir.resolve()),
        }
    )
    _atomic_json(monitor_path, monitor)
    log_worker_event(
        state_root,
        claim.worker,
        f"TASK_{status}",
        run_id=claim.run_id,
        ordinal=claim.ordinal,
        message=message,
    )


class ClaimHeartbeat:
    def __init__(
        self,
        state_root: Path,
        claim: TaskClaim,
        *,
        interval_seconds: float,
        evidence: dict[str, Any],
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.state_root = state_root
        self.claim = claim
        self.interval_seconds = interval_seconds
        self.evidence = evidence
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{claim.run_id}",
            daemon=True,
        )
        self.error: str | None = None

    def __enter__(self) -> "ClaimHeartbeat":
        self._write()
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))

    def _write(self) -> None:
        touch_claim(self.claim, phase="RUN_ONE")
        write_worker_status(
            self.state_root,
            self.claim.worker,
            state="RUNNING",
            active_run_id=self.claim.run_id,
            message="Frozen run_one is executing",
            evidence=self.evidence,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._write()
            except Exception as exc:  # heartbeat must not interrupt the scientific task
                self.error = f"{type(exc).__name__}: {exc}"
                log_worker_event(
                    self.state_root,
                    self.claim.worker,
                    "HEARTBEAT_ERROR",
                    run_id=self.claim.run_id,
                    error=self.error,
                )
                return


def pid_is_alive(pid: int, hostname: str) -> bool | None:
    if hostname.casefold() != socket.gethostname().casefold():
        return None
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # Access denied means the PID exists but is not queryable by this token.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def request_stop_after_current(
    run_root: Path,
    worker: WorkerIdentity,
    *,
    message: str,
) -> Path:
    """Create the global fail-closed marker without replacing an existing one."""

    sentinel = run_root / "STOP_AFTER_CURRENT"
    try:
        with sentinel.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema_version": "revision0710.fig04_fig05.corrected_primary.peer_stop.v1",
                        "requested_at": utc_now(),
                        "worker_id": worker.worker_id,
                        "session_id": worker.session_id,
                        "pid": worker.pid,
                        "message": message,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    except FileExistsError:
        pass
    return sentinel


def claim_runtime_state(state_root: Path, row: dict[str, str]) -> tuple[str | None, str]:
    directory = claim_directory(state_root, row)
    if not directory.is_dir():
        return None, ""
    terminal = _read_object(directory / "TERMINAL.json")
    if terminal is not None:
        return f"CLAIM_{terminal.get('status', 'TERMINAL')}", str(terminal.get("terminal_at", ""))
    record = _read_object(directory / "CLAIM.json")
    if record is None:
        return "CLAIMED", "claim directory exists; CLAIM.json is unavailable"
    worker = record.get("worker", {})
    alive = pid_is_alive(int(worker.get("pid", -1)), str(worker.get("hostname", "")))
    heartbeat = _read_object(directory / "HEARTBEAT.json") or {}
    detail = str(heartbeat.get("heartbeat_at", record.get("claimed_at", "")))
    if alive is True:
        return "RUNNING", detail
    if alive is False:
        return "CLAIMED_STALE", detail
    return "CLAIMED", detail


def task_state(
    row: dict[str, str],
    *,
    state_root: Path,
    task_paths: Callable[[dict[str, str]], dict[str, Path]],
) -> tuple[str, str]:
    paths = task_paths(row)
    if paths["complete"].is_file():
        payload = _read_object(paths["complete"]) or {}
        return "COMPLETE", str(payload.get("completed_at", ""))
    if paths["failure"].is_file():
        payload = _read_object(paths["failure"]) or {}
        return "FAILED", str(payload.get("error", ""))
    claim_state, detail = claim_runtime_state(state_root, row)
    if claim_state in {"RUNNING", "CLAIMED", "CLAIMED_STALE"}:
        return claim_state, detail
    if paths["root"].exists():
        return "INCOMPLETE", "task output directory exists without a terminal receipt"
    if claim_state is not None:
        return "INCOMPLETE", detail
    return "PENDING", ""


def scan_queue(
    rows: Sequence[dict[str, str]],
    *,
    state_root: Path,
    task_paths: Callable[[dict[str, str]], dict[str, Path]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        state, detail = task_state(row, state_root=state_root, task_paths=task_paths)
        records.append({**row, "run_state": state, "detail": detail})
    return pd.DataFrame(records)


def is_task_pending(
    row: dict[str, str],
    *,
    state_root: Path,
    task_paths: Callable[[dict[str, str]], dict[str, Path]],
) -> bool:
    state, _ = task_state(row, state_root=state_root, task_paths=task_paths)
    return state == "PENDING"


@contextmanager
def aggregate_lock(
    state_root: Path,
    worker: WorkerIdentity,
    *,
    timeout_seconds: float = 30.0,
    stale_after_seconds: float = 120.0,
) -> Iterator[None]:
    """Serialize shared summaries; only dead local owners may be reclaimed."""

    lock_dir = state_root / "two_worker_aggregate_status.lock"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            lock_dir.mkdir(exist_ok=False)
            _atomic_json(
                lock_dir / "OWNER.json",
                {
                    "worker_id": worker.worker_id,
                    "session_id": worker.session_id,
                    "pid": worker.pid,
                    "hostname": worker.hostname,
                    "acquired_at": utc_now(),
                },
            )
            break
        except FileExistsError:
            try:
                age = max(0.0, time.time() - lock_dir.stat().st_mtime)
            except FileNotFoundError:
                continue
            # Do not open OWNER.json while a live peer is releasing the lock.
            # On Windows a concurrent reader can make unlink fail with WinError 32.
            if age >= stale_after_seconds:
                owner_path = lock_dir / "OWNER.json"
                owner = _read_object(owner_path)
                owner_alive = (
                    None
                    if owner is None
                    else pid_is_alive(
                        int(owner.get("pid", -1)), str(owner.get("hostname", ""))
                    )
                )
                recoverable = owner is None or owner_alive is False
            else:
                owner_path = lock_dir / "OWNER.json"
                owner = None
                recoverable = False
            if recoverable:
                _append_jsonl(
                    state_root / "two_worker_aggregate_lock_recovery.jsonl",
                    {
                        "schema_version": "revision0710.fig04_fig05.corrected_primary.aggregate_lock_recovery.v1",
                        "recovered_at": utc_now(),
                        "lock_dir": str(lock_dir.resolve()),
                        "age_seconds": age,
                        "previous_owner": owner,
                        "reason": "missing_or_corrupt_owner" if owner is None else "dead_local_owner",
                        "recovering_worker": asdict(worker),
                    },
                )
                try:
                    owner_path.unlink(missing_ok=True)
                    lock_dir.rmdir()
                except (FileNotFoundError, OSError):
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for aggregate status lock: {lock_dir}")
            time.sleep(0.05)
    try:
        yield
    finally:
        release_deadline = time.monotonic() + 5.0
        while True:
            try:
                (lock_dir / "OWNER.json").unlink(missing_ok=True)
                lock_dir.rmdir()
                break
            except FileNotFoundError:
                break
            except OSError:
                if time.monotonic() >= release_deadline:
                    break
                time.sleep(0.05)


def _worker_snapshots(state_root: Path) -> list[dict[str, Any]]:
    if not state_root.is_dir():
        return []
    snapshots = []
    workers_root = state_root / "workers"
    if not workers_root.is_dir():
        return []
    for path in sorted(workers_root.glob("*.status.json")):
        payload = _read_object(path)
        if payload is not None:
            worker = payload.get("worker", {})
            payload["pid_alive"] = pid_is_alive(
                int(worker.get("pid", -1)), str(worker.get("hostname", ""))
            )
            snapshots.append(payload)
    return snapshots


def aggregate_run_status(
    rows: Sequence[dict[str, str]],
    *,
    state_root: Path,
    task_paths: Callable[[dict[str, str]], dict[str, Path]],
    worker: WorkerIdentity,
    status_path: Path,
    status_csv_path: Path,
    queue_sha256: str,
    contract_sha256: str,
    stop_after_current_present: bool,
    message: str,
) -> pd.DataFrame:
    with aggregate_lock(state_root, worker):
        table = scan_queue(rows, state_root=state_root, task_paths=task_paths)
        _atomic_csv(status_csv_path, table)
        counts = table["run_state"].value_counts().to_dict()
        active = table.loc[
            table["run_state"].isin(["RUNNING", "CLAIMED"]), "run_id"
        ].astype(str).tolist()
        _atomic_json(
            status_path,
            {
                "schema_version": RUN_STATUS_SCHEMA,
                "updated_at": utc_now(),
                "active_run_id": active[0] if len(active) == 1 else None,
                "active_run_ids": active,
                "message": message,
                "stop_after_current_present": stop_after_current_present,
                "counts": {str(key): int(value) for key, value in counts.items()},
                "queue_total": int(len(table)),
                "queue_sha256": queue_sha256,
                "frozen_contract_sha256": contract_sha256,
                "coordination": {
                    "mode": "atomic_mkdir_multiworker_v1",
                    "state_root": str(state_root.resolve()),
                    "claim_root": str((state_root / "claims").resolve()),
                    "automatic_task_claim_stealing": False,
                    "shared_status_writer": "aggregate_run_status under aggregate_status.lock",
                },
                "workers": _worker_snapshots(state_root),
            },
        )
        return table
