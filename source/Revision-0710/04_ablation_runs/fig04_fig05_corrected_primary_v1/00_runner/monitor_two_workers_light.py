from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = RUN_ROOT / "03_logs"
STATE_ROOT = LOG_ROOT / "multiworker_v1"
STATUS_PATH = RUN_ROOT / "RUN_STATUS.json"
LATEST_PATH = LOG_ROOT / "two_worker_monitor_latest.json"
HISTORY_PATH = LOG_ROOT / "two_worker_monitor_history.jsonl"
TERMINAL_PATH = LOG_ROOT / "TWO_WORKER_MONITOR_TERMINAL.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:12]}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    query_limited_information = 0x1000
    still_active = 259
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def worker_state(pid: int) -> dict[str, Any] | None:
    matches: list[tuple[float, dict[str, Any]]] = []
    for path in (STATE_ROOT / "workers").glob("*.status.json"):
        payload = read_json(path)
        worker = payload.get("worker", {}) if payload else {}
        if int(worker.get("pid", -1)) == pid:
            matches.append((path.stat().st_mtime, payload))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def active_claims() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for path in sorted((STATE_ROOT / "claims").glob("*.json")):
        payload = read_json(path)
        if payload and str(payload.get("status", "")).upper() == "CLAIMED":
            claims.append(
                {
                    "run_id": payload.get("run_id"),
                    "worker_name": payload.get("worker_name"),
                    "worker_pid": payload.get("worker_pid"),
                    "heartbeat_at": payload.get("heartbeat_at"),
                }
            )
    return claims


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
        return {
            "memory_used_mib": float(parts[0]),
            "memory_total_mib": float(parts[1]),
            "utilization_percent": float(parts[2]),
            "power_w": float(parts[3]),
            "temperature_c": float(parts[4]),
            "error": None,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def new_stderr(path: Path, offset: int) -> tuple[str, int]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(min(offset, size))
            text = handle.read().decode("utf-8", errors="replace")
        return text[-4000:], size
    except OSError:
        return "", offset


def snapshot(pids: list[int], offsets: dict[str, int]) -> dict[str, Any]:
    status = read_json(STATUS_PATH) or {}
    workers: list[dict[str, Any]] = []
    for index, pid in enumerate(pids):
        stderr_path = STATE_ROOT / f"worker{index}.stderr.log"
        stderr, offsets[str(stderr_path)] = new_stderr(stderr_path, offsets[str(stderr_path)])
        state = worker_state(pid) or {}
        workers.append(
            {
                "worker_name": f"worker{index}",
                "pid": pid,
                "alive": pid_alive(pid),
                "state": state.get("state"),
                "active_run_id": state.get("active_run_id"),
                "heartbeat_at": state.get("updated_at"),
                "new_stderr": stderr,
            }
        )
    claims = active_claims()
    duplicate_claims = len({claim["run_id"] for claim in claims}) != len(claims)
    new_error = any(worker["new_stderr"].strip() for worker in workers)
    alive_count = sum(bool(worker["alive"]) for worker in workers)
    counts = status.get("counts", {})
    complete = int(counts.get("COMPLETE", 0)) == int(status.get("queue_total", 60))
    if complete:
        monitor_state = "COMPLETE"
    elif duplicate_claims or alive_count == 0:
        monitor_state = "FAILURE"
    elif new_error or alive_count < 2 or int(counts.get("FAILED", 0)) > 0:
        monitor_state = "DEGRADED"
    else:
        monitor_state = "RUNNING"
    return {
        "schema_version": "revision0710.fig04_fig05.two_worker_light_monitor.v1",
        "checked_at": utc_now(),
        "monitor_state": monitor_state,
        "interval_seconds": None,
        "queue_counts": counts,
        "active_run_ids": status.get("active_run_ids", []),
        "workers": workers,
        "claims": claims,
        "duplicate_claims": duplicate_claims,
        "gpu": gpu_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-pids", nargs=2, type=int, required=True)
    parser.add_argument("--interval-seconds", type=int, default=7200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    offsets = {
        str(STATE_ROOT / f"worker{index}.stderr.log"): (STATE_ROOT / f"worker{index}.stderr.log").stat().st_size
        for index in range(2)
    }
    while True:
        record = snapshot(args.worker_pids, offsets)
        record["interval_seconds"] = args.interval_seconds
        write_json(LATEST_PATH, record)
        with HISTORY_PATH.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        if record["monitor_state"] in {"COMPLETE", "FAILURE"}:
            write_json(TERMINAL_PATH, record)
            return 0 if record["monitor_state"] == "COMPLETE" else 1
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
