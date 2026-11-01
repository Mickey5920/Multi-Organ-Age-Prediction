from __future__ import annotations

import datetime as dt
import json
import os
import platform
import secrets
import sys
from pathlib import Path
from typing import Any, BinaryIO


class RuntimeGuardError(RuntimeError):
    pass


LOCK_SCHEMA = "revision0710.fig04_fig05.single_gpu_lock.v8"
LOCK_BINDING_KEYS = {
    "required",
    "state_path",
    "os_lock_path",
    "token",
    "owner_pid",
    "run_id",
    "acquisition_ordinal",
}
LOCK_PATH_KEYS = ("state_path", "os_lock_path")


def lock_binding_paths(binding: dict[str, Any]) -> tuple[Path, Path]:
    """Return the two paths from the one frozen GPU-lock binding schema."""

    if not isinstance(binding, dict) or set(binding) != LOCK_BINDING_KEYS:
        raise RuntimeGuardError("formal GPU lock binding keys changed")
    if binding.get("required") is not True:
        raise RuntimeGuardError("formal GPU lock binding is not required")
    paths = tuple(Path(str(binding[key])).resolve() for key in LOCK_PATH_KEYS)
    if paths[0].name != "SINGLE_GPU_FORMAL_RUN_STATE.json":
        raise RuntimeGuardError("formal GPU lock state path name changed")
    if paths[1].name != "SINGLE_GPU_FORMAL_RUN.oslock":
        raise RuntimeGuardError("formal GPU OS lock path name changed")
    return paths


def observed_formal_environment() -> dict[str, Any]:
    import lifelines
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import torch

    configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if configured != ":4096:8":
        raise RuntimeGuardError("formal deterministic CuBLAS workspace setting changed")
    if not torch.cuda.is_available():
        raise RuntimeGuardError("formal environment has no available CUDA device")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__.split("+")[0],
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lifelines": lifelines.__version__,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpu_count": torch.cuda.device_count(),
        "cublas_workspace_config": configured,
    }


def validate_formal_environment(expected: dict[str, Any]) -> dict[str, Any]:
    observed = observed_formal_environment()
    if Path(observed["python_executable"]).resolve() != Path(expected["python_executable"]).resolve():
        raise RuntimeGuardError("formal Python executable changed")
    for key in set(expected) - {"python_executable"}:
        if observed.get(key) != expected[key]:
            raise RuntimeGuardError(f"formal environment changed: {key}")
    return observed


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _lock_byte(handle: BinaryIO, *, nonblocking: bool) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK
        msvcrt.locking(handle.fileno(), mode, 1)
    else:
        import fcntl

        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(handle.fileno(), flags)


def _unlock_byte(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, PermissionError):
        return False
    return True


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeGuardError("persistent GPU lock state is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != LOCK_SCHEMA:
        raise RuntimeGuardError("persistent GPU lock state schema changed")
    if payload.get("status") not in {"ACTIVE", "RELEASED"}:
        raise RuntimeGuardError("persistent GPU lock state status changed")
    if not isinstance(payload.get("events"), list):
        raise RuntimeGuardError("persistent GPU lock audit events changed")
    return payload


def _event(
    *,
    ordinal: int,
    event: str,
    owner_pid: int,
    run_id: str,
    token: str,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_ordinal": ordinal,
        "event": event,
        "created_utc": _utc_now(),
        "owner_pid": owner_pid,
        "run_id": run_id,
        "token": token,
        "previous_owner_pid": (
            int(previous["owner_pid"]) if previous is not None else None
        ),
        "previous_run_id": previous.get("run_id") if previous is not None else None,
        "previous_pid_alive_observation": (
            _pid_alive(int(previous["owner_pid"])) if previous is not None else None
        ),
        "os_lock_was_free": True,
    }


def acquire_global_gpu_lock(
    formal_root: Path, *, run_id: str
) -> tuple[BinaryIO, Path, dict[str, Any]]:
    formal_root.mkdir(parents=True, exist_ok=True)
    os_lock_path = formal_root / "SINGLE_GPU_FORMAL_RUN.oslock"
    state_path = formal_root / "SINGLE_GPU_FORMAL_RUN_STATE.json"
    handle = os_lock_path.open("a+b")
    if os_lock_path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _lock_byte(handle, nonblocking=True)
    except OSError as exc:
        handle.close()
        raise RuntimeGuardError("another formal GPU run holds the OS lock") from exc

    try:
        previous_state = _read_state(state_path)
        events = list(previous_state["events"]) if previous_state is not None else []
        previous_active = (
            previous_state.get("active")
            if previous_state is not None and previous_state.get("status") == "ACTIVE"
            else None
        )
        token = secrets.token_hex(32)
        if previous_active is not None:
            events.append(
                _event(
                    ordinal=len(events) + 1,
                    event="RECOVERED_STALE_ACTIVE_STATE_AFTER_OS_LOCK_BECAME_FREE",
                    owner_pid=os.getpid(),
                    run_id=str(run_id),
                    token=token,
                    previous=previous_active,
                )
            )
        acquisition_ordinal = int(
            previous_state.get("acquisition_count", 0) if previous_state else 0
        ) + 1
        active = {
            "owner_pid": os.getpid(),
            "run_id": str(run_id),
            "token": token,
            "acquisition_ordinal": acquisition_ordinal,
            "acquired_utc": _utc_now(),
        }
        events.append(
            _event(
                ordinal=len(events) + 1,
                event="ACQUIRED_OS_BACKED_LOCK",
                owner_pid=os.getpid(),
                run_id=str(run_id),
                token=token,
            )
        )
        payload = {
            "schema_version": LOCK_SCHEMA,
            "status": "ACTIVE",
            "os_lock_path": str(os_lock_path.resolve()),
            "acquisition_count": acquisition_ordinal,
            "active": active,
            "events": events,
        }
        _atomic_write_json(state_path, payload)
    except Exception:
        _unlock_byte(handle)
        handle.close()
        raise
    binding = {
        "required": True,
        "state_path": str(state_path.resolve()),
        "os_lock_path": str(os_lock_path.resolve()),
        "token": token,
        "owner_pid": os.getpid(),
        "run_id": str(run_id),
        "acquisition_ordinal": acquisition_ordinal,
    }
    return handle, state_path, binding


def _prove_os_lock_held_by_other(path: Path) -> None:
    handle = path.open("r+b")
    try:
        try:
            _lock_byte(handle, nonblocking=True)
        except OSError:
            return
        _unlock_byte(handle)
    finally:
        handle.close()
    raise RuntimeGuardError("GPU OS lock is not held by the executor")


def validate_global_gpu_lock(binding: dict[str, Any], *, task_run_id: str) -> dict[str, Any]:
    state_path, os_lock_path = lock_binding_paths(binding)
    if binding.get("required") is not True or binding.get("run_id") != task_run_id:
        raise RuntimeGuardError("formal GPU lock binding identity changed")
    if int(binding.get("owner_pid", -1)) != os.getppid():
        raise RuntimeGuardError("formal worker is not owned by the lock-holding executor")
    if state_path.name != "SINGLE_GPU_FORMAL_RUN_STATE.json" or not state_path.is_file():
        raise RuntimeGuardError("formal GPU lock state path is invalid")
    if os_lock_path.name != "SINGLE_GPU_FORMAL_RUN.oslock" or not os_lock_path.is_file():
        raise RuntimeGuardError("formal GPU OS lock path is invalid")
    state = _read_state(state_path)
    assert state is not None
    expected_active = {
        "owner_pid": int(binding["owner_pid"]),
        "run_id": task_run_id,
        "token": str(binding["token"]),
        "acquisition_ordinal": int(binding["acquisition_ordinal"]),
    }
    active = state.get("active")
    if state.get("status") != "ACTIVE" or not isinstance(active, dict):
        raise RuntimeGuardError("formal GPU lock is not active")
    for key, value in expected_active.items():
        if active.get(key) != value:
            raise RuntimeGuardError(f"formal GPU lock active binding changed: {key}")
    if Path(state.get("os_lock_path", "")).resolve() != os_lock_path:
        raise RuntimeGuardError("formal GPU lock OS path binding changed")
    _prove_os_lock_held_by_other(os_lock_path)
    return state


def release_global_gpu_lock(
    handle: BinaryIO,
    state_path: Path,
    binding: dict[str, Any],
) -> None:
    try:
        state = _read_state(state_path)
        if state is None or state.get("status") != "ACTIVE":
            raise RuntimeGuardError("global GPU lock state disappeared before release")
        active = state.get("active")
        if not isinstance(active, dict) or any(
            active.get(key) != binding.get(key)
            for key in ("token", "owner_pid", "run_id", "acquisition_ordinal")
        ):
            raise RuntimeGuardError("global GPU lock changed before release")
        events = list(state["events"])
        events.append(
            _event(
                ordinal=len(events) + 1,
                event="RELEASED_OS_BACKED_LOCK",
                owner_pid=int(binding["owner_pid"]),
                run_id=str(binding["run_id"]),
                token=str(binding["token"]),
            )
        )
        payload = {
            **state,
            "status": "RELEASED",
            "active": None,
            "last_release": {
                "owner_pid": int(binding["owner_pid"]),
                "run_id": str(binding["run_id"]),
                "token": str(binding["token"]),
                "acquisition_ordinal": int(binding["acquisition_ordinal"]),
                "released_utc": _utc_now(),
            },
            "events": events,
        }
        _atomic_write_json(state_path, payload)
    finally:
        try:
            _unlock_byte(handle)
        finally:
            handle.close()
