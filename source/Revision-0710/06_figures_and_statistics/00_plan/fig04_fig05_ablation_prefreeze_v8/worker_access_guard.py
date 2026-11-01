from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import v8_runtime_guard


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
SCHEMA = "revision0710.fig04_fig05.worker_access_audit.v8"


class AccessGuardError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(value: str | Path) -> Path:
    return Path(value).resolve()


def _production_allowed_files(task: dict[str, Any]) -> set[Path]:
    source = task["data_source"]
    manifest_path = _resolve(source["role_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_dir = _resolve(source["cache_dir"])
    allowed = {manifest_path}
    artifact = manifest["artifact"]
    allowed.update(
        {
            _resolve(artifact["age_targets"]["path"]),
            _resolve(artifact["joint_labels"]["path"]),
            cache_dir / "batch_manifest.json",
            cache_dir / "fold_transform_manifest.json",
            cache_dir / "embedding_tables.npz",
            cache_dir / "embedding_tables.pt",
            cache_dir / "cache_config.json",
        }
    )
    fold_ids = {int(value) for value in artifact["fold_ids"]}
    batch_manifest = json.loads((cache_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    for row in batch_manifest:
        if int(row["fold_id"]) in fold_ids:
            allowed.add(_resolve(cache_dir / row["path"]))
    transforms = json.loads(
        (cache_dir / "fold_transform_manifest.json").read_text(encoding="utf-8")
    )
    outer_fold = int(task["outer_fold"])
    allowed.add(_resolve(cache_dir / transforms[outer_fold]["path"]))
    return allowed


def allowed_files_for_task(task_path: Path, task: dict[str, Any]) -> tuple[set[Path], Path]:
    attempt = _resolve(task["output_dir"]).parent
    audit_dir = attempt / "00_worker_access_audit"
    audit_path = audit_dir / f"{task['phase']}.json"
    allowed = {
        task_path.resolve(),
        _resolve(task["claim_path"]),
        _resolve(task["contract_path"]),
    }
    authorization = task.get("authorization", {})
    if authorization.get("required") is True:
        allowed.add(_resolve(authorization["path"]))
    lock = task.get("gpu_lock", {})
    for key in v8_runtime_guard.LOCK_PATH_KEYS:
        if lock.get(key):
            allowed.add(_resolve(lock[key]))
    for key in (
        "training_complete_path",
        "checkpoint_path",
        "calibration_complete_path",
        "age_correction_path",
        "cox_nuisance_path",
        "outer_read_ledger_path",
    ):
        if task.get(key):
            allowed.add(_resolve(task[key]))
    source = task["data_source"]
    if source["format"] == "synthetic_role_bundle":
        allowed.add(_resolve(source["role_bundle_path"]))
    elif source["format"] == "revision0710_role_manifest":
        allowed.update(_production_allowed_files(task))
    else:
        raise AccessGuardError("worker data-source format changed")
    return allowed, audit_path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class AuditHook:
    def __init__(self, *, allowed_files: set[Path], allowed_write_root: Path, audit_path: Path):
        self.allowed_files = {path.resolve() for path in allowed_files}
        self.allowed_write_root = allowed_write_root.resolve()
        self.audit_path = audit_path.resolve()
        self.events: list[dict[str, Any]] = []
        self.denied: list[dict[str, Any]] = []
        self.active = True
        self.sensitive_roots = (
            (REPO_ROOT / "Revision-0710" / "01_data_rebuild").resolve(),
            (
                REPO_ROOT
                / "Revision-0710"
                / "03_ehr_age"
                / "04_fixed_replication"
                / "00_role_artifacts"
            ).resolve(),
            (REPO_ROOT / "Revision-0710" / "04_ablation_runs").resolve(),
            self.allowed_write_root,
        )

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        if not self.active or event != "open" or not args or isinstance(args[0], int):
            return
        try:
            path = _resolve(args[0])
        except (OSError, TypeError, ValueError):
            return
        if not any(_inside(path, root) for root in self.sensitive_roots):
            return
        mode = args[1] if len(args) > 1 else None
        flags = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 0
        write = bool(
            isinstance(mode, str) and any(value in mode for value in ("w", "a", "+", "x"))
        ) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
        access = "write" if write else "read"
        permitted = (
            _inside(path, self.allowed_write_root)
            if write
            else path in self.allowed_files or _inside(path, self.allowed_write_root)
        )
        row = {
            "event_ordinal": len(self.events) + len(self.denied) + 1,
            "path": str(path),
            "access": access,
            "permitted": permitted,
        }
        if permitted:
            self.events.append(row)
            return
        self.denied.append(row)
        raise PermissionError(f"worker access guard denied {access}: {path}")


def _write_audit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_guarded(task_path: Path) -> int:
    task_path = task_path.resolve()
    task_bytes = task_path.read_bytes()
    task = json.loads(task_bytes)
    allowed, audit_path = allowed_files_for_task(task_path, task)
    if audit_path.exists():
        raise AccessGuardError(f"create-only worker access audit exists: {audit_path}")
    hook = AuditHook(
        allowed_files=allowed,
        allowed_write_root=_resolve(task["output_dir"]),
        audit_path=audit_path,
    )
    sys.addaudithook(hook)
    os.environ["REVISION0710_WORKER_ACCESS_GUARD_ACTIVE"] = "1"
    os.environ["REVISION0710_WORKER_ACCESS_AUDIT_PATH"] = str(audit_path)
    exit_code = 1
    error = None
    try:
        sys.argv = [str((ROOT / "isolated_worker.py").resolve()), "--task", str(task_path)]
        runpy.run_path(str((ROOT / "isolated_worker.py").resolve()), run_name="__main__")
        exit_code = 0
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        hook.active = False
        unique = []
        seen: set[tuple[str, str]] = set()
        for row in hook.events:
            identity = (row["path"], row["access"])
            if identity not in seen:
                seen.add(identity)
                unique.append(row)
        payload = {
            "schema_version": SCHEMA,
            "status": "PASS_NO_DENIED_SENSITIVE_ACCESS" if not hook.denied and exit_code == 0 else "FAIL",
            "run_id": task["run_id"],
            "phase": task["phase"],
            "worker_pid": os.getpid(),
            "executor_pid": os.getppid(),
            "process_isolated": True,
            "python_audit_hook_active_before_worker_import": True,
            "task_path": str(task_path),
            "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
            "allowed_files_sha256": _canonical_sha(sorted(str(path) for path in allowed)),
            "actual_sensitive_accesses": unique,
            "actual_sensitive_access_count": len(hook.events),
            "denied_sensitive_accesses": hook.denied,
            "worker_exit_code": exit_code,
            "worker_error": error,
        }
        _write_audit(audit_path, payload)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    return run_guarded(args.task)


if __name__ == "__main__":
    raise SystemExit(main())
