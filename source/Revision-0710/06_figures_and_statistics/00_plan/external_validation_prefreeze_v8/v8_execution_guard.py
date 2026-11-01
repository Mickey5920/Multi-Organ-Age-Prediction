from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class V8GuardError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def current_environment_record(
    *, role: str, required_packages: Sequence[str]
) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    records = []
    for package in required_packages:
        distribution = importlib.metadata.distribution(package)
        distribution_path = Path(distribution._path)  # type: ignore[attr-defined]
        metadata_path = distribution_path / "METADATA"
        if not metadata_path.is_file():
            metadata_path = distribution_path / "PKG-INFO"
        record_path = distribution_path / "RECORD"
        if not record_path.is_file():
            record_path = distribution_path / "SOURCES.txt"
        if not metadata_path.is_file():
            raise V8GuardError(f"package metadata file is absent: {package}")
        records.append(
            {
                "name": package,
                "version": distribution.version,
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": sha256_file(metadata_path),
                "record_path": str(record_path.resolve()) if record_path.is_file() else None,
                "record_sha256": sha256_file(record_path) if record_path.is_file() else None,
            }
        )
    return {
        "role": role,
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": records,
    }


def validate_current_environment(
    expected: Mapping[str, Any], *, expected_role: str
) -> dict[str, Any]:
    if expected.get("role") != expected_role:
        raise V8GuardError("environment role mismatch")
    package_names = [str(record["name"]) for record in expected["packages"]]
    observed = current_environment_record(
        role=expected_role, required_packages=package_names
    )
    if observed != dict(expected):
        raise V8GuardError(
            "current environment differs from the byte-bound V8 environment lock"
        )
    return observed


def verify_local_code_bindings(
    *, repo_root: Path, bindings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Hash every authorized local code byte before external paths are touched."""
    verified: list[dict[str, Any]] = []
    for record in bindings:
        relative = str(record["relative_path"])
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise V8GuardError(f"code binding escapes repository: {relative}") from exc
        if not path.is_file():
            raise V8GuardError(f"bound code file is absent: {relative}")
        observed = {
            "name": str(record["name"]),
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if observed != dict(record):
            raise V8GuardError(f"bound code bytes changed: {relative}")
        verified.append(observed)
    return verified


def _atomic_json_create(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def file_manifest(paths: Sequence[Path], *, root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((value.resolve() for value in paths), key=lambda x: str(x)):
        if not path.is_file():
            raise V8GuardError(f"stage output is absent: {path}")
        records.append(
            {
                "relative_path": path.relative_to(root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def validate_stage_receipt(
    receipt_path: Path,
    *,
    stage_name: str,
    stage_root: Path,
    input_bindings: Sequence[Mapping[str, Any]],
    environment_sha256: str,
) -> dict[str, Any]:
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_keys = [
        "schema_version",
        "status",
        "stage_name",
        "environment_lock_sha256",
        "input_bindings",
        "outputs",
    ]
    if list(value) != expected_keys:
        raise V8GuardError("stage receipt is not a closed ordered object")
    if value["schema_version"] != "revision0710.external_validation.stage_receipt.v8":
        raise V8GuardError("stage receipt schema mismatch")
    if value["status"] != "STAGE_COMPLETE_CREATE_ONCE_REUSABLE":
        raise V8GuardError("stage receipt status mismatch")
    if value["stage_name"] != stage_name:
        raise V8GuardError("stage receipt name mismatch")
    if value["environment_lock_sha256"] != environment_sha256:
        raise V8GuardError("stage environment binding mismatch")
    if value["input_bindings"] != list(input_bindings):
        raise V8GuardError("stage input bindings mismatch")
    for record in value["outputs"]:
        if list(record) != ["relative_path", "bytes", "sha256"]:
            raise V8GuardError("stage output record is open or reordered")
        path = stage_root / record["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise V8GuardError(f"stage output hash mismatch: {path}")
    return value


def commit_stage_receipt(
    receipt_path: Path,
    *,
    stage_name: str,
    stage_root: Path,
    input_bindings: Sequence[Mapping[str, Any]],
    output_paths: Sequence[Path],
    environment_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "revision0710.external_validation.stage_receipt.v8",
        "status": "STAGE_COMPLETE_CREATE_ONCE_REUSABLE",
        "stage_name": stage_name,
        "environment_lock_sha256": environment_sha256,
        "input_bindings": list(input_bindings),
        "outputs": file_manifest(output_paths, root=stage_root),
    }
    _atomic_json_create(receipt_path, payload)
    return payload


def run_or_resume_stage(
    *,
    stage_name: str,
    stage_root: Path,
    input_bindings: Sequence[Mapping[str, Any]],
    environment_sha256: str,
    runner: Callable[[Path], Sequence[Path]],
) -> dict[str, Any]:
    """Preserve partial work and reuse only a fully rehashed completion receipt."""
    stage_root.mkdir(parents=True, exist_ok=True)
    receipt = stage_root / "STAGE_COMPLETION_V8.json"
    if receipt.is_file():
        return validate_stage_receipt(
            receipt,
            stage_name=stage_name,
            stage_root=stage_root,
            input_bindings=input_bindings,
            environment_sha256=environment_sha256,
        )
    outputs = list(runner(stage_root))
    return commit_stage_receipt(
        receipt,
        stage_name=stage_name,
        stage_root=stage_root,
        input_bindings=input_bindings,
        output_paths=outputs,
        environment_sha256=environment_sha256,
    )
