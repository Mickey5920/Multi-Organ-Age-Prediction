from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import v8_contract
import v8_schema_validator


class StateError(RuntimeError):
    pass


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise StateError(f"create-only artifact already exists: {path}") from exc


def claim_specific_attempt(
    run_root: Path,
    *,
    ordinal: int,
    run_id: str,
    seed: int,
) -> Path:
    if ordinal < 1:
        raise StateError("attempt ordinal must be positive")
    attempt = run_root / f"attempt_{ordinal:04d}"
    try:
        attempt.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise StateError(f"attempt already exists: {attempt.name}") from exc
    try:
        _write_json_exclusive(
            attempt / "CLAIM.json",
            {
                "schema_version": "revision0710.fig04_fig05.attempt_claim.v8",
                "status": "CLAIMED_CREATE_ONLY",
                "attempt_ordinal": ordinal,
                "run_id": run_id,
                "seed": int(seed),
                "restart_epoch": 1,
                "resume_training_state": False,
                "selection_uses_metrics": False,
            },
        )
    except Exception:
        attempt.rmdir()
        raise
    return attempt


def claim_new_attempt(run_root: Path, *, run_id: str, seed: int) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    for ordinal in range(1, 10000):
        try:
            return claim_specific_attempt(
                run_root, ordinal=ordinal, run_id=run_id, seed=seed
            )
        except StateError as exc:
            if "already exists" not in str(exc):
                raise
    raise StateError("attempt ordinal space exhausted")


def write_failure_receipt(attempt: Path, *, phase: str, reason: str) -> Path:
    if (attempt / "COMPLETE.json").exists():
        raise StateError("completed attempt cannot be failed")
    path = attempt / "FAILURE.json"
    _write_json_exclusive(
        path,
        {
            "schema_version": "revision0710.fig04_fig05.attempt_failure.v8",
            "status": "FAILED_SEALED_CREATE_NEW_ATTEMPT",
            "phase": phase,
            "reason": reason,
            "resume_allowed": False,
            "restart_epoch": 1,
        },
    )
    claim = json.loads((attempt / "CLAIM.json").read_text(encoding="utf-8"))
    receipt = {
        "schema_version": "revision0710.fig04_fig05.attempt_receipt.v8",
        "status": "FAILED",
        "run_id": claim["run_id"],
        "attempt_ordinal": int(claim["attempt_ordinal"]),
        "seed": int(claim["seed"]),
        "restart_epoch": 1,
        "training_resume_used": False,
        "fold_receipt_sha256": None,
        "failure_receipt_sha256": v8_contract.sha256_file(path),
        "selection_uses_metrics": False,
    }
    schemas = v8_schema_validator.load_schemas(
        Path(__file__).resolve().parent / "OUTPUT_SCHEMAS_V8.json"
    )
    v8_schema_validator.validate_json_payload(
        receipt, schemas["tables"]["attempt_receipt"], table="attempt_receipt"
    )
    _write_json_exclusive(attempt / "ATTEMPT_RECEIPT.json", receipt)
    return path


def write_completion_receipt(
    attempt: Path, *, run_id: str, seed: int, fold_receipt_path: Path
) -> Path:
    if (attempt / "FAILURE.json").exists():
        raise StateError("failed attempt cannot be completed")
    claim = json.loads((attempt / "CLAIM.json").read_text(encoding="utf-8"))
    if claim.get("run_id") != run_id or int(claim.get("seed", -1)) != int(seed):
        raise StateError("completion identity differs from claim")
    if not fold_receipt_path.is_file():
        raise StateError("fold receipt is required before attempt completion")
    path = attempt / "COMPLETE.json"
    _write_json_exclusive(
        path,
        {
            "schema_version": "revision0710.fig04_fig05.attempt_complete.v8",
            "status": "COMPLETE",
            "run_id": run_id,
            "seed": int(seed),
            "attempt_ordinal": int(claim["attempt_ordinal"]),
            "performance_metric_used_for_selection": False,
        },
    )
    receipt = {
        "schema_version": "revision0710.fig04_fig05.attempt_receipt.v8",
        "status": "COMPLETE",
        "run_id": run_id,
        "attempt_ordinal": int(claim["attempt_ordinal"]),
        "seed": int(seed),
        "restart_epoch": 1,
        "training_resume_used": False,
        "fold_receipt_sha256": v8_contract.sha256_file(fold_receipt_path),
        "failure_receipt_sha256": None,
        "selection_uses_metrics": False,
    }
    schemas = v8_schema_validator.load_schemas(
        Path(__file__).resolve().parent / "OUTPUT_SCHEMAS_V8.json"
    )
    v8_schema_validator.validate_json_payload(
        receipt, schemas["tables"]["attempt_receipt"], table="attempt_receipt"
    )
    _write_json_exclusive(attempt / "ATTEMPT_RECEIPT.json", receipt)
    return path


def select_completed_attempt(run_root: Path) -> Path:
    completed = []
    for attempt in sorted(run_root.glob("attempt_[0-9][0-9][0-9][0-9]")):
        receipt = attempt / "COMPLETE.json"
        if receipt.is_file() and not (attempt / "FAILURE.json").exists():
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if payload.get("status") == "COMPLETE":
                completed.append((int(payload["attempt_ordinal"]), attempt))
    if not completed:
        raise StateError("no valid completed attempt")
    return min(completed, key=lambda item: item[0])[1]


def training_resume_allowed(*, completed_epochs: int, target_epochs: int) -> bool:
    del completed_epochs, target_epochs
    return False


def evaluation_resume_allowed(*, training_complete: bool, e55_sealed: bool) -> bool:
    return bool(training_complete and e55_sealed)


def validate_complete_set_for_aggregate(receipts: dict[str, str]) -> None:
    expected = set(v8_contract.expected_run_ids())
    observed = set(receipts)
    if observed != expected or len(receipts) != 60:
        raise StateError(
            f"aggregate/source gate requires complete 60/60; observed={len(receipts)}"
        )
    if any(not isinstance(value, str) or len(value) != 64 for value in receipts.values()):
        raise StateError("aggregate receipt hashes are invalid")
