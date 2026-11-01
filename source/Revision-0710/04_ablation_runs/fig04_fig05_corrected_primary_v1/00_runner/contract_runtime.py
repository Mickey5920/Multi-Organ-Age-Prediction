from __future__ import annotations

"""Frozen-contract helpers for the corrected-primary Fig4/Fig5 queue."""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


CONTRACT_SCHEMA = "revision0710.fig04_fig05.corrected_primary.contract.v1"
QUEUE_COLUMNS = (
    "ordinal",
    "run_id",
    "figure",
    "variant_id",
    "outer_fold",
    "calibration_fold",
    "train_folds",
    "seed",
    "initialization",
    "train_through_epoch",
    "capture_epoch",
    "evaluate_only_epoch",
    "status",
    "intended_run_dir",
)
ROLES = ("train", "calibration", "outer_test")
ROLE_DIRECTORIES = {
    "train": "01_train",
    "calibration": "02_calibration",
    "outer_test": "03_outer_test",
}
FROZEN_QUEUE_STATUS = "FROZEN_CORRECTED_PRIMARY_V1"


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.resolve()
    if not _inside(resolved, repo_root):
        raise ContractError(f"Bound path escapes repository: {value}")
    return resolved


def repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    if not _inside(resolved, repo_root):
        raise ContractError(f"File is outside repository: {path}")
    return resolved.relative_to(repo_root.resolve()).as_posix()


def file_record(name: str, path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"Missing file for binding {name}: {resolved}")
    return {
        "name": name,
        "path": repo_relative(resolved, repo_root),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_record(record: dict[str, Any], repo_root: Path) -> Path:
    required = {"name", "path", "bytes", "sha256"}
    if set(record) != required:
        raise ContractError(f"File record schema changed: {record.get('name', '<unknown>')}")
    path = resolve_path(str(record["path"]), repo_root)
    if not path.is_file():
        raise ContractError(f"Bound file is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise ContractError(f"Bound file size changed: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise ContractError(f"Bound file SHA changed: {path}")
    return path


def read_queue(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != QUEUE_COLUMNS:
            raise ContractError(f"Queue columns changed: {path}")
        return [dict(row) for row in reader]


def write_queue_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def expected_fold_ids(outer_fold: int, role: str) -> tuple[int, ...]:
    calibration_fold = (outer_fold + 1) % 5
    if role == "outer_test":
        return (outer_fold,)
    if role == "calibration":
        return (calibration_fold,)
    if role == "train":
        return tuple(fold for fold in range(5) if fold not in {outer_fold, calibration_fold})
    raise ContractError(f"Unknown role: {role}")


def validate_source_queue(rows: list[dict[str, str]], variant_ids: tuple[str, ...]) -> None:
    if len(rows) != 60:
        raise ContractError(f"Expected 60 source queue rows, found {len(rows)}")
    expected_pairs = [(variant, fold) for variant in variant_ids for fold in range(5)]
    observed_pairs: list[tuple[str, int]] = []
    for ordinal, row in enumerate(rows, start=1):
        outer_fold = int(row["outer_fold"])
        variant_id = row["variant_id"]
        observed_pairs.append((variant_id, outer_fold))
        expected_train = "|".join(str(value) for value in expected_fold_ids(outer_fold, "train"))
        checks = {
            "ordinal": int(row["ordinal"]) == ordinal,
            "run_id": row["run_id"] == f"{variant_id}__outer{outer_fold}",
            "calibration_fold": int(row["calibration_fold"]) == (outer_fold + 1) % 5,
            "train_folds": row["train_folds"] == expected_train,
            "seed": int(row["seed"]) == 7210,
            "initialization": row["initialization"] == "fresh_random",
            "train_through_epoch": int(row["train_through_epoch"]) == 60,
            "capture_epoch": int(row["capture_epoch"]) == 55,
            "evaluate_only_epoch": int(row["evaluate_only_epoch"]) == 55,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ContractError(f"Source queue row {ordinal} violates: {', '.join(failed)}")
    if observed_pairs != expected_pairs:
        raise ContractError("Source queue variant/fold order differs from the V8 registry")


def build_queue_snapshot(
    source_rows: list[dict[str, str]], *, run_root: Path
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in source_rows:
        row = dict(source)
        ordinal = int(row["ordinal"])
        row["status"] = FROZEN_QUEUE_STATUS
        row["intended_run_dir"] = str(
            (run_root / "01_tasks" / f"{ordinal:02d}_{row['run_id']}").resolve()
        )
        rows.append(row)
    return rows


def validate_frozen_queue(rows: list[dict[str, str]], *, run_root: Path) -> None:
    variant_ids: list[str] = []
    for row in rows:
        if row["variant_id"] not in variant_ids:
            variant_ids.append(row["variant_id"])
    validate_source_queue(rows, tuple(variant_ids))
    if len(variant_ids) != 12:
        raise ContractError(f"Expected 12 variants, found {len(variant_ids)}")
    for row in rows:
        if row["status"] != FROZEN_QUEUE_STATUS:
            raise ContractError("Frozen queue status changed")
        expected = (
            run_root
            / "01_tasks"
            / f"{int(row['ordinal']):02d}_{row['run_id']}"
        ).resolve()
        if Path(row["intended_run_dir"]).resolve() != expected:
            raise ContractError(f"Frozen queue task path changed: {row['run_id']}")


def _require_inside_role_root(path: Path, role_root: Path, label: str) -> None:
    if not _inside(path, role_root):
        raise ContractError(f"{label} is outside corrected role root: {path}")


def _assert_source_binding(
    role_manifest: dict[str, Any],
    key: str,
    expected_record: dict[str, Any],
    repo_root: Path,
) -> None:
    observed = role_manifest.get("source_bindings", {}).get(key)
    if not isinstance(observed, dict):
        raise ContractError(f"Role manifest lacks corrected source binding: {key}")
    observed_path = resolve_path(str(observed.get("path", "")), repo_root)
    expected_path = resolve_path(str(expected_record["path"]), repo_root)
    if observed_path != expected_path or observed.get("sha256") != expected_record["sha256"]:
        raise ContractError(f"Role manifest source binding is not corrected-primary: {key}")


def inspect_role_artifact_chain(
    *,
    role_root: Path,
    repo_root: Path,
    corrected_sources: dict[str, dict[str, Any]],
    verify_payloads: bool,
) -> dict[str, Any]:
    role_root = role_root.resolve()
    _require_inside_role_root(role_root, repo_root, "Role root")
    registry_path = role_root / "ROLE_ARTIFACT_REGISTRY.FROZEN.json"
    validation_path = role_root / "ROLE_ARTIFACT_VALIDATION.PASS.json"
    registry_record = file_record("role_registry", registry_path, repo_root)
    validation_record = file_record("role_validation", validation_path, repo_root)
    registry = read_json(registry_path)
    validation = read_json(validation_path)
    if validation.get("status") != "PASS":
        raise ContractError("Corrected role validation is not PASS")
    if validation.get("role_registry_sha256") != registry_record["sha256"]:
        raise ContractError("Role validation does not bind the corrected registry SHA")
    if validation.get("all_role_sets_pairwise_disjoint_and_exhaustive") is not True:
        raise ContractError("Corrected role validation lacks disjoint/exhaustive PASS")
    folds = registry.get("folds")
    if not isinstance(folds, list) or [int(row.get("outer_fold", -1)) for row in folds] != list(range(5)):
        raise ContractError("Corrected role registry must contain outer folds 0-4 in order")

    fold_records: list[dict[str, Any]] = []
    for registry_fold in folds:
        outer_fold = int(registry_fold["outer_fold"])
        registry_outer = registry_fold.get("outer_role_manifest")
        if not isinstance(registry_outer, dict):
            raise ContractError(f"Registry lacks outer-role binding: fold {outer_fold}")
        outer_path = resolve_path(str(registry_outer.get("path", "")), repo_root)
        _require_inside_role_root(outer_path, role_root, "Outer role manifest")
        outer_record = file_record(f"outer_fold_{outer_fold}_manifest", outer_path, repo_root)
        if registry_outer.get("sha256") != outer_record["sha256"]:
            raise ContractError(f"Registry outer manifest SHA mismatch: fold {outer_fold}")
        outer_manifest = read_json(outer_path)
        if int(outer_manifest.get("outer_fold", -1)) != outer_fold:
            raise ContractError(f"Outer manifest identity mismatch: fold {outer_fold}")
        if outer_manifest.get("roles_are_physically_separate") is not True:
            raise ContractError(f"Roles are not declared physically separate: fold {outer_fold}")
        roles = outer_manifest.get("roles")
        if not isinstance(roles, dict) or set(roles) != set(ROLES):
            raise ContractError(f"Outer manifest role set changed: fold {outer_fold}")

        role_records: dict[str, Any] = {}
        for role in ROLES:
            outer_role = roles[role]
            role_manifest_binding = outer_role.get("role_manifest")
            if not isinstance(role_manifest_binding, dict):
                raise ContractError(f"Missing role manifest binding: fold {outer_fold} {role}")
            manifest_path = resolve_path(str(role_manifest_binding.get("path", "")), repo_root)
            expected_manifest_path = role_root / f"outer_fold_{outer_fold}" / ROLE_DIRECTORIES[role] / "role_manifest.json"
            if manifest_path != expected_manifest_path.resolve():
                raise ContractError(f"Unexpected role manifest path: fold {outer_fold} {role}")
            manifest_record = file_record(
                f"outer_fold_{outer_fold}_{role}_manifest", manifest_path, repo_root
            )
            if role_manifest_binding.get("sha256") != manifest_record["sha256"]:
                raise ContractError(f"Outer manifest role SHA mismatch: fold {outer_fold} {role}")
            manifest = read_json(manifest_path)
            if manifest.get("status") != "SEALED":
                raise ContractError(f"Role manifest is not SEALED: fold {outer_fold} {role}")
            if manifest.get("role") != role or int(manifest.get("outer_fold", -1)) != outer_fold:
                raise ContractError(f"Role manifest identity mismatch: fold {outer_fold} {role}")
            artifact = manifest.get("artifact")
            if not isinstance(artifact, dict):
                raise ContractError(f"Role artifact missing: fold {outer_fold} {role}")
            fold_ids = tuple(int(value) for value in artifact.get("fold_ids", []))
            if fold_ids != expected_fold_ids(outer_fold, role):
                raise ContractError(f"Role fold assignment changed: fold {outer_fold} {role}")
            if list(outer_role.get("fold_ids", [])) != list(fold_ids):
                raise ContractError(f"Outer/single role fold IDs disagree: fold {outer_fold} {role}")
            for source_key, expected in corrected_sources.items():
                _assert_source_binding(manifest, source_key, expected, repo_root)

            bound_artifacts: dict[str, Any] = {}
            for artifact_name in ("age_targets", "joint_labels"):
                item = artifact.get(artifact_name)
                outer_item = outer_role.get(artifact_name)
                if not isinstance(item, dict) or not isinstance(outer_item, dict):
                    raise ContractError(
                        f"Missing {artifact_name} binding: fold {outer_fold} {role}"
                    )
                path = resolve_path(str(item.get("path", "")), repo_root)
                _require_inside_role_root(path, role_root, artifact_name)
                record = file_record(
                    f"outer_fold_{outer_fold}_{role}_{artifact_name}", path, repo_root
                )
                if item.get("sha256") != record["sha256"]:
                    raise ContractError(
                        f"Role {artifact_name} SHA mismatch: fold {outer_fold} {role}"
                    )
                if outer_item.get("sha256") != record["sha256"]:
                    raise ContractError(
                        f"Outer {artifact_name} SHA mismatch: fold {outer_fold} {role}"
                    )
                if resolve_path(str(outer_item.get("path", "")), repo_root) != path:
                    raise ContractError(
                        f"Outer/single {artifact_name} paths disagree: fold {outer_fold} {role}"
                    )
                if verify_payloads:
                    verify_file_record(record, repo_root)
                bound_artifacts[artifact_name] = record

            for key in ("participant_count", "participant_organ_rows", "participant_ids_canonical_sha256"):
                if outer_role.get(key) != artifact.get(key):
                    raise ContractError(f"Outer/single role {key} disagree: fold {outer_fold} {role}")
            role_records[role] = {
                "manifest": manifest_record,
                "fold_ids": list(fold_ids),
                "participant_count": int(artifact["participant_count"]),
                "participant_organ_rows": int(artifact["participant_organ_rows"]),
                "participant_ids_canonical_sha256": str(
                    artifact["participant_ids_canonical_sha256"]
                ),
                **bound_artifacts,
            }
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "outer_manifest": outer_record,
                "roles": role_records,
            }
        )

    return {
        "role_root": repo_relative(role_root, repo_root),
        "registry": registry_record,
        "validation": validation_record,
        "registry_schema_version": registry.get("schema_version"),
        "validation_schema_version": validation.get("schema_version"),
        "folds": fold_records,
    }


def iter_role_file_records(role_artifacts: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield role_artifacts["registry"]
    yield role_artifacts["validation"]
    for fold in role_artifacts["folds"]:
        yield fold["outer_manifest"]
        for role in ROLES:
            role_record = fold["roles"][role]
            yield role_record["manifest"]
            yield role_record["age_targets"]
            yield role_record["joint_labels"]


def get_role_record(contract: dict[str, Any], outer_fold: int, role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise ContractError(f"Unknown role: {role}")
    matches = [
        fold for fold in contract["role_artifacts"]["folds"]
        if int(fold["outer_fold"]) == int(outer_fold)
    ]
    if len(matches) != 1:
        raise ContractError(f"Missing corrected role fold: {outer_fold}")
    return matches[0]["roles"][role]


def validate_frozen_contract(
    *,
    contract_path: Path,
    queue_path: Path,
    repo_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    contract = read_json(contract_path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ContractError("Runner accepts only corrected-primary contract v1")
    if contract.get("status") != "FROZEN_CORRECTED_PRIMARY_V1":
        raise ContractError("Corrected-primary contract is not frozen")
    if resolve_path(str(contract.get("run_root", "")), repo_root) != run_root.resolve():
        raise ContractError("Frozen contract run root changed")
    queue_binding = contract.get("queue")
    if not isinstance(queue_binding, dict):
        raise ContractError("Frozen queue binding missing")
    if sha256_file(queue_path) != queue_binding.get("sha256"):
        raise ContractError("Frozen queue SHA changed")
    if queue_path.stat().st_size != int(queue_binding.get("bytes", -1)):
        raise ContractError("Frozen queue size changed")
    rows = read_queue(queue_path)
    validate_frozen_queue(rows, run_root=run_root)
    if len(rows) != int(queue_binding.get("rows", -1)):
        raise ContractError("Frozen queue row count changed")

    for section in ("behavior_bindings", "corrected_primary_bindings", "runtime_bindings"):
        records = contract.get(section)
        if not isinstance(records, list) or not records:
            raise ContractError(f"Missing frozen binding section: {section}")
        names = [str(record.get("name")) for record in records]
        if len(names) != len(set(names)):
            raise ContractError(f"Duplicate binding names: {section}")
        for record in records:
            verify_file_record(record, repo_root)

    role_artifacts = contract.get("role_artifacts")
    if not isinstance(role_artifacts, dict):
        raise ContractError("Corrected role artifact bindings missing")
    for record in iter_role_file_records(role_artifacts):
        verify_file_record(record, repo_root)
    registry = read_json(verify_file_record(role_artifacts["registry"], repo_root))
    validation = read_json(verify_file_record(role_artifacts["validation"], repo_root))
    if validation.get("status") != "PASS":
        raise ContractError("Role validation is no longer PASS")
    if validation.get("role_registry_sha256") != sha256_file(
        resolve_path(str(role_artifacts["registry"]["path"]), repo_root)
    ):
        raise ContractError("Role validation/registry SHA link changed")
    if [int(item.get("outer_fold", -1)) for item in registry.get("folds", [])] != list(range(5)):
        raise ContractError("Role registry fold coverage changed")
    return contract


def stop_requested(run_root: Path) -> bool:
    return (run_root / "STOP_AFTER_CURRENT").is_file()
