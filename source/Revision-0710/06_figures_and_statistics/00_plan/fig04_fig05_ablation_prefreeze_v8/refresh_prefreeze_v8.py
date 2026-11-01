from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
CONTRACT_PATH = ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"
CACHE_DIR = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "04_cache_v1"

ARTIFACT_FILES = {
    "queue_v8": "ABLATION_QUEUE_V8.csv",
    "authorization_schema_v8": "FORMAL_AUTHORIZATION_SCHEMA_V8.json",
    "formal_executor_v8": "formal_executor.py",
    "isolated_worker_v8": "isolated_worker.py",
    "output_schemas_v8": "OUTPUT_SCHEMAS_V8.json",
    "resource_budget_v8": "RESOURCE_BUDGET_AND_PREFLIGHT_V8.json",
    "scaled_integration_runner_v8": "run_scaled_integration_smoke.py",
    "authorization_validator_v8": "v8_authorization.py",
    "contract_validator_v8": "v8_contract.py",
    "data_roles_v8": "v8_data.py",
    "input_ablation_v8": "v8_inputs.py",
    "loss_ablation_v8": "v8_loss.py",
    "model_ablation_v8": "v8_model.py",
    "schema_validator_v8": "v8_schema_validator.py",
    "state_machine_v8": "v8_state_machine.py",
    "training_v8": "v8_training.py",
    "independent_replay_v8": "v8_independent_replay.py",
    "formal_independent_verifier_v8": "verify_formal_attempt_v8_independent.py",
    "variant_registry_v8": "VARIANT_REGISTRY_V8.json",
    "tests_conftest": "tests/conftest.py",
    "tests_authorization_executor": "tests/test_authorization_executor.py",
    "tests_contract_queue": "tests/test_contract_queue_protocol.py",
    "tests_model_training": "tests/test_model_training_wiring.py",
    "tests_schema_state": "tests/test_schema_state_machine.py",
    "tests_v8_integrity": "tests/test_v8_integrity.py",
    "cuda_production_shape_preflight_v8": "run_cuda_production_shape_preflight.py",
    "worker_access_guard_v8": "worker_access_guard.py",
    "terminal_go_schema_v8": "TERMINAL_GO_SCHEMA_V8.json",
    "fresh_test_result_schema_v8": "FRESH_TEST_RESULT_SCHEMA_V8.json",
    "independent_verifier_v8": "verify_prefreeze_v8_independent.py",
    "refresh_prefreeze_v8": "refresh_prefreeze_v8.py",
    "runtime_guard_v8": "v8_runtime_guard.py",
    "tests_deep_replay_v8": "tests/test_deep_replay_v8.py",
    "finalize_prefreeze_v8": "finalize_prefreeze_v8.py",
    "readme_v8": "README_V8.md",
    "re_review_request_v8": "INDEPENDENT_RE_REVIEW_REQUEST_V8.md",
    "v7_terminal_review_no_go": "V7_TERMINAL_REVIEW_NO_GO.md",
    "tests_v8_terminal_hardening": "tests/test_v8_terminal_hardening.py",
}

V1_RUNTIME_FILES = {
    "ablation_model": "ablation_model.py",
    "ablation_loss": "ablation_loss.py",
    "ablation_inputs": "ablation_inputs.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact_record(name: str, path: Path, *, stored_path: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"bound artifact is missing: {name}: {path}")
    return {
        "name": name,
        "path": stored_path.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_cache_bindings() -> dict[str, object]:
    batch_manifest_path = CACHE_DIR / "batch_manifest.json"
    transform_manifest_path = CACHE_DIR / "fold_transform_manifest.json"
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    transform_manifest = json.loads(transform_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(batch_manifest, list) or len(batch_manifest) != 60:
        raise RuntimeError("frozen batch manifest must contain exactly 60 rows")
    if not isinstance(transform_manifest, list) or len(transform_manifest) != 5:
        raise RuntimeError("frozen transform manifest must contain exactly five rows")
    shards = []
    for ordinal, row in enumerate(batch_manifest):
        path = CACHE_DIR / Path(row["path"])
        shards.append(
            {
                "ordinal": ordinal,
                "path": str(row["path"]).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "fold_id": int(row["fold_id"]),
                "batch_in_fold": int(row["batch_in_fold"]),
                "n": int(row["n"]),
                "active_events": int(row["active_events"]),
                "storage_active_events": int(row["storage_active_events"]),
            }
        )
        if shards[-1]["sha256"] != row["sha256"]:
            raise RuntimeError(f"batch manifest SHA disagrees with shard: {row['path']}")
    transforms = []
    for ordinal, row in enumerate(transform_manifest):
        path = CACHE_DIR / Path(row["path"])
        transforms.append(
            {
                "ordinal": ordinal,
                "path": str(row["path"]).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "outer_fold": int(row["outer_fold"]),
                "validation_fold": int(row["validation_fold"]),
                "train_folds": [int(value) for value in row["train_folds"]],
                "train_n": int(row["train_n"]),
                "feature_hash": str(row["feature_hash"]),
                "train_id_hash": str(row["train_id_hash"]),
            }
        )
        if transforms[-1]["sha256"] != row["sha256"]:
            raise RuntimeError(f"transform manifest SHA disagrees with file: {row['path']}")
    return {
        "schema_version": "revision0710.fig04_fig05.cache_artifact_bindings.v8",
        "batch_manifest": artifact_record(
            "batch_manifest", batch_manifest_path, stored_path="batch_manifest.json"
        ),
        "shards": shards,
        "fold_transform_manifest": artifact_record(
            "fold_transform_manifest",
            transform_manifest_path,
            stored_path="fold_transform_manifest.json",
        ),
        "fold_transforms": transforms,
    }


def refresh_contract() -> dict[str, object]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    records = []
    for name, relative in ARTIFACT_FILES.items():
        path = ROOT / relative
        stored = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        records.append(artifact_record(name, path, stored_path=stored))
    payload["artifact_bindings"] = {"records": records}
    v1_root = REPO_ROOT / "Revision-0710" / "06_figures_and_statistics" / "00_plan" / "fig04_fig05_ablation_prefreeze_v1"
    v1_records = {
        str(record["name"]): record
        for record in payload["v1_immutable_bindings"]["records"]
    }
    for name, relative in V1_RUNTIME_FILES.items():
        path = v1_root / relative
        v1_records[name] = artifact_record(
            name,
            path,
            stored_path=path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        )
    payload["v1_immutable_bindings"] = {
        "records": [v1_records[name] for name in sorted(v1_records)]
    }
    payload["cache_artifact_bindings"] = build_cache_bindings()
    write_json_atomic(CONTRACT_PATH, payload)
    return {
        "status": "CONTRACT_ARTIFACT_BINDINGS_REFRESHED_NOT_AUTHORIZED",
        "contract_path": str(CONTRACT_PATH.resolve()),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "artifact_count": len(records),
        "artifact_names": [record["name"] for record in records],
        "cache_shard_count": len(payload["cache_artifact_bindings"]["shards"]),
        "fold_transform_count": len(payload["cache_artifact_bindings"]["fold_transforms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(refresh_contract(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
