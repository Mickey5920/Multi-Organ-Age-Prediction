from __future__ import annotations

"""Freeze the corrected-primary Fig4/Fig5 queue after role materialization."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contract_runtime import (
    CONTRACT_SCHEMA,
    ContractError,
    build_queue_snapshot,
    canonical_json_sha256,
    file_record,
    inspect_role_artifact_chain,
    read_json,
    read_queue,
    sha256_file,
    validate_frozen_contract,
    validate_source_queue,
    write_json_atomic,
    write_queue_atomic,
)


RUNNER_ROOT = Path(__file__).resolve().parent
RUN_ROOT = RUNNER_ROOT.parent
REPO_ROOT = RUNNER_ROOT.parents[3]
V8_ROOT = (
    REPO_ROOT
    / "Revision-0710"
    / "06_figures_and_statistics"
    / "00_plan"
    / "fig04_fig05_ablation_prefreeze_v8"
)
CORRECTED_ROOT = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "07_corrected_primary_v1"
ROLE_ROOT = (
    REPO_ROOT
    / "Revision-0710"
    / "03_ehr_age"
    / "05_corrected_primary_v1"
    / "00_role_artifacts"
)
SOURCE_QUEUE_PATH = V8_ROOT / "ABLATION_QUEUE_V8.csv"
VARIANT_REGISTRY_PATH = V8_ROOT / "VARIANT_REGISTRY_V8.json"
SOURCE_CONTRACT_PATH = V8_ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"
QUEUE_PATH = RUNNER_ROOT / "ABLATION_QUEUE_CORRECTED_PRIMARY_V1.csv"
CONTRACT_PATH = RUNNER_ROOT / "FROZEN_CONTRACT_CORRECTED_PRIMARY_V1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_status(path: Path, expected: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != expected:
        raise ContractError(f"Required {expected} status missing: {path}")
    return payload


def _records(paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [file_record(name, path, REPO_ROOT) for name, path in paths]


def _record_by_name(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [record for record in records if record["name"] == name]
    if len(matches) != 1:
        raise ContractError(f"Expected one binding named {name}")
    return matches[0]


def corrected_primary_bindings() -> list[dict[str, Any]]:
    codebook_root = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "06_icd_codebook_v1"
    paths = [
        ("corrected_rebuild_manifest", CORRECTED_ROOT / "MANIFEST.json"),
        ("corrected_rebuild_status", CORRECTED_ROOT / "RUN_STATUS.json"),
        (
            "corrected_delta_and_leakage_audit",
            CORRECTED_ROOT / "04_delta_audit" / "DELTA_AND_LEAKAGE_AUDIT.json",
        ),
        (
            "corrected_icd_rebuild_audit",
            CORRECTED_ROOT / "01_index_and_events" / "icd_rebuild_audit.json",
        ),
        (
            "corrected_outcome_rebuild_audit",
            CORRECTED_ROOT / "02_outcomes_60m" / "outcome_rebuild_audit.json",
        ),
        (
            "corrected_mapping",
            codebook_root / "02_v1_corrected_primary" / "mapping.csv",
        ),
        ("corrected_codebook_manifest", codebook_root / "MANIFEST.json"),
        (
            "corrected_mapping_export",
            CORRECTED_ROOT / "01_index_and_events" / "icd_mapping_export.csv",
        ),
        (
            "corrected_organ_outcomes",
            CORRECTED_ROOT / "02_outcomes_60m" / "organ_outcomes_60m.parquet",
        ),
        (
            "corrected_participant_masks",
            CORRECTED_ROOT
            / "02_outcomes_60m"
            / "participant_followup_and_evaluation_masks.parquet",
        ),
        (
            "corrected_fold_manifest",
            CORRECTED_ROOT / "03_folds" / "fold_manifest_5fold.parquet",
        ),
        (
            "corrected_outer_fold_roles",
            CORRECTED_ROOT / "03_folds" / "outer_fold_roles.parquet",
        ),
    ]
    records = _records(paths)
    rebuild_manifest = _require_status(CORRECTED_ROOT / "MANIFEST.json", "PASS")
    _require_status(CORRECTED_ROOT / "RUN_STATUS.json", "COMPLETE")
    _require_status(
        CORRECTED_ROOT / "04_delta_audit" / "DELTA_AND_LEAKAGE_AUDIT.json", "PASS"
    )
    if rebuild_manifest.get("audit_summary", {}).get("status") != "PASS":
        raise ContractError("Corrected rebuild audit summary is not PASS")
    if rebuild_manifest.get("mapping", {}).get("sha256") != _record_by_name(
        records, "corrected_mapping"
    )["sha256"]:
        raise ContractError("Corrected rebuild manifest does not bind corrected mapping SHA")
    output_bindings = rebuild_manifest.get("outputs", {})
    expected_output_shas = {
        "02_outcomes_60m\\organ_outcomes_60m.parquet": "corrected_organ_outcomes",
        "02_outcomes_60m\\participant_followup_and_evaluation_masks.parquet": "corrected_participant_masks",
        "03_folds\\fold_manifest_5fold.parquet": "corrected_fold_manifest",
        "03_folds\\outer_fold_roles.parquet": "corrected_outer_fold_roles",
        "04_delta_audit\\DELTA_AND_LEAKAGE_AUDIT.json": "corrected_delta_and_leakage_audit",
    }
    for manifest_key, record_name in expected_output_shas.items():
        item = output_bindings.get(manifest_key)
        if not isinstance(item, dict):
            # JSON written on non-Windows platforms may preserve forward slashes.
            item = output_bindings.get(manifest_key.replace("\\", "/"))
        if not isinstance(item, dict) or item.get("sha256") != _record_by_name(
            records, record_name
        )["sha256"]:
            raise ContractError(f"Corrected rebuild manifest output SHA mismatch: {manifest_key}")
    return records


def behavior_bindings() -> list[dict[str, Any]]:
    model_root = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "01_model"
    runner_root = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "02_runner"
    return _records(
        [
            ("source_v8_queue", SOURCE_QUEUE_PATH),
            ("source_v8_variant_registry", VARIANT_REGISTRY_PATH),
            ("source_v8_contract", SOURCE_CONTRACT_PATH),
            ("v8_contract", V8_ROOT / "v8_contract.py"),
            ("v8_data", V8_ROOT / "v8_data.py"),
            ("v8_inputs", V8_ROOT / "v8_inputs.py"),
            ("v8_loss", V8_ROOT / "v8_loss.py"),
            ("v8_model", V8_ROOT / "v8_model.py"),
            ("v8_training", V8_ROOT / "v8_training.py"),
            ("ehr_age_model", model_root / "ehr_age_model.py"),
            ("private_retain_age", model_root / "private_retain_age.py"),
            ("ehr_age_losses", model_root / "ehr_age_losses.py"),
            ("ehr_age_data", runner_root / "ehr_age_data.py"),
            (
                "compact_cache_dataset",
                REPO_ROOT
                / "Revision-0710"
                / "02_baselines_5fold"
                / "common"
                / "compact_cache_dataset.py",
            ),
            (
                "cache_hard_gate",
                REPO_ROOT
                / "Revision-0710"
                / "01_data_rebuild"
                / "04_cache_v1"
                / "cache_hard_gate.json",
            ),
            (
                "leakage_gate",
                REPO_ROOT
                / "Revision-0710"
                / "01_data_rebuild"
                / "05_leakage_audit"
                / "leakage_gate.json",
            ),
        ]
    )


def runtime_bindings() -> list[dict[str, Any]]:
    return _records(
        [
            ("corrected_contract_runtime", RUNNER_ROOT / "contract_runtime.py"),
            ("corrected_direct_runner", RUNNER_ROOT / "run_direct_5fold.py"),
            ("corrected_freeze_contract", RUNNER_ROOT / "freeze_contract.py"),
        ]
    )


def build_contract(role_root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    variant_registry = read_json(VARIANT_REGISTRY_PATH)
    variants = variant_registry.get("variants")
    if not isinstance(variants, list) or len(variants) != 12:
        raise ContractError("V8 registry must contain exactly 12 variants")
    variant_ids = tuple(str(record["variant_id"]) for record in variants)
    source_rows = read_queue(SOURCE_QUEUE_PATH)
    validate_source_queue(source_rows, variant_ids)
    queue_rows = build_queue_snapshot(source_rows, run_root=RUN_ROOT)

    corrected = corrected_primary_bindings()
    corrected_sources = {
        "corrected_organ_outcomes": _record_by_name(corrected, "corrected_organ_outcomes"),
        "corrected_participant_masks": _record_by_name(corrected, "corrected_participant_masks"),
        "corrected_fold_manifest": _record_by_name(corrected, "corrected_fold_manifest"),
        "corrected_outer_fold_roles": _record_by_name(corrected, "corrected_outer_fold_roles"),
    }
    role_artifacts = inspect_role_artifact_chain(
        role_root=role_root,
        repo_root=REPO_ROOT,
        corrected_sources=corrected_sources,
        verify_payloads=True,
    )
    source_contract = read_json(SOURCE_CONTRACT_PATH)
    cache_bindings = source_contract.get("cache_artifact_bindings")
    if not isinstance(cache_bindings, dict):
        raise ContractError("V8 source contract lacks cache artifact bindings")

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "FROZEN_CORRECTED_PRIMARY_V1",
        "created_utc": utc_now(),
        "run_root": RUN_ROOT.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "purpose": "Fresh formal Fig4/Fig5 12-variant x 5-fold queue on corrected-primary v1 labels",
        "protocol": {
            "variant_count": 12,
            "outer_fold_count": 5,
            "run_count": 60,
            "seed": 7210,
            "initialization": "fresh_random",
            "batch_size": 2048,
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "warmup_epochs": 3,
            "scheduler_horizon_epochs": 60,
            "train_through_epoch": 60,
            "capture_epoch": 55,
            "evaluate_only_epoch": 55,
            "age_correction_role": "calibration_only",
            "cox_nuisance_role": "calibration_only",
            "outer_test_use": "fixed_checkpoint_prediction_and_scoring_only",
            "performance_selection": False,
            "fold0_gate": {
                "run_id": "fig04_01_canonical_control__outer0",
                "automatic_pause_after_completion": True,
                "resume_requires_explicit_flag": "--continue-after-fold0-gate",
            },
        },
        "variant_ids": list(variant_ids),
        "behavior_bindings": behavior_bindings(),
        "corrected_primary_bindings": corrected,
        "runtime_bindings": runtime_bindings(),
        "cache_artifact_bindings": cache_bindings,
        "cache_artifact_bindings_sha256": canonical_json_sha256(cache_bindings),
        "role_artifacts": role_artifacts,
        "queue": {},
    }
    return contract, queue_rows


def freeze(role_root: Path) -> dict[str, Any]:
    if CONTRACT_PATH.exists():
        if not QUEUE_PATH.is_file():
            raise ContractError("Frozen contract exists without its queue snapshot")
        contract = validate_frozen_contract(
            contract_path=CONTRACT_PATH,
            queue_path=QUEUE_PATH,
            repo_root=REPO_ROOT,
            run_root=RUN_ROOT,
        )
        return {
            "status": "ALREADY_FROZEN_AND_VALID",
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "queue_sha256": sha256_file(QUEUE_PATH),
            "role_root": contract["role_artifacts"]["role_root"],
        }

    contract, queue_rows = build_contract(role_root)
    if QUEUE_PATH.exists():
        existing = read_queue(QUEUE_PATH)
        if existing != queue_rows:
            raise ContractError("Unbound queue snapshot already exists with different content")
    else:
        write_queue_atomic(QUEUE_PATH, queue_rows)
    contract["queue"] = {
        "path": QUEUE_PATH.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "bytes": QUEUE_PATH.stat().st_size,
        "sha256": sha256_file(QUEUE_PATH),
        "rows": len(queue_rows),
    }
    write_json_atomic(CONTRACT_PATH, contract)
    validate_frozen_contract(
        contract_path=CONTRACT_PATH,
        queue_path=QUEUE_PATH,
        repo_root=REPO_ROOT,
        run_root=RUN_ROOT,
    )
    return {
        "status": "FROZEN",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "queue_sha256": sha256_file(QUEUE_PATH),
        "role_root": contract["role_artifacts"]["role_root"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role-root",
        type=Path,
        default=ROLE_ROOT,
        help="Corrected-primary sealed role artifact root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate an existing frozen contract without creating files.",
    )
    args = parser.parse_args()
    if args.check:
        payload = validate_frozen_contract(
            contract_path=CONTRACT_PATH,
            queue_path=QUEUE_PATH,
            repo_root=REPO_ROOT,
            run_root=RUN_ROOT,
        )
        result = {
            "status": "PASS",
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "queue_sha256": sha256_file(QUEUE_PATH),
            "role_root": payload["role_artifacts"]["role_root"],
        }
    else:
        result = freeze(args.role_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
