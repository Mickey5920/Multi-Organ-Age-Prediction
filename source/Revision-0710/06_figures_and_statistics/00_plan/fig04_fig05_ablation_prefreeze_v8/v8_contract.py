from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "revision0710.fig04_fig05.ablation_prefreeze.contract.v8"
CACHE_BINDING_SCHEMA = "revision0710.fig04_fig05.cache_artifact_bindings.v8"
CACHE_RELATIVE_ROOT = Path("Revision-0710/01_data_rebuild/04_cache_v1")
ORGAN_SYSTEMS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
FIG04_VARIANTS = (
    "fig04_01_canonical_control",
    "fig04_02_no_organ_queries",
    "fig04_03_no_common_bidirectional_gru",
    "fig04_04_no_survival_supervision",
    "fig04_05_no_strict_offset_shrinkage",
    "fig04_06_no_consistency_objective",
    "fig04_07_seven_independent_organ_age_heads",
)
FIG05_VARIANTS = (
    "fig05_01_no_phenotypes_or_missingness",
    "fig05_02_no_hospital_icd",
    "fig05_03_no_gp_prescriptions",
    "fig05_04_no_event_text",
    "fig05_05_no_longitudinal_events",
)
VARIANT_IDS = (*FIG04_VARIANTS, *FIG05_VARIANTS)
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
EXPECTED_TRAINING_PROTOCOL = {
    "seed": 7210,
    "batch_size": 2048,
    "optimizer": "AdamW",
    "learning_rate": 0.0002,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "warmup_epochs": 3,
    "scheduler": "linear_warmup_then_cosine_decay",
    "scheduler_horizon_epochs": 60,
    "train_through_epoch": 60,
    "capture_epoch": 55,
    "evaluate_only_epoch": 55,
    "deterministic": True,
    "tf32": True,
    "amp": True,
    "restart_from_epoch_one_same_seed": True,
    "resume_training_state": False,
    "fresh_random_initialization_all_runs": True,
    "fold0_checkpoint_reuse": False,
    "warm_start": False,
    "fine_tune": False,
    "teacher": False,
    "distillation": False,
    "ensemble": False,
    "seed_retry": False,
    "performance_selection": False,
    "fold_drop": False,
}
EXPECTED_SCOPE = {
    "figures": ["Fig04", "Fig05"],
    "purpose": "Hash-bound formal comparison executor prefreeze with fresh canonical control and scaled integration evidence",
    "formal_training_executed": False,
    "formal_source_or_render_executed": False,
    "v1_modified": False,
}
EXPECTED_ENVIRONMENT = {
    "formal": {
        "python_executable": r"C:\Users\Mickey\.conda\envs\ICD-Prediction\python.exe",
        "python": "3.11.13",
        "torch": "2.5.1",
        "torch_cuda": "11.8",
        "cudnn": 90100,
        "numpy": "2.2.5",
        "pandas": "2.3.3",
        "scipy": "1.16.0",
        "scikit_learn": "1.7.1",
        "lifelines": "0.30.0",
        "device": "cuda",
        "gpu": "NVIDIA GeForce RTX 4090",
        "visible_gpu_count": 1,
        "cublas_workspace_config": ":4096:8",
    },
    "scaled_integration": {
        "device": "cpu",
        "formal_gpu_used": False,
        "formal_data_used": False,
    },
}
EXPECTED_EXECUTION_ROOTS = {
    "formal_run_root": r"E:\MPU\Code\Age-Prediction-0620\Revision-0710\04_ablation_runs\fig04_fig05_same_contract_5fold_v8",
    "scaled_smoke_root": r"E:\MPU\Code\Age-Prediction-0620\Revision-0710\06_figures_and_statistics\00_plan\fig04_fig05_ablation_prefreeze_v8\smoke",
    "formal_source_fig04": "Revision-0710/06_figures_and_statistics/05_source_data_by_figure/Fig04/source_data_v8",
    "formal_source_fig05": "Revision-0710/06_figures_and_statistics/05_source_data_by_figure/Fig05/source_data_v8",
    "formal_render_fig04": "Revision-0710/06_figures_and_statistics/06_rendered_figures/Fig04/render_v8_legacy_style",
    "formal_render_fig05": "Revision-0710/06_figures_and_statistics/06_rendered_figures/Fig05/render_v8_legacy_style",
}
EXPECTED_STATE_MACHINE = {
    "attempt_directories": "create_only_monotonic_attempt_ordinal",
    "failure_receipt_required": True,
    "failed_training_restart": "epoch_1_same_seed_fresh_state",
    "training_resume_allowed": False,
    "evaluation_resume_boundary": "training_complete_and_capture_sealed",
    "attempt_selection": "lowest_completed_attempt_ordinal_without_metric",
    "aggregate_gate": "exact_60_of_60_receipts",
    "gpu_lock": "os_backed_with_persistent_audit_and_stale_active_recovery",
    "independent_replay_resume": "create_only_cache_recomputed_from_primary_attempt_outputs_on_every_validation",
}
EXPECTED_AUTHORIZATION_PROTOCOL = {
    "formal_authorization_file": "FORMAL_AUTHORIZATION_V8.json",
    "must_be_absent_at_prefreeze": True,
    "exact_schema": "FORMAL_AUTHORIZATION_SCHEMA_V8.json",
    "all_60_together": True,
    "partial_authorization_allowed": False,
    "formal_executor_without_authorization": "HARD_FAIL",
    "hash_bindings_required": [
        "contract", "registry", "queue", "output_schema", "executor",
        "worker", "model", "loss", "input", "data", "schema_validator",
        "training", "contract_validator", "runtime_guard", "v1_runtime",
        "independent_replay", "formal_independent_verifier",
        "cache_artifact_bindings", "tests",
        "worker_access_guard", "independent_terminal_go",
        "mutually_exclusive_current_version_terminal_decision",
        "prefreeze_and_independent_cuda_resource_evidence",
    ],
    "independent_terminal_go_required": True,
}
EXPECTED_RESOURCE_PROTOCOL = {
    "budget_file": "RESOURCE_BUDGET_AND_PREFLIGHT_V8.json",
    "gpu": "NVIDIA GeForce RTX 4090",
    "max_concurrent_training_runs": 1,
    "raw_primary_training_gpu_hours": 36.8589,
    "raw_independent_replay_gpu_hours": 36.8589,
    "raw_combined_gpu_hours": 73.7177,
    "estimated_total_gpu_hours": 89.9356,
    "independent_full_replay_required": True,
    "budget_can_change_protocol": False,
    "micro_benchmark_authorized": False,
}
EXPECTED_PUBLICATION_FIREWALL = {
    "formal_training_allowed": False,
    "formal_source_allowed": False,
    "formal_render_allowed": False,
    "formal_targets_that_must_not_exist": [
        "Revision-0710/04_ablation_runs/fig04_fig05_same_contract_5fold_v8",
        "Revision-0710/06_figures_and_statistics/05_source_data_by_figure/Fig04/source_data_v8",
        "Revision-0710/06_figures_and_statistics/05_source_data_by_figure/Fig05/source_data_v8",
        "Revision-0710/06_figures_and_statistics/06_rendered_figures/Fig04/render_v8_legacy_style",
        "Revision-0710/06_figures_and_statistics/06_rendered_figures/Fig05/render_v8_legacy_style",
    ],
}
EXPECTED_SCALED_INTEGRATION_PROTOCOL = {
    "authorized": True,
    "device": "cpu",
    "variants": 12,
    "outer_fold": 0,
    "train_through_epoch": 2,
    "capture_epoch": 1,
    "evaluate_only_epoch": 1,
    "roles": ["train", "calibration", "outer_test"],
    "roles_physically_separate": True,
    "real_forward_backward_clip_scheduler_amp_path": True,
    "formal_data_allowed": False,
    "formal_gpu_allowed": False,
    "formal_output_allowed": False,
}
EXPECTED_BINDING_NAMES = {
    "v1_immutable_bindings": {
        "contract", "registry", "queue", "receipt",
        "ablation_model", "ablation_loss", "ablation_inputs",
    },
    "artifact_bindings": {
        "queue_v8", "authorization_schema_v8", "formal_executor_v8",
        "isolated_worker_v8", "output_schemas_v8", "resource_budget_v8",
        "scaled_integration_runner_v8", "authorization_validator_v8",
        "contract_validator_v8", "data_roles_v8", "input_ablation_v8",
        "loss_ablation_v8", "model_ablation_v8", "schema_validator_v8",
        "state_machine_v8", "training_v8", "variant_registry_v8",
        "independent_replay_v8",
        "formal_independent_verifier_v8",
        "tests_conftest", "tests_authorization_executor",
        "tests_contract_queue", "tests_model_training", "tests_schema_state",
        "tests_v8_integrity", "cuda_production_shape_preflight_v8",
        "independent_verifier_v8", "refresh_prefreeze_v8",
        "runtime_guard_v8", "tests_deep_replay_v8",
        "finalize_prefreeze_v8",
        "readme_v8", "re_review_request_v8",
        "v7_terminal_review_no_go", "worker_access_guard_v8",
        "terminal_go_schema_v8", "fresh_test_result_schema_v8",
        "tests_v8_terminal_hardening",
    },
    "canonical_bindings": {
        "cache_hard_gate", "leakage_gate", "embedding_tables",
        "compact_cache_dataset", "ehr_age_model", "private_retain_age",
        "ehr_age_losses", "ehr_age_data", "ehr_age_training",
        "role_registry", "role_validation",
        *{
            f"outer_fold_{fold}_{role}_manifest"
            for fold in range(5)
            for role in ("train", "calibration", "outer_test")
        },
    },
}


class ContractError(RuntimeError):
    pass


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_exact(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def read_json_list_exact(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ContractError(f"JSON root must be an array of objects: {path}")
    return value


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def repo_root(root: Path) -> Path:
    return root.resolve().parents[3]


def load_contract(root: Path) -> dict[str, Any]:
    return read_json_exact(root / "FROZEN_ABLATION_CONTRACT_V8.json")


def expected_run_ids() -> tuple[str, ...]:
    return tuple(f"{variant}__outer{fold}" for variant in VARIANT_IDS for fold in range(5))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root(root) / path


def _verify_record(root: Path, record: dict[str, Any], verify_hashes: bool) -> None:
    _require(set(record) == {"name", "path", "bytes", "sha256"}, "record keys changed")
    _require(isinstance(record["name"], str) and bool(record["name"]), "record name invalid")
    _require(isinstance(record["path"], str) and bool(record["path"]), "record path invalid")
    _require(isinstance(record["bytes"], int) and record["bytes"] > 0, "record bytes invalid")
    _require(isinstance(record["sha256"], str) and len(record["sha256"]) == 64, "record SHA invalid")
    if verify_hashes:
        path = _resolve(root, str(record["path"]))
        _require(path.is_file(), f"missing artifact: {record['name']}")
        _require(path.stat().st_size == int(record["bytes"]), f"bytes changed: {record['name']}")
        _require(sha256_file(path) == record["sha256"], f"SHA changed: {record['name']}")


def cache_artifact_bindings_sha256(bindings: dict[str, Any]) -> str:
    return canonical_json_sha256(bindings)


def _verify_cache_file_record(
    root: Path,
    record: dict[str, Any],
    *,
    expected_keys: set[str],
    verify_hashes: bool,
) -> Path:
    _require(set(record) == expected_keys, "cache artifact record keys changed")
    path_value = record.get("path")
    _require(isinstance(path_value, str) and bool(path_value), "cache artifact path invalid")
    _require(isinstance(record.get("bytes"), int) and int(record["bytes"]) > 0, "cache artifact bytes invalid")
    sha = record.get("sha256")
    _require(isinstance(sha, str) and len(sha) == 64, "cache artifact SHA invalid")
    expected_cache_root = (repo_root(root) / CACHE_RELATIVE_ROOT).resolve()
    candidate = Path(path_value)
    if candidate.is_absolute():
        path = candidate
    elif candidate.parts and candidate.parts[0].lower() == "revision-0710":
        path = repo_root(root) / candidate
    else:
        path = expected_cache_root / candidate
    try:
        path.resolve().relative_to(expected_cache_root)
    except ValueError as exc:
        raise ContractError("cache artifact escapes the frozen cache root") from exc
    if verify_hashes:
        _require(path.is_file(), f"missing cache artifact: {path_value}")
        _require(path.stat().st_size == int(record["bytes"]), f"cache artifact bytes changed: {path_value}")
        _require(sha256_file(path) == sha, f"cache artifact SHA changed: {path_value}")
    return path


def validate_cache_artifact_bindings(
    bindings: dict[str, Any],
    root: Path,
    *,
    verify_hashes: bool,
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "batch_manifest",
        "shards",
        "fold_transform_manifest",
        "fold_transforms",
    }
    _require(set(bindings) == expected_top, "cache_artifact_bindings keys changed")
    _require(bindings["schema_version"] == CACHE_BINDING_SCHEMA, "cache binding schema changed")
    base_keys = {"name", "path", "bytes", "sha256"}
    batch_path = _verify_cache_file_record(
        root,
        bindings["batch_manifest"],
        expected_keys=base_keys,
        verify_hashes=verify_hashes,
    )
    _require(bindings["batch_manifest"]["name"] == "batch_manifest", "batch manifest name changed")
    transform_manifest_path = _verify_cache_file_record(
        root,
        bindings["fold_transform_manifest"],
        expected_keys=base_keys,
        verify_hashes=verify_hashes,
    )
    _require(
        bindings["fold_transform_manifest"]["name"] == "fold_transform_manifest",
        "fold transform manifest name changed",
    )

    shards = bindings["shards"]
    shard_keys = {
        "ordinal", "path", "bytes", "sha256", "fold_id", "batch_in_fold",
        "n", "active_events", "storage_active_events",
    }
    _require(isinstance(shards, list) and len(shards) == 60, "cache shard binding count changed")
    _require([row.get("ordinal") for row in shards] == list(range(60)), "cache shard order changed")
    _require(len({row.get("path") for row in shards}) == 60, "cache shard paths are not unique")
    for row in shards:
        _verify_cache_file_record(
            root, row, expected_keys=shard_keys, verify_hashes=verify_hashes
        )
        for key in ("ordinal", "fold_id", "batch_in_fold", "n", "active_events", "storage_active_events"):
            _require(isinstance(row[key], int) and row[key] >= 0, f"cache shard {key} invalid")
        _require(int(row["fold_id"]) in range(5), "cache shard fold invalid")
        _require(int(row["n"]) > 0, "cache shard row count invalid")

    transforms = bindings["fold_transforms"]
    transform_keys = {
        "ordinal", "path", "bytes", "sha256", "outer_fold", "validation_fold",
        "train_folds", "train_n", "feature_hash", "train_id_hash",
    }
    _require(isinstance(transforms, list) and len(transforms) == 5, "fold transform binding count changed")
    _require([row.get("ordinal") for row in transforms] == list(range(5)), "fold transform order changed")
    _require([row.get("outer_fold") for row in transforms] == list(range(5)), "fold transform identity changed")
    for row in transforms:
        _verify_cache_file_record(
            root, row, expected_keys=transform_keys, verify_hashes=verify_hashes
        )
        _require(isinstance(row["validation_fold"], int), "validation fold invalid")
        _require(isinstance(row["train_folds"], list) and len(row["train_folds"]) == 3, "transform train folds invalid")
        _require(isinstance(row["train_n"], int) and row["train_n"] > 0, "transform train_n invalid")
        for key in ("feature_hash", "train_id_hash"):
            _require(isinstance(row[key], str) and len(row[key]) == 64, f"transform {key} invalid")

    if verify_hashes:
        observed_manifest = read_json_list_exact(batch_path)
        _require(len(observed_manifest) == 60, "batch manifest row count changed")
        for bound, observed in zip(shards, observed_manifest, strict=True):
            observed = dict(observed)
            observed["path"] = str(observed["path"]).replace("\\", "/")
            expected = {
                key: bound[key]
                for key in (
                    "path", "fold_id", "batch_in_fold", "n", "active_events",
                    "storage_active_events", "sha256",
                )
            }
            _require(observed == expected, f"batch manifest content/order changed at ordinal {bound['ordinal']}")
        observed_transforms = read_json_list_exact(transform_manifest_path)
        _require(len(observed_transforms) == 5, "fold transform manifest row count changed")
        for bound, observed in zip(transforms, observed_transforms, strict=True):
            observed = dict(observed)
            observed["path"] = str(observed["path"]).replace("\\", "/")
            expected = {
                key: bound[key]
                for key in (
                    "outer_fold", "validation_fold", "train_folds", "train_n",
                    "feature_hash", "train_id_hash", "path", "sha256",
                )
            }
            _require(observed == expected, f"fold transform content/order changed at outer fold {bound['outer_fold']}")
    return bindings


def verify_v1_immutable_bindings(payload: dict[str, Any], root: Path) -> dict[str, str]:
    bindings = payload["v1_immutable_bindings"]
    observed: dict[str, str] = {}
    for record in bindings["records"]:
        _verify_record(root, record, True)
        observed[str(record["name"])] = str(record["sha256"])
    return observed


def preflight_v1_runtime_bindings(root: Path) -> dict[str, str]:
    """Rehash immutable V1 runtime bytes before any dynamic V1 import."""

    payload = load_contract(root)
    records = {
        str(record.get("name")): record
        for record in payload.get("v1_immutable_bindings", {}).get("records", [])
    }
    required = ("ablation_model", "ablation_loss", "ablation_inputs")
    if any(name not in records for name in required):
        raise ContractError("V1 runtime pre-import binding is incomplete")
    observed: dict[str, str] = {}
    for name in required:
        record = records[name]
        _verify_record(root, record, True)
        observed[name] = str(record["sha256"])
    return observed


def validate_contract_payload(
    payload: dict[str, Any],
    root: Path,
    *,
    verify_hashes: bool,
    enforce_prefreeze_absence: bool = True,
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "created_date",
        "scope",
        "v1_immutable_bindings",
        "artifact_bindings",
        "canonical_bindings",
        "cache_artifact_bindings",
        "environment",
        "execution_roots",
        "variant_protocol",
        "training_protocol",
        "role_protocol",
        "evaluation_protocol",
        "state_machine",
        "authorization_protocol",
        "resource_protocol",
        "publication_firewall",
        "scaled_integration_protocol",
    }
    _require(set(payload) == expected_top, "contract top-level keys changed")
    _require(payload["schema_version"] == CONTRACT_SCHEMA, "schema_version changed")
    _require(payload["status"] == "PREFREEZE_V8_FORMAL_AUTHORIZATION_ABSENT", "status changed")
    _require(payload["created_date"] == "2026-07-11", "created_date changed")
    _require(payload["scope"] == EXPECTED_SCOPE, "scope changed")
    _require(payload["environment"] == EXPECTED_ENVIRONMENT, "environment changed")
    _require(payload["execution_roots"] == EXPECTED_EXECUTION_ROOTS, "execution_roots changed")
    _require(payload["state_machine"] == EXPECTED_STATE_MACHINE, "state_machine changed")
    _require(payload["authorization_protocol"] == EXPECTED_AUTHORIZATION_PROTOCOL, "authorization_protocol changed")
    _require(payload["resource_protocol"] == EXPECTED_RESOURCE_PROTOCOL, "resource_protocol changed")
    _require(payload["publication_firewall"] == EXPECTED_PUBLICATION_FIREWALL, "publication_firewall changed")
    _require(payload["scaled_integration_protocol"] == EXPECTED_SCALED_INTEGRATION_PROTOCOL, "scaled_integration_protocol changed")
    validate_cache_artifact_bindings(
        payload["cache_artifact_bindings"], root, verify_hashes=verify_hashes
    )
    training = payload["training_protocol"]
    _require(set(training) == set(EXPECTED_TRAINING_PROTOCOL), "training_protocol keys changed")
    for key, value in EXPECTED_TRAINING_PROTOCOL.items():
        _require(training.get(key) == value, f"training_protocol.{key} changed")

    roles = payload["role_protocol"]
    expected_roles = {
        "training_worker_allowed_roles": ["train"],
        "training_worker_calibration_or_outer_paths_allowed": False,
        "calibration_open_condition": "E60_COMPLETE_AND_E55_SEALED",
        "calibration_fits": ["sealed_age_correction", "sealed_cox_nuisance"],
        "calibration_model_selection_allowed": False,
        "outer_open_condition": "CALIBRATION_SEALED_AND_LEDGER_COMMITTED",
        "outer_refit_allowed": False,
        "cyclic_calibration_fold_by_outer": {"0": 1, "1": 2, "2": 3, "3": 4, "4": 0},
    }
    _require(set(roles) == set(expected_roles), "role_protocol keys changed")
    for key, value in expected_roles.items():
        _require(roles.get(key) == value, f"role_protocol.{key} changed")

    evaluation = payload["evaluation_protocol"]
    expected_evaluation = {
        "evaluate_only_epoch": 55,
        "strict_population": "outer_test strict_healthy_reference_60m == 1",
        "strict_metrics": ["raw_strict_mae", "raw_strict_pearson", "raw_strict_r2"],
        "risk_population": "all eligible outer_test at-risk participants",
        "risk_metric": "age+sex Harrell delta C-index",
        "organs": list(ORGAN_SYSTEMS),
        "organ_count": 7,
        "macro_rule": "unweighted mean across seven organ delta C-index values",
        "performance_selection_allowed": False,
        "all_folds_required": True,
        "validation_nuisance_verification": "independent_refit_7_age_corrections_and_14_cox_models",
        "cox_convergence_warning_policy": "HARD_FAIL",
        "prediction_label_alignment": "exact_frozen_role_row_order_age_sex_strict_duration_event_at_risk",
        "checkpoint_replay": "strict_production_restore_and_all_validation_and_outer_rows_bounded_deterministic_replay",
        "training_provenance": "seed7210_initial_reconstruction_and_independent_full_epoch_step_replay",
        "task_claim_opened_path_closure": "exact_expected_path_binding_plus_separate_process_python_audit_hook_sensitive_access_evidence",
        "deep_artifacts_per_run": 22,
    }
    _require(
        set(evaluation) == set(expected_evaluation),
        "evaluation_protocol keys changed",
    )
    for key, value in expected_evaluation.items():
        _require(evaluation.get(key) == value, f"evaluation_protocol.{key} changed")

    variants = payload["variant_protocol"]
    _require(
        set(variants) == {
            "variant_ids", "run_ids", "runs", "outer_folds",
            "fresh_model_and_optimizer_per_run", "unknown_variant_policy",
            "formal_comparator", "single_component_ablation_ids",
            "bundled_architecture_comparison_ids", "causal_component_language_allowed",
        },
        "variant_protocol keys changed",
    )
    _require(variants["variant_ids"] == list(VARIANT_IDS), "variant IDs changed")
    _require(variants["run_ids"] == list(expected_run_ids()), "run IDs changed")
    _require(variants["runs"] == 60, "run count changed")
    _require(variants["outer_folds"] == [0, 1, 2, 3, 4], "outer folds changed")
    _require(variants["fresh_model_and_optimizer_per_run"] is True, "fresh run policy changed")
    _require(variants["unknown_variant_policy"] == "FAIL_CLOSED", "unknown variant policy changed")
    _require(
        variants["formal_comparator"] == "fig04_01_canonical_control",
        "fresh canonical comparator changed",
    )
    _require(
        variants["single_component_ablation_ids"]
        == list(FIG04_VARIANTS[1:6]),
        "single-component estimands changed",
    )
    _require(
        variants["bundled_architecture_comparison_ids"]
        == ["fig04_07_seven_independent_organ_age_heads"],
        "bundled architecture comparison changed",
    )
    _require(
        variants["causal_component_language_allowed"] is False,
        "ablation interpretation policy changed",
    )
    roots = payload["execution_roots"]
    formal_root = Path(roots["formal_run_root"])
    _require(formal_root.is_absolute(), "formal_run_root must be absolute")
    _require(roots["formal_run_root"] != roots["scaled_smoke_root"], "formal/smoke roots overlap")

    for section in ("v1_immutable_bindings", "artifact_bindings", "canonical_bindings"):
        _require(set(payload[section]) == {"records"}, f"{section} keys changed")
        records = payload[section]["records"]
        _require(isinstance(records, list), f"{section} records changed")
        names = [str(record.get("name")) for record in records]
        _require(len(names) == len(set(names)), f"{section} duplicate names")
        _require(set(names) == EXPECTED_BINDING_NAMES[section], f"{section} names changed")
        for record in records:
            _verify_record(root, record, verify_hashes)
    if verify_hashes and enforce_prefreeze_absence:
        for target in payload["publication_firewall"]["formal_targets_that_must_not_exist"]:
            _require(not _resolve(root, target).exists(), f"formal target exists: {target}")
        _require(not (root / "FORMAL_AUTHORIZATION_V8.json").exists(), "formal authorization unexpectedly present")
    return payload


def read_queue(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != QUEUE_COLUMNS:
            raise ContractError("queue columns changed")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for key in (
                "ordinal",
                "outer_fold",
                "calibration_fold",
                "seed",
                "train_through_epoch",
                "capture_epoch",
                "evaluate_only_epoch",
            ):
                row[key] = int(row[key])
            rows.append(row)
    return rows


def validate_queue(rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    _require(len(rows) == 60, "queue must be complete 60/60")
    _require([row["ordinal"] for row in rows] == list(range(1, 61)), "queue ordinal changed")
    _require([row["run_id"] for row in rows] == list(expected_run_ids()), "queue run_id formula/order changed")
    _require(len({row["run_id"] for row in rows}) == 60, "queue run IDs are not unique")
    root = Path(payload["execution_roots"]["formal_run_root"]).resolve()
    paths: list[Path] = []
    for row in rows:
        variant = row["variant_id"]
        outer = row["outer_fold"]
        _require(variant in VARIANT_IDS and row["run_id"] == f"{variant}__outer{outer}", "queue variant/fold/run mismatch")
        figure = "Fig04" if variant in FIG04_VARIANTS else "Fig05"
        _require(row["figure"] == figure, "queue figure changed")
        _require(outer in range(5), "queue outer_fold changed")
        _require(row["calibration_fold"] == (outer + 1) % 5, "queue calibration fold changed")
        expected_train = "|".join(str(fold) for fold in range(5) if fold not in {outer, (outer + 1) % 5})
        _require(row["train_folds"] == expected_train, "queue train folds changed")
        _require(row["seed"] == 7210, "queue seed changed")
        _require(row["initialization"] == "fresh_random", "queue initialization changed")
        _require(row["train_through_epoch"] == 60 and row["capture_epoch"] == 55 and row["evaluate_only_epoch"] == 55, "queue epoch contract changed")
        _require(row["status"] == "PREFREEZE_NOT_AUTHORIZED", "queue status changed")
        intended = Path(row["intended_run_dir"])
        _require(intended.is_absolute(), "queue intended path must be absolute")
        resolved = intended.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ContractError("queue intended path escapes frozen root") from exc
        expected = root / variant / f"outer_fold_{outer}"
        _require(resolved == expected, "queue intended path formula changed")
        paths.append(resolved)
    _require(len(set(paths)) == 60, "queue intended paths are not unique")


def validate_prefreeze_bundle(root: Path, *, verify_hashes: bool) -> dict[str, Any]:
    payload = load_contract(root)
    validate_contract_payload(
        payload,
        root,
        verify_hashes=verify_hashes,
        enforce_prefreeze_absence=True,
    )
    queue = read_queue(root / "ABLATION_QUEUE_V8.csv")
    validate_queue(queue, payload)
    return {
        "status": "PASS",
        "contract_sha256": sha256_file(root / "FROZEN_ABLATION_CONTRACT_V8.json"),
        "queue_rows": len(queue),
        "formal_authorization_present": False,
        "formal_outputs_present": False,
    }
