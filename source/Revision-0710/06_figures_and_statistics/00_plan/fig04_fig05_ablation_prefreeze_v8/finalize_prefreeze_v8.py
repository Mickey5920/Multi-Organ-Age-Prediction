from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import v8_contract


ROOT = Path(__file__).resolve().parent


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": v8_contract.sha256_file(path),
    }


def finalize(
    *,
    smoke_root: Path,
    cuda_probe_root: Path,
    independent_receipt_path: Path,
    fresh_test_receipt_path: Path,
) -> dict[str, Any]:
    if (ROOT / "INDEPENDENT_TERMINAL_REVIEW_GO_V8.json").exists():
        raise RuntimeError("terminal GO must be created only by the later independent review")
    validation = v8_contract.validate_prefreeze_bundle(ROOT, verify_hashes=True)
    contract = v8_contract.load_contract(ROOT)
    smoke_manifest_path = smoke_root / "MANIFEST.json"
    smoke_receipts_path = smoke_root / "INTEGRATION_RUN_RECEIPTS.json"
    cuda_receipt_path = cuda_probe_root / "CUDA_PRODUCTION_SHAPE_PREFLIGHT_RECEIPT.json"
    smoke = v8_contract.read_json_exact(smoke_manifest_path)
    cuda = v8_contract.read_json_exact(cuda_receipt_path)
    independent = v8_contract.read_json_exact(independent_receipt_path)
    fresh_tests = v8_contract.read_json_exact(fresh_test_receipt_path)
    resource = v8_contract.read_json_exact(ROOT / "RESOURCE_BUDGET_AND_PREFLIGHT_V8.json")
    if not (
        smoke.get("status") == "PASS_SCALED_INTEGRATION_CPU_ONLY_NOT_FORMAL"
        and int(smoke.get("runs", -1)) == 12
        and int(smoke.get("deep_validated_runs", -1)) == 12
        and int(smoke.get("deep_artifacts_per_run", -1)) == 22
        and smoke.get("formal_authorization_present") is False
        and cuda.get("status") == "PASS_12_OF_12_BATCH2048_PRODUCTION_SHAPE_AMP_STEPS"
        and int(cuda.get("variant_count", -1)) == 12
        and cuda.get("contract_sha256") == validation["contract_sha256"]
        and cuda.get("shape_contract", {}).get("batch_size") == 2048
        and cuda.get("shape_contract", {}).get("sequence_length") == 32
        and cuda.get("all_backward") is True
        and cuda.get("all_parameter_state_changed") is True
        and cuda.get("oom_policy") == "HARD_FAIL_NO_BATCH_OR_SEQUENCE_DOWNGRADE"
        and cuda.get("oom_observed") is False
        and cuda.get("formal_authorization_used") is False
        and independent.get("status")
        == "PASS_INDEPENDENT_MECHANICAL_VERIFICATION_NOT_A_GO_DECISION"
        and independent.get("production_deep_validator_used_for_critical_replay") is False
        and int(independent.get("independent_seed7210_scaled_e1_e2_replays", -1)) == 12
        and independent.get("scaled_replay_scope")
        == "16 participants; train through E2; E1 checkpoint capture; CPU"
        and fresh_tests.get("status") == "PASS_FRESH_FULL_TEST_SUITE_NOT_A_GO"
        and int(fresh_tests.get("tests_failed", -1)) == 0
        and int(fresh_tests.get("tests_skipped", -1)) == 0
        and int(fresh_tests.get("tests_passed", -1))
        == int(fresh_tests.get("tests_collected", -2))
    ):
        raise RuntimeError("V8 final evidence is incomplete")
    raw = float(resource["estimate"]["raw_combined_training_and_replay_gpu_hours"])
    headroom = float(resource["estimate"]["evaluation_and_operational_headroom_fraction"])
    computed = raw * (1.0 + headroom)
    stated = float(resource["estimate"]["estimated_total_gpu_hours"])
    if not (
        math.isclose(raw, 73.7177207866, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(computed, 89.935619359652, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(stated, 89.9356193597, rel_tol=0.0, abs_tol=1e-10)
    ):
        raise RuntimeError("V8 resource budget arithmetic changed")
    if not (
        int(resource["estimate"]["estimated_checkpoint_bytes_per_run"]) == 236424792
        and int(resource["estimate"]["estimated_checkpoint_bytes_total"]) == 14185487520
        and int(resource["estimate"]["estimated_total_disk_bytes_with_headroom"])
        == 25071133776
        and int(resource["preflight_thresholds"]["minimum_free_disk_bytes"])
        == 40000000000
    ):
        raise RuntimeError("V8 disk budget does not cover observed production checkpoints")
    if smoke["contract_sha256"] != validation["contract_sha256"]:
        raise RuntimeError("scaled smoke was not run against the final contract")
    if independent["contract_sha256"] != validation["contract_sha256"]:
        raise RuntimeError("independent verification was not run against the final contract")
    if fresh_tests["contract_sha256"] != validation["contract_sha256"]:
        raise RuntimeError("fresh tests were not run against the final contract")
    index = {
        "schema_version": "revision0710.fig04_fig05.smoke_run_index.v8",
        "status": "FINAL_EVIDENCE_INDEX_NOT_FORMAL",
        "contract_sha256": validation["contract_sha256"],
        "scaled_integration": _record(smoke_manifest_path),
        "scaled_integration_run_receipts": _record(smoke_receipts_path),
        "cuda_production_shape_preflight": _record(cuda_receipt_path),
        "independent_mechanical_verification": _record(independent_receipt_path),
        "fresh_full_test_result": _record(fresh_test_receipt_path),
        "evidence_version": "v8_final",
        "formal_authorization_present": False,
        "formal_outputs_present": False,
        "terminal_go_present": False,
    }
    index_path = ROOT / "SMOKE_RUN_INDEX_V8.json"
    _write_create_only(index_path, index)
    receipt = {
        "schema_version": "revision0710.fig04_fig05.prefreeze_validation_receipt.v8",
        "status": "PASS_PREFREEZE_MECHANICAL_GATES_AWAITING_INDEPENDENT_TERMINAL_GO",
        "contract_sha256": validation["contract_sha256"],
        "contract_bound_artifact_count": len(contract["artifact_bindings"]["records"]),
        "v1_runtime_preimport_sha256": v8_contract.preflight_v1_runtime_bindings(ROOT),
        "cache_shards_rehashed": 60,
        "fold_transforms_rehashed": 5,
        "resource_budget": {
            "primary_plus_independent_replay_gpu_hours": raw,
            "headroom_fraction": headroom,
            "computed_total_gpu_hours": computed,
            "estimated_total_gpu_hours_with_headroom": stated,
            "planning_estimate_not_per_variant_runtime_guarantee": True,
        },
        "pytest": {
            "status": "PASS",
            "tests": int(fresh_tests["tests_passed"]),
            "receipt_sha256": v8_contract.sha256_file(fresh_test_receipt_path),
        },
        "scaled_integration": {
            "scope": independent["scaled_replay_scope"],
            "variants": 12,
            "deep_validated_runs": 12,
            "artifacts_per_run": 22,
            "prediction_sha_unique_count": smoke["prediction_sha_unique_count"],
            "manifest_sha256": v8_contract.sha256_file(smoke_manifest_path),
        },
        "adversarial_terminal_hardening_tests": [
            "v7_no_go_cannot_authorize_v8",
            "mixed_current_v8_go_and_no_go_hard_fails",
            "missing_or_moved_terminal_go_rejected",
            "terminal_go_evidence_sha_mismatch_rejected",
            "hard_kill_stale_gpu_lock_recovered_with_audit",
            "live_os_gpu_lock_contention_rejected",
            "persistent_replay_recomputed_from_primary_outputs_and_resealed_scientific_tamper_rejected",
            "shared_gpu_lock_state_and_os_path_schema_across_all_consumers",
            "direct_formal_worker_without_access_guard_rejected",
            "unexpected_sensitive_worker_path_denied",
            "production_shape_or_batch_downgrade_rejected",
            "canonical_control_and_estimand_labels_frozen",
            "resource_budget_recomputed_not_hard_coded",
        ],
        "cuda_production_shape_preflight": {
            "status": "PASS",
            "variant_count": cuda["variant_count"],
            "batch_size": cuda["shape_contract"]["batch_size"],
            "sequence_length": cuda["shape_contract"]["sequence_length"],
            "peak_cuda_memory_bytes": cuda["peak_cuda_memory_bytes"],
            "capacity_scope": cuda["capacity_scope"],
            "receipt_sha256": v8_contract.sha256_file(cuda_receipt_path),
        },
        "worker_access_evidence": {
            "mechanism": "separate process plus Python audit hook installed before worker import",
            "opened_paths_field_is_expected_closure_not_os_telemetry": True,
            "formal_access_audits_required_per_phase": 3,
        },
        "canonical_comparator": "fig04_01_canonical_control",
        "bundled_architecture_comparison_ids": [
            "fig04_07_seven_independent_organ_age_heads"
        ],
        "independent_mechanical_receipt_sha256": v8_contract.sha256_file(
            independent_receipt_path
        ),
        "smoke_run_index_sha256": v8_contract.sha256_file(index_path),
        "formal_authorization_present": False,
        "formal_targets_absent": validation["formal_outputs_present"] is False,
        "formal_training_executed": False,
        "scientific_go_issued": False,
    }
    receipt_path = ROOT / "PREFREEZE_VALIDATION_RECEIPT_V8.json"
    _write_create_only(receipt_path, receipt)
    return {**receipt, "receipt_sha256": v8_contract.sha256_file(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--cuda-probe-root", type=Path, required=True)
    parser.add_argument("--independent-receipt", type=Path, required=True)
    parser.add_argument("--fresh-test-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        smoke_root=args.smoke_root.resolve(),
        cuda_probe_root=args.cuda_probe_root.resolve(),
        independent_receipt_path=args.independent_receipt.resolve(),
        fresh_test_receipt_path=args.fresh_test_receipt.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
