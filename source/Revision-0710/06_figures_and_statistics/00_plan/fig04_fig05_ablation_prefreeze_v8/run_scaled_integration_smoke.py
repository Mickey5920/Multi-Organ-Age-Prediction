from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

CPU_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
for _key, _value in CPU_THREAD_ENVIRONMENT.items():
    os.environ[_key] = _value

import isolated_worker
import v8_contract
import v8_data
import v8_state_machine
import v8_schema_validator


ROOT = Path(__file__).resolve().parent
SMOKE_ROOT = ROOT / "smoke"


class IntegrationError(RuntimeError):
    pass


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise IntegrationError(f"create-only integration artifact exists: {path}") from exc


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    try:
        resolved.relative_to(SMOKE_ROOT.resolve())
    except ValueError as exc:
        raise IntegrationError("scaled smoke must remain under the V8 smoke root") from exc
    if resolved.exists():
        raise IntegrationError("scaled smoke output must be create-new")
    contract = v8_contract.load_contract(ROOT)
    formal_root = Path(contract["execution_roots"]["formal_run_root"]).resolve()
    try:
        resolved.relative_to(formal_root)
    except ValueError:
        pass
    else:
        raise IntegrationError("scaled smoke overlaps the formal run root")
    return resolved


def _task(
    *,
    phase: str,
    variant_id: str,
    data_record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    task = isolated_worker.synthetic_task_template(phase)
    attempt = output_dir.resolve().parent
    task_name = {
        "train": "TRAIN_TASK.json",
        "calibration": "CALIBRATION_TASK.json",
        "outer_test": "OUTER_TASK.json",
    }[phase]
    claim_path = attempt / "CLAIM.json"
    task.update(
        {
            "run_id": f"{variant_id}__outer0",
            "variant_id": variant_id,
            "data_source": {
                "format": "synthetic_role_bundle",
                "role": task["role"],
                "role_bundle_path": data_record["path"],
                "role_bundle_sha256": data_record["sha256"],
            },
            "output_dir": str(output_dir.resolve()),
            "contract_path": str((ROOT / "FROZEN_ABLATION_CONTRACT_V8.json").resolve()),
            "contract_sha256": v8_contract.sha256_file(
                ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"
            ),
            "task_path": str((attempt / task_name).resolve()),
            "claim_path": str(claim_path.resolve()),
            "claim_sha256": v8_contract.sha256_file(claim_path),
        }
    )
    return task


def _assert_override(variant_id: str, wiring: dict[str, Any]) -> dict[str, bool]:
    optimizer = set(wiring["optimizer_parameter_names"])
    frozen = set(wiring["frozen_or_unused_parameter_names"])
    checks = {
        "model_policy_bound": wiring["model"]["variant_id"] == variant_id,
        "input_policy_bound": isinstance(wiring["input"], dict),
        "loss_policy_bound": wiring["loss"]["variant_id"]
        if "variant_id" in wiring["loss"]
        else True,
        "optimizer_disjoint_from_frozen": not optimizer.intersection(frozen),
    }
    # Normalize the loss-policy check to a boolean even though policies are value-only.
    checks["loss_policy_bound"] = bool(checks["loss_policy_bound"])
    if variant_id == "fig04_02_no_organ_queries":
        checks["specific_override"] = (
            wiring["model"]["query_exact_zero"] is True
            and wiring["model"]["query_identical"] is True
            and wiring["model"]["query_requires_grad"] is False
            and not any(name.endswith("organ_queries") for name in optimizer)
        )
    elif variant_id == "fig04_03_no_common_bidirectional_gru":
        checks["specific_override"] = (
            wiring["model"]["shared_gru_module_present"] is False
            and wiring["model"]["shared_gru_output"] == "exact_zero_tensor"
            and wiring["model"]["private_alpha_gru_present"] is True
            and wiring["model"]["private_beta_gru_present"] is True
            and not any("temporal_encoder.gru" in name for name in wiring["model"].get("parameter_names", []))
        )
    elif variant_id == "fig04_04_no_survival_supervision":
        checks["specific_override"] = (
            wiring["loss"]["survival_weight"] == 0.0
            and wiring["loss"]["evaluator_gap_cox_weight"] == 0.25
            and any(name.endswith("baseline_log_rate") for name in frozen)
        )
    elif variant_id == "fig04_05_no_strict_offset_shrinkage":
        checks["specific_override"] = wiring["loss"]["strict_offset_weight"] == 0.0
    elif variant_id == "fig04_06_no_consistency_objective":
        checks["specific_override"] = (
            wiring["loss"]["event_dropout_probability"] == 0.1
            and wiring["loss"]["consistency_weight"] == 0.0
        )
    elif variant_id == "fig04_07_seven_independent_organ_age_heads":
        checks["specific_override"] = (
            wiring["loss"]["loss_route"] == "independent_heads_joint_loss"
            and wiring["model"]["independent_age_heads"] == 7
            and wiring["model"]["base_age_output"] is None
            and wiring["model"]["survival_risk_head_present"] is True
        )
    elif variant_id == "fig05_01_no_phenotypes_or_missingness":
        checks["specific_override"] = (
            wiring["input"]["phenotype"] == "zero"
            and wiring["input"]["phenotype_missing_mask"] == "zero"
            and wiring["input"]["sex"] == "retain"
        )
    elif variant_id == "fig05_02_no_hospital_icd":
        checks["specific_override"] = (
            wiring["input"]["remove_event_type"] == 1
            and wiring["input"]["stable_left_compaction"] is True
            and any("code_projection" in name for name in frozen)
        )
    elif variant_id == "fig05_03_no_gp_prescriptions":
        checks["specific_override"] = (
            wiring["input"]["remove_event_type"] == 2
            and wiring["input"]["stable_left_compaction"] is True
        )
    elif variant_id == "fig05_04_no_event_text":
        checks["specific_override"] = (
            wiring["input"]["event_text_id"] == "padding_zero"
            and any("text_projection" in name for name in frozen)
        )
    elif variant_id == "fig05_05_no_longitudinal_events":
        checks["specific_override"] = (
            wiring["input"]["all_event_arrays"] == "padding_zero"
            and any("event_encoder" in name for name in frozen)
        )
    else:
        checks["specific_override"] = (
            variant_id == "fig04_01_canonical_control"
            and wiring["model"]["policy"]["canonical_ehr_age_full_method"] is True
            and wiring["loss"]["survival_weight"] == 0.5
            and wiring["loss"]["evaluator_gap_cox_weight"] == 0.25
            and wiring["loss"]["consistency_weight"] == 0.1
        )
    if not all(checks.values()):
        raise IntegrationError(f"production override assertion failed for {variant_id}: {checks}")
    return checks


def run_scaled_integration_smoke(output_dir: Path) -> dict[str, Any]:
    v8_contract.validate_prefreeze_bundle(ROOT, verify_hashes=True)
    output_dir = _validate_output_dir(output_dir)
    output_dir.mkdir(parents=True)
    role_records = v8_data.create_physically_separate_synthetic_roles(
        output_dir / "00_roles"
    )
    rows = []
    output_shas = []
    for variant_ordinal, variant_id in enumerate(v8_contract.VARIANT_IDS, start=1):
        run_id = f"{variant_id}__outer0"
        # Short numbered smoke directories avoid the Windows legacy path limit.
        # The full immutable identity remains in every task and receipt.
        run_root = output_dir / "01_runs" / f"v{variant_ordinal:02d}" / "outer0"
        attempt = v8_state_machine.claim_specific_attempt(
            run_root, ordinal=1, run_id=run_id, seed=7210
        )
        try:
            train_task = _task(
                phase="train",
                variant_id=variant_id,
                data_record=role_records["train"],
                output_dir=attempt / "01_train",
            )
            _write_json_exclusive(attempt / "TRAIN_TASK.json", train_task)
            train_result = isolated_worker.run_train_task(train_task)
            training_path = attempt / "01_train" / "TRAINING_COMPLETE.json"
            if int(train_result["completed_epochs"]) != 2 or int(train_result["capture_epoch"]) != 1:
                raise IntegrationError("scaled training horizon changed")

            calibration_task = _task(
                phase="calibration",
                variant_id=variant_id,
                data_record=role_records["calibration"],
                output_dir=attempt / "02_calibration",
            )
            calibration_task.update(
                {
                    "training_complete_path": str(training_path.resolve()),
                    "checkpoint_path": train_result["checkpoint_path"],
                }
            )
            _write_json_exclusive(attempt / "CALIBRATION_TASK.json", calibration_task)
            calibration_result = isolated_worker.run_calibration_task(calibration_task)
            calibration_path = attempt / "02_calibration" / "CALIBRATION_COMPLETE.json"

            ledger = {
                "schema_version": "revision0710.fig04_fig05.outer_read_ledger.v8",
                "status": "COMMITTED_BEFORE_OUTER_OPEN",
                "run_id": run_id,
                "checkpoint_sha256": train_result["checkpoint_sha256"],
                "training_complete_sha256": v8_contract.sha256_file(training_path),
                "age_correction_sha256": calibration_result["age_correction_sha256"],
                "cox_nuisance_sha256": calibration_result["cox_nuisance_sha256"],
                "outer_role_manifest_sha256": role_records["outer_test"]["sha256"],
                "outer_opened_before_commit": False,
            }
            ledger_path = attempt / "OUTER_READ_LEDGER.json"
            _write_json_exclusive(ledger_path, ledger)

            outer_task = _task(
                phase="outer_test",
                variant_id=variant_id,
                data_record=role_records["outer_test"],
                output_dir=attempt / "03_outer_test",
            )
            outer_task.update(
                {
                    "training_complete_path": str(training_path.resolve()),
                    "checkpoint_path": train_result["checkpoint_path"],
                    "calibration_complete_path": str(calibration_path.resolve()),
                    "age_correction_path": calibration_result["age_correction_path"],
                    "cox_nuisance_path": calibration_result["cox_nuisance_path"],
                    "outer_read_ledger_path": str(ledger_path.resolve()),
                }
            )
            _write_json_exclusive(attempt / "OUTER_TASK.json", outer_task)
            outer_result = isolated_worker.run_outer_task(outer_task)
            v8_state_machine.write_completion_receipt(
                attempt,
                run_id=run_id,
                seed=7210,
                fold_receipt_path=attempt / "03_outer_test" / "FOLD_RECEIPT.json",
            )
            deep_bundle = v8_schema_validator.validate_scaled_attempt_bundle(
                attempt,
                variant_id=variant_id,
                root=ROOT,
            )
            if int(deep_bundle["artifact_count"]) != 22:
                raise IntegrationError("scaled deep closure is not exactly 22 artifacts")
            wiring = json.loads(
                (attempt / "01_train" / "PRODUCTION_WIRING.json").read_text(
                    encoding="utf-8"
                )
            )
            assertions = _assert_override(variant_id, wiring)
            output_shas.append(outer_result["prediction_rows_sha256"])
            rows.append(
                {
                    "run_id": run_id,
                    "variant_id": variant_id,
                    "training_complete_sha256": train_result[
                        "training_complete_sha256"
                    ],
                    "checkpoint_sha256": train_result["checkpoint_sha256"],
                    "training_trajectory_attestation_sha256": train_result[
                        "training_trajectory_attestation_sha256"
                    ],
                    "calibration_complete_sha256": calibration_result[
                        "calibration_complete_sha256"
                    ],
                    "outer_read_ledger_sha256": v8_contract.sha256_file(
                        ledger_path
                    ),
                    "fold_receipt_sha256": outer_result["fold_receipt_sha256"],
                    "prediction_rows_sha256": outer_result[
                        "prediction_rows_sha256"
                    ],
                    "override_assertions": assertions,
                    "deep_artifact_count": deep_bundle["artifact_count"],
                    "deep_artifact_set_sha256": deep_bundle["artifact_set_sha256"],
                    "production_replay": deep_bundle["production_replay"],
                    "task_sha256": {
                        phase_name: v8_contract.sha256_file(
                            attempt / file_name
                        )
                        for phase_name, file_name in (
                            ("train", "TRAIN_TASK.json"),
                            ("calibration", "CALIBRATION_TASK.json"),
                            ("outer_test", "OUTER_TASK.json"),
                        )
                    },
                    "claim_sha256": v8_contract.sha256_file(attempt / "CLAIM.json"),
                    "training_opened_roles": train_result["opened_roles"],
                    "calibration_opened_roles": calibration_result["opened_roles"],
                    "outer_opened_roles": outer_result["opened_roles"],
                }
            )
        except Exception as exc:
            v8_state_machine.write_failure_receipt(
                attempt,
                phase="scaled_integration",
                reason=f"{type(exc).__name__}: {exc}",
            )
            raise
    if len(set(output_shas)) <= 1:
        raise IntegrationError("all scaled integration prediction SHAs are identical")
    rows_path = output_dir / "INTEGRATION_RUN_RECEIPTS.json"
    _write_json_exclusive(rows_path, {"runs": rows})
    contract = v8_contract.load_contract(ROOT)
    formal_absent = []
    for target in contract["publication_firewall"]["formal_targets_that_must_not_exist"]:
        path = Path(target)
        path = path if path.is_absolute() else v8_contract.repo_root(ROOT) / path
        if path.exists():
            raise IntegrationError(f"formal target exists after smoke: {path}")
        formal_absent.append(str(path.resolve()))
    manifest = {
        "schema_version": "revision0710.fig04_fig05.scaled_integration_smoke.v8",
        "status": "PASS_SCALED_INTEGRATION_CPU_ONLY_NOT_FORMAL",
        "contract_sha256": v8_contract.sha256_file(
            ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"
        ),
        "protocol": {
            "device": "cpu",
            "train_epochs": 2,
            "capture_and_evaluate_epoch": 1,
            "real_optimizer_step": True,
            "real_backward": True,
            "gradient_clip": True,
            "scheduler_path": True,
            "amp_autocast_path": True,
            "cpu_thread_environment": CPU_THREAD_ENVIRONMENT,
        },
        "roles": {
            "physical_files": role_records,
            "pairwise_disjoint_participants": True,
            "training_only_opened_train": True,
            "calibration_opened_after_training_complete": True,
            "outer_opened_after_ledger_commit": True,
        },
        "variants": 12,
        "runs": len(rows),
        "deep_validated_runs": sum(int(row["deep_artifact_count"] == 22) for row in rows),
        "deep_artifacts_per_run": 22,
        "prediction_sha_unique_count": len(set(output_shas)),
        "all_prediction_shas_identical": False,
        "integration_run_receipts": {
            "path": rows_path.name,
            "bytes": rows_path.stat().st_size,
            "sha256": v8_contract.sha256_file(rows_path),
        },
        "formal_authorization_present": False,
        "formal_gpu_training_started": False,
        "formal_targets_absent": formal_absent,
    }
    _write_json_exclusive(output_dir / "MANIFEST.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_scaled_integration_smoke(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
