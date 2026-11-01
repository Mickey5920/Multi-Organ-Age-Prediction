from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

import formal_executor
import isolated_worker
import v8_contract
import v8_data
import v8_runtime_guard
import v8_schema_validator
import v8_state_machine
import v8_training
import verify_prefreeze_v8_independent as independent_verifier


ROOT = Path(__file__).resolve().parents[1]
VARIANT = "fig04_01_canonical_control"
RUN_ID = f"{VARIANT}__outer0"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _task(
    phase: str,
    *,
    role_record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    task = isolated_worker.synthetic_task_template(phase)
    attempt = output_dir.resolve().parent
    task_name = {
        "train": "TRAIN_TASK.json",
        "calibration": "CALIBRATION_TASK.json",
        "outer_test": "OUTER_TASK.json",
    }[phase]
    task.update(
        {
            "run_id": RUN_ID,
            "variant_id": VARIANT,
            "data_source": {
                "format": "synthetic_role_bundle",
                "role": task["role"],
                "role_bundle_path": role_record["path"],
                "role_bundle_sha256": role_record["sha256"],
            },
            "output_dir": str(output_dir.resolve()),
            "contract_path": str((ROOT / "FROZEN_ABLATION_CONTRACT_V8.json").resolve()),
            "contract_sha256": v8_contract.sha256_file(
                ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"
            ),
            "task_path": str((attempt / task_name).resolve()),
            "claim_path": str((attempt / "CLAIM.json").resolve()),
            "claim_sha256": v8_contract.sha256_file(attempt / "CLAIM.json"),
        }
    )
    return task


def _build_one_complete_attempt(root: Path) -> Path:
    roles = v8_data.create_physically_separate_synthetic_roles(root / "roles")
    attempt = v8_state_machine.claim_specific_attempt(
        root / "run" / "outer0", ordinal=1, run_id=RUN_ID, seed=7210
    )
    train_task = _task(
        "train", role_record=roles["train"], output_dir=attempt / "01_train"
    )
    _write_json(attempt / "TRAIN_TASK.json", train_task)
    train = isolated_worker.run_train_task(train_task)

    calibration_task = _task(
        "calibration",
        role_record=roles["calibration"],
        output_dir=attempt / "02_calibration",
    )
    calibration_task.update(
        {
            "training_complete_path": str(
                (attempt / "01_train" / "TRAINING_COMPLETE.json").resolve()
            ),
            "checkpoint_path": train["checkpoint_path"],
        }
    )
    _write_json(attempt / "CALIBRATION_TASK.json", calibration_task)
    calibration = isolated_worker.run_calibration_task(calibration_task)

    ledger = {
        "schema_version": "revision0710.fig04_fig05.outer_read_ledger.v8",
        "status": "COMMITTED_BEFORE_OUTER_OPEN",
        "run_id": RUN_ID,
        "checkpoint_sha256": train["checkpoint_sha256"],
        "training_complete_sha256": v8_contract.sha256_file(
            attempt / "01_train" / "TRAINING_COMPLETE.json"
        ),
        "age_correction_sha256": calibration["age_correction_sha256"],
        "cox_nuisance_sha256": calibration["cox_nuisance_sha256"],
        "outer_role_manifest_sha256": roles["outer_test"]["sha256"],
        "outer_opened_before_commit": False,
    }
    _write_json(attempt / "OUTER_READ_LEDGER.json", ledger)
    outer_task = _task(
        "outer_test",
        role_record=roles["outer_test"],
        output_dir=attempt / "03_outer_test",
    )
    outer_task.update(
        {
            "training_complete_path": str(
                (attempt / "01_train" / "TRAINING_COMPLETE.json").resolve()
            ),
            "checkpoint_path": train["checkpoint_path"],
            "calibration_complete_path": str(
                (attempt / "02_calibration" / "CALIBRATION_COMPLETE.json").resolve()
            ),
            "age_correction_path": calibration["age_correction_path"],
            "cox_nuisance_path": calibration["cox_nuisance_path"],
            "outer_read_ledger_path": str(
                (attempt / "OUTER_READ_LEDGER.json").resolve()
            ),
        }
    )
    _write_json(attempt / "OUTER_TASK.json", outer_task)
    isolated_worker.run_outer_task(outer_task)
    v8_state_machine.write_completion_receipt(
        attempt,
        run_id=RUN_ID,
        seed=7210,
        fold_receipt_path=attempt / "03_outer_test" / "FOLD_RECEIPT.json",
    )
    return attempt


@pytest.fixture(scope="module")
def complete_attempt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    attempt = _build_one_complete_attempt(tmp_path_factory.mktemp("v8_deep_bundle"))
    result = v8_schema_validator.validate_scaled_attempt_bundle(
        attempt, variant_id=VARIANT, root=ROOT
    )
    assert result["artifact_count"] == 22
    return attempt


def _copy_attempt(source: Path, root: Path) -> Path:
    # Absolute task/opened-path closure deliberately makes copied attempts invalid.
    # Build a fresh sealed attempt so each adversarial test reaches its intended gate.
    del source
    return _build_one_complete_attempt(root)


def _reseal_synchronized_bundle(attempt: Path) -> None:
    checkpoint_path = attempt / "01_train" / "fixed_epoch_0001.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_sha = v8_contract.sha256_file(checkpoint_path)

    checkpoint_manifest_path = attempt / "01_train" / "CHECKPOINT_MANIFEST.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint_manifest["checkpoint_sha256"] = checkpoint_sha
    checkpoint_manifest["checkpoint_bytes"] = checkpoint_path.stat().st_size
    checkpoint_manifest["initial_trainable_sha256"] = checkpoint.get(
        "initial_trainable_sha256", checkpoint_manifest["initial_trainable_sha256"]
    )
    _write_json(checkpoint_manifest_path, checkpoint_manifest)

    training_path = attempt / "01_train" / "TRAINING_COMPLETE.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["checkpoint_path"] = str(checkpoint_path.resolve())
    training["checkpoint_sha256"] = checkpoint_sha
    training["checkpoint_manifest_path"] = str(checkpoint_manifest_path.resolve())
    training["checkpoint_manifest_sha256"] = v8_contract.sha256_file(checkpoint_manifest_path)
    _write_json(training_path, training)

    correction_path = attempt / "02_calibration" / "age_correction.json"
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    correction["checkpoint_sha256"] = checkpoint_sha
    _write_json(correction_path, correction)
    correction_sha = v8_contract.sha256_file(correction_path)

    nuisance_path = attempt / "02_calibration" / "cox_nuisance.json"
    nuisance = json.loads(nuisance_path.read_text(encoding="utf-8"))
    nuisance["checkpoint_sha256"] = checkpoint_sha
    nuisance["age_correction_sha256"] = correction_sha
    _write_json(nuisance_path, nuisance)
    nuisance_sha = v8_contract.sha256_file(nuisance_path)

    validation_path = attempt / "02_calibration" / "validation_predictions.csv"
    validation = pd.read_csv(validation_path, dtype={"participant_id": str})
    validation["checkpoint_sha256"] = checkpoint_sha
    validation["age_correction_sha256"] = correction_sha
    validation["cox_nuisance_sha256"] = nuisance_sha
    validation.to_csv(validation_path, index=False, lineterminator="\n")

    calibration_path = attempt / "02_calibration" / "CALIBRATION_COMPLETE.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["training_complete_sha256"] = v8_contract.sha256_file(training_path)
    calibration["checkpoint_sha256"] = checkpoint_sha
    calibration["age_correction_path"] = str(correction_path.resolve())
    calibration["age_correction_sha256"] = correction_sha
    calibration["cox_nuisance_path"] = str(nuisance_path.resolve())
    calibration["cox_nuisance_sha256"] = nuisance_sha
    calibration["validation_predictions_sha256"] = v8_contract.sha256_file(validation_path)
    _write_json(calibration_path, calibration)

    ledger_path = attempt / "OUTER_READ_LEDGER.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["checkpoint_sha256"] = checkpoint_sha
    ledger["training_complete_sha256"] = v8_contract.sha256_file(training_path)
    ledger["age_correction_sha256"] = correction_sha
    ledger["cox_nuisance_sha256"] = nuisance_sha
    _write_json(ledger_path, ledger)

    prediction_path = attempt / "03_outer_test" / "prediction_rows.csv"
    prediction = pd.read_csv(prediction_path, dtype={"participant_id": str})
    prediction["checkpoint_sha256"] = checkpoint_sha
    prediction["age_correction_sha256"] = correction_sha
    prediction["cox_nuisance_sha256"] = nuisance_sha
    prediction.to_csv(prediction_path, index=False, lineterminator="\n")

    fold_path = attempt / "03_outer_test" / "fold_metrics.csv"
    fold = pd.read_csv(fold_path)
    fold["checkpoint_sha256"] = checkpoint_sha
    fold["prediction_rows_sha256"] = v8_contract.sha256_file(prediction_path)
    fold["age_correction_sha256"] = correction_sha
    fold["cox_nuisance_sha256"] = nuisance_sha
    fold["outer_read_ledger_sha256"] = v8_contract.sha256_file(ledger_path)
    fold.to_csv(fold_path, index=False, lineterminator="\n")

    receipt_path = attempt / "03_outer_test" / "FOLD_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "checkpoint_manifest_sha256": v8_contract.sha256_file(checkpoint_manifest_path),
            "prediction_rows_sha256": v8_contract.sha256_file(prediction_path),
            "age_correction_sha256": correction_sha,
            "cox_nuisance_sha256": nuisance_sha,
            "outer_read_ledger_sha256": v8_contract.sha256_file(ledger_path),
            "fold_metrics_sha256": v8_contract.sha256_file(fold_path),
            "organ_metrics_sha256": v8_contract.sha256_file(
                attempt / "03_outer_test" / "organ_metrics.csv"
            ),
        }
    )
    _write_json(receipt_path, receipt)
    attempt_receipt_path = attempt / "ATTEMPT_RECEIPT.json"
    attempt_receipt = json.loads(attempt_receipt_path.read_text(encoding="utf-8"))
    attempt_receipt["fold_receipt_sha256"] = v8_contract.sha256_file(receipt_path)
    _write_json(attempt_receipt_path, attempt_receipt)


def _assert_deep_failure(attempt: Path, pattern: str) -> None:
    with pytest.raises(v8_schema_validator.SchemaError, match=pattern):
        v8_schema_validator.validate_scaled_attempt_bundle(
            attempt, variant_id=VARIANT, root=ROOT
        )


def test_fake_nuisance_fails_after_all_receipts_are_resealed(
    complete_attempt: Path, tmp_path: Path
) -> None:
    attempt = _copy_attempt(complete_attempt, tmp_path)
    nuisance_path = attempt / "02_calibration" / "cox_nuisance.json"
    nuisance = json.loads(nuisance_path.read_text(encoding="utf-8"))
    nuisance["models"][0]["coefficients"]["actual_age"] += 0.25
    _write_json(nuisance_path, nuisance)
    _reseal_synchronized_bundle(attempt)
    _assert_deep_failure(attempt, "independent refit")


def test_predicted_equals_actual_fails_after_synchronized_reseal(
    complete_attempt: Path, tmp_path: Path
) -> None:
    attempt = _copy_attempt(complete_attempt, tmp_path)
    for relative in (
        Path("02_calibration/validation_predictions.csv"),
        Path("03_outer_test/prediction_rows.csv"),
    ):
        path = attempt / relative
        frame = pd.read_csv(path, dtype={"participant_id": str})
        frame["predicted_age"] = frame["actual_age"]
        frame["raw_age_gap"] = 0.0
        frame["corrected_age_gap"] = 0.0
        frame.to_csv(path, index=False, lineterminator="\n")
    _reseal_synchronized_bundle(attempt)
    _assert_deep_failure(attempt, "independent refit|checkpoint replay")


@pytest.mark.parametrize("replacement", ("empty", "arbitrary"))
def test_checkpoint_replacement_fails_after_synchronized_reseal(
    complete_attempt: Path, tmp_path: Path, replacement: str
) -> None:
    attempt = _copy_attempt(complete_attempt, tmp_path)
    checkpoint_path = attempt / "01_train" / "fixed_epoch_0001.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"] = (
        {} if replacement == "empty" else {"arbitrary": torch.zeros(1)}
    )
    torch.save(checkpoint, checkpoint_path)
    _reseal_synchronized_bundle(attempt)
    _assert_deep_failure(attempt, "state_dict|strictly restore|training trajectory")


def test_synchronized_prediction_and_receipt_tamper_fails_checkpoint_replay(
    complete_attempt: Path, tmp_path: Path
) -> None:
    attempt = _copy_attempt(complete_attempt, tmp_path)
    prediction_path = attempt / "03_outer_test" / "prediction_rows.csv"
    prediction = pd.read_csv(prediction_path, dtype={"participant_id": str})
    prediction["risk_score"] = prediction["risk_score"] + 0.5
    prediction.to_csv(prediction_path, index=False, lineterminator="\n")
    _reseal_synchronized_bundle(attempt)
    _assert_deep_failure(attempt, "checkpoint replay")


def test_validation_only_nuisance_manipulation_is_rejected_after_full_reseal(
    complete_attempt: Path, tmp_path: Path
) -> None:
    attempt = _copy_attempt(complete_attempt, tmp_path)
    validation_path = attempt / "02_calibration" / "validation_predictions.csv"
    validation = pd.read_csv(validation_path, dtype={"participant_id": str})
    validation["predicted_age"] = (
        validation["predicted_age"]
        + 0.75 * validation["event"]
        + 0.02 * validation["duration_months"]
    )
    validation["raw_age_gap"] = validation["predicted_age"] - validation["actual_age"]
    correction = v8_data.fit_age_correction(validation)
    old_correction = json.loads(
        (attempt / "02_calibration" / "age_correction.json").read_text(encoding="utf-8")
    )
    for key in (
        "run_id", "evaluation_epoch", "checkpoint_sha256",
        "participant_ids_sha256", "fit_role",
    ):
        correction[key] = old_correction[key]
    correction_path = attempt / "02_calibration" / "age_correction.json"
    _write_json(correction_path, correction)
    corrected_validation = v8_data.apply_age_correction(validation, correction)
    nuisance = v8_data.fit_cox_nuisance(corrected_validation, penalizer=0.1)
    nuisance.update(
        {
            "run_id": RUN_ID,
            "evaluation_epoch": 1,
            "checkpoint_sha256": old_correction["checkpoint_sha256"],
            "age_correction_sha256": v8_contract.sha256_file(correction_path),
            "fit_role": "validation",
        }
    )
    nuisance_path = attempt / "02_calibration" / "cox_nuisance.json"
    _write_json(nuisance_path, nuisance)
    corrected_validation.to_csv(validation_path, index=False, lineterminator="\n")

    outer_path = attempt / "03_outer_test" / "prediction_rows.csv"
    outer = pd.read_csv(outer_path, dtype={"participant_id": str})
    corrected_outer = v8_data.apply_age_correction(outer, correction)
    corrected_outer.to_csv(outer_path, index=False, lineterminator="\n")
    fold_values, organ_values = v8_data.score_outer_metrics(corrected_outer, nuisance)
    fold_path = attempt / "03_outer_test" / "fold_metrics.csv"
    fold = pd.read_csv(fold_path)
    for key, value in fold_values.items():
        fold.loc[0, key] = value
    fold.to_csv(fold_path, index=False, lineterminator="\n")
    organ_path = attempt / "03_outer_test" / "organ_metrics.csv"
    organ = pd.read_csv(organ_path)
    by_organ = {row["organ_system"]: row for row in organ_values}
    for index, row in organ.iterrows():
        for key, value in by_organ[str(row["organ_system"])].items():
            if key != "organ_system":
                organ.loc[index, key] = value
    organ.to_csv(organ_path, index=False, lineterminator="\n")
    _reseal_synchronized_bundle(attempt)
    _assert_deep_failure(
        attempt,
        "validation prediction predicted_age differs from production checkpoint replay",
    )


def test_same_shape_checkpoint_substitution_rejected_by_independent_full_replay(
    complete_attempt: Path, tmp_path: Path
) -> None:
    attempt = _copy_attempt(complete_attempt, tmp_path)
    checkpoint_path = attempt / "01_train" / "fixed_epoch_0001.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tensor = checkpoint["model_state_dict"]["model.organ_queries"].clone()
    tensor.view(-1)[0] += 0.123
    checkpoint["model_state_dict"]["model.organ_queries"] = tensor
    torch.save(checkpoint, checkpoint_path)
    attestation_path = attempt / "01_train" / "TRAINING_TRAJECTORY_ATTESTATION.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    forged_capture_sha = v8_training.canonical_state_sha256(
        checkpoint["model_state_dict"]
    )
    attestation["capture_checkpoint_sha256"] = v8_contract.sha256_file(checkpoint_path)
    attestation["capture_model_state_sha256"] = forged_capture_sha
    attestation["epochs"][0]["model_state_sha256"] = forged_capture_sha
    roles = {
        role: {
            "path": json.loads(
                (attempt / file_name).read_text(encoding="utf-8")
            )["data_source"]["role_bundle_path"]
        }
        for role, file_name in (
            ("train", "TRAIN_TASK.json"),
            ("calibration", "CALIBRATION_TASK.json"),
            ("outer_test", "OUTER_TASK.json"),
        )
    }
    with pytest.raises(
        independent_verifier.VerificationError,
        match="independent seed/training replay mismatch|same-shape checkpoint substitution",
    ):
        independent_verifier.independently_replay_training_and_both_roles(
            variant=VARIANT,
            train_task=json.loads(
                (attempt / "TRAIN_TASK.json").read_text(encoding="utf-8")
            ),
            checkpoint=checkpoint,
            attestation=attestation,
            role_records=roles,
            validation=pd.read_csv(
                attempt / "02_calibration" / "validation_predictions.csv",
                dtype={"participant_id": str},
            ),
            outer=pd.read_csv(
                attempt / "03_outer_test" / "prediction_rows.csv",
                dtype={"participant_id": str},
            ),
        )


def test_forged_initial_trainable_hash_rejected_by_seed_reconstruction(
    complete_attempt: Path, tmp_path: Path
) -> None:
    attempt = _copy_attempt(complete_attempt, tmp_path)
    checkpoint = torch.load(
        attempt / "01_train" / "fixed_epoch_0001.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["initial_trainable_sha256"] = "f" * 64
    attestation = json.loads(
        (attempt / "01_train" / "TRAINING_TRAJECTORY_ATTESTATION.json").read_text(
            encoding="utf-8"
        )
    )
    attestation["initial_trainable_sha256"] = "f" * 64
    roles = {
        role: {
            "path": json.loads((attempt / file_name).read_text(encoding="utf-8"))[
                "data_source"
            ]["role_bundle_path"]
        }
        for role, file_name in (
            ("train", "TRAIN_TASK.json"),
            ("calibration", "CALIBRATION_TASK.json"),
            ("outer_test", "OUTER_TASK.json"),
        )
    }
    with pytest.raises(independent_verifier.VerificationError, match="initial_trainable"):
        independent_verifier.independently_replay_training_and_both_roles(
            variant=VARIANT,
            train_task=json.loads((attempt / "TRAIN_TASK.json").read_text(encoding="utf-8")),
            checkpoint=checkpoint,
            attestation=attestation,
            role_records=roles,
            validation=pd.read_csv(
                attempt / "02_calibration" / "validation_predictions.csv",
                dtype={"participant_id": str},
            ),
            outer=pd.read_csv(
                attempt / "03_outer_test" / "prediction_rows.csv",
                dtype={"participant_id": str},
            ),
        )


def _formal_train_task(tmp_path: Path) -> dict[str, Any]:
    contract = v8_contract.load_contract(ROOT)
    row = v8_contract.read_queue(ROOT / "ABLATION_QUEUE_V8.csv")[0]
    fake_authorization = tmp_path / "shape_only_authorization.json"
    fake_authorization.write_text("{}\n", encoding="utf-8")
    lock = {
        "required": True,
        "state_path": str((tmp_path / "SINGLE_GPU_FORMAL_RUN_STATE.json").resolve()),
        "os_lock_path": str((tmp_path / "SINGLE_GPU_FORMAL_RUN.oslock").resolve()),
        "token": "a" * 64,
        "owner_pid": os.getpid(),
        "run_id": row["run_id"],
        "acquisition_ordinal": 1,
    }
    return formal_executor._common_task(
        phase="train",
        row=row,
        output_dir=Path(row["intended_run_dir"]) / "attempt_0001" / "01_train",
        data_source=formal_executor._data_source(contract, row["outer_fold"], "train"),
        authorization_path=fake_authorization,
        gpu_lock=lock,
    )


def test_executor_worker_canonical_task_schema_uses_real_validator(
    tmp_path: Path,
) -> None:
    task = _formal_train_task(tmp_path)
    assert set(task["data_source"]) == {
        "format", "role", "role_manifest_path", "role_manifest_sha256",
        "cache_dir", "cache_artifact_bindings_sha256",
    }
    isolated_worker.validate_task_structure(task, expected_phase="train")
    broken = copy.deepcopy(task)
    broken["data_source"].pop("cache_artifact_bindings_sha256")
    with pytest.raises(isolated_worker.WorkerError, match="data source keys"):
        isolated_worker.validate_task_structure(broken, expected_phase="train")


def test_direct_formal_worker_rechecks_environment_before_auth_or_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _formal_train_task(tmp_path)

    def fail_environment(expected: dict[str, Any]) -> dict[str, Any]:
        raise v8_runtime_guard.RuntimeGuardError("sentinel formal environment")

    monkeypatch.setattr(
        isolated_worker.v8_runtime_guard, "validate_formal_environment", fail_environment
    )
    with pytest.raises(v8_runtime_guard.RuntimeGuardError, match="sentinel"):
        isolated_worker.validate_task(task, expected_phase="train")


def test_direct_formal_worker_checks_auth_before_cache_or_role_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _formal_train_task(tmp_path)
    binding_called = False

    monkeypatch.setattr(
        isolated_worker.v8_runtime_guard,
        "validate_formal_environment",
        lambda expected: {"status": "TEST_ONLY"},
    )
    monkeypatch.setattr(
        isolated_worker.v8_runtime_guard,
        "validate_global_gpu_lock",
        lambda binding, task_run_id: {"status": "TEST_ONLY"},
    )

    def reject_authorization(
        path: Path, root: Path, **kwargs: Any
    ) -> dict[str, Any]:
        raise isolated_worker.v8_authorization.AuthorizationError("sentinel authorization")

    def record_binding(*args: Any, **kwargs: Any) -> None:
        nonlocal binding_called
        binding_called = True

    monkeypatch.setattr(
        isolated_worker.v8_authorization,
        "load_and_validate_authorization",
        reject_authorization,
    )
    monkeypatch.setattr(
        isolated_worker, "_validate_formal_task_bindings", record_binding
    )
    with pytest.raises(
        isolated_worker.v8_authorization.AuthorizationError, match="sentinel"
    ):
        isolated_worker.validate_task(task, expected_phase="train")
    assert binding_called is False


def test_gpu_lock_token_owner_and_run_are_all_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle, path, binding = v8_runtime_guard.acquire_global_gpu_lock(
        tmp_path, run_id=RUN_ID
    )
    monkeypatch.setattr(v8_runtime_guard.os, "getppid", lambda: os.getpid())
    try:
        v8_runtime_guard.validate_global_gpu_lock(binding, task_run_id=RUN_ID)
        for key, value in (
            ("token", "b" * 64), ("owner_pid", os.getpid() + 1), ("run_id", "wrong")
        ):
            broken = copy.deepcopy(binding)
            broken[key] = value
            with pytest.raises(v8_runtime_guard.RuntimeGuardError):
                v8_runtime_guard.validate_global_gpu_lock(
                    broken, task_run_id=str(broken["run_id"])
                )
    finally:
        v8_runtime_guard.release_global_gpu_lock(handle, path, binding)
