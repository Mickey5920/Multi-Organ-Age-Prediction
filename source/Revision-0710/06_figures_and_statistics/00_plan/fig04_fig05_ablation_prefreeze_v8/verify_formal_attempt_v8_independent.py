from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import v8_contract
import v8_data
import v8_independent_replay
import v8_model
import v8_runtime_guard


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
CANONICAL_MODEL_DIR = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "01_model"
if str(CANONICAL_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CANONICAL_MODEL_DIR))

from ehr_age_model import EHRAGEConfig  # noqa: E402


class FormalIndependentVerificationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalIndependentVerificationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root changed: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_role(
    task: dict[str, Any], contract: dict[str, Any]
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    source = task["data_source"]
    return v8_data.load_revision0710_role(
        role_manifest_path=Path(source["role_manifest_path"]),
        expected_manifest_sha256=source["role_manifest_sha256"],
        cache_dir=Path(source["cache_dir"]),
        outer_fold=int(task["outer_fold"]),
        expected_role=task["role"],
        cache_artifact_bindings=contract["cache_artifact_bindings"],
        expected_cache_artifact_bindings_sha256=source[
            "cache_artifact_bindings_sha256"
        ],
    )


def _expected_opened(
    task: dict[str, Any], contract: dict[str, Any], phase_inputs: list[Path]
) -> list[str]:
    source = task["data_source"]
    manifest_path = Path(source["role_manifest_path"])
    manifest = _read_json(manifest_path)
    cache_dir = Path(source["cache_dir"])
    bindings = contract["cache_artifact_bindings"]
    paths = [
        Path(task["task_path"]),
        Path(task["claim_path"]),
        Path(task["contract_path"]),
        ROOT / "ABLATION_QUEUE_V8.csv",
        Path(task["authorization"]["path"]),
        *v8_runtime_guard.lock_binding_paths(task["gpu_lock"]),
        *phase_inputs,
        manifest_path,
        Path(manifest["artifact"]["age_targets"]["path"]),
        Path(manifest["artifact"]["joint_labels"]["path"]),
        cache_dir / bindings["batch_manifest"]["path"],
        cache_dir / bindings["fold_transform_manifest"]["path"],
        cache_dir / bindings["fold_transforms"][int(task["outer_fold"])]["path"],
    ]
    selected = {int(value) for value in manifest["artifact"]["fold_ids"]}
    paths.extend(
        cache_dir / row["path"]
        for row in bindings["shards"]
        if int(row["fold_id"]) in selected
    )
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = str(path.resolve())
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def verify_attempt(attempt: Path, *, expected_run_id: str) -> dict[str, Any]:
    attempt = attempt.resolve()
    contract = v8_contract.load_contract(ROOT)
    tasks = {
        "train": _read_json(attempt / "TRAIN_TASK.json"),
        "calibration": _read_json(attempt / "CALIBRATION_TASK.json"),
        "outer_test": _read_json(attempt / "OUTER_TASK.json"),
    }
    claim_path = attempt / "CLAIM.json"
    claim_sha = _sha(claim_path)
    for phase, task in tasks.items():
        expected_name = {
            "train": "TRAIN_TASK.json",
            "calibration": "CALIBRATION_TASK.json",
            "outer_test": "OUTER_TASK.json",
        }[phase]
        _require(task["run_id"] == expected_run_id, f"{phase} run ID changed")
        _require(int(task["seed"]) == 7210, f"{phase} seed changed")
        _require(task["protocol"]["execution_mode"] == "formal", "not a formal task")
        _require(
            Path(task["task_path"]).resolve() == (attempt / expected_name).resolve()
            and Path(task["claim_path"]).resolve() == claim_path.resolve()
            and task["claim_sha256"] == claim_sha,
            f"{phase} task/claim closure changed",
        )

    train_bundle, code, text = _load_role(tasks["train"], contract)
    calibration_bundle, _, _ = _load_role(tasks["calibration"], contract)
    outer_bundle, _, _ = _load_role(tasks["outer_test"], contract)
    checkpoint_path = attempt / "01_train" / "fixed_epoch_0055.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    attestation_path = attempt / "01_train" / "TRAINING_TRAJECTORY_ATTESTATION.json"
    attestation = _read_json(attestation_path)
    training_path = attempt / "01_train" / "TRAINING_COMPLETE.json"
    calibration_complete_path = attempt / "02_calibration" / "CALIBRATION_COMPLETE.json"
    training = _read_json(training_path)
    calibration_complete = _read_json(calibration_complete_path)
    fold_receipt = _read_json(attempt / "03_outer_test" / "FOLD_RECEIPT.json")

    _require(
        training["opened_paths"]
        == _expected_opened(tasks["train"], contract, []),
        "TRAIN exact opened paths changed",
    )
    _require(
        calibration_complete["opened_paths"]
        == _expected_opened(
            tasks["calibration"], contract, [training_path, checkpoint_path]
        ),
        "CALIBRATION exact opened paths changed",
    )
    outer_inputs = [
        attempt / "OUTER_READ_LEDGER.json",
        training_path,
        calibration_complete_path,
        checkpoint_path,
        attempt / "02_calibration" / "age_correction.json",
        attempt / "02_calibration" / "cox_nuisance.json",
    ]
    _require(
        fold_receipt["opened_paths"]
        == _expected_opened(tasks["outer_test"], contract, outer_inputs),
        "OUTER exact opened paths changed",
    )

    protocol = tasks["train"]["protocol"]
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "formal independent replay requires CUDA")
    v8_independent_replay._seed(7210, device)
    age_anchor = float(train_bundle.labels["chronological_age"].float().mean())
    _require(
        abs(float(checkpoint["age_anchor"]) - age_anchor) <= 1e-7,
        "checkpoint age anchor differs from TRAIN",
    )
    model = v8_model.build_production_model(
        tasks["train"]["variant_id"],
        code,
        text,
        age_anchor=age_anchor,
        config=EHRAGEConfig(**checkpoint["model_config"]),
    )
    replay = v8_independent_replay.replay_training(
        model,
        variant_id=tasks["train"]["variant_id"],
        train_bundle=train_bundle,
        protocol=protocol,
        seed=7210,
    )
    expected_attestation = {
        "initial_full_model_sha256": replay.initial_full_model_sha256,
        "initial_trainable_sha256": replay.initial_trainable_sha256,
        "initial_optimizer_sha256": replay.initial_optimizer_sha256,
        "initial_scaler_sha256": replay.initial_scaler_sha256,
        "capture_model_state_sha256": replay.capture_model_state_sha256,
        "capture_trajectory_sha256": replay.capture_trajectory_sha256,
        "final_model_state_sha256": replay.final_model_state_sha256,
        "final_optimizer_state_sha256": replay.final_optimizer_state_sha256,
        "final_scaler_state_sha256": replay.final_scaler_state_sha256,
        "final_trajectory_sha256": replay.final_trajectory_sha256,
        "epochs": list(replay.epochs),
    }
    for key, expected in expected_attestation.items():
        _require(attestation.get(key) == expected, f"independent training mismatch: {key}")
    _require(
        checkpoint["training_trajectory_sha256_at_capture"]
        == replay.capture_trajectory_sha256,
        "checkpoint trajectory binding changed",
    )
    stored = checkpoint["model_state_dict"]
    _require(set(stored) == set(replay.capture_state_dict), "checkpoint keys changed")
    for name in stored:
        _require(
            torch.equal(stored[name].cpu(), replay.capture_state_dict[name]),
            f"checkpoint substitution detected: {name}",
        )

    restored = v8_model.build_production_model(
        tasks["train"]["variant_id"],
        code,
        text,
        age_anchor=age_anchor,
        config=EHRAGEConfig(**checkpoint["model_config"]),
    )
    restored.load_state_dict(replay.capture_state_dict, strict=True)
    role_receipts: dict[str, Any] = {}
    for role, bundle, path in (
        (
            "validation",
            calibration_bundle,
            attempt / "02_calibration" / "validation_predictions.csv",
        ),
        ("outer", outer_bundle, attempt / "03_outer_test" / "prediction_rows.csv"),
    ):
        frame = pd.read_csv(path, dtype={"participant_id": str})
        ages, risks = v8_independent_replay.replay_outputs(
            restored,
            bundle,
            device=device,
            batch_size=min(512, int(protocol["batch_size"])),
        )
        _require(
            np.allclose(frame["predicted_age"].to_numpy(float), ages, atol=2e-5, rtol=0.0),
            f"{role} predicted_age differs from checkpoint replay",
        )
        _require(
            np.allclose(frame["risk_score"].to_numpy(float), risks, atol=2e-5, rtol=0.0),
            f"{role} risk_score differs from checkpoint replay",
        )
        role_receipts[role] = {
            "participants": len(bundle.participant_ids),
            "rows": len(frame),
            "prediction_file_sha256": _sha(path),
        }
    return {
        "schema_version": "revision0710.fig04_fig05.formal_independent_replay.v8",
        "status": "PASS_SEED7210_FULL_TRAINING_AND_VALIDATION_OUTER_REPLAY",
        "run_id": expected_run_id,
        "variant_id": tasks["train"]["variant_id"],
        "outer_fold": int(tasks["train"]["outer_fold"]),
        "checkpoint_sha256": _sha(checkpoint_path),
        "capture_model_state_sha256": replay.capture_model_state_sha256,
        "final_trajectory_sha256": replay.final_trajectory_sha256,
        "claim_sha256": claim_sha,
        "task_set_sha256": v8_contract.canonical_json_sha256(
            {phase: _sha(Path(task["task_path"])) for phase, task in tasks.items()}
        ),
        "roles": role_receipts,
        "production_deep_validator_imported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_attempt(args.attempt, expected_run_id=args.run_id),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
