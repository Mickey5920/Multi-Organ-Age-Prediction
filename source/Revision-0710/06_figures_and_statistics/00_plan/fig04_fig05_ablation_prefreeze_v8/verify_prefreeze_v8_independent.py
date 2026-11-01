from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

import pandas as pd
import numpy as np
import torch

import v8_data
import v8_independent_replay
import v8_model


CANONICAL_MODEL_DIR = (
    Path(__file__).resolve().parents[4] / "Revision-0710" / "03_ehr_age" / "01_model"
)
if str(CANONICAL_MODEL_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(CANONICAL_MODEL_DIR))

from ehr_age_model import EHRAGEConfig


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
VARIANTS = (
    "fig04_01_canonical_control",
    "fig04_02_no_organ_queries",
    "fig04_03_no_common_bidirectional_gru",
    "fig04_04_no_survival_supervision",
    "fig04_05_no_strict_offset_shrinkage",
    "fig04_06_no_consistency_objective",
    "fig04_07_seven_independent_organ_age_heads",
    "fig05_01_no_phenotypes_or_missingness",
    "fig05_02_no_hospital_icd",
    "fig05_03_no_gp_prescriptions",
    "fig05_04_no_event_text",
    "fig05_05_no_longitudinal_events",
)
ORGANS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)
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


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_typed_csv(path: Path, schema: dict[str, Any]) -> pd.DataFrame:
    columns = list(schema["column_order"])
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    require(header == columns, f"CSV header changed: {path}")
    raw = pd.read_csv(
        path,
        dtype={column: "string" for column in columns},
        keep_default_na=False,
        na_filter=False,
    )
    converted = {}
    for column in columns:
        dtype = schema["columns"][column]["dtype"]
        if dtype == "string":
            converted[column] = raw[column].astype("string")
        elif dtype == "int64":
            converted[column] = pd.to_numeric(raw[column], errors="raise").astype(np.int64)
        elif dtype == "float64":
            converted[column] = pd.to_numeric(raw[column], errors="raise").astype(np.float64)
        elif dtype == "bool":
            mapping = {"True": True, "False": False, "true": True, "false": False}
            require(raw[column].isin(mapping).all(), f"boolean token changed: {path}/{column}")
            converted[column] = raw[column].map(mapping).astype(bool)
        else:
            raise VerificationError(f"unsupported CSV dtype: {dtype}")
    return pd.DataFrame(converted, columns=columns)


def independently_recompute_metrics(
    prediction: pd.DataFrame, nuisance: dict[str, Any]
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    from lifelines.utils import concordance_index

    strict = prediction.loc[prediction["strict_healthy_reference_60m"].eq(1)]
    actual = strict["actual_age"].to_numpy(np.float64)
    predicted = strict["predicted_age"].to_numpy(np.float64)
    residual = predicted - actual
    denominator = np.square(actual - actual.mean()).sum()
    require(len(strict) > 0 and denominator > 0, "strict metric population is invalid")
    fold = {
        "strict_n_participants": int(strict["participant_id"].nunique()),
        "strict_n_participant_organ_rows": int(len(strict)),
        "raw_strict_mae": float(np.abs(residual).mean()),
        "raw_strict_pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "raw_strict_r2": float(1.0 - np.square(residual).sum() / denominator),
    }
    models = {
        (row["organ_system"], row["model"]): row for row in nuisance["models"]
    }
    organs = []
    for organ in ORGANS:
        cell = prediction.loc[
            prediction["organ_system"].eq(organ) & prediction["at_risk"].eq(1)
        ]
        require(len(cell) > 0 and int(cell["event"].sum()) > 0, f"risk population invalid: {organ}")
        scores = {}
        for model_name in ("base", "full"):
            model = models[(organ, model_name)]
            risk = np.zeros(len(cell), dtype=np.float64)
            for name in model["covariates"]:
                risk += (
                    (cell[name].to_numpy(np.float64) - float(model["means"][name]))
                    / float(model["scales"][name])
                    * float(model["coefficients"][name])
                )
            scores[model_name] = float(
                concordance_index(
                    cell["duration_months"].to_numpy(np.float64),
                    -risk,
                    cell["event"].to_numpy(bool),
                )
            )
        organs.append(
            {
                "organ_system": organ,
                "at_risk_n": int(len(cell)),
                "event_n": int(cell["event"].sum()),
                "age_sex_harrell_base_c": scores["base"],
                "age_sex_gap_harrell_full_c": scores["full"],
                "age_sex_harrell_delta_c": scores["full"] - scores["base"],
            }
        )
    fold["age_sex_harrell_macro_delta_c"] = float(
        np.mean([row["age_sex_harrell_delta_c"] for row in organs])
    )
    return fold, organs


def verify_contract() -> dict[str, Any]:
    contract_path = ROOT / "FROZEN_ABLATION_CONTRACT_V8.json"
    contract = read_json(contract_path)
    require(
        contract.get("schema_version")
        == "revision0710.fig04_fig05.ablation_prefreeze.contract.v8",
        "contract schema changed",
    )
    require(
        contract.get("status") == "PREFREEZE_V8_FORMAL_AUTHORIZATION_ABSENT",
        "contract status changed",
    )
    require(contract["variant_protocol"]["variant_ids"] == list(VARIANTS), "variant order changed")
    expected_ids = [f"{variant}__outer{fold}" for variant in VARIANTS for fold in range(5)]
    require(contract["variant_protocol"]["run_ids"] == expected_ids, "60 run IDs changed")
    observed_v1 = {}
    record_count = 0
    for section in ("v1_immutable_bindings", "artifact_bindings", "canonical_bindings"):
        records = contract[section]["records"]
        names = [record["name"] for record in records]
        require(len(names) == len(set(names)), f"duplicate record names in {section}")
        for record in records:
            require(set(record) == {"name", "path", "bytes", "sha256"}, "record keys changed")
            path = resolve_record_path(record["path"])
            require(path.is_file(), f"bound file missing: {record['name']}")
            require(path.stat().st_size == record["bytes"], f"bound bytes changed: {record['name']}")
            require(sha256_file(path) == record["sha256"], f"bound SHA changed: {record['name']}")
            record_count += 1
            if section == "v1_immutable_bindings":
                observed_v1[record["name"]] = record["sha256"]
    expected_fixed_v1 = {
            "contract": "a1dbde3c0b684b45bdab75dc6ea7695a8bf854a9bc24e3ef3a2918e856bd602c",
            "registry": "1b4ec0e3f15bafdfcd103af68a2e54a3d2c1bdaff20ccc768a49b0831c18f574",
            "queue": "aa6ae492a56a0782b104591c6c145c017f2d5d5ad68edc515238f7050c734bc2",
            "receipt": "b288b164f2a94e81130db96041813ac016576b1444d7582cb293a60003994a74",
        }
    require(
        {key: observed_v1.get(key) for key in expected_fixed_v1} == expected_fixed_v1
        and set(observed_v1) == {
            *expected_fixed_v1,
            "ablation_model", "ablation_loss", "ablation_inputs",
        },
        "V1 immutable set changed",
    )
    cache = contract["cache_artifact_bindings"]
    require(
        set(cache)
        == {
            "schema_version", "batch_manifest", "shards",
            "fold_transform_manifest", "fold_transforms",
        },
        "cache binding envelope changed",
    )
    require(len(cache["shards"]) == 60, "cache binding is not 60/60 shards")
    require(len(cache["fold_transforms"]) == 5, "cache binding is not 5/5 transforms")
    require([row["ordinal"] for row in cache["shards"]] == list(range(60)), "cache shard order changed")
    cache_root = REPO_ROOT / "Revision-0710" / "01_data_rebuild" / "04_cache_v1"
    cache_records = [cache["batch_manifest"], *cache["shards"], cache["fold_transform_manifest"], *cache["fold_transforms"]]
    for record in cache_records:
        path = cache_root / record["path"]
        require(path.is_file(), f"cache-bound file missing: {record['path']}")
        require(path.stat().st_size == int(record["bytes"]), f"cache-bound bytes changed: {record['path']}")
        require(sha256_file(path) == record["sha256"], f"cache-bound SHA changed: {record['path']}")
    batch_rows = json.loads((cache_root / cache["batch_manifest"]["path"]).read_text(encoding="utf-8"))
    require(len(batch_rows) == 60, "batch manifest row count changed")
    for ordinal, (observed, bound) in enumerate(zip(batch_rows, cache["shards"], strict=True)):
        normalized = dict(observed)
        normalized["path"] = str(normalized["path"]).replace("\\", "/")
        expected = {key: bound[key] for key in (
            "path", "fold_id", "batch_in_fold", "n", "active_events",
            "storage_active_events", "sha256",
        )}
        require(normalized == expected and bound["ordinal"] == ordinal, "batch manifest content/order changed")
    transform_rows = json.loads((cache_root / cache["fold_transform_manifest"]["path"]).read_text(encoding="utf-8"))
    require(len(transform_rows) == 5, "transform manifest row count changed")
    for ordinal, (observed, bound) in enumerate(zip(transform_rows, cache["fold_transforms"], strict=True)):
        normalized = dict(observed)
        normalized["path"] = str(normalized["path"]).replace("\\", "/")
        expected = {key: bound[key] for key in (
            "outer_fold", "validation_fold", "train_folds", "train_n",
            "feature_hash", "train_id_hash", "path", "sha256",
        )}
        require(normalized == expected and bound["ordinal"] == ordinal, "transform manifest content/order changed")
    require(not (ROOT / "FORMAL_AUTHORIZATION_V8.json").exists(), "formal authorization exists")
    formal_absent = []
    for value in contract["publication_firewall"]["formal_targets_that_must_not_exist"]:
        path = resolve_record_path(value)
        require(not path.exists(), f"formal target exists: {path}")
        formal_absent.append(str(path.resolve()))
    return {
        "payload": contract,
        "sha256": sha256_file(contract_path),
        "bound_record_count": record_count,
        "cache_bound_file_count": len(cache_records),
        "formal_targets_absent": formal_absent,
    }


def verify_queue(contract: dict[str, Any]) -> dict[str, Any]:
    queue_path = ROOT / "ABLATION_QUEUE_V8.csv"
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == QUEUE_COLUMNS, "queue columns changed")
        rows = list(reader)
    expected_ids = [f"{variant}__outer{fold}" for variant in VARIANTS for fold in range(5)]
    require(len(rows) == 60, "queue is not 60/60")
    require([row["run_id"] for row in rows] == expected_ids, "queue run order changed")
    formal_root = Path(contract["execution_roots"]["formal_run_root"]).resolve()
    paths = []
    for ordinal, row in enumerate(rows, start=1):
        variant = VARIANTS[(ordinal - 1) // 5]
        outer = (ordinal - 1) % 5
        calibration = (outer + 1) % 5
        train = "|".join(str(fold) for fold in range(5) if fold not in {outer, calibration})
        expected_path = formal_root / variant / f"outer_fold_{outer}"
        require(int(row["ordinal"]) == ordinal, "queue ordinal changed")
        require(row["variant_id"] == variant and int(row["outer_fold"]) == outer, "queue identity changed")
        require(int(row["calibration_fold"]) == calibration and row["train_folds"] == train, "queue fold roles changed")
        require(int(row["seed"]) == 7210 and row["initialization"] == "fresh_random", "queue initialization changed")
        require(
            (int(row["train_through_epoch"]), int(row["capture_epoch"]), int(row["evaluate_only_epoch"]))
            == (60, 55, 55),
            "queue epoch protocol changed",
        )
        require(row["status"] == "PREFREEZE_NOT_AUTHORIZED", "queue status changed")
        observed_path = Path(row["intended_run_dir"]).resolve()
        require(observed_path == expected_path, "queue formal path formula changed")
        paths.append(observed_path)
    require(len(set(paths)) == 60, "queue paths are not unique")
    return {"rows": 60, "sha256": sha256_file(queue_path)}


def verify_roles(smoke_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest["roles"]["physical_files"]
    require(set(records) == {"train", "calibration", "outer_test"}, "role set changed")
    participant_sets: dict[str, set[str]] = {}
    paths = []
    for role, record in records.items():
        path = Path(record["path"])
        require(path.is_file() and path.resolve().is_relative_to(smoke_root.resolve()), "role path escaped smoke root")
        require(path.stat().st_size == record["bytes"], f"role bytes changed: {role}")
        require(sha256_file(path) == record["sha256"], f"role SHA changed: {role}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(
            isinstance(payload, dict)
            and set(payload) == {"schema_version", "role", "outer_fold", "participant_ids", "inputs", "labels"},
            f"role payload keys changed: {role}",
        )
        require(payload["role"] == role and int(payload["outer_fold"]) == 0, f"role identity changed: {role}")
        participants = {str(value) for value in payload["participant_ids"]}
        require(len(participants) == 16, f"role participant count changed: {role}")
        participant_sets[role] = participants
        paths.append(path.resolve())
    require(len(set(paths)) == 3, "role files are not physically separate")
    for left, right in (("train", "calibration"), ("train", "outer_test"), ("calibration", "outer_test")):
        require(not participant_sets[left].intersection(participant_sets[right]), "role participants overlap")
    return {"records": records, "participants_per_role": 16}


def verify_prediction_table(
    path: Path,
    *,
    expected_role: str,
    run_id: str,
    expected_schema: dict[str, Any],
) -> pd.DataFrame:
    frame = read_typed_csv(path, expected_schema)
    require(list(frame.columns) == expected_schema["column_order"], f"prediction columns changed: {path}")
    require(len(frame) == 16 * 7, f"prediction row count changed: {path}")
    require(set(frame["run_id"]) == {run_id}, "prediction run ID changed")
    require(set(frame["split_role"]) == {expected_role}, "prediction role changed")
    require(set(frame["evaluation_epoch"].astype(int)) == {1}, "prediction epoch changed")
    groups = frame.groupby(["run_id", "split_role", "participant_id"], dropna=False)
    require(groups.size().eq(7).all(), "prediction x7 cardinality changed")
    require(all(set(cell["organ_system"]) == set(ORGANS) for _, cell in groups), "prediction organ coverage changed")
    raw_expected = frame["predicted_age"].to_numpy(float) - frame["actual_age"].to_numpy(float)
    require((abs(frame["raw_age_gap"].to_numpy(float) - raw_expected) <= 1e-8).all(), "raw age-gap formula changed")
    require((frame["event"].astype(int) <= frame["at_risk"].astype(int)).all(), "event occurs outside risk set")
    numeric = frame.select_dtypes(include=["number"]).to_numpy(float)
    require(all(math.isfinite(value) for value in numeric.ravel()), "prediction contains non-finite values")
    return frame


def independently_replay_training_and_both_roles(
    *,
    variant: str,
    train_task: dict[str, Any],
    checkpoint: dict[str, Any],
    attestation: dict[str, Any],
    role_records: dict[str, dict[str, Any]],
    validation: pd.DataFrame,
    outer: pd.DataFrame,
) -> dict[str, Any]:
    bundles = {
        role: v8_data.load_role_bundle(
            Path(role_records[role]["path"]), expected_role=role
        )
        for role in ("train", "calibration", "outer_test")
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71002)
        code = torch.randn(64, 8)
        text = torch.randn(64, 8)
        code[0].zero_()
        text[0].zero_()
    device = torch.device(str(train_task["protocol"]["device"]))
    v8_independent_replay._seed(7210, device)
    expected_anchor = float(
        bundles["train"].labels["chronological_age"].float().mean()
    )
    require(
        abs(float(checkpoint["age_anchor"]) - expected_anchor) <= 1e-7,
        "checkpoint age anchor is not reconstructed from TRAIN",
    )
    model = v8_model.build_production_model(
        variant,
        code,
        text,
        age_anchor=expected_anchor,
        config=EHRAGEConfig(**checkpoint["model_config"]),
    )
    replay = v8_independent_replay.replay_training(
        model,
        variant_id=variant,
        train_bundle=bundles["train"],
        protocol=train_task["protocol"],
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
        require(
            attestation.get(key) == expected,
            f"independent seed/training replay mismatch: {variant}/{key}",
        )
    require(
        checkpoint["initial_full_model_sha256"] == replay.initial_full_model_sha256
        and checkpoint["initial_trainable_sha256"] == replay.initial_trainable_sha256
        and checkpoint["initial_optimizer_sha256"] == replay.initial_optimizer_sha256
        and checkpoint["training_trajectory_sha256_at_capture"]
        == replay.capture_trajectory_sha256,
        f"checkpoint provenance differs from independent replay: {variant}",
    )
    stored_state = checkpoint["model_state_dict"]
    require(set(stored_state) == set(replay.capture_state_dict), "checkpoint state keys changed")
    for name in stored_state:
        require(
            torch.equal(stored_state[name].cpu(), replay.capture_state_dict[name]),
            f"same-shape checkpoint substitution detected: {variant}/{name}",
        )

    restored = v8_model.build_production_model(
        variant,
        code,
        text,
        age_anchor=expected_anchor,
        config=EHRAGEConfig(**checkpoint["model_config"]),
    )
    restored.load_state_dict(replay.capture_state_dict, strict=True)
    role_results: dict[str, Any] = {}
    for role, frame in (("calibration", validation), ("outer_test", outer)):
        ages, risks = v8_independent_replay.replay_outputs(
            restored,
            bundles[role],
            device=device,
            batch_size=int(train_task["protocol"]["batch_size"]),
        )
        require(
            np.allclose(
                frame["predicted_age"].to_numpy(np.float64), ages, atol=2e-5, rtol=0.0
            ),
            f"independent checkpoint replay mismatch: {variant}/{role}/predicted_age",
        )
        require(
            np.allclose(
                frame["risk_score"].to_numpy(np.float64), risks, atol=2e-5, rtol=0.0
            ),
            f"independent checkpoint replay mismatch: {variant}/{role}/risk_score",
        )
        role_results[role] = {
            "participants": len(bundles[role].participant_ids),
            "rows": len(frame),
        }
    return {
        "status": "PASS_INDEPENDENT_SEED7210_FULL_TRAINING_AND_DUAL_ROLE_REPLAY",
        "capture_model_state_sha256": replay.capture_model_state_sha256,
        "final_trajectory_sha256": replay.final_trajectory_sha256,
        "roles": role_results,
    }


def verify_one_run(
    smoke_root: Path,
    *,
    ordinal: int,
    variant: str,
    row: dict[str, Any],
    role_records: dict[str, dict[str, Any]],
    schemas: dict[str, Any],
) -> dict[str, str]:
    run_id = f"{variant}__outer0"
    require(row["run_id"] == run_id and row["variant_id"] == variant, "integration receipt order changed")
    attempt = smoke_root / "01_runs" / f"v{ordinal:02d}" / "outer0" / "attempt_0001"
    require(attempt.is_dir(), f"attempt missing: {run_id}")
    claim = read_json(attempt / "CLAIM.json")
    require(
        claim["run_id"] == run_id
        and claim["attempt_ordinal"] == 1
        and claim["seed"] == 7210
        and claim["restart_epoch"] == 1
        and claim["resume_training_state"] is False
        and claim["selection_uses_metrics"] is False,
        f"attempt claim changed: {run_id}",
    )

    train_task = read_json(attempt / "TRAIN_TASK.json")
    require(train_task["role"] == "train" and train_task["phase"] == "train", "train task role changed")
    require(set(train_task).isdisjoint({"calibration_role_path", "outer_test_role_path"}), "train task exposes held-out path")
    require(Path(train_task["data_source"]["role_bundle_path"]).resolve() == Path(role_records["train"]["path"]).resolve(), "train role path changed")
    require(train_task["authorization"]["required"] is False, "scaled task requested formal authorization")
    claim_sha = sha256_file(attempt / "CLAIM.json")
    require(
        Path(train_task["task_path"]).resolve() == (attempt / "TRAIN_TASK.json").resolve()
        and Path(train_task["claim_path"]).resolve() == (attempt / "CLAIM.json").resolve()
        and train_task["claim_sha256"] == claim_sha,
        "TRAIN task/claim binding changed",
    )
    protocol = train_task["protocol"]
    require(
        protocol["execution_mode"] == "scaled_integration"
        and protocol["device"] == "cpu"
        and protocol["train_through_epoch"] == 2
        and protocol["capture_epoch"] == 1
        and protocol["evaluate_only_epoch"] == 1,
        "scaled training protocol changed",
    )
    training_path = attempt / "01_train" / "TRAINING_COMPLETE.json"
    training = read_json(training_path)
    require(sha256_file(training_path) == row["training_complete_sha256"], "training receipt SHA changed")
    require(
        training["completed_epochs"] == 2
        and training["capture_epoch"] == 1
        and training["opened_roles"] == ["train"]
        and training["calibration_read_count"] == 0
        and training["outer_test_read_count"] == 0
        and training["fresh_initialization"] is True
        and training["resume_used"] is False,
        f"training boundary changed: {run_id}",
    )
    for flag in ("forward_called", "backward_called", "gradient_clip_called", "scheduler_called", "amp_path_entered"):
        require(training[flag] is True, f"training operation missing: {run_id}/{flag}")
    checkpoint_path = Path(training["checkpoint_path"])
    require(sha256_file(checkpoint_path) == training["checkpoint_sha256"] == row["checkpoint_sha256"], "checkpoint SHA changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(
        checkpoint["epoch"] == 1
        and checkpoint["seed"] == 7210
        and checkpoint["variant_id"] == variant
        and "optimizer_state_dict" not in checkpoint
        and len(checkpoint["initial_full_model_sha256"]) == 64
        and len(checkpoint["initial_optimizer_sha256"]) == 64
        and len(checkpoint["training_trajectory_sha256_at_capture"]) == 64,
        f"checkpoint capture identity changed: {run_id}",
    )
    checkpoint_manifest_path = attempt / "01_train" / "CHECKPOINT_MANIFEST.json"
    checkpoint_manifest = read_json(checkpoint_manifest_path)
    require(
        checkpoint_manifest["status"] == "SEALED_SCALED_CAPTURE_ONLY"
        and checkpoint_manifest["evaluation_epoch"] == 1
        and checkpoint_manifest["checkpoint_sha256"] == row["checkpoint_sha256"]
        and checkpoint_manifest["checkpoint_bytes"] == checkpoint_path.stat().st_size,
        f"checkpoint manifest changed: {run_id}",
    )
    require(sha256_file(checkpoint_manifest_path) == training["checkpoint_manifest_sha256"], "checkpoint manifest SHA changed")
    history = read_json(attempt / "01_train" / "TRAINING_HISTORY.json")["history"]
    require([cell["epoch"] for cell in history] == [1, 2], f"training history epochs changed: {run_id}")
    require(all(cell["optimizer_steps"] >= 1 and cell["amp_autocast_entered"] is True for cell in history), "optimizer/AMP history changed")
    trajectory_path = attempt / "01_train" / "TRAINING_TRAJECTORY_ATTESTATION.json"
    trajectory = read_json(trajectory_path)
    require(
        training["training_trajectory_attestation_sha256"] == sha256_file(trajectory_path)
        and training["task_sha256"] == sha256_file(attempt / "TRAIN_TASK.json")
        and training["claim_sha256"] == claim_sha,
        "TRAIN trajectory/task/claim closure changed",
    )
    expected_train_opened = [
        str((attempt / "TRAIN_TASK.json").resolve()),
        str((attempt / "CLAIM.json").resolve()),
        str(Path(role_records["train"]["path"]).resolve()),
    ]
    require(training["opened_paths"] == expected_train_opened, "TRAIN exact opened paths changed")

    calibration_task = read_json(attempt / "CALIBRATION_TASK.json")
    require(calibration_task["role"] == "calibration", "calibration task role changed")
    require(Path(calibration_task["data_source"]["role_bundle_path"]).resolve() == Path(role_records["calibration"]["path"]).resolve(), "calibration role path changed")
    require(
        Path(calibration_task["task_path"]).resolve()
        == (attempt / "CALIBRATION_TASK.json").resolve()
        and calibration_task["claim_sha256"] == claim_sha,
        "CALIBRATION task/claim binding changed",
    )
    calibration_path = attempt / "02_calibration" / "CALIBRATION_COMPLETE.json"
    calibration = read_json(calibration_path)
    require(sha256_file(calibration_path) == row["calibration_complete_sha256"], "calibration receipt SHA changed")
    require(
        calibration["model_selection_used"] is False
        and calibration["opened_roles"] == ["calibration"]
        and calibration["outer_test_read_count"] == 0,
        f"calibration boundary changed: {run_id}",
    )
    correction_path = Path(calibration["age_correction_path"])
    nuisance_path = Path(calibration["cox_nuisance_path"])
    correction = read_json(correction_path)
    nuisance = read_json(nuisance_path)
    require(sha256_file(correction_path) == calibration["age_correction_sha256"], "age correction SHA changed")
    require(sha256_file(nuisance_path) == calibration["cox_nuisance_sha256"], "Cox nuisance SHA changed")
    require(
        correction["fit_role"] == "validation"
        and correction["evaluation_epoch"] == 1
        and correction["model_selection_used"] is False
        and len(correction["parameters"]) == 7
        and {cell["organ_system"] for cell in correction["parameters"]} == set(ORGANS),
        f"age correction contract changed: {run_id}",
    )
    require(
        nuisance["fit_role"] == "validation"
        and nuisance["evaluation_epoch"] == 1
        and nuisance["penalizer"] == 0.1
        and nuisance["model_selection_used"] is False
        and len(nuisance["models"]) == 14,
        f"Cox nuisance contract changed: {run_id}",
    )
    expected_cells = {(organ, model) for organ in ORGANS for model in ("base", "full")}
    require({(cell["organ_system"], cell["model"]) for cell in nuisance["models"]} == expected_cells, "Cox cell coverage changed")
    validation_path = attempt / "02_calibration" / "validation_predictions.csv"
    require(sha256_file(validation_path) == calibration["validation_predictions_sha256"], "validation prediction SHA changed")
    validation_frame = verify_prediction_table(
        validation_path,
        expected_role="validation",
        run_id=run_id,
        expected_schema=schemas["tables"]["prediction_rows"],
    )

    ledger_path = attempt / "OUTER_READ_LEDGER.json"
    ledger = read_json(ledger_path)
    require(sha256_file(ledger_path) == row["outer_read_ledger_sha256"], "outer ledger SHA changed")
    require(
        ledger["status"] == "COMMITTED_BEFORE_OUTER_OPEN"
        and ledger["outer_opened_before_commit"] is False
        and ledger["training_complete_sha256"] == sha256_file(training_path)
        and ledger["age_correction_sha256"] == sha256_file(correction_path)
        and ledger["cox_nuisance_sha256"] == sha256_file(nuisance_path),
        f"outer ledger binding changed: {run_id}",
    )
    outer_task = read_json(attempt / "OUTER_TASK.json")
    require(outer_task["role"] == "outer_test", "outer task role changed")
    require(Path(outer_task["data_source"]["role_bundle_path"]).resolve() == Path(role_records["outer_test"]["path"]).resolve(), "outer role path changed")
    require(
        Path(outer_task["task_path"]).resolve() == (attempt / "OUTER_TASK.json").resolve()
        and outer_task["claim_sha256"] == claim_sha,
        "OUTER task/claim binding changed",
    )
    prediction_path = attempt / "03_outer_test" / "prediction_rows.csv"
    require(sha256_file(prediction_path) == row["prediction_rows_sha256"], "outer prediction SHA changed")
    prediction = verify_prediction_table(
        prediction_path,
        expected_role="test",
        run_id=run_id,
        expected_schema=schemas["tables"]["prediction_rows"],
    )
    independent_replay = independently_replay_training_and_both_roles(
        variant=variant,
        train_task=train_task,
        checkpoint=checkpoint,
        attestation=trajectory,
        role_records=role_records,
        validation=validation_frame,
        outer=prediction,
    )
    fold_path = attempt / "03_outer_test" / "fold_metrics.csv"
    fold = read_typed_csv(fold_path, schemas["tables"]["fold_metrics"])
    require(list(fold.columns) == schemas["tables"]["fold_metrics"]["column_order"] and len(fold) == 1, "fold metric schema changed")
    strict = prediction.loc[prediction["strict_healthy_reference_60m"].eq(1)]
    require(int(fold.loc[0, "strict_n_participants"]) == strict["participant_id"].nunique(), "strict participant count changed")
    require(int(fold.loc[0, "strict_n_participant_organ_rows"]) == len(strict) == 7 * strict["participant_id"].nunique(), "strict x7 count changed")
    organ_path = attempt / "03_outer_test" / "organ_metrics.csv"
    organ = read_typed_csv(organ_path, schemas["tables"]["organ_metrics"])
    require(list(organ.columns) == schemas["tables"]["organ_metrics"]["column_order"] and len(organ) == 7, "organ metric schema changed")
    require(set(organ["organ_system"]) == set(ORGANS), "organ metric coverage changed")
    delta = organ["age_sex_gap_harrell_full_c"].to_numpy(float) - organ["age_sex_harrell_base_c"].to_numpy(float)
    require((abs(delta - organ["age_sex_harrell_delta_c"].to_numpy(float)) <= 1e-10).all(), "delta C formula changed")
    require(abs(float(fold.loc[0, "age_sex_harrell_macro_delta_c"]) - float(organ["age_sex_harrell_delta_c"].mean())) <= 1e-10, "macro delta C changed")
    expected_fold, expected_organs = independently_recompute_metrics(prediction, nuisance)
    for key, expected in expected_fold.items():
        observed = fold.loc[0, key]
        require(
            int(observed) == expected if isinstance(expected, int) else abs(float(observed) - float(expected)) <= 1e-10,
            f"independent fold metric changed: {run_id}/{key}",
        )
    organ_by_name = {row["organ_system"]: row for row in expected_organs}
    for _, observed in organ.iterrows():
        expected = organ_by_name[str(observed["organ_system"])]
        for key in (
            "at_risk_n", "event_n", "age_sex_harrell_base_c",
            "age_sex_gap_harrell_full_c", "age_sex_harrell_delta_c",
        ):
            value = expected[key]
            require(
                int(observed[key]) == value if isinstance(value, int) else abs(float(observed[key]) - float(value)) <= 1e-10,
                f"independent organ metric changed: {run_id}/{observed['organ_system']}/{key}",
            )
    fold_receipt_path = attempt / "03_outer_test" / "FOLD_RECEIPT.json"
    fold_receipt = read_json(fold_receipt_path)
    require(sha256_file(fold_receipt_path) == row["fold_receipt_sha256"], "fold receipt SHA changed")
    require(
        fold_receipt["status"] == "COMPLETE_SCALED_CAPTURE_ONLY_NO_SELECTION"
        and fold_receipt["opened_roles"] == ["outer_test"]
        and fold_receipt["outer_refit_calls"] == 0
        and fold_receipt["prediction_rows_sha256"] == sha256_file(prediction_path)
        and fold_receipt["fold_metrics_sha256"] == sha256_file(fold_path)
        and fold_receipt["organ_metrics_sha256"] == sha256_file(organ_path)
        and fold_receipt["checkpoint_manifest_sha256"] == sha256_file(checkpoint_manifest_path),
        f"fold receipt cross-link changed: {run_id}",
    )
    expected_calibration_opened = [
        str((attempt / "CALIBRATION_TASK.json").resolve()),
        str((attempt / "CLAIM.json").resolve()),
        str(training_path.resolve()),
        str(checkpoint_path.resolve()),
        str(Path(role_records["calibration"]["path"]).resolve()),
    ]
    expected_outer_opened = [
        str((attempt / "OUTER_TASK.json").resolve()),
        str((attempt / "CLAIM.json").resolve()),
        str(ledger_path.resolve()),
        str(training_path.resolve()),
        str(calibration_path.resolve()),
        str(checkpoint_path.resolve()),
        str(correction_path.resolve()),
        str(nuisance_path.resolve()),
        str(Path(role_records["outer_test"]["path"]).resolve()),
    ]
    require(
        calibration["task_sha256"] == sha256_file(attempt / "CALIBRATION_TASK.json")
        and calibration["claim_sha256"] == claim_sha
        and calibration["opened_paths"] == expected_calibration_opened,
        "CALIBRATION exact task/claim/opened paths changed",
    )
    require(
        fold_receipt["task_sha256"] == sha256_file(attempt / "OUTER_TASK.json")
        and fold_receipt["claim_sha256"] == claim_sha
        and fold_receipt["opened_paths"] == expected_outer_opened,
        "OUTER exact task/claim/opened paths changed",
    )
    attempt_receipt = read_json(attempt / "ATTEMPT_RECEIPT.json")
    require(
        attempt_receipt["status"] == "COMPLETE"
        and attempt_receipt["attempt_ordinal"] == 1
        and attempt_receipt["fold_receipt_sha256"] == sha256_file(fold_receipt_path)
        and attempt_receipt["failure_receipt_sha256"] is None
        and attempt_receipt["selection_uses_metrics"] is False,
        f"attempt receipt changed: {run_id}",
    )
    require(not (attempt / "FAILURE.json").exists(), f"completed smoke attempt also failed: {run_id}")
    assertions = row["override_assertions"]
    require(assertions and all(value is True for value in assertions.values()), f"variant override assertion changed: {run_id}")
    require(
        row["training_opened_roles"] == ["train"]
        and row["calibration_opened_roles"] == ["calibration"]
        and row["outer_opened_roles"] == ["outer_test"],
        f"receipt role boundary changed: {run_id}",
    )
    artifact_paths = {
        "claim": attempt / "CLAIM.json",
        "train_task": attempt / "TRAIN_TASK.json",
        "calibration_task": attempt / "CALIBRATION_TASK.json",
        "outer_task": attempt / "OUTER_TASK.json",
        "checkpoint": checkpoint_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "training_complete": training_path,
        "model_optimizer_manifest": attempt / "01_train" / "MODEL_OPTIMIZER_MANIFEST.json",
        "production_wiring": attempt / "01_train" / "PRODUCTION_WIRING.json",
        "training_history": attempt / "01_train" / "TRAINING_HISTORY.json",
        "training_trajectory_attestation": trajectory_path,
        "age_correction": correction_path,
        "cox_nuisance": nuisance_path,
        "validation_predictions": validation_path,
        "calibration_complete": calibration_path,
        "outer_read_ledger": ledger_path,
        "prediction_rows": prediction_path,
        "fold_metrics": fold_path,
        "organ_metrics": organ_path,
        "fold_receipt": fold_receipt_path,
        "attempt_receipt": attempt / "ATTEMPT_RECEIPT.json",
        "complete": attempt / "COMPLETE.json",
    }
    require(len(artifact_paths) == 22 and all(path.is_file() for path in artifact_paths.values()), f"22-artifact closure changed: {run_id}")
    require(training["model_optimizer_manifest_sha256"] == sha256_file(artifact_paths["model_optimizer_manifest"]), "optimizer manifest cross-hash changed")
    require(training["production_wiring_sha256"] == sha256_file(artifact_paths["production_wiring"]), "wiring cross-hash changed")
    require(training["training_history_sha256"] == sha256_file(artifact_paths["training_history"]), "history cross-hash changed")
    independent_artifact_sha = canonical_sha(
        [
            {
                "name": name,
                "path": path.resolve().relative_to(attempt.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        ]
    )
    require(
        row["deep_artifact_count"] == 22
        and row["deep_artifact_set_sha256"] == independent_artifact_sha,
        f"independent 22-artifact closure changed: {run_id}",
    )
    return {
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_trainable_sha256": checkpoint_manifest["initial_trainable_sha256"],
        "artifact_set_sha256": independent_artifact_sha,
        "independent_replay_sha256": canonical_sha(independent_replay),
    }


def verify_smoke(smoke_root: Path, contract_sha: str) -> dict[str, Any]:
    smoke_root = smoke_root.resolve()
    require(smoke_root.is_dir() and smoke_root.is_relative_to((ROOT / "smoke").resolve()), "smoke root changed")
    manifest_path = smoke_root / "MANIFEST.json"
    manifest = read_json(manifest_path)
    require(
        manifest["status"] == "PASS_SCALED_INTEGRATION_CPU_ONLY_NOT_FORMAL"
        and manifest["contract_sha256"] == contract_sha
        and manifest["variants"] == 12
        and manifest["runs"] == 12
        and manifest["deep_validated_runs"] == 12
        and manifest["deep_artifacts_per_run"] == 22
        and manifest["formal_authorization_present"] is False
        and manifest["formal_gpu_training_started"] is False,
        "smoke manifest changed",
    )
    protocol = manifest["protocol"]
    require(
        protocol
        == {
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
        "smoke protocol changed",
    )
    role_result = verify_roles(smoke_root, manifest)
    receipts_path = smoke_root / manifest["integration_run_receipts"]["path"]
    require(sha256_file(receipts_path) == manifest["integration_run_receipts"]["sha256"], "integration receipt set SHA changed")
    rows = read_json(receipts_path)["runs"]
    require(len(rows) == 12, "integration receipt count changed")
    schemas = read_json(ROOT / "OUTPUT_SCHEMAS_V8.json")
    observed = [
        verify_one_run(
            smoke_root,
            ordinal=ordinal,
            variant=variant,
            row=rows[ordinal - 1],
            role_records=role_result["records"],
            schemas=schemas,
        )
        for ordinal, variant in enumerate(VARIANTS, start=1)
    ]
    prediction_shas = [row["prediction_sha256"] for row in observed]
    checkpoint_shas = [row["checkpoint_sha256"] for row in observed]
    require(len(set(prediction_shas)) == 12, "prediction SHAs are not 12/12 unique")
    require(manifest["prediction_sha_unique_count"] == 12 and manifest["all_prediction_shas_identical"] is False, "prediction uniqueness manifest changed")
    require(len(set(checkpoint_shas)) == 12, "checkpoint SHAs are not 12/12 unique")
    require(
        len({row["artifact_set_sha256"] for row in observed}) == 12,
        "artifact-set SHAs are not 12/12 unique",
    )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "run_receipts_sha256": sha256_file(receipts_path),
        "runs": 12,
        "prediction_sha_unique_count": 12,
        "checkpoint_sha_unique_count": 12,
        "role_participants_per_role": role_result["participants_per_role"],
        "artifact_set_sha_unique_count": len({row["artifact_set_sha256"] for row in observed}),
        "independent_replay_sha_unique_count": len(
            {row["independent_replay_sha256"] for row in observed}
        ),
    }


def verify_cuda_probe(probe_root: Path) -> dict[str, Any]:
    probe_root = probe_root.resolve()
    require(probe_root.is_dir() and probe_root.is_relative_to((ROOT / "smoke").resolve()), "CUDA probe root changed")
    receipt_path = probe_root / "CUDA_PRODUCTION_SHAPE_PREFLIGHT_RECEIPT.json"
    receipt = read_json(receipt_path)
    require(
        receipt["status"]
        == "PASS_12_OF_12_BATCH2048_PRODUCTION_SHAPE_AMP_STEPS"
        and receipt["contract_sha256"] == sha256_file(ROOT / "FROZEN_ABLATION_CONTRACT_V8.json")
        and receipt["preflight_script_sha256"]
        == sha256_file(ROOT / "run_cuda_production_shape_preflight.py")
        and receipt["variant_count"] == 12
        and receipt["variant_ids"] == list(VARIANTS)
        and receipt["shape_contract"]["batch_size"] == 2048
        and receipt["shape_contract"]["sequence_length"] == 32
        and receipt["shape_contract"]["phenotype_features"] == 78
        and receipt["shape_contract"]["code_embedding_shape"] == [9906, 512]
        and receipt["shape_contract"]["text_embedding_shape"] == [44565, 512]
        and receipt["all_amp_autocast"] is True
        and receipt["all_backward"] is True
        and receipt["all_optimizer_step"] is True
        and receipt["all_gradient_clip"] is True
        and receipt["all_parameter_state_changed"] is True
        and receipt["participant_labels_opened"] is False
        and receipt["calibration_role_opened"] is False
        and receipt["outer_role_opened"] is False
        and receipt["capacity_scope"]
        == "one_full_batch_optimizer_step_per_variant_not_e60_runtime_benchmark"
        and receipt["formal_protocol_covers_same_shape"] is True
        and receipt["oom_policy"] == "HARD_FAIL_NO_BATCH_OR_SEQUENCE_DOWNGRADE"
        and receipt["oom_observed"] is False
        and receipt["formal_authorization_used"] is False
        and receipt["formal_output_used"] is False,
        "CUDA production-shape preflight changed",
    )
    variants = receipt["variants"]
    require(len(variants) == 12, "CUDA variant receipt count changed")
    for ordinal, (variant, expected_id) in enumerate(zip(variants, VARIANTS, strict=True), start=1):
        checkpoint_path = probe_root / variant["checkpoint_path"]
        require(
            variant["ordinal"] == ordinal
            and variant["variant_id"] == expected_id
            and variant["batch_size"] == 2048
            and variant["sequence_length"] == 32
            and variant["phenotype_features"] == 78
            and variant["amp_autocast"] is True
            and variant["backward"] is True
            and variant["optimizer_step"] is True
            and variant["gradient_clip"] is True
            and variant["initial_model_sha256"] != variant["post_step_model_sha256"]
            and int(variant["peak_cuda_memory_bytes"]) > 0
            and float(variant["elapsed_seconds_one_step_and_checkpoint"]) > 0.0
            and math.isfinite(float(variant["loss"]))
            and math.isfinite(float(variant["preclip_gradient_norm"])),
            f"CUDA variant evidence changed: {expected_id}",
        )
        require(checkpoint_path.is_file(), "CUDA probe checkpoint is missing")
        require(checkpoint_path.stat().st_size == variant["checkpoint_bytes"], "CUDA probe checkpoint bytes changed")
        require(sha256_file(checkpoint_path) == variant["checkpoint_sha256"], "CUDA probe checkpoint SHA changed")
    return {
        "receipt_sha256": sha256_file(receipt_path),
        "variant_count": 12,
        "checkpoint_set_sha256": canonical_sha(
            [variant["checkpoint_sha256"] for variant in variants]
        ),
        "peak_cuda_memory_bytes": receipt["peak_cuda_memory_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--cuda-probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract_result = verify_contract()
    queue_result = verify_queue(contract_result["payload"])
    smoke_result = verify_smoke(args.smoke_root, contract_result["sha256"])
    cuda_probe_result = verify_cuda_probe(args.cuda_probe_root)
    receipt = {
        "schema_version": "revision0710.fig04_fig05.independent_prefreeze_verification.v8",
        "status": "PASS_INDEPENDENT_MECHANICAL_VERIFICATION_NOT_A_GO_DECISION",
        "contract_sha256": contract_result["sha256"],
        "bound_record_count": contract_result["bound_record_count"],
        "cache_bound_file_count": contract_result["cache_bound_file_count"],
        "queue": queue_result,
        "scaled_integration": smoke_result,
        "cuda_production_shape_preflight": cuda_probe_result,
        "formal_authorization_present": False,
        "formal_targets_absent": contract_result["formal_targets_absent"],
        "production_ablation_modules_imported_for_smoke_replay": True,
        "production_deep_validator_used_for_critical_replay": False,
        "independent_seed7210_scaled_e1_e2_replays": 12,
        "scaled_replay_scope": "16 participants; train through E2; E1 checkpoint capture; CPU",
        "validation_checkpoint_replays": 12,
        "outer_checkpoint_replays": 12,
        "formal_training_executed": False,
        "formal_source_or_render_executed": False,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
