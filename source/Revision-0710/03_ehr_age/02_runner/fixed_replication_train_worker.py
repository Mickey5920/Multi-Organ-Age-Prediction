from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

RUNNER_DIR = Path(__file__).resolve().parent
EHR_AGE_ROOT = RUNNER_DIR.parent
MODEL_DIR = EHR_AGE_ROOT / "01_model"
REVISION_ROOT = EHR_AGE_ROOT.parent
for path in (RUNNER_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ehr_age_data import (  # noqa: E402
    LABEL_COLUMNS,
    Fold0DevelopmentData,
    JointLabels,
    JointVectorizedBatcher,
    canonical_hash,
    load_restricted_compact_cache,
    sha256_file,
)
from ehr_age_model import BASE_AGE_MODE_PRIVATE_RETAIN, EHRAGEModel, ORGAN_SYSTEMS  # noqa: E402
from ehr_age_training import (  # noqa: E402
    _loss_config,
    _model_config,
    _survival_multiplier,
    _train_epoch,
    configure_reproducibility,
    trainable_parameter_hash,
)


TRAIN_AUTHORIZATION_SCHEMA = "revision0710.ehr_age.fixed_replication.train_only_authorization.v1"
METHOD_SPEC_SCHEMA = "revision0710.ehr_age.fixed_replication.sanitized_method_spec.v1"
CHECKPOINT_SCHEMA = "revision0710.ehr_age.fixed_replication.e55_checkpoint.v1"
TRAINING_COMPLETE_SCHEMA = "revision0710.ehr_age.fixed_replication.training_complete.v1"
FAILURE_SCHEMA = "revision0710.ehr_age.fixed_replication.failure.v1"
PARENT_SHA256 = "73c46ce884458f39148aed47a5843be76d55e96db8253c4a88bb30f70ebaf84c"
SEED = 7210
FIXED_EPOCH = 55
MAX_EPOCHS = 60


class TrainWorkerError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")


def _fold_roles(outer: int) -> tuple[int, tuple[int, int, int]]:
    calibration = (outer + 1) % 5
    train = tuple(fold for fold in range(5) if fold not in {outer, calibration})
    return calibration, train  # type: ignore[return-value]


def _verify_code(method: Mapping[str, Any]) -> None:
    lineage = method.get("code_lineage")
    if not isinstance(lineage, Mapping):
        raise TrainWorkerError("method spec lacks code lineage")
    expected = {
        "ehr_age_model": MODEL_DIR / "ehr_age_model.py",
        "private_retain_age": MODEL_DIR / "private_retain_age.py",
        "ehr_age_losses": MODEL_DIR / "ehr_age_losses.py",
        "ehr_age_data": RUNNER_DIR / "ehr_age_data.py",
        "ehr_age_training": RUNNER_DIR / "ehr_age_training.py",
        "compact_cache_dataset": REVISION_ROOT / "02_baselines_5fold" / "common" / "compact_cache_dataset.py",
        "fixed_replication_train_worker": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        binding = lineage.get(name)
        if not isinstance(binding, Mapping) or sha256_file(path) != binding.get("sha256"):
            raise TrainWorkerError(f"training code lineage changed: {name}")


def load_authorized_inputs(
    authorization_path: Path,
    *,
    expected_authorization_sha256: str,
    expected_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_authorization_sha256) is None:
        raise TrainWorkerError("expected authorization SHA256 must be explicit")
    if re.fullmatch(r"[0-9a-f]{64}", expected_contract_sha256) is None:
        raise TrainWorkerError("expected contract SHA256 must be explicit")
    if sha256_file(authorization_path) != expected_authorization_sha256:
        raise TrainWorkerError("train-only authorization hash changed")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    allowed_authorization_keys = {
        "schema_version",
        "status",
        "replication_contract_sha256",
        "parent_d07_003_sha256",
        "outer_fold",
        "train_folds",
        "seed",
        "max_epochs",
        "fixed_checkpoint_epoch",
        "outer_control_manifest_sha256",
        "train_method_spec",
        "train_role_manifest_sha256",
        "train_age_targets",
        "train_joint_labels",
        "train_participant_count",
        "train_participant_ids_canonical_sha256",
        "label_free_cache",
        "outer_transform",
        "calibration_or_outer_paths_present",
        "learned_checkpoint_inputs",
    }
    if set(authorization) != allowed_authorization_keys:
        raise TrainWorkerError("train-only authorization top-level keys changed")
    outer = int(authorization.get("outer_fold", -1))
    if outer not in (1, 2, 3, 4):
        raise TrainWorkerError("training worker only accepts outer folds 1..4")
    _, expected_train = _fold_roles(outer)
    if (
        authorization.get("schema_version") != TRAIN_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "FROZEN_BEFORE_TRAINING"
        or authorization.get("replication_contract_sha256") != expected_contract_sha256
        or authorization.get("parent_d07_003_sha256") != PARENT_SHA256
        or tuple(authorization.get("train_folds", ())) != expected_train
        or authorization.get("seed") != SEED
        or authorization.get("max_epochs") != MAX_EPOCHS
        or authorization.get("fixed_checkpoint_epoch") != FIXED_EPOCH
        or re.fullmatch(r"[0-9a-f]{64}", str(authorization.get("outer_control_manifest_sha256", ""))) is None
        or authorization.get("calibration_or_outer_paths_present") is not False
        or authorization.get("learned_checkpoint_inputs") != []
    ):
        raise TrainWorkerError("invalid train-only authorization")
    encoded = json.dumps(authorization, sort_keys=True).lower()
    for forbidden in (
        "calibration_manifest",
        "calibration_age",
        "calibration_joint",
        "outer_test_manifest",
        "outer_test_age",
        "outer_test_joint",
        "role_registry_path",
    ):
        if forbidden in encoded:
            raise TrainWorkerError(f"train-only authorization exposes {forbidden}")
    method_binding = authorization.get("train_method_spec")
    if not isinstance(method_binding, Mapping):
        raise TrainWorkerError("train authorization lacks sanitized method spec")
    method_path = Path(str(method_binding["path"])).resolve()
    if sha256_file(method_path) != method_binding.get("sha256"):
        raise TrainWorkerError("sanitized method spec hash changed")
    method = json.loads(method_path.read_text(encoding="utf-8"))
    allowed_method_keys = {
        "schema_version",
        "status",
        "replication_contract_sha256",
        "parent_d07_003_sha256",
        "candidate_id",
        "model",
        "loss",
        "training",
        "cache_sha256",
        "code_lineage",
        "initial_trainable_parameter_sha256",
        "label_or_outcome_paths_present",
        "learned_checkpoint_inputs",
    }
    if set(method) != allowed_method_keys:
        raise TrainWorkerError("sanitized method spec top-level keys changed")
    if (
        method.get("schema_version") != METHOD_SPEC_SCHEMA
        or method.get("status") != "SANITIZED_LABEL_FREE_METHOD_FREEZE"
        or method.get("replication_contract_sha256") != expected_contract_sha256
        or method.get("parent_d07_003_sha256") != PARENT_SHA256
        or method.get("label_or_outcome_paths_present") is not False
        or method.get("learned_checkpoint_inputs") != []
    ):
        raise TrainWorkerError("invalid sanitized method spec")
    method_text = json.dumps(method, sort_keys=True).lower()
    for forbidden in (
        "master_contract_path",
        "role_registry_path",
        "calibration_path",
        "outer_test_path",
        "baseline_prediction_path",
        "learned_checkpoint_input_path",
    ):
        if forbidden in method_text:
            raise TrainWorkerError(f"sanitized method spec exposes {forbidden}")
    _verify_code(method)
    for key in ("train_age_targets", "train_joint_labels"):
        binding = authorization.get(key)
        if not isinstance(binding, Mapping):
            raise TrainWorkerError(f"train authorization lacks {key}")
        path = Path(str(binding["path"])).resolve()
        if sha256_file(path) != binding.get("sha256"):
            raise TrainWorkerError(f"authorized train artifact changed: {key}")
    cache = authorization.get("label_free_cache")
    if not isinstance(cache, Mapping) or set(cache) != set(method.get("cache_sha256", {})):
        raise TrainWorkerError("authorized cache binding set changed")
    for name, binding in cache.items():
        path = Path(str(binding["path"])).resolve()
        digest = sha256_file(path)
        if digest != binding.get("sha256") or digest != method["cache_sha256"].get(name):
            raise TrainWorkerError(f"authorized cache changed: {name}")
    return authorization, method


def _verify_shards(cache_dir: Path, folds: Sequence[int]) -> None:
    manifest = json.loads((cache_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest if int(row["fold_id"]) in set(folds)]
    if set(int(row["fold_id"]) for row in rows) != set(folds):
        raise TrainWorkerError("batch manifest lacks an authorized train fold")
    for row in rows:
        path = cache_dir / str(row["path"])
        if sha256_file(path) != row.get("sha256"):
            raise TrainWorkerError(f"authorized cache shard changed: {row['path']}")


def _aligned_labels(frame: pd.DataFrame, participant_ids: Sequence[str], folds: Sequence[int]) -> JointLabels:
    if set(frame.columns) != set(LABEL_COLUMNS):
        raise TrainWorkerError("train labels have an unexpected schema")
    frame = frame.loc[:, LABEL_COLUMNS].copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(frame["fold_id"].astype(int).unique()) != set(folds):
        raise TrainWorkerError("train labels contain an unauthorized fold")
    if set(frame["inner_role"].astype(str).unique()) != {"train"}:
        raise TrainWorkerError("train labels contain a non-train role")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise TrainWorkerError("train labels contain duplicate rows")
    if set(frame["organ_system"].astype(str).unique()) != set(ORGAN_SYSTEMS):
        raise TrainWorkerError("train labels changed the seven-organ axis")
    if len(frame) != len(participant_ids) * len(ORGAN_SYSTEMS):
        raise TrainWorkerError("train labels do not contain exactly seven rows per participant")
    for column in ("event", "at_risk", "strict_healthy_reference_60m"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not set(values.unique()).issubset({0, 1}):
            raise TrainWorkerError(f"train labels {column} must be binary")
        frame[column] = values.astype(np.int8)
    if (frame["event"].eq(1) & frame["at_risk"].ne(1)).any():
        raise TrainWorkerError("train event lies outside the at-risk set")
    duration = pd.to_numeric(frame["duration_months"], errors="coerce")
    risk = frame["at_risk"].eq(1)
    if duration.loc[risk].isna().any() or not np.isfinite(duration.loc[risk].to_numpy()).all():
        raise TrainWorkerError("at-risk train duration must be finite")
    if duration.loc[risk].le(0.0).any():
        raise TrainWorkerError("at-risk train duration must be positive")
    frame["duration_months"] = duration
    index = pd.MultiIndex.from_product(
        [pd.Index([str(value) for value in participant_ids]), list(ORGAN_SYSTEMS)],
        names=["participant_id", "organ_system"],
    )
    aligned = frame.set_index(["participant_id", "organ_system"]).reindex(index)
    if aligned[["event", "at_risk", "strict_healthy_reference_60m"]].isna().any().any():
        raise TrainWorkerError("train labels are incomplete")
    missing_duration = aligned["duration_months"].isna()
    if (missing_duration & aligned["at_risk"].ne(0)).any():
        raise TrainWorkerError("at-risk train row has missing duration")
    aligned["duration_months"] = aligned["duration_months"].fillna(0.0)
    n = len(participant_ids)
    strict = aligned["strict_healthy_reference_60m"].to_numpy(dtype=np.int8).reshape(n, -1)
    if not np.all(strict == strict[:, :1]):
        raise TrainWorkerError("train strict status disagrees across organs")
    labels = JointLabels(
        strict_healthy=torch.from_numpy(strict[:, 0].astype(bool)),
        duration_months=torch.from_numpy(
            aligned["duration_months"].to_numpy(dtype=np.float32).reshape(n, -1)
        ),
        event=torch.from_numpy(aligned["event"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)),
        at_risk=torch.from_numpy(
            aligned["at_risk"].to_numpy(dtype=np.int8).reshape(n, -1).astype(bool)
        ),
        inner_role=tuple(["train"] * n),
    )
    labels.validate(n)
    return labels


def load_train_data(authorization: Mapping[str, Any]) -> Fold0DevelopmentData:
    outer = int(authorization["outer_fold"])
    folds = tuple(int(value) for value in authorization["train_folds"])
    cache_dir = Path(str(authorization["label_free_cache"]["batch_manifest"]["path"])).resolve().parent
    transform = Path(str(authorization["outer_transform"]["path"])).resolve()
    expected_transform = (cache_dir / "fold_transforms" / f"outer_fold_{outer}.npz").resolve()
    if transform != expected_transform or sha256_file(transform) != authorization["outer_transform"]["sha256"]:
        raise TrainWorkerError("outer-fold transform is not authorized")
    _verify_shards(cache_dir, folds)
    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=outer,
        fold_ids=folds,
        age_target_cache_path=Path(str(authorization["train_age_targets"]["path"])),
    )
    labels = _aligned_labels(
        pd.read_parquet(Path(str(authorization["train_joint_labels"]["path"]))),
        cache.dataset.participant_ids,
        folds,
    )
    if len(cache.dataset) != int(authorization["train_participant_count"]):
        raise TrainWorkerError("authorized train count changed")
    ids_hash = canonical_hash(
        {"participant_ids": sorted(str(value) for value in cache.dataset.participant_ids)}
    )
    if ids_hash != authorization["train_participant_ids_canonical_sha256"]:
        raise TrainWorkerError("authorized train participant IDs changed")
    indices = torch.arange(len(cache.dataset), dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(cache, labels, indices, empty, empty, dict(authorization))


def _tensor_map_hash(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    import lifelines

    properties = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()[0].strip()
    except Exception:
        driver = "UNAVAILABLE"
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "lifelines_version": lifelines.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": properties.name,
        "gpu_total_memory": int(properties.total_memory),
        "gpu_count": torch.cuda.device_count(),
        "nvidia_driver_version": driver,
        "amp_enabled": True,
        "tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def run_worker(
    *,
    authorization_path: Path,
    expected_authorization_sha256: str,
    expected_contract_sha256: str,
    expected_role_validation_sha256: str,
    attempt_dir: Path,
) -> Path:
    attempt_dir = attempt_dir.resolve()
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stage = "LOAD_TRAIN_ONLY"
    outer = -1
    try:
        if re.fullmatch(r"[0-9a-f]{64}", expected_role_validation_sha256) is None:
            raise TrainWorkerError("expected role-validation SHA256 must be explicit")
        authorization, method = load_authorized_inputs(
            authorization_path.resolve(),
            expected_authorization_sha256=expected_authorization_sha256,
            expected_contract_sha256=expected_contract_sha256,
        )
        outer = int(authorization["outer_fold"])
        if not torch.cuda.is_available():
            raise TrainWorkerError("formal fixed replication requires CUDA")
        training = method["training"]
        if training.get("amp") is not True or training.get("tf32") is not True:
            raise TrainWorkerError("formal fixed replication requires frozen AMP and TF32")
        configure_reproducibility(SEED, deterministic=True, tf32=True)
        if not torch.backends.cuda.matmul.allow_tf32:
            raise TrainWorkerError("effective CUDA TF32 setting is disabled")
        run_config = {
            "worker": "ISOLATED_TRAIN_ONLY",
            "outer_fold": outer,
            "train_folds": list(authorization["train_folds"]),
            "replication_contract_sha256": expected_contract_sha256,
            "train_authorization_path": str(authorization_path.resolve()),
            "train_authorization_sha256": expected_authorization_sha256,
            "role_artifact_validation_sha256": expected_role_validation_sha256,
            "method_spec_path": authorization["train_method_spec"]["path"],
            "method_spec_sha256": authorization["train_method_spec"]["sha256"],
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "fixed_epoch": FIXED_EPOCH,
            "resume_supported": False,
            "heldout_inputs": [],
            "environment": _environment(),
        }
        run_config["materialized_config_hash"] = canonical_hash(run_config)
        run_config_path = attempt_dir / "run_config.json"
        _write_json_exclusive(run_config_path, run_config)
        data = load_train_data(authorization)
        stage = "MODEL_PREFLIGHT"
        age_anchor = float(data.cache.dataset.chronological_age.mean())
        cuda_rng_before_model = [state.clone() for state in torch.cuda.get_rng_state_all()]
        model = EHRAGEModel(
            data.cache.code_embedding,
            data.cache.text_embedding,
            age_anchor=age_anchor,
            config=_model_config(method["model"]),
        )
        cuda_rng_after_model = [state.clone() for state in torch.cuda.get_rng_state_all()]
        if len(cuda_rng_before_model) != len(cuda_rng_after_model) or any(
            not torch.equal(before, after)
            for before, after in zip(cuda_rng_before_model, cuda_rng_after_model)
        ):
            raise TrainWorkerError("private model construction mutated CUDA RNG state")
        if model.config.base_age_mode != BASE_AGE_MODE_PRIVATE_RETAIN:
            raise TrainWorkerError("sanitized method is not private RETAIN")
        device = torch.device("cuda")
        model.to(device)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        warmup = int(training["warmup_epochs"])

        def lr_factor(index: int) -> float:
            if index < warmup:
                return float(index + 1) / max(warmup, 1)
            progress = (index - warmup) / max(MAX_EPOCHS - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        if not scaler.is_enabled():
            raise TrainWorkerError("CUDA AMP GradScaler is not enabled")
        edges = torch.tensor(
            method["model"]["survival_interval_edges_months"], dtype=torch.float32, device=device
        )
        batcher = JointVectorizedBatcher(
            data,
            data.train_indices,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            seed=SEED,
        )
        initial_hash = trainable_parameter_hash(model)
        if initial_hash != method["initial_trainable_parameter_sha256"]:
            raise TrainWorkerError(
                "from-scratch initial trainable hash changed from frozen D07-003 initialization"
            )
        trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != trainable_ids:
            raise TrainWorkerError("optimizer parameters do not equal all trainable parameters")
        cache_embedding_parameters = {
            "common_code": model.event_encoder.code_embedding.weight,
            "common_text": model.event_encoder.text_embedding.weight,
            "private_code": model.private_retain_age_tower.backbone.token_encoder.code_embedding.weight,
            "private_text": model.private_retain_age_tower.backbone.token_encoder.text_embedding.weight,
        }
        if any(parameter.requires_grad for parameter in cache_embedding_parameters.values()):
            raise TrainWorkerError("frozen cache embedding parameter became trainable")
        expected_embedding_values = {
            "common_code": data.cache.code_embedding,
            "private_code": data.cache.code_embedding,
            "common_text": data.cache.text_embedding,
            "private_text": data.cache.text_embedding,
        }
        if any(
            not torch.equal(parameter.detach().cpu(), expected_embedding_values[name].detach().cpu())
            for name, parameter in cache_embedding_parameters.items()
        ):
            raise TrainWorkerError("model cache embedding values changed")
        if (
            cache_embedding_parameters["common_code"].data_ptr()
            == cache_embedding_parameters["private_code"].data_ptr()
            or cache_embedding_parameters["common_text"].data_ptr()
            == cache_embedding_parameters["private_text"].data_ptr()
        ):
            raise TrainWorkerError("common/private cache embeddings are not independent clones")
        preflight_path = attempt_dir / "model_preflight.json"
        _write_json_exclusive(
            preflight_path,
            {
                "status": "PASS",
                "outer_fold": outer,
                "device": "cuda",
                "amp_scaler_enabled": True,
                "initial_trainable_parameter_sha256": initial_hash,
                "initial_hash_matches_frozen_d07_003": True,
                "optimizer_binds_exact_trainable_set": True,
                "trainable_cache_embedding_parameters": [],
                "cache_embedding_values_match": True,
                "common_private_embedding_storage_independent": True,
                "private_constructor_cuda_rng_preserved": True,
                "base_age_mode": model.config.base_age_mode,
                "heldout_read_count": 0,
                "role_artifact_validation_sha256": expected_role_validation_sha256,
            },
        )
        history_path = attempt_dir / "training_history.jsonl"
        checkpoint_path = attempt_dir / "fixed_e55_checkpoint.pt"
        checkpoint_sha: str | None = None
        stage = "TRAIN_E01_E60_CAPTURE_E55"
        for epoch in range(1, MAX_EPOCHS + 1):
            started = time.perf_counter()
            losses = _train_epoch(
                model=model,
                batcher=batcher,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                interval_edges=edges,
                loss_config=_loss_config(method["loss"]),
                survival_multiplier=_survival_multiplier(epoch, method["loss"]),
                event_dropout_probability=float(training["event_dropout_probability"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                amp=True,
                max_batches=None,
            )
            if losses["optimizer_steps"] < 1:
                raise TrainWorkerError(f"epoch {epoch} had no optimizer update")
            scheduler.step()
            _append_jsonl(
                history_path,
                {
                    "epoch": epoch,
                    "losses": losses,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "seconds": time.perf_counter() - started,
                    "heldout_metric_calls": 0,
                },
            )
            print(
                f"outer={outer} epoch={epoch}/{MAX_EPOCHS} "
                f"loss={float(losses['total']):.9f} optimizer_steps={int(losses['optimizer_steps'])}",
                flush=True,
            )
            if epoch == FIXED_EPOCH:
                stage = "SEAL_E55"
                trainable = {
                    name: value.detach().cpu().clone()
                    for name, value in model.named_parameters()
                    if value.requires_grad
                }
                buffers = {
                    name: value.detach().cpu().clone() for name, value in model.named_buffers()
                }
                torch.save(
                    {
                        "schema_version": CHECKPOINT_SCHEMA,
                        "status": "FIXED_E55_CAPTURED_CONTINUE_TO_E60",
                        "outer_fold": outer,
                        "epoch": FIXED_EPOCH,
                        "replication_contract_sha256": expected_contract_sha256,
                        "parent_d07_003_sha256": PARENT_SHA256,
                        "train_authorization_sha256": expected_authorization_sha256,
                        "materialized_config_hash": run_config["materialized_config_hash"],
                        "age_anchor_train_role_only": age_anchor,
                        "initial_trainable_parameter_sha256": initial_hash,
                        "trainable_parameter_sha256": trainable_parameter_hash(model),
                        "trainable_state_dict": trainable,
                        "buffer_state_sha256": _tensor_map_hash(buffers),
                        "buffer_state_dict": buffers,
                        "from_random_initialization": True,
                        "learned_checkpoint_input": False,
                        "epoch_selection_performed": False,
                    },
                    checkpoint_path,
                )
                checkpoint_sha = sha256_file(checkpoint_path)
                stage = "TRAIN_E01_E60_CAPTURE_E55"
        if checkpoint_sha is None or sha256_file(checkpoint_path) != checkpoint_sha:
            raise TrainWorkerError("fixed E55 checkpoint missing or changed during E56-E60")
        final_hash = trainable_parameter_hash(model)
        if final_hash == initial_hash:
            raise TrainWorkerError("E60 trainable state did not change from initialization")
        history_rows = [
            json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
        ]
        if (
            [int(row.get("epoch", -1)) for row in history_rows]
            != list(range(1, MAX_EPOCHS + 1))
            or any(float(row.get("losses", {}).get("optimizer_steps", 0)) < 1 for row in history_rows)
        ):
            raise TrainWorkerError("training history coverage/optimizer-step audit failed")
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        completion_path = attempt_dir / "TRAINING_COMPLETE.json"
        _write_json_exclusive(
            completion_path,
            {
                "schema_version": TRAINING_COMPLETE_SCHEMA,
                "status": "FULL_60_EPOCHS_COMPLETE_E55_SEALED",
                "created_utc": _utc_now(),
                "outer_fold": outer,
                "completed_epochs": MAX_EPOCHS,
                "fixed_checkpoint_epoch": FIXED_EPOCH,
                "replication_contract_sha256": expected_contract_sha256,
                "train_authorization_path": str(authorization_path.resolve()),
                "train_authorization_sha256": expected_authorization_sha256,
                "role_artifact_validation_sha256": expected_role_validation_sha256,
                "outer_control_manifest_sha256": authorization[
                    "outer_control_manifest_sha256"
                ],
                "run_config_path": str(run_config_path.resolve()),
                "run_config_sha256": sha256_file(run_config_path),
                "model_preflight_path": str(preflight_path.resolve()),
                "model_preflight_sha256": sha256_file(preflight_path),
                "training_history_path": str(history_path.resolve()),
                "training_history_sha256": sha256_file(history_path),
                "fixed_e55_checkpoint_path": str(checkpoint_path.resolve()),
                "fixed_e55_checkpoint_sha256": checkpoint_sha,
                "fixed_e55_checkpoint_bytes": checkpoint_path.stat().st_size,
                "fixed_e55_trainable_parameter_sha256": checkpoint_payload[
                    "trainable_parameter_sha256"
                ],
                "initial_trainable_parameter_sha256": initial_hash,
                "epoch60_trainable_parameter_sha256": final_hash,
                "calibration_read_count": 0,
                "outer_read_count": 0,
                "resume_used": False,
                "environment": _environment(),
            },
        )
        return completion_path
    except BaseException as exc:
        failure = attempt_dir / "TRAINING_FAILURE.json"
        if not failure.exists():
            _write_json_exclusive(
                failure,
                {
                    "schema_version": FAILURE_SCHEMA,
                    "status": "FAILED_CREATE_NEW_ATTEMPT_FROM_EPOCH_1",
                    "stage": stage,
                    "outer_fold": outer,
                    "resume_allowed": False,
                    "exception": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        raise


def _sha(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("expected a 64-hex SHA256")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated D07-003 fixed-replication train worker")
    parser.add_argument("--train-authorization", type=Path, required=True)
    parser.add_argument("--expected-train-authorization-sha256", type=_sha, required=True)
    parser.add_argument("--expected-contract-sha256", type=_sha, required=True)
    parser.add_argument("--expected-role-validation-sha256", type=_sha, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_worker(
        authorization_path=args.train_authorization,
        expected_authorization_sha256=args.expected_train_authorization_sha256,
        expected_contract_sha256=args.expected_contract_sha256,
        expected_role_validation_sha256=args.expected_role_validation_sha256,
        attempt_dir=args.attempt_dir,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
