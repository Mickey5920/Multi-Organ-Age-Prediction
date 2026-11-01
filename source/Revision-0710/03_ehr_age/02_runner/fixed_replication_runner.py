from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from d07_candidate_runner import (
    _tensor_mapping_hash,
    load_frozen_candidate_contract,
    restore_trainable_checkpoint,
)
from d07_open_fold0_development import write_artifact_hash_registry
from ehr_age_data import (
    LABEL_COLUMNS,
    REVISION_ROOT,
    Fold0DevelopmentData,
    JointLabels,
    JointVectorizedBatcher,
    canonical_hash,
    load_restricted_compact_cache,
    sha256_file,
)
from ehr_age_model import EHRAGEModel, ORGAN_SYSTEMS
from ehr_age_training import (
    _model_config,
    _predict,
    configure_reproducibility,
    trainable_parameter_hash,
)

DOWNSTREAM_COMMON = REVISION_ROOT / "05_downstream" / "common"
if str(DOWNSTREAM_COMMON) not in sys.path:
    sys.path.insert(0, str(DOWNSTREAM_COMMON))
from crossfit_metrics import (  # noqa: E402
    LinearAgeCorrectionModel,
    OrganLinearCorrection,
    apply_linear_age_correction,
    apply_linear_age_correction_to_validation,
    fit_linear_age_correction,
)


RUNNER_DIR = Path(__file__).resolve().parent
EHR_AGE_ROOT = RUNNER_DIR.parent
WORKSPACE_ROOT = REVISION_ROOT.parent
DEFAULT_REPLICATION_CONTRACT = (
    EHR_AGE_ROOT
    / "00_contract"
    / "ehr_age_d07_003_e55_fixed_replication_v1.json"
)
DEFAULT_D07_003_CONTRACT = (
    EHR_AGE_ROOT / "00_contract" / "ehr_age_d07_003_private_retain_v1.json"
)
DEFAULT_D07_003_SHA256 = (
    "73c46ce884458f39148aed47a5843be76d55e96db8253c4a88bb30f70ebaf84c"
)
DEFAULT_CACHE_DIR = REVISION_ROOT / "01_data_rebuild" / "04_cache_v1"
DEFAULT_OUTCOMES = (
    REVISION_ROOT / "01_data_rebuild" / "02_outcomes_60m" / "organ_outcomes_60m.parquet"
)
DEFAULT_AGE_TARGETS = DEFAULT_CACHE_DIR / "targets" / "chronological_age_targets.parquet"
DEFAULT_FOLD_MANIFEST = (
    REVISION_ROOT / "01_data_rebuild" / "03_folds" / "fold_manifest_5fold.parquet"
)
DEFAULT_FOLD_ROLES = (
    REVISION_ROOT / "01_data_rebuild" / "03_folds" / "outer_fold_roles.parquet"
)
DEFAULT_PARTICIPANT_MASKS = (
    REVISION_ROOT
    / "01_data_rebuild"
    / "02_outcomes_60m"
    / "participant_followup_and_evaluation_masks.parquet"
)
DEFAULT_REPLICATION_ROOT = EHR_AGE_ROOT / "04_fixed_replication"
DEFAULT_ROLE_ROOT = DEFAULT_REPLICATION_ROOT / "00_role_artifacts"
DEFAULT_OUTPUT_ROOT = DEFAULT_REPLICATION_ROOT / "01_runs"
DEFAULT_AGGREGATE_ROOT = DEFAULT_REPLICATION_ROOT / "03_evaluation"
EXPECTED_CACHE_BINDINGS = {
    "hard_gate": DEFAULT_CACHE_DIR / "cache_hard_gate.json",
    "input_gate": DEFAULT_CACHE_DIR / "cache_input_gate.json",
    "leakage_gate": REVISION_ROOT / "01_data_rebuild" / "05_leakage_audit" / "leakage_gate.json",
    "batch_manifest": DEFAULT_CACHE_DIR / "batch_manifest.json",
    "embedding_tables": DEFAULT_CACHE_DIR / "embedding_tables.pt",
    "fold_transform_manifest": DEFAULT_CACHE_DIR / "fold_transform_manifest.json",
    "cache_run_manifest": DEFAULT_CACHE_DIR / "cache_run_manifest.json",
}

REPLICATION_SCHEMA = "revision0710_ehr_age_d07_003_e55_fixed_replication_v1"
ROLE_REGISTRY_SCHEMA = "revision0710_ehr_age_fixed_replication_role_registry_v1"
ROLE_MANIFEST_SCHEMA = "revision0710.ehr_age.fixed_replication.single_role_manifest.v1"
OUTER_CONTROL_MANIFEST_SCHEMA = "revision0710.ehr_age.fixed_replication.outer_control_manifest.v1"
TRAIN_AUTHORIZATION_SCHEMA = "revision0710.ehr_age.fixed_replication.train_only_authorization.v1"
TRAIN_METHOD_SPEC_SCHEMA = "revision0710.ehr_age.fixed_replication.sanitized_method_spec.v1"
AGE_CORRECTION_SCHEMA = "revision0710.ehr_age.fixed_replication.age_correction.v1"
COX_NUISANCE_SCHEMA = "revision0710.ehr_age.fixed_replication.cox_nuisance.v1"
NUISANCE_MANIFEST_SCHEMA = "revision0710.ehr_age.fixed_replication.nuisance_manifest.v1"
RUN_CONFIG_SCHEMA = "revision0710.ehr_age.fixed_replication.run.v1"
FIXED_CHECKPOINT_SCHEMA = "revision0710.ehr_age.fixed_replication.e55_checkpoint.v1"
TRAINING_COMPLETE_SCHEMA = "revision0710.ehr_age.fixed_replication.training_complete.v1"
REUSE_TRAINING_COMPLETE_SCHEMA = "revision0710.ehr_age.fixed_replication.reuse_training_complete.v1"
EVALUATION_MANIFEST_SCHEMA = "revision0710.ehr_age.fixed_replication.evaluation_manifest.v1"
OUTER_LEDGER_SCHEMA = "revision0710.ehr_age.fixed_replication.outer_read.v1"
RECEIPT_SCHEMA = "revision0710.ehr_age.fixed_replication.receipt.v1"
RECEIPT_SELECTION_SCHEMA = "revision0710.ehr_age.fixed_replication.receipt_selection.v1"
FAILURE_SCHEMA = "revision0710.ehr_age.fixed_replication.failure.v1"
PREDICTION_COLUMNS = (
    "participant_id",
    "organ_system",
    "actual_age",
    "predicted_age",
    "outer_fold",
    "split_role",
)
FIXED_EPOCH = 55
MAX_EPOCHS = 60
SEED = 7210
FORMAL_FOLDS = (1, 2, 3, 4)
ALL_FOLDS = (0, 1, 2, 3, 4)


class FixedReplicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FoldRoles:
    outer_fold: int
    calibration_fold: int
    train_folds: tuple[int, int, int]


@dataclass(frozen=True)
class ReplicationContract:
    path: Path
    sha256: str
    payload: dict[str, Any]
    parent: Any
    roles: dict[int, FoldRoles]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_parquet(path, index=False)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def cyclic_fold_roles(outer_fold: int) -> FoldRoles:
    outer = int(outer_fold)
    if outer not in ALL_FOLDS:
        raise ValueError("outer_fold must be one of 0..4")
    calibration = (outer + 1) % 5
    train = tuple(fold for fold in ALL_FOLDS if fold not in {outer, calibration})
    return FoldRoles(outer, calibration, train)  # type: ignore[arg-type]


def _relative_or_absolute(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _contract_roles(payload: Mapping[str, Any]) -> dict[int, FoldRoles]:
    raw_roles = payload.get("roles")
    if isinstance(raw_roles, Mapping):
        rows = list(raw_roles.values())
    elif isinstance(raw_roles, list):
        rows = raw_roles
    else:
        raise FixedReplicationError("replication contract roles must be a list or mapping")
    if len(rows) != 5:
        raise FixedReplicationError("replication contract must contain exactly five role rows")
    observed: dict[int, FoldRoles] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FixedReplicationError("replication role row must be an object")
        outer = int(row.get("outer_fold", -1))
        role = FoldRoles(
            outer_fold=outer,
            calibration_fold=int(row.get("calibration_fold", -1)),
            train_folds=tuple(int(value) for value in row.get("train_folds", ())),  # type: ignore[arg-type]
        )
        if outer in observed:
            raise FixedReplicationError(f"duplicate outer-fold role row: {outer}")
        if role != cyclic_fold_roles(outer):
            raise FixedReplicationError(f"contract violates cyclic role mapping at outer fold {outer}")
        expected_mode = "reuse_d07_003_e55" if outer == 0 else "train_from_scratch"
        if row.get("mode") != expected_mode or row.get("seed") != SEED:
            raise FixedReplicationError(f"contract role mode/seed changed at outer fold {outer}")
        observed[outer] = role
    if set(observed) != set(ALL_FOLDS):
        raise FixedReplicationError("replication contract must freeze roles for outer folds 0..4")
    return observed


def _binding_sha(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("sha256", ""))
    return str(value)


def validate_data_lineage(
    contract: ReplicationContract,
    *,
    outcomes_path: Path = DEFAULT_OUTCOMES,
    age_targets_path: Path = DEFAULT_AGE_TARGETS,
    fold_manifest_path: Path = DEFAULT_FOLD_MANIFEST,
    fold_roles_path: Path = DEFAULT_FOLD_ROLES,
    participant_masks_path: Path = DEFAULT_PARTICIPANT_MASKS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, str]:
    lineage = contract.payload.get("data_lineage")
    if not isinstance(lineage, Mapping):
        raise FixedReplicationError("replication contract lacks data_lineage")
    source_declared = lineage.get("source_sha256", lineage.get("sources"))
    cache_declared = lineage.get("cache_sha256", lineage.get("cache"))
    if not isinstance(source_declared, Mapping) or not isinstance(cache_declared, Mapping):
        raise FixedReplicationError("replication data_lineage must bind sources and cache")
    source_paths = {
        "organ_outcomes_60m": outcomes_path,
        "chronological_age_targets": age_targets_path,
        "fold_manifest_5fold": fold_manifest_path,
        "outer_fold_roles": fold_roles_path,
        "participant_masks": participant_masks_path,
    }
    observed: dict[str, str] = {}
    for name, path in source_paths.items():
        digest = sha256_file(path.resolve())
        if digest != _binding_sha(source_declared.get(name)):
            raise FixedReplicationError(f"replication source lineage changed: {name}")
        observed[name] = digest
    for name, default_path in EXPECTED_CACHE_BINDINGS.items():
        path = cache_dir / default_path.relative_to(DEFAULT_CACHE_DIR) if default_path.is_relative_to(DEFAULT_CACHE_DIR) else default_path
        digest = sha256_file(path.resolve())
        if digest != _binding_sha(cache_declared.get(name)):
            raise FixedReplicationError(f"replication cache lineage changed: {name}")
        observed[name] = digest
    transforms = json.loads((cache_dir / "fold_transform_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(transforms, list) or {int(row.get("outer_fold", -1)) for row in transforms} != set(ALL_FOLDS):
        raise FixedReplicationError("fold transform manifest coverage changed")
    for row in transforms:
        transform_path = cache_dir / str(row["path"])
        if sha256_file(transform_path) != row.get("sha256"):
            raise FixedReplicationError(f"outer-fold transform changed: {row.get('outer_fold')}")
    return observed


def _require_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise FixedReplicationError(f"replication contract field must be false: {key}")


def load_replication_contract(
    path: Path = DEFAULT_REPLICATION_CONTRACT,
    *,
    expected_sha256: str | None = None,
) -> ReplicationContract:
    path = path.resolve()
    observed_sha = sha256_file(path)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise FixedReplicationError(
            f"replication contract hash mismatch: expected={expected_sha256}, observed={observed_sha}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != REPLICATION_SCHEMA
        or payload.get("status") != "FROZEN_BEFORE_REPLICATION"
        or payload.get("decision") != "SUBSTANTIVE_BASELINE_SUPERIORITY_FREEZE_E55"
        or payload.get("execution_started") is not False
        or payload.get("formal_d07_status_preserved") != "NON_PROMOTABLE"
        or payload.get("formal_gate_change_allowed") is not False
    ):
        raise FixedReplicationError("replication contract is not the frozen E55 contract")

    parent_ref = payload.get("parent_d07_003")
    if not isinstance(parent_ref, Mapping):
        raise FixedReplicationError("replication contract lacks parent_d07_003")
    parent_path = _relative_or_absolute(WORKSPACE_ROOT, parent_ref.get("path", ""))
    parent_sha = str(parent_ref.get("sha256", ""))
    if parent_sha != DEFAULT_D07_003_SHA256:
        raise FixedReplicationError("replication contract changed the D07-003 substantive parent")
    parent = load_frozen_candidate_contract(parent_path, expected_sha256=parent_sha)
    if parent.candidate_id != "003_private_retain":
        raise FixedReplicationError("replication parent is not D07-003 private RETAIN")
    fixed_epoch = int(parent_ref.get("fixed_epoch", payload.get("fixed_epoch", -1)))
    if fixed_epoch != FIXED_EPOCH:
        raise FixedReplicationError("replication checkpoint must be fixed at epoch 55")

    training = payload.get("training_freeze", payload.get("training"))
    if not isinstance(training, Mapping):
        raise FixedReplicationError("replication contract lacks training_freeze")
    required_training = {
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "fixed_checkpoint_epoch": FIXED_EPOCH,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "scheduler_horizon_epochs": MAX_EPOCHS,
        "performance_early_stopping_enabled": False,
        "full_budget_required": True,
        "evaluate_only_epoch": FIXED_EPOCH,
        "batch_size": 2048,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": 3,
        "event_dropout_probability": 0.1,
        "deterministic": True,
        "tf32": True,
        "amp": True,
    }
    mismatches = {
        key: (training.get(key), expected)
        for key, expected in required_training.items()
        if training.get(key) != expected
    }
    if mismatches:
        raise FixedReplicationError(f"replication training freeze changed: {mismatches}")

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise FixedReplicationError("replication contract lacks protocol")
    for key in (
        "warm_start",
        "fine_tune",
        "teacher",
        "distillation",
        "outer_metric_selection",
        "outer_metric_tuning",
        "calibration_metric_selection",
        "calibration_metric_tuning",
        "checkpoint_averaging",
        "ensemble",
        "seed_retry",
        "fold_exclusion_after_result",
        "partial_results_for_hr_km_allowed",
        "target_metrics_other_epochs_allowed",
        "training_worker_reads_full_role_registry",
        "training_worker_reads_calibration_or_outer_role_manifest",
        "training_phase_resume_allowed",
    ):
        _require_false(protocol, key)
    if protocol.get("outer_1_to_4_from_scratch") is not True:
        raise FixedReplicationError("folds 1-4 must start from scratch")
    if protocol.get("outer_1_to_4_no_learned_input") is not True:
        raise FixedReplicationError("folds 1-4 cannot consume learned state")
    role_contract = payload.get("role_artifact_contract", {})
    if role_contract.get("physical_separation_required") is not True:
        raise FixedReplicationError("physical role separation is required")
    if protocol.get("no_calibration_label_read_before_epoch_60_and_e55_seal") is not True:
        raise FixedReplicationError("calibration may only be read after full training")
    if protocol.get("no_target_label_read_before_ledger") is not True:
        raise FixedReplicationError("outer access must follow the E55 evaluation manifest")
    roles = _contract_roles(payload)
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(key) is not True
        for key in (
            "role_build_authorized",
            "validation_authorized",
            "formal_training_authorized",
            "aggregation_authorized",
        )
    ):
        raise FixedReplicationError("replication contract has not authorized every formal stage")
    return ReplicationContract(path, observed_sha, payload, parent, roles)


def validate_code_lineage(contract: ReplicationContract) -> dict[str, str]:
    lineage = contract.payload.get("code_lineage")
    if not isinstance(lineage, Mapping):
        raise FixedReplicationError("replication contract lacks code_lineage")
    observed: dict[str, str] = {}
    policy_keys = {"code_hashes_must_match_before_role_build"}
    artifact_keys = set(lineage).difference(policy_keys)
    expected_keys = {
        "ehr_age_model",
        "private_retain_age",
        "ehr_age_losses",
        "ehr_age_data",
        "ehr_age_training",
        "d07_candidate_runner",
        "d07_open_fold0_development",
        "compact_cache_dataset",
        "crossfit_metrics",
        "post_receipt_parity_evaluator",
        "fixed_replication_runner",
        "fixed_replication_train_worker",
        "fixed_replication_entrypoint",
    }
    if artifact_keys != expected_keys or lineage.get("code_hashes_must_match_before_role_build") is not True:
        raise FixedReplicationError("replication code lineage key set changed")
    for name in sorted(artifact_keys):
        binding = lineage[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise FixedReplicationError(f"malformed code-lineage binding: {name}")
        path = _relative_or_absolute(WORKSPACE_ROOT, binding["path"])
        digest = sha256_file(path)
        if digest != binding["sha256"]:
            raise FixedReplicationError(f"replication code lineage changed: {name}")
        observed[name] = digest
    return observed


def _validate_people_and_outcomes(
    folds: pd.DataFrame,
    ages: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = folds.loc[:, ["participant_id", "fold_id"]].copy()
    ages = ages.loc[:, ["participant_id", "fold_id", "chronological_age"]].copy()
    outcome_columns = [
        column for column in LABEL_COLUMNS if column not in {"inner_role", "fold_id"}
    ]
    outcomes = outcomes.loc[:, outcome_columns].copy()
    for frame in (folds, ages, outcomes):
        frame["participant_id"] = frame["participant_id"].astype(str)
    folds["fold_id"] = folds["fold_id"].astype(np.int8)
    ages["fold_id"] = ages["fold_id"].astype(np.int8)
    if folds["participant_id"].duplicated().any() or ages["participant_id"].duplicated().any():
        raise FixedReplicationError("fold/age source contains duplicate participant IDs")
    if set(folds["fold_id"].unique()) != set(ALL_FOLDS):
        raise FixedReplicationError("fold manifest must contain folds 0..4")
    fold_age = folds.merge(
        ages,
        on=["participant_id", "fold_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not fold_age["_merge"].eq("both").all():
        raise FixedReplicationError("age targets and fold manifest do not have identical coverage")
    if not np.isfinite(fold_age["chronological_age"].to_numpy(dtype=np.float64)).all():
        raise FixedReplicationError("age targets contain a non-finite value")
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise FixedReplicationError("outcomes contain duplicate participant-organ rows")
    if set(outcomes["organ_system"].astype(str).unique()) != set(ORGAN_SYSTEMS):
        raise FixedReplicationError("outcomes changed the seven-organ axis")
    counts = outcomes.groupby("participant_id")["organ_system"].agg(["size", "nunique"])
    if not counts["size"].eq(len(ORGAN_SYSTEMS)).all() or not counts["nunique"].eq(
        len(ORGAN_SYSTEMS)
    ).all():
        raise FixedReplicationError("outcomes must contain seven unique organs per participant")
    if set(outcomes["participant_id"]) != set(folds["participant_id"]):
        raise FixedReplicationError("outcomes and fold manifest participant coverage changed")
    return folds, ages, outcomes


def _canonical_ids(frame: pd.DataFrame) -> str:
    return canonical_hash({"participant_ids": sorted(frame["participant_id"].astype(str).tolist())})


def _validate_authoritative_roles(
    folds: pd.DataFrame,
    roles: pd.DataFrame,
    participant_masks: pd.DataFrame,
) -> None:
    roles = roles.loc[:, ["participant_id", "fold_id", "outer_fold", "split_role"]].copy()
    roles["participant_id"] = roles["participant_id"].astype(str)
    masks = participant_masks.loc[:, ["participant_id"]].copy()
    masks["participant_id"] = masks["participant_id"].astype(str)
    if masks["participant_id"].duplicated().any() or set(masks["participant_id"]) != set(
        folds["participant_id"].astype(str)
    ):
        raise FixedReplicationError("participant masks and fold manifest coverage changed")
    if roles.duplicated(["participant_id", "outer_fold"]).any():
        raise FixedReplicationError("outer-fold role table contains duplicate rows")
    expected_rows = len(folds) * len(ALL_FOLDS)
    if len(roles) != expected_rows or set(roles["participant_id"]) != set(folds["participant_id"]):
        raise FixedReplicationError("outer-fold role table coverage changed")
    merged = roles.merge(
        folds.loc[:, ["participant_id", "fold_id"]],
        on="participant_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_manifest"),
    )
    if not merged["fold_id"].astype(int).eq(merged["fold_id_manifest"].astype(int)).all():
        raise FixedReplicationError("outer-fold roles disagree with fold assignment")
    expected_role = np.where(
        merged["fold_id"].astype(int).eq(merged["outer_fold"].astype(int)),
        "test",
        np.where(
            merged["fold_id"].astype(int).eq((merged["outer_fold"].astype(int) + 1) % 5),
            "validation",
            "train",
        ),
    )
    if not merged["split_role"].astype(str).eq(expected_role).all():
        raise FixedReplicationError("outer-fold roles violate the cyclic contract")


def _validate_staged_role_tree(
    *,
    staging_root: Path,
    contract: ReplicationContract,
    registry_path: Path,
    expected_participant_ids: set[str],
    write_report: bool = True,
) -> Path:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if len(registry.get("folds", ())) != 5:
        raise FixedReplicationError("staged registry does not contain five outer folds")
    authoritative_folds = pd.read_parquet(
        DEFAULT_FOLD_MANIFEST, columns=["participant_id", "fold_id"]
    )
    authoritative_folds["participant_id"] = authoritative_folds["participant_id"].astype(str)
    fold_map = authoritative_folds.set_index("participant_id")["fold_id"].astype(int)
    method_path = staging_root / "TRAIN_METHOD_SPEC.json"
    method = json.loads(method_path.read_text(encoding="utf-8"))
    exact_method_keys = set(
        contract.payload["role_artifact_contract"]["sanitized_method_spec"][
            "allowed_top_level_keys_exact"
        ]
    )
    if set(method) != exact_method_keys:
        raise FixedReplicationError("staged sanitized method spec key set changed")
    parquet_count = single_manifest_count = control_count = auth_count = 0
    for outer in ALL_FOLDS:
        control_path = staging_root / f"outer_fold_{outer}" / "outer_role_manifest.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        registry_row = next(row for row in registry["folds"] if int(row["outer_fold"]) == outer)
        expected_mode = "reuse_d07_003_e55" if outer == 0 else "train_from_scratch"
        if (
            control.get("schema_version") != OUTER_CONTROL_MANIFEST_SCHEMA
            or control.get("outer_fold") != outer
            or control.get("replication_contract_sha256") != contract.sha256
            or registry_row.get("manifest_sha256") != sha256_file(control_path)
            or registry_row.get("mode") != expected_mode
        ):
            raise FixedReplicationError(f"staged outer control manifest failed: {outer}")
        control_count += 1
        role_ids: dict[str, set[str]] = {}
        for ordinal, role in enumerate(("train", "calibration", "outer_test"), start=1):
            role_dir = staging_root / f"outer_fold_{outer}" / f"{ordinal:02d}_{role}"
            age_path = role_dir / "age_targets.parquet"
            label_path = role_dir / "joint_labels.parquet"
            role_manifest_path = role_dir / "role_manifest.json"
            age = pd.read_parquet(age_path)
            labels = pd.read_parquet(label_path)
            manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
            if list(age.columns) != ["participant_id", "fold_id", "chronological_age"]:
                raise FixedReplicationError(f"staged {outer}/{role} age schema changed")
            if list(labels.columns) != list(LABEL_COLUMNS):
                raise FixedReplicationError(f"staged {outer}/{role} label schema changed")
            ids = set(age["participant_id"].astype(str))
            expected_role_folds = {
                "train": set(contract.roles[outer].train_folds),
                "calibration": {contract.roles[outer].calibration_fold},
                "outer_test": {outer},
            }[role]
            expected_ids = set(fold_map.loc[fold_map.isin(expected_role_folds)].index)
            if (
                age["participant_id"].astype(str).duplicated().any()
                or labels.duplicated(["participant_id", "organ_system"]).any()
                or len(labels) != len(ids) * len(ORGAN_SYSTEMS)
                or set(labels["participant_id"].astype(str)) != ids
                or set(labels["organ_system"].astype(str)) != set(ORGAN_SYSTEMS)
                or set(labels["inner_role"].astype(str)) != {role}
                or ids != expected_ids
                or set(age["fold_id"].astype(int).unique()) != expected_role_folds
                or set(labels["fold_id"].astype(int).unique()) != expected_role_folds
            ):
                raise FixedReplicationError(f"staged {outer}/{role} identity coverage changed")
            for column in ("event", "at_risk", "strict_healthy_reference_60m"):
                if not set(pd.to_numeric(labels[column], errors="raise").unique()).issubset({0, 1}):
                    raise FixedReplicationError(f"staged {outer}/{role} {column} is not binary")
            if (labels["event"].eq(1) & labels["at_risk"].ne(1)).any():
                raise FixedReplicationError(f"staged {outer}/{role} event is outside risk set")
            risk_duration = pd.to_numeric(
                labels.loc[labels["at_risk"].eq(1), "duration_months"], errors="coerce"
            )
            if risk_duration.isna().any() or not np.isfinite(risk_duration).all() or risk_duration.le(0).any():
                raise FixedReplicationError(f"staged {outer}/{role} risk duration invalid")
            strict_counts = labels.groupby(labels["participant_id"].astype(str))[
                "strict_healthy_reference_60m"
            ].nunique()
            if strict_counts.ne(1).any() or not np.isfinite(
                pd.to_numeric(age["chronological_age"], errors="coerce").to_numpy()
            ).all():
                raise FixedReplicationError(f"staged {outer}/{role} age/strict consistency failed")
            entry = control["roles"][role]
            if (
                entry["participant_count"] != len(ids)
                or entry["participant_organ_rows"] != len(labels)
                or entry["participant_ids_canonical_sha256"]
                != canonical_hash({"participant_ids": sorted(ids)})
                or entry["age_targets"]["sha256"] != sha256_file(age_path)
                or entry["joint_labels"]["sha256"] != sha256_file(label_path)
                or entry["role_manifest"]["sha256"] != sha256_file(role_manifest_path)
                or manifest.get("schema_version") != ROLE_MANIFEST_SCHEMA
                or manifest.get("role") != role
            ):
                raise FixedReplicationError(f"staged {outer}/{role} manifest binding changed")
            role_ids[role] = ids
            parquet_count += 2
            single_manifest_count += 1
        if any(
            role_ids[left].intersection(role_ids[right])
            for left, right in (("train", "calibration"), ("train", "outer_test"), ("calibration", "outer_test"))
        ) or set().union(*role_ids.values()) != expected_participant_ids:
            raise FixedReplicationError(f"staged role partition failed for outer {outer}")
        auth_path = staging_root / f"outer_fold_{outer}" / "TRAIN_ONLY_AUTHORIZATION.json"
        if outer in FORMAL_FOLDS:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            exact_auth_keys = set(
                contract.payload["role_artifact_contract"]["train_only_authorization"][
                    "allowed_top_level_keys_exact"
                ]
            )
            if set(auth) != exact_auth_keys or auth.get("outer_fold") != outer:
                raise FixedReplicationError(f"staged train authorization failed for outer {outer}")
            if registry_row.get("train_authorization_sha256") != sha256_file(auth_path):
                raise FixedReplicationError(f"staged registry train authorization failed: {outer}")
            auth_count += 1
        elif auth_path.exists():
            raise FixedReplicationError("outer Fold0 must not have a train authorization")
    report_path = staging_root / "ROLE_ARTIFACT_VALIDATION.PASS.json"
    report_payload = {
            "schema_version": "revision0710.ehr_age.fixed_replication.role_artifact_validation.v1",
            "status": "PASS",
            "replication_contract_sha256": contract.sha256,
            "role_registry_sha256": sha256_file(registry_path),
            "parquet_count": parquet_count,
            "single_role_manifest_count": single_manifest_count,
            "outer_control_manifest_count": control_count,
            "train_authorization_count": auth_count,
            "sanitized_method_spec_sha256": sha256_file(method_path),
            "all_role_sets_pairwise_disjoint_and_exhaustive": True,
        }
    if write_report:
        _write_json_exclusive(report_path, report_payload)
    else:
        if json.loads(report_path.read_text(encoding="utf-8")) != report_payload:
            raise FixedReplicationError("published role-artifact PASS report changed")
    return report_path


def _build_role_artifacts_impl(
    *,
    contract_path: Path,
    expected_contract_sha256: str | None,
    outcomes_path: Path,
    age_targets_path: Path,
    fold_manifest_path: Path,
    outer_fold_roles_path: Path = DEFAULT_FOLD_ROLES,
    participant_masks_path: Path = DEFAULT_PARTICIPANT_MASKS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_root: Path,
    staging_root: Path | None = None,
) -> Path:
    """Trusted preprocessing step; no model process may read the source tables."""

    contract = load_replication_contract(contract_path, expected_sha256=expected_contract_sha256)
    validate_code_lineage(contract)
    validate_data_lineage(
        contract,
        outcomes_path=outcomes_path,
        age_targets_path=age_targets_path,
        fold_manifest_path=fold_manifest_path,
        fold_roles_path=outer_fold_roles_path,
        participant_masks_path=participant_masks_path,
        cache_dir=cache_dir,
    )
    canonical_root = output_root.resolve()
    if canonical_root.exists():
        raise FileExistsError(canonical_root)
    canonical_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = staging_root or canonical_root.with_name(
        f".{canonical_root.name}.building.{os.getpid()}.{time.time_ns()}"
    )
    staging_root.mkdir(exist_ok=False)
    def final_path(path: Path) -> Path:
        return canonical_root / path.resolve().relative_to(staging_root)
    try:
        folds, ages, outcomes = _validate_people_and_outcomes(
            pd.read_parquet(fold_manifest_path),
            pd.read_parquet(age_targets_path),
            pd.read_parquet(outcomes_path),
        )
        _validate_authoritative_roles(
            folds,
            pd.read_parquet(outer_fold_roles_path),
            pd.read_parquet(participant_masks_path),
        )
    except BaseException as exc:
        _write_json_exclusive(
            staging_root / "BUILD_FAILURE.json",
            {"status": "FAILED_BEFORE_MATERIALIZATION", "type": type(exc).__name__, "message": str(exc)},
        )
        raise
    source_bindings = {
        "fold_manifest": {"path": str(fold_manifest_path.resolve()), "sha256": sha256_file(fold_manifest_path)},
        "age_targets": {"path": str(age_targets_path.resolve()), "sha256": sha256_file(age_targets_path)},
        "organ_outcomes": {"path": str(outcomes_path.resolve()), "sha256": sha256_file(outcomes_path)},
        "outer_fold_roles": {"path": str(outer_fold_roles_path.resolve()), "sha256": sha256_file(outer_fold_roles_path)},
        "participant_masks": {"path": str(participant_masks_path.resolve()), "sha256": sha256_file(participant_masks_path)},
    }
    method_spec_path = staging_root / "TRAIN_METHOD_SPEC.json"
    _write_json_exclusive(
        method_spec_path,
        {
            "schema_version": TRAIN_METHOD_SPEC_SCHEMA,
            "status": "SANITIZED_LABEL_FREE_METHOD_FREEZE",
            "replication_contract_sha256": contract.sha256,
            "parent_d07_003_sha256": contract.parent.sha256,
            "candidate_id": contract.parent.candidate_id,
            "model": contract.parent.payload["model"],
            "loss": contract.parent.payload["loss"],
            "training": {
                key: contract.parent.payload["training"][key]
                for key in (
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                    "gradient_clip_norm",
                    "warmup_epochs",
                    "scheduler_horizon_epochs",
                    "event_dropout_probability",
                    "seed",
                    "deterministic",
                    "tf32",
                    "amp",
                )
            },
            "cache_sha256": contract.payload["data_lineage"]["cache_sha256"],
            "initial_trainable_parameter_sha256": (
                "cf54feb052f9b09556a1355e8e078c82dabbaa036c1e0387e05d85688373a8a8"
            ),
            "code_lineage": contract.payload["code_lineage"],
            "label_or_outcome_paths_present": False,
            "learned_checkpoint_inputs": [],
        },
    )
    registry_folds: list[dict[str, Any]] = []
    for outer in ALL_FOLDS:
        spec = contract.roles[outer]
        fold_dir = staging_root / f"outer_fold_{outer}"
        fold_dir.mkdir(exist_ok=False)
        role_folds: dict[str, tuple[int, ...]] = {
            "train": spec.train_folds,
            "calibration": (spec.calibration_fold,),
            "outer_test": (spec.outer_fold,),
        }
        role_entries: dict[str, Any] = {}
        role_id_sets: dict[str, set[str]] = {}
        for ordinal, (role, included_folds) in enumerate(role_folds.items(), start=1):
            role_dir = fold_dir / f"{ordinal:02d}_{role}"
            people = folds.loc[folds["fold_id"].isin(included_folds)].copy()
            role_age = ages.merge(
                people,
                on=["participant_id", "fold_id"],
                how="inner",
                validate="one_to_one",
            ).sort_values(["fold_id", "participant_id"])
            role_labels = outcomes.merge(
                people,
                on="participant_id",
                how="inner",
                validate="many_to_one",
            )
            role_labels["inner_role"] = role
            role_labels["organ_system"] = pd.Categorical(
                role_labels["organ_system"], categories=list(ORGAN_SYSTEMS), ordered=True
            )
            role_labels = role_labels.sort_values(["fold_id", "participant_id", "organ_system"])
            role_labels = role_labels.loc[:, LABEL_COLUMNS]
            if len(role_age) != len(people) or len(role_labels) != len(people) * len(ORGAN_SYSTEMS):
                raise FixedReplicationError(f"outer {outer} {role} materialization coverage failed")
            age_path = role_dir / "age_targets.parquet"
            labels_path = role_dir / "joint_labels.parquet"
            _write_parquet_exclusive(age_path, role_age)
            _write_parquet_exclusive(labels_path, role_labels)
            role_id_sets[role] = set(people["participant_id"])
            role_entry = {
                "fold_ids": list(included_folds),
                "participant_count": int(len(people)),
                "participant_organ_rows": int(len(role_labels)),
                "participant_ids_canonical_sha256": _canonical_ids(people),
                "age_targets": {"path": str(final_path(age_path)), "sha256": sha256_file(age_path)},
                "joint_labels": {"path": str(final_path(labels_path)), "sha256": sha256_file(labels_path)},
            }
            role_manifest_path = role_dir / "role_manifest.json"
            _write_json_exclusive(
                role_manifest_path,
                {
                    "schema_version": ROLE_MANIFEST_SCHEMA,
                    "status": "SEALED",
                    "phase": "trusted_fixed_preprocessing_before_replication",
                    "replication_contract_sha256": contract.sha256,
                    "outer_fold": outer,
                    "role": role,
                    "allowed_reader_stage": {
                        "train": "TRAIN_WORKER_ONLY",
                        "calibration": "EVALUATION_AFTER_FULL_TRAINING_ONLY",
                        "outer_test": "OUTER_WORKER_AFTER_READ_LEDGER_ONLY",
                    }[role],
                    "source_bindings": source_bindings,
                    "artifact": role_entry,
                },
            )
            role_entry["role_manifest"] = {
                "path": str(final_path(role_manifest_path)),
                "sha256": sha256_file(role_manifest_path),
            }
            role_entries[role] = role_entry
        if any(
            role_id_sets[left].intersection(role_id_sets[right])
            for left, right in (("train", "calibration"), ("train", "outer_test"), ("calibration", "outer_test"))
        ):
            raise FixedReplicationError(f"outer {outer} role artifacts overlap")
        if set().union(*role_id_sets.values()) != set(folds["participant_id"]):
            raise FixedReplicationError(f"outer {outer} role artifacts do not cover the cohort")
        manifest_path = fold_dir / "outer_role_manifest.json"
        manifest = {
            "schema_version": OUTER_CONTROL_MANIFEST_SCHEMA,
            "status": "SEALED",
            "phase": "trusted_fixed_preprocessing_before_replication",
            "outer_fold": outer,
            "calibration_fold": spec.calibration_fold,
            "train_folds": list(spec.train_folds),
            "replication_contract_sha256": contract.sha256,
            "source_bindings": source_bindings,
            "model_process_must_read_only_materialized_artifacts": True,
            "roles_are_physically_separate": True,
            "roles": role_entries,
        }
        _write_json_exclusive(manifest_path, manifest)
        transform_rows = json.loads(
            (cache_dir / "fold_transform_manifest.json").read_text(encoding="utf-8")
        )
        transform_row = next(row for row in transform_rows if int(row["outer_fold"]) == outer)
        registry_row: dict[str, Any] = {
            "outer_fold": outer,
            "manifest_path": str(final_path(manifest_path)),
            "manifest_sha256": sha256_file(manifest_path),
        }
        if outer in FORMAL_FOLDS:
            train_entry = role_entries["train"]
            train_authorization_path = fold_dir / "TRAIN_ONLY_AUTHORIZATION.json"
            _write_json_exclusive(
                train_authorization_path,
                {
                "schema_version": TRAIN_AUTHORIZATION_SCHEMA,
                "status": "FROZEN_BEFORE_TRAINING",
                "replication_contract_sha256": contract.sha256,
                "parent_d07_003_sha256": contract.parent.sha256,
                "outer_fold": outer,
                "train_folds": list(spec.train_folds),
                "seed": SEED,
                "max_epochs": MAX_EPOCHS,
                "fixed_checkpoint_epoch": FIXED_EPOCH,
                "outer_control_manifest_sha256": sha256_file(manifest_path),
                "train_method_spec": {
                    "path": str(final_path(method_spec_path)),
                    "sha256": sha256_file(method_spec_path),
                },
                "train_role_manifest_sha256": train_entry["role_manifest"]["sha256"],
                "train_age_targets": train_entry["age_targets"],
                "train_joint_labels": train_entry["joint_labels"],
                "train_participant_count": train_entry["participant_count"],
                "train_participant_ids_canonical_sha256": train_entry[
                    "participant_ids_canonical_sha256"
                ],
                "label_free_cache": {
                    name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for name, default_path in EXPECTED_CACHE_BINDINGS.items()
                    for path in [cache_dir / default_path.relative_to(DEFAULT_CACHE_DIR) if default_path.is_relative_to(DEFAULT_CACHE_DIR) else default_path]
                },
                "outer_transform": {
                    "path": str((cache_dir / str(transform_row["path"])).resolve()),
                    "sha256": transform_row["sha256"],
                },
                "calibration_or_outer_paths_present": False,
                "learned_checkpoint_inputs": [],
                },
            )
            registry_row.update(
                {
                    "mode": "train_from_scratch",
                    "train_authorization_path": str(final_path(train_authorization_path)),
                    "train_authorization_sha256": sha256_file(train_authorization_path),
                }
            )
        else:
            registry_row.update(
                {
                    "mode": "reuse_d07_003_e55",
                    "reuse_checkpoint_sha256": contract.payload["fold0_reuse"]["checkpoint"]["file_sha256"],
                    "reuse_test_predictions_sha256": contract.payload["fold0_reuse"]["existing_test_predictions"]["sha256"],
                }
            )
        registry_folds.append(registry_row)
    registry_path = staging_root / "ROLE_ARTIFACT_REGISTRY.FROZEN.json"
    registry = {
        "schema_version": ROLE_REGISTRY_SCHEMA,
        "status": "PASS",
        "replication_contract_path": str(contract.path),
        "replication_contract_sha256": contract.sha256,
        "all_outer_fold_configs_frozen_together": True,
        "train_method_spec_path": str(final_path(method_spec_path)),
        "train_method_spec_sha256": sha256_file(method_spec_path),
        "source_tables_forbidden_to_training_runner": [
            str(outcomes_path.resolve()),
            str(age_targets_path.resolve()),
            str(fold_manifest_path.resolve()),
        ],
        "folds": registry_folds,
    }
    _write_json_exclusive(registry_path, registry)
    _validate_staged_role_tree(
        staging_root=staging_root,
        contract=contract,
        registry_path=registry_path,
        expected_participant_ids=set(folds["participant_id"].astype(str)),
    )
    try:
        staging_root.rename(canonical_root)
    except BaseException as exc:
        failure = staging_root / "BUILD_PROMOTION_FAILURE.json"
        if not failure.exists():
            _write_json_exclusive(
                failure,
                {"status": "FAILED_ATOMIC_PROMOTION", "type": type(exc).__name__, "message": str(exc)},
            )
        raise
    return canonical_root / registry_path.name


def build_role_artifacts(**kwargs: Any) -> Path:
    """Guard the complete staging lifecycle and preserve every failed attempt."""

    output_root = Path(kwargs["output_root"]).resolve()
    staging = output_root.with_name(
        f".{output_root.name}.building.{os.getpid()}.{time.time_ns()}"
    )
    try:
        return _build_role_artifacts_impl(**kwargs, staging_root=staging)
    except BaseException as exc:
        staging.mkdir(parents=True, exist_ok=True)
        failure = staging / "BUILD_FAILURE.json"
        if not failure.exists():
            _write_json_exclusive(
                failure,
                {
                    "schema_version": FAILURE_SCHEMA,
                    "status": "FAILED_CANONICAL_ROOT_NOT_PUBLISHED",
                    "canonical_root_exists": output_root.exists(),
                    "exception": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        if output_root.exists():
            raise FixedReplicationError(
                "role build failed after canonical root unexpectedly appeared"
            ) from exc
        raise


def load_role_registry(
    path: Path,
    contract: ReplicationContract,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise FixedReplicationError("formal role registry SHA256 must be explicit")
    if sha256_file(path) != expected_sha256:
        raise FixedReplicationError("fixed-replication role registry hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("folds")
    if (
        payload.get("schema_version") != ROLE_REGISTRY_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("replication_contract_sha256") != contract.sha256
        or payload.get("all_outer_fold_configs_frozen_together") is not True
        or not isinstance(rows, list)
        or {int(row.get("outer_fold", -1)) for row in rows} != set(ALL_FOLDS)
    ):
        raise FixedReplicationError("invalid fixed-replication role registry")
    return payload


def validate_published_role_artifacts(
    *,
    contract: ReplicationContract,
    role_root: Path,
    expected_registry_sha256: str,
    expected_validation_sha256: str,
) -> dict[str, Any]:
    registry_path = role_root / "ROLE_ARTIFACT_REGISTRY.FROZEN.json"
    load_role_registry(registry_path, contract, expected_sha256=expected_registry_sha256)
    folds = pd.read_parquet(DEFAULT_FOLD_MANIFEST, columns=["participant_id"])
    report_path = _validate_staged_role_tree(
        staging_root=role_root.resolve(),
        contract=contract,
        registry_path=registry_path,
        expected_participant_ids=set(folds["participant_id"].astype(str)),
        write_report=False,
    )
    if sha256_file(report_path) != expected_validation_sha256:
        raise FixedReplicationError("published role-artifact PASS report hash changed")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _verify_authorized_shards(cache_dir: Path, fold_ids: Sequence[int]) -> None:
    rows = json.loads((cache_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    selected = [row for row in rows if int(row["fold_id"]) in set(int(x) for x in fold_ids)]
    if not selected or set(int(row["fold_id"]) for row in selected) != set(int(x) for x in fold_ids):
        raise FixedReplicationError("authorized shard manifest lacks a requested fold")
    for row in selected:
        shard = cache_dir / str(row["path"])
        if sha256_file(shard) != row.get("sha256"):
            raise FixedReplicationError(f"authorized cache shard changed: {row['path']}")


def _restore_fixed_checkpoint(
    *,
    path: Path,
    model: EHRAGEModel,
    replication_contract_sha256: str,
    parent_d07_003_sha256: str,
    outer_fold: int,
    authorization_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != FIXED_CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "FIXED_E55_CAPTURED_CONTINUE_TO_E60"
        or checkpoint.get("epoch") != FIXED_EPOCH
        or checkpoint.get("outer_fold") != int(outer_fold)
        or checkpoint.get("replication_contract_sha256") != replication_contract_sha256
        or checkpoint.get("parent_d07_003_sha256") != parent_d07_003_sha256
        or checkpoint.get("train_authorization_sha256") != authorization_sha256
        or checkpoint.get("from_random_initialization") is not True
        or checkpoint.get("learned_checkpoint_input") is not False
    ):
        raise FixedReplicationError("fixed E55 checkpoint provenance changed")
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    state = checkpoint.get("trainable_state_dict")
    buffers = {name: value for name, value in model.named_buffers()}
    stored_buffers = checkpoint.get("buffer_state_dict")
    if not isinstance(state, Mapping) or set(state) != set(parameters):
        raise FixedReplicationError("fixed E55 trainable state changed")
    if (
        not isinstance(stored_buffers, Mapping)
        or set(stored_buffers) != set(buffers)
        or _tensor_mapping_hash(stored_buffers) != checkpoint.get("buffer_state_sha256")
    ):
        raise FixedReplicationError("fixed E55 buffer state changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
        for name, buffer in buffers.items():
            buffer.copy_(stored_buffers[name].to(device=buffer.device, dtype=buffer.dtype))
    if trainable_parameter_hash(model) != checkpoint.get("trainable_parameter_sha256"):
        raise FixedReplicationError("fixed E55 restored parameter hash changed")
    return checkpoint


def _fold_manifest_entry(
    registry: Mapping[str, Any], outer_fold: int, contract: ReplicationContract
) -> tuple[Path, dict[str, Any]]:
    row = next(
        item for item in registry["folds"] if int(item["outer_fold"]) == int(outer_fold)
    )
    manifest_path = Path(str(row["manifest_path"])).resolve()
    if sha256_file(manifest_path) != row.get("manifest_sha256"):
        raise FixedReplicationError(f"outer fold {outer_fold} role manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = contract.roles[int(outer_fold)]
    if (
        manifest.get("schema_version") != OUTER_CONTROL_MANIFEST_SCHEMA
        or manifest.get("status") != "SEALED"
        or manifest.get("outer_fold") != spec.outer_fold
        or manifest.get("calibration_fold") != spec.calibration_fold
        or tuple(manifest.get("train_folds", ())) != spec.train_folds
        or manifest.get("replication_contract_sha256") != contract.sha256
        or manifest.get("roles_are_physically_separate") is not True
    ):
        raise FixedReplicationError(f"outer fold {outer_fold} role manifest changed")
    return manifest_path, manifest


def _aligned_role_labels(
    frame: pd.DataFrame,
    participant_ids: Sequence[str],
    *,
    role: str,
    allowed_folds: Sequence[int],
) -> JointLabels:
    missing = set(LABEL_COLUMNS).difference(frame.columns)
    if missing:
        raise FixedReplicationError(f"role labels lack columns: {sorted(missing)}")
    frame = frame.loc[:, LABEL_COLUMNS].copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if set(frame["fold_id"].astype(int).unique()) != set(int(value) for value in allowed_folds):
        raise FixedReplicationError(f"{role} labels contain unexpected folds")
    if set(frame["inner_role"].astype(str).unique()) != {role}:
        raise FixedReplicationError(f"{role} labels contain an unexpected role")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise FixedReplicationError(f"{role} labels contain duplicates")
    index = pd.MultiIndex.from_product(
        [pd.Index([str(value) for value in participant_ids]), list(ORGAN_SYSTEMS)],
        names=["participant_id", "organ_system"],
    )
    aligned = frame.set_index(["participant_id", "organ_system"]).reindex(index)
    if aligned[["event", "at_risk", "strict_healthy_reference_60m"]].isna().any().any():
        raise FixedReplicationError(f"{role} labels are incomplete after alignment")
    missing_duration = aligned["duration_months"].isna()
    if (missing_duration & aligned["at_risk"].ne(0)).any() or (
        missing_duration & aligned["event"].ne(0)
    ).any():
        raise FixedReplicationError(f"{role} has missing at-risk/event duration")
    aligned["duration_months"] = aligned["duration_months"].fillna(0.0)
    n = len(participant_ids)
    duration = aligned["duration_months"].to_numpy(dtype=np.float32).reshape(n, -1)
    event = aligned["event"].to_numpy(dtype=np.int8).reshape(n, -1)
    at_risk = aligned["at_risk"].to_numpy(dtype=np.int8).reshape(n, -1)
    strict = aligned["strict_healthy_reference_60m"].to_numpy(dtype=np.int8).reshape(n, -1)
    if not np.all(strict == strict[:, :1]):
        raise FixedReplicationError(f"{role} strict status disagrees across organs")
    labels = JointLabels(
        strict_healthy=torch.from_numpy(strict[:, 0].astype(bool)),
        duration_months=torch.from_numpy(duration),
        event=torch.from_numpy(event.astype(bool)),
        at_risk=torch.from_numpy(at_risk.astype(bool)),
        inner_role=tuple([role] * n),
    )
    labels.validate(n)
    return labels


def load_role_data(
    *,
    cache_dir: Path,
    outer_fold: int,
    role: str,
    manifest: Mapping[str, Any],
    contract: ReplicationContract,
    role_manifest_path: Path,
    fixed_checkpoint_path: Path | None = None,
    training_complete_path: Path | None = None,
    evaluation_manifest_path: Path | None = None,
    outer_read_ledger: Path | None = None,
) -> Fold0DevelopmentData:
    if role not in {"train", "calibration", "outer_test"}:
        raise ValueError(f"unsupported role: {role}")
    manifest_sha = sha256_file(role_manifest_path)
    if manifest.get("replication_contract_sha256") != contract.sha256:
        raise FixedReplicationError("role manifest and replication contract disagree")
    if role in {"calibration", "outer_test"}:
        if training_complete_path is None or fixed_checkpoint_path is None:
            raise FixedReplicationError(f"{role} cannot be opened before training completion")
        completion = json.loads(training_complete_path.read_text(encoding="utf-8"))
        if (
            completion.get("schema_version") != TRAINING_COMPLETE_SCHEMA
            or completion.get("status") != "FULL_60_EPOCHS_COMPLETE_E55_SEALED"
            or int(completion.get("outer_fold", -1)) != int(outer_fold)
            or completion.get("replication_contract_sha256") != contract.sha256
            or completion.get("role_manifest_sha256") != manifest_sha
            or completion.get("fixed_e55_checkpoint_sha256") != sha256_file(fixed_checkpoint_path)
            or int(completion.get("completed_epochs", -1)) != MAX_EPOCHS
        ):
            raise FixedReplicationError("invalid training-complete receipt")
    if role == "outer_test":
        if outer_read_ledger is None:
            raise FixedReplicationError("outer role cannot be opened before its read ledger")
        if evaluation_manifest_path is None:
            raise FixedReplicationError("outer role requires a sealed evaluation manifest")
        ledger = json.loads(outer_read_ledger.read_text(encoding="utf-8"))
        entry = manifest.get("roles", {}).get("outer_test", {})
        if (
            ledger.get("schema_version") != OUTER_LEDGER_SCHEMA
            or ledger.get("status") != "READ_COMMITTED_BEFORE_OUTER_ACCESS"
            or int(ledger.get("outer_fold", -1)) != int(outer_fold)
            or ledger.get("replication_contract_sha256") != contract.sha256
            or ledger.get("role_manifest_sha256") != manifest_sha
            or ledger.get("fixed_e55_checkpoint_sha256") != sha256_file(fixed_checkpoint_path)
            or ledger.get("evaluation_manifest_path") != str(evaluation_manifest_path.resolve())
            or ledger.get("evaluation_manifest_sha256") != sha256_file(evaluation_manifest_path)
            or ledger.get("outer_age_targets_sha256") != entry.get("age_targets", {}).get("sha256")
            or ledger.get("outer_joint_labels_sha256") != entry.get("joint_labels", {}).get("sha256")
        ):
            raise FixedReplicationError("invalid outer read ledger")
    entry = manifest.get("roles", {}).get(role)
    if not isinstance(entry, Mapping):
        raise FixedReplicationError(f"role manifest lacks {role}")
    age_path = Path(str(entry["age_targets"]["path"])).resolve()
    labels_path = Path(str(entry["joint_labels"]["path"])).resolve()
    if sha256_file(age_path) != entry["age_targets"]["sha256"]:
        raise FixedReplicationError(f"{role} age artifact hash changed")
    if sha256_file(labels_path) != entry["joint_labels"]["sha256"]:
        raise FixedReplicationError(f"{role} label artifact hash changed")
    folds = tuple(int(value) for value in entry["fold_ids"])
    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=int(outer_fold),
        fold_ids=folds,
        age_target_cache_path=age_path,
    )
    if set(cache.dataset.fold_id.tolist()) != set(folds):
        raise FixedReplicationError(f"{role} cache shard folds changed")
    labels = _aligned_role_labels(
        pd.read_parquet(labels_path), cache.dataset.participant_ids, role=role, allowed_folds=folds
    )
    ids = pd.DataFrame({"participant_id": cache.dataset.participant_ids})
    if _canonical_ids(ids) != entry.get("participant_ids_canonical_sha256"):
        raise FixedReplicationError(f"{role} participant ID hash changed")
    indices = torch.arange(len(cache.dataset), dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(
        cache=cache,
        labels=labels,
        train_indices=indices if role == "train" else empty,
        calibration_indices=indices if role == "calibration" else empty,
        screen_indices=indices if role == "outer_test" else empty,
        label_manifest=dict(manifest),
    )


def _ids_sha(values: Sequence[str]) -> str:
    return canonical_hash({"participant_ids": sorted(str(value) for value in values)})


def _fit_one_cox_nuisance(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
    calibration_fold: int,
    organ: str,
    model_role: str,
    covariates: tuple[str, ...],
    validation_predictions_sha256: str,
    provenance: Mapping[str, str],
    penalizer: float = 1e-4,
) -> dict[str, Any]:
    required = {
        "participant_id",
        "organ_system",
        "actual_age",
        "sex",
        "corrected_age_gap",
        "duration",
        "event",
        "at_risk",
    }
    if not required.issubset(frame.columns):
        raise FixedReplicationError(f"calibration Cox frame lacks {sorted(required.difference(frame.columns))}")
    subset = frame.loc[
        frame["organ_system"].astype(str).eq(organ) & frame["at_risk"].eq(1)
    ].copy()
    if subset.empty or subset["event"].sum() < 1:
        raise FixedReplicationError(f"calibration Cox set is not evaluable for {organ}")
    arrays = {
        name: pd.to_numeric(subset[name], errors="raise").to_numpy(dtype=np.float64)
        for name in (*covariates, "duration", "event")
    }
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise FixedReplicationError(f"calibration Cox set contains non-finite values for {organ}")
    means = {name: float(arrays[name].mean()) for name in covariates}
    scales = {name: float(arrays[name].std(ddof=0)) for name in covariates}
    if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise FixedReplicationError(f"calibration Cox covariate has no variation for {organ}")
    train = pd.DataFrame(
        {
            "duration": arrays["duration"],
            "event": arrays["event"].astype(np.int8),
        }
    )
    transformed = []
    for name in covariates:
        column = f"z_{name}"
        transformed.append(column)
        train[column] = (arrays[name] - means[name]) / scales[name]
    from lifelines import CoxPHFitter

    model = CoxPHFitter(penalizer=penalizer)
    model.fit(
        train[["duration", "event", *transformed]],
        duration_col="duration",
        event_col="event",
        show_progress=False,
    )
    coefficients = [float(model.params_.loc[column]) for column in transformed]
    if not np.isfinite(np.asarray(coefficients)).all():
        raise FixedReplicationError(f"calibration Cox coefficients are non-finite for {organ}")
    ids = subset["participant_id"].astype(str).tolist()
    return {
        "schema_version": COX_NUISANCE_SCHEMA,
        "status": "FITTED_AND_SEALED_BEFORE_OUTER_READ",
        "outer_fold": int(outer_fold),
        "calibration_fold": int(calibration_fold),
        "organ_system": organ,
        "model_role": model_role,
        "covariate_order": list(covariates),
        "calibration_participant_ids_sha256": _ids_sha(frame["participant_id"].astype(str).unique().tolist()),
        "at_risk_participant_ids_sha256": _ids_sha(ids),
        "feature_means": means,
        "feature_scales": scales,
        "penalizer": float(penalizer),
        "coefficients": coefficients,
        "fit_participant_count": int(len(subset)),
        "fit_event_count": int(subset["event"].sum()),
        "fit_diagnostics": {
            "converged_without_exception": True,
            "log_likelihood": float(model.log_likelihood_),
            "coefficient_count": len(coefficients),
        },
        "validation_predictions_sha256": validation_predictions_sha256,
        **dict(provenance),
    }


def fit_and_seal_calibration_nuisance(
    calibration_frame: pd.DataFrame,
    *,
    outer_fold: int,
    calibration_fold: int,
    validation_predictions_path: Path,
    output_dir: Path,
    replication_contract_sha256: str,
    calibration_role_manifest_sha256: str,
    training_complete_sha256: str,
    fixed_e55_checkpoint_sha256: str,
) -> tuple[Path, Path, pd.DataFrame]:
    """Fit every nuisance parameter before any outer-label read is authorized."""

    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_sha = sha256_file(validation_predictions_path)
    persisted = pd.read_parquet(validation_predictions_path)
    try:
        pd.testing.assert_frame_equal(
            persisted.reset_index(drop=True),
            calibration_frame.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise FixedReplicationError("calibration frame is not the hash-bound prediction artifact") from exc
    if (
        set(persisted["outer_fold"].astype(int).unique()) != {int(outer_fold)}
        or set(persisted["split_role"].astype(str).unique()) != {"validation"}
    ):
        raise FixedReplicationError("calibration prediction artifact has invalid fold/role")
    provenance = {
        "replication_contract_sha256": replication_contract_sha256,
        "calibration_role_manifest_sha256": calibration_role_manifest_sha256,
        "training_complete_sha256": training_complete_sha256,
        "fixed_e55_checkpoint_sha256": fixed_e55_checkpoint_sha256,
    }
    correction = fit_linear_age_correction(calibration_frame)
    strict_ids = calibration_frame.loc[
        calibration_frame["strict_healthy_reference_60m"].eq(1), "participant_id"
    ].astype(str)
    correction_payload = {
        "schema_version": AGE_CORRECTION_SCHEMA,
        "status": "FITTED_AND_SEALED_BEFORE_OUTER_READ",
        "outer_fold": int(outer_fold),
        "calibration_fold": int(calibration_fold),
        "calibration_participant_ids_sha256": _ids_sha(
            calibration_frame["participant_id"].astype(str).unique().tolist()
        ),
        "strict_reference_participant_ids_sha256": _ids_sha(strict_ids.unique().tolist()),
        "validation_predictions_sha256": prediction_sha,
        **provenance,
        "parameters": [
            {
                "organ_system": item.organ_system,
                "intercept": item.intercept,
                "slope": item.age_slope,
                "fit_population_count": item.n_reference_rows,
            }
            for item in correction.parameters
        ],
    }
    correction_path = output_dir / "age_correction.json"
    _write_json_exclusive(correction_path, correction_payload)
    corrected = apply_linear_age_correction_to_validation(calibration_frame, correction)
    nuisance_dir = output_dir / "cox_nuisance"
    nuisance_dir.mkdir(exist_ok=False)
    bindings: list[dict[str, Any]] = []
    for organ in ORGAN_SYSTEMS:
        for model_role, covariates in (
            ("base", ("actual_age", "sex")),
            ("full", ("actual_age", "sex", "corrected_age_gap")),
        ):
            payload = _fit_one_cox_nuisance(
                corrected,
                outer_fold=outer_fold,
                calibration_fold=calibration_fold,
                organ=organ,
                model_role=model_role,
                covariates=covariates,
                validation_predictions_sha256=prediction_sha,
                provenance=provenance,
            )
            path = nuisance_dir / f"{organ}__{model_role}.json"
            _write_json_exclusive(path, payload)
            bindings.append(
                {
                    "organ_system": organ,
                    "model_role": model_role,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
    manifest_path = output_dir / "COX_NUISANCE_MANIFEST.json"
    _write_json_exclusive(
        manifest_path,
        {
            "schema_version": NUISANCE_MANIFEST_SCHEMA,
            "status": "ALL_14_MODELS_SEALED_BEFORE_OUTER_READ",
            "outer_fold": int(outer_fold),
            "calibration_fold": int(calibration_fold),
            "validation_predictions_sha256": prediction_sha,
            **provenance,
            "age_correction_path": str(correction_path.resolve()),
            "age_correction_sha256": sha256_file(correction_path),
            "model_count": 14,
            "models": bindings,
        },
    )
    return correction_path, manifest_path, corrected


def _load_correction(path: Path, *, expected_sha256: str, outer_fold: int) -> LinearAgeCorrectionModel:
    if sha256_file(path) != expected_sha256:
        raise FixedReplicationError("sealed age-correction hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != AGE_CORRECTION_SCHEMA
        or payload.get("status") != "FITTED_AND_SEALED_BEFORE_OUTER_READ"
        or payload.get("outer_fold") != int(outer_fold)
    ):
        raise FixedReplicationError("invalid sealed age correction")
    parameters = tuple(
        OrganLinearCorrection(
            organ_system=str(row["organ_system"]),
            intercept=float(row["intercept"]),
            age_slope=float(row["slope"]),
            n_reference_rows=int(row["fit_population_count"]),
        )
        for row in payload["parameters"]
    )
    return LinearAgeCorrectionModel(
        parameters=parameters,
        validation_participant_ids=frozenset(),
    )


def score_outer_with_sealed_nuisance(
    outer_frame: pd.DataFrame,
    *,
    outer_fold: int,
    correction_path: Path,
    correction_sha256: str,
    nuisance_manifest_path: Path,
    nuisance_manifest_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score only. This function has no Cox fitting path."""

    correction = _load_correction(
        correction_path, expected_sha256=correction_sha256, outer_fold=outer_fold
    )
    corrected = apply_linear_age_correction(outer_frame, correction)
    if sha256_file(nuisance_manifest_path) != nuisance_manifest_sha256:
        raise FixedReplicationError("sealed Cox nuisance manifest hash changed")
    manifest = json.loads(nuisance_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != NUISANCE_MANIFEST_SCHEMA
        or manifest.get("status") != "ALL_14_MODELS_SEALED_BEFORE_OUTER_READ"
        or manifest.get("outer_fold") != int(outer_fold)
        or manifest.get("model_count") != 14
        or len(manifest.get("models", ())) != 14
    ):
        raise FixedReplicationError("invalid sealed Cox nuisance manifest")
    from lifelines.utils import concordance_index

    organ_rows: dict[str, Any] = {}
    deltas: list[float] = []
    for organ in ORGAN_SYSTEMS:
        subset = corrected.loc[
            corrected["organ_system"].astype(str).eq(organ) & corrected["at_risk"].eq(1)
        ].copy()
        scores: dict[str, float] = {}
        for model_role in ("base", "full"):
            binding = next(
                row
                for row in manifest["models"]
                if row["organ_system"] == organ and row["model_role"] == model_role
            )
            path = Path(str(binding["path"])).resolve()
            if sha256_file(path) != binding["sha256"]:
                raise FixedReplicationError(f"sealed Cox nuisance changed: {organ}/{model_role}")
            nuisance = json.loads(path.read_text(encoding="utf-8"))
            if (
                nuisance.get("schema_version") != COX_NUISANCE_SCHEMA
                or nuisance.get("status") != "FITTED_AND_SEALED_BEFORE_OUTER_READ"
                or nuisance.get("outer_fold") != int(outer_fold)
                or nuisance.get("organ_system") != organ
                or nuisance.get("model_role") != model_role
            ):
                raise FixedReplicationError(f"invalid Cox nuisance: {organ}/{model_role}")
            covariates = list(nuisance["covariate_order"])
            matrix = np.column_stack(
                [
                    (
                        pd.to_numeric(subset[name], errors="raise").to_numpy(dtype=np.float64)
                        - float(nuisance["feature_means"][name])
                    )
                    / float(nuisance["feature_scales"][name])
                    for name in covariates
                ]
            )
            risk = matrix @ np.asarray(nuisance["coefficients"], dtype=np.float64)
            c_value = concordance_index(
                pd.to_numeric(subset["duration"], errors="raise").to_numpy(dtype=np.float64),
                -risk,
                pd.to_numeric(subset["event"], errors="raise").to_numpy(dtype=np.int8).astype(bool),
            )
            if not math.isfinite(float(c_value)):
                raise FixedReplicationError(f"outer Cox score is non-finite: {organ}/{model_role}")
            scores[model_role] = float(c_value)
        delta = scores["full"] - scores["base"]
        deltas.append(delta)
        organ_rows[organ] = {
            "c_base": scores["base"],
            "c_plus": scores["full"],
            "delta_c": delta,
            "n_test": int(len(subset)),
            "test_events": int(subset["event"].sum()),
        }
    return (
        {
            "definition": "validation-fitted sealed Cox nuisance; outer score only",
            "macro_delta_c": float(np.mean(deltas)),
            "positive_organs": int(sum(value > 0.0 for value in deltas)),
            "evaluable_organs": len(deltas),
            "organs": organ_rows,
            "outer_refit_calls": 0,
        },
        corrected,
    )


def _validate_training_complete(
    path: Path,
    *,
    expected_sha256: str,
    contract: ReplicationContract,
    outer_fold: int,
    outer_control_manifest_sha256: str,
    expected_role_validation_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise FixedReplicationError("training-complete receipt hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_files = (
        ("run_config_path", "run_config_sha256"),
        ("model_preflight_path", "model_preflight_sha256"),
        ("training_history_path", "training_history_sha256"),
        ("fixed_e55_checkpoint_path", "fixed_e55_checkpoint_sha256"),
    )
    if (
        payload.get("schema_version") != TRAINING_COMPLETE_SCHEMA
        or payload.get("status") != "FULL_60_EPOCHS_COMPLETE_E55_SEALED"
        or payload.get("outer_fold") != int(outer_fold)
        or payload.get("completed_epochs") != MAX_EPOCHS
        or payload.get("fixed_checkpoint_epoch") != FIXED_EPOCH
        or payload.get("replication_contract_sha256") != contract.sha256
        or payload.get("outer_control_manifest_sha256") != outer_control_manifest_sha256
        or payload.get("role_artifact_validation_sha256") != expected_role_validation_sha256
        or payload.get("calibration_read_count") != 0
        or payload.get("outer_read_count") != 0
        or payload.get("resume_used") is not False
    ):
        raise FixedReplicationError("invalid training-complete receipt")
    for path_key, sha_key in required_files:
        artifact = Path(str(payload.get(path_key, ""))).resolve()
        if sha256_file(artifact) != payload.get(sha_key):
            raise FixedReplicationError(f"training receipt artifact changed: {path_key}")
    run_config = json.loads(Path(payload["run_config_path"]).read_text(encoding="utf-8"))
    preflight = json.loads(Path(payload["model_preflight_path"]).read_text(encoding="utf-8"))
    if (
        run_config.get("worker") != "ISOLATED_TRAIN_ONLY"
        or run_config.get("outer_fold") != int(outer_fold)
        or run_config.get("replication_contract_sha256") != contract.sha256
        or run_config.get("max_epochs") != MAX_EPOCHS
        or run_config.get("fixed_epoch") != FIXED_EPOCH
        or run_config.get("resume_supported") is not False
        or run_config.get("heldout_inputs") != []
        or preflight.get("status") != "PASS"
        or preflight.get("outer_fold") != int(outer_fold)
        or preflight.get("initial_hash_matches_frozen_d07_003") is not True
        or preflight.get("optimizer_binds_exact_trainable_set") is not True
        or preflight.get("heldout_read_count") != 0
    ):
        raise FixedReplicationError("training run-config/preflight provenance changed")
    rows = [json.loads(line) for line in Path(payload["training_history_path"]).read_text(encoding="utf-8").splitlines()]
    if (
        [int(row.get("epoch", -1)) for row in rows] != list(range(1, MAX_EPOCHS + 1))
        or any(row.get("heldout_metric_calls") != 0 for row in rows)
        or any(float(row.get("losses", {}).get("optimizer_steps", 0)) < 1 for row in rows)
        or any(
            not all(math.isfinite(float(value)) for value in row.get("losses", {}).values())
            for row in rows
        )
    ):
        raise FixedReplicationError("training history is not the complete E1-E60 trajectory")
    checkpoint = torch.load(Path(payload["fixed_e55_checkpoint_path"]), map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != FIXED_CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "FIXED_E55_CAPTURED_CONTINUE_TO_E60"
        or checkpoint.get("outer_fold") != int(outer_fold)
        or checkpoint.get("epoch") != FIXED_EPOCH
        or checkpoint.get("replication_contract_sha256") != contract.sha256
        or checkpoint.get("train_authorization_sha256") != payload["train_authorization_sha256"]
        or checkpoint.get("from_random_initialization") is not True
        or checkpoint.get("learned_checkpoint_input") is not False
    ):
        raise FixedReplicationError("fixed E55 checkpoint provenance changed")
    return payload


def _validate_cache_only(contract: ReplicationContract, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    declared = contract.payload["data_lineage"]["cache_sha256"]
    for name, default_path in EXPECTED_CACHE_BINDINGS.items():
        path = cache_dir / default_path.relative_to(DEFAULT_CACHE_DIR) if default_path.is_relative_to(DEFAULT_CACHE_DIR) else default_path
        if sha256_file(path) != declared.get(name):
            raise FixedReplicationError(f"evaluation cache lineage changed: {name}")


def _load_materialized_role(
    *,
    contract: ReplicationContract,
    cache_dir: Path,
    outer_fold: int,
    role: str,
    entry: Mapping[str, Any],
) -> Fold0DevelopmentData:
    role_manifest_path = Path(str(entry["role_manifest"]["path"])).resolve()
    if sha256_file(role_manifest_path) != entry["role_manifest"]["sha256"]:
        raise FixedReplicationError(f"{role} single-role manifest changed")
    role_manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    if (
        role_manifest.get("schema_version") != ROLE_MANIFEST_SCHEMA
        or role_manifest.get("status") != "SEALED"
        or role_manifest.get("replication_contract_sha256") != contract.sha256
        or role_manifest.get("outer_fold") != int(outer_fold)
        or role_manifest.get("role") != role
        or role_manifest.get("allowed_reader_stage")
        != {
            "calibration": "EVALUATION_AFTER_FULL_TRAINING_ONLY",
            "outer_test": "OUTER_WORKER_AFTER_READ_LEDGER_ONLY",
        }.get(role, "TRAIN_WORKER_ONLY")
    ):
        raise FixedReplicationError(f"invalid {role} single-role manifest")
    manifest_artifact = role_manifest.get("artifact")
    if not isinstance(manifest_artifact, Mapping):
        raise FixedReplicationError(f"{role} single-role manifest lacks artifact binding")
    for key in (
        "fold_ids",
        "participant_count",
        "participant_organ_rows",
        "participant_ids_canonical_sha256",
        "age_targets",
        "joint_labels",
    ):
        if manifest_artifact.get(key) != entry.get(key):
            raise FixedReplicationError(f"{role} single/control manifest disagreement: {key}")
    age_path = Path(str(entry["age_targets"]["path"])).resolve()
    labels_path = Path(str(entry["joint_labels"]["path"])).resolve()
    if sha256_file(age_path) != entry["age_targets"]["sha256"] or sha256_file(labels_path) != entry[
        "joint_labels"
    ]["sha256"]:
        raise FixedReplicationError(f"{role} materialized artifact changed")
    folds = tuple(int(value) for value in entry["fold_ids"])
    transforms = json.loads((cache_dir / "fold_transform_manifest.json").read_text(encoding="utf-8"))
    transform_row = next(row for row in transforms if int(row["outer_fold"]) == int(outer_fold))
    if sha256_file(cache_dir / str(transform_row["path"])) != transform_row["sha256"]:
        raise FixedReplicationError("outer-fold transform changed before role load")
    _verify_authorized_shards(cache_dir, folds)
    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=outer_fold,
        fold_ids=folds,
        age_target_cache_path=age_path,
    )
    labels = _aligned_role_labels(
        pd.read_parquet(labels_path), cache.dataset.participant_ids, role=role, allowed_folds=folds
    )
    if len(cache.dataset) != int(entry["participant_count"]):
        raise FixedReplicationError(f"{role} participant count changed")
    ids = pd.DataFrame({"participant_id": cache.dataset.participant_ids})
    if _canonical_ids(ids) != entry["participant_ids_canonical_sha256"]:
        raise FixedReplicationError(f"{role} participant IDs changed")
    indices = torch.arange(len(cache.dataset), dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    return Fold0DevelopmentData(
        cache,
        labels,
        empty,
        indices if role == "calibration" else empty,
        indices if role == "outer_test" else empty,
        dict(role_manifest),
    )


def _prediction_ids(data: Fold0DevelopmentData, prediction: Any) -> list[str]:
    return [
        str(data.cache.dataset.participant_ids[int(index)])
        for index in prediction.participant_indices.tolist()
    ]


def _authoritative_prediction_frame(
    *,
    predicted_age: np.ndarray,
    participant_ids: Sequence[str],
    data: Fold0DevelopmentData,
    role_entry: Mapping[str, Any],
    outer_fold: int,
    split_role: str,
) -> pd.DataFrame:
    n = len(participant_ids)
    matrix = np.asarray(predicted_age, dtype=np.float64)
    if matrix.shape != (n, len(ORGAN_SYSTEMS)) or not np.isfinite(matrix).all():
        raise FixedReplicationError("prediction matrix shape changed")
    if (
        n != int(role_entry["participant_count"])
        or canonical_hash({"participant_ids": sorted(str(value) for value in participant_ids)})
        != role_entry["participant_ids_canonical_sha256"]
    ):
        raise FixedReplicationError("prediction participant coverage changed")
    predictions = pd.DataFrame(
        {
            "participant_id": np.repeat(np.asarray(participant_ids, dtype=object), len(ORGAN_SYSTEMS)),
            "organ_system": np.tile(np.asarray(ORGAN_SYSTEMS, dtype=object), n),
            "predicted_age": matrix.reshape(-1),
        }
    )
    age = pd.read_parquet(Path(role_entry["age_targets"]["path"]))
    age["participant_id"] = age["participant_id"].astype(str)
    labels = pd.read_parquet(Path(role_entry["joint_labels"]["path"]))
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels = labels.rename(columns={"duration_months": "duration"})
    sex_map = {
        str(pid): float(value)
        for pid, value in zip(
            data.cache.dataset.participant_ids,
            data.cache.dataset.inputs["sex"].numpy().tolist(),
        )
    }
    frame = predictions.merge(
        age.loc[:, ["participant_id", "chronological_age"]].rename(
            columns={"chronological_age": "actual_age"}
        ),
        on="participant_id",
        how="left",
        validate="many_to_one",
    ).merge(
        labels.loc[:, [
            "participant_id",
            "organ_system",
            "strict_healthy_reference_60m",
            "duration",
            "event",
            "at_risk",
        ]],
        on=["participant_id", "organ_system"],
        how="left",
        validate="one_to_one",
    )
    frame["sex"] = frame["participant_id"].map(sex_map)
    frame["outer_fold"] = int(outer_fold)
    frame["split_role"] = split_role
    duration = pd.to_numeric(frame["duration"], errors="coerce")
    risk = frame["at_risk"].eq(1)
    if (
        duration.loc[risk].isna().any()
        or not np.isfinite(duration.loc[risk].to_numpy()).all()
        or duration.loc[risk].le(0.0).any()
        or (frame["event"].eq(1) & ~risk).any()
    ):
        raise FixedReplicationError("authoritative risk labels are invalid")
    frame["duration"] = duration.fillna(0.0)
    required_non_null = frame.columns.difference(["duration"])
    if frame.loc[:, required_non_null].isna().any().any() or len(frame) != n * len(ORGAN_SYSTEMS):
        raise FixedReplicationError("authoritative prediction frame is incomplete")
    return frame


def _raw_strict_frame_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    strict = frame["strict_healthy_reference_60m"].eq(1)
    actual = frame.loc[strict, "actual_age"].to_numpy(dtype=np.float64)
    predicted = frame.loc[strict, "predicted_age"].to_numpy(dtype=np.float64)
    residual = predicted - actual
    return {
        "row_n": int(len(actual)),
        "participant_n": int(frame.loc[strict, "participant_id"].nunique()),
        "mae": float(np.abs(residual).mean()),
        "pearson": float(np.corrcoef(actual, predicted)[0, 1]),
        "r2": float(1.0 - np.square(residual).sum() / np.square(actual - actual.mean()).sum()),
    }


def _restore_replication_model(
    *,
    contract: ReplicationContract,
    data: Fold0DevelopmentData,
    checkpoint_path: Path,
    outer_fold: int,
    train_authorization_sha256: str,
) -> tuple[EHRAGEModel, torch.device]:
    if not torch.cuda.is_available():
        raise FixedReplicationError("formal calibration/outer inference requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    age_anchor = float(checkpoint["age_anchor_train_role_only"])
    configure_reproducibility(SEED, deterministic=True, tf32=True)
    model = EHRAGEModel(
        data.cache.code_embedding,
        data.cache.text_embedding,
        age_anchor=age_anchor,
        config=_model_config(contract.parent.payload["model"]),
    )
    device = torch.device("cuda")
    model.to(device)
    _restore_fixed_checkpoint(
        path=checkpoint_path,
        model=model,
        replication_contract_sha256=contract.sha256,
        parent_d07_003_sha256=contract.parent.sha256,
        outer_fold=outer_fold,
        authorization_sha256=train_authorization_sha256,
    )
    return model, device


def prepare_fold_evaluation(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    role_registry_path: Path,
    expected_role_registry_sha256: str,
    expected_role_validation_sha256: str,
    training_complete_path: Path,
    expected_training_complete_sha256: str,
    evaluation_dir: Path,
) -> tuple[Path, Path]:
    """Calibration worker. It seals correction/Cox nuisance and never opens outer artifacts."""

    contract = load_replication_contract(contract_path, expected_sha256=expected_contract_sha256)
    validate_code_lineage(contract)
    _validate_cache_only(contract)
    registry = load_role_registry(
        role_registry_path, contract, expected_sha256=expected_role_registry_sha256
    )
    training_raw = json.loads(training_complete_path.read_text(encoding="utf-8"))
    outer = int(training_raw.get("outer_fold", -1))
    if outer not in FORMAL_FOLDS:
        raise FixedReplicationError("evaluation worker only accepts outer folds 1..4")
    control_path, control = _fold_manifest_entry(registry, outer, contract)
    control_sha = sha256_file(control_path)
    training = _validate_training_complete(
        training_complete_path,
        expected_sha256=expected_training_complete_sha256,
        contract=contract,
        outer_fold=outer,
        outer_control_manifest_sha256=control_sha,
        expected_role_validation_sha256=expected_role_validation_sha256,
    )
    checkpoint_path = Path(training["fixed_e55_checkpoint_path"]).resolve()
    checkpoint_sha = training["fixed_e55_checkpoint_sha256"]
    evaluation_dir = evaluation_dir.resolve()
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    stage = "LOAD_CALIBRATION_AFTER_E60"
    try:
        calibration_entry = control["roles"]["calibration"]
        calibration_manifest_sha = calibration_entry["role_manifest"]["sha256"]
        authorization_path = evaluation_dir / "CALIBRATION_READ_AUTHORIZATION.json"
        _write_json_exclusive(
            authorization_path,
            {
                "status": "AUTHORIZED_AFTER_E60_AND_E55_SEAL",
                "outer_fold": outer,
                "replication_contract_sha256": contract.sha256,
                "outer_control_manifest_sha256": control_sha,
                "training_complete_sha256": expected_training_complete_sha256,
                "fixed_e55_checkpoint_sha256": checkpoint_sha,
                "calibration_role_manifest_sha256": calibration_manifest_sha,
                "calibration_age_targets_sha256": calibration_entry["age_targets"]["sha256"],
                "calibration_joint_labels_sha256": calibration_entry["joint_labels"]["sha256"],
            },
        )
        calibration = _load_materialized_role(
            contract=contract,
            cache_dir=DEFAULT_CACHE_DIR,
            outer_fold=outer,
            role="calibration",
            entry=calibration_entry,
        )
        model, device = _restore_replication_model(
            contract=contract,
            data=calibration,
            checkpoint_path=checkpoint_path,
            outer_fold=outer,
            train_authorization_sha256=training["train_authorization_sha256"],
        )
        stage = "PREDICT_CALIBRATION_E55"
        batcher = JointVectorizedBatcher(
            calibration,
            calibration.calibration_indices,
            batch_size=int(contract.parent.payload["training"]["batch_size"]),
            shuffle=False,
            seed=SEED,
        )
        prediction = _predict(
            model=model,
            batcher=batcher,
            device=device,
            amp=bool(contract.parent.payload["training"]["amp"]),
            max_batches=None,
        )
        ids = _prediction_ids(calibration, prediction)
        internal = _authoritative_prediction_frame(
            predicted_age=prediction.predicted_age,
            participant_ids=ids,
            data=calibration,
            role_entry=calibration_entry,
            outer_fold=outer,
            split_role="validation",
        )
        internal_path = evaluation_dir / "validation_predictions_internal.parquet"
        _write_parquet_exclusive(internal_path, internal)
        standard_path = evaluation_dir / "validation_predictions_long.parquet"
        _write_parquet_exclusive(standard_path, internal.loc[:, PREDICTION_COLUMNS])
        stage = "SEAL_EVALUATION_BEFORE_OUTER"
        nuisance_dir = evaluation_dir / "sealed_nuisance"
        correction_path, nuisance_manifest_path, corrected = fit_and_seal_calibration_nuisance(
            internal,
            outer_fold=outer,
            calibration_fold=contract.roles[outer].calibration_fold,
            validation_predictions_path=internal_path,
            output_dir=nuisance_dir,
            replication_contract_sha256=contract.sha256,
            calibration_role_manifest_sha256=calibration_manifest_sha,
            training_complete_sha256=expected_training_complete_sha256,
            fixed_e55_checkpoint_sha256=checkpoint_sha,
        )
        corrected_path = evaluation_dir / "validation_predictions_corrected.parquet"
        _write_parquet_exclusive(corrected_path, corrected)
        nuisance_manifest = json.loads(nuisance_manifest_path.read_text(encoding="utf-8"))
        evaluation_manifest_path = evaluation_dir / "EVALUATION_MANIFEST_BEFORE_OUTER.json"
        outer_entry = control["roles"]["outer_test"]
        _write_json_exclusive(
            evaluation_manifest_path,
            {
                "schema_version": EVALUATION_MANIFEST_SCHEMA,
                "status": "SEALED_BEFORE_OUTER_READ",
                "outer_fold": outer,
                "replication_contract_sha256": contract.sha256,
                "role_registry_sha256": expected_role_registry_sha256,
                "outer_control_manifest_path": str(control_path.resolve()),
                "outer_control_manifest_sha256": control_sha,
                "training_complete_path": str(training_complete_path.resolve()),
                "training_complete_sha256": expected_training_complete_sha256,
                "fixed_e55_checkpoint_path": str(checkpoint_path),
                "fixed_e55_checkpoint_sha256": checkpoint_sha,
                "calibration_role_manifest_sha256": calibration_manifest_sha,
                "validation_predictions_internal": {"path": str(internal_path), "sha256": sha256_file(internal_path)},
                "validation_predictions_long": {"path": str(standard_path), "sha256": sha256_file(standard_path)},
                "validation_predictions_corrected": {"path": str(corrected_path), "sha256": sha256_file(corrected_path)},
                "age_correction": {"path": str(correction_path), "sha256": sha256_file(correction_path)},
                "cox_nuisance_manifest": {"path": str(nuisance_manifest_path), "sha256": sha256_file(nuisance_manifest_path)},
                "cox_nuisance_models": nuisance_manifest["models"],
                "outer_role_manifest_sha256": outer_entry["role_manifest"]["sha256"],
                "outer_age_targets_sha256": outer_entry["age_targets"]["sha256"],
                "outer_joint_labels_sha256": outer_entry["joint_labels"]["sha256"],
                "outer_metric_or_label_read_count": 0,
                "evaluated_epoch": FIXED_EPOCH,
            },
        )
        stage = "COMMIT_OUTER_READ"
        ledger_path = evaluation_dir / "OUTER_READ_LEDGER.json"
        _write_json_exclusive(
            ledger_path,
            {
                "schema_version": OUTER_LEDGER_SCHEMA,
                "status": "READ_COMMITTED_BEFORE_OUTER_ACCESS",
                "outer_fold": outer,
                "replication_contract_sha256": contract.sha256,
                "role_manifest_sha256": control_sha,
                "fixed_e55_checkpoint_sha256": checkpoint_sha,
                "evaluation_manifest_path": str(evaluation_manifest_path.resolve()),
                "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
                "validation_predictions_sha256": sha256_file(internal_path),
                "age_correction_sha256": sha256_file(correction_path),
                "cox_nuisance_manifest_sha256": sha256_file(nuisance_manifest_path),
                "outer_role_manifest_sha256": outer_entry["role_manifest"]["sha256"],
                "outer_age_targets_sha256": outer_entry["age_targets"]["sha256"],
                "outer_joint_labels_sha256": outer_entry["joint_labels"]["sha256"],
            },
        )
        return evaluation_manifest_path, ledger_path
    except BaseException as exc:
        failure = evaluation_dir / "CALIBRATION_STAGE_FAILURE.json"
        if not failure.exists():
            _write_json_exclusive(
                failure,
                {"schema_version": FAILURE_SCHEMA, "status": "FAILED", "stage": stage, "exception": {"type": type(exc).__name__, "message": str(exc)}},
            )
        raise


def _validate_evaluation_seal(
    *,
    contract: ReplicationContract,
    outer_fold: int,
    evaluation_manifest_path: Path,
    expected_evaluation_manifest_sha256: str,
    ledger_path: Path,
    expected_ledger_sha256: str,
    control_sha256: str,
    checkpoint_sha256: str,
    expected_role_registry_sha256: str,
    expected_training_complete_sha256: str,
    expected_calibration_role_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(evaluation_manifest_path) != expected_evaluation_manifest_sha256:
        raise FixedReplicationError("evaluation manifest hash changed")
    manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    required_bindings = (
        "validation_predictions_internal",
        "validation_predictions_long",
        "validation_predictions_corrected",
        "age_correction",
        "cox_nuisance_manifest",
    )
    if (
        manifest.get("schema_version") != EVALUATION_MANIFEST_SCHEMA
        or manifest.get("status") != "SEALED_BEFORE_OUTER_READ"
        or manifest.get("outer_fold") != int(outer_fold)
        or manifest.get("replication_contract_sha256") != contract.sha256
        or manifest.get("outer_control_manifest_sha256") != control_sha256
        or manifest.get("fixed_e55_checkpoint_sha256") != checkpoint_sha256
        or manifest.get("role_registry_sha256") != expected_role_registry_sha256
        or manifest.get("training_complete_sha256") != expected_training_complete_sha256
        or manifest.get("calibration_role_manifest_sha256")
        != expected_calibration_role_manifest_sha256
        or manifest.get("outer_metric_or_label_read_count") != 0
        or manifest.get("evaluated_epoch") != FIXED_EPOCH
        or len(manifest.get("cox_nuisance_models", ())) != 14
    ):
        raise FixedReplicationError("invalid evaluation manifest before outer read")
    for key in required_bindings:
        binding = manifest.get(key)
        if not isinstance(binding, Mapping) or sha256_file(Path(str(binding["path"]))) != binding.get("sha256"):
            raise FixedReplicationError(f"sealed evaluation artifact changed: {key}")
    expected_provenance = {
        "replication_contract_sha256": contract.sha256,
        "calibration_role_manifest_sha256": manifest.get("calibration_role_manifest_sha256"),
        "training_complete_sha256": manifest.get("training_complete_sha256"),
        "fixed_e55_checkpoint_sha256": checkpoint_sha256,
    }
    validation_sha = manifest["validation_predictions_internal"]["sha256"]
    correction_payload = json.loads(
        Path(str(manifest["age_correction"]["path"])).read_text(encoding="utf-8")
    )
    if (
        correction_payload.get("schema_version") != AGE_CORRECTION_SCHEMA
        or correction_payload.get("status") != "FITTED_AND_SEALED_BEFORE_OUTER_READ"
        or correction_payload.get("outer_fold") != int(outer_fold)
        or correction_payload.get("calibration_fold") != contract.roles[outer_fold].calibration_fold
        or correction_payload.get("validation_predictions_sha256") != validation_sha
        or any(correction_payload.get(key) != value for key, value in expected_provenance.items())
    ):
        raise FixedReplicationError("sealed age-correction provenance changed")
    correction_rows = correction_payload.get("parameters")
    if (
        not isinstance(correction_rows, list)
        or len(correction_rows) != len(ORGAN_SYSTEMS)
        or {row.get("organ_system") for row in correction_rows} != set(ORGAN_SYSTEMS)
        or any(
            not math.isfinite(float(row.get(key, math.nan)))
            for row in correction_rows
            for key in ("intercept", "slope")
        )
        or any(int(row.get("fit_population_count", 0)) <= 0 for row in correction_rows)
    ):
        raise FixedReplicationError("sealed age-correction parameters changed")
    nuisance_manifest_payload = json.loads(
        Path(str(manifest["cox_nuisance_manifest"]["path"])).read_text(encoding="utf-8")
    )
    if (
        nuisance_manifest_payload.get("schema_version") != NUISANCE_MANIFEST_SCHEMA
        or nuisance_manifest_payload.get("status") != "ALL_14_MODELS_SEALED_BEFORE_OUTER_READ"
        or nuisance_manifest_payload.get("outer_fold") != int(outer_fold)
        or nuisance_manifest_payload.get("calibration_fold")
        != contract.roles[outer_fold].calibration_fold
        or nuisance_manifest_payload.get("validation_predictions_sha256") != validation_sha
        or nuisance_manifest_payload.get("age_correction_path")
        != manifest["age_correction"]["path"]
        or nuisance_manifest_payload.get("age_correction_sha256")
        != manifest["age_correction"]["sha256"]
        or nuisance_manifest_payload.get("models") != manifest["cox_nuisance_models"]
        or any(nuisance_manifest_payload.get(key) != value for key, value in expected_provenance.items())
    ):
        raise FixedReplicationError("sealed Cox nuisance manifest provenance changed")
    combinations = {
        (row.get("organ_system"), row.get("model_role"))
        for row in manifest["cox_nuisance_models"]
    }
    if combinations != {(organ, role) for organ in ORGAN_SYSTEMS for role in ("base", "full")}:
        raise FixedReplicationError("sealed evaluation does not bind exactly 14 Cox models")
    for row in manifest["cox_nuisance_models"]:
        if sha256_file(Path(str(row["path"]))) != row.get("sha256"):
            raise FixedReplicationError("sealed Cox nuisance artifact changed")
        nuisance = json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))
        expected_covariates = (
            ["actual_age", "sex"]
            if row["model_role"] == "base"
            else ["actual_age", "sex", "corrected_age_gap"]
        )
        means = nuisance.get("feature_means", {})
        scales = nuisance.get("feature_scales", {})
        if (
            nuisance.get("outer_fold") != int(outer_fold)
            or nuisance.get("calibration_fold") != contract.roles[outer_fold].calibration_fold
            or nuisance.get("validation_predictions_sha256") != validation_sha
            or nuisance.get("covariate_order") != expected_covariates
            or nuisance.get("penalizer") != 1e-4
            or len(nuisance.get("coefficients", ())) != len(expected_covariates)
            or not np.isfinite(np.asarray(nuisance.get("coefficients", ()), dtype=np.float64)).all()
            or set(means) != set(expected_covariates)
            or set(scales) != set(expected_covariates)
            or not np.isfinite(np.asarray(list(means.values()), dtype=np.float64)).all()
            or not np.isfinite(np.asarray(list(scales.values()), dtype=np.float64)).all()
            or any(float(value) <= 0.0 for value in scales.values())
            or any(nuisance.get(key) != value for key, value in expected_provenance.items())
        ):
            raise FixedReplicationError("sealed Cox nuisance model provenance changed")
    if sha256_file(ledger_path) != expected_ledger_sha256:
        raise FixedReplicationError("outer read ledger hash changed")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if (
        ledger.get("schema_version") != OUTER_LEDGER_SCHEMA
        or ledger.get("status") != "READ_COMMITTED_BEFORE_OUTER_ACCESS"
        or ledger.get("outer_fold") != int(outer_fold)
        or ledger.get("replication_contract_sha256") != contract.sha256
        or ledger.get("role_manifest_sha256") != control_sha256
        or ledger.get("fixed_e55_checkpoint_sha256") != checkpoint_sha256
        or ledger.get("evaluation_manifest_path") != str(evaluation_manifest_path.resolve())
        or ledger.get("evaluation_manifest_sha256") != expected_evaluation_manifest_sha256
        or ledger.get("validation_predictions_sha256")
        != manifest["validation_predictions_internal"]["sha256"]
        or ledger.get("age_correction_sha256") != manifest["age_correction"]["sha256"]
        or ledger.get("cox_nuisance_manifest_sha256")
        != manifest["cox_nuisance_manifest"]["sha256"]
        or ledger.get("outer_role_manifest_sha256") != manifest["outer_role_manifest_sha256"]
        or ledger.get("outer_age_targets_sha256") != manifest["outer_age_targets_sha256"]
        or ledger.get("outer_joint_labels_sha256") != manifest["outer_joint_labels_sha256"]
    ):
        raise FixedReplicationError("outer read ledger is not bound to the sealed evaluation")
    return manifest, ledger


def score_fixed_outer(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    role_registry_path: Path,
    expected_role_registry_sha256: str,
    expected_role_validation_sha256: str,
    training_complete_path: Path,
    expected_training_complete_sha256: str,
    evaluation_manifest_path: Path,
    expected_evaluation_manifest_sha256: str,
    ledger_path: Path,
    expected_ledger_sha256: str,
    score_dir: Path,
) -> Path:
    """Outer worker. It can apply sealed nuisance parameters, but it has no fitting step."""

    contract = load_replication_contract(contract_path, expected_sha256=expected_contract_sha256)
    validate_code_lineage(contract)
    _validate_cache_only(contract)
    registry = load_role_registry(
        role_registry_path, contract, expected_sha256=expected_role_registry_sha256
    )
    training_raw = json.loads(training_complete_path.read_text(encoding="utf-8"))
    outer = int(training_raw.get("outer_fold", -1))
    control_path, control = _fold_manifest_entry(registry, outer, contract)
    control_sha = sha256_file(control_path)
    training = _validate_training_complete(
        training_complete_path,
        expected_sha256=expected_training_complete_sha256,
        contract=contract,
        outer_fold=outer,
        outer_control_manifest_sha256=control_sha,
        expected_role_validation_sha256=expected_role_validation_sha256,
    )
    checkpoint_path = Path(training["fixed_e55_checkpoint_path"]).resolve()
    checkpoint_sha = training["fixed_e55_checkpoint_sha256"]
    manifest, _ = _validate_evaluation_seal(
        contract=contract,
        outer_fold=outer,
        evaluation_manifest_path=evaluation_manifest_path,
        expected_evaluation_manifest_sha256=expected_evaluation_manifest_sha256,
        ledger_path=ledger_path,
        expected_ledger_sha256=expected_ledger_sha256,
        control_sha256=control_sha,
        checkpoint_sha256=checkpoint_sha,
        expected_role_registry_sha256=expected_role_registry_sha256,
        expected_training_complete_sha256=expected_training_complete_sha256,
        expected_calibration_role_manifest_sha256=control["roles"]["calibration"]["role_manifest"]["sha256"],
    )
    outer_entry = control["roles"]["outer_test"]
    if (
        outer_entry["role_manifest"]["sha256"] != manifest["outer_role_manifest_sha256"]
        or outer_entry["age_targets"]["sha256"] != manifest["outer_age_targets_sha256"]
        or outer_entry["joint_labels"]["sha256"] != manifest["outer_joint_labels_sha256"]
    ):
        raise FixedReplicationError("outer control artifacts changed after ledger commit")
    score_dir = score_dir.resolve()
    score_dir.mkdir(parents=True, exist_ok=False)
    stage = "LOAD_OUTER_AFTER_LEDGER"
    try:
        outer_data = _load_materialized_role(
            contract=contract,
            cache_dir=DEFAULT_CACHE_DIR,
            outer_fold=outer,
            role="outer_test",
            entry=outer_entry,
        )
        model, device = _restore_replication_model(
            contract=contract,
            data=outer_data,
            checkpoint_path=checkpoint_path,
            outer_fold=outer,
            train_authorization_sha256=training["train_authorization_sha256"],
        )
        batcher = JointVectorizedBatcher(
            outer_data,
            outer_data.screen_indices,
            batch_size=int(contract.parent.payload["training"]["batch_size"]),
            shuffle=False,
            seed=SEED,
        )
        prediction = _predict(
            model=model,
            batcher=batcher,
            device=device,
            amp=bool(contract.parent.payload["training"]["amp"]),
            max_batches=None,
        )
        ids = _prediction_ids(outer_data, prediction)
        internal = _authoritative_prediction_frame(
            predicted_age=prediction.predicted_age,
            participant_ids=ids,
            data=outer_data,
            role_entry=outer_entry,
            outer_fold=outer,
            split_role="test",
        )
        internal_path = score_dir / "test_predictions_internal.parquet"
        _write_parquet_exclusive(internal_path, internal)
        test_standard_path = score_dir / "test_predictions_long.parquet"
        _write_parquet_exclusive(test_standard_path, internal.loc[:, PREDICTION_COLUMNS])
        stage = "SCORE_OUTER_E55"
        raw_strict = _raw_strict_frame_metrics(internal)
        correction_binding = manifest["age_correction"]
        nuisance_binding = manifest["cox_nuisance_manifest"]
        risk, corrected = score_outer_with_sealed_nuisance(
            internal,
            outer_fold=outer,
            correction_path=Path(correction_binding["path"]),
            correction_sha256=correction_binding["sha256"],
            nuisance_manifest_path=Path(nuisance_binding["path"]),
            nuisance_manifest_sha256=nuisance_binding["sha256"],
        )
        corrected_path = score_dir / "test_predictions_corrected.parquet"
        _write_parquet_exclusive(corrected_path, corrected)
        validation_standard = pd.read_parquet(manifest["validation_predictions_long"]["path"])
        combined = pd.concat([validation_standard, internal.loc[:, PREDICTION_COLUMNS]], ignore_index=True)
        combined_path = score_dir / f"outer_fold_{outer}_validation_test_predictions_long.parquet"
        _write_parquet_exclusive(combined_path, combined)
        metrics_path = score_dir / "fold_metrics.json"
        _write_json_exclusive(
            metrics_path,
            {
                "outer_fold": outer,
                "fixed_epoch": FIXED_EPOCH,
                "raw_strict_age": raw_strict,
                "sealed_nuisance_harrell": risk,
                "outer_performance_used_for_selection_or_tuning": False,
            },
        )
        stage = "FINALIZE_RECEIPT"
        artifact_paths = {
            "test_predictions_internal": internal_path,
            "test_predictions_long": test_standard_path,
            "test_predictions_corrected": corrected_path,
            "validation_test_predictions_long": combined_path,
            "fold_metrics": metrics_path,
            "evaluation_manifest": evaluation_manifest_path,
            "outer_read_ledger": ledger_path,
        }
        registry_path = score_dir / "artifact_hashes.json"
        write_artifact_hash_registry(registry_path, artifact_paths)
        receipt_path = score_dir / "FOLD_COMPLETE_RECEIPT.json"
        _write_json_exclusive(
            receipt_path,
            {
                "schema_version": RECEIPT_SCHEMA,
                "status": "COMPLETE_FIXED_E55_OUTER_SCORE",
                "outer_fold": outer,
                "replication_contract_sha256": contract.sha256,
                "role_registry_sha256": expected_role_registry_sha256,
                "outer_control_manifest_sha256": control_sha,
                "training_complete_sha256": expected_training_complete_sha256,
                "fixed_e55_checkpoint_sha256": checkpoint_sha,
                "evaluation_manifest_sha256": expected_evaluation_manifest_sha256,
                "outer_read_ledger_sha256": expected_ledger_sha256,
                "age_correction_sha256": correction_binding["sha256"],
                "cox_nuisance_manifest_sha256": nuisance_binding["sha256"],
                "artifact_registry_path": str(registry_path),
                "artifact_registry_sha256": sha256_file(registry_path),
                "performance_changed_method": False,
            },
        )
        return receipt_path
    except BaseException as exc:
        failure = score_dir / "OUTER_STAGE_FAILURE.json"
        if not failure.exists():
            _write_json_exclusive(
                failure,
                {"schema_version": FAILURE_SCHEMA, "status": "FAILED_AFTER_LEDGER", "stage": stage, "exception": {"type": type(exc).__name__, "message": str(exc)}},
            )
        raise


def run_fold0_reuse(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    role_registry_path: Path,
    expected_role_registry_sha256: str,
    expected_role_validation_sha256: str,
    output_dir: Path,
) -> Path:
    """Reuse D07-003 E55; never regenerate Fold0 model predictions."""

    contract = load_replication_contract(contract_path, expected_sha256=expected_contract_sha256)
    validate_code_lineage(contract)
    _validate_cache_only(contract)
    registry = load_role_registry(
        role_registry_path, contract, expected_sha256=expected_role_registry_sha256
    )
    validate_published_role_artifacts(
        contract=contract,
        role_root=role_registry_path.resolve().parent,
        expected_registry_sha256=expected_role_registry_sha256,
        expected_validation_sha256=expected_role_validation_sha256,
    )
    control_path, control = _fold_manifest_entry(registry, 0, contract)
    control_sha = sha256_file(control_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    reuse = contract.payload["fold0_reuse"]
    evidence = contract.payload["d07_003_evidence"]
    checkpoint_path = _relative_or_absolute(WORKSPACE_ROOT, reuse["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != reuse["checkpoint"]["file_sha256"]:
        raise FixedReplicationError("historical D07-003 E55 checkpoint changed")
    for key in (
        "run_config",
        "training_history",
        "checkpoint_set",
        "checkpoint_decision",
        "model_preflight",
        "strict_age_history",
        "risk_history",
        "metrics",
        "summary",
        "artifact_registry",
        "read_ledger",
        "completion_receipt",
    ):
        binding = evidence[key]
        artifact = _relative_or_absolute(WORKSPACE_ROOT, binding["path"])
        if sha256_file(artifact) != binding["sha256"]:
            raise FixedReplicationError(f"historical D07-003 evidence changed: {key}")
    reuse_completion_path = output_dir / "REUSE_TRAINING_COMPLETE.json"
    _write_json_exclusive(
        reuse_completion_path,
        {
            "schema_version": REUSE_TRAINING_COMPLETE_SCHEMA,
            "status": "D07_003_E60_COMPLETE_E55_FIXED_REUSE",
            "outer_fold": 0,
            "replication_contract_sha256": contract.sha256,
            "parent_d07_003_sha256": contract.parent.sha256,
            "outer_control_manifest_sha256": control_sha,
            "role_artifact_validation_sha256": expected_role_validation_sha256,
            "source_run_config_sha256": evidence["run_config"]["sha256"],
            "source_training_history_sha256": evidence["training_history"]["sha256"],
            "source_checkpoint_set_sha256": evidence["checkpoint_set"]["sha256"],
            "fixed_e55_checkpoint_path": str(checkpoint_path),
            "fixed_e55_checkpoint_sha256": reuse["checkpoint"]["file_sha256"],
            "completed_epochs_in_source_run": 60,
            "fixed_checkpoint_epoch": 55,
            "retrained": False,
            "historical_test_predictions_opened": False,
        },
    )
    completion_sha = sha256_file(reuse_completion_path)
    calibration_entry = control["roles"]["calibration"]
    calibration_authorization_path = output_dir / "CALIBRATION_READ_AUTHORIZATION.json"
    _write_json_exclusive(
        calibration_authorization_path,
        {
            "status": "AUTHORIZED_AFTER_D07_003_E60_AND_E55_SEAL",
            "outer_fold": 0,
            "replication_contract_sha256": contract.sha256,
            "outer_control_manifest_sha256": control_sha,
            "training_complete_sha256": completion_sha,
            "fixed_e55_checkpoint_sha256": reuse["checkpoint"]["file_sha256"],
            "calibration_role_manifest_sha256": calibration_entry["role_manifest"]["sha256"],
            "calibration_age_targets_sha256": calibration_entry["age_targets"]["sha256"],
            "calibration_joint_labels_sha256": calibration_entry["joint_labels"]["sha256"],
        },
    )
    calibration = _load_materialized_role(
        contract=contract,
        cache_dir=DEFAULT_CACHE_DIR,
        outer_fold=0,
        role="calibration",
        entry=calibration_entry,
    )
    if not torch.cuda.is_available():
        raise FixedReplicationError("Fold0 full-calibration inference requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configure_reproducibility(SEED, deterministic=True, tf32=True)
    model = EHRAGEModel(
        calibration.cache.code_embedding,
        calibration.cache.text_embedding,
        age_anchor=float(checkpoint["age_anchor_train_folds_2_to_4_only"]),
        config=_model_config(contract.parent.payload["model"]),
    )
    model.to(torch.device("cuda"))
    restore_trainable_checkpoint(
        path=checkpoint_path,
        model=model,
        candidate_id=contract.parent.candidate_id,
        contract_sha256=contract.parent.sha256,
        materialized_config_hash=str(checkpoint["materialized_config_hash"]),
    )
    batcher = JointVectorizedBatcher(
        calibration,
        calibration.calibration_indices,
        batch_size=int(contract.parent.payload["training"]["batch_size"]),
        shuffle=False,
        seed=SEED,
    )
    prediction = _predict(
        model=model,
        batcher=batcher,
        device=torch.device("cuda"),
        amp=True,
        max_batches=None,
    )
    ids = _prediction_ids(calibration, prediction)
    validation = _authoritative_prediction_frame(
        predicted_age=prediction.predicted_age,
        participant_ids=ids,
        data=calibration,
        role_entry=calibration_entry,
        outer_fold=0,
        split_role="validation",
    )
    validation_internal_path = output_dir / "validation_predictions_internal.parquet"
    validation_standard_path = output_dir / "validation_predictions_long.parquet"
    _write_parquet_exclusive(validation_internal_path, validation)
    _write_parquet_exclusive(validation_standard_path, validation.loc[:, PREDICTION_COLUMNS])
    nuisance_dir = output_dir / "sealed_nuisance"
    correction_path, nuisance_manifest_path, corrected_validation = fit_and_seal_calibration_nuisance(
        validation,
        outer_fold=0,
        calibration_fold=1,
        validation_predictions_path=validation_internal_path,
        output_dir=nuisance_dir,
        replication_contract_sha256=contract.sha256,
        calibration_role_manifest_sha256=calibration_entry["role_manifest"]["sha256"],
        training_complete_sha256=completion_sha,
        fixed_e55_checkpoint_sha256=reuse["checkpoint"]["file_sha256"],
    )
    corrected_validation_path = output_dir / "validation_predictions_corrected.parquet"
    _write_parquet_exclusive(corrected_validation_path, corrected_validation)
    nuisance_manifest = json.loads(nuisance_manifest_path.read_text(encoding="utf-8"))
    outer_entry = control["roles"]["outer_test"]
    historical_prediction_path = _relative_or_absolute(
        WORKSPACE_ROOT, reuse["existing_test_predictions"]["path"]
    )
    evaluation_manifest_path = output_dir / "EVALUATION_MANIFEST_BEFORE_OUTER.json"
    _write_json_exclusive(
        evaluation_manifest_path,
        {
            "schema_version": EVALUATION_MANIFEST_SCHEMA,
            "status": "SEALED_BEFORE_OUTER_READ",
            "outer_fold": 0,
            "replication_contract_sha256": contract.sha256,
            "role_registry_sha256": expected_role_registry_sha256,
            "outer_control_manifest_path": str(control_path),
            "outer_control_manifest_sha256": control_sha,
            "training_complete_path": str(reuse_completion_path),
            "training_complete_sha256": completion_sha,
            "fixed_e55_checkpoint_path": str(checkpoint_path),
            "fixed_e55_checkpoint_sha256": reuse["checkpoint"]["file_sha256"],
            "calibration_role_manifest_sha256": calibration_entry["role_manifest"]["sha256"],
            "validation_predictions_internal": {"path": str(validation_internal_path), "sha256": sha256_file(validation_internal_path)},
            "validation_predictions_long": {"path": str(validation_standard_path), "sha256": sha256_file(validation_standard_path)},
            "validation_predictions_corrected": {"path": str(corrected_validation_path), "sha256": sha256_file(corrected_validation_path)},
            "age_correction": {"path": str(correction_path), "sha256": sha256_file(correction_path)},
            "cox_nuisance_manifest": {"path": str(nuisance_manifest_path), "sha256": sha256_file(nuisance_manifest_path)},
            "cox_nuisance_models": nuisance_manifest["models"],
            "outer_role_manifest_sha256": outer_entry["role_manifest"]["sha256"],
            "outer_age_targets_sha256": outer_entry["age_targets"]["sha256"],
            "outer_joint_labels_sha256": outer_entry["joint_labels"]["sha256"],
            "declared_historical_test_predictions_path": str(historical_prediction_path),
            "declared_historical_test_predictions_sha256": reuse["existing_test_predictions"]["sha256"],
            "outer_metric_or_label_read_count": 0,
            "evaluated_epoch": FIXED_EPOCH,
        },
    )
    ledger_path = output_dir / "OUTER_READ_LEDGER.json"
    _write_json_exclusive(
        ledger_path,
        {
            "schema_version": OUTER_LEDGER_SCHEMA,
            "status": "READ_COMMITTED_BEFORE_OUTER_ACCESS",
            "outer_fold": 0,
            "replication_contract_sha256": contract.sha256,
            "role_manifest_sha256": control_sha,
            "fixed_e55_checkpoint_sha256": reuse["checkpoint"]["file_sha256"],
            "evaluation_manifest_path": str(evaluation_manifest_path),
            "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
            "validation_predictions_sha256": sha256_file(validation_internal_path),
            "age_correction_sha256": sha256_file(correction_path),
            "cox_nuisance_manifest_sha256": sha256_file(nuisance_manifest_path),
            "outer_role_manifest_sha256": outer_entry["role_manifest"]["sha256"],
            "outer_age_targets_sha256": outer_entry["age_targets"]["sha256"],
            "outer_joint_labels_sha256": outer_entry["joint_labels"]["sha256"],
            "historical_test_predictions_sha256": reuse["existing_test_predictions"]["sha256"],
        },
    )
    _validate_evaluation_seal(
        contract=contract,
        outer_fold=0,
        evaluation_manifest_path=evaluation_manifest_path,
        expected_evaluation_manifest_sha256=sha256_file(evaluation_manifest_path),
        ledger_path=ledger_path,
        expected_ledger_sha256=sha256_file(ledger_path),
        control_sha256=control_sha,
        checkpoint_sha256=reuse["checkpoint"]["file_sha256"],
        expected_role_registry_sha256=expected_role_registry_sha256,
        expected_training_complete_sha256=completion_sha,
        expected_calibration_role_manifest_sha256=calibration_entry["role_manifest"]["sha256"],
    )
    # First access to historical Fold0 prediction/outcome values occurs after the ledger.
    if sha256_file(historical_prediction_path) != reuse["existing_test_predictions"]["sha256"]:
        raise FixedReplicationError("historical Fold0 predictions changed after ledger commit")
    historical = pd.read_parquet(historical_prediction_path)
    required = {
        "participant_id",
        "organ_system",
        "actual_age",
        "predicted_age",
        "strict_healthy_reference_60m",
        "sex",
        "duration",
        "event",
        "at_risk",
    }
    if not required.issubset(historical.columns) or historical.duplicated(["participant_id", "organ_system"]).any():
        raise FixedReplicationError("historical Fold0 prediction schema/identity changed")
    if set(historical["organ_system"].astype(str).unique()) != set(ORGAN_SYSTEMS):
        raise FixedReplicationError("historical Fold0 prediction organ axis changed")
    outer_data = _load_materialized_role(
        contract=contract,
        cache_dir=DEFAULT_CACHE_DIR,
        outer_fold=0,
        role="outer_test",
        entry=outer_entry,
    )
    outer_ids = [str(value) for value in outer_data.cache.dataset.participant_ids]
    if (
        set(historical["participant_id"].astype(str)) != set(outer_ids)
        or len(historical) != len(outer_ids) * len(ORGAN_SYSTEMS)
    ):
        raise FixedReplicationError("historical Fold0 participant-organ coverage changed")
    test = historical.loc[:, ["participant_id", "organ_system", "predicted_age"]].copy()
    test["participant_id"] = test["participant_id"].astype(str)
    ages = pd.read_parquet(Path(outer_entry["age_targets"]["path"]))
    ages["participant_id"] = ages["participant_id"].astype(str)
    labels = pd.read_parquet(Path(outer_entry["joint_labels"]["path"]))
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels = labels.rename(columns={"duration_months": "duration"})
    test = test.merge(
        ages.loc[:, ["participant_id", "chronological_age"]].rename(
            columns={"chronological_age": "actual_age"}
        ),
        on="participant_id",
        how="left",
        sort=False,
        validate="many_to_one",
    ).merge(
        labels.loc[:, ["participant_id", "organ_system", "strict_healthy_reference_60m", "duration", "event", "at_risk"]],
        on=["participant_id", "organ_system"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    sex_map = {
        str(pid): float(value)
        for pid, value in zip(
            outer_data.cache.dataset.participant_ids,
            outer_data.cache.dataset.inputs["sex"].numpy().tolist(),
        )
    }
    test["sex"] = test["participant_id"].map(sex_map)
    test["outer_fold"] = 0
    test["split_role"] = "test"
    test["duration"] = pd.to_numeric(test["duration"], errors="coerce").fillna(0.0)
    if not test.loc[:, ["participant_id", "organ_system"]].reset_index(drop=True).equals(
        historical.loc[:, ["participant_id", "organ_system"]]
        .assign(participant_id=lambda frame: frame["participant_id"].astype(str))
        .reset_index(drop=True)
    ):
        raise FixedReplicationError("Fold0 canonical repackaging changed historical row order")
    if not np.isfinite(test["predicted_age"].to_numpy()).all():
        raise FixedReplicationError("historical Fold0 raw predictions contain non-finite values")
    historical_age = historical.assign(
        participant_id=historical["participant_id"].astype(str)
    ).set_index(["participant_id", "organ_system"])["actual_age"]
    authoritative_index = test.set_index(["participant_id", "organ_system"])
    if not np.allclose(
        historical_age.reindex(authoritative_index.index).to_numpy(dtype=np.float64),
        authoritative_index["actual_age"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-5,
    ):
        raise FixedReplicationError("historical Fold0 actual_age changed from role artifact")
    test_internal_path = output_dir / "test_predictions_internal.normalized.parquet"
    test_standard_path = output_dir / "test_predictions_long.normalized.parquet"
    _write_parquet_exclusive(test_internal_path, test)
    _write_parquet_exclusive(test_standard_path, test.loc[:, PREDICTION_COLUMNS])
    risk, corrected_test = score_outer_with_sealed_nuisance(
        test,
        outer_fold=0,
        correction_path=correction_path,
        correction_sha256=sha256_file(correction_path),
        nuisance_manifest_path=nuisance_manifest_path,
        nuisance_manifest_sha256=sha256_file(nuisance_manifest_path),
    )
    corrected_test_path = output_dir / "test_predictions_corrected.parquet"
    _write_parquet_exclusive(corrected_test_path, corrected_test)
    raw_strict = _raw_strict_frame_metrics(test)
    metrics_path = output_dir / "fold_metrics.json"
    _write_json_exclusive(
        metrics_path,
        {"outer_fold": 0, "fixed_epoch": 55, "raw_strict_age": raw_strict, "sealed_nuisance_harrell": risk},
    )
    combined_path = output_dir / "outer_fold_0_validation_test_predictions_long.parquet"
    _write_parquet_exclusive(
        combined_path,
        pd.concat([validation.loc[:, PREDICTION_COLUMNS], test.loc[:, PREDICTION_COLUMNS]], ignore_index=True),
    )
    artifacts = {
        "reuse_training_complete": reuse_completion_path,
        "validation_predictions_long": validation_standard_path,
        "test_predictions_long": test_standard_path,
        "test_predictions_corrected": corrected_test_path,
        "validation_test_predictions_long": combined_path,
        "fold_metrics": metrics_path,
        "evaluation_manifest": evaluation_manifest_path,
        "outer_read_ledger": ledger_path,
    }
    artifact_registry_path = output_dir / "artifact_hashes.json"
    write_artifact_hash_registry(artifact_registry_path, artifacts)
    receipt_path = output_dir / "FOLD_COMPLETE_RECEIPT.json"
    _write_json_exclusive(
        receipt_path,
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "COMPLETE_FIXED_E55_OUTER_SCORE",
            "outer_fold": 0,
            "fold_role": "OPEN_DEVELOPMENT",
            "replication_contract_sha256": contract.sha256,
            "role_registry_sha256": expected_role_registry_sha256,
            "outer_control_manifest_sha256": control_sha,
            "training_complete_sha256": completion_sha,
            "fixed_e55_checkpoint_sha256": reuse["checkpoint"]["file_sha256"],
            "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
            "outer_read_ledger_sha256": sha256_file(ledger_path),
            "age_correction_sha256": sha256_file(correction_path),
            "cox_nuisance_manifest_sha256": sha256_file(nuisance_manifest_path),
            "artifact_registry_path": str(artifact_registry_path),
            "artifact_registry_sha256": sha256_file(artifact_registry_path),
            "historical_prediction_source_sha256": reuse["existing_test_predictions"]["sha256"],
            "normalized_test_prediction_sha256": sha256_file(test_standard_path),
            "retrained": False,
        },
    )
    return receipt_path


def _load_complete_receipt(path: Path, contract: ReplicationContract, outer: int) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "COMPLETE_FIXED_E55_OUTER_SCORE"
        or receipt.get("outer_fold") != outer
        or receipt.get("replication_contract_sha256") != contract.sha256
    ):
        raise FixedReplicationError(f"invalid completed fold receipt: {outer}")
    registry_path = Path(str(receipt["artifact_registry_path"])).resolve()
    if sha256_file(registry_path) != receipt["artifact_registry_sha256"]:
        raise FixedReplicationError(f"fold {outer} artifact registry changed")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for name, binding in registry.items():
        if sha256_file(Path(str(binding["path"]))) != binding["sha256"]:
            raise FixedReplicationError(f"fold {outer} artifact changed: {name}")
    return receipt, registry


def write_receipt_selection(
    *,
    path: Path,
    contract: ReplicationContract,
    role_registry_sha256: str,
    receipt_paths: Mapping[int, Path],
) -> Path:
    if set(receipt_paths) != set(ALL_FOLDS):
        raise FixedReplicationError("receipt selection requires folds 0..4")
    rows = []
    for outer in ALL_FOLDS:
        receipt_path = receipt_paths[outer].resolve()
        receipt, _ = _load_complete_receipt(receipt_path, contract, outer)
        if receipt.get("role_registry_sha256") != role_registry_sha256:
            raise FixedReplicationError("receipt selection mixes role registries")
        rows.append(
            {
                "outer_fold": outer,
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "selection_basis": "explicit_infrastructure_attempt_only_not_performance",
            }
        )
    _write_json_exclusive(
        path,
        {
            "schema_version": RECEIPT_SELECTION_SCHEMA,
            "status": "FROZEN_COMPLETE_FIVE_FOLD_SELECTION",
            "replication_contract_sha256": contract.sha256,
            "role_registry_sha256": role_registry_sha256,
            "performance_values_read_for_selection": False,
            "receipts": rows,
        },
    )
    return path


def load_receipt_selection(
    *,
    path: Path,
    expected_sha256: str,
    contract: ReplicationContract,
    role_registry_sha256: str,
) -> dict[int, Path]:
    if sha256_file(path) != expected_sha256:
        raise FixedReplicationError("receipt-selection manifest hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("receipts")
    if (
        payload.get("schema_version") != RECEIPT_SELECTION_SCHEMA
        or payload.get("status") != "FROZEN_COMPLETE_FIVE_FOLD_SELECTION"
        or payload.get("replication_contract_sha256") != contract.sha256
        or payload.get("role_registry_sha256") != role_registry_sha256
        or payload.get("performance_values_read_for_selection") is not False
        or not isinstance(rows, list)
        or {int(row.get("outer_fold", -1)) for row in rows} != set(ALL_FOLDS)
        or len(rows) != 5
    ):
        raise FixedReplicationError("invalid receipt-selection manifest")
    result: dict[int, Path] = {}
    for row in rows:
        outer = int(row["outer_fold"])
        receipt_path = Path(str(row["path"])).resolve()
        if sha256_file(receipt_path) != row["sha256"]:
            raise FixedReplicationError(f"selected receipt changed: outer {outer}")
        _load_complete_receipt(receipt_path, contract, outer)
        result[outer] = receipt_path
    return result


def _aggregate_fivefold_impl(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    role_registry_path: Path,
    expected_role_registry_sha256: str,
    expected_role_validation_sha256: str,
    receipt_paths: Mapping[int, Path],
    bundle_root: Path,
    receipt_selection_path: Path,
    expected_receipt_selection_sha256: str,
) -> dict[str, Path]:
    if set(receipt_paths) != set(ALL_FOLDS):
        raise FixedReplicationError("five-fold aggregation requires receipts for folds 0..4")
    contract = load_replication_contract(contract_path, expected_sha256=expected_contract_sha256)
    validate_code_lineage(contract)
    validate_data_lineage(contract)
    validate_published_role_artifacts(
        contract=contract,
        role_root=role_registry_path.resolve().parent,
        expected_registry_sha256=expected_role_registry_sha256,
        expected_validation_sha256=expected_role_validation_sha256,
    )
    load_receipt_selection(
        path=receipt_selection_path,
        expected_sha256=expected_receipt_selection_sha256,
        contract=contract,
        role_registry_sha256=expected_role_registry_sha256,
    )
    role_registry = load_role_registry(
        role_registry_path, contract, expected_sha256=expected_role_registry_sha256
    )
    prediction_root = bundle_root / "00_prediction_registry"
    evaluation_root = bundle_root / "01_evaluation"
    hrkm_root = bundle_root / "02_oof_hr_km_inputs"
    receipts_root = bundle_root / "03_receipts"
    for root in (prediction_root, evaluation_root, hrkm_root, receipts_root):
        root.mkdir(parents=True, exist_ok=False)
    prediction_entries: list[dict[str, str]] = []
    corrected_parts: list[pd.DataFrame] = []
    receipt_manifest_rows: list[dict[str, Any]] = []
    staged_metrics: dict[int, dict[str, Any]] = {}
    for outer in ALL_FOLDS:
        receipt_path = receipt_paths[outer].resolve()
        receipt, artifacts = _load_complete_receipt(receipt_path, contract, outer)
        evaluation_manifest = json.loads(
            Path(artifacts["evaluation_manifest"]["path"]).read_text(encoding="utf-8")
        )
        validation_path = Path(evaluation_manifest["validation_predictions_long"]["path"])
        test_path = Path(artifacts["test_predictions_long"]["path"])
        for split_role, path in (("validation", validation_path), ("test", test_path)):
            frame = pd.read_parquet(path, columns=list(PREDICTION_COLUMNS))
            if (
                set(frame["outer_fold"].astype(int).unique()) != {outer}
                or set(frame["split_role"].astype(str).unique()) != {split_role}
            ):
                raise FixedReplicationError(f"fold {outer} {split_role} prediction role changed")
            prediction_entries.append(
                {
                    "outer_fold": outer,
                    "split_role": split_role,
                    "rows": int(len(frame)),
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
            )
        corrected_path = Path(artifacts["test_predictions_corrected"]["path"])
        corrected = pd.read_parquet(corrected_path)
        training_complete = json.loads(
            Path(evaluation_manifest["training_complete_path"]).read_text(encoding="utf-8")
        )
        run_config_sha = (
            contract.payload["d07_003_evidence"]["run_config"]["sha256"]
            if outer == 0
            else training_complete["run_config_sha256"]
        )
        corrected["fold_role"] = "OPEN_DEVELOPMENT" if outer == 0 else "FIXED_REPLICATION"
        corrected["split_role"] = "test"
        corrected["contract_sha256"] = contract.sha256
        corrected["run_config_sha256"] = run_config_sha
        corrected["e55_checkpoint_sha256"] = receipt["fixed_e55_checkpoint_sha256"]
        corrected["outer_control_manifest_sha256"] = receipt["outer_control_manifest_sha256"]
        corrected["outer_test_role_manifest_sha256"] = evaluation_manifest[
            "outer_role_manifest_sha256"
        ]
        corrected["age_correction_sha256"] = receipt["age_correction_sha256"]
        corrected["cox_nuisance_manifest_sha256"] = receipt["cox_nuisance_manifest_sha256"]
        corrected["fold_receipt_sha256"] = sha256_file(receipt_path)
        corrected_parts.append(corrected)
        staged_metrics[outer] = json.loads(Path(artifacts["fold_metrics"]["path"]).read_text(encoding="utf-8"))
        receipt_manifest_rows.append(
            {"outer_fold": outer, "path": str(receipt_path), "sha256": sha256_file(receipt_path)}
        )
    if len(prediction_entries) != 10:
        raise FixedReplicationError("prediction registry must contain exactly ten files")
    prediction_registry_path = prediction_root / "ehr_age_d07_003_e55.json"
    _write_json_exclusive(
        prediction_registry_path,
        {
            "schema_version": "revision0710_prediction_registry_v1",
            "method": "ehr_age_d07_003_e55",
            "prediction_files": prediction_entries,
        },
    )
    evaluator_path = REVISION_ROOT / "05_downstream" / "scripts" / "evaluate_method_5fold.py"
    spec = importlib.util.spec_from_file_location("revision0710_evaluate_method_5fold", evaluator_path)
    if spec is None or spec.loader is None:
        raise FixedReplicationError("cannot load canonical five-fold evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = evaluator
    spec.loader.exec_module(evaluator)
    prediction_paths, _ = evaluator.resolve_prediction_files(
        method="ehr_age_d07_003_e55",
        prediction_registry=prediction_registry_path,
        predictions=None,
    )
    predictions = evaluator.load_predictions(prediction_paths)
    participants = evaluator.load_participant_masks(DEFAULT_PARTICIPANT_MASKS)
    outcomes = evaluator.load_organ_outcomes(DEFAULT_OUTCOMES)
    roles = evaluator.load_fold_roles(DEFAULT_FOLD_ROLES)
    result = evaluator.evaluate_method(
        method="ehr_age_d07_003_e55",
        predictions=predictions,
        participants=participants,
        outcomes=outcomes,
        roles=roles,
    )
    test_predictions = predictions.loc[predictions["split_role"].eq("test")].merge(
        participants.loc[:, ["participant_id", "strict_healthy_reference_60m"]],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    staged_strict = evaluator.raw_age_metrics(
        test_predictions.loc[test_predictions["strict_healthy_reference_60m"].eq(1)]
    )
    staged_organ_summary: dict[str, Any] = {}
    for organ in ORGAN_SYSTEMS:
        cells = [
            staged_metrics[outer]["sealed_nuisance_harrell"]["organs"][organ]
            for outer in ALL_FOLDS
        ]
        staged_organ_summary[organ] = {
            "mean_c_base": float(np.mean([row["c_base"] for row in cells])),
            "mean_c_plus": float(np.mean([row["c_plus"] for row in cells])),
            "mean_delta_c": float(np.mean([row["delta_c"] for row in cells])),
            "positive_folds": int(sum(row["delta_c"] > 0 for row in cells)),
        }
    staged_summary_path = evaluation_root / "STAGED_PRIMARY_SUMMARY.json"
    _write_json_exclusive(
        staged_summary_path,
        {
            "schema_version": "revision0710.ehr_age.fixed_replication.staged_primary.v1",
            "method": "ehr_age_d07_003_e55",
            "strict_age": {
                "n_rows": staged_strict.n_rows,
                "n_participants": staged_strict.n_participants,
                "mae": staged_strict.mae,
                "pearson": staged_strict.pearson,
                "r2": staged_strict.r2,
            },
            "cox_by_organ": staged_organ_summary,
            "macro_delta_c": float(
                np.mean([row["mean_delta_c"] for row in staged_organ_summary.values()])
            ),
            "canonical_post_receipt_evaluator_is_primary": False,
        },
    )
    canonical_audit_root = evaluation_root / "parity_audit"
    evaluator.write_outputs(
        result,
        method="ehr_age_d07_003_e55",
        output_dir=canonical_audit_root,
        prediction_files=prediction_paths,
        prediction_registry=prediction_registry_path,
        participant_masks=DEFAULT_PARTICIPANT_MASKS,
        organ_outcomes=DEFAULT_OUTCOMES,
        fold_roles=DEFAULT_FOLD_ROLES,
        overwrite=False,
    )
    canonical_cells = result["cox_cells"]
    parity_rows: list[dict[str, Any]] = []
    for outer in ALL_FOLDS:
        for organ in ORGAN_SYSTEMS:
            canonical = canonical_cells.loc[
                canonical_cells["fold"].eq(outer)
                & canonical_cells["organ_system"].astype(str).eq(organ)
            ].iloc[0]
            staged = staged_metrics[outer]["sealed_nuisance_harrell"]["organs"][organ]
            differences = {
                "c_base": abs(float(canonical["c_base"]) - float(staged["c_base"])),
                "c_plus": abs(float(canonical["c_plus"]) - float(staged["c_plus"])),
                "delta_c": abs(float(canonical["delta_c"]) - float(staged["delta_c"])),
            }
            parity_rows.append(
                {"outer_fold": outer, "organ_system": organ, **differences, "pass": max(differences.values()) <= 1e-12}
            )
    if not all(row["pass"] for row in parity_rows):
        raise FixedReplicationError("post-receipt staged/canonical Cox parity failed")
    canonical_strict = result["age_metrics"].loc[
        result["age_metrics"]["cohort"].eq("strict_healthy_reference_60m")
    ].iloc[0]
    strict_differences = {
        key: abs(float(canonical_strict[key]) - float(getattr(staged_strict, key)))
        for key in ("mae", "pearson", "r2")
    }
    if max(strict_differences.values()) > 1e-12:
        raise FixedReplicationError("post-receipt staged/canonical strict-age parity failed")
    parity_path = canonical_audit_root / "staged_canonical_parity.json"
    _write_json_exclusive(
        parity_path,
        {
            "status": "PASS",
            "absolute_tolerance": 1e-12,
            "strict_age_absolute_differences": strict_differences,
            "cells": parity_rows,
        },
    )
    oof = pd.concat(corrected_parts, ignore_index=True)
    if len(oof) != len(participants) * len(ORGAN_SYSTEMS) or oof.duplicated(
        ["participant_id", "organ_system"]
    ).any():
        raise FixedReplicationError("five-fold OOF corrected predictions lack exact participant-organ coverage")
    if oof.groupby("participant_id")["outer_fold"].nunique().ne(1).any():
        raise FixedReplicationError("a participant appears in multiple OOF outer folds")
    oof = oof.rename(
        columns={"predicted_age": "predicted_age_raw", "duration": "duration_months"}
    )
    hr_columns = contract.payload["outputs"]["hr_km_row_schema_exact"]
    missing = set(hr_columns).difference(oof.columns)
    if missing:
        raise FixedReplicationError(f"HR/KM OOF handoff lacks columns: {sorted(missing)}")
    oof_path = hrkm_root / "ehr_age_d07_003_e55_oof.parquet"
    _write_parquet_exclusive(oof_path, oof.loc[:, hr_columns])
    dictionary_path = hrkm_root / "DATA_DICTIONARY.json"
    _write_json_exclusive(
        dictionary_path,
        {
            "schema_version": "revision0710.ehr_age.fixed_replication.hrkm_dictionary.v1",
            "columns": hr_columns,
            "row_grain": "one held-out participant-organ row",
            "age_gap": "raw and validation-corrected E55 age gap",
            "risk_population_flag": "at_risk",
        },
    )
    hr_manifest_path = hrkm_root / "MANIFEST.json"
    _write_json_exclusive(
        hr_manifest_path,
        {
            "status": "COMPLETE_5FOLD_OOF",
            "contract_sha256": contract.sha256,
            "prediction_registry_sha256": sha256_file(prediction_registry_path),
            "receipt_selection_sha256": expected_receipt_selection_sha256,
            "oof_path": "02_oof_hr_km_inputs/ehr_age_d07_003_e55_oof.parquet",
            "oof_sha256": sha256_file(oof_path),
            "row_count": int(len(oof)),
            "participant_count": int(oof["participant_id"].nunique()),
            "folds": list(ALL_FOLDS),
            "partial_handoff": False,
        },
    )
    receipt_manifest_path = receipts_root / "FIVE_FOLD_RECEIPTS.json"
    _write_json_exclusive(
        receipt_manifest_path,
        {"contract_sha256": contract.sha256, "receipts": receipt_manifest_rows},
    )
    return {
        "prediction_registry": prediction_registry_path,
        "evaluation_summary": staged_summary_path,
        "parity_audit": parity_path,
        "hrkm_oof": oof_path,
        "hrkm_manifest": hr_manifest_path,
        "receipt_manifest": receipt_manifest_path,
    }


def aggregate_fivefold(
    *,
    contract_path: Path,
    expected_contract_sha256: str,
    role_registry_path: Path,
    expected_role_registry_sha256: str,
    expected_role_validation_sha256: str | None = None,
    receipt_paths: Mapping[int, Path],
    receipt_selection_path: Path | None = None,
    expected_receipt_selection_sha256: str | None = None,
) -> dict[str, Path]:
    if set(receipt_paths) != set(ALL_FOLDS):
        raise FixedReplicationError("five-fold aggregation requires receipts for folds 0..4")
    if receipt_selection_path is None or expected_receipt_selection_sha256 is None:
        raise FixedReplicationError("five-fold aggregation requires an explicit receipt selection")
    if expected_role_validation_sha256 is None:
        raise FixedReplicationError("five-fold aggregation requires the role-validation SHA256")
    final_bundle = DEFAULT_REPLICATION_ROOT / "02_published_fivefold"
    if final_bundle.exists():
        raise FileExistsError(f"aggregate canonical bundle already exists: {final_bundle}")
    staging = DEFAULT_REPLICATION_ROOT / f".02_published_fivefold.building.{os.getpid()}.{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        staged = _aggregate_fivefold_impl(
            contract_path=contract_path,
            expected_contract_sha256=expected_contract_sha256,
            role_registry_path=role_registry_path,
            expected_role_registry_sha256=expected_role_registry_sha256,
            expected_role_validation_sha256=expected_role_validation_sha256,
            receipt_paths=receipt_paths,
            bundle_root=staging,
            receipt_selection_path=receipt_selection_path,
            expected_receipt_selection_sha256=expected_receipt_selection_sha256,
        )
        for path in staged.values():
            if not path.is_file():
                raise FixedReplicationError(f"aggregate staged artifact missing: {path}")
        completion_path = staging / "03_receipts" / "FIVE_FOLD_AGGREGATION_COMPLETE.json"
        _write_json_exclusive(
            completion_path,
            {
                "status": "COMPLETE_ATOMIC_PUBLICATION_GATE",
                "replication_contract_sha256": expected_contract_sha256,
                "role_registry_sha256": expected_role_registry_sha256,
                "receipt_selection_sha256": expected_receipt_selection_sha256,
                "prediction_registry_sha256": sha256_file(
                    staging / "00_prediction_registry" / "ehr_age_d07_003_e55.json"
                ),
                "staged_primary_summary_sha256": sha256_file(
                    staging / "01_evaluation" / "STAGED_PRIMARY_SUMMARY.json"
                ),
                "hrkm_manifest_sha256": sha256_file(staging / "02_oof_hr_km_inputs" / "MANIFEST.json"),
                "single_bundle_atomic_publish_required": True,
            },
        )
        staging.rename(final_bundle)
        return {
            key: final_bundle / path.relative_to(staging)
            for key, path in staged.items()
        }
    except BaseException as exc:
        failure = staging / "AGGREGATION_FAILURE.json"
        if not failure.exists():
            _write_json_exclusive(
                failure,
                {
                    "status": "INCOMPLETE_NOT_CONSUMABLE",
                    "exception": {"type": type(exc).__name__, "message": str(exc)},
                    "completion_gate_written": False,
                },
            )
        raise
