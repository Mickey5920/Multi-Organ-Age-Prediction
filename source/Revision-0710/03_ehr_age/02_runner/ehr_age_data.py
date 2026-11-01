from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


REVISION_ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMMON = REVISION_ROOT / "02_baselines_5fold" / "common"
MODEL_DIR = Path(__file__).resolve().parents[1] / "01_model"
for _path in (BASELINE_COMMON, MODEL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from compact_cache_dataset import (  # noqa: E402
    CACHE_MODEL_INPUT_KEYS,
    CompactAgeDataset,
    CompactCacheBundle,
    PhenotypeTransform,
    _load_embeddings,
    _load_selected_shards,
)
from ehr_age_model import ORGAN_SYSTEMS  # noqa: E402


DEVELOPMENT_FOLDS = (1, 2, 3, 4)
TRAIN_FOLDS = (2, 3, 4)
VALIDATION_FOLD = 1
TEST_FOLD = 0
LABEL_COLUMNS = (
    "participant_id",
    "fold_id",
    "inner_role",
    "organ_system",
    "duration_months",
    "event",
    "at_risk",
    "strict_healthy_reference_60m",
)


def sha256_file(path: Path, block_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_pass_gate(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"required gate is not PASS: {path}")
    return sha256_file(path)


def prepare_fold0_development_labels(
    *,
    outcomes_path: Path,
    fold_manifest_path: Path,
    inner_split_path: Path,
    output_path: Path,
    age_targets_path: Path,
    age_output_path: Path,
    outer_test_age_output_path: Path,
    outer_test_label_output_path: Path,
    outer_test_manifest_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Materialize labels without Fold 0 before any model-development process starts."""

    folds = pd.read_parquet(fold_manifest_path, columns=["participant_id", "fold_id"])
    folds["participant_id"] = folds["participant_id"].astype(str)
    if folds["participant_id"].duplicated().any():
        raise ValueError("fold manifest contains duplicate participant_id")
    folds["fold_id"] = folds["fold_id"].astype(np.int8)
    development = folds.loc[folds["fold_id"].isin(DEVELOPMENT_FOLDS)].copy()
    if set(development["fold_id"].unique()) != set(DEVELOPMENT_FOLDS):
        raise ValueError("development fold manifest must contain folds 1, 2, 3, and 4")

    inner = pd.read_parquet(inner_split_path, columns=["participant_id", "fold_id", "inner_role"])
    inner["participant_id"] = inner["participant_id"].astype(str)
    if inner["participant_id"].duplicated().any() or set(inner["fold_id"].unique()) != {1}:
        raise ValueError("Fold 0 inner split must contain unique Fold 1 participants only")
    if set(inner["inner_role"].unique()) != {"calibration", "screen"}:
        raise ValueError("Fold 1 inner split must contain calibration and screen roles")

    all_outcomes = pd.read_parquet(
        outcomes_path,
        columns=[
            "participant_id",
            "organ_system",
            "duration_months",
            "event",
            "at_risk",
            "strict_healthy_reference_60m",
        ],
    )
    all_outcomes["participant_id"] = all_outcomes["participant_id"].astype(str)
    outcomes = all_outcomes.merge(development, on="participant_id", how="inner", validate="many_to_one")
    expected_rows = len(development) * len(ORGAN_SYSTEMS)
    if len(outcomes) != expected_rows:
        raise ValueError(f"development outcome join produced {len(outcomes):,}, expected {expected_rows:,}")
    if outcomes.duplicated(["participant_id", "organ_system"]).any():
        raise ValueError("development outcomes contain duplicate participant-organ rows")
    if set(outcomes["organ_system"].unique()) != set(ORGAN_SYSTEMS):
        raise ValueError("development outcomes do not contain the frozen seven-organ set")

    inner_role = inner.set_index("participant_id")["inner_role"]
    outcomes["inner_role"] = outcomes["participant_id"].map(inner_role)
    outcomes.loc[outcomes["fold_id"].isin(TRAIN_FOLDS), "inner_role"] = "train"
    if outcomes.loc[outcomes["fold_id"].eq(VALIDATION_FOLD), "inner_role"].isna().any():
        raise ValueError("some Fold 1 participants are missing calibration/screen assignment")
    if outcomes.loc[outcomes["fold_id"].isin(TRAIN_FOLDS), "inner_role"].ne("train").any():
        raise ValueError("training folds received an invalid inner role")
    if outcomes["fold_id"].eq(TEST_FOLD).any():
        raise RuntimeError("Fold 0 labels entered the development-label cache")

    outcomes["organ_system"] = pd.Categorical(
        outcomes["organ_system"], categories=list(ORGAN_SYSTEMS), ordered=True
    )
    outcomes = outcomes.sort_values(["fold_id", "participant_id", "organ_system"])
    outcomes = outcomes[list(LABEL_COLUMNS)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(output_path, index=False)

    all_age_targets = pd.read_parquet(
        age_targets_path, columns=["participant_id", "fold_id", "chronological_age"]
    )
    all_age_targets["participant_id"] = all_age_targets["participant_id"].astype(str)
    age_targets = all_age_targets.merge(
        development[["participant_id", "fold_id"]],
        on=["participant_id", "fold_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(age_targets) != len(development) or age_targets["fold_id"].eq(TEST_FOLD).any():
        raise ValueError("development age-target cache is incomplete or contains Fold 0")
    if not np.isfinite(age_targets["chronological_age"].to_numpy(dtype=np.float64)).all():
        raise ValueError("development age-target cache contains non-finite age")
    age_targets = age_targets.sort_values(["fold_id", "participant_id"])
    age_targets.to_parquet(age_output_path, index=False)

    outer_test_people = folds.loc[folds["fold_id"].eq(TEST_FOLD), ["participant_id", "fold_id"]]
    outer_test_age = all_age_targets.merge(
        outer_test_people,
        on=["participant_id", "fold_id"],
        how="inner",
        validate="one_to_one",
    ).sort_values("participant_id")
    outer_test_labels = all_outcomes.merge(
        outer_test_people,
        on="participant_id",
        how="inner",
        validate="many_to_one",
    )
    outer_test_labels["inner_role"] = "outer_test"
    outer_test_labels["organ_system"] = pd.Categorical(
        outer_test_labels["organ_system"], categories=list(ORGAN_SYSTEMS), ordered=True
    )
    outer_test_labels = outer_test_labels.sort_values(["participant_id", "organ_system"])
    outer_test_labels = outer_test_labels[list(LABEL_COLUMNS)]
    if len(outer_test_age) != len(outer_test_people):
        raise ValueError("sealed outer-test age cache is incomplete")
    if len(outer_test_labels) != len(outer_test_people) * len(ORGAN_SYSTEMS):
        raise ValueError("sealed outer-test outcome cache is incomplete")
    outer_test_age_output_path.parent.mkdir(parents=True, exist_ok=True)
    outer_test_age.to_parquet(outer_test_age_output_path, index=False)
    outer_test_labels.to_parquet(outer_test_label_output_path, index=False)
    sealed_manifest = {
        "schema_version": "revision0710_ehr_age_fold0_outer_test_materialization_v1",
        "status": "SEALED",
        "phase": "trusted_fixed_preprocessing_before_model_development",
        "outer_fold": 0,
        "participant_count": int(len(outer_test_people)),
        "participant_organ_rows": int(len(outer_test_labels)),
        "allowed_reader": "finalize_fold0_and_frozen_downstream_evaluator_only",
        "forbidden_reader": "train_fold0_development",
        "artifacts": {
            "age_targets": {
                "path": str(outer_test_age_output_path.resolve()),
                "sha256": sha256_file(outer_test_age_output_path),
            },
            "joint_labels": {
                "path": str(outer_test_label_output_path.resolve()),
                "sha256": sha256_file(outer_test_label_output_path),
            },
        },
    }
    outer_test_manifest_path.write_text(
        json.dumps(sealed_manifest, indent=2), encoding="utf-8"
    )

    counts = {
        str(int(fold)): int(count)
        for fold, count in development.groupby("fold_id").size().items()
    }
    manifest: dict[str, object] = {
        "schema_version": "revision0710_ehr_age_fold0_development_labels_v1",
        "status": "PASS",
        "phase": "trusted_fixed_preprocessing_before_model_development",
        "trusted_preprocessing_may_read_full_source_tables": True,
        "model_development_must_read_only_materialized_artifacts": True,
        "outer_test_fold_present": False,
        "outer_test_participant_rows": 0,
        "outer_test_participant_organ_rows": 0,
        "development_folds": list(DEVELOPMENT_FOLDS),
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALIDATION_FOLD,
        "participant_counts_by_fold": counts,
        "participant_count": int(len(development)),
        "participant_organ_rows": int(len(outcomes)),
        "organ_systems": list(ORGAN_SYSTEMS),
        "sources": {
            "organ_outcomes_60m": {
                "path": str(outcomes_path.resolve()),
                "sha256": sha256_file(outcomes_path),
            },
            "fold_manifest_5fold": {
                "path": str(fold_manifest_path.resolve()),
                "sha256": sha256_file(fold_manifest_path),
            },
            "fold0_validation_calibration_screen": {
                "path": str(inner_split_path.resolve()),
                "sha256": sha256_file(inner_split_path),
            },
            "chronological_age_targets": {
                "path": str(age_targets_path.resolve()),
                "sha256": sha256_file(age_targets_path),
            },
        },
        "artifacts": {
            "joint_labels": {
                "path": str(output_path.resolve()),
                "sha256": sha256_file(output_path),
            },
            "age_targets": {
                "path": str(age_output_path.resolve()),
                "sha256": sha256_file(age_output_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class JointLabels:
    strict_healthy: torch.Tensor
    duration_months: torch.Tensor
    event: torch.Tensor
    at_risk: torch.Tensor
    inner_role: tuple[str, ...]

    def validate(self, n: int) -> None:
        if self.strict_healthy.shape != (n,):
            raise ValueError("strict healthy label must have shape [participant]")
        expected = (n, len(ORGAN_SYSTEMS))
        if self.duration_months.shape != expected:
            raise ValueError("duration labels must have shape [participant,organ]")
        if self.event.shape != expected or self.at_risk.shape != expected:
            raise ValueError("event and at-risk labels must align with duration")
        if len(self.inner_role) != n:
            raise ValueError("inner roles must align with participants")
        if (self.event.bool() & ~self.at_risk.bool()).any():
            raise ValueError("event=1 outside the at-risk set")


@dataclass(frozen=True)
class Fold0DevelopmentData:
    cache: CompactCacheBundle
    labels: JointLabels
    train_indices: torch.Tensor
    calibration_indices: torch.Tensor
    screen_indices: torch.Tensor
    label_manifest: dict[str, object]


def _aligned_joint_labels(frame: pd.DataFrame, participant_ids: Sequence[str]) -> JointLabels:
    required = set(LABEL_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"development-label cache is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if frame["fold_id"].eq(TEST_FOLD).any():
        raise RuntimeError("Fold 0 is forbidden in the development-label cache")
    if frame.duplicated(["participant_id", "organ_system"]).any():
        raise ValueError("development-label cache contains duplicate participant-organ rows")
    expected_index = pd.MultiIndex.from_product(
        [pd.Index([str(value) for value in participant_ids]), list(ORGAN_SYSTEMS)],
        names=["participant_id", "organ_system"],
    )
    indexed = frame.set_index(["participant_id", "organ_system"]).reindex(expected_index)
    if indexed[["event", "at_risk"]].isna().any().any():
        raise ValueError("development labels are incomplete after cache alignment")
    missing_duration = indexed["duration_months"].isna()
    invalid_missing = missing_duration & indexed["at_risk"].ne(0)
    if invalid_missing.any():
        raise ValueError("an at-risk development row has missing duration")
    if (missing_duration & indexed["event"].ne(0)).any():
        raise ValueError("an event row has missing duration")
    indexed["duration_months"] = indexed["duration_months"].fillna(0.0)

    n = len(participant_ids)
    duration = indexed["duration_months"].to_numpy(dtype=np.float32).reshape(n, -1)
    event = indexed["event"].to_numpy(dtype=np.int8).reshape(n, -1)
    at_risk = indexed["at_risk"].to_numpy(dtype=np.int8).reshape(n, -1)
    strict_matrix = (
        indexed["strict_healthy_reference_60m"].to_numpy(dtype=np.int8).reshape(n, -1)
    )
    if not np.all(strict_matrix == strict_matrix[:, :1]):
        raise ValueError("strict healthy reference disagrees across organ rows")
    role_matrix = indexed["inner_role"].astype(str).to_numpy().reshape(n, -1)
    if not np.all(role_matrix == role_matrix[:, :1]):
        raise ValueError("inner role disagrees across organ rows")
    labels = JointLabels(
        strict_healthy=torch.from_numpy(strict_matrix[:, 0].astype(bool)),
        duration_months=torch.from_numpy(duration),
        event=torch.from_numpy(event.astype(bool)),
        at_risk=torch.from_numpy(at_risk.astype(bool)),
        inner_role=tuple(role_matrix[:, 0].tolist()),
    )
    labels.validate(n)
    return labels


def load_restricted_compact_cache(
    *,
    cache_dir: Path,
    outer_fold: int,
    fold_ids: Sequence[int],
    age_target_cache_path: Path,
) -> CompactCacheBundle:
    requested_folds = {int(value) for value in fold_ids}
    if not requested_folds or not requested_folds.issubset(set(range(5))):
        raise ValueError("restricted compact-cache folds must be a subset of 0..4")
    transform = PhenotypeTransform.load(
        cache_dir / "fold_transforms" / f"outer_fold_{int(outer_fold)}.npz"
    )
    arrays = _load_selected_shards(cache_dir, requested_folds)
    participant_ids = arrays["participant_id"].astype(str)
    fold_id_array = np.asarray(arrays["fold_id"], dtype=np.int64)
    phenotype = transform.apply(
        arrays["phenotype_raw_zero"], arrays["phenotype_missing_mask"]
    )
    age_frame = pd.read_parquet(age_target_cache_path)
    required_age = {"participant_id", "fold_id", "chronological_age"}
    if set(age_frame.columns) != required_age:
        raise ValueError("development age-target cache has an unexpected schema")
    age_frame["participant_id"] = age_frame["participant_id"].astype(str)
    if age_frame["participant_id"].duplicated().any():
        raise RuntimeError("restricted age-target cache contains duplicate participant_id")
    if set(age_frame["fold_id"].astype(int).unique()) != requested_folds:
        raise RuntimeError("restricted age-target cache contains unexpected folds")
    aligned_age = age_frame.set_index("participant_id").reindex(participant_ids)
    if aligned_age["chronological_age"].isna().any():
        raise ValueError("restricted age-target cache is incomplete after shard alignment")
    if not np.array_equal(aligned_age["fold_id"].to_numpy(dtype=np.int64), fold_id_array):
        raise ValueError("restricted age-target folds disagree with cache shards")
    age = aligned_age["chronological_age"].to_numpy(dtype=np.float32)
    inputs = {
        "phenotype": torch.from_numpy(np.ascontiguousarray(phenotype)),
        "phenotype_missing_mask": torch.from_numpy(
            np.ascontiguousarray(arrays["phenotype_missing_mask"], dtype=bool)
        ),
        "sex": torch.from_numpy(np.ascontiguousarray(arrays["sex"], dtype=np.int64)),
        "event_code_id": torch.from_numpy(
            np.ascontiguousarray(arrays["event_code_id"], dtype=np.int64)
        ),
        "event_text_id": torch.from_numpy(
            np.ascontiguousarray(arrays["event_text_id"], dtype=np.int64)
        ),
        "event_relative_days": torch.from_numpy(
            np.ascontiguousarray(arrays["event_relative_days"], dtype=np.int64)
        ),
        "event_type_id": torch.from_numpy(
            np.ascontiguousarray(arrays["event_type_id"], dtype=np.int64)
        ),
        "event_valid_mask": torch.from_numpy(
            np.ascontiguousarray(arrays["event_valid_mask"], dtype=bool)
        ),
    }
    dataset = CompactAgeDataset(
        participant_ids=participant_ids,
        inputs=inputs,
        chronological_age=torch.from_numpy(age),
        fold_id=torch.from_numpy(fold_id_array),
    )
    code_embedding, text_embedding = _load_embeddings(cache_dir)
    return CompactCacheBundle(dataset, code_embedding, text_embedding, transform)


def load_fold0_development_data(
    *,
    cache_dir: Path,
    label_cache_path: Path,
    age_target_cache_path: Path,
    label_manifest_path: Path,
) -> Fold0DevelopmentData:
    manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("phase") != "trusted_fixed_preprocessing_before_model_development"
        or manifest.get("model_development_must_read_only_materialized_artifacts") is not True
        or manifest.get("outer_test_fold_present") is not False
        or manifest.get("outer_test_participant_rows") != 0
    ):
        raise RuntimeError("development-label cache manifest did not pass the Fold 0 exclusion gate")
    artifacts = manifest.get("artifacts", {})
    if (
        not isinstance(artifacts, dict)
        or artifacts.get("joint_labels", {}).get("sha256") != sha256_file(label_cache_path)
        or artifacts.get("age_targets", {}).get("sha256") != sha256_file(age_target_cache_path)
    ):
        raise RuntimeError("development-label cache hash disagrees with its manifest")

    # This is the hard boundary: neither shards nor the age-target join open Fold 0.
    cache = load_restricted_compact_cache(
        cache_dir=cache_dir,
        outer_fold=0,
        fold_ids=DEVELOPMENT_FOLDS,
        age_target_cache_path=age_target_cache_path,
    )
    observed_folds = set(cache.dataset.fold_id.tolist())
    if observed_folds != set(DEVELOPMENT_FOLDS) or TEST_FOLD in observed_folds:
        raise RuntimeError("development cache contains the sealed outer test fold")
    labels_frame = pd.read_parquet(label_cache_path)
    labels = _aligned_joint_labels(labels_frame, cache.dataset.participant_ids)
    fold_id = cache.dataset.fold_id
    roles = np.asarray(labels.inner_role, dtype=object)
    train = torch.nonzero(
        torch.isin(fold_id, torch.tensor(TRAIN_FOLDS, dtype=fold_id.dtype)), as_tuple=False
    ).flatten()
    calibration = torch.from_numpy(np.flatnonzero(roles == "calibration")).long()
    screen = torch.from_numpy(np.flatnonzero(roles == "screen")).long()
    if len(train) == 0 or len(calibration) == 0 or len(screen) == 0:
        raise ValueError("training, calibration, and screen partitions must all be non-empty")
    if not fold_id.index_select(0, calibration).eq(VALIDATION_FOLD).all():
        raise ValueError("calibration partition is not restricted to Fold 1")
    if not fold_id.index_select(0, screen).eq(VALIDATION_FOLD).all():
        raise ValueError("screen partition is not restricted to Fold 1")
    return Fold0DevelopmentData(cache, labels, train, calibration, screen, manifest)


@dataclass(frozen=True)
class JointBatch:
    participant_indices: torch.Tensor
    inputs: dict[str, torch.Tensor]
    chronological_age: torch.Tensor
    strict_healthy: torch.Tensor
    duration_months: torch.Tensor
    event: torch.Tensor
    at_risk: torch.Tensor


class JointVectorizedBatcher:
    def __init__(
        self,
        data: Fold0DevelopmentData,
        indices: torch.Tensor,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.data = data
        self.indices = indices.detach().cpu().long()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.iteration = 0

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self) -> Iterator[JointBatch]:
        positions = self.indices
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.iteration)
            order = torch.randperm(len(positions), generator=generator)
            positions = positions.index_select(0, order)
        self.iteration += 1
        dataset = self.data.cache.dataset
        labels = self.data.labels
        for start in range(0, len(positions), self.batch_size):
            selected = positions[start : start + self.batch_size]
            yield JointBatch(
                participant_indices=selected,
                inputs={
                    key: dataset.inputs[key].index_select(0, selected)
                    for key in CACHE_MODEL_INPUT_KEYS
                },
                chronological_age=dataset.chronological_age.index_select(0, selected),
                strict_healthy=labels.strict_healthy.index_select(0, selected),
                duration_months=labels.duration_months.index_select(0, selected),
                event=labels.event.index_select(0, selected),
                at_risk=labels.at_risk.index_select(0, selected),
            )


def compact_event_dropout(
    inputs: Mapping[str, torch.Tensor],
    *,
    probability: float,
) -> dict[str, torch.Tensor]:
    """Drop valid events, then left-compact every event tuple without reordering."""

    if not 0.0 <= probability < 1.0:
        raise ValueError("event dropout probability must lie in [0,1)")
    result = {key: value for key, value in inputs.items()}
    if probability == 0.0:
        return result
    valid = inputs["event_valid_mask"].bool()
    keep = valid & torch.rand(valid.shape, device=valid.device).ge(probability)
    destination = keep.long().cumsum(dim=1) - 1
    rows, columns = torch.nonzero(keep, as_tuple=True)
    target_columns = destination[rows, columns]
    event_keys = (
        "event_code_id",
        "event_text_id",
        "event_relative_days",
        "event_type_id",
        "event_valid_mask",
    )
    for key in event_keys:
        source = inputs[key]
        compacted = torch.zeros_like(source)
        compacted[rows, target_columns] = source[rows, columns]
        result[key] = compacted
    return result
