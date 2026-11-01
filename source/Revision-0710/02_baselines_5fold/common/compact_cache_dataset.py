from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EVENT_INPUT_KEYS = (
    "event_code_id",
    "event_text_id",
    "event_relative_days",
    "event_type_id",
    "event_valid_mask",
)
CACHE_MODEL_INPUT_KEYS = (
    "phenotype",
    "phenotype_missing_mask",
    "sex",
    *EVENT_INPUT_KEYS,
)
_SHARD_ARRAY_KEYS = (
    "participant_id",
    "fold_id",
    "phenotype_raw_zero",
    "phenotype_missing_mask",
    "sex",
    *EVENT_INPUT_KEYS,
)
_FORBIDDEN_EXACT_KEYS = {
    "chronological_age",
    "reported_age_at_recruitment",
    "age_target",
    "target",
    "label",
    "event",
    "outcome",
    "duration",
    "death",
    "survival_time",
    "strict_healthy",
    "future_disease",
    "system_mask",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "duration",
    "death",
    "outcome",
    "strict_healthy",
    "future_disease",
    "survival",
    "system_mask",
    "chronological_age",
    "reported_age",
    "age_target",
)


@dataclass(frozen=True)
class PhenotypeTransform:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    outer_fold: int
    validation_fold: int

    @classmethod
    def load(cls, path: Path) -> "PhenotypeTransform":
        with np.load(path, allow_pickle=False) as payload:
            required = {"median", "mean", "std", "outer_fold", "validation_fold"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"phenotype transform is missing keys: {sorted(missing)}")
            median = np.asarray(payload["median"], dtype=np.float32)
            mean = np.asarray(payload["mean"], dtype=np.float32)
            std = np.asarray(payload["std"], dtype=np.float32)
            outer_fold = int(payload["outer_fold"])
            validation_fold = int(payload["validation_fold"])
        if median.ndim != 1 or mean.shape != median.shape or std.shape != median.shape:
            raise ValueError("phenotype transform vectors must be one-dimensional and aligned")
        if not np.isfinite(median).all() or not np.isfinite(mean).all():
            raise ValueError("phenotype transform contains non-finite location values")
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("phenotype transform standard deviations must be finite and positive")
        return cls(median, mean, std, outer_fold, validation_fold)

    def apply(self, raw_zero: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        raw_zero = np.asarray(raw_zero, dtype=np.float32)
        missing_mask = np.asarray(missing_mask, dtype=bool)
        if raw_zero.shape != missing_mask.shape:
            raise ValueError("phenotype values and missing mask have different shapes")
        if raw_zero.ndim != 2 or raw_zero.shape[1] != len(self.median):
            raise ValueError("phenotype feature count disagrees with the fold transform")
        imputed = np.where(missing_mask, self.median[None, :], raw_zero)
        transformed = (imputed - self.mean[None, :]) / self.std[None, :]
        if not np.isfinite(transformed).all():
            raise ValueError("fold-transformed phenotype contains non-finite values")
        return transformed.astype(np.float32, copy=False)


class CompactAgeDataset(Dataset[tuple[dict[str, torch.Tensor], torch.Tensor]]):
    def __init__(
        self,
        *,
        participant_ids: Iterable[str],
        inputs: dict[str, torch.Tensor],
        chronological_age: torch.Tensor,
        fold_id: torch.Tensor,
    ) -> None:
        self.participant_ids = tuple(str(value) for value in participant_ids)
        self.inputs = inputs
        self.chronological_age = chronological_age.float()
        self.fold_id = fold_id.long()
        n = len(self.participant_ids)
        if set(inputs) != set(CACHE_MODEL_INPUT_KEYS):
            raise ValueError("dataset inputs do not match the cache model-input contract")
        if len(self.chronological_age) != n or len(self.fold_id) != n:
            raise ValueError("dataset metadata length mismatch")
        if any(len(value) != n for value in inputs.values()):
            raise ValueError("dataset input length mismatch")

    def __len__(self) -> int:
        return len(self.participant_ids)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return ({key: value[index] for key, value in self.inputs.items()}, self.chronological_age[index])


@dataclass(frozen=True)
class CompactCacheBundle:
    dataset: CompactAgeDataset
    code_embedding: torch.Tensor
    text_embedding: torch.Tensor
    phenotype_transform: PhenotypeTransform


def _resolved_child(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"shard path escapes cache root: {relative_path}") from exc
    return candidate


def _load_embeddings(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    table_path = cache_dir / "embedding_tables.pt"
    try:
        payload = torch.load(table_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(table_path, map_location="cpu")
    required = {"code_embedding", "text_embedding", "padding_id", "embedding_dim"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"embedding table is missing keys: {sorted(missing)}")
    if int(payload["padding_id"]) != 0:
        raise ValueError("compact cache requires padding_id=0")
    code = torch.as_tensor(payload["code_embedding"]).detach().cpu()
    text = torch.as_tensor(payload["text_embedding"]).detach().cpu()
    dimension = int(payload["embedding_dim"])
    if code.ndim != 2 or text.ndim != 2 or code.shape[1] != dimension or text.shape[1] != dimension:
        raise ValueError("embedding-table dimensions disagree with embedding_dim")
    if not torch.isfinite(code).all() or not torch.isfinite(text).all():
        raise ValueError("embedding table contains non-finite values")
    if torch.count_nonzero(code[0]).item() or torch.count_nonzero(text[0]).item():
        raise ValueError("padding rows in embedding tables must be exactly zero")
    return code, text


def _forbidden_shard_keys(keys: Iterable[str]) -> list[str]:
    forbidden = []
    for key in keys:
        normalized = key.strip().lower()
        if normalized in _FORBIDDEN_EXACT_KEYS or any(
            fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS
        ):
            forbidden.append(key)
    return sorted(forbidden)


def _load_selected_shards(cache_dir: Path, fold_ids: set[int]) -> dict[str, np.ndarray]:
    manifest_path = cache_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [row for row in manifest if int(row["fold_id"]) in fold_ids]
    if not selected:
        raise ValueError(f"no compact-cache shards found for folds {sorted(fold_ids)}")

    chunks: dict[str, list[np.ndarray]] = {key: [] for key in _SHARD_ARRAY_KEYS}
    for row in selected:
        path = _resolved_child(cache_dir, str(row["path"]))
        with np.load(path, allow_pickle=False) as payload:
            forbidden = _forbidden_shard_keys(payload.files)
            if forbidden:
                raise ValueError(f"forbidden target/outcome key in input shard {path.name}: {forbidden}")
            missing = set(_SHARD_ARRAY_KEYS).difference(payload.files)
            if missing:
                raise ValueError(f"input shard {path.name} is missing keys: {sorted(missing)}")
            shard_n = len(payload["participant_id"])
            if int(row.get("n", shard_n)) != shard_n:
                raise ValueError(f"manifest row count disagrees with input shard {path.name}")
            observed_folds = set(np.asarray(payload["fold_id"], dtype=np.int64).tolist())
            if observed_folds != {int(row["fold_id"])}:
                raise ValueError(f"manifest fold disagrees with input shard {path.name}")
            for key in _SHARD_ARRAY_KEYS:
                chunks[key].append(np.asarray(payload[key]))
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    participant_ids = arrays["participant_id"].astype(str)
    if len(set(participant_ids.tolist())) != len(participant_ids):
        raise ValueError("participant_id is duplicated across selected compact-cache shards")
    return arrays


def _join_age_targets(
    target_path: Path,
    participant_ids: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    target = pd.read_parquet(target_path)
    required = {"participant_id", "fold_id", "chronological_age"}
    missing = required.difference(target.columns)
    if missing:
        raise ValueError(f"age-target table is missing columns: {sorted(missing)}")
    target = target[list(required)].copy()
    target["participant_id"] = target["participant_id"].astype(str)
    if target["participant_id"].duplicated().any():
        raise ValueError("age-target table contains duplicate participant_id")
    indexed = target.set_index("participant_id").reindex(participant_ids.astype(str))
    if indexed["chronological_age"].isna().any():
        missing_ids = participant_ids[indexed["chronological_age"].isna().to_numpy()][:5]
        raise ValueError(f"missing chronological-age target for participants: {missing_ids.tolist()}")
    joined_fold = indexed["fold_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(joined_fold, np.asarray(fold_id, dtype=np.int64)):
        raise ValueError("age-target fold_id disagrees with compact-cache shard")
    age = indexed["chronological_age"].to_numpy(dtype=np.float32)
    if not np.isfinite(age).all():
        raise ValueError("chronological-age target contains non-finite values")
    return age


def load_compact_cache(
    cache_dir: str | Path,
    *,
    outer_fold: int,
    fold_ids: Iterable[int],
) -> CompactCacheBundle:
    cache_dir = Path(cache_dir)
    requested_folds = {int(value) for value in fold_ids}
    if not requested_folds or not requested_folds.issubset(set(range(5))):
        raise ValueError("fold_ids must be a non-empty subset of {0,1,2,3,4}")
    transform = PhenotypeTransform.load(
        cache_dir / "fold_transforms" / f"outer_fold_{int(outer_fold)}.npz"
    )
    if transform.outer_fold != int(outer_fold):
        raise ValueError("requested outer fold disagrees with transform metadata")
    arrays = _load_selected_shards(cache_dir, requested_folds)
    participant_ids = arrays["participant_id"].astype(str)
    fold_id = np.asarray(arrays["fold_id"], dtype=np.int64)
    if not set(fold_id.tolist()).issubset(requested_folds):
        raise ValueError("loaded shard contains an unrequested fold")
    phenotype = transform.apply(
        arrays["phenotype_raw_zero"], arrays["phenotype_missing_mask"]
    )
    age = _join_age_targets(
        cache_dir / "targets" / "chronological_age_targets.parquet",
        participant_ids,
        fold_id,
    )
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
        fold_id=torch.from_numpy(fold_id),
    )
    code_embedding, text_embedding = _load_embeddings(cache_dir)
    return CompactCacheBundle(dataset, code_embedding, text_embedding, transform)
