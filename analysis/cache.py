from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from typing import Iterable

import numpy as np

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
