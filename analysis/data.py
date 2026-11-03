from __future__ import annotations

import math

from dataclasses import dataclass

from typing import Iterator, Mapping

import torch

from .cache import CACHE_MODEL_INPUT_KEYS, CompactCacheBundle

from .types import ORGAN_SYSTEMS

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
