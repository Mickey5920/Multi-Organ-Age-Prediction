from __future__ import annotations

from dataclasses import dataclass

import torch

ORGAN_SYSTEMS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)

@dataclass(frozen=True)
class EHRAGEOutput:
    predicted_age: torch.Tensor
    base_age: torch.Tensor
    risk_score: torch.Tensor
    risk_age_offset: torch.Tensor
    interval_log_rate: torch.Tensor
    base_view_ages: torch.Tensor | None = None
