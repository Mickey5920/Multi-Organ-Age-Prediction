from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = EXPERIMENT_DIR.parent
REVISION_ROOT = BASELINE_DIR.parent
COMMON_DIR = BASELINE_DIR / "common"
EHR_MODEL_DIR = REVISION_ROOT / "03_ehr_age" / "01_model"
for path in (COMMON_DIR, EHR_MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_models import StaticEncoder
from ehr_age_model import EHRAGEOutput, ORGAN_SYSTEMS
from masked_backbones import (
    BEHRTStyleBackbone,
    EHRFormerStyleBackbone,
    MedBERTStyleBackbone,
    RETAINBackbone,
    BackboneOutput,
    EventTokenEncoder,
    masked_mean,
)


CONTEMPORARY_MODELS = (
    "retain",
    "behrt_style",
    "medbert_style",
    "ehrformer_style",
)
GENERIC_NEURAL_MODELS = ("mlp", "cnn", "resnet", "transformer")


def _event_arguments(values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        values["event_code_id"],
        values["event_text_id"],
        values["event_relative_days"],
        values["event_type_id"],
        values["event_valid_mask"],
    )


def build_contemporary_backbone(
    name: str,
    code_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    *,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    freeze_embeddings: bool,
) -> nn.Module:
    token_encoder = EventTokenEncoder(
        code_embedding,
        text_embedding,
        hidden_dim=hidden_dim,
        type_vocab_size=3,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )
    if name == "retain":
        return RETAINBackbone(token_encoder, hidden_dim=hidden_dim, dropout=dropout)
    if name == "behrt_style":
        return BEHRTStyleBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name == "medbert_style":
        return MedBERTStyleBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name == "ehrformer_style":
        return EHRFormerStyleBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"unknown contemporary backbone: {name}")


class ProspectiveBaselineAdapter(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_dim: int,
        age_anchor: float,
        dropout: float,
        risk_offset_cap_years: float,
        risk_temperature: float,
        survival_intervals: int,
        baseline_log_rate_init: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.static_encoder = StaticEncoder(hidden_dim, dropout)
        self.patient_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_age_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.organ_embeddings = nn.Parameter(
            torch.randn(len(ORGAN_SYSTEMS), hidden_dim) * 0.02
        )
        self.organ_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.base_age_head[-1].weight)
        nn.init.zeros_(self.base_age_head[-1].bias)
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.zeros_(self.risk_head[-1].bias)
        self.baseline_log_rate = nn.Parameter(
            torch.full(
                (len(ORGAN_SYSTEMS), survival_intervals),
                float(baseline_log_rate_init),
            )
        )
        self.register_buffer(
            "age_anchor", torch.tensor(float(age_anchor), dtype=torch.float32)
        )
        self.risk_offset_cap_years = float(risk_offset_cap_years)
        self.risk_temperature = float(risk_temperature)

    def forward(
        self,
        phenotype: torch.Tensor,
        phenotype_missing_mask: torch.Tensor,
        sex: torch.Tensor,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> EHRAGEOutput:
        values = locals()
        temporal = self.backbone.encode(*_event_arguments(values))
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
        patient = self.patient_fusion(torch.cat((temporal.pooled, static), dim=1))
        base_residual = self.base_age_head(patient).squeeze(1)
        base_age = self.age_anchor.to(base_residual.dtype) + base_residual
        expanded_patient = patient.unsqueeze(1).expand(-1, len(ORGAN_SYSTEMS), -1)
        organ = self.organ_embeddings.unsqueeze(0).expand(len(patient), -1, -1)
        organ_context = self.organ_fusion(torch.cat((expanded_patient, organ), dim=2))
        risk_score = self.risk_head(organ_context).squeeze(2)
        offset = self.risk_offset_cap_years * torch.tanh(
            risk_score / self.risk_temperature
        )
        predicted_age = base_age.unsqueeze(1) + offset
        interval_log_rate = self.baseline_log_rate.unsqueeze(0) + risk_score.unsqueeze(2)
        return EHRAGEOutput(
            predicted_age=predicted_age,
            base_age=base_age,
            risk_score=risk_score,
            risk_age_offset=offset,
            interval_log_rate=interval_log_rate,
            base_view_ages=None,
        )


class MLPBackbone(nn.Module):
    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        maximum_events: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_encoder = token_encoder
        self.maximum_events = int(maximum_events)
        self.network = nn.Sequential(
            nn.Linear(self.maximum_events * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def encode(self, *events: torch.Tensor) -> BackboneOutput:
        mask = events[-1].bool()
        tokens = self.token_encoder(*events)
        if tokens.shape[1] != self.maximum_events:
            raise ValueError("MLP backbone received an unexpected event length")
        pooled = self.network(tokens.reshape(len(tokens), -1))
        return BackboneOutput(pooled=pooled, sequence=tokens, valid_mask=mask)


class CNNBackbone(nn.Module):
    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_encoder = token_encoder
        self.network = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def encode(self, *events: torch.Tensor) -> BackboneOutput:
        mask = events[-1].bool()
        tokens = self.token_encoder(*events)
        sequence = self.network(tokens.transpose(1, 2)).transpose(1, 2)
        sequence = self.normalization(sequence)
        sequence = sequence * mask.unsqueeze(2).to(sequence.dtype)
        return BackboneOutput(
            pooled=masked_mean(sequence, mask), sequence=sequence, valid_mask=mask
        )


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        bottleneck_dim = max(32, hidden_dim // 4)
        self.first = nn.Linear(3 * hidden_dim, bottleneck_dim)
        self.second = nn.Linear(bottleneck_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        left = torch.zeros_like(values)
        right = torch.zeros_like(values)
        left[:, 1:] = values[:, :-1]
        right[:, :-1] = values[:, 1:]
        hidden = torch.nn.functional.gelu(
            self.first(torch.cat((left, values, right), dim=2))
        )
        hidden = self.dropout(hidden)
        hidden = self.second(hidden)
        hidden = self.normalization(hidden + values)
        return torch.nn.functional.gelu(hidden) * mask.unsqueeze(2).to(hidden.dtype)


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_encoder = token_encoder
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden_dim, dropout)
            for _ in range(max(1, num_layers))
        )

    def encode(self, *events: torch.Tensor) -> BackboneOutput:
        mask = events[-1].bool()
        sequence = self.token_encoder(*events)
        for block in self.blocks:
            sequence = block(sequence, mask)
        return BackboneOutput(
            pooled=masked_mean(sequence, mask), sequence=sequence, valid_mask=mask
        )


class GenericTransformerBackbone(nn.Module):
    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        maximum_events: int,
    ) -> None:
        super().__init__()
        self.token_encoder = token_encoder
        self.position_embedding = nn.Parameter(
            torch.zeros(1, maximum_events, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.position_embedding, std=0.02)

    def encode(self, *events: torch.Tensor) -> BackboneOutput:
        mask = events[-1].bool()
        tokens = self.token_encoder(*events)
        if tokens.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("Transformer backbone received too many events")
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            tokens = tokens.clone()
            tokens[empty, 0] = 0.0
        sequence = self.transformer(tokens, src_key_padding_mask=~safe_mask)
        sequence = self.normalization(sequence)
        sequence = sequence * mask.unsqueeze(2).to(sequence.dtype)
        return BackboneOutput(
            pooled=masked_mean(sequence, mask), sequence=sequence, valid_mask=mask
        )


def build_generic_backbone(
    name: str,
    code_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    *,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    maximum_events: int,
    freeze_embeddings: bool,
) -> nn.Module:
    token_encoder = EventTokenEncoder(
        code_embedding,
        text_embedding,
        hidden_dim=hidden_dim,
        type_vocab_size=3,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )
    if name == "mlp":
        return MLPBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            maximum_events=maximum_events,
            dropout=dropout,
        )
    if name == "cnn":
        return CNNBackbone(token_encoder, hidden_dim=hidden_dim, dropout=dropout)
    if name == "resnet":
        return ResNetBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_layers=max(3, num_layers),
            dropout=dropout,
        )
    if name == "transformer":
        return GenericTransformerBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            maximum_events=maximum_events,
        )
    raise ValueError(f"unknown generic neural backbone: {name}")


class GenericOrganAgeModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_dim: int,
        age_anchor: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.static_encoder = StaticEncoder(hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.organ_embeddings = nn.Parameter(
            torch.randn(len(ORGAN_SYSTEMS), hidden_dim) * 0.02
        )
        self.age_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.age_head[-1].weight)
        nn.init.zeros_(self.age_head[-1].bias)
        self.register_buffer(
            "age_anchor", torch.tensor(float(age_anchor), dtype=torch.float32)
        )

    def forward(
        self,
        phenotype: torch.Tensor,
        phenotype_missing_mask: torch.Tensor,
        sex: torch.Tensor,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = locals()
        temporal = self.backbone.encode(*_event_arguments(values))
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
        patient = self.fusion(torch.cat((temporal.pooled, static), dim=1))
        expanded = patient.unsqueeze(1).expand(-1, len(ORGAN_SYSTEMS), -1)
        organ = self.organ_embeddings.unsqueeze(0).expand(len(patient), -1, -1)
        residual = self.age_head(torch.cat((expanded, organ), dim=2)).squeeze(2)
        return self.age_anchor.to(residual.dtype) + residual
