from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from compact_cache_dataset import CACHE_MODEL_INPUT_KEYS
from masked_backbones import (
    BEHRTStyleBackbone,
    EHRFormerStyleBackbone,
    MedBERTStyleBackbone,
    RETAINBackbone,
    EventTokenEncoder,
    event_inputs_from_batch,
)


ORGAN_SYSTEMS = (
    "cardiovascular",
    "pulmonary",
    "musculoskeletal",
    "immune",
    "renal",
    "hepatic",
    "metabolic",
)
PLAIN_MODEL_NAMES = ("retain", "behrt_style", "medbert_style", "ehrformer_style")
ORGANQUERY_MODEL_NAMES = tuple(f"{name}_organquery" for name in PLAIN_MODEL_NAMES)
MODEL_NAMES = PLAIN_MODEL_NAMES + ORGANQUERY_MODEL_NAMES


def model_inputs_from_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = set(CACHE_MODEL_INPUT_KEYS).difference(batch)
    if missing:
        raise KeyError(f"batch is missing model inputs: {sorted(missing)}")
    return {key: batch[key] for key in CACHE_MODEL_INPUT_KEYS}


class StaticEncoder(nn.Module):
    """Encode 78 normalized phenotypes, their 78 missing flags, and sex."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(78 + 78 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        phenotype: torch.Tensor,
        phenotype_missing_mask: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        if phenotype.ndim != 2 or phenotype.shape[1] != 78:
            raise ValueError("phenotype must have shape [batch,78]")
        if phenotype_missing_mask.shape != phenotype.shape:
            raise ValueError("phenotype missing mask must align with phenotype")
        if sex.ndim == 1:
            sex = sex.unsqueeze(1)
        if sex.shape != (len(phenotype), 1):
            raise ValueError("sex must have shape [batch] or [batch,1]")
        static = torch.cat(
            (
                phenotype.float(),
                phenotype_missing_mask.to(dtype=phenotype.dtype),
                sex.to(dtype=phenotype.dtype),
            ),
            dim=1,
        )
        return self.network(static)


class _AgeModel(nn.Module):
    def __init__(self, backbone: nn.Module, *, hidden_dim: int, age_anchor: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("age_anchor", torch.tensor(float(age_anchor), dtype=torch.float32))
        self.hidden_dim = int(hidden_dim)

    @staticmethod
    def _event_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return event_inputs_from_batch(batch)


class PlainOrganAgeModel(_AgeModel):
    """Architecture-faithful backbone with a minimal shared organ-conditioned head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_dim: int,
        age_anchor: float,
        dropout: float,
    ) -> None:
        super().__init__(backbone, hidden_dim=hidden_dim, age_anchor=age_anchor)
        self.static_encoder = StaticEncoder(hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.organ_embeddings = nn.Parameter(torch.randn(len(ORGAN_SYSTEMS), hidden_dim) * 0.02)
        self.age_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.age_head[-1].weight)
        nn.init.zeros_(self.age_head[-1].bias)

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
        batch = locals()
        temporal = self.backbone(**self._event_inputs(batch))
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
        patient = self.fusion(torch.cat((temporal, static), dim=1))
        patient = patient.unsqueeze(1).expand(-1, len(ORGAN_SYSTEMS), -1)
        organ = self.organ_embeddings.unsqueeze(0).expand(len(patient), -1, -1)
        residual = self.age_head(torch.cat((patient, organ), dim=2)).squeeze(2)
        return self.age_anchor.to(residual.dtype) + residual


class OrganQueryAgeModel(_AgeModel):
    """Seven learnable organ queries cross-attend only valid backbone states."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_dim: int,
        age_anchor: float,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__(backbone, hidden_dim=hidden_dim, age_anchor=age_anchor)
        self.static_encoder = StaticEncoder(hidden_dim, dropout)
        self.organ_queries = nn.Parameter(torch.randn(len(ORGAN_SYSTEMS), hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.age_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.age_head[-1].weight)
        nn.init.zeros_(self.age_head[-1].bias)

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
        encoded = self.backbone.encode(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        sequence = encoded.sequence
        safe_mask = encoded.valid_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            sequence = sequence.clone()
            safe_mask[empty, 0] = True
            sequence[empty, 0] = 0.0
        query = self.organ_queries.unsqueeze(0).expand(len(sequence), -1, -1)
        context, _ = self.cross_attention(
            query,
            sequence,
            sequence,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        if empty.any():
            context = context.clone()
            context[empty] = 0.0
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
        static = static.unsqueeze(1).expand(-1, len(ORGAN_SYSTEMS), -1)
        pooled = encoded.pooled.unsqueeze(1).expand(-1, len(ORGAN_SYSTEMS), -1)
        fused = self.fusion(torch.cat((context, pooled, static), dim=2))
        residual = self.age_head(fused).squeeze(2)
        return self.age_anchor.to(residual.dtype) + residual


def build_age_model(
    model_name: str,
    code_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    *,
    age_anchor: float,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.15,
    freeze_embeddings: bool = True,
) -> _AgeModel:
    normalized = model_name.lower().replace("-", "_")
    if normalized not in MODEL_NAMES:
        raise ValueError(f"unknown baseline model: {model_name}")
    base_name = normalized.removesuffix("_organquery")
    token_encoder = EventTokenEncoder(
        code_embedding,
        text_embedding,
        hidden_dim=hidden_dim,
        type_vocab_size=3,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )
    if base_name == "retain":
        backbone = RETAINBackbone(token_encoder, hidden_dim=hidden_dim, dropout=dropout)
    elif base_name == "behrt_style":
        backbone = BEHRTStyleBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif base_name == "medbert_style":
        backbone = MedBERTStyleBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        backbone = EHRFormerStyleBackbone(
            token_encoder,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    if normalized in ORGANQUERY_MODEL_NAMES:
        return OrganQueryAgeModel(
            backbone,
            hidden_dim=hidden_dim,
            age_anchor=age_anchor,
            num_heads=num_heads,
            dropout=dropout,
        )
    return PlainOrganAgeModel(
        backbone,
        hidden_dim=hidden_dim,
        age_anchor=age_anchor,
        dropout=dropout,
    )
