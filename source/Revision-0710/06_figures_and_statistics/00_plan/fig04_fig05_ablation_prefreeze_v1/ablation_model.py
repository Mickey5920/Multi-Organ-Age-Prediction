from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from ablation_inputs import apply_input_ablation, assert_left_aligned


REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_MODEL_DIR = REPO_ROOT / "Revision-0710" / "03_ehr_age" / "01_model"
if str(CANONICAL_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CANONICAL_MODEL_DIR))

from ehr_age_model import (  # noqa: E402
    BASE_AGE_HEAD_MODE_MLP,
    BASE_AGE_MODE_PRIVATE_RETAIN,
    BASE_AGE_MODE_V1,
    EHRAGEConfig,
    EHRAGEModel,
    ORGAN_SYSTEMS,
)


MODEL_ABLATION_IDS = (
    "fig04_01_shared_fusion_mlp_age_head",
    "fig04_02_no_organ_queries",
    "fig04_03_no_common_bidirectional_gru",
    "fig04_04_no_prognostic_supervision",
    "fig04_05_no_strict_offset_shrinkage",
    "fig04_06_no_event_thinning_consistency",
    "fig04_07_seven_independent_organ_age_heads",
)


@dataclass(frozen=True)
class IndependentOrganAgeOutput:
    predicted_age: torch.Tensor
    age_residual: torch.Tensor
    age_anchor: torch.Tensor
    base_age: None
    risk_score: torch.Tensor
    risk_age_offset: torch.Tensor
    interval_log_rate: torch.Tensor


class NoSharedGRUTemporalEncoder(nn.Module):
    """Canonical transformer path with the shared bidirectional-GRU branch removed."""

    shared_gru_output = "exact_zero_tensor"

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.transformer = source.transformer
        self.cls_token = source.cls_token
        self.position_embedding = source.position_embedding
        self.sequence_fusion = source.sequence_fusion
        self.global_fusion = source.global_fusion

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_base_views: bool = False,
        return_transformer_events: bool = False,
    ):
        if tokens.shape[1] > 32:
            raise ValueError("primary EHR-AGE comparison is restricted to cache L32")
        assert_left_aligned(mask)
        cls = self.cls_token.expand(len(tokens), -1, -1)
        transformer_input = torch.cat((cls, tokens), dim=1)
        transformer_input = transformer_input + self.position_embedding[
            :, : transformer_input.shape[1]
        ]
        padding = torch.cat(
            (
                torch.zeros(
                    (len(tokens), 1), dtype=torch.bool, device=tokens.device
                ),
                ~mask,
            ),
            dim=1,
        )
        transformed = self.transformer(
            transformer_input, src_key_padding_mask=padding
        )
        transformer_events = transformed[:, 1:] * mask.unsqueeze(-1).to(
            transformed.dtype
        )
        gru_events = torch.zeros_like(tokens)
        sequence = self.sequence_fusion(
            torch.cat((transformer_events, gru_events), dim=-1)
        )
        sequence = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        transformer_cls = transformed[:, 0]
        gru_pooled = torch.zeros_like(transformer_cls)
        pooled = self.global_fusion(
            torch.cat((transformer_cls, gru_pooled), dim=-1)
        )
        if return_base_views:
            return sequence, pooled, transformer_cls, gru_pooled
        if return_transformer_events:
            return sequence, pooled, transformer_events
        return sequence, pooled


class IndependentOrganAgeModel(nn.Module):
    """Seven direct organ-age heads plus the unchanged canonical survival head."""

    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        age_anchor: float,
        config: EHRAGEConfig,
    ) -> None:
        super().__init__()
        canonical = EHRAGEModel(
            code_embedding,
            text_embedding,
            age_anchor=age_anchor,
            config=config,
        )
        self.config = canonical.config
        self.event_encoder = canonical.event_encoder
        self.temporal_encoder = canonical.temporal_encoder
        self.static_encoder = canonical.static_encoder
        self.organ_queries = canonical.organ_queries
        self.organ_cross_attention = canonical.organ_cross_attention
        self.organ_fusion = canonical.organ_fusion
        self.organ_relation = canonical.organ_relation
        self.risk_head = canonical.risk_head
        self.baseline_log_rate = canonical.baseline_log_rate
        self.register_buffer(
            "age_anchor", canonical.age_anchor.detach().clone(), persistent=True
        )
        hidden = self.config.hidden_dim
        self.independent_age_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden // 2),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(hidden // 2, 1),
                )
                for _ in ORGAN_SYSTEMS
            ]
        )
        for head in self.independent_age_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _organ_context(
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
        mask = event_valid_mask.bool()
        tokens = self.event_encoder(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            mask,
        )
        sequence, pooled = self.temporal_encoder(tokens, mask)
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_sequence = sequence
        if empty.any():
            safe_mask[empty, 0] = True
            safe_sequence = sequence.clone()
            safe_sequence[empty, 0] = 0.0
        query = self.organ_queries.unsqueeze(0).expand(len(sequence), -1, -1)
        context, _ = self.organ_cross_attention(
            query,
            safe_sequence,
            safe_sequence,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        if empty.any():
            context = context.clone()
            context[empty] = 0.0
        organ = self.organ_fusion(
            torch.cat(
                (
                    context,
                    pooled.unsqueeze(1).expand(-1, len(ORGAN_SYSTEMS), -1),
                    static.unsqueeze(1).expand(-1, len(ORGAN_SYSTEMS), -1),
                    query,
                ),
                dim=-1,
            )
        )
        return self.organ_relation(organ)

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
    ) -> IndependentOrganAgeOutput:
        organ = self._organ_context(
            phenotype,
            phenotype_missing_mask,
            sex,
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        residual = torch.cat(
            [
                head(organ[:, index]).reshape(len(organ), 1)
                for index, head in enumerate(self.independent_age_heads)
            ],
            dim=1,
        )
        anchor = self.age_anchor.to(residual.dtype)
        predicted_age = anchor + residual
        risk_score = self.risk_head(organ).squeeze(-1)
        risk_offset = self.config.risk_offset_cap_years * torch.tanh(
            risk_score / self.config.risk_temperature
        )
        interval_log_rate = (
            self.baseline_log_rate.unsqueeze(0) + risk_score.unsqueeze(-1)
        )
        return IndependentOrganAgeOutput(
            predicted_age=predicted_age,
            age_residual=residual,
            age_anchor=anchor,
            base_age=None,
            risk_score=risk_score,
            risk_age_offset=risk_offset,
            interval_log_rate=interval_log_rate,
        )


class AblationModel(nn.Module):
    def __init__(self, variant_id: str, model: nn.Module) -> None:
        super().__init__()
        self.variant_id = str(variant_id)
        self.model = model

    def forward(self, **inputs: torch.Tensor):
        masked = apply_input_ablation(inputs, self.variant_id)
        return self.model(**masked)


def build_variant_model(
    variant_id: str,
    code_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    *,
    age_anchor: float,
    config: EHRAGEConfig,
) -> AblationModel:
    variant_id = str(variant_id)
    effective = config
    if variant_id == "fig04_01_shared_fusion_mlp_age_head":
        effective = replace(
            config,
            base_age_mode=BASE_AGE_MODE_V1,
            base_age_head_mode=BASE_AGE_HEAD_MODE_MLP,
        )
    elif config.base_age_mode != BASE_AGE_MODE_PRIVATE_RETAIN:
        raise ValueError("all non-shared-head ablations require private_retain base age")

    if variant_id == "fig04_07_seven_independent_organ_age_heads":
        model: nn.Module = IndependentOrganAgeModel(
            code_embedding,
            text_embedding,
            age_anchor=age_anchor,
            config=effective,
        )
    else:
        model = EHRAGEModel(
            code_embedding,
            text_embedding,
            age_anchor=age_anchor,
            config=effective,
        )
    if variant_id == "fig04_02_no_organ_queries":
        with torch.no_grad():
            model.organ_queries.zero_()
        model.organ_queries.requires_grad_(False)
    elif variant_id == "fig04_03_no_common_bidirectional_gru":
        model.temporal_encoder = NoSharedGRUTemporalEncoder(model.temporal_encoder)
    elif variant_id.startswith("fig04_") and variant_id not in MODEL_ABLATION_IDS:
        raise ValueError(f"unknown Fig04 model ablation: {variant_id}")
    return AblationModel(variant_id, model)


def model_ablation_receipt(model: AblationModel, variant_id: str) -> dict[str, object]:
    underlying = model.model
    if variant_id == "fig04_02_no_organ_queries":
        queries = underlying.organ_queries.detach()
        return {
            "all_seven_queries_identical": bool(
                torch.equal(queries, queries[0].expand_as(queries))
            ),
            "organ_queries_exact_zero": bool(torch.count_nonzero(queries).item() == 0),
            "organ_query_requires_grad": bool(underlying.organ_queries.requires_grad),
        }
    if variant_id == "fig04_03_no_common_bidirectional_gru":
        private = underlying.private_retain_age_tower.backbone
        return {
            "shared_temporal_bidirectional_gru_present": hasattr(
                underlying.temporal_encoder, "gru"
            ),
            "shared_gru_output": getattr(
                underlying.temporal_encoder, "shared_gru_output", None
            ),
            "private_retain_alpha_gru_present": isinstance(private.alpha_gru, nn.GRU),
            "private_retain_beta_gru_present": isinstance(private.beta_gru, nn.GRU),
        }
    if variant_id == "fig04_07_seven_independent_organ_age_heads":
        return {
            "independent_age_head_count": len(underlying.independent_age_heads),
            "shared_scalar_base_age_in_prediction": False,
            "survival_risk_head_retained": hasattr(underlying, "risk_head"),
        }
    return {"variant_id": variant_id, "model_constructed": True}


def build_tiny_variant_model(variant_id: str) -> AblationModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7210)
        code_embedding = torch.randn(64, 8)
        text_embedding = torch.randn(64, 8)
        code_embedding[0].zero_()
        text_embedding[0].zero_()
        config = EHRAGEConfig(
            hidden_dim=12,
            num_heads=3,
            num_transformer_layers=1,
            num_gru_layers=1,
            dropout=0.0,
            freeze_event_embeddings=True,
            risk_offset_cap_years=8.0,
            risk_temperature=2.0,
            survival_intervals=5,
            baseline_log_rate_init=-7.5,
            base_age_mode=BASE_AGE_MODE_PRIVATE_RETAIN,
            base_age_head_mode=BASE_AGE_HEAD_MODE_MLP,
        )
        return build_variant_model(
            variant_id,
            code_embedding,
            text_embedding,
            age_anchor=60.0,
            config=config,
        )
