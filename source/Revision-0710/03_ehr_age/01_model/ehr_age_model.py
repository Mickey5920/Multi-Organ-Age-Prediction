from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

try:
    from .private_retain_age import PrivateRETAINAgeTower
except ImportError:  # Support direct imports from the revision-local model directory.
    from private_retain_age import PrivateRETAINAgeTower


ORGAN_SYSTEMS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)

BASE_AGE_MODE_V1 = "shared_fusion"
BASE_AGE_MODE_THREE_VIEW_EQUAL_MEAN = "three_view_equal_mean"
BASE_AGE_MODE_PRIVATE_RETAIN = "private_retain"
BASE_AGE_MODE_SHARED_PRIVATE_EQUAL_MEAN = "shared_private_equal_mean"
BASE_AGE_MODES = (
    BASE_AGE_MODE_V1,
    BASE_AGE_MODE_THREE_VIEW_EQUAL_MEAN,
    BASE_AGE_MODE_PRIVATE_RETAIN,
    BASE_AGE_MODE_SHARED_PRIVATE_EQUAL_MEAN,
)
BASE_AGE_HEAD_MODE_LINEAR = "linear"
BASE_AGE_HEAD_MODE_MLP = "mlp"
BASE_AGE_HEAD_MODES = (
    BASE_AGE_HEAD_MODE_LINEAR,
    BASE_AGE_HEAD_MODE_MLP,
)
BASE_AGE_POOLING_MODE_GLOBAL_FUSED = "global_fused"
BASE_AGE_POOLING_MODE_TRANSFORMER_EVENT_MEAN = "transformer_event_mean"
BASE_AGE_POOLING_MODES = (
    BASE_AGE_POOLING_MODE_GLOBAL_FUSED,
    BASE_AGE_POOLING_MODE_TRANSFORMER_EVENT_MEAN,
)
BASE_AGE_VIEW_NAMES = (
    "transformer_cls",
    "gru_pooled",
    "semantic_mean",
)


@dataclass(frozen=True)
class EHRAGEConfig:
    hidden_dim: int = 192
    num_heads: int = 6
    num_transformer_layers: int = 3
    num_gru_layers: int = 2
    dropout: float = 0.15
    freeze_event_embeddings: bool = True
    risk_offset_cap_years: float = 8.0
    risk_temperature: float = 2.0
    survival_intervals: int = 5
    baseline_log_rate_init: float = -7.5
    base_age_mode: str = BASE_AGE_MODE_V1
    base_age_head_mode: str = BASE_AGE_HEAD_MODE_LINEAR
    base_age_pooling_mode: str = BASE_AGE_POOLING_MODE_GLOBAL_FUSED

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even for the bidirectional GRU")
        if self.num_transformer_layers <= 0 or self.num_gru_layers <= 0:
            raise ValueError("encoder layer counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.risk_offset_cap_years <= 0.0 or self.risk_temperature <= 0.0:
            raise ValueError("risk offset cap and temperature must be positive")
        if self.survival_intervals <= 0:
            raise ValueError("survival_intervals must be positive")
        if not math.isfinite(self.baseline_log_rate_init):
            raise ValueError("baseline_log_rate_init must be finite")
        if self.base_age_mode not in BASE_AGE_MODES:
            raise ValueError(
                f"base_age_mode must be one of {BASE_AGE_MODES}, got {self.base_age_mode!r}"
            )
        if self.base_age_head_mode not in BASE_AGE_HEAD_MODES:
            raise ValueError(
                "base_age_head_mode must be one of "
                f"{BASE_AGE_HEAD_MODES}, got {self.base_age_head_mode!r}"
            )
        if (
            self.base_age_mode == BASE_AGE_MODE_THREE_VIEW_EQUAL_MEAN
            and self.base_age_head_mode != BASE_AGE_HEAD_MODE_LINEAR
        ):
            raise ValueError(
                "three_view_equal_mean requires base_age_head_mode='linear'"
            )
        if self.base_age_mode in (
            BASE_AGE_MODE_PRIVATE_RETAIN,
            BASE_AGE_MODE_SHARED_PRIVATE_EQUAL_MEAN,
        ) and self.base_age_head_mode != BASE_AGE_HEAD_MODE_MLP:
            raise ValueError(
                f"{self.base_age_mode} requires base_age_head_mode='mlp'"
            )
        if self.base_age_pooling_mode not in BASE_AGE_POOLING_MODES:
            raise ValueError(
                "base_age_pooling_mode must be one of "
                f"{BASE_AGE_POOLING_MODES}, got {self.base_age_pooling_mode!r}"
            )
        if (
            self.base_age_mode != BASE_AGE_MODE_V1
            and self.base_age_pooling_mode != BASE_AGE_POOLING_MODE_GLOBAL_FUSED
        ):
            raise ValueError(
                f"{self.base_age_mode} requires base_age_pooling_mode='global_fused'"
            )


@dataclass(frozen=True)
class EHRAGEOutput:
    predicted_age: torch.Tensor
    base_age: torch.Tensor
    risk_score: torch.Tensor
    risk_age_offset: torch.Tensor
    interval_log_rate: torch.Tensor
    base_view_ages: torch.Tensor | None = None


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class FrozenSemanticEventEncoder(nn.Module):
    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        hidden_dim: int,
        dropout: float,
        freeze_embeddings: bool,
    ) -> None:
        super().__init__()
        code = torch.as_tensor(code_embedding, dtype=torch.float32).detach().clone()
        text = torch.as_tensor(text_embedding, dtype=torch.float32).detach().clone()
        if code.ndim != 2 or text.ndim != 2 or code.shape[1] != text.shape[1]:
            raise ValueError("code and text embedding tables must be aligned matrices")
        if code.shape[0] < 1 or text.shape[0] < 1:
            raise ValueError("embedding tables must include padding row zero")
        code[0].zero_()
        text[0].zero_()
        self.code_embedding = nn.Embedding.from_pretrained(
            code, freeze=freeze_embeddings, padding_idx=0
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            text, freeze=freeze_embeddings, padding_idx=0
        )
        semantic_dim = int(code.shape[1])
        self.code_projection = nn.Linear(semantic_dim, hidden_dim, bias=False)
        self.text_projection = nn.Linear(semantic_dim, hidden_dim, bias=False)
        self.semantic_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.time_projection = nn.Linear(5, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        if event_code_id.ndim != 2 or any(value.shape != event_code_id.shape for value in tensors[1:]):
            raise ValueError("event inputs must share [batch,sequence] shape")
        valid = event_valid_mask.bool()
        if (valid & event_relative_days.ge(0)).any():
            raise ValueError("active event times must be strictly pre-index")
        if (valid & event_type_id.eq(2) & event_code_id.ne(0)).any():
            raise ValueError("GP slots must use code_id=0")

        code = self.code_projection(self.code_embedding(event_code_id.long()))
        text = self.text_projection(self.text_embedding(event_text_id.long()))
        semantic = self.semantic_fusion(torch.cat((code, text), dim=-1))
        days = (-event_relative_days).clamp_min(0).to(semantic.dtype)
        years = days / 365.25
        time_features = torch.stack(
            (
                torch.log1p(days) / math.log(36_526.0),
                torch.sqrt(years.clamp_min(0.0)) / 10.0,
                torch.exp(-years),
                torch.exp(-years / 5.0),
                torch.exp(-years / 20.0),
            ),
            dim=-1,
        )
        encoded = semantic + self.type_embedding(event_type_id.long())
        encoded = encoded + self.time_projection(time_features)
        encoded = self.dropout(self.normalization(torch.nn.functional.gelu(encoded)))
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)


class StaticEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(157, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
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
            raise ValueError("phenotype missingness must align with phenotype")
        if sex.ndim == 1:
            sex = sex.unsqueeze(1)
        if sex.shape != (len(phenotype), 1):
            raise ValueError("sex must have shape [batch] or [batch,1]")
        inputs = torch.cat(
            (
                phenotype.float(),
                phenotype_missing_mask.to(phenotype.dtype),
                sex.to(phenotype.dtype),
            ),
            dim=1,
        )
        return self.network(inputs)


class DualTemporalEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, transformer_layers: int, gru_layers: int, dropout: float) -> None:
        super().__init__()
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
            layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, 33, hidden_dim) * 0.01)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.sequence_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _packed_gru(tokens: torch.Tensor, mask: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("valid compact-cache events must form a left-aligned prefix")
        lengths = mask.sum(dim=1).long()
        result = tokens.new_zeros(tokens.shape)
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if len(nonempty):
            active = tokens.index_select(0, nonempty)
            active_lengths = lengths.index_select(0, nonempty)
            packed = pack_padded_sequence(
                active, active_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            encoded, _ = gru(packed)
            padded, _ = pad_packed_sequence(
                encoded, batch_first=True, total_length=tokens.shape[1]
            )
            result.index_copy_(0, nonempty, padded.to(dtype=result.dtype))
        return result * mask.unsqueeze(-1).to(result.dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_base_views: bool = False,
        return_transformer_events: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if tokens.shape[1] > 32:
            raise ValueError("primary EHR-AGE comparison is restricted to cache L32")
        cls = self.cls_token.expand(len(tokens), -1, -1)
        transformer_input = torch.cat((cls, tokens), dim=1)
        transformer_input = transformer_input + self.position_embedding[:, : transformer_input.shape[1]]
        padding = torch.cat(
            (
                torch.zeros((len(tokens), 1), dtype=torch.bool, device=tokens.device),
                ~mask,
            ),
            dim=1,
        )
        transformed = self.transformer(transformer_input, src_key_padding_mask=padding)
        transformer_events = transformed[:, 1:] * mask.unsqueeze(-1).to(transformed.dtype)
        gru_events = self._packed_gru(tokens, mask, self.gru)
        sequence = self.sequence_fusion(torch.cat((transformer_events, gru_events), dim=-1))
        sequence = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        transformer_cls = transformed[:, 0]
        gru_pooled = _masked_mean(gru_events, mask)
        pooled = self.global_fusion(
            torch.cat((transformer_cls, gru_pooled), dim=-1)
        )
        if return_base_views:
            return sequence, pooled, transformer_cls, gru_pooled
        if return_transformer_events:
            return sequence, pooled, transformer_events
        return sequence, pooled


class EHRAGEModel(nn.Module):
    """From-scratch EHR-AGE with a shared age base and prognostic organ offset."""

    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        age_anchor: float,
        config: EHRAGEConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EHRAGEConfig()
        self.config.validate()
        hidden = self.config.hidden_dim
        self.event_encoder = FrozenSemanticEventEncoder(
            code_embedding,
            text_embedding,
            hidden_dim=hidden,
            dropout=self.config.dropout,
            freeze_embeddings=self.config.freeze_event_embeddings,
        )
        self.temporal_encoder = DualTemporalEncoder(
            hidden,
            self.config.num_heads,
            self.config.num_transformer_layers,
            self.config.num_gru_layers,
            self.config.dropout,
        )
        self.static_encoder = StaticEncoder(hidden, self.config.dropout)

        def make_base_fusion() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )

        if self.config.base_age_mode == BASE_AGE_MODE_V1:
            # Preserve the original module names so old v1 checkpoints load strictly.
            self.base_fusion = make_base_fusion()
            if self.config.base_age_head_mode == BASE_AGE_HEAD_MODE_LINEAR:
                self.base_age_head = nn.Linear(hidden, 1)
                terminal_age_layer = self.base_age_head
            else:
                self.base_age_head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(hidden, 1),
                )
                terminal_age_layer = self.base_age_head[-1]
            nn.init.zeros_(terminal_age_layer.weight)
            nn.init.zeros_(terminal_age_layer.bias)
        elif self.config.base_age_mode == BASE_AGE_MODE_THREE_VIEW_EQUAL_MEAN:
            self.base_view_fusions = nn.ModuleDict(
                {name: make_base_fusion() for name in BASE_AGE_VIEW_NAMES}
            )
            self.base_view_age_heads = nn.ModuleDict(
                {name: nn.Linear(hidden, 1) for name in BASE_AGE_VIEW_NAMES}
            )
            for head in self.base_view_age_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        elif self.config.base_age_mode == BASE_AGE_MODE_SHARED_PRIVATE_EQUAL_MEAN:
            # Build the shared branch from the private tower's formal shadow RNG
            # state, while leaving the global stream unchanged for the risk branch.
            entry_rng_state = torch.random.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(entry_rng_state)
                self.base_fusion = make_base_fusion()
                self.base_age_head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(hidden, 1),
                )
                nn.init.zeros_(self.base_age_head[-1].weight)
                nn.init.zeros_(self.base_age_head[-1].bias)
            self.private_retain_age_tower = PrivateRETAINAgeTower(
                code_embedding,
                text_embedding,
                age_anchor=age_anchor,
                dropout=self.config.dropout,
                freeze_embeddings=self.config.freeze_event_embeddings,
            )
        else:
            self.private_retain_age_tower = PrivateRETAINAgeTower(
                code_embedding,
                text_embedding,
                age_anchor=age_anchor,
                dropout=self.config.dropout,
                freeze_embeddings=self.config.freeze_event_embeddings,
            )

        self.organ_queries = nn.Parameter(torch.randn(len(ORGAN_SYSTEMS), hidden) * 0.02)
        self.organ_cross_attention = nn.MultiheadAttention(
            hidden, self.config.num_heads, dropout=self.config.dropout, batch_first=True
        )
        self.organ_fusion = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        relation_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.config.num_heads,
            dim_feedforward=2 * hidden,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.organ_relation = nn.TransformerEncoder(
            relation_layer, num_layers=1, enable_nested_tensor=False
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.zeros_(self.risk_head[-1].bias)
        self.baseline_log_rate = nn.Parameter(
            torch.full(
                (len(ORGAN_SYSTEMS), self.config.survival_intervals),
                float(self.config.baseline_log_rate_init),
            )
        )
        self.register_buffer("age_anchor", torch.tensor(float(age_anchor), dtype=torch.float32))

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
        mask = event_valid_mask.bool()
        tokens = self.event_encoder(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            mask,
        )
        if self.config.base_age_mode == BASE_AGE_MODE_THREE_VIEW_EQUAL_MEAN:
            sequence, pooled, transformer_cls, gru_pooled = self.temporal_encoder(
                tokens, mask, return_base_views=True
            )
        elif (
            self.config.base_age_pooling_mode
            == BASE_AGE_POOLING_MODE_TRANSFORMER_EVENT_MEAN
        ):
            sequence, pooled, transformer_events = self.temporal_encoder(
                tokens, mask, return_transformer_events=True
            )
        else:
            sequence, pooled = self.temporal_encoder(tokens, mask)
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
        base_view_ages = None
        if self.config.base_age_mode == BASE_AGE_MODE_V1:
            base_temporal = pooled
            if (
                self.config.base_age_pooling_mode
                == BASE_AGE_POOLING_MODE_TRANSFORMER_EVENT_MEAN
            ):
                base_temporal = _masked_mean(transformer_events, mask)
            base_context = self.base_fusion(
                torch.cat((base_temporal, static), dim=-1)
            )
            base_residual = self.base_age_head(base_context).squeeze(-1)
            base_age = self.age_anchor.to(base_residual.dtype) + base_residual
        elif self.config.base_age_mode == BASE_AGE_MODE_THREE_VIEW_EQUAL_MEAN:
            base_views = {
                "transformer_cls": transformer_cls,
                "gru_pooled": gru_pooled,
                "semantic_mean": _masked_mean(tokens, mask),
            }
            view_ages = []
            for name in BASE_AGE_VIEW_NAMES:
                context = self.base_view_fusions[name](
                    torch.cat((base_views[name], static), dim=-1)
                )
                residual = self.base_view_age_heads[name](context).squeeze(-1)
                view_ages.append(self.age_anchor.to(residual.dtype) + residual)
            base_view_ages = torch.stack(view_ages, dim=1)
            base_age = base_view_ages.mean(dim=1)
        elif self.config.base_age_mode == BASE_AGE_MODE_SHARED_PRIVATE_EQUAL_MEAN:
            shared_context = self.base_fusion(torch.cat((pooled, static), dim=-1))
            shared_residual = self.base_age_head(shared_context).squeeze(-1)
            shared_age = self.age_anchor.to(shared_residual.dtype) + shared_residual
            private_age = self.private_retain_age_tower(
                phenotype,
                phenotype_missing_mask,
                sex,
                event_code_id,
                event_text_id,
                event_relative_days,
                event_type_id,
                mask,
            )
            base_view_ages = torch.stack((shared_age, private_age), dim=1)
            base_age = base_view_ages.mean(dim=1)
        else:
            base_age = self.private_retain_age_tower(
                phenotype,
                phenotype_missing_mask,
                sex,
                event_code_id,
                event_text_id,
                event_relative_days,
                event_type_id,
                mask,
            )

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
        organ = self.organ_relation(organ)
        risk_score = self.risk_head(organ).squeeze(-1)
        offset = self.config.risk_offset_cap_years * torch.tanh(
            risk_score / self.config.risk_temperature
        )
        predicted_age = base_age.unsqueeze(1) + offset
        interval_log_rate = self.baseline_log_rate.unsqueeze(0) + risk_score.unsqueeze(-1)
        return EHRAGEOutput(
            predicted_age=predicted_age,
            base_age=base_age,
            risk_score=risk_score,
            risk_age_offset=offset,
            interval_log_rate=interval_log_rate,
            base_view_ages=base_view_ages,
        )
