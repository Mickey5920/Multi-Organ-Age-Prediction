from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


ORGAN_SYSTEMS = (
    "cardiovascular",
    "hepatic",
    "immune",
    "metabolic",
    "musculoskeletal",
    "pulmonary",
    "renal",
)

PRIVATE_RETAIN_HIDDEN_DIM = 128
FORMAL_COMMON_RISK_HIDDEN_DIM = 192
PRIVATE_RETAIN_RNG_NAMESPACE = b"revision0710.d07.private_retain.initialization.v1"


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
    max_events: int = 32
    base_age_mode: str = "shared_private_equal_mean"
    base_age_head_mode: str = "mlp"
    base_age_pooling_mode: str = "global_fused"

    def validate(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even")
        if self.num_transformer_layers <= 0 or self.num_gru_layers <= 0:
            raise ValueError("encoder layer counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.risk_offset_cap_years <= 0.0 or self.risk_temperature <= 0.0:
            raise ValueError("risk offset parameters must be positive")
        if self.survival_intervals <= 0 or self.max_events != 32:
            raise ValueError("the frozen model uses five intervals and 32 events")
        if self.base_age_mode != "shared_private_equal_mean":
            raise ValueError("the frozen model uses shared_private_equal_mean")
        if self.base_age_head_mode != "mlp" or self.base_age_pooling_mode != "global_fused":
            raise ValueError("the frozen base-age head configuration is fixed")


@dataclass(frozen=True)
class EHRAGEOutput:
    predicted_age: torch.Tensor
    base_age: torch.Tensor
    risk_score: torch.Tensor
    risk_age_offset: torch.Tensor
    interval_log_rate: torch.Tensor
    base_view_ages: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype).unsqueeze(-1)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~valid_mask, -torch.inf)
    empty = ~valid_mask.any(dim=1)
    if empty.any():
        masked = masked.clone()
        masked[empty] = 0.0
    probabilities = torch.softmax(masked, dim=1)
    return probabilities * valid_mask.to(probabilities.dtype)


def _derived_private_seed(entry_rng_state: torch.Tensor) -> int:
    payload = entry_rng_state.detach().cpu().contiguous().numpy().tobytes()
    digest = hashlib.sha256(PRIVATE_RETAIN_RNG_NAMESPACE + payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _formal_shared_mlp_post_init_rng_state(dropout: float) -> torch.Tensor:
    hidden = FORMAL_COMMON_RISK_HIDDEN_DIM
    fusion = nn.Sequential(
        nn.Linear(2 * hidden, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
    )
    head = nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    state = torch.random.get_rng_state().clone()
    del fusion, head
    return state


class FrozenSemanticEventEncoder(nn.Module):
    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        hidden_dim: int,
        dropout: float,
        freeze_embeddings: bool,
    ) -> None:
        super().__init__()
        code = torch.as_tensor(code_embedding, dtype=torch.float32).detach().clone()
        text = torch.as_tensor(text_embedding, dtype=torch.float32).detach().clone()
        if code.ndim != 2 or text.ndim != 2 or code.shape[1] != text.shape[1]:
            raise ValueError("embedding tables must be aligned matrices")
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
        values = (
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        if event_code_id.ndim != 2 or any(value.shape != event_code_id.shape for value in values[1:]):
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
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        transformer_layers: int,
        gru_layers: int,
        dropout: float,
    ) -> None:
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
            raise ValueError("valid events must form a left-aligned prefix")
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
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[1] > 32:
            raise ValueError("EHR-AGE is restricted to 32 retained events")
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
        pooled = self.global_fusion(torch.cat((transformer_cls, gru_pooled), dim=-1))
        return sequence, pooled


class PrivateRETAINEventTokenEncoder(nn.Module):
    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        hidden_dim: int = PRIVATE_RETAIN_HIDDEN_DIM,
        type_vocab_size: int = 3,
        dropout: float = 0.1,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        code = torch.as_tensor(code_embedding, dtype=torch.float32).detach().clone()
        text = torch.as_tensor(text_embedding, dtype=torch.float32).detach().clone()
        if code.ndim != 2 or text.ndim != 2 or code.shape[1] != text.shape[1]:
            raise ValueError("embedding tables must be aligned matrices")
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
        embedding_dim = int(code.shape[1])
        self.code_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.text_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.semantic_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(type_vocab_size, hidden_dim, padding_idx=0)
        self.time_projection = nn.Linear(3, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = int(hidden_dim)

    def forward(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = (
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        if event_code_id.ndim != 2 or any(value.shape != event_code_id.shape for value in values[1:]):
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
                years / 100.0,
                torch.exp(-years / 10.0),
            ),
            dim=-1,
        )
        encoded = semantic + self.type_embedding(event_type_id.long())
        encoded = encoded + self.time_projection(time_features)
        encoded = self.dropout(self.normalization(torch.nn.functional.gelu(encoded)))
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)


class PrivateRETAINBackbone(nn.Module):
    def __init__(
        self,
        token_encoder: PrivateRETAINEventTokenEncoder,
        hidden_dim: int = PRIVATE_RETAIN_HIDDEN_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if token_encoder.output_dim != hidden_dim:
            raise ValueError("token and RETAIN hidden dimensions must agree")
        self.token_encoder = token_encoder
        self.output_dim = int(hidden_dim)
        self.alpha_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.alpha_projection = nn.Linear(hidden_dim, 1)
        self.beta_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _reverse_compact(
        tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = valid_mask.sum(dim=1).long()
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        compact_mask = positions < lengths.unsqueeze(1)
        reverse_index = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
        reversed_tokens = tokens.gather(
            1, reverse_index.unsqueeze(-1).expand(-1, -1, tokens.shape[2])
        )
        reversed_tokens = reversed_tokens * compact_mask.unsqueeze(-1).to(tokens.dtype)
        return reversed_tokens, lengths

    def forward(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = event_valid_mask.bool()
        tokens = self.token_encoder(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            valid,
        )
        reversed_tokens, lengths = self._reverse_compact(tokens, valid)
        pooled = tokens.new_zeros((tokens.shape[0], self.output_dim))
        nonempty = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if len(nonempty):
            active_tokens = reversed_tokens.index_select(0, nonempty)
            active_lengths = lengths.index_select(0, nonempty)
            packed = pack_padded_sequence(
                active_tokens,
                active_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            alpha_packed, _ = self.alpha_gru(packed)
            beta_packed, _ = self.beta_gru(packed)
            alpha_hidden, _ = pad_packed_sequence(
                alpha_packed, batch_first=True, total_length=tokens.shape[1]
            )
            beta_hidden, _ = pad_packed_sequence(
                beta_packed, batch_first=True, total_length=tokens.shape[1]
            )
            compact_mask = (
                torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
                < active_lengths.unsqueeze(1)
            )
            alpha = _masked_softmax(
                self.alpha_projection(alpha_hidden).squeeze(-1), compact_mask
            )
            beta = torch.tanh(self.beta_projection(beta_hidden))
            active_pooled = (alpha.unsqueeze(-1) * beta * active_tokens).sum(dim=1)
            active_pooled = self.dropout(self.output_normalization(active_pooled))
            pooled.index_copy_(0, nonempty, active_pooled.to(dtype=pooled.dtype))
        return pooled


class PrivateRETAINStaticEncoder(nn.Module):
    def __init__(self, hidden_dim: int = PRIVATE_RETAIN_HIDDEN_DIM, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(157, hidden_dim),
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
            raise ValueError("phenotype missingness must align with phenotype")
        if sex.ndim == 1:
            sex = sex.unsqueeze(1)
        if sex.shape != (len(phenotype), 1):
            raise ValueError("sex must have shape [batch] or [batch,1]")
        static = torch.cat(
            (
                phenotype.float(),
                phenotype_missing_mask.to(phenotype.dtype),
                sex.to(phenotype.dtype),
            ),
            dim=1,
        )
        return self.network(static)


class PrivateRETAINAgeTower(nn.Module):
    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        age_anchor: float,
        dropout: float,
        freeze_embeddings: bool,
    ) -> None:
        super().__init__()
        hidden = PRIVATE_RETAIN_HIDDEN_DIM
        entry_rng_state = torch.random.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            reference_post_init_state = _formal_shared_mlp_post_init_rng_state(dropout)
        private_seed = _derived_private_seed(entry_rng_state)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(private_seed)
            token_encoder = PrivateRETAINEventTokenEncoder(
                code_embedding,
                text_embedding,
                hidden_dim=hidden,
                type_vocab_size=3,
                dropout=dropout,
                freeze_embeddings=freeze_embeddings,
            )
            self.backbone = PrivateRETAINBackbone(
                token_encoder, hidden_dim=hidden, dropout=dropout
            )
            self.static_encoder = PrivateRETAINStaticEncoder(hidden, dropout)
            self.fusion = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.age_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.age_head[-1].weight)
            nn.init.zeros_(self.age_head[-1].bias)
        torch.random.set_rng_state(reference_post_init_state)
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
    ) -> torch.Tensor:
        temporal = self.backbone(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
        patient = self.fusion(torch.cat((temporal, static), dim=1))
        residual = self.age_head(patient).squeeze(1)
        return self.age_anchor.to(residual.dtype) + residual


class EHRAGEModel(nn.Module):
    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
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
            hidden,
            self.config.dropout,
            self.config.freeze_event_embeddings,
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
            age_anchor,
            self.config.dropout,
            self.config.freeze_event_embeddings,
        )
        self.organ_queries = nn.Parameter(torch.randn(len(ORGAN_SYSTEMS), hidden) * 0.02)
        self.organ_cross_attention = nn.MultiheadAttention(
            hidden,
            self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
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
        sequence, pooled = self.temporal_encoder(tokens, mask)
        static = self.static_encoder(phenotype, phenotype_missing_mask, sex)
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
