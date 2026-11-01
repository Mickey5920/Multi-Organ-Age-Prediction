from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


EVENT_INPUT_KEYS = (
    "event_code_id",
    "event_text_id",
    "event_relative_days",
    "event_type_id",
    "event_valid_mask",
)

EHRFORMER_REFERENCE_REPOSITORY = "https://github.com/kaiwang13/EHRFormer"
EHRFORMER_REFERENCE_COMMIT = "e49bd6b610eed14ff5abad23cb2d475be043b6ac"


def event_inputs_from_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = set(EVENT_INPUT_KEYS).difference(batch)
    if missing:
        raise KeyError(f"batch is missing event inputs: {sorted(missing)}")
    return {key: batch[key] for key in EVENT_INPUT_KEYS}


def masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
        raise ValueError("masked_mean expects values [batch,sequence,hidden] and mask [batch,sequence]")
    weight = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    numerator = (values * weight).sum(dim=1)
    denominator = weight.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _last_valid(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
        raise ValueError("_last_valid expects values [batch,sequence,hidden] and mask [batch,sequence]")
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    last_positions = positions.masked_fill(~valid_mask, -1).max(dim=1).values
    safe_positions = last_positions.clamp_min(0)
    pooled = values.gather(
        1, safe_positions.view(-1, 1, 1).expand(-1, 1, values.shape[2])
    ).squeeze(1)
    return pooled * last_positions.ge(0).unsqueeze(1).to(values.dtype)


def _masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~valid_mask, -torch.inf)
    empty = ~valid_mask.any(dim=1)
    if empty.any():
        masked = masked.clone()
        masked[empty] = 0.0
    probabilities = torch.softmax(masked, dim=1)
    return probabilities * valid_mask.to(probabilities.dtype)


def _sinusoidal_positions(length: int, dimension: int, reference: torch.Tensor) -> torch.Tensor:
    position = torch.arange(length, device=reference.device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=reference.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(dimension, 1))
    )
    encoding = torch.zeros((length, dimension), device=reference.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=reference.dtype)


@dataclass(frozen=True)
class BackboneOutput:
    pooled: torch.Tensor
    sequence: torch.Tensor
    valid_mask: torch.Tensor


class EventTokenEncoder(nn.Module):
    """Fuse frozen code/name embeddings with event type and pre-index time."""

    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        hidden_dim: int,
        type_vocab_size: int = 3,
        dropout: float = 0.1,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        code_embedding = torch.as_tensor(code_embedding, dtype=torch.float32).detach().clone()
        text_embedding = torch.as_tensor(text_embedding, dtype=torch.float32).detach().clone()
        if code_embedding.ndim != 2 or text_embedding.ndim != 2:
            raise ValueError("code and text embedding tables must be two-dimensional")
        if code_embedding.shape[1] != text_embedding.shape[1]:
            raise ValueError("code and text embedding dimensions must agree")
        if code_embedding.shape[0] < 1 or text_embedding.shape[0] < 1:
            raise ValueError("embedding tables must contain the padding row")
        code_embedding[0].zero_()
        text_embedding[0].zero_()
        self.code_embedding = nn.Embedding.from_pretrained(
            code_embedding, freeze=freeze_embeddings, padding_idx=0
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            text_embedding, freeze=freeze_embeddings, padding_idx=0
        )
        embedding_dim = int(code_embedding.shape[1])
        self.code_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.text_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.semantic_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.type_embedding = nn.Embedding(type_vocab_size, hidden_dim, padding_idx=0)
        self.time_projection = nn.Linear(3, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = int(hidden_dim)

    @staticmethod
    def _validate_shapes(values: tuple[torch.Tensor, ...]) -> None:
        shape = values[0].shape
        if len(shape) != 2 or any(value.shape != shape for value in values[1:]):
            raise ValueError("all event arrays must have the same [batch,sequence] shape")

    def forward(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        code, text, event_type, event_time, valid_mask = self._component_tensors(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        semantic = self.semantic_fusion(torch.cat((code, text), dim=-1))
        encoded = semantic + event_type + event_time
        encoded = self.dropout(self.normalization(torch.nn.functional.gelu(encoded)))
        return encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)

    def _component_tensors(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_shapes(
            (event_code_id, event_text_id, event_relative_days, event_type_id, event_valid_mask)
        )
        valid_mask = event_valid_mask.bool()
        gp_with_code = valid_mask & event_type_id.eq(2) & event_code_id.ne(0)
        if gp_with_code.any():
            raise ValueError("GP prescription slots must use code_id=0")
        code = self.code_projection(self.code_embedding(event_code_id.long()))
        text = self.text_projection(self.text_embedding(event_text_id.long()))
        days_before = (-event_relative_days).clamp_min(0).to(dtype=code.dtype)
        years_before = days_before / 365.25
        time_features = torch.stack(
            (
                torch.log1p(days_before) / math.log(36_526.0),
                years_before / 100.0,
                torch.exp(-years_before / 10.0),
            ),
            dim=-1,
        )
        return (
            code,
            text,
            self.type_embedding(event_type_id.long()),
            self.time_projection(time_features),
            valid_mask,
        )

    def component_tokens(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return code, text, event-type, and time tokens for an examination encoder."""
        code, text, event_type, event_time, valid_mask = self._component_tensors(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        component_mask = torch.stack(
            (
                valid_mask & event_code_id.ne(0),
                valid_mask & event_text_id.ne(0),
                valid_mask & event_type_id.ne(0),
                valid_mask,
            ),
            dim=2,
        )
        components = torch.stack((code, text, event_type, event_time), dim=2)
        components = components * component_mask.unsqueeze(-1).to(components.dtype)
        return components, component_mask


class _EventBackbone(nn.Module):
    def __init__(self, token_encoder: EventTokenEncoder, hidden_dim: int) -> None:
        super().__init__()
        if token_encoder.output_dim != hidden_dim:
            raise ValueError("token-encoder and backbone hidden dimensions must agree")
        self.token_encoder = token_encoder
        self.output_dim = int(hidden_dim)

    def forward(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        ).pooled

    def encode(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> BackboneOutput:
        raise NotImplementedError


class RETAINBackbone(_EventBackbone):
    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(token_encoder, hidden_dim)
        self.alpha_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.alpha_projection = nn.Linear(hidden_dim, 1)
        self.beta_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _reverse_compact(tokens: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = valid_mask.sum(dim=1).long()
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        compact_mask = positions < lengths.unsqueeze(1)
        reverse_index = (lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
        reversed_tokens = tokens.gather(
            1, reverse_index.unsqueeze(-1).expand(-1, -1, tokens.shape[2])
        )
        reversed_tokens = reversed_tokens * compact_mask.unsqueeze(-1).to(tokens.dtype)
        return reversed_tokens, lengths

    def encode(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> BackboneOutput:
        mask = event_valid_mask.bool()
        tokens = self.token_encoder(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            mask,
        )
        reversed_tokens, lengths = self._reverse_compact(tokens, mask)
        pooled = tokens.new_zeros((tokens.shape[0], self.output_dim))
        contextual = tokens.new_zeros(tokens.shape)
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
            alpha = _masked_softmax(self.alpha_projection(alpha_hidden).squeeze(-1), compact_mask)
            beta = torch.tanh(self.beta_projection(beta_hidden))
            contributions = beta * active_tokens
            active_pooled = (alpha.unsqueeze(-1) * contributions).sum(dim=1)
            active_pooled = self.dropout(self.output_normalization(active_pooled))
            pooled[nonempty] = active_pooled
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            reverse_index = (active_lengths.unsqueeze(1) - 1 - positions).clamp_min(0)
            chronological = contributions.gather(
                1,
                reverse_index.unsqueeze(-1).expand(-1, -1, contributions.shape[2]),
            )
            chronological = chronological * compact_mask.unsqueeze(-1).to(chronological.dtype)
            contextual.index_copy_(0, nonempty, chronological)
        return BackboneOutput(pooled=pooled, sequence=contextual, valid_mask=mask)


class BEHRTStyleBackbone(_EventBackbone):
    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(token_encoder, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.output_normalization = nn.LayerNorm(hidden_dim)

    def encode(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> BackboneOutput:
        mask = event_valid_mask.bool()
        tokens = self.token_encoder(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            mask,
        )
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        sequence = torch.cat((cls, tokens), dim=1)
        sequence = sequence + _sinusoidal_positions(
            sequence.shape[1], sequence.shape[2], sequence
        ).unsqueeze(0)
        padding_mask = torch.cat(
            (
                torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device),
                ~mask,
            ),
            dim=1,
        )
        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)
        pooled = self.output_normalization(encoded[:, 0])
        event_sequence = encoded[:, 1:] * mask.unsqueeze(-1).to(encoded.dtype)
        return BackboneOutput(pooled=pooled, sequence=event_sequence, valid_mask=mask)


class MedBERTStyleBackbone(_EventBackbone):
    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(token_encoder, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_normalization = nn.LayerNorm(hidden_dim)

    def encode(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> BackboneOutput:
        mask = event_valid_mask.bool()
        tokens = self.token_encoder(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            mask,
        )
        if tokens.shape[1] == 0:
            raise ValueError("MedBERT-style backbone requires sequence length >= 1")
        tokens = tokens + _sinusoidal_positions(
            tokens.shape[1], tokens.shape[2], tokens
        ).unsqueeze(0)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            tokens = tokens.clone()
            tokens[empty, 0] = 0.0
        encoded = self.transformer(tokens, src_key_padding_mask=~safe_mask)
        encoded = self.output_normalization(encoded)
        event_sequence = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        pooled = masked_mean(event_sequence, mask)
        return BackboneOutput(pooled=pooled, sequence=event_sequence, valid_mask=mask)


class EHRFormerStyleBackbone(_EventBackbone):
    """Controlled EHRFormer architecture over the common pre-index event inputs.

    The public EHRFormer source encodes features within each examination, applies
    an autoregressive temporal Transformer, and passes the resulting sequence
    through variational encoders and a latent decoder. Here each dated EHR event
    is treated as one examination with four common-support features: code,
    description, event type, and relative time. Model width and depth remain
    controlled by the shared baseline configuration.
    """

    def __init__(
        self,
        token_encoder: EventTokenEncoder,
        *,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        latent_layers: int = 2,
    ) -> None:
        super().__init__(token_encoder, hidden_dim)
        if num_layers <= 0 or latent_layers <= 0:
            raise ValueError("EHRFormer-style layer counts must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        def encoder(layer_count: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            return nn.TransformerEncoder(
                layer, num_layers=layer_count, enable_nested_tensor=False
            )

        self.examination_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.feature_type_embedding = nn.Embedding(5, hidden_dim)
        self.examination_encoder = encoder(num_layers)
        self.examination_normalization = nn.LayerNorm(hidden_dim)
        self.temporal_decoder = encoder(num_layers)
        self.mu_encoder = encoder(latent_layers)
        self.logvar_encoder = encoder(latent_layers)
        self.latent_decoder = encoder(latent_layers)
        self.output_normalization = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.examination_cls, std=0.02)
        nn.init.normal_(self.feature_type_embedding.weight, std=0.02)

    def _examination_embeddings(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        components, component_mask = self.token_encoder.component_tokens(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            event_valid_mask,
        )
        batch_size, sequence_length, component_count, hidden_dim = components.shape
        flattened = components.reshape(-1, component_count, hidden_dim)
        flattened_mask = component_mask.reshape(-1, component_count)
        active = torch.nonzero(event_valid_mask.reshape(-1).bool(), as_tuple=False).flatten()
        examinations = components.new_zeros((batch_size * sequence_length, hidden_dim))
        if len(active):
            active_components = flattened.index_select(0, active)
            active_mask = flattened_mask.index_select(0, active)
            cls = self.examination_cls.expand(len(active), -1, -1)
            feature_sequence = torch.cat((cls, active_components), dim=1)
            feature_ids = torch.arange(
                component_count + 1, device=components.device, dtype=torch.long
            )
            feature_sequence = feature_sequence + self.feature_type_embedding(feature_ids).unsqueeze(0)
            padding_mask = torch.cat(
                (
                    torch.zeros((len(active), 1), dtype=torch.bool, device=components.device),
                    ~active_mask,
                ),
                dim=1,
            )
            encoded = self.examination_encoder(
                feature_sequence, src_key_padding_mask=padding_mask
            )
            pooled = self.examination_normalization(encoded[:, 0])
            examinations.index_copy_(0, active, pooled)
        return examinations.reshape(batch_size, sequence_length, hidden_dim)

    def encode(
        self,
        event_code_id: torch.Tensor,
        event_text_id: torch.Tensor,
        event_relative_days: torch.Tensor,
        event_type_id: torch.Tensor,
        event_valid_mask: torch.Tensor,
    ) -> BackboneOutput:
        mask = event_valid_mask.bool()
        if mask.ndim != 2 or mask.shape[1] == 0:
            raise ValueError("EHRFormer-style backbone requires a non-empty sequence dimension")
        examinations = self._examination_embeddings(
            event_code_id,
            event_text_id,
            event_relative_days,
            event_type_id,
            mask,
        )
        examinations = examinations + _sinusoidal_positions(
            examinations.shape[1], examinations.shape[2], examinations
        ).unsqueeze(0)
        examinations = examinations * mask.unsqueeze(-1).to(examinations.dtype)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            examinations = examinations.clone()
            examinations[empty, 0] = 0.0
        causal_mask = torch.triu(
            torch.ones(
                examinations.shape[1],
                examinations.shape[1],
                dtype=torch.bool,
                device=examinations.device,
            ),
            diagonal=1,
        )
        temporal = self.temporal_decoder(
            examinations,
            mask=causal_mask,
            src_key_padding_mask=~safe_mask,
        )
        temporal = temporal * safe_mask.unsqueeze(-1).to(temporal.dtype)
        mu = self.mu_encoder(temporal, src_key_padding_mask=~safe_mask)
        logvar = self.logvar_encoder(temporal, src_key_padding_mask=~safe_mask)
        if self.training:
            standard_deviation = torch.exp(0.5 * logvar.clamp(min=-10.0, max=10.0))
            latent = mu + torch.randn_like(standard_deviation) * standard_deviation
        else:
            latent = mu
        decoded = self.latent_decoder(latent, src_key_padding_mask=~safe_mask)
        decoded = self.output_normalization(decoded)
        event_sequence = decoded * mask.unsqueeze(-1).to(decoded.dtype)
        pooled = _last_valid(event_sequence, mask)
        return BackboneOutput(pooled=pooled, sequence=event_sequence, valid_mask=mask)
