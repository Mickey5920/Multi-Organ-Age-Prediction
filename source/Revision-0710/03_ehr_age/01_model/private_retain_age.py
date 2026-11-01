from __future__ import annotations

"""Private RETAIN age tower derived from the Revision-0710 plain baseline.

The event encoder, reverse-time alpha/beta attention, and static encoder mirror
``02_baselines_5fold/common/{masked_backbones,baseline_models}.py``.  They live
here because those revision-local baseline modules rely on top-level imports
and cannot be imported reliably through both supported EHR-AGE import paths.
No baseline checkpoint or prediction is consumed.
"""

import hashlib
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


PRIVATE_RETAIN_HIDDEN_DIM = 128
FORMAL_COMMON_RISK_HIDDEN_DIM = 192
PRIVATE_RETAIN_RNG_NAMESPACE = b"revision0710.d07.private_retain.initialization.v1"


def _derived_private_seed(entry_rng_state: torch.Tensor) -> int:
    digest = hashlib.sha256(
        PRIVATE_RETAIN_RNG_NAMESPACE
        + entry_rng_state.detach().cpu().contiguous().numpy().tobytes()
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _formal_shared_mlp_post_init_rng_state(dropout: float) -> torch.Tensor:
    """Return the D07-002 post-base-branch CPU RNG state without leaking modules."""

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


def _masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~valid_mask, -torch.inf)
    empty = ~valid_mask.any(dim=1)
    if empty.any():
        masked = masked.clone()
        masked[empty] = 0.0
    probabilities = torch.softmax(masked, dim=1)
    return probabilities * valid_mask.to(probabilities.dtype)


class PrivateRETAINEventTokenEncoder(nn.Module):
    """Baseline-lineage code/text/type encoder with three pre-index time features."""

    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
        hidden_dim: int = PRIVATE_RETAIN_HIDDEN_DIM,
        type_vocab_size: int = 3,
        dropout: float = 0.1,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        code = torch.as_tensor(code_embedding, dtype=torch.float32).detach().clone()
        text = torch.as_tensor(text_embedding, dtype=torch.float32).detach().clone()
        if code.ndim != 2 or text.ndim != 2:
            raise ValueError("code and text embedding tables must be two-dimensional")
        if code.shape[1] != text.shape[1]:
            raise ValueError("code and text embedding dimensions must agree")
        if code.shape[0] < 1 or text.shape[0] < 1:
            raise ValueError("embedding tables must contain the padding row")
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
        shape = values[0].shape
        if len(shape) != 2 or any(value.shape != shape for value in values[1:]):
            raise ValueError("all event arrays must have the same [batch,sequence] shape")
        valid = event_valid_mask.bool()
        if (valid & event_relative_days.ge(0)).any():
            raise ValueError("active event times must be strictly pre-index")
        if (valid & event_type_id.eq(2) & event_code_id.ne(0)).any():
            raise ValueError("GP prescription slots must use code_id=0")

        code = self.code_projection(self.code_embedding(event_code_id.long()))
        text = self.text_projection(self.text_embedding(event_text_id.long()))
        semantic = self.semantic_fusion(torch.cat((code, text), dim=-1))
        days_before = (-event_relative_days).clamp_min(0).to(dtype=semantic.dtype)
        years_before = days_before / 365.25
        time_features = torch.stack(
            (
                torch.log1p(days_before) / math.log(36_526.0),
                years_before / 100.0,
                torch.exp(-years_before / 10.0),
            ),
            dim=-1,
        )
        encoded = semantic + self.type_embedding(event_type_id.long())
        encoded = encoded + self.time_projection(time_features)
        encoded = self.dropout(self.normalization(torch.nn.functional.gelu(encoded)))
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)


class PrivateRETAINBackbone(nn.Module):
    """Baseline-lineage reverse-time RETAIN alpha/beta attention."""

    def __init__(
        self,
        token_encoder: PrivateRETAINEventTokenEncoder,
        *,
        hidden_dim: int = PRIVATE_RETAIN_HIDDEN_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if token_encoder.output_dim != hidden_dim:
            raise ValueError("token-encoder and RETAIN hidden dimensions must agree")
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
            contributions = beta * active_tokens
            active_pooled = (alpha.unsqueeze(-1) * contributions).sum(dim=1)
            active_pooled = self.dropout(self.output_normalization(active_pooled))
            pooled.index_copy_(0, nonempty, active_pooled.to(dtype=pooled.dtype))
        return pooled


class PrivateRETAINStaticEncoder(nn.Module):
    """Exact H128 static layout used by the Revision-0710 baselines."""

    def __init__(
        self, hidden_dim: int = PRIVATE_RETAIN_HIDDEN_DIM, dropout: float = 0.1
    ) -> None:
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


class PrivateRETAINAgeTower(nn.Module):
    """Independent H128 RETAIN patient encoder with one shared scalar age head."""

    def __init__(
        self,
        code_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        *,
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
