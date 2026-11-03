from __future__ import annotations

import torch

from torch import nn

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
