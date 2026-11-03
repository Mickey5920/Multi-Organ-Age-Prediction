from __future__ import annotations

import math

import torch

def _scheduler(
    optimizer: torch.optim.Optimizer, *, warmup_epochs: int, horizon_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(index: int) -> float:
        if index < warmup_epochs:
            return float(index + 1) / max(warmup_epochs, 1)
        progress = (index - warmup_epochs) / max(horizon_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
