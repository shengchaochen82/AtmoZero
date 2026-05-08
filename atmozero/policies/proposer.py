"""Proposer policy pi_P over (window, regime) selections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ProposerConfig:
    n_windows: int = 8192
    n_regimes: int = 16
    entropy_bonus: float = 0.05
    infeasibility_penalty: float = 1.0
    softmax_temperature: float = 1.0
    init_uniform: bool = True


class Proposer(nn.Module):
    def __init__(self, cfg: ProposerConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.n_windows * cfg.n_regimes
        init = torch.zeros(n) if cfg.init_uniform else torch.randn(n) * 0.01
        # Held to a uniform prior until cold-start engages requires_grad.
        self.logits = nn.Parameter(init, requires_grad=False)

    def engage(self) -> None:
        self.logits.requires_grad_(True)

    def forward(self) -> torch.Tensor:
        return torch.log_softmax(self.logits / self.cfg.softmax_temperature, dim=-1)

    @torch.no_grad()
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = self.forward()
        probs = torch.exp(log_probs)
        flat_idx = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        window_idx = flat_idx // self.cfg.n_regimes
        regime_idx = flat_idx % self.cfg.n_regimes
        return window_idx, regime_idx

    def entropy(self) -> torch.Tensor:
        log_probs = self.forward()
        probs = torch.exp(log_probs)
        return -(probs * log_probs).sum()
