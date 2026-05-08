"""Cycle reward R_cycle(c, x) = -||x - G_phi(c)||_F^2 / T_w."""
from __future__ import annotations

import torch
import torch.nn as nn


class CycleReward(nn.Module):
    def __init__(self, G_phi: nn.Module):
        super().__init__()
        self.G_phi = G_phi
        for p in self.G_phi.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, caption: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``caption`` is either (B, L) int64 token-ids or (B, text_dim) float features.
        Returns (B,) cycle rewards in maximisation form (sign already flipped)."""
        x_hat = self.G_phi(caption)
        T_w = x.shape[1]
        sq = ((x - x_hat) ** 2).sum(dim=(-2, -1))
        return -sq / float(T_w)
