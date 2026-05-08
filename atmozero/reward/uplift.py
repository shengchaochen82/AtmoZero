"""Forecaster-uplift reward via frozen F_psi (PatchTST).

R_uplift = L(x_>t0 | x_<=t0) - L(x_>t0 | x_<=t0, c)

i.e., the loss reduction the caption induces on the frozen forecaster.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UpliftReward(nn.Module):
    def __init__(self, F_psi: nn.Module):
        super().__init__()
        self.F_psi = F_psi
        for p in self.F_psi.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        caption_features: torch.Tensor,  # (B, d_z) caption conditioning vector
        x_past: torch.Tensor,             # (B, T_past, d) input window
        x_future: torch.Tensor,           # (B, H, d) target horizon
    ) -> torch.Tensor:
        """Returns (B,) uplift rewards."""
        loss_no_caption = _per_sample_mse(self.F_psi(x_past), x_future)
        loss_with_caption = _per_sample_mse(
            self.F_psi(x_past, caption=caption_features), x_future
        )
        return loss_no_caption - loss_with_caption


def _per_sample_mse(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((yhat - y) ** 2).mean(dim=tuple(range(1, y.ndim)))
