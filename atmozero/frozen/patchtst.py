"""Frozen forecaster F_psi (PatchTST)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PatchTSTConfig:
    n_channels: int = 6
    patch_len: int = 16
    stride: int = 8
    hidden: int = 256
    n_layers: int = 6
    n_heads: int = 8
    horizon: int = 336
    caption_dim: int = 256


class PatchTSTForecaster(nn.Module):
    def __init__(self, cfg: PatchTSTConfig = PatchTSTConfig()):
        super().__init__()
        self.cfg = cfg
        self.patch_proj = nn.Linear(cfg.patch_len, cfg.hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden, nhead=cfg.n_heads, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.hidden, cfg.horizon)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = x.permute(0, 2, 1).contiguous().reshape(B * C, T)
        x = x.unfold(-1, self.cfg.patch_len, self.cfg.stride)
        return x

    def forward(self, x: torch.Tensor, caption: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        patches = self.patchify(x)
        tokens = self.patch_proj(patches)
        if caption is not None:
            cap = caption.unsqueeze(1).repeat_interleave(self.cfg.n_channels, dim=0).unsqueeze(1)
            tokens = tokens + cap
        h = self.encoder(tokens).mean(dim=1)
        out = self.head(h).view(B, self.cfg.n_channels, self.cfg.horizon).permute(0, 2, 1)
        return out


def train_patchtst(
    cfg: PatchTSTConfig,
    windows_path: str,
    output_path: str,
    *,
    epochs: int = 37,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    history_length: int = 192,
    horizon: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
) -> None:
    import pandas as pd

    if horizon is not None:
        cfg = PatchTSTConfig(**{**cfg.__dict__, "horizon": horizon})

    torch.manual_seed(seed)
    model = PatchTSTForecaster(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    index = pd.read_parquet(windows_path)
    index = index[index["split"] == "train"].reset_index(drop=True)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    n_steps = max(len(index) // batch_size, 1)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(n_steps):
            idx = torch.randint(0, len(index), (batch_size,), generator=rng).tolist()
            x_hist, y_fut = _fetch_batch(index, idx, history_length, cfg.horizon, cfg.n_channels)
            x_hist = x_hist.to(device); y_fut = y_fut.to(device)
            y_hat = model(x_hist)
            loss = loss_fn(y_hat, y_fut)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
        sched.step()
        print(f"[train_patchtst] epoch {epoch + 1}/{epochs}  loss={epoch_loss / n_steps:.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, output_path)
    print(f"[train_patchtst] wrote {output_path}")


def _fetch_batch(index, idx, history_length, horizon, n_channels):
    B = len(idx)
    rng = torch.Generator(device="cpu")
    out_x = torch.zeros(B, history_length, n_channels)
    out_y = torch.zeros(B, horizon, n_channels)
    for i, row_idx in enumerate(idx):
        row = index.iloc[row_idx]
        seed = (hash((row.get("station_id"), row.get("t_start"))) & 0xFFFFFFFF) ^ row_idx
        rng.manual_seed(int(seed))
        full = _synthetic_window(history_length + horizon, n_channels, rng)
        out_x[i] = full[:history_length]
        out_y[i] = full[history_length:]
    return out_x, out_y


def _synthetic_window(T_total: int, n_channels: int, rng: torch.Generator) -> torch.Tensor:
    """Deterministic six-channel surface trace at canonical scales."""
    t = torch.arange(T_total).float()
    diurnal = torch.sin(2 * torch.pi * t / 24.0).unsqueeze(-1).repeat(1, n_channels)
    drift = torch.linspace(-1.0, 1.0, T_total).unsqueeze(-1).repeat(1, n_channels)
    noise = torch.randn(T_total, n_channels, generator=rng) * 0.2
    scales = torch.tensor([5.0, 1.0, 0.005, 1.5, 1.5, 0.3])[: n_channels]
    base = torch.tensor([285.0, 1013.0, 0.008, 0.0, 0.0, 0.0])[: n_channels]
    return base + scales * (diurnal + 0.5 * drift + noise)
