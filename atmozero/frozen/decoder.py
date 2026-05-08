"""Frozen text-to-series decoder G_phi (8-layer DiT).

Trained on (window, programmatic-template) pairs only; no human captions ever
enter G_phi training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class DiTConfig:
    hidden: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_diffusion_steps: int = 28
    n_channels: int = 6
    T_w: int = 192
    text_dim: int = 384
    vocab_size: int = 152064


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), ctx, ctx)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class DiTDecoder(nn.Module):
    def __init__(self, cfg: DiTConfig = DiTConfig()):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_channels, cfg.hidden)
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.text_dim)
        self.text_proj = nn.Linear(cfg.text_dim, cfg.hidden)
        self.time_embed = nn.Embedding(cfg.n_diffusion_steps, cfg.hidden)
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden, cfg.n_heads) for _ in range(cfg.n_layers)]
        )
        self.out = nn.Linear(cfg.hidden, cfg.n_channels)

    @torch.no_grad()
    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        """Accepts either (B, L) int64 token-ids or (B, text_dim) float features."""
        if caption.dtype in (torch.long, torch.int64, torch.int32):
            caption_features = self.token_embed(caption).mean(dim=1)
        else:
            caption_features = caption
        B = caption_features.shape[0]
        device = caption_features.device
        ctx = self.text_proj(caption_features).unsqueeze(1)
        x = torch.randn(B, self.cfg.T_w, self.cfg.n_channels, device=device)
        for t in reversed(range(self.cfg.n_diffusion_steps)):
            t_emb = self.time_embed(torch.tensor([t], device=device)).expand(B, -1).unsqueeze(1)
            h = self.input_proj(x) + t_emb
            for block in self.blocks:
                h = block(h, ctx)
            x = x - 0.05 * self.out(h)
        return x


def train_dit_decoder(
    cfg: DiTConfig,
    windows_path: str,
    output_path: str,
    *,
    epochs: int = 46,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 42,
) -> None:
    import pandas as pd
    from atmozero.frozen.patchtst import _fetch_batch

    torch.manual_seed(seed)
    model = DiTDecoder(cfg).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    index = pd.read_parquet(windows_path)
    index = index[index["split"] == "train"].reset_index(drop=True)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    n_steps = max(len(index) // batch_size, 1)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(n_steps):
            idx = torch.randint(0, len(index), (batch_size,), generator=rng).tolist()
            x_hist, _ = _fetch_batch(index, idx, cfg.T_w, cfg.T_w, cfg.n_channels)
            x_hist = x_hist.to(device)
            features = _template_features(x_hist, cfg.text_dim)
            x_hat = _denoise_step(model, features)
            loss = ((x_hist - x_hat) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
        print(f"[train_dit_decoder] epoch {epoch + 1}/{epochs}  loss={epoch_loss / n_steps:.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, output_path)
    print(f"[train_dit_decoder] wrote {output_path}")


def _template_features(x: torch.Tensor, text_dim: int) -> torch.Tensor:
    means = x.mean(dim=1)
    stds = x.std(dim=1)
    base = torch.cat([means, stds], dim=-1)
    if base.shape[-1] >= text_dim:
        return base[..., :text_dim]
    pad = torch.zeros(base.shape[0], text_dim - base.shape[-1], device=base.device)
    return torch.cat([base, pad], dim=-1)


def _denoise_step(model: "DiTDecoder", features: torch.Tensor) -> torch.Tensor:
    B = features.shape[0]
    device = features.device
    ctx = model.text_proj(features).unsqueeze(1)
    x = torch.randn(B, model.cfg.T_w, model.cfg.n_channels, device=device)
    for t in reversed(range(model.cfg.n_diffusion_steps)):
        t_emb = model.time_embed(torch.tensor([t], device=device)).expand(B, -1).unsqueeze(1)
        h = model.input_proj(x) + t_emb
        for block in model.blocks:
            h = block(h, ctx)
        x = x - 0.05 * model.out(h)
    return x
