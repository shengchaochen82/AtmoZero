"""Frozen text-to-series decoder G_phi (8-layer DiT).

Trained on (window, programmatic-template) pairs only; templates are
produced deterministically from the rule-family lexicons V_k by running the
verifier on each window and one-hot-encoding the supported (family, value)
claims into the lexicon vocabulary V_total = ⋃_k V_k. No human captions ever
enter G_phi training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from atmozero.frozen.patchtst import CANONICAL_CHANNELS, _fetch_batch, _stack_channels
from atmozero.verifier import VerifierGrader
from atmozero.verifier.base import WindowContext
from atmozero.verifier.lexicons import LEXICONS


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


def _build_lexicon_index() -> Tuple[Dict[Tuple[int, str], int], int]:
    """Map every (family_id, value) in V_total to a unique integer index."""
    idx: Dict[Tuple[int, str], int] = {}
    for fid, lex in LEXICONS.items():
        for v in lex:
            idx[(fid, v)] = len(idx)
    return idx, len(idx)


LEXICON_INDEX, V_TOTAL_SIZE = _build_lexicon_index()


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
        # Programmatic-template path: V_total → text_dim. Random orthogonal-ish
        # projection seeded by a fixed value so the embedding is deterministic
        # across runs without depending on the LM tokenizer.
        proj = torch.empty(V_TOTAL_SIZE, cfg.text_dim)
        torch.nn.init.orthogonal_(proj.unsqueeze(0))
        self.register_buffer("lexicon_proj", proj.squeeze(0))
        self.time_embed = nn.Embedding(cfg.n_diffusion_steps, cfg.hidden)
        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden, cfg.n_heads) for _ in range(cfg.n_layers)]
        )
        self.out = nn.Linear(cfg.hidden, cfg.n_channels)

    def encode_template(self, multihot: torch.Tensor) -> torch.Tensor:
        """Project a (B, V_total) multi-hot of supported (k, v) claims to (B, text_dim)."""
        return multihot @ self.lexicon_proj

    @torch.no_grad()
    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        """Accepts (B, L) int64 token-ids, (B, V_total) float multi-hot, or (B, text_dim) float features."""
        if caption.dtype in (torch.long, torch.int64, torch.int32):
            caption_features = self.token_embed(caption).mean(dim=1)
        elif caption.shape[-1] == V_TOTAL_SIZE:
            caption_features = self.encode_template(caption.float())
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
    stations_path: Optional[str] = None,
    epochs: int = 46,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 128,
    seed: int = 42,
) -> None:
    """Train G_phi on real ARCO-ERA5 windows conditioned on lexicon-derived
    programmatic templates. Multi-GPU via ``accelerate launch``."""
    import pandas as pd
    from accelerate import Accelerator

    accelerator = Accelerator()
    torch.manual_seed(seed + accelerator.process_index)

    model = DiTDecoder(cfg)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model, opt = accelerator.prepare(model, opt)

    index = pd.read_parquet(windows_path)
    index = index[index["split"] == "train"].reset_index(drop=True)
    if stations_path is not None:
        stations = pd.read_parquet(stations_path).set_index("station_id")
        index = index.join(stations, on="station_id", how="left")

    grader = VerifierGrader()
    rng = np.random.default_rng(seed + accelerator.process_index)
    per_rank_batch = max(1, batch_size // accelerator.num_processes)
    n_steps = max(len(index) // batch_size, 1)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(n_steps):
            idx = rng.integers(0, len(index), size=per_rank_batch).tolist()
            x_batch, multihot = _fetch_with_templates(index, idx, cfg, grader)
            x_batch = x_batch.to(accelerator.device)
            features = accelerator.unwrap_model(model).encode_template(multihot.to(accelerator.device))
            x_hat = _denoise_step(accelerator.unwrap_model(model), features)
            loss = ((x_batch - x_hat) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
        if accelerator.is_main_process:
            print(f"[train_dit_decoder] epoch {epoch + 1}/{epochs}  loss={epoch_loss / n_steps:.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": accelerator.unwrap_model(model).state_dict(),
                    "config": cfg.__dict__}, output_path)
        print(f"[train_dit_decoder] wrote {output_path}")


def _fetch_with_templates(index, idx: List[int], cfg: DiTConfig, grader: VerifierGrader):
    """Fetch (B, T_w, C) windows and (B, V_total) lexicon multi-hot templates."""
    x_hist, _ = _fetch_batch(index, idx, cfg.T_w, 0, cfg.n_channels)
    multihots = []
    for i, row_idx in enumerate(idx):
        row = index.iloc[row_idx]
        x_dict = _channels_dict_from_tensor(x_hist[i])
        ctx = WindowContext(
            x=x_dict,
            metadata={
                "lat": float(row.get("lat", 0.0)),
                "lon": float(row.get("lon", 0.0)),
                "elevation": float(row.get("elevation", 0.0)),
                "koppen_zone": str(row.get("koppen_zone", "Cfa")),
            },
            neighbors=None,
            climatology=None,
        )
        multihots.append(_lexicon_multihot(ctx, grader))
    return x_hist, torch.stack(multihots, dim=0)


def _channels_dict_from_tensor(x: torch.Tensor) -> Dict[str, np.ndarray]:
    arr = x.cpu().numpy()
    return {ch: arr[:, i] for i, ch in enumerate(CANONICAL_CHANNELS[: arr.shape[1]])}


def _lexicon_multihot(ctx: WindowContext, grader: VerifierGrader) -> torch.Tensor:
    """Multi-hot over V_total of (k, v) pairs the verifier supports on this window."""
    mh = torch.zeros(V_TOTAL_SIZE)
    available = {k for k, vals in ctx.x.items() if vals is not None}
    for fid, rule in grader.rules.items():
        if not rule.has_required_channels(available):
            continue
        for v in LEXICONS[fid]:
            score = rule.grade(v, ctx)
            if score >= rule.tau_plus:
                mh[LEXICON_INDEX[(fid, v)]] = 1.0
    return mh


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
