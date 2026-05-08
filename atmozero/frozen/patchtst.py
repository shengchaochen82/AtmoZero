"""Frozen forecaster F_psi (PatchTST)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from atmozero.data.era5 import ERA5Loader, WindowSpec


CANONICAL_CHANNELS = ("T", "P", "q", "u", "v", "r")


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
    stations_path: Optional[str] = None,
    epochs: int = 37,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    history_length: int = 192,
    horizon: Optional[int] = None,
    seed: int = 42,
) -> None:
    """Train F_psi on real ARCO-ERA5 windows. Multi-GPU via ``accelerate launch``."""
    import pandas as pd
    from accelerate import Accelerator

    if horizon is not None:
        cfg = PatchTSTConfig(**{**cfg.__dict__, "horizon": horizon})

    accelerator = Accelerator()
    torch.manual_seed(seed + accelerator.process_index)

    model = PatchTSTForecaster(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    model, opt, sched = accelerator.prepare(model, opt, sched)

    index = pd.read_parquet(windows_path)
    index = index[index["split"] == "train"].reset_index(drop=True)
    if stations_path is not None:
        stations = pd.read_parquet(stations_path).set_index("station_id")
        index = index.join(stations, on="station_id", how="left")
    _require_station_metadata(index)

    loader = ERA5Loader()
    rng = np.random.default_rng(seed + accelerator.process_index)
    per_rank_batch = max(1, batch_size // accelerator.num_processes)
    n_steps = max(len(index) // batch_size, 1)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(n_steps):
            idx = rng.integers(0, len(index), size=per_rank_batch).tolist()
            x_hist, y_fut = _fetch_batch(index, idx, history_length, cfg.horizon, cfg.n_channels, loader)
            x_hist = x_hist.to(accelerator.device); y_fut = y_fut.to(accelerator.device)
            y_hat = model(x_hist)
            loss = loss_fn(y_hat, y_fut)
            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach())
        sched.step()
        if accelerator.is_main_process:
            print(f"[train_patchtst] epoch {epoch + 1}/{epochs}  loss={epoch_loss / n_steps:.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": accelerator.unwrap_model(model).state_dict(),
                    "config": cfg.__dict__}, output_path)
        print(f"[train_patchtst] wrote {output_path}")


def _require_station_metadata(index) -> None:
    missing = [c for c in ("lat", "lon", "elevation", "koppen_zone") if c not in index.columns]
    if missing:
        raise ValueError(
            f"window index is missing station columns {missing}; pass --stations to "
            "train_frozen_backbones.py / pre-join the stations parquet"
        )


def _fetch_batch(index, idx, history_length, horizon, n_channels, loader: Optional[ERA5Loader] = None):
    """Pull ``len(idx)`` (history, future) pairs from ARCO-ERA5 and stack them."""
    if loader is None:
        loader = ERA5Loader()
    B = len(idx)
    out_x = torch.zeros(B, history_length, n_channels)
    out_y = torch.zeros(B, horizon, n_channels)
    needed = history_length + horizon
    for i, row_idx in enumerate(idx):
        row = index.iloc[row_idx]
        spec = WindowSpec(
            station_id=int(row["station_id"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            elevation=float(row["elevation"]),
            koppen_zone=str(row["koppen_zone"]),
            t_start=np.datetime64(str(row["t_start"]), "h"),
            T_w=needed,
        )
        window = loader.fetch(spec)
        full = _stack_channels(window, needed, n_channels)
        out_x[i] = full[:history_length]
        out_y[i] = full[history_length:history_length + horizon]
    return out_x, out_y


def _stack_channels(window: dict, T_total: int, n_channels: int) -> torch.Tensor:
    cols = []
    for ch in CANONICAL_CHANNELS[:n_channels]:
        arr = window.get(ch)
        if arr is None:
            raise KeyError(f"ARCO-ERA5 window missing canonical channel {ch!r}")
        cols.append(np.asarray(arr, dtype=np.float32)[:T_total])
    stacked = np.stack(cols, axis=-1)  # (T, C)
    if stacked.shape[0] < T_total:
        raise ValueError(f"ARCO-ERA5 window shorter than expected ({stacked.shape[0]} < {T_total})")
    return torch.from_numpy(stacked)
