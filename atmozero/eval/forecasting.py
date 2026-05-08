"""Forecasting MSE/MAE at horizons {96, 192, 336, 720} hours."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def mse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((yhat - y) ** 2))


def mae(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(yhat - y)))


def horizon_table(yhat: Dict[int, np.ndarray], y: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
    return {h: {"MSE": mse(yhat[h], y[h]), "MAE": mae(yhat[h], y[h])} for h in yhat}


def evaluate_forecasting(
    captioner_checkpoint: str,
    forecaster_checkpoint: str,
    windows_path: str,
    horizons: Iterable[int] = (96, 192, 336, 720),
    history_length: int = 192,
    device: str = "cpu",
    max_windows: int = 4096,
) -> Dict:
    """Score a captioner / frozen-PatchTST pair on the held-out test split.

    Captioner-feature cache is loaded if present alongside ``captioner_checkpoint``;
    otherwise the no-caption forecaster is scored.
    """
    import pandas as pd
    import torch

    from atmozero.frozen.patchtst import (
        PatchTSTConfig,
        PatchTSTForecaster,
        _fetch_batch,
    )

    blob = torch.load(forecaster_checkpoint, map_location=device)
    cfg = PatchTSTConfig(**blob.get("config", {}))
    forecaster = PatchTSTForecaster(cfg).to(device)
    forecaster.load_state_dict(blob["state_dict"])
    forecaster.eval()

    index = pd.read_parquet(windows_path)
    index = index[index["split"] == "test"].reset_index(drop=True)
    index = index.head(max_windows).reset_index(drop=True)
    n = len(index)
    if n == 0:
        return {"MSE": {}, "MAE": {}, "window_ids": []}

    horizons = list(horizons)
    max_h = max(horizons)
    cfg = PatchTSTConfig(**{**cfg.__dict__, "horizon": max_h})
    forecaster.cfg = cfg
    if forecaster.head.out_features != max_h:
        forecaster.head = torch.nn.Linear(cfg.hidden, max_h).to(device)

    caption_features = _maybe_load_caption_features(captioner_checkpoint, n, cfg.caption_dim, device)

    yhat: Dict[int, List[np.ndarray]] = {h: [] for h in horizons}
    y_true: Dict[int, List[np.ndarray]] = {h: [] for h in horizons}
    with torch.no_grad():
        batch = 32
        for start in range(0, n, batch):
            idx = list(range(start, min(start + batch, n)))
            x_hist, y_fut = _fetch_batch(index, idx, history_length, max_h, cfg.n_channels)
            x_hist = x_hist.to(device); y_fut = y_fut.to(device)
            cap = caption_features[start: start + len(idx)] if caption_features is not None else None
            preds = forecaster(x_hist, caption=cap).cpu().numpy()
            true = y_fut.cpu().numpy()
            for h in horizons:
                yhat[h].append(preds[:, :h])
                y_true[h].append(true[:, :h])

    return {
        "MSE": {f"H={h}": mse(np.concatenate(yhat[h]), np.concatenate(y_true[h])) for h in horizons},
        "MAE": {f"H={h}": mae(np.concatenate(yhat[h]), np.concatenate(y_true[h])) for h in horizons},
        "window_ids": index["station_id"].astype(int).tolist(),
    }


def _maybe_load_caption_features(captioner_checkpoint: str, n: int, dim: int, device: str):
    import torch

    cache = Path(captioner_checkpoint) / "caption_features.pt"
    if cache.exists():
        feats = torch.load(cache, map_location=device)
        if feats.shape[0] >= n and feats.shape[-1] == dim:
            return feats[:n].to(device)
    return None
