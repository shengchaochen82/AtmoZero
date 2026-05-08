"""Window-level paired bootstrap for confidence intervals."""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def window_paired_bootstrap(
    metric_fn: Callable,
    payload: Dict,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """95% half-width on ``metric_fn(payload)`` via window-level resampling.

    Resamples are paired across captioners on the same window subsample.
    """
    rng = np.random.default_rng(seed)
    window_ids = np.asarray(payload["window_ids"])
    unique_windows = np.unique(window_ids)
    n_windows = len(unique_windows)

    estimates = []
    for _ in range(n_resamples):
        resampled = rng.choice(unique_windows, size=n_windows, replace=True)
        index_map = {w: np.where(window_ids == w)[0] for w in resampled}
        idx = np.concatenate([index_map[w] for w in resampled])
        estimates.append(metric_fn(_subset(payload, idx)))

    point = float(metric_fn(payload))
    arr = np.asarray(estimates, dtype=float)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "estimate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "half_width": float((hi - lo) / 2.0),
        "std": float(arr.std()),
    }


def _subset(payload: Dict, idx: np.ndarray) -> Dict:
    out = {}
    n = len(payload["window_ids"])
    for k, v in payload.items():
        if isinstance(v, list) and len(v) == n:
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    out["window_ids"] = [payload["window_ids"][i] for i in idx]
    return out
