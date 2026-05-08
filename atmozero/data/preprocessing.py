"""De-trending, quality control, split utilities, and window-index construction."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def detrend(window: Dict[str, np.ndarray], slopes: Dict[str, float]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in window.items():
        if v.ndim != 1:
            out[k] = v
            continue
        slope = slopes.get(k, 0.0)
        t = np.arange(v.size, dtype=v.dtype)
        out[k] = v - slope * t
    return out


def quality_control(window: Dict[str, np.ndarray], climatology: Dict[str, float]) -> bool:
    r = window.get("r")
    if r is not None and climatology.get("r_p9997") is not None:
        if np.max(r) > climatology["r_p9997"]:
            return False
    for v in window.values():
        if v is None:
            continue
        if not np.isfinite(v).all():
            return False
    return True


def time_split(t: np.datetime64) -> str:
    year = int(str(t)[:4])
    if year <= 2022:
        return "train"
    if year == 2023:
        return "val"
    return "test"


def build_window_index(
    stations,
    year_range: Tuple[int, int] = (2014, 2025),
    T_w: int = 192,
    stride: int = 12,
):
    """Enumerate every (station, t_start) that yields a length-T_w window."""
    import pandas as pd

    if hasattr(stations, "iterrows"):
        station_ids = [int(r.station_id) for _, r in stations.iterrows()]
    else:
        station_ids = [int(s.station_id) for s in stations]

    y_lo, y_hi = year_range
    starts = []
    t = np.datetime64(f"{y_lo:04d}-01-01T00", "h")
    end = np.datetime64(f"{y_hi:04d}-12-31T23", "h") - np.timedelta64(T_w, "h")
    while t <= end:
        starts.append(t)
        t += np.timedelta64(stride, "h")

    rows = []
    for sid in station_ids:
        for t_start in starts:
            rows.append({
                "station_id": sid,
                "t_start": np.datetime_as_string(t_start, unit="h"),
                "T_w": T_w,
                "split": time_split(t_start),
            })
    return pd.DataFrame(rows, columns=["station_id", "t_start", "T_w", "split"])
