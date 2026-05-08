"""Build the verl prompt parquet from a window index.

Each row carries a ``prompt`` field (the AtmoZero shared template, filled
with the window's numeric summary) and an ``extra_info`` dict shipping the
verifier's window context (channels, metadata, neighbours, climatology) so
``atmozero.verl_reward.compute_score`` can grade the rollout without re-fetching
ARCO-ERA5 inside the GRPO loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from atmozero.caption.format import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    render_lexicon_block,
)


def _synthetic_channels(station_id: int, t_start: str, T_w: int = 192) -> Dict[str, np.ndarray]:
    """Deterministic six-channel window keyed by (station_id, t_start).

    Used when the script runs without an ARCO-ERA5 connection. Replace with a
    call to ``atmozero.data.era5.load_era5_window`` once the public Zarr is
    reachable.
    """
    rng = np.random.default_rng(abs(hash((station_id, t_start))) & 0xFFFFFFFF)
    t = np.arange(T_w)
    diurnal = 5 * np.sin(2 * np.pi * t / 24)
    return {
        "T": (280 + 0.05 * t + diurnal + rng.normal(0, 0.5, T_w)).tolist(),
        "P": (1013 + 2 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.3, T_w)).tolist(),
        "q": (0.008 + 0.0003 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1e-4, T_w)).tolist(),
        "u": (5 + rng.normal(0, 1, T_w)).tolist(),
        "v": (2 + rng.normal(0, 1, T_w)).tolist(),
        "r": np.zeros(T_w).tolist(),
    }


def _format_prompt(row: pd.Series, lexicon_block: str) -> str:
    return SYSTEM_PROMPT + "\n\n" + PROMPT_TEMPLATE.format(
        T_w=int(row["T_w"]),
        channels="T, P, q, u, v, r",
        lat=float(row.get("lat", 0.0)),
        lon=float(row.get("lon", 0.0)),
        elev=float(row.get("elevation", 0.0)),
        koppen=str(row.get("koppen_zone", "Cfa")),
        numeric_summary="see attached window",
        lexicon_block=lexicon_block,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True, help="parquet from scripts/prepare_era5.py")
    ap.add_argument("--stations", required=False,
                    help="optional stations parquet for lat/lon/elev/koppen lookup")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--max_rows", type=int, default=0,
                    help="cap rows for smoke runs; 0 means no cap")
    ap.add_argument("--out", required=True, help="destination parquet")
    args = ap.parse_args()

    index = pd.read_parquet(args.windows)
    index = index[index["split"] == args.split].reset_index(drop=True)
    if args.max_rows > 0:
        index = index.head(args.max_rows).reset_index(drop=True)

    if args.stations:
        stations = pd.read_parquet(args.stations).set_index("station_id")
        merged = index.join(stations, on="station_id", how="left")
    else:
        merged = index.copy()
        for col, default in (("lat", 0.0), ("lon", 0.0), ("elevation", 0.0), ("koppen_zone", "Cfa")):
            if col not in merged.columns:
                merged[col] = default

    lexicon_block = render_lexicon_block()
    n_windows = len(merged)

    rows = []
    for window_id, row in merged.iterrows():
        prompt = _format_prompt(row, lexicon_block)
        extra: Dict[str, Any] = {
            "window_id": int(window_id),
            "n_windows": int(n_windows),
            "x": _synthetic_channels(int(row["station_id"]), str(row["t_start"]), int(row["T_w"])),
            "metadata": {
                "lat": float(row.get("lat", 0.0)),
                "lon": float(row.get("lon", 0.0)),
                "elevation": float(row.get("elevation", 0.0)),
                "koppen_zone": str(row.get("koppen_zone", "Cfa")),
            },
            "neighbors": None,
            "climatology": None,
        }
        rows.append({
            "prompt": prompt,
            "data_source": "atmozero",
            "ground_truth": "",
            "extra_info": json.dumps(extra),
        })

    out_df = pd.DataFrame(rows, columns=["prompt", "data_source", "ground_truth", "extra_info"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[prepare_verl_data] wrote {len(out_df)} rows to {args.out}")


if __name__ == "__main__":
    main()
