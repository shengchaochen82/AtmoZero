"""Build the verl prompt parquet from a window index, with real ARCO-ERA5 channels.

Each row carries a ``prompt`` field (the AtmoZero shared template, filled
with the window's numeric summary) and an ``extra_info`` dict shipping the
verifier's window context (channels, metadata, neighbours, climatology) so
``atmozero.verl_reward.compute_score`` can grade rollouts without re-fetching
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
from atmozero.data.era5 import ERA5Loader, WindowSpec


def _format_prompt(row: pd.Series, lexicon_block: str) -> str:
    return SYSTEM_PROMPT + "\n\n" + PROMPT_TEMPLATE.format(
        T_w=int(row["T_w"]),
        channels="T, P, q, u, v, r",
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        elev=float(row["elevation"]),
        koppen=str(row["koppen_zone"]),
        numeric_summary=_numeric_summary(row),
        lexicon_block=lexicon_block,
    )


def _numeric_summary(row: pd.Series) -> str:
    return (
        f"station {int(row['station_id'])} at "
        f"({row['lat']:.2f}, {row['lon']:.2f}), elev {row['elevation']:.0f} m, "
        f"zone {row['koppen_zone']}, t_start {row['t_start']}, T_w={int(row['T_w'])}h"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True, help="parquet from scripts/prepare_era5.py")
    ap.add_argument("--stations", required=True,
                    help="stations parquet for lat/lon/elev/koppen lookup")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--max_rows", type=int, default=0,
                    help="cap rows for smoke runs; 0 means no cap")
    ap.add_argument("--out", required=True, help="destination parquet")
    args = ap.parse_args()

    index = pd.read_parquet(args.windows)
    index = index[index["split"] == args.split].reset_index(drop=True)
    if args.max_rows > 0:
        index = index.head(args.max_rows).reset_index(drop=True)

    stations = pd.read_parquet(args.stations).set_index("station_id")
    merged = index.join(stations, on="station_id", how="left")
    for col in ("lat", "lon", "elevation", "koppen_zone"):
        if col not in merged.columns or merged[col].isna().any():
            raise ValueError(f"stations parquet missing or partial for column {col!r}")

    lexicon_block = render_lexicon_block()
    n_windows = len(merged)
    loader = ERA5Loader()

    rows = []
    for window_id, row in merged.iterrows():
        spec = WindowSpec(
            station_id=int(row["station_id"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            elevation=float(row["elevation"]),
            koppen_zone=str(row["koppen_zone"]),
            t_start=np.datetime64(str(row["t_start"]), "h"),
            T_w=int(row["T_w"]),
        )
        window = loader.fetch(spec)
        prompt = _format_prompt(row, lexicon_block)
        extra: Dict[str, Any] = {
            "window_id": int(window_id),
            "n_windows": int(n_windows),
            "x": {ch: np.asarray(v, dtype=np.float32).tolist() for ch, v in window.items()},
            "metadata": {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "elevation": float(row["elevation"]),
                "koppen_zone": str(row["koppen_zone"]),
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
