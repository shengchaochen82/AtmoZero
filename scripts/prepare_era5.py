"""Pull ARCO-ERA5 windows for the virtual-station network.

This is a thin orchestration wrapper around the modules in ``atmozero.data``
(``era5.py``, ``stations.py``, ``neighbors.py``, ``preprocessing.py``). The
real work of opening the public Zarr store, sampling stations stratified by
Köppen zone, and building the K-nearest-neighbour graph lives there; this
script wires them into a single CLI entry point.

Usage::

    python scripts/prepare_era5.py --years 2014-2025 --n_stations 8192 \\
        --output_dir data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

from atmozero.data.era5 import ARCO_ERA5_URL, ERA5Loader
from atmozero.data.stations import sample_virtual_stations
from atmozero.data.neighbors import build_neighbor_graph
from atmozero.data.preprocessing import build_window_index


def parse_year_range(spec: str) -> tuple[int, int]:
    a, b = spec.split("-")
    return int(a), int(b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2014-2025", help="inclusive year range, e.g. 2014-2025")
    ap.add_argument("--n_stations", type=int, default=8192)
    ap.add_argument("--K", type=int, default=8, help="K nearest spatial neighbours")
    ap.add_argument("--neighbor_radius_km", type=float, default=250.0)
    ap.add_argument("--T_w", type=int, default=192, help="window length in hours")
    ap.add_argument("--stride", type=int, default=12, help="window stride in hours")
    ap.add_argument("--source", default=ARCO_ERA5_URL)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--skip_zarr_probe", action="store_true",
                    help="skip the live ARCO-ERA5 connectivity check")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y0, y1 = parse_year_range(args.years)

    print(f"[prepare_era5] sampling {args.n_stations} virtual stations across all 30 Köppen zones")
    stations = sample_virtual_stations(n=args.n_stations, seed=42)
    stations.to_parquet(out / "stations.parquet", index=False)

    print(f"[prepare_era5] building K={args.K} neighbour graph (radius {args.neighbor_radius_km} km)")
    graph = build_neighbor_graph(stations, K=args.K, radius_km=args.neighbor_radius_km)
    graph.to_parquet(out / "neighbors.parquet", index=False)

    print(f"[prepare_era5] indexing {y1 - y0 + 1}-year window grid (T_w={args.T_w}, stride={args.stride})")
    windows = build_window_index(stations, year_range=(y0, y1), T_w=args.T_w, stride=args.stride)
    windows.to_parquet(out / "windows.parquet", index=False)

    if not args.skip_zarr_probe:
        print(f"[prepare_era5] verifying ARCO-ERA5 access at {args.source}")
        loader = ERA5Loader(source=args.source)
        loader._open()  # touch the store; raises if Zarr is unreachable

    print(f"[prepare_era5] wrote stations/neighbors/windows to {out}")


if __name__ == "__main__":
    main()
