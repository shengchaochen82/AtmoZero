"""Evaluate caption-conditioned forecasting against the frozen PatchTST F_psi.

Reports MSE and MAE at horizons {96, 192, 336, 720} on the 2024-2025 test
split, with window-paired bootstrap confidence intervals.

Usage::

    python scripts/eval_forecast.py --checkpoint runs/atmozero/final \\
        --patchtst runs/frozen/patchtst.pt \\
        --windows data/processed/windows.parquet \\
        --horizons 96 192 336 720 \\
        --bootstrap 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atmozero.eval.forecasting import evaluate_forecasting
from atmozero.eval.bootstrap import window_paired_bootstrap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Solver checkpoint directory")
    ap.add_argument("--patchtst", required=True, help="frozen PatchTST checkpoint")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    summary = evaluate_forecasting(
        captioner_checkpoint=args.checkpoint,
        forecaster_checkpoint=args.patchtst,
        windows_path=args.windows,
        horizons=args.horizons,
    )

    cis = {}
    for h in args.horizons:
        cis[f"H={h}"] = window_paired_bootstrap(
            lambda payload, hh=h: payload["MSE"][f"H={hh}"],
            summary,
            n_resamples=args.bootstrap,
        )

    out = {"point": summary, "ci": cis}
    text = json.dumps(out, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
