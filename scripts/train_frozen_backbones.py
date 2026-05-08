"""Train the two frozen backbones used at AtmoZero post-training time.

- F_psi : PatchTST forecaster (no caption channel) — used as the no-caption
          reference and as the conditioning backbone for the forecaster-uplift
          reward R_uplift.
- G_phi : 8-layer DiT text-to-series decoder — used as the cycle-reward
          decoder R_cycle. Trained on (window, programmatic-template) pairs
          only; no human captions ever enter G_phi training.

Once trained, both checkpoints are loaded with ``requires_grad_(False)``
during AtmoZero post-training.

Usage::

    python scripts/train_frozen_backbones.py \\
        --config configs/atmozero_default.yaml \\
        --windows data/processed/windows.parquet \\
        --output_dir runs/frozen
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from atmozero.frozen.patchtst import PatchTSTConfig, train_patchtst
from atmozero.frozen.decoder import DiTConfig, train_dit_decoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/atmozero_default.yaml")
    ap.add_argument("--windows", required=True, help="parquet of (station, t_start, T_w) windows")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--skip_patchtst", action="store_true")
    ap.add_argument("--skip_decoder", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_patchtst:
        pcfg = PatchTSTConfig(**cfg.get("frozen", {}).get("patchtst", {}))
        print(f"[train_frozen_backbones] training PatchTST: {pcfg}")
        train_patchtst(pcfg, windows_path=args.windows, output_path=str(out / "patchtst.pt"))

    if not args.skip_decoder:
        dcfg = DiTConfig(**cfg.get("frozen", {}).get("decoder", {}))
        print(f"[train_frozen_backbones] training DiT decoder G_phi: {dcfg}")
        train_dit_decoder(dcfg, windows_path=args.windows, output_path=str(out / "dit.pt"))


if __name__ == "__main__":
    main()
