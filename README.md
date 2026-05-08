# AtmoZero

## Repository layout

```
atmozero/                  # the importable package
├── data/                  # ARCO-ERA5 loader, virtual stations, K-NN graph
├── verifier/              # seven rule families, lexicons, 5-fold cross-fit
│   ├── thermodynamic.py   # k=1, τ_1⁻ = 0.18
│   ├── rain_moisture.py   # k=2, τ_2⁻ = 0.20
│   ├── frontal.py         # k=3, τ_3⁻ = 0.15
│   ├── wind_regime.py     # k=4, τ_4⁻ = 0.22
│   ├── diurnal.py         # k=5, τ_5⁻ = 0.18
│   ├── climatological.py  # k=6, τ_6⁻ = 0.20
│   ├── spatial.py         # k=7, τ_7⁻ = 0.20
│   ├── lexicons.py        # V_k surface forms
│   ├── grader.py          # exposes (S_k, U, C) signals
│   └── thresholds.py      # stratified 5-fold cross-fit
├── caption/               # typed-claim list grammar + permissive parser
├── reward/                # R = Σ wₖ Sₖ − λU + μC + νD; shaping, cycle, uplift
├── policies/              # π_S Solver, π_P Proposer, cold-start protocol
├── grpo/                  # group-relative trainer with bi-level objective
├── frozen/                # F_ψ PatchTST and G_φ 8-layer DiT
├── eval/                  # faithfulness metrics, forecasting, bootstrap CIs
└── utils/

configs/atmozero_default.yaml   # the canonical training config

scripts/
├── prepare_era5.py             # ARCO-ERA5 + station network + window index
├── train_frozen_backbones.py   # train F_ψ (PatchTST) and G_φ (DiT)
├── calibrate_thresholds.py     # 5-fold cross-fit on the held-out 200-window set
├── train_atmozero.py           # GRPO post-training (Proposer–Solver self-play)
├── eval_faithfulness.py        # eight-metric faithfulness audit
└── eval_forecast.py            # caption-conditioned forecasting MSE/MAE

pyproject.toml
```

## Installation

```bash
git clone <repo-url> atmozero
cd atmozero
pip install -e .
```

## Base model

The Solver and Proposer initialise from `Qwen/Qwen2.5-7B-Instruct`. Pulled automatically by `transformers.AutoModelForCausalLM` on first use; no manual download needed.

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)
```

To pre-download:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir ~/.cache/huggingface/hub/Qwen2.5-7B-Instruct
```

## ERA5 dataset

We pull the public ARCO-ERA5 Zarr store on Google Cloud Storage; no auth required:

```
gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
```

`scripts/prepare_era5.py` opens this store via `xarray.open_zarr`, samples 8,192 virtual stations stratified across all 30 Köppen-Geiger zones, builds the K=8 neighbour graph (250 km radius), and writes a sliding-window index over 2014-2025 to `data/processed/{stations,neighbors,windows}.parquet`.

```bash
python scripts/prepare_era5.py \
    --years 2014-2025 \
    --n_stations 8192 \
    --output_dir data/processed
```

## Quick start

```bash
# 1. Pull ARCO-ERA5, build stations + neighbour graph
python scripts/prepare_era5.py \
    --years 2014-2025 \
    --n_stations 8192 \
    --output_dir data/processed

# 2. Train F_ψ (PatchTST) and G_φ (8-layer DiT)
python scripts/train_frozen_backbones.py \
    --config configs/atmozero_default.yaml \
    --windows data/processed/windows.parquet \
    --output_dir runs/frozen

# 3. Calibrate verifier thresholds via stratified 5-fold cross-fit
python scripts/calibrate_thresholds.py \
    --scores     data/annotation/scores.npz \
    --labels     data/annotation/labels.npz \
    --window_ids data/annotation/window_ids.npz \
    --strata     data/annotation/strata.npy \
    --output     runs/thresholds.json

# 4. AtmoZero post-training
python scripts/train_atmozero.py --config configs/atmozero_default.yaml

# 5a. Faithfulness (eight metrics + window-paired bootstrap CIs)
python scripts/eval_faithfulness.py \
    --captions runs/atmozero/captions.jsonl \
    --labels   data/annotation/labels.jsonl \
    --bootstrap 1000

# 5b. Caption-conditioned forecasting (MSE/MAE × {96, 192, 336, 720})
python scripts/eval_forecast.py \
    --checkpoint runs/atmozero/final \
    --patchtst   runs/frozen/patchtst.pt \
    --windows    data/processed/windows.parquet
```
