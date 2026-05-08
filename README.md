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
├── frozen/                # F_ψ PatchTST and G_φ 8-layer DiT
├── eval/                  # faithfulness metrics, forecasting, bootstrap CIs
├── proposer.py            # window sampler + cold-start gate (τ_cold = 0.84)
├── verl_reward.py         # verl compute_score bridge → verifier + reward
├── verl_dataset.py        # AtmoZero RLHFDataset subclass for verl
└── utils/

configs/
├── atmozero_default.yaml   # paper-anchored hyperparameters
└── verl_grpo.yaml          # verl Hydra config (GRPO)

scripts/
├── prepare_era5.py             # ARCO-ERA5 + station network + window index
├── train_frozen_backbones.py   # train F_ψ (PatchTST) and G_φ (DiT)
├── calibrate_thresholds.py     # 5-fold cross-fit on the held-out 200-window set
├── prepare_verl_data.py        # parquet → verl prompt dataset (with extra_info)
├── run_verl.sh                 # GRPO post-training launch (verl)
├── generate_captions.py        # vLLM caption generation on held-out windows
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

# 2. Train F_ψ (PatchTST) and G_φ (8-layer DiT) on real ARCO-ERA5 windows.
#    Multi-GPU via accelerate (set --num_processes to your GPU count).
accelerate launch --num_processes 4 scripts/train_frozen_backbones.py \
    --config configs/atmozero_default.yaml \
    --windows  data/processed/windows.parquet \
    --stations data/processed/stations.parquet \
    --output_dir runs/frozen

# 3. Calibrate verifier thresholds via stratified 5-fold cross-fit
python scripts/calibrate_thresholds.py \
    --scores     data/annotation/scores.npz \
    --labels     data/annotation/labels.npz \
    --window_ids data/annotation/window_ids.npz \
    --strata     data/annotation/strata.npy \
    --output     runs/thresholds.json

# 4a. Build the verl prompt parquet (one-time)
python scripts/prepare_verl_data.py \
    --windows  data/processed/windows.parquet \
    --stations data/processed/stations.parquet \
    --split    train \
    --out      data/processed/verl_train.parquet

# 4b. AtmoZero GRPO post-training (verl on 4 × H100)
bash scripts/run_verl.sh

# 5. Generate held-out captions via vLLM (set --tensor_parallel_size to your GPU count)
python scripts/generate_captions.py \
    --checkpoint runs/atmozero/final \
    --windows    data/processed/windows.parquet \
    --stations   data/processed/stations.parquet \
    --split      test \
    --tensor_parallel_size 4 \
    --out        runs/atmozero/captions.jsonl

# 6a. Faithfulness (eight metrics + window-paired bootstrap CIs)
python scripts/eval_faithfulness.py \
    --captions runs/atmozero/captions.jsonl \
    --labels   data/annotation/labels.jsonl \
    --bootstrap 1000

# 6b. Caption-conditioned forecasting (MSE/MAE × {96, 192, 336, 720})
python scripts/eval_forecast.py \
    --checkpoint runs/atmozero/final \
    --patchtst   runs/frozen/patchtst.pt \
    --windows    data/processed/windows.parquet
```
