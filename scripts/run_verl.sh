#!/usr/bin/env bash
# AtmoZero GRPO post-training via verl.
#
# Prerequisites:
#   1. Install verl: `pip install verl`
#   2. Build the verl prompt parquet: `python scripts/prepare_verl_data.py ...`
#   3. Train the frozen backbones: `python scripts/train_frozen_backbones.py ...`
#   4. (Optional) Calibrate thresholds: `python scripts/calibrate_thresholds.py ...`

set -euo pipefail

CONFIG="${CONFIG:-configs/verl_grpo.yaml}"
STATE_PATH="${ATMOZERO_STATE_PATH:-runs/atmozero/state.json}"
N_WINDOWS="${ATMOZERO_N_WINDOWS:-8192}"

mkdir -p "$(dirname "$STATE_PATH")"
rm -f "$STATE_PATH"

export ATMOZERO_STATE_PATH="$STATE_PATH"
export ATMOZERO_N_WINDOWS="$N_WINDOWS"
export PYTHONPATH="${PYTHONPATH:-.}"

python -m verl.trainer.main_ppo \
    --config-path="$(realpath "$(dirname "$CONFIG")")" \
    --config-name="$(basename "$CONFIG" .yaml)" \
    "$@"
