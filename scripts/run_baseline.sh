#!/usr/bin/env bash
# usage: scripts/run_baseline.sh <gpu> <train-config> <exp-name> [extra args]   (train + test inference + evaluation)
#   scripts/run_baseline.sh 0 configs/train/baseline_unet.yaml unet_r34_seed42
set -euo pipefail
GPU="$1"; CONFIG="$2"; EXP="$3"; shift 3
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
python -m medplib_bt.train_baseline --config "$CONFIG" --exp-name "$EXP" "$@"
python -m medplib_bt.infer_baseline --config "$CONFIG" --weights "outputs/runs/$EXP/final/model.pt" --out-dir "outputs/preds/$EXP" "$@"
python -m medplib_bt.evaluate --pred-dir "outputs/preds/$EXP"
