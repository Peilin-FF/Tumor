#!/usr/bin/env bash
# usage: scripts/run_baseline_infer.sh <gpu> <config> <out-dir> [--weights ...] [extra args]   (inference + evaluation only)
#   scripts/run_baseline_infer.sh 2 configs/infer/sam_med2d_box.yaml outputs/preds/sam_med2d_oracle_box
set -euo pipefail
GPU="$1"; CONFIG="$2"; OUT="$3"; shift 3
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
python -m medplib_bt.infer_baseline --config "$CONFIG" --out-dir "$OUT" "$@"
python -m medplib_bt.evaluate --pred-dir "$OUT"
