#!/usr/bin/env bash
# usage: scripts/run_infer.sh <gpus> <out-dir> (--adapter <dir> | --no-adapter) [--subset test] [extra args]
#   scripts/run_infer.sh 0,1,2,3 outputs/preds/stageB_seed42 --adapter outputs/runs/stageB_seed42/final
set -euo pipefail
GPUS="$1"; OUT="$2"; shift 2
source "$(dirname "$0")/env.sh"
PORT=${MASTER_PORT:-$((29500 + RANDOM % 500))}
deepspeed --include="localhost:$GPUS" --master_port "$PORT" --module medplib_bt.infer --out-dir "$OUT" "$@"
python -m medplib_bt.evaluate --pred-dir "$OUT"
