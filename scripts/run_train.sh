#!/usr/bin/env bash
# usage: scripts/run_train.sh <gpus e.g. 0,1,2,3> <config> <exp-name> [extra args for medplib_bt.train]
#   scripts/run_train.sh 0,1,2,3,4,5,6,7 configs/train/stage_a.yaml stageA_seed42
#   scripts/run_train.sh 0,1,2,3,4,5,6,7 configs/train/stage_b.yaml stageB_seed42 --init-from outputs/runs/stageA_seed42/final
set -euo pipefail
GPUS="$1"; CONFIG="$2"; EXP="$3"; shift 3
source "$(dirname "$0")/env.sh"
PORT=${MASTER_PORT:-$((29500 + RANDOM % 500))}
deepspeed --include="localhost:$GPUS" --master_port "$PORT" --module medplib_bt.train --config "$CONFIG" --exp-name "$EXP" "$@"
