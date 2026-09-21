#!/usr/bin/env bash
# Full pipeline for one seed: stage A -> stage B -> test inference -> evaluation.
# usage: scripts/run_pipeline.sh <gpus> <seed>
set -euo pipefail
GPUS="$1"; SEED="${2:-42}"
cd "$(dirname "$0")/.."
scripts/run_train.sh "$GPUS" configs/train/stage_a.yaml "stageA_seed$SEED" --set seed="$SEED"
scripts/run_train.sh "$GPUS" configs/train/stage_b.yaml "stageB_seed$SEED" --set seed="$SEED" --init-from "outputs/runs/stageA_seed$SEED/final"
scripts/run_infer.sh "$GPUS" "outputs/preds/stageB_seed$SEED" --adapter "outputs/runs/stageB_seed$SEED/final" --subset test
