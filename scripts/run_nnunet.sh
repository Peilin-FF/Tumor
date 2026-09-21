#!/usr/bin/env bash
# nnU-Net v2, 2D configuration, fold 0, 100-epoch trainer; then predicts the test split and imports the
# result into outputs/preds/<name> for evaluate.py.
# usage: scripts/run_nnunet.sh <gpu> [name=nnunet_2d] [trainer=nnUNetTrainer_100epochs]
set -euo pipefail
GPU="$1"; NAME="${2:-nnunet_2d}"; TRAINER="${3:-nnUNetTrainer_100epochs}"
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
export nnUNet_raw="$PWD/outputs/nnunet/raw" nnUNet_preprocessed="$PWD/outputs/nnunet/preprocessed" nnUNet_results="$PWD/outputs/nnunet/results"
export nnUNet_n_proc_DA=8 nnUNet_def_n_proc=8
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
[ -f "$nnUNet_raw/Dataset501_BRISC/dataset.json" ] || python -m medplib_bt.datasets.export_nnunet --raw-dir "$nnUNet_raw" --dataset-id 501
[ -d "$nnUNet_preprocessed/Dataset501_BRISC" ] || nnUNetv2_plan_and_preprocess -d 501 -c 2d -np 8 --verify_dataset_integrity
nnUNetv2_train 501 2d 0 -tr "$TRAINER" --npz
nnUNetv2_predict -i "$nnUNet_raw/Dataset501_BRISC/imagesTs" -o "outputs/nnunet/pred_${NAME}" -d 501 -c 2d -f 0 -tr "$TRAINER" --save_probabilities
python -m medplib_bt.import_nnunet --pred-dir "outputs/nnunet/pred_${NAME}" --out-dir "outputs/preds/${NAME}"
python -m medplib_bt.evaluate --pred-dir "outputs/preds/${NAME}"
