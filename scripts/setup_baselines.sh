#!/usr/bin/env bash
# Install the packages for the conventional baselines into the `medplib` env and pre-download the
# ImageNet encoder weights (timm, via the HF mirror). Writes logs/pip_baselines2.log; the last line
# is done=<pip exit>/<resnet34 weights exit>/<efficientnet weights exit>.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/pip_baselines2.log"; mkdir -p "$ROOT/logs"
source /home/peilin/miniconda3/etc/profile.d/conda.sh
conda activate medplib
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
pip install --timeout 30 --retries 20 -i https://mirrors.aliyun.com/pypi/simple \
  segmentation-models-pytorch==0.3.4 albumentations==1.4.3 nnunetv2==2.5.1 SimpleITK > "$LOG" 2>&1
pe=$?
# the baseline packages pull huggingface-hub >= 1.0, which transformers 4.31 (MedPLIB) cannot import; pin it back
pip install --timeout 30 -i https://mirrors.aliyun.com/pypi/simple "huggingface-hub==0.21.4" "numpy==1.26.4" >> "$LOG" 2>&1
python -c 'import timm, segmentation_models_pytorch, nnunetv2; print("timm", timm.__version__, "smp", segmentation_models_pytorch.__version__)' >> "$LOG" 2>&1
conda activate sigma
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download timm/resnet34.a1_in1k >> "$LOG" 2>&1; w1=$?
huggingface-cli download timm/efficientnet_b0.ra_in1k >> "$LOG" 2>&1; w2=$?
echo "done=$pe/$w1/$w2" >> "$LOG"
tail -3 "$LOG"
