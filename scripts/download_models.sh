#!/usr/bin/env bash
# Download the pretrained weights. Idempotent (huggingface-cli skips finished files).
# huggingface.co is broken through the server's proxy but hf-mirror.com works directly,
# so the proxy is unset and HF_ENDPOINT points at the mirror.
#   MedPLIB-7b-2e (safetensors only, ~24 GB)  -> $HF_MODEL/MedPLIB-7b-2e
#   openai/clip-vit-large-patch14-336         -> $HF_MODEL/clip-vit-large-patch14-336
#   SAM-Med2D ViT-B checkpoint (sam-med2d_b.pth, mirror of the official Google-Drive file)
#                                             -> $HF_MODEL/SAM-Med2D/sam-med2d_b.pth
set -euo pipefail
HF_MODEL="${HF_MODEL:-/mnt/data/peilin/HF_MODEL}"
source /home/peilin/miniconda3/etc/profile.d/conda.sh
conda activate sigma   # any env with huggingface-cli
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
huggingface-cli download Huangxs/MedPLIB-7b-2e --local-dir "$HF_MODEL/MedPLIB-7b-2e" --exclude '*.bin' --max-workers 8
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir "$HF_MODEL/clip-vit-large-patch14-336" --exclude '*.h5' --max-workers 4
huggingface-cli download schengal1/SAM-Med2D_model sam-med2d_b.pth --local-dir "$HF_MODEL/SAM-Med2D"
du -sh "$HF_MODEL/MedPLIB-7b-2e" "$HF_MODEL/clip-vit-large-patch14-336" "$HF_MODEL/SAM-Med2D"
