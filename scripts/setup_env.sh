#!/usr/bin/env bash
# Create/refresh the `medplib` conda env on the server. Idempotent.
# PyPI is reachable directly from the server but not through the local proxy, so
# proxies are unset for pip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /home/peilin/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q '^medplib ' || conda create -n medplib python=3.10 -y
conda activate medplib
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
pip install --upgrade pip
pip install -r "$ROOT/requirements-medplib.txt"
python - <<'PY'
import torch, transformers, deepspeed, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "deepspeed", deepspeed.__version__, "peft", peft.__version__)
PY
