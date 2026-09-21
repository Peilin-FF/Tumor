# Sourced by the run scripts: conda env + offline HF + MedPLIB on the path.
source /home/peilin/miniconda3/etc/profile.d/conda.sh
conda activate medplib
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset all_proxy ALL_PROXY   # httpx (gradio) has no SOCKS support without extra packages
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}"
