#!/usr/bin/env bash
# Download BRISC 2025 (CC BY 4.0) from Kaggle into data/brisc2025. Idempotent.
# Needs a Kaggle API token in $KAGGLE_API_TOKEN or ~/.kaggle/access_token.
# kaggle.com is reachable directly from the server (not via the local proxy).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
mkdir -p "$DATA_DIR"
if [ -d "$DATA_DIR/brisc2025/classification_task" ]; then
  echo "BRISC already present at $DATA_DIR/brisc2025"; exit 0
fi
TOKEN="${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token 2>/dev/null || true)}"
[ -n "$TOKEN" ] || { echo "no Kaggle token (KAGGLE_API_TOKEN or ~/.kaggle/access_token)" >&2; exit 1; }
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
ZIP="$DATA_DIR/brisc2025.zip"
[ -s "$ZIP" ] || curl -L --fail --retry 5 --retry-delay 5 -C - \
  -H "Authorization: Bearer $TOKEN" -o "$ZIP" \
  https://www.kaggle.com/api/v1/datasets/download/briscdataset/brisc2025
rm -rf "$DATA_DIR/_brisc_extract"; mkdir -p "$DATA_DIR/_brisc_extract"
unzip -q "$ZIP" -d "$DATA_DIR/_brisc_extract"
# the archive contains a single top-level folder 'brisc2025/'
mv "$DATA_DIR/_brisc_extract/brisc2025" "$DATA_DIR/brisc2025"
rmdir "$DATA_DIR/_brisc_extract"
echo "done: $(find "$DATA_DIR/brisc2025" -name '*.jpg' | wc -l) images, $(find "$DATA_DIR/brisc2025" -name '*.png' | wc -l) masks"
