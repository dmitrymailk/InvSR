#!/usr/bin/env bash
set -euo pipefail

# Этот скрипт запускает InvSR для sample_000217.png без внешнего разрезания на патчи
# и без tiled VAE, сохраняя итог в save_dir/sample_000217_no_patch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
. .venv/bin/activate

python inference_invsr.py \
  -i sample_000217.png \
  -o save_dir/sample_000217_no_patch \
  --num_steps 1 \
  --chopping_size 512 \
  --tiled_vae false