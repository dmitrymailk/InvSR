#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
. .venv/bin/activate

if [[ ! -d ./data/LSDIR-prepared/train_hr_flat ]]; then
  echo "Missing prepared LSDIR dataset. Run ./prepare_lsdir_working_dataset.sh first."
  exit 1
fi

if [[ ! -d ./data/LSDIR-prepared/val_lq_x4_mod8 || ! -d ./data/LSDIR-prepared/val_gt_x4_mod8 ]]; then
  echo "Missing prepared LSDIR validation folders. Run ./prepare_lsdir_working_dataset.sh first."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SAVE_ROOT="${SAVE_ROOT:-./save_dir/lsdir_x4_single_gpu/$(date +%Y-%m-%d-%H-%M-%S)}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  --nnodes=1 \
  main.py \
  --cfg_path ./configs/sd-turbo-sr-ldis-lsdir.yaml \
  --save_dir "$SAVE_ROOT" \
  "$@"