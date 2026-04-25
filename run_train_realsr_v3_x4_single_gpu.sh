#!/usr/bin/env bash
set -euo pipefail

# Этот скрипт запускает штатный train InvSR на реальном датасете RealSRV3,
# который уже подготовлен внутри репозитория в data/RealSR-prepared.
# Он нужен как отдельный воспроизводимый entrypoint для полного real-data
# сценария, чтобы не менять smoke/mini-real конфиги и не собирать команду вручную.
#
# Что использует:
# - configs/sd-turbo-sr-ldis-realsr.yaml
# - train: data/RealSR-prepared/train_hr
# - val: data/RealSR-prepared/val_lq_x4_native_lr_mod8 + data/RealSR-prepared/val_gt_x4_native_hr_mod8
#
# Полезные переменные окружения:
# - CUDA_VISIBLE_DEVICES: какой GPU использовать, по умолчанию 0
# - SAVE_ROOT: корневая папка для логов и чекпоинтов; по умолчанию создаётся
#   отдельный timestamped run внутри ./save_dir/realsr_v3_x4_single_gpu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
. .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SAVE_ROOT="${SAVE_ROOT:-./save_dir/realsr_v3_x4_single_gpu/$(date +%Y-%m-%d-%H-%M-%S)}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  --nnodes=1 \
  main.py \
  --cfg_path ./configs/sd-turbo-sr-ldis-realsr.yaml \
  --save_dir "$SAVE_ROOT" \
  "$@"