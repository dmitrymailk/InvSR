#!/usr/bin/env bash
set -euo pipefail

# Этот скрипт запускает локальный mini-real train для InvSR.
# Он создан как воспроизводимый entrypoint для конфига
# configs/sd-turbo-sr-ldis-mini-real.yaml, чтобы не запускать train вручную.
#
# Что делает скрипт:
# - активирует локальное окружение .venv;
# - запускает train на одном GPU через torch.distributed.run;
# - пишет каждый запуск в отдельную директорию внутри save_dir_mini_real_runs.
#
# Полезные переменные окружения:
# - CUDA_VISIBLE_DEVICES: какой GPU использовать, по умолчанию 0;
# - SAVE_ROOT: корневая папка для артефактов текущего запуска.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
export NO_ALBUMENTATIONS_UPDATE=1
. .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SAVE_ROOT="${SAVE_ROOT:-./save_dir_mini_real_runs/$(date +%Y-%m-%d-%H-%M-%S)}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=1 \
  --nnodes=1 \
  main.py \
  --cfg_path ./configs/sd-turbo-sr-ldis-mini-real.yaml \
  --save_dir "$SAVE_ROOT" \
  "$@"