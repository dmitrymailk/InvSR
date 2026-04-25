#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCHIVE_PATH="${REALSR_ARCHIVE:-$SCRIPT_DIR/data/RealSR (ICCV2019).tar.gz}"
RAW_ROOT="${REALSR_RAW_ROOT:-$SCRIPT_DIR/data/RealSR-raw/RealSR (ICCV2019)}"
PREPARED_ROOT="${REALSR_PREPARED_ROOT:-$SCRIPT_DIR/data/RealSR-prepared}"
TRAIN_OUT="$PREPARED_ROOT/train_hr"
VAL_LQ_OUT="$PREPARED_ROOT/val_lq_x4_native_lr_mod8"
VAL_GT_OUT="$PREPARED_ROOT/val_gt_x4_native_hr_mod8"

if [[ ! -d "$RAW_ROOT" ]]; then
  if [[ ! -f "$ARCHIVE_PATH" ]]; then
    echo "RealSR raw dataset not found." >&2
    echo "Expected extracted root: $RAW_ROOT" >&2
    echo "Expected archive: $ARCHIVE_PATH" >&2
    echo "Download the official RealSR archive and place it there, or set REALSR_ARCHIVE/REALSR_RAW_ROOT." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$RAW_ROOT")"
  tar -xzf "$ARCHIVE_PATH" -C "$(dirname "$RAW_ROOT")"
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python interpreter not found." >&2
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/scripts/prepare_realsr_v3_working_dataset.py" \
  --raw-root "$RAW_ROOT" \
  --train-out "$TRAIN_OUT" \
  --val-lq-out "$VAL_LQ_OUT" \
  --val-gt-out "$VAL_GT_OUT"