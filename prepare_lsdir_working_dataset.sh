#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
. .venv/bin/activate

python ./scripts/prepare_lsdir_working_dataset.py "$@"