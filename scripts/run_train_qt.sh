#!/usr/bin/env bash
set -euo pipefail

# Run PySide6 launcher for OpenGaussian training.
# Usage:
#   bash scripts/run_train_qt.sh
#   bash scripts/run_train_qt.sh /home/ysy/miniconda3/envs/opengaussian/bin/python

PYTHON_BIN="${1:-/home/ysy/miniconda3/envs/opengaussian/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QT_SCRIPT="${SCRIPT_DIR}/train_qt.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$QT_SCRIPT" ]]; then
  echo "[ERROR] script not found: $QT_SCRIPT" >&2
  exit 1
fi

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] QT_SCRIPT=$QT_SCRIPT"
export TRAIN_QT_PYTHON="$PYTHON_BIN"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$QT_SCRIPT"
