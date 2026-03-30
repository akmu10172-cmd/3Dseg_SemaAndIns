#!/usr/bin/env bash
set -euo pipefail

# Install PySide6 for the OpenGaussian env.
# Usage:
#   bash scripts/install_train_qt_deps.sh
#   bash scripts/install_train_qt_deps.sh /home/ysy/miniconda3/envs/opengaussian/bin/python

PYTHON_BIN="${1:-/home/ysy/miniconda3/envs/opengaussian/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[INFO] Using python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install PySide6

echo "[DONE] PySide6 installed."

