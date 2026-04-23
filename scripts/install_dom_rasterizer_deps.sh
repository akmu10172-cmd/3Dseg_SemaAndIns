#!/usr/bin/env bash
set -euo pipefail

# Install DOM orthographic rasterizer from local source tree.
# Usage:
#   bash scripts/install_dom_rasterizer_deps.sh
#   bash scripts/install_dom_rasterizer_deps.sh /home/ysy/miniconda3/envs/opengaussian/bin/python

PYTHON_BIN="${1:-/home/ysy/miniconda3/envs/opengaussian/bin/python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOM_SRC="$ROOT_DIR/submodules/dom-rasterizer"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$DOM_SRC" ]]; then
  echo "[ERROR] dom-rasterizer source not found: $DOM_SRC" >&2
  exit 1
fi

export PATH="/usr/local/cuda/bin:/usr/bin:/bin:$(dirname "$PYTHON_BIN"):$PATH"
export CUDA_HOME="/usr/local/cuda"

if [[ -x /usr/bin/gcc-11 && -x /usr/bin/g++-11 ]]; then
  export CC=/usr/bin/gcc-11
  export CXX=/usr/bin/g++-11
fi

echo "[INFO] Using python: $PYTHON_BIN"
echo "[INFO] Source: $DOM_SRC"
"$PYTHON_BIN" -m pip install --no-build-isolation "$DOM_SRC"

echo "[DONE] dom_gaussian_rasterization installed."

