#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_sam3_scene_0320.sh [INPUT_DIR] [OUTPUT_DIR] [CHECKPOINT] [DEVICE]
# Examples:
#   bash scripts/run_sam3_scene_0320.sh
#   bash scripts/run_sam3_scene_0320.sh "D:\\Scene_0320\\images_8"
#   bash scripts/run_sam3_scene_0320.sh "D:\\Scene_0320\\images_8" "D:\\Scene_0320\\sam3_masks_images_8" "C:\\Users\\ysy\\.cache\\modelscope\\hub\\models\\facebook\\sam3\\sam3.pt" cuda

INPUT_RAW="${1:-D:\\Scene_0320\\images_8}"
OUTPUT_RAW="${2:-D:\\Scene_0320\\sam3_masks_images_8}"
CHECKPOINT_RAW="${3:-C:\\Users\\ysy\\.cache\\modelscope\\hub\\models\\facebook\\sam3\\sam3.pt}"
DEVICE="${4:-cuda}"

SAM3_PY="/mnt/d/sam3/.conda/envs/sam3/bin/python"
SAM3_SCRIPT="/mnt/d/sam3/sam3/scripts/batch_sam3_dji_masks.py"

to_wsl_path() {
  local p="$1"
  # Convert backslashes first
  p="${p//\\\\//}"
  # Windows drive letter path: D:/xxx -> /mnt/d/xxx
  if [[ "$p" =~ ^([A-Za-z]):/(.*)$ ]]; then
    local drive
    drive="${BASH_REMATCH[1],,}"
    local rest
    rest="${BASH_REMATCH[2]}"
    echo "/mnt/${drive}/${rest}"
    return
  fi
  echo "$p"
}

INPUT_DIR="$(to_wsl_path "$INPUT_RAW")"
OUTPUT_DIR="$(to_wsl_path "$OUTPUT_RAW")"
CHECKPOINT_PATH="$(to_wsl_path "$CHECKPOINT_RAW")"

if [[ ! -x "$SAM3_PY" ]]; then
  echo "[ERROR] sam3 python not found: $SAM3_PY" >&2
  exit 1
fi
if [[ ! -f "$SAM3_SCRIPT" ]]; then
  echo "[ERROR] sam3 script not found: $SAM3_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "[ERROR] input dir not found: $INPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[INFO] INPUT_DIR=$INPUT_DIR"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "[INFO] DEVICE=$DEVICE"

"$SAM3_PY" "$SAM3_SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --no-hf-download \
  --device "$DEVICE"

echo "[DONE] SAM3 finished. Output: $OUTPUT_DIR"
