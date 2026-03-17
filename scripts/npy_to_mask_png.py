import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = [0] * (256 * 3)
# id -> RGB
_id2rgb = {
    0: (0, 0, 0),        # background
    1: (255, 0, 0),      # vehicle
    2: (0, 255, 0),      # person
    3: (255, 128, 0),    # bicycle
    4: (34, 139, 34),    # vegetation
    5: (255, 180, 105),  # reserved
    6: (255, 0, 255),    # traffic facility
    7: (128, 200, 200),  # others
}
for _i, (r, g, b) in _id2rgb.items():
    PALETTE[_i * 3:_i * 3 + 3] = [r, g, b]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert *_s.npy masks to editable PNG masks.")
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Directory containing *_s.npy files (e.g., language_features)",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="Directory to save PNG masks",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_s.npy",
        help="Mask npy suffix to scan",
    )
    parser.add_argument(
        "--palette_png",
        action="store_true",
        help="Save palette-colored indexed PNG instead of grayscale",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(f"*{args.suffix}"))
    if not files:
        raise SystemExit(f"No files matched in {src_dir}: *{args.suffix}")

    saved = 0
    for npy_path in files:
        arr = np.load(npy_path)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 2:
            print(f"[skip] {npy_path.name}: unsupported shape {arr.shape}")
            continue

        max_id = int(arr.max())
        min_id = int(arr.min())
        if min_id < 0 or max_id > 255:
            print(f"[skip] {npy_path.name}: value range [{min_id}, {max_id}] not in [0,255]")
            continue

        if args.palette_png:
            img = Image.fromarray(arr.astype(np.uint8), mode="P")
            img.putpalette(PALETTE)
        else:
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
        out_name = npy_path.name.replace(".npy", ".png")
        img.save(dst_dir / out_name)
        saved += 1

    print(f"done: {saved}/{len(files)} saved to {dst_dir}")


if __name__ == "__main__":
    main()
