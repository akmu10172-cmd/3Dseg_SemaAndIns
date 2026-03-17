import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Convert edited PNG masks back to *_s.npy format.")
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Directory containing edited PNG masks",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="Directory to save *_s.npy files (e.g., language_features)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_s.png",
        help="Mask png suffix to scan",
    )
    parser.add_argument(
        "--max_class_id",
        type=int,
        default=255,
        help="Maximum allowed class id in edited masks",
    )
    return parser.parse_args()


def image_to_index_array(path: Path):
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and arr.shape[2] >= 3:
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        if np.array_equal(r, g) and np.array_equal(r, b):
            return r
        raise ValueError(
            "RGB mask is not grayscale-index. "
            "Please edit as grayscale/index mask (no color-coded palette)."
        )

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(f"*{args.suffix}"))
    if not files:
        raise SystemExit(f"No files matched in {src_dir}: *{args.suffix}")

    saved = 0
    for png_path in files:
        try:
            arr = image_to_index_array(png_path).astype(np.int32)
        except Exception as exc:
            print(f"[skip] {png_path.name}: {exc}")
            continue

        min_id = int(arr.min())
        max_id = int(arr.max())
        if min_id < 0 or max_id > args.max_class_id:
            print(
                f"[skip] {png_path.name}: value range [{min_id}, {max_id}] "
                f"outside [0,{args.max_class_id}]"
            )
            continue

        out_name = png_path.name.replace(".png", ".npy")
        np.save(dst_dir / out_name, arr[np.newaxis, :, :])
        saved += 1

    print(f"done: {saved}/{len(files)} saved to {dst_dir}")


if __name__ == "__main__":
    main()
