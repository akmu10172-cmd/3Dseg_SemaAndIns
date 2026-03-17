import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binarize semantic masks: all non-zero ids -> 1, zero stays 0."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Directory containing *_s.npy masks",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_s.npy",
        help="Filename suffix to match",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite source files in place",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="",
        help="Output directory when not using --inplace",
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        default="",
        help="Backup original files when using --inplace",
    )
    return parser.parse_args()


def to_1hw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    raise ValueError(f"Unsupported mask shape: {arr.shape}")


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    files = sorted(src_dir.glob(f"*{args.suffix}"))
    if not files:
        raise SystemExit(f"No masks found in {src_dir} with suffix {args.suffix}")

    if args.inplace:
        out_dir = src_dir
    else:
        if not args.dst_dir:
            raise SystemExit("When not using --inplace, --dst_dir is required.")
        out_dir = Path(args.dst_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = None
    if args.inplace and args.backup_dir:
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    for p in files:
        m = np.load(p)
        m = to_1hw(m)
        b = (m > 0).astype(np.int32)

        if backup_dir is not None:
            np.save(backup_dir / p.name, m.astype(np.int32))

        np.save(out_dir / p.name, b)
        done += 1

    print(f"done: {done}/{len(files)}")
    if backup_dir is not None:
        print(f"backup saved to: {backup_dir}")
    print(f"output dir: {out_dir}")


if __name__ == "__main__":
    main()
