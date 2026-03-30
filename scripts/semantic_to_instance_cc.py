#!/usr/bin/env python3
"""Convert semantic *_s.npy masks to instance *_inst.npy via connected components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required. Install with: pip install opencv-python") from exc


def parse_int_list(text: str) -> List[int]:
    text = (text or "").strip()
    if not text:
        return []
    out: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def load_semantic_mask(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        return arr.astype(np.int32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0].astype(np.int32)
        return arr[0].astype(np.int32)
    raise ValueError(f"Unsupported mask shape {arr.shape} in {path}")


def semantic_to_instance(
    sem: np.ndarray,
    ignore_ids: List[int],
    min_area: int,
    connectivity: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    h, w = sem.shape
    inst = np.zeros((h, w), dtype=np.int32)
    ignore = set(ignore_ids)
    next_id = 1
    num_components = 0

    for cid in sorted(int(x) for x in np.unique(sem).tolist()):
        if cid in ignore:
            continue
        binary = (sem == cid).astype(np.uint8)
        if int(binary.sum()) == 0:
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=connectivity
        )
        for comp_id in range(1, int(num_labels)):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < int(min_area):
                continue
            inst[labels == comp_id] = next_id
            next_id += 1
            num_components += 1

    summary = {
        "num_instances": int(next_id - 1),
        "num_components_kept": int(num_components),
        "num_pixels_instance": int((inst > 0).sum()),
    }
    return inst, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate instance masks from semantic masks using connected components."
    )
    parser.add_argument("--src_dir", required=True, help="Directory containing semantic *_s.npy masks")
    parser.add_argument("--dst_dir", required=True, help="Directory to save instance *_inst.npy masks")
    parser.add_argument("--suffix", default="_s.npy", help="Input semantic mask suffix")
    parser.add_argument("--out_suffix", default="_inst.npy", help="Output instance mask suffix")
    parser.add_argument("--ignore_ids", default="0", help="Comma-separated semantic ids to ignore")
    parser.add_argument("--min_area", type=int, default=20, help="Minimum connected-component area")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8], help="Connected-components connectivity")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob(f"*{args.suffix}"))
    if not files:
        raise SystemExit(f"No files found in {src_dir} matching *{args.suffix}")

    ignore_ids = parse_int_list(args.ignore_ids)
    records: List[Dict[str, int | str]] = []
    saved = 0
    for p in files:
        sem = load_semantic_mask(p)
        inst, stat = semantic_to_instance(
            sem=sem,
            ignore_ids=ignore_ids,
            min_area=int(args.min_area),
            connectivity=int(args.connectivity),
        )
        stem = p.name[: -len(args.suffix)]
        out_path = dst_dir / f"{stem}{args.out_suffix}"
        np.save(out_path, inst.astype(np.int32))
        saved += 1
        records.append(
            {
                "input": str(p.name),
                "output": str(out_path.name),
                **stat,
            }
        )

    report = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "suffix": args.suffix,
        "out_suffix": args.out_suffix,
        "ignore_ids": ignore_ids,
        "min_area": int(args.min_area),
        "connectivity": int(args.connectivity),
        "files_total": int(len(files)),
        "files_saved": int(saved),
        "items": records[:200],
    }
    (dst_dir / "semantic_to_instance_cc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {saved}/{len(files)} instance masks -> {dst_dir}")


if __name__ == "__main__":
    main()
