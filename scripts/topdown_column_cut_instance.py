#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from semantic_instance_pipeline import write_semantic_instance_ply, write_subset_ply


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Column-cut instances from one topdown instance mask (extrude along Z).")
    p.add_argument("--input_ply", required=True)
    p.add_argument("--pose_json", required=True)
    p.add_argument("--instance_mask_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--semantic_id", type=int, default=3)
    p.add_argument("--mask_image", default="auto", help="e.g. top_001.png; auto picks mask with most instances")
    p.add_argument("--min_instance_points", type=int, default=3000)
    p.add_argument("--mode", default="single_view", choices=["single_view", "all_views_fused"])
    p.add_argument("--fused_instance_npy", default="", help="point_instance_id.npy from multi-view fusion")
    p.add_argument("--xoy_stride_multiplier", type=float, default=1.0, help="XOY quantization stride multiplier")
    p.add_argument("--save_instance_parts", action="store_true")
    return p.parse_args()


def project_all_points(xyz: np.ndarray, w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float, w: int, h: int):
    xyz_h = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    cam = xyz_h @ w2c.T
    z = cam[:, 2]
    valid = z > 1e-6

    u = np.full((len(xyz),), -1, dtype=np.int32)
    v = np.full((len(xyz),), -1, dtype=np.int32)
    if not np.any(valid):
        return u, v, valid

    x = cam[valid, 0]
    y = cam[valid, 1]
    zz = cam[valid, 2]
    uu = np.round(fx * (x / zz) + cx).astype(np.int32)
    vv = np.round(fy * (y / zz) + cy).astype(np.int32)
    inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)

    idx = np.where(valid)[0]
    idx_in = idx[inside]
    u[idx_in] = uu[inside]
    v[idx_in] = vv[inside]
    valid2 = (u >= 0) & (v >= 0)
    return u, v, valid2


def relabel_contiguous(labels: np.ndarray, min_points: int):
    out = np.zeros_like(labels, dtype=np.int32)
    nid = 1
    for old in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        m = labels == old
        if int(np.sum(m)) < int(min_points):
            continue
        out[m] = nid
        nid += 1
    return out, int(nid - 1)


def infer_fused_instance_npy(mask_dir: Path) -> Path:
    # <pipe_root>/sam3_instance/instance_index_npy -> <pipe_root>/instance_3d/point_instance_id.npy
    pipe_root = mask_dir.parent.parent
    return pipe_root / "instance_3d" / "point_instance_id.npy"


def build_column_cut_from_fused(
    xyz: np.ndarray,
    fused_labels: np.ndarray,
    min_instance_points: int,
    xoy_stride_multiplier: float,
):
    if fused_labels.shape[0] != xyz.shape[0]:
        raise ValueError(f"fused_labels size mismatch: labels={len(fused_labels)} points={len(xyz)}")

    base_labels, _ = relabel_contiguous(fused_labels.astype(np.int32), min_points=max(1, int(min_instance_points)))
    if int(np.sum(base_labels > 0)) == 0:
        return np.zeros((len(xyz),), dtype=np.int32), 0

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    span_xy = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    base_step = max(span_xy / 2048.0, 1e-6)
    step = max(base_step * float(max(0.05, xoy_stride_multiplier)), 1e-6)

    qx = np.floor((xyz[:, 0] - mins[0]) / step).astype(np.int64)
    qy = np.floor((xyz[:, 1] - mins[1]) / step).astype(np.int64)
    key_all = (qx << 32) ^ (qy & 0xFFFFFFFF)

    unique_keys, inv = np.unique(key_all, return_inverse=True)
    owner = np.zeros((len(unique_keys),), dtype=np.int32)

    inst_ids = [int(x) for x in np.unique(base_labels) if int(x) > 0]
    inst_sizes = {inst: int(np.sum(base_labels == inst)) for inst in inst_ids}
    inst_ids = sorted(inst_ids, key=lambda i: inst_sizes[i], reverse=True)  # take larger first

    for inst in inst_ids:
        pos = np.unique(inv[base_labels == inst])
        free = owner[pos] == 0
        owner[pos[free]] = int(inst)

    extruded = owner[inv]
    extruded, ninst = relabel_contiguous(extruded, min_points=max(1, int(min_instance_points)))
    return extruded, int(ninst)


def main() -> None:
    args = parse_args()

    input_ply = Path(args.input_ply)
    pose_json = Path(args.pose_json)
    mask_dir = Path(args.instance_mask_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(str(input_ply))
    vertex = np.array(ply["vertex"].data)
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)

    poses = json.loads(pose_json.read_text(encoding="utf-8"))
    items = poses.get("items", [])
    if not items:
        raise RuntimeError(f"No camera items in {pose_json}")

    target_img = ""
    mode_used = str(args.mode)
    if str(args.mode) == "all_views_fused":
        fused_npy = Path(args.fused_instance_npy) if (args.fused_instance_npy or "").strip() else infer_fused_instance_npy(mask_dir)
        if not fused_npy.exists():
            raise FileNotFoundError(f"fused instance npy not found: {fused_npy}")
        fused_labels = np.load(fused_npy).astype(np.int32)
        labels, ninst = build_column_cut_from_fused(
            xyz=xyz,
            fused_labels=fused_labels,
            min_instance_points=int(args.min_instance_points),
            xoy_stride_multiplier=float(args.xoy_stride_multiplier),
        )
        target_img = "all_views_fused"
        mask_path = fused_npy
    else:
        summary_path = mask_dir.parent / "summary.json"
        best_name = None
        if args.mask_image == "auto" and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = summary.get("items", [])
            if rows:
                rows = sorted(rows, key=lambda r: int(r.get("num_instances", 0)), reverse=True)
                best_name = Path(rows[0]["image"]).name

        if args.mask_image != "auto":
            target_img = Path(args.mask_image).name
        elif best_name is not None:
            target_img = best_name
        else:
            target_img = Path(items[0]["image"]).name

        cam = None
        for it in items:
            if Path(it["image"]).name == target_img:
                cam = it
                break
        if cam is None:
            raise RuntimeError(f"Camera pose not found for image: {target_img}")

        mask_path = mask_dir / f"{Path(target_img).stem}_inst.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        inst2d = np.load(mask_path).astype(np.int32)
        h, w = inst2d.shape

        w2c = np.array(cam["w2c"], dtype=np.float64)
        fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])

        u, v, ok = project_all_points(xyz, w2c, fx, fy, cx, cy, w, h)
        labels = np.zeros((len(xyz),), dtype=np.int32)
        labels[ok] = inst2d[v[ok], u[ok]]

        labels, ninst = relabel_contiguous(labels, min_points=int(args.min_instance_points))

    np.save(out_dir / "point_instance_id_columncut.npy", labels.astype(np.int32))
    out_ply = out_dir / "instance_projected_sem_ins_columncut.ply"
    write_semantic_instance_ply(vertex, out_ply, int(args.semantic_id), labels)

    if args.save_instance_parts and ninst > 0:
        parts = out_dir / "instance_parts"
        parts.mkdir(parents=True, exist_ok=True)
        for inst in range(1, int(ninst) + 1):
            m = labels == inst
            if np.any(m):
                write_subset_ply(vertex[m], parts / f"instance_{inst:03d}.ply")

    report = {
        "input_ply": str(input_ply),
        "pose_json": str(pose_json),
        "instance_mask": str(mask_path),
        "mask_image": target_img,
        "num_points": int(len(xyz)),
        "num_instances_3d": int(ninst),
        "labeled_points": int(np.sum(labels > 0)),
        "unlabeled_points": int(np.sum(labels == 0)),
        "params": {
            "semantic_id": int(args.semantic_id),
            "min_instance_points": int(args.min_instance_points),
            "mode": mode_used,
            "xoy_stride_multiplier": float(args.xoy_stride_multiplier),
        },
    }
    (out_dir / "columncut_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"num_instances_3d": int(ninst), "mask_image": target_img, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
