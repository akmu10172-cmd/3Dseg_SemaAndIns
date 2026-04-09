#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from semantic_instance_pipeline import write_semantic_instance_ply, write_subset_ply


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine 3D instance labels: keep largest component + KNN fill unlabeled points.")
    p.add_argument("--input_ply", required=True)
    p.add_argument("--instance_npy", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--semantic_id", type=int, default=3)

    p.add_argument("--keep_largest_component", action="store_true")
    p.add_argument("--cc_knn", type=int, default=12)
    p.add_argument("--cc_eps_quantile", type=float, default=0.90)

    p.add_argument("--fill_unlabeled", action="store_true")
    p.add_argument("--fill_max_dist", type=float, default=-1.0, help="<0 means auto")
    p.add_argument("--fill_dist_factor", type=float, default=3.0, help="auto max dist factor")

    p.add_argument("--save_instance_parts", action="store_true")
    return p.parse_args()


def load_xyz_vertex(input_ply: Path):
    ply = PlyData.read(str(input_ply))
    vertex = np.array(ply["vertex"].data)
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    return vertex, xyz


def auto_eps(pts: np.ndarray, k: int, q: float) -> float:
    if len(pts) <= 2:
        return 0.0
    k = max(1, min(int(k), len(pts) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(pts)
    d, _ = nn.kneighbors(pts)
    kth = d[:, k]
    return float(np.quantile(kth, np.clip(q, 0.0, 1.0)))


def keep_largest_components(xyz: np.ndarray, labels: np.ndarray, k: int, q: float) -> tuple[np.ndarray, int]:
    out = labels.copy()
    removed = 0
    for inst in sorted(int(x) for x in np.unique(out) if int(x) > 0):
        idx = np.where(out == inst)[0]
        if len(idx) <= 2:
            continue
        pts = xyz[idx]
        eps = auto_eps(pts, k=k, q=q)
        if eps <= 0:
            continue
        cl = DBSCAN(eps=eps, min_samples=1, metric="euclidean", n_jobs=-1).fit_predict(pts)
        uniq, cnt = np.unique(cl, return_counts=True)
        keep = int(uniq[np.argmax(cnt)])
        drop_mask = cl != keep
        if np.any(drop_mask):
            out[idx[drop_mask]] = 0
            removed += int(np.sum(drop_mask))
    return out, removed


def fill_unlabeled_knn(xyz: np.ndarray, labels: np.ndarray, max_dist: float) -> tuple[np.ndarray, int]:
    out = labels.copy()
    unl = np.where(out == 0)[0]
    lbl = np.where(out > 0)[0]
    if len(unl) == 0 or len(lbl) == 0:
        return out, 0

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(xyz[lbl])
    d, ind = nn.kneighbors(xyz[unl])
    d = d[:, 0]
    ind = ind[:, 0]

    take = d <= float(max_dist)
    if np.any(take):
        out[unl[take]] = out[lbl[ind[take]]]
    return out, int(np.sum(take))


def relabel_contiguous(labels: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros_like(labels, dtype=np.int32)
    nid = 1
    for old in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        m = labels == old
        if np.any(m):
            out[m] = nid
            nid += 1
    return out, int(nid - 1)


def main() -> None:
    args = parse_args()
    input_ply = Path(args.input_ply)
    instance_npy = Path(args.instance_npy)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vertex, xyz = load_xyz_vertex(input_ply)
    labels = np.load(instance_npy).astype(np.int32)
    if len(labels) != len(xyz):
        raise ValueError(f"label length mismatch: labels={len(labels)} xyz={len(xyz)}")

    before_instances = int(len([x for x in np.unique(labels) if int(x) > 0]))
    before_labeled = int(np.sum(labels > 0))

    removed_cc = 0
    if args.keep_largest_component:
        labels, removed_cc = keep_largest_components(
            xyz=xyz,
            labels=labels,
            k=int(args.cc_knn),
            q=float(args.cc_eps_quantile),
        )

    auto_max_dist = None
    filled = 0
    if args.fill_unlabeled:
        if float(args.fill_max_dist) > 0:
            max_dist = float(args.fill_max_dist)
        else:
            labeled_idx = np.where(labels > 0)[0]
            if len(labeled_idx) > 1:
                pts = xyz[labeled_idx]
                nn = NearestNeighbors(n_neighbors=2)
                nn.fit(pts)
                d, _ = nn.kneighbors(pts)
                base = float(np.median(d[:, 1]))
                max_dist = float(base * float(args.fill_dist_factor))
            else:
                max_dist = 0.0
            auto_max_dist = max_dist

        labels, filled = fill_unlabeled_knn(
            xyz=xyz,
            labels=labels,
            max_dist=max_dist,
        )

    labels, ninst = relabel_contiguous(labels)

    np.save(out_dir / "point_instance_id_refined.npy", labels.astype(np.int32))
    out_ply = out_dir / "instance_projected_sem_ins_refined.ply"
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
        "instance_npy": str(instance_npy),
        "num_points": int(len(xyz)),
        "before": {
            "instances": int(before_instances),
            "labeled_points": int(before_labeled),
            "unlabeled_points": int(len(xyz) - before_labeled),
        },
        "after": {
            "instances": int(ninst),
            "labeled_points": int(np.sum(labels > 0)),
            "unlabeled_points": int(np.sum(labels == 0)),
        },
        "ops": {
            "keep_largest_component": bool(args.keep_largest_component),
            "removed_by_component": int(removed_cc),
            "fill_unlabeled": bool(args.fill_unlabeled),
            "filled_points": int(filled),
            "auto_fill_max_dist": auto_max_dist,
        },
        "params": {
            "semantic_id": int(args.semantic_id),
            "cc_knn": int(args.cc_knn),
            "cc_eps_quantile": float(args.cc_eps_quantile),
            "fill_max_dist": float(args.fill_max_dist),
            "fill_dist_factor": float(args.fill_dist_factor),
        },
    }
    (out_dir / "refine_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"instances_after": int(ninst), "filled": int(filled), "removed_cc": int(removed_cc), "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
