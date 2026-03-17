import argparse
import json
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_int_list(text: str):
    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("class_ids is empty")
    return vals


def normalize_rows(x: np.ndarray):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n < 1e-8] = 1.0
    return x / n


def get_npz_key(npz_obj, candidates):
    for k in candidates:
        if k in npz_obj:
            return npz_obj[k]
    raise KeyError(f"None of keys exist in npz: {candidates}")


def find_latest_iteration(point_cloud_dir: Path):
    iters = []
    for d in point_cloud_dir.glob("iteration_*"):
        if not d.is_dir():
            continue
        name = d.name
        try:
            iters.append(int(name.split("_")[-1]))
        except ValueError:
            pass
    if not iters:
        raise FileNotFoundError(f"No iteration_* directories under {point_cloud_dir}")
    return max(iters)


def load_prototypes_from_dir(language_feature_dir: Path, class_ids):
    files = sorted(language_feature_dir.glob("*_f.npy"))
    if not files:
        raise FileNotFoundError(f"No *_f.npy found in {language_feature_dir}")

    feat_dim = None
    feat_sum = None
    per_class_count = None
    max_rows_seen = -1
    for p in files:
        arr = np.load(p)
        if arr.ndim != 2:
            continue
        max_rows_seen = max(max_rows_seen, arr.shape[0] - 1)
        if feat_dim is None:
            feat_dim = arr.shape[1]
            feat_sum = np.zeros((len(class_ids), feat_dim), dtype=np.float64)
            per_class_count = np.zeros((len(class_ids),), dtype=np.int64)
        elif arr.shape[1] != feat_dim:
            # Skip malformed feature files with inconsistent feature dims.
            continue

        # Aggregate per class across files. A file may not contain all classes.
        curr = np.zeros((len(class_ids), feat_dim), dtype=np.float64)
        valid_mask = np.zeros((len(class_ids),), dtype=np.bool_)
        for i, cid in enumerate(class_ids):
            if 0 <= cid < arr.shape[0]:
                curr[i] = arr[cid].astype(np.float32)
                valid_mask[i] = True

        if feat_sum is None:
            feat_sum = np.zeros_like(curr, dtype=np.float64)
            per_class_count = np.zeros((len(class_ids),), dtype=np.int64)

        feat_sum[valid_mask] += curr[valid_mask]
        per_class_count[valid_mask] += 1

    if feat_sum is None or per_class_count is None:
        raise ValueError(
            f"No valid 2D *_f.npy found in {language_feature_dir}"
        )

    missing = [cid for cid, c in zip(class_ids, per_class_count.tolist()) if c == 0]
    if missing:
        raise ValueError(
            "Some class_ids never appear in *_f.npy. "
            f"missing={missing}, seen_id_range=[0,{max_rows_seen}] in {language_feature_dir}. "
            "Check your class_ids mapping (e.g. maybe 0-based ids like 0..7)."
        )

    proto = feat_sum / np.maximum(per_class_count[:, None], 1)
    return proto.astype(np.float32), int(len(files))


def load_prototypes_from_npy(path: Path, class_ids):
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"prototype_npy must be 2D, got {arr.shape} from {path}")
    if max(class_ids) >= arr.shape[0]:
        raise ValueError(
            f"class_ids max={max(class_ids)} out of range for prototype_npy shape={arr.shape}"
        )
    return arr[class_ids].astype(np.float32)


def write_labeled_ply(input_ply: Path, output_ply: Path, semantic_id, instance_id):
    ply = PlyData.read(str(input_ply))
    vertex = ply["vertex"].data
    if len(vertex) != len(semantic_id) or len(vertex) != len(instance_id):
        raise ValueError(
            f"Vertex count mismatch: ply={len(vertex)} sem={len(semantic_id)} ins={len(instance_id)}"
        )

    old_dtype = vertex.dtype
    new_dtype = np.dtype(old_dtype.descr + [("semantic_id", "i4"), ("instance_id", "i4")])
    out = np.empty(len(vertex), dtype=new_dtype)
    for name in old_dtype.names:
        out[name] = vertex[name]
    out["semantic_id"] = semantic_id.astype(np.int32)
    out["instance_id"] = instance_id.astype(np.int32)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(output_ply))


def main():
    parser = argparse.ArgumentParser(
        description="Export point-wise semantic and instance labels from OpenGaussian Stage3 outputs."
    )
    parser.add_argument("--model_path", required=True, help="OpenGaussian output path, e.g. /mnt/d/Scene/out_clip_full")
    parser.add_argument("--class_ids", required=True, help="Comma-separated semantic ids, e.g. 1,2,3,4,5,6,7,8")
    parser.add_argument(
        "--language_feature_dir",
        default=None,
        help="Directory containing *_f.npy. Used to build class prototypes by averaging.",
    )
    parser.add_argument(
        "--prototype_npy",
        default=None,
        help="Optional class-feature table npy [num_ids, feat_dim]. If set, overrides language_feature_dir.",
    )
    parser.add_argument("--background_id", type=int, default=0, help="Semantic id for invalid/unmatched leaves")
    parser.add_argument(
        "--min_occu_count",
        type=int,
        default=2,
        help="Leaves with occu_count < this are set to background",
    )
    parser.add_argument(
        "--min_leaf_score",
        type=float,
        default=0.0,
        help="Leaves with leaf_score < this are set to background",
    )
    parser.add_argument(
        "--reindex_instance",
        action="store_true",
        help="Reindex instance ids to contiguous [1..N] for non-background points",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration for labeled PLY output. -1 means latest iteration_*",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory, default: <model_path>/segmentation")
    parser.add_argument("--no_write_ply", action="store_true", help="Do not write labeled PLY")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    class_ids = parse_int_list(args.class_ids)
    output_dir = Path(args.output_dir) if args.output_dir else model_path / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_path = model_path / "cluster_lang.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing file: {cluster_path}")

    cluster = np.load(cluster_path)
    leaf_feat = get_npz_key(cluster, ["leaf_feat", "leaf_feat.npy"]).astype(np.float32)
    leaf_score = get_npz_key(cluster, ["leaf_score", "leaf_score.npy"]).astype(np.float32)
    occu_count = get_npz_key(cluster, ["occu_count", "occu_count.npy"]).astype(np.float32)
    leaf_ind = get_npz_key(cluster, ["leaf_ind", "leaf_ind.npy"]).astype(np.int64)

    if args.prototype_npy:
        class_proto = load_prototypes_from_npy(Path(args.prototype_npy), class_ids)
        proto_source = f"prototype_npy:{args.prototype_npy}"
        proto_count = 1
    else:
        if not args.language_feature_dir:
            raise ValueError("Provide either --prototype_npy or --language_feature_dir")
        class_proto, proto_count = load_prototypes_from_dir(Path(args.language_feature_dir), class_ids)
        proto_source = f"language_feature_dir:{args.language_feature_dir}"

    if class_proto.shape[1] != leaf_feat.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: class_proto={class_proto.shape[1]}, leaf_feat={leaf_feat.shape[1]}"
        )

    class_proto = normalize_rows(class_proto)
    leaf_feat_n = normalize_rows(leaf_feat)

    sim = class_proto @ leaf_feat_n.T
    best_class_idx = np.argmax(sim, axis=0)
    best_sim = np.max(sim, axis=0)
    class_ids_arr = np.array(class_ids, dtype=np.int32)
    leaf_semantic = class_ids_arr[best_class_idx].astype(np.int32)

    valid_leaf = (occu_count >= args.min_occu_count) & (leaf_score >= args.min_leaf_score)
    leaf_semantic[~valid_leaf] = np.int32(args.background_id)

    safe_leaf_ind = leaf_ind.copy()
    safe_leaf_ind = np.clip(safe_leaf_ind, 0, len(leaf_semantic) - 1)
    point_semantic = leaf_semantic[safe_leaf_ind]

    point_instance = safe_leaf_ind.astype(np.int32)
    point_instance[point_semantic == args.background_id] = 0
    if args.reindex_instance:
        fg = point_semantic != args.background_id
        uniq = np.unique(point_instance[fg])
        remap = {old: new_id for new_id, old in enumerate(uniq, start=1)}
        out = np.zeros_like(point_instance, dtype=np.int32)
        for old, new_id in remap.items():
            out[point_instance == old] = new_id
        point_instance = out

    np.save(output_dir / "point_semantic_id.npy", point_semantic.astype(np.int32))
    np.save(output_dir / "point_instance_id.npy", point_instance.astype(np.int32))
    np.save(output_dir / "leaf_to_semantic_id.npy", leaf_semantic.astype(np.int32))
    np.save(output_dir / "leaf_best_similarity.npy", best_sim.astype(np.float32))

    meta = {
        "model_path": str(model_path),
        "class_ids": class_ids,
        "background_id": int(args.background_id),
        "min_occu_count": int(args.min_occu_count),
        "min_leaf_score": float(args.min_leaf_score),
        "prototype_source": proto_source,
        "prototype_file_count": int(proto_count),
        "num_points": int(point_semantic.shape[0]),
        "num_leaves": int(leaf_semantic.shape[0]),
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if not args.no_write_ply:
        point_cloud_dir = model_path / "point_cloud"
        if args.iteration < 0:
            iteration = find_latest_iteration(point_cloud_dir)
        else:
            iteration = args.iteration
        in_ply = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
        out_ply = output_dir / f"point_cloud_sem_ins_iter_{iteration}.ply"
        if in_ply.exists():
            write_labeled_ply(in_ply, out_ply, point_semantic, point_instance)
        else:
            print(f"[warn] Skip PLY writing. File not found: {in_ply}")

    fg_points = int(np.sum(point_semantic != args.background_id))
    print("done")
    print(f"output_dir={output_dir}")
    print(f"points={len(point_semantic)}, foreground_points={fg_points}")


if __name__ == "__main__":
    main()
