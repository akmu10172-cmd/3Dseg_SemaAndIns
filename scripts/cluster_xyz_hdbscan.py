import argparse
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

try:
    import hdbscan
except ImportError as exc:
    raise SystemExit("hdbscan is required. Install it in your environment.") from exc


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_xyz(value):
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("seed_xyz must be x,y,z")
    return np.array(parts, dtype=np.float32)


def parse_bbox(value):
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("seed_bbox must be xmin,xmax,ymin,ymax,zmin,zmax")
    return np.array(parts, dtype=np.float32)


def get_ins_feat(vertex):
    prop_names = [p.name for p in vertex.properties]
    numeric = [name for name in prop_names if name.startswith("ins_feat_")]
    if numeric:
        def key_fn(n):
            try:
                return int(n.split("_")[-1])
            except ValueError:
                return 0
        numeric = sorted(numeric, key=key_fn)
        feats = np.vstack([vertex[name] for name in numeric]).T
        return feats, numeric
    legacy = ["ins_feat_r", "ins_feat_g", "ins_feat_b", "ins_feat_r2", "ins_feat_g2", "ins_feat_b2"]
    if all(name in prop_names for name in legacy):
        feats = np.vstack([vertex[name] for name in legacy]).T
        return feats, legacy
    return None, []


def normalize_minmax(data):
    d_min = data.min(axis=0)
    d_max = data.max(axis=0)
    rng = d_max - d_min
    rng[rng == 0] = 1.0
    return (data - d_min) / rng


def build_semantic_reference(features, xyz, args):
    if args.semantic_ref is not None:
        ref = np.load(args.semantic_ref)
        if ref.ndim == 2:
            ref = ref.mean(axis=0)
        return ref.astype(np.float32)
    if args.seed_xyz is not None:
        deltas = xyz - args.seed_xyz.reshape(1, 3)
        dist2 = np.sum(deltas ** 2, axis=1)
        mask = dist2 <= args.seed_radius ** 2
        if not np.any(mask):
            raise SystemExit("No points found within seed_radius.")
        return features[mask].mean(axis=0)
    if args.seed_bbox is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = args.seed_bbox
        mask = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
        )
        if not np.any(mask):
            raise SystemExit("No points found inside seed_bbox.")
        return features[mask].mean(axis=0)
    return None


def main():
    parser = argparse.ArgumentParser(description="Cluster points with HDBSCAN using XYZ (optional semantic filtering).")
    parser.add_argument("--input_ply", required=True, help="Input PLY path")
    parser.add_argument("--output_dir", required=True, help="Output directory for clusters")
    parser.add_argument("--downsample_rate", type=int, default=1, help="Use every Nth point for clustering")
    parser.add_argument("--opacity_threshold", type=float, default=0.05, help="Opacity threshold after sigmoid")
    parser.add_argument("--scale_threshold_log", type=float, default=2.0, help="Scale threshold in log space")
    parser.add_argument("--min_cluster_size", type=int, default=300, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=None, help="HDBSCAN min_samples")
    parser.add_argument("--use_feature_cat", action="store_true", help="Concatenate features with XYZ for clustering")
    parser.add_argument("--feature_weight", type=float, default=1.0, help="Feature weight when use_feature_cat")
    parser.add_argument("--spatial_weight", type=float, default=7.0, help="XYZ weight when use_feature_cat")
    parser.add_argument("--semantic_ref", type=str, default=None, help="Path to .npy feature vector for target")
    parser.add_argument("--seed_xyz", type=parse_xyz, default=None, help="Seed xyz as x,y,z")
    parser.add_argument("--seed_radius", type=float, default=0.2, help="Radius for seed_xyz")
    parser.add_argument("--seed_bbox", type=parse_bbox, default=None, help="Seed bbox xmin,xmax,ymin,ymax,zmin,zmax")
    parser.add_argument("--semantic_threshold", type=float, default=0.2, help="Cosine similarity threshold")
    parser.add_argument("--feature_norm_threshold", type=float, default=None,
                        help="Fallback filter: keep points with ||ins_feat|| > threshold (when no semantic ref)")
    parser.add_argument("--assign_full", action="store_true", help="Assign labels to all valid points")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading PLY: {args.input_ply}")
    plydata = PlyData.read(args.input_ply)
    vertex = plydata["vertex"]

    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    opacities = sigmoid(vertex["opacity"]) if "opacity" in vertex.data.dtype.names else np.ones(len(xyz))
    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(name in vertex.data.dtype.names for name in scale_names):
        scales_log = np.max(np.vstack([vertex[n] for n in scale_names]).T, axis=1)
    else:
        scales_log = np.zeros(len(xyz), dtype=np.float32)

    mask = (opacities > args.opacity_threshold) & (scales_log < args.scale_threshold_log)

    features, feat_names = get_ins_feat(vertex)
    if args.semantic_ref or args.seed_xyz is not None or args.seed_bbox is not None:
        if features is None:
            raise SystemExit("No ins_feat fields found for semantic filtering.")
        feats_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
        ref = build_semantic_reference(feats_norm, xyz, args)
        ref = ref / (np.linalg.norm(ref) + 1e-6)
        sim = np.dot(feats_norm, ref)
        mask &= sim >= args.semantic_threshold
    elif args.feature_norm_threshold is not None:
        if features is None:
            raise SystemExit("No ins_feat fields found for feature norm filtering.")
        feat_norm = np.linalg.norm(features, axis=1)
        mask &= feat_norm >= args.feature_norm_threshold

    valid_indices = np.where(mask)[0]
    print(f"Points total: {len(xyz)}")
    print(f"Valid after filtering: {len(valid_indices)}")
    if len(valid_indices) == 0:
        raise SystemExit("No valid points after filtering; relax thresholds.")

    if args.downsample_rate > 1:
        sample_indices = valid_indices[::args.downsample_rate]
    else:
        sample_indices = valid_indices

    print(f"Clustering points: {len(sample_indices)}")
    xyz_sample = xyz[sample_indices]
    if args.use_feature_cat:
        if features is None:
            raise SystemExit("No ins_feat fields found for feature concatenation.")
        feat_sample = features[sample_indices]
        combined = np.hstack([
            normalize_minmax(feat_sample) * args.feature_weight,
            normalize_minmax(xyz_sample) * args.spatial_weight,
        ])
    else:
        combined = normalize_minmax(xyz_sample)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels_sample = clusterer.fit_predict(combined)

    if args.assign_full:
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            print("scipy not available; cannot assign full labels. Using sampled labels.")
            args.assign_full = False
        if args.assign_full:
            tree = cKDTree(xyz_sample)
            _, nn_idx = tree.query(xyz[valid_indices], k=1)
            labels_full = labels_sample[nn_idx]
        else:
            labels_full = labels_sample
    else:
        labels_full = labels_sample

    prop_names = [p.name for p in vertex.properties]
    if args.assign_full:
        save_indices = valid_indices
    else:
        save_indices = sample_indices

    labels_save = labels_full
    unique_labels = np.unique(labels_save)
    print(f"Clusters found: {len(unique_labels)} (including -1)")

    vertex_data_map = {name: vertex[name][save_indices] for name in prop_names}
    for label in unique_labels:
        if label == -1:
            continue
        mask_cluster = labels_save == label
        count = int(np.sum(mask_cluster))
        if count < args.min_cluster_size:
            continue
        out_vertex = np.empty(count, dtype=vertex.data.dtype)
        for name in prop_names:
            out_vertex[name] = vertex_data_map[name][mask_cluster]
        out_path = os.path.join(args.output_dir, f"obj_{label}.ply")
        PlyData([PlyElement.describe(out_vertex, "vertex")], text=False).write(out_path)
        print(f"Saved {out_path} ({count} points)")

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
