import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement

try:
    from sklearn.cluster import KMeans
except ImportError as exc:
    raise SystemExit("scikit-learn is required. Install it with: pip install scikit-learn") from exc


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_ins_feat(vertex):
    prop_names = [p.name for p in vertex.properties]
    numeric = [name for name in prop_names if name.startswith("ins_feat_")]
    if numeric:
        def key_fn(name):
            try:
                return int(name.split("_")[-1])
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
    dmin = data.min(axis=0)
    dmax = data.max(axis=0)
    rng = dmax - dmin
    rng[rng == 0] = 1.0
    return (data - dmin) / rng


def build_feature_space(features, xyz, use_xyz, feature_weight, spatial_weight):
    feat_part = normalize_minmax(features) * feature_weight
    if not use_xyz:
        return feat_part
    xyz_part = normalize_minmax(xyz) * spatial_weight
    return np.hstack([feat_part, xyz_part])


def assign_by_centers(data, centers):
    # squared euclidean distance, vectorized
    d2 = (
        np.sum(data * data, axis=1, keepdims=True)
        + np.sum(centers * centers, axis=1).reshape(1, -1)
        - 2.0 * data @ centers.T
    )
    return np.argmin(d2, axis=1)


def save_cluster_ply(vertex, indices, out_path):
    out_vertex = np.empty(len(indices), dtype=vertex.data.dtype)
    for name in vertex.data.dtype.names:
        out_vertex[name] = vertex[name][indices]
    PlyData([PlyElement.describe(out_vertex, "vertex")], text=False).write(out_path)


def save_semantic_labeled_ply(vertex, indices, labels, out_path):
    old_dtype = vertex.data.dtype
    new_dtype = np.dtype(old_dtype.descr + [("semantic_id", "i4")])
    out_vertex = np.empty(len(indices), dtype=new_dtype)
    for name in old_dtype.names:
        out_vertex[name] = vertex[name][indices]
    out_vertex["semantic_id"] = labels.astype(np.int32)
    PlyData([PlyElement.describe(out_vertex, "vertex")], text=False).write(out_path)


def main():
    parser = argparse.ArgumentParser(description="K-Means semantic-style clustering on Gaussian point features.")
    parser.add_argument("--input_ply", required=True, help="Input PLY path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_clusters", type=int, default=8, help="Number of clusters (e.g., semantic classes)")
    parser.add_argument("--downsample_rate", type=int, default=1, help="Use every Nth valid point for fitting KMeans")
    parser.add_argument("--opacity_threshold", type=float, default=0.05, help="Opacity threshold after sigmoid")
    parser.add_argument("--scale_threshold_log", type=float, default=2.0, help="Scale threshold in log space")
    parser.add_argument("--feature_norm_threshold", type=float, default=None, help="Keep points with ||ins_feat|| >= th")
    parser.add_argument("--use_xyz", action="store_true", help="Concatenate XYZ with ins_feat for clustering")
    parser.add_argument("--feature_weight", type=float, default=1.0, help="Feature weight")
    parser.add_argument("--spatial_weight", type=float, default=7.0, help="XYZ weight (only when --use_xyz)")
    parser.add_argument("--assign_full", action="store_true", help="Assign all valid points by nearest center")
    parser.add_argument("--min_points", type=int, default=1, help="Minimum points to export one class PLY")
    parser.add_argument("--random_state", type=int, default=0, help="KMeans random seed")
    parser.add_argument("--n_init", type=int, default=10, help="KMeans n_init")
    parser.add_argument("--max_iter", type=int, default=300, help="KMeans max_iter")
    parser.add_argument("--save_npz", action="store_true", help="Save labels/centers metadata to .npz")
    parser.add_argument("--save_semantic_ply", action="store_true", help="Save one merged PLY with semantic_id field")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading PLY: {args.input_ply}")
    plydata = PlyData.read(args.input_ply)
    vertex = plydata["vertex"]

    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    features, feat_names = get_ins_feat(vertex)
    if features is None:
        raise SystemExit("No ins_feat fields found in PLY.")

    opacities = sigmoid(vertex["opacity"]) if "opacity" in vertex.data.dtype.names else np.ones(len(xyz))
    if all(n in vertex.data.dtype.names for n in ["scale_0", "scale_1", "scale_2"]):
        scales_log = np.max(np.vstack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]]).T, axis=1)
    else:
        scales_log = np.zeros(len(xyz), dtype=np.float32)

    mask = (opacities > args.opacity_threshold) & (scales_log < args.scale_threshold_log)
    if args.feature_norm_threshold is not None:
        mask &= np.linalg.norm(features, axis=1) >= args.feature_norm_threshold

    valid_indices = np.where(mask)[0]
    print(f"Points total: {len(xyz)}")
    print(f"Valid after filtering: {len(valid_indices)}")
    if len(valid_indices) == 0:
        raise SystemExit("No valid points after filtering.")

    if args.downsample_rate > 1:
        sample_indices = valid_indices[::args.downsample_rate]
    else:
        sample_indices = valid_indices
    print(f"Points for fitting KMeans: {len(sample_indices)}")

    if len(sample_indices) < args.n_clusters:
        raise SystemExit(
            f"sample points ({len(sample_indices)}) < n_clusters ({args.n_clusters}). "
            f"Lower n_clusters or relax filters."
        )

    sample_feats = features[sample_indices]
    sample_xyz = xyz[sample_indices]
    data_sample = build_feature_space(
        sample_feats,
        sample_xyz,
        use_xyz=args.use_xyz,
        feature_weight=args.feature_weight,
        spatial_weight=args.spatial_weight,
    )

    km = KMeans(
        n_clusters=args.n_clusters,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    labels_sample = km.fit_predict(data_sample)
    centers = km.cluster_centers_

    if args.assign_full:
        full_feats = features[valid_indices]
        full_xyz = xyz[valid_indices]
        data_full = build_feature_space(
            full_feats,
            full_xyz,
            use_xyz=args.use_xyz,
            feature_weight=args.feature_weight,
            spatial_weight=args.spatial_weight,
        )
        labels_save = assign_by_centers(data_full, centers)
        save_indices = valid_indices
    else:
        labels_save = labels_sample
        save_indices = sample_indices

    unique_labels = np.unique(labels_save)
    print(f"Clusters found: {len(unique_labels)}")

    for label in unique_labels:
        mask_c = labels_save == label
        count = int(np.sum(mask_c))
        if count < args.min_points:
            continue
        cluster_indices = save_indices[mask_c]
        out_path = os.path.join(args.output_dir, f"class_{int(label)}.ply")
        save_cluster_ply(vertex, cluster_indices, out_path)
        print(f"Saved {out_path} ({count} points)")

    if args.save_semantic_ply:
        sem_ply = os.path.join(args.output_dir, "semantic_labeled.ply")
        save_semantic_labeled_ply(vertex, save_indices, labels_save, sem_ply)
        print(f"Saved {sem_ply}")

    if args.save_npz:
        out_npz = os.path.join(args.output_dir, "kmeans_result.npz")
        np.savez_compressed(
            out_npz,
            labels=labels_save.astype(np.int32),
            indices=save_indices.astype(np.int64),
            centers=centers.astype(np.float32),
            feature_names=np.array(feat_names),
        )
        print(f"Saved {out_npz}")

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
