import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData
from torch import nn

from scene.gaussian_model import GaussianModel
from scene.colmap_loader import read_points3D_binary, read_points3D_text, rotmat2qvec


def build_training_args(args):
    # Minimal optimization args used by GaussianModel.training_setup(...)
    return SimpleNamespace(
        percent_dense=args.percent_dense,
        position_lr_init=args.position_lr_init,
        position_lr_final=args.position_lr_final,
        position_lr_delay_mult=args.position_lr_delay_mult,
        position_lr_max_steps=args.position_lr_max_steps,
        feature_lr=args.feature_lr,
        ins_feat_lr=args.ins_feat_lr,
        opacity_lr=args.opacity_lr,
        scaling_lr=args.scaling_lr,
        rotation_lr=args.rotation_lr,
        frozen_init_pts=args.frozen_init_pts,
    )


def resolve_sparse_points_path(path_like: str):
    p = Path(path_like)
    candidates = []
    if p.is_file():
        candidates.append(p)
    else:
        candidates.extend(
            [
                p / "points3D.bin",
                p / "points3D.txt",
                p / "sparse" / "0" / "points3D.bin",
                p / "sparse" / "0" / "points3D.txt",
                p / "sparse" / "points3D.bin",
                p / "sparse" / "points3D.txt",
                p / "0" / "points3D.bin",
                p / "0" / "points3D.txt",
            ]
        )

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find points3D.bin/txt from align_sparse_path={path_like}")


def load_sparse_points(path_like: str):
    p = resolve_sparse_points_path(path_like)
    if p.suffix.lower() == ".bin":
        xyz, _, _ = read_points3D_binary(str(p))
    else:
        xyz, _, _ = read_points3D_text(str(p))
    return xyz.astype(np.float64), p


def _eval_transform(src_sample, tgt_tree, R, s, t, rng, n_eval=50000, trim_quantile=0.85):
    if src_sample.shape[0] > n_eval:
        idx = rng.choice(src_sample.shape[0], size=n_eval, replace=False)
        X = src_sample[idx]
    else:
        X = src_sample
    Y = (s * (X @ R.T)) + t
    d, _ = tgt_tree.query(Y, k=1, workers=-1)
    cut = np.quantile(d, trim_quantile)
    d_trim = d[d <= cut]
    return {
        "trim_mean": float(d_trim.mean()),
        "median": float(np.median(d)),
        "mean": float(d.mean()),
    }


def _umeyama_similarity(X, Y):
    # Solve Y ≈ s R X + t
    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my

    cov = (Yc.T @ Xc) / X.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    var_x = np.mean(np.sum(Xc**2, axis=1))
    s = float(np.trace(np.diag(D) @ S) / (var_x + 1e-12))
    t = my - s * (R @ mx)
    return R, s, t


def estimate_similarity_transform(
    src_xyz,
    tgt_xyz,
    init_sample=120000,
    icp_sample=250000,
    icp_iters=12,
    trim_quantile=0.7,
    seed=42,
):
    # Returns transform that maps src -> tgt: X' = s * R * X + t
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    src_xyz = src_xyz.astype(np.float64)
    tgt_xyz = tgt_xyz.astype(np.float64)

    sid = rng.choice(src_xyz.shape[0], size=min(init_sample, src_xyz.shape[0]), replace=False)
    tid = rng.choice(tgt_xyz.shape[0], size=min(init_sample, tgt_xyz.shape[0]), replace=False)
    src = src_xyz[sid]
    tgt = tgt_xyz[tid]

    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    Xs = src - mu_s
    Xt = tgt - mu_t

    Cs = (Xs.T @ Xs) / len(Xs)
    Ct = (Xt.T @ Xt) / len(Xt)
    ws, Us = np.linalg.eigh(Cs)
    wt, Ut = np.linalg.eigh(Ct)
    Us = Us[:, np.argsort(ws)[::-1]]
    Ut = Ut[:, np.argsort(wt)[::-1]]

    tgt_tree_init = cKDTree(tgt)

    # Enumerate axis permutations + sign flips for robust PCA init.
    I = np.eye(3, dtype=np.float64)
    perm_mats = []
    for p in itertools.permutations(range(3)):
        P = I[:, p]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            S = np.diag(signs)
            M = P @ S
            if np.linalg.det(M) > 0.5:
                perm_mats.append(M)

    best = None
    rs = np.linalg.norm(Xs, axis=1)
    rt = np.linalg.norm(Xt, axis=1)
    s0 = float(np.median(rt) / (np.median(rs) + 1e-12))

    for M in perm_mats:
        R = Ut @ M @ Us.T
        t = mu_t - s0 * (R @ mu_s)
        met = _eval_transform(src, tgt_tree_init, R, s0, t, rng)
        key = (met["trim_mean"], met["median"], met["mean"])
        if best is None or key < best[0]:
            best = (key, R, s0, t, met)

    _, R, s, t, init_metrics = best

    # ICP refinement with similarity transform.
    iid = rng.choice(src_xyz.shape[0], size=min(icp_sample, src_xyz.shape[0]), replace=False)
    X = src_xyz[iid]
    tgt_tree_full = cKDTree(tgt_xyz)

    icp_trace = []
    for k in range(icp_iters):
        Y = (s * (X @ R.T)) + t
        d, nn = tgt_tree_full.query(Y, k=1, workers=-1)

        q = np.quantile(d, trim_quantile)
        m = d <= q
        if m.sum() < 128:
            break

        Rn, sn, tn = _umeyama_similarity(X[m], tgt_xyz[nn[m]])
        R, s, t = Rn, sn, tn

        icp_trace.append(
            {
                "iter": int(k + 1),
                "trim_mean": float(d[m].mean()),
                "trim_median": float(np.median(d[m])),
                "raw_median": float(np.median(d)),
            }
        )

    final_metrics = _eval_transform(src, tgt_tree_init, R, s, t, rng)

    return {
        "R": R,
        "s": float(s),
        "t": t,
        "init_metrics": init_metrics,
        "final_metrics": final_metrics,
        "icp_trace": icp_trace,
        "trim_quantile": float(trim_quantile),
    }


def apply_similarity_to_gaussians(gaussians, R, s, t):
    # Apply X' = s * R * X + t to geometry-related params.
    with torch.no_grad():
        dev = gaussians._xyz.device
        R_t = torch.tensor(R, dtype=torch.float32, device=dev)
        t_t = torch.tensor(t, dtype=torch.float32, device=dev)

        # xyz
        xyz = gaussians._xyz.data
        gaussians._xyz.data = s * (xyz @ R_t.T) + t_t

        # scaling is log-space in this repo: scale_world = exp(_scaling)
        gaussians._scaling.data = gaussians._scaling.data + float(np.log(s))

        # rotation quaternion: q' = q_g * q (left multiplication)
        qg = rotmat2qvec(R)  # [w, x, y, z]
        qg_t = torch.tensor(qg, dtype=torch.float32, device=dev)

        q = gaussians._rotation.data
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        W, X, Y, Z = qg_t[0], qg_t[1], qg_t[2], qg_t[3]

        out_w = W * qw - X * qx - Y * qy - Z * qz
        out_x = W * qx + X * qw + Y * qz - Z * qy
        out_y = W * qy - X * qz + Y * qw + Z * qx
        out_z = W * qz + X * qy - Y * qx + Z * qw
        q_new = torch.stack([out_w, out_x, out_y, out_z], dim=1)
        q_new = q_new / (q_new.norm(dim=1, keepdim=True) + 1e-12)
        gaussians._rotation.data = q_new



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a trainable OpenGaussian checkpoint from point_cloud.ply. "
            "The output can be used by train.py --start_checkpoint."
        )
    )
    parser.add_argument("--input_ply", required=True, help="Path to point_cloud.ply")
    parser.add_argument("--output_pth", default="", help="Output checkpoint path, e.g. .../chkpnt30000.pth")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration index saved in checkpoint tuple")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree used by GaussianModel")
    parser.add_argument(
        "--spatial_lr_scale",
        type=float,
        default=1.0,
        help="spatial_lr_scale in checkpoint (affects xyz lr after resume)",
    )

    # Optional: align local PLY to target sparse coords before writing checkpoint.
    parser.add_argument(
        "--align_sparse_path",
        default="",
        help=(
            "Optional path to sparse model root/file for alignment. "
            "Examples: scene_root, scene_root/sparse/0, .../points3D.bin"
        ),
    )
    parser.add_argument("--align_init_sample", type=int, default=120000, help="Sample size for PCA init")
    parser.add_argument("--align_icp_sample", type=int, default=250000, help="Sample size for ICP")
    parser.add_argument("--align_icp_iters", type=int, default=12, help="ICP iteration count")
    parser.add_argument("--align_trim_quantile", type=float, default=0.7, help="Trim quantile for ICP inlier set")
    parser.add_argument("--align_seed", type=int, default=42)
    parser.add_argument(
        "--align_transform_json",
        default="",
        help="Optional output json for solved transform and metrics (default: alongside output_pth)",
    )

    # Optimization defaults aligned with arguments/OptimizationParams.
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30000)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--ins_feat_lr", type=float, default=0.001)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)
    parser.add_argument("--frozen_init_pts", action="store_true", default=False)
    args = parser.parse_args()

    in_ply = Path(args.input_ply)
    if not in_ply.exists():
        raise FileNotFoundError(f"input_ply not found: {in_ply}")

    if args.output_pth:
        out_pth = Path(args.output_pth)
    else:
        out_pth = in_ply.parent / f"chkpnt{int(args.iteration)}.pth"
    out_pth.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required in this repo to load/save GaussianModel tensors.")

    gaussians = GaussianModel(args.sh_degree)
    try:
        gaussians.load_ply(str(in_ply))
    except Exception as e:
        # Fallback for simplified PLYs (e.g., only x/y/z + f_dc + opacity + scale + rot)
        # by synthesizing missing fields (f_rest and ins_feat).
        ply = PlyData.read(str(in_ply))
        names = {p.name for p in ply.elements[0].properties}

        xyz = np.stack(
            (
                np.asarray(ply.elements[0]["x"]),
                np.asarray(ply.elements[0]["y"]),
                np.asarray(ply.elements[0]["z"]),
            ),
            axis=1,
        )

        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        for i in range(3):
            k = f"f_dc_{i}"
            if k in names:
                features_dc[:, i, 0] = np.asarray(ply.elements[0][k], dtype=np.float32)

        extra_names = sorted(
            [n for n in names if n.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        expect_extra = 3 * ((args.sh_degree + 1) ** 2) - 3
        features_extra = np.zeros((xyz.shape[0], expect_extra), dtype=np.float32)
        fill_n = min(expect_extra, len(extra_names))
        for idx in range(fill_n):
            features_extra[:, idx] = np.asarray(ply.elements[0][extra_names[idx]], dtype=np.float32)
        features_extra = features_extra.reshape((xyz.shape[0], 3, (args.sh_degree + 1) ** 2 - 1))

        if "opacity" in names:
            opacities = np.asarray(ply.elements[0]["opacity"], dtype=np.float32)[..., np.newaxis]
        else:
            opacities = np.full((xyz.shape[0], 1), -2.1972246, dtype=np.float32)  # inverse_sigmoid(0.1)

        scale_names = sorted(
            [n for n in names if n.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        for idx, key in enumerate(scale_names[:3]):
            scales[:, idx] = np.asarray(ply.elements[0][key], dtype=np.float32)

        rot_names = sorted(
            [n for n in names if n.startswith("rot_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        if len(rot_names) >= 4:
            for idx in range(4):
                rots[:, idx] = np.asarray(ply.elements[0][rot_names[idx]], dtype=np.float32)
        else:
            rots[:, 0] = 1.0

        if {"ins_feat_r", "ins_feat_g", "ins_feat_b", "ins_feat_r2", "ins_feat_g2", "ins_feat_b2"}.issubset(names):
            ins_feat = np.stack(
                (
                    np.asarray(ply.elements[0]["ins_feat_r"], dtype=np.float32),
                    np.asarray(ply.elements[0]["ins_feat_g"], dtype=np.float32),
                    np.asarray(ply.elements[0]["ins_feat_b"], dtype=np.float32),
                    np.asarray(ply.elements[0]["ins_feat_r2"], dtype=np.float32),
                    np.asarray(ply.elements[0]["ins_feat_g2"], dtype=np.float32),
                    np.asarray(ply.elements[0]["ins_feat_b2"], dtype=np.float32),
                ),
                axis=1,
            )
        else:
            ins_feat = np.random.uniform(-0.01, 0.01, size=(xyz.shape[0], 6)).astype(np.float32)

        gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        gaussians._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)
        )
        gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians._ins_feat = nn.Parameter(torch.tensor(ins_feat, dtype=torch.float, device="cuda").requires_grad_(True))
        gaussians.active_sh_degree = args.sh_degree

        print("warning=fallback_loader_used")
        print(f"warning_reason={type(e).__name__}: {e}")

    # Optional: align to sparse points before creating optimizer/checkpoint.
    align_report = None
    sparse_points_path = None
    if args.align_sparse_path:
        src_xyz = gaussians.get_xyz.detach().cpu().numpy()
        tgt_xyz, sparse_points_path = load_sparse_points(args.align_sparse_path)
        fit = estimate_similarity_transform(
            src_xyz=src_xyz,
            tgt_xyz=tgt_xyz,
            init_sample=args.align_init_sample,
            icp_sample=args.align_icp_sample,
            icp_iters=args.align_icp_iters,
            trim_quantile=args.align_trim_quantile,
            seed=args.align_seed,
        )

        apply_similarity_to_gaussians(gaussians, fit["R"], fit["s"], fit["t"])

        align_report = {
            "align_sparse_path": str(args.align_sparse_path),
            "resolved_sparse_points": str(sparse_points_path),
            "transform_local_to_sparse": {
                "scale": float(fit["s"]),
                "R": fit["R"].tolist(),
                "t": fit["t"].tolist(),
            },
            "metrics": {
                "init": fit["init_metrics"],
                "final": fit["final_metrics"],
                "icp_trace": fit["icp_trace"],
                "trim_quantile": fit["trim_quantile"],
            },
        }

        if args.align_transform_json:
            out_json = Path(args.align_transform_json)
        else:
            out_json = out_pth.with_suffix(".align.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(align_report, f, ensure_ascii=False, indent=2)

        print("align=enabled")
        print(f"align_sparse_points={sparse_points_path}")
        print(
            "align_metrics="
            f"init_trim_mean={fit['init_metrics']['trim_mean']:.6f},"
            f"final_trim_mean={fit['final_metrics']['trim_mean']:.6f},"
            f"final_median={fit['final_metrics']['median']:.6f}"
        )
        print(f"align_json={out_json}")

    gaussians.spatial_lr_scale = float(args.spatial_lr_scale)

    training_args = build_training_args(args)
    gaussians.training_setup(training_args)

    n_pts = gaussians.get_xyz.shape[0]
    gaussians.max_radii2D = torch.zeros((n_pts,), device="cuda")
    gaussians.xyz_gradient_accum = torch.zeros((n_pts, 1), device="cuda")
    gaussians.denom = torch.zeros((n_pts, 1), device="cuda")

    model_params = gaussians.capture()
    checkpoint = (model_params, int(args.iteration))
    torch.save(checkpoint, str(out_pth))

    print("done")
    print(f"input_ply={in_ply}")
    print(f"output_pth={out_pth}")
    print(f"iteration={int(args.iteration)}")
    print(f"num_points={int(n_pts)}")
    if align_report is not None:
        print("note=checkpoint was aligned to sparse coordinates before save")
    else:
        print("note=alignment disabled")
    print("note=optimizer state is newly initialized, not recovered from original training run")


if __name__ == "__main__":
    main()
