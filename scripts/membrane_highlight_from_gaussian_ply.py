#!/usr/bin/env python3
"""Build a membrane-like red highlight PLY from a Gaussian point PLY.

The output keeps the original vertex dtype/fields (Supersplat-friendly),
while replacing point positions with voxel-shell centers and forcing a red look.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_closing, binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree


C0 = 0.28209479177387814


def rgb_to_sh_dc(rgb01: np.ndarray) -> np.ndarray:
    return (rgb01 - 0.5) / C0


def estimate_voxel_size(xyz: np.ndarray) -> float:
    rng = np.random.default_rng(42)
    n = len(xyz)
    take = min(n, 30000)
    idx = rng.choice(n, size=take, replace=False)
    sample = xyz[idx]
    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=2, workers=-1)
    nn = d[:, 1]
    # Slightly larger than local spacing to bridge small gaps.
    return float(max(np.quantile(nn, 0.75) * 1.0, np.median(nn) * 1.3, 1e-3))


def build_shell_points(
    xyz: np.ndarray,
    voxel_size: float,
    closing_iters: int,
    max_voxels: int = 80_000_000,
) -> tuple[np.ndarray, float]:
    mins = xyz.min(axis=0)

    # Keep memory bounded by adaptively enlarging voxel size when needed.
    vs = float(voxel_size)
    while True:
        ijk = np.floor((xyz - mins) / vs).astype(np.int32)
        max_ijk = ijk.max(axis=0)
        dims = max_ijk + 3  # +padding
        total = int(dims[0]) * int(dims[1]) * int(dims[2])
        if total <= max_voxels:
            break
        vs *= 1.25

    occ = np.zeros(dims, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    st = generate_binary_structure(3, 1)
    filled = binary_closing(occ, structure=st, iterations=int(max(closing_iters, 1)))
    eroded = binary_erosion(filled, structure=st, iterations=1)
    shell = filled & (~eroded)

    shell_idx = np.argwhere(shell).astype(np.float32)
    shell_xyz = (shell_idx + 0.5) * vs + mins
    return shell_xyz, vs


def uniform_resample_grid(xyz: np.ndarray, radius: float) -> np.ndarray:
    """Approximate uniform resampling by keeping one point per grid cell."""
    r = float(max(radius, 1e-6))
    mins = xyz.min(axis=0)
    ijk = np.floor((xyz - mins) / r).astype(np.int32)
    # deterministic keep-first per cell
    _, first_idx = np.unique(ijk, axis=0, return_index=True)
    first_idx = np.sort(first_idx)
    return xyz[first_idx]


def main() -> None:
    p = argparse.ArgumentParser(description="Create membrane-style red highlight Gaussian PLY.")
    p.add_argument("--input_ply", required=True)
    p.add_argument("--output_ply", required=True)
    p.add_argument("--voxel_size", type=float, default=0.0, help="<=0 means auto-estimate")
    p.add_argument(
        "--uniform_resample_radius",
        type=float,
        default=0.0,
        help=">0 enables uniform resampling before voxelization (grid radius in world units)",
    )
    p.add_argument("--closing_iters", type=int, default=2)
    p.add_argument("--max_points", type=int, default=220000, help="Randomly cap shell points")
    p.add_argument("--opacity_logit", type=float, default=2.0, help="Set/raise opacity logits to this value")
    p.add_argument("--scale_factor", type=float, default=0.55, help="scale_i := log(voxel_size * scale_factor)")
    args = p.parse_args()

    in_p = Path(args.input_ply)
    out_p = Path(args.output_ply)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(str(in_p))
    v = ply["vertex"].data
    names = list(v.dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("Input PLY must contain x/y/z.")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    xyz_shell = xyz
    if float(args.uniform_resample_radius) > 0:
        xyz_shell = uniform_resample_grid(xyz, float(args.uniform_resample_radius))

    voxel = float(args.voxel_size) if float(args.voxel_size) > 0 else estimate_voxel_size(xyz_shell)
    shell_xyz, voxel_used = build_shell_points(xyz_shell, voxel_size=voxel, closing_iters=args.closing_iters)

    if len(shell_xyz) == 0:
        raise RuntimeError("No shell points generated. Try smaller voxel size.")

    if args.max_points > 0 and len(shell_xyz) > int(args.max_points):
        rng = np.random.default_rng(123)
        keep = rng.choice(len(shell_xyz), size=int(args.max_points), replace=False)
        shell_xyz = shell_xyz[keep]

    tree = cKDTree(xyz)
    _, nn_idx = tree.query(shell_xyz, k=1, workers=-1)

    out = np.empty(len(shell_xyz), dtype=v.dtype)
    for n in names:
        out[n] = v[n][nn_idx]

    out["x"] = shell_xyz[:, 0].astype(out["x"].dtype)
    out["y"] = shell_xyz[:, 1].astype(out["y"].dtype)
    out["z"] = shell_xyz[:, 2].astype(out["z"].dtype)

    # Force red highlight in both rgb and SH-DC channels.
    sh_dc = rgb_to_sh_dc(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    if "f_dc_0" in names:
        out["f_dc_0"] = np.float32(sh_dc[0])
    if "f_dc_1" in names:
        out["f_dc_1"] = np.float32(sh_dc[1])
    if "f_dc_2" in names:
        out["f_dc_2"] = np.float32(sh_dc[2])
    if "red" in names:
        out["red"] = np.uint8(255)
    if "green" in names:
        out["green"] = np.uint8(0)
    if "blue" in names:
        out["blue"] = np.uint8(0)

    # Increase visual presence.
    if "opacity" in names:
        out["opacity"] = np.maximum(out["opacity"].astype(np.float32), np.float32(args.opacity_logit))
    target_scale = np.float32(np.log(max(voxel_used * float(args.scale_factor), 1e-4)))
    for k in ("scale_0", "scale_1", "scale_2"):
        if k in names:
            out[k] = target_scale

    # Use identity rotations for a cleaner shell.
    if "rot_0" in names:
        out["rot_0"] = np.float32(1.0)
    for k in ("rot_1", "rot_2", "rot_3"):
        if k in names:
            out[k] = np.float32(0.0)

    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(out_p))
    print(f"input={in_p}")
    print(f"output={out_p}")
    print(f"in_points={len(v)}")
    print(f"shell_seed_points={len(xyz_shell)}")
    print(f"out_points={len(out)}")
    print(f"voxel_size={voxel_used:.6f}")


if __name__ == "__main__":
    main()
