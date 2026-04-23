#!/usr/bin/env python3
"""Convert Gaussian PLY to a surface mesh through voxelized implicit occupancy.

This script keeps dependencies minimal (numpy/scipy/plyfile) and can run inside
the existing OpenGaussian environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, generate_binary_structure


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def adaptive_voxel_size(xyz: np.ndarray, voxel_size: float, max_voxels: int) -> tuple[float, np.ndarray]:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    vs = float(max(voxel_size, 1e-4))
    while True:
        dims = np.ceil((maxs - mins) / vs).astype(np.int64) + 3
        total = int(dims[0]) * int(dims[1]) * int(dims[2])
        if total <= max_voxels:
            return vs, mins
        vs *= 1.2


def uniform_resample_keep_first(xyz: np.ndarray, radius: float) -> np.ndarray:
    r = float(max(radius, 1e-6))
    mins = xyz.min(axis=0)
    ijk = np.floor((xyz - mins) / r).astype(np.int32)
    _, first_idx = np.unique(ijk, axis=0, return_index=True)
    return np.sort(first_idx)


def build_occupancy(
    xyz: np.ndarray,
    weights: np.ndarray,
    voxel_size: float,
    mins: np.ndarray,
    occ_quantile: float,
) -> tuple[np.ndarray, float]:
    ijk = np.floor((xyz - mins) / voxel_size).astype(np.int32)
    dims = ijk.max(axis=0) + 3
    accum = np.zeros(dims, dtype=np.float32)
    np.add.at(accum, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), weights.astype(np.float32))
    nz = accum[accum > 0]
    if nz.size == 0:
        raise RuntimeError("No occupied voxels after accumulation.")
    q = float(np.clip(occ_quantile, 0.01, 0.99))
    thr = float(np.quantile(nz, q))
    occ = accum >= thr
    return occ, thr


def make_shell(occ: np.ndarray, closing_iters: int, fill_holes_flag: bool, shell_thickness: int) -> np.ndarray:
    st = generate_binary_structure(3, 1)
    out = binary_closing(occ, structure=st, iterations=max(int(closing_iters), 0))
    if fill_holes_flag:
        out = binary_fill_holes(out)
    if shell_thickness > 0:
        eroded = binary_erosion(out, structure=st, iterations=int(shell_thickness))
        out = out & (~eroded)
    return out


def _quads_for_mask(idx: np.ndarray, face_name: str) -> np.ndarray:
    i = idx[:, 0]
    j = idx[:, 1]
    k = idx[:, 2]
    if face_name == "x+":
        return np.stack(
            [
                np.stack([i + 1, j, k], axis=1),
                np.stack([i + 1, j + 1, k], axis=1),
                np.stack([i + 1, j + 1, k + 1], axis=1),
                np.stack([i + 1, j, k + 1], axis=1),
            ],
            axis=1,
        )
    if face_name == "x-":
        return np.stack(
            [
                np.stack([i, j, k], axis=1),
                np.stack([i, j, k + 1], axis=1),
                np.stack([i, j + 1, k + 1], axis=1),
                np.stack([i, j + 1, k], axis=1),
            ],
            axis=1,
        )
    if face_name == "y+":
        return np.stack(
            [
                np.stack([i, j + 1, k], axis=1),
                np.stack([i, j + 1, k + 1], axis=1),
                np.stack([i + 1, j + 1, k + 1], axis=1),
                np.stack([i + 1, j + 1, k], axis=1),
            ],
            axis=1,
        )
    if face_name == "y-":
        return np.stack(
            [
                np.stack([i, j, k], axis=1),
                np.stack([i + 1, j, k], axis=1),
                np.stack([i + 1, j, k + 1], axis=1),
                np.stack([i, j, k + 1], axis=1),
            ],
            axis=1,
        )
    if face_name == "z+":
        return np.stack(
            [
                np.stack([i, j, k + 1], axis=1),
                np.stack([i + 1, j, k + 1], axis=1),
                np.stack([i + 1, j + 1, k + 1], axis=1),
                np.stack([i, j + 1, k + 1], axis=1),
            ],
            axis=1,
        )
    if face_name == "z-":
        return np.stack(
            [
                np.stack([i, j, k], axis=1),
                np.stack([i, j + 1, k], axis=1),
                np.stack([i + 1, j + 1, k], axis=1),
                np.stack([i + 1, j, k], axis=1),
            ],
            axis=1,
        )
    raise ValueError(face_name)


def voxel_surface_to_mesh(shell: np.ndarray, mins: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    x_pos = shell & (~np.pad(shell[1:, :, :], ((0, 1), (0, 0), (0, 0)), constant_values=False))
    x_neg = shell & (~np.pad(shell[:-1, :, :], ((1, 0), (0, 0), (0, 0)), constant_values=False))
    y_pos = shell & (~np.pad(shell[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=False))
    y_neg = shell & (~np.pad(shell[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=False))
    z_pos = shell & (~np.pad(shell[:, :, 1:], ((0, 0), (0, 0), (0, 1)), constant_values=False))
    z_neg = shell & (~np.pad(shell[:, :, :-1], ((0, 0), (0, 0), (1, 0)), constant_values=False))

    entries = [
        ("x+", np.argwhere(x_pos)),
        ("x-", np.argwhere(x_neg)),
        ("y+", np.argwhere(y_pos)),
        ("y-", np.argwhere(y_neg)),
        ("z+", np.argwhere(z_pos)),
        ("z-", np.argwhere(z_neg)),
    ]

    verts_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
    v_offset = 0
    for face_name, idx in entries:
        if idx.size == 0:
            continue
        quads = _quads_for_mask(idx, face_name).astype(np.float32)  # (N,4,3)
        xyz = mins[None, None, :] + quads * np.float32(voxel_size)
        n = xyz.shape[0]
        verts = xyz.reshape(-1, 3)
        base = (np.arange(n, dtype=np.int64) * 4 + v_offset)[:, None]
        tri1 = np.concatenate([base + 0, base + 1, base + 2], axis=1)
        tri2 = np.concatenate([base + 0, base + 2, base + 3], axis=1)
        faces = np.concatenate([tri1, tri2], axis=0)
        verts_chunks.append(verts)
        faces_chunks.append(faces)
        v_offset += verts.shape[0]

    if not verts_chunks:
        raise RuntimeError("No boundary faces found; occupancy is empty.")

    vertices = np.concatenate(verts_chunks, axis=0)
    faces = np.concatenate(faces_chunks, axis=0).astype(np.int32)
    return vertices, faces


def dedup_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    key = np.round(vertices, 6)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    faces_new = inv[faces]
    return uniq.astype(np.float32), faces_new.astype(np.int32)


def write_mesh_ply(
    out_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    color_rgb: tuple[int, int, int],
    alpha: int,
) -> None:
    r, g, b = [int(np.clip(v, 0, 255)) for v in color_rgb]
    a = int(np.clip(alpha, 0, 255))

    vtx = np.empty(
        len(vertices),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ],
    )
    vtx["x"] = vertices[:, 0]
    vtx["y"] = vertices[:, 1]
    vtx["z"] = vertices[:, 2]
    vtx["red"] = np.uint8(r)
    vtx["green"] = np.uint8(g)
    vtx["blue"] = np.uint8(b)
    vtx["alpha"] = np.uint8(a)

    fdata = np.empty(len(faces), dtype=[("vertex_indices", "O")])
    fdata["vertex_indices"] = [f.tolist() for f in faces]

    PlyData(
        [
            PlyElement.describe(vtx, "vertex"),
            PlyElement.describe(fdata, "face"),
        ],
        text=False,
    ).write(str(out_path))


def main() -> None:
    p = argparse.ArgumentParser(description="Gaussian PLY -> voxel surface mesh (PLY).")
    p.add_argument("--input_ply", required=True)
    p.add_argument("--output_mesh", required=True)
    p.add_argument("--voxel_size", type=float, default=2.2)
    p.add_argument("--max_voxels", type=int, default=45_000_000)
    p.add_argument("--uniform_resample_radius", type=float, default=2.0)
    p.add_argument("--max_points", type=int, default=300_000)
    p.add_argument("--opacity_min", type=float, default=-2.2)
    p.add_argument("--occ_quantile", type=float, default=0.55)
    p.add_argument("--closing_iters", type=int, default=1)
    p.add_argument("--fill_holes", action="store_true")
    p.add_argument("--shell_thickness_vox", type=int, default=1)
    p.add_argument("--dedup_vertices", action="store_true")
    p.add_argument("--color_r", type=int, default=0)
    p.add_argument("--color_g", type=int, default=0)
    p.add_argument("--color_b", type=int, default=255)
    p.add_argument("--alpha", type=int, default=90)
    args = p.parse_args()

    in_path = Path(args.input_ply)
    out_path = Path(args.output_mesh)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(str(in_path))
    v = ply["vertex"].data
    names = set(v.dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("Input PLY must contain x,y,z.")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    keep = np.ones(len(xyz), dtype=bool)
    if "opacity" in names:
        keep &= v["opacity"].astype(np.float32) >= np.float32(args.opacity_min)
    xyz = xyz[keep]
    idx_kept = np.nonzero(keep)[0]

    if len(xyz) == 0:
        raise RuntimeError("No points left after filtering.")

    if float(args.uniform_resample_radius) > 0:
        k = uniform_resample_keep_first(xyz, float(args.uniform_resample_radius))
        xyz = xyz[k]
        idx_kept = idx_kept[k]

    if args.max_points > 0 and len(xyz) > int(args.max_points):
        rng = np.random.default_rng(42)
        sel = rng.choice(len(xyz), size=int(args.max_points), replace=False)
        xyz = xyz[sel]
        idx_kept = idx_kept[sel]

    voxel_size, mins = adaptive_voxel_size(xyz, float(args.voxel_size), int(args.max_voxels))

    weights = np.ones(len(xyz), dtype=np.float32)
    if "opacity" in names:
        op = v["opacity"][idx_kept].astype(np.float32)
        weights *= np.clip(sigmoid(op) * 1.2, 0.05, 1.5)
    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        s = (
            v["scale_0"][idx_kept].astype(np.float32)
            + v["scale_1"][idx_kept].astype(np.float32)
            + v["scale_2"][idx_kept].astype(np.float32)
        ) / 3.0
        sigma = np.exp(s)
        weights *= np.clip(sigma / np.float32(voxel_size), 0.35, 3.0)

    occ, thr = build_occupancy(
        xyz=xyz,
        weights=weights,
        voxel_size=voxel_size,
        mins=mins,
        occ_quantile=float(args.occ_quantile),
    )
    shell = make_shell(
        occ,
        closing_iters=int(args.closing_iters),
        fill_holes_flag=bool(args.fill_holes),
        shell_thickness=int(args.shell_thickness_vox),
    )

    vertices, faces = voxel_surface_to_mesh(shell, mins=mins, voxel_size=voxel_size)
    if args.dedup_vertices:
        vertices, faces = dedup_vertices(vertices, faces)

    write_mesh_ply(
        out_path=out_path,
        vertices=vertices,
        faces=faces,
        color_rgb=(args.color_r, args.color_g, args.color_b),
        alpha=args.alpha,
    )

    print(f"input={in_path}")
    print(f"output={out_path}")
    print(f"points_used={len(xyz)}")
    print(f"voxel_size={voxel_size:.6f}")
    print(f"occ_threshold={thr:.6f}")
    print(f"shell_voxels={int(shell.sum())}")
    print(f"mesh_vertices={len(vertices)}")
    print(f"mesh_faces={len(faces)}")


if __name__ == "__main__":
    main()
