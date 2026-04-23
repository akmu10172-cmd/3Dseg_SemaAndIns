#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaussian_renderer import render as gs_render  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from semantic_instance_pipeline import (  # noqa: E402
    CameraRecord,
    load_instance_masks_for_cameras,
    proj2d_instances,
    write_semantic_instance_ply,
    write_subset_ply,
)
from utils.graphics_utils import getProjectionMatrix  # noqa: E402
from utils.sh_utils import C0, eval_sh  # noqa: E402

try:  # noqa: E402
    import dom_gaussian_rasterization as dom_gs_raster

    HAS_DOM_RASTER = True
    DOM_RASTER_IMPORT_ERR = ""
except Exception as _dom_exc:  # noqa: E402
    dom_gs_raster = None
    HAS_DOM_RASTER = False
    DOM_RASTER_IMPORT_ERR = str(_dom_exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Topdown render -> SAM3 instance -> 2D projection to 3D instances. "
            "Supports strict orthographic DOM mode (no perspective)."
        )
    )
    p.add_argument("--scene_path", required=True)
    p.add_argument("--input_ply", required=True)
    p.add_argument("--output_subdir", default="topdown_instance_pipeline")

    p.add_argument("--num_views", type=int, default=8)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--fov_deg", type=float, default=58.0, help="Used only in perspective mode")
    p.add_argument("--projection_mode", default="orthographic", choices=["orthographic", "perspective"])
    p.add_argument(
        "--camera_grid_side",
        type=int,
        default=1,
        help="Orthographic-only: use NxN grid camera layout around scene center when >1",
    )
    p.add_argument(
        "--camera_grid_step",
        type=float,
        default=0.0,
        help="Orthographic-only: XY spacing between grid cameras in world units (<=0 uses auto spacing)",
    )
    p.add_argument(
        "--camera_height_offset",
        type=float,
        default=0.0,
        help="Additive Z offset on top of auto-computed camera height (supports negative values)",
    )
    p.add_argument("--ortho_padding_ratio", type=float, default=0.02)
    p.add_argument(
        "--ortho_span_scale",
        type=float,
        default=1.0,
        help="Orthographic view span multiplier. <1 zooms in, >1 zooms out.",
    )
    p.add_argument(
        "--ortho_grid_window_ratio",
        type=float,
        default=2.0,
        help=(
            "Grid orthographic only: per-view local window span = camera_grid_step * ratio "
            "when ortho_grid_window_size <= 0."
        ),
    )
    p.add_argument(
        "--ortho_grid_window_size",
        type=float,
        default=0.0,
        help="Grid orthographic only: fixed per-view local window span in world units (>0 overrides ratio).",
    )
    p.add_argument(
        "--ortho_world_units_per_pixel",
        type=float,
        default=0.0,
        help=(
            "Grid orthographic only: world units represented by one pixel. "
            "<=0 disables this override; >0 overrides ortho_grid_window_ratio when "
            "ortho_grid_window_size <= 0."
        ),
    )
    p.add_argument("--ortho_splat_radius", type=int, default=0)
    p.add_argument("--ortho_opacity_threshold", type=float, default=0.08)
    p.add_argument("--max_points", type=int, default=0, help="0 means no downsample for projection/output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--xy_jitter_ratio", type=float, default=0.0, help="0 keeps exact center topdown")

    p.add_argument("--sam3_python", default="/mnt/d/sam3/.conda/envs/sam3/bin/python")
    p.add_argument("--sam3_checkpoint", default="/mnt/d/modelscope/hub/models/facebook/sam3/sam3.pt")
    p.add_argument("--sam3_device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--sam3_resolution", type=int, default=1008)
    p.add_argument("--sam3_conf", type=float, default=0.35)
    p.add_argument("--sam3_prompts", default="building")
    p.add_argument("--sam3_min_mask_area", type=int, default=40)
    p.add_argument("--sam3_overwrite", action="store_true")
    p.add_argument("--semantic_probe_summary", default="", help="summary.json from semantic probe step")
    p.add_argument("--semantic_probe_fallback_prompts", default="", help="fallback prompts when semantic probe fails")

    p.add_argument("--vote_stride", type=int, default=1)
    p.add_argument("--instance_mask_ignore_ids", default="0")
    p.add_argument("--instance_min_mask_points", type=int, default=30)
    p.add_argument("--instance_match_iou", type=float, default=0.2)
    p.add_argument("--instance_min_point_votes", type=int, default=2)
    p.add_argument("--min_instance_points", type=int, default=120)

    p.add_argument("--semantic_id", type=int, default=3)
    p.add_argument("--save_instance_parts", action="store_true")
    p.add_argument("--render_only", action="store_true", help="Only render topdown images and save poses, then exit")
    p.add_argument("--skip_render", action="store_true", help="Reuse existing renders/poses and continue SAM3+projection")
    # Backward compat: transparently accept legacy arg name.
    argv = ["--ortho_world_units_per_pixel" if x == "--ortho_rate" else x for x in sys.argv[1:]]
    return p.parse_args(argv)


def parse_int_list(text: str) -> List[int]:
    s = (text or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_vertex_xyz(input_ply: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(input_ply))
    v = np.array(ply["vertex"].data)
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return v, xyz


def extract_display_rgb(vertex_arr: np.ndarray) -> np.ndarray:
    names = set(vertex_arr.dtype.names or [])
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.vstack([vertex_arr["f_dc_0"], vertex_arr["f_dc_1"], vertex_arr["f_dc_2"]]).T.astype(np.float32)
        return np.clip(dc * float(C0) + 0.5, 0.0, 1.0)
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.vstack([vertex_arr["red"], vertex_arr["green"], vertex_arr["blue"]]).T.astype(np.float32) / 255.0
        return np.clip(rgb, 0.0, 1.0)
    return np.full((len(vertex_arr), 3), 0.75, dtype=np.float32)


def extract_opacity(vertex_arr: np.ndarray) -> np.ndarray:
    names = set(vertex_arr.dtype.names or [])
    if "opacity" in names:
        return sigmoid(vertex_arr["opacity"].astype(np.float32))
    return np.ones((len(vertex_arr),), dtype=np.float32)


def look_at_w2c(cam_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    z_cam = normalize(target - cam_pos)
    if np.linalg.norm(z_cam) < 1e-6:
        z_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    x_cam = np.cross(z_cam, up_hint)
    if np.linalg.norm(x_cam) < 1e-6:
        x_cam = np.cross(z_cam, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    x_cam = normalize(x_cam)
    y_cam = normalize(np.cross(x_cam, z_cam))

    r_c2w = np.stack([x_cam, y_cam, z_cam], axis=1)
    r_w2c = r_c2w.T
    t = -r_w2c @ cam_pos

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r_w2c
    w2c[:3, 3] = t
    return w2c


def strict_topdown_w2c(cam_pos: np.ndarray) -> np.ndarray:
    # Camera axes in world frame:
    # x_cam -> +X_world, y_cam -> -Y_world, z_cam -> -Z_world (pure top-down).
    r_w2c = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    t = -r_w2c @ cam_pos.astype(np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = r_w2c
    w2c[:3, 3] = t
    return w2c


class TopdownView:
    def __init__(self, w2c: np.ndarray, width: int, height: int, fov_deg: float):
        self.image_width = int(width)
        self.image_height = int(height)
        self.FoVx = math.radians(float(fov_deg))
        self.FoVy = math.radians(float(fov_deg))
        self.znear = 0.01
        self.zfar = 1000.0
        self.bClusterOccur = None

        wv = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
        proj = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).to(device="cuda", dtype=torch.float32)

        self.world_view_transform = wv
        self.full_proj_transform = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]


def get_orthographic_projection_matrix(
    left: float,
    right: float,
    bottom: float,
    top: float,
    znear: float,
    zfar: float,
) -> torch.Tensor:
    p = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
    p[0, 0] = 2.0 / max((right - left), 1e-8)
    p[1, 1] = 2.0 / max((top - bottom), 1e-8)
    p[0, 3] = -(right + left) / max((right - left), 1e-8)
    p[1, 3] = -(top + bottom) / max((top - bottom), 1e-8)
    p[2, 2] = 2.0 / max((zfar - znear), 1e-8)
    p[2, 3] = -(zfar + znear) / max((zfar - znear), 1e-8)
    p[3, 3] = 1.0
    return p


class TopdownOrthoView:
    def __init__(
        self,
        w2c: np.ndarray,
        width: int,
        height: int,
        left: float,
        right: float,
        bottom: float,
        top: float,
    ):
        self.image_width = int(width)
        self.image_height = int(height)
        self.left = float(left)
        self.right = float(right)
        self.bottom = float(bottom)
        self.top = float(top)
        # Placeholder FOV used by raster settings tanfovx/tanfovy.
        self.FoVx = math.radians(90.0)
        self.FoVy = math.radians(90.0)
        self.znear = 0.01
        self.zfar = 1000.0
        self.bClusterOccur = None

        wv = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
        proj = get_orthographic_projection_matrix(
            left=self.left,
            right=self.right,
            bottom=self.bottom,
            top=self.top,
            znear=self.znear,
            zfar=self.zfar,
        )
        self.world_view_transform = wv
        self.full_proj_transform = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        self.camera_center = torch.inverse(self.world_view_transform)[3, :3]


def render_orthographic_dom_gaussian(
    view: TopdownOrthoView,
    pc: GaussianModel,
    pipe: SimpleNamespace,
    bg_color: torch.Tensor,
) -> np.ndarray:
    if not HAS_DOM_RASTER or dom_gs_raster is None:
        raise RuntimeError(f"dom_gaussian_rasterization not available: {DOM_RASTER_IMPORT_ERR}")

    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)

    raster_settings = dom_gs_raster.GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        left=float(view.left),
        right=float(view.right),
        bottom=float(view.bottom),
        top=float(view.top),
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=view.world_view_transform,
        projmatrix=view.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=view.camera_center,
        prefiltered=False,
        debug=bool(getattr(pipe, "debug", False)),
    )
    rasterizer = dom_gs_raster.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if bool(getattr(pipe, "compute_cov3D_python", False)):
        cov3d_precomp = pc.get_covariance(1.0)
        scales = None
        rotations = None
    else:
        cov3d_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if bool(getattr(pipe, "convert_SHs_python", False)):
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - view.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        shs = None
    else:
        colors_precomp = None
        shs = pc.get_features

    out = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3d_precomp,
    )
    if not isinstance(out, tuple) or len(out) < 1:
        raise RuntimeError("Unexpected output from dom rasterizer")
    rendered = out[0]
    rgb = rendered.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (rgb * 255.0).astype(np.uint8)


def build_topdown_cameras(
    xyz: np.ndarray,
    num_views: int,
    image_size: int,
    fov_deg: float,
    xy_jitter_ratio: float,
    projection_mode: str = "orthographic",
    camera_grid_side: int = 1,
    camera_grid_step: float = 0.0,
    camera_height_offset: float = 0.0,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    x_span = float(maxs[0] - mins[0])
    y_span = float(maxs[1] - mins[1])
    z_span = float(maxs[2] - mins[2])
    xy_diag = float(math.hypot(x_span, y_span))

    base_h = max(0.9 * z_span, 0.2 * xy_diag, 1e-3)
    h_off = float(camera_height_offset)
    levels = (
        [max(base_h + h_off, 1e-3)]
        if z_span < 0.35 * max(xy_diag, 1e-6)
        else [max(base_h + h_off, 1e-3), max(1.45 * base_h + h_off, 1e-3)]
    )

    n = max(1, int(num_views))
    grid_side = max(1, int(camera_grid_side))
    grid_step_input = max(0.0, float(camera_grid_step))
    auto_grid_step = False
    use_grid = str(projection_mode).lower() == "orthographic" and grid_side > 1
    if use_grid:
        if grid_step_input > 0.0:
            grid_step = grid_step_input
        else:
            # Auto spacing: spread NxN cameras over scene span when user leaves step at 0.
            scene_span = max(x_span, y_span, 1e-6)
            denom = float(max(grid_side - 1, 1))
            grid_step = max(scene_span / denom, 1e-6)
            auto_grid_step = True
    else:
        grid_step = grid_step_input

    yaws = [360.0 * i / n for i in range(n)]
    jitter_r = max(0.0, float(xy_jitter_ratio)) * max(xy_diag, 1e-3)
    target_z = mins[2] + 0.5 * z_span
    w2cs: List[np.ndarray] = []
    if use_grid:
        idxs = [float(i) - (float(grid_side - 1) * 0.5) for i in range(grid_side)]
        hh = max(base_h + h_off, 1e-3)
        for gy in idxs:
            for gx in idxs:
                cam_pos = np.array(
                    [center[0] + gx * grid_step, center[1] + gy * grid_step, target_z + hh],
                    dtype=np.float64,
                )
                w2cs.append(strict_topdown_w2c(cam_pos))
    else:
        for i, yaw in enumerate(yaws):
            a = math.radians(yaw)
            ox = jitter_r * math.cos(a)
            oy = jitter_r * math.sin(a)
            hh = levels[i % len(levels)]
            cam_pos = np.array([center[0] + ox, center[1] + oy, target_z + hh], dtype=np.float64)
            if str(projection_mode).lower() == "orthographic":
                # Keep a fixed Z-down frame in orthographic mode even for single-view/ring layout.
                # This avoids per-view in-plane yaw rotation from look_at.
                w2cs.append(strict_topdown_w2c(cam_pos))
            else:
                target = np.array([center[0] + ox, center[1] + oy, target_z], dtype=np.float64)
                up_hint = np.array([math.cos(a), math.sin(a), 0.0], dtype=np.float64)
                w2cs.append(look_at_w2c(cam_pos, target, up_hint))

    f = 0.5 * float(image_size) / math.tan(math.radians(float(fov_deg)) / 2.0)
    meta = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "x_span": x_span,
        "y_span": y_span,
        "z_span": z_span,
        "xy_diag": xy_diag,
        "base_height": float(base_h),
        "topdown": True,
        "xy_jitter_ratio": float(xy_jitter_ratio),
        "fov_deg": float(fov_deg),
        "camera_height_offset": float(h_off),
        "camera_grid_side": int(grid_side),
        "camera_grid_step": float(grid_step),
        "camera_grid_step_auto": bool(auto_grid_step),
        "camera_layout": "grid" if use_grid else "ring",
        "num_views_rendered": int(len(w2cs)),
        "fx": float(f),
        "fy": float(f),
        "cx": float((image_size - 1) / 2.0),
        "cy": float((image_size - 1) / 2.0),
    }
    return w2cs, meta


def compute_ortho_bounds(xyz: np.ndarray, w2c: np.ndarray, padding_ratio: float) -> Tuple[float, float, float, float]:
    xyz_h = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    cam = xyz_h @ w2c.T
    x = cam[:, 0]
    y = cam[:, 1]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    pad = max(span_x, span_y) * max(0.0, float(padding_ratio))
    return x_min - pad, x_max + pad, y_min - pad, y_max + pad


def compute_centered_ortho_bounds(
    x_span: float,
    y_span: float,
    padding_ratio: float,
    span_scale: float,
) -> Tuple[float, float, float, float]:
    pad_ratio = max(0.0, float(padding_ratio))
    span = max(float(x_span), float(y_span), 1e-6) * max(0.05, float(span_scale))
    span = span * (1.0 + 2.0 * pad_ratio)
    half = 0.5 * span
    return -half, half, -half, half


def scale_ortho_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    span_scale: float,
) -> Tuple[float, float, float, float]:
    s = max(0.05, float(span_scale))
    if abs(s - 1.0) < 1e-6:
        return float(x_min), float(x_max), float(y_min), float(y_max)
    cx = 0.5 * (float(x_min) + float(x_max))
    cy = 0.5 * (float(y_min) + float(y_max))
    hx = 0.5 * max(float(x_max) - float(x_min), 1e-6) * s
    hy = 0.5 * max(float(y_max) - float(y_min), 1e-6) * s
    return cx - hx, cx + hx, cy - hy, cy + hy


def render_orthographic_dom(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    splat_radius: int,
    opacity_threshold: float,
) -> np.ndarray:
    img = np.ones((height, width, 3), dtype=np.float32)
    if len(xyz) == 0:
        return (img * 255.0).astype(np.uint8)

    xyz_h = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    cam = xyz_h @ w2c.T
    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2]

    valid = z > 1e-6
    if opacity_threshold > 0.0 and len(opacity) == len(valid):
        valid &= (opacity >= float(opacity_threshold))
    if not np.any(valid):
        return (img * 255.0).astype(np.uint8)

    x = x[valid]
    y = y[valid]
    z = z[valid]
    c = rgb[valid]

    if not (x_max > x_min and y_max > y_min):
        return (img * 255.0).astype(np.uint8)

    u = np.round((x - x_min) / (x_max - x_min) * float(width - 1)).astype(np.int32)
    v = np.round((y_max - y) / (y_max - y_min) * float(height - 1)).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return (img * 255.0).astype(np.uint8)

    u = u[inside]
    v = v[inside]
    z = z[inside]
    c = c[inside]

    depth = np.full((height * width,), np.inf, dtype=np.float32)
    img_flat = img.reshape(-1, 3)
    r = max(0, int(splat_radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            uu = u + dx
            vv = v + dy
            keep = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not np.any(keep):
                continue
            uu = uu[keep]
            vv = vv[keep]
            zz = z[keep]
            cc = c[keep]

            linear = (vv.astype(np.int64) * np.int64(width) + uu.astype(np.int64))
            order = np.lexsort((zz, linear))
            linear_s = linear[order]
            zz_s = zz[order]
            cc_s = cc[order]
            uniq = np.ones((len(order),), dtype=np.bool_)
            uniq[1:] = linear_s[1:] != linear_s[:-1]
            pix = linear_s[uniq]
            dep = zz_s[uniq]
            col = cc_s[uniq]
            old = depth[pix]
            better = dep < old
            if np.any(better):
                pp = pix[better]
                depth[pp] = dep[better]
                img_flat[pp] = col[better]

    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def select_prompt_from_probe_summary(summary_json: Path, fallback_prompts: str, default_prompts: str) -> str:
    if not summary_json.exists():
        fb = (fallback_prompts or "").strip()
        return fb if fb else default_prompts
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        fb = (fallback_prompts or "").strip()
        return fb if fb else default_prompts

    score: Dict[str, int] = {}
    for it in items:
        d = it.get("prompt_raw_counts", {})
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if not str(k).strip():
                continue
            score[str(k).strip()] = int(score.get(str(k).strip(), 0) + int(v))

    if not score:
        fb = (fallback_prompts or "").strip()
        return fb if fb else default_prompts

    best = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    print(f"[semantic-probe] selected prompt: {best} from {summary_json}")
    return best


def load_cameras_from_pose_json(pose_json: Path) -> Tuple[List[CameraRecord], Dict[str, object], List[float]]:
    if not pose_json.exists():
        raise FileNotFoundError(f"pose json not found: {pose_json}")
    data = json.loads(pose_json.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"no camera items in pose json: {pose_json}")
    meta = data.get("camera_adaptive_meta", {})
    default_mode = str(data.get("projection_mode", "perspective"))
    cameras: List[CameraRecord] = []
    down_angles: List[float] = []
    for it in items:
        name = str(it.get("image", ""))
        if not name:
            continue
        w2c = np.array(it["w2c"], dtype=np.float64)
        fx = float(it.get("fx", 0.0))
        fy = float(it.get("fy", 0.0))
        cx = float(it.get("cx", 0.0))
        cy = float(it.get("cy", 0.0))
        mode = str(it.get("projection_mode", default_mode))
        ortho_v_mode = str(it.get("ortho_v_mode", "auto"))
        cameras.append(
            CameraRecord(
                name=name,
                mask_path=Path(name),
                w2c=w2c,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                projection_mode=mode,
                ortho_x_min=float(it.get("ortho_x_min", 0.0)),
                ortho_x_max=float(it.get("ortho_x_max", 0.0)),
                ortho_y_min=float(it.get("ortho_y_min", 0.0)),
                ortho_y_max=float(it.get("ortho_y_max", 0.0)),
                ortho_v_mode=ortho_v_mode,
            )
        )
        down_angles.append(float(it.get("angle_to_down_deg", 0.0)))
    if not cameras:
        raise RuntimeError(f"no valid cameras in pose json: {pose_json}")
    return cameras, meta, down_angles


def main() -> None:
    args = parse_args()
    if args.render_only and args.skip_render:
        raise ValueError("--render_only and --skip_render cannot be used together")
    np.random.seed(int(args.seed))

    scene_path = Path(args.scene_path)
    input_ply = Path(args.input_ply)
    if not scene_path.exists():
        raise FileNotFoundError(f"scene_path not found: {scene_path}")
    if not input_ply.exists():
        raise FileNotFoundError(f"input_ply not found: {input_ply}")

    projection_mode = str(args.projection_mode).lower()
    if projection_mode not in {"orthographic", "perspective"}:
        raise ValueError(f"Invalid projection_mode={args.projection_mode}")

    xy_jitter_ratio = float(args.xy_jitter_ratio)
    if projection_mode == "orthographic":
        # Keep strict Z-down DOM behavior by default in orthographic mode.
        xy_jitter_ratio = 0.0
        if abs(float(args.camera_height_offset)) > 1e-6:
            print(
                "[note] orthographic DOM is almost invariant to camera height. "
                "height_offset mostly changes pose metadata, not image scale."
            )

    pipe_root = scene_path / args.output_subdir
    render_dir = pipe_root / "renders_topdown"
    sam3_dir = pipe_root / "sam3_instance"
    pose_dir = pipe_root / "render_view_poses_topdown"
    legacy_pose_dir = scene_path / "render_view_poses_topdown"
    pose_json_path = pose_dir / "topdown_camera_poses.json"
    legacy_pose_json_path = legacy_pose_dir / "topdown_camera_poses.json"
    out3d_dir = pipe_root / "instance_3d"
    for d in [pipe_root, render_dir, sam3_dir, pose_dir, out3d_dir]:
        d.mkdir(parents=True, exist_ok=True)

    vertex_full, xyz_full = load_vertex_xyz(input_ply)
    n_full = int(len(xyz_full))
    if n_full == 0:
        raise RuntimeError("empty point cloud")

    if int(args.max_points) > 0 and n_full > int(args.max_points):
        idx = np.random.choice(n_full, size=int(args.max_points), replace=False)
        idx = np.sort(idx)
        vertex_proj = vertex_full[idx]
        xyz_proj = xyz_full[idx]
        print(f"[info] projection downsample: {n_full} -> {len(xyz_proj)}")
    else:
        vertex_proj = vertex_full
        xyz_proj = xyz_full

    cameras: List[CameraRecord] = []
    cam_meta: Dict[str, object] = {}
    down_angles: List[float] = []
    if not args.skip_render:
        w2cs, cam_meta = build_topdown_cameras(
            xyz=xyz_proj,
            num_views=int(args.num_views),
            image_size=int(args.image_size),
            fov_deg=float(args.fov_deg),
            xy_jitter_ratio=xy_jitter_ratio,
            projection_mode=projection_mode,
            camera_grid_side=int(args.camera_grid_side),
            camera_grid_step=float(args.camera_grid_step),
            camera_height_offset=float(args.camera_height_offset),
        )
        if projection_mode == "orthographic" and str(cam_meta.get("camera_layout", "")) == "grid":
            if bool(cam_meta.get("camera_grid_step_auto", False)):
                print(
                    "[info] grid spacing auto-computed: "
                    f"side={int(cam_meta.get('camera_grid_side', 1))}, "
                    f"step={float(cam_meta.get('camera_grid_step', 0.0)):.6f}"
                )
            else:
                print(
                    "[info] grid spacing from arg: "
                    f"side={int(cam_meta.get('camera_grid_side', 1))}, "
                    f"step={float(cam_meta.get('camera_grid_step', 0.0)):.6f}"
                )

        W = int(args.image_size)
        H = int(args.image_size)
        pose_items = []
        if projection_mode == "perspective":
            gaussians = GaussianModel(sh_degree=3)
            gaussians.load_ply(str(input_ply))
            pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
            bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

            fx, fy = float(cam_meta["fx"]), float(cam_meta["fy"])
            cx, cy = float(cam_meta["cx"]), float(cam_meta["cy"])

            for i, w2c in enumerate(w2cs):
                name = f"top_{i:03d}.png"
                view = TopdownView(w2c=w2c, width=W, height=H, fov_deg=float(args.fov_deg))
                pkg = gs_render(
                    viewpoint_camera=view,
                    pc=gaussians,
                    pipe=pipe,
                    bg_color=bg,
                    iteration=0,
                    rescale=False,
                    render_feat_map=False,
                    render_cluster=False,
                    render_color=True,
                )
                rgb = pkg["render"].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                img = (rgb * 255.0).astype(np.uint8)
                Image.fromarray(img, mode="RGB").save(render_dir / name)

                c2w = np.linalg.inv(w2c)
                forward = c2w[:3, 2]
                down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                ang = math.degrees(
                    math.acos(np.clip(float(np.dot(forward, down) / (np.linalg.norm(forward) + 1e-9)), -1.0, 1.0))
                )
                down_angles.append(float(ang))

                cameras.append(
                    CameraRecord(
                        name=name,
                        mask_path=render_dir / name,
                        w2c=w2c.astype(np.float64),
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        projection_mode="perspective",
                    )
                )
                pose_items.append(
                    {
                        "image": name,
                        "projection_mode": "perspective",
                        "w2c": w2c.tolist(),
                        "c2w": c2w.tolist(),
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "width": W,
                        "height": H,
                        "angle_to_down_deg": float(ang),
                    }
                )
        else:
            ortho_backend = "dom_gaussian_rasterization" if HAS_DOM_RASTER else "numpy_fallback"
            cam_meta["orthographic_backend"] = ortho_backend
            if HAS_DOM_RASTER:
                print("[info] orthographic backend: dom_gaussian_rasterization")
            else:
                print(
                    "[warn] dom_gaussian_rasterization not available, fallback to numpy renderer: "
                    f"{DOM_RASTER_IMPORT_ERR}"
                )

            if HAS_DOM_RASTER:
                gaussians_ortho = GaussianModel(sh_degree=3)
                gaussians_ortho.load_ply(str(input_ply))
                pipe_ortho = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
                bg_ortho = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
                rgb_proj = None
                opacity_proj = None
            else:
                rgb_proj = extract_display_rgb(vertex_proj)
                opacity_proj = extract_opacity(vertex_proj)
                gaussians_ortho = None
                pipe_ortho = None
                bg_ortho = None

            use_centered_bounds = str(cam_meta.get("camera_layout", "")) == "grid"
            grid_window_span = 0.0
            grid_window_mode = "tile"
            effective_ortho_world_units_per_pixel = 0.0
            if use_centered_bounds:
                grid_step = float(cam_meta.get("camera_grid_step", 0.0))
                scene_span = max(
                    float(cam_meta.get("x_span", 0.0)),
                    float(cam_meta.get("y_span", 0.0)),
                    1e-6,
                )
                grid_side = max(1, int(cam_meta.get("camera_grid_side", 1)))
                scene_tile_span = scene_span / float(grid_side)
                if float(args.ortho_grid_window_size) > 0.0:
                    grid_window_span = float(args.ortho_grid_window_size)
                    grid_window_mode = "fixed_size"
                elif float(args.ortho_world_units_per_pixel) > 0.0:
                    grid_window_span = float(args.ortho_world_units_per_pixel) * float(max(W, 1))
                    grid_window_mode = "world_units_per_pixel"
                elif grid_step > 0.0:
                    step_span = grid_step * max(0.05, float(args.ortho_grid_window_ratio))
                    grid_window_span = max(
                        scene_tile_span,
                        step_span,
                    )
                    grid_window_mode = "ratio_or_tile"
                    if step_span + 1e-6 < scene_tile_span:
                        print(
                            "[note] grid_step is small vs scene extent; using scene_tile_span as minimum "
                            "window size for stable coverage."
                        )
                else:
                    grid_window_span = scene_tile_span
                    grid_window_mode = "tile"
                pad_ratio = max(0.0, float(args.ortho_padding_ratio))
                span_scale = max(0.05, float(args.ortho_span_scale))
                effective_ortho_world_units_per_pixel = (
                    float(grid_window_span) * span_scale * (1.0 + 2.0 * pad_ratio) / float(max(W, 1))
                )
                print(
                    f"[info] orthographic grid local span={grid_window_span:.6f} "
                    f"(mode={grid_window_mode}, effective_world_units_per_pixel={effective_ortho_world_units_per_pixel:.6f}, "
                    f"scene_tile_span={scene_tile_span:.6f}, step={grid_step:.6f}, "
                    f"ratio={float(args.ortho_grid_window_ratio):.3f})"
                )
            cam_meta["ortho_span_scale"] = float(args.ortho_span_scale)
            cam_meta["ortho_grid_window_ratio"] = float(args.ortho_grid_window_ratio)
            cam_meta["ortho_grid_window_size"] = float(args.ortho_grid_window_size)
            cam_meta["ortho_world_units_per_pixel"] = float(args.ortho_world_units_per_pixel)
            cam_meta["ortho_grid_window_mode"] = str(grid_window_mode)
            cam_meta["ortho_effective_world_units_per_pixel"] = float(effective_ortho_world_units_per_pixel)
            cam_meta["ortho_grid_window_effective_span"] = float(grid_window_span)
            centered_bounds = compute_centered_ortho_bounds(
                x_span=float(grid_window_span),
                y_span=float(grid_window_span),
                padding_ratio=float(args.ortho_padding_ratio),
                span_scale=float(args.ortho_span_scale),
            ) if use_centered_bounds else None
            for i, w2c in enumerate(w2cs):
                name = f"top_{i:03d}.png"
                if centered_bounds is None:
                    x_min, x_max, y_min, y_max = compute_ortho_bounds(
                        xyz=xyz_proj,
                        w2c=w2c,
                        padding_ratio=float(args.ortho_padding_ratio),
                    )
                    x_min, x_max, y_min, y_max = scale_ortho_bounds(
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        span_scale=float(args.ortho_span_scale),
                    )
                else:
                    x_min, x_max, y_min, y_max = centered_bounds
                if HAS_DOM_RASTER:
                    view = TopdownOrthoView(
                        w2c=w2c,
                        width=W,
                        height=H,
                        left=float(x_min),
                        right=float(x_max),
                        bottom=float(y_min),
                        top=float(y_max),
                    )
                    img = render_orthographic_dom_gaussian(
                        view=view,
                        pc=gaussians_ortho,
                        pipe=pipe_ortho,
                        bg_color=bg_ortho,
                    )
                else:
                    img = render_orthographic_dom(
                        xyz=xyz_proj,
                        rgb=rgb_proj,
                        opacity=opacity_proj,
                        w2c=w2c,
                        width=W,
                        height=H,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        splat_radius=int(args.ortho_splat_radius),
                        opacity_threshold=float(args.ortho_opacity_threshold),
                    )
                Image.fromarray(img, mode="RGB").save(render_dir / name)

                c2w = np.linalg.inv(w2c)
                forward = c2w[:3, 2]
                down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                ang = math.degrees(
                    math.acos(np.clip(float(np.dot(forward, down) / (np.linalg.norm(forward) + 1e-9)), -1.0, 1.0))
                )
                down_angles.append(float(ang))

                cameras.append(
                    CameraRecord(
                        name=name,
                        mask_path=render_dir / name,
                        w2c=w2c.astype(np.float64),
                        fx=0.0,
                        fy=0.0,
                        cx=float((W - 1) / 2.0),
                        cy=float((H - 1) / 2.0),
                        projection_mode="orthographic",
                        ortho_x_min=float(x_min),
                        ortho_x_max=float(x_max),
                        ortho_y_min=float(y_min),
                        ortho_y_max=float(y_max),
                        ortho_v_mode=("y_minus_y_min" if float(w2c[1, 1]) < 0.0 else "y_max_minus_y"),
                    )
                )
                pose_items.append(
                    {
                        "image": name,
                        "projection_mode": "orthographic",
                        "w2c": w2c.tolist(),
                        "c2w": c2w.tolist(),
                        "fx": 0.0,
                        "fy": 0.0,
                        "cx": float((W - 1) / 2.0),
                        "cy": float((H - 1) / 2.0),
                        "ortho_x_min": float(x_min),
                        "ortho_x_max": float(x_max),
                        "ortho_y_min": float(y_min),
                        "ortho_y_max": float(y_max),
                        "ortho_v_mode": ("y_minus_y_min" if float(w2c[1, 1]) < 0.0 else "y_max_minus_y"),
                        "width": W,
                        "height": H,
                        "angle_to_down_deg": float(ang),
                    }
                )

        pose_json_path.write_text(
            json.dumps(
                {
                    "scene_path": str(scene_path),
                    "input_ply": str(input_ply),
                    "projection_mode": projection_mode,
                    "camera_adaptive_meta": cam_meta,
                    "topdown_angle_to_down_deg": {
                        "min": float(min(down_angles)),
                        "mean": float(sum(down_angles) / len(down_angles)),
                        "max": float(max(down_angles)),
                    },
                    "items": pose_items,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        pose_json = pose_json_path
        if (not pose_json.exists()) and legacy_pose_json_path.exists():
            print(
                "[warn] pose json not found in output_subdir; fallback to legacy scene-level path: "
                f"{legacy_pose_json_path}"
            )
            pose_json_path.write_bytes(legacy_pose_json_path.read_bytes())
            pose_json = pose_json_path
        cameras, cam_meta, down_angles = load_cameras_from_pose_json(pose_json)
        if cameras:
            projection_mode = str(cameras[0].projection_mode or projection_mode).lower()

    if args.render_only:
        print("[done] render-only completed")
        print(f"[out] render_dir: {render_dir}")
        print(f"[out] poses     : {pose_json_path}")
        return

    effective_prompts = str(args.sam3_prompts)
    if (args.semantic_probe_summary or "").strip():
        effective_prompts = select_prompt_from_probe_summary(
            summary_json=Path(str(args.semantic_probe_summary)),
            fallback_prompts=str(args.semantic_probe_fallback_prompts or ""),
            default_prompts=str(args.sam3_prompts),
        )

    sam3_script = SCRIPT_DIR / "sam3_instance_from_images.py"
    sam_cmd = [
        str(args.sam3_python),
        str(sam3_script),
        "--input-dir",
        str(render_dir),
        "--output-dir",
        str(sam3_dir),
        "--device",
        str(args.sam3_device),
        "--resolution",
        str(int(args.sam3_resolution)),
        "--confidence-threshold",
        str(float(args.sam3_conf)),
        "--prompts",
        str(effective_prompts),
        "--min-mask-area",
        str(int(args.sam3_min_mask_area)),
        "--checkpoint-path",
        str(args.sam3_checkpoint),
        "--no-hf-download",
    ]
    if args.sam3_overwrite:
        sam_cmd.append("--overwrite")
    run_cmd(sam_cmd)

    instance_mask_dir = sam3_dir / "instance_index_npy"
    instance_mask_paths = load_instance_masks_for_cameras(
        cameras=cameras,
        instance_mask_dir=instance_mask_dir,
        mode="name",
        mask_suffix="_inst.npy",
        index_pattern="sam_mask_instance_view_{index:04d}.png",
    )
    if not instance_mask_paths:
        raise RuntimeError(f"No instance masks matched views in {instance_mask_dir}")

    instance_id, num_instances, stats = proj2d_instances(
        xyz=xyz_proj,
        cameras=cameras,
        instance_mask_paths=instance_mask_paths,
        vote_stride=int(args.vote_stride),
        instance_mask_level=0,
        instance_mask_ignore_ids=parse_int_list(args.instance_mask_ignore_ids),
        min_mask_points=int(args.instance_min_mask_points),
        merge_iou=float(args.instance_match_iou),
        min_point_votes=int(args.instance_min_point_votes),
        min_instance_points=int(args.min_instance_points),
    )

    np.save(out3d_dir / "point_instance_id.npy", instance_id.astype(np.int32))
    out_ply = out3d_dir / "instance_projected_sem_ins.ply"
    write_semantic_instance_ply(vertex_proj, out_ply, int(args.semantic_id), instance_id)

    if args.save_instance_parts and num_instances > 0:
        inst_dir = out3d_dir / "instance_parts"
        inst_dir.mkdir(parents=True, exist_ok=True)
        for inst in range(1, int(num_instances) + 1):
            m = instance_id == inst
            if int(np.sum(m)) == 0:
                continue
            write_subset_ply(vertex_proj[m], inst_dir / f"instance_{inst:03d}.ply")

    report = {
        "scene_path": str(scene_path),
        "input_ply": str(input_ply),
        "projection_mode": projection_mode,
        "pipeline_root": str(pipe_root),
        "render_dir": str(render_dir),
        "pose_dir": str(pose_dir),
        "sam3_dir": str(sam3_dir),
        "instance_mask_dir": str(instance_mask_dir),
        "output_3d_dir": str(out3d_dir),
        "num_points_projected": int(len(xyz_proj)),
        "num_points_original": int(len(xyz_full)),
        "num_views": int(len(cameras)),
        "matched_instance_masks": int(len(instance_mask_paths)),
        "num_instances_3d": int(num_instances),
        "proj2d_stats": stats,
        "params": {
            "sam3_prompts": str(args.sam3_prompts),
            "sam3_conf": float(args.sam3_conf),
            "sam3_min_mask_area": int(args.sam3_min_mask_area),
            "vote_stride": int(args.vote_stride),
            "instance_match_iou": float(args.instance_match_iou),
            "instance_min_point_votes": int(args.instance_min_point_votes),
            "min_instance_points": int(args.min_instance_points),
            "xy_jitter_ratio": float(xy_jitter_ratio),
            "fov_deg": float(args.fov_deg),
            "ortho_padding_ratio": float(args.ortho_padding_ratio),
            "ortho_span_scale": float(args.ortho_span_scale),
            "ortho_grid_window_ratio": float(args.ortho_grid_window_ratio),
            "ortho_grid_window_size": float(args.ortho_grid_window_size),
            "ortho_world_units_per_pixel": float(args.ortho_world_units_per_pixel),
            "ortho_splat_radius": int(args.ortho_splat_radius),
            "ortho_opacity_threshold": float(args.ortho_opacity_threshold),
            "orthographic_backend": str(cam_meta.get("orthographic_backend", "")),
        },
    }
    (pipe_root / "pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] pipeline completed")
    print(f"[out] report: {pipe_root / 'pipeline_report.json'}")
    print(f"[out] poses : {pose_dir / 'topdown_camera_poses.json'}")
    print(f"[out] 3d ply: {out_ply}")


if __name__ == "__main__":
    main()
