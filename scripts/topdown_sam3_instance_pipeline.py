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
from utils.graphics_utils import getProjectionMatrix  # noqa: E402
from semantic_instance_pipeline import (  # noqa: E402
    CameraRecord,
    load_instance_masks_for_cameras,
    proj2d_instances,
    write_semantic_instance_ply,
    write_subset_ply,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Topdown Gaussian-render -> SAM3 instance -> 2D projection to 3D instances. "
            "Cameras are strict topdown to XOY (forward = -Z)."
        )
    )
    p.add_argument("--scene_path", required=True)
    p.add_argument("--input_ply", required=True)
    p.add_argument("--output_subdir", default="topdown_instance_pipeline")

    p.add_argument("--num_views", type=int, default=8)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--fov_deg", type=float, default=58.0)
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
    return p.parse_args()


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


def load_vertex_xyz(input_ply: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(input_ply))
    v = np.array(ply["vertex"].data)
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return v, xyz


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


def build_topdown_cameras(
    xyz: np.ndarray,
    num_views: int,
    image_size: int,
    fov_deg: float,
    xy_jitter_ratio: float,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    x_span = float(maxs[0] - mins[0])
    y_span = float(maxs[1] - mins[1])
    z_span = float(maxs[2] - mins[2])
    xy_diag = float(math.hypot(x_span, y_span))

    base_h = max(0.9 * z_span, 0.2 * xy_diag, 1e-3)
    levels = [base_h] if z_span < 0.35 * max(xy_diag, 1e-6) else [base_h, 1.45 * base_h]

    n = max(1, int(num_views))
    yaws = [360.0 * i / n for i in range(n)]
    jitter_r = max(0.0, float(xy_jitter_ratio)) * max(xy_diag, 1e-3)

    target_z = mins[2] + 0.5 * z_span
    w2cs: List[np.ndarray] = []
    for i, yaw in enumerate(yaws):
        a = math.radians(yaw)
        ox = jitter_r * math.cos(a)
        oy = jitter_r * math.sin(a)
        hh = levels[i % len(levels)]
        cam_pos = np.array([center[0] + ox, center[1] + oy, target_z + hh], dtype=np.float64)
        # Strict topdown to XOY
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
        "fx": float(f),
        "fy": float(f),
        "cx": float((image_size - 1) / 2.0),
        "cy": float((image_size - 1) / 2.0),
    }
    return w2cs, meta


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


def load_cameras_from_pose_json(pose_json: Path) -> Tuple[List[CameraRecord], Dict[str, float], List[float]]:
    if not pose_json.exists():
        raise FileNotFoundError(f"pose json not found: {pose_json}")
    data = json.loads(pose_json.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"no camera items in pose json: {pose_json}")
    meta = data.get("camera_adaptive_meta", {})
    cameras: List[CameraRecord] = []
    down_angles: List[float] = []
    for it in items:
        name = str(it.get("image", ""))
        if not name:
            continue
        w2c = np.array(it["w2c"], dtype=np.float64)
        fx = float(it["fx"])
        fy = float(it["fy"])
        cx = float(it["cx"])
        cy = float(it["cy"])
        cameras.append(
            CameraRecord(
                name=name,
                mask_path=Path(name),
                w2c=w2c,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
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

    pipe_root = scene_path / args.output_subdir
    render_dir = pipe_root / "renders_topdown"
    sam3_dir = pipe_root / "sam3_instance"
    pose_dir = scene_path / "render_view_poses_topdown"
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
    cam_meta: Dict[str, float] = {}
    down_angles: List[float] = []
    if not args.skip_render:
        w2cs, cam_meta = build_topdown_cameras(
            xyz=xyz_proj,
            num_views=int(args.num_views),
            image_size=int(args.image_size),
            fov_deg=float(args.fov_deg),
            xy_jitter_ratio=float(args.xy_jitter_ratio),
        )

        # True Gaussian rendering
        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(str(input_ply))
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

        fx, fy = float(cam_meta["fx"]), float(cam_meta["fy"])
        cx, cy = float(cam_meta["cx"]), float(cam_meta["cy"])
        W = int(args.image_size)
        H = int(args.image_size)

        pose_items = []
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
                )
            )
            pose_items.append(
                {
                    "image": name,
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

        (pose_dir / "topdown_camera_poses.json").write_text(
            json.dumps(
                {
                    "scene_path": str(scene_path),
                    "input_ply": str(input_ply),
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
        pose_json = pose_dir / "topdown_camera_poses.json"
        cameras, cam_meta, down_angles = load_cameras_from_pose_json(pose_json)

    if args.render_only:
        print("[done] render-only completed")
        print(f"[out] render_dir: {render_dir}")
        print(f"[out] poses     : {pose_dir / 'topdown_camera_poses.json'}")
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
            "xy_jitter_ratio": float(args.xy_jitter_ratio),
        },
    }
    (pipe_root / "pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] pipeline completed")
    print(f"[out] report: {pipe_root / 'pipeline_report.json'}")
    print(f"[out] poses : {pose_dir / 'topdown_camera_poses.json'}")
    print(f"[out] 3d ply: {out_ply}")


if __name__ == "__main__":
    main()
