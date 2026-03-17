import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:
    raise SystemExit("scikit-learn is required. Install it with: pip install scikit-learn") from exc


DEFAULT_ID2LABEL = {
    0: "background",
    1: "vehicle",
    2: "person",
    3: "bicycle",
    4: "vegetation",
    5: "road",
    6: "traffic_facility",
    7: "other",
}


@dataclass
class CameraRecord:
    name: str
    mask_path: Path
    w2c: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class _ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class _ColmapImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def parse_id2label(text: str) -> Dict[int, str]:
    if not text:
        return DEFAULT_ID2LABEL.copy()
    out: Dict[int, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid id2label item: {item}")
        k, v = item.split(":", 1)
        out[int(k.strip())] = v.strip()
    return out


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    out = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch in (" ", "/"):
            safe.append("_")
    s = "".join(safe).strip("_")
    return s if s else "unknown"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_mask(path: Path, level: int) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        return arr.astype(np.int32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0].astype(np.int32)
        if level < 0 or level >= arr.shape[0]:
            raise ValueError(f"mask_level={level} out of range for {path}, shape={arr.shape}")
        return arr[level].astype(np.int32)
    raise ValueError(f"Unsupported mask shape {arr.shape} in {path}")


def load_index_mask(path: Path, level: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_mask(path, level)

    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return arr.astype(np.int32)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0].astype(np.int32)
        # For RGB-like masks, use the first channel.
        return arr[:, :, 0].astype(np.int32)
    raise ValueError(f"Unsupported image mask shape {arr.shape} in {path}")


def get_focal_from_fov(camera_angle_x: float, width: int) -> float:
    return width / (2.0 * math.tan(camera_angle_x / 2.0))


def _find_mask_file(language_feature_dir: Path, frame_file_path: str, mask_suffix: str) -> Optional[Path]:
    base = Path(frame_file_path).name
    stem = Path(frame_file_path).stem
    candidates = [
        language_feature_dir / f"{base}{mask_suffix}",
        language_feature_dir / f"{stem}{mask_suffix}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _format_index_pattern(pattern: str, index: int) -> str:
    try:
        return pattern.format(index=index)
    except Exception:
        pass
    try:
        return pattern.format(index)
    except Exception as exc:
        raise ValueError(
            f"Invalid instance_mask_index_pattern={pattern!r}. "
            "Use style like 'sam_mask_instance_view_{index:04d}.png'."
        ) from exc


def _find_instance_mask_file(
    instance_mask_dir: Path,
    camera_name: str,
    camera_index: int,
    mode: str,
    mask_suffix: str,
    index_pattern: str,
) -> Optional[Path]:
    mode = (mode or "name").lower()
    candidates: List[Path] = []
    if mode == "index":
        candidates.append(instance_mask_dir / _format_index_pattern(index_pattern, camera_index))
    else:
        cam_name = Path(camera_name).name
        cam_stem = Path(camera_name).stem
        candidates.extend(
            [
                instance_mask_dir / f"{cam_name}{mask_suffix}",
                instance_mask_dir / f"{cam_stem}{mask_suffix}",
            ]
        )

    for p in candidates:
        if p.exists():
            return p
    return None


def load_instance_masks_for_cameras(
    cameras: List[CameraRecord],
    instance_mask_dir: Path,
    mode: str,
    mask_suffix: str,
    index_pattern: str,
) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for i, cam in enumerate(cameras):
        p = _find_instance_mask_file(
            instance_mask_dir=instance_mask_dir,
            camera_name=cam.name,
            camera_index=i,
            mode=mode,
            mask_suffix=mask_suffix,
            index_pattern=index_pattern,
        )
        if p is not None:
            out[i] = p
    return out


def load_cameras_from_transforms(
    scene_path: Path,
    transforms_name: str,
    language_feature_dir: Path,
    mask_suffix: str,
) -> List[CameraRecord]:
    path = scene_path / transforms_name
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    global_w = data.get("w")
    global_h = data.get("h")
    global_fx = data.get("fl_x")
    global_fy = data.get("fl_y")
    global_cx = data.get("cx")
    global_cy = data.get("cy")
    camera_angle_x = data.get("camera_angle_x")

    records: List[CameraRecord] = []
    for frame in data.get("frames", []):
        frame_file = str(frame.get("file_path", ""))
        if not frame_file:
            continue

        mask_path = _find_mask_file(language_feature_dir, frame_file, mask_suffix)
        if mask_path is None:
            continue

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        # Keep the same coordinate convention used in OpenGaussian dataset loader.
        c2w[:3, 1:3] *= -1.0
        w2c = np.linalg.inv(c2w)

        k = frame.get("K")
        if k is not None:
            fx = float(k[0][0])
            fy = float(k[1][1])
            cx = float(k[0][2])
            cy = float(k[1][2])
        else:
            fw = frame.get("w", global_w)
            fh = frame.get("h", global_h)
            if fw is None or fh is None:
                # Will be filled after loading mask shape.
                fw = 0
                fh = 0

            fx = float(frame.get("fl_x", global_fx) if frame.get("fl_x", global_fx) is not None else 0.0)
            fy = float(frame.get("fl_y", global_fy) if frame.get("fl_y", global_fy) is not None else 0.0)
            cx = float(frame.get("cx", global_cx) if frame.get("cx", global_cx) is not None else -1.0)
            cy = float(frame.get("cy", global_cy) if frame.get("cy", global_cy) is not None else -1.0)

            if fx <= 0.0 and camera_angle_x is not None and fw > 0:
                fx = get_focal_from_fov(float(camera_angle_x), int(fw))
            if fy <= 0.0:
                fy = fx
            if cx < 0.0 and fw > 0:
                cx = float(fw) / 2.0
            if cy < 0.0 and fh > 0:
                cy = float(fh) / 2.0

        records.append(
            CameraRecord(
                name=Path(frame_file).name,
                mask_path=mask_path,
                w2c=w2c.astype(np.float64),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
            )
        )
    return records


def load_cameras_from_colmap(
    scene_path: Path,
    language_feature_dir: Path,
    mask_suffix: str,
) -> List[CameraRecord]:
    sparse0 = scene_path / "sparse" / "0"
    if not sparse0.exists():
        return []

    ext_txt = sparse0 / "images.txt"
    int_txt = sparse0 / "cameras.txt"
    if not (ext_txt.exists() and int_txt.exists()):
        return []

    cam_intrinsics = _read_colmap_intrinsics_text(int_txt)
    cam_extrinsics = _read_colmap_extrinsics_text(ext_txt)

    records: List[CameraRecord] = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics.get(extr.camera_id)
        if intr is None:
            continue

        mask_path = _find_mask_file(language_feature_dir, str(extr.name), mask_suffix)
        if mask_path is None:
            continue

        # OpenGaussian's COLMAP camera convention from dataset_readers.py:
        # R = transpose(qvec2rotmat), T = tvec, and world2view uses R.transpose() internally.
        # So here we reconstruct w2c exactly as getWorld2View(R, T) does.
        R = np.transpose(_qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec, dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R.transpose()
        w2c[:3, 3] = T

        width = int(intr.width)
        height = int(intr.height)
        model = intr.model.upper()
        if model == "SIMPLE_PINHOLE":
            fx = float(intr.params[0])
            fy = float(intr.params[0])
            cx = float(intr.params[1]) if len(intr.params) > 1 else (width / 2.0)
            cy = float(intr.params[2]) if len(intr.params) > 2 else (height / 2.0)
        elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "RADIAL", "SIMPLE_RADIAL"):
            fx = float(intr.params[0])
            fy = float(intr.params[1]) if len(intr.params) > 1 else fx
            cx = float(intr.params[2]) if len(intr.params) > 2 else (width / 2.0)
            cy = float(intr.params[3]) if len(intr.params) > 3 else (height / 2.0)
        else:
            # Skip unsupported intrinsics models in this script.
            continue

        records.append(
            CameraRecord(
                name=Path(str(extr.name)).name,
                mask_path=mask_path,
                w2c=w2c,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )
    return records


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3])
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _read_colmap_intrinsics_text(path: Path) -> Dict[int, _ColmapCamera]:
    cameras: Dict[int, _ColmapCamera] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            if len(elems) < 5:
                continue
            cam_id = int(elems[0])
            model = elems[1]
            width = int(float(elems[2]))
            height = int(float(elems[3]))
            params = [float(x) for x in elems[4:]]
            cameras[cam_id] = _ColmapCamera(
                id=cam_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def _read_colmap_extrinsics_text(path: Path) -> Dict[int, _ColmapImage]:
    images: Dict[int, _ColmapImage] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    # COLMAP images.txt uses two lines per image:
    # line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    # line 2: POINTS2D ...
    i = 0
    while i < len(lines):
        elems = lines[i].split()
        if len(elems) >= 10 and elems[0].isdigit():
            image_id = int(elems[0])
            qvec = np.array([float(elems[1]), float(elems[2]), float(elems[3]), float(elems[4])], dtype=np.float64)
            tvec = np.array([float(elems[5]), float(elems[6]), float(elems[7])], dtype=np.float64)
            camera_id = int(elems[8])
            name = elems[9]
            images[image_id] = _ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
            i += 2
        else:
            i += 1
    return images


def read_xyz_and_quality(vertex) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    quality = np.ones((len(xyz),), dtype=np.float32)
    names = set(vertex.data.dtype.names)

    if "opacity" in names:
        quality *= sigmoid(vertex["opacity"].astype(np.float32))
    if all(k in names for k in ("scale_0", "scale_1", "scale_2")):
        scales = np.vstack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]]).T.astype(np.float32)
        scale_penalty = np.exp(np.max(scales, axis=1))
        quality /= (1.0 + scale_penalty)
    return xyz, quality


def normalize_minmax(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data
    d_min = data.min(axis=0, keepdims=True)
    d_max = data.max(axis=0, keepdims=True)
    rng = d_max - d_min
    rng[rng < 1e-8] = 1.0
    return (data - d_min) / rng


def get_ins_feat_array(vertex_arr: np.ndarray) -> Optional[np.ndarray]:
    names = list(vertex_arr.dtype.names or [])
    numeric = [n for n in names if n.startswith("ins_feat_")]
    if numeric:
        def _key(nm: str) -> int:
            try:
                return int(nm.split("_")[-1])
            except ValueError:
                return 0
        numeric = sorted(numeric, key=_key)
        return np.vstack([vertex_arr[n] for n in numeric]).T.astype(np.float32)

    legacy = ["ins_feat_r", "ins_feat_g", "ins_feat_b", "ins_feat_r2", "ins_feat_g2", "ins_feat_b2"]
    if all(n in names for n in legacy):
        return np.vstack([vertex_arr[n] for n in legacy]).T.astype(np.float32)
    return None


def build_instance_space(
    vertex_arr: np.ndarray,
    sem_id: int,
    instance_space: str,
    spatial_weight: float,
    feature_weight: float,
    z_weight: float,
    vehicle_sem_id: int,
    vehicle_use_ins_feat: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    xyz = np.vstack([vertex_arr["x"], vertex_arr["y"], vertex_arr["z"]]).T.astype(np.float32)
    xyz_scaled = xyz.copy()
    xyz_scaled[:, 2] *= float(z_weight)
    xyz_part = normalize_minmax(xyz_scaled) * float(spatial_weight)

    use_feat = instance_space == "xyz_feat" or (vehicle_use_ins_feat and sem_id == vehicle_sem_id)
    if use_feat:
        feat = get_ins_feat_array(vertex_arr)
        if feat is not None and feat.shape[1] > 0:
            feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-6)
            feat_part = normalize_minmax(feat) * float(feature_weight)
            return np.hstack([xyz_part, feat_part]), xyz, "xyz_feat"
    return xyz_part, xyz, "xyz"


def resolve_instance_params(
    sem_id: int,
    args,
) -> Tuple[Optional[float], float, int, int]:
    eps = args.dbscan_eps if args.dbscan_eps > 0 else None
    eps_quantile = float(args.dbscan_eps_quantile)
    min_samples = int(args.dbscan_min_samples)
    min_instance_points = int(args.min_instance_points)

    if sem_id == int(args.vehicle_sem_id):
        if args.vehicle_dbscan_eps > 0:
            eps = float(args.vehicle_dbscan_eps)
        if args.vehicle_eps_quantile > 0:
            eps_quantile = float(args.vehicle_eps_quantile)
        if args.vehicle_min_samples > 0:
            min_samples = int(args.vehicle_min_samples)
        if args.vehicle_min_instance_points > 0:
            min_instance_points = int(args.vehicle_min_instance_points)

    return eps, eps_quantile, min_samples, min_instance_points


def pick_core_points(
    xyz: np.ndarray,
    quality: np.ndarray,
    core_ratio: float,
    min_core_points: int,
    max_core_points: int,
    mode: str,
    geo_weight: float,
) -> np.ndarray:
    n = len(xyz)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    target = int(round(n * core_ratio))
    target = max(target, min_core_points)
    target = min(target, n, max_core_points)
    if target <= 0:
        target = min(n, max(min_core_points, 1))

    mode = (mode or "quality").lower()
    if mode == "centroid":
        center = xyz.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(xyz - center, axis=1)
        idx = np.argsort(dist)[:target]
    elif mode == "mixed":
        center = xyz.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(xyz - center, axis=1)
        d_min, d_max = float(dist.min()), float(dist.max())
        if d_max - d_min < 1e-8:
            dist_score = np.ones_like(dist, dtype=np.float32)
        else:
            dist_score = 1.0 - ((dist - d_min) / (d_max - d_min))

        q = quality.astype(np.float32)
        q_min, q_max = float(q.min()), float(q.max())
        if q_max - q_min < 1e-8:
            q_score = np.ones_like(q, dtype=np.float32)
        else:
            q_score = (q - q_min) / (q_max - q_min)

        w = float(np.clip(geo_weight, 0.0, 1.0))
        score = w * dist_score + (1.0 - w) * q_score
        idx = np.argsort(-score)[:target]
    else:
        idx = np.argsort(-quality)[:target]
    return xyz[idx]


def project_points(
    xyz: np.ndarray,
    w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xyz) == 0:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty, empty

    xyz_h = np.concatenate([xyz.astype(np.float64), np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    cam = xyz_h @ w2c.T
    z = cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty, empty

    valid_idx = np.where(valid)[0].astype(np.int32)
    x = cam[valid, 0]
    y = cam[valid, 1]
    z = z[valid]
    u = np.round(fx * (x / z) + cx).astype(np.int32)
    v = np.round(fy * (y / z) + cy).astype(np.int32)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty, empty
    u = u[inside]
    v = v[inside]
    z = z[inside]
    valid_idx = valid_idx[inside]

    # Simple per-cluster z-buffer: keep nearest projected point per pixel.
    linear = (v.astype(np.int64) * np.int64(width) + u.astype(np.int64))
    order = np.lexsort((z, linear))
    linear_s = linear[order]
    keep = np.ones((len(order),), dtype=np.bool_)
    keep[1:] = linear_s[1:] != linear_s[:-1]
    sel = order[keep]
    return u[sel], v[sel], valid_idx[sel]


def vote_cluster_to_mask_ids(
    core_xyz: np.ndarray,
    cameras: List[CameraRecord],
    mask_level: int,
    vote_stride: int,
    ignore_ids: List[int],
) -> Dict[int, int]:
    counter: Counter = Counter()
    if len(core_xyz) == 0:
        return {}

    if vote_stride <= 0:
        vote_stride = 1

    for i, cam in enumerate(cameras):
        if i % vote_stride != 0:
            continue
        if cam.mask_path is None:
            continue

        mask = load_mask(cam.mask_path, mask_level)
        h, w = mask.shape
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        if fx <= 0 or fy <= 0:
            # Fallback from mask size if intrinsics missing.
            fx = float(max(w, 1))
            fy = fx
            cx = float(w) / 2.0
            cy = float(h) / 2.0

        u, v, _ = project_points(core_xyz, cam.w2c, fx, fy, cx, cy, w, h)
        if len(u) == 0:
            continue
        ids = mask[v, u].astype(np.int32)
        if ignore_ids:
            keep = ~np.isin(ids, np.array(ignore_ids, dtype=np.int32))
            ids = ids[keep]
            if len(ids) == 0:
                continue
        uniq, cnt = np.unique(ids, return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist()):
            counter[int(k)] += int(c)
    return dict(counter)


def entropy_from_counts(counts: Dict[int, int]) -> float:
    total = int(sum(counts.values()))
    if total <= 0:
        return 0.0
    p = np.array(list(counts.values()), dtype=np.float64) / float(total)
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def get_vote_stats(counts: Dict[int, int]) -> Dict[str, float]:
    if not counts:
        return {
            "total_votes": 0,
            "top1_id": -1,
            "top1_count": 0,
            "top1_ratio": 0.0,
            "top2_id": -1,
            "top2_count": 0,
            "top2_ratio": 0.0,
            "entropy": 0.0,
        }
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    total = int(sum(v for _, v in items))
    top1_id, top1_count = items[0]
    if len(items) > 1:
        top2_id, top2_count = items[1]
    else:
        top2_id, top2_count = -1, 0
    return {
        "total_votes": total,
        "top1_id": int(top1_id),
        "top1_count": int(top1_count),
        "top1_ratio": float(top1_count / max(total, 1)),
        "top2_id": int(top2_id),
        "top2_count": int(top2_count),
        "top2_ratio": float(top2_count / max(total, 1)),
        "entropy": entropy_from_counts(counts),
    }


def write_subset_ply(vertex, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(out_path))


def merge_vertices(vertices: List[np.ndarray]) -> np.ndarray:
    if not vertices:
        return np.empty((0,), dtype=np.float32)
    if len(vertices) == 1:
        return vertices[0]
    return np.concatenate(vertices, axis=0)


def write_semantic_instance_ply(
    vertex: np.ndarray,
    out_path: Path,
    semantic_id: int,
    instance_id: np.ndarray,
):
    old_dtype = vertex.dtype
    new_dtype = np.dtype(old_dtype.descr + [("semantic_id", "i4"), ("instance_id", "i4")])
    out = np.empty(len(vertex), dtype=new_dtype)
    for n in old_dtype.names:
        out[n] = vertex[n]
    out["semantic_id"] = np.int32(semantic_id)
    out["instance_id"] = instance_id.astype(np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(out_path))


def auto_eps_from_knn(xyz: np.ndarray, k: int, quantile: float) -> float:
    if len(xyz) <= 1:
        return 0.0
    k = int(max(1, min(k, len(xyz) - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
    nn.fit(xyz)
    dists, _ = nn.kneighbors(xyz)
    kth = dists[:, k]
    return float(np.quantile(kth, np.clip(quantile, 0.0, 1.0)))


def dbscan_instances(
    data: np.ndarray,
    min_samples: int,
    eps: Optional[float],
    eps_quantile: float,
    eps_knn: int,
    min_instance_points: int,
) -> Tuple[np.ndarray, float, int]:
    n = len(data)
    if n == 0:
        return np.zeros((0,), dtype=np.int32), 0.0, 0

    if eps is None or eps <= 0.0:
        eps_used = auto_eps_from_knn(data, eps_knn, eps_quantile)
        if eps_used <= 0.0:
            eps_used = 1e-4
    else:
        eps_used = float(eps)

    model = DBSCAN(eps=eps_used, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = model.fit_predict(data).astype(np.int32)

    # Remove tiny clusters and remap to contiguous 1..N. Noise is 0.
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for old in sorted([x for x in np.unique(labels).tolist() if x >= 0]):
        m = labels == old
        if int(np.sum(m)) < min_instance_points:
            continue
        out[m] = next_id
        next_id += 1
    return out, eps_used, next_id - 1


def proj2d_instances(
    xyz: np.ndarray,
    cameras: List[CameraRecord],
    instance_mask_paths: Dict[int, Path],
    vote_stride: int,
    instance_mask_level: int,
    instance_mask_ignore_ids: List[int],
    min_mask_points: int,
    merge_iou: float,
    min_point_votes: int,
    min_instance_points: int,
) -> Tuple[np.ndarray, int, Dict[str, int]]:
    n = len(xyz)
    if n == 0:
        return np.zeros((0,), dtype=np.int32), 0, {
            "views_considered": 0,
            "views_with_masks": 0,
            "views_used": 0,
            "local_instances_used": 0,
            "global_hypotheses": 0,
        }

    if vote_stride <= 0:
        vote_stride = 1
    ignore_ids = set(instance_mask_ignore_ids)
    min_mask_points = max(int(min_mask_points), 1)
    min_point_votes = max(int(min_point_votes), 1)
    min_instance_points = max(int(min_instance_points), 1)
    merge_iou = float(np.clip(merge_iou, 0.0, 1.0))

    global_sets: List[set] = []
    global_votes: List[Counter] = []
    views_considered = 0
    views_with_masks = 0
    views_used = 0
    local_instances_used = 0

    for i, cam in enumerate(cameras):
        if i % vote_stride != 0:
            continue
        views_considered += 1
        mask_path = instance_mask_paths.get(i)
        if mask_path is None:
            continue
        views_with_masks += 1

        mask = load_index_mask(mask_path, instance_mask_level)
        if mask.ndim != 2:
            continue
        h, w = mask.shape
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        if fx <= 0 or fy <= 0:
            fx = float(max(w, 1))
            fy = fx
            cx = float(w) / 2.0
            cy = float(h) / 2.0

        u, v, pidx = project_points(xyz, cam.w2c, fx, fy, cx, cy, w, h)
        if len(pidx) == 0:
            continue
        ids = mask[v, u].astype(np.int32)
        if len(ignore_ids) > 0:
            keep = np.ones((len(ids),), dtype=np.bool_)
            for ig in ignore_ids:
                keep &= (ids != int(ig))
            ids = ids[keep]
            pidx = pidx[keep]
            if len(pidx) == 0:
                continue

        views_used += 1
        uniq_ids, uniq_cnt = np.unique(ids, return_counts=True)
        for inst_id, inst_cnt in zip(uniq_ids.tolist(), uniq_cnt.tolist()):
            if inst_cnt < min_mask_points:
                continue
            local_pts = np.unique(pidx[ids == inst_id]).astype(np.int32)
            if len(local_pts) < min_mask_points:
                continue
            local_set = set(local_pts.tolist())
            if not local_set:
                continue
            local_instances_used += 1

            best_idx = -1
            best_iou = 0.0
            for gi, gset in enumerate(global_sets):
                inter = len(local_set & gset)
                if inter <= 0:
                    continue
                union = len(local_set) + len(gset) - inter
                if union <= 0:
                    continue
                iou = float(inter / union)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_idx >= 0 and best_iou >= merge_iou:
                global_sets[best_idx].update(local_set)
                global_votes[best_idx].update(local_set)
            else:
                global_sets.append(set(local_set))
                c = Counter()
                c.update(local_set)
                global_votes.append(c)

    if not global_votes:
        return np.zeros((n,), dtype=np.int32), 0, {
            "views_considered": int(views_considered),
            "views_with_masks": int(views_with_masks),
            "views_used": int(views_used),
            "local_instances_used": int(local_instances_used),
            "global_hypotheses": 0,
        }

    best_votes = np.zeros((n,), dtype=np.int32)
    best_inst = np.zeros((n,), dtype=np.int32)
    for inst_idx, vote_counter in enumerate(global_votes, start=1):
        for pid, votes in vote_counter.items():
            p = int(pid)
            v = int(votes)
            if v > int(best_votes[p]):
                best_votes[p] = v
                best_inst[p] = inst_idx
            elif v == int(best_votes[p]) and v > 0 and inst_idx < int(best_inst[p]):
                best_inst[p] = inst_idx

    best_inst[best_votes < min_point_votes] = 0

    out = np.zeros_like(best_inst, dtype=np.int32)
    next_id = 1
    for old in sorted(np.unique(best_inst).tolist()):
        if old <= 0:
            continue
        m = best_inst == old
        if int(np.sum(m)) < min_instance_points:
            continue
        out[m] = next_id
        next_id += 1

    return out, (next_id - 1), {
        "views_considered": int(views_considered),
        "views_with_masks": int(views_with_masks),
        "views_used": int(views_used),
        "local_instances_used": int(local_instances_used),
        "global_hypotheses": int(len(global_votes)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Semantic + instance pipeline: 3D cluster voting to 2D semantic ids and per-semantic instance segmentation."
    )
    parser.add_argument("--scene_path", required=True, help="Scene path containing transforms*.json and language_features/")
    parser.add_argument("--cluster_dir", required=True, help="Directory with class_*.ply from KMeans")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--cluster_glob", default="class_*.ply", help="Glob pattern for cluster files")
    parser.add_argument("--language_features_subdir", default="language_features", help="Subdir for *_s.npy masks")
    parser.add_argument("--mask_suffix", default="_s.npy", help="Mask filename suffix")
    parser.add_argument("--mask_level", type=int, default=0, help="Mask level if mask is [L,H,W]")
    parser.add_argument("--transforms_files", default="transforms_train.json,transforms_test.json",
                        help="Comma-separated transforms files to load")
    parser.add_argument("--vote_stride", type=int, default=2, help="Use every Nth camera for voting")
    parser.add_argument("--core_ratio", type=float, default=0.3, help="Core point ratio in each cluster")
    parser.add_argument("--min_core_points", type=int, default=500, help="Minimum core points per cluster")
    parser.add_argument("--max_core_points", type=int, default=20000, help="Maximum core points per cluster")
    parser.add_argument("--core_mode", default="quality", choices=["quality", "centroid", "mixed"],
                        help="Core point sampling mode")
    parser.add_argument("--core_geo_weight", type=float, default=0.7,
                        help="Geometry weight for --core_mode mixed")
    parser.add_argument("--min_votes", type=int, default=2000, help="Minimum vote count to accept semantic assignment")
    parser.add_argument("--min_top1_ratio", type=float, default=0.55, help="Minimum top-1 ratio to accept assignment")
    parser.add_argument("--vote_ignore_ids", default="", help="Ignore these ids during voting, e.g. 0,7")
    parser.add_argument("--unknown_id", type=int, default=255, help="Semantic id for low-confidence clusters")
    parser.add_argument("--id2label",
                        default="0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other",
                        help="id->label mapping, e.g. '1:vehicle,5:road'")
    parser.add_argument("--stuff_ids", default="0,4,5,6,7,255",
                        help="Semantic ids that skip instance segmentation")
    parser.add_argument("--instance_only_ids", default="",
                        help="Optional semantic ids to run instance segmentation, e.g. 1,2")
    parser.add_argument("--instance_space", default="xyz", choices=["xyz", "xyz_feat"],
                        help="Instance clustering space")
    parser.add_argument("--instance_spatial_weight", type=float, default=1.0,
                        help="Spatial weight in instance clustering")
    parser.add_argument("--instance_feature_weight", type=float, default=1.2,
                        help="Feature weight when using xyz_feat")
    parser.add_argument("--instance_z_weight", type=float, default=0.6,
                        help="Scale Z before clustering to reduce vertical bridge effects")
    parser.add_argument("--vehicle_sem_id", type=int, default=1, help="Vehicle semantic id")
    parser.add_argument("--vehicle_use_ins_feat", action="store_true",
                        help="Use ins_feat+xyz for vehicle even when instance_space=xyz")
    parser.add_argument("--vehicle_dbscan_eps", type=float, default=-1.0, help="Vehicle fixed eps override")
    parser.add_argument("--vehicle_eps_quantile", type=float, default=-1.0, help="Vehicle eps quantile override")
    parser.add_argument("--vehicle_min_samples", type=int, default=-1, help="Vehicle min_samples override")
    parser.add_argument("--vehicle_min_instance_points", type=int, default=-1,
                        help="Vehicle min_instance_points override")
    parser.add_argument("--dbscan_eps", type=float, default=-1.0, help="Fixed eps. <=0 means auto from kNN quantile")
    parser.add_argument("--dbscan_eps_quantile", type=float, default=0.92, help="kNN distance quantile for auto eps")
    parser.add_argument("--dbscan_eps_knn", type=int, default=20, help="k for kNN distance when auto eps")
    parser.add_argument("--dbscan_min_samples", type=int, default=20, help="DBSCAN min_samples")
    parser.add_argument("--min_instance_points", type=int, default=300, help="Drop DBSCAN clusters smaller than this")
    parser.add_argument("--instance_method", default="dbscan", choices=["dbscan", "proj2d"],
                        help="Instance segmentation method: dbscan or projection voting from 2D instance masks")
    parser.add_argument("--instance_mask_dir", default="",
                        help="Directory containing 2D instance masks (required when --instance_method proj2d)")
    parser.add_argument("--instance_mask_mode", default="index", choices=["index", "name"],
                        help="How to match camera to instance mask file")
    parser.add_argument("--instance_mask_suffix", default="_instance.png",
                        help="Mask suffix for --instance_mask_mode name")
    parser.add_argument("--instance_mask_index_pattern", default="sam_mask_instance_view_{index:04d}.png",
                        help="Filename pattern for --instance_mask_mode index")
    parser.add_argument("--instance_mask_level", type=int, default=0,
                        help="Mask level when instance masks are .npy files")
    parser.add_argument("--instance_mask_ignore_ids", default="0",
                        help="Ignore these mask ids when projecting instances")
    parser.add_argument("--instance_min_mask_points", type=int, default=40,
                        help="Minimum projected points in one view-mask to use it as an instance cue")
    parser.add_argument("--instance_match_iou", type=float, default=0.2,
                        help="IoU threshold to merge per-view instance cues across views")
    parser.add_argument("--instance_min_point_votes", type=int, default=2,
                        help="Minimum multi-view votes required to keep point-level instance assignment")
    args = parser.parse_args()

    scene_path = Path(args.scene_path)
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_models_dir = output_dir / "semantic_models"
    semantic_models_dir.mkdir(parents=True, exist_ok=True)
    semantic_labeled_dir = output_dir / "semantic_instance"
    semantic_labeled_dir.mkdir(parents=True, exist_ok=True)

    id2label = parse_id2label(args.id2label)
    stuff_ids = set(parse_int_list(args.stuff_ids))
    instance_only_ids = set(parse_int_list(args.instance_only_ids))
    vote_ignore_ids = parse_int_list(args.vote_ignore_ids)
    instance_mask_ignore_ids = parse_int_list(args.instance_mask_ignore_ids)

    language_feature_dir = scene_path / args.language_features_subdir
    if not language_feature_dir.exists():
        raise FileNotFoundError(f"language feature dir not found: {language_feature_dir}")

    tf_names = [x.strip() for x in args.transforms_files.split(",") if x.strip()]
    cameras: List[CameraRecord] = []
    for name in tf_names:
        cameras.extend(
            load_cameras_from_transforms(
                scene_path=scene_path,
                transforms_name=name,
                language_feature_dir=language_feature_dir,
                mask_suffix=args.mask_suffix,
            )
        )
    camera_source = "transforms"
    if not cameras:
        cameras = load_cameras_from_colmap(
            scene_path=scene_path,
            language_feature_dir=language_feature_dir,
            mask_suffix=args.mask_suffix,
        )
        camera_source = "colmap"
    if not cameras:
        raise SystemExit(
            "No valid cameras loaded from transforms or COLMAP. "
            "Check scene_path, sparse/0, and language_features/*_s.npy names."
        )

    instance_mask_paths: Dict[int, Path] = {}
    if args.instance_method == "proj2d":
        if not args.instance_mask_dir:
            raise SystemExit("--instance_mask_dir is required when --instance_method proj2d")
        instance_mask_dir = Path(args.instance_mask_dir)
        if not instance_mask_dir.exists():
            raise FileNotFoundError(f"instance mask dir not found: {instance_mask_dir}")
        instance_mask_paths = load_instance_masks_for_cameras(
            cameras=cameras,
            instance_mask_dir=instance_mask_dir,
            mode=args.instance_mask_mode,
            mask_suffix=args.instance_mask_suffix,
            index_pattern=args.instance_mask_index_pattern,
        )
        if not instance_mask_paths:
            raise SystemExit(
                "No instance masks matched cameras. "
                "Check --instance_mask_mode and --instance_mask_index_pattern/--instance_mask_suffix."
            )
        print(f"[proj2d] matched instance masks: {len(instance_mask_paths)}/{len(cameras)} cameras")

    cluster_files = sorted(cluster_dir.glob(args.cluster_glob))
    if not cluster_files:
        raise SystemExit(f"No cluster files found in {cluster_dir} with glob {args.cluster_glob}")

    cluster_infos = []
    semantic_to_vertices: Dict[int, List[np.ndarray]] = defaultdict(list)

    for p in cluster_files:
        ply = PlyData.read(str(p))
        vertex = ply["vertex"].data
        xyz, quality = read_xyz_and_quality(ply["vertex"])
        core_xyz = pick_core_points(
            xyz=xyz,
            quality=quality,
            core_ratio=args.core_ratio,
            min_core_points=args.min_core_points,
            max_core_points=args.max_core_points,
            mode=args.core_mode,
            geo_weight=args.core_geo_weight,
        )
        votes = vote_cluster_to_mask_ids(
            core_xyz=core_xyz,
            cameras=cameras,
            mask_level=args.mask_level,
            vote_stride=args.vote_stride,
            ignore_ids=vote_ignore_ids,
        )
        stats = get_vote_stats(votes)

        assigned_sem_id = int(stats["top1_id"])
        accepted = (stats["total_votes"] >= args.min_votes) and (stats["top1_ratio"] >= args.min_top1_ratio)
        if not accepted:
            assigned_sem_id = int(args.unknown_id)

        sem_label = id2label.get(assigned_sem_id, f"id_{assigned_sem_id}")
        sem_safe = sanitize_name(sem_label)
        dst = semantic_models_dir / f"{assigned_sem_id:03d}_{sem_safe}" / p.name
        write_subset_ply(vertex, dst)
        semantic_to_vertices[assigned_sem_id].append(vertex)

        cluster_infos.append(
            {
                "cluster_file": str(p),
                "num_points": int(len(vertex)),
                "num_core_points": int(len(core_xyz)),
                "votes": {str(k): int(v) for k, v in sorted(votes.items(), key=lambda x: x[0])},
                "vote_stats": stats,
                "accepted": bool(accepted),
                "assigned_semantic_id": int(assigned_sem_id),
                "assigned_semantic_label": sem_label,
                "saved_path": str(dst),
            }
        )
        print(
            f"[cluster] {p.name}: sem={assigned_sem_id}({sem_label}) "
            f"top1_ratio={stats['top1_ratio']:.3f} votes={stats['total_votes']}"
        )

    semantic_summary = []
    for sem_id, parts in sorted(semantic_to_vertices.items(), key=lambda x: x[0]):
        sem_label = id2label.get(sem_id, f"id_{sem_id}")
        sem_safe = sanitize_name(sem_label)
        merged = merge_vertices(parts)
        merged_ply = semantic_models_dir / f"{sem_id:03d}_{sem_safe}" / "merged_semantic.ply"
        write_subset_ply(merged, merged_ply)

        if instance_only_ids:
            run_instance = (sem_id not in stuff_ids) and (sem_id in instance_only_ids)
        else:
            run_instance = sem_id not in stuff_ids

        proj2d_stats = {
            "views_considered": 0,
            "views_with_masks": 0,
            "views_used": 0,
            "local_instances_used": 0,
            "global_hypotheses": 0,
        }

        if run_instance:
            if args.instance_method == "proj2d":
                xyz_raw = np.vstack([merged["x"], merged["y"], merged["z"]]).T.astype(np.float32)
                _, _, _, min_inst_pts_used = resolve_instance_params(
                    sem_id=sem_id,
                    args=args,
                )
                instance_id, num_instances, proj2d_stats = proj2d_instances(
                    xyz=xyz_raw,
                    cameras=cameras,
                    instance_mask_paths=instance_mask_paths,
                    vote_stride=args.vote_stride,
                    instance_mask_level=args.instance_mask_level,
                    instance_mask_ignore_ids=instance_mask_ignore_ids,
                    min_mask_points=args.instance_min_mask_points,
                    merge_iou=args.instance_match_iou,
                    min_point_votes=args.instance_min_point_votes,
                    min_instance_points=min_inst_pts_used,
                )
                instance_space_used = "proj2d"
                min_samples_used = 0
                eps_quantile_used = 0.0
                eps_used = 0.0
            else:
                data_cluster, xyz_raw, instance_space_used = build_instance_space(
                    vertex_arr=merged,
                    sem_id=sem_id,
                    instance_space=args.instance_space,
                    spatial_weight=args.instance_spatial_weight,
                    feature_weight=args.instance_feature_weight,
                    z_weight=args.instance_z_weight,
                    vehicle_sem_id=args.vehicle_sem_id,
                    vehicle_use_ins_feat=args.vehicle_use_ins_feat,
                )
                eps_arg, eps_quantile_used, min_samples_used, min_inst_pts_used = resolve_instance_params(
                    sem_id=sem_id,
                    args=args,
                )
                instance_id, eps_used, num_instances = dbscan_instances(
                    data=data_cluster,
                    min_samples=min_samples_used,
                    eps=eps_arg,
                    eps_quantile=eps_quantile_used,
                    eps_knn=args.dbscan_eps_knn,
                    min_instance_points=min_inst_pts_used,
                )
            out_ply = semantic_labeled_dir / f"{sem_id:03d}_{sem_safe}_sem_ins.ply"
            write_semantic_instance_ply(merged, out_ply, sem_id, instance_id)
        else:
            xyz_raw = np.vstack([merged["x"], merged["y"], merged["z"]]).T.astype(np.float32)
            instance_space_used = "disabled"
            min_samples_used = 0
            min_inst_pts_used = 0
            eps_quantile_used = 0.0
            eps_used = 0.0
            num_instances = 0
            instance_id = np.zeros((len(merged),), dtype=np.int32)
            out_ply = semantic_labeled_dir / f"{sem_id:03d}_{sem_safe}_sem_ins.ply"
            write_semantic_instance_ply(merged, out_ply, sem_id, instance_id)

        # Export each instance as separate PLY.
        if run_instance and num_instances > 0:
            inst_dir = semantic_labeled_dir / f"{sem_id:03d}_{sem_safe}_instances"
            inst_dir.mkdir(parents=True, exist_ok=True)
            for inst in range(1, num_instances + 1):
                m = instance_id == inst
                if not np.any(m):
                    continue
                inst_ply = inst_dir / f"instance_{inst:03d}.ply"
                write_subset_ply(merged[m], inst_ply)

        semantic_summary.append(
            {
                "semantic_id": int(sem_id),
                "semantic_label": sem_label,
                "num_clusters": int(len(parts)),
                "num_points": int(len(merged)),
                "instance_enabled": bool(run_instance),
                "instance_method": str(args.instance_method if run_instance else "disabled"),
                "num_instances": int(num_instances),
                "dbscan_eps_used": float(eps_used),
                "dbscan_eps_quantile_used": float(eps_quantile_used),
                "dbscan_min_samples_used": int(min_samples_used),
                "min_instance_points_used": int(min_inst_pts_used),
                "proj2d_views_considered": int(proj2d_stats["views_considered"]),
                "proj2d_views_with_masks": int(proj2d_stats["views_with_masks"]),
                "proj2d_views_used": int(proj2d_stats["views_used"]),
                "proj2d_local_instances_used": int(proj2d_stats["local_instances_used"]),
                "proj2d_global_hypotheses": int(proj2d_stats["global_hypotheses"]),
                "instance_space_used": str(instance_space_used),
                "merged_ply": str(merged_ply),
                "semantic_instance_ply": str(out_ply),
            }
        )
        if run_instance and args.instance_method == "proj2d":
            print(
                f"[semantic] id={sem_id} label={sem_label} points={len(merged)} "
                f"instances={num_instances} method=proj2d views_used={proj2d_stats['views_used']}"
            )
        else:
            print(
                f"[semantic] id={sem_id} label={sem_label} points={len(merged)} "
                f"instances={num_instances} eps={eps_used:.6f}"
            )

    report = {
        "scene_path": str(scene_path),
        "cluster_dir": str(cluster_dir),
        "num_cameras_used": int(len(cameras)),
        "camera_source": camera_source,
        "num_clusters": int(len(cluster_files)),
        "cluster_infos": cluster_infos,
        "semantic_summary": semantic_summary,
        "params": {
            "mask_level": int(args.mask_level),
            "vote_stride": int(args.vote_stride),
            "core_ratio": float(args.core_ratio),
            "min_core_points": int(args.min_core_points),
            "max_core_points": int(args.max_core_points),
            "core_mode": str(args.core_mode),
            "core_geo_weight": float(args.core_geo_weight),
            "min_votes": int(args.min_votes),
            "min_top1_ratio": float(args.min_top1_ratio),
            "vote_ignore_ids": vote_ignore_ids,
            "unknown_id": int(args.unknown_id),
            "stuff_ids": sorted(list(stuff_ids)),
            "instance_only_ids": sorted(list(instance_only_ids)),
            "instance_space": str(args.instance_space),
            "instance_spatial_weight": float(args.instance_spatial_weight),
            "instance_feature_weight": float(args.instance_feature_weight),
            "instance_z_weight": float(args.instance_z_weight),
            "instance_method": str(args.instance_method),
            "instance_mask_dir": str(args.instance_mask_dir),
            "instance_mask_mode": str(args.instance_mask_mode),
            "instance_mask_suffix": str(args.instance_mask_suffix),
            "instance_mask_index_pattern": str(args.instance_mask_index_pattern),
            "instance_mask_level": int(args.instance_mask_level),
            "instance_mask_ignore_ids": instance_mask_ignore_ids,
            "instance_min_mask_points": int(args.instance_min_mask_points),
            "instance_match_iou": float(args.instance_match_iou),
            "instance_min_point_votes": int(args.instance_min_point_votes),
            "vehicle_sem_id": int(args.vehicle_sem_id),
            "vehicle_use_ins_feat": bool(args.vehicle_use_ins_feat),
            "vehicle_dbscan_eps": float(args.vehicle_dbscan_eps),
            "vehicle_eps_quantile": float(args.vehicle_eps_quantile),
            "vehicle_min_samples": int(args.vehicle_min_samples),
            "vehicle_min_instance_points": int(args.vehicle_min_instance_points),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_eps_quantile": float(args.dbscan_eps_quantile),
            "dbscan_eps_knn": int(args.dbscan_eps_knn),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "min_instance_points": int(args.min_instance_points),
        },
    }
    report_path = output_dir / "semantic_instance_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"done: {output_dir}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
