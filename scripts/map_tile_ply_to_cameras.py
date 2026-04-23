import argparse
import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass
class TileInfo:
    path: Path
    vertex_count: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    expanded_min: np.ndarray
    expanded_max: np.ndarray
    image_counts: Dict[int, int] = field(default_factory=dict)
    matched_sparse_points: int = 0

    @property
    def stem(self) -> str:
        return self.path.stem

    def contains(self, xyz: Tuple[float, float, float]) -> bool:
        x, y, z = xyz
        return (
            self.expanded_min[0] <= x <= self.expanded_max[0]
            and self.expanded_min[1] <= y <= self.expanded_max[1]
            and self.expanded_min[2] <= z <= self.expanded_max[2]
        )


@dataclass
class TileGrid:
    origin: np.ndarray
    cell_size: np.ndarray
    global_min: np.ndarray
    global_max: np.ndarray
    cells: Dict[Tuple[int, int, int], List[int]]

    def lookup(self, xyz: Tuple[float, float, float]) -> List[int]:
        x, y, z = xyz
        if (
            x < self.global_min[0]
            or x > self.global_max[0]
            or y < self.global_min[1]
            or y > self.global_max[1]
            or z < self.global_min[2]
            or z > self.global_max[2]
        ):
            return []
        key = (
            int(math.floor((x - self.origin[0]) / self.cell_size[0])),
            int(math.floor((y - self.origin[1]) / self.cell_size[1])),
            int(math.floor((z - self.origin[2]) / self.cell_size[2])),
        )
        return self.cells.get(key, [])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Map each tile PLY to the COLMAP cameras/images that observe sparse "
            "points inside that tile. Outputs a per-tile images.txt whitelist "
            "plus cameras.json pose metadata."
        )
    )
    parser.add_argument("--scene_root", type=str, required=True, help="Scene root containing sparse/ and images/")
    parser.add_argument(
        "--tile_path",
        type=str,
        nargs="*",
        default=None,
        help="One or more explicit tile PLY paths. If omitted, --tile_glob is used.",
    )
    parser.add_argument(
        "--tile_glob",
        type=str,
        default="Tile_*_point_cloud_clip.ply",
        help="Glob pattern used under scene_root when --tile_path is omitted.",
    )
    parser.add_argument(
        "--sparse_dir",
        type=str,
        default="sparse/0",
        help="Relative path from scene_root to COLMAP sparse model directory.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="",
        help="Relative path from scene_root to images directory. Auto-detected when empty.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <scene_root>/tile_camera_map.",
    )
    parser.add_argument(
        "--bbox_expand_ratio",
        type=float,
        default=0.0,
        help="Expand each tile bbox by this fraction of its axis extent.",
    )
    parser.add_argument(
        "--bbox_expand_abs",
        type=float,
        default=0.0,
        help="Expand each tile bbox by this many world units on each side.",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=5,
        help="Keep only images supported by at least this many sparse points.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Optionally keep only the top-K images per tile after counting support points.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=200000,
        help="Print streaming progress every N sparse points.",
    )
    return parser.parse_args()


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Expected {num_bytes} bytes, got {len(data)}")
    return struct.unpack("<" + format_char_sequence, data)


def auto_detect_images_dir(scene_root: Path) -> str:
    for candidate in ("images", "images_8", "input", "rgb", "imgs"):
        if (scene_root / candidate).exists():
            return candidate
    raise SystemExit(
        "Could not auto-detect images directory. Please pass --images_dir explicitly."
    )


def resolve_tile_paths(scene_root: Path, explicit_tiles: Optional[Iterable[str]], tile_glob: str) -> List[Path]:
    if explicit_tiles:
        tiles = [Path(p) for p in explicit_tiles]
    else:
        tiles = sorted(scene_root.glob(tile_glob))
    if not tiles:
        raise SystemExit("No tile PLY files found.")
    missing = [str(p) for p in tiles if not p.exists()]
    if missing:
        raise SystemExit("Missing tile PLY files:\n" + "\n".join(missing))
    return tiles


def read_binary_ply_bbox(path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        fmt = None
        vertex_count = None
        in_vertex_block = False
        vertex_props = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while parsing PLY header: {path}")
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element "):
                parts = text.split()
                element_name = parts[1]
                in_vertex_block = element_name == "vertex"
                if in_vertex_block:
                    vertex_count = int(parts[2])
            elif text.startswith("property ") and in_vertex_block:
                parts = text.split()
                if parts[1] == "list":
                    raise ValueError(f"PLY list properties are not supported: {path}")
                prop_type = parts[1]
                prop_name = parts[2]
                if prop_type not in PLY_DTYPE_MAP:
                    raise ValueError(f"Unsupported PLY property type '{prop_type}' in {path}")
                vertex_props.append((prop_name, PLY_DTYPE_MAP[prop_type]))
            elif text == "end_header":
                header_end = f.tell()
                break

    if fmt != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got '{fmt}' in {path}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"Invalid vertex count in {path}")

    dtype = np.dtype(vertex_props)
    names = set(dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex properties must include x/y/z: {path}")

    vertices = np.memmap(path, dtype=dtype, mode="r", offset=header_end, shape=(vertex_count,))
    bbox_min = np.array(
        [float(vertices["x"].min()), float(vertices["y"].min()), float(vertices["z"].min())],
        dtype=np.float64,
    )
    bbox_max = np.array(
        [float(vertices["x"].max()), float(vertices["y"].max()), float(vertices["z"].max())],
        dtype=np.float64,
    )
    del vertices
    return vertex_count, bbox_min, bbox_max


def build_tiles(tile_paths: List[Path], expand_ratio: float, expand_abs: float) -> List[TileInfo]:
    tiles = []
    for tile_path in tile_paths:
        vertex_count, bbox_min, bbox_max = read_binary_ply_bbox(tile_path)
        pad = (bbox_max - bbox_min) * expand_ratio + expand_abs
        tiles.append(
            TileInfo(
                path=tile_path,
                vertex_count=vertex_count,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                expanded_min=bbox_min - pad,
                expanded_max=bbox_max + pad,
            )
        )
        print(
            f"[tile] {tile_path.name}: vertices={vertex_count}, "
            f"bbox_min={bbox_min.tolist()}, bbox_max={bbox_max.tolist()}"
        )
    return tiles


def build_tile_grid(tiles: List[TileInfo]) -> TileGrid:
    mins = np.vstack([tile.expanded_min for tile in tiles])
    maxs = np.vstack([tile.expanded_max for tile in tiles])
    sizes = maxs - mins
    global_min = mins.min(axis=0)
    global_max = maxs.max(axis=0)
    scene_extent = np.maximum(global_max - global_min, 1e-6)
    median_tile_extent = np.median(np.maximum(sizes, 1e-6), axis=0)
    cell_size = np.maximum(median_tile_extent, scene_extent / 128.0)

    cells: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, tile in enumerate(tiles):
        start = np.floor((tile.expanded_min - global_min) / cell_size).astype(np.int64)
        end = np.floor((tile.expanded_max - global_min) / cell_size).astype(np.int64)
        for ix in range(int(start[0]), int(end[0]) + 1):
            for iy in range(int(start[1]), int(end[1]) + 1):
                for iz in range(int(start[2]), int(end[2]) + 1):
                    cells.setdefault((ix, iy, iz), []).append(idx)
    return TileGrid(
        origin=global_min,
        cell_size=cell_size,
        global_min=global_min,
        global_max=global_max,
        cells=cells,
    )


def accumulate_sparse_tracks(points3d_path: Path, tiles: List[TileInfo], tile_grid: TileGrid, progress_every: int):
    with points3d_path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        print(f"[points3D] streaming {num_points} sparse points from {points3d_path}")
        for idx in range(num_points):
            point_meta = read_next_bytes(fid, 43, "QdddBBBd")
            xyz = (point_meta[1], point_meta[2], point_meta[3])
            candidate_tiles = tile_grid.lookup(xyz)

            track_length = read_next_bytes(fid, 8, "Q")[0]
            if not candidate_tiles:
                fid.seek(8 * track_length, 1)
            else:
                track_blob = fid.read(8 * track_length)
                track_pairs = struct.unpack("<" + "ii" * track_length, track_blob) if track_length else ()
                image_ids = track_pairs[0::2]
                for tile_idx in candidate_tiles:
                    tile = tiles[tile_idx]
                    if not tile.contains(xyz):
                        continue
                    tile.matched_sparse_points += 1
                    counts = tile.image_counts
                    for image_id in image_ids:
                        counts[image_id] = counts.get(image_id, 0) + 1

            if progress_every > 0 and (idx + 1) % progress_every == 0:
                print(f"[points3D] processed {idx + 1}/{num_points}")


def read_cameras_binary(path: Path) -> Dict[int, dict]:
    cameras = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            camera = {
                "camera_id": camera_id,
                "model_id": model_id,
                "model_name": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
            if model_name == "PINHOLE":
                camera.update({"fx": params[0], "fy": params[1], "cx": params[2], "cy": params[3]})
            elif model_name == "SIMPLE_PINHOLE":
                camera.update({"fx": params[0], "fy": params[0], "cx": params[1], "cy": params[2]})
            cameras[camera_id] = camera
    return cameras


def read_images_binary_light(path: Path) -> Dict[int, dict]:
    images = {}
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        print(f"[images] loading {num_images} camera poses from {path}")
        for _ in range(num_images):
            data = read_next_bytes(fid, 64, "idddddddi")
            image_id = data[0]
            qvec = list(data[1:5])
            tvec = list(data[5:8])
            camera_id = data[8]

            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                if not c:
                    raise EOFError(f"Unexpected EOF while reading image name from {path}")
                name_bytes.extend(c)
            image_name = name_bytes.decode("utf-8")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2d, 1)

            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "image_name": image_name,
                "image_basename": Path(image_name).name,
            }
    return images


def qvec2rotmat(qvec: List[float]) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[3] * q[1] + 2 * q[0] * q[2]],
            [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
            [2 * q[3] * q[1] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2],
        ],
        dtype=np.float64,
    )


def camera_center_from_qt(qvec: List[float], tvec: List[float]) -> List[float]:
    rotation = qvec2rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64)
    center = -rotation.T @ t
    return center.tolist()


def resolve_image_path(scene_root: Path, images_dir: str, image_name: str) -> Optional[str]:
    basename = Path(image_name).name
    candidates = []
    if images_dir:
        candidates.append(scene_root / images_dir / image_name)
        candidates.append(scene_root / images_dir / basename)
    candidates.append(scene_root / image_name)
    candidates.append(scene_root / basename)
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.relative_to(scene_root))
            except ValueError:
                return str(candidate)
    return None


def write_outputs(
    out_root: Path,
    scene_root: Path,
    images_dir: str,
    tiles: List[TileInfo],
    images: Dict[int, dict],
    cameras: Dict[int, dict],
    total_sparse_points: int,
    total_scene_images: int,
    min_observations: int,
    top_k: int,
):
    out_root.mkdir(parents=True, exist_ok=True)

    for tile in tiles:
        ranked = sorted(tile.image_counts.items(), key=lambda kv: (-kv[1], images.get(kv[0], {}).get("image_name", "")))
        ranked = [kv for kv in ranked if kv[1] >= min_observations]
        if top_k > 0:
            ranked = ranked[:top_k]

        tile_out = out_root / tile.stem
        tile_out.mkdir(parents=True, exist_ok=True)

        image_lines = []
        camera_records = []
        for image_id, support_count in ranked:
            image_info = images.get(image_id)
            if image_info is None:
                continue
            camera_info = cameras.get(image_info["camera_id"], {})
            record = {
                **image_info,
                **camera_info,
                "support_sparse_points": support_count,
                "camera_center": camera_center_from_qt(image_info["qvec"], image_info["tvec"]),
                "resolved_image_path": resolve_image_path(scene_root, images_dir, image_info["image_name"]),
            }
            camera_records.append(record)
            image_lines.append(image_info["image_name"])

        summary = {
            "tile_name": tile.path.name,
            "tile_ply": str(tile.path),
            "vertex_count": tile.vertex_count,
            "bbox_min": tile.bbox_min.tolist(),
            "bbox_max": tile.bbox_max.tolist(),
            "expanded_bbox_min": tile.expanded_min.tolist(),
            "expanded_bbox_max": tile.expanded_max.tolist(),
            "matched_sparse_points": tile.matched_sparse_points,
            "matched_images_before_filter": len(tile.image_counts),
            "matched_images_after_filter": len(camera_records),
            "min_observations": min_observations,
            "top_k": top_k,
            "scene_root": str(scene_root),
            "images_dir": images_dir,
            "total_scene_sparse_points": total_sparse_points,
            "total_scene_images": total_scene_images,
        }

        (tile_out / "images.txt").write_text("\n".join(image_lines) + ("\n" if image_lines else ""), encoding="utf-8")
        (tile_out / "cameras.json").write_text(json.dumps(camera_records, indent=2, ensure_ascii=False), encoding="utf-8")
        (tile_out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[done] {tile.path.name}: sparse_points={tile.matched_sparse_points}, "
            f"images={len(camera_records)}, out={tile_out}"
        )


def main():
    args = parse_args()

    scene_root = Path(args.scene_root).resolve()
    sparse_root = (scene_root / args.sparse_dir).resolve()
    out_root = (Path(args.out_dir).resolve() if args.out_dir else (scene_root / "tile_camera_map").resolve())
    images_dir = args.images_dir or auto_detect_images_dir(scene_root)

    tile_paths = resolve_tile_paths(scene_root, args.tile_path, args.tile_glob)
    cameras_path = sparse_root / "cameras.bin"
    images_path = sparse_root / "images.bin"
    points3d_path = sparse_root / "points3D.bin"

    for required in (cameras_path, images_path, points3d_path):
        if not required.exists():
            raise SystemExit(f"Missing required COLMAP file: {required}")

    print(f"[scene] scene_root={scene_root}")
    print(f"[scene] images_dir={images_dir}")
    print(f"[scene] sparse_root={sparse_root}")
    print(f"[scene] out_root={out_root}")

    tiles = build_tiles(tile_paths, args.bbox_expand_ratio, args.bbox_expand_abs)
    tile_grid = build_tile_grid(tiles)
    accumulate_sparse_tracks(points3d_path, tiles, tile_grid, args.progress_every)

    with points3d_path.open("rb") as fid:
        total_sparse_points = read_next_bytes(fid, 8, "Q")[0]

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary_light(images_path)

    write_outputs(
        out_root=out_root,
        scene_root=scene_root,
        images_dir=images_dir,
        tiles=tiles,
        images=images,
        cameras=cameras,
        total_sparse_points=total_sparse_points,
        total_scene_images=len(images),
        min_observations=args.min_observations,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
