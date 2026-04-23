import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


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


@dataclass
class CameraRecord:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: Tuple[float, ...]


@dataclass
class ImageRecord:
    image_id: int
    qvec: Tuple[float, float, float, float]
    tvec: Tuple[float, float, float]
    camera_id: int
    name: str
    xys: List[Tuple[float, float]]
    point3d_ids: List[int]


@dataclass
class PointRecord:
    point3d_id: int
    xyz: Tuple[float, float, float]
    rgb: Tuple[int, int, int]
    error: float
    track: List[Tuple[int, int]]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a COLMAP sparse subset model from an image whitelist. "
            "Output is cameras.bin/images.bin/points3D.bin."
        )
    )
    parser.add_argument("--src_sparse", type=str, required=True, help="Source sparse/0 directory")
    parser.add_argument("--keep_list", type=str, required=True, help="Text file: one image name (or stem) per line")
    parser.add_argument("--out_sparse", type=str, required=True, help="Output sparse/0 directory")
    parser.add_argument(
        "--min_track_len",
        type=int,
        default=1,
        help="Keep points3D only if filtered track length >= this value",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=200000,
        help="Progress print interval for streaming points3D.bin",
    )
    return parser.parse_args()


def read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Expected {num_bytes} bytes, got {len(data)}")
    return struct.unpack("<" + fmt, data)


def read_c_string(fid) -> str:
    buf = bytearray()
    while True:
        c = fid.read(1)
        if not c:
            raise EOFError("Unexpected EOF while reading c-string")
        if c == b"\x00":
            break
        buf.extend(c)
    return buf.decode("utf-8")


def load_keep_set(path: Path):
    keep_names = set()
    keep_stems = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if p.suffix:
            keep_names.add(p.name.lower())
            keep_stems.add(p.stem)
        else:
            keep_stems.add(p.name)
    return keep_names, keep_stems


def is_kept_name(image_name: str, keep_names: Set[str], keep_stems: Set[str]) -> bool:
    base = Path(image_name).name
    stem = Path(base).stem
    return base.lower() in keep_names or stem in keep_stems


def read_cameras_binary(path: Path) -> Dict[int, CameraRecord]:
    cameras: Dict[int, CameraRecord] = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown camera model_id={model_id}")
            _, num_params = CAMERA_MODELS[model_id]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = CameraRecord(
                camera_id=camera_id,
                model_id=model_id,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_kept_images_binary(
    images_path: Path, keep_names: Set[str], keep_stems: Set[str]
) -> Dict[int, ImageRecord]:
    kept_images: Dict[int, ImageRecord] = {}
    with images_path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        print(f"[images] scanning {num_images} images from {images_path}")
        for idx in range(num_images):
            meta = read_next_bytes(fid, 64, "idddddddi")
            image_id = meta[0]
            qvec = (meta[1], meta[2], meta[3], meta[4])
            tvec = (meta[5], meta[6], meta[7])
            camera_id = meta[8]
            image_name = read_c_string(fid)
            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            if is_kept_name(image_name, keep_names, keep_stems):
                raw = read_next_bytes(fid, 24 * num_points2d, "ddq" * num_points2d) if num_points2d else ()
                xys = []
                point3d_ids = []
                for i in range(0, len(raw), 3):
                    xys.append((raw[i], raw[i + 1]))
                    point3d_ids.append(raw[i + 2])
                kept_images[image_id] = ImageRecord(
                    image_id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3d_ids=point3d_ids,
                )
            else:
                fid.seek(24 * num_points2d, 1)
            if (idx + 1) % 5000 == 0:
                print(f"[images] processed {idx + 1}/{num_images}")
    print(f"[images] kept {len(kept_images)} images")
    return kept_images


def filter_points3d_binary(
    points3d_path: Path,
    keep_image_ids: Set[int],
    min_track_len: int,
    progress_every: int,
) -> Tuple[List[PointRecord], Set[int], int]:
    kept_points: List[PointRecord] = []
    kept_point_ids: Set[int] = set()
    with points3d_path.open("rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        print(f"[points3D] streaming {num_points} points from {points3d_path}")
        for idx in range(num_points):
            pmeta = read_next_bytes(fid, 43, "QdddBBBd")
            point3d_id = pmeta[0]
            xyz = (pmeta[1], pmeta[2], pmeta[3])
            rgb = (pmeta[4], pmeta[5], pmeta[6])
            error = pmeta[7]
            track_len = read_next_bytes(fid, 8, "Q")[0]
            if track_len == 0:
                continue
            raw_track = read_next_bytes(fid, 8 * track_len, "ii" * track_len)
            filtered_track = []
            for j in range(0, len(raw_track), 2):
                image_id = raw_track[j]
                point2d_idx = raw_track[j + 1]
                if image_id in keep_image_ids:
                    filtered_track.append((image_id, point2d_idx))
            if len(filtered_track) >= min_track_len:
                kept_points.append(
                    PointRecord(
                        point3d_id=point3d_id,
                        xyz=xyz,
                        rgb=rgb,
                        error=error,
                        track=filtered_track,
                    )
                )
                kept_point_ids.add(point3d_id)
            if progress_every > 0 and (idx + 1) % progress_every == 0:
                print(f"[points3D] processed {idx + 1}/{num_points}")
    print(f"[points3D] kept {len(kept_points)} points")
    return kept_points, kept_point_ids, num_points


def sanitize_image_point_ids(images: Dict[int, ImageRecord], valid_point_ids: Set[int]):
    for img in images.values():
        for i, pid in enumerate(img.point3d_ids):
            if pid != -1 and pid not in valid_point_ids:
                img.point3d_ids[i] = -1


def write_cameras_binary(path: Path, cameras: Dict[int, CameraRecord]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fid:
        ordered = sorted(cameras.values(), key=lambda c: c.camera_id)
        fid.write(struct.pack("<Q", len(ordered)))
        for cam in ordered:
            fid.write(struct.pack("<iiQQ", cam.camera_id, cam.model_id, cam.width, cam.height))
            fid.write(struct.pack("<" + "d" * len(cam.params), *cam.params))


def write_images_binary(path: Path, images: Dict[int, ImageRecord]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fid:
        ordered = sorted(images.values(), key=lambda x: x.image_id)
        fid.write(struct.pack("<Q", len(ordered)))
        for img in ordered:
            fid.write(
                struct.pack(
                    "<idddddddi",
                    img.image_id,
                    img.qvec[0],
                    img.qvec[1],
                    img.qvec[2],
                    img.qvec[3],
                    img.tvec[0],
                    img.tvec[1],
                    img.tvec[2],
                    img.camera_id,
                )
            )
            fid.write(img.name.encode("utf-8"))
            fid.write(b"\x00")
            num_points2d = len(img.point3d_ids)
            if len(img.xys) != num_points2d:
                raise ValueError(
                    f"Image {img.image_id} has mismatch: len(xys)={len(img.xys)} "
                    f"vs len(point3d_ids)={num_points2d}"
                )
            fid.write(struct.pack("<Q", num_points2d))
            for i in range(num_points2d):
                x, y = img.xys[i]
                pid = img.point3d_ids[i]
                fid.write(struct.pack("<ddq", x, y, pid))


def write_points3d_binary(path: Path, points: List[PointRecord]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fid:
        ordered = sorted(points, key=lambda p: p.point3d_id)
        fid.write(struct.pack("<Q", len(ordered)))
        for pt in ordered:
            fid.write(
                struct.pack(
                    "<QdddBBBd",
                    pt.point3d_id,
                    pt.xyz[0],
                    pt.xyz[1],
                    pt.xyz[2],
                    pt.rgb[0],
                    pt.rgb[1],
                    pt.rgb[2],
                    pt.error,
                )
            )
            fid.write(struct.pack("<Q", len(pt.track)))
            for image_id, point2d_idx in pt.track:
                fid.write(struct.pack("<ii", image_id, point2d_idx))


def main():
    args = parse_args()
    src_sparse = Path(args.src_sparse).resolve()
    keep_list = Path(args.keep_list).resolve()
    out_sparse = Path(args.out_sparse).resolve()

    cameras_path = src_sparse / "cameras.bin"
    images_path = src_sparse / "images.bin"
    points3d_path = src_sparse / "points3D.bin"
    for p in (cameras_path, images_path, points3d_path, keep_list):
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    keep_names, keep_stems = load_keep_set(keep_list)
    if not keep_names and not keep_stems:
        raise SystemExit("keep_list is empty after filtering blank/comment lines")

    src_cameras = read_cameras_binary(cameras_path)
    kept_images = read_kept_images_binary(images_path, keep_names, keep_stems)
    if not kept_images:
        raise SystemExit("No images matched keep_list.")

    keep_image_ids = set(kept_images.keys())
    kept_points, kept_point_ids, total_points = filter_points3d_binary(
        points3d_path=points3d_path,
        keep_image_ids=keep_image_ids,
        min_track_len=args.min_track_len,
        progress_every=args.progress_every,
    )
    sanitize_image_point_ids(kept_images, kept_point_ids)

    used_camera_ids = sorted({img.camera_id for img in kept_images.values()})
    kept_cameras = {cid: src_cameras[cid] for cid in used_camera_ids}

    write_cameras_binary(out_sparse / "cameras.bin", kept_cameras)
    write_images_binary(out_sparse / "images.bin", kept_images)
    write_points3d_binary(out_sparse / "points3D.bin", kept_points)

    print("=== Sparse Subset Summary ===")
    print(f"source sparse:   {src_sparse}")
    print(f"keep list:       {keep_list}")
    print(f"output sparse:   {out_sparse}")
    print(f"kept cameras:    {len(kept_cameras)}")
    print(f"kept images:     {len(kept_images)}")
    print(f"source points3D: {total_points}")
    print(f"kept points3D:   {len(kept_points)}")


if __name__ == "__main__":
    main()
