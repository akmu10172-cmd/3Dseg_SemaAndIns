import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np


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


def parse_id2label(text: str):
    if not text:
        return DEFAULT_ID2LABEL.copy()
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid id2label item: {item}")
        k, v = item.split(":", 1)
        out[int(k.strip())] = v.strip()
    return out


def load_mask(path: Path):
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape {arr.shape} in {path}")
    return arr.astype(np.int32)


def find_image_path(images_dir: Path, stem: str):
    if images_dir is None:
        return None
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def contour_to_points(contour):
    pts = contour.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in pts]


def encode_binary_mask_png(binary_mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (binary_mask.astype(np.uint8) * 255))
    if not ok:
        raise RuntimeError("Failed to encode mask png")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def encode_image_base64(image_path: Path):
    if image_path is None or not image_path.exists():
        return None
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert *_s.npy masks to Labelme json.")
    parser.add_argument("--src_dir", required=True, help="Directory containing *_s.npy")
    parser.add_argument("--dst_dir", required=True, help="Output directory for json")
    parser.add_argument("--suffix", default="_s.npy", help="Input mask suffix")
    parser.add_argument("--images_dir", default="", help="Optional image dir for imagePath/image size")
    parser.add_argument("--skip_ids", default="0", help="Comma-separated class ids to skip (default: background 0)")
    parser.add_argument(
        "--id2label",
        default="",
        help="Mapping string, e.g. '1:vehicle,2:person,5:road'. Defaults to built-in mapping.",
    )
    parser.add_argument("--min_area", type=float, default=20.0, help="Skip tiny polygons")
    parser.add_argument("--approx_eps", type=float, default=1.0, help="Polygon approximation epsilon in pixels")
    parser.add_argument(
        "--shape_type",
        default="polygon",
        choices=["polygon", "mask"],
        help="Export Labelme shapes as polygon or mask",
    )
    parser.add_argument(
        "--embed_image_data",
        action="store_true",
        help="Embed image bytes into json.imageData (larger files but better compatibility)",
    )
    parser.add_argument("--json_version", default="5.11.3", help="Value for json.version")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir) if args.images_dir else None

    skip_ids = set()
    if args.skip_ids.strip():
        skip_ids = {int(x.strip()) for x in args.skip_ids.split(",") if x.strip()}

    id2label = parse_id2label(args.id2label)

    files = sorted(src_dir.glob(f"*{args.suffix}"))
    if not files:
        raise SystemExit(f"No files found in {src_dir} matching *{args.suffix}")

    saved = 0
    for npy_path in files:
        mask = load_mask(npy_path)
        h, w = mask.shape
        stem = npy_path.name[: -len(args.suffix)]

        image_path = find_image_path(images_dir, stem) if images_dir else None
        image_name = image_path.name if image_path else ""
        image_data_b64 = encode_image_base64(image_path) if args.embed_image_data else None

        shapes = []
        class_ids = sorted(np.unique(mask).tolist())
        for cid in class_ids:
            if cid in skip_ids:
                continue
            binary = (mask == cid).astype(np.uint8)
            if binary.sum() == 0:
                continue
            label = id2label.get(int(cid), f"class_{cid}")

            if args.shape_type == "polygon":
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < args.min_area:
                        continue
                    if args.approx_eps > 0:
                        cnt = cv2.approxPolyDP(cnt, epsilon=float(args.approx_eps), closed=True)
                    if cnt.shape[0] < 3:
                        continue
                    shapes.append(
                        {
                            "label": label,
                            "points": contour_to_points(cnt),
                            "group_id": None,
                            "description": "",
                            "shape_type": "polygon",
                            "flags": {},
                            "mask": None,
                        }
                    )
            else:
                num_labels, comp, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                for comp_id in range(1, num_labels):
                    area = int(stats[comp_id, cv2.CC_STAT_AREA])
                    if area < int(args.min_area):
                        continue
                    x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                    y = int(stats[comp_id, cv2.CC_STAT_TOP])
                    ww = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                    hh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                    local = (comp[y : y + hh, x : x + ww] == comp_id)
                    if not local.any():
                        continue
                    shapes.append(
                        {
                            "label": label,
                            # AnyLabeling-style mask uses top-left and bottom-right points.
                            "points": [[float(x), float(y)], [float(x + ww - 1), float(y + hh - 1)]],
                            "group_id": None,
                            "description": "",
                            "shape_type": "mask",
                            "flags": {},
                            "mask": encode_binary_mask_png(local),
                        }
                    )

        out = {
            "version": args.json_version,
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": image_data_b64,
            "imageHeight": int(h),
            "imageWidth": int(w),
            "description": "",
        }
        out_path = dst_dir / f"{stem}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        saved += 1

    print(f"done: {saved}/{len(files)} json files -> {dst_dir}")


if __name__ == "__main__":
    main()
