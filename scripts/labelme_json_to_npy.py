import argparse
import base64
import binascii
import json
from pathlib import Path

import cv2
import numpy as np


def parse_label_map(text: str):
    """
    Parse 'label:id,label2:id2' into dict.
    Example: 'car:1,truck:1,person:2,road:5'
    """
    mapping = {}
    if not text:
        return mapping
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid label_map item: {item}")
        k, v = item.split(":", 1)
        mapping[k.strip()] = int(v.strip())
    return mapping


def decode_mask_bitmap(mask_field):
    """
    Decode a Labelme brush/mask bitmap from base64.
    Supports:
    - shape["mask"] = "<base64_png>"
    - shape["mask"] = {"data": "<base64_png>", "origin": [x, y]}
    - shape["bitmap"] in the same formats above
    """
    origin = None
    payload = mask_field

    if isinstance(mask_field, dict):
        payload = mask_field.get("data")
        origin = mask_field.get("origin")

    if not isinstance(payload, str) or not payload:
        return None, origin

    try:
        raw = base64.b64decode(payload)
    except (binascii.Error, ValueError):
        return None, origin

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, origin

    if img.ndim == 3:
        img = img[:, :, 0]
    bitmap = img > 0
    return bitmap, origin


def draw_bitmap_shape(mask, shape, class_id):
    bitmap, origin = decode_mask_bitmap(shape.get("mask"))
    if bitmap is None:
        bitmap, origin = decode_mask_bitmap(shape.get("bitmap"))
    if bitmap is None:
        return

    h, w = mask.shape[:2]

    if bitmap.shape == (h, w):
        mask[bitmap] = int(class_id)
        return

    if origin is None:
        pts = shape.get("points", [])
        if pts:
            origin = pts[0]
        else:
            origin = [0, 0]

    x0 = int(round(float(origin[0])))
    y0 = int(round(float(origin[1])))
    bh, bw = bitmap.shape[:2]

    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(w, x0 + bw)
    y2 = min(h, y0 + bh)
    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x0
    sy1 = y1 - y0
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    sub_bitmap = bitmap[sy1:sy2, sx1:sx2]
    roi = mask[y1:y2, x1:x2]
    roi[sub_bitmap] = int(class_id)
    mask[y1:y2, x1:x2] = roi


def draw_shape(mask, shape, class_id):
    shape_type = shape.get("shape_type", "polygon")
    if shape_type in ("mask", "brush"):
        draw_bitmap_shape(mask, shape, class_id)
        return

    points = np.array(shape["points"], dtype=np.float32)

    if shape_type == "polygon":
        pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], int(class_id))
        return

    if shape_type == "rectangle":
        if points.shape[0] < 2:
            return
        p1, p2 = points[0], points[1]
        x1, y1 = np.round(np.minimum(p1, p2)).astype(np.int32)
        x2, y2 = np.round(np.maximum(p1, p2)).astype(np.int32)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), int(class_id), thickness=-1)
        return

    if shape_type == "circle":
        if points.shape[0] < 2:
            return
        c = points[0]
        p = points[1]
        r = int(np.linalg.norm(p - c))
        cx, cy = np.round(c).astype(np.int32)
        cv2.circle(mask, (int(cx), int(cy)), int(r), int(class_id), thickness=-1)
        return

    if shape_type in ("linestrip", "line"):
        pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], isClosed=False, color=int(class_id), thickness=3)
        return

    if shape_type == "point":
        p = np.round(points[0]).astype(np.int32)
        cv2.circle(mask, (int(p[0]), int(p[1])), 2, int(class_id), thickness=-1)
        return


def main():
    parser = argparse.ArgumentParser(description="Convert Labelme json masks to *_s.npy")
    parser.add_argument("--src_dir", required=True, help="Directory containing Labelme .json files")
    parser.add_argument("--dst_dir", required=True, help="Output directory for *_s.npy")
    parser.add_argument(
        "--label_map",
        default="car:1,truck:1,bus:1,motorcycle:1,train:1,person:2,bicycle:3,plant:4,tree:4,vegetation:4,road:5,street:5,lane:5,pavement:5,asphalt:5,sidewalk:5,traffic light:6,stop sign:6,parking meter:6",
        help="Mapping string: 'label:id,label2:id2'",
    )
    parser.add_argument("--default_id", type=int, default=7, help="Class id for unknown labels")
    parser.add_argument("--background_id", type=int, default=0, help="Background class id")
    parser.add_argument("--strict", action="store_true", help="Fail if unknown label appears")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    label_map = parse_label_map(args.label_map)

    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No json found in {src_dir}")

    saved = 0
    unknown_labels = {}

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        h = int(data["imageHeight"])
        w = int(data["imageWidth"])
        mask = np.full((h, w), int(args.background_id), dtype=np.int32)

        for shape in data.get("shapes", []):
            label = str(shape.get("label", "")).strip()
            if not label:
                continue
            if label in label_map:
                cid = label_map[label]
            else:
                unknown_labels[label] = unknown_labels.get(label, 0) + 1
                if args.strict:
                    raise SystemExit(f"Unknown label '{label}' in {jf.name}")
                cid = int(args.default_id)
            draw_shape(mask, shape, cid)

        out_name = f"{jf.stem}_s.npy"
        np.save(dst_dir / out_name, mask[np.newaxis, ...].astype(np.int32))
        saved += 1

    print(f"done: {saved}/{len(json_files)} -> {dst_dir}")
    if unknown_labels:
        print("unknown labels mapped to default_id:")
        for k, v in sorted(unknown_labels.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
