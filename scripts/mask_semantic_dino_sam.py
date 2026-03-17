import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.ops import box_convert, nms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-semantic masks (_s.npy) with Grounding DINO + SAM."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Scene root path")
    parser.add_argument("--images_subdir", type=str, default="images", help="Image folder under data_path")
    parser.add_argument(
        "--dino_model_id",
        type=str,
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace Grounding DINO model id",
    )
    parser.add_argument(
        "--dino_checkpoint",
        type=str,
        default="",
        help="Local GroundingDINO checkpoint (.pth). If set, use local backend.",
    )
    parser.add_argument(
        "--dino_config",
        type=str,
        default="",
        help="Local GroundingDINO config (.py). Optional if auto-discovery finds SwinT OGC config.",
    )
    parser.add_argument(
        "--dino_backend",
        type=str,
        default="auto",
        choices=["auto", "hf", "local"],
        help="auto: local when --dino_checkpoint exists, otherwise hf",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="car . truck . bus . motorcycle . train . person . bicycle . "
                "traffic light . stop sign . parking meter . potted plant . "
                "road . street . lane . pavement . asphalt . sidewalk",
        help="Grounding prompt (dot-separated phrases recommended)",
    )
    parser.add_argument("--box_threshold", type=float, default=0.2, help="DINO box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="DINO text threshold")
    parser.add_argument("--nms_iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--min_box_area", type=float, default=32.0 * 32.0, help="Drop tiny boxes before SAM")
    parser.add_argument("--max_boxes", type=int, default=128, help="Max boxes per image after NMS")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM model type")
    parser.add_argument("--start_idx", type=int, default=0, help="Start image index")
    parser.add_argument("--end_idx", type=int, default=10**9, help="End image index (exclusive)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save_vis", action="store_true", help="Save overlay visualizations")
    return parser.parse_args()


def phrase_to_sem_id(phrase: str) -> int:
    p = phrase.lower().strip()

    if any(k in p for k in ["car", "truck", "bus", "motorcycle", "train", "vehicle"]):
        return 1
    if "person" in p or "pedestrian" in p:
        return 2
    if "bicycle" in p or "bike" in p:
        return 3
    if any(k in p for k in ["plant", "tree", "shrub", "bush", "vegetation"]):
        return 4
    if any(k in p for k in ["road", "street", "lane", "pavement", "asphalt", "sidewalk"]):
        return 5
    if any(k in p for k in ["traffic light", "stop sign", "parking meter", "traffic sign"]):
        return 6
    return 7


SEM_COLORS_BGR = {
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (0, 128, 255),
    4: (34, 139, 34),
    5: (180, 105, 255),
    6: (255, 0, 255),
    7: (200, 128, 128),
}


def draw_overlay(image_bgr: np.ndarray, sem_map: np.ndarray) -> np.ndarray:
    vis = image_bgr.copy()
    for sid, color in SEM_COLORS_BGR.items():
        m = sem_map == sid
        if np.any(m):
            vis[m] = (vis[m] * 0.45 + np.array(color) * 0.55).astype(np.uint8)
    return vis


def find_swint_ogc_config(user_config: str) -> Path:
    if user_config:
        p = Path(user_config)
        if p.exists():
            return p
        raise SystemExit(f"DINO config not found: {user_config}")

    candidates = []
    try:
        import groundingdino  # type: ignore

        pkg_dir = Path(groundingdino.__file__).resolve().parent
        candidates.extend(
            [
                pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py",
                pkg_dir / "config" / "GroundingDINO_SwinT_OGC.cfg.py",
                pkg_dir.parent / "config" / "GroundingDINO_SwinT_OGC.py",
                pkg_dir.parent / "config" / "GroundingDINO_SwinT_OGC.cfg.py",
            ]
        )
    except Exception:
        pass

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "scripts" / "groundingdino_configs" / "GroundingDINO_SwinT_OGC.py",
            cwd / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
            cwd / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.cfg.py",
        ]
    )

    for c in candidates:
        if c.exists():
            return c

    raise SystemExit(
        "Cannot find GroundingDINO SwinT OGC config. "
        "Please provide --dino_config /path/to/GroundingDINO_SwinT_OGC.py"
    )


def build_local_dino_predictor(checkpoint_path: str, config_path: str, device: str):
    try:
        from groundingdino.datasets import transforms as T  # type: ignore
        from groundingdino.util.inference import load_model, predict  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Local GroundingDINO backend requires package 'groundingdino'."
        ) from exc

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = load_model(config_path, checkpoint_path, device=device)

    def infer(image_rgb: np.ndarray, text_prompt: str, box_th: float, text_th: float):
        pil_img = Image.fromarray(image_rgb)
        image_t, _ = transform(pil_img, None)
        boxes, logits, phrases = predict(
            model=model,
            image=image_t,
            caption=text_prompt,
            box_threshold=box_th,
            text_threshold=text_th,
            device=device,
        )
        if boxes is None or len(boxes) == 0:
            return torch.empty((0, 4)), [], torch.empty((0,))
        if not torch.is_tensor(boxes):
            boxes = torch.tensor(boxes)
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits)
        labels = [str(p) for p in phrases]
        return boxes, labels, logits

    return infer


def build_hf_dino_predictor(model_id: str, device: str):
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except Exception as exc:
        raise SystemExit(
            "HF backend requires transformers. Install with: pip install transformers"
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    def infer(image_rgb: np.ndarray, text_prompt: str, box_th: float, text_th: float):
        h, w = image_rgb.shape[:2]
        image_pil = Image.fromarray(image_rgb)
        with torch.no_grad():
            inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
            outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_th,
                text_threshold=text_th,
                target_sizes=[(h, w)],
            )[0]
        boxes = results.get("boxes", torch.empty((0, 4), device=device))
        labels = results.get("labels", [])
        scores = results.get("scores", torch.empty((0,), device=device))
        return boxes, labels, scores

    return infer


def run():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to cpu.")
        device = "cpu"

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as exc:
        raise SystemExit(
            "segment_anything is required."
        ) from exc

    if not os.path.exists(args.sam_checkpoint):
        raise SystemExit(f"SAM checkpoint not found: {args.sam_checkpoint}")

    data_path = Path(args.data_path)
    images_dir = data_path / args.images_subdir
    out_npy_dir = data_path / "language_features"
    out_vis_dir = data_path / "vis_semantic_dino"
    out_npy_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    image_files = image_files[args.start_idx: min(args.end_idx, len(image_files))]
    print(f"Images to process: {len(image_files)}")
    use_local = args.dino_backend == "local" or (
        args.dino_backend == "auto" and args.dino_checkpoint and Path(args.dino_checkpoint).exists()
    )
    if use_local:
        ckpt = Path(args.dino_checkpoint)
        if not ckpt.exists():
            raise SystemExit(f"DINO checkpoint not found: {ckpt}")
        cfg = find_swint_ogc_config(args.dino_config)
        print(f"DINO backend: local")
        print(f"DINO checkpoint: {ckpt}")
        print(f"DINO config: {cfg}")
        dino_infer = build_local_dino_predictor(str(ckpt), str(cfg), device)
    else:
        print(f"DINO backend: hf")
        print(f"DINO model: {args.dino_model_id}")
        dino_infer = build_hf_dino_predictor(args.dino_model_id, device)

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for img_path in tqdm(image_files, desc="Masking"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, labels, scores = dino_infer(image_rgb, args.text_prompt, args.box_threshold, args.text_threshold)
        boxes = boxes.to(device)
        scores = scores.to(device)
        if use_local and boxes.numel() > 0:
            # Local DINO outputs normalized cxcywh; convert to absolute xyxy.
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            boxes = box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")

        if boxes.numel() == 0:
            sem_map = np.zeros((1, h, w), dtype=np.int32)
            np.save(out_npy_dir / f"{img_path.stem}_s.npy", sem_map)
            if args.save_vis:
                cv2.imwrite(str(out_vis_dir / f"{img_path.stem}.jpg"), image_bgr)
            continue

        keep = []
        sem_ids = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            area = float((x2 - x1) * (y2 - y1))
            if area < args.min_box_area:
                continue
            phrase = labels[i] if isinstance(labels[i], str) else str(labels[i])
            sem_ids.append(phrase_to_sem_id(phrase))
            keep.append(i)

        if not keep:
            sem_map = np.zeros((1, h, w), dtype=np.int32)
            np.save(out_npy_dir / f"{img_path.stem}_s.npy", sem_map)
            if args.save_vis:
                cv2.imwrite(str(out_vis_dir / f"{img_path.stem}.jpg"), image_bgr)
            continue

        boxes = boxes[keep]
        scores = scores[keep]
        sem_ids = torch.tensor(sem_ids, device=device, dtype=torch.int64)

        final_keep = []
        for sid in sem_ids.unique():
            idx = torch.where(sem_ids == sid)[0]
            sid_keep = nms(boxes[idx], scores[idx], args.nms_iou)
            final_keep.extend(idx[sid_keep].tolist())

        if not final_keep:
            sem_map = np.zeros((1, h, w), dtype=np.int32)
            np.save(out_npy_dir / f"{img_path.stem}_s.npy", sem_map)
            if args.save_vis:
                cv2.imwrite(str(out_vis_dir / f"{img_path.stem}.jpg"), image_bgr)
            continue

        if len(final_keep) > args.max_boxes:
            final_keep = sorted(final_keep, key=lambda i: float(scores[i]), reverse=True)[: args.max_boxes]

        boxes = boxes[final_keep]
        sem_ids = sem_ids[final_keep]

        predictor.set_image(image_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image_rgb.shape[:2])
        masks_t, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        sem_map_2d = np.zeros((h, w), dtype=np.int32)
        order = torch.argsort(scores[final_keep].to(device), descending=False)
        for idx in order.tolist():
            sem_id = int(sem_ids[idx].item())
            mask = masks_t[idx, 0].detach().cpu().numpy().astype(bool)
            sem_map_2d[mask] = sem_id

        sem_map = sem_map_2d[np.newaxis, ...]
        np.save(out_npy_dir / f"{img_path.stem}_s.npy", sem_map)

        if args.save_vis:
            vis = draw_overlay(image_bgr, sem_map_2d)
            cv2.imwrite(str(out_vis_dir / f"{img_path.stem}.jpg"), vis)

    print(f"Done. _s.npy saved to: {out_npy_dir}")
    if args.save_vis:
        print(f"Visualizations saved to: {out_vis_dir}")


if __name__ == "__main__":
    run()
