#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


PALETTE = np.array(
    [[0, 0, 0]] + [[(i * 53) % 256, (i * 97) % 256, (i * 193) % 256] for i in range(1, 10000)],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAM3 instance segmentation on a folder of images.")
    p.add_argument("--input-dir", type=Path, required=True, help="Image folder")
    p.add_argument("--output-dir", type=Path, required=True, help="Output folder")
    p.add_argument("--checkpoint-path", type=str, default=None, help="sam3.pt local path")
    p.add_argument("--no-hf-download", action="store_true", help="Disable HF auto-download")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--resolution", type=int, default=1008)
    p.add_argument("--confidence-threshold", type=float, default=0.35)
    p.add_argument("--max-images", type=int, default=0, help="0 means all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--prompts", type=str, default="vehicle", help="Comma-separated prompts, e.g. vehicle,car,bus")
    p.add_argument("--min-mask-area", type=int, default=40, help="Drop tiny masks")
    p.add_argument("--save-semantic", action="store_true", help="Also save semantic id map by prompt order")
    return p.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in exts])


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    d = {
        "instance_npy": out_dir / "instance_index_npy",
        "instance_png": out_dir / "instance_color_png",
        "instance_gray_png": out_dir / "instance_index_png",
        "semantic_npy": out_dir / "semantic_index_npy",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def instance_map_from_masks(
    masks: List[np.ndarray],
    scores: List[float],
    prompt_ids: List[int],
    min_mask_area: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if not masks:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32), 0

    h, w = masks[0].shape
    inst = np.zeros((h, w), dtype=np.int32)
    sem = np.zeros((h, w), dtype=np.int32)

    order = np.argsort(np.array(scores, dtype=np.float32))[::-1]
    next_id = 1
    for oi in order.tolist():
        m = masks[oi]
        if m.shape != (h, w):
            continue
        if int(m.sum()) < int(min_mask_area):
            continue
        assign = m & (inst == 0)
        if int(assign.sum()) < int(min_mask_area):
            continue
        inst[assign] = next_id
        sem[assign] = int(prompt_ids[oi])
        next_id += 1

    return inst, sem, int(next_id - 1)


def colorize_instance(inst: np.ndarray) -> np.ndarray:
    if inst.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    max_id = int(inst.max())
    if max_id < len(PALETTE):
        lut = PALETTE
    else:
        rng = np.random.default_rng(42)
        extra = rng.integers(0, 255, size=(max_id - len(PALETTE) + 1, 3), dtype=np.uint8)
        lut = np.concatenate([PALETTE, extra], axis=0)
    return lut[inst]


def main() -> None:
    args = parse_args()

    in_dir = args.input_dir
    if not in_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {in_dir}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    d = ensure_dirs(out_dir)

    prompts = [x.strip() for x in str(args.prompts or "").split(",") if x.strip()]
    if not prompts:
        raise ValueError("prompts cannot be empty")

    imgs = list_images(in_dir)
    if not imgs:
        raise RuntimeError(f"no images in: {in_dir}")
    if args.max_images > 0:
        imgs = imgs[: int(args.max_images)]

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=not args.no_hf_download,
        compile=args.compile,
    )
    processor = Sam3Processor(
        model=model,
        resolution=int(args.resolution),
        device=args.device,
        confidence_threshold=float(args.confidence_threshold),
    )
    print(f"Model ready on {args.device}")

    cfg = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "prompts": prompts,
        "resolution": int(args.resolution),
        "confidence_threshold": float(args.confidence_threshold),
        "device": args.device,
        "checkpoint_path": args.checkpoint_path,
        "load_from_hf": bool(not args.no_hf_download),
        "images": int(len(imgs)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = []
    total = len(imgs)
    for i, p in enumerate(imgs, start=1):
        stem = p.stem
        out_inst_npy = d["instance_npy"] / f"{stem}_inst.npy"
        if out_inst_npy.exists() and not args.overwrite:
            print(f"[{i}/{total}] skip existing: {p.name}")
            continue

        print(f"[{i}/{total}] processing: {p.name}")
        image = Image.open(p).convert("RGB")
        state = processor.set_image(image)

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []
        all_prompt_ids: List[int] = []
        prompt_counts: Dict[str, int] = {}

        for prompt_idx, prompt in enumerate(prompts, start=1):
            state = processor.set_text_prompt(prompt=prompt, state=state)
            masks_t = state["masks"]
            scores_t = state["scores"]
            n_inst = int(scores_t.numel())
            prompt_counts[prompt] = n_inst
            if n_inst <= 0:
                continue

            masks_np = masks_t.squeeze(1).detach().cpu().numpy().astype(bool)
            scores_np = scores_t.detach().cpu().numpy().astype(np.float32)
            for mi in range(n_inst):
                all_masks.append(masks_np[mi])
                all_scores.append(float(scores_np[mi]))
                all_prompt_ids.append(prompt_idx)

        if all_masks:
            inst, sem, num_inst = instance_map_from_masks(
                masks=all_masks,
                scores=all_scores,
                prompt_ids=all_prompt_ids,
                min_mask_area=int(args.min_mask_area),
            )
        else:
            w, h = image.size
            inst = np.zeros((h, w), dtype=np.int32)
            sem = np.zeros((h, w), dtype=np.int32)
            num_inst = 0

        np.save(out_inst_npy, inst.astype(np.int32))
        Image.fromarray(np.clip(inst, 0, 255).astype(np.uint8), mode="L").save(d["instance_gray_png"] / f"{stem}_inst.png")
        Image.fromarray(colorize_instance(inst), mode="RGB").save(d["instance_png"] / f"{stem}_inst.png")
        if args.save_semantic:
            np.save(d["semantic_npy"] / f"{stem}_s.npy", sem.astype(np.int32))

        summary.append(
            {
                "image": p.name,
                "num_instances": int(num_inst),
                "prompt_raw_counts": prompt_counts,
                "assigned_pixels": int((inst > 0).sum()),
            }
        )

    (out_dir / "summary.json").write_text(
        json.dumps({"items": summary, "prompts": prompts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] processed {len(summary)} images -> {out_dir}")


if __name__ == "__main__":
    main()
