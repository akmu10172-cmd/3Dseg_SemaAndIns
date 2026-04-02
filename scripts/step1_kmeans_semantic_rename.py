#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def list_class_plys(kmeans_dir: Path) -> List[Path]:
    files = sorted(kmeans_dir.glob("class*.ply"))
    if not files:
        files = sorted(kmeans_dir.glob("class_*.ply"))
    return [p for p in files if p.is_file()]


def choose_prompt_from_summary(summary_json: Path, fallback: str) -> str:
    if not summary_json.exists():
        return fallback
    try:
        obj = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    score: Dict[str, int] = {}
    for it in obj.get("items", []):
        d = it.get("prompt_raw_counts", {})
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            kk = str(k).strip()
            if not kk:
                continue
            score[kk] = int(score.get(kk, 0) + int(v))
    if not score:
        return fallback
    return sorted(score.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def sanitize_name(s: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in s.strip())
    return out or "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Step1: assign semantic to each kmeans class ply and rename outputs.")
    p.add_argument("--scene_path", required=True)
    p.add_argument("--kmeans_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--python_exec", default="python")
    p.add_argument("--sam3_python", default="/mnt/d/sam3/.conda/envs/sam3/bin/python")
    p.add_argument("--sam3_checkpoint", default="/mnt/c/Users/ysy/.cache/modelscope/hub/models/facebook/sam3/sam3.pt")
    p.add_argument("--sam3_device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--semantic_prompts", default="building,vehicle,person,bicycle,vegetation,road,traffic_facility,other")
    p.add_argument("--semantic_probe_views", type=int, default=6)
    p.add_argument("--num_views", type=int, default=6)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--fov_deg", type=float, default=58.0)
    p.add_argument("--xy_jitter_ratio", type=float, default=0.22)
    args = p.parse_args()

    scene_path = Path(args.scene_path)
    kmeans_dir = Path(args.kmeans_dir)
    output_dir = Path(args.output_dir)
    if not scene_path.exists():
        raise FileNotFoundError(f"scene_path not found: {scene_path}")
    if not kmeans_dir.exists():
        raise FileNotFoundError(f"kmeans_dir not found: {kmeans_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = list_class_plys(kmeans_dir)
    if not classes:
        raise RuntimeError(f"no class*.ply found in {kmeans_dir}")

    topdown_script = SCRIPT_DIR / "topdown_sam3_instance_pipeline.py"
    sam3_img_script = SCRIPT_DIR / "sam3_instance_from_images.py"
    if not topdown_script.exists():
        raise FileNotFoundError(f"missing script: {topdown_script}")
    if not sam3_img_script.exists():
        raise FileNotFoundError(f"missing script: {sam3_img_script}")

    mapping_rows = []
    for i, cls_ply in enumerate(classes, start=1):
        stem = cls_ply.stem
        subdir = f"_step1_semprobe/{stem}"
        print(f"[{i}/{len(classes)}] probing {cls_ply.name}")

        run_cmd(
            [
                str(args.python_exec),
                str(topdown_script),
                "--scene_path",
                str(scene_path),
                "--input_ply",
                str(cls_ply),
                "--output_subdir",
                subdir,
                "--num_views",
                str(int(args.num_views)),
                "--image_size",
                str(int(args.image_size)),
                "--fov_deg",
                str(float(args.fov_deg)),
                "--xy_jitter_ratio",
                str(float(args.xy_jitter_ratio)),
                "--sam3_python",
                str(args.sam3_python),
                "--sam3_checkpoint",
                str(args.sam3_checkpoint),
                "--sam3_device",
                str(args.sam3_device),
                "--render_only",
            ]
        )

        render_dir = scene_path / subdir / "renders_topdown"
        probe_dir = scene_path / subdir / "semantic_probe"
        run_cmd(
            [
                str(args.sam3_python),
                str(sam3_img_script),
                "--input-dir",
                str(render_dir),
                "--output-dir",
                str(probe_dir),
                "--device",
                str(args.sam3_device),
                "--resolution",
                "1008",
                "--confidence-threshold",
                "0.35",
                "--prompts",
                str(args.semantic_prompts),
                "--min-mask-area",
                "40",
                "--checkpoint-path",
                str(args.sam3_checkpoint),
                "--no-hf-download",
                "--max-images",
                str(int(args.semantic_probe_views)),
            ]
        )

        best = choose_prompt_from_summary(probe_dir / "summary.json", fallback="unknown")
        safe = sanitize_name(best)
        out_name = f"{safe}__{stem}.ply"
        out_path = output_dir / out_name
        if out_path.exists():
            out_path = output_dir / f"{safe}__{stem}_{i:03d}.ply"
        shutil.copy2(cls_ply, out_path)

        mapping_rows.append(
            {
                "input_class_ply": str(cls_ply),
                "semantic": best,
                "output_ply": str(out_path),
                "probe_dir": str(probe_dir),
            }
        )
        print(f"[ok] {cls_ply.name} -> {best} -> {out_path.name}")

    (output_dir / "semantic_mapping.json").write_text(
        json.dumps({"items": mapping_rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] output_dir={output_dir}")


if __name__ == "__main__":
    main()

