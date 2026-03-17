import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData

from semantic_instance_pipeline import (
    load_cameras_from_colmap,
    load_cameras_from_transforms,
    load_instance_masks_for_cameras,
    parse_id2label,
    parse_int_list,
    proj2d_instances,
    sanitize_name,
    write_semantic_instance_ply,
    write_subset_ply,
)


def parse_semantic_dir_name(dirname: str) -> Tuple[Optional[int], Optional[str]]:
    m = re.match(r"^(\d+)(?:_(.*))?$", dirname.strip())
    if not m:
        return None, None
    sem_id = int(m.group(1))
    sem_label = (m.group(2) or "").strip()
    return sem_id, sem_label


def collect_semantic_models(semantic_models_dir: Path, merged_name: str) -> List[Tuple[int, str, Path]]:
    out: List[Tuple[int, str, Path]] = []
    if not semantic_models_dir.exists():
        return out

    for d in sorted(semantic_models_dir.iterdir()):
        if not d.is_dir():
            continue
        sem_id, sem_label = parse_semantic_dir_name(d.name)
        if sem_id is None:
            continue
        merged = d / merged_name
        if merged.exists():
            out.append((sem_id, sem_label, merged))
            continue

        # Fallback: some folders may only keep class_*.ply
        classes = sorted(d.glob("class_*.ply"))
        if classes:
            out.append((sem_id, sem_label, classes[0]))
    return out


def load_cameras(scene_path: Path, language_features_subdir: str, mask_suffix: str, transforms_files: str):
    language_feature_dir = scene_path / language_features_subdir
    if not language_feature_dir.exists():
        raise FileNotFoundError(f"language feature dir not found: {language_feature_dir}")

    tf_names = [x.strip() for x in transforms_files.split(",") if x.strip()]
    cameras = []
    for name in tf_names:
        cameras.extend(
            load_cameras_from_transforms(
                scene_path=scene_path,
                transforms_name=name,
                language_feature_dir=language_feature_dir,
                mask_suffix=mask_suffix,
            )
        )
    camera_source = "transforms"
    if not cameras:
        cameras = load_cameras_from_colmap(
            scene_path=scene_path,
            language_feature_dir=language_feature_dir,
            mask_suffix=mask_suffix,
        )
        camera_source = "colmap"
    if not cameras:
        raise SystemExit(
            "No valid cameras loaded from transforms or COLMAP. "
            "Check scene_path, sparse/0, and language_features/*_s.npy names."
        )
    return cameras, camera_source


def main():
    parser = argparse.ArgumentParser(
        description="Run instance segmentation (proj2d) directly on semantic models produced by semantic voting."
    )
    parser.add_argument("--scene_path", required=True, help="Scene path containing transforms*.json and sparse/0")
    parser.add_argument("--semantic_models_dir", required=True, help="Directory like <sem_ins>/semantic_models")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--merged_name", default="merged_semantic.ply", help="Merged semantic ply filename")
    parser.add_argument("--semantic_only_ids", default="", help="Optional semantic ids to process, e.g. 1,2")
    parser.add_argument("--stuff_ids", default="0,4,5,6,7,255",
                        help="Semantic ids that skip instance segmentation")
    parser.add_argument("--id2label",
                        default="0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other",
                        help="id->label mapping")
    parser.add_argument("--transforms_files", default="transforms_train.json,transforms_test.json",
                        help="Comma-separated transforms files to load")
    parser.add_argument("--language_features_subdir", default="language_features", help="Subdir for *_s.npy masks")
    parser.add_argument("--mask_suffix", default="_s.npy", help="Mask filename suffix for camera matching")
    parser.add_argument("--vote_stride", type=int, default=1, help="Use every Nth camera for instance voting")
    parser.add_argument("--instance_mask_dir", required=True, help="Directory containing 2D instance masks")
    parser.add_argument("--instance_mask_mode", default="name", choices=["index", "name"],
                        help="How to match camera to instance mask file")
    parser.add_argument("--instance_mask_suffix", default="_inst.npy", help="Mask suffix for name mode")
    parser.add_argument("--instance_mask_index_pattern", default="sam_mask_instance_view_{index:04d}.png",
                        help="Filename pattern for index mode")
    parser.add_argument("--instance_mask_level", type=int, default=0, help="Mask level when mask is .npy [L,H,W]")
    parser.add_argument("--instance_mask_ignore_ids", default="0", help="Ignore these 2D mask ids")
    parser.add_argument("--instance_min_mask_points", type=int, default=30,
                        help="Minimum projected points in one view-mask to use it")
    parser.add_argument("--instance_match_iou", type=float, default=0.2,
                        help="IoU threshold to merge local view instances across views")
    parser.add_argument("--instance_min_point_votes", type=int, default=2,
                        help="Minimum votes to keep point-level assignment")
    parser.add_argument("--min_instance_points", type=int, default=180,
                        help="Drop 3D instances smaller than this")
    parser.add_argument("--save_instance_parts", action="store_true",
                        help="Save each instance as separate PLY")
    args = parser.parse_args()

    scene_path = Path(args.scene_path)
    semantic_models_dir = Path(args.semantic_models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_models_dir = output_dir / "semantic_models"
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_instance_dir = output_dir / "semantic_instance"
    out_instance_dir.mkdir(parents=True, exist_ok=True)

    id2label = parse_id2label(args.id2label)
    stuff_ids = set(parse_int_list(args.stuff_ids))
    semantic_only_ids = set(parse_int_list(args.semantic_only_ids))
    mask_ignore_ids = parse_int_list(args.instance_mask_ignore_ids)

    cameras, camera_source = load_cameras(
        scene_path=scene_path,
        language_features_subdir=args.language_features_subdir,
        mask_suffix=args.mask_suffix,
        transforms_files=args.transforms_files,
    )

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

    semantic_inputs = collect_semantic_models(semantic_models_dir, args.merged_name)
    if not semantic_inputs:
        raise SystemExit(f"No semantic merged models found in {semantic_models_dir}")

    summary = []
    for sem_id, sem_label_src, in_ply in semantic_inputs:
        if semantic_only_ids and sem_id not in semantic_only_ids:
            continue

        sem_label = id2label.get(sem_id, sem_label_src if sem_label_src else f"id_{sem_id}")
        sem_safe = sanitize_name(sem_label)
        out_sem_dir = out_models_dir / f"{sem_id:03d}_{sem_safe}"
        out_sem_dir.mkdir(parents=True, exist_ok=True)

        ply = PlyData.read(str(in_ply))
        vertex = ply["vertex"].data
        xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
        out_merged = out_sem_dir / "merged_semantic.ply"
        write_subset_ply(vertex, out_merged)

        run_instance = sem_id not in stuff_ids
        proj2d_stats = {
            "views_considered": 0,
            "views_with_masks": 0,
            "views_used": 0,
            "local_instances_used": 0,
            "global_hypotheses": 0,
        }
        if run_instance:
            instance_id, num_instances, proj2d_stats = proj2d_instances(
                xyz=xyz,
                cameras=cameras,
                instance_mask_paths=instance_mask_paths,
                vote_stride=args.vote_stride,
                instance_mask_level=args.instance_mask_level,
                instance_mask_ignore_ids=mask_ignore_ids,
                min_mask_points=args.instance_min_mask_points,
                merge_iou=args.instance_match_iou,
                min_point_votes=args.instance_min_point_votes,
                min_instance_points=args.min_instance_points,
            )
        else:
            instance_id = np.zeros((len(vertex),), dtype=np.int32)
            num_instances = 0

        out_sem_ins = out_instance_dir / f"{sem_id:03d}_{sem_safe}_sem_ins.ply"
        write_semantic_instance_ply(vertex, out_sem_ins, sem_id, instance_id)

        if args.save_instance_parts and num_instances > 0:
            inst_dir = out_instance_dir / f"{sem_id:03d}_{sem_safe}_instances"
            inst_dir.mkdir(parents=True, exist_ok=True)
            for inst in range(1, num_instances + 1):
                m = instance_id == inst
                if not np.any(m):
                    continue
                write_subset_ply(vertex[m], inst_dir / f"instance_{inst:03d}.ply")

        summary.append(
            {
                "semantic_id": int(sem_id),
                "semantic_label": sem_label,
                "input_ply": str(in_ply),
                "num_points": int(len(vertex)),
                "instance_enabled": bool(run_instance),
                "num_instances": int(num_instances),
                "proj2d_views_considered": int(proj2d_stats["views_considered"]),
                "proj2d_views_with_masks": int(proj2d_stats["views_with_masks"]),
                "proj2d_views_used": int(proj2d_stats["views_used"]),
                "proj2d_local_instances_used": int(proj2d_stats["local_instances_used"]),
                "proj2d_global_hypotheses": int(proj2d_stats["global_hypotheses"]),
                "merged_ply": str(out_merged),
                "semantic_instance_ply": str(out_sem_ins),
            }
        )
        print(f"[semantic] id={sem_id} label={sem_label} points={len(vertex)} instances={num_instances}")

    report = {
        "scene_path": str(scene_path),
        "semantic_models_dir": str(semantic_models_dir),
        "num_cameras_used": int(len(cameras)),
        "camera_source": camera_source,
        "num_semantic_inputs": int(len(semantic_inputs)),
        "semantic_summary": summary,
        "params": {
            "semantic_only_ids": sorted(list(semantic_only_ids)),
            "stuff_ids": sorted(list(stuff_ids)),
            "vote_stride": int(args.vote_stride),
            "instance_mask_dir": str(instance_mask_dir),
            "instance_mask_mode": str(args.instance_mask_mode),
            "instance_mask_suffix": str(args.instance_mask_suffix),
            "instance_mask_index_pattern": str(args.instance_mask_index_pattern),
            "instance_mask_level": int(args.instance_mask_level),
            "instance_mask_ignore_ids": mask_ignore_ids,
            "instance_min_mask_points": int(args.instance_min_mask_points),
            "instance_match_iou": float(args.instance_match_iou),
            "instance_min_point_votes": int(args.instance_min_point_votes),
            "min_instance_points": int(args.min_instance_points),
            "merged_name": str(args.merged_name),
        },
    }
    report_path = output_dir / "semantic_models_seg3d_instance_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"done: {output_dir}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
