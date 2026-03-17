import argparse
import json
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a scene subset by image whitelist for faster training."
    )
    parser.add_argument("--src", type=str, required=True, help="Source scene path")
    parser.add_argument("--dst", type=str, required=True, help="Destination subset scene path")
    parser.add_argument(
        "--keep_list",
        type=str,
        required=True,
        help="Text file with one image name (or stem) per line",
    )
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--features_subdir", type=str, default="language_features")
    parser.add_argument(
        "--copy_sparse",
        action="store_true",
        help="Copy sparse/ directory from source scene",
    )
    parser.add_argument(
        "--copy_database",
        action="store_true",
        help="Copy database.db if exists",
    )
    parser.add_argument(
        "--filter_transforms",
        action="store_true",
        help="Filter transforms_train/test.json frames by selected images",
    )
    return parser.parse_args()


def load_keep_set(keep_list_path: Path):
    keep_names = set()
    keep_stems = set()
    for raw in keep_list_path.read_text(encoding="utf-8").splitlines():
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


def is_kept(img_path: Path, keep_names, keep_stems):
    return img_path.name.lower() in keep_names or img_path.stem in keep_stems


def filter_transforms_file(src_file: Path, dst_file: Path, keep_names, keep_stems):
    if not src_file.exists():
        return 0, 0
    data = json.loads(src_file.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    kept = []
    for fr in frames:
        fp = str(fr.get("file_path", ""))
        base = Path(fp).name
        stem = Path(fp).stem if Path(fp).suffix else Path(fp).name
        if base.lower() in keep_names or stem in keep_stems:
            kept.append(fr)
    data["frames"] = kept
    dst_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return len(frames), len(kept)


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    keep_list = Path(args.keep_list)
    if not src.exists():
        raise SystemExit(f"Source scene not found: {src}")
    if not keep_list.exists():
        raise SystemExit(f"keep_list not found: {keep_list}")

    keep_names, keep_stems = load_keep_set(keep_list)
    if not keep_names and not keep_stems:
        raise SystemExit("keep_list is empty after removing comments/blank lines.")

    src_images = src / args.images_subdir
    src_feats = src / args.features_subdir
    dst_images = dst / args.images_subdir
    dst_feats = dst / args.features_subdir
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_feats.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted([p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS])
    kept_imgs = [p for p in all_imgs if is_kept(p, keep_names, keep_stems)]

    copied_img = 0
    copied_mask_s = 0
    copied_mask_f = 0
    missing_mask_s = 0
    for img in kept_imgs:
        shutil.copy2(img, dst_images / img.name)
        copied_img += 1

        mask_s = src_feats / f"{img.stem}_s.npy"
        if mask_s.exists():
            shutil.copy2(mask_s, dst_feats / mask_s.name)
            copied_mask_s += 1
        else:
            missing_mask_s += 1

        mask_f = src_feats / f"{img.stem}_f.npy"
        if mask_f.exists():
            shutil.copy2(mask_f, dst_feats / mask_f.name)
            copied_mask_f += 1

    if args.copy_sparse and (src / "sparse").exists():
        shutil.copytree(src / "sparse", dst / "sparse", dirs_exist_ok=True)

    if args.copy_database and (src / "database.db").exists():
        shutil.copy2(src / "database.db", dst / "database.db")

    tf_stats = []
    if args.filter_transforms:
        for name in ("transforms_train.json", "transforms_test.json"):
            src_tf = src / name
            dst_tf = dst / name
            total, kept = filter_transforms_file(src_tf, dst_tf, keep_names, keep_stems)
            if total > 0:
                tf_stats.append((name, total, kept))
    else:
        for name in ("transforms_train.json", "transforms_test.json"):
            src_tf = src / name
            if src_tf.exists():
                shutil.copy2(src_tf, dst / name)

    print("=== Subset Build Summary ===")
    print(f"source images: {len(all_imgs)}")
    print(f"kept images:   {len(kept_imgs)}")
    print(f"copied images: {copied_img}")
    print(f"copied _s.npy: {copied_mask_s}")
    print(f"missing _s.npy:{missing_mask_s}")
    print(f"copied _f.npy: {copied_mask_f}")
    if tf_stats:
        for name, total, kept in tf_stats:
            print(f"{name}: {kept}/{total} frames kept")
    print(f"done: {dst}")


if __name__ == "__main__":
    main()
