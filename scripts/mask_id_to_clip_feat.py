import argparse
from pathlib import Path

import numpy as np


DEFAULT_ID2LABEL = {
    0: "background",
    1: "vehicle",
    2: "person",
    3: "bicycle",
    4: "vegetation",
    5: "road",
    6: "traffic facility",
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


def load_mask_2d(path: Path, mask_level: int):
    arr = np.load(path)
    if arr.ndim == 2:
        return arr.astype(np.int32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0].astype(np.int32)
        if mask_level < 0 or mask_level >= arr.shape[0]:
            raise ValueError(f"mask_level={mask_level} out of range for {path.name}, shape={arr.shape}")
        return arr[mask_level].astype(np.int32)
    raise ValueError(f"Unsupported mask shape {arr.shape} in {path}")


def build_texts(class_ids, id2label, template):
    texts = []
    for cid in class_ids:
        label = id2label.get(int(cid), f"class {int(cid)}")
        texts.append(template.format(label=label))
    return texts


def encode_text_open_clip(texts, model_name, pretrained, device, normalize):
    import torch
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens).float()
        if normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
    return feats.cpu().numpy()


def encode_text_transformers(texts, model_path, device, normalize):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(model_path).to(device)
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_path)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        feats = model.get_text_features(**inputs).float()
        if normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
    return feats.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Build CLIP *_f.npy from semantic *_s.npy masks.")
    parser.add_argument("--src_dir", required=True, help="Directory containing *_s.npy")
    parser.add_argument("--dst_dir", required=True, help="Directory to save *_f.npy")
    parser.add_argument("--suffix", default="_s.npy", help="Input mask suffix")
    parser.add_argument("--mask_level", type=int, default=0, help="Mask level if mask is [L,H,W]")
    parser.add_argument(
        "--id2label",
        default="0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic facility,7:other",
        help="id->label mapping, e.g. '1:car,2:person,5:road'",
    )
    parser.add_argument(
        "--template",
        default="a photo of {label}",
        help="Prompt template, use {label} placeholder",
    )
    parser.add_argument("--backend", choices=["open_clip", "transformers"], default="open_clip")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--no_normalize", action="store_true", help="Do not L2-normalize CLIP text features")
    parser.add_argument("--skip_background", action="store_true", help="Force id=0 feature to all-zero vector")

    # open_clip
    parser.add_argument("--clip_model", default="ViT-B-32", help="open_clip model name")
    parser.add_argument("--clip_pretrained", default="openai", help="open_clip pretrained tag or checkpoint path")

    # transformers
    parser.add_argument("--clip_model_path", default="openai/clip-vit-base-patch32",
                        help="Transformers CLIP model path or local dir")

    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    id2label = parse_id2label(args.id2label)
    normalize = not args.no_normalize

    files = sorted(src_dir.glob(f"*{args.suffix}"))
    if not files:
        raise SystemExit(f"No files found in {src_dir} matching *{args.suffix}")

    # Gather all class ids used in dataset.
    all_ids = set()
    per_file_max = {}
    for p in files:
        m = load_mask_2d(p, args.mask_level)
        ids = np.unique(m).astype(np.int32)
        all_ids.update(ids.tolist())
        per_file_max[p] = int(ids.max())

    all_ids = sorted(int(x) for x in all_ids if x >= 0)
    texts = build_texts(all_ids, id2label, args.template)

    if args.backend == "open_clip":
        feats = encode_text_open_clip(
            texts=texts,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
            normalize=normalize,
        )
    else:
        feats = encode_text_transformers(
            texts=texts,
            model_path=args.clip_model_path,
            device=args.device,
            normalize=normalize,
        )

    feat_dim = feats.shape[1]
    id2feat = {cid: feats[i] for i, cid in enumerate(all_ids)}
    print(f"Encoded {len(all_ids)} class ids, feat_dim={feat_dim}")
    if feat_dim != 512:
        print(f"[warn] feat_dim is {feat_dim}, while original Stage3 expects 512.")

    saved = 0
    for p in files:
        max_id = per_file_max[p]
        out = np.zeros((max_id + 1, feat_dim), dtype=np.float32)
        for cid in range(max_id + 1):
            if cid in id2feat:
                out[cid] = id2feat[cid]
        if args.skip_background and max_id >= 0:
            out[0] = 0.0

        stem = p.name[: -len(args.suffix)]
        out_path = dst_dir / f"{stem}_f.npy"
        np.save(out_path, out.astype(np.float32))
        saved += 1

    print(f"done: {saved}/{len(files)} -> {dst_dir}")


if __name__ == "__main__":
    main()
