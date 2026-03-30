#!/usr/bin/env python3
"""Crop OpenGaussian checkpoint to keep only the center spatial region."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Tuple

import torch


def normalize_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")


def _slice_first_dim(x: Any, idx: torch.Tensor, n_points: int) -> Any:
    if torch.is_tensor(x) and x.ndim >= 1 and x.shape[0] == n_points:
        use_idx = idx.to(x.device) if x.device != idx.device else idx
        y = x.index_select(0, use_idx).contiguous()
        if isinstance(x, torch.nn.Parameter):
            return torch.nn.Parameter(y.detach(), requires_grad=x.requires_grad)
        if y.requires_grad != x.requires_grad:
            return y.detach().requires_grad_(x.requires_grad)
        return y
    return x


def _to_device_keep_kind(x: Any, device: torch.device) -> Any:
    if not torch.is_tensor(x):
        return x
    y = x.to(device=device, non_blocking=False)
    if isinstance(x, torch.nn.Parameter):
        return torch.nn.Parameter(y, requires_grad=x.requires_grad)
    if y.requires_grad != x.requires_grad:
        return y.detach().requires_grad_(x.requires_grad)
    return y


def _target_device(mode: str) -> torch.device:
    mode = mode.lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("target_device=cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_center_keep_index(
    xyz: torch.Tensor, keep_ratio_xy: float, keep_ratio_z: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz_cpu = xyz.detach().float().cpu()
    mins = xyz_cpu.min(dim=0).values
    maxs = xyz_cpu.max(dim=0).values
    center = (mins + maxs) * 0.5
    half = (maxs - mins) * 0.5

    half_x = half[0] * keep_ratio_xy
    half_y = half[1] * keep_ratio_xy
    half_z = half[2] * keep_ratio_z

    mask = (
        (xyz_cpu[:, 0] >= center[0] - half_x)
        & (xyz_cpu[:, 0] <= center[0] + half_x)
        & (xyz_cpu[:, 1] >= center[1] - half_y)
        & (xyz_cpu[:, 1] <= center[1] + half_y)
        & (xyz_cpu[:, 2] >= center[2] - half_z)
        & (xyz_cpu[:, 2] <= center[2] + half_z)
    )
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    return idx, mins, maxs, center, half


def _crop_model_params(
    model_params: Any,
    keep_ratio_xy: float,
    keep_ratio_z: float,
    target_device: torch.device,
) -> Tuple[Any, int, int]:
    if not isinstance(model_params, (list, tuple)) or len(model_params) < 2:
        raise ValueError("Unsupported model_params format.")
    if not torch.is_tensor(model_params[1]):
        raise ValueError("Cannot infer xyz tensor from model_params[1].")

    xyz = model_params[1]
    n_points = int(xyz.shape[0])
    idx, mins, maxs, center, half = _build_center_keep_index(
        xyz, keep_ratio_xy=keep_ratio_xy, keep_ratio_z=keep_ratio_z
    )
    if idx.numel() <= 0:
        raise ValueError("Center crop keeps 0 points. Increase keep ratios.")

    data = list(model_params)
    for i, x in enumerate(data):
        if torch.is_tensor(x):
            cropped = _slice_first_dim(x, idx, n_points)
            data[i] = _to_device_keep_kind(cropped, target_device)

    # 14-field format: optimizer is index 12; old format: index 10.
    opt_idx = 12 if len(data) >= 14 else (10 if len(data) >= 11 else None)
    if opt_idx is not None and isinstance(data[opt_idx], dict):
        opt = data[opt_idx]
        new_opt = {
            "state": {},
            "param_groups": opt.get("param_groups", []),
        }
        for k, st in opt.get("state", {}).items():
            if isinstance(st, dict):
                st_new = {}
                for sk, sv in st.items():
                    sliced = _slice_first_dim(sv, idx, n_points)
                    st_new[sk] = _to_device_keep_kind(sliced, target_device)
                new_opt["state"][k] = st_new
            else:
                new_opt["state"][k] = st
        data[opt_idx] = new_opt

    out = tuple(data) if isinstance(model_params, tuple) else data
    print(
        "bbox_min=", mins.tolist(),
        "bbox_max=", maxs.tolist(),
        "center=", center.tolist(),
        "half=", half.tolist(),
    )
    return out, n_points, int(idx.numel())


def crop_checkpoint(
    input_ckpt: str,
    output_ckpt: str,
    keep_ratio_xy: float,
    keep_ratio_z: float,
    target_device: str,
) -> Tuple[str, int, int]:
    in_path = Path(normalize_path(input_ckpt))
    out_path = Path(normalize_path(output_ckpt))
    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = torch.load(str(in_path), map_location="cpu")
    dev = _target_device(target_device)

    ckpt_out: Any
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and isinstance(raw[1], int):
        model_params, iteration = raw
        model_params_new, old_n, new_n = _crop_model_params(
            model_params, keep_ratio_xy, keep_ratio_z, dev
        )
        ckpt_out = (model_params_new, int(iteration))
    elif isinstance(raw, dict) and "model_params" in raw:
        model_params = raw["model_params"]
        iteration = int(raw.get("iteration", 0))
        model_params_new, old_n, new_n = _crop_model_params(
            model_params, keep_ratio_xy, keep_ratio_z, dev
        )
        ckpt_out = dict(raw)
        ckpt_out["model_params"] = model_params_new
        ckpt_out["iteration"] = int(iteration)
    else:
        raise ValueError("Unsupported checkpoint format.")

    torch.save(ckpt_out, str(out_path))
    return str(out_path), old_n, new_n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep only center region points from an OpenGaussian checkpoint."
    )
    parser.add_argument("--input_ckpt", required=True, help="Input checkpoint path")
    parser.add_argument("--output_ckpt", default="", help="Output checkpoint path")
    parser.add_argument(
        "--keep_ratio_xy",
        type=float,
        default=0.5,
        help="Keep central XY ratio in each direction. Range: (0, 1].",
    )
    parser.add_argument(
        "--keep_ratio_z",
        type=float,
        default=1.0,
        help="Keep central Z ratio in each direction. Range: (0, 1].",
    )
    parser.add_argument(
        "--target_device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for tensors in output checkpoint.",
    )
    args = parser.parse_args()

    if not (0.0 < args.keep_ratio_xy <= 1.0):
        raise ValueError("--keep_ratio_xy must be in (0, 1].")
    if not (0.0 < args.keep_ratio_z <= 1.0):
        raise ValueError("--keep_ratio_z must be in (0, 1].")

    in_path = Path(normalize_path(args.input_ckpt))
    if args.output_ckpt:
        out_path = Path(normalize_path(args.output_ckpt))
    else:
        suffix = f".center_xy{int(args.keep_ratio_xy*100)}_z{int(args.keep_ratio_z*100)}.pth"
        out_path = in_path.with_name(in_path.stem + suffix)

    out, old_n, new_n = crop_checkpoint(
        input_ckpt=str(in_path),
        output_ckpt=str(out_path),
        keep_ratio_xy=float(args.keep_ratio_xy),
        keep_ratio_z=float(args.keep_ratio_z),
        target_device=args.target_device,
    )
    print(f"input_ckpt={in_path}")
    print(f"output_ckpt={out}")
    print(f"points={old_n}->{new_n}")


if __name__ == "__main__":
    main()
