#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, List, Tuple

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


def _quarter_indices(xyz: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    xyz_cpu = xyz.detach().float().cpu()
    mins = xyz_cpu.min(dim=0).values
    maxs = xyz_cpu.max(dim=0).values
    center = (mins + maxs) * 0.5
    cx, cy = float(center[0]), float(center[1])
    x = xyz_cpu[:, 0]
    y = xyz_cpu[:, 1]
    masks = [
        (x <= cx) & (y <= cy),  # q0
        (x > cx) & (y <= cy),   # q1
        (x <= cx) & (y > cy),   # q2
        (x > cx) & (y > cy),    # q3
    ]
    idxs = [torch.nonzero(m, as_tuple=False).squeeze(1) for m in masks]
    return idxs, mins, maxs


def _crop_model_params(model_params: Any, idx: torch.Tensor, target_device: torch.device) -> Tuple[Any, int, int]:
    if not isinstance(model_params, (list, tuple)) or len(model_params) < 2:
        raise ValueError("Unsupported model_params format.")
    if not torch.is_tensor(model_params[1]):
        raise ValueError("Cannot infer xyz tensor from model_params[1].")

    xyz = model_params[1]
    n_points = int(xyz.shape[0])
    if idx.numel() <= 0:
        raise ValueError("Quarter keeps 0 points.")

    data = list(model_params)
    for i, x in enumerate(data):
        if torch.is_tensor(x):
            cropped = _slice_first_dim(x, idx, n_points)
            data[i] = _to_device_keep_kind(cropped, target_device)

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
    return out, n_points, int(idx.numel())


def split_checkpoint(
    input_ckpt: str,
    output_dir: str,
    target_device: str,
    prefix: str = "CHKPNT31000_q",
) -> List[str]:
    in_path = Path(normalize_path(input_ckpt))
    out_dir = Path(normalize_path(output_dir))
    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(str(in_path), map_location="cpu")
    dev = _target_device(target_device)

    if isinstance(raw, (list, tuple)) and len(raw) == 2 and isinstance(raw[1], int):
        model_params, iteration = raw
    elif isinstance(raw, dict) and "model_params" in raw:
        model_params = raw["model_params"]
        iteration = int(raw.get("iteration", 0))
    else:
        raise ValueError("Unsupported checkpoint format.")

    if not torch.is_tensor(model_params[1]):
        raise ValueError("Unsupported checkpoint model_params[1].")
    idxs, mins, maxs = _quarter_indices(model_params[1])
    print("bbox_min=", mins.tolist(), "bbox_max=", maxs.tolist(), "iter=", int(iteration))

    outs: List[str] = []
    for qi, idx in enumerate(idxs):
        mp_new, old_n, new_n = _crop_model_params(model_params, idx, dev)
        if isinstance(raw, (list, tuple)):
            ckpt_out: Any = (mp_new, int(iteration))
        else:
            ckpt_out = dict(raw)
            ckpt_out["model_params"] = mp_new
            ckpt_out["iteration"] = int(iteration)
        out_path = out_dir / f"{prefix}{qi}.pth"
        torch.save(ckpt_out, str(out_path))
        outs.append(str(out_path))
        print(f"q{qi}: points={old_n}->{new_n} out={out_path}")
    return outs


def main() -> None:
    p = argparse.ArgumentParser(description="Split OpenGaussian checkpoint into 4 XOY quarters.")
    p.add_argument("--input_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--prefix", default="CHKPNT31000_q")
    args = p.parse_args()

    split_checkpoint(
        input_ckpt=args.input_ckpt,
        output_dir=args.output_dir,
        target_device=args.target_device,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()

