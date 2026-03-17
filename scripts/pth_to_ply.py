import argparse
import os
import re

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _flatten_features(features, name):
    if not isinstance(features, torch.Tensor):
        raise ValueError(f"{name} is not a torch.Tensor")
    if features.ndim == 3:
        return features.transpose(1, 2).reshape(features.shape[0], -1)
    if features.ndim == 2:
        return features
    raise ValueError(f"{name} has unexpected shape {tuple(features.shape)}")


def _infer_iteration(path):
    match = re.search(r"chkpnt(\\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, (tuple, list)) and len(checkpoint) == 2 and isinstance(checkpoint[1], int):
        model_params = checkpoint[0]
    elif isinstance(checkpoint, dict) and "model_params" in checkpoint:
        model_params = checkpoint["model_params"]
    else:
        model_params = checkpoint
    if not isinstance(model_params, (tuple, list)):
        raise ValueError("Unsupported checkpoint format; expected tuple or list of model params.")
    if len(model_params) < 9:
        raise ValueError(f"Model params too short: {len(model_params)}")
    return model_params


def _to_numpy(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} is not a torch.Tensor")
    return tensor.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Convert OpenGaussian checkpoint to PLY with dynamic ins_feat dims.")
    parser.add_argument("--checkpoint", required=True, help="Path to chkpntXXXX.pth")
    parser.add_argument("--output", default=None, help="Output PLY path")
    parser.add_argument("--ins_feat_dim", type=int, default=None, help="Override ins_feat dims (pads or truncates)")
    parser.add_argument("--use_ins_feat_q", action="store_true", help="Use quantized ins_feat if available")
    parser.add_argument("--add_rgb", action="store_true", default=True, help="Add red/green/blue from ins_feat")
    parser.add_argument("--no_add_rgb", action="store_false", dest="add_rgb", help="Disable red/green/blue output")
    args = parser.parse_args()

    model_params = _load_checkpoint(args.checkpoint)
    _, xyz, f_dc, f_rest, scaling, rotation, opacity, ins_feat, ins_feat_q, *_ = model_params

    if args.use_ins_feat_q and isinstance(ins_feat_q, torch.Tensor) and ins_feat_q.numel() > 0:
        ins_feat_tensor = ins_feat_q
    else:
        ins_feat_tensor = ins_feat

    xyz_np = _to_numpy(xyz, "xyz")
    f_dc_np = _flatten_features(f_dc, "f_dc").detach().cpu().numpy()
    f_rest_np = _flatten_features(f_rest, "f_rest").detach().cpu().numpy()
    opacity_np = _to_numpy(opacity, "opacity")
    scaling_np = _to_numpy(scaling, "scaling")
    rotation_np = _to_numpy(rotation, "rotation")
    ins_feat_np = _to_numpy(ins_feat_tensor, "ins_feat")

    n_points = xyz_np.shape[0]
    if args.ins_feat_dim is None:
        ins_feat_dim = ins_feat_np.shape[1]
    else:
        ins_feat_dim = args.ins_feat_dim
        if ins_feat_np.shape[1] < ins_feat_dim:
            pad = np.zeros((n_points, ins_feat_dim - ins_feat_np.shape[1]), dtype=ins_feat_np.dtype)
            ins_feat_np = np.concatenate([ins_feat_np, pad], axis=1)
        else:
            ins_feat_np = ins_feat_np[:, :ins_feat_dim]

    if f_dc_np.shape[0] != n_points:
        raise ValueError("f_dc point count mismatch")
    if f_rest_np.shape[0] != n_points:
        raise ValueError("f_rest point count mismatch")

    if args.output is None:
        iteration = _infer_iteration(args.checkpoint)
        if iteration is None:
            raise ValueError("Cannot infer iteration from checkpoint name; pass --output.")
        base_dir = os.path.dirname(args.checkpoint)
        out_dir = os.path.join(base_dir, "point_cloud", f"iteration_{iteration}")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "point_cloud.ply")
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    for i in range(f_dc_np.shape[1]):
        dtype_list.append((f"f_dc_{i}", "f4"))
    for i in range(f_rest_np.shape[1]):
        dtype_list.append((f"f_rest_{i}", "f4"))
    dtype_list.append(("opacity", "f4"))
    for i in range(scaling_np.shape[1]):
        dtype_list.append((f"scale_{i}", "f4"))
    for i in range(rotation_np.shape[1]):
        dtype_list.append((f"rot_{i}", "f4"))
    for i in range(ins_feat_dim):
        dtype_list.append((f"ins_feat_{i}", "f4"))
    if args.add_rgb:
        dtype_list.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

    elements = np.empty(n_points, dtype=dtype_list)
    elements["x"] = xyz_np[:, 0]
    elements["y"] = xyz_np[:, 1]
    elements["z"] = xyz_np[:, 2]
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0
    for i in range(f_dc_np.shape[1]):
        elements[f"f_dc_{i}"] = f_dc_np[:, i]
    for i in range(f_rest_np.shape[1]):
        elements[f"f_rest_{i}"] = f_rest_np[:, i]
    elements["opacity"] = opacity_np.reshape(-1)
    for i in range(scaling_np.shape[1]):
        elements[f"scale_{i}"] = scaling_np[:, i]
    for i in range(rotation_np.shape[1]):
        elements[f"rot_{i}"] = rotation_np[:, i]
    for i in range(ins_feat_dim):
        elements[f"ins_feat_{i}"] = ins_feat_np[:, i]
    if args.add_rgb:
        if ins_feat_np.shape[1] >= 3:
            vis = (ins_feat_np[:, :3] + 1.0) * 0.5 * 255.0
        else:
            vis = np.zeros((n_points, 3), dtype=ins_feat_np.dtype)
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        elements["red"] = vis[:, 0]
        elements["green"] = vis[:, 1]
        elements["blue"] = vis[:, 2]

    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
