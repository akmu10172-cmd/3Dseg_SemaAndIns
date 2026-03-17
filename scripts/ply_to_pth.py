import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from scene.gaussian_model import GaussianModel


def build_training_args(args):
    # Minimal optimization args used by GaussianModel.training_setup(...)
    return SimpleNamespace(
        percent_dense=args.percent_dense,
        position_lr_init=args.position_lr_init,
        position_lr_final=args.position_lr_final,
        position_lr_delay_mult=args.position_lr_delay_mult,
        position_lr_max_steps=args.position_lr_max_steps,
        feature_lr=args.feature_lr,
        ins_feat_lr=args.ins_feat_lr,
        opacity_lr=args.opacity_lr,
        scaling_lr=args.scaling_lr,
        rotation_lr=args.rotation_lr,
        frozen_init_pts=args.frozen_init_pts,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a trainable OpenGaussian checkpoint from point_cloud.ply. "
            "The output can be used by train.py --start_checkpoint."
        )
    )
    parser.add_argument("--input_ply", required=True, help="Path to point_cloud.ply")
    parser.add_argument("--output_pth", default="", help="Output checkpoint path, e.g. .../chkpnt30000.pth")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration index saved in checkpoint tuple")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree used by GaussianModel")
    parser.add_argument(
        "--spatial_lr_scale",
        type=float,
        default=1.0,
        help="spatial_lr_scale in checkpoint (affects xyz lr after resume)",
    )

    # Optimization defaults aligned with arguments/OptimizationParams.
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--position_lr_init", type=float, default=0.00016)
    parser.add_argument("--position_lr_final", type=float, default=0.0000016)
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    parser.add_argument("--position_lr_max_steps", type=int, default=30000)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--ins_feat_lr", type=float, default=0.001)
    parser.add_argument("--opacity_lr", type=float, default=0.05)
    parser.add_argument("--scaling_lr", type=float, default=0.005)
    parser.add_argument("--rotation_lr", type=float, default=0.001)
    parser.add_argument("--frozen_init_pts", action="store_true", default=False)
    args = parser.parse_args()

    in_ply = Path(args.input_ply)
    if not in_ply.exists():
        raise FileNotFoundError(f"input_ply not found: {in_ply}")

    if args.output_pth:
        out_pth = Path(args.output_pth)
    else:
        out_pth = in_ply.parent / f"chkpnt{int(args.iteration)}.pth"
    out_pth.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required in this repo to load/save GaussianModel tensors.")

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(in_ply))
    gaussians.spatial_lr_scale = float(args.spatial_lr_scale)

    training_args = build_training_args(args)
    gaussians.training_setup(training_args)

    n_pts = gaussians.get_xyz.shape[0]
    gaussians.max_radii2D = torch.zeros((n_pts,), device="cuda")
    gaussians.xyz_gradient_accum = torch.zeros((n_pts, 1), device="cuda")
    gaussians.denom = torch.zeros((n_pts, 1), device="cuda")

    model_params = gaussians.capture()
    checkpoint = (model_params, int(args.iteration))
    torch.save(checkpoint, str(out_pth))

    print("done")
    print(f"input_ply={in_ply}")
    print(f"output_pth={out_pth}")
    print(f"iteration={int(args.iteration)}")
    print(f"num_points={int(n_pts)}")
    print("note=optimizer state is newly initialized, not recovered from original training run")


if __name__ == "__main__":
    main()
