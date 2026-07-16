import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

# Ensure src/ is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bts_nvs.data.dataset import SceneDataset, estimate_image_cache_bytes
from bts_nvs.data.holdout import HoldoutSplit, load_holdout_split
from bts_nvs.data.manifest import (
    build_scene_manifest,
    load_scene_manifest,
    save_scene_manifest,
)
from bts_nvs.data.sparse_subset import build_split_sparse_initialization
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.initialization import initialize_from_manifest
from bts_nvs.rendering.gsplat_renderer import render_gaussians
from bts_nvs.training.trainer import Trainer
from bts_nvs.training.qualification import (
    CALIBRATION_SCENES,
    evaluate_internal_validation,
    save_qualification_report,
)
from bts_nvs.evaluation.metrics import LpipsBackend
from bts_nvs.training.resources import (
    linux_memory_status,
    require_cache_capacity,
    require_no_swap,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Main training entry point for BTS Digital Twin Novel View Synthesis."
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=None,
        help="Optional writable manifest artifact directory.",
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Path to the scene directory containing train and test folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for camera sampling and PyTorch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save config, environments, logs, preview renders, and checkpoints.",
    )
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=4,
        help="Downscaling division factor for train images (e.g. 4 for 4x downscaling).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum training/optimization steps.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=3000,
        help="Save checkpoint interval frequency.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--cache_images",
        action="store_true",
        help="Decode, undistort, and resize each selected train image once.",
    )
    parser.add_argument(
        "--pinned_transfer",
        action="store_true",
        help="Use a two-sample pinned host-memory ring for CUDA transfers.",
    )
    parser.add_argument(
        "--profile_input",
        action="store_true",
        help="Run the fixed 50 warm-up + 500 measured-step input profile.",
    )
    parser.add_argument(
        "--internal_holdout",
        action="store_true",
        help="Train only on the prepared leakage-controlled internal split.",
    )
    parser.add_argument(
        "--qualification_candidate",
        choices=("B0-reference", "B0-compact"),
        default=None,
        help="Run one locked Phase 4.4 candidate on the internal holdout.",
    )
    return parser.parse_args()


def validate_resize_factor(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("resize_factor must be a positive integer")


def validate_output_directory(output_dir: Path, resume: str | Path | None) -> None:
    output = Path(output_dir)
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume}")
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; use --resume or a new path: {output}"
        )


def validate_profile_args(args) -> None:
    if not args.profile_input:
        return
    if args.max_steps != 550:
        raise ValueError("--profile_input requires exactly 550 max steps")
    if args.resume is not None:
        raise ValueError("--profile_input requires a fresh run without --resume")


def validate_qualification_args(args) -> None:
    if args.qualification_candidate is None:
        return
    if (
        args.max_steps != 7000
        or args.resize_factor != 1
        or args.seed != 0
        or not args.cache_images
        or not args.pinned_transfer
        or args.resume is not None
    ):
        raise ValueError(
            "qualification requires a fresh factor-1, seed-0, 7000-step run "
            "with cached images and pinned transfer"
        )


def internal_holdout_enabled(args) -> bool:
    return bool(
        args.internal_holdout
        or args.profile_input
        or args.qualification_candidate is not None
    )


def should_save_checkpoints(args) -> bool:
    return not args.profile_input and args.qualification_candidate is None


def load_internal_holdout(
    manifest_dir: Path,
    manifest,
    *,
    enabled: bool,
) -> HoldoutSplit | None:
    if not enabled:
        return None
    path = Path(manifest_dir) / "holdout.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"required internal holdout artifact does not exist: {path}"
        )
    return load_holdout_split(path, manifest)


def build_initialization_manifest(manifest, scene_root: Path, split):
    if split is None:
        return manifest
    sparse = build_split_sparse_initialization(manifest, scene_root, split)
    return replace(
        manifest,
        sparse_points=sparse.points,
        sparse_colors=sparse.colors,
    )


def validate_host_resources(
    *,
    cache_bytes: int,
    memory_status: tuple[int, int] | None,
) -> None:
    if memory_status is None:
        return
    available_bytes, swap_used_bytes = memory_status
    require_no_swap(swap_used_bytes)
    if cache_bytes:
        require_cache_capacity(cache_bytes, available_bytes=available_bytes)


def validate_preflight_gradients(parameters) -> None:
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients or any(
        not torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError(
            "gsplat preflight did not produce a finite parameter gradient"
        )


def run_cuda_preflight() -> None:
    """Fail before scene loading unless real gsplat CUDA forward/backward works."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-enabled PyTorch is required; torch.cuda.is_available() is False"
        )
    device = torch.device("cuda")
    gaussians = GaussianParameters(
        means=torch.tensor([[0.0, 0.0, 5.0]], device=device),
        scales=torch.full((1, 3), math.log(0.1), device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        opacities=torch.zeros(1, device=device),
        sh0=torch.zeros((1, 1, 3), device=device),
        shN=torch.zeros((1, 15, 3), device=device),
    )
    try:
        result = render_gaussians(
            gaussians=gaussians,
            viewmat=torch.eye(4, device=device),
            intrinsics=CameraIntrinsics(32, 32, 24.0, 24.0, 16.0, 16.0),
            active_sh_degree=0,
        )
        result.rgb.sum().backward()
        validate_preflight_gradients(gaussians.parameters())
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError("gsplat CUDA forward/backward preflight failed") from error


def build_training_config(
    args,
    manifest,
    resize: tuple[int, int],
    *,
    split: HoldoutSplit | None = None,
) -> dict:
    width, height = resize
    config = {
        "scene_id": manifest.scene_id,
        "resize_factor": args.resize_factor,
        "resize_width": width,
        "resize_height": height,
        "undistort": True,
        "cache_images": bool(args.cache_images),
        "pinned_transfer": bool(args.pinned_transfer),
        "profile_input": bool(args.profile_input),
        "internal_holdout": split is not None,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "prune_opa": 0.005,
        "grow_grad2d": (
            0.0003 if args.qualification_candidate == "B0-compact" else 0.0002
        ),
        "grow_scale3d": 0.01,
        "refine_start_step": 500,
        "refine_stop_step": 15000,
        "refine_every": 100,
        "reset_every": 3000,
        "lambda_dssim": 0.2,
        "data_range": 1.0,
    }
    if args.qualification_candidate is not None:
        config["qualification_candidate"] = args.qualification_candidate
    if split is not None:
        config.update(
            {
                "holdout_algorithm": split.algorithm,
                "holdout_manifest_sha256": split.manifest_sha256,
                "internal_train_count": len(split.train_image_names),
                "guard_count": len(split.guard_image_names),
                "validation_count": len(split.validation_image_names),
            }
        )
    return config


def write_convergence_report(
    path: Path,
    initial: dict[str, float | bool | None],
    final: dict[str, float | bool | None],
) -> dict[str, object]:
    initial_psnr = initial["psnr_db"]
    final_psnr = final["psnr_db"]
    psnr_delta = (
        None
        if initial_psnr is None or final_psnr is None
        else float(final_psnr - initial_psnr)
    )
    ssim_delta = float(final["ssim"] - initial["ssim"])
    report = {
        "train_camera_index": 0,
        "initial": initial,
        "final": final,
        "psnr_delta_db": psnr_delta,
        "ssim_delta": ssim_delta,
        "quality_improved": bool(
            psnr_delta is not None and psnr_delta > 0.0 and ssim_delta > 0.0
        ),
        "final_render_non_blank": bool(
            final["alpha_coverage"] >= 0.01 and final["rgb_std"] >= 0.01
        ),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def write_json_record(path: Path, record: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json_record(path: Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required run record does not exist: {source}")
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"run record must contain a JSON object: {source}")
    return record


def main():
    args = parse_args()
    validate_resize_factor(args.resize_factor)
    validate_profile_args(args)
    validate_qualification_args(args)
    validate_output_directory(Path(args.output_dir), args.resume)
    run_cuda_preflight()

    scene_root = Path(args.scene_dir)
    manifest_dir = (
        Path(args.manifest_dir) if args.manifest_dir else scene_root / "manifest"
    )
    manifest_json = manifest_dir / "manifest.json"

    # 1. Build or Load Scene Manifest
    if not manifest_json.is_file():
        print(f"Building scene manifest for: {scene_root}")
        manifest = build_scene_manifest(scene_root)
        save_scene_manifest(manifest, manifest_dir)
    else:
        print(f"Loading scene manifest from: {manifest_json}")
        manifest = load_scene_manifest(manifest_json, scene_root)
    if (
        args.qualification_candidate is not None
        and manifest.scene_id not in CALIBRATION_SCENES
    ):
        raise ValueError("qualification scene is not in the locked calibration set")
    split = load_internal_holdout(
        manifest_dir,
        manifest,
        enabled=internal_holdout_enabled(args),
    )

    # 2. Setup resolution resizing
    ref_camera = manifest.train_intrinsics[0]
    w = ref_camera.width // args.resize_factor
    h = ref_camera.height // args.resize_factor
    resize = (w, h)
    print(f"Downscaling images by factor {args.resize_factor} to resolution: {resize}")
    validate_host_resources(
        cache_bytes=(
            estimate_image_cache_bytes(
                manifest,
                resize=resize,
                indices=(
                    tuple(
                        manifest.train_image_names.index(name)
                        for name in split.train_image_names
                    )
                    if split is not None
                    else None
                ),
            )
            if args.cache_images
            else 0
        ),
        memory_status=linux_memory_status(),
    )

    # 3. Setup Dataset
    dataset = SceneDataset(
        manifest,
        scene_root,
        image_names=(split.train_image_names if split is not None else None),
        undistort=True,
        resize=resize,
        cache_images=args.cache_images,
    )
    validation_dataset = (
        SceneDataset(
            manifest,
            scene_root,
            image_names=split.validation_image_names,
            undistort=True,
            resize=resize,
            cache_images=args.cache_images,
        )
        if args.qualification_candidate is not None and split is not None
        else None
    )

    # 4. Initialize adaptively from sparse point cloud
    print("Adaptive scale initialization from SfM sparse point cloud...")
    initialization_manifest = build_initialization_manifest(
        manifest,
        scene_root,
        split,
    )
    gaussians = initialize_from_manifest(initialization_manifest)
    print(f"Initialized {gaussians.num_gaussians} 3D Gaussian primitives.")

    # 5. Build Config baseline B0, including preprocessing identity.
    config = build_training_config(args, manifest, resize, split=split)

    # 6. Instantiate Trainer
    trainer = Trainer(
        gaussians=gaussians,
        dataset=dataset,
        output_dir=args.output_dir,
        config=config,
        manifest_json_path=manifest_json,
    )

    # 7. Check for resume
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        trainer.resume(args.resume)

    previews = Path(args.output_dir) / "train_previews"
    initial_metrics_path = previews / "initial_metrics.json"
    if args.resume:
        initial_metrics = read_json_record(initial_metrics_path)
    else:
        initial_metrics = trainer.evaluate_train_view(
            0,
            render_path=previews / "step_000000000.png",
            reference_path=previews / "reference.png",
        )
        write_json_record(initial_metrics_path, initial_metrics)

    # 8. Start optimization run
    print(f"Starting optimization for {args.max_steps} iterations...")
    trainer.train(
        stop_step=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        save_checkpoints=should_save_checkpoints(args),
    )
    final_metrics = trainer.evaluate_train_view(
        0,
        render_path=previews / f"step_{args.max_steps:09d}.png",
    )
    report = write_convergence_report(
        Path(args.output_dir) / "convergence.json",
        initial_metrics,
        final_metrics,
    )
    print(
        "Convergence: "
        f"PSNR delta={report['psnr_delta_db']}, "
        f"SSIM delta={report['ssim_delta']:.6f}, "
        f"non_blank={report['final_render_non_blank']}"
    )
    if validation_dataset is not None:
        validation = evaluate_internal_validation(
            trainer,
            validation_dataset,
            LpipsBackend(backbone="alex", device=str(trainer.device)),
            Path(args.output_dir) / "validation_renders",
        )
        summary = read_json_record(Path(args.output_dir) / "summary.json")
        metric_records = [
            json.loads(line)
            for line in (Path(args.output_dir) / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        qualification_report = {
            "schema_version": 1,
            "scene_id": manifest.scene_id,
            "candidate_id": args.qualification_candidate,
            "step": args.max_steps,
            "image_count": validation["image_count"],
            "psnr_db_mean": validation["psnr_db_mean"],
            "ssim_mean": validation["ssim_mean"],
            "lpips_mean": validation["lpips_mean"],
            "valid_fraction_mean": validation["valid_fraction_mean"],
            "peak_gaussians": max(item["num_gaussians"] for item in metric_records),
            "max_vram_mb": summary["max_vram_mb"],
            "total_time_seconds": summary["total_time_seconds"],
            "config_sha256": trainer.config_hash,
            "holdout_sha256": split.manifest_sha256,
            "images": validation["images"],
        }
        save_qualification_report(
            qualification_report,
            Path(args.output_dir) / "qualification_report.json",
        )
        print(
            "Validation: "
            f"PSNR={validation['psnr_db_mean']:.4f}, "
            f"SSIM={validation['ssim_mean']:.6f}, "
            f"LPIPS={validation['lpips_mean']:.6f}"
        )
    print(f"Optimization run completed. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
