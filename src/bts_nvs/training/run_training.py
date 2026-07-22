import argparse
import json
import math
import subprocess
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
from bts_nvs.evaluation.detail_metrics import evaluate_detail_directory
from bts_nvs.evaluation.experiment_report import (
    build_experiment_report,
    save_experiment_report,
)
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.initialization import initialize_from_manifest
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.evaluation.pose_strata import build_pose_strata, save_pose_strata
from bts_nvs.experiments.candidates import candidate_training_overrides
from bts_nvs.experiments.experiment import (
    COHORT_SCENE_IDS,
    Experiment,
    ExperimentStage,
)
from bts_nvs.rendering.density_strategy import GsplatStrategy
from bts_nvs.rendering.gsplat_renderer import render_gaussians
from bts_nvs.training.checkpoint import load_checkpoint
from bts_nvs.training.trainer import Trainer
from bts_nvs.training.qualification import (
    CALIBRATION_SCENES,
    build_full_length_report,
    evaluate_internal_validation,
    save_full_length_report,
    save_qualification_report,
)
from bts_nvs.evaluation.metrics import LpipsBackend
from bts_nvs.training.resources import (
    linux_memory_status,
    require_cache_capacity,
    require_no_swap,
)
from bts_nvs.training.precision import TrainingPrecision


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
        "--stop_step",
        type=int,
        default=None,
        help="Optional execution stop distinct from the configured schedule horizon.",
    )
    parser.add_argument(
        "--candidate_id",
        type=str,
        default=None,
        help="Registered generic experiment candidate identity.",
    )
    parser.add_argument(
        "--experiment_stage",
        choices=tuple(stage.value for stage in ExperimentStage),
        default=None,
        help="Generic experiment stage.",
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
    parser.add_argument(
        "--full_length_qualification",
        action="store_true",
        help="Run the locked HCM0181 Phase 4.5 30k internal-holdout dry run.",
    )
    parser.add_argument(
        "--optimizer_backend",
        choices=("adam", "adam-fused"),
        default="adam",
        help="Adam implementation; fused is CUDA-only and never falls back.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp-fp16"),
        default="fp32",
        help="Training precision; AMP keeps Gaussian parameters in FP32.",
    )
    parser.add_argument(
        "--backend_qualification",
        action="store_true",
        help="Run the fixed Phase 4.6 1000-step HCM0181 backend qualification.",
    )
    parser.add_argument(
        "--rolling_checkpoint",
        action="store_true",
        help="Atomically reuse checkpoints/recovery.pt instead of numbered files.",
    )
    return parser.parse_args()


def validate_resize_factor(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("resize_factor must be a positive integer")


def validate_training_backend(args) -> None:
    if args.precision == "amp-fp16" and args.optimizer_backend != "adam-fused":
        raise ValueError("amp-fp16 requires adam-fused")


def validate_backend_qualification_args(args) -> None:
    if not args.backend_qualification:
        return
    if (
        args.max_steps != 1000
        or args.resize_factor != 1
        or args.seed != 0
        or not args.cache_images
        or not args.pinned_transfer
        or args.resume is not None
        or args.profile_input
        or args.qualification_candidate is not None
        or args.full_length_qualification
    ):
        raise ValueError(
            "backend qualification requires a fresh factor-1, seed-0, "
            "1000-step run with cached images and pinned transfer"
        )


def validate_backend_qualification_scene(scene_id: str, args) -> None:
    if args.backend_qualification and scene_id != "HCM0181":
        raise ValueError("backend qualification requires HCM0181")


def validate_rolling_checkpoint_args(args) -> None:
    if not args.rolling_checkpoint:
        return
    if (
        args.profile_input
        or args.backend_qualification
        or args.qualification_candidate is not None
    ):
        raise ValueError(
            "rolling checkpoint is only supported for ordinary or full-length runs"
        )


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


def validate_full_length_args(args) -> None:
    if not args.full_length_qualification:
        return
    if args.qualification_candidate is not None:
        raise ValueError("full-length qualification is mutually exclusive with candidate mode")
    if (
        args.max_steps != 30_000
        or args.resize_factor != 1
        or args.checkpoint_every != 3_000
        or args.seed != 0
        or not args.cache_images
        or not args.pinned_transfer
    ):
        raise ValueError(
            "full-length qualification requires factor-1, seed-0, 30000 steps, "
            "3000-step checkpoints, cached images, and pinned transfer"
        )
    if args.resume is not None:
        expected = Path(args.output_dir) / "checkpoints" / "recovery.pt"
        if Path(args.resume).resolve() != expected.resolve():
            raise ValueError("full-length resume must use output checkpoints/recovery.pt")


def _generic_experiment(args, scene_id: str) -> Experiment:
    stage = ExperimentStage(args.experiment_stage)
    authorization = {}
    if stage is ExperimentStage.CONFIRM:
        authorization["authorized_scene_winner"] = args.candidate_id
    elif stage is ExperimentStage.PRODUCTION:
        authorization["authorized_cohort_candidate"] = args.candidate_id
    return Experiment(
        stage=stage,
        scene_id=scene_id,
        candidate_id=args.candidate_id,
        **authorization,
    )


def validate_generic_experiment_args(args) -> None:
    candidate_id = args.candidate_id
    stage_name = args.experiment_stage
    if candidate_id is None and stage_name is None:
        if args.stop_step is not None:
            raise ValueError("stop_step is only supported for a generic experiment")
        return
    if candidate_id is None or stage_name is None:
        raise ValueError("candidate_id and experiment_stage must be supplied together")
    if (
        args.profile_input
        or args.qualification_candidate is not None
        or args.full_length_qualification
        or args.backend_qualification
    ):
        raise ValueError("generic experiment identity cannot be combined with legacy modes")

    experiment = _generic_experiment(args, COHORT_SCENE_IDS[0])
    common_runtime = (
        args.resize_factor == 1
        and args.seed == 0
        and args.cache_images
        and args.pinned_transfer
    )
    if experiment.stage in (ExperimentStage.REFERENCE, ExperimentStage.SCREEN):
        if (
            not common_runtime
            or args.max_steps != 7_000
            or args.stop_step not in (None, 7_000)
            or not args.internal_holdout
            or args.resume is not None
            or args.rolling_checkpoint
        ):
            raise ValueError(
                f"{experiment.stage.value} requires a fresh factor-1, seed-0, "
                "7000-step internal-holdout run with cached images, pinned "
                "transfer, and no checkpoints"
            )
        return

    expected_recovery = Path(args.output_dir) / "checkpoints" / "recovery.pt"
    recovery_is_valid = args.resume is None or (
        Path(args.resume).resolve() == expected_recovery.resolve()
    )
    rolling_runtime = (
        common_runtime
        and args.max_steps == 30_000
        and args.checkpoint_every == 3_000
        and args.rolling_checkpoint
        and recovery_is_valid
    )
    if experiment.stage is ExperimentStage.CONFIRM:
        if (
            not rolling_runtime
            or args.stop_step not in (15_000, 30_000)
            or not args.internal_holdout
        ):
            raise ValueError(
                "confirm requires a factor-1, seed-0, 30000-step schedule "
                "stopped at 15000 or 30000 with internal holdout, cached "
                "images, pinned transfer, and 3000-step rolling recovery"
            )
        return
    if (
        not rolling_runtime
        or args.stop_step != 30_000
        or args.internal_holdout
    ):
        raise ValueError(
            "production requires a factor-1, seed-0, 30000-step run without "
            "internal holdout, with cached images, pinned transfer, and "
            "3000-step rolling recovery"
        )


def validate_generic_experiment_scene(scene_id: str, args) -> None:
    if args.candidate_id is not None:
        _generic_experiment(args, scene_id)


def training_target_step(args) -> int:
    return args.stop_step if args.stop_step is not None else args.max_steps


def validate_full_length_scene(scene_id: str, args) -> None:
    if args.full_length_qualification and scene_id != "HCM0181":
        raise ValueError("full-length qualification requires HCM0181")


def read_clean_git_commit(repo_root: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("full-length qualification rejects a dirty tracked worktree")
    if len(commit) != 40:
        raise RuntimeError("git did not return a full commit SHA")
    return commit


def validate_recovery_checkpoint(
    path: Path,
    manifest_hash: str,
    config_hash: str,
    expected_step: int,
) -> None:
    state = load_checkpoint(
        path,
        expected_manifest_hash=manifest_hash,
        expected_config_hash=config_hash,
    )
    required = {
        "step",
        "gaussians",
        "optimizers",
        "scheduler",
        "strategy_state",
        "active_sh_degree",
        "rng_states",
        "manifest_hash",
        "config_hash",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"recovery checkpoint is missing: {', '.join(missing)}")
    if state["step"] != expected_step:
        raise ValueError(
            f"recovery checkpoint ended at step {state['step']}, "
            f"expected {expected_step}"
        )


def optimization_required(start_step: int, target_step: int) -> bool:
    if start_step > target_step:
        raise ValueError("checkpoint step is beyond the requested target")
    return start_step < target_step


def internal_holdout_enabled(args) -> bool:
    return bool(
        args.internal_holdout
        or args.profile_input
        or args.qualification_candidate is not None
        or args.full_length_qualification
        or args.backend_qualification
    )


def should_save_checkpoints(args) -> bool:
    if args.candidate_id is not None:
        return args.experiment_stage in (
            ExperimentStage.CONFIRM.value,
            ExperimentStage.PRODUCTION.value,
        )
    return (
        not args.profile_input
        and args.qualification_candidate is None
        and not args.backend_qualification
    )


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


def run_cuda_preflight(
    optimizer_backend: str = "adam",
    precision_mode: str = "fp32",
) -> None:
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
        optimizers = setup_optimizers(gaussians, backend=optimizer_backend)
        precision = TrainingPrecision(precision_mode, device)
        strategy = GsplatStrategy(gaussians, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=1.0)
        with precision.autocast():
            result = render_gaussians(
                gaussians=gaussians,
                viewmat=torch.eye(4, device=device),
                intrinsics=CameraIntrinsics(32, 32, 24.0, 24.0, 16.0, 16.0),
                active_sh_degree=0,
            )
            loss = result.rgb.square().mean()
        strategy.step_pre_backward(strategy_state, step=1, info=result.info)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        means2d = result.info.get("means2d")
        if not isinstance(means2d, torch.Tensor):
            raise RuntimeError("gsplat preflight did not return projected means")
        precision.backward_and_unscale(loss, optimizers, means2d)
        validate_preflight_gradients(gaussians.parameters())
        if means2d.grad is None or not torch.isfinite(means2d.grad).all():
            raise RuntimeError(
                "gsplat preflight did not produce a finite projected gradient"
            )
        strategy.step_post_backward(
            strategy_state,
            step=1,
            info=result.info,
            packed=True,
        )
        precision.step(optimizers)
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
        "full_length_qualification": bool(args.full_length_qualification),
        "optimizer_backend": args.optimizer_backend,
        "precision": args.precision,
        "backend_qualification": bool(args.backend_qualification),
        "rolling_checkpoint": bool(args.rolling_checkpoint),
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
    if args.candidate_id is not None:
        config["experiment_stage"] = args.experiment_stage
        config.update(candidate_training_overrides(args.candidate_id))
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


def _read_metric_records(path: Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required metric record does not exist: {source}")
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"metric record must contain JSON objects: {source}")
    return records


def build_experiment_resource_summary(output_dir: Path) -> dict[str, float | int]:
    output = Path(output_dir)
    summary = read_json_record(output / "summary.json")
    metric_records = _read_metric_records(output / "metrics.jsonl")
    return {
        "total_time_seconds": summary["total_time_seconds"],
        "max_vram_mb": summary["max_vram_mb"],
        "peak_gaussians": max(
            int(record["num_gaussians"]) for record in metric_records
        ),
        "final_num_gaussians": int(summary["final_num_gaussians"]),
    }


def _qualification_report(
    *,
    scene_id: str,
    candidate_id: str,
    step: int,
    validation: dict,
    resources: dict[str, float | int],
    config_sha256: str,
    holdout_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "candidate_id": candidate_id,
        "step": step,
        "image_count": validation["image_count"],
        "psnr_db_mean": validation["psnr_db_mean"],
        "ssim_mean": validation["ssim_mean"],
        "lpips_mean": validation["lpips_mean"],
        "valid_fraction_mean": validation["valid_fraction_mean"],
        "peak_gaussians": resources["peak_gaussians"],
        "max_vram_mb": resources["max_vram_mb"],
        "total_time_seconds": resources["total_time_seconds"],
        "config_sha256": config_sha256,
        "holdout_sha256": holdout_sha256,
        "images": validation["images"],
    }


def generate_generic_experiment_reports(
    *,
    args,
    trainer,
    manifest,
    split: HoldoutSplit,
    validation_dataset,
) -> dict[str, object]:
    output = Path(args.output_dir)
    render_dir = output / "validation_renders"
    validation = evaluate_internal_validation(
        trainer,
        validation_dataset,
        LpipsBackend(backbone="alex", device=str(trainer.device)),
        render_dir,
    )
    resources = build_experiment_resource_summary(output)
    qualification = _qualification_report(
        scene_id=manifest.scene_id,
        candidate_id=args.candidate_id,
        step=training_target_step(args),
        validation=validation,
        resources=resources,
        config_sha256=trainer.config_hash,
        holdout_sha256=split.manifest_sha256,
    )
    write_json_record(output / "qualification_report.json", qualification)

    detail = evaluate_detail_directory(validation_dataset, render_dir)
    write_json_record(output / "detail_metrics.json", detail)
    pose_strata = build_pose_strata(manifest, split)
    save_pose_strata(pose_strata, output / "pose_strata.json")
    experiment = build_experiment_report(
        scene_id=manifest.scene_id,
        candidate_id=args.candidate_id,
        step=training_target_step(args),
        config_sha256=trainer.config_hash,
        manifest_sha256=trainer.manifest_hash,
        holdout_sha256=split.manifest_sha256,
        full_frame_report=validation,
        detail_report=detail,
        pose_strata_report=pose_strata,
        resource_summary=resources,
    )
    save_experiment_report(experiment, output / "experiment_report.json")
    return experiment


def main():
    args = parse_args()
    validate_resize_factor(args.resize_factor)
    validate_training_backend(args)
    validate_backend_qualification_args(args)
    validate_rolling_checkpoint_args(args)
    validate_profile_args(args)
    validate_qualification_args(args)
    validate_full_length_args(args)
    validate_generic_experiment_args(args)
    validate_output_directory(Path(args.output_dir), args.resume)
    source_commit = (
        read_clean_git_commit(Path(__file__).resolve().parents[3])
        if args.full_length_qualification
        else None
    )
    run_cuda_preflight(args.optimizer_backend, args.precision)

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
    validate_full_length_scene(manifest.scene_id, args)
    validate_backend_qualification_scene(manifest.scene_id, args)
    validate_generic_experiment_scene(manifest.scene_id, args)
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
        if (
            args.qualification_candidate is not None
            or args.full_length_qualification
            or args.candidate_id is not None
        )
        and split is not None
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

    lpips_backend = None
    initial_validation = None
    if args.full_length_qualification:
        lpips_backend = LpipsBackend(backbone="alex", device=str(trainer.device))
        initial_validation_path = Path(args.output_dir) / "initial_validation.json"
        if args.resume:
            initial_validation = read_json_record(initial_validation_path)
        else:
            assert validation_dataset is not None
            initial_validation = evaluate_internal_validation(
                trainer, validation_dataset, lpips_backend, None
            )
            write_json_record(initial_validation_path, initial_validation)

    # 8. Start optimization run
    target_step = training_target_step(args)
    if optimization_required(trainer.start_step, target_step):
        print(f"Starting optimization through step {target_step}...")
        trainer.train(
            stop_step=target_step,
            checkpoint_every=args.checkpoint_every,
            save_checkpoints=should_save_checkpoints(args),
            rolling_checkpoint=(
                args.rolling_checkpoint or args.full_length_qualification
            ),
        )
    else:
        print(f"Checkpoint is already at step {target_step}; finalizing artifacts...")
    final_metrics = trainer.evaluate_train_view(
        0,
        render_path=previews / f"step_{target_step:09d}.png",
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
    if args.full_length_qualification:
        assert validation_dataset is not None
        assert lpips_backend is not None
        assert initial_validation is not None
        final_train = evaluate_internal_validation(
            trainer, dataset, lpips_backend, None
        )
        final_validation = evaluate_internal_validation(
            trainer,
            validation_dataset,
            lpips_backend,
            Path(args.output_dir) / "validation_renders",
        )
        summary = read_json_record(Path(args.output_dir) / "summary.json")
        timing_records = read_json_record(Path(args.output_dir) / "timing.json")
        convergence = read_json_record(Path(args.output_dir) / "convergence.json")
        recovery_path = Path(args.output_dir) / "checkpoints" / "recovery.pt"
        validate_recovery_checkpoint(
            recovery_path,
            trainer.manifest_hash,
            trainer.config_hash,
            args.max_steps,
        )
        metric_records = [
            json.loads(line)
            for line in (Path(args.output_dir) / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        full_length_report = build_full_length_report(
            scene_id=manifest.scene_id,
            git_commit=source_commit,
            initial_validation=initial_validation,
            final_train=final_train,
            final_validation=final_validation,
            summary=summary,
            metric_records=metric_records,
            timing_records=timing_records,
            convergence=convergence,
        )
        save_full_length_report(
            full_length_report,
            Path(args.output_dir) / "full_length_report.json",
        )
        print(
            "Full-length validation: "
            f"PSNR={final_validation['psnr_db_mean']:.4f}, "
            f"SSIM={final_validation['ssim_mean']:.6f}, "
            f"LPIPS={final_validation['lpips_mean']:.6f}, "
            f"automated_gates={full_length_report['automated_gates_passed']}"
        )
    elif args.candidate_id is not None and validation_dataset is not None:
        assert split is not None
        experiment_report = generate_generic_experiment_reports(
            args=args,
            trainer=trainer,
            manifest=manifest,
            split=split,
            validation_dataset=validation_dataset,
        )
        print(
            "Validation: "
            f"score50={experiment_report['overall']['score50']:.6f}, "
            f"step={target_step}"
        )
    elif validation_dataset is not None:
        validation = evaluate_internal_validation(
            trainer,
            validation_dataset,
            LpipsBackend(backbone="alex", device=str(trainer.device)),
            Path(args.output_dir) / "validation_renders",
        )
        assert split is not None
        resources = build_experiment_resource_summary(Path(args.output_dir))
        qualification_report = _qualification_report(
            scene_id=manifest.scene_id,
            candidate_id=args.qualification_candidate,
            step=args.max_steps,
            validation=validation,
            resources=resources,
            config_sha256=trainer.config_hash,
            holdout_sha256=split.manifest_sha256,
        )
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
