import hashlib
import json
import os
import platform
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from bts_nvs.cameras.poses import camera_center_from_world_to_camera
from bts_nvs.data.dataset import SceneDataset
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.loss import JointLoss
from bts_nvs.models.optimizer import setup_mean_scheduler, setup_optimizers
from bts_nvs.rendering.density_strategy import GsplatStrategy
from bts_nvs.rendering.gsplat_renderer import render_gaussians
from bts_nvs.training.checkpoint import load_checkpoint, save_checkpoint, set_rng_states


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _truncate_run_records(output_dir: Path, completed_step: int) -> None:
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.is_file():
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        retained = [
            record for record in records if record.get("step", -1) <= completed_step
        ]
        text = "".join(
            json.dumps(record, allow_nan=False) + "\n" for record in retained
        )
        _atomic_write_text(metrics_path, text)

    timing_path = output_dir / "timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if not isinstance(timing, dict):
            raise ValueError("timing.json must contain an object")
        retained_timing = {
            key: value for key, value in timing.items() if int(key) <= completed_step
        }
        _atomic_write_text(
            timing_path,
            json.dumps(retained_timing, indent=2, allow_nan=False),
        )


def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA-256 hash of a file's content."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_config_sha256(config: Dict[str, Any]) -> str:
    """Hash the exact optimization configuration deterministically."""
    try:
        payload = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("training config must be finite JSON data") from error
    return hashlib.sha256(payload).hexdigest()


def compute_manifest_sha256(manifest_json_path: Path) -> str:
    """Hash both manifest metadata and its referenced NumPy arrays."""
    manifest_path = Path(manifest_json_path)
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest JSON: {manifest_path}") from error
    arrays_name = metadata.get("arrays_file")
    if (
        not isinstance(arrays_name, str)
        or Path(arrays_name).is_absolute()
        or Path(arrays_name).name != arrays_name
    ):
        raise ValueError("manifest arrays_file must be a relative basename")
    arrays_path = manifest_path.parent / arrays_name
    if not arrays_path.is_file():
        raise FileNotFoundError(f"Manifest arrays file not found: {arrays_path}")

    digest = hashlib.sha256()
    for label, path in (("manifest.json", manifest_path), (arrays_name, arrays_path)):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8192):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_dataset_manifest(dataset: SceneDataset, manifest_path: Path) -> None:
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        arrays_path = manifest_path.parent / metadata["arrays_file"]
        with np.load(arrays_path, allow_pickle=False) as arrays:
            matches = (
                metadata.get("scene_id") == dataset.manifest.scene_id
                and np.array_equal(
                    arrays["train_world_to_camera"],
                    dataset.manifest.train_world_to_camera,
                )
                and np.array_equal(
                    arrays["normalization_transform"],
                    dataset.manifest.normalization_transform,
                )
                and np.array_equal(
                    arrays["inverse_normalization_transform"],
                    dataset.manifest.inverse_normalization_transform,
                )
            )
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError("dataset does not match manifest artifact") from error
    if not matches:
        raise ValueError("dataset does not match manifest artifact")


def _normalize_world_to_camera(
    world_to_camera: np.ndarray,
    normalization_transform: np.ndarray,
) -> np.ndarray:
    """Express a rigid raw-world camera pose in the normalized world domain."""
    pose = np.asarray(world_to_camera, dtype=np.float64)
    transform = np.asarray(normalization_transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("normalization transform must be finite with shape (4, 4)")
    center = camera_center_from_world_to_camera(pose)
    normalized_center = transform[:3, :3] @ center + transform[:3, 3]
    normalized = pose.copy()
    normalized[:3, 3] = -pose[:3, :3] @ normalized_center
    return normalized


class Trainer:
    """Optimization engine for per-scene Gaussian Splatting baseline B0."""

    def __init__(
        self,
        gaussians: GaussianParameters,
        dataset: SceneDataset,
        output_dir: str | Path,
        config: Dict[str, Any],
        manifest_json_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes output paths, saves environmental data, and builds optimizer group.

        Args:
            gaussians (GaussianParameters): Initialized Gaussian parameters container.
            dataset (SceneDataset): Ground truth training data container.
            output_dir (str | Path): Path to output workspace.
            config (Dict[str, Any]): Optimization parameters dict.
            manifest_json_path (str | Path): Path to manifest JSON.
            device (torch.device, optional): Specific CUDA/CPU execution target.
        """
        self.gaussians = gaussians
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.config = dict(config)
        self.manifest_json_path = Path(manifest_json_path)
        self.max_steps = self.config.get("max_steps", 30000)
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("config max_steps must be a positive integer")
        seed = self.config.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("config seed must be a non-negative integer")
        if len(self.dataset) == 0:
            raise ValueError("training dataset must contain at least one camera")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Determine device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gaussians.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # Calculate SHA256 of manifest
        if not self.manifest_json_path.is_file():
            raise FileNotFoundError(
                f"Manifest file not found: {self.manifest_json_path}"
            )
        self.manifest_hash = compute_manifest_sha256(self.manifest_json_path)
        self.config_hash = compute_config_sha256(self.config)
        _validate_dataset_manifest(self.dataset, self.manifest_json_path)

        # Setup workspaces
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_previews").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "validation_renders").mkdir(parents=True, exist_ok=True)

        # Output files
        # 1. config.yaml
        config_path = self.output_dir / "config.yaml"
        config_text = yaml.safe_dump(
            self.config,
            default_flow_style=False,
            sort_keys=True,
        )
        if config_path.exists():
            if config_path.read_text(encoding="utf-8") != config_text:
                raise ValueError("existing run config does not match requested config")
        else:
            config_path.write_text(config_text, encoding="utf-8")

        # 2. environment.json
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pytorch_version": torch.__version__,
            "gsplat_version": version("gsplat"),
            "numpy_version": version("numpy"),
            "opencv_version": cv2.__version__,
            "pillow_version": version("Pillow"),
            "pycolmap_version": version("pycolmap"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device": str(self.device),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else "CPU"
            ),
            "device_capability": (
                list(torch.cuda.get_device_capability(self.device))
                if self.device.type == "cuda"
                else None
            ),
        }
        environment_path = self.output_dir / "environment.json"
        if not environment_path.exists():
            environment_path.write_text(
                json.dumps(env_info, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        # 3. manifest_hash.json
        hash_path = self.output_dir / "manifest_hash.json"
        hash_record = {"manifest_hash": self.manifest_hash}
        if hash_path.exists():
            if json.loads(hash_path.read_text(encoding="utf-8")) != hash_record:
                raise ValueError(
                    "existing run manifest does not match requested manifest"
                )
        else:
            hash_path.write_text(
                json.dumps(hash_record, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        # Setup Optimizers
        self.optimizers = setup_optimizers(self.gaussians)

        # Setup LR Scheduler
        self.scheduler = setup_mean_scheduler(self.optimizers, max_steps=self.max_steps)

        # Setup Density Strategy matching workspace signature
        self.strategy = GsplatStrategy(
            self.gaussians,
            self.optimizers,
            prune_opa=self.config.get("prune_opa", 0.005),
            grow_grad2d=self.config.get("grow_grad2d", 0.0002),
            grow_scale3d=self.config.get("grow_scale3d", 0.01),
            refine_start_step=self.config.get("refine_start_step", 500),
            refine_stop_step=self.config.get("refine_stop_step", 15000),
            refine_every=self.config.get("refine_every", 100),
            reset_every=self.config.get("reset_every", 3000),
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

        # Setup Differentiable Loss Module
        self.loss_fn = JointLoss(
            lambda_dssim=self.config.get("lambda_dssim", 0.2),
            ssim_kernel_size=self.config.get("ssim_kernel_size", 11),
            ssim_sigma=self.config.get("ssim_sigma", 1.5),
            ssim_k1=self.config.get("ssim_k1", 0.01),
            ssim_k2=self.config.get("ssim_k2", 0.03),
            data_range=self.config.get("data_range", 1.0),
        ).to(self.device)

        # Start state
        self.start_step = 0
        self.active_sh_degree = 0

    @torch.no_grad()
    def evaluate_train_view(
        self,
        index: int,
        *,
        render_path: str | Path,
        reference_path: str | Path | None = None,
    ) -> dict[str, float | bool | None]:
        """Render one fixed train camera and report masked smoke metrics."""
        sample = self.dataset[index]
        if sample.distortion.model != "PINHOLE":
            raise ValueError("diagnostics require an undistorted PINHOLE sample")
        normalized_w2c = _normalize_world_to_camera(
            sample.world_to_camera,
            self.dataset.manifest.normalization_transform,
        )
        result = render_gaussians(
            gaussians=self.gaussians,
            viewmat=torch.from_numpy(normalized_w2c).to(self.device),
            intrinsics=sample.intrinsics,
            active_sh_degree=self.active_sh_degree,
            render_mode="RGB",
        )
        prediction = result.rgb.float()
        target = (
            torch.from_numpy(sample.image).to(device=self.device, dtype=torch.float32)
            / 255.0
        )
        mask = torch.from_numpy(sample.valid_mask).to(self.device)
        if (
            not torch.isfinite(prediction).all()
            or not torch.isfinite(result.alpha).all()
        ):
            raise FloatingPointError("non-finite diagnostic render")

        metric_prediction = prediction.clamp(0.0, 1.0)
        valid_prediction = metric_prediction[mask]
        valid_target = target[mask]
        mse = torch.mean((valid_prediction - valid_target).square()).item()
        psnr_db = None if mse == 0.0 else float(-10.0 * np.log10(mse))
        ssim = float(
            1.0
            - self.loss_fn.ssim_loss(metric_prediction, target, mask)
            .detach()
            .cpu()
            .item()
        )
        alpha = result.alpha[..., 0] if result.alpha.ndim == 3 else result.alpha
        alpha_coverage = float(((alpha > 1e-4) & mask).sum().item() / mask.sum().item())
        rgb_std = float(valid_prediction.std(unbiased=False).item())

        render_file = Path(render_path)
        render_file.parent.mkdir(parents=True, exist_ok=True)
        rendered_uint8 = metric_prediction.mul(255.0).round().byte().cpu().numpy()
        Image.fromarray(rendered_uint8).save(render_file)
        if reference_path is not None:
            reference_file = Path(reference_path)
            reference_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sample.image).save(reference_file)

        return {
            "psnr_db": psnr_db,
            "psnr_is_infinite": mse == 0.0,
            "ssim": ssim,
            "alpha_coverage": alpha_coverage,
            "rgb_std": rgb_std,
            "valid_fraction": float(mask.float().mean().item()),
        }

    def resume(self, checkpoint_path: str | Path) -> None:
        """Loads states from a checkpoint and validates SHA256 integrity.

        Args:
            checkpoint_path (str | Path): Target checkpoint file.
        """
        checkpoint_state = load_checkpoint(
            checkpoint_path,
            expected_manifest_hash=self.manifest_hash,
            expected_config_hash=self.config_hash,
        )

        self.start_step = checkpoint_state["step"]
        self.active_sh_degree = checkpoint_state["active_sh_degree"]

        # 1. Resize GaussianParameters module parameters to match target shapes
        for name in ["means", "scales", "quats", "opacities", "sh0", "shN"]:
            param_val = checkpoint_state["gaussians"][name]
            setattr(
                self.gaussians,
                name,
                torch.nn.Parameter(torch.zeros_like(param_val, device=self.device)),
            )

        # Load Gaussian parameters
        self.gaussians.load_state_dict(checkpoint_state["gaussians"])

        # 2. Re-create optimizers for resized parameters
        self.optimizers = setup_optimizers(self.gaussians)
        for name, opt in self.optimizers.items():
            opt.load_state_dict(checkpoint_state["optimizers"][name])

        # 3. Re-create scheduler tracking the newly reconstructed optimizer
        self.scheduler = setup_mean_scheduler(self.optimizers, max_steps=self.max_steps)
        self.scheduler.load_state_dict(checkpoint_state["scheduler"])

        # 4. Re-bind parameters and optimizers inside the density strategy
        self.strategy.params = self.gaussians.parameter_map()
        self.strategy.optimizers = self.optimizers

        # 5. Restore strategy state
        # Copy elements to match device
        loaded_state = checkpoint_state["strategy_state"]
        for k, v in loaded_state.items():
            if isinstance(v, torch.Tensor):
                loaded_state[k] = v.to(self.device)
        self.strategy_state = loaded_state

        # 6. Restore RNG state
        set_rng_states(checkpoint_state["rng_states"])

        # 7. Remove records written after the resumed checkpoint.
        _truncate_run_records(self.output_dir, self.start_step)

    def train(
        self,
        *,
        stop_step: int | None = None,
        checkpoint_every: int = 3000,
    ) -> None:
        """Runs the main optimization training loop.

        Args:
            stop_step (int, optional): Completed step at which this invocation stops.
            checkpoint_every (int): Frequency in steps to save checkpoints.
        """
        target_step = self.max_steps if stop_step is None else stop_step
        if (
            isinstance(target_step, bool)
            or not isinstance(target_step, int)
            or target_step <= self.start_step
        ):
            raise ValueError(
                "stop_step must be an integer greater than the current step"
            )
        if target_step > self.max_steps:
            raise ValueError("stop_step cannot exceed configured max_steps")
        if (
            isinstance(checkpoint_every, bool)
            or not isinstance(checkpoint_every, int)
            or checkpoint_every <= 0
        ):
            raise ValueError("checkpoint_every must be a positive integer")
        metrics_file = self.output_dir / "metrics.jsonl"
        timing_file = self.output_dir / "timing.json"
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        training_start_time = time.perf_counter()

        if timing_file.exists():
            timing_records = json.loads(timing_file.read_text(encoding="utf-8"))
            if not isinstance(timing_records, dict):
                raise ValueError("timing.json must contain an object")
        else:
            timing_records = {}

        for step_index in range(self.start_step, target_step):
            completed_step = step_index + 1
            step_start_time = time.perf_counter()

            # Spherical harmonics degree scheduling
            active_sh_degree = min(step_index // 1000, 3)
            self.active_sh_degree = active_sh_degree

            # Select uniform random camera sample
            sample_idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[sample_idx]
            if sample.distortion.model != "PINHOLE":
                raise ValueError(
                    "training requires an undistorted PINHOLE dataset sample"
                )

            # Move data tensors to device
            rgb_gt = torch.from_numpy(sample.image).to(self.device)
            mask = torch.from_numpy(sample.valid_mask).to(self.device)
            normalized_w2c = _normalize_world_to_camera(
                sample.world_to_camera,
                self.dataset.manifest.normalization_transform,
            )
            viewmat = torch.from_numpy(normalized_w2c).to(self.device)

            # 1. Forward Pass
            t_fwd_start = time.perf_counter()
            result = render_gaussians(
                gaussians=self.gaussians,
                viewmat=viewmat,
                intrinsics=sample.intrinsics,
                active_sh_degree=active_sh_degree,
                render_mode="RGB",
            )
            t_fwd = time.perf_counter() - t_fwd_start

            # 2. Strategy pre-backward step (step + 1 because strategy step is 1-based)
            self.strategy.step_pre_backward(
                state=self.strategy_state,
                step=completed_step,
                info=result.info,
            )

            # 3. Compute loss
            loss = self.loss_fn(result.rgb, rgb_gt, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at completed step {completed_step}"
                )

            # 4. Zero grad
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            # 5. Backward Pass
            t_bwd_start = time.perf_counter()
            loss.backward()
            for name, parameter in self.gaussians.named_parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(
                        f"non-finite gradient for {name} at completed step "
                        f"{completed_step}"
                    )
            means2d = result.info.get("means2d")
            if (
                isinstance(means2d, torch.Tensor)
                and (means2d.is_leaf or means2d.retains_grad)
                and means2d.grad is not None
                and not torch.isfinite(means2d.grad).all()
            ):
                raise FloatingPointError(
                    "non-finite projected means gradient at completed step "
                    f"{completed_step}"
                )
            t_bwd = time.perf_counter() - t_bwd_start

            # 6. Strategy post-backward step (updates params/optimizers in-place)
            t_strategy_start = time.perf_counter()
            self.strategy.step_post_backward(
                state=self.strategy_state,
                step=completed_step,
                info=result.info,
                packed=True,
            )
            for name, parameter in self.gaussians.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"non-finite Gaussian parameter {name} after density "
                        f"control at completed step {completed_step}"
                    )
            t_strategy = time.perf_counter() - t_strategy_start

            # 7. Optimizer step and scheduler decay
            t_opt_start = time.perf_counter()
            for opt in self.optimizers.values():
                opt.step()
            self.scheduler.step()
            for name, parameter in self.gaussians.named_parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError(
                        f"non-finite Gaussian parameter {name} after completed "
                        f"step {completed_step}"
                    )
            t_opt = time.perf_counter() - t_opt_start

            step_duration = time.perf_counter() - step_start_time

            # Record timing
            timing_records[str(completed_step)] = {
                "forward": t_fwd,
                "backward": t_bwd,
                "strategy": t_strategy,
                "optimizer": t_opt,
                "total": step_duration,
            }

            # Log metrics JSONL
            metrics = {
                "step": completed_step,
                "loss": loss.item(),
                "num_gaussians": self.gaussians.num_gaussians,
                "lr_means": self.scheduler.get_last_lr()[0],
            }
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, allow_nan=False) + "\n")

            # Checkpoint
            if completed_step % checkpoint_every == 0 or completed_step == target_step:
                checkpoint_name = f"step_{completed_step:09d}.pt"
                checkpoint_path = self.output_dir / "checkpoints" / checkpoint_name
                save_checkpoint(
                    path=checkpoint_path,
                    step=completed_step,
                    gaussians_state_dict=self.gaussians.state_dict(),
                    optimizers_state_dict={
                        name: opt.state_dict() for name, opt in self.optimizers.items()
                    },
                    scheduler_state_dict=self.scheduler.state_dict(),
                    strategy_state=self.strategy_state,
                    active_sh_degree=self.active_sh_degree,
                    manifest_hash=self.manifest_hash,
                    config_hash=self.config_hash,
                )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_time_seconds = time.perf_counter() - training_start_time

        # Write final outputs
        # 1. timing.json
        with open(timing_file, "w", encoding="utf-8") as f:
            json.dump(timing_records, f, indent=2, allow_nan=False)

        # 2. summary.json
        summary = {
            "total_steps": target_step,
            "total_time_seconds": total_time_seconds,
            "final_loss": loss.item(),
            "final_num_gaussians": self.gaussians.num_gaussians,
            "max_vram_mb": (
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                if self.device.type == "cuda"
                else 0.0
            ),
        }
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
