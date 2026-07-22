from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import bts_nvs.rendering.density_strategy as density_strategy
import bts_nvs.training.trainer as trainer_module
from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.dataset import CameraSample
from bts_nvs.experiments.candidates import candidate_training_overrides
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.rendering.render_result import RenderResult
from bts_nvs.training.checkpoint import load_checkpoint, save_checkpoint
from bts_nvs.training.trainer import (
    Trainer,
    compute_config_sha256,
    compute_file_sha256,
    compute_manifest_sha256,
)


class _FakeDefaultStrategy:
    def __init__(self, **kwargs) -> None:
        self.config = kwargs

    def check_sanity(self, params, optimizers) -> None:
        assert params.keys() == optimizers.keys()

    def initialize_state(self, scene_scale: float = 1.0):
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info) -> None:
        return None

    def step_post_backward(
        self, params, optimizers, state, step, info, packed=False
    ) -> None:
        return None


class _MockDataset:
    def __init__(self, *, distortion: CameraDistortion | None = None) -> None:
        scale = 0.1
        center = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] *= scale
        transform[:3, 3] = -scale * center
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] /= scale
        inverse[:3, 3] = center
        self.manifest = SimpleNamespace(
            scene_id="mock_scene",
            normalization_transform=transform,
            inverse_normalization_transform=inverse,
            train_world_to_camera=np.stack(
                [self._raw_pose(10.0), self._raw_pose(12.0)]
            ),
        )
        self.distortion = distortion or CameraDistortion("PINHOLE", ())
        self.sampled_indices: list[int] = []

    @staticmethod
    def _raw_pose(center_x: float) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = -center_x
        return pose

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> CameraSample:
        self.sampled_indices.append(index)
        image = np.full((16, 16, 3), index * 255, dtype=np.uint8)
        return CameraSample(
            image=image,
            world_to_camera=self.manifest.train_world_to_camera[index],
            intrinsics=CameraIntrinsics(16, 16, 8.0, 8.0, 8.0, 8.0),
            distortion=self.distortion,
            valid_mask=np.ones((16, 16), dtype=bool),
            image_name=f"img_{index}.png",
        )


@pytest.fixture(autouse=True)
def fake_density_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(density_strategy, "DefaultStrategy", _FakeDefaultStrategy)


@pytest.fixture(autouse=True)
def isolate_checkpoint_tests_from_host_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trainer_module, "require_checkpoint_path_capacity", lambda *args: None
    )


@pytest.fixture(autouse=True)
def isolate_trainer_tests_from_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "version",
        lambda package: f"test-{package}-version",
    )


@pytest.fixture
def manifest_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "manifest"
    artifact.mkdir()
    arrays_path = artifact / "arrays.npz"
    dataset = _MockDataset()
    np.savez(
        arrays_path,
        train_world_to_camera=dataset.manifest.train_world_to_camera,
        normalization_transform=dataset.manifest.normalization_transform,
        inverse_normalization_transform=dataset.manifest.inverse_normalization_transform,
    )
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": "mock_scene",
                "arrays_file": "arrays.npz",
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _gaussians() -> GaussianParameters:
    return GaussianParameters(
        means=torch.zeros((5, 3)),
        scales=torch.zeros((5, 3)),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 5),
        opacities=torch.zeros(5),
        sh0=torch.zeros((5, 1, 3)),
        shN=torch.zeros((5, 15, 3)),
    )


def _config(max_steps: int = 4) -> dict[str, int | float]:
    return {
        "max_steps": max_steps,
        "seed": 42,
        "refine_every": 2,
        "reset_every": 5,
    }


def test_trainer_rejects_fused_adam_on_cpu(
    tmp_path: Path,
    manifest_artifact: Path,
) -> None:
    config = _config()
    config["optimizer_backend"] = "adam-fused"

    with pytest.raises(ValueError, match="adam-fused requires CUDA"):
        _trainer(tmp_path / "run", manifest_artifact, _MockDataset(), config=config)


def _differentiable_render(
    gaussians,
    viewmat,
    intrinsics,
    active_sh_degree,
    backgrounds=None,
    render_mode="RGB",
    absgrad=False,
    rasterize_mode="classic",
):
    value = torch.sigmoid(gaussians.sh0.mean())
    rgb = value.expand(intrinsics.height, intrinsics.width, 3)
    alpha = torch.ones((*rgb.shape[:2], 1), device=rgb.device)
    info = {
        "means2d": gaussians.means[:, :2],
        "radii": torch.ones(
            gaussians.num_gaussians, dtype=torch.int32, device=rgb.device
        ),
    }
    return RenderResult(rgb, alpha, None, info)


def _trainer(
    tmp_path: Path,
    manifest_artifact: Path,
    dataset: _MockDataset,
    *,
    config: dict | None = None,
) -> Trainer:
    return Trainer(
        gaussians=_gaussians(),
        dataset=dataset,
        output_dir=tmp_path,
        config=config or _config(),
        manifest_json_path=manifest_artifact,
        device=torch.device("cpu"),
    )


def test_trainer_normalizes_raw_camera_pose_before_render(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def capture_render(*args, **kwargs):
        captured["viewmat"] = kwargs["viewmat"].detach().cpu()
        return _differentiable_render(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "render_gaussians", capture_render)
    dataset = _MockDataset()
    trainer = _trainer(tmp_path / "run", manifest_artifact, dataset)
    trainer.train(stop_step=1, checkpoint_every=1)

    expected = torch.eye(4, dtype=torch.float64)
    expected[0, 3] = -0.2 * dataset.sampled_indices[0]
    torch.testing.assert_close(captured["viewmat"], expected)


@pytest.mark.parametrize(
    "candidate_id,expected_absgrad,expected_grow_grad2d",
    [
        ("B0-reference", False, 0.0002),
        ("E1-density-absgrad-t04-v1", True, 0.0004),
    ],
)
def test_trainer_forwards_candidate_density_settings(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_id: str,
    expected_absgrad: bool,
    expected_grow_grad2d: float,
) -> None:
    captured: dict[str, object] = {}

    def capture_render(*args, **kwargs):
        if "absgrad" in kwargs:
            captured["render_absgrad"] = kwargs.pop("absgrad")
            captured["rasterize_mode"] = kwargs.pop("rasterize_mode")
        return _differentiable_render(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "render_gaussians", capture_render)
    config = _config(max_steps=1)
    config.update(candidate_training_overrides(candidate_id))
    trainer = _trainer(
        tmp_path / candidate_id,
        manifest_artifact,
        _MockDataset(),
        config=config,
    )

    trainer.train(stop_step=1, checkpoint_every=1)

    assert captured["render_absgrad"] is expected_absgrad
    assert captured["rasterize_mode"] == "classic"
    assert trainer.strategy.backend.config["absgrad"] is expected_absgrad
    assert trainer.strategy.backend.config["grow_grad2d"] == pytest.approx(
        expected_grow_grad2d
    )


def test_trainer_rejects_distorted_training_sample(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    dataset = _MockDataset(distortion=CameraDistortion("SIMPLE_RADIAL", (0.01,)))
    trainer = _trainer(tmp_path / "run", manifest_artifact, dataset)

    with pytest.raises(ValueError, match="undistorted PINHOLE"):
        trainer.train(stop_step=1, checkpoint_every=1)


def test_trainer_moves_loss_module_to_training_device(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moved_to: list[torch.device] = []

    class _SpyLoss(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()

        def to(self, device):
            moved_to.append(device)
            return self

    monkeypatch.setattr(trainer_module, "JointLoss", _SpyLoss)
    _trainer(tmp_path / "run", manifest_artifact, _MockDataset())

    assert moved_to == [torch.device("cpu")]


def test_environment_snapshot_records_rendering_dependencies(
    tmp_path: Path,
    manifest_artifact: Path,
) -> None:
    _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    environment = json.loads(
        (tmp_path / "run" / "environment.json").read_text(encoding="utf-8")
    )

    for key in (
        "gsplat_version",
        "numpy_version",
        "opencv_version",
        "pillow_version",
        "pycolmap_version",
    ):
        assert environment[key]


def test_environment_snapshot_does_not_require_gui_opencv_distribution(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_version = trainer_module.version

    def reject_gui_opencv(package: str) -> str:
        if package == "opencv-python":
            raise AssertionError("GUI OpenCV distribution must not be queried")
        return real_version(package)

    monkeypatch.setattr(trainer_module, "version", reject_gui_opencv)
    _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    environment = json.loads(
        (tmp_path / "run" / "environment.json").read_text(encoding="utf-8")
    )

    assert environment["opencv_version"] == cv2.__version__


def test_train_view_diagnostic_writes_preview_and_masked_metrics(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    render_path = tmp_path / "preview" / "initial.png"
    reference_path = tmp_path / "preview" / "reference.png"

    metrics = trainer.evaluate_train_view(
        0,
        render_path=render_path,
        reference_path=reference_path,
    )

    assert render_path.is_file()
    assert reference_path.is_file()
    assert metrics["psnr_db"] is not None
    assert -1.0 <= metrics["ssim"] <= 1.0
    assert 0.0 <= metrics["alpha_coverage"] <= 1.0
    assert metrics["valid_fraction"] == 1.0


def test_train_view_diagnostic_metrics_match_clamped_preview(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overbright_render(**kwargs) -> RenderResult:
        rgb = torch.full((16, 16, 3), 2.0)
        alpha = torch.ones((16, 16, 1))
        return RenderResult(rgb=rgb, alpha=alpha, info={})

    monkeypatch.setattr(trainer_module, "render_gaussians", overbright_render)
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())

    metrics = trainer.evaluate_train_view(
        1,
        render_path=tmp_path / "preview.png",
    )

    assert metrics["psnr_is_infinite"] is True
    assert metrics["psnr_db"] is None


def test_fresh_trainers_use_config_seed_for_camera_sampling(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    dataset_a = _MockDataset()
    dataset_b = _MockDataset()

    _trainer(tmp_path / "run_a", manifest_artifact, dataset_a).train(
        stop_step=4, checkpoint_every=4
    )
    _trainer(tmp_path / "run_b", manifest_artifact, dataset_b).train(
        stop_step=4, checkpoint_every=4
    )

    assert dataset_a.sampled_indices == dataset_b.sampled_indices


def test_trainer_uses_completed_step_numbers_in_metrics(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    trainer.train(stop_step=2, checkpoint_every=2)

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]


def test_trainer_records_sample_and_input_timings(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    config = _config(max_steps=1)
    config["pinned_transfer"] = True
    dataset = _MockDataset()
    trainer = _trainer(tmp_path / "run", manifest_artifact, dataset, config=config)

    trainer.train(stop_step=1, checkpoint_every=1)

    timing = json.loads((tmp_path / "run" / "timing.json").read_text())["1"]
    metric = json.loads((tmp_path / "run" / "metrics.jsonl").read_text())
    assert timing["data"] >= 0.0
    assert timing["transfer"] >= 0.0
    assert metric["sample_index"] == dataset.sampled_indices[0]
    assert trainer.input_pipeline.slot_count == 0


def test_summary_records_total_wall_time_and_peak_vram(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    trainer.train(stop_step=1, checkpoint_every=1)

    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["total_time_seconds"] >= 0.0
    assert summary["max_vram_mb"] == 0.0


@pytest.mark.parametrize("checkpoint_every", [0, -1, True])
def test_trainer_rejects_invalid_checkpoint_interval(
    tmp_path: Path,
    manifest_artifact: Path,
    checkpoint_every,
) -> None:
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    with pytest.raises(ValueError, match="checkpoint_every"):
        trainer.train(stop_step=1, checkpoint_every=checkpoint_every)


def test_trainer_rejects_stop_beyond_configured_horizon(
    tmp_path: Path,
    manifest_artifact: Path,
) -> None:
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    with pytest.raises(ValueError, match="configured max_steps"):
        trainer.train(stop_step=5, checkpoint_every=1)


def test_trainer_rejects_empty_dataset(
    tmp_path: Path,
    manifest_artifact: Path,
) -> None:
    class _EmptyDataset(_MockDataset):
        def __len__(self) -> int:
            return 0

    dataset = _EmptyDataset()
    with pytest.raises(ValueError, match="dataset must contain"):
        _trainer(tmp_path / "run", manifest_artifact, dataset)


def test_non_finite_loss_fails_before_optimizer_update(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nan_render(*args, **kwargs):
        result = _differentiable_render(*args, **kwargs)
        return RenderResult(result.rgb * torch.nan, result.alpha, None, result.info)

    monkeypatch.setattr(trainer_module, "render_gaussians", nan_render)
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    before = trainer.gaussians.sh0.detach().clone()

    with pytest.raises(FloatingPointError, match="loss"):
        trainer.train(stop_step=1, checkpoint_every=1)
    torch.testing.assert_close(trainer.gaussians.sh0, before)


def test_non_finite_gradient_fails_before_optimizer_update(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FiniteForwardNanBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, grad_output):
            return torch.full_like(grad_output, torch.nan)

    def bad_gradient_render(*args, **kwargs):
        gaussians = kwargs["gaussians"]
        intrinsics = kwargs["intrinsics"]
        value = _FiniteForwardNanBackward.apply(torch.sigmoid(gaussians.sh0.mean()))
        rgb = value.expand(intrinsics.height, intrinsics.width, 3)
        alpha = torch.ones((*rgb.shape[:2], 1))
        return RenderResult(
            rgb,
            alpha,
            None,
            {
                "means2d": gaussians.means[:, :2],
                "radii": torch.ones(gaussians.num_gaussians, dtype=torch.int32),
            },
        )

    monkeypatch.setattr(trainer_module, "render_gaussians", bad_gradient_render)
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())
    before = trainer.gaussians.sh0.detach().clone()

    with pytest.raises(FloatingPointError, match="gradient"):
        trainer.train(stop_step=1, checkpoint_every=1)
    torch.testing.assert_close(trainer.gaussians.sh0, before)


def test_non_finite_projected_mean_gradient_fails_before_density_update(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FiniteForwardNanBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, grad_output):
            return torch.full_like(grad_output, torch.nan)

    def bad_projected_gradient_render(*args, **kwargs):
        gaussians = kwargs["gaussians"]
        intrinsics = kwargs["intrinsics"]
        means2d = torch.zeros((gaussians.num_gaussians, 2), requires_grad=True)
        projected = _FiniteForwardNanBackward.apply(means2d.mean())
        value = torch.sigmoid(gaussians.sh0.mean()) + projected
        rgb = value.expand(intrinsics.height, intrinsics.width, 3)
        return RenderResult(
            rgb,
            torch.ones((*rgb.shape[:2], 1)),
            None,
            {
                "means2d": means2d,
                "radii": torch.ones(gaussians.num_gaussians, dtype=torch.int32),
            },
        )

    monkeypatch.setattr(
        trainer_module, "render_gaussians", bad_projected_gradient_render
    )
    trainer = _trainer(tmp_path / "run", manifest_artifact, _MockDataset())

    with pytest.raises(FloatingPointError, match="projected means gradient"):
        trainer.train(stop_step=1, checkpoint_every=1)


def test_checkpoint_manifest_hash_validation(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path=checkpoint_path,
        step=5,
        gaussians_state_dict=_gaussians().state_dict(),
        optimizers_state_dict={},
        scheduler_state_dict={},
        strategy_state={},
        active_sh_degree=0,
        manifest_hash="expected",
        config_hash="config",
        precision_state={"scale": 1024.0},
    )

    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        load_checkpoint(checkpoint_path, expected_manifest_hash="wrong")
    loaded = load_checkpoint(checkpoint_path, expected_manifest_hash="expected")
    assert loaded["step"] == 5
    assert loaded["precision_state"] == {"scale": 1024.0}


def test_compute_file_sha256_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("first", encoding="utf-8")
    first = compute_file_sha256(path)
    path.write_text("second", encoding="utf-8")
    assert compute_file_sha256(path) != first


def test_manifest_hash_includes_npz_arrays(manifest_artifact: Path) -> None:
    first = compute_manifest_sha256(manifest_artifact)
    with np.load(manifest_artifact.parent / "arrays.npz") as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values["train_world_to_camera"] = values["train_world_to_camera"].copy()
    values["train_world_to_camera"][0, 0, 3] -= 1.0
    np.savez(manifest_artifact.parent / "arrays.npz", **values)

    assert compute_manifest_sha256(manifest_artifact) != first


def test_trainer_rejects_manifest_artifact_unrelated_to_dataset(
    tmp_path: Path,
    manifest_artifact: Path,
) -> None:
    with np.load(manifest_artifact.parent / "arrays.npz") as arrays:
        values = {name: arrays[name] for name in arrays.files}
    values["normalization_transform"] = np.eye(4, dtype=np.float64)
    np.savez(manifest_artifact.parent / "arrays.npz", **values)

    with pytest.raises(ValueError, match="dataset does not match manifest artifact"):
        _trainer(tmp_path / "run", manifest_artifact, _MockDataset())


def test_resume_rejects_changed_training_config(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    first = _trainer(
        tmp_path / "run", manifest_artifact, _MockDataset(), config=_config()
    )
    first.train(stop_step=2, checkpoint_every=2)
    checkpoint = tmp_path / "run" / "checkpoints" / "step_000000002.pt"
    changed = _config()
    changed["lambda_dssim"] = 0.1
    resumed = _trainer(
        tmp_path / "resumed", manifest_artifact, _MockDataset(), config=changed
    )

    with pytest.raises(ValueError, match="Config hash mismatch"):
        resumed.resume(checkpoint)


def test_rolling_checkpoint_retains_only_latest_recovery(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    output = tmp_path / "run"
    trainer = _trainer(
        output,
        manifest_artifact,
        _MockDataset(),
        config=_config(max_steps=2),
    )

    trainer.train(
        stop_step=2,
        checkpoint_every=1,
        rolling_checkpoint=True,
    )

    checkpoints = tuple((output / "checkpoints").iterdir())
    assert [path.name for path in checkpoints] == ["recovery.pt"]
    assert load_checkpoint(
        checkpoints[0], expected_manifest_hash=trainer.manifest_hash
    )["step"] == 2


def test_trainer_checks_capacity_before_checkpoint_save(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        trainer_module,
        "require_checkpoint_path_capacity",
        lambda path, count: calls.append((Path(path), count)),
    )
    output = tmp_path / "run"
    trainer = _trainer(
        output,
        manifest_artifact,
        _MockDataset(),
        config=_config(max_steps=1),
    )

    trainer.train(stop_step=1, checkpoint_every=1, rolling_checkpoint=True)

    assert calls == [(output / "checkpoints" / "recovery.pt", 5)]


def test_checkpoint_persists_timing_before_save_failure(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    def fail_save_checkpoint(**kwargs: object) -> None:
        raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(trainer_module, "save_checkpoint", fail_save_checkpoint)
    output = tmp_path / "run"
    trainer = _trainer(
        output,
        manifest_artifact,
        _MockDataset(),
        config=_config(max_steps=2),
    )

    with pytest.raises(RuntimeError, match="simulated checkpoint failure"):
        trainer.train(
            stop_step=2,
            checkpoint_every=2,
            rolling_checkpoint=True,
        )

    timing = json.loads((output / "timing.json").read_text(encoding="utf-8"))
    assert list(timing) == ["1", "2"]


def test_existing_run_config_is_not_overwritten(
    tmp_path: Path,
    manifest_artifact: Path,
) -> None:
    output = tmp_path / "run"
    _trainer(output, manifest_artifact, _MockDataset(), config=_config())
    original = (output / "config.yaml").read_bytes()
    changed = _config()
    changed["lambda_dssim"] = 0.1

    with pytest.raises(ValueError, match="existing run config"):
        _trainer(output, manifest_artifact, _MockDataset(), config=changed)
    assert (output / "config.yaml").read_bytes() == original


def test_fresh_resume_matches_continuous_training_with_real_updates(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    config = _config(max_steps=4)

    continuous = _trainer(
        tmp_path / "continuous", manifest_artifact, _MockDataset(), config=config
    )
    initial_sh0 = continuous.gaussians.sh0.detach().clone()
    continuous.train(stop_step=4, checkpoint_every=2)

    first_half = _trainer(
        tmp_path / "split", manifest_artifact, _MockDataset(), config=config
    )
    first_half.train(stop_step=2, checkpoint_every=2)
    checkpoint = tmp_path / "split" / "checkpoints" / "step_000000002.pt"

    resumed = _trainer(
        tmp_path / "split", manifest_artifact, _MockDataset(), config=config
    )
    resumed.resume(checkpoint)
    resumed.train(stop_step=4, checkpoint_every=2)

    assert not torch.equal(continuous.gaussians.sh0, initial_sh0)
    for name, expected in continuous.gaussians.state_dict().items():
        torch.testing.assert_close(resumed.gaussians.state_dict()[name], expected)
    assert resumed.scheduler.state_dict() == continuous.scheduler.state_dict()
    assert compute_config_sha256(config) == resumed.config_hash
    timing = json.loads((tmp_path / "split" / "timing.json").read_text())
    assert list(timing) == ["1", "2", "3", "4"]


def test_resume_discards_records_newer_than_checkpoint(
    tmp_path: Path,
    manifest_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "render_gaussians", _differentiable_render)
    output = tmp_path / "run"
    config = _config(max_steps=4)
    first = _trainer(output, manifest_artifact, _MockDataset(), config=config)
    first.train(stop_step=3, checkpoint_every=1)

    resumed = _trainer(output, manifest_artifact, _MockDataset(), config=config)
    resumed.resume(output / "checkpoints" / "step_000000002.pt")

    metrics = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    timing = json.loads((output / "timing.json").read_text())
    assert [record["step"] for record in metrics] == [1, 2]
    assert list(timing) == ["1", "2"]
