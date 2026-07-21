from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest
import torch

import bts_nvs.training.run_training as run_training


def _args(**changes):
    values = {
        "scene_dir": "scene",
        "output_dir": "run",
        "manifest_dir": None,
        "resize_factor": 4,
        "max_steps": 500,
        "checkpoint_every": 100,
        "seed": 7,
        "resume": None,
        "cache_images": False,
        "pinned_transfer": False,
        "profile_input": False,
        "internal_holdout": False,
        "qualification_candidate": None,
        "full_length_qualification": False,
        "full_length_candidate": None,
        "backend_qualification": False,
        "optimizer_backend": "adam",
        "precision": "fp32",
        "rolling_checkpoint": False,
    }
    values.update(changes)
    return Namespace(**values)


def test_cuda_preflight_rejects_cpu_only_torch(monkeypatch):
    monkeypatch.setattr(run_training.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        run_training.run_cuda_preflight("adam-fused", "amp-fp16")


def test_preflight_requires_a_finite_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(RuntimeError, match="finite parameter gradient"):
        run_training.validate_preflight_gradients([parameter])

    (parameter * 2.0).sum().backward()
    run_training.validate_preflight_gradients([parameter])


@pytest.mark.parametrize("factor", [0, -1, True])
def test_resize_factor_must_be_positive_integer(factor):
    with pytest.raises(ValueError, match="resize_factor"):
        run_training.validate_resize_factor(factor)


def test_training_config_binds_preprocessing_identity():
    manifest = SimpleNamespace(scene_id="HCM0181")
    config = run_training.build_training_config(
        _args(resize_factor=2, seed=19),
        manifest,
        resize=(660, 494),
    )

    assert config["scene_id"] == "HCM0181"
    assert config["resize_factor"] == 2
    assert config["resize_width"] == 660
    assert config["resize_height"] == 494
    assert config["undistort"] is True
    assert config["seed"] == 19
    assert config["cache_images"] is False
    assert config["pinned_transfer"] is False
    assert config["optimizer_backend"] == "adam"
    assert config["precision"] == "fp32"
    assert config["rolling_checkpoint"] is False


@pytest.mark.parametrize(
    "backend,precision",
    [
        ("adam", "fp32"),
        ("adam-fused", "fp32"),
        ("adam-fused", "amp-fp16"),
    ],
)
def test_training_backend_contract_accepts_supported_pairs(backend, precision):
    run_training.validate_training_backend(
        _args(optimizer_backend=backend, precision=precision)
    )


def test_training_backend_contract_rejects_amp_with_unfused_adam():
    with pytest.raises(ValueError, match="amp-fp16 requires adam-fused"):
        run_training.validate_training_backend(
            _args(optimizer_backend="adam", precision="amp-fp16")
        )


def test_rolling_checkpoint_is_allowed_only_for_ordinary_or_full_length_runs():
    run_training.validate_rolling_checkpoint_args(_args(rolling_checkpoint=True))
    run_training.validate_rolling_checkpoint_args(
        _args(rolling_checkpoint=True, full_length_qualification=True)
    )

    for change in (
        {"profile_input": True},
        {"backend_qualification": True},
        {"qualification_candidate": "B0-reference"},
    ):
        with pytest.raises(ValueError, match="rolling checkpoint"):
            run_training.validate_rolling_checkpoint_args(
                _args(rolling_checkpoint=True, **change)
            )


def test_profile_mode_requires_fresh_exact_550_step_run():
    run_training.validate_profile_args(_args(profile_input=True, max_steps=550))

    with pytest.raises(ValueError, match="550"):
        run_training.validate_profile_args(_args(profile_input=True, max_steps=549))
    with pytest.raises(ValueError, match="fresh"):
        run_training.validate_profile_args(
            _args(profile_input=True, max_steps=550, resume="checkpoint.pt")
        )


def test_profile_implies_internal_holdout():
    assert run_training.internal_holdout_enabled(
        _args(profile_input=True, max_steps=550)
    )
    assert run_training.internal_holdout_enabled(_args(internal_holdout=True))
    assert not run_training.internal_holdout_enabled(_args())


def test_backend_qualification_locks_1000_step_l4_contract():
    valid = _args(
        backend_qualification=True,
        resize_factor=1,
        max_steps=1000,
        checkpoint_every=1000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
        optimizer_backend="adam-fused",
        precision="amp-fp16",
    )
    run_training.validate_backend_qualification_args(valid)
    assert run_training.internal_holdout_enabled(valid)
    assert run_training.should_save_checkpoints(valid) is False

    for change in (
        {"max_steps": 999},
        {"resize_factor": 2},
        {"seed": 1},
        {"cache_images": False},
        {"pinned_transfer": False},
        {"resume": "checkpoint.pt"},
        {"full_length_qualification": True},
    ):
        with pytest.raises(ValueError, match="backend qualification"):
            run_training.validate_backend_qualification_args(
                _args(**(vars(valid) | change))
            )


def test_qualification_candidate_locks_run_contract():
    valid = _args(
        qualification_candidate="B0-compact",
        resize_factor=1,
        max_steps=7000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
    )
    run_training.validate_qualification_args(valid)
    assert run_training.internal_holdout_enabled(valid)

    for change in (
        {"max_steps": 6999},
        {"resize_factor": 2},
        {"seed": 1},
        {"cache_images": False},
        {"pinned_transfer": False},
    ):
        with pytest.raises(ValueError, match="qualification"):
            run_training.validate_qualification_args(_args(**(vars(valid) | change)))


def test_full_length_qualification_locks_30k_contract(tmp_path):
    output = tmp_path / "run"
    valid = _args(
        output_dir=str(output),
        full_length_qualification=True,
        resize_factor=1,
        max_steps=30_000,
        checkpoint_every=3_000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
    )

    run_training.validate_full_length_args(valid)
    assert run_training.internal_holdout_enabled(valid)
    assert run_training.should_save_checkpoints(valid)
    run_training.validate_full_length_scene("HCM0181", valid)

    for change in (
        {"max_steps": 7_000},
        {"resize_factor": 2},
        {"checkpoint_every": 6_000},
        {"seed": 1},
        {"cache_images": False},
        {"pinned_transfer": False},
        {"qualification_candidate": "B0-reference"},
    ):
        with pytest.raises(ValueError, match="full-length"):
            run_training.validate_full_length_args(_args(**(vars(valid) | change)))

    with pytest.raises(ValueError, match="HCM0181"):
        run_training.validate_full_length_scene("HCM0421", valid)


def test_full_length_resume_is_fixed_to_rolling_checkpoint(tmp_path):
    output = tmp_path / "run"
    recovery = output / "checkpoints" / "recovery.pt"
    recovery.parent.mkdir(parents=True)
    recovery.write_bytes(b"checkpoint")
    valid = _args(
        output_dir=str(output),
        full_length_qualification=True,
        resize_factor=1,
        max_steps=30_000,
        checkpoint_every=3_000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
        resume=str(recovery),
    )

    run_training.validate_full_length_args(valid)

    valid.resume = str(output / "checkpoints" / "other.pt")
    with pytest.raises(ValueError, match="recovery.pt"):
        run_training.validate_full_length_args(valid)


def test_recovery_checkpoint_must_be_complete_and_at_expected_step(
    monkeypatch, tmp_path
):
    complete = {
        "step": 30_000,
        "gaussians": {},
        "optimizers": {},
        "scheduler": {},
        "strategy_state": {},
        "active_sh_degree": 3,
        "rng_states": {},
        "manifest_hash": "manifest",
        "config_hash": "config",
    }
    monkeypatch.setattr(
        run_training,
        "load_checkpoint",
        lambda *args, **kwargs: complete,
    )

    run_training.validate_recovery_checkpoint(
        tmp_path / "recovery.pt", "manifest", "config", 30_000
    )

    monkeypatch.setattr(
        run_training,
        "load_checkpoint",
        lambda *args, **kwargs: {
            key: value for key, value in complete.items() if key != "rng_states"
        },
    )
    with pytest.raises(ValueError, match="rng_states"):
        run_training.validate_recovery_checkpoint(
            tmp_path / "recovery.pt", "manifest", "config", 30_000
        )


def test_completed_resume_skips_optimization_but_allows_finalization():
    assert run_training.optimization_required(27_000, 30_000) is True
    assert run_training.optimization_required(30_000, 30_000) is False

    with pytest.raises(ValueError, match="beyond"):
        run_training.optimization_required(30_001, 30_000)


def test_clean_git_commit_rejects_tracked_changes(monkeypatch, tmp_path):
    responses = iter(("abc123\n", " M src/file.py\n"))
    monkeypatch.setattr(
        run_training.subprocess,
        "check_output",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="dirty"):
        run_training.read_clean_git_commit(tmp_path)


def test_full_length_config_uses_reference_threshold():
    args = _args(full_length_qualification=True)
    config = run_training.build_training_config(
        args, SimpleNamespace(scene_id="HCM0181"), resize=(1320, 989)
    )

    assert config["full_length_qualification"] is True
    assert config["grow_grad2d"] == pytest.approx(0.0002)


def test_full_length_candidate_locks_phase_c_contract(tmp_path):
    output = tmp_path / "run"
    valid = _args(
        output_dir=str(output),
        full_length_candidate="C1-absgrad-t08-revopacity-v1",
        resize_factor=1,
        max_steps=30_000,
        checkpoint_every=3_000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
    )

    run_training.validate_full_length_args(valid)
    run_training.validate_full_length_scene("HCM0181", valid)
    config = run_training.build_training_config(
        valid, SimpleNamespace(scene_id="HCM0181"), resize=(1320, 989)
    )

    assert config["full_length_candidate"] == "C1-absgrad-t08-revopacity-v1"
    assert config["grow_grad2d"] == pytest.approx(0.0008)
    assert config["absgrad"] is True
    assert config["revised_opacity"] is True
    assert config["rolling_checkpoint"] is True
    assert run_training.internal_holdout_enabled(valid) is True


def test_full_length_candidate_resume_is_colocated(tmp_path):
    output = tmp_path / "run"
    recovery = output / "checkpoints" / "recovery.pt"
    recovery.parent.mkdir(parents=True)
    recovery.write_bytes(b"checkpoint")
    valid = _args(
        output_dir=str(output),
        full_length_candidate="C1-absgrad-t08-revopacity-v1",
        resize_factor=1,
        max_steps=30_000,
        checkpoint_every=3_000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
        resume=str(recovery),
    )

    run_training.validate_full_length_args(valid)
    with pytest.raises(ValueError, match="recovery.pt"):
        run_training.validate_full_length_args(
            _args(**(vars(valid) | {"resume": str(tmp_path / "foreign.pt")}))
        )


def test_training_config_applies_locked_candidate_threshold():
    manifest = SimpleNamespace(scene_id="HCM0181")
    config = run_training.build_training_config(
        _args(qualification_candidate="B0-compact"),
        manifest,
        resize=(8, 6),
    )

    assert config["qualification_candidate"] == "B0-compact"
    assert config["grow_grad2d"] == pytest.approx(0.0003)
    assert run_training.should_save_checkpoints(_args()) is True
    assert run_training.should_save_checkpoints(
        _args(qualification_candidate="B0-compact")
    ) is False


def test_training_config_applies_locked_c1_density_settings():
    manifest = SimpleNamespace(scene_id="HCM0421")
    config = run_training.build_training_config(
        _args(qualification_candidate="C1-absgrad-t08-revopacity-v1"),
        manifest,
        resize=(8, 6),
    )

    assert config["qualification_candidate"] == "C1-absgrad-t08-revopacity-v1"
    assert config["grow_grad2d"] == pytest.approx(0.0008)
    assert config["absgrad"] is True
    assert config["revised_opacity"] is True


def test_internal_holdout_requires_colocated_artifact(tmp_path, monkeypatch):
    manifest = SimpleNamespace(scene_id="HCM0181")
    with pytest.raises(FileNotFoundError, match="holdout.json"):
        run_training.load_internal_holdout(tmp_path, manifest, enabled=True)
    assert run_training.load_internal_holdout(tmp_path, manifest, enabled=False) is None

    path = tmp_path / "holdout.json"
    path.write_text("{}")
    expected = SimpleNamespace(train_image_names=("a.JPG",))
    monkeypatch.setattr(
        run_training,
        "load_holdout_split",
        lambda source, value: expected,
    )
    assert run_training.load_internal_holdout(tmp_path, manifest, enabled=True) is expected


def test_training_config_records_holdout_identity():
    manifest = SimpleNamespace(scene_id="HCM0181")
    split = SimpleNamespace(
        algorithm="pose_fps_guard2_v1",
        manifest_sha256="abc",
        train_image_names=("a", "b"),
        guard_image_names=("c",),
        validation_image_names=("d",),
    )

    config = run_training.build_training_config(
        _args(), manifest, resize=(8, 6), split=split
    )

    assert config["holdout_algorithm"] == "pose_fps_guard2_v1"
    assert config["holdout_manifest_sha256"] == "abc"
    assert config["internal_train_count"] == 2
    assert config["guard_count"] == 1
    assert config["validation_count"] == 1


def test_internal_holdout_replaces_only_sparse_initialization(monkeypatch):
    @dataclass(frozen=True)
    class Manifest:
        sparse_points: np.ndarray
        sparse_colors: np.ndarray

    manifest = Manifest(
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.uint8),
    )
    split = SimpleNamespace()
    sparse = SimpleNamespace(
        points=np.ones((2, 3), dtype=np.float64),
        colors=np.full((2, 3), 7, dtype=np.uint8),
    )
    monkeypatch.setattr(
        run_training,
        "build_split_sparse_initialization",
        lambda value, root, holdout: sparse,
    )

    result = run_training.build_initialization_manifest(manifest, Path("scene"), split)

    np.testing.assert_array_equal(result.sparse_points, sparse.points)
    np.testing.assert_array_equal(result.sparse_colors, sparse.colors)
    np.testing.assert_array_equal(manifest.sparse_points, np.zeros((1, 3)))


def test_host_resource_preflight_rejects_swap_and_low_cache_ram():
    run_training.validate_host_resources(cache_bytes=0, memory_status=None)
    with pytest.raises(RuntimeError, match="swap"):
        run_training.validate_host_resources(
            cache_bytes=0,
            memory_status=(8 * 1024**3, 1),
        )
    with pytest.raises(MemoryError, match="host RAM"):
        run_training.validate_host_resources(
            cache_bytes=5 * 1024**3,
            memory_status=(8 * 1024**3, 0),
        )


def test_non_resume_run_rejects_non_empty_output(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "metrics.jsonl").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        run_training.validate_output_directory(output, resume=None)


def test_resume_requires_existing_checkpoint(tmp_path: Path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        run_training.validate_output_directory(tmp_path / "run", resume=missing)


def test_convergence_report_records_deltas_and_non_blank_render(tmp_path: Path):
    initial = {
        "psnr_db": 10.0,
        "psnr_is_infinite": False,
        "ssim": 0.20,
        "alpha_coverage": 0.4,
        "rgb_std": 0.1,
        "valid_fraction": 0.99,
    }
    final = {**initial, "psnr_db": 12.5, "ssim": 0.28}

    report = run_training.write_convergence_report(
        tmp_path / "convergence.json", initial, final
    )

    assert report["psnr_delta_db"] == pytest.approx(2.5)
    assert report["ssim_delta"] == pytest.approx(0.08)
    assert report["quality_improved"] is True
    assert report["final_render_non_blank"] is True
    assert json.loads((tmp_path / "convergence.json").read_text()) == report


def test_initial_metrics_round_trip_for_resume(tmp_path: Path):
    path = tmp_path / "initial_metrics.json"
    metrics = {"psnr_db": 10.0, "ssim": 0.2}

    run_training.write_json_record(path, metrics)

    assert run_training.read_json_record(path) == metrics
