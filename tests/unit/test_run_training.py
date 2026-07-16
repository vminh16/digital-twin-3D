from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import json
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
    }
    values.update(changes)
    return Namespace(**values)


def test_cuda_preflight_rejects_cpu_only_torch(monkeypatch):
    monkeypatch.setattr(run_training.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        run_training.run_cuda_preflight()


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


def test_profile_mode_requires_fresh_exact_550_step_run():
    run_training.validate_profile_args(_args(profile_input=True, max_steps=550))

    with pytest.raises(ValueError, match="550"):
        run_training.validate_profile_args(_args(profile_input=True, max_steps=549))
    with pytest.raises(ValueError, match="fresh"):
        run_training.validate_profile_args(
            _args(profile_input=True, max_steps=550, resume="checkpoint.pt")
        )


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
