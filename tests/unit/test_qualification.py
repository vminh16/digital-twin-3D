import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bts_nvs.training.qualification as qualification
from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.dataset import CameraSample
from bts_nvs.rendering.render_result import RenderResult
from bts_nvs.training.qualification import (
    CALIBRATION_SCENES,
    build_full_length_report,
    build_qualification_decision,
    evaluate_internal_validation,
    hash_validation_renders,
    save_full_length_report,
    save_qualification_decision,
    save_qualification_report,
)


def test_validation_render_hashes_bind_original_names_to_png_payloads(tmp_path):
    render_dir = tmp_path / "renders"
    render_dir.mkdir()
    (render_dir / "a.png").write_bytes(b"render-a")
    (render_dir / "b.png").write_bytes(b"render-b")

    hashes = hash_validation_renders(render_dir, ("a.JPG", "b.JPG"))

    assert tuple(hashes) == ("a.JPG", "b.JPG")
    assert all(len(value) == 64 for value in hashes.values())

    (render_dir / "b.png").unlink()
    with pytest.raises(FileNotFoundError, match="b.png"):
        hash_validation_renders(render_dir, ("a.JPG", "b.JPG"))


def _reports(*, compact_psnr_delta=-0.1, compact_resource_ratio=0.8):
    reports = []
    for scene in CALIBRATION_SCENES:
        reports.append(
            {
                "schema_version": 1,
                "scene_id": scene,
                "candidate_id": "B0-reference",
                "step": 7000,
                "image_count": 8,
                "psnr_db_mean": 24.0,
                "ssim_mean": 0.85,
                "lpips_mean": 0.20,
                "peak_gaussians": 1_000_000,
                "max_vram_mb": 5000.0,
                "total_time_seconds": 1000.0,
            }
        )
        reports.append(
            {
                "schema_version": 1,
                "scene_id": scene,
                "candidate_id": "B0-compact",
                "step": 7000,
                "image_count": 8,
                "psnr_db_mean": 24.0 + compact_psnr_delta,
                "ssim_mean": 0.847,
                "lpips_mean": 0.205,
                "peak_gaussians": int(1_000_000 * compact_resource_ratio),
                "max_vram_mb": 4500.0,
                "total_time_seconds": 950.0,
            }
        )
    return reports


def test_compact_wins_only_with_quality_and_resource_bounds():
    decision = build_qualification_decision(_reports())
    assert decision["selected_candidate"] == "B0-compact"
    assert decision["compact_passed"] is True

    quality_fail = build_qualification_decision(_reports(compact_psnr_delta=-0.3))
    assert quality_fail["selected_candidate"] == "B0-reference"

    resource_fail = build_qualification_decision(
        _reports(compact_resource_ratio=0.90)
    )
    assert resource_fail["selected_candidate"] == "B0-reference"


def test_decision_rejects_missing_or_duplicate_run():
    reports = _reports()
    with pytest.raises(ValueError, match="exactly"):
        build_qualification_decision(reports[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        build_qualification_decision(reports[:-1] + [reports[0]])


def test_decision_json_is_standard_and_deterministic(tmp_path):
    path = tmp_path / "decision.json"
    decision = build_qualification_decision(_reports())
    save_qualification_decision(decision, path)
    first = path.read_bytes()
    save_qualification_decision(decision, path)

    assert path.read_bytes() == first
    assert json.loads(first)["selected_candidate"] == "B0-compact"
    assert b"NaN" not in first and b"Infinity" not in first


def test_qualification_report_accepts_c1_without_changing_b0_matrix(tmp_path):
    report = _reports()[0] | {"candidate_id": "C1-absgrad-t08-v1"}

    save_qualification_report(report, tmp_path / "report.json")

    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert qualification.CANDIDATES == ("B0-reference", "B0-compact")


def test_internal_validation_reports_every_image_and_ignores_invalid_border(
    tmp_path, monkeypatch
):
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.ones((16, 16), dtype=bool)
    mask[0] = False
    sample = CameraSample(
        image=image,
        world_to_camera=np.eye(4),
        intrinsics=CameraIntrinsics(16, 16, 8.0, 8.0, 8.0, 8.0),
        distortion=CameraDistortion("PINHOLE", ()),
        valid_mask=mask,
        image_name="validation.JPG",
    )
    dataset = SimpleNamespace(
        manifest=SimpleNamespace(normalization_transform=np.eye(4)),
        __len__=lambda: 1,
    )

    class Dataset:
        manifest = dataset.manifest

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return sample

    prediction = torch.zeros((16, 16, 3))
    prediction[0] = 1.0
    monkeypatch.setattr(
        qualification,
        "render_gaussians",
        lambda **kwargs: RenderResult(
            prediction, torch.ones((16, 16, 1)), None, {}
        ),
    )
    class Backend:
        package = "fake"
        version = "1"
        device = "cpu"
        dtype = "float32"

        def __call__(self, prediction, target):
            return 0.25

    backend = Backend()
    trainer = SimpleNamespace(
        gaussians=object(), device=torch.device("cpu"), active_sh_degree=0
    )

    report = evaluate_internal_validation(
        trainer, Dataset(), backend, tmp_path / "renders"
    )

    assert report["image_count"] == 1
    assert report["valid_fraction_mean"] == pytest.approx(15 / 16)
    assert "validation.JPG" in report["images"]
    assert (tmp_path / "renders" / "validation.png").is_file()


def test_internal_validation_can_skip_render_files(tmp_path, monkeypatch):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    sample = CameraSample(
        image=image,
        world_to_camera=np.eye(4),
        intrinsics=CameraIntrinsics(16, 16, 8.0, 8.0, 8.0, 8.0),
        distortion=CameraDistortion("PINHOLE", ()),
        valid_mask=np.ones((16, 16), dtype=bool),
        image_name="validation.JPG",
    )

    class Dataset:
        manifest = SimpleNamespace(normalization_transform=np.eye(4))

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return sample

    monkeypatch.setattr(
        qualification,
        "render_gaussians",
        lambda **kwargs: RenderResult(
            torch.full((16, 16, 3), 0.1), torch.ones((16, 16, 1)), None, {}
        ),
    )
    trainer = SimpleNamespace(
        gaussians=object(), device=torch.device("cpu"), active_sh_degree=0
    )
    backend = lambda prediction, target: 0.0

    report = evaluate_internal_validation(trainer, Dataset(), backend, None)

    assert report["image_count"] == 1
    assert tuple(tmp_path.iterdir()) == ()


def test_full_length_report_applies_quality_and_resource_gates(tmp_path):
    initial = {
        "psnr_db_mean": 18.0,
        "ssim_mean": 0.60,
        "lpips_mean": 0.30,
    }
    final_train = {"psnr_db_mean": 27.0}
    final_validation = {
        "image_count": 1,
        "psnr_db_mean": 23.0,
        "ssim_mean": 0.70,
        "lpips_mean": 0.20,
        "images": {
            "frame.JPG": {"psnr_db": 23.0, "ssim": 0.70, "lpips": 0.20}
        },
    }
    metrics = [
        {"step": step, "loss": 0.1, "num_gaussians": 5_000_000}
        for step in range(1, 30_001)
    ]
    timing = {str(step): {"total": 0.1} for step in range(1, 30_001)}

    report = build_full_length_report(
        scene_id="HCM0181",
        git_commit="a" * 40,
        candidate_id="C1-absgrad-t08-revopacity-v1",
        config_sha256="b" * 64,
        manifest_sha256="c" * 64,
        validation_render_sha256={"frame.JPG": "d" * 64},
        initial_validation=initial,
        final_train=final_train,
        final_validation=final_validation,
        summary={
            "total_steps": 30_000,
            "max_vram_mb": 8_000.0,
            "total_time_seconds": 100.0,
            "final_num_gaussians": 4_000_000,
        },
        metric_records=metrics,
        timing_records=timing,
        convergence={"final_render_non_blank": True},
    )
    path = tmp_path / "report.json"
    save_full_length_report(report, path)

    assert report["automated_gates_passed"] is True
    assert report["validation_psnr_delta_db"] == pytest.approx(5.0)
    assert report["train_validation_psnr_gap_db"] == pytest.approx(4.0)
    assert report["candidate_id"] == "C1-absgrad-t08-revopacity-v1"
    assert report["config_sha256"] == "b" * 64
    assert report["manifest_sha256"] == "c" * 64
    assert report["validation_render_sha256"] == {"frame.JPG": "d" * 64}
    assert report["total_time_seconds"] == pytest.approx(100.0)
    assert report["final_num_gaussians"] == 4_000_000
    assert json.loads(path.read_text()) == report

    final_validation["lpips_mean"] = 0.27
    failed = build_full_length_report(
        scene_id="HCM0181",
        git_commit="a" * 40,
        candidate_id="C1-absgrad-t08-revopacity-v1",
        config_sha256="b" * 64,
        manifest_sha256="c" * 64,
        validation_render_sha256={"frame.JPG": "d" * 64},
        initial_validation=initial,
        final_train=final_train,
        final_validation=final_validation,
        summary={
            "total_steps": 30_000,
            "max_vram_mb": 8_000.0,
            "total_time_seconds": 100.0,
            "final_num_gaussians": 4_000_000,
        },
        metric_records=metrics,
        timing_records=timing,
        convergence={"final_render_non_blank": True},
    )
    assert failed["automated_gates_passed"] is False
