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
    build_qualification_decision,
    evaluate_internal_validation,
    save_qualification_decision,
)


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
