import copy
import json

import pytest

from bts_nvs.evaluation.experiment_report import (
    build_experiment_report,
    local_score50,
    save_experiment_report,
)


NAMES = ("easy.JPG", "medium.JPG", "hard.JPG")


def _full_frame() -> dict:
    images = {
        "easy.JPG": {"psnr_db": 25.0, "ssim": 0.90, "lpips": 0.10},
        "medium.JPG": {"psnr_db": 23.0, "ssim": 0.85, "lpips": 0.15},
        "hard.JPG": {"psnr_db": 20.0, "ssim": 0.80, "lpips": 0.20},
    }
    return {
        "image_count": 3,
        "psnr_db_mean": 68.0 / 3.0,
        "ssim_mean": 0.85,
        "lpips_mean": 0.15,
        "images": images,
    }


def _detail() -> dict:
    images = {
        name: {
            "hf_l1": 0.01 * (index + 1),
            "missing_edge": 0.02 * (index + 1),
            "spurious_edge": 0.03 * (index + 1),
            "symmetric_edge_distance": 0.001 * (index + 1),
        }
        for index, name in enumerate(NAMES)
    }
    return {
        "schema_version": 1,
        "scene_id": "HCM0539",
        "image_count": 3,
        "images": images,
    }


def _pose() -> dict:
    images = {
        name: {
            "nearest_train_image_name": f"train_{index}.JPG",
            "pose_distance": float(index + 1),
            "center_distance": float(index + 1) / 2.0,
            "rotation_angle_deg": float(index * 5),
            "stratum": stratum,
        }
        for index, (name, stratum) in enumerate(
            zip(NAMES, ("easy", "medium", "hard"))
        )
    }
    return {
        "schema_version": 1,
        "scene_id": "HCM0539",
        "algorithm": "nearest_train_tertiles_v1",
        "holdout_algorithm": "pose_fps_guard2_v1",
        "holdout_manifest_sha256": "c" * 64,
        "image_count": 3,
        "images": images,
    }


def _resources() -> dict:
    return {
        "total_time_seconds": 100.0,
        "max_vram_mb": 5000.0,
        "peak_gaussians": 1_200_000,
        "final_num_gaussians": 1_100_000,
    }


def _build(**overrides) -> dict:
    arguments = {
        "scene_id": "HCM0539",
        "candidate_id": "B0-reference",
        "step": 7000,
        "config_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "holdout_sha256": "c" * 64,
        "full_frame_report": _full_frame(),
        "detail_report": _detail(),
        "pose_strata_report": _pose(),
        "resource_summary": _resources(),
    }
    arguments.update(overrides)
    return build_experiment_report(**arguments)


def test_build_experiment_report_aggregates_matching_images_by_stratum() -> None:
    report = _build()

    assert report["schema_version"] == 1
    assert report["scene_id"] == "HCM0539"
    assert report["candidate_id"] == "B0-reference"
    assert report["image_count"] == 3
    assert report["strata"]["hard"]["image_count"] == 1
    assert report["strata"]["hard"]["score50"] < report["strata"]["easy"][
        "score50"
    ]
    assert report["overall"]["score50"] == pytest.approx(
        local_score50(report["overall"])
    )
    assert report["resources"] == _resources()
    assert set(report["images"]) == set(NAMES)


def test_local_score50_uses_locked_diagnostic_formula() -> None:
    assert local_score50({"psnr_db": 25.0, "ssim": 0.9, "lpips": 0.1}) == pytest.approx(
        78.0
    )


def test_report_rejects_mismatched_image_sets_and_counts() -> None:
    detail = _detail()
    detail["images"].pop("hard.JPG")
    with pytest.raises(ValueError, match="image names"):
        _build(detail_report=detail)

    full_frame = _full_frame()
    full_frame["image_count"] = 2
    with pytest.raises(ValueError, match="image_count"):
        _build(full_frame_report=full_frame)


def test_report_rejects_unknown_strata_and_holdout_identity() -> None:
    pose = _pose()
    pose["images"]["hard.JPG"]["stratum"] = "extreme"
    with pytest.raises(ValueError, match="stratum"):
        _build(pose_strata_report=pose)

    pose = _pose()
    pose["holdout_manifest_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="holdout"):
        _build(pose_strata_report=pose)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"scene_id": ""}, "scene_id"),
        ({"candidate_id": ""}, "candidate_id"),
        ({"step": 0}, "step"),
        ({"config_sha256": "not-a-hash"}, "SHA-256"),
        ({"manifest_sha256": "A" * 64}, "SHA-256"),
        ({"holdout_sha256": "c" * 63}, "SHA-256"),
    ],
)
def test_report_rejects_invalid_identity(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _build(**overrides)


def test_report_rejects_non_finite_metrics_and_invalid_resources() -> None:
    full_frame = _full_frame()
    full_frame["images"]["hard.JPG"]["lpips"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _build(full_frame_report=full_frame)

    for field in _resources():
        resources = _resources()
        resources[field] = -1
        with pytest.raises(ValueError, match=field):
            _build(resource_summary=resources)


def test_experiment_report_save_is_canonical(tmp_path) -> None:
    report = _build()
    path = tmp_path / "experiment_report.json"

    save_experiment_report(report, path)
    first = path.read_bytes()
    save_experiment_report(copy.deepcopy(report), path)

    assert path.read_bytes() == first
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
    assert b"NaN" not in first and b"Infinity" not in first
    assert json.loads(first)["candidate_id"] == "B0-reference"
