import json
from dataclasses import replace

import numpy as np
import pytest

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.holdout import build_pose_holdout
from bts_nvs.data.manifest import SceneManifest
from bts_nvs.data.validation import DataContractError
from bts_nvs.evaluation.pose_strata import (
    assign_pose_strata,
    build_pose_strata,
    save_pose_strata,
)


def _manifest(count: int = 80) -> SceneManifest:
    names = tuple(f"image_{index:03d}.JPG" for index in range(count))
    camera_to_world = np.repeat(np.eye(4)[None], count, axis=0)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    camera_to_world[:, 0, 3] = np.cos(angles)
    camera_to_world[:, 1, 3] = np.sin(angles)
    camera_to_world[:, 2, 3] = np.linspace(-0.5, 0.5, count)
    world_to_camera = np.linalg.inv(camera_to_world)
    intrinsics = CameraIntrinsics(64, 48, 50.0, 50.0, 32.0, 24.0)
    distortion = CameraDistortion("PINHOLE", ())
    return SceneManifest(
        schema_version=1,
        scene_id="scene",
        train_image_paths=tuple(f"train/images/{name}" for name in names),
        train_image_names=names,
        train_world_to_camera=world_to_camera,
        train_camera_to_world=camera_to_world,
        train_intrinsics=(intrinsics,) * count,
        train_distortion=(distortion,) * count,
        test_image_names=("test.JPG",),
        test_output_names=("test.png",),
        test_world_to_camera=np.eye(4)[None],
        test_intrinsics=(intrinsics,),
        test_distortion=(distortion,),
        sparse_points=np.empty((0, 3)),
        sparse_colors=np.empty((0, 3), dtype=np.uint8),
        normalization_transform=np.eye(4),
        inverse_normalization_transform=np.eye(4),
    )


def test_assign_pose_strata_is_order_independent_and_exhaustive() -> None:
    distances = {f"v{index}.JPG": float(index) for index in range(8)}

    forward = assign_pose_strata(distances)
    reverse = assign_pose_strata(dict(reversed(tuple(distances.items()))))

    assert forward == reverse
    assert tuple(forward.values()).count("easy") == 3
    assert tuple(forward.values()).count("medium") == 3
    assert tuple(forward.values()).count("hard") == 2


def test_assign_pose_strata_breaks_distance_ties_by_filename() -> None:
    assigned = assign_pose_strata({"z.JPG": 1.0, "a.JPG": 1.0, "m.JPG": 1.0})

    assert assigned == {"a.JPG": "easy", "m.JPG": "medium", "z.JPG": "hard"}


@pytest.mark.parametrize(
    "distances",
    [
        {},
        {"a.JPG": float("nan")},
        {"a.JPG": -1.0},
        {"": 1.0},
    ],
)
def test_assign_pose_strata_rejects_invalid_distances(distances) -> None:
    with pytest.raises(ValueError):
        assign_pose_strata(distances)


def test_build_pose_strata_records_nearest_retained_train_camera() -> None:
    manifest = _manifest()
    split = build_pose_holdout(manifest)

    report = build_pose_strata(manifest, split)

    assert report["schema_version"] == 1
    assert report["scene_id"] == "scene"
    assert report["algorithm"] == "nearest_train_tertiles_v1"
    assert report["holdout_algorithm"] == split.algorithm
    assert report["holdout_manifest_sha256"] == split.manifest_sha256
    assert report["image_count"] == len(split.validation_image_names)
    assert set(report["images"]) == set(split.validation_image_names)
    assert {item["stratum"] for item in report["images"].values()} == {
        "easy",
        "medium",
        "hard",
    }
    for item in report["images"].values():
        assert item["nearest_train_image_name"] in split.train_image_names
        assert item["pose_distance"] >= 0.0
        assert item["center_distance"] >= 0.0
        assert 0.0 <= item["rotation_angle_deg"] <= 180.0


def test_build_pose_strata_rejects_split_that_does_not_match_manifest() -> None:
    manifest = _manifest()
    split = replace(build_pose_holdout(manifest), algorithm="tampered")

    with pytest.raises(DataContractError, match="algorithm"):
        build_pose_strata(manifest, split)


def test_pose_strata_save_is_canonical(tmp_path) -> None:
    manifest = _manifest()
    report = build_pose_strata(manifest, build_pose_holdout(manifest))
    path = tmp_path / "pose_strata.json"

    save_pose_strata(report, path)
    first = path.read_bytes()
    save_pose_strata(report, path)

    assert path.read_bytes() == first
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
    assert json.loads(first)["algorithm"] == "nearest_train_tertiles_v1"
