import csv

import numpy as np
import pytest

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.colmap import (
    ColmapCameraRecord,
    ColmapImageRecord,
    ColmapModel,
    ColmapPointRecord,
)
from bts_nvs.data.manifest import TEST_POSE_COLUMNS, load_scene_source_data
from bts_nvs.data.validation import DataContractError


def _w2c(tx=0.0):
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = tx
    return result


def _model(test_tx=0.0):
    camera = ColmapCameraRecord(
        camera_id=1,
        intrinsics=CameraIntrinsics(1320, 989, 926.0, 926.0, 660.0, 494.5),
        distortion=CameraDistortion("SIMPLE_RADIAL", (0.01,)),
    )
    images = {
        10: ColmapImageRecord(10, "train_b.JPG", 1, _w2c()),
        11: ColmapImageRecord(11, "train_a.JPG", 1, _w2c(1.0)),
        20: ColmapImageRecord(20, "test_001.JPG", 1, _w2c(test_tx)),
        30: ColmapImageRecord(30, "extra.JPG", 1, _w2c()),
    }
    points = {
        100: ColmapPointRecord(100, [1.0, 2.0, 3.0], [1, 2, 3], 0.5, (10, 20)),
        101: ColmapPointRecord(101, [4.0, 5.0, 6.0], [4, 5, 6], 0.4, (20,)),
        102: ColmapPointRecord(102, [np.nan, 0.0, 0.0], [7, 8, 9], 0.3, (11,)),
        103: ColmapPointRecord(103, [7.0, 8.0, 9.0], [10, 11, 12], 0.2, (11,)),
    }
    return ColmapModel(cameras={1: camera}, images=images, points3d=points)


def _make_scene(tmp_path, *, test_name="test_001.JPG", test_tx="0", test_fx="926"):
    scene = tmp_path / "scene_001"
    image_dir = scene / "train" / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "train_b.JPG").write_bytes(b"b")
    (image_dir / "train_a.JPG").write_bytes(b"a")
    test_dir = scene / "test"
    test_dir.mkdir()
    row = {
        "image_name": test_name,
        "qw": "-1",
        "qx": "0",
        "qy": "0",
        "qz": "0",
        "tx": test_tx,
        "ty": "0",
        "tz": "0",
        "fx": test_fx,
        "fy": test_fx,
        "cx": "660",
        "cy": "494.5",
        "width": "1320",
        "height": "989",
    }
    with (test_dir / "test_poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEST_POSE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return scene


def test_scene_source_uses_physical_train_split_and_train_supported_points(tmp_path):
    scene = _make_scene(tmp_path)

    source = load_scene_source_data(scene, colmap_model=_model())

    assert source.scene_id == "scene_001"
    assert source.train_image_names == ("train_a.JPG", "train_b.JPG")
    assert source.train_image_paths == (
        "train/images/train_a.JPG",
        "train/images/train_b.JPG",
    )
    assert tuple(image.image_id for image in source.train_images) == (11, 10)
    assert tuple(point.point_id for point in source.sparse_points) == (100, 103)
    assert source.test_poses[0].image_name == "test_001.JPG"
    assert source.test_distortions == (CameraDistortion("SIMPLE_RADIAL", (0.01,)),)
    assert 30 not in {image.image_id for image in source.train_images}


def test_scene_source_requires_every_physical_train_image_to_have_pose(tmp_path):
    scene = _make_scene(tmp_path)
    (scene / "train" / "images" / "missing.JPG").write_bytes(b"missing")

    with pytest.raises(DataContractError, match="missing.JPG"):
        load_scene_source_data(scene, colmap_model=_model())


def test_scene_source_requires_matching_test_registration(tmp_path):
    scene = _make_scene(tmp_path, test_name="unknown.JPG")

    with pytest.raises(DataContractError, match="unknown.JPG"):
        load_scene_source_data(scene, colmap_model=_model())


def test_scene_source_cross_checks_translation(tmp_path):
    scene = _make_scene(tmp_path, test_tx="0.01")

    with pytest.raises(DataContractError, match="translation"):
        load_scene_source_data(scene, colmap_model=_model())


def test_scene_source_cross_checks_csv_intrinsics(tmp_path):
    scene = _make_scene(tmp_path, test_fx="927")

    with pytest.raises(DataContractError, match="intrinsics"):
        load_scene_source_data(scene, colmap_model=_model())


def test_colmap_records_use_canonical_read_only_arrays():
    model = _model()

    assert model.images[10].world_to_camera.dtype == np.float64
    assert not model.images[10].world_to_camera.flags.writeable
    assert model.points3d[100].xyz.dtype == np.float64
    assert not model.points3d[100].xyz.flags.writeable
    assert model.points3d[100].rgb.dtype == np.uint8
    assert not model.points3d[100].rgb.flags.writeable
