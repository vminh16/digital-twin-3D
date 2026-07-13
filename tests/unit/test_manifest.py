from pathlib import Path

import numpy as np
import pytest

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.cameras.poses import invert_rigid_transform
from bts_nvs.data.colmap import (
    ColmapCameraRecord,
    ColmapImageRecord,
    ColmapPointRecord,
)
from bts_nvs.data import (
    SceneManifest,
    build_scene_manifest,
    load_scene_manifest,
    save_scene_manifest,
    validate_scene_manifest,
)
from bts_nvs.data.manifest import SceneSourceData, TestPoseRecord as PoseRecord
from bts_nvs.data.validation import DataContractError


def _pose(center_x: float) -> tuple[np.ndarray, np.ndarray]:
    c2w = np.eye(4, dtype=np.float64)
    c2w[0, 3] = center_x
    return invert_rigid_transform(c2w), c2w


def _scene_root(tmp_path: Path) -> Path:
    root = tmp_path / "scene_001"
    images = root / "train" / "images"
    images.mkdir(parents=True)
    (images / "a.JPG").write_bytes(b"a")
    (images / "b.JPG").write_bytes(b"b")
    return root


def _manifest(root: Path, **changes) -> SceneManifest:
    w2c_a, c2w_a = _pose(0.0)
    w2c_b, c2w_b = _pose(2.0)
    intrinsics = CameraIntrinsics(16, 12, 10.0, 10.0, 8.0, 6.0)
    distortion = CameraDistortion("SIMPLE_RADIAL", (0.01,))
    values = dict(
        schema_version=1,
        scene_id="scene_001",
        train_image_paths=("train/images/a.JPG", "train/images/b.JPG"),
        train_image_names=("a.JPG", "b.JPG"),
        train_world_to_camera=np.stack((w2c_a, w2c_b)),
        train_camera_to_world=np.stack((c2w_a, c2w_b)),
        train_intrinsics=(intrinsics, intrinsics),
        train_distortion=(distortion, distortion),
        test_image_names=("target.JPG",),
        test_output_names=("target.png",),
        test_world_to_camera=w2c_a[None],
        test_intrinsics=(intrinsics,),
        test_distortion=(distortion,),
        sparse_points=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        sparse_colors=np.asarray([[4, 5, 6]], dtype=np.int64),
        normalization_transform=np.asarray(
            [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        ),
        inverse_normalization_transform=np.asarray(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        ),
    )
    values.update(changes)
    return SceneManifest(**values)


def test_manifest_canonicalizes_arrays_as_read_only(tmp_path):
    manifest = _manifest(_scene_root(tmp_path))

    for field in (
        "train_world_to_camera",
        "train_camera_to_world",
        "test_world_to_camera",
        "sparse_points",
        "sparse_colors",
        "normalization_transform",
        "inverse_normalization_transform",
    ):
        array = getattr(manifest, field)
        assert not array.flags.writeable
    assert manifest.train_world_to_camera.dtype == np.float64
    assert manifest.sparse_points.dtype == np.float64
    assert manifest.sparse_colors.dtype == np.uint8


def test_manifest_round_trip_preserves_contract(tmp_path):
    root = _scene_root(tmp_path)
    manifest = _manifest(root)
    validate_scene_manifest(manifest, root)
    output = tmp_path / "artifact"

    save_scene_manifest(manifest, output)
    loaded = load_scene_manifest(output / "manifest.json", root)

    assert loaded.schema_version == 1
    assert loaded.test_distortion == manifest.test_distortion
    assert loaded.test_output_names == ("target.png",)
    np.testing.assert_array_equal(loaded.sparse_colors, manifest.sparse_colors)
    np.testing.assert_allclose(
        loaded.train_world_to_camera, manifest.train_world_to_camera
    )
    assert not loaded.train_world_to_camera.flags.writeable


def test_save_is_non_overwriting(tmp_path):
    root = _scene_root(tmp_path)
    output = tmp_path / "artifact"
    save_scene_manifest(_manifest(root), output)

    with pytest.raises(FileExistsError):
        save_scene_manifest(_manifest(root), output)


def test_validation_rejects_path_escape(tmp_path):
    root = _scene_root(tmp_path)
    manifest = _manifest(
        root,
        train_image_paths=("../outside.JPG", "train/images/b.JPG"),
    )

    with pytest.raises(DataContractError, match="path"):
        validate_scene_manifest(manifest, root)


def test_validation_rejects_dot_dot_even_when_target_stays_inside_scene(tmp_path):
    root = _scene_root(tmp_path)
    manifest = _manifest(
        root,
        train_image_paths=("train/../train/images/a.JPG", "train/images/b.JPG"),
    )

    with pytest.raises(DataContractError, match="path"):
        validate_scene_manifest(manifest, root)


def test_validation_rejects_normalization_unrelated_to_train_trajectory(tmp_path):
    root = _scene_root(tmp_path)
    identity = np.eye(4, dtype=np.float64)
    manifest = _manifest(
        root,
        normalization_transform=identity,
        inverse_normalization_transform=identity,
    )

    with pytest.raises(DataContractError, match="normalization"):
        validate_scene_manifest(manifest, root)


def test_load_rejects_unsupported_schema(tmp_path):
    root = _scene_root(tmp_path)
    output = tmp_path / "artifact"
    save_scene_manifest(_manifest(root), output)
    manifest_json = output / "manifest.json"
    text = manifest_json.read_text(encoding="utf-8").replace(
        '"schema_version": 1', '"schema_version": 2'
    )
    manifest_json.write_text(text, encoding="utf-8")

    with pytest.raises(DataContractError, match="schema version"):
        load_scene_manifest(manifest_json, root)


def test_build_manifest_uses_source_data(monkeypatch, tmp_path):
    root = _scene_root(tmp_path)
    w2c_a, _ = _pose(0.0)
    w2c_b, _ = _pose(2.0)
    intrinsics = CameraIntrinsics(16, 12, 10.0, 10.0, 8.0, 6.0)
    distortion = CameraDistortion("SIMPLE_RADIAL", (0.01,))
    camera = ColmapCameraRecord(1, intrinsics, distortion)
    test_pose = PoseRecord("target.JPG", "target.png", w2c_a, intrinsics)
    source = SceneSourceData(
        scene_id="scene_001",
        train_image_paths=("train/images/a.JPG", "train/images/b.JPG"),
        train_image_names=("a.JPG", "b.JPG"),
        train_images=(
            ColmapImageRecord(1, "a.JPG", 1, w2c_a),
            ColmapImageRecord(2, "b.JPG", 1, w2c_b),
        ),
        test_poses=(test_pose,),
        test_distortions=(distortion,),
        sparse_points=(
            ColmapPointRecord(7, [1, 2, 3], [4, 5, 6], 0.1, (1,)),
        ),
        cameras={1: camera},
    )
    monkeypatch.setattr(
        "bts_nvs.data.manifest.load_scene_source_data", lambda _: source
    )

    manifest = build_scene_manifest(root)

    assert manifest.train_intrinsics == (intrinsics, intrinsics)
    assert manifest.test_image_names == ("target.JPG",)
    assert manifest.test_output_names == ("target.png",)
    np.testing.assert_array_equal(manifest.sparse_colors, [[4, 5, 6]])
    validate_scene_manifest(manifest, root)
