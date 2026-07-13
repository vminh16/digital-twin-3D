from pathlib import Path

import numpy as np
from PIL import Image

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.dataset import CameraSample, SceneDataset
from bts_nvs.data.manifest import SceneManifest


def _dataset(tmp_path: Path, *, undistort=False, resize=None):
    root = tmp_path / "scene"
    image_dir = root / "train" / "images"
    image_dir.mkdir(parents=True)
    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(8, dtype=np.uint8)
    Image.fromarray(pixels).save(image_dir / "a.png")
    identity = np.eye(4, dtype=np.float64)
    intrinsics = CameraIntrinsics(8, 6, 5.0, 5.0, 4.0, 3.0)
    distortion = CameraDistortion("SIMPLE_RADIAL", (0.5,))
    manifest = SceneManifest(
        1,
        "scene",
        ("train/images/a.png",),
        ("a.png",),
        identity[None],
        identity[None],
        (intrinsics,),
        (distortion,),
        ("target.JPG",),
        ("target.png",),
        identity[None],
        (intrinsics,),
        (distortion,),
        np.empty((0, 3)),
        np.empty((0, 3), dtype=np.uint8),
        identity,
        identity,
    )
    return SceneDataset(manifest, root, undistort=undistort, resize=resize), manifest


def test_raw_dataset_sample_preserves_camera_contract(tmp_path):
    dataset, manifest = _dataset(tmp_path)

    sample = dataset[0]

    assert isinstance(sample, CameraSample)
    assert sample.image.shape == (6, 8, 3)
    assert sample.image_name == "a.png"
    assert sample.intrinsics == manifest.train_intrinsics[0]
    assert sample.distortion == manifest.train_distortion[0]
    assert sample.valid_mask.dtype == np.bool_
    assert sample.valid_mask.all()


def test_resize_scales_intrinsics_and_mask_with_nearest_neighbor(tmp_path):
    dataset, _ = _dataset(tmp_path, resize=(4, 3))

    sample = dataset[0]

    assert sample.image.shape == (3, 4, 3)
    assert sample.valid_mask.shape == (3, 4)
    assert sample.intrinsics == CameraIntrinsics(4, 3, 2.5, 2.5, 2.0, 1.5)


def test_undistort_then_resize_returns_zero_distortion_and_valid_mask(tmp_path):
    dataset, _ = _dataset(tmp_path, undistort=True, resize=(4, 3))

    sample = dataset[0]

    assert sample.intrinsics == CameraIntrinsics(4, 3, 2.5, 2.5, 2.0, 1.5)
    assert sample.distortion == CameraDistortion("PINHOLE", ())
    assert sample.valid_mask.dtype == np.bool_
    assert not sample.valid_mask.all()


def test_dataset_does_not_mutate_manifest_arrays(tmp_path):
    dataset, manifest = _dataset(tmp_path, undistort=True, resize=(4, 3))
    before = manifest.train_world_to_camera.copy()

    sample = dataset[0]
    sample.world_to_camera[0, 3] = 99.0

    np.testing.assert_array_equal(manifest.train_world_to_camera, before)


def test_dataset_rejects_image_resolution_that_disagrees_with_intrinsics(tmp_path):
    dataset, _ = _dataset(tmp_path)
    Image.fromarray(np.zeros((6, 7, 3), dtype=np.uint8)).save(
        tmp_path / "scene" / "train" / "images" / "a.png"
    )

    try:
        dataset[0]
    except ValueError as error:
        assert "resolution" in str(error)
    else:
        raise AssertionError("resolution mismatch was accepted")
