from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import bts_nvs.data.dataset as dataset_module
from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.dataset import CameraSample, SceneDataset
from bts_nvs.data.manifest import SceneManifest


def _dataset(tmp_path: Path, *, undistort=False, resize=None, cache_images=False):
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
    return (
        SceneDataset(
            manifest,
            root,
            undistort=undistort,
            resize=resize,
            cache_images=cache_images,
        ),
        manifest,
    )


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


def test_cached_and_uncached_samples_are_bit_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "available_host_ram_bytes", lambda: 8 * 1024**3)
    uncached, _ = _dataset(tmp_path / "uncached", undistort=True, resize=(4, 3))
    cached, _ = _dataset(
        tmp_path / "cached", undistort=True, resize=(4, 3), cache_images=True
    )

    expected = uncached[0]
    actual = cached[0]

    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.valid_mask, expected.valid_mask)
    np.testing.assert_array_equal(actual.world_to_camera, expected.world_to_camera)
    assert actual.intrinsics == expected.intrinsics
    assert actual.distortion == expected.distortion
    assert actual.image.flags.c_contiguous
    assert actual.image.dtype == np.uint8
    assert actual.valid_mask.flags.c_contiguous
    assert actual.valid_mask.dtype == np.bool_


def test_cache_decodes_once_and_returned_samples_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "available_host_ram_bytes", lambda: 8 * 1024**3)
    original_open = dataset_module.Image.open
    opened = 0

    def counted_open(*args, **kwargs):
        nonlocal opened
        opened += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(dataset_module.Image, "open", counted_open)
    dataset, _ = _dataset(tmp_path, cache_images=True)
    first = dataset[0]
    first.image[:] = 255
    first.valid_mask[:] = False
    second = dataset[0]

    assert opened == 1
    assert not np.all(second.image == 255)
    assert second.valid_mask.all()


def test_cache_rejects_insufficient_host_ram_before_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "available_host_ram_bytes", lambda: 0)
    opened = False

    def reject_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("image must not be decoded before RAM preflight")

    monkeypatch.setattr(dataset_module.Image, "open", reject_open)

    with pytest.raises(MemoryError, match="4 GiB"):
        _dataset(tmp_path, cache_images=True)
    assert opened is False


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


def test_dataset_subset_preserves_requested_order_and_excludes_other_images(tmp_path):
    _, manifest = _dataset(tmp_path)
    image_dir = tmp_path / "scene" / "train" / "images"
    for name, value in (("b.png", 20), ("c.png", 30)):
        Image.fromarray(np.full((6, 8, 3), value, dtype=np.uint8)).save(
            image_dir / name
        )
    manifest = replace(
        manifest,
        train_image_paths=(
            "train/images/a.png",
            "train/images/b.png",
            "train/images/c.png",
        ),
        train_image_names=("a.png", "b.png", "c.png"),
        train_world_to_camera=np.repeat(manifest.train_world_to_camera, 3, axis=0),
        train_camera_to_world=np.repeat(manifest.train_camera_to_world, 3, axis=0),
        train_intrinsics=manifest.train_intrinsics * 3,
        train_distortion=manifest.train_distortion * 3,
    )

    dataset = SceneDataset(
        manifest,
        tmp_path / "scene",
        image_names=("c.png", "a.png"),
    )

    assert len(dataset) == 2
    assert [dataset[index].image_name for index in range(len(dataset))] == [
        "c.png",
        "a.png",
    ]


def test_dataset_subset_rejects_duplicate_unknown_and_test_names(tmp_path):
    _, manifest = _dataset(tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        SceneDataset(manifest, tmp_path / "scene", image_names=("a.png", "a.png"))
    with pytest.raises(ValueError, match="unknown"):
        SceneDataset(manifest, tmp_path / "scene", image_names=("missing.png",))
    with pytest.raises(ValueError, match="unknown"):
        SceneDataset(manifest, tmp_path / "scene", image_names=("target.JPG",))
