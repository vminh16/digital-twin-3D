from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.colmap import (
    ColmapCameraRecord,
    ColmapImageRecord,
    ColmapModel,
    ColmapPointRecord,
)
from bts_nvs.data.holdout import build_pose_holdout
from bts_nvs.data.manifest import SceneManifest
from bts_nvs.data.sparse_subset import SparseInitialization, build_split_sparse_initialization


def _manifest(scene_root, count: int = 150) -> SceneManifest:
    names = tuple(f"image_{index:03d}.png" for index in range(count))
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    c2w[:, 0, 3] = np.arange(count, dtype=np.float64)
    intrinsics = CameraIntrinsics(2, 2, 2.0, 2.0, 1.0, 1.0)
    distortion = CameraDistortion("PINHOLE", ())
    image_dir = scene_root / "train" / "images"
    image_dir.mkdir(parents=True)
    for name in names:
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_dir / name)
    return SceneManifest(
        schema_version=1,
        scene_id=scene_root.name,
        train_image_paths=tuple(f"train/images/{name}" for name in names),
        train_image_names=names,
        train_world_to_camera=np.linalg.inv(c2w),
        train_camera_to_world=c2w,
        train_intrinsics=(intrinsics,) * count,
        train_distortion=(distortion,) * count,
        test_image_names=("target.JPG",),
        test_output_names=("target.png",),
        test_world_to_camera=np.eye(4, dtype=np.float64)[None],
        test_intrinsics=(intrinsics,),
        test_distortion=(distortion,),
        sparse_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        sparse_colors=np.asarray([[0, 0, 0]], dtype=np.uint8),
        normalization_transform=np.eye(4, dtype=np.float64),
        inverse_normalization_transform=np.eye(4, dtype=np.float64),
    )


def _save_color(scene_root, name: str, color: tuple[int, int, int]) -> None:
    Image.fromarray(np.full((2, 2, 3), color, dtype=np.uint8)).save(
        scene_root / "train" / "images" / name
    )


def test_sparse_initialization_uses_only_valid_internal_train_observations(
    tmp_path, monkeypatch
) -> None:
    scene_root = tmp_path / "scene"
    manifest = _manifest(scene_root)
    split = build_pose_holdout(manifest)
    train_a, train_b, train_outside = split.train_image_names[:3]
    validation = split.validation_image_names[0]
    _save_color(scene_root, train_a, (10, 20, 30))
    _save_color(scene_root, train_b, (30, 40, 50))
    _save_color(scene_root, validation, (200, 210, 220))

    camera = ColmapCameraRecord(1, manifest.train_intrinsics[0], manifest.train_distortion[0])
    image_ids = {name: index + 1 for index, name in enumerate(manifest.train_image_names)}
    images = {
        image_id: ColmapImageRecord(image_id, name, 1, np.eye(4))
        for name, image_id in image_ids.items()
    }
    images[image_ids[train_a]] = ColmapImageRecord(
        image_ids[train_a], train_a, 1, np.eye(4),
        points2d_xy=np.asarray([[1.5, 1.5], [1.5, 1.5], [1.9, 1.9]]),
        point3d_ids=np.asarray([1, 4, -1]),
    )
    images[image_ids[train_b]] = ColmapImageRecord(
        image_ids[train_b], train_b, 1, np.eye(4),
        points2d_xy=np.asarray([[1.5, 1.5]]), point3d_ids=np.asarray([4]),
    )
    images[image_ids[train_outside]] = ColmapImageRecord(
        image_ids[train_outside], train_outside, 1, np.eye(4),
        points2d_xy=np.asarray([[-2.0, 0.0]]), point3d_ids=np.asarray([3]),
    )
    images[image_ids[validation]] = ColmapImageRecord(
        image_ids[validation], validation, 1, np.eye(4),
        points2d_xy=np.asarray([[1.5, 1.5], [1.5, 1.5]]),
        point3d_ids=np.asarray([1, 2]),
    )
    points = {
        point_id: ColmapPointRecord(
            point_id,
            [float(point_id), 0.0, 0.0],
            [0, 0, 0],
            0.0,
            tuple(image_ids.values()),
        )
        for point_id in (1, 2, 3, 4)
    }
    model = ColmapModel(cameras={1: camera}, images=images, points3d=points)
    monkeypatch.setattr(
        "bts_nvs.data.sparse_subset.read_colmap_model", lambda path: model
    )

    first = build_split_sparse_initialization(manifest, scene_root, split)
    _save_color(scene_root, validation, (1, 2, 3))
    second = build_split_sparse_initialization(manifest, scene_root, split)

    np.testing.assert_array_equal(first.point_ids, [1, 4])
    np.testing.assert_allclose(first.points, [[1, 0, 0], [4, 0, 0]])
    np.testing.assert_array_equal(first.colors, [[10, 20, 30], [20, 30, 40]])
    np.testing.assert_array_equal(second.colors, first.colors)
    assert first.points.dtype == np.float64
    assert first.colors.dtype == np.uint8
    assert not first.colors.flags.writeable


def test_sparse_initialization_rejects_color_cast_overflow() -> None:
    with pytest.raises(ValueError, match="colors"):
        SparseInitialization(
            point_ids=[1],
            points=[[0.0, 0.0, 0.0]],
            colors=[[256.0, 0.0, 0.0]],
        )


def test_continuous_observation_mapping_recovers_noninteger_scale(
    tmp_path, monkeypatch
) -> None:
    scene_root = tmp_path / "chair"
    manifest = _manifest(scene_root)
    intrinsics = CameraIntrinsics(4, 4, 1.0, 1.0, 2.0, 2.0)
    manifest = replace(
        manifest,
        train_intrinsics=(intrinsics,) * len(manifest.train_image_names),
    )
    image = np.fromfunction(
        lambda y, x, channel: x * 10 + y * 20 + channel,
        (4, 4, 3),
        dtype=int,
    ).astype(np.uint8)
    for name in manifest.train_image_names:
        Image.fromarray(image).save(scene_root / "train" / "images" / name)

    split = build_pose_holdout(manifest)
    observed_name = split.train_image_names[0]
    observed_index = manifest.train_image_names.index(observed_name)
    camera_to_world = manifest.train_camera_to_world[observed_index]
    camera_points = np.asarray(
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    world_points = (camera_to_world @ camera_points.T).T[:, :3]

    image_ids = {
        name: index + 1 for index, name in enumerate(manifest.train_image_names)
    }
    images = {
        image_id: ColmapImageRecord(
            image_id,
            name,
            1,
            manifest.train_world_to_camera[index],
        )
        for index, (name, image_id) in enumerate(image_ids.items())
    }
    images[image_ids[observed_name]] = ColmapImageRecord(
        image_ids[observed_name],
        observed_name,
        1,
        manifest.train_world_to_camera[observed_index],
        points2d_xy=np.asarray([[3.0, 3.0], [4.5, 4.5]]),
        point3d_ids=np.asarray([1, 2]),
    )
    points = {
        point_id: ColmapPointRecord(
            point_id,
            xyz,
            image[y, x],
            0.0,
            (image_ids[observed_name],),
        )
        for point_id, xyz, x, y in (
            (1, world_points[0], 2, 2),
            (2, world_points[1], 3, 3),
        )
    }
    model = ColmapModel(
        cameras={
            1: ColmapCameraRecord(
                1,
                intrinsics,
                CameraDistortion("PINHOLE", ()),
            )
        },
        images=images,
        points3d=points,
    )
    monkeypatch.setattr(
        "bts_nvs.data.sparse_subset.read_colmap_model", lambda path: model
    )

    legacy = build_split_sparse_initialization(manifest, scene_root, split)
    corrected = build_split_sparse_initialization(
        manifest,
        scene_root,
        split,
        observation_mapping_mode="continuous-reprojection",
    )

    np.testing.assert_array_equal(corrected.colors, [image[2, 2], image[3, 3]])
    assert not np.array_equal(legacy.colors, corrected.colors)
    diagnostics = corrected.observation_mapping_diagnostics
    assert diagnostics is not None
    assert diagnostics.scale == pytest.approx((1.5, 1.5))
    assert diagnostics.legacy_scale == pytest.approx((2.0, 2.0))
    assert diagnostics.fit_observation_count == 2
    assert diagnostics.mapped_reprojection_p95_px == pytest.approx(0.0)
    assert diagnostics.legacy_reprojection_p95_px > 0.0
    assert diagnostics.selected_color_mae_mean == pytest.approx(0.0)
    assert diagnostics.legacy_color_mae_mean > 0.0
