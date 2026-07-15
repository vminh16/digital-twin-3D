from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.holdout import (
    ALGORITHM,
    _pose_distance_matrix,
    build_pose_holdout,
    load_holdout_split,
    manifest_pose_distance_matrix,
    manifest_holdout_sha256,
    save_holdout_split,
)
from bts_nvs.data.manifest import SceneManifest
from bts_nvs.data.validation import DataContractError


def _manifest(count: int = 150) -> SceneManifest:
    names = tuple(f"image_{index:03d}.JPG" for index in range(count))
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    c2w[:, 0, 3] = np.cos(angles)
    c2w[:, 1, 3] = np.sin(angles)
    c2w[:, 2, 3] = np.linspace(-0.5, 0.5, count)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    c2w[:, 0, 0] = cosines
    c2w[:, 0, 2] = sines
    c2w[:, 2, 0] = -sines
    c2w[:, 2, 2] = cosines
    intrinsics = CameraIntrinsics(8, 6, 5.0, 5.0, 4.0, 3.0)
    distortion = CameraDistortion("PINHOLE", ())
    return SceneManifest(
        schema_version=1,
        scene_id="scene",
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


def test_pose_distance_combines_normalized_translation_and_angle() -> None:
    centers = np.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    axes = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    distances = _pose_distance_matrix(centers, axes)

    assert distances[0, 1] == pytest.approx(5.25)
    assert distances[1, 0] == pytest.approx(5.25)
    assert distances[0, 0] == pytest.approx(0.0)


def test_manifest_pose_distances_are_sorted_and_symmetric() -> None:
    manifest = _manifest()

    names, distances = manifest_pose_distance_matrix(manifest)

    assert names == tuple(sorted(manifest.train_image_names))
    assert distances.shape == (150, 150)
    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_array_equal(np.diag(distances), np.zeros(150))
    assert not distances.flags.writeable


def test_pose_holdout_satisfies_partition_and_minimums() -> None:
    manifest = _manifest()

    split = build_pose_holdout(manifest)

    train = set(split.train_image_names)
    validation = set(split.validation_image_names)
    guard = set(split.guard_image_names)
    assert split.algorithm == ALGORITHM
    assert len(validation) >= 8
    assert len(train) >= 120
    assert len(train) >= int(np.ceil(0.70 * len(manifest.train_image_names)))
    assert train.isdisjoint(validation)
    assert train.isdisjoint(guard)
    assert validation.isdisjoint(guard)
    assert train | validation | guard == set(manifest.train_image_names)
    assert not (train | validation | guard) & set(manifest.test_image_names)


def test_holdout_is_order_independent_and_rejects_official_test_overlap() -> None:
    manifest = _manifest()
    order = np.arange(len(manifest.train_image_names))[::-1]
    reordered = replace(
        manifest,
        train_image_paths=tuple(manifest.train_image_paths[index] for index in order),
        train_image_names=tuple(manifest.train_image_names[index] for index in order),
        train_world_to_camera=manifest.train_world_to_camera[order],
        train_camera_to_world=manifest.train_camera_to_world[order],
        train_intrinsics=tuple(manifest.train_intrinsics[index] for index in order),
        train_distortion=tuple(manifest.train_distortion[index] for index in order),
    )

    assert build_pose_holdout(reordered) == build_pose_holdout(manifest)
    assert manifest_holdout_sha256(reordered) == manifest_holdout_sha256(manifest)
    changed_test_pose = manifest.test_world_to_camera.copy()
    changed_test_pose[0, 0, 3] = 99.0
    changed_test = replace(
        manifest,
        test_image_names=("other_target.JPG",),
        test_world_to_camera=changed_test_pose,
    )
    assert build_pose_holdout(changed_test) == build_pose_holdout(manifest)
    with pytest.raises(DataContractError, match="official test"):
        build_pose_holdout(
            replace(manifest, test_image_names=(manifest.train_image_names[0],))
        )


def test_holdout_round_trip_is_canonical_and_detects_manifest_change(tmp_path) -> None:
    manifest = _manifest()
    split = build_pose_holdout(manifest)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_holdout_split(split, first)
    save_holdout_split(split, second)

    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert load_holdout_split(first, manifest) == split

    changed_c2w = manifest.train_camera_to_world.copy()
    changed_c2w[0, 0, 3] += 0.1
    changed = replace(
        manifest,
        train_camera_to_world=changed_c2w,
        train_world_to_camera=np.linalg.inv(changed_c2w),
    )
    with pytest.raises(DataContractError, match="hash"):
        load_holdout_split(first, changed)

    changed_paths = replace(
        manifest,
        train_image_paths=("train/images/replaced.JPG",) + manifest.train_image_paths[1:],
    )
    with pytest.raises(DataContractError, match="hash"):
        load_holdout_split(first, changed_paths)


def test_holdout_load_rejects_tampered_partition(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "holdout.json"
    save_holdout_split(build_pose_holdout(manifest), path)
    text = path.read_text(encoding="utf-8")
    text = text.replace('"schema_version": 1', '"schema_version": 2')
    path.write_text(text, encoding="utf-8", newline="\n")

    with pytest.raises(DataContractError, match="schema"):
        load_holdout_split(path, manifest)

    split = build_pose_holdout(manifest)
    save_holdout_split(split, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_image_names"][0], payload["validation_image_names"][0] = (
        payload["validation_image_names"][0],
        payload["train_image_names"][0],
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(DataContractError, match="deterministic algorithm"):
        load_holdout_split(path, manifest)
