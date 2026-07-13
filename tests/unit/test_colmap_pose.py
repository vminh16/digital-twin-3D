import numpy as np
import pytest

from bts_nvs.cameras.poses import (
    camera_center_from_world_to_camera,
    compute_scene_normalization,
    invert_rigid_transform,
    invert_similarity_transform,
    normalize_camera_poses,
    normalize_points,
    qvec_wxyz_to_rotation_matrix,
    world_to_camera_from_qt,
)


def _project(k: np.ndarray, w2c: np.ndarray, point: np.ndarray) -> np.ndarray:
    camera_point = w2c[:3, :3] @ point + w2c[:3, 3]
    pixel = k @ camera_point
    return pixel[:2] / pixel[2]


def test_qvec_wxyz_converts_known_rotation_and_normalizes_small_drift():
    angle = np.pi / 2
    qvec = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    rotation = qvec_wxyz_to_rotation_matrix(qvec * 1.0001)

    np.testing.assert_allclose(
        rotation,
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-12,
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "qvec",
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.01, 0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0, 0.0],
    ],
)
def test_qvec_wxyz_rejects_invalid_quaternions(qvec):
    with pytest.raises(ValueError):
        qvec_wxyz_to_rotation_matrix(qvec)


def test_world_to_camera_inverse_and_camera_center_follow_colmap_convention():
    angle = np.pi / 3
    qvec = [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
    tvec = np.array([1.5, -2.0, 3.0])

    w2c = world_to_camera_from_qt(qvec, tvec)
    c2w = invert_rigid_transform(w2c)
    center = camera_center_from_world_to_camera(w2c)

    np.testing.assert_allclose(w2c @ c2w, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(center, -w2c[:3, :3].T @ tvec, atol=1e-12)
    np.testing.assert_allclose(c2w[:3, 3], center, atol=1e-12)


def test_rigid_inverse_rejects_similarity_transform():
    similarity = np.diag([2.0, 2.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="rigid"):
        invert_rigid_transform(similarity)


def test_rigid_inverse_rejects_near_unit_similarity_transform():
    similarity = np.diag([1.000001, 1.000001, 1.000001, 1.0])
    with pytest.raises(ValueError, match="rigid"):
        invert_rigid_transform(similarity)


@pytest.mark.parametrize(
    ("center", "scale", "transform", "inverse"),
    [
        (
            np.array([np.nan, 0.0, 0.0]),
            1.0,
            np.eye(4),
            np.eye(4),
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            2.0,
            np.eye(4),
            np.eye(4),
        ),
    ],
)
def test_scene_normalization_rejects_inconsistent_state(
    center, scale, transform, inverse
):
    from bts_nvs.cameras.poses import SceneNormalization

    with pytest.raises(ValueError):
        SceneNormalization(
            center=center,
            scale=scale,
            transform=transform,
            inverse_transform=inverse,
        )


def test_normalization_has_unit_p90_radius_and_preserves_projection():
    centers = np.array(
        [
            [-3.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [4.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(centers), axis=0)
    c2w[:, :3, 3] = centers

    normalization = compute_scene_normalization(c2w)
    normalized_c2w = normalize_camera_poses(c2w, normalization)
    normalized_centers = normalized_c2w[:, :3, 3]

    radius_p90 = np.percentile(np.linalg.norm(normalized_centers, axis=1), 90)
    assert radius_p90 == pytest.approx(1.0)
    np.testing.assert_allclose(normalized_c2w[:, :3, :3], c2w[:, :3, :3])
    np.testing.assert_allclose(
        normalization.transform @ normalization.inverse_transform,
        np.eye(4),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        invert_similarity_transform(normalization.transform),
        normalization.inverse_transform,
        atol=1e-12,
    )
    assert not normalization.center.flags.writeable
    assert not normalization.transform.flags.writeable
    assert not normalization.inverse_transform.flags.writeable

    point_world = np.array([0.2, -0.3, 5.0])
    point_normalized = normalize_points(point_world[None], normalization)[0]
    raw_w2c = invert_rigid_transform(c2w[0])
    normalized_w2c = invert_rigid_transform(normalized_c2w[0])
    k = np.array([[900.0, 0.0, 660.0], [0.0, 900.0, 494.5], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        _project(k, raw_w2c, point_world),
        _project(k, normalized_w2c, point_normalized),
        atol=1e-10,
    )


def test_normalization_rejects_zero_radius_trajectory():
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    with pytest.raises(ValueError, match="radius"):
        compute_scene_normalization(c2w)


def test_normalization_supports_large_world_coordinate_origin():
    centers = np.array(
        [
            [1.0e9 - 3.0, -1.0e9, 5.0e8],
            [1.0e9 - 1.0, -1.0e9 + 1.0, 5.0e8],
            [1.0e9 + 1.0, -1.0e9 - 1.0, 5.0e8],
            [1.0e9 + 4.0, -1.0e9 + 0.5, 5.0e8],
        ],
        dtype=np.float64,
    )
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(centers), axis=0)
    c2w[:, :3, 3] = centers

    normalization = compute_scene_normalization(c2w)
    normalized = normalize_camera_poses(c2w, normalization)

    assert np.percentile(np.linalg.norm(normalized[:, :3, 3], axis=1), 90) == pytest.approx(1.0)
