from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly_float64(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _transform_array(transform: np.ndarray) -> np.ndarray:
    array = np.asarray(transform, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError("transform must be finite")
    if not np.allclose(
        array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0
    ):
        raise ValueError("transform must have homogeneous bottom row")
    return array


def _validate_rotation(rotation: np.ndarray) -> None:
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0
    ):
        raise ValueError("transform is not rigid: rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("transform is not rigid: rotation determinant is not one")


@dataclass(frozen=True)
class SceneNormalization:
    center: np.ndarray
    scale: float
    transform: np.ndarray
    inverse_transform: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("normalization scale must be positive and finite")
        center = _readonly_float64(self.center)
        if center.shape != (3,):
            raise ValueError("normalization center must have shape (3,)")
        if not np.all(np.isfinite(center)):
            raise ValueError("normalization center must be finite")
        transform = _readonly_float64(_transform_array(self.transform))
        inverse = _readonly_float64(_transform_array(self.inverse_transform))

        expected_transform = np.eye(4, dtype=np.float64)
        expected_transform[:3, :3] *= self.scale
        expected_transform[:3, 3] = -self.scale * center
        expected_inverse = np.eye(4, dtype=np.float64)
        expected_inverse[:3, :3] /= self.scale
        expected_inverse[:3, 3] = center
        if not np.allclose(
            transform, expected_transform, atol=1e-12, rtol=0.0
        ):
            raise ValueError("normalization transform is inconsistent with center and scale")
        if not np.allclose(
            inverse, expected_inverse, atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                "inverse normalization transform is inconsistent with center and scale"
            )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "inverse_transform", inverse)


def qvec_wxyz_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(qvec, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must have shape (4,) in qw,qx,qy,qz order")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion norm is too small")
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("quaternion norm differs from one by more than 1e-3")

    qw, qx, qy, qz = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    _validate_rotation(rotation)
    return rotation


def world_to_camera_from_qt(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    translation = np.asarray(tvec, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation must be a finite vector with shape (3,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = qvec_wxyz_to_rotation_matrix(qvec)
    transform[:3, 3] = translation
    return transform


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    array = _transform_array(transform)
    rotation = array[:3, :3]
    _validate_rotation(rotation)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ array[:3, 3]
    return inverse


def camera_center_from_world_to_camera(transform: np.ndarray) -> np.ndarray:
    array = _transform_array(transform)
    rotation = array[:3, :3]
    _validate_rotation(rotation)
    return -rotation.T @ array[:3, 3]


def invert_similarity_transform(transform: np.ndarray) -> np.ndarray:
    array = _transform_array(transform)
    linear = array[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("similarity transform must have positive finite scale")
    rotation = linear / scale
    _validate_rotation(rotation)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T / scale
    inverse[:3, 3] = -(rotation.T / scale) @ array[:3, 3]
    return inverse


def compute_scene_normalization(camera_to_world: np.ndarray) -> SceneNormalization:
    poses = np.asarray(camera_to_world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] == 0:
        raise ValueError("camera_to_world must have shape (N, 4, 4) with N > 0")
    for pose in poses:
        invert_rigid_transform(pose)

    centers = poses[:, :3, 3]
    center = centers.mean(axis=0)
    radii = np.linalg.norm(centers - center, axis=1)
    radius_p90 = float(np.percentile(radii, 90))
    if not np.isfinite(radius_p90) or radius_p90 <= 1e-12:
        raise ValueError("camera trajectory radius must be greater than 1e-12")

    scale = 1.0 / radius_p90
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] /= scale
    inverse[:3, 3] = center
    return SceneNormalization(
        center=center,
        scale=scale,
        transform=transform,
        inverse_transform=inverse,
    )


def normalize_camera_poses(
    camera_to_world: np.ndarray,
    normalization: SceneNormalization,
) -> np.ndarray:
    poses = np.asarray(camera_to_world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("camera_to_world must have shape (N, 4, 4)")
    normalized = poses.copy()
    for pose in poses:
        invert_rigid_transform(pose)
    normalized[:, :3, 3] = normalization.scale * (
        poses[:, :3, 3] - normalization.center
    )
    return normalized


def normalize_points(points: np.ndarray, normalization: SceneNormalization) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite")
    return normalization.scale * (array - normalization.center)
