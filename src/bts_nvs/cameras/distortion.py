from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


class DistortionConvergenceError(ValueError):
    def __init__(self, point_indices: tuple[int, ...]) -> None:
        self.point_indices = point_indices
        super().__init__(f"distortion inversion did not converge for point indices {point_indices}")


@dataclass(frozen=True)
class CameraDistortion:
    model: Literal["PINHOLE", "SIMPLE_RADIAL"]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        coefficients = tuple(float(value) for value in self.coefficients)
        if not np.all(np.isfinite(np.asarray(coefficients, dtype=np.float64))):
            raise ValueError("distortion coefficients must be finite")
        if self.model == "PINHOLE":
            if coefficients:
                raise ValueError("PINHOLE requires zero distortion coefficients")
        elif self.model == "SIMPLE_RADIAL":
            if len(coefficients) != 1:
                raise ValueError("SIMPLE_RADIAL requires exactly one coefficient")
        else:
            raise ValueError(f"unsupported camera model: {self.model}")
        object.__setattr__(self, "coefficients", coefficients)


def _points_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 2:
        raise ValueError("normalized points must have shape (..., 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError("normalized points must be finite")
    return array


def distort_normalized_points(
    points: np.ndarray,
    distortion: CameraDistortion,
) -> np.ndarray:
    array = _points_array(points)
    if distortion.model == "PINHOLE":
        return array.copy()

    k1 = distortion.coefficients[0]
    radius_squared = np.sum(array * array, axis=-1, keepdims=True)
    return array * (1.0 + k1 * radius_squared)


def undistort_normalized_points(
    points: np.ndarray,
    distortion: CameraDistortion,
    *,
    max_iterations: int = 10,
    tolerance: float = 1e-10,
) -> np.ndarray:
    array = _points_array(points)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if distortion.model == "PINHOLE":
        return array.copy()

    original_shape = array.shape
    flat = array.reshape(-1, 2)
    distorted_radius = np.linalg.norm(flat, axis=1)
    radius = distorted_radius.copy()
    k1 = distortion.coefficients[0]

    for _ in range(max_iterations):
        residual = radius + k1 * radius**3 - distorted_radius
        if np.all(np.abs(residual) <= tolerance):
            break
        derivative = 1.0 + 3.0 * k1 * radius**2
        invalid = np.flatnonzero(np.abs(derivative) <= 1e-12)
        if invalid.size:
            raise DistortionConvergenceError(tuple(int(index) for index in invalid))
        radius -= residual / derivative

    scale = 1.0 + k1 * radius**2
    invalid_scale = np.flatnonzero(np.abs(scale) <= 1e-12)
    if invalid_scale.size:
        raise DistortionConvergenceError(tuple(int(index) for index in invalid_scale))

    recovered = flat / scale[:, None]
    forward_residual = np.linalg.norm(
        distort_normalized_points(recovered, distortion) - flat,
        axis=1,
    )
    failed = np.flatnonzero(~np.isfinite(forward_residual) | (forward_residual > tolerance))
    if failed.size:
        raise DistortionConvergenceError(tuple(int(index) for index in failed))
    return recovered.reshape(original_shape)
