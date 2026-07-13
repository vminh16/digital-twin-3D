import numpy as np
import pytest

from bts_nvs.cameras.distortion import (
    CameraDistortion,
    DistortionConvergenceError,
    distort_normalized_points,
    undistort_normalized_points,
)


def test_pinhole_distortion_is_identity():
    points = np.array([[0.0, 0.0], [0.4, -0.2]], dtype=np.float64)
    distortion = CameraDistortion(model="PINHOLE", coefficients=())

    np.testing.assert_array_equal(distort_normalized_points(points, distortion), points)
    np.testing.assert_array_equal(undistort_normalized_points(points, distortion), points)


def test_simple_radial_round_trip_at_high_distortion_corner():
    distortion = CameraDistortion(
        model="SIMPLE_RADIAL",
        coefficients=(-0.11479405370758,),
    )
    points = np.array(
        [[0.0, 0.0], [660.0 / 925.477, 494.5 / 925.477], [-0.65, 0.5]],
        dtype=np.float64,
    )

    distorted = distort_normalized_points(points, distortion)
    recovered = undistort_normalized_points(distorted, distortion)

    np.testing.assert_allclose(recovered, points, atol=1e-10)
    np.testing.assert_allclose(
        distort_normalized_points(recovered, distortion),
        distorted,
        atol=1e-10,
    )


def test_simple_radial_reports_non_convergence():
    distortion = CameraDistortion(model="SIMPLE_RADIAL", coefficients=(-1.0,))
    with pytest.raises(DistortionConvergenceError):
        undistort_normalized_points(
            np.array([[1.0, 0.0]], dtype=np.float64),
            distortion,
            max_iterations=2,
        )


@pytest.mark.parametrize(
    ("model", "coefficients"),
    [
        ("PINHOLE", (0.1,)),
        ("SIMPLE_RADIAL", ()),
        ("SIMPLE_RADIAL", (0.1, 0.2)),
        ("OPENCV", (0.1,)),
    ],
)
def test_distortion_contract_rejects_invalid_models_and_coefficients(model, coefficients):
    with pytest.raises(ValueError):
        CameraDistortion(model=model, coefficients=coefficients)
