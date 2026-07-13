import numpy as np
import pytest

from bts_nvs.cameras.intrinsics import CameraIntrinsics


def test_intrinsics_matrix_and_non_uniform_resize():
    intrinsics = CameraIntrinsics(
        width=1320,
        height=989,
        fx=926.0,
        fy=925.0,
        cx=660.0,
        cy=494.5,
    )

    resized = intrinsics.resized(width=660, height=989)

    np.testing.assert_allclose(
        intrinsics.matrix,
        [[926.0, 0.0, 660.0], [0.0, 925.0, 494.5], [0.0, 0.0, 1.0]],
    )
    assert resized.width == 660
    assert resized.height == 989
    assert resized.fx == pytest.approx(463.0)
    assert resized.fy == pytest.approx(925.0)
    assert resized.cx == pytest.approx(330.0)
    assert resized.cy == pytest.approx(494.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0, "height": 10, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"width": 10, "height": 10, "fx": -1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        {"width": 10, "height": 10, "fx": 1.0, "fy": np.nan, "cx": 0.0, "cy": 0.0},
    ],
)
def test_intrinsics_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CameraIntrinsics(**kwargs)
