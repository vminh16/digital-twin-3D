import numpy as np
import pytest

from bts_nvs.evaluation.metrics import MetricConfig, evaluate_image


class RecordingLpips:
    package = "fake-lpips"
    version = "test"
    device = "cpu"
    dtype = "float32"

    def __init__(self, value=0.0):
        self.value = value
        self.inputs = []

    def __call__(self, prediction, target):
        self.inputs.append((prediction, target))
        return self.value


def test_identical_images_have_unit_composite_and_json_safe_psnr():
    image = np.linspace(0.0, 1.0, 16 * 16 * 3).reshape(16, 16, 3)
    backend = RecordingLpips(0.0)

    result = evaluate_image(image, image, MetricConfig(psnr_max=40.0), backend)

    assert result["psnr_db"] is None
    assert result["psnr_is_infinite"] is True
    assert result["psnr_normalized"] == 1.0
    assert result["ssim"] == pytest.approx(1.0)
    assert result["lpips"] == 0.0
    assert result["composite"] == pytest.approx(1.0)
    prediction_input, target_input = backend.inputs[0]
    assert prediction_input.dtype == np.float32
    assert prediction_input.shape == (1, 3, 16, 16)
    assert prediction_input.min() >= -1.0 and prediction_input.max() <= 1.0
    np.testing.assert_array_equal(prediction_input, target_input)


def test_controlled_difference_uses_per_image_clipped_psnr():
    prediction = np.zeros((16, 16, 3), dtype=np.float64)
    target = np.ones((16, 16, 3), dtype=np.float64)

    result = evaluate_image(
        prediction,
        target,
        MetricConfig(psnr_max=40.0),
        RecordingLpips(0.5),
    )

    assert result["psnr_db"] == pytest.approx(0.0)
    assert result["psnr_normalized"] == 0.0
    assert result["ssim"] < 0.001
    assert result["composite"] == pytest.approx(
        0.4 * 0.5 + 0.3 * result["ssim"]
    )


@pytest.mark.parametrize("lpips_value", [np.nan, -0.1, 1.1])
def test_invalid_lpips_result_is_rejected(lpips_value):
    image = np.zeros((16, 16, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="LPIPS"):
        evaluate_image(
            image,
            image,
            MetricConfig(psnr_max=40.0),
            RecordingLpips(lpips_value),
        )


def test_metric_input_must_be_finite_rgb_in_unit_range():
    image = np.zeros((16, 16, 3), dtype=np.float64)
    image[0, 0, 0] = 2.0

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_image(image, image, MetricConfig(psnr_max=40.0), RecordingLpips())
