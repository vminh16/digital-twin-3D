import json

import numpy as np
import pytest
from PIL import Image

from bts_nvs.evaluation.evaluator import (
    EvaluationError,
    evaluate_benchmark,
    load_image_pairs,
    save_metric_report,
)
from bts_nvs.evaluation.metrics import MetricConfig


class ConstantLpips:
    package = "fake-lpips"
    version = "test"
    device = "cpu"
    dtype = "float32"

    def __init__(self, value=0.0):
        self.value = value
        self.calls = 0

    def __call__(self, prediction, target):
        self.calls += 1
        return self.value


def test_benchmark_averages_images_then_scenes_and_serializes_standard_json(tmp_path):
    image = np.zeros((16, 16, 3), dtype=np.float64)
    report = evaluate_benchmark(
        {"scene_b": {"b.png": (image, image)}, "scene_a": {"a.png": (image, image)}},
        MetricConfig(psnr_max=40.0),
        ConstantLpips(),
    )
    output = tmp_path / "metrics.json"

    save_metric_report(report, output)
    encoded = output.read_text(encoding="utf-8")
    loaded = json.loads(encoded)

    assert loaded["final_score"] == pytest.approx(1.0)
    assert list(loaded["scenes"]) == ["scene_a", "scene_b"]
    assert "Infinity" not in encoded and "NaN" not in encoded
    assert loaded["metadata"]["ssim_padding"] == "valid"
    assert loaded["metadata"]["lpips"]["package"] == "fake-lpips"


def test_zero_image_scene_fails():
    with pytest.raises(EvaluationError, match="zero images"):
        evaluate_benchmark(
            {"scene": {}}, MetricConfig(psnr_max=40.0), ConstantLpips()
        )


def test_directory_preflight_rejects_missing_file_before_metrics(tmp_path):
    prediction_dir = tmp_path / "prediction"
    reference_dir = tmp_path / "reference"
    prediction_dir.mkdir()
    reference_dir.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
        reference_dir / "a.png"
    )

    with pytest.raises(EvaluationError, match="missing"):
        load_image_pairs(
            {"a.png": (16, 16)}, prediction_dir, reference_dir
        )


def test_directory_preflight_rejects_wrong_resolution(tmp_path):
    prediction_dir = tmp_path / "prediction"
    reference_dir = tmp_path / "reference"
    prediction_dir.mkdir()
    reference_dir.mkdir()
    for directory in (prediction_dir, reference_dir):
        Image.fromarray(np.zeros((15, 16, 3), dtype=np.uint8)).save(
            directory / "a.png"
        )

    with pytest.raises(EvaluationError, match="resolution"):
        load_image_pairs(
            {"a.png": (16, 16)}, prediction_dir, reference_dir
        )
