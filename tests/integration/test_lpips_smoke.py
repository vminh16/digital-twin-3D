import os

import numpy as np
import pytest

from bts_nvs.evaluation.metrics import LpipsBackend, MetricConfig, evaluate_image


@pytest.mark.lpips_smoke
def test_pretrained_lpips_backend_initializes_and_scores_identical_images():
    if os.environ.get("BTS_RUN_LPIPS_SMOKE") != "1":
        pytest.skip("set BTS_RUN_LPIPS_SMOKE=1 after caching pretrained weights")
    image = np.zeros((64, 64, 3), dtype=np.float64)

    result = evaluate_image(
        image,
        image,
        MetricConfig(psnr_max=40.0),
        LpipsBackend("alex", "cpu"),
    )

    assert result["lpips"] == pytest.approx(0.0, abs=1e-7)
