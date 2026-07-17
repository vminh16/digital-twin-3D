from __future__ import annotations

import os

import pytest

from bts_nvs.training.run_training import run_cuda_preflight


@pytest.mark.skipif(
    os.environ.get("BTS_RUN_CUDA_BACKEND_SMOKE") != "1",
    reason="set BTS_RUN_CUDA_BACKEND_SMOKE=1 on the NVIDIA L4",
)
@pytest.mark.parametrize(
    "backend,precision",
    [
        ("adam", "fp32"),
        ("adam-fused", "fp32"),
        ("adam-fused", "amp-fp16"),
    ],
)
def test_real_gsplat_gradient_path(backend: str, precision: str) -> None:
    run_cuda_preflight(backend, precision)

