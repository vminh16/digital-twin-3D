import os

import pytest

from bts_nvs.training.spectral_preflight import run_spectral_density_preflight


@pytest.mark.skipif(
    os.environ.get("BTS_RUN_SPECTRAL_SMOKE") != "1",
    reason="set BTS_RUN_SPECTRAL_SMOKE=1 on the NVIDIA L4",
)
def test_real_spectral_cuda_split() -> None:
    run_spectral_density_preflight("adam-fused")
