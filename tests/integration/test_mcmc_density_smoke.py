from __future__ import annotations

import os

import pytest

from bts_nvs.training.mcmc_preflight import run_mcmc_density_preflight


@pytest.mark.skipif(
    os.environ.get("BTS_RUN_MCMC_SMOKE") != "1",
    reason="set BTS_RUN_MCMC_SMOKE=1 on the NVIDIA L4",
)
def test_real_mcmc_cuda_relocation_and_growth() -> None:
    run_mcmc_density_preflight("adam-fused")
