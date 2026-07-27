from __future__ import annotations

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.evaluation.gaussian_diagnostics import build_gaussian_diagnostics
from bts_nvs.models.gaussian_parameters import GaussianParameters


@pytest.mark.cuda
@pytest.mark.skipif(
    os.environ.get("BTS_RUN_RESEARCH_DIAGNOSTICS_SMOKE") != "1",
    reason="set BTS_RUN_RESEARCH_DIAGNOSTICS_SMOKE=1 on the NVIDIA L4",
)
def test_real_research_radius_diagnostics_and_filtered_render(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.fail("research diagnostics smoke requires CUDA-enabled PyTorch")
    device = torch.device("cuda")
    gaussians = GaussianParameters(
        means=torch.tensor(
            [[0.0, 0.0, 5.0], [0.2, 0.0, 5.0]],
            device=device,
        ),
        scales=torch.full((2, 3), math.log(0.1), device=device),
        quats=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            device=device,
        ),
        opacities=torch.zeros(2, device=device),
        sh0=torch.zeros((2, 1, 3), device=device),
        shN=torch.zeros((2, 15, 3), device=device),
    )
    intrinsics = CameraIntrinsics(32, 32, 24.0, 24.0, 16.0, 16.0)
    sample = SimpleNamespace(
        image_name="frame_000390.jpg",
        world_to_camera=np.eye(4, dtype=np.float64),
        intrinsics=intrinsics,
    )
    manifest = SimpleNamespace(
        scene_id="bonsai",
        normalization_transform=np.eye(4, dtype=np.float64),
    )

    class Dataset:
        def __init__(self):
            self.manifest = manifest

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return sample

    report = build_gaussian_diagnostics(
        SimpleNamespace(
            gaussians=gaussians,
            device=device,
            active_sh_degree=0,
        ),
        Dataset(),
        tmp_path,
        scale_threshold=0.05,
        radius_threshold_pixels=1.0,
        density_summary={
            "event_count": 0,
            "observed_net_added": 0,
            "observed_net_removed": 0,
            "net_change": 0,
        },
    )

    assert report["diagnostic_only"] is True
    assert report["projected_radius"]["projected_radius_max"] > 0.0
    assert (tmp_path / "frame_000390.png").is_file()
