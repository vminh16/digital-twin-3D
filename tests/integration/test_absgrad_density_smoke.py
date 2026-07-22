from __future__ import annotations

import math
import os

import pytest
import torch

from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.experiments.candidates import candidate_settings
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering.density_strategy import GsplatStrategy
from bts_nvs.rendering.gsplat_renderer import render_gaussians
from bts_nvs.training.precision import TrainingPrecision


@pytest.mark.skipif(
    os.environ.get("BTS_RUN_ABSGRAD_SMOKE") != "1",
    reason="set BTS_RUN_ABSGRAD_SMOKE=1 on the NVIDIA L4",
)
def test_real_absgrad_reaches_first_density_event() -> None:
    if not torch.cuda.is_available():
        pytest.fail("BTS_RUN_ABSGRAD_SMOKE=1 requires CUDA-enabled PyTorch")

    device = torch.device("cuda")
    settings = candidate_settings("E1-density-absgrad-t04-v1")
    gaussians = GaussianParameters(
        means=torch.tensor([[0.0, 0.0, 5.0]], device=device),
        scales=torch.full((1, 3), math.log(0.1), device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        opacities=torch.zeros(1, device=device),
        sh0=torch.zeros((1, 1, 3), device=device),
        shN=torch.zeros((1, 15, 3), device=device),
    )
    optimizers = setup_optimizers(gaussians, backend="adam-fused")
    precision = TrainingPrecision("amp-fp16", device)
    strategy = GsplatStrategy(
        gaussians,
        optimizers,
        prune_opa=settings.prune_opa,
        grow_grad2d=settings.grow_grad2d,
        grow_scale3d=settings.grow_scale3d,
        refine_stop_step=settings.refine_stop_step,
        absgrad=settings.absgrad,
    )
    strategy_state = strategy.initialize_state(scene_scale=1.0)
    intrinsics = CameraIntrinsics(32, 32, 24.0, 24.0, 16.0, 16.0)
    target = torch.ones((32, 32, 3), dtype=torch.float32, device=device)
    event_absgrad: torch.Tensor | None = None

    for step in range(1, 501):
        with precision.autocast():
            result = render_gaussians(
                gaussians=gaussians,
                viewmat=torch.eye(4, device=device),
                intrinsics=intrinsics,
                active_sh_degree=0,
                absgrad=settings.absgrad,
                rasterize_mode=settings.rasterize_mode,
            )
            loss = (result.rgb - target).square().mean()
        strategy.step_pre_backward(strategy_state, step=step, info=result.info)
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        means2d = result.info.get("means2d")
        assert isinstance(means2d, torch.Tensor)
        precision.backward_and_unscale(loss, optimizers, means2d)
        absolute_gradient = getattr(means2d, "absgrad", None)
        assert isinstance(absolute_gradient, torch.Tensor)
        assert torch.isfinite(absolute_gradient).all()
        if step == 500:
            event_absgrad = absolute_gradient.detach().clone()
        strategy.step_post_backward(
            strategy_state,
            step=step,
            info=result.info,
            packed=True,
        )
        precision.step(optimizers)

    torch.cuda.synchronize(device)
    assert event_absgrad is not None and torch.isfinite(event_absgrad).all()
    assert isinstance(strategy_state["grad2d"], torch.Tensor)
    assert torch.isfinite(strategy_state["grad2d"]).all()
    assert gaussians.num_gaussians > 0
