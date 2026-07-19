from __future__ import annotations

import importlib.util

import pytest
import torch

from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.rendering import gsplat_renderer
from bts_nvs.rendering.render_result import RenderResult


def _gaussians(device: torch.device | str = "cpu") -> GaussianParameters:
    return GaussianParameters(
        means=torch.tensor([[0.0, 0.0, 5.0]], device=device),
        scales=torch.full((1, 3), -4.0, device=device),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        opacities=torch.tensor([1.0], device=device),
        sh0=torch.zeros((1, 1, 3), device=device),
        shN=torch.zeros((1, 15, 3), device=device),
    )


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(16, 16, 10.0, 10.0, 8.0, 8.0)


def test_render_result_validates_shapes() -> None:
    rgb = torch.zeros((16, 16, 3))
    alpha = torch.zeros((16, 16, 1))
    depth = torch.zeros((16, 16, 1))

    result = RenderResult(rgb=rgb, alpha=alpha, depth=depth)
    assert result.rgb.shape == (16, 16, 3)

    with pytest.raises(ValueError, match="RGB"):
        RenderResult(rgb=torch.zeros((16, 16)), alpha=alpha)
    with pytest.raises(ValueError, match="Alpha"):
        RenderResult(rgb=rgb, alpha=torch.zeros((16, 16, 2)))
    with pytest.raises(ValueError, match="Depth"):
        RenderResult(rgb=rgb, alpha=alpha, depth=torch.zeros((16, 16)))
    with pytest.raises(ValueError, match="dimensions must match"):
        RenderResult(rgb=torch.zeros((15, 16, 3)), alpha=alpha)


def test_renderer_passes_single_camera_contract_to_gsplat(monkeypatch) -> None:
    captured = {}

    def fake_rasterization(**kwargs):
        captured.update(kwargs)
        signal = sum(
            kwargs[name].sum()
            for name in ("means", "quats", "scales", "opacities", "colors")
        )
        rendered = torch.ones((1, 16, 16, 4), device=signal.device) * signal
        alpha = torch.ones((1, 16, 16, 1), device=signal.device) * signal
        return rendered, alpha, {"means2d": torch.zeros((1, 1, 2))}

    monkeypatch.setattr(gsplat_renderer, "rasterization", fake_rasterization)
    gaussians = _gaussians()
    result = gsplat_renderer.render_gaussians(
        gaussians,
        torch.eye(4, dtype=torch.float64),
        _intrinsics(),
        active_sh_degree=2,
        backgrounds=torch.ones(3, dtype=torch.float64),
        render_mode="RGB+D",
    )

    assert captured["viewmats"].shape == (1, 4, 4)
    assert captured["Ks"].shape == (1, 3, 3)
    assert captured["backgrounds"].shape == (1, 3)
    assert captured["viewmats"].dtype == gaussians.means.dtype
    assert captured["Ks"].dtype == gaussians.means.dtype
    assert captured["backgrounds"].dtype == gaussians.means.dtype
    assert captured["colors"].shape == (1, 16, 3)
    assert torch.equal(captured["colors"], gaussians.get_shs())
    assert captured["quats"] is gaussians.quats
    assert captured["sh_degree"] == 2
    assert captured["packed"] is True
    assert captured["sparse_grad"] is False
    assert captured["absgrad"] is False
    assert captured["rasterize_mode"] == "classic"
    assert result.rgb.shape == (16, 16, 3)
    assert result.alpha.shape == (16, 16, 1)
    assert result.depth is not None and result.depth.shape == (16, 16, 1)
    assert result.info is not None
    assert torch.equal(result.info["means2d"], torch.zeros((1, 1, 2)))

    (result.rgb.sum() + result.alpha.sum() + result.depth.sum()).backward()
    for parameter in gaussians.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_renderer_forwards_absgrad_to_gsplat(monkeypatch) -> None:
    captured = {}

    def fake_rasterization(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros((1, 16, 16, 3)),
            torch.zeros((1, 16, 16, 1)),
            {"means2d": torch.zeros((1, 1, 2))},
        )

    monkeypatch.setattr(gsplat_renderer, "rasterization", fake_rasterization)
    gsplat_renderer.render_gaussians(
        _gaussians(),
        torch.eye(4),
        _intrinsics(),
        active_sh_degree=0,
        absgrad=True,
    )

    assert captured["absgrad"] is True


@pytest.mark.parametrize(
    "viewmat",
    [torch.eye(3), torch.eye(4).repeat(2, 1, 1), torch.full((4, 4), float("nan"))],
)
def test_renderer_rejects_invalid_single_camera_pose(monkeypatch, viewmat) -> None:
    monkeypatch.setattr(gsplat_renderer, "rasterization", lambda **_: None)
    with pytest.raises(ValueError, match="viewmat"):
        gsplat_renderer.render_gaussians(
            _gaussians(), viewmat, _intrinsics(), active_sh_degree=0
        )


@pytest.mark.parametrize("degree", [-1, 4, True])
def test_renderer_rejects_invalid_sh_degree(monkeypatch, degree) -> None:
    monkeypatch.setattr(gsplat_renderer, "rasterization", lambda **_: None)
    with pytest.raises(ValueError, match="active_sh_degree"):
        gsplat_renderer.render_gaussians(
            _gaussians(), torch.eye(4), _intrinsics(), active_sh_degree=degree
        )


@pytest.mark.parametrize("mode", ["D", "ED", "RGB+ED", "invalid"])
def test_renderer_rejects_unsupported_render_mode(monkeypatch, mode) -> None:
    monkeypatch.setattr(gsplat_renderer, "rasterization", lambda **_: None)
    with pytest.raises(ValueError, match="render_mode"):
        gsplat_renderer.render_gaussians(
            _gaussians(),
            torch.eye(4),
            _intrinsics(),
            active_sh_degree=0,
            render_mode=mode,
        )


HAS_REAL_GSPLAT = importlib.util.find_spec("gsplat") is not None


@pytest.mark.skipif(
    not HAS_REAL_GSPLAT or not torch.cuda.is_available(),
    reason="requires gsplat and CUDA",
)
def test_real_gsplat_camera_translation_and_backward() -> None:
    gaussians = _gaussians("cuda")
    identity = gsplat_renderer.render_gaussians(
        gaussians,
        torch.eye(4, device="cuda"),
        _intrinsics(),
        active_sh_degree=3,
        render_mode="RGB+D",
    )
    moved_view = torch.eye(4, device="cuda")
    moved_view[0, 3] = -1.0
    moved = gsplat_renderer.render_gaussians(
        gaussians,
        moved_view,
        _intrinsics(),
        active_sh_degree=3,
    )

    identity_x = torch.argmax(identity.alpha[..., 0]).remainder(16)
    moved_x = torch.argmax(moved.alpha[..., 0]).remainder(16)
    assert moved_x < identity_x

    assert identity.depth is not None
    (identity.rgb.sum() + identity.alpha.sum() + identity.depth.sum()).backward()
    for parameter in gaussians.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
