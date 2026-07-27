from types import SimpleNamespace

import pytest
import torch

from bts_nvs.evaluation.gaussian_diagnostics import (
    _visible_radii,
    summarize_gaussians,
)


def test_gaussian_summary_reports_scale_opacity_tail() -> None:
    gaussians = SimpleNamespace(
        get_scales=lambda: torch.tensor(
            [[0.01, 0.02, 0.03], [0.20, 0.10, 0.05]]
        ),
        get_opacities=lambda: torch.tensor([0.25, 0.75]),
    )

    report = summarize_gaussians(gaussians, scale_threshold=0.1)

    assert report["scale3d_max"] == pytest.approx(0.2)
    assert report["scale_above_threshold_count"] == 1
    assert report["scale_above_threshold_opacity_mass"] == pytest.approx(0.75)
    assert report["scale_above_threshold_opacity_fraction"] == pytest.approx(0.75)


def test_visible_radii_supports_packed_and_dense_renderer_info() -> None:
    ids, radii = _visible_radii(
        {
            "gaussian_ids": torch.tensor([4, 2, 3]),
            "radii": torch.tensor([0.0, 5.0, 7.0]),
        },
        5,
    )
    assert ids.tolist() == [2, 3]
    assert radii.tolist() == [5.0, 7.0]

    ids, radii = _visible_radii({"radii": torch.tensor([0.0, 2.0])}, 2)
    assert ids.tolist() == [1]
    assert radii.tolist() == [2.0]
