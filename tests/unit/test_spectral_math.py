from __future__ import annotations

import math

import pytest
import torch

from bts_nvs.rendering.spectral_math import (
    shape_aware_child_scales,
    spectral_entropy_and_condition,
)


def test_isotropic_gaussian_has_maximum_entropy_and_unit_condition() -> None:
    entropy, condition = spectral_entropy_and_condition(
        torch.tensor([[2.0, 2.0, 2.0]])
    )

    assert entropy.item() == pytest.approx(math.log(3.0))
    assert condition.item() == pytest.approx(1.0)


def test_elongation_reduces_entropy_and_increases_condition() -> None:
    entropy, condition = spectral_entropy_and_condition(
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.1, 0.1]])
    )

    assert entropy[1] < entropy[0]
    assert condition[1] > condition[0]


def test_shape_aware_split_reduces_principal_axis_condition() -> None:
    parents = torch.tensor([[1.0, 0.1, 0.05], [0.2, 0.2, 0.01]])
    children = shape_aware_child_scales(parents, k=0.6, k0=1.0)
    _, parent_condition = spectral_entropy_and_condition(parents)
    _, child_condition = spectral_entropy_and_condition(children)

    assert torch.all(child_condition <= parent_condition)
    assert children[0, 0] == pytest.approx(torch.tensor(0.625))
    assert children[0, 1] == parents[0, 1]


def test_near_zero_scales_remain_finite_and_empty_input_is_supported() -> None:
    entropy, condition = spectral_entropy_and_condition(
        torch.tensor([[1e-30, 1e-30, 1e-30]])
    )
    empty_entropy, empty_condition = spectral_entropy_and_condition(
        torch.empty((0, 3))
    )

    assert torch.isfinite(entropy).all()
    assert torch.isfinite(condition).all()
    assert empty_entropy.shape == (0,)
    assert empty_condition.shape == (0,)


@pytest.mark.parametrize(
    "scales",
    [
        torch.ones(3),
        torch.ones((2, 2)),
        torch.tensor([[1.0, -1.0, 1.0]]),
        torch.tensor([[1.0, math.inf, 1.0]]),
    ],
)
def test_spectral_math_rejects_invalid_scales(scales: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError)):
        spectral_entropy_and_condition(scales)
