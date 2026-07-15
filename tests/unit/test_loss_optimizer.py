from __future__ import annotations

import math

import pytest
import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.loss import JointLoss, MaskedL1Loss, MaskedSSIMLoss
from bts_nvs.models.optimizer import setup_mean_scheduler, setup_optimizers


def _gaussians() -> GaussianParameters:
    return GaussianParameters(
        means=torch.zeros((5, 3)),
        scales=torch.zeros((5, 3)),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 5),
        opacities=torch.zeros(5),
        sh0=torch.zeros((5, 1, 3)),
        shN=torch.zeros((5, 15, 3)),
    )


def test_l1_accepts_dataset_image_and_mask_contract() -> None:
    target = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
    prediction = target.float() / 255.0
    mask = torch.ones((32, 32), dtype=torch.bool)

    torch.testing.assert_close(
        MaskedL1Loss()(prediction, target, mask), torch.tensor(0.0)
    )

    mask[:5, :5] = False
    changed = prediction.clone()
    changed[:5, :5] = 1.0 - changed[:5, :5]
    torch.testing.assert_close(
        MaskedL1Loss()(changed, target, mask),
        torch.tensor(0.0),
    )


def test_ssim_accepts_dataset_image_and_mask_contract() -> None:
    target = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
    prediction = target.float() / 255.0
    mask = torch.ones((32, 32), dtype=torch.bool)

    loss = MaskedSSIMLoss()(prediction, target, mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_joint_loss_ignores_pixels_outside_mask() -> None:
    target = torch.rand((32, 32, 3))
    prediction = target.clone()
    mask = torch.ones((32, 32), dtype=torch.bool)
    mask[:8, :] = False

    changed = prediction.clone()
    changed[:8, :] = 1.0 - changed[:8, :]

    torch.testing.assert_close(
        JointLoss()(changed, target, mask),
        JointLoss()(prediction, target, mask),
        atol=1e-6,
        rtol=0.0,
    )


def test_losses_reject_empty_supervision() -> None:
    prediction = torch.rand((16, 16, 3))
    target = torch.rand((16, 16, 3))
    empty = torch.zeros((16, 16), dtype=torch.bool)
    fragmented = torch.ones((16, 16), dtype=torch.bool)
    fragmented[8, 8] = False

    with pytest.raises(ValueError, match="no valid pixels"):
        MaskedL1Loss()(prediction, target, empty)
    with pytest.raises(ValueError, match="no valid pixels"):
        MaskedSSIMLoss()(prediction, target, empty)
    with pytest.raises(ValueError, match="no valid SSIM windows"):
        MaskedSSIMLoss()(prediction, target, fragmented)


@pytest.mark.parametrize(
    ("target", "mask", "message"),
    [
        (
            torch.rand((16, 16, 1)),
            torch.ones((16, 16), dtype=torch.bool),
            "same RGB shape",
        ),
        (
            torch.rand((1, 16, 16, 3)),
            torch.ones((16, 16), dtype=torch.bool),
            "same RGB shape",
        ),
        (torch.rand((16, 16, 3)), torch.ones((16, 16)), "boolean"),
        (
            torch.rand((16, 16, 3)),
            torch.ones((16, 16, 2), dtype=torch.bool),
            "mask shape",
        ),
    ],
)
def test_l1_rejects_ambiguous_input_contract(target, mask, message) -> None:
    with pytest.raises(ValueError, match=message):
        MaskedL1Loss()(torch.rand((16, 16, 3)), target, mask)


def test_ssim_is_differentiable() -> None:
    prediction = torch.rand((32, 32, 3), requires_grad=True)
    target = torch.rand((32, 32, 3))
    mask = torch.ones((32, 32), dtype=torch.bool)

    loss = MaskedSSIMLoss()(prediction, target, mask)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_joint_loss_validates_weight_and_identity() -> None:
    with pytest.raises(ValueError, match="lambda_dssim"):
        JointLoss(lambda_dssim=1.1)

    target = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
    prediction = target.float() / 255.0
    mask = torch.ones((32, 32), dtype=torch.bool)
    assert torch.allclose(
        JointLoss()(prediction, target, mask), torch.tensor(0.0), atol=1e-6
    )


def test_ssim_validates_configuration_and_resolution() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        MaskedSSIMLoss(kernel_size=10)
    with pytest.raises(ValueError, match="sigma"):
        MaskedSSIMLoss(sigma=0.0)

    with pytest.raises(ValueError, match="smaller"):
        MaskedSSIMLoss()(
            torch.rand((10, 16, 3)),
            torch.rand((10, 16, 3)),
            torch.ones((10, 16), dtype=torch.bool),
        )


def test_setup_optimizers_matches_default_strategy_contract() -> None:
    gaussians = _gaussians()
    optimizers = setup_optimizers(gaussians)
    parameters = dict(gaussians.named_parameters())
    expected_lrs = {
        "means": 1.6e-4,
        "scales": 5.0e-3,
        "quats": 1.0e-3,
        "opacities": 5.0e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }

    assert optimizers.keys() == parameters.keys() == expected_lrs.keys()
    for name, optimizer in optimizers.items():
        assert len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group["name"] == name
        assert group["params"] == [parameters[name]]
        assert group["lr"] == expected_lrs[name]
        assert optimizer.defaults["eps"] == 1e-15


def test_mean_scheduler_has_exact_decay_and_restorable_state() -> None:
    optimizers = setup_optimizers(_gaussians())
    scheduler = setup_mean_scheduler(optimizers, max_steps=30_000)

    assert math.isclose(scheduler.gamma**30_000, 0.01, rel_tol=1e-12)
    optimizers["means"].step()
    scheduler.step()
    state = scheduler.state_dict()

    restored_optimizers = setup_optimizers(_gaussians())
    restored = setup_mean_scheduler(restored_optimizers, max_steps=30_000)
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_mean_scheduler_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        setup_mean_scheduler(setup_optimizers(_gaussians()), max_steps=0)
