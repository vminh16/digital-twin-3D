from __future__ import annotations

import numpy as np
import pytest
import torch

from bts_nvs.data.perceptual_sensitivity import (
    extract_perceptual_sensitivity,
)
from bts_nvs.models.perceptual_loss import perceptual_sensitivity_loss


def test_sensitivity_extraction_is_binary_deterministic_and_masked() -> None:
    image = np.zeros((15, 15, 3), dtype=np.uint8)
    image[:, 7:] = 255
    mask = np.ones((15, 15), dtype=bool)
    mask[:, :3] = False

    first = extract_perceptual_sensitivity(image, mask)
    second = extract_perceptual_sensitivity(image, mask)

    np.testing.assert_array_equal(first, second)
    assert set(np.unique(first)) <= {0, 255}
    assert not first[:, :3].any()
    assert first[:, 5:10].any()


def test_sensitivity_extraction_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="RGB uint8"):
        extract_perceptual_sensitivity(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="valid_mask"):
        extract_perceptual_sensitivity(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((7, 8), dtype=bool),
        )


def test_perceptual_loss_uses_only_valid_pixels() -> None:
    prediction = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[True, True], [False, False]])

    loss = perceptual_sensitivity_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(-torch.log(torch.tensor(0.9)).item())
