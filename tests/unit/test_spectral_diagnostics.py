from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from bts_nvs.evaluation.spectral_diagnostics import (
    build_spectral_diagnostics,
    validate_spectral_diagnostics,
)
from bts_nvs.experiments.spectral_policy import SPECTRAL_CANDIDATE_ID
from bts_nvs.models.gaussian_parameters import GaussianParameters


def _trainer():
    gaussians = GaussianParameters(
        means=torch.zeros((2, 3)),
        scales=torch.log(torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.1, 0.1]])),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        opacities=torch.zeros(2),
        sh0=torch.zeros((2, 1, 3)),
        shN=torch.zeros((2, 15, 3)),
    )
    return SimpleNamespace(
        gaussians=gaussians,
        config={
            "candidate_id": SPECTRAL_CANDIDATE_ID,
            "spectral_entropy_threshold": 0.5,
            "spectral_split_k": 0.6,
            "spectral_split_k0": 1.0,
            "spectral_cap_max": 2_100_000,
        },
        strategy_state={
            "adc_clone_count": 3,
            "adc_split_count": 2,
            "spectral_split_count": 1,
            "spectral_cap_hit_step": None,
            "spectral_child_condition_violations": 0,
        },
    )


def test_spectral_diagnostics_round_trip_contract() -> None:
    report = build_spectral_diagnostics(_trainer())

    validate_spectral_diagnostics(
        report,
        candidate_id=SPECTRAL_CANDIDATE_ID,
        threshold=0.5,
        cap_max=2_100_000,
    )
    assert report["low_entropy"]["count"] == 1
    assert report["density"]["spectral_split_count"] == 1


def test_spectral_diagnostics_reject_condition_violation() -> None:
    report = build_spectral_diagnostics(_trainer())
    report["density"]["child_condition_violations"] = 1

    with pytest.raises(ValueError, match="condition"):
        validate_spectral_diagnostics(
            report,
            candidate_id=SPECTRAL_CANDIDATE_ID,
            threshold=0.5,
            cap_max=2_100_000,
        )
