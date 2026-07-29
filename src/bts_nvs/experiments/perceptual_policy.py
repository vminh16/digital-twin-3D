from __future__ import annotations

import math
from dataclasses import asdict, dataclass


PERCEPTUAL_CANDIDATE_ID = "E6-chair-observation-scale-perceptual-v1"
PERCEPTUAL_ADC_CANDIDATE_ID = "E7-chair-perceptual-adc-corrected-v1"


@dataclass(frozen=True)
class PerceptualDensityPolicy:
    density_strategy: str = "perceptual"
    perceptual_loss_weight: float = 0.1
    perceptual_sensitivity_lr: float = 0.05
    perceptual_high_threshold: float = 0.9
    perceptual_low_threshold: float = 0.3
    perceptual_high_interval: int = 1_000
    perceptual_medium_interval: int = 1_500
    perceptual_high_contribution: float = 25.0
    perceptual_medium_contribution: float = 10.0
    perceptual_opacity_exponent: float = 1.2
    perceptual_cap_max: int = 2_100_000

    def __post_init__(self) -> None:
        if self.density_strategy not in {"perceptual", "perceptual-adc"}:
            raise ValueError("density_strategy must be perceptual or perceptual-adc")
        finite_fields = (
            "perceptual_loss_weight",
            "perceptual_sensitivity_lr",
            "perceptual_high_threshold",
            "perceptual_low_threshold",
            "perceptual_high_contribution",
            "perceptual_medium_contribution",
            "perceptual_opacity_exponent",
        )
        for field in finite_fields:
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field} must be positive and finite")
        if not 0.0 < self.perceptual_loss_weight < 1.0:
            raise ValueError("perceptual_loss_weight must be in (0, 1)")
        if not (
            0.0
            < self.perceptual_low_threshold
            < self.perceptual_high_threshold
            < 1.0
        ):
            raise ValueError("perceptual sensitivity thresholds are invalid")
        if self.perceptual_opacity_exponent < 1.0:
            raise ValueError("perceptual_opacity_exponent must be at least one")
        for field in (
            "perceptual_high_interval",
            "perceptual_medium_interval",
            "perceptual_cap_max",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")

    def training_overrides(self) -> dict[str, float | int | str]:
        return asdict(self)


_POLICY = PerceptualDensityPolicy()
_ADC_POLICY = PerceptualDensityPolicy(density_strategy="perceptual-adc")


def perceptual_policy_overrides(
    candidate_id: str,
) -> dict[str, float | int | str]:
    policies = {
        PERCEPTUAL_CANDIDATE_ID: _POLICY,
        PERCEPTUAL_ADC_CANDIDATE_ID: _ADC_POLICY,
    }
    policy = policies.get(candidate_id)
    return policy.training_overrides() if policy is not None else {}


def is_perceptual_candidate(candidate_id: str | None) -> bool:
    return candidate_id in {
        PERCEPTUAL_CANDIDATE_ID,
        PERCEPTUAL_ADC_CANDIDATE_ID,
    }
