from __future__ import annotations

import math
from dataclasses import asdict, dataclass


SPECTRAL_CANDIDATE_ID = "E8-chair-observation-scale-spectral-split-v1"


@dataclass(frozen=True)
class SpectralDensityPolicy:
    density_strategy: str = "spectral"
    spectral_cap_max: int = 2_100_000
    spectral_entropy_threshold: float = 0.5
    spectral_split_k: float = 0.6
    spectral_split_k0: float = 1.0

    def __post_init__(self) -> None:
        if self.density_strategy != "spectral":
            raise ValueError("density_strategy must be spectral")
        if (
            isinstance(self.spectral_cap_max, bool)
            or not isinstance(self.spectral_cap_max, int)
            or self.spectral_cap_max <= 0
        ):
            raise ValueError("spectral_cap_max must be a positive integer")
        for field in (
            "spectral_entropy_threshold",
            "spectral_split_k",
            "spectral_split_k0",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field} must be positive and finite")
        if self.spectral_split_k0 < 1.0:
            raise ValueError("spectral_split_k0 must be at least one")

    def training_overrides(self) -> dict[str, float | int | str]:
        return asdict(self)


_POLICY = SpectralDensityPolicy()


def spectral_policy_overrides(
    candidate_id: str,
) -> dict[str, float | int | str]:
    if candidate_id == SPECTRAL_CANDIDATE_ID:
        return _POLICY.training_overrides()
    return {}


def is_spectral_candidate(candidate_id: str | None) -> bool:
    return candidate_id == SPECTRAL_CANDIDATE_ID
