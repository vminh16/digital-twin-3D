from __future__ import annotations

import math
from dataclasses import asdict, dataclass


MCMC_CANDIDATE_ID = "E5-chair-observation-scale-mcmc-v1"


@dataclass(frozen=True)
class MCMCDensityPolicy:
    density_strategy: str = "mcmc"
    mcmc_cap_max: int = 2_000_000
    mcmc_noise_lr: float = 500_000.0
    mcmc_opacity_reg: float = 0.001
    mcmc_scale_reg: float = 0.01

    def __post_init__(self) -> None:
        if self.density_strategy != "mcmc":
            raise ValueError("density_strategy must be mcmc")
        if (
            isinstance(self.mcmc_cap_max, bool)
            or not isinstance(self.mcmc_cap_max, int)
            or self.mcmc_cap_max <= 0
        ):
            raise ValueError("mcmc_cap_max must be a positive integer")
        for field in (
            "mcmc_noise_lr",
            "mcmc_opacity_reg",
            "mcmc_scale_reg",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{field} must be finite and nonnegative")

    def training_overrides(self) -> dict[str, int | float | str]:
        return asdict(self)


_MCMC_POLICY = MCMCDensityPolicy()


def density_policy_overrides(candidate_id: str) -> dict[str, int | float | str]:
    if candidate_id == MCMC_CANDIDATE_ID:
        return _MCMC_POLICY.training_overrides()
    return {}


def is_full_horizon_research_candidate(candidate_id: str | None) -> bool:
    return candidate_id == MCMC_CANDIDATE_ID
