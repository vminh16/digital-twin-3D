from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CandidateSettings:
    candidate_id: str
    absgrad: bool
    grow_grad2d: float
    grow_scale3d: float
    prune_opa: float
    refine_stop_step: int
    rasterize_mode: str
    appearance_mode: str
    sampling_mode: str
    max_sh_degree: int
    pixel_weight_mode: str
    pixel_weight_floor: float
    pixel_weight_patch_size: int
    observation_mapping_mode: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id.strip()
            or self.candidate_id != self.candidate_id.strip()
        ):
            raise ValueError("candidate_id must be a non-empty trimmed string")
        if not isinstance(self.absgrad, bool):
            raise ValueError("absgrad must be boolean")
        for field in ("grow_grad2d", "grow_scale3d", "prune_opa"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field} must be positive and finite")
        if self.prune_opa >= 1.0:
            raise ValueError("prune_opa must be less than one")
        if (
            isinstance(self.refine_stop_step, bool)
            or not isinstance(self.refine_stop_step, int)
            or self.refine_stop_step <= 0
        ):
            raise ValueError("refine_stop_step must be a positive integer")
        if self.rasterize_mode not in {"classic", "antialiased"}:
            raise ValueError("rasterize_mode is unsupported")
        if self.appearance_mode not in {"baseline", "sh4"}:
            raise ValueError("appearance_mode is unsupported")
        if self.sampling_mode != "uniform":
            raise ValueError("sampling_mode is unsupported")
        if (
            isinstance(self.max_sh_degree, bool)
            or not isinstance(self.max_sh_degree, int)
            or self.max_sh_degree not in {3, 4}
        ):
            raise ValueError("max_sh_degree must be 3 or 4")
        if self.appearance_mode == "sh4" and self.max_sh_degree != 4:
            raise ValueError("appearance_mode sh4 requires max_sh_degree 4")
        if self.appearance_mode == "baseline" and self.max_sh_degree != 3:
            raise ValueError("max_sh_degree 4 requires appearance_mode sh4")
        if self.pixel_weight_mode not in {"uniform", "local-laplacian"}:
            raise ValueError("pixel_weight_mode is unsupported")
        if self.observation_mapping_mode not in {
            "legacy-ceil",
            "continuous-reprojection",
        }:
            raise ValueError("observation_mapping_mode is unsupported")
        if (
            isinstance(self.pixel_weight_floor, bool)
            or not isinstance(self.pixel_weight_floor, (int, float))
            or not math.isfinite(float(self.pixel_weight_floor))
            or not 0.0 < float(self.pixel_weight_floor) <= 1.0
        ):
            raise ValueError("pixel_weight_floor must be finite and in (0, 1]")
        if (
            isinstance(self.pixel_weight_patch_size, bool)
            or not isinstance(self.pixel_weight_patch_size, int)
            or self.pixel_weight_patch_size < 3
            or self.pixel_weight_patch_size % 2 == 0
        ):
            raise ValueError(
                "pixel_weight_patch_size must be an odd integer of at least 3"
            )

    def training_overrides(self) -> dict[str, bool | float | int | str]:
        return asdict(self)
