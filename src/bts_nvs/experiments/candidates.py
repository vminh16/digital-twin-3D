from __future__ import annotations

from dataclasses import replace

from bts_nvs.experiments.contracts import CandidateSettings
from bts_nvs.experiments.density_policies import (
    MCMC_CANDIDATE_ID,
    density_policy_overrides,
)


CANDIDATE_IDS = (
    "B0-reference",
    "E1-density-absgrad-t04-v1",
    "E1-density-scale005-v1",
    "E2-raster-aa-v1",
    "E2-loss-local-laplacian-v1",
    "E2-appearance-sh4-v1",
    "E3-chair-observation-scale-v1",
    "E4-chair-observation-scale-absgrad-v1",
    MCMC_CANDIDATE_ID,
)

CHAIR_RESEARCH_CANDIDATE_IDS = frozenset(
    (
        "E3-chair-observation-scale-v1",
        "E4-chair-observation-scale-absgrad-v1",
        MCMC_CANDIDATE_ID,
    )
)

_BASELINE = CandidateSettings(
        candidate_id="B0-reference",
        absgrad=False,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_opa=0.005,
        refine_stop_step=15_000,
        rasterize_mode="classic",
        appearance_mode="baseline",
        sampling_mode="uniform",
        max_sh_degree=3,
        pixel_weight_mode="uniform",
        pixel_weight_floor=0.5,
        pixel_weight_patch_size=31,
        observation_mapping_mode="legacy-ceil",
    )

_CANDIDATES = {
    "B0-reference": _BASELINE,
    "E1-density-absgrad-t04-v1": CandidateSettings(
        candidate_id="E1-density-absgrad-t04-v1",
        absgrad=True,
        grow_grad2d=0.0004,
        grow_scale3d=0.01,
        prune_opa=0.005,
        refine_stop_step=15_000,
        rasterize_mode="classic",
        appearance_mode="baseline",
        sampling_mode="uniform",
        max_sh_degree=3,
        pixel_weight_mode="uniform",
        pixel_weight_floor=0.5,
        pixel_weight_patch_size=31,
        observation_mapping_mode="legacy-ceil",
    ),
    "E1-density-scale005-v1": CandidateSettings(
        candidate_id="E1-density-scale005-v1",
        absgrad=False,
        grow_grad2d=0.0002,
        grow_scale3d=0.005,
        prune_opa=0.005,
        refine_stop_step=15_000,
        rasterize_mode="classic",
        appearance_mode="baseline",
        sampling_mode="uniform",
        max_sh_degree=3,
        pixel_weight_mode="uniform",
        pixel_weight_floor=0.5,
        pixel_weight_patch_size=31,
        observation_mapping_mode="legacy-ceil",
    ),
    "E2-raster-aa-v1": CandidateSettings(
        candidate_id="E2-raster-aa-v1",
        absgrad=False,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_opa=0.005,
        refine_stop_step=15_000,
        rasterize_mode="antialiased",
        appearance_mode="baseline",
        sampling_mode="uniform",
        max_sh_degree=3,
        pixel_weight_mode="uniform",
        pixel_weight_floor=0.5,
        pixel_weight_patch_size=31,
        observation_mapping_mode="legacy-ceil",
    ),
    "E2-loss-local-laplacian-v1": CandidateSettings(
        candidate_id="E2-loss-local-laplacian-v1",
        absgrad=False,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_opa=0.005,
        refine_stop_step=15_000,
        rasterize_mode="classic",
        appearance_mode="baseline",
        sampling_mode="uniform",
        max_sh_degree=3,
        pixel_weight_mode="local-laplacian",
        pixel_weight_floor=0.5,
        pixel_weight_patch_size=31,
        observation_mapping_mode="legacy-ceil",
    ),
    "E2-appearance-sh4-v1": CandidateSettings(
        candidate_id="E2-appearance-sh4-v1",
        absgrad=False,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_opa=0.005,
        refine_stop_step=15_000,
        rasterize_mode="classic",
        appearance_mode="sh4",
        sampling_mode="uniform",
        max_sh_degree=4,
        pixel_weight_mode="uniform",
        pixel_weight_floor=0.5,
        pixel_weight_patch_size=31,
        observation_mapping_mode="legacy-ceil",
    ),
    "E3-chair-observation-scale-v1": replace(
        _BASELINE,
        candidate_id="E3-chair-observation-scale-v1",
        pixel_weight_mode="local-laplacian",
        observation_mapping_mode="continuous-reprojection",
    ),
    "E4-chair-observation-scale-absgrad-v1": replace(
        _BASELINE,
        candidate_id="E4-chair-observation-scale-absgrad-v1",
        absgrad=True,
        grow_grad2d=0.0004,
        pixel_weight_mode="local-laplacian",
        observation_mapping_mode="continuous-reprojection",
    ),
    MCMC_CANDIDATE_ID: replace(
        _BASELINE,
        candidate_id=MCMC_CANDIDATE_ID,
        refine_stop_step=25_000,
        pixel_weight_mode="local-laplacian",
        observation_mapping_mode="continuous-reprojection",
    ),
}


def candidate_settings(candidate_id: str) -> CandidateSettings:
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    try:
        return _CANDIDATES[candidate_id]
    except KeyError as error:
        raise ValueError(f"unknown candidate: {candidate_id}") from error


def candidate_training_overrides(
    candidate_id: str,
) -> dict[str, bool | float | int | str]:
    overrides = candidate_settings(candidate_id).training_overrides()
    overrides.update(density_policy_overrides(candidate_id))
    return overrides
