from __future__ import annotations

from bts_nvs.experiments.contracts import CandidateSettings


CANDIDATE_IDS = (
    "B0-reference",
    "E1-density-absgrad-t04-v1",
    "E1-density-scale005-v1",
)

_CANDIDATES = {
    "B0-reference": CandidateSettings(
        candidate_id="B0-reference",
        absgrad=False,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_opa=0.005,
        refine_stop_step=15_000,
        rasterize_mode="classic",
        appearance_mode="baseline",
        sampling_mode="uniform",
    ),
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
    return candidate_settings(candidate_id).training_overrides()
