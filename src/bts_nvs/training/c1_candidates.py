from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateSettings:
    grow_grad2d: float
    absgrad: bool
    revised_opacity: bool


C1_CANDIDATES = (
    "C1-absgrad-t08-v1",
    "C1-absgrad-t08-revopacity-v1",
)
FULL_LENGTH_CANDIDATES = ("C1-absgrad-t08-revopacity-v1",)
QUALIFICATION_CANDIDATES = (
    "B0-reference",
    "B0-compact",
    *C1_CANDIDATES,
)

_SETTINGS = {
    "B0-reference": CandidateSettings(0.0002, False, False),
    "B0-compact": CandidateSettings(0.0003, False, False),
    "C1-absgrad-t08-v1": CandidateSettings(0.0008, True, False),
    "C1-absgrad-t08-revopacity-v1": CandidateSettings(0.0008, True, True),
}


def candidate_settings(candidate_id: str | None) -> CandidateSettings:
    if candidate_id is None:
        return _SETTINGS["B0-reference"]
    try:
        return _SETTINGS[candidate_id]
    except KeyError as error:
        raise ValueError(f"unknown qualification candidate: {candidate_id}") from error


def full_length_mode_enabled(args: object) -> bool:
    return bool(
        getattr(args, "full_length_qualification", False)
        or getattr(args, "full_length_candidate", None) is not None
    )


def selected_density_candidate(args: object) -> str | None:
    full_length_candidate = getattr(args, "full_length_candidate", None)
    if full_length_candidate is not None:
        if (
            getattr(args, "qualification_candidate", None) is not None
            or getattr(args, "full_length_qualification", False)
        ):
            raise ValueError("full-length candidate modes are mutually exclusive")
        return full_length_candidate
    return getattr(args, "qualification_candidate", None)
