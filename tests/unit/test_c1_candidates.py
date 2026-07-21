from argparse import Namespace

import pytest

from bts_nvs.training.c1_candidates import (
    C1_CANDIDATES,
    FULL_LENGTH_CANDIDATES,
    QUALIFICATION_CANDIDATES,
    candidate_settings,
    full_length_mode_enabled,
    selected_density_candidate,
)


def test_c1_candidate_settings_are_exact() -> None:
    plain = candidate_settings("C1-absgrad-t08-v1")
    revised = candidate_settings("C1-absgrad-t08-revopacity-v1")

    assert C1_CANDIDATES == (
        "C1-absgrad-t08-v1",
        "C1-absgrad-t08-revopacity-v1",
    )
    assert set(C1_CANDIDATES).issubset(QUALIFICATION_CANDIDATES)
    assert (plain.grow_grad2d, plain.absgrad, plain.revised_opacity) == (
        pytest.approx(0.0008),
        True,
        False,
    )
    assert (revised.grow_grad2d, revised.absgrad, revised.revised_opacity) == (
        pytest.approx(0.0008),
        True,
        True,
    )


def test_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown qualification candidate"):
        candidate_settings("unknown")


def test_none_uses_b0_reference_defaults() -> None:
    settings = candidate_settings(None)

    assert (settings.grow_grad2d, settings.absgrad, settings.revised_opacity) == (
        pytest.approx(0.0002),
        False,
        False,
    )


def test_full_length_candidate_identity_is_locked_to_phase_c_winner() -> None:
    assert FULL_LENGTH_CANDIDATES == ("C1-absgrad-t08-revopacity-v1",)

    args = Namespace(
        qualification_candidate=None,
        full_length_qualification=False,
        full_length_candidate="C1-absgrad-t08-revopacity-v1",
    )

    assert full_length_mode_enabled(args) is True
    assert selected_density_candidate(args) == "C1-absgrad-t08-revopacity-v1"
    settings = candidate_settings(selected_density_candidate(args))
    assert (settings.grow_grad2d, settings.absgrad, settings.revised_opacity) == (
        pytest.approx(0.0008),
        True,
        True,
    )


def test_b0_full_length_mode_retains_reference_density_defaults() -> None:
    args = Namespace(
        qualification_candidate=None,
        full_length_qualification=True,
        full_length_candidate=None,
    )

    assert full_length_mode_enabled(args) is True
    assert selected_density_candidate(args) is None
    settings = candidate_settings(selected_density_candidate(args))
    assert settings.grow_grad2d == pytest.approx(0.0002)


@pytest.mark.parametrize(
    "args",
    (
        Namespace(
            qualification_candidate="B0-reference",
            full_length_qualification=False,
            full_length_candidate="C1-absgrad-t08-revopacity-v1",
        ),
        Namespace(
            qualification_candidate=None,
            full_length_qualification=True,
            full_length_candidate="C1-absgrad-t08-revopacity-v1",
        ),
    ),
)
def test_full_length_candidate_rejects_mixed_experiment_modes(args: Namespace) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        selected_density_candidate(args)
