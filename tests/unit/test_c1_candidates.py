import pytest

from bts_nvs.training.c1_candidates import (
    C1_CANDIDATES,
    QUALIFICATION_CANDIDATES,
    candidate_settings,
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
