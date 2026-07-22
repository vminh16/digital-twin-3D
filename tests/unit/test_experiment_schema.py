from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bts_nvs.experiments.experiment import (
    COHORT_SCENE_IDS,
    MAX_PAIRED_WALL_TIME_RATIO,
    MAX_PEAK_VRAM_MB,
    STAGE_HORIZONS,
    Experiment,
    ExperimentStage,
    validate_paired_wall_time_ratio,
    validate_peak_vram_mb,
)


def test_schema_locks_cohort_stages_horizons_and_resource_limits() -> None:
    assert COHORT_SCENE_IDS == (
        "HCM0644",
        "HCM0674",
        "HCM0540",
        "HCM0539",
        "HCM0421",
        "chair",
        "bonsai",
    )
    assert tuple(ExperimentStage) == (
        ExperimentStage.REFERENCE,
        ExperimentStage.SCREEN,
        ExperimentStage.CONFIRM,
        ExperimentStage.PRODUCTION,
    )
    assert STAGE_HORIZONS == {
        ExperimentStage.REFERENCE: 7_000,
        ExperimentStage.SCREEN: 7_000,
        ExperimentStage.CONFIRM: 30_000,
        ExperimentStage.PRODUCTION: 30_000,
    }
    assert MAX_PEAK_VRAM_MB == 23 * 1024
    assert MAX_PAIRED_WALL_TIME_RATIO == 1.25


@pytest.mark.parametrize("ratio", [0.0, 0.5, MAX_PAIRED_WALL_TIME_RATIO])
def test_paired_wall_time_ratio_accepts_finite_values_within_limit(
    ratio: float,
) -> None:
    assert validate_paired_wall_time_ratio(ratio) is None


@pytest.mark.parametrize(
    "ratio",
    [
        True,
        False,
        -0.01,
        MAX_PAIRED_WALL_TIME_RATIO + 0.01,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_paired_wall_time_ratio_rejects_invalid_or_over_budget_values(
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match="paired_wall_time_ratio"):
        validate_paired_wall_time_ratio(ratio)


@pytest.mark.parametrize("peak_vram_mb", [0.0, MAX_PEAK_VRAM_MB - 0.5])
def test_peak_vram_accepts_finite_values_strictly_below_limit(
    peak_vram_mb: float,
) -> None:
    assert validate_peak_vram_mb(peak_vram_mb) is None


@pytest.mark.parametrize(
    "peak_vram_mb",
    [
        True,
        False,
        -0.01,
        MAX_PEAK_VRAM_MB,
        MAX_PEAK_VRAM_MB + 0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_peak_vram_rejects_invalid_or_at_limit_values(peak_vram_mb: float) -> None:
    with pytest.raises(ValueError, match="peak_vram_mb"):
        validate_peak_vram_mb(peak_vram_mb)


@pytest.mark.parametrize(
    ("stage", "candidate_id", "authorization"),
    [
        (ExperimentStage.REFERENCE, "B0-reference", {}),
        (ExperimentStage.SCREEN, "E1-density-absgrad-t04-v1", {}),
        (ExperimentStage.CONFIRM, "B0-reference", {}),
        (
            ExperimentStage.CONFIRM,
            "E1-density-absgrad-t04-v1",
            {"authorized_scene_winner": "E1-density-absgrad-t04-v1"},
        ),
        (
            ExperimentStage.PRODUCTION,
            "E1-density-scale005-v1",
            {"authorized_cohort_candidate": "E1-density-scale005-v1"},
        ),
    ],
)
def test_experiment_accepts_only_legal_stage_candidate_pairs(
    stage: ExperimentStage, candidate_id: str, authorization: dict[str, str]
) -> None:
    experiment = Experiment(
        stage=stage,
        scene_id="HCM0644",
        candidate_id=candidate_id,
        **authorization,
    )

    assert experiment.horizon == STAGE_HORIZONS[stage]


@pytest.mark.parametrize(
    ("stage", "candidate_id", "authorization", "error"),
    [
        (ExperimentStage.REFERENCE, "E1-density-absgrad-t04-v1", {}, "reference"),
        (ExperimentStage.SCREEN, "B0-reference", {}, "screen"),
        (
            ExperimentStage.CONFIRM,
            "E1-density-absgrad-t04-v1",
            {},
            "authorized_scene_winner",
        ),
        (
            ExperimentStage.CONFIRM,
            "E1-density-absgrad-t04-v1",
            {"authorized_scene_winner": "E1-density-scale005-v1"},
            "authorized_scene_winner",
        ),
        (
            ExperimentStage.PRODUCTION,
            "B0-reference",
            {},
            "authorized_cohort_candidate",
        ),
        (
            ExperimentStage.PRODUCTION,
            "E1-density-absgrad-t04-v1",
            {"authorized_cohort_candidate": "E1-density-scale005-v1"},
            "authorized_cohort_candidate",
        ),
    ],
)
def test_experiment_rejects_illegal_stage_candidate_pairs(
    stage: ExperimentStage,
    candidate_id: str,
    authorization: dict[str, str],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        Experiment(
            stage=stage,
            scene_id="HCM0644",
            candidate_id=candidate_id,
            **authorization,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("stage", "screen", "stage"),
        ("scene_id", "HCM0181", "scene_id"),
        ("candidate_id", "unknown-candidate", "unknown candidate"),
        ("authorized_scene_winner", "unknown-candidate", "authorized_scene_winner"),
        ("authorized_cohort_candidate", "unknown-candidate", "authorized_cohort_candidate"),
    ],
)
def test_experiment_rejects_invalid_schema_identities(
    field: str, value: object, error: str
) -> None:
    values = {
        "stage": ExperimentStage.SCREEN,
        "scene_id": "HCM0644",
        "candidate_id": "E1-density-absgrad-t04-v1",
    }
    values[field] = value

    with pytest.raises(ValueError, match=error):
        Experiment(**values)


def test_experiment_is_immutable_and_paths_are_stage_first() -> None:
    screen = Experiment(
        stage=ExperimentStage.SCREEN,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
    )
    confirm = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_scene_winner="E1-density-absgrad-t04-v1",
    )
    reference = Experiment(
        stage=ExperimentStage.REFERENCE,
        scene_id="HCM0644",
        candidate_id="B0-reference",
    )

    with pytest.raises(FrozenInstanceError):
        screen.scene_id = "chair"

    root = Path("runs") / "scene_opt_v1"
    assert reference.run_path(root) == root / "reference" / "HCM0644"
    assert screen.run_path(root) == (
        root / "screen" / "HCM0644" / "E1-density-absgrad-t04-v1"
    )
    assert confirm.run_path(root) == (
        root / "confirm" / "HCM0644" / "E1-density-absgrad-t04-v1"
    )
    assert screen.run_path(root) != confirm.run_path(root)
