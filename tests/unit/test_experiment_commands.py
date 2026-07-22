from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bts_nvs.experiments.commands import build_training_command
from bts_nvs.experiments.experiment import Experiment, ExperimentStage


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "repo_root": tmp_path / "repo",
        "scene_dir": tmp_path / "scene",
        "manifest_dir": tmp_path / "manifests",
        "output_dir": tmp_path / "output",
    }


def _build(tmp_path: Path, experiment: Experiment, **overrides) -> list[str]:
    arguments = {
        **_paths(tmp_path),
        "optimizer_backend": "adam-fused",
        "precision": "amp-fp16",
        "experiment": experiment,
        "stop_step": experiment.horizon,
    }
    arguments.update(overrides)
    return build_training_command(**arguments)


def _option_values(command: list[str]) -> dict[str, str | bool]:
    values: dict[str, str | bool] = {}
    index = 2
    while index < len(command):
        option = command[index]
        if index + 1 == len(command) or command[index + 1].startswith("--"):
            values[option] = True
            index += 1
        else:
            values[option] = command[index + 1]
            index += 2
    return values


def test_reference_command_pins_the_locked_7k_runtime(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    experiment = Experiment(
        stage=ExperimentStage.REFERENCE,
        scene_id="HCM0644",
        candidate_id="B0-reference",
    )

    command = _build(tmp_path, experiment)

    assert command == [
        sys.executable,
        str(paths["repo_root"] / "src" / "bts_nvs" / "training" / "run_training.py"),
        "--scene_dir",
        str(paths["scene_dir"]),
        "--manifest_dir",
        str(paths["manifest_dir"]),
        "--output_dir",
        str(paths["output_dir"]),
        "--seed",
        "0",
        "--resize_factor",
        "1",
        "--max_steps",
        "7000",
        "--stop_step",
        "7000",
        "--candidate_id",
        "B0-reference",
        "--experiment_stage",
        "reference",
        "--cache_images",
        "--pinned_transfer",
        "--internal_holdout",
        "--optimizer_backend",
        "adam-fused",
        "--precision",
        "amp-fp16",
    ]


def test_screen_command_is_fresh_7k_without_authorization_or_checkpoints(
    tmp_path: Path,
) -> None:
    experiment = Experiment(
        stage=ExperimentStage.SCREEN,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
    )

    options = _option_values(_build(tmp_path, experiment))

    assert options["--max_steps"] == "7000"
    assert options["--stop_step"] == "7000"
    assert options["--internal_holdout"] is True
    assert "--authorized_candidate_id" not in options
    assert "--checkpoint_every" not in options
    assert "--rolling_checkpoint" not in options
    assert "--resume" not in options


def test_confirmation_15k_command_uses_30k_schedule_and_rolling_recovery(
    tmp_path: Path,
) -> None:
    experiment = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_scene_winner="E1-density-absgrad-t04-v1",
    )

    options = _option_values(_build(tmp_path, experiment, stop_step=15_000))

    assert options["--max_steps"] == "30000"
    assert options["--stop_step"] == "15000"
    assert options["--checkpoint_every"] == "3000"
    assert options["--rolling_checkpoint"] is True
    assert options["--internal_holdout"] is True
    assert options["--authorized_candidate_id"] == experiment.candidate_id
    assert "--resume" not in options


def test_confirmation_resume_targets_its_own_recovery_file(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    experiment = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_scene_winner="E1-density-absgrad-t04-v1",
    )
    recovery = paths["output_dir"] / "checkpoints" / "recovery.pt"

    options = _option_values(
        _build(
            tmp_path,
            experiment,
            stop_step=30_000,
            resume_path=recovery,
        )
    )

    assert options["--max_steps"] == "30000"
    assert options["--stop_step"] == "30000"
    assert options["--resume"] == str(recovery)


@pytest.mark.parametrize(
    "resume_path",
    (
        "absolute-output/checkpoints/recovery.pt",
        "relative-output/checkpoints/../checkpoints/recovery.pt",
    ),
)
def test_confirmation_resume_accepts_equivalent_recovery_path_spellings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_path: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    experiment = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_scene_winner="E1-density-absgrad-t04-v1",
    )
    output_dir = Path("relative-output")
    if resume_path.startswith("absolute-output"):
        output_dir = Path("absolute-output")
        resume = tmp_path / resume_path
    else:
        resume = Path(resume_path)

    options = _option_values(
        _build(
            tmp_path,
            experiment,
            output_dir=output_dir,
            stop_step=30_000,
            resume_path=resume,
        )
    )

    assert options["--resume"] == str(resume)


def test_fresh_b0_confirmation_supplies_the_cli_authorization(tmp_path: Path) -> None:
    experiment = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="B0-reference",
    )

    options = _option_values(_build(tmp_path, experiment, stop_step=30_000))

    assert options["--authorized_candidate_id"] == "B0-reference"
    assert "--resume" not in options


def test_production_is_30k_without_holdout_and_only_resumes_explicitly(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    experiment = Experiment(
        stage=ExperimentStage.PRODUCTION,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_cohort_candidate="E1-density-absgrad-t04-v1",
    )
    recovery = paths["output_dir"] / "checkpoints" / "recovery.pt"

    fresh = _option_values(_build(tmp_path, experiment))
    resumed = _option_values(_build(tmp_path, experiment, resume_path=recovery))

    assert fresh["--max_steps"] == "30000"
    assert fresh["--stop_step"] == "30000"
    assert fresh["--checkpoint_every"] == "3000"
    assert fresh["--rolling_checkpoint"] is True
    assert "--internal_holdout" not in fresh
    assert "--resume" not in fresh
    assert resumed["--resume"] == str(recovery)


def test_paired_confirmation_commands_keep_all_fairness_settings_identical(
    tmp_path: Path,
) -> None:
    b0 = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="B0-reference",
    )
    candidate = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_scene_winner="E1-density-absgrad-t04-v1",
    )
    b0_command = _build(tmp_path, b0, output_dir=tmp_path / "b0", stop_step=30_000)
    candidate_command = _build(
        tmp_path,
        candidate,
        output_dir=tmp_path / "candidate",
        stop_step=30_000,
    )
    allowed_differences = {
        "--candidate_id",
        "--authorized_candidate_id",
        "--output_dir",
    }

    b0_options = _option_values(b0_command)
    candidate_options = _option_values(candidate_command)

    assert {
        key: value
        for key, value in b0_options.items()
        if key not in allowed_differences
    } == {
        key: value
        for key, value in candidate_options.items()
        if key not in allowed_differences
    }
    assert not set(b0_options).intersection(
        {"--absgrad", "--grow_grad2d", "--grow_scale3d", "--prune_opa"}
    )


@pytest.mark.parametrize(
    ("experiment", "stop_step", "resume_path", "match"),
    [
        (
            Experiment(ExperimentStage.REFERENCE, "HCM0644", "B0-reference"),
            15_000,
            None,
            "reference",
        ),
        (
            Experiment(ExperimentStage.SCREEN, "HCM0644", "E1-density-absgrad-t04-v1"),
            7_000,
            "not-recovery.pt",
            "screen",
        ),
        (
            Experiment(
                ExperimentStage.CONFIRM,
                "HCM0644",
                "E1-density-absgrad-t04-v1",
                authorized_scene_winner="E1-density-absgrad-t04-v1",
            ),
            15_000,
            "not-recovery.pt",
            "resume_path",
        ),
        (
            Experiment(
                ExperimentStage.PRODUCTION,
                "HCM0644",
                "E1-density-absgrad-t04-v1",
                authorized_cohort_candidate="E1-density-absgrad-t04-v1",
            ),
            15_000,
            None,
            "production",
        ),
    ],
)
def test_builder_rejects_illegal_stage_stop_and_resume_combinations(
    tmp_path: Path,
    experiment: Experiment,
    stop_step: int,
    resume_path: str | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _build(
            tmp_path,
            experiment,
            stop_step=stop_step,
            resume_path=resume_path,
        )


def test_confirmation_rejects_resuming_the_15k_invocation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    experiment = Experiment(
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_scene_winner="E1-density-absgrad-t04-v1",
    )

    with pytest.raises(ValueError, match="confirm resume"):
        _build(
            tmp_path,
            experiment,
            stop_step=15_000,
            resume_path=paths["output_dir"] / "checkpoints" / "recovery.pt",
        )


@pytest.mark.parametrize(
    ("optimizer_backend", "precision", "match"),
    [
        ("sgd", "fp32", "optimizer_backend"),
        ("adam", "bf16", "precision"),
        ("adam", "amp-fp16", "amp-fp16 requires adam-fused"),
    ],
)
def test_builder_rejects_unaccepted_runtime_settings(
    tmp_path: Path,
    optimizer_backend: str,
    precision: str,
    match: str,
) -> None:
    experiment = Experiment(ExperimentStage.REFERENCE, "HCM0644", "B0-reference")

    with pytest.raises(ValueError, match=match):
        _build(
            tmp_path,
            experiment,
            optimizer_backend=optimizer_backend,
            precision=precision,
        )
