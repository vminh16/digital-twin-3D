from pathlib import Path
from types import SimpleNamespace

import pytest

from bts_nvs.experiments import run_mvp_production
from bts_nvs.experiments.experiment import ExperimentStage


def test_validate_scene_ids_defaults_to_locked_pair() -> None:
    assert run_mvp_production.validate_scene_ids(None) == (
        "HCM0421",
        "HCM0539",
    )


def test_validate_scene_ids_rejects_unknown_or_duplicate_scene() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        run_mvp_production.validate_scene_ids(("HCM0644",))
    with pytest.raises(ValueError, match="unique"):
        run_mvp_production.validate_scene_ids(("HCM0421", "HCM0421"))


def test_screen_evidence_validates_internal_holdout_reference_and_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        run_mvp_production,
        "validate_existing",
        lambda **kwargs: calls.append(kwargs),
    )

    run_mvp_production.validate_screen_evidence(
        repo_root=tmp_path,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        backend_root=tmp_path / "backend",
        experiment_root=tmp_path / "experiments",
        scene_id="HCM0421",
    )

    assert [call["stage"] for call in calls] == [
        ExperimentStage.REFERENCE,
        ExperimentStage.SCREEN,
    ]
    assert calls[0]["candidate_id"] == "B0-reference"
    assert calls[1]["candidate_id"] == "E1-density-absgrad-t04-v1"
    assert calls[1]["stop_step"] == 7_000
    assert "b0_report_path" not in calls[1]


def test_production_command_uses_all_images_and_resumes_only_valid_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []
    output_root = tmp_path / "output"
    run_dir = output_root / "scenes" / "HCM0421"
    recovery = run_dir / "checkpoints" / "recovery.pt"
    monkeypatch.setattr(run_mvp_production, "validate_screen_evidence", lambda **_: None)
    monkeypatch.setattr(
        run_mvp_production,
        "load_or_create_backend_decision",
        lambda _: SimpleNamespace(optimizer_backend="adam", precision="fp32"),
    )
    monkeypatch.setattr(
        run_mvp_production,
        "inspect_scene_run",
        lambda *_: "resume",
    )
    monkeypatch.setattr(
        run_mvp_production,
        "_validate_completed_production",
        lambda **_: None,
    )
    monkeypatch.setattr(run_mvp_production, "_validate_mvp_config", lambda *_: {})

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    run_mvp_production.run_mvp_production(
        repo_root=tmp_path,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        backend_root=tmp_path / "backend",
        experiment_root=tmp_path / "experiments",
        output_root=output_root,
        scene_ids=("HCM0421",),
        run_process=fake_run,
    )

    command, options = commands[0]
    assert command[command.index("--max_steps") + 1] == "30000"
    assert command[command.index("--stop_step") + 1] == "30000"
    assert command[command.index("--candidate_id") + 1] == (
        "E1-density-absgrad-t04-v1"
    )
    assert command[command.index("--experiment_stage") + 1] == "production"
    assert command[command.index("--resume") + 1] == str(recovery)
    assert "--internal_holdout" not in command
    assert "--rolling_checkpoint" in command
    assert options == {"cwd": tmp_path, "check": False, "shell": False}


def test_completed_scene_is_validated_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = []
    monkeypatch.setattr(
        run_mvp_production,
        "validate_screen_evidence",
        lambda **kwargs: evidence.append(kwargs["scene_id"]),
    )
    monkeypatch.setattr(
        run_mvp_production,
        "load_or_create_backend_decision",
        lambda _: SimpleNamespace(optimizer_backend="adam", precision="fp32"),
    )
    monkeypatch.setattr(run_mvp_production, "inspect_scene_run", lambda *_: "trained")
    monkeypatch.setattr(run_mvp_production, "_validate_mvp_config", lambda *_: {})
    monkeypatch.setattr(
        run_mvp_production,
        "_validate_completed_production",
        lambda **_: None,
    )

    run_mvp_production.run_mvp_production(
        repo_root=tmp_path,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        backend_root=tmp_path / "backend",
        experiment_root=tmp_path / "experiments",
        output_root=tmp_path / "output",
        scene_ids=("HCM0539",),
        run_process=lambda *_args, **_kwargs: pytest.fail("must not retrain"),
    )

    assert evidence == ["HCM0539"]


def test_existing_mvp_config_rejects_internal_holdout(
    tmp_path: Path,
) -> None:
    experiment = run_mvp_production.Experiment(
        stage=ExperimentStage.PRODUCTION,
        scene_id="HCM0421",
        candidate_id="E1-density-absgrad-t04-v1",
        authorized_cohort_candidate="E1-density-absgrad-t04-v1",
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            (
                "scene_id: HCM0421",
                "candidate_id: E1-density-absgrad-t04-v1",
                "experiment_stage: production",
                "max_steps: 30000",
                "internal_holdout: true",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MVP production identity"):
        run_mvp_production._validate_mvp_config(path, experiment)
