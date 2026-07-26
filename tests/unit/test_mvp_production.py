import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bts_nvs.experiments import run_mvp_production
from bts_nvs.experiments.experiment import ExperimentStage
from bts_nvs.experiments.provenance import canonical_json_sha256


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


def test_validate_scene_ids_accepts_auxiliary_winners() -> None:
    assert run_mvp_production.validate_scene_ids(("chair", "bonsai")) == (
        "chair",
        "bonsai",
    )


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
    assert calls[1]["b0_report_path"] == (
        tmp_path / "experiments/reference/HCM0421/experiment_report.json"
    )


def test_auxiliary_evidence_uses_separate_b0_root_and_selected_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    decisions = []
    monkeypatch.setattr(
        run_mvp_production,
        "validate_existing",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        run_mvp_production,
        "_validate_selected_screen_decision",
        lambda path, **kwargs: decisions.append((path, kwargs)),
    )

    run_mvp_production.validate_screen_evidence(
        repo_root=tmp_path,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        backend_root=tmp_path / "backend",
        reference_root=tmp_path / "v1",
        experiment_root=tmp_path / "v2",
        scene_id="bonsai",
    )

    assert calls[0]["experiment_root"] == tmp_path / "v1"
    assert calls[1]["experiment_root"] == tmp_path / "v2"
    assert calls[1]["candidate_id"] == "E2-appearance-sh4-v1"
    assert calls[1]["b0_report_path"] == (
        tmp_path / "v1/reference/bonsai/experiment_report.json"
    )
    assert decisions == [
        (
            tmp_path / "v2/decisions/screen/bonsai.json",
            {
                "scene_id": "bonsai",
                "candidate_id": "E2-appearance-sh4-v1",
            },
        )
    ]


def test_production_command_uses_all_images_and_resumes_only_valid_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []
    output_root = tmp_path / "output"
    run_dir = output_root / "scenes" / "HCM0421"
    recovery = run_dir / "checkpoints" / "recovery.pt"
    monkeypatch.setattr(
        run_mvp_production,
        "validate_screen_evidence",
        lambda **_: None,
    )
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


def test_auxiliary_production_uses_locked_scene_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = []
    monkeypatch.setattr(run_mvp_production, "validate_screen_evidence", lambda **_: None)
    monkeypatch.setattr(
        run_mvp_production,
        "load_or_create_backend_decision",
        lambda _: SimpleNamespace(optimizer_backend="adam", precision="fp32"),
    )
    monkeypatch.setattr(run_mvp_production, "inspect_scene_run", lambda *_: "fresh")
    monkeypatch.setattr(
        run_mvp_production,
        "_validate_completed_production",
        lambda **_: None,
    )

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    run_mvp_production.run_mvp_production(
        repo_root=tmp_path,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        backend_root=tmp_path / "backend",
        reference_root=tmp_path / "v1",
        experiment_root=tmp_path / "v2",
        output_root=tmp_path / "output",
        scene_ids=("chair", "bonsai"),
        run_process=fake_run,
    )

    assert [
        command[command.index("--candidate_id") + 1] for command in commands
    ] == [
        "E2-loss-local-laplacian-v1",
        "E2-appearance-sh4-v1",
    ]


def test_selected_screen_decision_must_be_hashed_and_qualified(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision.json"
    decision = {
        "schema_version": 1,
        "scene_id": "chair",
        "step": 7_000,
        "decision_stage": "screen",
        "selected_candidate_id": "E2-loss-local-laplacian-v1",
        "fallback_to_b0": False,
        "evaluations": [
            {
                "candidate_id": "E2-loss-local-laplacian-v1",
                "screen_qualified": True,
            }
        ],
    }
    decision["decision_sha256"] = canonical_json_sha256(decision)
    path.write_text(json.dumps(decision), encoding="utf-8")

    run_mvp_production._validate_selected_screen_decision(
        path,
        scene_id="chair",
        candidate_id="E2-loss-local-laplacian-v1",
    )

    decision["selected_candidate_id"] = "B0-reference"
    path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        run_mvp_production._validate_selected_screen_decision(
            path,
            scene_id="chair",
            candidate_id="E2-loss-local-laplacian-v1",
        )
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
