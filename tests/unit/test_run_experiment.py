from pathlib import Path
from types import SimpleNamespace

import pytest

from bts_nvs.experiments.artifacts import ArtifactValidationResult
from bts_nvs.experiments.decisions import build_cohort_decision
from bts_nvs.experiments.experiment import COHORT_SCENE_IDS, ExperimentStage
from bts_nvs.experiments.provenance import canonical_json_sha256, save_json_artifact
from bts_nvs.experiments.run_experiment import (
    _validate_backend_config,
    main,
    run_one,
    validate_existing,
)


def _scene_decision(
    scene_id: str,
    candidate_id: str,
    *,
    step: int = 30_000,
    decision_stage: str = "confirmation",
) -> dict[str, object]:
    decision: dict[str, object] = {
        "schema_version": 1,
        "scene_id": scene_id,
        "step": step,
        "decision_stage": decision_stage,
        "selected_candidate_id": candidate_id,
    }
    decision["decision_sha256"] = canonical_json_sha256(decision)
    return decision


def _paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "repo_root": tmp_path / "repo",
        "scenes_root": tmp_path / "scenes",
        "manifests_root": tmp_path / "manifests",
        "backend_root": tmp_path / "backend",
        "experiment_root": tmp_path / "runs",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    trainer = paths["repo_root"] / "src" / "bts_nvs" / "training" / "run_training.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# trainer placeholder\n", encoding="utf-8")
    (paths["scenes_root"] / "HCM0644").mkdir()
    manifest_dir = paths["manifests_root"] / "HCM0644"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (manifest_dir / "holdout.json").write_text("{}", encoding="utf-8")
    return paths


def _patch_valid_inputs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def _run(argv, **kwargs):
        assert kwargs["shell"] is False
        calls.append(list(argv))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.load_or_create_backend_decision",
        lambda path: SimpleNamespace(
            optimizer_backend="adam-fused",
            precision="amp-fp16",
            report_sha256="b" * 64,
        ),
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.load_scene_manifest",
        lambda manifest, scene: SimpleNamespace(scene_id="HCM0644"),
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.load_holdout_split",
        lambda path, manifest: SimpleNamespace(
            manifest_sha256="h" * 64,
            validation_image_names=("a.jpg", "b.jpg"),
        ),
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.compute_manifest_sha256",
        lambda path: "m" * 64,
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.subprocess.run",
        _run,
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment._config_sha256",
        lambda path: "c" * 64,
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment._validate_backend_config",
        lambda path, decision: None,
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.validate_run_artifacts",
        lambda *args, **kwargs: ArtifactValidationResult(None, None, True, True),
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.validate_recovery_checkpoint",
        lambda *args, **kwargs: 15_000,
    )
    return calls


def test_run_one_launches_existing_trainer_once_then_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    calls = _patch_valid_inputs(monkeypatch)

    result = run_one(
        **paths,
        stage=ExperimentStage.SCREEN,
        scene_id="HCM0644",
        candidate_id="E1-density-absgrad-t04-v1",
        stop_step=7_000,
    )

    assert result.integrity_passed is True
    assert len(calls) == 1
    command = calls[0]
    assert Path(command[1]).as_posix().endswith("src/bts_nvs/training/run_training.py")
    assert Path(command[command.index("--output_dir") + 1]).as_posix().endswith(
        "screen/HCM0644/E1-density-absgrad-t04-v1"
    )
    assert "--internal_holdout" in command


def test_preflight_rejects_nonempty_fresh_output_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    calls = _patch_valid_inputs(monkeypatch)
    output = paths["experiment_root"] / "screen" / "HCM0644" / "E1-density-absgrad-t04-v1"
    output.mkdir(parents=True)
    (output / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        run_one(
            **paths,
            stage=ExperimentStage.SCREEN,
            scene_id="HCM0644",
            candidate_id="E1-density-absgrad-t04-v1",
            stop_step=7_000,
        )

    assert calls == []


def test_confirm_requires_verified_scene_winner_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    calls = _patch_valid_inputs(monkeypatch)
    decision_path = tmp_path / "scene.json"
    save_json_artifact(
        _scene_decision(
            "HCM0644",
            "E1-density-scale005-v1",
            step=7_000,
            decision_stage="screen",
        ),
        decision_path,
    )

    with pytest.raises(ValueError, match="authorize"):
        run_one(
            **paths,
            stage=ExperimentStage.CONFIRM,
            scene_id="HCM0644",
            candidate_id="E1-density-absgrad-t04-v1",
            stop_step=15_000,
            scene_decision_path=decision_path,
        )

    assert calls == []


def test_confirm_launches_only_a_screen_qualified_scene_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    calls = _patch_valid_inputs(monkeypatch)
    candidate_id = "E1-density-absgrad-t04-v1"
    decision = _scene_decision(
        "HCM0644", candidate_id, step=7_000, decision_stage="screen"
    )
    unhashed = dict(decision)
    del unhashed["decision_sha256"]
    unhashed["fallback_to_b0"] = False
    unhashed["evaluations"] = [
        {"candidate_id": candidate_id, "screen_qualified": True}
    ]
    unhashed["decision_sha256"] = canonical_json_sha256(unhashed)
    decision_path = tmp_path / "screen-winner.json"
    save_json_artifact(unhashed, decision_path)

    run_one(
        **paths,
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id=candidate_id,
        stop_step=15_000,
        scene_decision_path=decision_path,
    )

    assert len(calls) == 1


def test_production_requires_verified_cohort_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    calls = _patch_valid_inputs(monkeypatch)
    decisions = [_scene_decision(scene, "B0-reference") for scene in COHORT_SCENE_IDS]
    cohort_path = tmp_path / "cohort.json"
    save_json_artifact(build_cohort_decision(decisions), cohort_path)

    run_one(
        **paths,
        stage=ExperimentStage.PRODUCTION,
        scene_id="HCM0644",
        candidate_id="B0-reference",
        stop_step=30_000,
        cohort_decision_path=cohort_path,
    )

    assert len(calls) == 1
    assert "--internal_holdout" not in calls[0]


def test_resume_uses_only_the_rolling_recovery_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    calls = _patch_valid_inputs(monkeypatch)
    output = paths["experiment_root"] / "confirm" / "HCM0644" / "B0-reference"
    recovery = output / "checkpoints" / "recovery.pt"
    recovery.parent.mkdir(parents=True)
    recovery.write_bytes(b"checkpoint")

    run_one(
        **paths,
        stage=ExperimentStage.CONFIRM,
        scene_id="HCM0644",
        candidate_id="B0-reference",
        stop_step=30_000,
        resume=True,
    )

    assert calls[0][calls[0].index("--resume") + 1] == str(recovery)


def test_subprocess_failure_is_recorded_and_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    _patch_valid_inputs(monkeypatch)
    records = []
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.append_failure",
        lambda *args, **kwargs: records.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="code 9"):
        run_one(
            **paths,
            stage=ExperimentStage.SCREEN,
            scene_id="HCM0644",
            candidate_id="E1-density-absgrad-t04-v1",
            stop_step=7_000,
        )

    assert len(records) == 1
    assert "code 9" in records[0]["reason"]


def test_cli_parses_one_run_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.run_one",
        lambda **kwargs: captured.append(kwargs),
    )

    assert main(
        [
            "run",
            "--repo-root", str(tmp_path),
            "--scenes-root", str(tmp_path),
            "--manifests-root", str(tmp_path),
            "--backend-root", str(tmp_path),
            "--experiment-root", str(tmp_path),
            "--stage", "screen",
            "--scene-id", "HCM0644",
            "--candidate-id", "E1-density-absgrad-t04-v1",
            "--stop-step", "7000",
        ]
    ) == 0
    assert captured[0]["stage"] is ExperimentStage.SCREEN
    assert captured[0]["resume"] is False


def test_validate_existing_reuses_full_artifact_validation_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    calls = _patch_valid_inputs(monkeypatch)
    output = paths["experiment_root"] / "reference" / "HCM0644"
    output.mkdir(parents=True)
    (output / "config.yaml").write_text("candidate_id: B0-reference\n", encoding="utf-8")

    result = validate_existing(
        **paths,
        stage=ExperimentStage.REFERENCE,
        scene_id="HCM0644",
        candidate_id="B0-reference",
        stop_step=7_000,
    )

    assert result.integrity_passed is True
    assert calls == []


def test_validate_existing_requires_a_completed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    _patch_valid_inputs(monkeypatch)

    with pytest.raises(FileNotFoundError, match="existing experiment"):
        validate_existing(
            **paths,
            stage=ExperimentStage.REFERENCE,
            scene_id="HCM0644",
            candidate_id="B0-reference",
            stop_step=7_000,
        )


def test_existing_config_must_match_accepted_backend(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "optimizer_backend: adam\nprecision: fp32\n",
        encoding="utf-8",
    )
    decision = SimpleNamespace(
        optimizer_backend="adam-fused",
        precision="amp-fp16",
    )

    with pytest.raises(ValueError, match="backend|precision"):
        _validate_backend_config(config, decision)


def test_cli_validate_dispatches_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    monkeypatch.setattr(
        "bts_nvs.experiments.run_experiment.validate_existing",
        lambda **kwargs: captured.append(kwargs),
    )

    assert main(
        [
            "validate",
            "--repo-root", str(tmp_path),
            "--scenes-root", str(tmp_path),
            "--manifests-root", str(tmp_path),
            "--backend-root", str(tmp_path),
            "--experiment-root", str(tmp_path),
            "--stage", "reference",
            "--scene-id", "HCM0644",
            "--candidate-id", "B0-reference",
            "--stop-step", "7000",
        ]
    ) == 0
    assert captured[0]["stage"] is ExperimentStage.REFERENCE
    assert "resume" not in captured[0]
