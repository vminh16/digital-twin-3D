from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from bts_nvs.data.holdout import load_holdout_split
from bts_nvs.data.manifest import load_scene_manifest
from bts_nvs.experiments.artifacts import (
    ArtifactValidationResult,
    append_failure,
    validate_run_artifacts,
)
from bts_nvs.experiments.commands import build_training_command
from bts_nvs.experiments.experiment import Experiment, ExperimentStage
from bts_nvs.experiments.provenance import (
    canonical_json_sha256,
    load_json_artifact,
)
from bts_nvs.training.full_training import (
    BackendDecision,
    load_or_create_backend_decision,
)
from bts_nvs.training.run_training import validate_recovery_checkpoint
from bts_nvs.training.trainer import compute_config_sha256, compute_manifest_sha256


_RECOVERY_PATH = Path("checkpoints") / "recovery.pt"


@dataclass(frozen=True)
class _PreparedRun:
    experiment: Experiment
    repo: Path
    root: Path
    output: Path
    scene_dir: Path
    manifest_dir: Path
    manifest_sha256: str
    holdout_sha256: str | None
    expected_names: tuple[str, ...]
    b0_report: dict[str, object] | None
    backend_decision: BackendDecision


def run_one(
    *,
    repo_root: str | Path,
    scenes_root: str | Path,
    manifests_root: str | Path,
    backend_root: str | Path,
    experiment_root: str | Path,
    stage: ExperimentStage,
    scene_id: str,
    candidate_id: str,
    stop_step: int,
    resume: bool = False,
    scene_decision_path: str | Path | None = None,
    cohort_decision_path: str | Path | None = None,
    b0_report_path: str | Path | None = None,
) -> ArtifactValidationResult:
    """Preflight, launch, and validate one stage-first experiment run."""
    prepared = _prepare_run(
        repo_root=repo_root,
        scenes_root=scenes_root,
        manifests_root=manifests_root,
        backend_root=backend_root,
        experiment_root=experiment_root,
        stage=stage,
        scene_id=scene_id,
        candidate_id=candidate_id,
        b0_report_path=b0_report_path,
        scene_decision_path=scene_decision_path,
        cohort_decision_path=cohort_decision_path,
    )
    recovery = prepared.output / _RECOVERY_PATH if resume else None
    _validate_output_state(prepared.output, recovery)
    if recovery is not None:
        _validate_backend_config(
            prepared.output / "config.yaml", prepared.backend_decision
        )
        recovery_step = validate_recovery_checkpoint(
            recovery,
            prepared.manifest_sha256,
            _config_sha256(prepared.output / "config.yaml"),
            15_000
            if prepared.experiment.stage is ExperimentStage.CONFIRM
            else None,
            require_precision_state=True,
        )
        if recovery_step >= stop_step:
            raise ValueError("recovery checkpoint must precede the requested stop step")

    command = build_training_command(
        repo_root=prepared.repo,
        scene_dir=prepared.scene_dir,
        manifest_dir=prepared.manifest_dir,
        output_dir=prepared.output,
        optimizer_backend=prepared.backend_decision.optimizer_backend,
        precision=prepared.backend_decision.precision,
        experiment=prepared.experiment,
        stop_step=stop_step,
        resume_path=recovery,
    )
    provenance = {
        "backend_report_sha256": prepared.backend_decision.report_sha256,
        "manifest_sha256": prepared.manifest_sha256,
        "holdout_sha256": prepared.holdout_sha256,
    }
    try:
        completed = subprocess.run(
            command, cwd=prepared.repo, check=False, shell=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"experiment training exited with code {completed.returncode}"
            )
        _validate_backend_config(
            prepared.output / "config.yaml", prepared.backend_decision
        )
        config_sha256 = _config_sha256(prepared.output / "config.yaml")
        return validate_run_artifacts(
            prepared.output,
            prepared.experiment,
            manifest_sha256=prepared.manifest_sha256,
            config_sha256=config_sha256,
            expected_image_names=prepared.expected_names,
            holdout_sha256=prepared.holdout_sha256,
            step=stop_step,
            b0_report=prepared.b0_report,
        )
    except Exception as error:
        append_failure(
            prepared.root / "failures.jsonl",
            experiment=prepared.experiment,
            command_argv=command,
            reason=f"{type(error).__name__}: {error}",
            provenance=provenance,
        )
        raise


def validate_existing(
    *,
    repo_root: str | Path,
    scenes_root: str | Path,
    manifests_root: str | Path,
    backend_root: str | Path,
    experiment_root: str | Path,
    stage: ExperimentStage,
    scene_id: str,
    candidate_id: str,
    stop_step: int,
    scene_decision_path: str | Path | None = None,
    cohort_decision_path: str | Path | None = None,
    b0_report_path: str | Path | None = None,
) -> ArtifactValidationResult:
    """Validate a completed run without launching training or mutating it."""
    prepared = _prepare_run(
        repo_root=repo_root,
        scenes_root=scenes_root,
        manifests_root=manifests_root,
        backend_root=backend_root,
        experiment_root=experiment_root,
        stage=stage,
        scene_id=scene_id,
        candidate_id=candidate_id,
        b0_report_path=b0_report_path,
        scene_decision_path=scene_decision_path,
        cohort_decision_path=cohort_decision_path,
    )
    if not prepared.output.is_dir() or not any(prepared.output.iterdir()):
        raise FileNotFoundError(
            f"existing experiment output does not exist: {prepared.output}"
        )
    _validate_backend_config(
        prepared.output / "config.yaml", prepared.backend_decision
    )
    return validate_run_artifacts(
        prepared.output,
        prepared.experiment,
        manifest_sha256=prepared.manifest_sha256,
        config_sha256=_config_sha256(prepared.output / "config.yaml"),
        expected_image_names=prepared.expected_names,
        holdout_sha256=prepared.holdout_sha256,
        step=stop_step,
        b0_report=prepared.b0_report,
    )


def _prepare_run(
    *,
    repo_root: str | Path,
    scenes_root: str | Path,
    manifests_root: str | Path,
    backend_root: str | Path,
    experiment_root: str | Path,
    stage: ExperimentStage,
    scene_id: str,
    candidate_id: str,
    b0_report_path: str | Path | None,
    scene_decision_path: str | Path | None,
    cohort_decision_path: str | Path | None,
) -> _PreparedRun:
    stage = _stage(stage)
    authorized_scene_winner, authorized_cohort_candidate = _authorization(
        stage=stage,
        scene_id=scene_id,
        candidate_id=candidate_id,
        scene_decision_path=scene_decision_path,
        cohort_decision_path=cohort_decision_path,
    )
    experiment = Experiment(
        stage=stage,
        scene_id=scene_id,
        candidate_id=candidate_id,
        authorized_scene_winner=authorized_scene_winner,
        authorized_cohort_candidate=authorized_cohort_candidate,
    )
    repo = Path(repo_root).resolve()
    scenes = Path(scenes_root).resolve()
    manifests = Path(manifests_root).resolve()
    root = Path(experiment_root).resolve()
    scene_dir = scenes / scene_id
    manifest_dir = manifests / scene_id
    manifest_path = manifest_dir / "manifest.json"
    _require_file(
        repo / "src" / "bts_nvs" / "training" / "run_training.py", "trainer"
    )
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory does not exist: {scene_dir}")
    _require_file(manifest_path, "manifest")
    manifest = load_scene_manifest(manifest_path, scene_dir)
    if manifest.scene_id != scene_id:
        raise ValueError("manifest scene_id does not match requested scene")

    expected_names: tuple[str, ...] = ()
    holdout_sha256 = None
    if stage is not ExperimentStage.PRODUCTION:
        _require_file(manifest_dir / "holdout.json", "holdout")
        split = load_holdout_split(manifest_dir / "holdout.json", manifest)
        holdout_sha256 = split.manifest_sha256
        expected_names = tuple(split.validation_image_names)

    return _PreparedRun(
        experiment=experiment,
        repo=repo,
        root=root,
        output=experiment.run_path(root),
        scene_dir=scene_dir,
        manifest_dir=manifest_dir,
        manifest_sha256=compute_manifest_sha256(manifest_path),
        holdout_sha256=holdout_sha256,
        expected_names=expected_names,
        b0_report=(
            None
            if b0_report_path is None
            else load_json_artifact(Path(b0_report_path))
        ),
        backend_decision=load_or_create_backend_decision(
            Path(backend_root).resolve()
        ),
    )


def _authorization(
    *,
    stage: ExperimentStage,
    scene_id: str,
    candidate_id: str,
    scene_decision_path: str | Path | None,
    cohort_decision_path: str | Path | None,
) -> tuple[str | None, str | None]:
    if stage in (ExperimentStage.REFERENCE, ExperimentStage.SCREEN):
        if scene_decision_path is not None or cohort_decision_path is not None:
            raise ValueError("reference/screen do not accept authorization artifacts")
        return None, None
    if stage is ExperimentStage.CONFIRM:
        if cohort_decision_path is not None:
            raise ValueError("confirm does not accept a cohort decision")
        if candidate_id == "B0-reference":
            if scene_decision_path is not None:
                raise ValueError("B0 confirmation does not require scene authorization")
            return candidate_id, None
        if scene_decision_path is None:
            raise ValueError("confirm candidate requires a scene decision")
        decision = _verified_embedded_hash(
            Path(scene_decision_path), "decision_sha256"
        )
        if (
            decision.get("scene_id") != scene_id
            or decision.get("decision_stage") != "screen"
            or decision.get("step") != 7_000
            or decision.get("selected_candidate_id") != candidate_id
            or decision.get("fallback_to_b0") is not False
            or not _screen_evaluation_passed(decision, candidate_id)
        ):
            raise ValueError("scene decision does not authorize requested candidate")
        return candidate_id, None

    if scene_decision_path is not None:
        raise ValueError("production does not accept a scene decision")
    if cohort_decision_path is None:
        raise ValueError("production requires a cohort decision")
    cohort = _verified_embedded_hash(Path(cohort_decision_path), "cohort_sha256")
    scenes = cohort.get("scenes")
    if not isinstance(scenes, Mapping):
        raise ValueError("cohort decision scenes must be a mapping")
    choice = scenes.get(scene_id)
    if not isinstance(choice, Mapping) or choice.get("candidate_id") != candidate_id:
        raise ValueError("cohort decision does not authorize requested candidate")
    return None, candidate_id


def _verified_embedded_hash(path: Path, hash_field: str) -> dict[str, object]:
    record = load_json_artifact(path)
    digest = record.get(hash_field)
    if not isinstance(digest, str):
        raise ValueError(f"{hash_field} is missing")
    unhashed = dict(record)
    del unhashed[hash_field]
    if canonical_json_sha256(unhashed) != digest:
        raise ValueError(f"{hash_field} does not match artifact contents")
    return record


def _validate_output_state(output: Path, recovery: Path | None) -> None:
    if recovery is not None:
        if not recovery.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {recovery}")
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output}")


def _screen_evaluation_passed(
    decision: Mapping[str, object], candidate_id: str
) -> bool:
    evaluations = decision.get("evaluations")
    if not isinstance(evaluations, list):
        return False
    return any(
        isinstance(evaluation, Mapping)
        and evaluation.get("candidate_id") == candidate_id
        and evaluation.get("screen_qualified") is True
        for evaluation in evaluations
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} does not exist: {path}")


def _config_sha256(path: Path) -> str:
    return compute_config_sha256(_load_config(path))


def _validate_backend_config(path: Path, decision: BackendDecision) -> None:
    config = _load_config(path)
    expected = {
        "optimizer_backend": decision.optimizer_backend,
        "precision": decision.precision,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"completed config {field} does not match accepted backend"
            )


def _load_config(path: Path) -> dict[str, object]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load completed config: {path}") from error
    if not isinstance(config, dict):
        raise ValueError("completed config must contain a mapping")
    return dict(config)


def _stage(value: ExperimentStage | str) -> ExperimentStage:
    if isinstance(value, ExperimentStage):
        return value
    try:
        return ExperimentStage(value)
    except (TypeError, ValueError) as error:
        raise ValueError("stage is unsupported") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one generic BTS experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one scene/candidate/stage")
    validate = subparsers.add_parser(
        "validate", help="validate one existing scene/candidate/stage"
    )
    for command in (run, validate):
        _add_common_arguments(command)
    run.add_argument("--resume", action="store_true")
    return parser


def _add_common_arguments(command: argparse.ArgumentParser) -> None:
    for name in (
        "repo-root",
        "scenes-root",
        "manifests-root",
        "backend-root",
        "experiment-root",
        "scene-id",
        "candidate-id",
    ):
        command.add_argument(f"--{name}", required=True)
    command.add_argument(
        "--stage",
        choices=tuple(stage.value for stage in ExperimentStage),
        required=True,
    )
    command.add_argument("--stop-step", type=int, required=True)
    command.add_argument("--scene-decision")
    command.add_argument("--cohort-decision")
    command.add_argument("--b0-report")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = dict(
        repo_root=args.repo_root,
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        backend_root=args.backend_root,
        experiment_root=args.experiment_root,
        stage=ExperimentStage(args.stage),
        scene_id=args.scene_id,
        candidate_id=args.candidate_id,
        stop_step=args.stop_step,
        scene_decision_path=args.scene_decision,
        cohort_decision_path=args.cohort_decision,
        b0_report_path=args.b0_report,
    )
    if args.command == "run":
        run_one(**common, resume=args.resume)
    elif args.command == "validate":
        validate_existing(**common)
    else:
        raise RuntimeError("unsupported command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
