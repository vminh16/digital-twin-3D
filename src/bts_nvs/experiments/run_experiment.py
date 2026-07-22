from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping, Sequence
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
from bts_nvs.training.full_training import load_or_create_backend_decision
from bts_nvs.training.run_training import validate_recovery_checkpoint
from bts_nvs.training.trainer import compute_config_sha256, compute_manifest_sha256


_RECOVERY_PATH = Path("checkpoints") / "recovery.pt"


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
    backend = Path(backend_root).resolve()
    root = Path(experiment_root).resolve()
    output = experiment.run_path(root)
    recovery = output / _RECOVERY_PATH if resume else None
    _require_file(repo / "src" / "bts_nvs" / "training" / "run_training.py", "trainer")
    _validate_output_state(output, recovery)

    scene_dir = scenes / scene_id
    manifest_dir = manifests / scene_id
    manifest_path = manifest_dir / "manifest.json"
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory does not exist: {scene_dir}")
    _require_file(manifest_path, "manifest")
    manifest = load_scene_manifest(manifest_path, scene_dir)
    if manifest.scene_id != scene_id:
        raise ValueError("manifest scene_id does not match requested scene")
    manifest_sha256 = compute_manifest_sha256(manifest_path)
    if recovery is not None:
        recovery_step = validate_recovery_checkpoint(
            recovery,
            manifest_sha256,
            _config_sha256(output / "config.yaml"),
            15_000 if stage is ExperimentStage.CONFIRM else None,
            require_precision_state=True,
        )
        if recovery_step >= stop_step:
            raise ValueError("recovery checkpoint must precede the requested stop step")

    expected_names: tuple[str, ...] = ()
    holdout_sha256 = None
    if stage is not ExperimentStage.PRODUCTION:
        _require_file(manifest_dir / "holdout.json", "holdout")
        split = load_holdout_split(manifest_dir / "holdout.json", manifest)
        holdout_sha256 = split.manifest_sha256
        expected_names = tuple(split.validation_image_names)

    b0_report = (
        None
        if b0_report_path is None
        else load_json_artifact(Path(b0_report_path))
    )

    backend_decision = load_or_create_backend_decision(backend)
    command = build_training_command(
        repo_root=repo,
        scene_dir=scene_dir,
        manifest_dir=manifest_dir,
        output_dir=output,
        optimizer_backend=backend_decision.optimizer_backend,
        precision=backend_decision.precision,
        experiment=experiment,
        stop_step=stop_step,
        resume_path=recovery,
    )
    provenance = {
        "backend_report_sha256": backend_decision.report_sha256,
        "manifest_sha256": manifest_sha256,
        "holdout_sha256": holdout_sha256,
    }
    try:
        completed = subprocess.run(command, cwd=repo, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"experiment training exited with code {completed.returncode}"
            )
        config_sha256 = _config_sha256(output / "config.yaml")
        return validate_run_artifacts(
            output,
            experiment,
            manifest_sha256=manifest_sha256,
            config_sha256=config_sha256,
            expected_image_names=expected_names,
            holdout_sha256=holdout_sha256,
            step=stop_step,
            b0_report=b0_report,
        )
    except Exception as error:
        append_failure(
            root / "failures.jsonl",
            experiment=experiment,
            command_argv=command,
            reason=f"{type(error).__name__}: {error}",
            provenance=provenance,
        )
        raise


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
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load completed config: {path}") from error
    if not isinstance(config, dict):
        raise ValueError("completed config must contain a mapping")
    return compute_config_sha256(config)


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
    for name in (
        "repo-root",
        "scenes-root",
        "manifests-root",
        "backend-root",
        "experiment-root",
        "scene-id",
        "candidate-id",
    ):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--stage", choices=tuple(stage.value for stage in ExperimentStage), required=True)
    run.add_argument("--stop-step", type=int, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--scene-decision")
    run.add_argument("--cohort-decision")
    run.add_argument("--b0-report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "run":
        raise RuntimeError("unsupported command")
    run_one(
        repo_root=args.repo_root,
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        backend_root=args.backend_root,
        experiment_root=args.experiment_root,
        stage=ExperimentStage(args.stage),
        scene_id=args.scene_id,
        candidate_id=args.candidate_id,
        stop_step=args.stop_step,
        resume=args.resume,
        scene_decision_path=args.scene_decision,
        cohort_decision_path=args.cohort_decision,
        b0_report_path=args.b0_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
