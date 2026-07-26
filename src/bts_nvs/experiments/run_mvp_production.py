from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from bts_nvs.experiments.artifacts import validate_run_artifacts
from bts_nvs.experiments.commands import build_training_command
from bts_nvs.experiments.experiment import Experiment, ExperimentStage
from bts_nvs.experiments.provenance import (
    canonical_json_sha256,
    load_json_artifact,
)
from bts_nvs.experiments.run_experiment import validate_existing
from bts_nvs.training.full_training import (
    inspect_scene_run,
    load_or_create_backend_decision,
)
from bts_nvs.training.trainer import compute_config_sha256, compute_manifest_sha256


MVP_SCENE_IDS = ("HCM0421", "HCM0539")
MVP_CANDIDATE_IDS = {
    "HCM0421": "E1-density-absgrad-t04-v1",
    "HCM0539": "E1-density-absgrad-t04-v1",
    "chair": "E2-loss-local-laplacian-v1",
    "bonsai": "E2-appearance-sh4-v1",
}
DECISION_REQUIRED_SCENES = frozenset(("chair", "bonsai"))


def validate_scene_ids(scene_ids: Sequence[str] | None) -> tuple[str, ...]:
    selected = MVP_SCENE_IDS if scene_ids is None else tuple(scene_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("scene_ids must be a non-empty unique sequence")
    unsupported = tuple(
        scene_id for scene_id in selected if scene_id not in MVP_CANDIDATE_IDS
    )
    if unsupported:
        raise ValueError(f"unsupported MVP scene_ids: {unsupported}")
    return selected


def validate_screen_evidence(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    backend_root: Path,
    experiment_root: Path,
    reference_root: Path | None = None,
    scene_id: str,
) -> None:
    candidate_id = MVP_CANDIDATE_IDS[scene_id]
    b0_root = experiment_root if reference_root is None else reference_root
    b0_report = b0_root / "reference" / scene_id / "experiment_report.json"
    common = {
        "repo_root": repo_root,
        "scenes_root": scenes_root,
        "manifests_root": manifests_root,
        "backend_root": backend_root,
        "scene_id": scene_id,
        "stop_step": 7_000,
    }
    validate_existing(
        **common,
        experiment_root=b0_root,
        stage=ExperimentStage.REFERENCE,
        candidate_id="B0-reference",
    )
    validate_existing(
        **common,
        experiment_root=experiment_root,
        stage=ExperimentStage.SCREEN,
        candidate_id=candidate_id,
        b0_report_path=b0_report,
    )
    if scene_id in DECISION_REQUIRED_SCENES:
        _validate_selected_screen_decision(
            experiment_root / "decisions" / "screen" / f"{scene_id}.json",
            scene_id=scene_id,
            candidate_id=candidate_id,
        )


def run_mvp_production(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    backend_root: Path,
    experiment_root: Path,
    reference_root: Path | None = None,
    output_root: Path,
    python_bin: str = sys.executable,
    scene_ids: Sequence[str] | None = None,
    run_process: Callable[..., object] = subprocess.run,
) -> None:
    selected = validate_scene_ids(scene_ids)
    decision = load_or_create_backend_decision(backend_root)

    for scene_id in selected:
        validate_screen_evidence(
            repo_root=repo_root,
            scenes_root=scenes_root,
            manifests_root=manifests_root,
            backend_root=backend_root,
            experiment_root=experiment_root,
            reference_root=reference_root,
            scene_id=scene_id,
        )

        candidate_id = MVP_CANDIDATE_IDS[scene_id]
        experiment = Experiment(
            stage=ExperimentStage.PRODUCTION,
            scene_id=scene_id,
            candidate_id=candidate_id,
            authorized_cohort_candidate=candidate_id,
        )
        run_dir = output_root / "scenes" / scene_id
        manifest_path = manifests_root / scene_id / "manifest.json"
        state = inspect_scene_run(run_dir, scene_id, manifest_path, decision)
        if state != "fresh":
            _validate_mvp_config(run_dir / "config.yaml", experiment)
        if state == "trained":
            _validate_completed_production(
                run_dir=run_dir,
                manifest_path=manifest_path,
                experiment=experiment,
            )
            print(f"MVP production already complete: {scene_id}")
            continue

        recovery = (
            run_dir / "checkpoints" / "recovery.pt" if state == "resume" else None
        )
        command = build_training_command(
            repo_root=repo_root,
            scene_dir=scenes_root / scene_id,
            manifest_dir=manifests_root / scene_id,
            output_dir=run_dir,
            optimizer_backend=decision.optimizer_backend,
            precision=decision.precision,
            experiment=experiment,
            stop_step=30_000,
            resume_path=recovery,
        )
        completed = run_process(command, cwd=repo_root, check=False, shell=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"MVP production training failed for {scene_id}: "
                f"exit code {completed.returncode}"
            )
        _validate_completed_production(
            run_dir=run_dir,
            manifest_path=manifest_path,
            experiment=experiment,
        )


def _validate_mvp_config(path: Path, experiment: Experiment) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "scene_id": experiment.scene_id,
        "candidate_id": experiment.candidate_id,
        "experiment_stage": ExperimentStage.PRODUCTION.value,
        "max_steps": 30_000,
        "internal_holdout": False,
    }
    if not isinstance(config, dict) or any(
        config.get(field) != value for field, value in expected.items()
    ):
        raise ValueError(
            f"existing run does not match the MVP production identity: {path}"
        )
    return dict(config)


def _validate_selected_screen_decision(
    path: Path,
    *,
    scene_id: str,
    candidate_id: str,
) -> None:
    decision = load_json_artifact(path)
    digest = decision.get("decision_sha256")
    unhashed = dict(decision)
    unhashed.pop("decision_sha256", None)
    evaluations = decision.get("evaluations")
    selected_passed = isinstance(evaluations, list) and any(
        isinstance(evaluation, dict)
        and evaluation.get("candidate_id") == candidate_id
        and evaluation.get("screen_qualified") is True
        for evaluation in evaluations
    )
    if (
        not isinstance(digest, str)
        or canonical_json_sha256(unhashed) != digest
        or decision.get("scene_id") != scene_id
        or decision.get("step") != 7_000
        or decision.get("decision_stage") != "screen"
        or decision.get("selected_candidate_id") != candidate_id
        or decision.get("fallback_to_b0") is not False
        or not selected_passed
    ):
        raise ValueError(f"screen decision does not authorize MVP production: {path}")


def _validate_completed_production(
    *,
    run_dir: Path,
    manifest_path: Path,
    experiment: Experiment,
) -> None:
    config_path = run_dir / "config.yaml"
    config = _validate_mvp_config(config_path, experiment)
    validate_run_artifacts(
        run_dir,
        experiment,
        manifest_sha256=compute_manifest_sha256(manifest_path),
        config_sha256=compute_config_sha256(config),
        expected_image_names=(),
        step=30_000,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train approved deadline-exception MVP production scenes"
    )
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--backend_root", type=Path, required=True)
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("scene_ids", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    run_mvp_production(
        repo_root=args.repo_root.resolve(),
        scenes_root=args.scenes_root.resolve(),
        manifests_root=args.manifests_root.resolve(),
        backend_root=args.backend_root.resolve(),
        experiment_root=args.experiment_root.resolve(),
        reference_root=(
            None if args.reference_root is None else args.reference_root.resolve()
        ),
        output_root=args.output_root.resolve(),
        python_bin=args.python_bin,
        scene_ids=args.scene_ids or None,
    )


if __name__ == "__main__":
    main()
