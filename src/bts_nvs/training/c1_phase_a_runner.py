from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from bts_nvs.data.dataset import SceneDataset
from bts_nvs.data.holdout import load_holdout_split
from bts_nvs.data.manifest import load_scene_manifest
from bts_nvs.evaluation.high_frequency import evaluate_render_directory
from bts_nvs.training.c1_candidates import C1_CANDIDATES
from bts_nvs.training.c1_phase_a import (
    BASELINE_CANDIDATE,
    PHASE_A_SCENES,
    build_phase_a_decision,
    save_phase_a_decision,
    score50,
)
from bts_nvs.training.full_training import (
    BackendDecision,
    load_or_create_backend_decision,
)


def _atomic_json(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)


def _load_report(path: Path, scene_id: str, candidate_id: str) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing qualification report: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"qualification report must contain an object: {source}")
    if (
        report.get("schema_version") != 1
        or report.get("step") != 7000
        or report.get("scene_id") != scene_id
        or report.get("candidate_id") != candidate_id
    ):
        raise ValueError(f"qualification report identity mismatch: {source}")
    score50(report)
    return report


def build_phase_a_command(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    output_root: Path,
    scene_id: str,
    candidate_id: str,
    decision: BackendDecision,
    python_bin: str,
) -> list[str]:
    if scene_id not in PHASE_A_SCENES:
        raise ValueError(f"unexpected Phase A scene: {scene_id}")
    if candidate_id not in C1_CANDIDATES:
        raise ValueError(f"unexpected Phase A candidate: {candidate_id}")
    return [
        python_bin,
        str(Path(repo_root) / "src" / "bts_nvs" / "training" / "run_training.py"),
        "--scene_dir",
        str(Path(scenes_root) / scene_id),
        "--manifest_dir",
        str(Path(manifests_root) / scene_id),
        "--output_dir",
        str(Path(output_root) / scene_id / candidate_id),
        "--resize_factor",
        "1",
        "--max_steps",
        "7000",
        "--seed",
        "0",
        "--cache_images",
        "--pinned_transfer",
        "--qualification_candidate",
        candidate_id,
        "--optimizer_backend",
        decision.optimizer_backend,
        "--precision",
        decision.precision,
    ]


def _diagnostics_for_run(
    *,
    scene_id: str,
    candidate_id: str,
    scene_root: Path,
    manifest_root: Path,
    render_dir: Path,
) -> dict:
    manifest = load_scene_manifest(Path(manifest_root) / "manifest.json", scene_root)
    split = load_holdout_split(Path(manifest_root) / "holdout.json", manifest)
    reference = manifest.train_intrinsics[0]
    dataset = SceneDataset(
        manifest,
        scene_root,
        image_names=split.validation_image_names,
        undistort=True,
        resize=(reference.width, reference.height),
        cache_images=False,
    )
    report = evaluate_render_directory(dataset, render_dir)
    if report["scene_id"] != scene_id:
        raise ValueError("diagnostic scene identity mismatch")
    report["candidate_id"] = candidate_id
    return report


def _require_phase_a_inputs(
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    baseline_root: Path,
) -> None:
    entrypoint = Path(repo_root) / "src" / "bts_nvs" / "training" / "run_training.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"training entry point does not exist: {entrypoint}")
    for scene_id in PHASE_A_SCENES:
        if not (Path(scenes_root) / scene_id).is_dir():
            raise FileNotFoundError(f"Phase A scene does not exist: {scene_id}")
        for name in ("manifest.json", "holdout.json"):
            path = Path(manifests_root) / scene_id / name
            if not path.is_file():
                raise FileNotFoundError(f"Phase A artifact does not exist: {path}")
        baseline = Path(baseline_root) / scene_id / BASELINE_CANDIDATE
        _load_report(
            baseline / "qualification_report.json",
            scene_id,
            BASELINE_CANDIDATE,
        )
        if not (baseline / "validation_renders").is_dir():
            raise FileNotFoundError(
                f"baseline validation renders do not exist: {baseline}"
            )


def run_phase_a(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    backend_root: Path,
    baseline_root: Path,
    output_root: Path,
    python_bin: str,
    run_process: Callable[..., object] = subprocess.run,
) -> dict:
    _require_phase_a_inputs(repo_root, scenes_root, manifests_root, baseline_root)
    backend = load_or_create_backend_decision(backend_root)
    candidate_reports: list[dict] = []

    for scene_id in PHASE_A_SCENES:
        for candidate_id in C1_CANDIDATES:
            run_dir = Path(output_root) / scene_id / candidate_id
            report_path = run_dir / "qualification_report.json"
            if report_path.is_file():
                report = _load_report(report_path, scene_id, candidate_id)
            else:
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise ValueError(
                        f"Phase A run directory is non-empty without a report: {run_dir}"
                    )
                command = build_phase_a_command(
                    repo_root=repo_root,
                    scenes_root=scenes_root,
                    manifests_root=manifests_root,
                    output_root=output_root,
                    scene_id=scene_id,
                    candidate_id=candidate_id,
                    decision=backend,
                    python_bin=python_bin,
                )
                result = run_process(command, check=False)
                returncode = getattr(result, "returncode", None)
                if returncode != 0:
                    raise RuntimeError(
                        f"Phase A training exited with code {returncode} for "
                        f"{scene_id}/{candidate_id}"
                    )
                report = _load_report(report_path, scene_id, candidate_id)
            candidate_reports.append(report)

    baseline_reports: list[dict] = []
    diagnostics: list[dict] = []
    for scene_id in PHASE_A_SCENES:
        baseline_run = Path(baseline_root) / scene_id / BASELINE_CANDIDATE
        baseline_reports.append(
            _load_report(
                baseline_run / "qualification_report.json",
                scene_id,
                BASELINE_CANDIDATE,
            )
        )
        baseline_diagnostic = _diagnostics_for_run(
            scene_id=scene_id,
            candidate_id=BASELINE_CANDIDATE,
            scene_root=Path(scenes_root) / scene_id,
            manifest_root=Path(manifests_root) / scene_id,
            render_dir=baseline_run / "validation_renders",
        )
        _atomic_json(
            Path(output_root) / "baseline_diagnostics" / f"{scene_id}.json",
            baseline_diagnostic,
        )
        diagnostics.append(baseline_diagnostic)

        for candidate_id in C1_CANDIDATES:
            run_dir = Path(output_root) / scene_id / candidate_id
            diagnostic = _diagnostics_for_run(
                scene_id=scene_id,
                candidate_id=candidate_id,
                scene_root=Path(scenes_root) / scene_id,
                manifest_root=Path(manifests_root) / scene_id,
                render_dir=run_dir / "validation_renders",
            )
            _atomic_json(run_dir / "high_frequency.json", diagnostic)
            diagnostics.append(diagnostic)

    decision = build_phase_a_decision(
        baseline_reports,
        candidate_reports,
        diagnostics,
    )
    save_phase_a_decision(decision, Path(output_root) / "phase_a_decision.json")
    return decision
