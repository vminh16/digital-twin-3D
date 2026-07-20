from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from bts_nvs.training.c1_candidates import C1_CANDIDATES
from bts_nvs.training.c1_phase_a import (
    BASELINE_CANDIDATE,
    PHASE_A_SCENES,
    build_phase_a_decision,
    save_phase_a_decision,
)
from bts_nvs.training.full_training import (
    BackendDecision,
    load_or_create_backend_decision,
)
from bts_nvs.training.c1_screening import (
    atomic_json,
    build_screening_command,
    diagnostics_for_run,
    load_completed_run,
    load_report,
)


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
    return build_screening_command(
        repo_root=repo_root,
        scenes_root=scenes_root,
        manifests_root=manifests_root,
        run_dir=Path(output_root) / scene_id / candidate_id,
        scene_id=scene_id,
        candidate_id=candidate_id,
        decision=decision,
        python_bin=python_bin,
    )


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
        load_report(
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
                report = load_completed_run(
                    run_dir,
                    scene_id,
                    candidate_id,
                    backend,
                )
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
                report = load_completed_run(
                    run_dir,
                    scene_id,
                    candidate_id,
                    backend,
                )
            candidate_reports.append(report)

    baseline_reports: list[dict] = []
    diagnostics: list[dict] = []
    for scene_id in PHASE_A_SCENES:
        baseline_run = Path(baseline_root) / scene_id / BASELINE_CANDIDATE
        baseline_reports.append(
            load_report(
                baseline_run / "qualification_report.json",
                scene_id,
                BASELINE_CANDIDATE,
            )
        )
        baseline_diagnostic = diagnostics_for_run(
            scene_id=scene_id,
            candidate_id=BASELINE_CANDIDATE,
            scene_root=Path(scenes_root) / scene_id,
            manifest_root=Path(manifests_root) / scene_id,
            render_dir=baseline_run / "validation_renders",
        )
        atomic_json(
            Path(output_root) / "baseline_diagnostics" / f"{scene_id}.json",
            baseline_diagnostic,
        )
        diagnostics.append(baseline_diagnostic)

        for candidate_id in C1_CANDIDATES:
            run_dir = Path(output_root) / scene_id / candidate_id
            diagnostic = diagnostics_for_run(
                scene_id=scene_id,
                candidate_id=candidate_id,
                scene_root=Path(scenes_root) / scene_id,
                manifest_root=Path(manifests_root) / scene_id,
                render_dir=run_dir / "validation_renders",
            )
            atomic_json(run_dir / "high_frequency.json", diagnostic)
            diagnostics.append(diagnostic)

    decision = build_phase_a_decision(
        baseline_reports,
        candidate_reports,
        diagnostics,
    )
    save_phase_a_decision(decision, Path(output_root) / "phase_a_decision.json")
    return decision
