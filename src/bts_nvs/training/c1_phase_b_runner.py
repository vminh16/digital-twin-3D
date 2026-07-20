from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from bts_nvs.training.c1_phase_a import PHASE_A_SCENES
from bts_nvs.training.c1_phase_b import (
    LOCKED_CANDIDATE,
    PHASE_B_SCENES,
    SCREENING_SCENES,
    build_phase_b_decision,
    save_phase_b_decision,
)
from bts_nvs.training.c1_screening import (
    BASELINE_CANDIDATE,
    atomic_json,
    build_screening_command,
    diagnostics_for_run,
    load_completed_run,
    load_report,
)
from bts_nvs.training.full_training import (
    BackendDecision,
    load_or_create_backend_decision,
)


def _load_json(path: Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing JSON artifact: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {source}")
    return payload


def load_phase_a_lock(path: Path) -> dict:
    decision = _load_json(path)
    if decision.get("schema_version") != 1 or decision.get("phase") != "C1-phase-A":
        raise ValueError("Phase A decision identity mismatch")
    if decision.get("phase_a_passed") is not True:
        raise ValueError("Phase A did not pass")
    if decision.get("selected_candidate") != LOCKED_CANDIDATE:
        raise ValueError("Phase A winner mismatch")
    return decision


def build_phase_b_command(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    output_root: Path,
    scene_id: str,
    decision: BackendDecision,
    python_bin: str,
) -> list[str]:
    if scene_id not in PHASE_B_SCENES:
        raise ValueError(f"unexpected Phase B scene: {scene_id}")
    return build_screening_command(
        repo_root=repo_root,
        scenes_root=scenes_root,
        manifests_root=manifests_root,
        run_dir=Path(output_root) / scene_id,
        scene_id=scene_id,
        candidate_id=LOCKED_CANDIDATE,
        decision=decision,
        python_bin=python_bin,
    )


def _require_phase_b_inputs(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    baseline_root: Path,
    phase_a_root: Path,
) -> None:
    required_files = [
        Path(repo_root) / "src" / "bts_nvs" / "training" / "run_training.py",
        Path(phase_a_root) / "phase_a_decision.json",
    ]
    required_directories: list[Path] = []
    for scene_id in PHASE_B_SCENES:
        required_directories.append(Path(scenes_root) / scene_id)
        required_files.extend(
            [
                Path(manifests_root) / scene_id / "manifest.json",
                Path(manifests_root) / scene_id / "holdout.json",
            ]
        )
    for scene_id in SCREENING_SCENES:
        baseline_run = Path(baseline_root) / scene_id / BASELINE_CANDIDATE
        required_files.append(baseline_run / "qualification_report.json")
        required_directories.append(baseline_run / "validation_renders")
    for scene_id in PHASE_A_SCENES:
        candidate_run = Path(phase_a_root) / scene_id / LOCKED_CANDIDATE
        required_files.extend(
            [
                candidate_run / "qualification_report.json",
                candidate_run / "config.yaml",
                candidate_run / "high_frequency.json",
                Path(phase_a_root) / "baseline_diagnostics" / f"{scene_id}.json",
            ]
        )
        required_directories.append(candidate_run / "validation_renders")
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_directories if not path.is_dir())
    if missing:
        details = "\n".join(f"- {path}" for path in sorted(missing))
        raise FileNotFoundError(f"missing Phase B inputs:\n{details}")

    load_phase_a_lock(Path(phase_a_root) / "phase_a_decision.json")
    for scene_id in SCREENING_SCENES:
        load_report(
            Path(baseline_root)
            / scene_id
            / BASELINE_CANDIDATE
            / "qualification_report.json",
            scene_id,
            BASELINE_CANDIDATE,
        )
    for scene_id in PHASE_A_SCENES:
        load_report(
            Path(phase_a_root)
            / scene_id
            / LOCKED_CANDIDATE
            / "qualification_report.json",
            scene_id,
            LOCKED_CANDIDATE,
        )


def run_phase_b(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    backend_root: Path,
    baseline_root: Path,
    phase_a_root: Path,
    output_root: Path,
    python_bin: str,
    run_process: Callable[..., object] = subprocess.run,
) -> dict:
    _require_phase_b_inputs(
        repo_root=repo_root,
        scenes_root=scenes_root,
        manifests_root=manifests_root,
        baseline_root=baseline_root,
        phase_a_root=phase_a_root,
    )
    backend = load_or_create_backend_decision(backend_root)
    candidate_reports = [
        load_completed_run(
            Path(phase_a_root) / scene_id / LOCKED_CANDIDATE,
            scene_id,
            LOCKED_CANDIDATE,
            backend,
        )
        for scene_id in PHASE_A_SCENES
    ]

    for scene_id in PHASE_B_SCENES:
        run_dir = Path(output_root) / scene_id
        report_path = run_dir / "qualification_report.json"
        if report_path.is_file():
            report = load_completed_run(
                run_dir,
                scene_id,
                LOCKED_CANDIDATE,
                backend,
            )
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise ValueError(
                    f"Phase B run directory is non-empty without a report: {run_dir}"
                )
            command = build_phase_b_command(
                repo_root=repo_root,
                scenes_root=scenes_root,
                manifests_root=manifests_root,
                output_root=output_root,
                scene_id=scene_id,
                decision=backend,
                python_bin=python_bin,
            )
            result = run_process(command, check=False)
            returncode = getattr(result, "returncode", None)
            if returncode != 0:
                raise RuntimeError(
                    f"Phase B training exited with code {returncode} for {scene_id}"
                )
            report = load_completed_run(
                run_dir,
                scene_id,
                LOCKED_CANDIDATE,
                backend,
            )
        candidate_reports.append(report)

    baseline_reports = [
        load_report(
            Path(baseline_root)
            / scene_id
            / BASELINE_CANDIDATE
            / "qualification_report.json",
            scene_id,
            BASELINE_CANDIDATE,
        )
        for scene_id in SCREENING_SCENES
    ]
    diagnostics = []
    for scene_id in PHASE_A_SCENES:
        diagnostics.append(
            _load_json(Path(phase_a_root) / "baseline_diagnostics" / f"{scene_id}.json")
        )
        diagnostics.append(
            _load_json(
                Path(phase_a_root)
                / scene_id
                / LOCKED_CANDIDATE
                / "high_frequency.json"
            )
        )
    for scene_id in PHASE_B_SCENES:
        baseline_run = Path(baseline_root) / scene_id / BASELINE_CANDIDATE
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

        run_dir = Path(output_root) / scene_id
        candidate_diagnostic = diagnostics_for_run(
            scene_id=scene_id,
            candidate_id=LOCKED_CANDIDATE,
            scene_root=Path(scenes_root) / scene_id,
            manifest_root=Path(manifests_root) / scene_id,
            render_dir=run_dir / "validation_renders",
        )
        atomic_json(run_dir / "high_frequency.json", candidate_diagnostic)
        diagnostics.append(candidate_diagnostic)

    decision = build_phase_b_decision(
        baseline_reports,
        candidate_reports,
        diagnostics,
    )
    save_phase_b_decision(decision, Path(output_root) / "phase_b_decision.json")
    return decision
