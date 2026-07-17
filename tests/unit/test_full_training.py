from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import numpy as np
import torch
import yaml
from PIL import Image

from bts_nvs.training.backend_qualification import write_backend_profile
from bts_nvs.training.checkpoint import save_checkpoint
from bts_nvs.training.full_training import (
    CANONICAL_SCENES,
    TrainedRun,
    build_scene_command,
    inspect_scene_run,
    load_or_create_backend_decision,
    load_or_create_ledger,
    set_scene_status,
    run_full_training,
    validate_trained_run,
    validate_scene_pool,
)
from bts_nvs.training.trainer import compute_config_sha256, compute_manifest_sha256


def _backend_profile(backend: str, precision: str, cuda_ms: float) -> dict:
    return {
        "schema_version": 1,
        "optimizer_backend": backend,
        "precision": precision,
        "steps": 1000,
        "device_name": "NVIDIA L4",
        "device_capability": [8, 9],
        "median_cuda_step_ms": cuda_ms,
        "peak_vram_bytes": 1024,
        "sample_indices": [index % 5 for index in range(1000)],
        "losses": [1.0] * 1000,
        "gaussian_counts": [100] * 1000,
        "density_event_steps": [600, 700, 800, 900, 1000],
        "gradient_audits": [
            {
                "step": step,
                "finite": True,
                "strategy_gradient_unscaled": True,
                "loss_scale": 1.0,
                "projected_grad_max": 0.1,
                "leaf_grad_max": {"means": 0.1},
                "parameter_dtypes": {"means": "torch.float32"},
                "render_dtype": "torch.float32",
                "loss_dtype": "torch.float32",
            }
            for step in (1, 499, 500, 501, 600, 1000)
        ],
    }


def _backend_root(tmp_path: Path) -> Path:
    root = tmp_path / "backend"
    for name, backend, precision, milliseconds in (
        ("reference", "adam", "fp32", 10.0),
        ("fused", "adam-fused", "fp32", 9.8),
        ("amp", "adam-fused", "amp-fp16", 9.7),
    ):
        write_backend_profile(
            root / name / "backend_profile.json",
            _backend_profile(backend, precision, milliseconds),
        )
    return root


def _scene_pool(tmp_path: Path) -> tuple[Path, Path]:
    scenes = tmp_path / "scenes"
    manifests = tmp_path / "manifests"
    for scene_id in CANONICAL_SCENES:
        (scenes / scene_id).mkdir(parents=True)
        manifest_dir = manifests / scene_id
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "arrays.npz").write_bytes(b"npz")
        (manifest_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scene_id": scene_id,
                    "arrays_file": "arrays.npz",
                }
            ),
            encoding="utf-8",
        )
    return scenes, manifests


def test_backend_decision_is_generated_without_rerunning_profiles(tmp_path: Path):
    root = _backend_root(tmp_path)

    decision = load_or_create_backend_decision(root)

    assert decision.optimizer_backend == "adam"
    assert decision.precision == "fp32"
    assert len(decision.report_sha256) == 64
    report = json.loads((root / "backend_qualification.json").read_text())
    assert report["accepted"] is True


def test_backend_decision_rejects_tampered_or_unaccepted_aggregate(tmp_path: Path):
    root = _backend_root(tmp_path)
    decision = load_or_create_backend_decision(root)
    path = root / "backend_qualification.json"
    report = json.loads(path.read_text())
    report["selected_precision"] = "amp-fp16"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match backend profiles"):
        load_or_create_backend_decision(root)

    path.unlink()
    load_or_create_backend_decision(root)
    report = json.loads(path.read_text())
    report["accepted"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match backend profiles"):
        load_or_create_backend_decision(root)


def test_scene_pool_accepts_exact_case_sensitive_18_scene_contract(tmp_path: Path):
    scenes, manifests = _scene_pool(tmp_path)
    assert validate_scene_pool(scenes, manifests) == CANONICAL_SCENES


@pytest.mark.parametrize("problem", ["missing", "extra", "wrong_case", "npz", "identity"])
def test_scene_pool_rejects_incomplete_or_ambiguous_layout(
    tmp_path: Path,
    problem: str,
):
    scenes, manifests = _scene_pool(tmp_path)
    target = CANONICAL_SCENES[0]
    if problem == "missing":
        (scenes / target).rmdir()
    elif problem == "extra":
        (scenes / "EXTRA").mkdir()
    elif problem == "wrong_case":
        (scenes / target).rename(scenes / target.upper())
    elif problem == "npz":
        (manifests / target / "arrays.npz").unlink()
    else:
        path = manifests / target / "manifest.json"
        payload = json.loads(path.read_text())
        payload["scene_id"] = "other"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((ValueError, FileNotFoundError)):
        validate_scene_pool(scenes, manifests)


def _decision() -> object:
    from bts_nvs.training.full_training import BackendDecision

    return BackendDecision("adam", "fp32", "a" * 64)


def _manifest_artifact(tmp_path: Path, scene_id: str = "HCM0181") -> Path:
    root = tmp_path / "manifest"
    root.mkdir(parents=True)
    np.savez(root / "arrays.npz", value=np.array([1.0]))
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scene_id": scene_id,
                "arrays_file": "arrays.npz",
            }
        ),
        encoding="utf-8",
    )
    return path


def _trained_run(tmp_path: Path) -> tuple[Path, Path, dict]:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "train_previews").mkdir()
    manifest = _manifest_artifact(tmp_path)
    config = {
        "scene_id": "HCM0181",
        "resize_factor": 1,
        "max_steps": 30_000,
        "seed": 0,
        "cache_images": True,
        "pinned_transfer": True,
        "optimizer_backend": "adam",
        "precision": "fp32",
        "rolling_checkpoint": True,
        "internal_holdout": False,
    }
    (run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    manifest_hash = compute_manifest_sha256(manifest)
    config_hash = compute_config_sha256(config)
    (run / "manifest_hash.json").write_text(
        json.dumps({"manifest_hash": manifest_hash}), encoding="utf-8"
    )
    save_checkpoint(
        run / "checkpoints" / "recovery.pt",
        step=30_000,
        gaussians_state_dict={},
        optimizers_state_dict={},
        scheduler_state_dict={},
        strategy_state={},
        active_sh_degree=3,
        manifest_hash=manifest_hash,
        config_hash=config_hash,
    )
    (run / "summary.json").write_text(
        json.dumps({"total_steps": 30_000}), encoding="utf-8"
    )
    (run / "convergence.json").write_text(
        json.dumps({"final_render_non_blank": True}), encoding="utf-8"
    )
    with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(1, 30_001):
            handle.write(json.dumps({"step": step, "loss": 1.0 / step}) + "\n")
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(
        run / "train_previews" / "step_000030000.png"
    )
    return run, manifest, config


def test_ledger_is_atomic_deterministic_and_identity_bound(tmp_path: Path):
    root = tmp_path / "full"
    decision = _decision()
    ledger = load_or_create_ledger(root, decision, CANONICAL_SCENES)
    path = root / "ledger.json"
    first = path.read_bytes()
    assert [record["scene_id"] for record in ledger["scenes"]] == list(
        CANONICAL_SCENES
    )
    assert {record["status"] for record in ledger["scenes"]} == {"pending"}
    assert load_or_create_ledger(root, decision, CANONICAL_SCENES) == ledger
    assert path.read_bytes() == first

    set_scene_status(path, ledger, CANONICAL_SCENES[0], "running")
    loaded = json.loads(path.read_text())
    assert loaded["scenes"][0]["status"] == "running"
    assert not path.with_name(".ledger.json.tmp").exists()

    with pytest.raises(ValueError, match="identity"):
        load_or_create_ledger(
            root,
            type(decision)("adam-fused", "fp32", "b" * 64),
            CANONICAL_SCENES,
        )
    with pytest.raises(ValueError, match="status"):
        set_scene_status(path, loaded, CANONICAL_SCENES[0], "complete")


def test_trained_run_validation_and_scene_inspection(tmp_path: Path):
    run, manifest, _ = _trained_run(tmp_path)

    trained = validate_trained_run(run, "HCM0181", manifest, _decision())

    assert trained.completed_step == 30_000
    assert inspect_scene_run(run, "HCM0181", manifest, _decision()) == "trained"
    assert inspect_scene_run(
        tmp_path / "absent", "HCM0181", manifest, _decision()
    ) == "fresh"


@pytest.mark.parametrize(
    "problem",
    ["metrics_count", "metrics_order", "nan", "config", "manifest", "preview", "checkpoint", "blank"],
)
def test_trained_run_rejects_incomplete_or_inconsistent_artifacts(
    tmp_path: Path,
    problem: str,
):
    run, manifest, _ = _trained_run(tmp_path)
    if problem == "metrics_count":
        lines = (run / "metrics.jsonl").read_text().splitlines()
        (run / "metrics.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    elif problem == "metrics_order":
        path = run / "metrics.jsonl"
        lines = path.read_text().splitlines()
        lines[10] = json.dumps({"step": 99, "loss": 0.1})
        path.write_text("\n".join(lines) + "\n")
    elif problem == "nan":
        path = run / "metrics.jsonl"
        lines = path.read_text().splitlines()
        lines[0] = '{"step": 1, "loss": NaN}'
        path.write_text("\n".join(lines) + "\n")
    elif problem == "config":
        path = run / "config.yaml"
        config = yaml.safe_load(path.read_text())
        config["max_steps"] = 29_999
        path.write_text(yaml.safe_dump(config))
    elif problem == "manifest":
        with np.load(manifest.parent / "arrays.npz") as arrays:
            values = {name: arrays[name] for name in arrays.files}
        values["value"] = np.array([2.0])
        np.savez(manifest.parent / "arrays.npz", **values)
    elif problem == "preview":
        (run / "train_previews" / "step_000030000.png").write_bytes(b"bad")
    elif problem == "checkpoint":
        state = torch.load(
            run / "checkpoints" / "recovery.pt", weights_only=False
        )
        state["step"] = 29_999
        torch.save(state, run / "checkpoints" / "recovery.pt")
    else:
        (run / "convergence.json").write_text(
            json.dumps({"final_render_non_blank": False})
        )

    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        validate_trained_run(run, "HCM0181", manifest, _decision())


def test_scene_inspection_allows_valid_recovery_but_rejects_ambiguous_content(
    tmp_path: Path,
):
    run, manifest, _ = _trained_run(tmp_path)
    (run / "summary.json").unlink()
    assert inspect_scene_run(run, "HCM0181", manifest, _decision()) == "resume"

    (run / "checkpoints" / "recovery.pt").unlink()
    with pytest.raises(ValueError, match="non-empty"):
        inspect_scene_run(run, "HCM0181", manifest, _decision())


def test_scene_command_locks_full_training_contract(tmp_path: Path):
    command = build_scene_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        output_root=tmp_path / "output",
        scene_id="HCM0181",
        decision=_decision(),
        python_bin="python3",
        resume_path=tmp_path / "output/scenes/HCM0181/checkpoints/recovery.pt",
    )

    assert command[0] == "python3"
    assert Path(command[1]).parts[-4:] == (
        "src",
        "bts_nvs",
        "training",
        "run_training.py",
    )
    for pair in (
        ("--resize_factor", "1"),
        ("--max_steps", "30000"),
        ("--checkpoint_every", "3000"),
        ("--seed", "0"),
        ("--optimizer_backend", "adam"),
        ("--precision", "fp32"),
    ):
        index = command.index(pair[0])
        assert command[index + 1] == pair[1]
    for flag in ("--cache_images", "--pinned_transfer", "--rolling_checkpoint"):
        assert flag in command
    for forbidden in (
        "--internal_holdout",
        "--qualification_candidate",
        "--backend_qualification",
        "--full_length_qualification",
    ):
        assert forbidden not in command
    assert "--resume" in command


def test_full_training_runs_sorted_scenes_resumes_and_skips_trained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import bts_nvs.training.full_training as module

    scenes, manifests = _scene_pool(tmp_path)
    backend = _backend_root(tmp_path)
    output = tmp_path / "full"
    states = {CANONICAL_SCENES[0]: "trained", CANONICAL_SCENES[1]: "resume"}
    monkeypatch.setattr(
        module,
        "inspect_scene_run",
        lambda run, scene, manifest, decision: states.get(scene, "fresh"),
    )
    monkeypatch.setattr(
        module,
        "validate_trained_run",
        lambda *args: TrainedRun(30_000, "c" * 64, "m" * 64),
    )
    commands: list[list[str]] = []

    def succeed(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    run_full_training(
        repo_root=tmp_path,
        scenes_root=scenes,
        manifests_root=manifests,
        backend_root=backend,
        output_root=output,
        python_bin="python3",
        run_process=succeed,
    )

    assert len(commands) == 17
    assert commands[0][commands[0].index("--scene_dir") + 1] == str(
        scenes / CANONICAL_SCENES[1]
    )
    assert "--resume" in commands[0]
    assert [command[command.index("--scene_dir") + 1] for command in commands] == [
        str(scenes / scene_id) for scene_id in CANONICAL_SCENES[1:]
    ]
    ledger = json.loads((output / "ledger.json").read_text())
    assert {record["status"] for record in ledger["scenes"]} == {"trained"}


def test_full_training_failure_stops_later_scenes_and_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import bts_nvs.training.full_training as module

    scenes, manifests = _scene_pool(tmp_path)
    backend = _backend_root(tmp_path)
    output = tmp_path / "full"
    monkeypatch.setattr(module, "inspect_scene_run", lambda *args: "fresh")
    calls = 0

    def fail_first(command, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=7)

    with pytest.raises(RuntimeError, match=CANONICAL_SCENES[0]):
        run_full_training(
            repo_root=tmp_path,
            scenes_root=scenes,
            manifests_root=manifests,
            backend_root=backend,
            output_root=output,
            python_bin="python3",
            run_process=fail_first,
        )

    assert calls == 1
    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["scenes"][0]["status"] == "failed"
    assert {record["status"] for record in ledger["scenes"][1:]} == {"pending"}


def test_full_training_records_invalid_existing_artifacts_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import bts_nvs.training.full_training as module

    scenes, manifests = _scene_pool(tmp_path)
    backend = _backend_root(tmp_path)
    output = tmp_path / "full"
    monkeypatch.setattr(
        module,
        "inspect_scene_run",
        lambda *args: (_ for _ in ()).throw(ValueError("ambiguous artifacts")),
    )

    with pytest.raises(ValueError, match="ambiguous artifacts"):
        run_full_training(
            repo_root=tmp_path,
            scenes_root=scenes,
            manifests_root=manifests,
            backend_root=backend,
            output_root=output,
            python_bin="python3",
        )

    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["scenes"][0]["status"] == "failed"
    assert ledger["scenes"][0]["error_type"] == "ValueError"
    assert {record["status"] for record in ledger["scenes"][1:]} == {"pending"}


def test_full_training_cli_parses_only_runtime_paths(tmp_path: Path):
    from bts_nvs.training.run_full_training import parse_args

    args = parse_args(
        [
            "--repo_root",
            str(tmp_path / "repo"),
            "--scenes_root",
            str(tmp_path / "scenes"),
            "--manifests_root",
            str(tmp_path / "manifests"),
            "--backend_root",
            str(tmp_path / "backend"),
            "--output_root",
            str(tmp_path / "output"),
            "--python_bin",
            "python3",
        ]
    )

    assert args.repo_root == tmp_path / "repo"
    assert args.scenes_root == tmp_path / "scenes"
    assert args.manifests_root == tmp_path / "manifests"
    assert args.backend_root == tmp_path / "backend"
    assert args.output_root == tmp_path / "output"
    assert args.python_bin == "python3"
