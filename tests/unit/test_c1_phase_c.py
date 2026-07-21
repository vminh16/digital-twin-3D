import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bts_nvs.training.c1_phase_c import (
    LEGACY_B0_MANIFEST_SHA256,
    LOCKED_CANDIDATE,
    build_phase_c_command,
    build_phase_c_decision,
    classify_run_directory,
    compute_phase_c_config_sha256,
    compute_render_set_sha256,
    validate_phase_b_authorization,
    validate_phase_c_baseline,
    validate_phase_c_baseline_artifacts,
    validate_phase_c_candidate,
    validate_phase_c_pair,
)
from bts_nvs.training.c1_phase_c_runner import run_phase_c


def _report(
    *, psnr: float, ssim: float, lpips: float, candidate: bool = False
) -> dict:
    report = {
        "schema_version": 1,
        "scene_id": "HCM0181",
        "step": 30_000,
        "max_vram_mb": 12_000.0,
        "peak_gaussians": 6_000_000,
        "final_num_gaussians": 5_000_000,
        "total_time_seconds": 10_000.0,
        "timing_record_count": 30_000,
        "final_validation": {
            "image_count": 2,
            "psnr_db_mean": psnr,
            "ssim_mean": ssim,
            "lpips_mean": lpips,
            "images": {
                "a.JPG": {"psnr_db": psnr, "ssim": ssim, "lpips": lpips},
                "b.JPG": {"psnr_db": psnr, "ssim": ssim, "lpips": lpips},
            },
        },
    }
    if candidate:
        report.update(
            {
                "candidate_id": LOCKED_CANDIDATE,
                "config_sha256": "c" * 64,
                "manifest_sha256": "d" * 64,
                "validation_render_sha256": {
                    "a.JPG": "a" * 64,
                    "b.JPG": "b" * 64,
                },
            }
        )
    return report


def _config(*, candidate: bool) -> dict:
    config = {
        "scene_id": "HCM0181",
        "resize_factor": 1,
        "max_steps": 30_000,
        "seed": 0,
        "cache_images": True,
        "pinned_transfer": True,
        "internal_holdout": True,
        "holdout_manifest_sha256": "h" * 64,
        "rolling_checkpoint": candidate,
        "grow_grad2d": 0.0008 if candidate else 0.0002,
        "absgrad": candidate,
        "revised_opacity": candidate,
    }
    if candidate:
        config["full_length_candidate"] = LOCKED_CANDIDATE
        config["optimizer_backend"] = "adam-fused"
        config["precision"] = "amp-fp16"
    else:
        config["full_length_qualification"] = True
    return config


def _diagnostic(*, missing: float, spurious: float) -> dict:
    return {
        "schema_version": 1,
        "scene_id": "HCM0181",
        "image_count": 2,
        "missing_edge_mean": missing,
        "spurious_edge_mean": spurious,
        "hf_l1_mean": 0.2,
    }


def _historical_baseline_config() -> dict:
    return {
        "cache_images": True,
        "data_range": 1.0,
        "full_length_qualification": True,
        "grow_grad2d": 0.0002,
        "grow_scale3d": 0.01,
        "guard_count": 46,
        "holdout_algorithm": "pose_fps_guard2_v1",
        "holdout_manifest_sha256": (
            "0d39a9f2ec144f06640470cc3fca10cf355b37c9c04c619ec756991d2d36aec5"
        ),
        "internal_holdout": True,
        "internal_train_count": 169,
        "lambda_dssim": 0.2,
        "max_steps": 30_000,
        "pinned_transfer": True,
        "profile_input": False,
        "prune_opa": 0.005,
        "refine_every": 100,
        "refine_start_step": 500,
        "refine_stop_step": 15_000,
        "reset_every": 3_000,
        "resize_factor": 1,
        "resize_height": 989,
        "resize_width": 1320,
        "scene_id": "HCM0181",
        "seed": 0,
        "undistort": True,
        "validation_count": 25,
    }


def _historical_baseline_report() -> dict:
    baseline = _report(
        psnr=22.691555943105403,
        ssim=0.8053435290993863,
        lpips=0.11125879257917404,
    )
    baseline.update(
        {
            "git_commit": "aa55a21704f6d40375662d669dbf52ec1684a528",
            "peak_gaussians": 6_861_805,
            "max_vram_mb": 11_789.9814453125,
        }
    )
    baseline["final_validation"]["image_count"] = 25
    baseline["final_validation"]["images"] = {
        f"frame_{index:02d}.JPG": {
            "psnr_db": 22.691555943105403,
            "ssim": 0.8053435290993863,
            "lpips": 0.11125879257917404,
        }
        for index in range(25)
    }
    return baseline


def test_phase_c_pins_historical_b0_and_legacy_adam_fp32() -> None:
    baseline = _historical_baseline_report()

    validate_phase_c_baseline(
        baseline,
        _historical_baseline_config(),
        SimpleNamespace(optimizer_backend="adam", precision="fp32"),
    )

    with pytest.raises(ValueError, match="legacy Adam/FP32"):
        validate_phase_c_baseline(
            baseline,
            _historical_baseline_config(),
            SimpleNamespace(optimizer_backend="adam-fused", precision="amp-fp16"),
        )


def test_phase_c_render_set_hash_is_canonical_and_baseline_is_pinned() -> None:
    hashes = {"b.png": "b" * 64, "a.png": "a" * 64}
    assert compute_render_set_sha256(hashes) == compute_render_set_sha256(
        {"a.png": "a" * 64, "b.png": "b" * 64}
    )

    with pytest.raises(ValueError, match="manifest"):
        validate_phase_c_baseline_artifacts("0" * 64, hashes)


def test_phase_c_accepts_the_real_historical_b0_render_authority() -> None:
    # SHA-256 values recomputed from the 25 validation PNG blobs at Git 411c8de.
    hashes = dict(
        line.split(" ", 1)
        for line in """
DJI_20241229103156_0001_V.JPG c3178c75397d998d08797785e4b1bbb4c878e86c6ad427426462e50aa5cb3d9b
DJI_20241229103333_0019_V.JPG 66fabeecaf9aab80460066c7a38f65c4b3cf78b3664bca2dcd54a05d032b85be
DJI_20241229103347_0033_V.JPG 01a60dfad1822282ad792aa17ef9430919f1aeba2d2f88f6077bc22912fd5d3f
DJI_20241229103413_0055_V.JPG 8d27016b85b7ee0b67ccb92b6fc368113bab67c6253693ae33bfd9fad5bfd535
DJI_20241229103441_0073_V.JPG c34d36d53baf7de5823074ca04968bc6147f11f8f7f36f5fd18952bb1dfc29f0
DJI_20241229103452_0084_V.JPG dc9e4f2f755fb9fc34538a49484ee3ad29305ad5897091ed9b449cc2befed43b
DJI_20241229103457_0089_V.JPG 3104110a1b0fb2365d9ce163af9de89376c0d7f7c9ce3202345a3aef8abfe7ea
DJI_20241229103502_0094_V.JPG ece1a7f5fb6c08ad8ee67d6fab955ec99feb483690a912a59a52d7c08353ac92
DJI_20241229103507_0099_V.JPG 81d712bfddcd1944ea21617996d2edbba26cdcc8ba355829de2cdcf15737c7c3
DJI_20241229103807_0187_V.JPG e8841972f223840072845bc3c99a7dc9042d134168a8020e199adfcc9ab40484
DJI_20241229103814_0194_V.JPG 168d38dbee736f2f633a6fa8a3f6a62ede44507419d72e731478ae80bca61a65
DJI_20241229103817_0197_V.JPG 6a49b6a43c58cf9d16bf77a4aa051fd1568ad10ef7ec349c18f22a5117f2b747
DJI_20241229103819_0199_V.JPG bda365ba608dd2b97812c86e27f7360fed291cdfc24d3b9e2aeede34da1d1683
DJI_20241229103823_0203_V.JPG 1676a47ac2dd0f6d10ad182a27eddbcfba322a4395936148855ce7bb099c0bbb
DJI_20241229103826_0206_V.JPG cab5811f8e2b07036fb808a911fb420e7ee87a987719b90a34f3dbaca28f2c71
DJI_20241229103829_0209_V.JPG 2a9a109c34d7b2d624feeacbc02045756bc1cfc72999d57182c89f55ae11d288
DJI_20241229103851_0211_V.JPG e41aa9c3829de156d489b0f75c05fb13ecc56e98d13ac1ac023e80f0dce00a0b
DJI_20241229103903_0223_V.JPG 2b6338bced28117efdd68c3ee73c58ce1aba7d8dc7a7ca4fd70efc50acb714b3
DJI_20241229103906_0226_V.JPG d3f4faa6546e2838df40f0a92f3fa018cac89b676f8223ab2a645ca68965d17c
DJI_20241229103911_0231_V.JPG 1a4ef1d237b3e23fc804b76497a25be6de1228a701c97fae96d10ba89158caba
DJI_20241229103916_0236_V.JPG ec4d0ead1d8f7f4fee2f4efd7f90f8eebcfe6c2e6aa1c8faf79d2350793b4230
DJI_20241229104005_0248_V.JPG 352ad08f1884712eefa59ae392b91922491d0f7d8caafcbe0dfcd52a77f0eb36
DJI_20241229104012_0255_V.JPG 7cdeee48cd7635e643a86a4c929fe724e4993c6022b6a4d08feee73b11c5892f
DJI_20241229104056_0285_V.JPG 4291fb7efb699277df541960fabcdd0425bcc7e2e0258cd39b63061a06c6594c
DJI_20241229104114_0297_V.JPG 97469fca44652dddd46c9ba50cd0b2e736a6561f15b37924f1cd4e816212c521
""".strip().splitlines()
    )

    validate_phase_c_baseline_artifacts(LEGACY_B0_MANIFEST_SHA256, hashes)


def test_phase_c_command_is_locked_and_uses_one_rolling_recovery(
    tmp_path: Path,
) -> None:
    backend = SimpleNamespace(optimizer_backend="adam-fused", precision="amp-fp16")
    command = build_phase_c_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        output_root=tmp_path / "phase_c",
        decision=backend,
        python_bin="python",
    )

    assert command[-6:] == [
        "--optimizer_backend",
        "adam-fused",
        "--precision",
        "amp-fp16",
        "--full_length_candidate",
        LOCKED_CANDIDATE,
    ]
    assert command[command.index("--max_steps") + 1] == "30000"
    assert command[command.index("--checkpoint_every") + 1] == "3000"
    assert "--cache_images" in command and "--pinned_transfer" in command
    assert "--resume" not in command

    recovery = tmp_path / "phase_c" / "HCM0181" / "checkpoints" / "recovery.pt"
    resumed = build_phase_c_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        output_root=tmp_path / "phase_c",
        decision=backend,
        python_bin="python",
        resume_path=recovery,
    )
    assert resumed[-2:] == ["--resume", str(recovery)]


def test_phase_c_pair_requires_same_holdout_and_images() -> None:
    baseline = _report(psnr=22.69, ssim=0.805, lpips=0.111)
    candidate = _report(psnr=23.0, ssim=0.81, lpips=0.10, candidate=True)

    validate_phase_c_pair(
        baseline,
        candidate,
        _config(candidate=False),
        _config(candidate=True),
    )

    mismatched = _config(candidate=True) | {"holdout_manifest_sha256": "x" * 64}
    with pytest.raises(ValueError, match="holdout"):
        validate_phase_c_pair(baseline, candidate, _config(candidate=False), mismatched)


def test_phase_c_candidate_binds_config_manifest_report_and_renders() -> None:
    config = _config(candidate=True)
    report = _report(psnr=23.0, ssim=0.81, lpips=0.10, candidate=True)
    report["config_sha256"] = compute_phase_c_config_sha256(config)
    report["manifest_sha256"] = LEGACY_B0_MANIFEST_SHA256
    render_hashes = {"a.JPG": "a" * 64, "b.JPG": "b" * 64}

    validate_phase_c_candidate(
        report, config, LEGACY_B0_MANIFEST_SHA256, render_hashes
    )

    with pytest.raises(ValueError, match="render hashes"):
        validate_phase_c_candidate(
            report,
            config,
            LEGACY_B0_MANIFEST_SHA256,
            render_hashes | {"b.JPG": "f" * 64},
        )


def test_phase_c_decision_passes_only_when_every_preregistered_gate_passes() -> None:
    baseline = _report(psnr=22.69, ssim=0.805, lpips=0.111)
    candidate = _report(psnr=23.0, ssim=0.81, lpips=0.10)
    baseline_diag = _diagnostic(missing=0.30, spurious=0.03)
    candidate_diag = _diagnostic(missing=0.20, spurious=0.04)

    decision = build_phase_c_decision(
        baseline,
        candidate,
        baseline_diag,
        candidate_diag,
        integrity_passed=True,
        provenance={"baseline_root": "baseline", "candidate_root": "candidate"},
    )

    assert decision["phase_c_passed"] is True
    assert decision["selected_candidate"] == LOCKED_CANDIDATE
    assert decision["delta_score50"] > 0
    assert all(decision["gates"].values())
    assert decision["failed_gates"] == []
    assert decision["provenance"]["baseline_root"] == "baseline"


@pytest.mark.parametrize(
    "candidate,candidate_diag,integrity_passed,failed_gate",
    (
        (
            _report(psnr=22.0, ssim=0.80, lpips=0.12),
            _diagnostic(missing=0.2, spurious=0.02),
            True,
            "score_improved",
        ),
        (
            _report(psnr=24.0, ssim=0.84, lpips=0.12),
            _diagnostic(missing=0.2, spurious=0.02),
            True,
            "lpips_not_worse",
        ),
        (
            _report(psnr=24.0, ssim=0.84, lpips=0.10),
            _diagnostic(missing=0.4, spurious=0.04),
            True,
            "edge_errors_not_both_worse",
        ),
        (
            _report(psnr=24.0, ssim=0.84, lpips=0.10),
            _diagnostic(missing=0.2, spurious=0.02),
            False,
            "candidate_integrity_passed",
        ),
    ),
)
def test_phase_c_decision_rejects_any_failed_gate(
    candidate: dict,
    candidate_diag: dict,
    integrity_passed: bool,
    failed_gate: str,
) -> None:
    decision = build_phase_c_decision(
        _report(psnr=22.69, ssim=0.805, lpips=0.111),
        candidate,
        _diagnostic(missing=0.30, spurious=0.03),
        candidate_diag,
        integrity_passed=integrity_passed,
        provenance={},
    )

    assert decision["phase_c_passed"] is False
    assert decision["selected_candidate"] is None
    assert decision["gates"][failed_gate] is False
    assert failed_gate in decision["failed_gates"]


def test_phase_c_uses_preregistered_23gb_vram_gate() -> None:
    candidate = _report(psnr=24.0, ssim=0.84, lpips=0.10)
    candidate["max_vram_mb"] = 22 * 1024

    decision = build_phase_c_decision(
        _report(psnr=22.69, ssim=0.805, lpips=0.111),
        candidate,
        _diagnostic(missing=0.30, spurious=0.03),
        _diagnostic(missing=0.20, spurious=0.04),
        integrity_passed=True,
        provenance={},
    )

    assert decision["phase_c_passed"] is True
    assert decision["gates"]["peak_vram_below_23gb"] is True


def test_phase_c_run_directory_is_fresh_resumable_or_complete(tmp_path: Path) -> None:
    run = tmp_path / "HCM0181"
    assert classify_run_directory(run) == "fresh"

    recovery = run / "checkpoints" / "recovery.pt"
    recovery.parent.mkdir(parents=True)
    recovery.write_bytes(b"recovery")
    assert classify_run_directory(run) == "resume"

    (run / "full_length_report.json").write_text("{}", encoding="utf-8")
    assert classify_run_directory(run) == "complete"

    recovery.unlink()
    with pytest.raises(ValueError, match="recovery"):
        classify_run_directory(run)

    (run / "full_length_report.json").unlink()
    (run / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        classify_run_directory(run)


def test_phase_c_requires_the_passing_locked_phase_b_winner() -> None:
    validate_phase_b_authorization(
        {"phase_b_passed": True, "selected_candidate": LOCKED_CANDIDATE}
    )

    with pytest.raises(ValueError, match="did not authorize"):
        validate_phase_b_authorization(
            {"phase_b_passed": False, "selected_candidate": LOCKED_CANDIDATE}
        )


def test_phase_c_preflights_current_manifest_before_expensive_training(
    tmp_path: Path,
) -> None:
    phase_b_root = tmp_path / "phase_b"
    phase_b_root.mkdir()
    (phase_b_root / "phase_b_decision.json").write_text(
        json.dumps(
            {"phase_b_passed": True, "selected_candidate": LOCKED_CANDIDATE}
        ),
        encoding="utf-8",
    )
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    (baseline_root / "full_length_report.json").write_text(
        json.dumps(_historical_baseline_report()), encoding="utf-8"
    )
    (baseline_root / "config.yaml").write_text(
        yaml.safe_dump(_historical_baseline_config()), encoding="utf-8"
    )
    (baseline_root / "manifest_hash.json").write_text(
        json.dumps({"manifest_hash": LEGACY_B0_MANIFEST_SHA256}), encoding="utf-8"
    )
    scene_root = tmp_path / "scenes" / "HCM0181"
    scene_root.mkdir(parents=True)
    manifest_root = tmp_path / "manifests" / "HCM0181"
    manifest_root.mkdir(parents=True)
    (manifest_root / "manifest.json").write_text("{}", encoding="utf-8")
    (manifest_root / "holdout.json").write_text("{}", encoding="utf-8")
    process_called = False

    def run_process(*args, **kwargs):
        nonlocal process_called
        process_called = True
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="input manifest"):
        run_phase_c(
            repo_root=tmp_path,
            scenes_root=tmp_path / "scenes",
            manifests_root=tmp_path / "manifests",
            backend_root=tmp_path / "backend",
            baseline_root=baseline_root,
            phase_b_root=phase_b_root,
            output_root=tmp_path / "phase_c",
            python_bin="python",
            run_process=run_process,
            load_backend_fn=lambda _: SimpleNamespace(
                optimizer_backend="adam",
                precision="fp32",
                report_sha256="q" * 64,
            ),
            compute_manifest_hash_fn=lambda _: "0" * 64,
        )

    assert process_called is False
    ledger = json.loads(
        (tmp_path / "phase_c" / "phase_c_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["status"] == "failed"
    assert ledger["error_type"] == "ValueError"


def test_phase_c_validates_and_reuses_a_bound_completed_run(tmp_path: Path) -> None:
    phase_b_root = tmp_path / "phase_b"
    phase_b_root.mkdir()
    (phase_b_root / "phase_b_decision.json").write_text(
        json.dumps(
            {"phase_b_passed": True, "selected_candidate": LOCKED_CANDIDATE}
        ),
        encoding="utf-8",
    )
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline_report = _historical_baseline_report()
    (baseline_root / "full_length_report.json").write_text(
        json.dumps(baseline_report), encoding="utf-8"
    )
    (baseline_root / "config.yaml").write_text(
        yaml.safe_dump(_historical_baseline_config()), encoding="utf-8"
    )
    (baseline_root / "manifest_hash.json").write_text(
        json.dumps({"manifest_hash": LEGACY_B0_MANIFEST_SHA256}), encoding="utf-8"
    )
    scene_root = tmp_path / "scenes" / "HCM0181"
    scene_root.mkdir(parents=True)
    manifest_root = tmp_path / "manifests" / "HCM0181"
    manifest_root.mkdir(parents=True)
    (manifest_root / "manifest.json").write_text("{}", encoding="utf-8")
    (manifest_root / "holdout.json").write_text("{}", encoding="utf-8")

    output_root = tmp_path / "phase_c"
    run_dir = output_root / "HCM0181"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "recovery.pt").write_bytes(b"checkpoint")
    candidate_config = _config(candidate=True) | {
        "holdout_manifest_sha256": _historical_baseline_config()[
            "holdout_manifest_sha256"
        ],
        "optimizer_backend": "adam",
        "precision": "fp32",
    }
    candidate_report = _report(
        psnr=23.1,
        ssim=0.82,
        lpips=0.10,
        candidate=True,
    )
    candidate_report["final_validation"]["image_count"] = 25
    candidate_report["final_validation"]["images"] = {
        name: {"psnr_db": 23.1, "ssim": 0.82, "lpips": 0.10}
        for name in baseline_report["final_validation"]["images"]
    }
    render_hashes = {
        name: f"{index % 16:x}" * 64
        for index, name in enumerate(candidate_report["final_validation"]["images"])
    }
    candidate_report.update(
        {
            "config_sha256": compute_phase_c_config_sha256(candidate_config),
            "manifest_sha256": LEGACY_B0_MANIFEST_SHA256,
            "validation_render_sha256": render_hashes,
        }
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(candidate_config), encoding="utf-8"
    )
    (run_dir / "full_length_report.json").write_text(
        json.dumps(candidate_report), encoding="utf-8"
    )
    (run_dir / "manifest_hash.json").write_text(
        json.dumps({"manifest_hash": LEGACY_B0_MANIFEST_SHA256}), encoding="utf-8"
    )
    recovery_checked = False

    def validate_recovery(*args):
        nonlocal recovery_checked
        recovery_checked = True

    def diagnostics(*, candidate_id: str, **kwargs) -> dict:
        if candidate_id == "B0-reference":
            return _diagnostic(missing=0.30, spurious=0.03)
        return _diagnostic(missing=0.20, spurious=0.04)

    def hash_renders(path: Path, image_names: tuple[str, ...]) -> dict[str, str]:
        if Path(path) == run_dir / "validation_renders":
            return {name: render_hashes[name] for name in image_names}
        return {name: "f" * 64 for name in image_names}

    def reject_training(*args, **kwargs):
        raise AssertionError("completed run must not retrain")

    decision = run_phase_c(
        repo_root=tmp_path,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        backend_root=tmp_path / "backend",
        baseline_root=baseline_root,
        phase_b_root=phase_b_root,
        output_root=output_root,
        python_bin="python",
        run_process=reject_training,
        load_backend_fn=lambda _: SimpleNamespace(
            optimizer_backend="adam",
            precision="fp32",
            report_sha256="a" * 64,
        ),
        hash_renders_fn=hash_renders,
        diagnostics_fn=diagnostics,
        validate_recovery_fn=validate_recovery,
        compute_manifest_hash_fn=lambda _: LEGACY_B0_MANIFEST_SHA256,
        validate_baseline_artifacts_fn=lambda *_: None,
    )

    assert recovery_checked is True
    assert decision["phase_c_passed"] is True
    assert (
        decision["provenance"]["candidate_manifest_sha256"]
        == LEGACY_B0_MANIFEST_SHA256
    )
    assert (
        decision["provenance"]["baseline_manifest_sha256"]
        == LEGACY_B0_MANIFEST_SHA256
    )
    ledger = json.loads((output_root / "phase_c_ledger.json").read_text())
    assert ledger["status"] == "passed"


@pytest.mark.parametrize("initial_state", ("fresh", "resume"))
def test_phase_c_executes_and_validates_fresh_and_resume_paths(
    tmp_path: Path,
    initial_state: str,
) -> None:
    phase_b_root = tmp_path / "phase_b"
    phase_b_root.mkdir()
    (phase_b_root / "phase_b_decision.json").write_text(
        json.dumps(
            {"phase_b_passed": True, "selected_candidate": LOCKED_CANDIDATE}
        ),
        encoding="utf-8",
    )
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline_report = _historical_baseline_report()
    (baseline_root / "full_length_report.json").write_text(
        json.dumps(baseline_report), encoding="utf-8"
    )
    (baseline_root / "config.yaml").write_text(
        yaml.safe_dump(_historical_baseline_config()), encoding="utf-8"
    )
    (baseline_root / "manifest_hash.json").write_text(
        json.dumps({"manifest_hash": LEGACY_B0_MANIFEST_SHA256}), encoding="utf-8"
    )
    (tmp_path / "scenes" / "HCM0181").mkdir(parents=True)
    manifest_root = tmp_path / "manifests" / "HCM0181"
    manifest_root.mkdir(parents=True)
    (manifest_root / "manifest.json").write_text("{}", encoding="utf-8")
    (manifest_root / "holdout.json").write_text("{}", encoding="utf-8")

    output_root = tmp_path / "phase_c"
    run_dir = output_root / "HCM0181"
    recovery = run_dir / "checkpoints" / "recovery.pt"
    if initial_state == "resume":
        recovery.parent.mkdir(parents=True)
        recovery.write_bytes(b"partial checkpoint")

    candidate_config = _config(candidate=True) | {
        "holdout_manifest_sha256": _historical_baseline_config()[
            "holdout_manifest_sha256"
        ],
        "optimizer_backend": "adam",
        "precision": "fp32",
    }
    candidate_report = _report(
        psnr=23.1,
        ssim=0.82,
        lpips=0.10,
        candidate=True,
    )
    candidate_report["final_validation"]["image_count"] = 25
    candidate_report["final_validation"]["images"] = {
        name: {"psnr_db": 23.1, "ssim": 0.82, "lpips": 0.10}
        for name in baseline_report["final_validation"]["images"]
    }
    candidate_hashes = {
        name: f"{index % 16:x}" * 64
        for index, name in enumerate(candidate_report["final_validation"]["images"])
    }
    candidate_report.update(
        {
            "config_sha256": compute_phase_c_config_sha256(candidate_config),
            "manifest_sha256": LEGACY_B0_MANIFEST_SHA256,
            "validation_render_sha256": candidate_hashes,
        }
    )
    observed_command: list[str] = []

    def run_process(command: list[str], *, check: bool) -> SimpleNamespace:
        observed_command.extend(command)
        assert check is False
        recovery.parent.mkdir(parents=True, exist_ok=True)
        recovery.write_bytes(b"complete checkpoint")
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(candidate_config), encoding="utf-8"
        )
        (run_dir / "full_length_report.json").write_text(
            json.dumps(candidate_report), encoding="utf-8"
        )
        (run_dir / "manifest_hash.json").write_text(
            json.dumps({"manifest_hash": LEGACY_B0_MANIFEST_SHA256}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    recovery_calls: list[tuple] = []

    def hash_renders(path: Path, image_names: tuple[str, ...]) -> dict[str, str]:
        if Path(path) == run_dir / "validation_renders":
            return {name: candidate_hashes[name] for name in image_names}
        return {name: "f" * 64 for name in image_names}

    decision = run_phase_c(
        repo_root=tmp_path,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        backend_root=tmp_path / "backend",
        baseline_root=baseline_root,
        phase_b_root=phase_b_root,
        output_root=output_root,
        python_bin="python",
        run_process=run_process,
        load_backend_fn=lambda _: SimpleNamespace(
            optimizer_backend="adam",
            precision="fp32",
            report_sha256="a" * 64,
        ),
        hash_renders_fn=hash_renders,
        diagnostics_fn=lambda **kwargs: _diagnostic(
            missing=0.30 if kwargs["candidate_id"] == "B0-reference" else 0.20,
            spurious=0.03 if kwargs["candidate_id"] == "B0-reference" else 0.04,
        ),
        validate_recovery_fn=lambda *args: recovery_calls.append(args),
        compute_manifest_hash_fn=lambda _: LEGACY_B0_MANIFEST_SHA256,
        validate_baseline_artifacts_fn=lambda *_: None,
    )

    assert observed_command[observed_command.index("--max_steps") + 1] == "30000"
    assert ("--resume" in observed_command) is (initial_state == "resume")
    if initial_state == "resume":
        assert observed_command[-2:] == ["--resume", str(recovery)]
    assert recovery_calls == [
        (
            recovery,
            LEGACY_B0_MANIFEST_SHA256,
            compute_phase_c_config_sha256(candidate_config),
            30_000,
        )
    ]
    assert decision["phase_c_passed"] is True
    ledger = json.loads((output_root / "phase_c_ledger.json").read_text())
    assert ledger["status"] == "passed"
