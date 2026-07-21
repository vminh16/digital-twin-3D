from pathlib import Path

import pytest

import bts_nvs.training.run_c1_confirmation as cli


def test_confirmation_cli_dispatches_all_explicit_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}
    monkeypatch.setattr(cli, "run_phase_c", lambda **kwargs: observed.update(kwargs))

    cli.main(
        [
            "--repo_root",
            str(tmp_path / "repo"),
            "--scenes_root",
            str(tmp_path / "scenes"),
            "--manifests_root",
            str(tmp_path / "manifests"),
            "--backend_root",
            str(tmp_path / "backend"),
            "--baseline_root",
            str(tmp_path / "baseline"),
            "--phase_b_root",
            str(tmp_path / "phase_b"),
            "--output_root",
            str(tmp_path / "phase_c"),
            "--python_bin",
            "python3",
        ]
    )

    assert observed == {
        "repo_root": tmp_path / "repo",
        "scenes_root": tmp_path / "scenes",
        "manifests_root": tmp_path / "manifests",
        "backend_root": tmp_path / "backend",
        "baseline_root": tmp_path / "baseline",
        "phase_b_root": tmp_path / "phase_b",
        "output_root": tmp_path / "phase_c",
        "python_bin": "python3",
    }
