from __future__ import annotations

from pathlib import Path

import pytest

import bts_nvs.training.run_c1_screening as cli


def _common_args(tmp_path: Path) -> list[str]:
    return [
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
        "--output_root",
        str(tmp_path / "output"),
    ]


def test_phase_b_requires_phase_a_root(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "phase-b", *_common_args(tmp_path)])
    assert "--phase_a_root is required for phase-b" in capsys.readouterr().err


def test_main_dispatches_phase_a_without_phase_a_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}
    monkeypatch.setattr(cli, "run_phase_a", lambda **kwargs: observed.update(kwargs))

    cli.main(["--stage", "phase-a", *_common_args(tmp_path)])

    assert observed["output_root"] == tmp_path / "output"
    assert "phase_a_root" not in observed


def test_main_dispatches_phase_b_with_phase_a_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}
    monkeypatch.setattr(cli, "run_phase_b", lambda **kwargs: observed.update(kwargs))

    cli.main(
        [
            "--stage",
            "phase-b",
            "--phase_a_root",
            str(tmp_path / "phase_a"),
            *_common_args(tmp_path),
        ]
    )

    assert observed["phase_a_root"] == tmp_path / "phase_a"
    assert observed["output_root"] == tmp_path / "output"
