from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bts_nvs.training.c1_phase_a_runner import run_phase_a
from bts_nvs.training.c1_phase_b_runner import run_phase_b


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a C1 screening stage.")
    parser.add_argument("--stage", choices=("phase-a", "phase-b"), required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--backend_root", type=Path, required=True)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--phase_a_root", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args(argv)
    if args.stage == "phase-b" and args.phase_a_root is None:
        parser.error("--phase_a_root is required for phase-b")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    common = {
        "repo_root": args.repo_root,
        "scenes_root": args.scenes_root,
        "manifests_root": args.manifests_root,
        "backend_root": args.backend_root,
        "baseline_root": args.baseline_root,
        "output_root": args.output_root,
        "python_bin": args.python_bin,
    }
    if args.stage == "phase-b":
        run_phase_b(phase_a_root=args.phase_a_root, **common)
    else:
        run_phase_a(**common)


if __name__ == "__main__":
    main()
