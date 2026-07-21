from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bts_nvs.training.c1_phase_c_runner import run_phase_c


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the C1 30k confirmation.")
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--backend_root", type=Path, required=True)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--phase_b_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python_bin", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_phase_c(
        repo_root=args.repo_root,
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        backend_root=args.backend_root,
        baseline_root=args.baseline_root,
        phase_b_root=args.phase_b_root,
        output_root=args.output_root,
        python_bin=args.python_bin,
    )


if __name__ == "__main__":
    main()
