from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from bts_nvs.training.full_training import run_full_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked Phase 4 full-training cohort sequentially."
    )
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--backend_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python_bin", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_full_training(
        repo_root=args.repo_root,
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        backend_root=args.backend_root,
        output_root=args.output_root,
        python_bin=args.python_bin,
    )


if __name__ == "__main__":
    main()
