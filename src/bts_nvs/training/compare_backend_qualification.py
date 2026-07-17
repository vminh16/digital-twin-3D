from __future__ import annotations

import argparse
import json
from pathlib import Path

from bts_nvs.training.backend_qualification import (
    compare_backend_profiles,
    load_backend_profile,
    write_backend_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 4.6 training backends")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--amp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = compare_backend_profiles(
        load_backend_profile(args.reference),
        load_backend_profile(args.fused),
        load_backend_profile(args.amp),
    )
    write_backend_comparison(args.output, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

