from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bts_nvs.training.profiling import (
    compare_input_profiles,
    load_input_profile,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Phase 4.3 input profiles")
    parser.add_argument("--uncached", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = compare_input_profiles(
        load_input_profile(args.uncached),
        load_input_profile(args.cached),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Phase 4.3 profile: accepted={report['accepted']}, "
        f"speedup={report['speedup_fraction']:.3f}"
    )
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
