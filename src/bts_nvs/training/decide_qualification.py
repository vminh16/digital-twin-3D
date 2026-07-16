from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bts_nvs.training.qualification import (
    build_qualification_decision,
    load_qualification_reports,
    save_qualification_decision,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decide the locked Phase 4.4 candidate")
    parser.add_argument("--reports_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    decision = build_qualification_decision(load_qualification_reports(args.reports_root))
    save_qualification_decision(decision, args.output)
    print(f"Selected qualification candidate: {decision['selected_candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
