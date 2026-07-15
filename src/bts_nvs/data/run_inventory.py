from __future__ import annotations

import argparse
import ctypes
import shutil
import sys
from pathlib import Path

# Match the repository's training entry point: scripts run from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bts_nvs.data.inventory import (
    EXPECTED_SCENE_COUNT,
    Phase4InventoryReport,
    audit_phase4_inventory,
    save_phase4_inventory_report,
)


def available_host_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("cannot query available host RAM")
    return int(status.available_physical)


def check_local_feasibility(
    report: Phase4InventoryReport,
    *,
    available_host_ram_bytes: int,
    available_artifact_disk_bytes: int,
) -> tuple[str, ...]:
    errors = []
    if available_host_ram_bytes < report.required_host_ram_bytes:
        errors.append("insufficient_host_ram")
    if available_artifact_disk_bytes < report.required_artifact_disk_bytes:
        errors.append("insufficient_artifact_disk")
    return tuple(errors)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Phase 4 scene readiness")
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_scenes", type=int, default=EXPECTED_SCENE_COUNT)
    parser.add_argument("--require_ready", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_phase4_inventory(
        args.scenes_root,
        args.manifests_root,
        expected_scene_count=args.expected_scenes,
    )
    save_phase4_inventory_report(report, args.output)
    capacity_errors = check_local_feasibility(
        report,
        available_host_ram_bytes=available_host_ram_bytes(),
        available_artifact_disk_bytes=shutil.disk_usage(args.output.parent).free,
    )
    print(
        f"Phase 4 inventory: status={report.status}, "
        f"valid_scenes={len(report.scenes)}, issues={len(report.issues)}"
    )
    if capacity_errors:
        print("Capacity check failed: " + ", ".join(capacity_errors), file=sys.stderr)
        return 3
    if args.require_ready and report.status != "ready":
        print("Cohort is not ready", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
