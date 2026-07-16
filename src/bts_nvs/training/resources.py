from __future__ import annotations

import shutil
from pathlib import Path


HOST_RAM_HEADROOM_BYTES = 4 * 1024**3
DISK_HEADROOM_BYTES = 10 * 1024**3
MAX_PEAK_VRAM_BYTES = 20 * 1024**3
CHECKPOINT_BYTES_PER_GAUSSIAN = 768


def parse_linux_meminfo(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            values[parts[0][:-1]] = int(parts[1]) * 1024
    try:
        available = values["MemAvailable"]
        swap_used = values["SwapTotal"] - values["SwapFree"]
    except KeyError as error:
        raise ValueError("meminfo is missing RAM or swap fields") from error
    if available < 0 or swap_used < 0:
        raise ValueError("meminfo contains invalid values")
    return available, swap_used


def linux_memory_status() -> tuple[int, int] | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    return parse_linux_meminfo(path.read_text(encoding="ascii"))


def require_cache_capacity(cache_bytes: int, *, available_bytes: int) -> None:
    if cache_bytes < 0 or available_bytes < cache_bytes + HOST_RAM_HEADROOM_BYTES:
        raise MemoryError("image cache would leave less than 4 GiB free host RAM")


def require_no_swap(swap_used_bytes: int) -> None:
    if swap_used_bytes > 0:
        raise RuntimeError("active host swap is not allowed for Phase 4 training")


def checkpoint_required_bytes(gaussian_count: int) -> int:
    if gaussian_count <= 0:
        raise ValueError("gaussian_count must be positive")
    estimate = gaussian_count * CHECKPOINT_BYTES_PER_GAUSSIAN
    return 2 * estimate + DISK_HEADROOM_BYTES


def require_checkpoint_capacity(
    gaussian_count: int,
    *,
    available_bytes: int,
) -> None:
    if available_bytes < checkpoint_required_bytes(gaussian_count):
        raise OSError("insufficient disk space for atomic checkpoint and headroom")


def require_checkpoint_path_capacity(path: Path, gaussian_count: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    require_checkpoint_capacity(
        gaussian_count,
        available_bytes=shutil.disk_usage(target.parent).free,
    )


def require_peak_vram(peak_allocated_bytes: int) -> None:
    if peak_allocated_bytes >= MAX_PEAK_VRAM_BYTES:
        raise RuntimeError("peak allocated VRAM must remain below 20 GiB")
