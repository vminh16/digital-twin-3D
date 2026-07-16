from pathlib import Path

import pytest

from bts_nvs.training.resources import (
    CHECKPOINT_BYTES_PER_GAUSSIAN,
    DISK_HEADROOM_BYTES,
    HOST_RAM_HEADROOM_BYTES,
    MAX_PEAK_VRAM_BYTES,
    parse_linux_meminfo,
    require_cache_capacity,
    require_checkpoint_capacity,
    require_no_swap,
    require_peak_vram,
)


def test_linux_meminfo_reports_available_ram_and_used_swap():
    available, swap_used = parse_linux_meminfo(
        "MemAvailable: 6291456 kB\nSwapTotal: 1048576 kB\nSwapFree: 786432 kB\n"
    )

    assert available == 6 * 1024**3
    assert swap_used == 256 * 1024**2


def test_cache_capacity_requires_four_gib_after_allocation():
    require_cache_capacity(1024, available_bytes=HOST_RAM_HEADROOM_BYTES + 1024)
    with pytest.raises(MemoryError, match="host RAM"):
        require_cache_capacity(1025, available_bytes=HOST_RAM_HEADROOM_BYTES + 1024)


def test_active_swap_is_fatal():
    with pytest.raises(RuntimeError, match="swap"):
        require_no_swap(1)
    require_no_swap(0)


def test_checkpoint_capacity_uses_double_estimate_plus_ten_gib():
    count = 100
    required = 2 * count * CHECKPOINT_BYTES_PER_GAUSSIAN + DISK_HEADROOM_BYTES
    require_checkpoint_capacity(count, available_bytes=required)
    with pytest.raises(OSError, match="checkpoint"):
        require_checkpoint_capacity(count, available_bytes=required - 1)


def test_peak_vram_must_remain_below_twenty_gib():
    require_peak_vram(MAX_PEAK_VRAM_BYTES - 1)
    with pytest.raises(RuntimeError, match="VRAM"):
        require_peak_vram(MAX_PEAK_VRAM_BYTES)
