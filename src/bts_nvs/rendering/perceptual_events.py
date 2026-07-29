from __future__ import annotations


def is_perceptual_density_event(
    step: int,
    *,
    high_interval: int,
    medium_interval: int,
) -> bool:
    return step % high_interval == 0 or step % medium_interval == 0
