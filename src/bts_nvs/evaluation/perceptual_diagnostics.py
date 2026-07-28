from __future__ import annotations

import json
from pathlib import Path

import torch


@torch.no_grad()
def build_perceptual_diagnostics(trainer, output_dir: str | Path) -> dict[str, object]:
    sensitivity = trainer.gaussians.get_sensitivities().detach()
    opacity = trainer.gaussians.get_opacities().detach()
    low_threshold = float(trainer.config["perceptual_low_threshold"])
    high_threshold = float(trainer.config["perceptual_high_threshold"])
    bins = {
        "low": sensitivity <= low_threshold,
        "medium": (sensitivity > low_threshold)
        & (sensitivity <= high_threshold),
        "high": sensitivity > high_threshold,
    }

    def summarize(mask: torch.Tensor) -> dict[str, float | int]:
        count = int(mask.sum())
        return {
            "count": count,
            "fraction": float(count / max(1, sensitivity.numel())),
            "opacity_mass": float(opacity[mask].sum()) if count else 0.0,
        }

    losses = []
    metrics_path = Path(output_dir) / "metrics.jsonl"
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line:
            value = float(json.loads(line).get("sensitivity_loss", 0.0))
            if value > 0.0:
                losses.append(value)
    state = trainer.strategy_state
    return {
        "schema_version": 1,
        "candidate_id": trainer.config["candidate_id"],
        "sensitivity_manifest_sha256": trainer.config[
            "perceptual_sensitivity_sha256"
        ],
        "scene_mean_sensitivity": float(
            trainer.config["perceptual_scene_sensitivity"]
        ),
        "sensitivity_loss_first": losses[0] if losses else 0.0,
        "sensitivity_loss_final": losses[-1] if losses else 0.0,
        "sensitivity_mean": float(sensitivity.mean()),
        "sensitivity_min": float(sensitivity.min()),
        "sensitivity_max": float(sensitivity.max()),
        "bins": {name: summarize(mask) for name, mask in bins.items()},
        "density": {
            "clone_count": int(state.get("perceptual_clone_count", 0)),
            "split_count": int(state.get("perceptual_split_count", 0)),
            "cap_hit_step": state.get("perceptual_cap_hit_step"),
            "cap_max": int(trainer.config["perceptual_cap_max"]),
        },
    }
