import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def get_rng_states() -> Dict[str, Any]:
    """Captures the current state of all random number generators.

    Returns:
        Dict[str, Any]: Container of RNG states.
    """
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_gpu"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restores the state of all random number generators.

    Args:
        states (Dict[str, Any]): Dict of captured RNG states.
    """
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if "torch_gpu" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_gpu"])


def save_checkpoint(
    path: str | Path,
    step: int,
    gaussians_state_dict: dict,
    optimizers_state_dict: dict,
    scheduler_state_dict: dict,
    strategy_state: dict,
    active_sh_degree: int,
    manifest_hash: str,
    config_hash: str,
    precision_state: dict | None = None,
) -> None:
    """Saves training checkpoint atomically.

    Args:
        path (str | Path): Final target file path (.pt).
        step (int): Current iteration step count.
        gaussians_state_dict (dict): State dictionary of the Gaussian model.
        optimizers_state_dict (dict): Dictionary of optimizer state dicts.
        scheduler_state_dict (dict): State dictionary of the scheduler.
        strategy_state (dict): Internal state of the density strategy.
        active_sh_degree (int): Active degree of Spherical Harmonics.
        manifest_hash (str): SHA-256 hash of the scene manifest.
        config_hash (str): SHA-256 hash of the optimization configuration.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")

    state = {
        "step": step,
        "gaussians": gaussians_state_dict,
        "optimizers": optimizers_state_dict,
        "scheduler": scheduler_state_dict,
        "strategy_state": strategy_state,
        "active_sh_degree": active_sh_degree,
        "rng_states": get_rng_states(),
        "manifest_hash": manifest_hash,
        "config_hash": config_hash,
        "precision_state": precision_state or {},
    }

    torch.save(state, tmp_path)
    # Perform atomic replace
    os.replace(tmp_path, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    expected_manifest_hash: Optional[str] = None,
    expected_config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads a training checkpoint and verifies SHA256 manifest hash integrity.

    Args:
        path (str | Path): Path to checkpoint file.
        expected_manifest_hash (str, optional): Target scene manifest SHA256 hash.
        expected_config_hash (str, optional): Target optimization config SHA256 hash.

    Returns:
        Dict[str, Any]: Deserialized training state.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if expected_manifest_hash is not None:
        loaded_hash = state.get("manifest_hash")
        if loaded_hash != expected_manifest_hash:
            raise ValueError(
                f"Manifest hash mismatch! Checkpoint hash: {loaded_hash}, "
                f"Expected scene hash: {expected_manifest_hash}"
            )
    if expected_config_hash is not None:
        loaded_hash = state.get("config_hash")
        if loaded_hash != expected_config_hash:
            raise ValueError(
                f"Config hash mismatch! Checkpoint hash: {loaded_hash}, "
                f"Expected config hash: {expected_config_hash}"
            )

    return state
