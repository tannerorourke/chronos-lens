from pathlib import Path
import random
import torch

import numpy as np
import yaml

DEFAULT_SEED: int = 42

"""
Training loop, analyses, functions, etc. all use same global.
"""
SEED: int = DEFAULT_SEED


def set_global_seed(seed: int | None = None) -> int:
    """Set torch/numpy/stdlib seeds globally. Updates module-level SEED. Returns seed used."""
    
    global SEED
    SEED = seed if seed is not None else DEFAULT_SEED

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    return SEED


def _restore_rng(checkpoint: dict) -> None:
    """Restore RNG states from a checkpoint dict."""
    if "rng_states" not in checkpoint:
        raise ValueError("Checkpoint has no RNG states. Activating agresssive angry sounds.")
    
    rng = checkpoint["rng_states"]
    torch.random.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["cuda"])
        
    return rng


def load_seed(exp_dir: Path) -> int:
    """Read seed from an experiment's config.yaml (meta.seed), fallback to DEFAULT_SEED."""
    cfg_path = Path(exp_dir) / "config.yaml"
    with open(cfg_path) as f:
        params = yaml.safe_load(f)
    return params.get("meta", {}).get("seed", DEFAULT_SEED)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh numpy Generator seeded with given seed or module-level SEED."""
    return np.random.default_rng(seed if seed is not None else SEED)
