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


def load_exp_seed(exp_dir: Path) -> int:
    """Read meta.seed from an experiment's config. Accepts either flat input config 
    file (experiments/<run-id>.yaml) or a run dir holding a frozen config.yaml."""
    exp_dir = Path(exp_dir)
    cfg_path = exp_dir if exp_dir.suffix == ".yaml" else exp_dir / "config.yaml"
    with open(cfg_path) as f:
        params = yaml.safe_load(f)
    seed = params.get("meta", {}).get("seed", -1)
    if seed < 0:
        raise ValueError(f"No seed found in {cfg_path}")
    return seed


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh numpy Generator seeded with given seed or module-level SEED."""
    return np.random.default_rng(seed if seed is not None else SEED)
