"""
Reproducibility — single source of truth for the global random seed.

Training scripts:
    from src.utils.seed import set_global_seed
    set_global_seed(params.get("meta", {}).get("seed"))

Analysis modules:
    from src.utils.seed import SEED, get_rng
    pca = PCA(random_state=SEED)
    rng = get_rng()

Notebooks (after setting exp_dir):
    from src.utils.seed import load_seed, set_global_seed
    set_global_seed(load_seed(exp_dir))
"""

from pathlib import Path
import random

import numpy as np
import yaml

DEFAULT_SEED: int = 42

# Module-level seed, updated by set_global_seed()
SEED: int = DEFAULT_SEED


def set_global_seed(seed: int | None = None) -> int:
    """Set torch/numpy/stdlib seeds globally. Updates module-level SEED. Returns seed used."""
    import torch

    global SEED
    SEED = seed if seed is not None else DEFAULT_SEED

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    return SEED


def load_seed(exp_dir: Path) -> int:
    """Read seed from an experiment's config.yaml (meta.seed), fallback to DEFAULT_SEED."""
    cfg_path = Path(exp_dir) / "config.yaml"
    with open(cfg_path) as f:
        params = yaml.safe_load(f)
    return params.get("meta", {}).get("seed", DEFAULT_SEED)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a fresh numpy Generator seeded with given seed or module-level SEED."""
    return np.random.default_rng(seed if seed is not None else SEED)
