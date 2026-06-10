import os
from pathlib import Path
import random
import torch

import numpy as np
import yaml

# optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv() 
except:
    pass

# --- Env vars ---
AWS_S3_BUCKET: str = os.environ.get("AWS_S3_BUCKET", "chronos-ml")
AWS_REGION: str = os.environ.get("AWS_REGION", "")
MIMIC_BQ_PID: str = os.environ.get("CHRONOS_BQ_PID", "")
MIMIC_BQ_DATASET: str = os.environ.get("MIMIC_BQ_DATASET", "physionet-data.mimiciv_3_1_hosp")

# --- App globals ---
GCP_AUTH_URL = "https://www.googleapis.com/auth/cloud-platform"
# ----------------------------------------------------------------------------


def set_cuda_precision(use_bf16: bool) -> None:
    """Configure CUDA matmul / cuDNN precision for a run.

    bf16 runs allow TF32 in cuDNN convolutions but keep fp32 matmul accumulation;
    fp32 runs pin TF32 off and raise matmul precision to 'high'. Single source of
    truth for both the training entrypoint and the analysis loaders.
    """
    torch.backends.cudnn.benchmark = True
    if use_bf16:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("high")

# =============================================================================
# RNG / Seeding
# =============================================================================
DEFAULT_SEED: int = 42
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


def _restore_rng(rng: dict) -> None:
    """Restore RNG states from a checkpoint dict.

    Checkpoints loaded with `torch.load(map_location='cuda')` have their saved
    RNG tensors mapped onto the GPU, but `set_rng_state` requires a CPU uint8
    ByteTensor - so the states are forced back to CPU/uint8 before restoring.
    """
    assert all(k in rng for k in ["torch","python","numpy","cuda"])
    
    torch.random.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["cuda"].cpu().to(torch.uint8))


def load_exp_seed(run_id: Path) -> int:
    """Read meta.seed from an experiment's config """
    run_id = Path(run_id)
    cfg_path = run_id if run_id.suffix == ".yaml" else run_id / "config.yaml"
    with open(cfg_path) as f:
        params = yaml.safe_load(f)
    seed = params.get("meta", {}).get("seed", -1)
    if seed < 0:
        raise ValueError(f"No seed found in {cfg_path}")
    return seed


def get_numpy_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else SEED)
