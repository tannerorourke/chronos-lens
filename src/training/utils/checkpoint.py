"""Checkpoint save / load and model construction.

- Serialization of model weights, optimizer state, and RNG state
for training resumption. 
- build_model() for constructing the correct architecture from a 
params dict.
"""

from pathlib import Path
import random

import numpy as np
import torch

from src.models import MODEL_TYPE, SparseAutoencoder, build_model
from src.utils.system import SEED, _restore_rng
from src.utils.io import EXPS_DIR


def save_checkpoint(
    state_dict: dict,
    model_params: dict,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    loss_history: list,
    save_dir: Path,
    filename: str | None = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    file = save_dir / (filename or f"checkpoint_{epoch}.pt")

    torch.save({
        "model":        state_dict,
        "model_params": model_params,
        "seed":         SEED,
        "optimizer":    None if optimizer is None else optimizer.state_dict(),
        "scheduler":    None if scheduler is None else scheduler.state_dict(),
        "epoch":        epoch,
        "global_step":  global_step,
        "loss_history": loss_history,
        "rng_states": {
            "torch":    torch.random.get_rng_state(),
            "numpy":    np.random.get_state(),
            "cuda":     torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python":   random.getstate()
        },
    }, file)
    print(f"   Checkpoint saved: {save_dir.name}/{file.name} (epoch {epoch})")
    return file


def sync_model_checkpoint(
    model,
    optimizer,
    scheduler,
    path: Path,
    device: torch.device,
    restore_rng: bool = True,
) -> tuple:
    """ 
    Sync created model and optimizers to checkpoint.
    Use for restarting training.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    epoch = checkpoint["epoch"]
    
    model_dict = checkpoint["model"]
    model.load_state_dict(model_dict)
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    
    if restore_rng:
        _restore_rng(rng=checkpoint["rng_states"])
    
    model_params = checkpoint["model_params"]
    global_step = checkpoint["global_step"]
    loss_hist = checkpoint.get("loss_history", [])

    model.eval()
    return model, model_params, optimizer, scheduler, epoch, global_step, loss_hist
  
  
def load_model_eval(
    device: torch.device,
    run_dir: Path = EXPS_DIR,
    restore_rng: bool = True,
    run_id: str | None = None,
    filename: str | None = None,
    abs_path: Path | None = None,
) -> tuple[MODEL_TYPE, dict]:
    """
    Load model in eval mode at given checkpoint.
    """
    if run_id is not None and filename is not None:
        path = run_dir / run_id / "checkpoints" / filename
        assert path.exists(), f"[load_model_from_checkpoint] Checkpoint not found."
    elif abs_path is not None:
        path = abs_path
        assert path.exists(), f"[load_model_from_checkpoint] Checkpoint not found."
    else:
        raise FileNotFoundError(f"[load_model_from_checkpoint] Invalid parameters.")
    
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model_params = ckpt["model_params"]
    model = build_model(model_params, device)
    model.load_state_dict(ckpt["model"])

    if restore_rng:
        _restore_rng(ckpt)

    model.eval()
    return model, ckpt
    
    
def load_sae_eval(checkpoint_path: Path, device: torch.device) -> SparseAutoencoder:
    """Load a trained SAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SparseAutoencoder(
        embed_dim=ckpt["embed_dim"],
        n_features=ckpt["n_features"],
        top_k=ckpt["top_k"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model