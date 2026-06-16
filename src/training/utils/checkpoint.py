"""Checkpoint save / load and model construction.

- Serialization of model weights, optimizer state, and RNG state
for training resumption. 
- build_model() for constructing the correct architecture from a 
params dict.
"""

from pathlib import Path
import random
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch

from src.models import MODEL_TYPE, SparseAutoencoder, build_model
from src.utils.system import SEED, _restore_rng
from src.utils.io import EXPS_DIR, save_json


def save_checkpoint(
    state_dict: dict,
    model_params: dict,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    loss_history: list,
    file: Path,
) -> Path:
    file.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"   Checkpoint saved: {file.parent.name}/{file.name} (epoch {epoch})")
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

    if restore_rng and "rng_states" in ckpt:
        _restore_rng(ckpt["rng_states"])

    model.eval()
    return model, ckpt
    

def save_sae_results(
    model: SparseAutoencoder,
    data_vec: np.ndarray,
    loss_history: list,
    sae_exp_dir: Path,
    device: torch.device
):
    """Save SAE checkpoint, dictionary, activations, and loss curve"""
    # -- Checkpoint
    ckpt_path = sae_exp_dir / "sae.pt"
    torch.save({
        "model": model.state_dict(),
        "model_params": {
            "embed_dim": model.embed_dim,
            "n_features": model.n_features,
            "top_k": model.top_k,
        },
        "loss_history": loss_history,
    }, ckpt_path)

    # -- Dictionary (decoder weights)
    # decoder.weight is (embed_dim, n_features); save as (n_features, embed_dim)
    dictionary = model.D.weight.data.cpu().numpy().T
    dict_path = sae_exp_dir / "decoder_weights.npy"
    np.save(dict_path, dictionary)

    # -- Activations on full dataset
    with torch.no_grad():
        x = torch.tensor(data_vec, dtype=torch.float32, device=device)
        _, act = model(x)
    act_np = act.cpu().numpy()
    np.save(sae_exp_dir / "activations.npy", act_np)

    # -- Loss curve
    from src.training.utils.logging import plot_loss_curve
    plot_loss_curve(loss_history, save_path=sae_exp_dir / "sae_loss_curve",
                    title="SAE Training Loss")

    # -- Summary JSON
    summary_path = sae_exp_dir / "sae_summary.json"
    save_json({
        "embed_dim": model.embed_dim,
        "n_features": model.n_features,
        "top_k": model.top_k,
        "n_samples": data_vec.shape[0],
        "final_loss": loss_history[-1] if loss_history else None,
        "n_epochs": len(loss_history),
        "n_dead_features": int(((act_np != 0).sum(axis=0) == 0).sum()),
        "mean_active_per_sample": float((act_np != 0).sum(axis=1).mean()),
    }, summary_path)

    logger.info(f"SAE results saved in {sae_exp_dir}")