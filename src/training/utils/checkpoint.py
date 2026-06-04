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

from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.jepa_ema import JEPA_EMA
from src.models.supervised_transformer import SupervisedTransformer
from src.utils.seed import SEED, _restore_rng


def count_improvement(current, best, since_best, delta=0.0, gt=False):
    """ helper to calc lowest so far for metric"""
    condition = (current < best - delta) if gt else (current > best + delta)
    if condition:
        return current, 0, True
    return best, since_best + 1, False


def build_model(model_params: dict, device: torch.device) -> JEPA_EMA | JEPAStopGrad | SupervisedTransformer:
    arch = model_params.get("architecture", "")
    if arch == "ema":
        return JEPA_EMA(**model_params).to(device) 
    elif arch == "stopgrad":
        return JEPAStopGrad(**model_params).to(device)
    elif arch == "supervised":
        return SupervisedTransformer(**model_params).to(device)
    raise ValueError(f"Unknown architecture: '{arch}'")


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


def save_periodic(
    model,
    model_params: dict,
    optimizer,
    scheduler,
    epoch: int,
    total_epochs: int,
    save_cycle: int | None,
    global_step: int,
    loss_history: list,
    ckpt_dir: Path,
    logger,
) -> str | None:
    """Save a rolling `last.pt` every `save_cycle` epochs and a preserved 
    `checkpoint_<epoch>.pt` written on the final epoch. shared by every 
    training loop.

    Decouples persistence from training dynamics. Each write is followed by 
    a blocking, verified S3 push via the logger (a no-op unless `sync_s3` 
    is enabled). Returns the final-epoch checkpoint filename on the last 
    epoch (so the orchestrator can extract embeddings from it), else None.
    """
    is_final = (epoch == total_epochs)
    is_cycle = bool(save_cycle) and (epoch % save_cycle == 0)
    if not (is_final or is_cycle):
        return None

    filename = f"checkpoint_{epoch}.pt" if is_final else "last.pt"
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    path = save_checkpoint(
        state, model_params, optimizer, scheduler,
        epoch, global_step, loss_history, ckpt_dir, filename=filename)
    logger.push_checkpoint(path)  # verified blocking push of the checkpoint
    logger.sync()                 # non-blocking bulk sync of the rest (metrics/logs)
    return filename


def load_model_checkpoint(
    path: Path | None,
    device: torch.device,
    restore_rng: bool = True,
    exp_dir: Path | None = None,
    ckpt_name: str | None = None
) -> tuple[JEPA_EMA | JEPAStopGrad | SupervisedTransformer, dict]:
    """ Load model.eval() at a given checkpoint. """
    if path is None:
        assert exp_dir is not None and ckpt_name is not None
        path = exp_dir / "checkpoints" / ckpt_name
    if exp_dir is not None and exp_dir.exists() and ckpt_name is not None:
        path = exp_dir / "checkpoints" / ckpt_name
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_params = checkpoint["model_params"]
    model = build_model(model_params, device)
    model.load_state_dict(checkpoint["model"])

    if restore_rng:
        _restore_rng(checkpoint)

    model.eval()
    return model, checkpoint
  

def sync_model_checkpoint(
    model,
    optimizer,
    scheduler,
    path: Path,
    device: torch.device,
    restore_rng: bool = True,
) -> tuple:
    """ Sync created model and optimizers to checkpoint. """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    epoch = checkpoint["epoch"]
    
    model_dict = checkpoint["model"]
    model.load_state_dict(model_dict)
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    
    if restore_rng:
        _restore_rng(checkpoint)
    
    model_params = checkpoint["model_params"]
    global_step = checkpoint["global_step"]
    loss_hist = checkpoint.get("loss_history", [])

    model.eval()
    return model, model_params, optimizer, scheduler, epoch, global_step, loss_hist
