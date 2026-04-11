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

from src.models.jepa_ema import JEPA_EMA
from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.supervised_transformer import SupervisedTransformer
from src.utils.seed import _restore_rng


def build_model(model_params: dict, device: torch.device) -> JEPA_EMA | JEPAStopGrad | SupervisedTransformer:
    arch = model_params.get("architecture", "")

    if arch == "stopgrad":
        return JEPAStopGrad(**model_params).to(device)
    elif arch == "ema":
        return JEPA_EMA(**model_params).to(device)
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
    seed: int | None = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    file = save_dir / f"checkpoint_{epoch}.pt"

    torch.save({
        "model":        state_dict,
        "model_params": model_params,
        "seed":         seed,
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


def load_model_notrain(
    path: Path,
    device: torch.device,
    restore_rng: bool = True,
) -> tuple[JEPA_EMA | JEPAStopGrad | SupervisedTransformer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_params = checkpoint["model_params"]
    model = build_model(model_params, device)
    model.load_state_dict(checkpoint["model"])

    if restore_rng:
        _restore_rng(checkpoint)

    model.eval()
    return model, checkpoint


def load_model_checkpoint(
    model,
    optimizer,
    scheduler,
    path: Path,
    device: torch.device,
    restore_rng: bool = True,
) -> tuple:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    epoch = checkpoint["epoch"]
    
    model_dict = checkpoint["model"]
    model.load_state_dict(model_dict)
    print(f"loaded {model.architecture} model from epoch {epoch}")
    
    optimizer_dict = checkpoint["optimizer"]
    optimizer.load_state_dict(optimizer_dict)
    print(f"loaded optimizer from epoch {epoch}")
    
    scheduler_dict = checkpoint["scheduler"]
    if scheduler is not None and scheduler_dict is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"loaded scheduler from epoch {epoch}")
    
    if restore_rng:
        _restore_rng(checkpoint)

    model_params = checkpoint["model_params"]
    global_step = checkpoint["global_step"]
    loss_hist = checkpoint.get("loss_history", [])

    model.eval()
    return model, model_params, optimizer, scheduler, epoch, global_step, loss_hist
