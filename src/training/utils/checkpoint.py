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
from src.utils.seed import _restore_rng


def save_checkpoint(
    model: JEPA_EMA | JEPAStopGrad,
    model_params: dict,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    loss_history: list,
    save_dir: Path,
    seed: int | None = None,
) -> None:
    """Save a training checkpoint to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    file = save_dir / f"checkpoint_{epoch}.pt"

    torch.save({
        "model":        model.state_dict(),
        "model_params": model_params,
        "seed":         seed,
        "optimizer":    None if optimizer is None else optimizer.state_dict(),
        "scheduler":    None if scheduler is None else scheduler.state_dict(),
        "scaler":       None if scaler is None else scaler.state_dict(),
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


def build_model(model_params: dict, device: torch.device) -> JEPA_EMA | JEPAStopGrad:
    """Construct a model from a params dict, filtering non-constructor keys."""
    arch = model_params.get("architecture", "stopgrad")
    ctor_params = {k: v for k, v in model_params.items()}

    if arch == "stopgrad":
        model = JEPAStopGrad(**ctor_params)
    elif arch == "ema":
        model = JEPA_EMA(**ctor_params)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model.to(device)


def load_model_notrain(
    path: Path,
    device: torch.device,
    restore_rng: bool = True,
) -> tuple[JEPA_EMA | JEPAStopGrad, dict]:
    """Load a model checkpoint for analysis / freezing (no training)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_params = checkpoint["model_params"]
    model = build_model(model_params, device)
    model.load_state_dict(checkpoint["model"])

    if restore_rng:
        _restore_rng(checkpoint)

    model.eval()
    return model, checkpoint


def load_model_checkpoint(
    path: Path,
    device: torch.device,
    restore_rng: bool = True,
) -> tuple[JEPA_EMA | JEPAStopGrad, dict, int, int, list]:
    """Load a model checkpoint for training resumption."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_params = checkpoint["model_params"]
    model = build_model(model_params, device)
    model.load_state_dict(checkpoint["model"])

    if restore_rng:
        _restore_rng(checkpoint)

    epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    loss_hist = checkpoint.get("loss_history", [])

    model.eval()
    return model, checkpoint, epoch, global_step, loss_hist
