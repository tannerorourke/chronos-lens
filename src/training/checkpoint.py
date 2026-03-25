from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.models.sequential_jepa import JEPA
from src.models.jepa_stopgrad import JEPAStopGrad


def load_embedding_vecs(emb_path: Path):
    npz = np.load(emb_path, allow_pickle=True)
    return { k: npz[k] for k in npz }

    
def save_embedding_vecs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    save_dir: Path | None = None
):
    """ Calculate embeddings for all masked tokens in the dataset.
        This is run on a checkpoint cycle during training.
    """
    from src.analysis.displacement import calc_embedding_vecs, run_embedding_stat_check
    
    model_vecs = calc_embedding_vecs(model, loader, device)
    z_context      = model_vecs["z_context"]
    z_pred         = model_vecs["z_pred"]
    z_target       = model_vecs["z_target"]
    z_pc_delta     = model_vecs["delta"]
    pred_error     = model_vecs["pred_error"]
    observed_traj  = model_vecs["observed_traj"]
    subject_ids    = model_vecs["subject_ids"]
    mask_positions = model_vecs["mask_positions"]
    labels         = model_vecs["labels"]

    stat_log = run_embedding_stat_check(
        z_context, z_pred, z_target, z_pc_delta,
        pred_error=pred_error, observed_traj=observed_traj, labels=labels
    )
    if not all([stat_log['z_pred_ok'], stat_log['zpc_delta_ok'], stat_log["pred_ok"]]):
        print("Embeddings NOT saved: one or more sanity checks failed.")
        return model_vecs, stat_log

    if save_dir is not None:
        ep_str = f"_{epoch}" if epoch is not None else ""
        file = (save_dir / f"embeddings{ep_str}").with_suffix(".npz")
        np.savez(file,
                z_context = z_context,
                z_pred = z_pred,
                z_target = z_target,
                delta = z_pc_delta,
                pred_error = pred_error,
                observed_traj = observed_traj,
                subject_ids = subject_ids,
                mask_positions = mask_positions,
                labels = labels)
        print(f"   Embeddings saved: {save_dir.name}/{file.name} (epoch {epoch})")
    
    return model_vecs, stat_log


def save_checkpoint(model: JEPA | JEPAStopGrad, model_params: dict,
                    optimizer, scheduler, scaler,
                    epoch: int, global_step: int, loss_history: list,
                    save_dir: Path
) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        file = save_dir / f"checkpoint_{epoch}.pt"
        torch.save({
            "model":        model.state_dict(),
            "model_params": model_params,
            "optimizer":    optimizer.state_dict(),
            "scheduler":    None if scheduler is None else scheduler.state_dict(),
            "scaler":       None if scaler is None else scaler.state_dict(),
            "epoch":        epoch,
            "global_step":  global_step,
            "loss_history": loss_history,
            "rng_states": {
                "torch":    torch.random.get_rng_state(),
                "numpy":    np.random.get_state(),
                "cuda":     torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }, file)
        print(f"   Checkpoint saved: {save_dir.name}/{file.name} (epoch {epoch})")


def build_model(model_params: dict, device: torch.device):
    arch = model_params.get("architecture", "stopgrad") # ema, stopgrad
    
    if arch == "stopgrad":
        from src.models.jepa_stopgrad import JEPAStopGrad
        model = JEPAStopGrad(**model_params)
    elif arch == "ema":
        from src.models.sequential_jepa import JEPA
        model = JEPA(**model_params)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model.to(device)


def load_model_notrain(path, device, restore_rng = True):
    """Load model model checkpoint for analysis/freezing, no training"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model_params = checkpoint["model_params"]
    model = build_model(model_params, device)
    model.load_state_dict(checkpoint["model"])
            
    if restore_rng and "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        torch.random.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
    
    model.eval()
    return model, checkpoint


def load_model_checkpoint(path: Path, device: torch.device, restore_rng: bool = True):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model_params = checkpoint["model_params"]
    model = build_model(model_params, device)
    model.load_state_dict(checkpoint["model"])
    
    if restore_rng and "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        torch.random.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
            
    epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    loss_hist = checkpoint.get("loss_history", [])
    
    model.eval()
    return model, checkpoint, epoch, global_step, loss_hist