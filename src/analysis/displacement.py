"""Displacement field extraction, embedding statistics, and embedding I/O.

- Core embedding extraction pipeline (calc_embedding_vecs),
- sanity checks (run_embedding_stat_check)
save/load helpers used
by the training loop's checkpoint cycle.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats



rng = np.random.default_rng(42)


@torch.no_grad()
def calc_embedding_vecs(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """ Calculate context, predicted, and target embeddings for all masked tokens in the dataset.
    
        Returns:
        
            - z_context:      (N, D) context embeddings
            - z_pred:         (N, D) predicted embeddings
            - z_target:       (N, D) target embeddings
            - delta:          (N, D) displacement vectors (pred - ctx)
            - pred_error:     (N, D) prediction error vectors (pred - target)
            - observed_traj:  (N, D) observed trajectory vectors (target - ctx)
            - subject_ids:    (N,) arr of subject IDs
            - mask_positions: (N,) arr of masked positions in sequence
            - labels:         (N,) arr of true tokens at masked positions
    """
    model.eval()
    all_z_ctx:  list[np.ndarray] = []
    all_z_pred: list[np.ndarray] = []
    all_z_tgt:  list[np.ndarray] = []
    all_sids:   list[str]        = []
    all_mpos:   list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        z_ctx, z_pred, z_tgt = model(batch_dev)

        all_z_ctx.append(z_ctx.cpu().numpy())
        all_z_pred.append(z_pred.cpu().numpy())
        all_z_tgt.append(z_tgt.cpu().numpy())
        all_sids.extend(batch["subject_ids"])
        all_mpos.append(batch["mask_pos"].numpy())
        all_labels.append(batch["labels"].numpy())

    z_context   = np.concatenate(all_z_ctx,  axis=0)
    z_pred_arr  = np.concatenate(all_z_pred, axis=0)
    z_target    = np.concatenate(all_z_tgt,  axis=0)
    z_pc_delta  = z_pred_arr - z_context
    pred_error  = z_pred_arr - z_target
    observed_traj = z_target - z_context
    subject_ids = np.array(all_sids)
    mask_positions = np.concatenate(all_mpos,  axis=0)
    labels      = np.concatenate(all_labels, axis=0)

    return {
        "z_context": z_context,
        "z_pred": z_pred_arr,
        "z_target": z_target,
        "delta": z_pc_delta,
        "pred_error": pred_error,
        "observed_traj": observed_traj,
        "subject_ids": subject_ids,
        "mask_positions": mask_positions,
        "labels": labels
    }


def run_embedding_stat_check(
    z_context:  np.ndarray,
    z_pred:     np.ndarray,
    z_target:   np.ndarray,
    z_pc_delta: np.ndarray,
    pred_error: np.ndarray = None,
    observed_traj: np.ndarray = None,
    labels:     np.ndarray = None,
    frac_nontrivial_thresh: float = 0.5,
) -> dict:
    """ Run useful checks on extracted embeddings to ensure
        predictor hasn't collapsed, delta remains statistically significant, etc
    """
    stat_log: Dict[str, Any] = {}

    # The variance of a significant portion of z_pred's dimensions
    # should remain non-trivial (> 0.01) over time to avoid overfitting
    pred_std        = z_pred.std(axis=0)
    frac_nontrivial = float((pred_std > 0.01).mean())
    stat_log['z_pred_ok'] = frac_nontrivial > frac_nontrivial_thresh
    stat_log['z_pred_min_std'] = pred_std.min()
    stat_log['z_pred_frac_nt_dims'] = frac_nontrivial
    
    # Check displacement field ||Delta|| hasn't collapsed by ensuring
    # its full variance hasn't collapsed
    delta_norms = np.linalg.norm(z_pc_delta, axis=-1)
    if labels is not None:
        norms0 = delta_norms[labels == 0]
        norms1 = delta_norms[labels == 1]
        
        if len(norms0) >= 1 and len(norms1) >= 1:
            mw_stat, mw_p = stats.mannwhitneyu(norms0, norms1, alternative="two-sided")
            mw_stat, mw_p = float(mw_stat), float(mw_p)
        else:
            mw_stat, mw_p = np.nan, np.nan
        stat_log["mannwhitney_stat"] = mw_stat
        stat_log["mannwhitney_p"] = mw_p
    
    stat_log["zpc_delta_ok"] = delta_norms.std() > 1e-4
    stat_log["zpc_delta_norm_min"] = delta_norms.min()
    stat_log["zpc_delta_norm_max"] = delta_norms.max()
    stat_log["zpc_delta_norm_std"] = delta_norms.std()
    stat_log["zpc_delta_norm_mean"] = delta_norms.mean()
    
    # Check the predictor beats a random baseline scaled to z_target's distribution
    rand_pred = rng.standard_normal(z_pred.shape).astype(np.float32) * z_target.std() + z_target.mean()
    pred_dist = float(np.linalg.norm(z_pred    - z_target, axis=-1).mean())
    rand_dist = float(np.linalg.norm(rand_pred - z_target, axis=-1).mean())
    stat_log["pred_ok"] = pred_dist < rand_dist
    stat_log["||z_pred-z_tgt||"] = pred_dist
    stat_log["||rand-z_tgt||"] = rand_dist

    # Norm stats for pred_error (P - T) and observed_traj (T - C)
    if pred_error is not None:
        pe_norms = np.linalg.norm(pred_error, axis=-1)
        stat_log["pred_error_norm_mean"] = float(pe_norms.mean())
        stat_log["pred_error_norm_std"]  = float(pe_norms.std())
        stat_log["pred_error_norm_min"]  = float(pe_norms.min())
        stat_log["pred_error_norm_max"]  = float(pe_norms.max())

    if observed_traj is not None:
        ot_norms = np.linalg.norm(observed_traj, axis=-1)
        stat_log["observed_traj_norm_mean"] = float(ot_norms.mean())
        stat_log["observed_traj_norm_std"]  = float(ot_norms.std())
        stat_log["observed_traj_norm_min"]  = float(ot_norms.min())
        stat_log["observed_traj_norm_max"]  = float(ot_norms.max())

    return stat_log


def load_embedding_vecs(emb_path: Path) -> dict:
    """Load a previously saved embeddings .npz file."""
    npz = np.load(emb_path, allow_pickle=True)
    return {k: npz[k] for k in npz}


def save_embedding_vecs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    save_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Extract embeddings, run sanity checks, and optionally save to disk.

    Called on a checkpoint cycle during training.
    """
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
        pred_error=pred_error, observed_traj=observed_traj, labels=labels,
    )
    if not all([stat_log["z_pred_ok"], stat_log["zpc_delta_ok"], stat_log["pred_ok"]]):
        print("Embeddings NOT saved: one or more sanity checks failed.")
        return model_vecs, stat_log

    if save_dir is not None:
        ep_str = f"_{epoch}" if epoch is not None else ""
        file = (save_dir / f"embeddings{ep_str}").with_suffix(".npz")
        np.savez(
            file,
            z_context=z_context,
            z_pred=z_pred,
            z_target=z_target,
            delta=z_pc_delta,
            pred_error=pred_error,
            observed_traj=observed_traj,
            subject_ids=subject_ids,
            mask_positions=mask_positions,
            labels=labels,
        )
        print(f"   Embeddings saved: {save_dir.name}/{file.name} (epoch {epoch})")

    return model_vecs, stat_log