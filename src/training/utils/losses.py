"""
Loss functions for JEPA stop-gradient training.

VICReg regularization follows Bardes et al., 2022
("VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning").
Only the variance and covariance terms are used here - the invariance role is
filled by the JEPA prediction loss.
"""

import torch
import torch.nn.functional as F


def vicreg_regularization(
    z: torch.Tensor,
    var_weight: float,
    cov_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variance + covariance regularization for an embedding matrix.

    Parameters
    ----------
    z : (B, D) embedding batch
    var_weight : scalar multiplier for the variance (anti-collapse) term
    cov_weight : scalar multiplier for the covariance (decorrelation) term

    Returns
    -------
    weighted_var  : var_weight * var_loss
    weighted_cov  : cov_weight * cov_loss
    raw_total     : var_loss + cov_loss  (unweighted, for logging)
    """
    B, D = z.shape

    # Need at least 2 samples for meaningful variance/covariance
    if B < 2:
        zero = z.new_tensor(0.0)
        return zero, zero, zero

    # --- Variance term
    # Per-dimension std across batch -> hinge penalty for dims with std <1
    std = torch.sqrt(z.var(dim=0) + 1e-4) # (D,)
    var_loss = torch.mean(F.relu(1.0 - std))

    # --- Covariance term
    # Center -> compute (D, D) covariance matrix
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (B - 1)    # (D, D)
    # Mean of squared off-diagonal elements
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    cov_loss = (cov[off_diag_mask] ** 2).mean()

    return var_weight * var_loss, cov_weight * cov_loss, var_loss + cov_loss


def jepa_stopgrad_loss(
    z_enc: torch.Tensor,
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    z_target_sg: torch.Tensor,
    ctx_pad_mask: torch.Tensor,
    sim_weight: float,
    var_weight: float,
    cov_weight: float,
) -> dict[str, torch.Tensor]:
    # --- Sim term: normalized, using stop-grad target
    z_pred_n   = F.normalize(z_pred, dim=-1)
    z_target_n = F.normalize(z_target_sg, dim=-1)
    sim_loss = 1.0 - (z_pred_n * z_target_n).sum(dim=-1).mean()  # = -cos + 1
    
    # --- VICReg on z_pred
    var_pred, cov_pred, _ = vicreg_regularization(z_pred, var_weight, cov_weight)
    
    # --- VICReg on z_enc (flatten valid encounters)
    z_enc_flat = z_enc[~ctx_pad_mask]
    var_enc, cov_enc, _ = vicreg_regularization(z_enc_flat, var_weight, cov_weight)
    
    # --- VICReg on z_target (WITH grad - this is the symmetry fix) ---
    var_tgt, cov_tgt, _ = vicreg_regularization(z_target, var_weight, cov_weight)
    
    total = (
        sim_weight * sim_loss + 
        var_pred + cov_pred + 
        var_enc + cov_enc + 
        var_tgt + cov_tgt
    )

    return {
        "loss":     total,
        "sim":      sim_loss.detach(),
        "var_tgt":  var_tgt.detach(),
        "cov_tgt":  cov_tgt.detach(),
        "var_pred": var_pred.detach(),
        "cov_pred": cov_pred.detach(),
        "var_enc":  var_enc.detach(),
        "cov_enc":  cov_enc.detach()
    }
