"""
Loss functions for JEPA stop-gradient training.

VICReg regularization follows Bardes et al., 2022
("VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning").
Only the variance and covariance terms are used here - the invariance role is
filled by the JEPA prediction loss (MSE between z_pred and z_target).
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
        return var_weight * zero, cov_weight * zero, zero

    # --- Variance term ---
    # Per-dimension std across the batch.
    std = torch.sqrt(z.var(dim=0) + 1e-4) # (D,)
    # Hinge: penalise dimensions whose std drops below 1
    var_loss = torch.mean(F.relu(1.0 - std))

    # --- Covariance term ---
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
    ctx_pad_mask: torch.Tensor,
    sim_weight: float,
    var_weight: float,
    cov_weight: float,
) -> dict[str, torch.Tensor]:
    """Combined prediction + VICReg-style regularization loss.

    Parameters
    ----------
    z_enc     : (B, D) per encounter representation (has grad)
    z_pred    : (B, D) predictor output (has grad)
    z_target  : (B, D) stop-grad target encoding (detached)
    ctx_pad_mask : (B, C) True=padding
    sim_weight : multiplier for MSE prediction term
    var_weight : multiplier for variance terms
    cov_weight : multiplier for covariance terms

    Returns
    -------
    dict with: loss, sim, var_pred, var_enc, cov_pred, cov_enc
    """
    # --- Similarity (prediction) loss ---
    sim_loss = F.mse_loss(z_pred, z_target)

    # --- VICReg regularization on z_pred ---
    var_pred, cov_pred, _ = vicreg_regularization(z_pred,    var_weight, cov_weight)
    
    # --- VICReg on z_enc (flatten valid encounters) ---
    valid_mask = ~ctx_pad_mask                          # (B, C) True=real
    z_enc_flat = z_enc[valid_mask]                      # (N_valid, D)
    var_enc, cov_enc, _ = vicreg_regularization(z_enc_flat, var_weight, cov_weight)

    total = sim_weight * sim_loss + var_pred + cov_pred + var_enc + cov_enc

    return {
        "loss":     total,
        "sim":      sim_loss.detach(),
        "var_pred": var_pred.detach(),
        "var_enc":  var_enc.detach(),
        "cov_pred": cov_pred.detach(),
        "cov_enc":  cov_enc.detach(),
    }
