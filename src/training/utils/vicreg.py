import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    As done in SimCLR, VICReg, BYOL, Barlow Twins, apply the 
    contrastive/predictive/regularization loss on a projected version of
    the encoder output directly. This keeps encoder manifold structure
    representation clean for downstream analysis
    - z_proj = unconstrained, all loss terms computed
    - z_enc is the analysis object (on unit sphere)
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # NO LayerNorm

    def forward(self, x):
        return self.net(x)
    

class VicRegLoss(nn.Module):
    """
    VICReg regularization loss function, following Bardes et al., 2022
    ("VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning").
    
    Invariance = MSE prediction residual (forward); variance = two-sided (std->1)
    penalty and covariance = off-diagonal decorrelation (vicreg helper), both on the
    projected embeddings. Auto-accumulates per-batch loss terms for epoch logging.

    Parameters
    ----------
    sim_wt : scalar multiplier for the invariance (MSE prediction residual) term
    var_wt : scalar multiplier for the variance (anti-collapse) term
    cov_wt : scalar multiplier for the covariance (decorrelation) term
    on_enc : bool, whether to apply VICReg regularization on z_enc
    on_pred : bool, whether to apply VICReg regularization on z_pred
    """
    def __init__(
        self, 
        sim_wt: float, 
        var_wt: float, 
        cov_wt: float, 
        on_enc: bool = True,
        on_pred: bool = False,
        on_tgt: bool = False
    ):
        super().__init__()
        self.on_enc = on_enc
        self.on_tgt = on_tgt
        self.on_pred = on_pred
        self.sim_wt = sim_wt
        self.var_wt = var_wt
        self.cov_wt = cov_wt
        
        self.vicreg_accum = { "inv": 0.0, "cos_dist": 0.0 }
        if self.on_enc:
            self.vicreg_accum["var_enc"] = 0.0
            self.vicreg_accum["cov_enc"] = 0.0
        if self.on_tgt:
            self.vicreg_accum["var_tgt"] = 0.0
            self.vicreg_accum["cov_tgt"] = 0.0
        if self.on_pred:
            self.vicreg_accum["var_pred"] = 0.0
            self.vicreg_accum["cov_pred"] = 0.0
            
    def reset_accum(self):
        for k in self.vicreg_accum.keys():
            self.vicreg_accum[k] = 0.0
            
    def compute_accum(self, n_batches: int):
        return {
            k: self.vicreg_accum[k] / max(n_batches, 1)
            for k in self.vicreg_accum.keys()
        }
        
    def vicreg(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variance + covariance regularization for an embedding matrix.
           - Takes embedding matrix z (B, D)
           - Returns weighted_var, weighted_cov, raw_total
        """
        B, D = z.shape

        # Need >=2 samples for meaningful variance/covariance
        if B < 2:
            zero = z.new_tensor(0.0)
            return zero, zero, zero

        # -- Var term: per-dim std over batch -> two-sided penalty toward std=1.
        #    (std-1)^2 floors collapse (std->0) AND caps runaway (std>>1); the cap is
        #    the Phase-15 fix - a one-sided relu(1-std) hinge let the encoder satisfy
        #    the floor by inflating scale without bound (Phase-14 magnitude runaway).
        std = torch.sqrt(z.var(dim=0) + 1e-4) # (D,)
        var_loss = torch.mean((std - 1.0) ** 2)

        # --- Cov term: Center -> compute (D, D) covariance matrix
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (B - 1) # (D, D)
        # Mean of squared off-diagonal elements
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        cov_loss = (cov[off_diag_mask] ** 2).mean()

        return self.var_wt * var_loss, self.cov_wt * cov_loss, var_loss + cov_loss

    def forward(
        self, 
        z_enc: torch.Tensor,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        ctx_pad_mask: torch.Tensor,
        z_target_sg: torch.Tensor = None,
        projector: nn.Module = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # -- invariance: MSE prediction residual on the raw (un-normalized) vectors,
        #    using the detached target (SG) or no-grad target (EMA). MSE scores the
        #    full displacement P-T incl. magnitude, so radius stays a meaningful axis
        #    (evidence-magnitude; origin = null) - see TIMELINE Phase 15.
        tgt_for_sim = z_target_sg if z_target_sg is not None else z_target
        inv_loss = F.mse_loss(z_pred, tgt_for_sim)

        loss = (self.sim_wt * inv_loss)
        # -- cosine distance kept as a scale-free DIAGNOSTIC (not in the loss) so a
        #    run shows direction- vs magnitude-learning side by side with the MSE
        cos_dist = 1.0 - (F.normalize(z_pred, dim=-1)
                          * F.normalize(tgt_for_sim, dim=-1)).sum(dim=-1).mean()
        loss_dict = { "inv": inv_loss.detach(), "cos_dist": cos_dist.detach() }
        self.vicreg_accum["inv"] += inv_loss.detach().item()
        self.vicreg_accum["cos_dist"] += cos_dist.detach().item()

        if self.on_enc:
            z_enc_flat = z_enc[~ctx_pad_mask]
            
            # -- provide gradient thru projector
            z_enc_reg = projector(z_enc_flat) if projector is not None else z_enc_flat
            var_enc, cov_enc, _ = self.vicreg(z_enc_reg)
            loss += (var_enc + cov_enc)
            loss_dict["var_enc"] = var_enc.detach()
            loss_dict["cov_enc"] = cov_enc.detach()
            self.vicreg_accum["var_enc"] += var_enc.detach().item()
            self.vicreg_accum["cov_enc"] += cov_enc.detach().item()
            
        if self.on_tgt:
            assert z_target.requires_grad, \
                "z_target_sg must be a have grad for vicreg loss (not from a frozen encoder)"
                
            # -- provide gradient thru projector
            z_tgt_reg = projector(z_target) if projector is not None else z_target
            var_tgt, cov_tgt, _ = self.vicreg(z_tgt_reg)
            loss += (var_tgt + cov_tgt)
            loss_dict["var_tgt"] = var_tgt.detach()
            loss_dict["cov_tgt"] = cov_tgt.detach()
            self.vicreg_accum["var_tgt"] += var_tgt.detach().item()
            self.vicreg_accum["cov_tgt"] += cov_tgt.detach().item()
            
        if self.on_pred:
            # -- no projection, always ln'd
            var_pred, cov_pred, _ = self.vicreg(z_pred)
            loss += (var_pred + cov_pred)
            loss_dict["var_pred"] = var_pred.detach()
            loss_dict["cov_pred"] = cov_pred.detach()
            self.vicreg_accum["var_pred"] += var_pred.detach().item()
            self.vicreg_accum["cov_pred"] += cov_pred.detach().item()
        
        loss_dict["loss"] = loss
        return loss, loss_dict