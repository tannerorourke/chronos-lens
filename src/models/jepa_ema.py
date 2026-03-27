"""EMA-based JEPA orchestrator.

Composes EncounterEncoder, Predictor, and an EMA target encoder.
The training loop calls update_target_encoder() after each optimizer step.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import EncounterEncoder
from src.models.predictor import Predictor


class JEPA_EMA(nn.Module):
    """EMA-based Joint Embedding Predictive Architecture.

    Composes a context encoder (online), a predictor, and an EMA target
    encoder (deepcopy of context encoder, no gradients).

    Components
    ----------
    encoder        : EncounterEncoder — online context path (with grads)
    predictor      : Predictor MLP — maps (z_context, mask_pos) to z_pred
    target_encoder : EncounterEncoder — EMA shadow, no backprop
    """

    def __init__(
        self,
        vocab_size:       int,
        embed_dim:        int = 64,
        num_heads:        int = 2,
        num_layers:       int = 2,
        max_seq_len:      int = 256,
        ffn_dim:          int = 256,
        predictor_hidden: int = 128,
        pad_idx:          int = 0,
        tau:              float = 0.996,
    ):
        super().__init__()
        self.architecture     = "ema"
        self.vocab_size       = vocab_size
        self.embed_dim        = embed_dim
        self.num_heads        = num_heads
        self.num_layers       = num_layers
        self.ffn_dim          = ffn_dim
        self.max_seq_len      = max_seq_len
        self.predictor_hidden = predictor_hidden
        self.momentum         = tau

        self.encoder = EncounterEncoder(
            vocab_size, embed_dim, num_heads, num_layers, max_seq_len, ffn_dim, pad_idx)
        self.predictor = Predictor(embed_dim, predictor_hidden, max_seq_len)

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def set_momentum(self, m: float):
        """Set the EMA momentum coefficient (called by the training loop scheduler)."""
        self.momentum = m

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update. Called from training loop after optimizer.step()."""
        m = self.momentum
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(m).add_((1. - m) * p_online.detach().data)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_context : (B, D) context representation (differentiable)
        z_pred    : (B, D) predictor output       (differentiable)
        z_target  : (B, D) EMA target encoding    (no grad)
        """
        ctx_tokens   = batch["ctx_tokens"]    # (B, C, T_tok)
        ctx_tok_mask = batch["ctx_tok_mask"]   # (B, C, T_tok)
        ctx_pad_mask = batch["ctx_pad_mask"]   # (B, C)
        tgt_tokens   = batch["tgt_tokens"]     # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"]   # (B, T_tok)
        mask_pos     = batch["mask_pos"]       # (B,)

        # -- Context path (with grads) ----------------------------------------
        z_context = self.encoder(ctx_tokens, ctx_tok_mask, ctx_pad_mask)  # (B, D)

        # -- Target path (no grads — EMA encoder) -----------------------------
        with torch.no_grad():
            z_target = self.target_encoder(tgt_tokens, tgt_tok_mask)      # (B, D)
            z_target = F.layer_norm(z_target, (z_target.size(-1),))

        # -- Predictor ---------------------------------------------------------
        z_pred = self.predictor(z_context, mask_pos)                      # (B, D)

        return z_context, z_pred, z_target

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.encoder.layers
