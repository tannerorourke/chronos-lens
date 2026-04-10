import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import EncounterEncoder
from src.models.predictor import Predictor


class JEPA_EMA(nn.Module):
    """
    EMA-based Joint Embedding Predictive Architecture.

    Composes a context encoder (online), a transformer predictor, and an EMA
    target encoder (deepcopy of context encoder, no gradients).

    Components
    ----------
    encoder        : EncounterEncoder - online context path (with grads)
    predictor      : Predictor - attends over (B, C, D) -> z_pred (B, D)
    target_encoder : EncounterEncoder - EMA shadow, no backprop
    
    Returns
    -------
    z_enc : (B, D) encoded representation (differentiable)
    z_pred    : (B, D) predictor output (differentiable)
    z_target  : (B, D) EMA target encoding (no grad)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int           = 64,
        num_heads: int           = 2,
        num_layers: int          = 2,
        max_seq_len: int         = 256,
        ffn_dim: int             = 256,
        predictor_embed_dim: int = 32,
        predictor_depth: int     = 2,
        pad_idx: int             = 0,
        architecture: str        = "ema",
    ):
        super().__init__()
        self.architecture        = architecture
        self.vocab_size          = vocab_size
        self.embed_dim           = embed_dim
        self.num_heads           = num_heads
        self.num_layers          = num_layers
        self.ffn_dim             = ffn_dim
        self.max_seq_len         = max_seq_len
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_depth     = predictor_depth
        self.encoder = EncounterEncoder(
            vocab_size, embed_dim, num_heads, num_layers, max_seq_len, ffn_dim, pad_idx)
        self.predictor = Predictor(
            embed_dim, predictor_embed_dim, max_seq_len, num_heads, predictor_depth)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_tokens   = batch["ctx_tokens"]    # (B, C, T_tok)
        ctx_tok_mask = batch["ctx_tok_mask"]  # (B, C, T_tok)
        ctx_pad_mask = batch["ctx_pad_mask"]  # (B, C)
        tgt_tokens   = batch["tgt_tokens"]    # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"]  # (B, T_tok)
        mask_pos     = batch["mask_pos"]      # (B,)

        # --- Encode context (with grads)
        z_enc = self.encoder(ctx_tokens, ctx_tok_mask, ctx_pad_mask, pool=False) # (B, C, D)

        # -- Predict masked encounters --
        z_pred = self.predictor(z_enc, ctx_pad_mask, mask_pos) # (B, D)

        # -- Target path (no grad)
        with torch.no_grad():
            z_target = self.target_encoder(tgt_tokens, tgt_tok_mask) # (B, D)

        return z_enc, z_pred, z_target

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.encoder.layers
