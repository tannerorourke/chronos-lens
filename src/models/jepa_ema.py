import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import EncounterEncoder
from src.models.predictor import Predictor
from src.training.utils.vicreg import Projector


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
    z_enc: (B, D) encoded representation
    z_pred: (B, D) predictor output
    z_target: (B, D) EMA target encoding
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int                  = 128,
        encoder_heads: int              = 8,
        encoder_depth: int              = 4,
        encoder_ffn_dim: int            = 256,
        token_enc_heads: int            = 2,
        token_enc_depth: int            = 2,
        token_enc_ffn_dim: int | None   = None,
        predictor_embed_dim: int        = 64,
        predictor_heads: int            = 2,
        predictor_depth: int            = 2,
        predictor_ffn_dim: int          = 128,
        projector_dim: int | None       = None,
        architecture: str               = "ema",
    ):
        super().__init__()
        self.architecture        = architecture
        self.vocab_size          = vocab_size
        self.embed_dim           = embed_dim
        self.encoder_heads       = encoder_heads
        self.encoder_depth       = encoder_depth
        self.encoder_ffn_dim     = encoder_ffn_dim
        self.token_enc_heads     = token_enc_heads
        self.token_enc_depth     = token_enc_depth
        self.token_enc_ffn_dim   = token_enc_ffn_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_heads     = predictor_heads
        self.predictor_depth     = predictor_depth
        _proj_dim = projector_dim if projector_dim is not None else 2*embed_dim
        self.projector = Projector(embed_dim, _proj_dim, _proj_dim)
        
        self.encoder = EncounterEncoder(vocab_size, embed_dim, 
            encoder_heads, encoder_depth, encoder_ffn_dim,
            token_enc_heads, token_enc_depth, token_enc_ffn_dim,
            pad_idx=0
        )
        self.predictor = Predictor(embed_dim,
            predictor_embed_dim, predictor_heads, predictor_depth, predictor_ffn_dim
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
    
    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_tokens   = batch["ctx_tokens"]   # (B, C, T_tok)
        ctx_tok_mask = batch["ctx_tok_mask"] # (B, C, T_tok)
        ctx_pad_mask = batch["ctx_pad_mask"] # (B, C)
        ctx_times    = batch["ctx_times"]    # (B, C)
        tgt_tokens   = batch["tgt_tokens"]   # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"] # (B, T_tok)
        tgt_times    = batch["tgt_times"]    # (B,)

        # --- Encode context -> (B, C, D)
        z_enc = self.encoder(ctx_tokens, ctx_tok_mask, ctx_times, ctx_pad_mask, pool=False)

        # -- Predict masked encounters -> (B, D)
        z_pred = self.predictor(z_enc, ctx_times, tgt_times, ctx_pad_mask)

        # -- Target path (no grad) -> (B, D)
        with torch.no_grad():
            z_target = self.target_encoder(tgt_tokens, tgt_tok_mask, tgt_times, pool=True) 

        return z_enc, z_pred, z_target

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.encoder.layers
