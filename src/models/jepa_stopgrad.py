import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import EncounterEncoder
from src.models.predictor import Predictor


class JEPAStopGrad(nn.Module):
    """Joint Embedding Predictive Architecture with a stop-gradient variant.

    Both predictor and target paths share the same encoder; the target
    path blocks gradient flow via torch.no_grad() + detach().

    Components
    ----------
    encoder   : EncounterEncoder (shared by both paths)
    predictor : Predictor - attends over (B, C, D) -> z_pred (B, D)
    
    Returns
    -------
    z_enc: (B, D) encoded representation
    z_pred: (B, D) predictor output
    z_target: (B, D) stop-grad target
    z_target_nograd: (B, D) stop-gradtarget (no grad)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int           = 128,
        encoder_heads: int       = 8,
        encoder_depth: int       = 4,
        encoder_ffn_dim: int     = 256,
        token_enc_heads: int     = 2,
        token_enc_depth: int     = 2,
        token_enc_ffn_dim: int | None = None,
        predictor_embed_dim: int = 64,
        predictor_heads: int     = 2,
        predictor_depth: int     = 2,
        architecture: str        = "stopgrad",
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
        
        self.encoder = EncounterEncoder(vocab_size, embed_dim,
            encoder_heads, encoder_depth, encoder_ffn_dim,
            token_enc_heads, token_enc_depth, token_enc_ffn_dim,
            pad_idx=0
        )
        self.predictor = Predictor(embed_dim,
            predictor_embed_dim, predictor_heads, predictor_depth
        )
        # No target encoder is created - predictor and target paths share the same encoder. 
        # The target path blocks gradient flow via torch.no_grad() + detach() for full symmetry.

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        # -- Target path (with grad) for symmetric VICReg -> (B, D)
        z_target = self.encoder(tgt_tokens, tgt_tok_mask, tgt_times, pool=True)
        
        # Target path stop-grad, for sim loss
        z_target_nograd = z_target.detach()

        return z_enc, z_pred, z_target, z_target_nograd

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.encoder.layers
