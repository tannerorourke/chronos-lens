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
    z_enc : (B, D) encoded representation (differentiable)
    z_pred    : (B, D) predictor output (differentiable)
    z_target  : (B, D) stop-grad target (no grad)
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
        architecture: str        = "stopgrad",
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
        # No target encoder is created - predictor and target paths share the same encoder. 
        # The target path blocks gradient flow via torch.no_grad() + detach()

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_tokens   = batch["ctx_tokens"]    # (B, C, T_tok)
        ctx_tok_mask = batch["ctx_tok_mask"]  # (B, C, T_tok)
        ctx_pad_mask = batch["ctx_pad_mask"]  # (B, C)
        tgt_tokens   = batch["tgt_tokens"]    # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"]  # (B, T_tok)
        mask_pos     = batch["mask_pos"]      # (B,)

        z_enc = self.encoder(ctx_tokens, ctx_tok_mask, ctx_pad_mask, pool=False) # (B, C, D)
        z_pred = self.predictor(z_enc, ctx_pad_mask, mask_pos) # (B, D)
        
        # Target path WITH grad (for symmetric VICReg)
        z_target = self.encoder(tgt_tokens, tgt_tok_mask)  # (B, D)
        
        # Target path STOP-grad (for sim loss)
        z_target_sg = z_target.detach()

        return z_enc, z_pred, z_target, z_target_sg

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.encoder.layers
