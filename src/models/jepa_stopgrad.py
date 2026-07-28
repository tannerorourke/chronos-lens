"""
Joint Embedding Predictive Architecture with a stop-gradient on the target.

Both predictor and target paths share the same encoder; the target
path blocks gradient flow via torch.no_grad() + detach() for full symmetry.

Components
----------
encoder   : EncounterEncoder (shared by both paths)
predictor : Predictor - attends over (B, C, D) -> z_pred (B, D)
"""
    
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import EncounterEncoder
from src.models.predictor import Predictor
from src.training.utils.vicreg import Projector


class JEPAStopGrad(nn.Module):
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
        architecture: str               = "stopgrad",
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
        # No target encoder is created

    def encode(self, batch: dict) -> torch.Tensor:
        """Per-encounter context representation (B, C, D), the z_enc that
        forward returns first. Used for analysis/extraction.
        """
        return self.encoder(
            batch["ctx_tokens"], batch["ctx_tok_mask"],
            batch["ctx_times"], batch["ctx_pad_mask"], pool=False)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx_pad_mask = batch["ctx_pad_mask"] # (B, C)
        ctx_times    = batch["ctx_times"]    # (B, C)
        tgt_tokens   = batch["tgt_tokens"]   # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"] # (B, T_tok)
        tgt_times    = batch["tgt_times"]    # (B,)

        # --- Encode context -> (B, C, D)
        z_enc = self.encode(batch)
        
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
