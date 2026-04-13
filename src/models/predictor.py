import torch
import torch.nn as nn

from src.models.encoder import MHSABlock, TemporalEncoding


class Predictor(nn.Module):
    """
    iJEPA-style transformer predictor: 
    - Attention over per-encounter context representations (B, C, D) 
    - Learnable mask token (at the masked position) to predict the target 
      encounter representation in encoder space.
    - Sinusoidal temporal encoding of admittime_days (days since first 
      admission), no positional embedding.
    """
    def __init__(
        self,
        embed_dim: int,
        predictor_embed_dim: int,
        predictor_heads: int,
        predictor_depth: int,
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim

        self.input_proj        = nn.Linear(embed_dim, predictor_embed_dim)
        self.temporal_encoding = TemporalEncoding(predictor_embed_dim)
        self.layers            = nn.ModuleList([
            MHSABlock(predictor_embed_dim, predictor_heads, predictor_embed_dim * 4)
            for _ in range(predictor_depth)
        ])
        self.norm        = nn.LayerNorm(predictor_embed_dim)
        
        self.mask_token        = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.output_proj = nn.Sequential(
            nn.Linear(predictor_embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim) # ensure output is on unit sphere-ish
        )

        

    def forward(
        self,
        z_enc:        torch.Tensor, # (B, C, D)
        ctx_times:    torch.Tensor, # (B, C)
        tgt_times:    torch.Tensor, # (B,)
        ctx_pad_mask: torch.Tensor, # (B, C)
    ) -> torch.Tensor:
        B, C, _ = z_enc.shape

        # -- Project to predictor dim (B, C, D_pred)
        h = self.input_proj(z_enc)

        # -- Temporal encoding for context encounters
        h = h + self.temporal_encoding(ctx_times)  # (B, C, D_pred)

        # -- Mask token with temporal encoding for the target position
        mask_time_emb = self.temporal_encoding(tgt_times)       # (B, D_pred)
        mask_tokens   = self.mask_token.expand(B, 1, -1) + mask_time_emb.unsqueeze(1)  # (B, 1, D_pred)

        # -- append mask token to EOS -> (B, C+1, D_pred)
        h = torch.cat([h, mask_tokens], dim=1)

        # -- Extend key_padding_mask (mask token is never padding)
        mask_tok_pad = torch.zeros(B, 1, dtype=torch.bool, device=z_enc.device)
        key_pad      = torch.cat([ctx_pad_mask, mask_tok_pad], dim=1)  # (B, C+1)

        for layer in self.layers:
            h = layer(h, key_pad)
        h = self.norm(h)

        z_pred_hidden = h[:, C, :]

        # -- Project back to encoder dim
        return self.output_proj(z_pred_hidden)
