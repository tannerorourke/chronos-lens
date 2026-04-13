import torch
import torch.nn as nn

from src.models.encoder import MHSABlock, TemporalEncoding


class Predictor(nn.Module):
    """
    iJEPA-style transformer predictor: 
    - Attention over per-encounter context representations (B, C, D) 
    - Learnable mask token (at the masked position) to predict the target 
      encounter representation in encoder space.
    
    Encoding:
    - Context tokens receive a
    - Sinusoidal temporal encoding of "days since first admission" for all 
      context AND target tokens
    - STE of the "days since last context encounter" for target tokens, added 
      to the mask positions
    """
    def __init__(
        self,
        embed_dim: int,
        predictor_embed_dim: int,
        predictor_heads: int,
        predictor_depth: int,
        predictor_ffn_dim: int
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim

        self.input_proj        = nn.Linear(embed_dim, predictor_embed_dim)
        self.temporal_encoding = TemporalEncoding(predictor_embed_dim)
        self.relative_encoding = TemporalEncoding(predictor_embed_dim)
        self.layers            = nn.ModuleList([
            MHSABlock(predictor_embed_dim, predictor_heads, predictor_ffn_dim)
            for _ in range(predictor_depth)
        ])
        self.norm        = nn.LayerNorm(predictor_embed_dim)
        
        self.mask_token        = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        self.output_proj = nn.Sequential(
            nn.Linear(predictor_embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim) # no vic-reg
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

        # -- Mask token with absolute + relative temporal encoding -> (B, 1, D_pred)
        rel_to_last_ctx = tgt_times - ctx_times[:, -1]
        # last non-padded index per row
        # last_valid = (~ctx_pad_mask).float().cumsum(dim=1).argmax(dim=1)  # (B,)
        # last_ctx_times = ctx_times.gather(1, last_valid.unsqueeze(1)).squeeze(1)
        # rel_to_last_ctx = tgt_times - last_ctx_times
        mask_tokens = (self.mask_token.expand(B, 1, -1)
                       + self.temporal_encoding(tgt_times).unsqueeze(1)
                       + self.relative_encoding(rel_to_last_ctx).unsqueeze(1))

        # -- append mask token to EOS -> (B, C+1, D_pred)
        h = torch.cat([h, mask_tokens], dim=1)

        # -- Extend key_padding_mask (mask token is never padding)
        mask_tok_pad = torch.zeros(B, 1, dtype=torch.bool, device=z_enc.device)
        key_pad      = torch.cat([ctx_pad_mask, mask_tok_pad], dim=1)  # (B, C+1)

        for layer in self.layers:
            h = layer(h, key_pad)
        h = self.norm(h)

        # -- Extract preds
        z_pred_hidden = h[:, C, :]

        return self.output_proj(z_pred_hidden)
