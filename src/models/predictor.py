"""Since we're dealing with sequences, not spatial grid patches, we diverge
   from vision variants and take a per-encounter context representation (B, C, D),
   project to predictor dim, add positional embeddings, concatenate a learnable
   mask token at the target position with its positional embedding, run through
   transformer blocks, extract the mask token output, and project back to encoder dim.
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, embed_dim: int, predictor_hidden: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mask_pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, embed_dim),
        )

    def forward(self, z_context: torch.Tensor, mask_pos: torch.Tensor) -> torch.Tensor:
        pos_emb = self.mask_pos_emb(mask_pos.clamp(max=self.max_seq_len - 1))  # (B, D)
        return self.mlp(torch.cat([z_context, pos_emb], dim=-1))               # (B, D)
