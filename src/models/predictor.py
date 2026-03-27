"""MLP predictor module for JEPA variants.

Maps (z_context, mask_position) to z_pred via learned positional
embedding concatenated with the context representation.
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """MLP predictor: (z_context || pos_emb) -> z_pred.

    Parameters
    ----------
    embed_dim        : model dimension (input and output)
    predictor_hidden : hidden layer width
    max_seq_len      : maximum sequence length (for positional embedding)
    """

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
        """
        Parameters
        ----------
        z_context : (B, D) context encoder output
        mask_pos  : (B,) integer position of the masked target encounter

        Returns
        -------
        z_pred : (B, D) predicted target representation
        """
        pos_emb = self.mask_pos_emb(mask_pos.clamp(max=self.max_seq_len - 1))  # (B, D)
        return self.mlp(torch.cat([z_context, pos_emb], dim=-1))               # (B, D)
