"""
Shared encoder, used by all JEPA variants and supervised transformer.

Token embedding, transformer encoder, and EncounterEncoder that wraps
them into a single module suitable for deepcopy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def embed_and_pool(
    embedding: nn.Embedding,
    tokens: torch.Tensor,       # (..., max_tok)
    tok_mask: torch.Tensor,     # True=real
) -> torch.Tensor:              # (..., embed_dim)
    """Embed tokens then mean-pool over the token dimension, ignoring padding."""
    emb  = embedding(tokens)
    mask = tok_mask.float().unsqueeze(-1)
    return (emb * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1.0)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        Q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, Dh).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), # (B,1,1,T)
                float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        
        # -- guard fully-padded rows
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, V) # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer:
       x -> LN -> MHSA -> residual -> LN -> FFN (GELU) -> residual
    """

    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ffn_dim: int
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of N TransformerEncoderLayers + learned positional embedding."""

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        num_layers:  int,
        max_seq_len: int,
        ffn_dim:     int,
    ):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,    # (B, T, D)
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:      # (B, D)
        # -- Flatten batch to sequence vector (1, T)
        B, T, _ = x.shape
        max_pos = self.pos_embedding.num_embeddings - 1
        positions = torch.arange(T, device=x.device).clamp(max=max_pos).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.norm(x)

        # -- Mean-pool over valid positions
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            z = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            z = x.mean(dim=1)

        return z


class EncounterEncoder(nn.Module):
    """Maps raw encounter tokens to z in R^d.  Wraps both the embedding table
       and the transformer so deepcopy copies everything together.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        num_heads: int,
        num_layers: int, 
        max_seq_len: int, 
        ffn_dim: int,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers, max_seq_len, ffn_dim)

    def forward(
        self,
        tokens: torch.Tensor,   # (..., T_tok)
        tok_mask: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:          # (B, D)
        """
        Parameters
        ----------
        tokens   : (..., T_tok) LongTensor — (B, C, T_tok) multi-encounter or (B, T_tok) single
        tok_mask : (..., T_tok) BoolTensor  True=real token
        pad_mask : (B, C) BoolTensor True=padding, or None for single encounter

        Returns
        -------
        z : (B, D) sequence-level representation
        """
        emb = embed_and_pool(self.token_embedding, tokens, tok_mask)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)   # (B, D) -> (B, 1, D) for single encounter
        return self.encoder(emb, pad_mask)
