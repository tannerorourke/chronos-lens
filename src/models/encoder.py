import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoding(nn.Module):
    """Continuous sinusoidal encoding of calendar-day deltas.

    Uses log-spaced periods spanning from 1 day 5 years, ensuring equal representational
    capacity per octave of temporal scale, rather than packing dimensions into the 
    shortest periods.

    Single learnable `time_scale` parameter lets the model stretch or compress
    the entire frequency bank during training, adapting to the actual temporal
    distribution of the cohort without overfitting individual frequencies.
    """

    def __init__(
        self,
        embed_dim: int,
        min_period_days: float = 1.0,
        max_period_days: float = 1825.0,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos pairs"

        half = embed_dim // 2
        periods = min_period_days * (max_period_days / min_period_days) ** (
            torch.arange(half, dtype=torch.float32) / (half - 1)
        )
        freqs = 2.0 * math.pi / periods  # (D/2,)

        self.register_buffer("freqs", freqs)
        self.time_scale = nn.Parameter(torch.ones(()))

    def forward(self, days: torch.Tensor) -> torch.Tensor:
        """days: LongTensor (B,), non-negative integers (days since first admission) --> FloatTensor (B, D)"""
        scaled = days.float() * self.time_scale
        phases = scaled.unsqueeze(-1) * self.freqs # type: ignore # (B, D/2)
        return torch.cat([phases.sin(), phases.cos()], dim=-1) # (B, D)


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


class MHSABlock(nn.Module):
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


class TokenEncoder(nn.Module):
    """Learned [CLS]-token transformer over the tokens within a single encounter.

    Prepended [CLS] token attends over ICD codes and medications (an unordered set
    w/ no positional encoding). final hidden [CLS] state becomes the encounter vector.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 3,
        depth: int = 2,
        ffn_dim: int | None = None,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = embed_dim * 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList([
            MHSABlock(embed_dim, num_heads, ffn_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens_emb: torch.Tensor, # (N, T_tok, D)
        tok_mask: torch.Tensor,   # (N, T_tok) True=real token
    ) -> torch.Tensor:            # (N, D)
        N = tokens_emb.shape[0]

        # Prepend [CLS]
        cls = self.cls_token.expand(N, 1, -1)   # (N, 1, D)
        x = torch.cat([cls, tokens_emb], dim=1) # (N, T_tok+1, D)

        # Build key padding mask (True=padding)
        cls_mask = torch.ones(N, 1, dtype=torch.bool, device=tok_mask.device)
        tok_mask_ext = torch.cat([cls_mask, tok_mask], dim=1) # (N, T_tok+1)
        key_pad = ~tok_mask_ext

        for layer in self.layers:
            x = layer(x, key_pad)
        x = self.norm(x)

        return x[:, 0, :]


class TransformerEncoder(nn.Module):
    """Stack of N MultiHead Self Attn Blocks + sinusoidal temporal encoding."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        ffn_dim: int,
    ):
        super().__init__()
        self.temporal_encoding = TemporalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            MHSABlock(embed_dim, num_heads, ffn_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        pool: bool = False,
    ) -> torch.Tensor:
        # x (B, T, D) -> z (B, D) if pool, (B, T, D) if not
        x = x + self.temporal_encoding(times)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.norm(x)

        if not pool:
            return x # (B, T, D)

        # -- Mean-pool over valid positions (if MLP predictor)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            z = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            z = x.mean(dim=1)

        return z


class EncounterEncoder(nn.Module):
    """Map raw encounter tokens to z in R^d (target encoder deepcopy's this)
       - shared token embedding
       - token-level [CLS] transformer and the encounter-level transformer
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        ffn_dim: int,
        token_enc_heads: int,
        token_enc_depth: int,
        token_enc_ffn_dim: int | None = None,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.token_encoder = TokenEncoder(embed_dim, token_enc_heads, token_enc_depth, token_enc_ffn_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, depth, ffn_dim)

    def forward(
        self,
        tokens: torch.Tensor,                 # (B, C, T_tok)
        tok_mask: torch.Tensor,               # (B, C, T_tok) True=real token
        times: torch.Tensor,                  # (B, C)
        pad_mask: torch.Tensor | None = None, # (B, C) True=padding
        pool: bool = False,
    ) -> torch.Tensor: # -> z (B,D) if pool=True, (B,C,D) if pool=False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1) # -> (B, 1, T_tok)
            tok_mask = tok_mask.unsqueeze(1)

        B, C, T_tok = tokens.shape

        # Flatten encounters: (B*C, T_tok)
        tokens_flat = tokens.reshape(B * C, T_tok)
        tok_mask_flat = tok_mask.reshape(B * C, T_tok)

        # -- token-level attn -> (B*C, D)
        tokens_emb = self.token_embedding(tokens_flat) # (B*C, T_tok, D)
        enc = self.token_encoder(tokens_emb, tok_mask_flat) # (B*C, D)
        enc = enc.view(B, C, -1)

        if times.dim() == 1:
            times = times.unsqueeze(1)
        return self.encoder(enc, times, pad_mask, pool=pool)
