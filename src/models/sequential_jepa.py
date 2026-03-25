import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



# =============================================================================
# Embedding utility
# =============================================================================

def embed_and_pool(
    embedding: nn.Embedding,
    tokens: torch.Tensor,       # (..., max_tok)  LongTensor
    tok_mask: torch.Tensor,     # (..., max_tok)  BoolTensor  True=real
) -> torch.Tensor:              # (..., embed_dim)
    """Embed tokens then mean-pool over the token dimension, ignoring padding."""
    emb  = embedding(tokens)                # (..., max_tok, D)
    mask = tok_mask.float().unsqueeze(-1)   # (..., max_tok, 1)
    return (emb * mask).sum(dim=-2) / mask.sum(dim=-2).clamp(min=1.0)


# =============================================================================
# JEPA model
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

    Parameters
    ----------
    embed_dim : total model dimension (must be divisible by num_heads)
    num_heads : number of attention heads
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Simpler to implement as separate linear layers for q, k, v than a single big one
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        Q = self.q(x).view(B, T, H, Dh).transpose(1, 2)
        K = self.k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.v(x).view(B, T, H, Dh).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # Mask out padded key positions, broadcast over (H, T_query)
            # (B, T) True=ignore
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B,1,1,T)
                float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)      # guard fully-padded rows

        out = torch.matmul(attn, V)                 # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer:
      x -> LN -> MHSA -> residual -> LN -> FFN (GELU) -> residual
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, 
                x: torch.Tensor, 
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """ Stack of N TransformerEncoderLayers + learned positional embedding """

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
        x: torch.Tensor,                              # (B, T, D)
        key_padding_mask: torch.Tensor | None = None, # (B, T) True=padding
    ) -> torch.Tensor:                                # (B, D)
        """
        Returns:
            single sequence-level vector (mean-pool over valid positions), i.e. shape (B, embed_dim).
        """
        B, T, _ = x.shape
        max_pos = self.pos_embedding.num_embeddings - 1
        positions = torch.arange(T, device=x.device).clamp(max=max_pos).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.norm(x)

        # Mean-pool over valid (non-padded) positions
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)   # (B, T, 1)
            z = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            z = x.mean(dim=1)

        return z                                                 # (B, D)


class JEPA(nn.Module):
    """Joint Embedding Predictive Architecture for patient sequences.

    Components
    ----------
    token_embedding : shared Embedding table (ICD codes + med names)
    context_encoder : TransformerEncoder trained with gradient descent
    target_encoder  : EMA copy of context_encoder, no gradients
    mask_pos_emb    : learnable position embedding fed to predictor
    predictor       : MLP(z_context (outer) pos_emb) -> z_pred
    """

    def __init__(
        self,
        vocab_size:       int,
        embed_dim:        int = 64,
        num_heads:        int = 2,
        num_layers:       int = 2,
        max_seq_len:      int = 256,
        ffn_dim:          int = 256,
        predictor_hidden: int = 128,
        tau:              float = 0.996,
        pad_idx:          int = 0,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.tau         = tau
        self.max_seq_len = max_seq_len

        # Online token embedding (context path; target gets its own EMA copy)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.context_encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers, max_seq_len, ffn_dim)

        # Target encoder: EMA shadow of context_encoder. No backprop.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Target token embedding: EMA shadow of token_embedding. No backprop.
        self.target_token_embedding = copy.deepcopy(self.token_embedding)
        for p in self.target_token_embedding.parameters():
            p.requires_grad_(False)

        # Positional embedding for the mask (target) position
        self.mask_pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # Predictor: (z_context || pos_emb) -> z_pred
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, embed_dim),
        )

    @torch.no_grad()
    def update_target_encoder(self):
        """Exponential moving average update of target encoder and embedding weights."""
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(self.tau).add_((1.0 - self.tau) * p_ctx.data)
        for p_ctx, p_tgt in zip(
            self.token_embedding.parameters(),
            self.target_token_embedding.parameters(),
        ):
            p_tgt.data.mul_(self.tau).add_((1.0 - self.tau) * p_ctx.data)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_context : (B, D) context representation (differentiable)
        z_pred    : (B, D) predictor output       (differentiable)
        z_target  : (B, D) EMA target encoding    (no grad)
        """
        ctx_tokens   = batch["ctx_tokens"]    # (B, C, T_tok)
        ctx_tok_mask = batch["ctx_tok_mask"]  # (B, C, T_tok)
        ctx_pad_mask = batch["ctx_pad_mask"]  # (B, C)
        tgt_tokens   = batch["tgt_tokens"]    # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"]  # (B, T_tok)
        mask_pos     = batch["mask_pos"]      # (B,)

        B, C, T_tok = ctx_tokens.shape

        # -- Context path (with grads) --------------------------------
        # Embed and mean-pool each context encounter: (B, C, D)
        ctx_embs = embed_and_pool(
            self.token_embedding,
            ctx_tokens.view(B * C, T_tok),
            ctx_tok_mask.view(B * C, T_tok),
        ).view(B, C, self.embed_dim)

        z_context = self.context_encoder(ctx_embs, ctx_pad_mask)  # (B, D)

        # -- Target path (no grads - EMA encoder) ---------------------
        with torch.no_grad():
            # (B, 1, D) single encounter as target — use EMA embedding
            tgt_embs = embed_and_pool(self.target_token_embedding, tgt_tokens, tgt_tok_mask).unsqueeze(1)
            z_target = self.target_encoder(tgt_embs)  # (B, D)  single encounter, no padding

        # -- Predictor ----------------------------------------------------
        # Clamp mask_pos to stay within positional embedding capacity
        mask_pos_clamped = mask_pos.clamp(max=self.max_seq_len - 1)
        pos_emb = self.mask_pos_emb(mask_pos_clamped)                     # (B, D)
        z_pred  = self.predictor(torch.cat([z_context, pos_emb], dim=-1)) # (B, D)

        return z_context, z_pred, z_target