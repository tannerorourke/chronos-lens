import torch
import torch.nn as nn

from src.models.sequential_jepa import embed_and_pool, TransformerEncoder


class JEPAStopGrad(nn.Module):
    """JEPA variant that replaces the EMA target encoder with a shared
    encoder + stop-gradient.  Both context and target paths use the same
    token_embedding and context_encoder; the target path simply blocks
    gradient flow via torch.no_grad() + detach().

    forward() returns (z_context, z_pred, z_target) with the same shapes
    as JEPA so all downstream code works unchanged.
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
        pad_idx:          int = 0,
        **kwargs # absorb saving architecture and use_bfloat16 in model params
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.context_encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers, max_seq_len, ffn_dim)

        # Positional embedding for the mask (target) position
        self.mask_pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # Predictor: (z_context || pos_emb) -> z_pred
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, predictor_hidden),
            nn.GELU(),
            nn.Linear(predictor_hidden, embed_dim),
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_context : (B, D) context representation (differentiable)
        z_pred    : (B, D) predictor output       (differentiable)
        z_target  : (B, D) stop-grad target       (no grad)
        """
        ctx_tokens   = batch["ctx_tokens"]    # (B, C, T_tok)
        ctx_tok_mask = batch["ctx_tok_mask"]   # (B, C, T_tok)
        ctx_pad_mask = batch["ctx_pad_mask"]   # (B, C)
        tgt_tokens   = batch["tgt_tokens"]     # (B, T_tok)
        tgt_tok_mask = batch["tgt_tok_mask"]   # (B, T_tok)
        mask_pos     = batch["mask_pos"]       # (B,)

        B, C, T_tok = ctx_tokens.shape

        # -- Context path (with grads) ------------------------------------
        ctx_embs = embed_and_pool(
            self.token_embedding,
            ctx_tokens.view(B * C, T_tok),
            ctx_tok_mask.view(B * C, T_tok),
        ).view(B, C, self.embed_dim)

        z_context = self.context_encoder(ctx_embs, ctx_pad_mask)  # (B, D)

        # -- Target path (shared weights, stop-gradient) ------------------
        with torch.no_grad():
            tgt_embs = embed_and_pool(
                self.token_embedding, tgt_tokens, tgt_tok_mask
            ).unsqueeze(1)                                         # (B, 1, D)
            z_target = self.context_encoder(tgt_embs).detach()     # (B, D)

        # -- Predictor ----------------------------------------------------
        mask_pos_clamped = mask_pos.clamp(max=self.max_seq_len - 1)
        pos_emb = self.mask_pos_emb(mask_pos_clamped)                      # (B, D)
        z_pred  = self.predictor(torch.cat([z_context, pos_emb], dim=-1))  # (B, D)

        return z_context, z_pred, z_target
