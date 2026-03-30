import torch
import torch.nn as nn

from src.models.encoder import TransformerEncoderLayer


class MLPPredictor(nn.Module):
    """
    Original lightweight MLP predictor used in early experiments. Requires
    the context encoder to produce a single mean-pooled representation per sample.
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
        pos_emb = self.mask_pos_emb(mask_pos.clamp(max=self.max_seq_len - 1))  # (B, D)
        return self.mlp(torch.cat([z_context, pos_emb], dim=-1))               # (B, D)



class Predictor(nn.Module):
    """
    iJEPA-style transformer predictor: Attends over per-encounter context 
    representations (B, C, D) plus a learnable mask token (at the masked 
    position) to predict the target encounter representation in encoder space.
    """
    def __init__(
        self,
        embed_dim: int,
        predictor_embed_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.max_seq_len         = max_seq_len

        self.input_proj    = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embedding = nn.Embedding(max_seq_len, predictor_embed_dim)
        self.layers        = nn.ModuleList([
            TransformerEncoderLayer(predictor_embed_dim, num_heads, predictor_embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.norm        = nn.LayerNorm(predictor_embed_dim)
        self.output_proj = nn.Linear(predictor_embed_dim, embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        z_enc:        torch.Tensor, # (B, C, D)
        ctx_pad_mask: torch.Tensor, # (B, C) True=padding
        mask_pos:     torch.Tensor, # (B,)
    ) -> torch.Tensor:              # (B, D)
        B, C, _ = z_enc.shape

        # -- Project to predictor dim (B, C, D_pred) --
        h = self.input_proj(z_enc)  # (B, C, D_pred)

        # -- Build context position indices: range(C+1) minus mask_pos per sample
        #    Original sequence had C+1 encounters; context skips the masked one --
        all_pos       = torch.arange(C + 1, device=z_enc.device).unsqueeze(0).expand(B, -1) # (B, C+1)
        keep          = all_pos != mask_pos.unsqueeze(1)
        ctx_positions = all_pos[keep].view(B, C)

        ctx_pos_emb = self.pos_embedding(ctx_positions.clamp(max=self.max_seq_len - 1)) # (B, C, D_pred)
        h = h + ctx_pos_emb

        # -- Mask token with positional embedding for the masked position
        mask_pos_emb = self.pos_embedding(mask_pos.clamp(max=self.max_seq_len - 1)) # (B, D_pred)
        mask_tokens  = self.mask_token.expand(B, 1, -1) + mask_pos_emb.unsqueeze(1) # (B, 1, D_pred)

        # -- Concatenate context encounters + mask token -> (B, C+1, D_pred) --
        h = torch.cat([h, mask_tokens], dim=1)

        # -- Extend key_padding_mask (mask token is never padding)
        mask_tok_pad = torch.zeros(B, 1, dtype=torch.bool, device=z_enc.device)
        key_pad      = torch.cat([ctx_pad_mask, mask_tok_pad], dim=1)  # (B, C+1)

        # -- Run layers and norm
        for layer in self.layers:
            h = layer(h, key_pad)
        h = self.norm(h)

        # -- Extract mask token output (B, 1, D_pred) --
        z_pred_hidden = h[:, C, :]

        # -- Project back to encoder dim --
        return self.output_proj(z_pred_hidden)
