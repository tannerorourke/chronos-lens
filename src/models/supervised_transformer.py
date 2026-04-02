"""Supervised Transformer baseline.

Same encoder architecture as the JEPA variants, but trained with explicit 
label supervision (BCEWithLogitsLoss) instead of self-supervised prediction.
Provides a representation-quality baseline for comparing JEPA embeddings.
"""

import torch
import torch.nn as nn

from src.models.encoder import TransformerEncoder, embed_and_pool


class SupervisedTransformer(nn.Module):
    """Supervised encoder + linear classifier.

    Components
    ----------
    token_embedding : nn.Embedding - shared token table
    encoder         : TransformerEncoder - same architecture as JEPA
    classifier      : nn.Linear(embed_dim, 1) - binary logit head
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 64,
        num_heads:   int = 2,
        num_layers:  int = 2,
        max_seq_len: int = 256,
        ffn_dim:     int = 256,
        pad_idx:     int = 0,
        architecture: str = "supervised",
    ):
        super().__init__()
        self.architecture = architecture
        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.num_layers   = num_layers
        self.ffn_dim      = ffn_dim
        self.max_seq_len  = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers, max_seq_len, ffn_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean pool and encode the full encounter sequence and classify."""
        tokens   = batch["ctx_tokens"]
        tok_mask = batch["ctx_tok_mask"]
        pad_mask = batch["ctx_pad_mask"]

        # embed and pool (B, E, T_tok) to (B, E, D)
        enc_repr = embed_and_pool(self.token_embedding, tokens, tok_mask) 
        if enc_repr.dim() == 2:
            enc_repr = enc_repr.unsqueeze(1)

        # run transformer encoder (B, E, D) to (B, D)
        z_enc_pooled = self.encoder(enc_repr, key_padding_mask=pad_mask) 

        logits = self.classifier(z_enc_pooled).squeeze(-1)
        return z_enc_pooled, logits

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.layers
