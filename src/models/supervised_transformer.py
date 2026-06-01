"""Supervised Transformer baseline.

Same encoder architecture as the JEPA variants (EncounterEncoder), but
trained with explicit label supervision (BCEWithLogitsLoss) instead of
self-supervised prediction.  Provides a representation-quality baseline
for comparing JEPA embeddings.
"""

import torch
import torch.nn as nn

from src.models.encoder import EncounterEncoder


class SupervisedTransformer(nn.Module):
    """Supervised encoder + linear classifier.

    Components
    ----------
    - encoder: EncounterEncoder - identical to JEPA context path
    - classifier: nn.Linear(embed_dim, 1) - binary logit head
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 64,
        encoder_heads:   int = 2,
        encoder_depth:  int = 2,
        encoder_ffn_dim:     int = 256,
        token_enc_heads: int = 3,
        token_enc_depth: int = 2,
        token_enc_ffn_dim: int | None = None,
        pad_idx:     int = 0,
        architecture: str = "supervised",
    ):
        super().__init__()
        self.architecture    = architecture
        self.vocab_size      = vocab_size
        self.embed_dim       = embed_dim
        self.encoder_heads   = encoder_heads
        self.encoder_depth   = encoder_depth
        self.encoder_ffn_dim = encoder_ffn_dim

        self.encoder = EncounterEncoder(
            vocab_size, embed_dim, encoder_heads, encoder_depth, encoder_ffn_dim,
            token_enc_heads, token_enc_depth, token_enc_ffn_dim, pad_idx=pad_idx)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens per encounter via [CLS], then encode encounter sequence."""
        tokens   = batch["ctx_tokens"]     # (B, C, T_tok)
        tok_mask = batch["ctx_tok_mask"]   # (B, C, T_tok)
        pad_mask = batch["ctx_pad_mask"]   # (B, C)
        times    = batch["ctx_times"]      # (B, C)

        z_enc_pooled = self.encoder(tokens, tok_mask, times, pad_mask, pool=True)
        logits = self.classifier(z_enc_pooled).squeeze(-1)
        return z_enc_pooled, logits

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.encoder.layers
