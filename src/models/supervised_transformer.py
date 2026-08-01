"""Supervised Transformer baseline.

Same encoder architecture as the JEPA variants (EncounterEncoder), but
trained with explicit label supervision (BCEWithLogitsLoss) instead of
self-supervised prediction.  Provides a representation-quality baseline
for comparing JEPA embeddings.

Components
----------
- encoder: EncounterEncoder - identical to JEPA context path
- classifier: nn.Linear(embed_dim, 1) - binary logit head
"""

import torch
import torch.nn as nn

from src.models.encoder import EncounterEncoder


class SupervisedTransformer(nn.Module):
    """Supervised encoder + linear classifier.

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

    def encode(self, batch: dict) -> torch.Tensor:
        """Per-encounter context representation (B, C, D), identical in
        shape/semantics to the JEPA z_enc. Used for analysis/extraction.
        Supervised model makes a prediction on the next encounter based
        on this representation.
        """
        return self.encoder(
            batch["ctx_tokens"], batch["ctx_tok_mask"],
            batch["ctx_times"], batch["ctx_pad_mask"])

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode context, then classify the most recent encounter's vector.

        Returns per-encounter representation 'z_enc', and logits for loss.
        'mask_pos' is the target index 'k', so 'mask_pos - 1'
        is the last valid context slot (context is the chronological prefix,
        right-padded with no interior gaps).
        """
        z_enc = self.encode(batch)                  # (B, C, D)
        last_idx = (batch["mask_pos"] - 1).long()   # (B,)
        rows = torch.arange(z_enc.size(0), device=z_enc.device)
        z_last = z_enc[rows, last_idx]              # (B, D) recency readout
        logits = self.classifier(z_last).squeeze(-1)
        return z_enc, logits

    @property
    def transformer_layers(self):
        """Convenience access to transformer layers for probing hooks."""
        return self.encoder.encoder.layers
