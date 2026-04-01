"""
Sparse Autoencoder for JEPA vector analysis.

Learns a sparse dictionary of basis directions that reconstruct the chosen
latent vector (z_enc, z_pred, z_target, or derived). Each direction is a
"feature" discovered from the geometry itself.  Sparsity enforced via 
TopK activation (keep top-k, zero the rest; no L1 penalty needed).

Architecture
------------
Encoder : Linear(embed_dim -> n_features) -> top k activation
Decoder : Linear(n_features -> embed_dim) - decoder rows ARE the dictionary

Loss
----
||x - x_hat||^2 (reconstruction MSE)
"""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """TopK sparse autoencoder.

    Parameters
    ----------
    embed_dim  : input dimension (= JEPA embed_dim)
    n_features : dictionary size (overcomplete, ~4x embed_dim)
    top_k      : number of active features per sample
    """

    def __init__(
        self, 
        embed_dim: int, 
        n_features: int, 
        top_k: int
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.top_k = top_k

        self.encoder = nn.Linear(embed_dim, n_features)
        self.decoder = nn.Linear(n_features, embed_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + TopK sparsity.  Returns (B, n_features) with <= k nonzeros."""
        h = self.encoder(x) # (B, n_features)
        topk_vals, topk_idx = h.topk(self.top_k, dim=-1)
        sparse = torch.zeros_like(h)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """Decode from sparse activations. Returns (B, embed_dim)."""
        return self.decoder(sparse)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Returns reconstruction x_hat (B, embed_dim) and sparse activations (B, n_features) """
        activations = self.encode(x)
        x_hat = self.decode(activations)
        return x_hat, activations
