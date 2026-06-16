"""
Sparse Autoencoder for JEPA vector analysis.

Learns a sparse dictionary of basis directions that reconstruct the chosen
latent vector x = (z_enc, z_pred, z_target).
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

        self.chi = nn.Linear(embed_dim, n_features)
        self.D = nn.Linear(n_features, embed_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + TopK sparsity.  Returns (B, n_features) with <= k nonzeros."""
        h = self.chi(x)
        topk_vals, topk_idx = h.topk(self.top_k, dim=-1)
        sparse = torch.zeros_like(h)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """Decode from sparse activations. Returns (B, embed_dim)."""
        return self.D(sparse)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Returns reconstruction x_hat(chi(x)) < (B, embed_dim) and 
            sparse activations chi(x) (B, n_features) 
        """
        chi_x = self.encode(x)
        x_hat = self.decode(chi_x)
        return x_hat, chi_x
