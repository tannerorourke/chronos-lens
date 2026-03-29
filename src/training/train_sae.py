"""
Trains a TopK sparse autoencoder on a JEPA latent vector.
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.sae import SparseAutoencoder
from src.training.utils.checkpoint import load_model_notrain


from src.utils.seed import SEED


# =========================================================================
# Training
# =========================================================================

def train_sae(
    disp_vec: np.ndarray,
    n_features: int = 256,
    top_k: int = 12,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    seed: int = SEED,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Dataset ---
    N, embed_dim = disp_vec.shape
    tensor_disp_vec = torch.tensor(disp_vec, dtype=torch.float32)
    dataset = TensorDataset(tensor_disp_vec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SparseAutoencoder(embed_dim, n_features, top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat, activations = model(batch_x)
            loss = F.mse_loss(x_hat, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        # --- Sanity diagnostics every 10% of training or last epoch ------
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                all_x = tensor_disp_vec.to(device)
                _, all_act = model(all_x)
                all_act_np = all_act.cpu().numpy()

                active_per_sample = (all_act_np != 0).sum(axis=1).mean()
                dead_features = int(((all_act_np != 0).sum(axis=0) == 0).sum())

                # Dictionary diversity: mean pairwise cosine of decoder rows
                W = model.decoder.weight.data  # (embed_dim, n_features)
                D_cols = W / W.norm(dim=0, keepdim=True).clamp(min=1e-10)
                cos_matrix = D_cols.T @ D_cols  # (n_features, n_features)
                n_f = cos_matrix.shape[0]
                if n_f > 1:
                    triu_idx = torch.triu_indices(n_f, n_f, offset=1)
                    mean_cos = cos_matrix[triu_idx[0], triu_idx[1]].mean().item()
                else:
                    mean_cos = 0.0

            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"loss={mean_loss:.6f}  "
                f"active/sample={active_per_sample:.1f}  "
                f"dead={dead_features}/{n_features}  "
                f"dict_cos={mean_cos:.3f}"
            )
            model.train()

    model.eval()
    return model, loss_history


# =========================================================================
# Save results
# =========================================================================

def save_sae_results(
    model: SparseAutoencoder,
    disp_vec: np.ndarray,
    loss_history: list,
    output_dir: Path,
) -> dict:
    """Save SAE checkpoint, dictionary, activations, and loss curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpoint ---
    ckpt_path = output_dir / "sae_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "embed_dim": model.embed_dim,
        "n_features": model.n_features,
        "top_k": model.top_k,
        "loss_history": loss_history,
    }, ckpt_path)

    # --- Dictionary (decoder weights) ---
    # decoder.weight is (embed_dim, n_features); save as (n_features, embed_dim)
    dictionary = model.decoder.weight.data.cpu().numpy().T
    dict_path = output_dir / "sae_dictionary.npy"
    np.save(dict_path, dictionary)

    # --- Activations on full dataset ---
    model.eval()
    with torch.no_grad():
        tensor_disp_vec = torch.tensor(disp_vec, dtype=torch.float32)
        _, activations = model(tensor_disp_vec.to(next(model.parameters()).device))
        act_np = activations.cpu().numpy()
    act_path = output_dir / "sae_activations.npy"
    np.save(act_path, act_np)

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("SAE Training Loss")
    fig.tight_layout()
    loss_fig_path = output_dir / "sae_loss_curve.png"
    fig.savefig(loss_fig_path, dpi=150)
    plt.close(fig)

    # --- Summary JSON ---
    summary = {
        "embed_dim": model.embed_dim,
        "n_features": model.n_features,
        "top_k": model.top_k,
        "n_samples": disp_vec.shape[0],
        "final_loss": loss_history[-1] if loss_history else None,
        "n_epochs": len(loss_history),
        "n_dead_features": int(((act_np != 0).sum(axis=0) == 0).sum()),
        "mean_active_per_sample": float((act_np != 0).sum(axis=1).mean()),
    }
    summary_path = output_dir / "sae_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"SAE results saved → {output_dir}")
    print(f"  checkpoint:   {ckpt_path.name}")
    print(f"  dictionary:   {dict_path.name}  shape={dictionary.shape}")
    print(f"  activations:  {act_path.name}  shape={act_np.shape}")
    print(f"  loss curve:   {loss_fig_path.name}")
    print(f"  summary:      {summary_path.name}")

    return {
        "checkpoint": ckpt_path,
        "dictionary": dict_path,
        "activations": act_path,
        "loss_curve": loss_fig_path,
        "summary": summary_path,
    }
