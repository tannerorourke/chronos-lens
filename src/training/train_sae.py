"""
Trains a TopK sparse autoencoder on a JEPA latent vector.
"""

import gc
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.models.sae import SparseAutoencoder
from src.infra.inference import load_embeddings_for_analysis
from src.analysis.sae import extract_activations
from src.utils.io import save_json
from src.training.utils.logging import plot_loss_curve



def save_sae_results(
    model: SparseAutoencoder,
    data_vec: np.ndarray,
    loss_history: list,
    sae_exp_dir: Path,
    device: torch.device
):
    """Save SAE checkpoint, dictionary, activations, and loss curve"""
    # -- Checkpoint
    ckpt_path = sae_exp_dir / "sae.pt"
    torch.save({
        "model": model.state_dict(),
        "model_params": {
            "embed_dim": model.embed_dim,
            "n_features": model.n_features,
            "top_k": model.top_k,
        },
        "loss_history": loss_history,
    }, ckpt_path)

    # -- Dictionary (decoder weights)
    # decoder.weight is (embed_dim, n_features); save as (n_features, embed_dim)
    dictionary = model.decoder.weight.data.cpu().numpy().T
    dict_path = sae_exp_dir / "decoder_weights.npy"
    np.save(dict_path, dictionary)

    # -- Activations on full dataset
    with torch.no_grad():
        x = torch.tensor(data_vec, dtype=torch.float32, device=device)
        _, act = model(x)
    act_np = act.cpu().numpy()
    np.save(sae_exp_dir / "activations.npy", act_np)

    # -- Loss curve
    plot_loss_curve(loss_history, save_path=sae_exp_dir / "sae_loss_curve",
                    title="SAE Training Loss")

    # -- Summary JSON
    summary_path = sae_exp_dir / "sae_summary.json"
    save_json({
        "embed_dim": model.embed_dim,
        "n_features": model.n_features,
        "top_k": model.top_k,
        "n_samples": data_vec.shape[0],
        "final_loss": loss_history[-1] if loss_history else None,
        "n_epochs": len(loss_history),
        "n_dead_features": int(((act_np != 0).sum(axis=0) == 0).sum()),
        "mean_active_per_sample": float((act_np != 0).sum(axis=1).mean()),
    }, summary_path)

    logger.info(f"SAE results saved in {sae_exp_dir}")

# =========================================================================
# Training
# =========================================================================

def main(
    sae_params: dict,
    sae_exp_dir: Path,
    target: str,
    embeddings: str,
    device: torch.device,
) -> None:
    base_model_dir = sae_exp_dir.parent
    
    logger.info("Setting up SAE training..")
    logger.info(f"  Target: {target}")
    n_features  = sae_params["n_features"]
    top_k       = sae_params["top_k"]
    epochs      = sae_params["epochs"]
    lr          = float(sae_params["lr"])
    batch_size  = sae_params["batch_size"]
    
    # --- Load target vector into tensor dataset
    logger.info(f"Loading embedding vec to reconstruct from {embeddings}")
    emb_stream, _ = load_embeddings_for_analysis(base_model_dir.name, name=embeddings, device=device)
    with emb_stream as emb:
        if target == "z_enc":
            ctx_pad_mask = emb["ctx_pad_mask"].astype(bool)
            data_vec = emb["z_encs"][~ctx_pad_mask].astype(np.float64)
        elif target == "pred_error":
            data_vec = (emb["z_pred"] - emb["z_target"]).astype(np.float64)
        else:
            data_vec = emb[target].astype(np.float64)
            
    dv_N, dv_D = data_vec.shape
    logger.info(f"    N={dv_N} samples, D={dv_D} embed_dim")
    
    logger.info("Initializing all the parts..")
    tensor_data_vec = torch.tensor(data_vec, dtype=torch.float32)
    ds = TensorDataset(tensor_data_vec)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    del ds; gc.collect()
    
    model = SparseAutoencoder(dv_D, n_features, top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Training loop
    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_hat, _ = model(batch_x)
            loss = F.mse_loss(x_hat, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        # --- Sanity diagnostics every 10% of training or last epoch
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                all_x = tensor_data_vec.to(device)
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

            logger.info(
                f"[Epoch {epoch:4d}/{epochs}]  "
                f"  loss={mean_loss:.6f}  "
                f"  active/sample={active_per_sample:.1f}  "
                f"  dead={dead_features}/{n_features}  "
                f"  dict_cos={mean_cos:.3f}"
            )
            model.train()

    logger.info("\nTraining Complete!")
    model.eval()
    save_sae_results(model, data_vec, loss_history, sae_exp_dir, device)
