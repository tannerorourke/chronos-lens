"""
Trains a TopK sparse autoencoder on a JEPA latent vector.
"""

from pathlib import Path

import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.models.sae import SparseAutoencoder
from src.utils.io import load_embeddings, save_json
from src.analysis.plotting import plot_loss_curve

# =========================================================================
# Training
# =========================================================================

def main(
    params: dict,
    exp_dir: Path,
    target: str,
    embeddings: str,
    device: torch.device,
) -> None:
    print(f"Configuring SAE for {target} :)")
    
    # --- Locate embeddings file
    emb_npz, emb_path = load_embeddings(exp_dir, embeddings)
    print(f"  Embeddings: {emb_path.name}")
    
    # -- get output directory
    output_dir = exp_dir / f"sae_{target}"
    i = 0
    if output_dir.exists(): i = 2
    while output_dir.exists():
        output_dir = exp_dir / f"sae_{target}_v{i}"; i += 1
    print(f"  Output directory: {'/'.join(output_dir.parts[-3:])}")

    # --- Load target vector
    if target == "z_enc":
        z_encs = emb_npz["z_encs"]                              # (N, C, D)
        ctx_pad_mask = emb_npz["ctx_pad_mask"].astype(bool)    # (N, C)
        data_vec = z_encs[~ctx_pad_mask].astype(np.float64)     # (N_valid, D)
        print(f"  Flattened z_encs: {z_encs.shape} -> {data_vec.shape} valid encounters")
    elif target == "pred_error":
        data_vec = (emb_npz["z_pred"] - emb_npz["z_target"]).astype(np.float64)
        print(f"  Computed pred_error = z_pred - z_target")
    else:
        data_vec = emb_npz[target].astype(np.float64)
    N, D = data_vec.shape
    
    print("  Training config:")
    print(f"    N={N} samples, D={D} embed_dim")
    for k, v in params.items():
        print(f"    {k}: {v}")
    
    n_features  = params["n_features"]
    top_k       = params["top_k"]
    epochs      = params["epochs"]
    lr          = params["lr"]
    batch_size  = params["batch_size"]

    # --- Dataset
    N, embed_dim = data_vec.shape
    tensor_data_vec = torch.tensor(data_vec, dtype=torch.float32)
    dataset = TensorDataset(tensor_data_vec)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # --- Model
    model = SparseAutoencoder(embed_dim, n_features, top_k).to(device)
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

            print(
                f"[Epoch {epoch:4d}/{epochs}]  "
                f"  loss={mean_loss:.6f}  "
                f"  active/sample={active_per_sample:.1f}  "
                f"  dead={dead_features}/{n_features}  "
                f"  dict_cos={mean_cos:.3f}"
            )
            model.train()

    print("\nTraining Complete.")
    print("=" * 60)
    save_sae_results(model, data_vec, loss_history, output_dir)


# =========================================================================
# Save results
# =========================================================================

def save_sae_results(
    model: SparseAutoencoder,
    disp_vec: np.ndarray,
    loss_history: list,
    output_dir: Path,
) -> dict:
    """Save SAE checkpoint, dictionary, activations, and loss curve"""
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

    # --- Dictionary (decoder weights)
    # decoder.weight is (embed_dim, n_features); save as (n_features, embed_dim)
    dictionary = model.decoder.weight.data.cpu().numpy().T
    dict_path = output_dir / "sae_dictionary.npy"
    np.save(dict_path, dictionary)

    # --- Activations on full dataset
    model.eval()
    with torch.no_grad():
        tensor_disp_vec = torch.tensor(disp_vec, dtype=torch.float32)
        _, activations = model(tensor_disp_vec.to(next(model.parameters()).device))
        act_np = activations.cpu().numpy()
    act_path = output_dir / "sae_activations.npy"
    np.save(act_path, act_np)

    # --- Loss curve
    loss_fig_path = output_dir / "sae_loss_curve.png"
    plot_loss_curve(loss_history, show=False, save=True, fig_dir=output_dir,
                    fig_name="sae_loss_curve", title="SAE Training Loss")

    # --- Summary JSON
    summary_path = output_dir / "sae_summary.json"
    save_json({
        "embed_dim": model.embed_dim,
        "n_features": model.n_features,
        "top_k": model.top_k,
        "n_samples": disp_vec.shape[0],
        "final_loss": loss_history[-1] if loss_history else None,
        "n_epochs": len(loss_history),
        "n_dead_features": int(((act_np != 0).sum(axis=0) == 0).sum()),
        "mean_active_per_sample": float((act_np != 0).sum(axis=1).mean()),
    }, summary_path)

    print(f"SAE results saved")
    print(f"  checkpoint:   {ckpt_path.name}")
    print(f"  dictionary:   {dict_path.name} | shape={dictionary.shape}")
    print(f"  activations:  {act_path.name} | shape={act_np.shape}")
    print(f"  loss curve:   {loss_fig_path.name}")
    print(f"  summary:      {summary_path.name}")

    return {
        "checkpoint": ckpt_path,
        "dictionary": dict_path,
        "activations": act_path,
        "loss_curve": loss_fig_path,
        "summary": summary_path,
    }
