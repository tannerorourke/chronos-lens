import argparse
import gc
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.models.sae import SparseAutoencoder
from src.training.utils.checkpoint import save_sae_results
from src.infra.inference import load_embeddings_for_analysis
from src.utils.io import init_exp_config
from src.utils.constants import SAE_TARGETS
from src.utils.system import set_global_seed, load_exp_seed, set_cuda_precision


parser = argparse.ArgumentParser(description="""Trains a TopK sparse autoencoder on a JEPA latent vector.""")

parser.add_argument(
    '--exp', required=True,
    help="Run-id naming the input config 'experiments/<exp>.yaml'. If a run "
        "with this id already has artifacts, a new versioned run dir is "
        "populated. New models should add a unique 'experiments/<run-id>.yaml'.")
parser.add_argument(
    "--target", type=str, required=True, choices=SAE_TARGETS,
    help="Which vector to train on: z_enc (flattened encoder), z_pred, z_target, "
         "pred_error (z_pred - z_target). Must be present in the run's config['sae_config'].")
parser.add_argument(
    "--embeddings", type=str, required=True,
    help="Embeddings .npz filename within the run's embeddings/ dir (e.g., embedding_40.npz)")


def train(
    sae_params: dict,
    sae_exp_dir: Path,
    data_vec: np.ndarray,
    device: torch.device
):
    n_features  = sae_params["n_features"]
    top_k       = sae_params["top_k"]
    epochs      = sae_params["epochs"]
    lr          = float(sae_params["lr"])
    batch_size  = sae_params["batch_size"]
            
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
        logger.info(f"[{epoch:4d}/{epochs}] | loss={mean_loss:.6f}")
        
        # -- Periodic diagnostics every 10% of training or last epoch
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            model.eval()
    
            all_x = tensor_data_vec.to(device)
            _, all_act = model(all_x)
            all_act_np = all_act.cpu().numpy()

            active_per_sample = (all_act_np != 0).sum(axis=1).mean()
            dead_features = int(((all_act_np != 0).sum(axis=0) == 0).sum())

            # mean pairwise cosine of decoder rows
            W = model.D.weight.data # (embed_dim, n_features)
            D_cols = W / W.norm(dim=0, keepdim=True).clamp(min=1e-10)
            cos_matrix = D_cols.T @ D_cols # (n_features, n_features)
            n_f = cos_matrix.shape[0]
            
            mean_cos = 0.0
            if n_f > 1:
                triu_idx = torch.triu_indices(n_f, n_f, offset=1)
                mean_cos = cos_matrix[triu_idx[0], triu_idx[1]].mean().item()
                
            logger.info(
                f"  active/sample={active_per_sample:.1f}  "
                f"  dead={dead_features}/{n_features}  "
                f"  dict_cos={mean_cos:.3f}"
            )
            model.train()

    logger.info("\nTraining Complete!")
    model.eval()
    save_sae_results(model, data_vec, loss_history, sae_exp_dir, device)
    logger.info("Done.")

    
def main():
    args = parser.parse_args()
    target = args.target
    embeddings = args.embeddings if args.embeddings.endswith(".npz") else f"{args.embeddings}.npz"
    run_dir, sae_params = init_exp_config(args.exp, "sae", target)
    set_global_seed(load_exp_seed(run_dir))
    
    logger.info(f"  Experiment: '{run_dir.parent.name}' -> '{run_dir.name}'"
                f"  Artifact dir: {run_dir}"
                f"  Target: {target}"
                f"  Embeddings: {embeddings}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        set_cuda_precision(use_bf16=True)
    logger.info(f"  Device: {torch.cuda.get_device_name() if device.type == 'cuda' else device}")
    
    with load_embeddings_for_analysis(
        run_id=run_dir.parent.name, name=embeddings, 
        device=device
    )[0] as embed:
        if target == "z_enc":
            ctx_pad_mask = embed["ctx_pad_mask"].astype(bool)
            data_vec = embed["z_encs"][~ctx_pad_mask].astype(np.float64)
        elif target == "pred_error":
            data_vec = (embed["z_pred"] - embed["z_target"]).astype(np.float64)
        else:
            data_vec = embed[target].astype(np.float64)
        train(sae_params, run_dir, data_vec, device)
    
    
if __name__ == "__main__":
    main()