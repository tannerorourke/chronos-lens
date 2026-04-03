#!/usr/bin/env python3
"""
Train a TopK Sparse Autoencoder on a JEPA displacement vector.
Don't need to freeze the JEPA model, since we already saved embedding vectors.
Hyperparameters are read from experiments/<model>/config_sae.yaml.
The target vector can be set via --target CLI arg, or the `target` field.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.utils.io import EXPERIMENTS_DIR
from src.utils.seed import load_exp_seed, set_global_seed
from training.train_sae import train_sae, save_sae_results


TARGETS = ("z_enc", "z_pred", "z_target", "pred_error")

parser = argparse.ArgumentParser(
    description="Train a TopK Sparse Autoencoder on a latent vector of a JEPA model.")
parser.add_argument(
    "--model", type=str, required=True,
    help="Experiment subdir under experiments/ (e.g. test_01)")
parser.add_argument(
    "--target", type=str, default=None, choices=TARGETS,
    help="Which vector to train on: z_enc (flattened encoder), z_pred, "
         "z_target, pred_error (z_pred - z_target). Overrides config_sae.yaml.")
parser.add_argument(
    "--embeddings", type=str, default=None,
    help="Embeddings .npz filename within the model dir (e.g. embedding_ep_40.npz). "
            "If not provided, pick latest embeddings file.")
parser.add_argument(
    "--output-dir", type=str, default=None,
    help="Output directory for SAE results (default: experiments/<model>/sae_<target>/)")


def load_sae_config(model_dir: Path) -> dict:
    """Load SAE hyperparameters from config_sae.yaml in the experiment dir."""
    config_path = model_dir / "config_sae.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"SAE config not found: {config_path}\n"
            f"Create experiments/{model_dir.name}/config_sae.yaml with SAE hyperparameters.")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parser.parse_args()

    model_dir = EXPERIMENTS_DIR / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # --- Seed from parent model config ---
    set_global_seed(load_exp_seed(model_dir))

    # --- Load SAE config ---
    cfg = load_sae_config(model_dir)
    sae_cfg = cfg["sae"]
    print(f"Loaded SAE config from: {model_dir / 'config_sae.yaml'}")

    # --- Resolve target vector: CLI > config ---
    target = ""
    if "target" in sae_cfg:
        target = sae_cfg["target"]
    elif args.target:
        target = args.target
        
    assert target in TARGETS, f"Invalid target '{target}'. Must be one of {TARGETS}."
    print(f"Target vector: {target}")

    # --- Locate embeddings file ---
    if args.embeddings:
        emb_path = model_dir / args.embeddings
    else:
        candidates = sorted(model_dir.glob("**/embeddings*.npz"))
        if not candidates:
            candidates = sorted(model_dir.glob("**/embedding*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No embeddings .npz found in model_dir. Provide --embeddings explicitly.")
        # PICK LAST
        emb_path = candidates[-1]

    print(f"Loading {target} from: {emb_path}")
    npz = np.load(emb_path, allow_pickle=True)

    # --- Load target vector ---
    if target == "z_enc":
        z_encs = npz["z_encs"]                              # (N, C, D)
        ctx_pad_mask = npz["ctx_pad_mask"].astype(bool)    # (N, C)
        data = z_encs[~ctx_pad_mask].astype(np.float64)     # (N_valid, D)
        print(f"  Flattened z_encs: {z_encs.shape} -> {data.shape} valid encounters")
    elif target == "pred_error":
        data = (npz["z_pred"] - npz["z_target"]).astype(np.float64)
        print(f"  Computed pred_error = z_pred - z_target")
    else:
        data = npz[target].astype(np.float64)

    N, D = data.shape
    print(f"  N={N} samples, D={D} embed_dim")

    # --- Train SAE ---
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / f"sae_{target}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"SAE config: n_features={sae_cfg['n_features']}, top_k={sae_cfg['top_k']}, "
          f"epochs={sae_cfg['epochs']}, lr={sae_cfg['lr']}, batch_size={sae_cfg['batch_size']}")

    model, loss_history = train_sae(
        disp_vec    =data,
        n_features  =sae_cfg["n_features"],
        top_k       =sae_cfg["top_k"],
        epochs      =sae_cfg["epochs"],
        lr          =sae_cfg["lr"],
        batch_size  =sae_cfg["batch_size"],
        device      =device,
        seed        =sae_cfg.get("seed", 42),
    )

    # --- Save results ---
    save_sae_results(model, data, loss_history, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
