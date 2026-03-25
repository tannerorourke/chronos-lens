#!/usr/bin/env python3
"""
Train a TopK Sparse Autoencoder on a JEPA displacement vector.

Usage
-----
  python scripts/train_sae.py --model test_01
  python scripts/train_sae.py --model test_01 --target pred_error
  python scripts/train_sae.py --model test_01 --embeddings embedding_ep_40.npz

Hyperparameters are read from experiments/<model>/config_sae.yaml.
The target vector can be set via --target CLI arg, or the `target` field
in config_sae.yaml, falling back to "delta" (P - C).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.utils.io import EXPERIMENTS_DIR
from src.training.sae.train_sae import freeze_jepa, train_sae, save_sae_results


TARGETS = ("delta", "pred_error", "observed_traj")

parser = argparse.ArgumentParser(
    description="Train a TopK Sparse Autoencoder on a JEPA displacement vector")
parser.add_argument(
    "--model", type=str, required=True,
    help="Experiment subdir under experiments/ (e.g. test_01)")
parser.add_argument(
    "--target", type=str, default=None, choices=TARGETS,
    help="Which vector to train on: delta (P-C), pred_error (P-T), "
         "observed_traj (T-C). Overrides config_sae.yaml. Default: delta.")
parser.add_argument(
    "--embeddings", type=str, default=None,
    help="Embeddings .npz filename within the model dir (e.g. embedding_ep_40.npz). "
            "If not provided, pick latest embeddings file.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="JEPA checkpoint .pt to freeze (optional, for verification only). "
            "Not required for SAE training — SAE trains on precomputed delta.")
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

    # --- Load SAE config ---
    cfg = load_sae_config(model_dir)
    sae_cfg = cfg["sae"]
    print(f"Loaded SAE config from: {model_dir / 'config_sae.yaml'}")

    # --- Resolve target vector: CLI > config > default ---
    if args.target:
        target = args.target
    elif "target" in sae_cfg:
        target = sae_cfg["target"]
        assert target in TARGETS, \
            f"Invalid target '{target}' in config_sae.yaml. Must be one of {TARGETS}."
    else:
        target = "delta"
    print(f"Target vector: {target}")

    # --- Locate embeddings file ---
    if args.embeddings:
        emb_path = model_dir / args.embeddings
    else:
        candidates = sorted(model_dir.glob("**/embeddings*.npz"))
        if not candidates:
            candidates = sorted(model_dir.glob("**/embedding*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No embeddings .npz found in {model_dir}. "
                f"Provide --embeddings explicitly.")
        emb_path = candidates[-1]

    print(f"Loading {target} from: {emb_path}")
    npz = np.load(emb_path, allow_pickle=True)

    # Load named vector, or compute from base embeddings if not in npz
    if target in npz:
        data = npz[target].astype(np.float64)
    else:
        z_context = npz["z_context"]
        z_pred    = npz["z_pred"]
        z_target  = npz["z_target"]
        compute_map = {
            "delta":         lambda: (z_pred - z_context).astype(np.float64),
            "pred_error":    lambda: (z_pred - z_target).astype(np.float64),
            "observed_traj": lambda: (z_target - z_context).astype(np.float64),
        }
        data = compute_map[target]()
        print(f"  (computed {target} from base embeddings — not stored in npz)")

    N, D = data.shape
    print(f"  N={N} samples, D={D} embed_dim")

    # --- Optionally verify JEPA is loadable ---
    if args.checkpoint:
        ckpt_path = model_dir / "checkpoints" / args.checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        jepa = freeze_jepa(ckpt_path, device)
        print(f"  JEPA loaded and frozen (embed_dim={jepa.embed_dim})")
        del jepa

    # --- Output directory ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir / f"sae_{target}"

    # --- Train SAE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"SAE config: n_features={sae_cfg['n_features']}, top_k={sae_cfg['top_k']}, "
          f"epochs={sae_cfg['epochs']}, lr={sae_cfg['lr']}, batch_size={sae_cfg['batch_size']}")

    model, loss_history = train_sae(
        delta=data,
        n_features=sae_cfg["n_features"],
        top_k=sae_cfg["top_k"],
        epochs=sae_cfg["epochs"],
        lr=sae_cfg["lr"],
        batch_size=sae_cfg["batch_size"],
        device=device,
        seed=sae_cfg.get("seed", 42),
    )

    # --- Save results ---
    save_sae_results(model, data, loss_history, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
