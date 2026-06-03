#!/usr/bin/env python3
"""
Compositional decomposition analysis script.

Loads embeddings + trained SAE + labels, runs label subspace extraction,
effective rank, compositional decomposition (greedy matching pursuit of
label subspace via SAE decoder directions), and cross-label subspace
alignment.  Saves structured results to JSON + NPZ.

Usage
-----
  python -m scripts.analyze_composition --exp stopg_42_v01 --emb embeddings_40.npz --sae sae_pred_error
  python -m scripts.analyze_composition --exp stopg_42_v01 --ckpt checkpoint_100.pt --sae sae_pred_error
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.infra.loaders import load_scaffolding, extract_embeddings
from src.infra.vector_computation import compute_derived_vectors
from src.infra.labels import (
    load_escalation_labels, load_label_30d_at_k, extract_icd_block_targets)
from src.analysis.geometry import (
    label_subspace, multi_label_subspace, effective_rank_of_label)
from src.analysis.composition import (
    compositional_decomposition)
from src.analysis.sae import load_sae
from src.utils.io import (
    RUNS_DIR, DATA_DIR,
    load_embeddings, load_sequences_dict)
from src.utils.seed import load_exp_seed, set_global_seed


# =============================================================================
# Helpers
# =============================================================================

def _find_sae_dir(exp_dir: Path, sae_name: str) -> Path:
    """Locate SAE directory under experiment. Accepts name or path."""
    sae_dir = exp_dir / sae_name
    if sae_dir.is_dir():
        return sae_dir
    sae_dir = exp_dir / f"sae_{sae_name}"
    if sae_dir.is_dir():
        return sae_dir
    candidates = list(exp_dir.glob(f"sae_{sae_name}*"))
    if candidates:
        candidates.sort()
        return candidates[0]
    raise FileNotFoundError(
        f"SAE directory not found: tried {exp_dir / sae_name} and {exp_dir / f'sae_{sae_name}'}")


def _label_subspace_alignment(dirs_a: np.ndarray, dirs_b: np.ndarray) -> dict:
    """Principal angles between two direction matrices (rank_a, D) and (rank_b, D)."""
    k = min(dirs_a.shape[0], dirs_b.shape[0])
    M = dirs_a @ dirs_b.T
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    cos_angles = np.clip(s[:k], 0.0, 1.0)
    return {
        "cos_principal_angles": cos_angles.tolist(),
        "mean_alignment": float(cos_angles.mean()),
        "min_alignment": float(cos_angles.min()),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compositional decomposition analysis")
    parser.add_argument("--exp", type=str, required=True,
                        help="Run-id of a completed run (under artifacts/training-runs/)")
    parser.add_argument("--sae", type=str, required=True,
                        help="SAE directory name (e.g. sae_pred_error)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emb", type=str, default=None,
                       help="Embeddings .npz file name (e.g. embeddings_40.npz)")
    group.add_argument("--ckpt", type=str, default=None,
                       help="Checkpoint .pt file to extract embeddings from")
    parser.add_argument("--rank", type=int, default=5,
                        help="Subspace rank for label_subspace (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Coverage threshold for compositional decomposition (default: 0.8)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = RUNS_DIR / args.exp   # --exp is the run-id under RUNS_DIR

    # -- Load or extract embeddings -------------------------------------------
    if args.emb:
        set_global_seed(load_exp_seed(exp_dir))
        emb, emb_path = load_embeddings(exp_dir, args.emb)
        print(f"Loaded embeddings: {emb_path}")
    else:
        model, loader, exp_dir, (ckpt, config), _ = \
            load_scaffolding(args.ckpt, args.exp, device)
        emb = extract_embeddings(model, loader, device)
        emb = compute_derived_vectors(emb)
        print(f"Extracted embeddings from {args.ckpt}")

    subject_ids = emb["subject_ids"]    # (N,)
    mask_pos = emb["mask_pos"]          # (N,)
    N = len(subject_ids)

    # Per-sample analysis vector: the recency encounter z_enc[k-1] (fall back to z_pred)
    z_enc = emb.get("z_enc_recency")
    if z_enc is None:
        z_enc = emb.get("z_pred")
    if z_enc is None:
        raise KeyError(
            f"No 'z_enc_recency' or 'z_pred' in embeddings for run '{args.exp}' "
            f"(available keys: {sorted(emb.keys())}).")
    D = z_enc.shape[1]
    print(f"z_enc: ({N}, {D})")

    # -- Load SAE -------------------------------------------------------------
    sae_dir = _find_sae_dir(exp_dir, args.sae)
    sae_ckpt_path = sae_dir / "sae_checkpoint.pt"
    sae_model = load_sae(sae_ckpt_path, device)
    print(f"Loaded SAE: {sae_ckpt_path} "
          f"(n_features={sae_model.n_features}, top_k={sae_model.top_k})")

    # -- Load labels ----------------------------------------------------------
    sequences_path = DATA_DIR / "sequences.jsonl"
    patients_dict = load_sequences_dict(sequences_path)

    label_esc = load_escalation_labels(patients_dict, subject_ids, mask_pos)
    label_30d = load_label_30d_at_k(patients_dict, subject_ids, mask_pos)
    icd_targets, icd_chapters = extract_icd_block_targets(
        sequences_path, subject_ids, mask_pos)

    labels_dict: dict[str, np.ndarray] = {
        "escalation": label_esc,
        "30d_readmit": label_30d,
    }
    for ch_idx, ch_name in enumerate(icd_chapters):
        ch_labels = icd_targets[:, ch_idx]
        if ch_labels.sum() >= 10:
            labels_dict[f"icd_{ch_name}"] = ch_labels

    label_names = sorted(labels_dict.keys())
    print(f"Labels: {label_names}")

    # -- Per-label: label_subspace + effective_rank + compositional decomp -----
    print("\n--- Label Subspace Analysis ---")
    subspace_results: dict[str, dict] = {}
    rank_results: dict[str, int] = {}
    decomp_results: dict[str, dict] = {}
    label_dirs: dict[str, np.ndarray] = {}

    for lname in label_names:
        lbl = labels_dict[lname]

        # Label subspace
        sub = label_subspace(z_enc, lbl, rank=args.rank)
        subspace_results[lname] = {
            "eigenvalues": sub["eigenvalues"].tolist(),
            "explained_separation": sub["explained_separation"],
        }
        label_dirs[lname] = sub["directions"]

        # Effective rank
        eff_rank = effective_rank_of_label(z_enc, lbl)
        rank_results[lname] = eff_rank

        # Compositional decomposition
        decomp = compositional_decomposition(sub, sae_model, threshold=args.threshold)
        decomp_results[lname] = {
            "selected_features": decomp["selected_features"],
            "principal_angles": decomp["principal_angles"],
            "n_features_needed": decomp["n_features_needed"],
            "residual": decomp["residual"],
        }

        n_feat = decomp["n_features_needed"]
        residual = decomp["residual"]
        final_cov = decomp["principal_angles"][-1] if decomp["principal_angles"] else 0.0
        print(f"  {lname}: eff_rank={eff_rank}, "
              f"n_features_needed={n_feat}, "
              f"coverage={final_cov:.4f}, residual={residual:.4f}")

    # -- Multi-label CCA subspace ---------------------------------------------
    print("\n--- Multi-label CCA ---")
    label_matrix = np.column_stack([labels_dict[ln] for ln in label_names])
    multi_sub = multi_label_subspace(z_enc, label_matrix, rank=args.rank,
                                     label_names=label_names)
    print(f"  Top canonical correlations: "
          f"{[round(c, 4) for c in multi_sub['correlations'][:5]]}")

    # -- Cross-label subspace alignment ---------------------------------------
    print("\n--- Cross-label Subspace Alignment ---")
    alignment_results: dict[str, dict] = {}
    for i, ln_a in enumerate(label_names):
        for j, ln_b in enumerate(label_names):
            if j <= i:
                continue
            alignment = _label_subspace_alignment(label_dirs[ln_a], label_dirs[ln_b])
            key = f"{ln_a}_vs_{ln_b}"
            alignment_results[key] = alignment
            print(f"  {key}: mean={alignment['mean_alignment']:.4f}, "
                  f"min={alignment['min_alignment']:.4f}")

    # -- Summary table --------------------------------------------------------
    print(f"\n{'label':<20s} {'eff_rank':>10s} {'n_feat':>8s} "
          f"{'coverage':>10s} {'residual':>10s}")
    print("-" * 62)
    for lname in label_names:
        decomp = decomp_results[lname]
        final_cov = decomp["principal_angles"][-1] if decomp["principal_angles"] else 0.0
        print(f"{lname:<20s} "
              f"{rank_results[lname]:>10d} "
              f"{decomp['n_features_needed']:>8d} "
              f"{final_cov:>10.4f} "
              f"{decomp['residual']:>10.4f}")

    # -- Save results ---------------------------------------------------------
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON: scalars and per-label results
    json_output = {
        "exp": args.exp,
        "sae": args.sae,
        "n_samples": N,
        "embed_dim": D,
        "rank": args.rank,
        "threshold": args.threshold,
        "label_names": label_names,
        "subspaces": subspace_results,
        "effective_ranks": rank_results,
        "decomposition": decomp_results,
        "multi_label_cca": {
            "correlations": multi_sub["correlations"].tolist(),
            "label_names": multi_sub["label_names"],
        },
        "alignment": alignment_results,
    }

    json_path = results_dir / "composition.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=float)
    print(f"\nScalar results -> {json_path}")

    # NPZ: large arrays
    npz_data = {
        "multi_cca_directions": multi_sub["canonical_directions"],
        "multi_cca_correlations": multi_sub["correlations"],
    }
    # Per-label subspace directions
    for lname in label_names:
        safe_name = lname.replace("/", "_")
        npz_data[f"label_dirs_{safe_name}"] = label_dirs[lname]
        npz_data[f"label_eigenvalues_{safe_name}"] = \
            np.array(subspace_results[lname]["eigenvalues"])

    # Alignment matrix for heatmap
    n_labels = len(label_names)
    alignment_matrix = np.eye(n_labels, dtype=np.float64)
    for i, ln_a in enumerate(label_names):
        for j, ln_b in enumerate(label_names):
            if j <= i:
                continue
            key = f"{ln_a}_vs_{ln_b}"
            alignment_matrix[i, j] = alignment_results[key]["mean_alignment"]
            alignment_matrix[j, i] = alignment_matrix[i, j]
    npz_data["alignment_matrix"] = alignment_matrix
    npz_data["label_names"] = np.array(label_names)

    npz_path = results_dir / "composition.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Array results  -> {npz_path}")


if __name__ == "__main__":
    main()
