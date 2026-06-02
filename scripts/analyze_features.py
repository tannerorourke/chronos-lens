#!/usr/bin/env python3
"""
SAE feature analysis script.

Loads embeddings + trained SAE checkpoint + labels, runs label enrichment,
specificity, co-activation, boolean composition, minimal feature set,
and temporal enrichment analyses.  Saves structured results to JSON + NPZ.

Usage
-----
  python -m scripts.analyze_features --exp stopg_42_v01 --emb embeddings_40.npz --sae sae_pred_error
  python -m scripts.analyze_features --exp stopg_42_v01 --ckpt checkpoint_100.pt --sae sae_pred_error
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.eval_infra import (
    extract_jepa_embeddings, compute_derived_vectors,
    load_escalation_labels, load_label_30d_at_k,
    extract_icd_block_targets)
from src.training.utils.inference import load_scaffolding
from src.analysis.sae import (
    load_sae, extract_sae_activations,
    sae_label_enrichment, feature_label_specificity,
    sae_coactivation_matrix, sae_temporal_enrichment)
from src.analysis.composition import (
    sae_boolean_composition, minimal_feature_set)
from src.utils.io import (
    RUNS_DIR, DATA_DIR,
    load_embeddings, load_sequences_dict)
from src.utils.seed import load_exp_seed, set_global_seed


# =============================================================================
# Helpers
# =============================================================================

def _encounter_times(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """Extract days_since_first for each (subject_id, mask_pos) sample."""
    N = len(subject_ids)
    times = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_pos[i])
        encs = patients_dict[sid]["encounters"]
        if pos < len(encs):
            times[i] = encs[pos].get("days_since_first", pos)
        else:
            times[i] = pos
    return times


def _relative_times(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """Extract days since previous encounter for each sample.

    First encounters get 0.
    """
    N = len(subject_ids)
    rel = np.zeros(N, dtype=np.float64)
    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_pos[i])
        encs = patients_dict[sid]["encounters"]
        if pos > 0 and pos < len(encs):
            t_cur = encs[pos].get("days_since_first", pos)
            t_prev = encs[pos - 1].get("days_since_first", pos - 1)
            rel[i] = t_cur - t_prev
    return rel


def _find_sae_dir(exp_dir: Path, sae_name: str) -> Path:
    """Locate SAE directory under experiment. Accepts name or path."""
    sae_dir = exp_dir / sae_name
    if sae_dir.is_dir():
        return sae_dir
    # Try with sae_ prefix
    sae_dir = exp_dir / f"sae_{sae_name}"
    if sae_dir.is_dir():
        return sae_dir
    # Search for matching directories
    candidates = list(exp_dir.glob(f"sae_{sae_name}*"))
    if candidates:
        candidates.sort()
        return candidates[0]
    raise FileNotFoundError(
        f"SAE directory not found: tried {exp_dir / sae_name} and {exp_dir / f'sae_{sae_name}'}")


def _top_coactivation_cliques(
    lift_matrix: np.ndarray,
    feature_indices: list[int],
    top_n: int = 5,
) -> list[tuple[int, int, float]]:
    """Extract top co-activation pairs from the lift matrix."""
    n = lift_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((feature_indices[i], feature_indices[j],
                          float(lift_matrix[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def _resolve_target_vec(emb: dict, sae_dir: Path) -> np.ndarray:
    """Resolve the SAE's input vector from an embeddings dict via the SAE dir name.

    Raises a clear, actionable error if no usable vector is present, rather than
    silently passing ``None`` into the SAE forward pass.
    """
    target_key = sae_dir.name.replace("sae_", "")
    if emb.get(target_key) is not None:
        return emb[target_key]
    if (target_key == "pred_error"
            and emb.get("z_pred") is not None and emb.get("z_target") is not None):
        return emb["z_pred"] - emb["z_target"]
    for fallback in ("z_enc_pooled", "z_pred"):
        if emb.get(fallback) is not None:
            return emb[fallback]
    raise KeyError(
        f"Could not resolve SAE target vector for '{sae_dir.name}': none of "
        f"'{target_key}', 'z_enc_pooled', 'z_pred' are present in the embeddings "
        f"(available keys: {sorted(emb.keys())}).")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE feature analysis")
    parser.add_argument("--exp", type=str, required=True,
                        help="Run-id of a completed run (under artifacts/training-runs/)")
    parser.add_argument("--sae", type=str, required=True,
                        help="SAE directory name (e.g. sae_pred_error)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emb", type=str, default=None,
                       help="Embeddings .npz file name (e.g. embeddings_40.npz)")
    group.add_argument("--ckpt", type=str, default=None,
                       help="Checkpoint .pt file to extract embeddings from")
    parser.add_argument("--target-auroc", type=float, default=0.8,
                        help="Target AUROC for minimal feature set (default: 0.8)")
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
        emb = extract_jepa_embeddings(model, loader, device)
        emb = compute_derived_vectors(emb)
        print(f"Extracted embeddings from {args.ckpt}")

    subject_ids = emb["subject_ids"]    # (N,)
    mask_pos = emb["mask_pos"]          # (N,)
    N = len(subject_ids)

    # -- Load SAE and get activations -----------------------------------------
    sae_dir = _find_sae_dir(exp_dir, args.sae)
    sae_ckpt_path = sae_dir / "sae_checkpoint.pt"
    sae_model = load_sae(sae_ckpt_path, device)
    print(f"Loaded SAE: {sae_ckpt_path} "
          f"(n_features={sae_model.n_features}, top_k={sae_model.top_k})")

    # Use pre-computed activations if available, else extract
    precomputed_path = sae_dir / "sae_activations.npy"
    if precomputed_path.exists():
        activations = np.load(precomputed_path)
        print(f"Loaded pre-computed activations: {activations.shape}")
        # Activations might be from flattened z_enc (N_valid) not sample-level (N).
        # If shape matches N, use directly; otherwise re-extract at sample level.
        if activations.shape[0] != N:
            print(f"  Shape mismatch ({activations.shape[0]} vs {N}), re-extracting...")
            vec = _resolve_target_vec(emb, sae_dir)
            activations = extract_sae_activations(sae_model, vec)
    else:
        # Infer target vector from SAE directory name
        vec = _resolve_target_vec(emb, sae_dir)
        activations = extract_sae_activations(sae_model, vec)

    n_features = activations.shape[1]
    n_active = int((activations != 0).any(axis=0).sum())
    print(f"Activations: {activations.shape}, active features: {n_active}/{n_features}")

    # -- Load patient data for labels and times --------------------------------
    sequences_path = DATA_DIR / "sequences.jsonl"
    patients_dict = load_sequences_dict(sequences_path)
    times = _encounter_times(patients_dict, subject_ids, mask_pos)
    rel_times = _relative_times(patients_dict, subject_ids, mask_pos)

    # -- Labels ----------------------------------------------------------------
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

    print(f"Labels: {list(labels_dict.keys())}")

    # -- Label enrichment per label --------------------------------------------
    print("\n--- Label Enrichment ---")
    enrichment_results: dict[str, list[dict]] = {}
    for label_name, label_vec in labels_dict.items():
        enriched = sae_label_enrichment(activations, label_vec, label_name=label_name)
        enrichment_results[label_name] = enriched
        n_sig = sum(1 for e in enriched if e["fdr_q"] < 0.05)
        print(f"  {label_name}: {len(enriched)} features tested, {n_sig} significant (FDR<0.05)")

    # -- Feature-label specificity (lift matrix) --------------------------------
    print("\n--- Feature-Label Specificity ---")
    specificity = feature_label_specificity(activations, labels_dict)
    print(f"  Lift matrix: {specificity['lift_matrix'].shape} "
          f"({len(specificity['feature_indices'])} features × {len(specificity['label_names'])} labels)")

    # -- Co-activation matrix --------------------------------------------------
    print("\n--- Co-Activation Matrix ---")
    coactivation = sae_coactivation_matrix(activations)
    print(f"  Matrix: {coactivation['lift_matrix'].shape}")
    top_cliques = _top_coactivation_cliques(
        coactivation["lift_matrix"], coactivation["feature_indices"], top_n=10)
    print("  Top co-activation pairs:")
    for f1, f2, lift in top_cliques[:5]:
        print(f"    F{f1} × F{f2}: lift={lift:.2f}")

    # -- Boolean composition per label -----------------------------------------
    print("\n--- Boolean Composition ---")
    composition_results: dict[str, dict] = {}
    for label_name, label_vec in labels_dict.items():
        comp = sae_boolean_composition(activations, label_vec)
        composition_results[label_name] = comp
        print(f"  {label_name}: tree={comp['tree_auroc']:.4f}, "
              f"single={comp['best_single_feature_auroc']:.4f}, "
              f"gap={comp['compositional_gap']:+.4f}, "
              f"features_used={comp['n_features_used']}")

    # -- Minimal feature set per label -----------------------------------------
    print("\n--- Minimal Feature Set ---")
    minimal_results: dict[str, dict] = {}
    for label_name, label_vec in labels_dict.items():
        mfs = minimal_feature_set(activations, label_vec,
                                  target_auroc=args.target_auroc)
        minimal_results[label_name] = mfs
        final_auc = mfs["auroc_curve"][-1] if mfs["auroc_curve"] else 0.0
        print(f"  {label_name}: {mfs['n_features_needed']} features needed, "
              f"final AUROC={final_auc:.4f}")

    # -- Temporal enrichment ---------------------------------------------------
    print("\n--- Temporal Enrichment ---")
    temporal = sae_temporal_enrichment(activations, times, rel_times)
    n_time_corr = sum(1 for t in temporal if abs(t["time_corr"]) > 0.1)
    print(f"  {len(temporal)} features analyzed, "
          f"{n_time_corr} with |time_corr| > 0.1")

    # -- Summary table ---------------------------------------------------------
    print(f"\n{'label':<20s} {'comp_gap':>10s} {'n_feat':>8s} {'tree_auc':>10s} "
          f"{'single_auc':>11s} {'min_feat':>9s}")
    print("-" * 72)
    for label_name in labels_dict:
        comp = composition_results[label_name]
        mfs = minimal_results[label_name]
        print(f"{label_name:<20s} "
              f"{comp['compositional_gap']:>+10.4f} "
              f"{comp['n_features_used']:>8d} "
              f"{comp['tree_auroc']:>10.4f} "
              f"{comp['best_single_feature_auroc']:>11.4f} "
              f"{mfs['n_features_needed']:>9d}")

    # -- Save results ----------------------------------------------------------
    analysis_dir = exp_dir / "results"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # JSON: scalars and per-label results
    json_output = {
        "exp": args.exp,
        "sae": args.sae,
        "n_samples": N,
        "n_features": n_features,
        "n_active_features": n_active,
        "enrichment": {
            label_name: results
            for label_name, results in enrichment_results.items()
        },
        "specificity": {
            "label_names": specificity["label_names"],
            "feature_indices": specificity["feature_indices"],
        },
        "coactivation": {
            "feature_indices": coactivation["feature_indices"],
            "top_pairs": [
                {"f1": f1, "f2": f2, "lift": round(lift, 4)}
                for f1, f2, lift in top_cliques
            ],
        },
        "composition": {
            label_name: {
                "tree_auroc": comp["tree_auroc"],
                "best_single_feature_auroc": comp["best_single_feature_auroc"],
                "compositional_gap": comp["compositional_gap"],
                "n_features_used": comp["n_features_used"],
                "rules": comp["rules"],
            }
            for label_name, comp in composition_results.items()
        },
        "minimal_feature_set": {
            label_name: mfs
            for label_name, mfs in minimal_results.items()
        },
        "temporal": temporal,
    }

    json_path = analysis_dir / "features.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=float)
    print(f"\nScalar results -> {json_path}")

    # NPZ: large arrays
    npz_data = {
        "activations": activations,                         # (N, n_features)
        "lift_matrix": specificity["lift_matrix"],          # (n_feat, n_labels)
        "coactivation_matrix": coactivation["lift_matrix"], # (n_active, n_active)
        "coactivation_indices": np.array(coactivation["feature_indices"]),
        "specificity_indices": np.array(specificity["feature_indices"]),
        "label_names": np.array(specificity["label_names"]),
        "times": times,                                     # (N,)
        "rel_times": rel_times,                             # (N,)
        "subject_ids": subject_ids,                         # (N,)
        "mask_pos": mask_pos,                               # (N,)
        "label_escalation": label_esc,                      # (N,)
        "label_30d_readmit": label_30d,                     # (N,)
    }
    npz_path = analysis_dir / "features.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Array results  -> {npz_path}")


if __name__ == "__main__":
    main()
