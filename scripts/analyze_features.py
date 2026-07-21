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

import logging
logger = logging.getLogger(__name__)

from src.infra.inference import (
    load_embeddings_for_analysis, load_sae_info
)
from src.infra.labels import (
    load_escalation_labels, load_label_30d_at_k, 
    extract_icd_block_targets, get_absolute_enc_times, get_relative_enc_times
)
from src.analysis.sae import (
    sae_label_enrichment, feature_label_specificity,
    sae_coactivation_matrix, sae_temporal_enrichment,
    inspect_sae_feature_content
)
from src.analysis.composition import (
    sae_boolean_composition, minimal_feature_set
)
from src.utils.io import EXPS_DIR, DATA_DIR, load_sequences_dict
from src.utils.system import load_exp_seed, set_global_seed


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

parser.add_argument("--save-res", default=False, action="store_true")
parser.add_argument("--cards", default=False, action="store_true",
                    help="Emit per-feature clinical content cards into features.json (top-activator ICD/med enrichment)."
                         "Costs an O(N,features) sweep over sequences.jsonl")

# =============================================================================
# Helpers
# =============================================================================

def _top_coactivation_cliques(
    lift_matrix: np.ndarray,
    feature_indices: list[int],
    top_n: int = 5,
    _verbose: bool = False
) -> list[tuple[int, int, float]]:
    """Extract top co-activation pairs from the lift matrix."""
    if _verbose: print("\n Top co-activation pairs:")
    n = lift_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(
                ( feature_indices[i], feature_indices[j], float(lift_matrix[i, j]) )
            )
    pairs.sort(key=lambda x: x[2], reverse=True)
    topn_pairs = pairs[:top_n]
    
    if _verbose:
        for f1, f2, lift in topn_pairs:
            print(f"    F{f1} x F{f2}: lift={lift:.2f}")
    
    return topn_pairs


# =============================================================================
# =============================================================================


def main():
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_id = args.exp
    exp_dir = EXPS_DIR / exp_id
    
    set_global_seed(load_exp_seed(exp_dir))

    # -- Load or extract embeddings (the .npz stem pairs with checkpoints/<stem>.pt).
    #    Only the alignment fields are read; activations come from the SAE dir.
    emb_name = args.emb if args.emb else f"{Path(args.ckpt).stem}.npz"
    with load_embeddings_for_analysis(
        exp_id, emb_name, 
        device, sync_ckpts=False
    )[0] as es:
        subject_ids = es["subject_ids"]    # (N,)
        mask_pos = es["mask_pos"]          # (N,)

    N = len(subject_ids)

    # -- Load SAE and get activations
    sae_model, _, sae_params, dec_weights, activations = load_sae_info(exp_id, args.sae, device)
    print(f"Loaded SAE: n_features={sae_model.n_features}, top_k={sae_model.top_k}")

    n_features = sae_model.n_features
    n_active = int((activations != 0).any(axis=0).sum())
    print(f"Activations: {activations.shape}, active features: {n_active}/{n_features}")

    # -- Load patient data for labels and times
    sequences_path = DATA_DIR / "sequences.jsonl"
    patients_dict = load_sequences_dict(sequences_path)
    times = get_absolute_enc_times(patients_dict, subject_ids, mask_pos)
    rel_times = get_relative_enc_times(patients_dict, subject_ids, mask_pos)

    # -- Load Labels
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


    # --- ANALYSIS
    enrichment_results: dict[str, list[dict]] = {}
    for label_name, label_vec in labels_dict.items():
        print(f"\n-- Label Enrichment --> label: {label_name}")
        enriched = sae_label_enrichment(activations, label_vec, _verbose=True)
        enrichment_results[label_name] = enriched

    print("\n-- Feature-Label Specificity (ie, lift matrix) --")
    specificity = feature_label_specificity(activations, labels_dict, _verbose=True)
    
    
    print("\n-- Co-Activation Matrix --")
    coactivation = sae_coactivation_matrix(activations, _verbose=True)
    top_cliques = _top_coactivation_cliques(
        coactivation["lift_matrix"], 
        coactivation["feature_indices"], 
        top_n=10,
        _verbose=True
    )

    # -- Boolean composition per label -----------------------------------------
    print("\n--- Boolean Composition ---")
    composition_results: dict[str, dict] = {}
    for label_name, label_vec in labels_dict.items():
        comp = sae_boolean_composition(activations, label_vec)
        composition_results[label_name] = comp
        print(
            f"  {label_name}: tree={comp['tree_auroc']:.4f}, "
            f"single={comp['best_single_feature_auroc']:.4f}, "
            f"gap={comp['compositional_gap']:+.4f}, "
            f"features_used={comp['n_features_used']}"
        )

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
    temporal = sae_temporal_enrichment(activations, times, rel_times, _verbose=True)

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

    # -- Per-feature clinical content (opt-in) ---------------------------------
    # Auto-interp each live feature from its top activators' raw ICD/med vocabulary at the recency
    # encounter (mask_pos). This is the label-agnostic "what does this feature encode" pass that the
    # sae-labeler reads; a feature whose content contradicts its label-first identity is the
    # mislabeling signal.
    feature_cards = None
    if args.cards:
        print("\n--- Feature Content Cards (--cards) ---")
        feature_cards = inspect_sae_feature_content(
            activations, subject_ids, sequences_path,
            encounter_indices=mask_pos, encounter_level=True,
            top_n_samples=50, top_n_enriched=10, min_activation_frac=0.01,
        )
        print(f"  {len(feature_cards)} feature cards (activation_frac >= 0.01)")

    # -- Save results ----------------------------------------------------------
    if args.save_res:
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
        if feature_cards is not None:
            json_output["feature_cards"] = feature_cards

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
