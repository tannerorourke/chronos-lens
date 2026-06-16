#!/usr/bin/env python3
"""
Cross-architecture comparison between JEPA and supervised embeddings.

Usage
-----
  python -m scripts.analyze_comparison \
      --jepa-exp stopg_42_v01 --jepa-emb embeddings_40.npz \
      --sup-exp supervised_64_42 --sup-emb embeddings_20.npz \
      [--jepa-sae sae_pred_error] [--sup-sae sae_z_enc]
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

import numpy as np
import torch

from src.infra.labels import (
    load_escalation_labels, load_label_30d_at_k, 
    extract_icd_block_targets, get_absolute_enc_times
)
from src.analysis.geometry import (
    fit_pca, 
    get_pca_stats, 
    linear_cka, 
    subspace_alignment, 
    label_subspace,
    label_subspace_alignment
)
from src.analysis.sae import cross_sae_overlap
from src.analysis.trajectories import (
    extract_trajectories,
    trajectory_velocity, trajectory_curvature,
    concept_centroid,
    prospective_trajectory_probe
)
from src.infra.inference import load_embeddings_for_analysis, load_sae_info
from src.utils.io import EXPS_DIR, DATA_DIR, load_sequences_dict
from src.utils.system import load_exp_seed, set_global_seed


parser = argparse.ArgumentParser(description="Cross-architecture comparison: JEPA vs supervised")
parser.add_argument("--jepa-exp", type=str, required=True,
                    help="JEPA experiment name")
parser.add_argument("--jepa-emb", type=str, required=True,
                    help="JEPA embeddings .npz filename")

parser.add_argument("--sup-exp", type=str, required=True,
                    help="Supervised experiment name")
parser.add_argument("--sup-emb", type=str, required=True,
                    help="Supervised embeddings .npz filename")

parser.add_argument("--jepa-sae", type=str, default=None,
                    help="JEPA SAE directory name (optional)")
parser.add_argument("--sup-sae", type=str, default=None,
                    help="Supervised SAE directory name (optional)")

parser.add_argument("--top-k", type=int, default=10,
                    help="Top-k PCs for subspace analysis (default: 10)")
parser.add_argument("--rank", type=int, default=5,
                    help="Rank for label subspace (default: 5)")
parser.add_argument("--cosine-threshold", type=float, default=0.85,
                    help="SAE feature overlap stability threshold (default: 0.85)")

# =============================================================================
# Helpers
# =============================================================================

def _build_labels_per_step(
    sample_labels: np.ndarray,
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
    traj_dict: dict,
) -> np.ndarray:
    """Map sample-level labels (N,) into trajectory-aligned (P, T_max) matrix.

    Entries without a matching sample are set to -1 (ignore).
    """
    patient_ids = traj_dict["patient_ids"]
    T_max = traj_dict["trajectories"].shape[1]
    P = len(patient_ids)
    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}

    patient_steps: dict[str, list[tuple[int, int]]] = {}
    for i in range(len(subject_ids)):
        sid = str(subject_ids[i])
        patient_steps.setdefault(sid, []).append((int(mask_pos[i]), i))
    for sid in patient_steps:
        patient_steps[sid].sort(key=lambda x: x[0])

    labels_mat = np.full((P, T_max), -1, dtype=int)
    for sid, steps in patient_steps.items():
        if sid not in pid_to_idx:
            continue
        p = pid_to_idx[sid]
        for t, (_, sample_idx) in enumerate(steps):
            if t < T_max:
                labels_mat[p, t] = int(sample_labels[sample_idx])

    return labels_mat


# =============================================================================
# Main
# =============================================================================

def main():
    args = parser.parse_args()
    
    jepa_exp_id = args.jepa_exp
    sup_exp_id = args.sup_exp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa_exp_dir = EXPS_DIR / jepa_exp_id
    sup_exp_dir = EXPS_DIR / sup_exp_id
    
    set_global_seed(load_exp_seed(jepa_exp_dir))

    # -- Load embeddings ----------------------------------------------------
    with load_embeddings_for_analysis(
        jepa_exp_id, args.jepa_emb, 
        device, sync_ckpts=False
    )[0] as jes:
        jepa_sids = jes["subject_ids"]
        jepa_mpos = jes["mask_pos"]

        # -- recency slice: the last valid context slot per sample
        last_idx = (np.asarray(jepa_mpos) - 1).astype(int)
        rows = np.arange(jes["z_encs"].shape[0])
        jepa_z = jes["z_encs"][rows, last_idx].astype(np.float32)  # (N, D)
        D = jepa_z.shape[1]
        print(f"JEPA:       N={len(jepa_sids)}, D={D}")
    
    with load_embeddings_for_analysis(
        sup_exp_id, args.sup_emb, 
        device, sync_ckpts=False
    )[0] as ses:
        sup_sids = ses["subject_ids"]
        sup_mpos = ses["mask_pos"]

        last_idx = (np.asarray(sup_mpos) - 1).astype(int)
        rows = np.arange(ses["z_encs"].shape[0])
        sup_z = ses["z_encs"][rows, last_idx].astype(np.float32)  # (N, D)

        print(f"Supervised: N={len(sup_sids)}, D={sup_z.shape[1]}")
    
    # -- Match samples by (subject_id, mask_pos) ----------------------------
    sup_lookup: dict[tuple[str, int], int] = {}
    for i in range(len(sup_sids)):
        key = (str(sup_sids[i]), int(sup_mpos[i]))
        sup_lookup[key] = i

    jepa_idx, sup_idx = [], []
    for i in range(len(jepa_sids)):
        key = (str(jepa_sids[i]), int(jepa_mpos[i]))
        if key in sup_lookup:
            jepa_idx.append(i)
            sup_idx.append(sup_lookup[key])
    jepa_idx = np.array(jepa_idx)
    sup_idx = np.array(sup_idx)

    N_matched = len(jepa_idx)
    print(f"Matched samples: {N_matched}")
    if N_matched == 0:
        raise ValueError("No matching (subject_id, mask_pos) pairs found")

    z_j = jepa_z[jepa_idx]
    z_s = sup_z[sup_idx]
    matched_sids = jepa_sids[jepa_idx]
    matched_mpos = jepa_mpos[jepa_idx]

    # -- PCA stats ----------------------------------------------------------
    print("\n--- PCA Stats ---")
    top_k = args.top_k

    jepa_pca, _, _ = fit_pca(z_j, top_k)
    sup_pca, _, _ = fit_pca(z_s, top_k)
    jepa_pca_stats = get_pca_stats(jepa_pca, top_k, N_matched)
    sup_pca_stats = get_pca_stats(sup_pca, top_k, N_matched)

    print(f"  JEPA eff. dim:       {jepa_pca_stats['effective_dimensionality']:.1f}")
    print(f"  Supervised eff. dim: {sup_pca_stats['effective_dimensionality']:.1f}")

    # -- CKA ----------------------------------------------------------------
    print("\n--- CKA ---")
    cka = linear_cka(z_j, z_s)
    print(f"  Linear CKA: {cka:.4f}")

    # -- Subspace alignment -------------------------------------------------
    print("\n--- Subspace Alignment ---")
    sa = subspace_alignment(jepa_pca, sup_pca, top_k)
    print(f"  Mean alignment: {sa['mean_alignment']:.4f}")
    print(f"  Min alignment:  {sa['min_alignment']:.4f}")

    # -- SAE feature overlap ------------------------------------------------
    sae_overlap_result = None
    sae_matched_cosines = None
    if args.jepa_sae and args.sup_sae:
        print("\n--- SAE Feature Overlap ---")
        _, _, jepa_ckpt, jepa_dec_weights, _ = \
            load_sae_info(jepa_exp_id, args.jepa_sae, device)
        _, _, _, spv_dec_weights, _ = \
            load_sae_info(sup_exp_id, args.sup_sae, device)
        
        sae_overlap_result = cross_sae_overlap(
            jepa_dec_weights,
            spv_dec_weights,
            cosine_threshold=args.cosine_threshold)
        sae_matched_cosines = sae_overlap_result.pop("matched_cosines")
        print(f"  Mean cosine: {sae_overlap_result['mean_cosine']:.4f}")
        print(f"  Frac stable: {sae_overlap_result['frac_stable']:.2%}")
        print(f"  N matched:   {sae_overlap_result['n_matched']}")

    # -- Load labels --------------------------------------------------------
    print("\n--- Loading labels ---")
    sequences_path = DATA_DIR / "sequences.jsonl"
    patients_dict = load_sequences_dict(sequences_path)

    label_esc = load_escalation_labels(patients_dict, matched_sids, matched_mpos)
    label_30d = load_label_30d_at_k(patients_dict, matched_sids, matched_mpos)
    icd_targets, icd_chapters = extract_icd_block_targets(
        sequences_path, matched_sids, matched_mpos)

    labels_dict: dict[str, np.ndarray] = {
        "escalation": label_esc,
        "30d_readmit": label_30d,
    }
    for ch_idx, ch_name in enumerate(icd_chapters):
        ch_labels = icd_targets[:, ch_idx]
        if ch_labels.sum() >= 10:
            labels_dict[f"icd_{ch_name}"] = ch_labels

    label_names = sorted(labels_dict.keys())
    print(f"  Labels: {label_names}")

    # -- Trajectory comparison ----------------------------------------------
    print("\n--- Trajectory Comparison ---")
    times_matched = get_absolute_enc_times(patients_dict, matched_sids, matched_mpos)

    jepa_traj = extract_trajectories(z_j, matched_sids, matched_mpos, times_matched)
    sup_traj = extract_trajectories(z_s, matched_sids, matched_mpos, times_matched)

    jepa_vel, jepa_vel_mask = trajectory_velocity(jepa_traj)
    sup_vel, sup_vel_mask = trajectory_velocity(sup_traj)
    jepa_curv, jepa_curv_mask = trajectory_curvature(jepa_traj)
    sup_curv, sup_curv_mask = trajectory_curvature(sup_traj)

    # Flatten valid entries for stats and NPZ
    jv_flat = np.linalg.norm(jepa_vel[jepa_vel_mask], axis=-1)
    sv_flat = np.linalg.norm(sup_vel[sup_vel_mask], axis=-1)
    jc_flat = jepa_curv[jepa_curv_mask]
    sc_flat = sup_curv[sup_curv_mask]

    traj_comparison = {
        "jepa_velocity_mean": float(jv_flat.mean()) if len(jv_flat) else 0.0,
        "sup_velocity_mean": float(sv_flat.mean()) if len(sv_flat) else 0.0,
        "jepa_curvature_mean": float(jc_flat.mean()) if len(jc_flat) else 0.0,
        "sup_curvature_mean": float(sc_flat.mean()) if len(sc_flat) else 0.0,
    }
    print(f"  JEPA vel mean:  {traj_comparison['jepa_velocity_mean']:.4f}")
    print(f"  Sup  vel mean:  {traj_comparison['sup_velocity_mean']:.4f}")
    print(f"  JEPA curv mean: {traj_comparison['jepa_curvature_mean']:.4f}")
    print(f"  Sup  curv mean: {traj_comparison['sup_curvature_mean']:.4f}")

    # -- Prospective probe comparison ---------------------------------------
    print("\n--- Prospective Probe ---")
    probe_comparison: dict[str, dict] = {}

    for lname in label_names:
        lbl = labels_dict[lname]

        # Build trajectory-aligned labels for each model
        jepa_labels_step = _build_labels_per_step(lbl, matched_sids, matched_mpos, jepa_traj)
        sup_labels_step = _build_labels_per_step(lbl, matched_sids, matched_mpos, sup_traj)

        pos_mask = lbl == 1
        if pos_mask.sum() < 5:
            continue

        # -- per-model centroids and baselines: the two embedding spaces are
        #    not interchangeable, and the probe baseline expects (P, T_max, D)
        centroid_j = concept_centroid(z_j, lbl)
        centroid_s = concept_centroid(z_s, lbl)

        try:
            jepa_probe = prospective_trajectory_probe(
                jepa_traj, jepa_labels_step,
                baseline_z_enc=jepa_traj["trajectories"],
                centroid_mean=centroid_j["mean"])
            sup_probe = prospective_trajectory_probe(
                sup_traj, sup_labels_step,
                baseline_z_enc=sup_traj["trajectories"],
                centroid_mean=centroid_s["mean"])
        except (ValueError, np.linalg.LinAlgError):
            continue

        probe_comparison[lname] = {
            "jepa_traj_auroc": float(jepa_probe["traj_auroc"]),
            "jepa_baseline_auroc": float(jepa_probe["baseline_auroc"]),
            "sup_traj_auroc": float(sup_probe["traj_auroc"]),
            "sup_baseline_auroc": float(sup_probe["baseline_auroc"]),
        }
        print(f"  {lname}: JEPA traj={jepa_probe['traj_auroc']:.4f}, "
              f"Sup traj={sup_probe['traj_auroc']:.4f}")

    # -- Label subspace alignment -------------------------------------------
    print("\n--- Label Subspace Alignment ---")
    label_sub_alignment: dict[str, dict] = {}
    jepa_label_dirs: dict[str, np.ndarray] = {}
    sup_label_dirs: dict[str, np.ndarray] = {}

    for lname in label_names:
        lbl = labels_dict[lname]
        if lbl.sum() < 10 or (lbl == 0).sum() < 10:
            continue

        jepa_sub = label_subspace(z_j, lbl, rank=args.rank)
        sup_sub = label_subspace(z_s, lbl, rank=args.rank)
        jepa_label_dirs[lname] = jepa_sub["directions"]
        sup_label_dirs[lname] = sup_sub["directions"]

        alignment = label_subspace_alignment(
            jepa_sub["directions"], sup_sub["directions"])
        label_sub_alignment[lname] = {
            "mean_alignment": alignment["mean_alignment"],
            "min_alignment": alignment["min_alignment"],
        }
        print(f"  {lname}: mean={alignment['mean_alignment']:.4f}, "
              f"min={alignment['min_alignment']:.4f}")

    # Build cross-model alignment matrix (n_labels, n_labels):
    # entry (i,j) = alignment between label_i subspace in JEPA and
    # label_j subspace in supervised
    aligned_labels = sorted(set(jepa_label_dirs.keys()) & set(sup_label_dirs.keys()))
    n_labels = len(aligned_labels)
    alignment_matrix = np.zeros((n_labels, n_labels), dtype=np.float64)
    for i, ln_a in enumerate(aligned_labels):
        for j, ln_b in enumerate(aligned_labels):
            aln = label_subspace_alignment(
                jepa_label_dirs[ln_a], sup_label_dirs[ln_b])
            alignment_matrix[i, j] = aln["mean_alignment"]

    # -- Summary table ------------------------------------------------------
    print(f"\n{'metric':<30s} {'JEPA':>12s} {'Supervised':>12s}")
    print("-" * 56)
    print(f"{'eff_dimensionality':<30s} "
          f"{jepa_pca_stats['effective_dimensionality']:>12.1f} "
          f"{sup_pca_stats['effective_dimensionality']:>12.1f}")
    print(f"{'PCs for 90%':<30s} "
          f"{jepa_pca_stats['components_for_90pct']:>12d} "
          f"{sup_pca_stats['components_for_90pct']:>12d}")
    print(f"{'PCs for 95%':<30s} "
          f"{jepa_pca_stats['components_for_95pct']:>12d} "
          f"{sup_pca_stats['components_for_95pct']:>12d}")
    print(f"{'CKA':<30s} {cka:>12.4f} {'-':>12s}")
    print(f"{'subspace mean alignment':<30s} {sa['mean_alignment']:>12.4f} {'-':>12s}")
    print(f"{'velocity mean':<30s} "
          f"{traj_comparison['jepa_velocity_mean']:>12.4f} "
          f"{traj_comparison['sup_velocity_mean']:>12.4f}")
    print(f"{'curvature mean':<30s} "
          f"{traj_comparison['jepa_curvature_mean']:>12.4f} "
          f"{traj_comparison['sup_curvature_mean']:>12.4f}")
    if sae_overlap_result:
        print(f"{'SAE mean cosine':<30s} "
              f"{sae_overlap_result['mean_cosine']:>12.4f} {'-':>12s}")
        print(f"{'SAE frac stable':<30s} "
              f"{sae_overlap_result['frac_stable']:>12.4f} {'-':>12s}")

    # -- Save results -------------------------------------------------------
    results_dir = jepa_exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_output = {
        "jepa_exp": args.jepa_exp,
        "sup_exp": args.sup_exp,
        "n_matched": N_matched,
        "embed_dim": D,
        "top_k": top_k,
        "rank": args.rank,
        "pca_stats": {
            "jepa": jepa_pca_stats,
            "supervised": sup_pca_stats,
        },
        "cka": cka,
        "subspace_alignment": sa,
        "sae_feature_overlap": sae_overlap_result,
        "trajectory_comparison": traj_comparison,
        "probe_comparison": probe_comparison,
        "label_subspace_alignment": label_sub_alignment,
    }

    json_path = results_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=float)
    print(f"\nScalar results -> {json_path}")

    # NPZ: arrays for notebook plots
    npz_data: dict[str, np.ndarray] = {
        "jepa_velocities": jv_flat,
        "sup_velocities": sv_flat,
        "jepa_curvatures": jc_flat,
        "sup_curvatures": sc_flat,
        "label_alignment_matrix": alignment_matrix,
        "label_names": np.array(aligned_labels),
    }
    if sae_matched_cosines is not None:
        npz_data["sae_matched_cosines"] = sae_matched_cosines

    npz_path = results_dir / "comparison.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Array results  -> {npz_path}")


if __name__ == "__main__":
    main()
