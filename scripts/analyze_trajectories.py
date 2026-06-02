#!/usr/bin/env python3
"""
Trajectory analysis script for JEPA patient embeddings.

Loads embeddings (from a saved .npz or via load_scaffolding), computes
trajectory geometry (velocity, curvature, arc length, concept drift),
runs prospective trajectory probes per label, and saves structured
results (scalars to JSON, large arrays to NPZ).

Usage
-----
  python -m scripts.analyze_trajectories --exp stopg_42_v01 --emb embeddings_40.npz
  python -m scripts.analyze_trajectories --exp stopg_42_v01 --ckpt checkpoint_100.pt
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

import numpy as np
import torch

from src.analysis.eval_infra import (
    extract_jepa_embeddings, compute_derived_vectors,
    load_escalation_labels, load_label_30d_at_k,
    extract_icd_block_targets)
from src.training.utils.inference import load_scaffolding
from src.analysis.trajectories import (
    extract_trajectories,
    trajectory_velocity, trajectory_temporal_velocity,
    trajectory_curvature, trajectory_arc_length,
    concept_centroid, drift_toward_concept, temporal_drift_rate,
    prospective_trajectory_probe)
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
    """Extract days_since_first for each (subject_id, mask_pos) sample.

    Falls back to encounter index if days_since_first is not available.
    """
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


def _build_labels_per_step(
    sample_labels: np.ndarray,
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
    traj_dict: dict,
) -> np.ndarray:
    """Map sample-level labels (N,) into trajectory-aligned (P, T_max) matrix.

    Entries without a matching sample are set to -1 (ignore).
    """
    patient_ids = traj_dict["patient_ids"]    # (P,) str
    T_max = traj_dict["trajectories"].shape[1]
    P = len(patient_ids)

    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}

    # We need mask_pos → trajectory step mapping per patient.
    # extract_trajectories sorts by mask_pos and assigns sequential steps.
    # Rebuild that mapping.
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
    parser = argparse.ArgumentParser(description="Trajectory analysis for JEPA embeddings")
    parser.add_argument("--exp", type=str, required=True,
                        help="Run-id of a completed run (under artifacts/training-runs/)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emb", type=str, default=None,
                       help="Embeddings .npz file name (e.g. embeddings_40.npz)")
    group.add_argument("--ckpt", type=str, default=None,
                       help="Checkpoint .pt file to extract embeddings from")
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

    z_enc_pooled = emb["z_enc_pooled"]     # (N, D)
    subject_ids = emb["subject_ids"]       # (N,)
    mask_pos = emb["mask_pos"]             # (N,)

    N, D = z_enc_pooled.shape
    print(f"Samples: {N}, Dim: {D}, Patients: {len(np.unique(subject_ids))}")

    # -- Load patient data for labels and times --------------------------------
    sequences_path = DATA_DIR / "sequences.jsonl"
    patients_dict = load_sequences_dict(sequences_path)
    times = _encounter_times(patients_dict, subject_ids, mask_pos)

    # -- Extract trajectories --------------------------------------------------
    traj_dict = extract_trajectories(z_enc_pooled, subject_ids, mask_pos, times=times)
    P, T_max = traj_dict["trajectories"].shape[:2]
    print(f"Trajectories: {P} patients, T_max={T_max}")

    # -- Geometric primitives --------------------------------------------------
    vel, vel_mask = trajectory_velocity(traj_dict)
    temp_vel, _ = trajectory_temporal_velocity(traj_dict)
    curv, curv_mask = trajectory_curvature(traj_dict)
    arc_lengths = trajectory_arc_length(traj_dict)

    vel_mag = np.linalg.norm(vel, axis=-1)  # (P, T_max-1)
    vel_mag[~vel_mask] = np.nan
    temp_vel_mag = np.linalg.norm(temp_vel, axis=-1)
    temp_vel_mag[~vel_mask] = np.nan

    print(f"  Velocity:  median={np.nanmedian(vel_mag):.4f}, "
          f"mean={np.nanmean(vel_mag):.4f}")
    print(f"  Curvature: median={np.nanmedian(curv):.4f}")
    print(f"  Arc length: median={np.median(arc_lengths):.4f}")

    # -- Labels ----------------------------------------------------------------
    label_esc = load_escalation_labels(patients_dict, subject_ids, mask_pos)
    label_30d = load_label_30d_at_k(patients_dict, subject_ids, mask_pos)
    icd_targets, icd_chapters = extract_icd_block_targets(
        sequences_path, subject_ids, mask_pos)

    labels_dict = {
        "escalation": label_esc,
        "30d_readmit": label_30d,
    }
    # Add ICD blocks with enough positives
    for ch_idx, ch_name in enumerate(icd_chapters):
        ch_labels = icd_targets[:, ch_idx]
        if ch_labels.sum() >= 10:
            labels_dict[f"icd_{ch_name}"] = ch_labels

    print(f"  Labels: {list(labels_dict.keys())}")

    # -- Concept centroids per label -------------------------------------------
    centroids: dict[str, np.ndarray] = {}
    centroid_results: dict[str, dict] = {}
    for label_name, label_vec in labels_dict.items():
        cc = concept_centroid(z_enc_pooled, label_vec)
        centroids[label_name] = cc["mean"]
        centroid_results[label_name] = {
            "n_positive": cc["n_positive"],
            "centroid_norm": float(np.linalg.norm(cc["mean"])),
        }

    # -- Drift toward each concept centroid ------------------------------------
    drift_arrays: dict[str, np.ndarray] = {}
    drift_summaries: dict[str, dict] = {}
    for label_name, centroid_mean in centroids.items():
        drift, drift_mask = drift_toward_concept(traj_dict, centroid_mean)
        drift_safe = drift.copy()
        drift_safe[~drift_mask] = np.nan
        drift_arrays[label_name] = drift

        tdrift, _ = temporal_drift_rate(traj_dict, centroid_mean)
        tdrift_safe = tdrift.copy()
        tdrift_safe[~drift_mask] = np.nan

        drift_summaries[label_name] = {
            "drift_mean": float(np.nanmean(drift_safe)),
            "drift_std": float(np.nanstd(drift_safe)),
            "temporal_drift_mean": float(np.nanmean(tdrift_safe)),
            "temporal_drift_std": float(np.nanstd(tdrift_safe)),
        }

    # -- Prospective trajectory probes -----------------------------------------
    probe_results: dict[str, dict] = {}
    # Focus probes on the main clinical labels
    probe_labels = {k: v for k, v in labels_dict.items()
                    if not k.startswith("icd_")}

    for label_name, label_vec in probe_labels.items():
        labels_per_step = _build_labels_per_step(
            label_vec, subject_ids, mask_pos, traj_dict)

        centroid_mean = centroids.get(label_name)

        result = prospective_trajectory_probe(
            traj_dict,
            labels_per_step,
            baseline_z_enc=traj_dict["trajectories"],
            centroid_mean=centroid_mean,
            n_splits=5)
        probe_results[label_name] = result
        print(f"  Probe [{label_name}]: traj={result['traj_auroc']:.4f}, "
              f"base={result['baseline_auroc']:.4f}, "
              f"delta={result['delta_auroc']:+.4f} "
              f"(n={result['n_samples']})")

    # -- Summary table ---------------------------------------------------------
    print(f"\n{'label':<20s} {'traj_auroc':>12s} {'baseline':>12s} {'delta':>12s}")
    print("-" * 58)
    for label_name, res in probe_results.items():
        print(f"{label_name:<20s} "
              f"{res['traj_auroc']:>12.4f} "
              f"{res['baseline_auroc']:>12.4f} "
              f"{res['delta_auroc']:>+12.4f}")

    # -- Save results ----------------------------------------------------------
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON: scalars and per-label probe results
    json_output = {
        "exp": args.exp,
        "n_samples": N,
        "n_patients": P,
        "T_max": T_max,
        "embed_dim": D,
        "geometry": {
            "velocity_median": float(np.nanmedian(vel_mag)),
            "velocity_mean": float(np.nanmean(vel_mag)),
            "curvature_median": float(np.nanmedian(curv)),
            "arc_length_median": float(np.median(arc_lengths)),
            "arc_length_mean": float(np.mean(arc_lengths)),
        },
        "centroids": centroid_results,
        "drift": drift_summaries,
        "probes": {
            label_name: {
                "traj_auroc": res["traj_auroc"],
                "baseline_auroc": res["baseline_auroc"],
                "delta_auroc": res["delta_auroc"],
                "n_samples": res["n_samples"],
                "feature_names": res["feature_names"],
            }
            for label_name, res in probe_results.items()
        },
    }

    json_path = results_dir / "trajectories.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=float)
    print(f"\nScalar results -> {json_path}")

    # NPZ: large arrays
    npz_data = {
        "trajectories": traj_dict["trajectories"],       # (P, T_max, D)
        "validity_mask": traj_dict["validity_mask"],      # (P, T_max)
        "patient_ids": traj_dict["patient_ids"],          # (P,)
        "times": traj_dict["times"],                      # (P, T_max)
        "velocity": vel,                                  # (P, T_max-1, D)
        "velocity_mask": vel_mask,                        # (P, T_max-1)
        "velocity_magnitude": vel_mag,                    # (P, T_max-1)
        "temporal_velocity_magnitude": temp_vel_mag,      # (P, T_max-1)
        "curvature": curv,                                # (P, T_max-2)
        "curvature_mask": curv_mask,                      # (P, T_max-2)
        "arc_lengths": arc_lengths,                       # (P,)
        # Per-label drift arrays
        **{f"drift_{name}": arr for name, arr in drift_arrays.items()},
        # Sample-level labels for the notebook
        "label_escalation": label_esc,                    # (N,)
        "label_30d_readmit": label_30d,                   # (N,)
        "subject_ids": subject_ids,                       # (N,)
        "mask_pos": mask_pos,                             # (N,)
        "z_enc_pooled": z_enc_pooled,                     # (N, D)
    }
    npz_path = results_dir / "trajectories.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Array results  -> {npz_path}")


if __name__ == "__main__":
    main()
