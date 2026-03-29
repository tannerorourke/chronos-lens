"""
SAE analysis utilities - activation extraction, open-ended feature
inspection, and SAE-cluster cross-reference.

Tier C of the three-tier partial labeling bridge (thesis §5.5).

Functions
---------
  extract_sae_activations : run a trained SAE on delta → sparse activations
  load_sae                : reconstruct a SparseAutoencoder from checkpoint
  inspect_sae_features    : open-ended enrichment against raw ICD/med vocabulary
  sae_cluster_crossref    : cross-reference SAE features with HDBSCAN clusters
"""

from pathlib import Path
from collections import Counter

import numpy as np
import torch

from src.models.sae import SparseAutoencoder
from src.utils.io import load_sequences_dict


from src.utils.seed import SEED


# =========================================================================
# Loading & extraction
# =========================================================================

def load_sae(checkpoint_path: Path, device: torch.device = torch.device("cpu")) -> SparseAutoencoder:
    """Load a trained SAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SparseAutoencoder(
        embed_dim=ckpt["embed_dim"],
        n_features=ckpt["n_features"],
        top_k=ckpt["top_k"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract_sae_activations(
    model: SparseAutoencoder,
    vec: np.ndarray,
) -> np.ndarray:
    """Run a trained SAE on displacement vectors and return sparse activations.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(vec, dtype=torch.float32, device=device)
        _, activations = model(x)
    return activations.cpu().numpy()


# =========================================================================
# feature inspection
# =========================================================================

def _odds_ratio(freq_group: float, freq_pop: float) -> float:
    """Compute odds ratio, clamped to avoid division by zero."""
    p_g = np.clip(freq_group, 1e-10, 1 - 1e-10)
    p_p = np.clip(freq_pop, 1e-10, 1 - 1e-10)
    return (p_g / (1 - p_g)) / (p_p / (1 - p_p))


def inspect_sae_features(
    sae_activations: np.ndarray,
    subject_ids: np.ndarray,
    sequences_path,
    cluster_labels: np.ndarray | None = None,
    metadata_features: np.ndarray | None = None,
    metadata_feature_names: list | None = None,
    top_n_samples: int = 50,
    top_n_enriched: int = 10,
    min_activation_frac: float = 0.01,
) -> list[dict]:
    """Open-ended inspection of SAE features against raw clinical data.

    For each non-dead feature that activates on >= min_activation_frac of
    samples, pulls top activators and computes enrichment over the FULL
    raw ICD/medication vocabulary from sequences.jsonl.

    Parameters
    ----------
    sae_activations        : (N, n_features) sparse activation matrix
    subject_ids            : (N,) subject IDs per sample
    sequences_path         : path to sequences.jsonl
    cluster_labels         : (N,) HDBSCAN cluster assignments (optional)
    metadata_features      : (N, n_meta) sample-level metadata (optional)
    metadata_feature_names : column names for metadata (optional)
    top_n_samples          : max samples to use as "top activators"
    top_n_enriched         : max enriched codes/meds to report per feature
    min_activation_frac    : minimum fraction of samples a feature must
                             activate on to be inspected

    Returns
    -------
    list of feature card dicts, one per feature
    """
    N, n_features = sae_activations.shape
    patients = load_sequences_dict(sequences_path)
    subject_ids = np.asarray(subject_ids, dtype=str)

    # Pre-compute population-level ICD and med frequencies
    # For each sample, collect its ICD codes and meds (from the patient record)
    sample_icd_sets: list[set] = []
    sample_med_sets: list[set] = []

    for sid in subject_ids:
        p = patients.get(str(sid))
        if p is None:
            sample_icd_sets.append(set())
            sample_med_sets.append(set())
            continue
        icds = set()
        meds = set()
        for enc in p.get("encounters", []):
            icds.update(str(c) for c in enc.get("icd_codes", []))
            meds.update(str(m).lower() for m in enc.get("meds", []))
        sample_icd_sets.append(icds)
        sample_med_sets.append(meds)

    # Population frequencies (fraction of samples with each code/med)
    all_icds: Counter = Counter()
    all_meds: Counter = Counter()
    for icds in sample_icd_sets:
        for c in icds:
            all_icds[c] += 1
    for meds in sample_med_sets:
        for m in meds:
            all_meds[m] += 1

    pop_icd_freq = {c: count / N for c, count in all_icds.items()}
    pop_med_freq = {m: count / N for m, count in all_meds.items()}

    # Population temporal stats
    pop_intervals = []
    pop_n_encounters = []
    pop_med_burden = []
    for sid in subject_ids:
        p = patients.get(str(sid))
        if p is None:
            continue
        encs = p.get("encounters", [])
        pop_n_encounters.append(len(encs))
        pop_med_burden.append(
            np.mean([len(enc.get("meds", [])) for enc in encs]) if encs else 0
        )
        if len(encs) >= 2:
            from datetime import datetime
            times = []
            for enc in encs:
                t = enc.get("admittime")
                if isinstance(t, str):
                    try:
                        times.append(datetime.fromisoformat(t))
                    except (ValueError, TypeError):
                        pass
            times.sort()
            if len(times) >= 2:
                gaps = [(times[i+1] - times[i]).total_seconds() / 86400
                        for i in range(len(times) - 1)]
                pop_intervals.extend(gaps)

    pop_mean_interval = float(np.mean(pop_intervals)) if pop_intervals else 0.0
    pop_mean_encounters = float(np.mean(pop_n_encounters)) if pop_n_encounters else 0.0
    pop_mean_med_burden = float(np.mean(pop_med_burden)) if pop_med_burden else 0.0

    # Inspect each feature
    feature_cards: list[dict] = []
    active_mask = sae_activations != 0  # (N, n_features)

    for feat_idx in range(n_features):
        feat_active = active_mask[:, feat_idx]
        n_active = int(feat_active.sum())
        activation_frac = n_active / N

        if activation_frac < min_activation_frac:
            continue

        # Top activating samples (by activation magnitude)
        feat_vals = sae_activations[:, feat_idx]
        n_top = min(top_n_samples, max(1, int(N * 0.05)))
        n_top = min(n_top, top_n_samples)
        top_indices = np.argsort(feat_vals)[::-1][:n_top]

        # ICD enrichment
        top_icd_counts: Counter = Counter()
        for idx in top_indices:
            for c in sample_icd_sets[idx]:
                top_icd_counts[c] += 1
        icd_enrichment = []
        for code, count in top_icd_counts.items():
            freq_group = count / len(top_indices)
            freq_pop = pop_icd_freq.get(code, 1e-10)
            odds = _odds_ratio(freq_group, freq_pop)
            icd_enrichment.append({"code": code, "odds_ratio": round(odds, 2),
                                   "freq_group": round(freq_group, 3),
                                   "freq_pop": round(freq_pop, 3)})
        icd_enrichment.sort(key=lambda x: x["odds_ratio"], reverse=True)
        top_icd = icd_enrichment[:top_n_enriched]

        # Med enrichment
        top_med_counts: Counter = Counter()
        for idx in top_indices:
            for m in sample_med_sets[idx]:
                top_med_counts[m] += 1
        med_enrichment = []
        for med, count in top_med_counts.items():
            freq_group = count / len(top_indices)
            freq_pop = pop_med_freq.get(med, 1e-10)
            odds = _odds_ratio(freq_group, freq_pop)
            med_enrichment.append({"med": med, "odds_ratio": round(odds, 2),
                                   "freq_group": round(freq_group, 3),
                                   "freq_pop": round(freq_pop, 3)})
        med_enrichment.sort(key=lambda x: x["odds_ratio"], reverse=True)
        top_meds = med_enrichment[:top_n_enriched]

        # Temporal stats for top activators
        from datetime import datetime as dt
        top_intervals = []
        top_n_enc = []
        top_med_burd = []
        for idx in top_indices:
            sid = subject_ids[idx]
            p = patients.get(str(sid))
            if p is None:
                continue
            encs = p.get("encounters", [])
            top_n_enc.append(len(encs))
            top_med_burd.append(
                np.mean([len(enc.get("meds", [])) for enc in encs]) if encs else 0
            )
            times = []
            for enc in encs:
                t = enc.get("admittime")
                if isinstance(t, str):
                    try:
                        times.append(dt.fromisoformat(t))
                    except (ValueError, TypeError):
                        pass
            times.sort()
            if len(times) >= 2:
                gaps = [(times[i+1] - times[i]).total_seconds() / 86400
                        for i in range(len(times) - 1)]
                top_intervals.extend(gaps)

        temporal = {
            "mean_interval_days": round(float(np.mean(top_intervals)), 1) if top_intervals else None,
            "pop_mean_interval_days": round(pop_mean_interval, 1),
            "mean_n_encounters": round(float(np.mean(top_n_enc)), 1) if top_n_enc else None,
            "pop_mean_n_encounters": round(pop_mean_encounters, 1),
            "mean_med_burden": round(float(np.mean(top_med_burd)), 1) if top_med_burd else None,
            "pop_mean_med_burden": round(pop_mean_med_burden, 1),
        }

        # Cluster overlap (if cluster labels provided)
        cluster_overlap = None
        if cluster_labels is not None:
            top_clusters = cluster_labels[top_indices]
            cluster_counts = Counter(int(c) for c in top_clusters)
            total_top = len(top_indices)
            cluster_overlap = {
                str(cid): round(count / total_top, 3)
                for cid, count in cluster_counts.most_common()
            }

        # Metadata correlation (if provided)
        metadata_corr = None
        if metadata_features is not None and metadata_feature_names is not None:
            from scipy import stats as sp_stats
            corrs = []
            for meta_idx, meta_name in enumerate(metadata_feature_names):
                meta_col = metadata_features[:, meta_idx]
                if np.std(meta_col) < 1e-10 or np.std(feat_vals) < 1e-10:
                    continue
                r, p = sp_stats.pearsonr(feat_vals, meta_col)
                corrs.append({"feature": meta_name, "r": round(float(r), 3),
                              "p": float(p)})
            corrs.sort(key=lambda x: abs(x["r"]), reverse=True)
            metadata_corr = corrs[:5]

        card = {
            "feature_idx": feat_idx,
            "activation_frac": round(activation_frac, 4),
            "n_active_samples": n_active,
            "top_enriched_icd": top_icd,
            "top_enriched_meds": top_meds,
            "temporal": temporal,
            "cluster_overlap": cluster_overlap,
            "metadata_correlation": metadata_corr,
        }
        feature_cards.append(card)

    return feature_cards


# =========================================================================
# SAE-Cluster cross-reference
# =========================================================================

def sae_cluster_crossref(
    sae_activations: np.ndarray,
    cluster_labels: np.ndarray,
    top_n_samples: int = 50,
    concentration_threshold: float = 0.60,
) -> dict:
    """Cross-reference SAE features with HDBSCAN cluster assignments.

    For each active SAE feature, compute the distribution of its top
    activators across clusters.  Classify the relationship:
      - single cluster (>threshold in one cluster)
      - multi-cluster (spread across 3+)

    For each cluster, find which SAE features are most active in it.

    Parameters
    ----------
    sae_activations          : (N, n_features) sparse activations
    cluster_labels           : (N,) HDBSCAN cluster labels (-1 = noise)
    top_n_samples            : how many top activators to consider per feature
    concentration_threshold  : fraction in one cluster to be "concentrated"

    Returns
    -------
    dict with:
      feature_cluster_map     : per-feature cluster distribution
      cluster_feature_map     : per-cluster active features
      heatmap                 : (n_active_features, n_clusters) mean activation
      feature_indices         : which features are in the heatmap rows
      cluster_ids             : which clusters are in the heatmap columns
      summary                 : narrative counts
    """
    N, n_features = sae_activations.shape
    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    n_clusters = len(cluster_ids)

    # Mean activation per cluster (all features × all clusters)
    heatmap_full = np.zeros((n_features, n_clusters))
    for j, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        if mask.sum() > 0:
            heatmap_full[:, j] = sae_activations[mask].mean(axis=0)

    # Active features: activate on >0 samples
    active_mask = (sae_activations != 0).sum(axis=0) > 0
    active_indices = np.where(active_mask)[0]

    heatmap = heatmap_full[active_indices]

    # Per-feature: cluster distribution of top activators
    feature_cluster_map = {}
    n_single_cluster = 0
    n_multi_cluster = 0

    for feat_idx in active_indices:
        feat_vals = sae_activations[:, feat_idx]
        n_top = min(top_n_samples, max(1, int(N * 0.05)))
        n_top = min(n_top, top_n_samples)
        top_indices = np.argsort(feat_vals)[::-1][:n_top]
        top_clusters = cluster_labels[top_indices]

        cluster_dist = Counter(int(c) for c in top_clusters)
        total_top = len(top_indices)
        dist_frac = {
            str(cid): round(count / total_top, 3)
            for cid, count in cluster_dist.most_common()
        }

        max_frac = max(dist_frac.values()) if dist_frac else 0
        n_clusters_hit = sum(1 for v in dist_frac.values() if v > 0.05)

        if max_frac >= concentration_threshold:
            relationship = "single_cluster"
            n_single_cluster += 1
        elif n_clusters_hit >= 3:
            relationship = "multi_cluster"
            n_multi_cluster += 1
        else:
            relationship = "moderate"

        feature_cluster_map[int(feat_idx)] = {
            "cluster_distribution": dist_frac,
            "max_concentration": round(max_frac, 3),
            "n_clusters_hit": n_clusters_hit,
            "relationship": relationship,
        }

    # Per-cluster: which features are most active?
    cluster_feature_map = {}
    n_multi_feature_clusters = 0
    for j, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        if mask.sum() == 0:
            continue
        mean_act = sae_activations[mask].mean(axis=0)
        top_feats = np.argsort(mean_act)[::-1][:10]
        active_feats = [int(f) for f in top_feats if mean_act[f] > 0]
        cluster_feature_map[int(cid)] = {
            "top_features": active_feats,
            "n_active_features": len(active_feats),
        }
        if len(active_feats) >= 3:
            n_multi_feature_clusters += 1

    summary = {
        "n_active_features": int(len(active_indices)),
        "n_clusters": n_clusters,
        "n_features_single_cluster": n_single_cluster,
        "n_features_multi_cluster": n_multi_cluster,
        "n_multi_feature_clusters": n_multi_feature_clusters,
        "interpretation": (
            f"{n_single_cluster} features map to single clusters (SAE and UMAP agree). "
            f"{n_multi_cluster} features cut across clusters (SAE finds finer structure). "
            f"{n_multi_feature_clusters} clusters contain 3+ active features "
            f"(UMAP is coarser than SAE)."
        ),
    }

    return {
        "feature_cluster_map": feature_cluster_map,
        "cluster_feature_map": cluster_feature_map,
        "heatmap": heatmap,
        "feature_indices": active_indices.tolist(),
        "cluster_ids": [int(c) for c in cluster_ids],
        "summary": summary,
    }


# =========================================================================
# Validation of dictionary directions across seed (as done in Anthropic papers)
# =========================================================================


def sae_seed_stability(
    checkpoint_paths: list[Path],
    cosine_threshold: float = 0.85,
    device: str = "cpu",
) -> dict:
    """Compare SAE dictionary directions across seeds via Hungarian matching.

    Args:
        checkpoint_paths: Paths to SAE checkpoint .pt files (one per seed).
        cosine_threshold: Cosine similarity above which a matched pair is "stable".
        device: Device for loading checkpoints.

    Returns:
        Dict with per-pair and aggregate stability metrics.
    """
    import torch
    from scipy.optimize import linear_sum_assignment
    from itertools import combinations

    # Load decoder weights: each is (embed_dim, n_features), columns = dictionary directions
    decoders = []
    for p in checkpoint_paths:
        ckpt = torch.load(p, map_location=device)
        # Adjust key if your checkpoint wraps in a "model" dict
        state = ckpt if isinstance(ckpt, dict) and "decoder.weight" in ckpt else ckpt.get("model", ckpt)
        W = state["decoder.weight"]  # (embed_dim, n_features)
        # L2-normalize columns (each dictionary direction)
        W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
        decoders.append(W.float().cpu())

    pair_results = {}
    for (i, Wi), (j, Wj) in combinations(enumerate(decoders), 2):
        # Cosine similarity matrix: (n_features_i, n_features_j)
        cos_sim = (Wi.T @ Wj).numpy()

        # Hungarian on cost = 1 - cosine
        row_idx, col_idx = linear_sum_assignment(1.0 - cos_sim)
        matched_cosines = cos_sim[row_idx, col_idx]

        stable_mask = matched_cosines > cosine_threshold
        pair_results[f"seed_{i}_vs_{j}"] = {
            "n_features": (Wi.shape[1], Wj.shape[1]),
            "matched_cosines": matched_cosines.tolist(),
            "mean_cosine": float(matched_cosines.mean()),
            "median_cosine": float(np.median(matched_cosines)),
            "frac_stable": float(stable_mask.mean()),
            "n_stable": int(stable_mask.sum()),
            "n_matched": len(matched_cosines),
        }

    # Aggregate across all pairs
    all_cosines = np.concatenate([
        r["matched_cosines"] for r in pair_results.values()
    ])
    summary = {
        "pairs": pair_results,
        "aggregate": {
            "mean_cosine": float(all_cosines.mean()),
            "median_cosine": float(np.median(all_cosines)),
            "frac_stable": float((all_cosines > cosine_threshold).mean()),
            "cosine_threshold": cosine_threshold,
            "n_checkpoints": len(checkpoint_paths),
        },
    }
    return summary