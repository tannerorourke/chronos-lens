"""
SAE analysis utilities.

Primary workflow: label-driven feature identification, then clinical content
inspection for interpretation.

Functions
---------
  load_sae                    : reconstruct a SparseAutoencoder from checkpoint
  extract_sae_activations     : run a trained SAE on a latent vector -> sparse activations
  sae_label_enrichment        : per-feature Fisher exact test + BH FDR correction
  feature_label_specificity   : (n_features, n_labels) lift matrix
  sae_coactivation_matrix     : (n_features, n_features) normalized co-activation lift
  sae_temporal_enrichment     : per-feature temporal correlation and quartile activation
  inspect_sae_feature_content : clinical content enrichment (secondary, post-identification)
  sae_cluster_crossref        : cross-reference SAE features with HDBSCAN clusters
  sae_seed_stability          : dictionary direction stability across seeds
  decompose_patient           : per-patient SAE decomposition
"""

from pathlib import Path
from collections import Counter

import numpy as np
import torch

from src.analysis.eval_infra import odds_ratio
from src.models.sae import SparseAutoencoder
from src.utils.io import load_sequences_dict
from src.mimic.helper import parse_dt

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
    """Run a trained SAE on a latent vector and return sparse activations."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(vec, dtype=torch.float32, device=device)
        _, activations = model(x)
    return activations.cpu().numpy()


# =========================================================================
# Label-driven feature analysis (primary workflow)
# =========================================================================

def sae_label_enrichment(
    activations: np.ndarray,
    labels: np.ndarray,
    label_name: str = "",
) -> list[dict]:
    """Per-feature Fisher exact test with Benjamini-Hochberg FDR correction.

    For each feature, builds a 2x2 contingency table (feature active/inactive
    vs label positive/negative) and runs a one-sided Fisher exact test for
    enrichment among active samples.

    Parameters
    ----------
    activations : (N, n_features) sparse activation matrix
    labels      : (N,) binary labels (0/1)
    label_name  : optional label name for context

    Returns
    -------
    list of dicts (one per feature with activation_frac >= 0.01):
        feature_idx, odds_ratio, p_value, fdr_q, n_active,
        n_pos_active, activation_frac
    """
    from scipy.stats import fisher_exact
    N, n_features = activations.shape
    labels = np.asarray(labels, dtype=int)
    active_mask = activations != 0  # (N, n_features)

    raw_results: list[tuple] = []
    for feat_idx in range(n_features):
        feat_on = active_mask[:, feat_idx]
        n_active = int(feat_on.sum())
        activation_frac = n_active / N
        if activation_frac < 0.01:
            continue

        a = int((feat_on & (labels == 1)).sum())   # active + pos
        b = int((feat_on & (labels == 0)).sum())   # active + neg
        c = int((~feat_on & (labels == 1)).sum())  # inactive + pos
        d = int((~feat_on & (labels == 0)).sum())  # inactive + neg

        oddsratio, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        raw_results.append((feat_idx, oddsratio, p, a, n_active, activation_frac))

    # Benjamini-Hochberg FDR
    raw_results.sort(key=lambda x: x[2])  # sort by p-value ascending
    m = len(raw_results)
    fdr_q_vals = np.ones(m)
    for rank, (_, _, p, _, _, _) in enumerate(raw_results, 1):
        fdr_q_vals[rank - 1] = p * m / rank
    # enforce monotonicity (step-up)
    for i in range(m - 2, -1, -1):
        fdr_q_vals[i] = min(fdr_q_vals[i], fdr_q_vals[i + 1])
    fdr_q_vals = np.clip(fdr_q_vals, 0, 1)

    result = []
    for i, (feat_idx, oddsratio, p, n_pos_active, n_active, act_frac) in enumerate(raw_results):
        result.append({
            "feature_idx": feat_idx,
            "odds_ratio": round(float(oddsratio), 4),
            "p_value": float(p),
            "fdr_q": round(float(fdr_q_vals[i]), 6),
            "n_active": n_active,
            "n_pos_active": n_pos_active,
            "activation_frac": round(act_frac, 4),
        })
    return result


def feature_label_specificity(
    activations: np.ndarray,
    labels_dict: dict[str, np.ndarray],
) -> dict:
    """Lift matrix: P(label=1 | feature active) / P(label=1).

    Parameters
    ----------
    activations : (N, n_features) sparse activation matrix
    labels_dict : {label_name: (N,) binary array}

    Returns
    -------
    dict with:
        lift_matrix     : (n_features, n_labels) lift values (non-dead features only)
        label_names     : column order
        feature_indices : row order (non-dead features only)
    """
    N, n_features = activations.shape
    active_mask = activations != 0
    label_names = sorted(labels_dict.keys())
    n_labels = len(label_names)

    # Active features only
    feat_active_counts = active_mask.sum(axis=0)
    active_feat_idx = np.where(feat_active_counts > 0)[0]
    n_active = len(active_feat_idx)

    lift_matrix = np.zeros((n_active, n_labels), dtype=np.float64)

    for j, lname in enumerate(label_names):
        lbl = np.asarray(labels_dict[lname], dtype=int)
        p_label = lbl.mean()
        if p_label < 1e-10:
            continue
        for row, feat_idx in enumerate(active_feat_idx):
            feat_on = active_mask[:, feat_idx]
            n_on = int(feat_on.sum())
            p_label_given_active = lbl[feat_on].mean() if n_on > 0 else 0.0
            lift_matrix[row, j] = p_label_given_active / p_label

    return {
        "lift_matrix": lift_matrix,
        "label_names": label_names,
        "feature_indices": active_feat_idx.tolist(),
    }


def sae_coactivation_matrix(
    activations: np.ndarray,
) -> dict:
    """Normalized co-activation lift between feature pairs.

    Entry (i, j) = P(i active AND j active) / (P(i active) * P(j active)).
    Values > 1 indicate features that co-fire more than chance.

    Parameters
    ----------
    activations : (N, n_features) sparse activation matrix

    Returns
    -------
    dict with:
        lift_matrix     : (n_active, n_active) symmetric lift values, diagonal = 1
        feature_indices : which original feature indices are included (non-dead)
    """
    N, n_features = activations.shape
    active_mask = (activations != 0).astype(np.float64)
    feat_freq = active_mask.mean(axis=0)  # P(feature active)
    active_feat_idx = np.where(feat_freq > 0)[0]
    n_active = len(active_feat_idx)

    sub = active_mask[:, active_feat_idx]  # (N, n_active)
    sub_freq = feat_freq[active_feat_idx]  # (n_active,)

    # Co-occurrence matrix: (n_active, n_active)
    cooccur = (sub.T @ sub) / N  # P(i AND j)
    expected = np.outer(sub_freq, sub_freq)  # P(i) * P(j)
    expected = np.clip(expected, 1e-12, None)

    lift_matrix = cooccur / expected
    np.fill_diagonal(lift_matrix, 1.0)

    return {
        "lift_matrix": lift_matrix,
        "feature_indices": active_feat_idx.tolist(),
    }


def sae_temporal_enrichment(
    activations: np.ndarray,
    times: np.ndarray,
    rel_times: np.ndarray,
) -> list[dict]:
    """Per-feature temporal enrichment analysis.

    For each non-dead feature (activation_frac >= 0.01), computes correlation
    of activation magnitude with absolute and relative time, plus activation
    rates in the first and last quartiles of the time distribution.

    Parameters
    ----------
    activations : (N, n_features) sparse activation matrix
    times       : (N,) absolute encounter times (e.g. days_since_first)
    rel_times   : (N,) relative times (e.g. days since previous encounter)

    Returns
    -------
    list of dicts per feature:
        feature_idx, time_corr, rel_time_corr,
        early_activation_frac, late_activation_frac
    """
    from scipy import stats

    N, n_features = activations.shape
    active_mask = activations != 0

    # Quartile boundaries for absolute time
    t_q1 = np.percentile(times, 25)
    t_q4 = np.percentile(times, 75)
    early_mask = times <= t_q1
    late_mask = times >= t_q4

    results = []
    for feat_idx in range(n_features):
        feat_on = active_mask[:, feat_idx]
        n_active = int(feat_on.sum())
        if n_active / N < 0.01:
            continue

        feat_vals = activations[:, feat_idx]

        # Pearson with absolute time
        if np.std(feat_vals) > 1e-10 and np.std(times) > 1e-10:
            r_time, _ = stats.pearsonr(feat_vals, times)
        else:
            r_time = 0.0

        # Pearson with relative time
        if np.std(feat_vals) > 1e-10 and np.std(rel_times) > 1e-10:
            r_rel, _ = stats.pearsonr(feat_vals, rel_times)
        else:
            r_rel = 0.0

        # Activation rates in quartiles
        early_frac = float(feat_on[early_mask].mean()) if early_mask.sum() > 0 else 0.0
        late_frac = float(feat_on[late_mask].mean()) if late_mask.sum() > 0 else 0.0

        results.append({
            "feature_idx": feat_idx,
            "time_corr": round(float(r_time), 4),
            "rel_time_corr": round(float(r_rel), 4),
            "early_activation_frac": round(early_frac, 4),
            "late_activation_frac": round(late_frac, 4),
        })

    return results


# =========================================================================
# feature content inspection (secondary - "what does this feature detect?")
# =========================================================================

def inspect_sae_feature_content(
    sae_activations: np.ndarray,
    subject_ids: np.ndarray,
    sequences_path,
    cluster_labels: np.ndarray | None = None,
    metadata_features: np.ndarray | None = None,
    metadata_feature_names: list | None = None,
    encounter_indices: np.ndarray | None = None,
    encounter_level: bool = False,
    top_n_samples: int = 50,
    top_n_enriched: int = 10,
    min_activation_frac: float = 0.01,
) -> list[dict]:
    """Open-ended inspection of SAE features against raw clinical data.

    For each non-dead feature that activates on >= min_activation_frac of
    samples, pulls top activators and computes enrichment over the raw
    ICD/medication vocabulary from sequences.jsonl.

    When encounter_level=True, ICD/med enrichment uses only the specific
    encounter for each sample (identified by encounter_indices) rather
    than aggregating across all encounters for the patient.

    Parameters
    ----------
    sae_activations        : (N, n_features) sparse activation matrix
    subject_ids            : (N,) subject IDs per sample
    sequences_path         : path to sequences.jsonl
    cluster_labels         : (N,) HDBSCAN cluster assignments (optional)
    metadata_features      : (N, n_meta) sample-level metadata (optional)
    metadata_feature_names : column names for metadata (optional)
    encounter_indices      : (N,) int encounter index per sample (optional,
                             required when encounter_level=True)
    encounter_level        : if True, use per-encounter ICD/med lookup
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

    for i, sid in enumerate(subject_ids):
        p = patients.get(str(sid))
        if p is None:
            sample_icd_sets.append(set())
            sample_med_sets.append(set())
            continue
        if encounter_level and encounter_indices is not None:
            enc_idx = int(encounter_indices[i])
            encs = p.get("encounters", [])
            enc = encs[enc_idx] if enc_idx < len(encs) else {}
            icds = set(str(c) for c in enc.get("icd_codes", []))
            meds = set(str(m).lower() for m in enc.get("meds", []))
        else:
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
            times = [parse_dt(enc.get("admittime")) for enc in encs]
            times = sorted(t for t in times if t is not None)
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
            odds = odds_ratio(freq_group, freq_pop)
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
            odds = odds_ratio(freq_group, freq_pop)
            med_enrichment.append({"med": med, "odds_ratio": round(odds, 2),
                                   "freq_group": round(freq_group, 3),
                                   "freq_pop": round(freq_pop, 3)})
        med_enrichment.sort(key=lambda x: x["odds_ratio"], reverse=True)
        top_meds = med_enrichment[:top_n_enriched]

        # Temporal stats for top activators
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
            times = [parse_dt(enc.get("admittime")) for enc in encs]
            times = sorted(t for t in times if t is not None)
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
            from scipy import stats
            corrs = []
            for meta_idx, meta_name in enumerate(metadata_feature_names):
                meta_col = metadata_features[:, meta_idx]
                if np.std(meta_col) < 1e-10 or np.std(feat_vals) < 1e-10:
                    continue
                r, p = stats.pearsonr(feat_vals, meta_col)
                corrs.append({"feature": meta_name, 
                              "r": round(float(r), 3), # type: ignore
                              "p": float(p)}) # type: ignore
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

    # Mean activation per cluster (all features x all clusters)
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


# =========================================================================
# SAE decomposition for a single patient
# =========================================================================

def decompose_patient(
    patient_idx: int,
    vectors_dict: dict[str, np.ndarray],
    sae_models_dict: dict[str, SparseAutoencoder],
    feature_cards_dict: dict[str, list[dict]],
    top_k: int = 8,
) -> dict:
    """Decompose one patient's vectors through their respective SAEs.

    Parameters
    ----------
    patient_idx    : row index into the (N, D) arrays
    vectors_dict   : {name: (N, D)} arrays (e.g. pred_error, z_pred, z_target)
    sae_models_dict    : {name: SparseAutoencoder} - trained, eval-mode
    feature_cards_dict : {name: [card, ...]} - from inspect_sae_features
    top_k          : max active features to return per vector

    Returns
    -------
    dict keyed by vector name, each value a list of dicts:
        {feature_idx, magnitude, label, decoder_direction}
    sorted descending by magnitude.
    """
    def _card_lookup(cards: list[dict]) -> dict[int, str | None]:
        lookup: dict[int, str | None] = {}
        for card in cards:
            idx = card["feature_idx"]
            label = None
            enriched = card.get("top_enriched_icd", [])
            if enriched:
                label = enriched[0].get("code")
            if label is None:
                enriched_meds = card.get("top_enriched_meds", [])
                if enriched_meds:
                    label = enriched_meds[0].get("med")
            lookup[idx] = label
        return lookup

    result = {}
    for name, sae in sae_models_dict.items():
        if name not in vectors_dict:
            continue
        vec = vectors_dict[name][patient_idx]
        cards = feature_cards_dict.get(name, [])
        card_lookup = _card_lookup(cards)

        device = next(sae.parameters()).device
        with torch.no_grad():
            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
            activations = sae.encode(x).squeeze(0).cpu().numpy()

        decoder_weight = sae.decoder.weight.detach().cpu().numpy()  # (embed_dim, n_features)

        active_idx = np.nonzero(activations)[0]
        magnitudes = activations[active_idx]
        order = np.argsort(-np.abs(magnitudes))[:top_k]
        active_idx = active_idx[order]
        magnitudes = magnitudes[order]

        features = []
        for fi, mag in zip(active_idx, magnitudes):
            features.append({
                "feature_idx": int(fi),
                "magnitude": float(mag),
                "label": card_lookup.get(int(fi)),
                "decoder_direction": decoder_weight[:, int(fi)].tolist(),
            })
        result[name] = features

    return result