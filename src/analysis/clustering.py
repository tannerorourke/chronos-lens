"""
Partial labeling bridge — Tiers A (LASSO) and B (UMAP + HDBSCAN cluster
enrichment) for the geometric latent space analysis (thesis §5.5).

Connects the model's geometric structure back to clinical concepts using
a rich metadata vocabulary (~60-80 features across 4 tiers).  The richer
the vocabulary, the stronger the "no clinical match" claim for
unexplained geometric structure.

Tier A (LASSO on PCA axes):
    Linear bridge — regress metadata against PC scores.
    Assumes geometry is organised along linear axes.
    Unexplained variance = 1 − R².

Tier B (UMAP + HDBSCAN cluster enrichment):
    Nonlinear bridge — cluster UMAP embedding with HDBSCAN, compute
    enrichment of metadata features per cluster.  Clusters with no
    clear enrichment are the mislabeling problem made visible.

Downstream consumers
--------------------
  lasso/                          → Tier A results
  clusters/                       → Tier B results
"""

import warnings

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN

from src.utils.seed import SEED, get_rng
from src.mimic.features import extract_metadata, is_binary  # noqa: F401
rng  = get_rng()

def _lasso_stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 100,
    subsample_frac: float = 0.5,
) -> tuple:
    """
    Bootstrap stability selection for LASSO.

    Returns
    -------
    sel_probs : (n_features,) selection probability [0, 1]
    ci_low    : (n_features,) 2.5th percentile of bootstrap coefficients
    ci_high   : (n_features,) 97.5th percentile of bootstrap coefficients
    """
    n_samples, n_features = X.shape
    sub_n   = max(4, int(n_samples * subsample_frac))
    counts  = np.zeros(n_features)
    coef_samples = []
    n_valid = 0

    for _ in range(n_iter):
        idx = rng.choice(n_samples, size=sub_n, replace=False)
        Xs, ys = X[idx], y[idx]
        if ys.std() < 1e-8:
            continue
        cv_folds = min(5, sub_n - 1)
        if cv_folds < 2:
            continue
        lasso = LassoCV(cv=cv_folds, random_state=SEED,
                        max_iter=5000, n_jobs=1)
        try:
            lasso.fit(Xs, ys)
            counts  += (np.abs(lasso.coef_) > 1e-10).astype(float)
            coef_samples.append(lasso.coef_.copy())
            n_valid += 1
        except Exception:
            pass

    sel_probs = counts / n_valid if n_valid > 0 else counts

    if coef_samples:
        coef_arr = np.array(coef_samples)
        ci_low  = np.percentile(coef_arr, 2.5, axis=0)
        ci_high = np.percentile(coef_arr, 97.5, axis=0)
    else:
        ci_low  = np.zeros(n_features)
        ci_high = np.zeros(n_features)

    return sel_probs, ci_low, ci_high


# =============================================================================
# Public utilities
# =============================================================================

def pool_to_patients(
    values: np.ndarray,
    subject_ids: np.ndarray,
    patient_ids: np.ndarray,
) -> np.ndarray:
    """Mean-pool sample-level values (N, ...) to patient level (P, ...)."""
    sid_str = np.asarray(subject_ids, dtype=str)
    return np.vstack([
        values[sid_str == pid].mean(axis=0)
        for pid in patient_ids
    ])


def broadcast_to_samples(
    metadata: np.ndarray,
    patient_ids: np.ndarray,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """Broadcast patient-level metadata (P, F) to sample-level (N, F)."""
    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}
    indices = np.array([pid_to_idx[str(sid)] for sid in subject_ids])
    return metadata[indices]


# =============================================================================
# Tier A: LASSO Bridge
# =============================================================================

def run_lasso_bridge(
    pc_projections: np.ndarray,
    metadata: np.ndarray,
    feature_names: list,
    top_k: int = 10,
    n_bootstrap: int = 100,
) -> dict:
    """
    Tier A — LASSO regression of clinical metadata against PC scores.

    For each top-k PC, regresses PC score ~ metadata features with LassoCV.
    Bootstrap stability selection identifies robustly selected features.

    Both pc_projections and metadata should be at patient level
    (use pool_to_patients() on sample-level PC projections first).

    Parameters
    ----------
    pc_projections : (n_patients, k) patient-level PC scores
    metadata       : (n_patients, n_features) patient-level metadata
    feature_names  : list of n_features feature names
    top_k          : number of PCs to regress (clamped to available)
    n_bootstrap    : iterations for stability selection

    Returns
    -------
    dict with r2_per_pc, mean_r2, unexplained_variance_fraction,
    coeff_matrix, stability_matrix, ci_low, ci_high,
    top_predictors_per_pc, feature_names, n_patients.
    """
    n_patients, k_avail = pc_projections.shape
    n_features = metadata.shape[1]
    k = min(top_k, k_avail)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(metadata)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    r2_values        = np.zeros(k)
    coeff_matrix     = np.zeros((n_features, k))
    stability_matrix = np.zeros((n_features, k))
    ci_low_matrix    = np.zeros((n_features, k))
    ci_high_matrix   = np.zeros((n_features, k))

    for pc_idx in range(k):
        y = pc_projections[:, pc_idx]
        if y.std() < 1e-8:
            continue

        cv_folds = min(5, n_patients - 1)
        if cv_folds < 2:
            continue

        lasso = LassoCV(cv=cv_folds, random_state=SEED,
                        max_iter=5000, n_jobs=1)
        try:
            lasso.fit(X_scaled, y)
        except Exception as exc:
            warnings.warn(f"LassoCV failed for PC{pc_idx + 1}: {exc}")
            continue

        y_pred = lasso.predict(X_scaled)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_values[pc_idx]      = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        coeff_matrix[:, pc_idx] = lasso.coef_

        sel_probs, ci_lo, ci_hi = _lasso_stability_selection(
            X_scaled, y, n_iter=n_bootstrap,
        )
        stability_matrix[:, pc_idx] = sel_probs
        ci_low_matrix[:, pc_idx]    = ci_lo
        ci_high_matrix[:, pc_idx]   = ci_hi

    mean_r2     = float(r2_values.mean())
    unexplained = max(0.0, 1.0 - mean_r2)

    # Top predictors per PC (by |coefficient|, only non-zero)
    top_predictors = {}
    for pc_idx in range(k):
        coefs = coeff_matrix[:, pc_idx]
        order = np.argsort(np.abs(coefs))[::-1]
        top_predictors[f"PC{pc_idx + 1}"] = [
            {
                "feature":     feature_names[i],
                "coefficient": float(coefs[i]),
                "stability":   float(stability_matrix[i, pc_idx]),
                "ci_low":      float(ci_low_matrix[i, pc_idx]),
                "ci_high":     float(ci_high_matrix[i, pc_idx]),
            }
            for i in order[:5]
            if abs(coefs[i]) > 1e-10
        ]

    return {
        "n_patients":                   n_patients,
        "n_features":                   n_features,
        "top_k":                        k,
        "feature_names":                feature_names,
        "r2_per_pc":                    {f"PC{i + 1}": float(v) for i, v in enumerate(r2_values)},
        "mean_r2":                      mean_r2,
        "unexplained_variance_fraction": unexplained,
        "coeff_matrix":                 coeff_matrix,        # (n_features, k)
        "stability_matrix":             stability_matrix,    # (n_features, k)
        "ci_low":                       ci_low_matrix,       # (n_features, k)
        "ci_high":                      ci_high_matrix,      # (n_features, k)
        "top_predictors_per_pc":        top_predictors,
        "n_bootstrap_iterations":       n_bootstrap,
        "interpretation_note": (
            "Unexplained variance fraction is a positive scientific finding: "
            "the model has learned geometric structure that falls outside the "
            "conceptual vocabulary captured by these metadata features."
        ),
    }


# =============================================================================
# Step 5c — Tier B: UMAP + HDBSCAN Cluster Enrichment
# =============================================================================

def run_cluster_enrichment(
    umap_embedding: np.ndarray,
    metadata: np.ndarray,
    feature_names: list,
    min_cluster_size: int = 10,
) -> dict:
    """
    Tier B — UMAP + HDBSCAN cluster enrichment analysis.
    https://hdbscan.readthedocs.io/en/latest/index.html
    
    Clusters the UMAP embedding with HDBSCAN, then computes enrichment of
    metadata features in each cluster relative to the population baseline.

    Both umap_embedding and metadata must be at the same granularity
    (sample-level).  Use broadcast_to_samples() to expand patient-level
    metadata before calling.

    Parameters
    ----------
    umap_embedding   : (N, 2) UMAP coordinates
    metadata         : (N, n_features) sample-level metadata
    feature_names    : list of n_features feature names
    min_cluster_size : HDBSCAN minimum cluster size

    Returns
    -------
    dict with cluster_labels, n_clusters, n_noise, enrichment_matrix,
    cluster_profiles, cluster_sizes, unlabeled_clusters.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,                  # defaults to min_cluster_size
    )
    cluster_labels = clusterer.fit_predict(umap_embedding)

    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    n_clusters  = len(cluster_ids)
    n_noise     = int((cluster_labels == -1).sum())

    if n_clusters == 0:
        warnings.warn("HDBSCAN found 0 clusters (all noise). "
                       "Try reducing min_cluster_size.")
        return {
            "cluster_labels": cluster_labels,
            "n_clusters": 0, "n_noise": n_noise,
            "cluster_ids": [],
            "enrichment_matrix": np.empty((0, metadata.shape[1])),
            "cluster_profiles": {},
            "cluster_sizes": {},
            "unlabeled_clusters": [],
            "feature_names": feature_names,
        }

    n_features = metadata.shape[1]

    # Population statistics (non-noise samples only)
    non_noise  = cluster_labels >= 0
    pop_data   = metadata[non_noise]
    pop_mean   = pop_data.mean(axis=0)
    pop_std    = pop_data.std(axis=0)
    pop_std[pop_std < 1e-10] = 1e-10

    # Enrichment z-scores  (n_clusters × n_features)
    enrichment_matrix = np.zeros((n_clusters, n_features))
    cluster_sizes     = {}

    for i, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        cluster_data = metadata[mask]
        cluster_mean = cluster_data.mean(axis=0)
        enrichment_matrix[i] = (cluster_mean - pop_mean) / pop_std
        cluster_sizes[int(cid)] = int(mask.sum())

    # Cluster profiles: top enriched/depleted features per cluster
    binary_mask = np.array([is_binary(metadata[:, j])
                            for j in range(n_features)])
    cluster_profiles = {}

    for i, cid in enumerate(cluster_ids):
        z = enrichment_matrix[i]
        order = np.argsort(np.abs(z))[::-1]
        features = []
        for j in order[:7]:
            if abs(z[j]) < 0.3:
                break
            entry = {
                "feature":   feature_names[j],
                "z_score":   round(float(z[j]), 3),
                "direction": "enriched" if z[j] > 0 else "depleted",
            }
            # Odds ratio for binary features
            if binary_mask[j]:
                mask_c = cluster_labels == cid
                p_clust = metadata[mask_c, j].mean()
                p_pop   = pop_data[:, j].mean()
                if p_pop > 0 and p_pop < 1:
                    or_val = (p_clust / (1 - p_clust + 1e-10)) / \
                             (p_pop   / (1 - p_pop   + 1e-10))
                    entry["odds_ratio"] = round(float(or_val), 3)
            features.append(entry)
        cluster_profiles[int(cid)] = features

    # Identify unlabeled clusters (max |z| < 1.0 — no strong enrichment)
    max_z_per_cluster = np.abs(enrichment_matrix).max(axis=1)
    unlabeled_clusters = [
        int(cluster_ids[i]) for i in range(n_clusters)
        if max_z_per_cluster[i] < 1.0
    ]

    return {
        "cluster_labels":      cluster_labels,
        "n_clusters":          n_clusters,
        "n_noise":             n_noise,
        "cluster_ids":         cluster_ids,
        "enrichment_matrix":   enrichment_matrix,
        "cluster_profiles":    cluster_profiles,
        "cluster_sizes":       cluster_sizes,
        "unlabeled_clusters":  unlabeled_clusters,
        "feature_names":       feature_names,
    }