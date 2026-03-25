"""
geometric latent-space analysis of the JEPA displacement field. Operates 
on the raw displacement vectors (no external metadata). 
Load z_context, z_pred, and labels, compute ||Delta||, PCA, divergence, and 
ICC analysis.

- See 'src/analysis/displacement.py' for displacement field construction
- Plotting from these functions done in 'notebooks/*.ipynb'
"""

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA
import umap as umap_module
import pingouin as pg


SEED = 42
rng  = np.random.default_rng(SEED)


# =============================================================================
# PCA Decomp
# =============================================================================

# --- Utilities ---
def marchenko_pastur_upper(n: int, p: int, trace: float) -> float:
    """
    upper bound of the null hypothesis that the eigenvalues of a covariance matrix 
    are due to noise. The number of signal eigenvalues is the number of eigenvalues 
    above this bound.
    """
    sigma_sq = trace / p
    gamma    = p / n
    return sigma_sq * (1.0 + np.sqrt(gamma)) ** 2

# -----------------

def fit_pca(delta: np.ndarray, k: int, seed: int = SEED) -> tuple[PCA, np.ndarray, np.ndarray]:
    """ Uses n_components=D_embed for PCA so the complete eigenvalue spectrum is available """
    _, D = delta.shape
    pca  = PCA(n_components=D, random_state=seed, svd_solver="full")
    pca.fit(delta)
    
    pca_projections = pca.transform(delta)
    topk_projections = pca_projections[:, :k].astype(np.float64)
    
    return pca, pca_projections, topk_projections


def get_pca_stats(pca: PCA, k: int, n_samples: int):
    """
    Compute stats on the PCA decomposition of the displacement field. Uses 
    n_components=D_embed for PCA so the complete eigenvalue spectrum is available
    
    Parameters
    ----------
    delta : np.ndarray (N, D_embed)
        Displacement field Δ = z_pred - z_context
    top_k : int
        Maximum number of principal components to compute
    n_samples : int
        Number of samples
    seed : int
        Random seed

    References
    ----------
    [1] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of
    eigenvalues for a multivariate normal distribution. Journal of
    Multivariate Analysis, 6(3), 407-412.
    """
    D = pca.n_components_
    eigenvalues = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    trace_cov = float(eigenvalues.sum())
    mp_upper  = marchenko_pastur_upper(n_samples, D, trace_cov)
    n_signal  = int((eigenvalues.tolist() > mp_upper).sum())
    
    ev_sq_sum = float(np.sum(eigenvalues ** 2))
    d_eff     = (trace_cov ** 2) / ev_sq_sum if ev_sq_sum > 0 else 0

    thresh90 = int(np.searchsorted(cumvar, 0.90)) + 1
    thresh95 = int(np.searchsorted(cumvar, 0.95)) + 1
    
    stats = {
        "n_components": D,
        "n_signal_components": n_signal,
        "mp_upper_bound": float(mp_upper),
        "effective_dimensionality": float(d_eff),
        "components_for_90pct": thresh90,
        "components_for_95pct": thresh95,
        "top_k_explained_variance": float(evr[:k].sum()),
        "eigenvalues_all": eigenvalues.tolist(),
    }

    return stats

# =============================================================================
# UMAP
# =============================================================================

def fit_umap_2d(
    delta:       np.ndarray,
    n_neighbors: int = 15,
    metric:      str = "cosine",
    seed:        int = SEED,
) -> np.ndarray:
    """
    Fit UMAP on the displacement field.

    "Cosine metric (default) is consistent with the pairwise distance
    measure used throughout the divergence and clustering analyses
    and avoids the curse of dimensionality"

    Parameters
    ----------
    delta       : (N, D) displacement vectors
    n_neighbors : UMAP locality parameter (automatically clamped to N-1)
    metric      : distance metric — 'cosine' recommended; 'euclidean' also valid
    seed        : random state for reproducibility

    Returns
    -------
    umap_embedding : (N, 2) UMAP embedding
    """

    N = delta.shape[0]
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, N - 1),
        metric=metric,
        random_state=seed,
    )
    return np.asarray(reducer.fit_transform(delta), dtype=np.float64)

# =============================================================================
# Divergent pairs analysis
# =============================================================================

# --- Utilities ---
def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """(N, N) pairwise cosine similarity. Zero-norm rows -> zero similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xn    = X / norms
    return Xn @ Xn.T

def cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    """(N, N) pairwise cosine distance  (= 1 - cosine_similarity)."""
    return 1.0 - cosine_sim_matrix(X)

# -----------------

def find_divergent_pairs(
    context_sims: np.ndarray,
    pred_dists:   np.ndarray,
    eps:          float = 0.9,
    delta:        float = 0.5,
) -> tuple:
    """
    Upper-triangle pairs where context histories are cos_similar > eps
    but cosine_dist of predictions diverge by > delta.

    Parameters
    ----------
    context_sims : (N, N) cosine similarity matrix
    pred_dists   : (N, N) cosine distance matrix
    eps          : context similarity lower bound
    delta        : prediction distance lower bound

    Returns
    -------
    (row_idx, col_idx) : integer arrays indexing the N samples
                         (upper triangle only;  row_idx < col_idx)
    """
    N  = context_sims.shape[0]
    iu = np.triu_indices(N, k=1)
    mask = (context_sims[iu] > eps) & (pred_dists[iu] > delta)
    return iu[0][mask], iu[1][mask]


def regress_divergence(
    context_sims: np.ndarray,
    pred_dists:   np.ndarray,
) -> dict:
    """
    OLS regression of prediction cosine distance on context cosine similarity
    over all upper-triangle pairs.

    The continuous version of divergence analysis: instead of hard
    thresholds, study residuals from the regression as the "unexplained"
    divergence signal.

    Parameters
    ----------
    context_sims : (N, N) cosine similarity matrix
    pred_dists   : (N, N) cosine distance matrix

    Returns
    -------
    dict with keys:
      slope, intercept, r, p, stderr — OLS statistics
      residuals                       — (n_pairs,) signed residuals (y − ŷ)
      ctx_sim_flat                    — (n_pairs,) upper-triangle context sims
      pred_dist_flat                  — (n_pairs,) upper-triangle pred dists
      n_pairs                         — number of unique pairs evaluated
    """
    N  = context_sims.shape[0]
    iu = np.triu_indices(N, k=1)
    x  = context_sims[iu]
    y  = pred_dists[iu]

    lr        = stats.linregress(x, y)
    slope     = float(lr.slope) # type: ignore[arg-type]
    intercept = float(lr.intercept) # type: ignore[arg-type]
    residuals = y - (slope * x + intercept)

    return {
        "slope":          slope,
        "intercept":      intercept,
        "r":              float(lr.rvalue),  # type: ignore[arg-type]
        "p":              float(lr.pvalue),  # type: ignore[arg-type]
        "stderr":         float(lr.stderr),  # type: ignore[arg-type]
        "residuals":      residuals,
        "ctx_sim_flat":   x,
        "pred_dist_flat": y,
        "n_pairs":        int(len(x)),
    }
    

def project_pair_divergence_vectors(
    z_pred:         np.ndarray,
    pair_indices:   tuple,
    pca_components: np.ndarray,
) -> np.ndarray:
    """
    - Compute divergence vectors  d_ij = z_pred_i - z_pred_j
    - project onto top-k PCA axes from the geometry step.

    Parameters
    ----------
    z_pred         : (N, D) predicted embeddings
    pair_indices   : (row_idx, col_idx) from find_divergent_pairs
    pca_components : (k, D) from pca.components_[:k]

    Returns
    -------
    projections : (n_pairs, k)  projection of each divergence vector onto PCs
                  Returns shape (0, k) when pair_indices is empty.
    """
    row_idx, col_idx = pair_indices
    k = pca_components.shape[0]
    if len(row_idx) == 0:
        return np.empty((0, k), dtype=np.float64)
    div_vecs = z_pred[row_idx] - z_pred[col_idx]    # (n_pairs, D)
    return div_vecs @ pca_components.T              # (n_pairs, k)


def divergence_variance_comparison(
    z_pred:         np.ndarray,
    pair_indices:   tuple,
    pca_components: np.ndarray,
) -> tuple:
    """
    Per-PC projection variance of divergent pairs vs. a matched random baseline.

    If divergent-pair vectors are concentrated along a PC axis (high variance
    ratio), that axis encodes the divergence structure — not noise.

    Parameters
    ----------
    z_pred          : (N, D) predicted embeddings
    pair_indices    : (row_idx, col_idx) divergent pair indices
    pca_components  : (k, D) PCA components

    Returns
    -------
    (div_pc_var, rand_pc_var) : each (k,) float arrays
                                np.nan entries when pair_indices is empty.
    """
    row_idx, col_idx = pair_indices
    k     = pca_components.shape[0]
    n_div = len(row_idx)

    if n_div == 0:
        return np.full(k, np.nan), np.full(k, np.nan)

    # Divergent pair projections
    div_proj   = project_pair_divergence_vectors(z_pred, pair_indices, pca_components)
    div_pc_var = div_proj.var(axis=0)

    # Random baseline - same count, drawn uniformly from all upper-triangle pairs
    N           = z_pred.shape[0]
    n_total     = N * (N - 1) // 2
    rand_linear = rng.choice(n_total, size=n_div, replace=(n_div > n_total))
    iu          = np.triu_indices(N, k=1)
    rand_proj   = project_pair_divergence_vectors(
        z_pred, (iu[0][rand_linear], iu[1][rand_linear]), pca_components
    )
    rand_pc_var = rand_proj.var(axis=0)

    return div_pc_var, rand_pc_var

# =============================================================================
# ICC Analysis
# =============================================================================

def compute_icc(
    pc_projections:  np.ndarray,
    subject_ids:     np.ndarray,
    top_k:           int,
    min_samples:     int = 3,
    trait_threshold: float = 0.8,
    state_threshold: float = 0.2
) -> dict:
    """
    Intraclass correlation coefficient per top-k PC axis across encounter
    windows within each patient.

    Interpretation
    --------------
    ICC > trait_threshold (0.8) -> axis tracks stable patient-level trait
    ICC < state_threshold (0.2) -> axis reflects dynamic encounter state

    Parameters
    ----------
    pc_projections : (N, >= top_k) PC score matrix from the geometry step
    subject_ids    : (N,) patient identifier per sample
    top_k          : number of PC axes to evaluate (clamped to shape[1])
    min_samples    : patients with fewer encounter samples are excluded

    Returns
    -------
    dict with keys:
      icc_per_pc       : {"PC1": float|None, ...}
      trait_pcs        : list of PC labels with ICC > trait_threshold
      state_pcs        : list of PC labels with ICC < state_threshold
      eligible_patients: count of qualifying patients
      trait_threshold  : TRAIT_THRESHOLD
      state_threshold  : STATE_THRESHOLD
    """
    def _nan_to_none(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v
    
    # ---------------------------------------------------------------- #
    k = min(top_k, pc_projections.shape[1])

    # Group encounter-level PC scores by patient
    unique_sids   = np.unique(subject_ids)
    groups_per_pc = [[] for _ in range(k)]
    eligible_pids = []

    for sid in unique_sids:
        m = subject_ids == sid
        if int(m.sum()) < min_samples:
            continue
        eligible_pids.append(sid)
        for pc_idx in range(k):
            groups_per_pc[pc_idx].append(pc_projections[m, pc_idx])

    n_eligible = len(eligible_pids)
    icc_values = np.full(k, np.nan)

    icc_list = []
    for pc_idx in range(k):
        rows = []
        for pid in eligible_pids:
            m = subject_ids == pid
            for val in pc_projections[m, pc_idx]:
                rows.append({"subject": pid, "score": float(val)})
        if len(rows) < 4:
            icc_list.append(np.nan)
            continue
        df = pd.DataFrame(rows)
        df["rater"] = df.groupby("subject").cumcount()
        try:
            result = pg.intraclass_corr(
                data=df, targets="subject", raters="rater", ratings="score",
                # nan_policy="omit"
            )
            row_31 = result[result["Type"] == "ICC3,1"]
            icc_list.append(
                float(row_31["ICC"].values[0]) if len(row_31) else np.nan
            )
        except Exception:
            icc_list.append(np.nan)

    icc_values = np.array(icc_list)

    trait_pcs = [f"PC{i+1}" for i, v in enumerate(icc_values)
                 if not np.isnan(v) and v > trait_threshold]
    state_pcs = [f"PC{i+1}" for i, v in enumerate(icc_values)
                 if not np.isnan(v) and v < state_threshold]

    return {
        "icc_per_pc":        {f"PC{i+1}": _nan_to_none(float(v))
                               for i, v in enumerate(icc_values)},
        "trait_pcs":         trait_pcs,
        "state_pcs":         state_pcs,
        "eligible_patients": n_eligible,
        "trait_threshold":   float(trait_threshold),
        "state_threshold":   float(state_threshold),
    }