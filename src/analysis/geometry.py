"""
Geometric analysis of JEPA latent-space vectors.

Operates on raw embedding vectors (no external metadata).
PCA, divergence, ICC, subspace alignment, CKA, and
label-supervised subspace analysis.

- Plotting from these functions done in 'notebooks/*.ipynb'
"""

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA
import umap as umap_module
import phate as phate_module
import pingouin as pg


from src.utils.seed import SEED

# =============================================================================
# PCA Decomp
# =============================================================================

# --- Utilities ---
def marchenko_pastur_upper(n: int, p: int, trace: float) -> float:
    """
    null hypothesis upper bound that the eigenvalues of a covariance matrix are due to noise. 
    The number of signal eigenvalues is the number of eigenvalues above this bound.
    """
    sigma_sq = trace / p
    gamma    = float(p / n)
    return sigma_sq * (1.0 + np.sqrt(gamma)) ** 2

# -----------------

def fit_pca(vec: np.ndarray, k: int) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit full PCA on a (N, D) embedding matrix and return top-k projections."""
    _, D = vec.shape
    pca  = PCA(n_components=D, random_state=SEED, svd_solver="full")
    pca.fit(vec)
    
    pca_projections = pca.transform(vec)
    topk_projections = pca_projections[:, :k].astype(np.float64)
    
    return pca, pca_projections, topk_projections


def get_pca_stats(pca: PCA, k: int, n_samples: int):
    """
    Compute stats on a PCA decomposition of a latent-space vector set.

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
    n_signal  = int((eigenvalues > mp_upper).sum())
    
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

def fit_tform_umap_2d(
    vec:       np.ndarray,
    n_neighbors: int = 15,
    metric:      str = "cosine"
) -> np.ndarray:
    N = vec.shape[0]
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, N - 1),
        metric=metric,
        random_state=SEED,
    )
    return np.asarray(reducer.fit_transform(vec), dtype=np.float64)

# =============================================================================
# PHATE
# =============================================================================

def fit_tform_phate_2d(
    vec:  np.ndarray,
    knn:   int = 15,
    t = "auto"
) -> np.ndarray:
    N = vec.shape[0]
    reducer = phate_module.PHATE(
        n_components=2,
        knn=min(knn, N - 1),
        t=t,
        random_state=SEED,
        verbose=0,
    )
    return np.asarray(reducer.fit_transform(vec), dtype=np.float64)


def fit_phate(
    vec:  np.ndarray,
    knn:   int = 15,
    dims: int  = 2,
    t = "auto"
) -> phate_module.PHATE:
    N = vec.shape[0]
    reducer = phate_module.PHATE(
        n_components=min(dims, 3), 
        knn=min(knn, N - 1),
        t=t, 
        random_state=SEED, 
        verbose=0
    )
    return reducer.fit(vec)

# =============================================================================
# Divergent pairs analysis
# =============================================================================

# -----------------

def find_divergent_pairs(
    context_sims: np.ndarray,
    pred_dists:   np.ndarray,
    eps:          float = 0.9,
    delta:        float = 0.5,
) -> tuple:
    """
    Find pairs with similar contexts but divergent predictions.

    Upper-triangle pairs where context cosine similarity > eps
    but prediction cosine distance > delta.

    Parameters
    ----------
    context_sims : (N, N) cosine similarity matrix (e.g. z_target or z_enc)
    pred_dists   : (N, N) cosine distance matrix of predictions (z_pred)
    eps          : context similarity lower bound
    delta        : prediction divergence lower bound

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
    OLS regression of prediction divergence on context similarity.

    Continuous version of divergence analysis: residuals from the
    regression are the "unexplained" prediction divergence signal.

    Parameters
    ----------
    context_sims : (N, N) cosine similarity matrix (e.g. z_target or z_enc)
    pred_dists   : (N, N) cosine distance matrix of predictions (z_pred)

    Returns
    -------
    dict with keys:
      slope, intercept, r, p, stderr - OLS statistics
      residuals                       - (n_pairs,) signed residuals (y - ŷ)
      ctx_sim_flat                    - (n_pairs,) upper-triangle context sims
      pred_dist_flat                  - (n_pairs,) upper-triangle pred dists
      n_pairs                         - number of unique pairs evaluated
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
    Compute prediction divergence vectors d_ij = z_pred_i - z_pred_j
    and project onto top-k PCA axes.
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
    Per-PC projection variance of prediction-divergent pairs vs. random baseline.

    If divergent-pair vectors concentrate along a PC axis (high variance
    ratio), that axis encodes prediction divergence structure - not noise.
    """
    from src.utils.seed import get_rng
    rng  = get_rng()
    
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
      icc_per_pc       : {"PC1": float, ...}
      trait_pcs        : list of PC labels with ICC > trait_threshold
      state_pcs        : list of PC labels with ICC < state_threshold
      eligible_patients: num qualifying patients
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


# =============================================================================
# Subspace Alignment
# =============================================================================

def subspace_alignment(pca_a: PCA, pca_b: PCA, top_k: int) -> dict:
    """
    Principal angles between the top-k PC subspaces of two PCA decompositions.

    Measures whether two representations (e.g. z_pred vs z_target) share the
    same dominant subspace. Cosines near 1.0 mean the axes are aligned;
    near 0.0 means orthogonal subspaces.

    Parameters
    ----------
    pca_a, pca_b : fitted sklearn PCA objects (must share embedding dimension)
    top_k        : number of top PC axes to compare

    Returns
    -------
    dict with keys:
      cos_principal_angles : (top_k,) cosines of canonical angles
      mean_alignment       : mean of cos_principal_angles
      min_alignment        : worst-case alignment across axes
    """
    k = min(top_k, pca_a.n_components_, pca_b.n_components_)
    A = pca_a.components_[:k]   # (k, D)
    B = pca_b.components_[:k]   # (k, D)

    # SVD of A @ B^T gives canonical correlations as singular values
    M = A @ B.T                 # (k, k)
    _, s, _ = np.linalg.svd(M)
    cos_angles = np.clip(s[:k], 0.0, 1.0)

    return {
        "cos_principal_angles": cos_angles.tolist(),
        "mean_alignment":      float(cos_angles.mean()),
        "min_alignment":       float(cos_angles.min()),
    }


# =============================================================================
# CKA (Centered Kernel Alignment)
# =============================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment between two representation matrices.

    CKA measures similarity between two sets of representations independent
    of orthogonal rotation and isotropic scaling. Values in [0, 1]; 1.0
    means identical representational geometry.

    Parameters
    ----------
    X : (N, D1) representation matrix
    Y : (N, D2) representation matrix  (same N)

    Returns
    -------
    float : CKA similarity score

    References
    ----------
    Kornblith et al. (2019). "Similarity of Neural Network Representations
    Revisited." ICML.
    """
    N = X.shape[0]
    assert Y.shape[0] == N, "X and Y must have the same number of samples"

    # Center both matrices
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Linear HSIC: ||Y^T X||_F^2
    hsic_xy = float(np.linalg.norm(Y.T @ X, "fro") ** 2)
    hsic_xx = float(np.linalg.norm(X.T @ X, "fro") ** 2)
    hsic_yy = float(np.linalg.norm(Y.T @ Y, "fro") ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return hsic_xy / denom


# =============================================================================
# Label-supervised subspace analysis
# =============================================================================

def label_subspace(
    z_enc: np.ndarray,
    label: np.ndarray,
    rank: int = 5,
) -> dict:
    """Supervised subspace that separates label classes in z_enc.

    Uses LDA (between-class scatter eigenvectors) for binary labels.
    Falls back to supervised PCA if LDA is degenerate (singular within-class
    scatter or fewer than 2 classes with sufficient samples).

    Parameters
    ----------
    z_enc : (N, D) embedding matrix
    label : (N,) binary labels (0/1)
    rank  : number of top directions to return

    Returns
    -------
    dict with:
        directions          : (rank, D) supervised directions (unit-norm rows)
        eigenvalues         : (rank,) separation strength per direction
        explained_separation: fraction of total between-class variance in top rank
    """
    label = np.asarray(label, dtype=int)
    N, D = z_enc.shape
    rank = min(rank, D)

    classes = np.unique(label)
    n_classes = len(classes)

    # Global mean
    mu = z_enc.mean(axis=0)  # (D,)

    # Between-class scatter S_B
    S_B = np.zeros((D, D), dtype=np.float64)
    for c in classes:
        mask = label == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        mu_c = z_enc[mask].mean(axis=0)
        diff = (mu_c - mu).reshape(-1, 1)  # (D, 1)
        S_B += n_c * (diff @ diff.T)

    # Within-class scatter S_W
    S_W = np.zeros((D, D), dtype=np.float64)
    for c in classes:
        mask = label == c
        if mask.sum() == 0:
            continue
        X_c = z_enc[mask] - z_enc[mask].mean(axis=0)
        S_W += X_c.T @ X_c

    # Try LDA: solve S_W^{-1} S_B via generalized eigenvalue problem
    try:
        # Regularize S_W for numerical stability
        S_W_reg = S_W + 1e-6 * np.eye(D) * np.trace(S_W) / D
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(S_B, S_W_reg)
        # eigh returns ascending order; flip to descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except np.linalg.LinAlgError:
        # Fallback: supervised PCA on between-class scatter directly
        eigenvalues, eigenvectors = np.linalg.eigh(S_B)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues to zero (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Take top rank
    top_vals = eigenvalues[:rank]
    top_vecs = eigenvectors[:, :rank].T  # (rank, D)

    # L2-normalize each direction
    norms = np.linalg.norm(top_vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    top_vecs = top_vecs / norms

    total_separation = float(eigenvalues.sum())
    explained = float(top_vals.sum()) / total_separation if total_separation > 0 else 0.0

    return {
        "directions": top_vecs,
        "eigenvalues": top_vals,
        "explained_separation": explained,
    }


def multi_label_subspace(
    z_enc: np.ndarray,
    label_matrix: np.ndarray,
    rank: int = 5,
    label_names: list[str] | None = None,
) -> dict:
    """CCA directions between z_enc and a multi-label matrix.

    Finds linear directions in z_enc space that are maximally correlated
    with the label matrix.  Uses SVD of the cross-covariance matrix.

    Parameters
    ----------
    z_enc        : (N, D) embedding matrix
    label_matrix : (N, L) binary label matrix (one column per label)
    rank         : number of canonical directions
    label_names  : optional column names for label_matrix

    Returns
    -------
    dict with:
        canonical_directions : (rank, D) CCA directions in z_enc space
        correlations         : (rank,) canonical correlations
        label_names          : list[str]
    """
    N, D = z_enc.shape
    L = label_matrix.shape[1]
    rank = min(rank, D, L)

    if label_names is None:
        label_names = [f"label_{i}" for i in range(L)]

    # Center both matrices
    X = z_enc - z_enc.mean(axis=0)
    Y = label_matrix.astype(np.float64) - label_matrix.mean(axis=0)

    # Covariance matrices
    C_xx = (X.T @ X) / (N - 1)  # (D, D)
    C_yy = (Y.T @ Y) / (N - 1)  # (L, L)
    C_xy = (X.T @ Y) / (N - 1)  # (D, L)

    # Regularize
    C_xx += 1e-6 * np.eye(D) * np.trace(C_xx) / D
    C_yy += 1e-6 * np.eye(L) * np.trace(C_yy) / L

    # Whitening transforms
    from scipy.linalg import sqrtm, inv
    C_xx_inv_sqrt = np.real(inv(sqrtm(C_xx)))  # (D, D)
    C_yy_inv_sqrt = np.real(inv(sqrtm(C_yy)))  # (L, L)

    # SVD of whitened cross-covariance
    M = C_xx_inv_sqrt @ C_xy @ C_yy_inv_sqrt  # (D, L)
    U, s, _ = np.linalg.svd(M, full_matrices=False)

    # CCA directions in original z_enc space
    directions = (C_xx_inv_sqrt @ U[:, :rank]).T  # (rank, D)

    # L2-normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    directions = directions / norms

    correlations = np.clip(s[:rank], 0.0, 1.0)

    return {
        "canonical_directions": directions,
        "correlations": correlations,
        "label_names": list(label_names),
    }


def effective_rank_of_label(
    z_enc: np.ndarray,
    label: np.ndarray,
) -> int:
    """Intrinsic dimensionality of a label concept in z_enc.

    Number of LDA eigenvalues (from the between-class scatter) above the
    Marchenko-Pastur noise floor.

    Parameters
    ----------
    z_enc : (N, D) embedding matrix
    label : (N,) binary labels (0/1)

    Returns
    -------
    int : number of signal eigenvalues (>= 1 for any non-trivial label)
    """
    label = np.asarray(label, dtype=int)
    N, D = z_enc.shape

    classes = np.unique(label)
    mu = z_enc.mean(axis=0)

    # Between-class scatter
    S_B = np.zeros((D, D), dtype=np.float64)
    for c in classes:
        mask = label == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        mu_c = z_enc[mask].mean(axis=0)
        diff = (mu_c - mu).reshape(-1, 1)
        S_B += n_c * (diff @ diff.T)

    # Within-class scatter
    S_W = np.zeros((D, D), dtype=np.float64)
    for c in classes:
        mask = label == c
        if mask.sum() == 0:
            continue
        X_c = z_enc[mask] - z_enc[mask].mean(axis=0)
        S_W += X_c.T @ X_c

    # Solve generalized eigenvalue problem
    try:
        S_W_reg = S_W + 1e-6 * np.eye(D) * np.trace(S_W) / D
        from scipy.linalg import eigh
        eigenvalues, _ = eigh(S_B, S_W_reg)
        eigenvalues = np.sort(eigenvalues)[::-1]
    except np.linalg.LinAlgError:
        eigenvalues, _ = np.linalg.eigh(S_B)
        eigenvalues = np.sort(eigenvalues)[::-1]

    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Marchenko-Pastur threshold on the between-class scatter eigenvalues
    trace = float(eigenvalues.sum())
    if trace < 1e-12:
        return 0
    mp_upper = marchenko_pastur_upper(N, D, trace)
    n_signal = int((eigenvalues > mp_upper).sum())
    return max(n_signal, 1) if trace > 1e-8 else 0