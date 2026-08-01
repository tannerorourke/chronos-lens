"""
Composition tooling for SAE features.

Asks how many SAE features a clinical concept needs. Two views: how many features a
predictor needs to reach a target held-out AUROC on the label, and how many decoder
directions are needed to span a label subspace in embedding space. A small set means a
near-monosemantic concept; a large one means the concept is distributed over features.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.infra.metrics import make_cv_splitter


def minimal_feature_set(
    activations: np.ndarray,
    label: np.ndarray,
    target_auroc: float = 0.8,
    max_features: int = 25,
    binary_active: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
) -> dict:
    """Greedy forward selection of features to reach target AUROC.

    At each step, adds the feature that maximally improves held-out AUROC when combined
    with the already-selected set (using logistic regression on the binary
    active/inactive indicators). Set size is the concept's compositionality
    measure: 1 = monosemantic, many = distributed/compositional.

    Parameters
    ----------
    activations   : (N, n_features) sparse activation matrix
    label         : (N,) binary labels (0/1)
    target_auroc  : stop when this AUROC is reached
    max_features  : hard cap on selected features; reports reached_target=False when unmet
    binary_active : optional precomputed (activations != 0); computed here when None
    groups        : (N,) group ids held out whole, e.g. patient id; None for row-level folds
    n_splits      : number of cross-validation folds scoring each candidate

    Returns
    -------
    dict with:
        selected_features : ordered list of feature indices
        auroc_curve       : mean held-out AUROC after adding each feature
        n_features_needed : number of features selected
        reached_target    : whether target_auroc was met before the cap / plateau
        cv_scheme         : fold scheme used for the curve
    """
    from sklearn.linear_model import LogisticRegression

    label = np.asarray(label, dtype=int)
    if binary_active is None:
        binary_active = (activations != 0).astype(np.float64)

    # Candidate features: those that activate on at least one sample
    candidates = set(np.where(binary_active.sum(axis=0) > 0)[0].tolist())
    selected: list[int] = []
    auroc_curve: list[float] = []
    current_auroc = 0.0

    # -- folds are label-fixed, so cut them once and reuse for every candidate
    splitter, cv_scheme = make_cv_splitter(n_splits=n_splits, groups=groups)
    folds = [(tr, te) for tr, te in splitter.split(binary_active, label, groups)
             if len(np.unique(label[te])) > 1]

    while candidates and current_auroc < target_auroc and len(selected) < max_features:
        best_feat = -1
        best_auc = current_auroc

        # -- each candidate costs n_splits fits; max_features bounds the sweep
        for feat_idx in candidates:
            trial = selected + [feat_idx]
            X_trial = binary_active[:, trial]
            fold_aurocs = []
            for train_idx, test_idx in folds:
                try:
                    clf = LogisticRegression(
                        max_iter=200, solver="lbfgs",
                        class_weight="balanced", C=1.0)
                    clf.fit(X_trial[train_idx], label[train_idx])
                    proba = clf.predict_proba(X_trial[test_idx])[:, 1]
                    fold_aurocs.append(roc_auc_score(label[test_idx], proba))
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if not fold_aurocs:
                continue
            auc = float(np.mean(fold_aurocs))
            if auc > best_auc:
                best_auc = auc
                best_feat = feat_idx

        if best_feat < 0:
            break                       # -- no remaining feature improves AUROC: plateau

        selected.append(best_feat)
        candidates.discard(best_feat)
        current_auroc = best_auc
        auroc_curve.append(round(current_auroc, 4))

    return {
        "selected_features": selected,
        "auroc_curve": auroc_curve,
        "n_features_needed": len(selected),
        "reached_target": current_auroc >= target_auroc,
        "cv_scheme": cv_scheme,
    }


def sae_feature_subspace(
    sae,
    feature_indices: list[int],
) -> np.ndarray:
    """L2-normalized decoder weight columns as a feature basis.

    Parameters
    ----------
    sae             : SparseAutoencoder (or any module with sae.decoder.weight)
    feature_indices : which features to extract

    Returns
    -------
    (len(feature_indices), D) array - rows are unit-norm decoder directions
    """
    W = sae.decoder.weight.detach().cpu().numpy()  # (embed_dim, n_features)
    cols = W[:, feature_indices].T  # (len(indices), embed_dim)
    norms = np.linalg.norm(cols, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    return cols / norms


def sae_decomposition(
    label_sub: dict,
    sae,
    threshold: float = 0.8,
    max_features: int = 25,
) -> dict:
    """Greedy matching pursuit: cover a label subspace with SAE decoder directions.

    At each step, adds the SAE feature whose decoder direction maximally
    increases the maximum principal-angle cosine coverage of the label
    subspace.

    Parameters
    ----------
    label_sub    : dict from label_subspace() with "directions" (rank, D) and "eigenvalues" (rank,)
    sae          : SparseAutoencoder (needs sae.decoder.weight and sae.n_features)
    threshold    : stop when max principal angle cosine >= this value
    max_features : hard cap on selected features; reports reached_target=False when unmet

    Returns
    -------
    dict with:
        selected_features : list[int] in greedy selection order
        principal_angles  : list[float] max principal angle cosine after each addition
        n_features_needed : int to reach threshold
        residual          : 1 - coverage at termination
        reached_target    : whether threshold was met before the cap / plateau
    """

    label_dirs = label_sub["directions"]  # (rank, D)
    rank, D = label_dirs.shape

    # All SAE decoder directions (L2-normalized)
    W = sae.decoder.weight.detach().cpu().numpy()  # (embed_dim, n_features)
    n_features = W.shape[1]
    W_norm = W / np.clip(np.linalg.norm(W, axis=0, keepdims=True), 1e-10, None)

    def _max_cos_principal_angle(basis: np.ndarray) -> float:
        """Max cosine of principal angle between basis rows and label_dirs."""
        if basis.shape[0] == 0:
            return 0.0
        # SVD of basis @ label_dirs^T
        M = basis @ label_dirs.T  # (n_basis, rank)
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        return float(np.clip(s[0], 0.0, 1.0))

    selected: list[int] = []
    principal_angles: list[float] = []
    candidates = set(range(n_features))
    current_coverage = 0.0

    while candidates and current_coverage < threshold and len(selected) < max_features:
        best_feat = -1
        best_cov = current_coverage

        # Current basis from selected features
        if selected:
            current_basis = sae_feature_subspace(sae, selected)
        else:
            current_basis = np.empty((0, D), dtype=np.float64)

        for feat_idx in candidates:
            trial_basis = np.vstack([current_basis,
                                     W_norm[:, feat_idx].reshape(1, -1)])
            # Orthogonalize for numerical stability
            Q, _ = np.linalg.qr(trial_basis.T)
            trial_orth = Q[:, :trial_basis.shape[0]].T  # (n_basis, D)
            cov = _max_cos_principal_angle(trial_orth)
            if cov > best_cov:
                best_cov = cov
                best_feat = feat_idx

        if best_feat < 0:
            break

        selected.append(best_feat)
        candidates.discard(best_feat)
        current_coverage = best_cov
        principal_angles.append(round(current_coverage, 6))

    return {
        "selected_features": selected,
        "principal_angles": principal_angles,
        "n_features_needed": len(selected),
        "residual": round(1.0 - current_coverage, 6),
        "reached_target": current_coverage >= threshold,
    }
