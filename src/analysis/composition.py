"""
Boolean composition tooling for SAE features.

Answers if a clinical concept is captured not by a single SAE feature but
with a boolean combination of features?  A large AUROC gap between the
best single feature and a shallow decision tree is the compositional signal.

Functions
---------
  sae_boolean_composition       : decision-tree extraction of boolean rules over features
  minimal_feature_set           : greedy forward selection to reach target AUROC
  sae_feature_subspace          : L2-normalized decoder weight columns as a basis
  compositional_decomposition   : greedy matching pursuit of label subspace via SAE features
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def sae_boolean_composition(
    activations: np.ndarray,
    label: np.ndarray,
    max_depth: int = 4,
    _verbose: bool = False
) -> dict:
    """Fit a shallow decision tree on sparse activations and extract boolean rules.

    The tree splits on feature activation (nonzero vs zero), producing
    interpretable AND/OR-style rules over SAE features.

    Parameters
    ----------
    activations : (N, n_features) sparse activation matrix
    label       : (N,) binary labels (0/1)
    max_depth   : maximum tree depth (controls rule complexity)

    Returns
    -------
    dict with:
        rules                     : list of rule dicts (positive-class paths only)
        tree_auroc                : AUROC of the full tree
        best_single_feature_auroc : AUROC of the best individual feature
        compositional_gap         : tree_auroc - best_single_feature_auroc
        n_features_used           : number of unique features in tree splits
    """
    label = np.asarray(label, dtype=int)
    N, n_features = activations.shape
    binary_active = (activations != 0).astype(np.float64)

    # Best single feature AUROC
    best_single_auroc = 0.0
    for feat_idx in range(n_features):
        feat = binary_active[:, feat_idx]
        if feat.sum() == 0 or feat.sum() == N:
            continue
        try:
            auc = roc_auc_score(label, activations[:, feat_idx])
        except ValueError:
            continue
        if auc > best_single_auroc:
            best_single_auroc = auc
    
    if _verbose: print(f"  Best single feature AUROC: {best_single_auroc:.4f}")

    # Fit decision tree on binary activations
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(5, int(0.01 * N)),
        class_weight="balanced",
    )
    tree.fit(binary_active, label)
    
    if _verbose:
        print(f"  Tree fitted:")
        print(f"  - Tree depth: {tree.get_depth()}")
        print(f"  - Tree nodes: {tree.tree_.node_count}")

    tree_proba = tree.predict_proba(binary_active) # (n_samples, n_classes)
    tree_auroc = 0.0
    if tree_proba.shape[1] == 2:
        try:
            tree_auroc = roc_auc_score(label, tree_proba[:, 1])
        except ValueError:
            pass
        
    if _verbose: 
        print(f"  - Tree AUROC: {tree_auroc:.4f}")
        print(f"  - Compositional Gap: {tree_auroc - best_single_auroc:.5f}")

    # Extract rules from root-to-leaf paths (positive class only)
    rules = _extract_tree_rules(tree, binary_active, label)

    # Count unique features used in splits
    tree_ = tree.tree_
    split_features = set(
        int(tree_.feature[i]) for i in range(tree_.node_count)
        if tree_.children_left[i] != tree_.children_right[i]
    )
    
    if _verbose:
        print(f"  Unique features used: {len(split_features)}")

    return {
        "rules": rules,
        "tree_auroc": round(tree_auroc, 4),
        "best_single_feature_auroc": round(best_single_auroc, 4),
        "compositional_gap": round(tree_auroc - best_single_auroc, 4),
        "n_features_used": len(split_features),
    }


def _extract_tree_rules(
    tree: DecisionTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> list[dict]:
    """Walk the fitted tree and extract root-to-leaf paths predicting positive class."""
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value

    rules = []

    def _walk(node: int, conditions: list[dict]):
        if tree_.children_left[node] == tree_.children_right[node]:
            # Leaf node - only keep positive-class predictions
            counts = value[node][0]
            prediction = int(np.argmax(counts))
            if prediction != 1:
                return

            leaf_mask = np.ones(len(y), dtype=bool)
            for cond in conditions:
                feat_col = X[:, cond["feature"]]
                if cond["direction"] == ">":
                    leaf_mask &= feat_col > cond["threshold"]
                else:
                    leaf_mask &= feat_col <= cond["threshold"]

            n_leaf = int(leaf_mask.sum())
            if n_leaf == 0:
                return

            prec = float(precision_score(y[leaf_mask],
                                         np.full(n_leaf, 1),
                                         zero_division=0))
            rec_denom = int((y == 1).sum())
            rec = int((y[leaf_mask] == 1).sum()) / rec_denom if rec_denom > 0 else 0.0

            rules.append({
                "conditions": list(conditions),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "support": n_leaf,
            })
            return

        feat_idx = feature[node]
        thresh = float(threshold[node])
        # Left child: feature <= threshold
        _walk(tree_.children_left[node],
              conditions + [{"feature": int(feat_idx), "threshold": thresh, "direction": "<="}])
        # Right child: feature > threshold
        _walk(tree_.children_right[node],
              conditions + [{"feature": int(feat_idx), "threshold": thresh, "direction": ">"}])

    _walk(0, [])
    rules.sort(key=lambda r: r["precision"], reverse=True)
    return rules


def minimal_feature_set(
    activations: np.ndarray,
    label: np.ndarray,
    target_auroc: float = 0.8,
) -> dict:
    """Greedy forward selection of features to reach target AUROC.

    At each step, adds the feature that maximally improves AUROC when combined
    with the already-selected set (using logistic regression on the binary
    active/inactive indicators). Set size is the concept's compositionality
    measure: 1 = monosemantic, many = distributed/compositional.

    Parameters
    ----------
    activations  : (N, n_features) sparse activation matrix
    label        : (N,) binary labels (0/1)
    target_auroc : stop when this AUROC is reached

    Returns
    -------
    dict with:
        selected_features : ordered list of feature indices
        auroc_curve       : AUROC after adding each feature
        n_features_needed : number of features selected
    """
    from sklearn.linear_model import LogisticRegression

    label = np.asarray(label, dtype=int)
    N, n_features = activations.shape
    binary_active = (activations != 0).astype(np.float64)

    # Candidate features: those that activate on at least one sample
    candidates = set(np.where(binary_active.sum(axis=0) > 0)[0].tolist())
    selected: list[int] = []
    auroc_curve: list[float] = []
    current_auroc = 0.0

    while candidates and current_auroc < target_auroc:
        best_feat = -1
        best_auc = current_auroc

        for feat_idx in candidates:
            trial = selected + [feat_idx]
            X_trial = binary_active[:, trial]
            try:
                clf = LogisticRegression(
                    max_iter=200, solver="lbfgs",
                    class_weight="balanced", C=1.0)
                clf.fit(X_trial, label)
                proba = clf.predict_proba(X_trial)[:, 1]
                auc = roc_auc_score(label, proba)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if auc > best_auc:
                best_auc = auc
                best_feat = feat_idx

        if best_feat < 0:
            break

        selected.append(best_feat)
        candidates.discard(best_feat)
        current_auroc = best_auc
        auroc_curve.append(round(current_auroc, 4))

    return {
        "selected_features": selected,
        "auroc_curve": auroc_curve,
        "n_features_needed": len(selected),
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
) -> dict:
    """Greedy matching pursuit: cover a label subspace with SAE decoder directions.

    At each step, adds the SAE feature whose decoder direction maximally
    increases the maximum principal-angle cosine coverage of the label
    subspace.

    Parameters
    ----------
    label_sub : dict from label_subspace() with "directions" (rank, D)
                and "eigenvalues" (rank,)
    sae       : SparseAutoencoder (needs sae.decoder.weight and sae.n_features)
    threshold : stop when max principal angle cosine >= this value

    Returns
    -------
    dict with:
        selected_features : list[int] in greedy selection order
        principal_angles  : list[float] max principal angle cosine after each addition
        n_features_needed : int to reach threshold
        residual          : 1 - coverage at termination
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

    while candidates and current_coverage < threshold:
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
    }
