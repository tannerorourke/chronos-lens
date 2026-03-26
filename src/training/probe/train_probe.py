"""
Linear probing — train logistic regression probes on transformer
layer representations to localize the readmission signal.

Public API
----------
  train_linear_probe(X, y, cv, seed)
      -> per-fold AUC, accuracy, F1; mean +/- std

  run_probing_sweep(layer_representations)
      -> probe at every layer + final output, full results dict
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


from src.utils.seed import SEED


# =========================================================================
# Single probe
# =========================================================================

def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = SEED,
) -> dict:
    """Train a logistic regression probe with stratified cross-validation.

    Uses class_weight='balanced' to handle label imbalance (~16% positive).
    StandardScaler is fit independently per fold to avoid leakage.

    Parameters
    ----------
    X    : (N, D) representation matrix
    y    : (N,) binary labels
    cv   : number of CV folds
    seed : random seed

    Returns
    -------
    dict with:
        fold_auc      : list[float]
        fold_accuracy : list[float]
        fold_f1       : list[float]
        mean_auc      : float
        std_auc       : float
        mean_accuracy : float
        std_accuracy  : float
        mean_f1       : float
        std_f1        : float
        n_samples     : int
        n_features    : int
        n_positive    : int
        positive_rate : float
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    fold_auc = []
    fold_acc = []
    fold_f1 = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegressionCV(
            cv=3,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            scoring="roc_auc",
            n_jobs=1,
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        fold_auc.append(float(roc_auc_score(y_test, y_prob)))
        fold_acc.append(float(accuracy_score(y_test, y_pred)))
        fold_f1.append(float(f1_score(y_test, y_pred)))

    n_pos = int(y.sum())
    return {
        "fold_auc": fold_auc,
        "fold_accuracy": fold_acc,
        "fold_f1": fold_f1,
        "mean_auc": float(np.mean(fold_auc)),
        "std_auc": float(np.std(fold_auc)),
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
        "n_samples": len(y),
        "n_features": X.shape[1],
        "n_positive": n_pos,
        "positive_rate": round(n_pos / len(y), 4),
    }


# =========================================================================
# Full sweep
# =========================================================================

def run_probing_sweep(
    layer_representations: dict,
    cv: int = 5,
    seed: int = SEED,
) -> dict:
    """Run linear probes at every layer and the final output.

    Parameters
    ----------
    layer_representations : output from extract_layer_representations,
        dict with "layer_0", "layer_1", ..., "final", "labels", "n_layers"
    cv   : number of CV folds
    seed : random seed

    Returns
    -------
    dict with:
        per_layer : dict mapping layer key -> probe results
        summary   : ordered list of (layer_key, mean_auc) for plotting
        best_layer: layer key with highest mean AUC
        interpretation : narrative string
    """
    labels = layer_representations["labels"]
    n_layers = layer_representations["n_layers"]

    # Build ordered list of layer keys
    layer_keys = [f"layer_{i}" for i in range(n_layers)] + ["final"]

    per_layer = {}
    summary = []

    for key in layer_keys:
        X = layer_representations[key]
        result = train_linear_probe(X, labels, cv=cv, seed=seed)
        per_layer[key] = result
        summary.append((key, result["mean_auc"], result["std_auc"]))
        print(f"  {key:12s}  AUC={result['mean_auc']:.4f} +/- {result['std_auc']:.4f}  "
              f"F1={result['mean_f1']:.4f}  acc={result['mean_accuracy']:.4f}")

    best_key = max(per_layer, key=lambda k: per_layer[k]["mean_auc"])
    best_auc = per_layer[best_key]["mean_auc"]

    # Localization interpretation
    early_auc = per_layer["layer_0"]["mean_auc"]
    final_auc = per_layer["final"]["mean_auc"]
    delta = final_auc - early_auc

    if delta < 0.02:
        localization = (
            f"Signal is already present at layer 0 (AUC={early_auc:.3f}) "
            f"and does not improve substantially through the encoder "
            f"(final AUC={final_auc:.3f}). The token embedding table "
            f"already separates the classes — transformer attention "
            f"primarily organizes geometric structure."
        )
    elif delta < 0.05:
        localization = (
            f"Modest signal gain from layer 0 (AUC={early_auc:.3f}) to "
            f"final (AUC={final_auc:.3f}). Both embedding-level features "
            f"and attention-derived structure contribute to separability."
        )
    else:
        localization = (
            f"Substantial signal gain from layer 0 (AUC={early_auc:.3f}) "
            f"to final (AUC={final_auc:.3f}), delta={delta:.3f}. "
            f"Transformer attention is actively building the readmission "
            f"signal through its layers."
        )

    return {
        "per_layer": per_layer,
        "summary": summary,
        "best_layer": best_key,
        "best_auc": best_auc,
        "interpretation": localization,
    }
