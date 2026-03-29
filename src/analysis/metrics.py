import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ Expected Calibration Error (ECE) with equal-width bins.
        - Partition [0, 1] into *n_bins* equal-width intervals.
        - For each non-empty bin, computes |mean(y_true) - mean(y_prob)| 
          weighted by fraction of samples in that bin. Return the weighted avg.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i == 0:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        avg_true = float(y_true[mask].mean())
        avg_pred = float(y_prob[mask].mean())
        ece += (n_bin / n) * abs(avg_true - avg_pred)
    return float(ece)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> dict:
    """ Function to evaluate provide all post-training evaluations to ensure
        consistent metric definitions.
    """
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1":    float(f1_score(y_true, y_pred)),
        "brier": brier_score(y_true, y_prob),
        "ece":   expected_calibration_error(y_true, y_prob),
    }
