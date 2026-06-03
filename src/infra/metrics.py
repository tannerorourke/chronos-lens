"""Base evaluation metrics (infra).

Pure metric definitions used across the analysis stack - kept in one place so
AUROC/AUPRC/F1/Brier/ECE and result-dict formatting stay consistent everywhere.
No model or I/O dependencies.
"""
import numpy as np


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
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "f1":    float(f1_score(y_true, y_pred)),
        "brier": brier_score(y_true, y_prob),
        "ece":   expected_calibration_error(y_true, y_prob),
    }


def odds_ratio(freq_group: float, freq_pop: float) -> float:
    """Clamped odds ratio to avoid division by zero."""
    p_g = np.clip(freq_group, 1e-10, 1 - 1e-10)
    p_p = np.clip(freq_pop, 1e-10, 1 - 1e-10)
    return (p_g / (1 - p_g)) / (p_p / (1 - p_p))


def format_binary_result(res: dict) -> dict:
    """Extract standard metrics from evaluate_binary_probe result."""
    return {
        "auroc": res["mean_auroc"],
        "auprc": res["mean_auprc"],
        "f1":    res["mean_f1"],
        "brier": res["mean_brier"],
        "std_auroc": res["std_auroc"],
        "std_auprc": res["std_auprc"],
        "std_f1":    res["std_f1"],
        "std_brier": res["std_brier"],
        "n_samples":  res["n_samples"],
        "n_positive": res["n_positive"],
        "positive_rate": res["positive_rate"],
    }
