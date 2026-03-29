"""
Logistic Regression and XGBoost baselines predicting readmission on 
the precomputed metadata features (from src.mimic.features).
Falls back to flatten_sequences() if metadata files don't exist yet.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.io import PROCESSED_DIR, load_metadata



def flatten_sequences(sequences: list[dict], label_key: str = "label") -> tuple:
    records = []
    labels = []
    for seq in sequences:
        encs = seq["encounters"]
        n_enc = len(encs)

        all_icd = [c for enc in encs for c in enc["icd_codes"]]
        all_meds = [m for enc in encs for m in enc["meds"]]

        times = sorted(enc["admittime"] for enc in encs)
        span_days = (times[-1] - times[0]).total_seconds() / 86400 if n_enc > 1 else 0
        gaps = [(times[i + 1] - times[i]).total_seconds() / 86400
                for i in range(len(times) - 1)]

        records.append([
            float(n_enc),
            float(len(set(all_icd))),
            float(len(set(all_meds))),
            float(len(all_meds) / n_enc),
            float(len(all_icd) / n_enc),
            float(span_days),
            float(np.mean(gaps)) if gaps else 0.0,
        ])
        labels.append(int(seq.get(label_key, 0)))

    feature_names = [
        "n_encounters", "n_unique_icd", "n_unique_meds",
        "mean_meds_per_enc", "mean_icd_per_enc",
        "span_days", "mean_gap_days",
    ]
    return np.array(records, dtype=np.float64), np.array(labels), feature_names


# =============================================================================
# Shared evaluation
# =============================================================================

def _evaluate_folds(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_auroc, fold_auprc, fold_f1, fold_brier = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_cls(**model_kwargs)
        model.fit(X_train_s, y_train)

        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_auroc.append(roc_auc_score(y_test, y_prob))
        fold_auprc.append(average_precision_score(y_test, y_prob))
        fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
        fold_brier.append(brier_score_loss(y_test, y_prob))

    return {
        "fold_auroc": fold_auroc,
        "fold_auprc": fold_auprc,
        "fold_f1":    fold_f1,
        "fold_brier": fold_brier,
        "mean_auroc": float(np.mean(fold_auroc)),
        "std_auroc":  float(np.std(fold_auroc)),
        "mean_auprc": float(np.mean(fold_auprc)),
        "std_auprc":  float(np.std(fold_auprc)),
        "mean_f1":    float(np.mean(fold_f1)),
        "std_f1":     float(np.std(fold_f1)),
        "mean_brier": float(np.mean(fold_brier)),
        "std_brier":  float(np.std(fold_brier)),
    }


# =============================================================================
# Logistic Regression
# =============================================================================

def run_logistic(
    metadata: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Logistic regression baseline with stratified k-fold CV"""
    print("\n" + "=" * 60)
    print("Logistic Regression Baseline")
    print("=" * 60)

    n_pos = int(labels.sum())
    print(f"  Samples:  {len(labels)} (pos={n_pos}, neg={len(labels) - n_pos})")
    print(f"  Features: {len(feature_names)}")
    print(f"  Folds:    {n_splits}")

    X = np.nan_to_num(metadata, nan=0.0)

    results = _evaluate_folds(
        LogisticRegression,
        dict(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed),
        X, labels, n_splits=n_splits, seed=seed,
    )
    results["model"] = "logistic_regression"
    results["n_samples"] = len(labels)
    results["n_positive"] = n_pos
    results["n_features"] = len(feature_names)

    print(f"\n  AUROC: {results['mean_auroc']:.4f} +/- {results['std_auroc']:.4f}")
    print(f"  AUPRC: {results['mean_auprc']:.4f} +/- {results['std_auprc']:.4f}")
    print(f"  F1:    {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}")
    print(f"  Brier: {results['mean_brier']:.4f} +/- {results['std_brier']:.4f}")

    return results


# =============================================================================
# XGBoost
# =============================================================================

def run_xgboost(
    metadata: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """XGBoost baseline with stratified k-fold CV."""
    print("\n" + "=" * 60)
    print("XGBoost Baseline")
    print("=" * 60)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"  Samples:  {len(labels)} (pos={n_pos}, neg={n_neg})")
    print(f"  Features: {len(feature_names)}")
    print(f"  Folds:    {n_splits}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    X = np.nan_to_num(metadata, nan=0.0)

    results = _evaluate_folds(
        XGBClassifier,
        dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
        ),
        X, labels, n_splits=n_splits, seed=seed,
    )
    results["model"] = "xgboost"
    results["n_samples"] = len(labels)
    results["n_positive"] = n_pos
    results["n_features"] = len(feature_names)

    print(f"\n  AUROC: {results['mean_auroc']:.4f} +/- {results['std_auroc']:.4f}")
    print(f"  AUPRC: {results['mean_auprc']:.4f} +/- {results['std_auprc']:.4f}")
    print(f"  F1:    {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}")
    print(f"  Brier: {results['mean_brier']:.4f} +/- {results['std_brier']:.4f}")

    return results


# =============================================================================
# CLI
# =============================================================================

def _serializable(obj):
    """Make results JSON-serializable (drop numpy arrays)."""
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run logistic regression and XGBoost baselines")
    parser.add_argument(
        "--sequences", type=str,
        default=str(PROCESSED_DIR / "sequences.jsonl"),
        help="Path to sequences.jsonl (fallback featurization)")
    parser.add_argument(
        "--metadata-dir", type=str,
        default=str(PROCESSED_DIR),
        help="Directory with precomputed metadata files")
    parser.add_argument(
        "--label-key", type=str, default="label",
        help="Label column name in metadata feature matrix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON (default: metadata-dir/baseline_results.json)")
    args = parser.parse_args()

    print("=" * 60)
    print("Baseline Evaluation")
    print("=" * 60)

    # Try loading precomputed metadata
    meta_dir = Path(args.metadata_dir)
    try:
        metadata, feature_names, patient_ids = load_metadata(meta_dir)
        print(f"  Loaded precomputed metadata: {metadata.shape[0]} patients x {metadata.shape[1]} features")
    except FileNotFoundError:
        print("  Precomputed metadata not found, falling back to flatten_sequences()")
        from src.utils.io import load_sequences
        sequences = load_sequences()
        metadata, labels_arr, feature_names = flatten_sequences(sequences, args.label_key)
        patient_ids = np.array([s["subject_id"] for s in sequences], dtype=str)

    # Extract labels from the metadata feature matrix
    label_idx = feature_names.index(args.label_key) if args.label_key in feature_names else None
    if label_idx is not None:
        labels = metadata[:, label_idx].astype(int)
    else:
        # Fallback: load from sequences
        from src.utils.io import load_sequences_dict
        seq_dict = load_sequences_dict(Path(args.sequences))
        labels = np.array([
            int(seq_dict[pid].get(args.label_key, 0))
            for pid in patient_ids
        ])

    lr_results = run_logistic(metadata, labels, feature_names, seed=args.seed)
    xgb_results = run_xgboost(metadata, labels, feature_names, seed=args.seed)

    # Save combined results
    output_path = Path(args.output) if args.output else meta_dir / "baseline_results.json"
    combined = {
        "logistic_regression": _serializable(lr_results),
        "xgboost": _serializable(xgb_results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved -> {output_path}")
