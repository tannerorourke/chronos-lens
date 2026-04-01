"""
Baseline models for clinical prediction tasks.

Two tiers of baselines:

1. Token-level (fair comparison)
   Same input information as the JEPA - multi-hot encoded ICD codes and
   medications from the same vocabulary.  High-dimensional, sparse.

2. Metadata (ceiling comparison)
   Expert-engineered features from extract_metadata(). Uses derived features
   encoding domain knowledge (drug class groupings, temporal statistics,
   escalation indicators). Establishes what's achievable with feature engineering.

Both tiers are evaluated with logistic regression and XGBoost under
stratified k-fold CV, for each available label (label_30d, label_escalation).
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

from src.utils.io import DATA_DIR, load_metadata


# =============================================================================
# Token-level feature construction
# =============================================================================

def build_token_features(
    sequences: list[dict],
    vocab: dict[str, int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build multi-hot token features from patient sequences.

    For each patient, encodes the presence of every ICD code and medication
    across all encounters as binary indicators, using the same token vocabulary
    as the JEPA. Adds n_encounters as the only structural feature.

    Parameters
    ----------
    sequences : list[dict] with "encounters" containing "icd_codes" and "meds"
    vocab     : {token: index} mapping. If None, built from the sequences.

    Returns
    -------
    X             : (N_patients, vocab_size + 1) float64
    feature_names : list[str]
    """
    if vocab is None:
        tokens: set[str] = set()
        for seq in sequences:
            for enc in seq["encounters"]:
                tokens.update(str(c) for c in enc.get("icd_codes", []))
                tokens.update(str(m) for m in enc.get("meds", []))
        tokens.discard("[PAD]")
        vocab = {tok: i for i, tok in enumerate(sorted(tokens))}

    # Filter out special tokens
    token_list = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])
                  if tok != "[PAD]"]
    tok_to_col = {tok: i for i, tok in enumerate(token_list)}
    n_tok = len(token_list)

    N = len(sequences)
    X = np.zeros((N, n_tok + 1), dtype=np.float64)

    for i, seq in enumerate(sequences):
        encs = seq["encounters"]
        X[i, n_tok] = float(len(encs))  # n_encounters column
        for enc in encs:
            for code in enc.get("icd_codes", []):
                col = tok_to_col.get(str(code))
                if col is not None:
                    X[i, col] = 1.0
            for med in enc.get("meds", []):
                col = tok_to_col.get(str(med))
                if col is not None:
                    X[i, col] = 1.0

    feature_names = [f"tok_{tok}" for tok in token_list] + ["n_encounters"]
    return X, feature_names


# =============================================================================
# Legacy feature flattening (kept for standalone CLI fallback)
# =============================================================================

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
# Combined baseline runner
# =============================================================================

_LABEL_KEYS = ["label_30d", "label_escalation"]


def _drop_label_cols(
    X: np.ndarray,
    feature_names: list[str],
    label_keys: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Remove columns whose name matches any label key."""
    keep = [i for i, name in enumerate(feature_names) if name not in label_keys]
    return X[:, keep], [feature_names[i] for i in keep]


def _extract_labels(
    sequences: list[dict],
    label_key: str,
) -> np.ndarray:
    """Extract binary label array from patient dicts."""
    return np.array([int(s.get(label_key, 0)) for s in sequences])


def run_all_baselines(
    sequences: list[dict],
    metadata: np.ndarray,
    feature_names: list[str],
    patient_ids: np.ndarray,
    vocab: dict[str, int] | None = None,
    seed: int = 42,
) -> dict:
    """Run all baseline models across all labels and feature tiers.

    Parameters
    ----------
    sequences     : list[dict] of patient dicts (with label fields attached)
    metadata      : (N, F) metadata feature matrix from extract_metadata()
    feature_names : list[str] metadata column names
    patient_ids   : (N,) str array of subject IDs
    vocab         : {token: index} vocabulary dict, or None to build from sequences
    seed          : random seed

    Returns
    -------
    dict keyed by label name, each containing metrics for all 4 model variants.
    """
    print("\n" + "=" * 60)
    print("Baseline Evaluation")
    print("=" * 60)

    # Build token features once
    print("\nBuilding token-level features...")
    X_tok, tok_names = build_token_features(sequences, vocab=vocab)
    print(f"  Token features: {X_tok.shape[0]} patients x {X_tok.shape[1]} features")
    print(f"  Metadata features: {metadata.shape[0]} patients x {metadata.shape[1]} features")

    # Align sequences to patient_ids order (metadata row order is authoritative)
    pid_to_seq = {str(s["subject_id"]): s for s in sequences}
    ordered_sequences = [pid_to_seq[pid] for pid in patient_ids]

    results: dict = {}

    for label_key in _LABEL_KEYS:
        labels = _extract_labels(ordered_sequences, label_key)
        n_pos = int(labels.sum())

        if n_pos == 0:
            print(f"\n  [{label_key}] No positive samples - skipping")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Label: {label_key}  (pos={n_pos}, neg={len(labels) - n_pos})")
        print(f"{'─' * 60}")

        label_results: dict = {}

        # Prepare feature matrices with label columns removed
        X_tok_clean, tok_names_clean = _drop_label_cols(X_tok, tok_names, _LABEL_KEYS)
        X_meta_clean, meta_names_clean = _drop_label_cols(metadata, feature_names, _LABEL_KEYS)

        n_neg = len(labels) - n_pos
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

        configs = [
            ("token_logistic", X_tok_clean, tok_names_clean,
             LogisticRegression,
             dict(max_iter=1000, class_weight="balanced", solver="lbfgs",
                  random_state=seed)),
            ("token_xgboost", X_tok_clean, tok_names_clean,
             XGBClassifier,
             dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                  scale_pos_weight=scale_pos, eval_metric="logloss",
                  random_state=seed, verbosity=0)),
            ("metadata_logistic", X_meta_clean, meta_names_clean,
             LogisticRegression,
             dict(max_iter=1000, class_weight="balanced", solver="lbfgs",
                  random_state=seed)),
            ("metadata_xgboost", X_meta_clean, meta_names_clean,
             XGBClassifier,
             dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                  scale_pos_weight=scale_pos, eval_metric="logloss",
                  random_state=seed, verbosity=0)),
        ]

        for name, X, _, model_cls, model_kwargs in configs:
            print(f"    {name:<22s} ({X.shape[1]} features) ... ", end="", flush=True)
            X_clean = np.nan_to_num(X, nan=0.0)
            fold_results = _evaluate_folds(
                model_cls, model_kwargs, X_clean, labels, n_splits=5, seed=seed,
            )
            fold_results["model"] = name
            fold_results["n_samples"] = len(labels)
            fold_results["n_positive"] = n_pos
            fold_results["n_features"] = X.shape[1]
            label_results[name] = fold_results
            print(f"AUROC={fold_results['mean_auroc']:.4f}  "
                  f"AUPRC={fold_results['mean_auprc']:.4f}  "
                  f"F1={fold_results['mean_f1']:.4f}")

        results[label_key] = label_results

    # Print summary table
    _print_summary_table(results)

    return results


def _print_summary_table(results: dict) -> None:
    """Print a formatted comparison table across all labels and models."""
    labels = [k for k in _LABEL_KEYS if k in results]
    if not labels:
        return

    models = ["token_logistic", "token_xgboost", "metadata_logistic", "metadata_xgboost"]
    model_display = {
        "token_logistic":    "Token LR",
        "token_xgboost":     "Token XGB",
        "metadata_logistic": "Metadata LR",
        "metadata_xgboost":  "Metadata XGB",
    }
    metrics = ["mean_auroc", "mean_auprc", "mean_f1"]
    metric_display = {"mean_auroc": "AUROC", "mean_auprc": "AUPRC", "mean_f1": "F1"}

    print(f"\n{'=' * 60}")
    print("Baseline Summary")
    print(f"{'=' * 60}")

    # Header
    header = f"{'':22s}"
    for label in labels:
        header += f"  {label:^27s}"
    print(header)

    sub_header = f"{'':22s}"
    for _ in labels:
        sub_header += "  " + "  ".join(f"{metric_display[m]:>7s}" for m in metrics) + "  "
    print(sub_header)
    print("─" * len(sub_header))

    # Rows
    for model in models:
        row = f"{model_display[model]:<22s}"
        for label in labels:
            label_results = results.get(label, {})
            model_results = label_results.get(model)
            if model_results:
                vals = "  ".join(f"{model_results[m]:7.3f}" for m in metrics)
                row += f"  {vals}  "
            else:
                row += "  " + "  ".join(f"{'-':>7s}" for _ in metrics) + "  "
        print(row)

    print(f"{'=' * 60}")


# =============================================================================
# JSON serialization helper
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


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline models (logistic regression and XGBoost)")
    parser.add_argument("--data-dir", type=str,
                        default=str(DATA_DIR),
                        help="Directory containing sequences.jsonl and precomputed metadata files")
    parser.add_argument("--label-key", type=str, default="label",
                        help="Label column name in metadata feature matrix")
    parser.add_argument("--seed", type=int, 
                        default=42)
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Path to save results JSON (default: data-dir/baseline_results.json)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Try loading precomputed metadata
    try:
        metadata, feature_names, patient_ids = load_metadata(data_dir)
        print(f"  Loaded precomputed metadata: {metadata.shape[0]} patients x {metadata.shape[1]} features")
    except FileNotFoundError:
        print("  Precomputed metadata not found, falling back to flatten_sequences")
        from src.utils.io import load_sequences
        sequences = load_sequences(path=data_dir / "sequences.jsonl")
        metadata, labels_arr, feature_names = flatten_sequences(sequences, args.label_key)
        patient_ids = np.array([s["subject_id"] for s in sequences], dtype=str)

    # Extract labels from the metadata feature matrix
    label_idx = feature_names.index(args.label_key) if args.label_key in feature_names else None
    if label_idx is not None:
        labels = metadata[:, label_idx].astype(int)
    else:
        # Fallback: load from sequences
        from src.utils.io import load_sequences_dict
        seq_dict = load_sequences_dict(data_dir / "sequences.jsonl")
        labels = np.array([
            int(seq_dict[pid].get(args.label_key, 0))
            for pid in patient_ids
        ])

    lr_results = run_logistic(metadata, labels, feature_names, seed=args.seed)
    xgb_results = run_xgboost(metadata, labels, feature_names, seed=args.seed)

    # Save combined results
    output_path = Path(args.output) if args.output else data_dir / "baseline_results.json"
    combined = {
        "logistic_regression": _serializable(lr_results),
        "xgboost": _serializable(xgb_results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved -> {output_path}")
