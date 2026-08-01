"""
Baselines that bracket the JEPA on the clinical prediction tasks.

Token-level is the fair comparison: multi-hot ICD codes and medications over the same
vocabulary the model sees, so it isolates what the representation adds over the raw
input. Metadata is the ceiling: expert-engineered features from extract_metadata(),
establishing what hand-built domain knowledge achieves.

Both tiers run logistic regression and XGBoost under stratified k-fold CV for every
available label.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.io import DATA_DIR, load_json, load_sequences, save_json
from src.utils.system import SEED, set_global_seed


LABEL_KEYS = ["label_30d", "label_escalation"]

# Columns to exclude when predicting each label.
# These are features derived from or proxying the prediction target.
_EXCLUDE_FOR_LABEL: dict[str, list[str]] = {
    "label_escalation": [
        "label_escalation",
        "label_escalation_per_enc",
        "escalation_criteria_fired",
        "label_30d",
        # Tier 5: all escalation-derived features
        "n_escalation_events",
        "first_escalation_position",
        "escalation_rate",
        "has_new_subcategory",
        "has_severity_increase",
        "has_new_specifier",
        "has_f32_to_f33",
        "has_med_initiation",
        "has_new_drug_class",
        # Tier 6 trajectory (derived from the same encounter-over-encounter changes)
        "f_block_growth",       # approx has_new_subcategory as a count
        "n_unique_f_blocks",    # correlated with f_block_growth
        "max_f_severity",       # correlated with severity_increase
    ],
    "label_30d": [
        # The label itself
        "label_30d",
        "label_escalation",
        "label_escalation_per_enc",
        "escalation_criteria_fired",
        # Tier 4 temporal: min gap directly encodes readmission timing
        "min_days_between_admissions",
        "mean_days_between_admissions",
    ],
}

# =============================================================================
# Private helpers
# =============================================================================

def _drop_cols_for_label(X, feature_names: list[str], label_key: str):
    """Remove columns that leak information about the target label.

    Always drops ALL label columns (don't let label_30d predict escalation
    either - they may be correlated). Then drops label-specific exclusions
    (escalation-derived features, temporal proxies, etc.).
    """
    exclude: set[str] = set()
    for cols in _EXCLUDE_FOR_LABEL.values():
        exclude.update(cols)
    exclude.update(_EXCLUDE_FOR_LABEL.get(label_key, []))

    keep = [i for i, name in enumerate(feature_names) if name not in exclude]
    new_names = [feature_names[i] for i in keep]
    if scipy.sparse.issparse(X):
        return X[:, keep], new_names
    return X[:, keep], new_names


def _build_token_features(
    sequences: list[dict],
    vocab: dict[str, int] | None = None,
) -> tuple[scipy.sparse.csr_array, list[str]]:
    """Build multi-hot token features from patient sequences. For each patient, 
    encode presence of every ICD code and medication across all encounters as 
    binary indicators, using the (same token vocab as core model). Adds 
    n_encounters as the only structural feature.

    Parameters
    ----------
    sequences : list[dict] with encounters containing "icd_codes" and "meds"
    vocab     : {token: index} mapping. If None, built from the sequences.

    Returns
    -------
    X             : (N_patients, vocab_size + 1) csr_array, float32
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
    token_list = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]) if tok != "[PAD]"]
    tok_to_col = {tok: i for i, tok in enumerate(token_list)}
    n_tok = len(token_list)

    N = len(sequences)
    # Build COO data for sparse construction (deduplicate per patient)
    rows, cols, vals = [], [], []

    for i, seq in enumerate(sequences):
        encs = seq["encounters"]
        # Collect unique token columns for patient
        active_cols: set[int] = set()
        for enc in encs:
            for code in enc.get("icd_codes", []):
                col = tok_to_col.get(str(code))
                if col is not None:
                    active_cols.add(col)
            for med in enc.get("meds", []):
                col = tok_to_col.get(str(med))
                if col is not None:
                    active_cols.add(col)
        for col in active_cols:
            rows.append(i)
            cols.append(col)
            vals.append(1.0)
        # n_encounters column (last column)
        rows.append(i)
        cols.append(n_tok)
        vals.append(float(len(encs)))

    X = scipy.sparse.coo_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))
        ),
        shape=(N, n_tok + 1),
    ).tocsr()

    feature_names = [f"tok_{tok}" for tok in token_list] + ["n_encounters"]
    return X, feature_names


def _run_a_baseline(
    model_cls,
    model_kwargs: dict,
    X,
    y: np.ndarray,
    n_splits: int = 5,
    name: str = "",
) -> dict:
    """
        Evaluate baseline (XGB, LR) model under stratified k-fold CV, 
        return fold-level + mean metrics.
    """
    n_pos = int(y.sum())
    print(f"\n    {name:<22s} ({X.shape[1]} features) ... ", end="", flush=True)
    
    is_sparse = scipy.sparse.issparse(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_auroc, fold_auprc, fold_f1, fold_brier = [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler(with_mean=not is_sparse)
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

        del model, scaler, X_train_s, X_test_s, y_prob, y_pred
    
    results = {
        "name": name,
        "n_samples": len(y),
        "n_positive": n_pos,
        "n_features": X.shape[1],
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
    
    print(f"\n      AUROC: {results['mean_auroc']:.4f} +/- {results['std_auroc']:.4f}")
    print(f"      AUPRC: {results['mean_auprc']:.4f} +/- {results['std_auprc']:.4f}")
    print(f"      F1:    {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}")
    print(f"      Brier: {results['mean_brier']:.4f} +/- {results['std_brier']:.4f}")
    
    return results

# =============================================================================
# Combined baseline runner
# =============================================================================

def run_baselines(
    sequences: list[dict],
    metadata: np.ndarray,
    feature_names: list[str],
    patient_ids: np.ndarray,
    vocab: dict[str, int] | None = None,
) -> dict:
    """Run all baseline models across all labels and feature tiers.
       
       Baselines are run on SEED=42 (baseline model seed) for reproducibility.
    """
    print(f"{'=' * 60}\n")
    print("Evaluating baselines..")
    
    set_global_seed(42)

    # Align sequences to patient_ids order (metadata row order is authoritative)
    pid_to_seq = {str(s["subject_id"]): s for s in sequences}
    ordered_sequences = [pid_to_seq[pid] for pid in patient_ids]
    del pid_to_seq

    results: dict = {}
    for label_key in LABEL_KEYS:
        labels = np.array([int(s.get(label_key, 0)) for s in ordered_sequences])
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        print(f"  Label: {label_key}  (pos={n_pos}, neg={n_neg})")

        scale_pos = n_neg / n_pos
        label_results: dict = {}

        # -- Metadata baselines (small, per-label column exclusion) --
        X_meta_clean, meta_names_clean = _drop_cols_for_label(
            metadata, feature_names, label_key)
        X_meta_clean = np.nan_to_num(X_meta_clean, nan=0.0)
        
        # print(f"Remaining features ({len(meta_names_clean)}):")
        # for name in meta_names_clean:
        #     print(f"  {name}")
        
        print(f"  Metadata features: {X_meta_clean.shape[1]} features "
              f"({len(feature_names) - len(meta_names_clean)} excluded for {label_key})")

        label_results["metadata_logistic"] = _run_a_baseline(
            LogisticRegression, dict(max_iter=1000, class_weight="balanced",
                                     solver="lbfgs", random_state=SEED),
            X_meta_clean, labels, name="metadata_logistic",
        )

        label_results["metadata_xgboost"] = _run_a_baseline(
            XGBClassifier, dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=scale_pos, eval_metric="logloss",
                                random_state=SEED, verbosity=0),
            X_meta_clean, labels, name="metadata_xgboost",
        )
        del X_meta_clean

        # -- Token baselines (large, sparse, per-label column exclusion) --
        print("  Building token-level features...")
        X_tok, tok_names = _build_token_features(ordered_sequences, vocab=vocab)
        X_tok_clean, tok_names_clean = _drop_cols_for_label(
            X_tok, tok_names, label_key)
        del X_tok, tok_names
        print(f"  Token features: {X_tok_clean.shape[0]} patients x {X_tok_clean.shape[1]} features"
              f" (sparse, {X_tok_clean.nnz} nonzeros)")

        label_results["token_logistic"] = _run_a_baseline(
            LogisticRegression, dict(max_iter=1000, class_weight="balanced",
                                     solver="lbfgs", random_state=SEED),
            X_tok_clean, labels, name="token_logistic",
        )
        gc.collect()

        label_results["token_xgboost"] = _run_a_baseline(
            XGBClassifier, dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=scale_pos, eval_metric="logloss",
                                random_state=SEED, verbosity=0),
            X_tok_clean, labels, name="token_xgboost",
        )
        del X_tok_clean
        gc.collect()

        results[label_key] = label_results

    del ordered_sequences
    gc.collect()

    _print_summary_table(results)
    return results


# =============================================================================
# Private helpers
# =============================================================================

def _print_summary_table(results: dict, label_keys: list[str] = LABEL_KEYS) -> None:
    labels = [k for k in label_keys if k in results]
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
    print("-" * len(sub_header))

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
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline models (logistic regression and XGBoost)")
    parser.add_argument("--data-dir", type=str,
                        default=str(DATA_DIR),
                        help="Directory containing sequences.jsonl and precomputed metadata files")
    parser.add_argument("--label-key", type=str, default="label",
                        choices=["label_30d", "label_escalation"],
                        help="Label column name in metadata feature matrix")
    parser.add_argument("--output-dir", type=str,
                        default=DATA_DIR,
                        help="Path to save results JSON (default: data-dir/baseline_results.json)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    set_global_seed(42)

    # -- Load sequences --
    sequences = load_sequences(path=data_dir / "sequences.jsonl")

    # -- Load metadata - if not available, compute from sequences --
    try:
        from src.utils.io import load_metadata
        print(f"  Loaded metadata from ../{'/'.join(data_dir.parts[-3])}")
        metadata, feature_names, patient_ids = load_metadata(data_dir)
        print(f"    - {metadata.shape[0]} patients x {metadata.shape[1]} features")
    except FileNotFoundError:
        from src.mimic.metadata import extract_metadata
        print("  Precomputed metadata not found, computing..")
        metadata, feature_names, patient_ids = extract_metadata(sequences, subject_ids=None)
        patient_ids = np.array([s["subject_id"] for s in sequences], dtype=str)

    # -- load vocab --
    print(f"  Loaded metadata from {'/'.join(data_dir.parts[-3])}")
    vocab = load_json(data_dir / "vocab.json")

    # -- Extract labels --
    label_idx = feature_names.index(args.label_key) if args.label_key in feature_names else None
    labels = metadata[:, label_idx].astype(int)

    # -- Run models --
    baseline_results = run_baselines(
        sequences, metadata, feature_names, patient_ids, vocab=vocab)

    save_json(baseline_results, Path(args.output_dir) / "baseline_results.json")