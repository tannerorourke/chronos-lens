import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.analysis.metrics import compute_all_metrics


# =============================================================================
# ICD-10 chapter target extraction
# =============================================================================

def extract_icd_block_targets(
    sequences_path: Path,
    subject_ids: np.ndarray,
    mask_positions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build a binary multi-label matrix of ICD-10 chapters for masked encounters.

    For each (subject_id, mask_position) pair, looks up the masked encounter
    in sequences.jsonl and extracts the first character of every ICD code
    (the ICD-10 chapter letter).  ICD-9 numeric-prefix codes are ignored.

    Parameters
    ----------
    sequences_path : path to sequences.jsonl
    subject_ids    : (N,) str array - patient IDs per sample
    mask_positions : (N,) int array - which encounter was masked (0-indexed)

    Returns
    -------
    targets       : (N, C) int8 binary matrix, columns = active chapters
    chapter_names : list[str] of length C, sorted chapter letters
    """
    patients: dict[str, dict] = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    N = len(subject_ids)
    chapter_sets: list[set[str]] = []
    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_positions[i])
        enc = patients[sid]["encounters"][pos]
        codes = enc.get("icd_codes", [])
        chapters = {c[0] for c in codes if c and c[0].isalpha()}
        chapter_sets.append(chapters)

    all_chapters = sorted(set().union(*chapter_sets)) if chapter_sets else []
    ch_to_idx = {ch: i for i, ch in enumerate(all_chapters)}

    targets = np.zeros((N, len(all_chapters)), dtype=np.int8)
    for i, chapters in enumerate(chapter_sets):
        for ch in chapters:
            targets[i, ch_to_idx[ch]] = 1

    return targets, all_chapters


# =============================================================================
# ICD-10 chapter probing (multi-label one-vs-rest)
# =============================================================================

def probe_icd_blocks(
    embeddings: np.ndarray,
    targets: np.ndarray,
    chapter_names: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """One-vs-rest logistic regression probe for ICD-10 chapter prediction.

    Train a separate LogisticRegression(max_iter=1000) per chapter column
    with per-fold StandardScaler.  Chapters with fewer than *n_splits* +/- 
    samples are skipped (insufficient for stratification).

    Returns
    -------
    dict with:
        per_chapter          : dict mapping chapter key -> metrics dict
        n_chapters_evaluated : int - chapters with enough samples
        macro_auroc          : float
        macro_auprc          : float
        macro_f1             : float
    """
    N, C = targets.shape

    per_chapter: dict[str, dict] = {}
    all_aurocs: list[float] = []
    all_auprcs: list[float] = []
    all_f1s: list[float] = []
    all_briers: list[float] = []
    all_eces: list[float] = []

    for c in range(C):
        y = targets[:, c]
        n_pos = int(y.sum())
        n_neg = N - n_pos

        if n_pos < n_splits or n_neg < n_splits:
            continue

        key = chapter_names[c] if chapter_names is not None else str(c)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_auroc: list[float] = []
        fold_auprc: list[float] = []
        fold_f1: list[float] = []
        fold_brier: list[float] = []
        fold_ece: list[float] = []

        for train_idx, test_idx in skf.split(embeddings, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(embeddings[train_idx])
            X_te = scaler.transform(embeddings[test_idx])
            y_tr, y_te = y[train_idx], y[test_idx]

            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            )
            clf.fit(X_tr, y_tr)

            y_prob = clf.predict_proba(X_te)[:, 1]
            y_pred = clf.predict(X_te)

            m = compute_all_metrics(y_te, y_prob, y_pred)
            fold_auroc.append(m["auroc"])
            fold_auprc.append(m["auprc"])
            fold_f1.append(m["f1"])
            fold_brier.append(m["brier"])
            fold_ece.append(m["ece"])

        mean_auroc = float(np.mean(fold_auroc))
        mean_auprc = float(np.mean(fold_auprc))
        mean_f1 = float(np.mean(fold_f1))
        mean_brier = float(np.mean(fold_brier))
        mean_ece = float(np.mean(fold_ece))

        per_chapter[key] = {
            "n_positive": n_pos,
            "prevalence": round(n_pos / N, 4),
            "mean_auroc": mean_auroc,
            "mean_auprc": mean_auprc,
            "std_auroc": float(np.std(fold_auroc)),
            "std_auprc": float(np.std(fold_auprc)),
            "mean_f1": mean_f1,
            "std_f1": float(np.std(fold_f1)),
            "mean_brier": mean_brier,
            "std_brier": float(np.std(fold_brier)),
            "mean_ece": mean_ece,
            "std_ece": float(np.std(fold_ece)),
        }
        all_aurocs.append(mean_auroc)
        all_auprcs.append(mean_auprc)
        all_f1s.append(mean_f1)
        all_briers.append(mean_brier)
        all_eces.append(mean_ece)

    return {
        "per_chapter": per_chapter,
        "n_chapters_evaluated": len(per_chapter),
        "macro_auroc": float(np.mean(all_aurocs)) if all_aurocs else float("nan"),
        "macro_auprc": float(np.mean(all_auprcs)) if all_auprcs else float("nan"),
        "macro_f1": float(np.mean(all_f1s)) if all_f1s else float("nan"),
        "macro_brier": float(np.mean(all_briers)) if all_briers else float("nan"),
        "macro_ece": float(np.mean(all_eces)) if all_eces else float("nan"),
    }


# =============================================================================
# Readmission probing (binary)
# =============================================================================

def evaluate_readmission(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Probe for binary readmission prediction using:
        - Logistic Regression
        - Stratified k-fold CV with per-fold StandardScaler
        - balanced class weights.

    Parameters
    ----------
    embeddings : (N, D) embedding matrix
    labels     : (N,) binary labels {0, 1}
    n_splits   : number of stratified CV folds
    seed       : random seed

    Returns
    -------
    dict with:
        fold_auroc / fold_auprc / fold_f1 / fold_brier / fold_ece : list[float]
        mean_* / std_* for each metric
        n_samples, n_positive, positive_rate
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_auroc: list[float] = []
    fold_auprc: list[float] = []
    fold_f1: list[float] = []
    fold_brier: list[float] = []
    fold_ece: list[float] = []

    for train_idx, test_idx in skf.split(embeddings, labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(embeddings[train_idx])
        X_te = scaler.transform(embeddings[test_idx])
        y_tr, y_te = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        m = compute_all_metrics(y_te, y_prob, y_pred)
        fold_auroc.append(m["auroc"])
        fold_auprc.append(m["auprc"])
        fold_f1.append(m["f1"])
        fold_brier.append(m["brier"])
        fold_ece.append(m["ece"])

    n_pos = int(labels.sum())
    return {
        "fold_auroc": fold_auroc,
        "fold_auprc": fold_auprc,
        "fold_f1": fold_f1,
        "fold_brier": fold_brier,
        "fold_ece": fold_ece,
        "mean_auroc": float(np.mean(fold_auroc)),
        "std_auroc": float(np.std(fold_auroc)),
        "mean_auprc": float(np.mean(fold_auprc)),
        "std_auprc": float(np.std(fold_auprc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
        "mean_brier": float(np.mean(fold_brier)),
        "std_brier": float(np.std(fold_brier)),
        "mean_ece": float(np.mean(fold_ece)),
        "std_ece": float(np.std(fold_ece)),
        "n_samples": len(labels),
        "n_positive": n_pos,
        "positive_rate": round(n_pos / len(labels), 4),
    }
