"""
Linear probing utilities for layer-wise signal localization and
representation quality assessment.

Functions
---------
  extract_layer_representations : forward-hook extraction at every encoder layer
  probe_vectors                 : generic binary probe across named vectors
  probe_encounter_level         : per-encounter probing with patient-grouped CV
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, f1_score, 
    roc_auc_score, brier_score_loss)

from src.models.jepa_ema import JEPA_EMA
from analysis.eval_infra import (
    flatten_valid_encounters, 
    pool_to_patients, 
    compute_all_metrics)
from src.utils.seed import SEED

# =============================================================================
# Setup
# =============================================================================

def extract_layer_representations(
    model: JEPA_EMA | torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Extract per-layer transformer representations via forward hooks.

    Registers a forward hook on each transformer encoder layer to capture
    intermediate outputs.  Both layer outputs and the final z_enc are
    mean-pooled over valid context positions.

    Returns
    -------
    dict with:
        layer_0 .. layer_{n-1} : (N, D) mean-pooled per-layer outputs
        final                  : (N, D) mean-pooled z_enc from model output
        subject_ids            : (N,) str array
        n_layers               : int
    """
    model.eval()
    layers = model.transformer_layers
    n_layers = len(layers)

    hook_outputs: dict[int, list] = {i: [] for i in range(n_layers)}
    hooks = []

    def _make_hook(layer_idx):
        def hook_fn(module, input, output):
            hook_outputs[layer_idx].append(output.detach())
        return hook_fn

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(_make_hook(i))
        hooks.append(h)

    all_final = []
    all_sids = []
    all_pad_masks = []

    try:
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                z_enc, z_pred, z_target = model(batch)
                # z_enc: (B, C, D) - mean-pool over valid context positions
                ctx_pad_mask = batch["ctx_pad_mask"]  # (B, C) True=padding
                valid = (~ctx_pad_mask).float().unsqueeze(-1)  # (B, C, 1)
                z_enc_pooled = (
                    (z_enc * valid).sum(dim=1) /
                    valid.sum(dim=1).clamp(min=1.0)
                )  # (B, D)
                all_final.append(z_enc_pooled.cpu())
                all_sids.extend(batch["subject_ids"])
                all_pad_masks.append(ctx_pad_mask.cpu())
    finally:
        for h in hooks:
            h.remove()

    # Mean-pool each layer's sequence outputs using padding masks
    result = {}
    for i in range(n_layers):
        layer_seqs = torch.cat(hook_outputs[i], dim=0)  # (N, C, D)
        pad_masks = torch.cat(all_pad_masks, dim=0)     # (N, C)
        valid = (~pad_masks).float().unsqueeze(-1)       # (N, C, 1)
        pooled = (layer_seqs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        result[f"layer_{i}"] = pooled.numpy()

    result["final"] = torch.cat(all_final, dim=0).numpy()
    result["subject_ids"] = np.array(all_sids, dtype=str)
    result["n_layers"] = n_layers

    return result


# =============================================================================
# Probes
# =============================================================================

def probe_vectors(
    vectors: dict[str, np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    pool_to_patient: bool = False,
    n_splits: int = 5,
) -> dict[str, dict]:
    """ Generic binary probe across named vectors.

    Parameters
    ----------
    vectors         : named (N, D) arrays to probe
    labels          : (N,) binary labels
    subject_ids     : (N,) patient IDs
    pool_to_patient : if True, mean-pool vectors per patient before probing
                      (prevents data leakage for patient-level labels with
                      multiple mask positions per patient)
    n_splits        : number of stratified CV folds

    Returns
    -------
    dict mapping vector name -> evaluate_binary_probe metrics dict
    """
    results = {}
    for name, emb in vectors.items():
        if pool_to_patient:
            emb_p, unique_ids = pool_to_patients(emb, subject_ids)
            _, first_idx = np.unique(np.asarray(subject_ids, dtype=str), return_index=True)
            labels_p = labels[first_idx]
            results[name] = evaluate_binary_probe(emb_p, labels_p, n_splits=n_splits)
        else:
            results[name] = evaluate_binary_probe(emb, labels, n_splits=n_splits)
    return results


def probe_encounter_level(
    z_encs: np.ndarray,
    ctx_pad_mask: np.ndarray,
    enc_labels: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5
) -> dict:
    """Probe individual encounter representations with patient-grouped CV.

    Flattens valid encounters across all samples and runs a binary
    LogisticRegression probe.  Uses GroupKFold so that encounters from
    the same patient never appear in both train and test splits.

    Parameters
    ----------
    z_encs        : (N, C, D) encoder outputs per context encounter
    ctx_pad_mask : (N, C) bool, True = padding
    enc_labels    : (N, C) binary labels per encounter
    subject_ids   : (N,) patient IDs per sample
    n_splits      : number of GroupKFold splits

    Returns
    -------
    dict with fold-level and mean/std metrics matching the structure of
    evaluate_binary_probe, plus n_encounters and n_patients.
    """
    valid_mask = ~ctx_pad_mask  # (N, C) True=valid

    # Flatten valid encounters
    X, groups, _ = flatten_valid_encounters(z_encs, ctx_pad_mask, subject_ids)
    y = enc_labels[valid_mask]      # (N_valid,)

    gkf = GroupKFold(n_splits=n_splits)

    fold_auroc: list[float] = []
    fold_auprc: list[float] = []
    fold_f1: list[float] = []
    fold_brier: list[float] = []
    fold_ece: list[float] = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        if y_tr.sum() < 1 or (len(y_tr) - y_tr.sum()) < 1:
            continue
        if y_te.sum() < 1 or (len(y_te) - y_te.sum()) < 1:
            continue

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        m = compute_all_metrics(y_te, y_prob, y_pred)
        fold_auroc.append(m["auroc"])
        fold_auprc.append(m["auprc"])
        fold_f1.append(m["f1"])
        fold_brier.append(m["brier"])
        fold_ece.append(m["ece"])

    n_valid = int(valid_mask.sum())
    n_pos = int(y.sum())
    return {
        "fold_auroc": fold_auroc,
        "fold_auprc": fold_auprc,
        "fold_f1": fold_f1,
        "fold_brier": fold_brier,
        "fold_ece": fold_ece,
        "mean_auroc": float(np.mean(fold_auroc)) if fold_auroc else float("nan"),
        "std_auroc": float(np.std(fold_auroc)) if fold_auroc else float("nan"),
        "mean_auprc": float(np.mean(fold_auprc)) if fold_auprc else float("nan"),
        "std_auprc": float(np.std(fold_auprc)) if fold_auprc else float("nan"),
        "mean_f1": float(np.mean(fold_f1)) if fold_f1 else float("nan"),
        "std_f1": float(np.std(fold_f1)) if fold_f1 else float("nan"),
        "mean_brier": float(np.mean(fold_brier)) if fold_brier else float("nan"),
        "std_brier": float(np.std(fold_brier)) if fold_brier else float("nan"),
        "mean_ece": float(np.mean(fold_ece)) if fold_ece else float("nan"),
        "std_ece": float(np.std(fold_ece)) if fold_ece else float("nan"),
        "n_encounters": n_valid,
        "n_positive": n_pos,
        "positive_rate": round(n_pos / n_valid, 4) if n_valid > 0 else 0.0,
        "n_patients": len(np.unique(groups)),
    }
    

def probe_icd_blocks(
    embeddings: np.ndarray,
    targets: np.ndarray,
    chapter_names: list[str] | None = None,
    n_splits: int = 5,
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
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

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

            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
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


def probe_icd_blocks_temporal(
    embeddings: np.ndarray,
    targets: np.ndarray,
    chapter_names: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    """Single temporal split ICD chapter probing."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embeddings[train_mask])
    X_te = scaler.transform(embeddings[test_mask])

    N_te = int(test_mask.sum())
    C = targets.shape[1]
    per_chapter: dict[str, dict] = {}
    macro_aurocs: list[float] = []
    macro_auprcs: list[float] = []
    macro_f1s: list[float] = []

    for c in range(C):
        y_tr = targets[train_mask, c]
        y_te = targets[test_mask, c]

        if y_tr.sum() < 1 or (len(y_tr) - y_tr.sum()) < 1:
            continue
        if y_te.sum() < 1 or (len(y_te) - y_te.sum()) < 1:
            continue

        key = chapter_names[c] if chapter_names else str(c)
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        auroc = float(roc_auc_score(y_te, y_prob))
        auprc = float(average_precision_score(y_te, y_prob))
        f1_val = float(f1_score(y_te, y_pred))

        n_pos = int(y_te.sum())
        per_chapter[key] = {
            "n_positive": n_pos,
            "prevalence": round(n_pos / N_te, 4),
            "mean_auroc": auroc, "std_auroc": 0.0,
            "mean_auprc": auprc, "std_auprc": 0.0,
            "mean_f1": f1_val, "std_f1": 0.0,
        }
        macro_aurocs.append(auroc)
        macro_auprcs.append(auprc)
        macro_f1s.append(f1_val)

    return {
        "per_chapter": per_chapter,
        "n_chapters_evaluated": len(per_chapter),
        "macro_auroc": float(np.mean(macro_aurocs)) if macro_aurocs else float("nan"),
        "macro_auprc": float(np.mean(macro_auprcs)) if macro_auprcs else float("nan"),
        "macro_f1": float(np.mean(macro_f1s)) if macro_f1s else float("nan"),
    }

# =============================================================================
# Probe Evaluation
# =============================================================================

def evaluate_binary_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5
) -> dict:
    """Binary logistic-regression probe with stratified k-fold CV.

    Uses per-fold StandardScaler and balanced class weights.

    Parameters
    ----------
    embeddings : (N, D) embedding matrix
    labels     : (N,) binary labels {0, 1}
    n_splits   : number of stratified CV folds

    Returns
    -------
    dict with:
        fold_auroc / fold_auprc / fold_f1 / fold_brier / fold_ece : list[float]
        mean_* / std_* for each metric
        n_samples, n_positive, positive_rate
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

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

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
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


def evaluate_binary_probe_temporal(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    """Single temporal train/test split binary probe.

    Returns the same dict structure as evaluate_binary_probe so that
    downstream formatting code works unchanged (fold lists have length 1,
    std values are 0.0).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embeddings[train_mask])
    X_te = scaler.transform(embeddings[test_mask])
    y_tr, y_te = labels[train_mask], labels[test_mask]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    auroc = float(roc_auc_score(y_te, y_prob))
    auprc = float(average_precision_score(y_te, y_prob))
    f1_val = float(f1_score(y_te, y_pred))
    brier = float(brier_score_loss(y_te, y_prob))

    n_test = int(test_mask.sum())
    n_pos = int(y_te.sum())
    return {
        "fold_auroc": [auroc], "fold_auprc": [auprc],
        "fold_f1": [f1_val], "fold_brier": [brier],
        "mean_auroc": auroc, "std_auroc": 0.0,
        "mean_auprc": auprc, "std_auprc": 0.0,
        "mean_f1": f1_val, "std_f1": 0.0,
        "mean_brier": brier, "std_brier": 0.0,
        "n_samples": n_test,
        "n_positive": n_pos,
        "positive_rate": round(n_pos / n_test, 4) if n_test > 0 else 0.0,
        "n_train": int(train_mask.sum()),
        "n_test": n_test,
    }