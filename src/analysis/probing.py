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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.jepa_ema import JEPA_EMA
from src.analysis.eval_tasks import evaluate_binary_probe
from src.analysis.metrics import compute_all_metrics


def extract_layer_representations(
    model: JEPA_EMA | JEPAStopGrad,
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
# Patient-level pooling
# =============================================================================

def _pool_to_patients(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids, inverse = np.unique(subject_ids, return_inverse=True)
    n_patients = len(unique_ids)
    D = embeddings.shape[1]

    patient_embs = np.zeros((n_patients, D), dtype=embeddings.dtype)
    counts = np.zeros(n_patients, dtype=np.int64)
    patient_labels = np.empty(n_patients, dtype=labels.dtype)

    for i, inv in enumerate(inverse):
        patient_embs[inv] += embeddings[i]
        counts[inv] += 1
        patient_labels[inv] = labels[i]

    patient_embs /= counts[:, None]
    return patient_embs, patient_labels


# =============================================================================
# Generic vector probing
# =============================================================================

def probe_vectors(
    vectors: dict[str, np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    pool_to_patient: bool = False,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, dict]:
    """Binary probe across named vectors.

    Parameters
    ----------
    vectors         : named (N, D) arrays to probe
    labels          : (N,) binary labels
    subject_ids     : (N,) patient IDs
    pool_to_patient : if True, mean-pool vectors per patient before probing
                      (prevents data leakage for patient-level labels with
                      multiple mask positions per patient)
    n_splits        : number of stratified CV folds
    seed            : random seed

    Returns
    -------
    dict mapping vector name -> evaluate_binary_probe metrics dict
    """
    results = {}
    for name, emb in vectors.items():
        if pool_to_patient:
            emb_p, labels_p = _pool_to_patients(emb, subject_ids, labels)
            results[name] = evaluate_binary_probe(
                emb_p, labels_p, n_splits=n_splits, seed=seed)
        else:
            results[name] = evaluate_binary_probe(
                emb, labels, n_splits=n_splits, seed=seed)
    return results


# =============================================================================
# Encounter-level probing
# =============================================================================

def probe_encounter_level(
    z_encs: np.ndarray,
    ctx_pad_masks: np.ndarray,
    enc_labels: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Probe individual encounter representations with patient-grouped CV.

    Flattens valid encounters across all samples and runs a binary
    LogisticRegression probe.  Uses GroupKFold so that encounters from
    the same patient never appear in both train and test splits.

    Parameters
    ----------
    z_encs        : (N, C, D) encoder outputs per context encounter
    ctx_pad_masks : (N, C) bool, True = padding
    enc_labels    : (N, C) binary labels per encounter
    subject_ids   : (N,) patient IDs per sample
    n_splits      : number of GroupKFold splits
    seed          : random seed

    Returns
    -------
    dict with fold-level and mean/std metrics matching the structure of
    evaluate_binary_probe, plus n_encounters and n_patients.
    """
    N, C, D = z_encs.shape
    valid_mask = ~ctx_pad_masks  # (N, C) True=valid

    # Flatten valid encounters
    X = z_encs[valid_mask]          # (N_valid, D)
    y = enc_labels[valid_mask]      # (N_valid,)

    # Expand subject_ids to (N, C) then flatten with same mask
    sid_expanded = np.repeat(subject_ids[:, np.newaxis], C, axis=1)
    groups = sid_expanded[valid_mask]  # (N_valid,)

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

        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed)
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
