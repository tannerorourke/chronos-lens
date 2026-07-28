"""
Linear probing utilities for layer-wise signal localization and
representation quality assessment.

Functions
---------
  extract_layer_representations : forward-hook extraction at every encoder layer
  probe_vectors                 : generic binary probe across named vectors
  probe_encounter_level         : per-encounter probing with patient-grouped CV
  probe_icd_blocks              : one-vs-rest ICD-10 chapter probes (+ temporal variant)
  evaluate_binary_probe         : stratified-CV logistic probe (+ temporal variant)
  run_probing_sweep             : layer-by-layer probe sweep for signal localization
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, f1_score, 
    roc_auc_score, brier_score_loss)

from src.models import MODEL_TYPE
from src.infra.vector_computation import (
    flatten_valid_encounters,
    select_terminal_by_patient)
from src.infra.metrics import compute_all_metrics, make_cv_splitter
from src.utils.system import SEED

# =============================================================================
# Setup
# =============================================================================

def extract_layer_representations(
    model: MODEL_TYPE,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Extract per-layer transformer representations via forward hooks.

    Registers a forward hook on each transformer encoder layer to capture
    intermediate outputs. Both layer outputs and the final z_enc are reduced to
    the recency encounter `z_enc[k-1]` (the most-recent context slot), not a
    context mean - the recency point is the consistent per-encounter vector.

    Encodes context only, so each hook fires once per batch.

    Returns
    -------
    dict with:
        layer_0 .. layer_{n-1} : (N, D) recency per-layer outputs
        final                  : (N, D) recency z_enc from model output
        subject_ids            : (N,) str array
        mask_pos               : (N,) int array - target encounter index per
                                 sample, for aligning causal per-encounter labels
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
    all_mask_pos = []

    def _recency(seq: torch.Tensor, mpos: torch.Tensor) -> torch.Tensor:
        # (B, C, D) -> (B, D): the last valid context slot, index mask_pos-1
        idx = (mpos - 1).long()
        assert int(idx.max()) < seq.size(1), (
            f"recency index {int(idx.max())} outside context length {seq.size(1)} - "
            "captured activations are not the context pass")
        rows = torch.arange(seq.size(0))
        return seq[rows, idx]

    try:
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                # -- context pass only: (B, C, D), the same z_enc every arch's forward returns first
                z_enc = model.encode(batch)
                mpos = batch["mask_pos"].cpu()
                all_final.append(_recency(z_enc.cpu(), mpos))   # slice recency -> (B, D)
                all_sids.extend(batch["subject_ids"])
                all_mask_pos.append(mpos)
    finally:
        for h in hooks:
            h.remove()

    # -- one capture per layer per batch, or the hooks saw a pass we did not intend
    for i in range(n_layers):
        assert len(hook_outputs[i]) == len(all_mask_pos), (
            f"layer {i} captured {len(hook_outputs[i])} batches for {len(all_mask_pos)} "
            "encoded - hooks fired on an unintended forward pass")

    # Slice each layer's recency encounter per batch *before* concatenating:
    # context length C varies across batches, so raw (B, C, D) tensors can't be
    # stacked, but the recency (B, D) slices can.
    result = {}
    for i in range(n_layers):
        rec_batches = []
        for layer_seqs, mpos in zip(hook_outputs[i], all_mask_pos):
            rec_batches.append(_recency(layer_seqs.cpu(), mpos))   # (B, D)
        result[f"layer_{i}"] = torch.cat(rec_batches, dim=0).numpy()

    result["final"] = torch.cat(all_final, dim=0).numpy()
    result["subject_ids"] = np.array(all_sids, dtype=str)
    result["mask_pos"] = torch.cat(all_mask_pos, dim=0).numpy()
    result["n_layers"] = n_layers

    return result


# =============================================================================
# Probes
# =============================================================================

def probe_vectors(
    vectors: dict[str, np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    to_patient: bool = False,
    mask_pos: np.ndarray | None = None,
    n_splits: int = 5,
) -> dict[str, dict]:
    """ Generic binary probe across named vectors.

    Parameters
    ----------
    vectors     : named (N, D) arrays to probe
    labels      : (N,) binary labels
    subject_ids : (N,) patient IDs
    to_patient  : if True, reduce to one row per patient (their terminal sample,
                  largest mask_pos) before probing - prevents data leakage for
                  patient-level labels with multiple mask positions per patient.
                  Requires `mask_pos`.
    mask_pos    : (N,) target encounter index per sample; required when
                  `to_patient` is True.
    n_splits    : number of stratified CV folds

    Returns
    -------
    dict mapping vector name -> evaluate_binary_probe metrics dict
    """
    results = {}
    for name, emb in vectors.items():
        if to_patient:
            assert mask_pos is not None, "probe_vectors(to_patient=True) requires mask_pos"
            emb_p, _ = select_terminal_by_patient(emb, subject_ids, mask_pos)
            # -- reduce labels through the same selection, so each row's label is the one at the
            #    encounter its embedding came from. Causal labels vary with mask_pos, so taking
            #    the patient's first occurrence instead would pair encounter k's vector with
            #    encounter 0's label.
            labels_p, _ = select_terminal_by_patient(
                np.asarray(labels).reshape(-1, 1), subject_ids, mask_pos)
            results[name] = evaluate_binary_probe(emb_p, labels_p.ravel(), n_splits=n_splits)
        else:
            results[name] = evaluate_binary_probe(
                emb, labels, n_splits=n_splits,
                groups=np.asarray(subject_ids, dtype=str))
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
    LogisticRegression probe.  GroupKFold keeps encounters from the same
    patient out of both train and test in any one split.

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
        cv_scheme            : str
    """
    N, C = targets.shape

    # -- no group key is passed in, so folds are not patient-held-out
    skf, cv_scheme = make_cv_splitter(n_splits=n_splits)

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
        "cv_scheme": cv_scheme,
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
    n_splits: int = 5,
    groups: np.ndarray | None = None,
) -> dict:
    """Binary logistic-regression probe with stratified k-fold CV.

    Uses per-fold StandardScaler and balanced class weights.

    groups is held out whole; a patient's rows are not independent.

    Parameters
    ----------
    embeddings : (N, D) embedding matrix
    labels     : (N,) binary labels {0, 1}
    n_splits   : number of stratified CV folds
    groups     : (N,) group key per row, held out whole. None = rows are independent.

    Returns
    -------
    dict with:
        fold_auroc / fold_auprc / fold_f1 / fold_brier / fold_ece : list[float]
        mean_* / std_* for each metric
        n_samples, n_positive, positive_rate, cv_scheme
    """
    splitter, cv_scheme = make_cv_splitter(n_splits=n_splits, groups=groups)

    fold_auroc: list[float] = []
    fold_auprc: list[float] = []
    fold_f1: list[float] = []
    fold_brier: list[float] = []
    fold_ece: list[float] = []

    for train_idx, test_idx in splitter.split(embeddings, labels, groups=groups):
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
        "cv_scheme": cv_scheme,
        "n_groups": int(len(np.unique(groups))) if groups is not None else None,
    }


def evaluate_binary_probe_temporal(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    """Single temporal train/test split binary probe.

    Returns the same dict structure as evaluate_binary_probe; fold lists
    have length 1 and std values are 0.0.
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


# =============================================================================
# Layer-wise probing sweep (signal localization)
# =============================================================================

def run_probing_sweep(
    layer_representations: dict,
    labels: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Probe every transformer layer + the final z_enc to localize signal.

    Runs :func:`evaluate_binary_probe` (stratified k-fold, AUROC/AUPRC/F1/Brier/
    ECE) on each layer's recency representation and reports where prediction
    signal emerges through the encoder.

    Folds are grouped on subject_id; a patient's rows are not independent.

    Parameters
    ----------
    layer_representations : output of :func:`extract_layer_representations` -
        dict with `layer_0` .. `layer_{n-1}`, `final`, `subject_ids`, and `n_layers`.
    labels : (N,) binary labels aligned to the representation rows.
    n_splits : number of stratified CV folds.

    Returns
    -------
    dict with:
        per_layer      : layer key -> evaluate_binary_probe metrics dict
        summary        : ordered list of (layer_key, mean_auroc, std_auroc)
        best_layer     : layer key with highest mean AUROC
        best_auroc     : that layer's mean AUROC
        interpretation : narrative string on signal localization
    """
    n_layers = layer_representations["n_layers"]
    layer_keys = [f"layer_{i}" for i in range(n_layers)] + ["final"]
    groups = np.asarray(layer_representations["subject_ids"], dtype=str)

    per_layer: dict[str, dict] = {}
    summary: list[tuple[str, float, float]] = []

    for key in layer_keys:
        X = layer_representations[key]
        result = evaluate_binary_probe(X, labels, n_splits=n_splits, groups=groups)
        per_layer[key] = result
        summary.append((key, result["mean_auroc"], result["std_auroc"]))
        print(f"  {key:12s}  AUROC={result['mean_auroc']:.4f} +/- {result['std_auroc']:.4f}  "
              f"AUPRC={result['mean_auprc']:.4f}  F1={result['mean_f1']:.4f}")

    best_key = max(per_layer, key=lambda k: per_layer[k]["mean_auroc"])
    best_auroc = per_layer[best_key]["mean_auroc"]

    # Localization: how much does signal grow from the first layer to z_enc?
    early_auroc = per_layer["layer_0"]["mean_auroc"]
    final_auroc = per_layer["final"]["mean_auroc"]
    delta = final_auroc - early_auroc

    if delta < 0.02:
        interpretation = (
            f"Signal is already present at layer 0 (AUROC={early_auroc:.3f}) and "
            f"does not improve substantially through the encoder (final "
            f"AUROC={final_auroc:.3f}). The token embeddings already separate the "
            f"classes - attention primarily organizes geometric structure."
        )
    elif delta < 0.05:
        interpretation = (
            f"Modest signal gain from layer 0 (AUROC={early_auroc:.3f}) to final "
            f"(AUROC={final_auroc:.3f}). Both embedding-level features and "
            f"attention-derived structure contribute to separability."
        )
    else:
        interpretation = (
            f"Substantial signal gain from layer 0 (AUROC={early_auroc:.3f}) to "
            f"final (AUROC={final_auroc:.3f}), delta={delta:.3f}. Attention is "
            f"actively building prediction signal through its layers."
        )

    return {
        "per_layer": per_layer,
        "summary": summary,
        "best_layer": best_key,
        "best_auroc": best_auroc,
        "interpretation": interpretation,
    }