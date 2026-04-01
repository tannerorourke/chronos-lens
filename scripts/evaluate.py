#!/usr/bin/env python3
"""
Unified evaluation script for all trained models (stopgrad, ema, supervised).

Loads a checkpoint, extracts embeddings, and runs downstream probing tasks
on each available latent vector.  Outputs a structured JSON and a summary
table to stdout.

Tasks
-----
  readmit_30d      : 30-day F-code readmission (patient-level, pooled vectors)
  escalation       : per-encounter escalation binary probe
  icd_block        : ICD-10 chapter prediction for the masked encounter
  escalation_type  : per-criterion escalation probes (6 binary probes)

Usage
-----
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --tasks readmit_30d,escalation
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --split temporal
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --eval-subset fcode
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.training.utils.checkpoint import load_model_notrain
from src.training.utils.datasets import (
    MimicDataset, collate_fn,
    SupervisedDataset, supervised_collate_fn,
    build_vocab,
)
from src.analysis.eval_tasks import (
    evaluate_binary_probe,
    extract_icd_block_targets,
    probe_icd_blocks,
)
from src.mimic.labels import (
    _check_escalation,
    _get_f_codes_full,
    _update_state,
)
from src.utils.io import load_sequences, DATA_DIR


ALL_TASKS = ["readmit_30d", "escalation", "icd_block", "escalation_type"]

ESCALATION_CRITERIA = [
    "new_subcategory", "severity_increase", "new_specifier",
    "f32_to_f33", "med_initiation", "new_drug_class",
]

# Vectors available per architecture family
JEPA_VECTORS = ["z_pred", "z_target", "pred_error", "z_enc_pooled"]
SUPERVISED_VECTORS = ["z_enc_pooled"]


# =============================================================================
# Embedding extraction
# =============================================================================

def extract_jepa_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run JEPA model inference and collect embeddings.

    Returns dict with keys: z_encs, z_pred, z_target, ctx_pad_masks,
    subject_ids, mask_pos.
    """
    all_z_encs: list[np.ndarray] = []
    all_z_pred: list[np.ndarray] = []
    all_z_target: list[np.ndarray] = []
    all_ctx_pad_masks: list[np.ndarray] = []
    all_subject_ids: list[str] = []
    all_mask_pos: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            z_enc, z_pred, z_target = model(batch)
            # z_enc: (B, C, D), z_pred: (B, D), z_target: (B, D)

            all_z_encs.append(z_enc.cpu().numpy())
            all_z_pred.append(z_pred.cpu().numpy())
            all_z_target.append(z_target.cpu().numpy())
            all_ctx_pad_masks.append(batch["ctx_pad_mask"].cpu().numpy())
            all_subject_ids.extend(batch["subject_ids"])
            all_mask_pos.append(batch["mask_pos"].cpu().numpy())

    # Pad z_encs and ctx_pad_masks to uniform context length across batches
    max_C = max(arr.shape[1] for arr in all_z_encs)
    D = all_z_encs[0].shape[2]
    padded_z_encs: list[np.ndarray] = []
    padded_masks: list[np.ndarray] = []
    for z, m in zip(all_z_encs, all_ctx_pad_masks):
        B, C = z.shape[0], z.shape[1]
        if C < max_C:
            z = np.concatenate(
                [z, np.zeros((B, max_C - C, D), dtype=z.dtype)], axis=1)
            m = np.concatenate(
                [m, np.ones((B, max_C - C), dtype=m.dtype)], axis=1)
        padded_z_encs.append(z)
        padded_masks.append(m)

    return {
        "z_encs": np.concatenate(padded_z_encs),           # (N, C_max, D)
        "z_pred": np.concatenate(all_z_pred),               # (N, D)
        "z_target": np.concatenate(all_z_target),           # (N, D)
        "ctx_pad_masks": np.concatenate(padded_masks),      # (N, C_max)
        "subject_ids": np.array(all_subject_ids),           # (N,)
        "mask_pos": np.concatenate(all_mask_pos),           # (N,)
    }


def extract_supervised_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run supervised model inference and collect encoder output.

    Returns dict with keys: z_enc_pooled, subject_ids.
    """
    all_z_enc_pooled: list[np.ndarray] = []
    all_subject_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            z_context, _logits = model(batch)
            all_z_enc_pooled.append(z_context.cpu().numpy())
            all_subject_ids.extend(batch["subject_ids"])

    return {
        "z_enc_pooled": np.concatenate(all_z_enc_pooled),  # (N, D)
        "subject_ids": np.array(all_subject_ids),           # (N,)
    }


# =============================================================================
# Derived vectors
# =============================================================================

def compute_derived_vectors(raw_vecs: dict) -> dict:
    """Compute pred_error and z_enc_pooled from raw embedding arrays."""
    vecs = dict(raw_vecs)

    if "z_pred" in vecs and "z_target" in vecs:
        vecs["pred_error"] = vecs["z_pred"] - vecs["z_target"]

    if "z_encs" in vecs and "ctx_pad_masks" in vecs:
        z_encs = vecs["z_encs"]              # (N, C, D)
        pad_masks = vecs["ctx_pad_masks"]    # (N, C)  True=padding
        valid = (~pad_masks).astype(np.float32)[..., np.newaxis]  # (N, C, 1)
        vecs["z_enc_pooled"] = (
            (z_encs * valid).sum(axis=1) /
            valid.sum(axis=1).clip(min=1.0)
        )  # (N, D)

    return vecs


# =============================================================================
# Patient-level pooling
# =============================================================================

def pool_to_patients(
    vecs_dict: dict[str, np.ndarray],
    subject_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Average sample-level vectors per unique subject_id.

    Returns (pooled_dict, unique_subject_ids).  Pooled keys get a
    ``_pooled`` suffix unless the name already ends with it.
    """
    unique_ids, inverse = np.unique(subject_ids, return_inverse=True)
    n_patients = len(unique_ids)

    pooled: dict[str, np.ndarray] = {}
    for name, emb in vecs_dict.items():
        if emb is None:
            continue
        D = emb.shape[1]
        acc = np.zeros((n_patients, D), dtype=np.float64)
        counts = np.zeros(n_patients, dtype=np.float64)
        for i in range(len(subject_ids)):
            idx = inverse[i]
            acc[idx] += emb[i]
            counts[idx] += 1
        pooled_name = name if name.endswith("_pooled") else name + "_pooled"
        pooled[pooled_name] = (acc / counts[:, np.newaxis]).astype(np.float32)

    return pooled, unique_ids


# =============================================================================
# Label loading
# =============================================================================

def load_patients_dict(sequences_path: Path) -> dict[str, dict]:
    """Load sequences.jsonl into {subject_id_str: patient_dict}."""
    patients: dict[str, dict] = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p
    return patients


def load_label_30d(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
) -> np.ndarray:
    """Patient-level 30-day F-code readmission labels."""
    return np.array([
        patients_dict[str(sid)].get("label_30d", 0)
        for sid in subject_ids
    ], dtype=np.int64)


def load_escalation_labels(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """Per-encounter escalation labels at each sample's mask position."""
    labels = np.zeros(len(subject_ids), dtype=np.int64)
    for i, (sid, pos) in enumerate(zip(subject_ids, mask_pos)):
        patient = patients_dict[str(sid)]
        per_enc = patient.get("label_escalation_per_enc", [])
        pos = int(pos)
        if pos < len(per_enc):
            labels[i] = per_enc[pos]
    return labels


def compute_escalation_criteria_labels(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute per-encounter escalation criteria for each sample.

    Replays the escalation state machine from encounter 0 to mask_pos-1,
    then checks which criteria fire at mask_pos.

    Returns dict mapping criterion_name -> (N,) binary int64 array.
    """
    N = len(subject_ids)
    criteria_labels = {c: np.zeros(N, dtype=np.int64) for c in ESCALATION_CRITERIA}

    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_pos[i])
        encs = patients_dict[sid]["encounters"]

        if pos == 0:
            continue

        # Build prior state from encounters 0 .. pos-1
        prior_subcats: dict[str, int] = {}
        prior_f_codes: set[str] = set()
        prior_drug_classes: set[str] = set()
        has_prior_psych_meds = False

        for j in range(pos):
            f_codes = _get_f_codes_full(encs[j])
            meds = [m.lower() for m in encs[j].get("meds", [])]
            had_meds = _update_state(
                f_codes, meds, prior_subcats, prior_f_codes, prior_drug_classes)
            has_prior_psych_meds = has_prior_psych_meds or had_meds

        # Check escalation at mask_pos
        f_codes = _get_f_codes_full(encs[pos])
        meds = [m.lower() for m in encs[pos].get("meds", [])]
        fired = _check_escalation(
            f_codes, meds,
            prior_subcats, prior_f_codes, prior_drug_classes,
            has_prior_psych_meds,
        )

        for criterion in fired:
            if criterion in criteria_labels:
                criteria_labels[criterion][i] = 1

    return criteria_labels


# =============================================================================
# Temporal train/test split
# =============================================================================

def compute_temporal_split(
    sequences_path: Path,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split samples into train/test by median latest admission date.

    All encounter windows from the same patient are assigned to the same
    split.  Patients whose latest admission is strictly before the median
    cutoff go to train; the rest go to test.

    Returns
    -------
    train_mask  : (N,) bool array over subject_ids
    test_mask   : (N,) bool array over subject_ids
    cutoff_iso  : ISO-format string of the cutoff date
    """
    patients: dict[str, dict] = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    latest: dict[str, datetime] = {}
    for pid, p in patients.items():
        dates = [datetime.fromisoformat(e["admittime"])
                 for e in p["encounters"] if "admittime" in e]
        if dates:
            latest[pid] = max(dates)

    all_dates = sorted(latest.values())
    cutoff = all_dates[len(all_dates) // 2]

    train_mask = np.array([latest.get(str(sid), cutoff) < cutoff
                           for sid in subject_ids])
    test_mask = ~train_mask

    return train_mask, test_mask, cutoff.isoformat()


def evaluate_binary_probe_temporal(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int = 42,
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

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed)
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


def probe_icd_blocks_temporal(
    embeddings: np.ndarray,
    targets: np.ndarray,
    chapter_names: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int = 42,
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
            max_iter=1000, class_weight="balanced", random_state=seed)
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
# Cohort subset filtering
# =============================================================================

def compute_subset_mask(
    sequences_path: Path,
    subject_ids: np.ndarray,
    subset: str,
) -> np.ndarray:
    """Compute boolean mask for --eval-subset filtering.

    Parameters
    ----------
    sequences_path : path to sequences.jsonl
    subject_ids    : (N,) sample-level patient IDs
    subset         : "all", "fcode", or "non_fcode"

    Returns
    -------
    (N,) bool array - True for samples to keep
    """
    if subset == "all":
        return np.ones(len(subject_ids), dtype=bool)

    patients: dict[str, dict] = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    fcode_pids: set[str] = set()
    for pid, p in patients.items():
        for enc in p["encounters"]:
            if any(c.upper().startswith("F3") for c in enc.get("icd_codes", [])):
                fcode_pids.add(pid)
                break

    is_fcode = np.array([str(sid) in fcode_pids for sid in subject_ids])
    return is_fcode if subset == "fcode" else ~is_fcode


def filter_vecs(vecs: dict, mask: np.ndarray) -> dict:
    """Apply a boolean sample mask to all arrays in an embedding dict."""
    filtered = {}
    for key, val in vecs.items():
        if val is None:
            filtered[key] = None
        elif isinstance(val, np.ndarray):
            filtered[key] = val[mask]
        else:
            filtered[key] = val
    return filtered


# =============================================================================
# Result formatting
# =============================================================================

def _format_binary_result(res: dict) -> dict:
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


# =============================================================================
# Core evaluation
# =============================================================================

def run_tasks(
    vecs: dict,
    patients_dict: dict[str, dict],
    sequences_path: Path,
    tasks: list[str],
    is_supervised: bool,
    seed: int,
    split: str = "random",
    temporal_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Run requested evaluation tasks on all available latent vectors.

    Patient-level tasks (readmit_30d) use vectors pooled across mask
    positions per patient to prevent data leakage.  Encounter-level tasks
    (escalation, icd_block, escalation_type) use sample-level vectors.
    """
    subject_ids = vecs["subject_ids"]
    mask_pos = vecs.get("mask_pos")
    results: dict = {}

    # -- Build encounter-level vector dict ------------------------------------
    if is_supervised:
        enc_vectors = {"z_enc_pooled": vecs["z_enc_pooled"]}
    else:
        enc_vectors = {
            name: vecs[name] for name in JEPA_VECTORS
            if vecs.get(name) is not None
        }

    # -- Build patient-level vector dict --------------------------------------
    if is_supervised:
        pat_vectors = {"z_enc_pooled": vecs["z_enc_pooled"]}
        pat_ids = subject_ids  # one sample per patient already
    else:
        pat_vectors, pat_ids = pool_to_patients(enc_vectors, subject_ids)

    # Patient-level temporal masks (computed separately from sample-level)
    pat_temporal: tuple[np.ndarray, np.ndarray] | None = None
    if split == "temporal":
        pat_train, pat_test, _ = compute_temporal_split(
            sequences_path, pat_ids)
        pat_temporal = (pat_train, pat_test)

    # -- readmit_30d (patient-level) ------------------------------------------
    if "readmit_30d" in tasks:
        labels_30d = load_label_30d(patients_dict, pat_ids)
        task_results: dict = {}
        for vec_name, emb in pat_vectors.items():
            if split == "temporal":
                res = evaluate_binary_probe_temporal(
                    emb, labels_30d,
                    pat_temporal[0], pat_temporal[1], seed=seed)
            else:
                res = evaluate_binary_probe(emb, labels_30d, seed=seed)
            task_results[vec_name] = _format_binary_result(res)
        results["readmit_30d"] = task_results

    # -- escalation (encounter-level) -----------------------------------------
    if "escalation" in tasks:
        if is_supervised or mask_pos is None:
            print("  [escalation] skipped - requires JEPA model with mask positions")
        else:
            labels_esc = load_escalation_labels(
                patients_dict, subject_ids, mask_pos)
            task_results = {}
            for vec_name, emb in enc_vectors.items():
                if split == "temporal":
                    assert temporal_masks is not None
                    res = evaluate_binary_probe_temporal(
                        emb, labels_esc,
                        temporal_masks[0], temporal_masks[1], seed=seed)
                else:
                    res = evaluate_binary_probe(emb, labels_esc, seed=seed)
                task_results[vec_name] = _format_binary_result(res)
            results["escalation"] = task_results

    # -- icd_block (encounter-level) ------------------------------------------
    if "icd_block" in tasks:
        if is_supervised or mask_pos is None:
            print("  [icd_block] skipped - requires JEPA model with mask positions")
        else:
            targets, chapter_names = extract_icd_block_targets(
                sequences_path, subject_ids, mask_pos)
            task_results = {}
            for vec_name, emb in enc_vectors.items():
                if split == "temporal":
                    assert temporal_masks is not None
                    res = probe_icd_blocks_temporal(
                        emb, targets, chapter_names,
                        temporal_masks[0], temporal_masks[1], seed=seed)
                else:
                    res = probe_icd_blocks(
                        emb, targets, chapter_names, seed=seed)
                task_results[vec_name] = {
                    "macro_auroc": res["macro_auroc"],
                    "macro_auprc": res["macro_auprc"],
                    "macro_f1":    res["macro_f1"],
                    "n_chapters_evaluated": res["n_chapters_evaluated"],
                }
            results["icd_block"] = task_results

    # -- escalation_type (encounter-level) ------------------------------------
    if "escalation_type" in tasks:
        if is_supervised or mask_pos is None:
            print("  [escalation_type] skipped - requires JEPA model with mask positions")
        else:
            criteria_labels = compute_escalation_criteria_labels(
                patients_dict, subject_ids, mask_pos)
            task_results = {}
            for vec_name, emb in enc_vectors.items():
                per_criterion: dict = {}
                macro_aurocs: list[float] = []
                macro_auprcs: list[float] = []
                macro_f1s: list[float] = []

                for criterion in ESCALATION_CRITERIA:
                    labels = criteria_labels[criterion]
                    n_pos = int(labels.sum())
                    n_neg = len(labels) - n_pos

                    if split == "temporal":
                        assert temporal_masks is not None
                        y_tr = labels[temporal_masks[0]]
                        y_te = labels[temporal_masks[1]]
                        if (y_tr.sum() < 1 or (len(y_tr) - y_tr.sum()) < 1
                                or y_te.sum() < 1
                                or (len(y_te) - y_te.sum()) < 1):
                            continue
                        res = evaluate_binary_probe_temporal(
                            emb, labels,
                            temporal_masks[0], temporal_masks[1], seed=seed)
                    else:
                        if n_pos < 5 or n_neg < 5:
                            continue
                        res = evaluate_binary_probe(emb, labels, seed=seed)

                    per_criterion[criterion] = _format_binary_result(res)
                    macro_aurocs.append(res["mean_auroc"])
                    macro_auprcs.append(res["mean_auprc"])
                    macro_f1s.append(res["mean_f1"])

                task_results[vec_name] = {
                    "per_criterion": per_criterion,
                    "n_criteria_evaluated": len(per_criterion),
                    "macro_auroc": float(np.mean(macro_aurocs)) if macro_aurocs else float("nan"),
                    "macro_auprc": float(np.mean(macro_auprcs)) if macro_auprcs else float("nan"),
                    "macro_f1": float(np.mean(macro_f1s)) if macro_f1s else float("nan"),
                }
            results["escalation_type"] = task_results

    return results


# =============================================================================
# Summary table
# =============================================================================

def print_summary(
    architecture: str,
    results: dict,
    split: str = "random",
    eval_subset: str = "all",
) -> None:
    """Print a formatted summary table to stdout."""
    label = f"{architecture} ({split} split"
    if eval_subset != "all":
        label += f", {eval_subset} subset"
    label += ")"
    print("\n" + "=" * 80)
    print(f"  Evaluation Summary - {label}")
    print("=" * 80)

    for task_name, task_results in results.items():
        print(f"\n  {task_name}")
        print(f"  {'-' * 70}")

        if task_name in ("icd_block", "escalation_type"):
            col = "n_ch" if task_name == "icd_block" else "n_crit"
            n_key = ("n_chapters_evaluated" if task_name == "icd_block"
                     else "n_criteria_evaluated")
            header = (f"  {'vector':<20s} {'macro_AUROC':>12s} "
                      f"{'macro_AUPRC':>12s} {'macro_F1':>12s} {col:>6s}")
            print(header)
            for vec_name, metrics in task_results.items():
                print(f"  {vec_name:<20s} "
                      f"{metrics['macro_auroc']:>12.4f} "
                      f"{metrics['macro_auprc']:>12.4f} "
                      f"{metrics['macro_f1']:>12.4f} "
                      f"{metrics[n_key]:>6d}")
        else:
            header = (f"  {'vector':<20s} {'AUROC':>12s} {'AUPRC':>12s} "
                      f"{'F1':>12s} {'Brier':>12s}")
            print(header)
            for vec_name, metrics in task_results.items():
                print(f"  {vec_name:<20s} "
                      f"{metrics['auroc']:>12.4f} "
                      f"{metrics['auprc']:>12.4f} "
                      f"{metrics['f1']:>12.4f} "
                      f"{metrics['brier']:>12.4f}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on downstream tasks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--sequences", type=str,
                        default=str(DATA_DIR / "sequences.jsonl"),
                        help="Path to sequences.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for results JSON (default: <run_dir>/eval_results.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str,
                        default="readmit_30d,escalation,icd_block,escalation_type",
                        help="Comma-separated task list")
    parser.add_argument("--split", type=str, default="random",
                        choices=["random", "temporal"],
                        help="Split strategy: random (stratified CV) or temporal")
    parser.add_argument("--eval-subset", type=str, default="all",
                        choices=["all", "fcode", "non_fcode"],
                        help="Evaluate on a patient subset: fcode (F30-F39), non_fcode, or all")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    tasks = [t.strip() for t in args.tasks.split(",")]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"Unknown task '{t}'. Available: {ALL_TASKS}")

    # -- Resolve experiment directory -----------------------------------------
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.yaml"
    vocab_path = run_dir / "vocab.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config["data"]
    sequences_path = Path(args.sequences)

    # -- Load model -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_notrain(ckpt_path, device, restore_rng=False)
    model_params = checkpoint["model_params"]
    architecture = model_params.get("architecture", "unknown")
    epoch = checkpoint.get("epoch", "?")
    is_supervised = architecture == "supervised"

    print(f"  Model:        {architecture}")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Epoch:        {epoch}")
    print(f"  Device:       {device}")
    print(f"  Tasks:        {tasks}")
    print(f"  Split:        {args.split}")
    print(f"  Eval subset:  {args.eval_subset}")

    # -- Load vocab & build dataset -------------------------------------------
    n_patients = data_params.get("n_patients", None)
    patients = load_sequences(n=n_patients)

    if vocab_path.exists():
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"  Vocab:        {vocab_path} ({len(vocab)} tokens)")
    else:
        vocab = build_vocab(patients, pad_idx=0, dir=run_dir)

    if is_supervised:
        dataset = SupervisedDataset(patients, vocab, data_params, pad_idx=0)
        loader = DataLoader(
            dataset, batch_size=data_params.get("batch_size", 64),
            shuffle=False, collate_fn=supervised_collate_fn, drop_last=False,
            num_workers=0)
    else:
        dataset = MimicDataset(patients, vocab, data_params, pad_idx=0)
        loader = DataLoader(
            dataset, batch_size=data_params.get("batch_size", 64),
            shuffle=False, collate_fn=collate_fn, drop_last=False,
            num_workers=0)

    print(f"  Samples:      {len(dataset)}")

    # -- Extract embeddings ---------------------------------------------------
    print("\n  Extracting embeddings...")
    if is_supervised:
        vecs = extract_supervised_embeddings(model, loader, device)
    else:
        vecs = extract_jepa_embeddings(model, loader, device)
        vecs = compute_derived_vectors(vecs)

    if is_supervised:
        print(f"  z_enc_pooled shape: {vecs['z_enc_pooled'].shape}")
    else:
        print(f"  z_pred shape:       {vecs['z_pred'].shape}")
        print(f"  z_enc_pooled shape: {vecs['z_enc_pooled'].shape}")

    # -- Load patient data for label lookups ----------------------------------
    patients_dict = load_patients_dict(sequences_path)

    # -- Filter by eval subset if requested -----------------------------------
    if args.eval_subset != "all":
        subset_mask = compute_subset_mask(
            sequences_path, vecs["subject_ids"], args.eval_subset)
        n_before = len(vecs["subject_ids"])
        vecs = filter_vecs(vecs, subset_mask)
        print(f"\n  Subset filter: {args.eval_subset} - "
              f"{int(subset_mask.sum())}/{n_before} samples kept")

    # -- Compute temporal split if requested ----------------------------------
    temporal_masks = None
    if args.split == "temporal":
        print("\n  Computing temporal split...")
        train_mask, test_mask, cutoff = compute_temporal_split(
            sequences_path, vecs["subject_ids"])
        temporal_masks = (train_mask, test_mask)
        print(f"  Cutoff date:  {cutoff}")
        print(f"  Train:        {int(train_mask.sum())} samples")
        print(f"  Test:         {int(test_mask.sum())} samples")

    # -- Run evaluation tasks -------------------------------------------------
    print("\n  Running evaluation tasks...")
    task_results = run_tasks(
        vecs, patients_dict, sequences_path, tasks, is_supervised, args.seed,
        split=args.split, temporal_masks=temporal_masks)

    # -- Build output ---------------------------------------------------------
    n_patients_unique = len(np.unique(vecs["subject_ids"]))
    output = {
        "model": architecture,
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "seed": args.seed,
        "split": args.split,
        "eval_subset": args.eval_subset,
        "n_samples": len(vecs["subject_ids"]),
        "n_patients": n_patients_unique,
        "modality": data_params.get("modality", "all"),
        "tasks": task_results,
    }

    # -- Print summary table --------------------------------------------------
    print_summary(architecture, task_results,
                  split=args.split, eval_subset=args.eval_subset)

    # -- Save JSON ------------------------------------------------------------
    output_path = Path(args.output) if args.output else run_dir / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"  Results saved -> {output_path}")


if __name__ == "__main__":
    main()
