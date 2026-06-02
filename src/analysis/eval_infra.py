import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.jepa_ema import JEPA_EMA
from src.models.supervised_transformer import SupervisedTransformer

# ``load_scaffolding`` (model+loader reconstruction) now lives on the training
# side at ``src.training.utils.inference``; analysis depends on training infra,
# not the reverse. Re-export here for backwards-compatible imports.
from src.training.utils.inference import load_scaffolding  # noqa: F401


# =============================================================================
# Live Inference
# =============================================================================

def extract_jepa_embeddings(model, loader, device) -> dict:
    """Run JEPA model inference and collect embeddings.

    Returns dict with keys: z_encs, z_pred, z_target, ctx_pad_mask, subject_ids, mask_pos
    """
    all_z_encs: list[np.ndarray] = []
    all_z_pred: list[np.ndarray] = []
    all_z_target: list[np.ndarray] = []
    all_ctx_pad_mask: list[np.ndarray] = []
    all_mask_pos: list[np.ndarray] = []
    all_subject_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            z_enc, z_pred, z_target = model(batch_dev)
            # z_enc: (B, C, D), z_pred: (B, D), z_target: (B, D)

            all_z_encs.append(z_enc.cpu().numpy())
            all_z_pred.append(z_pred.cpu().numpy())
            all_z_target.append(z_target.cpu().numpy())
            all_ctx_pad_mask.append(batch_dev["ctx_pad_mask"].cpu().numpy())
            all_mask_pos.append(batch_dev["mask_pos"].cpu().numpy())
            all_subject_ids.extend(batch["subject_ids"])

    # Pad z_encs and ctx_pad_mask to uniform context length across batches
    max_C = max(arr.shape[1] for arr in all_z_encs)
    D = all_z_encs[0].shape[2]
    padded_z_encs: list[np.ndarray] = []
    padded_masks: list[np.ndarray] = []
    for z, m in zip(all_z_encs, all_ctx_pad_mask):
        B, C = z.shape[0], z.shape[1]
        if C < max_C:
            z = np.concatenate([z, np.zeros((B, max_C - C, D), dtype=z.dtype)], axis=1)
            m = np.concatenate([m, np.ones((B, max_C - C), dtype=m.dtype)], axis=1)
        padded_z_encs.append(z)
        padded_masks.append(m)

    return {
        "z_encs": np.concatenate(padded_z_encs),    # (N, C_max, D)
        "z_pred": np.concatenate(all_z_pred),       # (N, D)
        "z_target": np.concatenate(all_z_target),   # (N, D)
        "ctx_pad_mask": np.concatenate(padded_masks), # (N, C_max)
        "subject_ids": np.array(all_subject_ids),   # (N,)
        "mask_pos": np.concatenate(all_mask_pos),   # (N,)
    }


def extract_supervised_embeddings(model, loader, device) -> dict:
    """Run supervised model inference and collect per-encounter encoder output.

    Returns the same shape contract as :func:`extract_jepa_embeddings` (minus
    z_pred/z_target, which the supervised model lacks): per-encounter ``z_encs``
    + ``ctx_pad_mask``, so callers derive ``z_enc_pooled`` via
    :func:`compute_derived_vectors` exactly as for JEPA. Uses the encoder with
    ``pool=False`` - the same per-encounter representation - rather than the
    classifier-facing pooled vector.

    Returns dict with keys: z_encs, ctx_pad_mask, subject_ids, mask_pos.
    """
    all_z_encs: list[np.ndarray] = []
    all_ctx_pad_mask: list[np.ndarray] = []
    all_subject_ids: list[str] = []
    all_mask_pos: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            z_enc = model.encode(batch_dev, pool=False)   # (B, C, D)
            del batch_dev

            all_z_encs.append(z_enc.cpu().numpy())
            all_ctx_pad_mask.append(batch["ctx_pad_mask"].cpu().numpy())
            all_subject_ids.extend(batch["subject_ids"])
            all_mask_pos.append(batch["mask_pos"].cpu().numpy())

    # Pad z_encs and ctx_pad_mask to uniform context length across batches
    max_C = max(arr.shape[1] for arr in all_z_encs)
    D = all_z_encs[0].shape[2]
    padded_z_encs: list[np.ndarray] = []
    padded_masks: list[np.ndarray] = []
    for z, m in zip(all_z_encs, all_ctx_pad_mask):
        B, C = z.shape[0], z.shape[1]
        if C < max_C:
            z = np.concatenate([z, np.zeros((B, max_C - C, D), dtype=z.dtype)], axis=1)
            m = np.concatenate([m, np.ones((B, max_C - C), dtype=m.dtype)], axis=1)
        padded_z_encs.append(z)
        padded_masks.append(m)

    return {
        "z_encs": np.concatenate(padded_z_encs),        # (N, C_max, D)
        "ctx_pad_mask": np.concatenate(padded_masks),   # (N, C_max)
        "subject_ids": np.array(all_subject_ids),       # (N,)
        "mask_pos": np.concatenate(all_mask_pos),       # (N,)
    }


# =============================================================================

def compute_derived_vectors(raw_vecs: dict) -> dict:
    """Compute
        pred_error = z_pred - z_target
        z_enc_pooled = mean(z_encs) over valid (non-padded) positions
    """
    vecs = dict(raw_vecs)

    if "z_pred" in vecs and "z_target" in vecs:
        vecs["pred_error"] = vecs["z_pred"] - vecs["z_target"]

    if "z_encs" in vecs and "ctx_pad_mask" in vecs:
        z_encs = vecs["z_encs"]              # (N, C, D)
        pad_masks = vecs["ctx_pad_mask"]
        valid = (~pad_masks).astype(np.float32)[..., np.newaxis] # (N, C, 1)
        vecs["z_enc_pooled"] = (
            (z_encs * valid).sum(axis=1) /
            valid.sum(axis=1).clip(min=1.0)
        )  # (N, D)

    return vecs


def broadcast_to_samples(patient_data, patient_ids, subject_ids) -> np.ndarray:
    """Expand patient-level (P, ..) to sample-level (N, ..) by subject_id lookup."""
    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}
    indices = np.array([pid_to_idx[str(sid)] for sid in subject_ids])
    return patient_data[indices]

# =============================================================================
# Label loading
# =============================================================================

def load_label_30d_at_k(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """Load per-sample causal 30d readmission label at each sample's mask position.

    Reads patient["label_30d_per_enc"][k] for each (subject_id, mask_pos=k) pair.
    """
    labels = np.zeros(len(subject_ids), dtype=np.int64)
    for i, (sid, pos) in enumerate(zip(subject_ids, mask_pos)):
        patient = patients_dict[str(sid)]
        per_enc = patient.get("label_30d_per_enc", [])
        pos = int(pos)
        if pos < len(per_enc):
            labels[i] = per_enc[pos]
    return labels


def load_label(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    label_key: str,
    mask_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Load per-sample binary label from patients_dict.

    For per-encounter labels (label_30d, label_escalation), mask_pos is required
    to select the correct encounter position.
    """
    if label_key == "label_30d":
        if mask_pos is not None:
            return load_label_30d_at_k(patients_dict, subject_ids, mask_pos)
        # Fallback for patient-level callers (e.g. supervised): use last encounter
        return np.array([
            patients_dict[str(sid)].get("label_30d_per_enc", [0])[-1]
            for sid in subject_ids
        ], dtype=np.int64)
    elif label_key == "label_escalation":
        if mask_pos is not None:
            return load_escalation_labels(patients_dict, subject_ids, mask_pos)
        return np.array([
            patients_dict[str(sid)].get("label_escalation", 0)
            for sid in subject_ids
        ], dtype=np.int64)
    else:
        raise ValueError(f"[load_label] Unknown label key: {label_key}")
    
    
def load_escalation_labels(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> np.ndarray:
    """Load per-sample per-encounter escalation labels at each sample's mask position."""
    labels = np.zeros(len(subject_ids), dtype=np.int64)
    for i, (sid, pos) in enumerate(zip(subject_ids, mask_pos)):
        patient = patients_dict[str(sid)]
        per_enc = patient.get("label_escalation_per_enc", [])
        pos = int(pos)
        if pos < len(per_enc):
            labels[i] = per_enc[pos]
    return labels


def compute_escalation_criterions(
    patients_dict: dict[str, dict],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recompute per-encounter escalation criteria for each sample.

    Replays the escalation state machine from encounter 0 to mask_pos-1,
    then checks which criteria fire at mask_pos.

    Returns dict mapping criterion_name -> (N,) binary int64 array.
    """
    from src.utils.constants import ESCALATION_CRITERIA
    from src.mimic.labels import _check_enc_escalation, _update_state
    from src.mimic.helper import get_encounter_f_codes
    
    
    N = len(subject_ids)
    criteria_labels = {c: np.zeros(N, dtype=np.int64) for c in ESCALATION_CRITERIA}

    for i in range(N):
        sid = str(subject_ids[i])
        pos = int(mask_pos[i])
        encs = patients_dict[sid]["encounters"]

        if pos == 0:
            continue

        # Build prior state from encounters 0 .. pos-1
        # Causal assertion: only encounters [0:pos+1] are accessed
        assert pos < len(encs), (
            f"mask_pos {pos} >= n_encounters {len(encs)} for patient {sid}")

        prior_subcats: dict[str, int] = {}
        prior_f_codes: set[str] = set()
        prior_drug_classes: set[str] = set()
        has_prior_psych_meds = False

        for j in range(pos):
            f_codes = get_encounter_f_codes(encs[j], full=True)
            meds = [m.lower() for m in encs[j].get("meds", [])]
            had_meds = _update_state(
                f_codes, meds, prior_subcats, prior_f_codes, prior_drug_classes)
            has_prior_psych_meds = has_prior_psych_meds or had_meds

        # Check escalation at mask_pos (encounter at pos only - no future data)
        f_codes = get_encounter_f_codes(encs[pos], full=True)
        meds = [m.lower() for m in encs[pos].get("meds", [])]
        fired = _check_enc_escalation(
            f_codes, meds,
            prior_subcats, prior_f_codes, prior_drug_classes,
            has_prior_psych_meds,
        )

        for criterion in fired:
            if criterion in criteria_labels:
                criteria_labels[criterion][i] = 1

    return criteria_labels

# =============================================================================
# Subset filtering
# =============================================================================

def compute_subset_mask(patients: dict[str, dict], subject_ids: np.ndarray, subset):
    n_tot = len(subject_ids)
    if subset == "all":
        return np.ones(n_tot, dtype=bool)
    
    # f-code subset
    fcode_pids: set[str] = set()
    for pid, p in patients.items():
        for enc in p["encounters"]:
            if any(c.upper().startswith("F3") for c in enc.get("icd_codes", [])):
                fcode_pids.add(pid)
                break
    is_fcode = np.array([str(sid) in fcode_pids for sid in subject_ids])
    subset_mask = is_fcode if subset == "fcode" else ~is_fcode
    print(f"  Subset: {subset} -> {int(subset_mask.sum())}/{n_tot} samples")
    
    return subset_mask


def flatten_valid_encounters(z_encs, ctx_pad_mask, subject_ids) -> tuple:
    """Flatten z_enc from (N, C, D) to (N_valid, D) using pad masks. Usually called
       in order to pool encounters over patients. enc_positions[i] is the context 
       position index for the i-th valid encounter.

    Returns (z_enc_flat, enc_subject_ids, enc_positions).
    
    """
    valid_mask = ~ctx_pad_mask.astype(bool)
    z_enc_flat = z_encs[valid_mask]
    sample_idx, ctx_pos = np.where(valid_mask)
    enc_subject_ids = np.asarray(subject_ids, dtype=str)[sample_idx]
    return z_enc_flat, enc_subject_ids, ctx_pos


def pool_to_patients(
    vecs: np.ndarray | dict[str, np.ndarray],
    subject_ids: np.ndarray,
    key_suffix: str = ""
) -> tuple:
    """Mean-pool sample-level vector(s) to patient-level. Accepts a single (N, D) 
    array (e.g., cluster enrichment, probing) or a dict of {name: (N, D)} 
    arrays (e.g., pooling subject ids for evaluation). 
    
    Returns (pooled, unique_subject_ids).
    If dict input, pooled is a dict with same keys.
    """
    unique_ids, inverse = np.unique(
        np.asarray(subject_ids, dtype=str), return_inverse=True)
    n_patients = len(unique_ids)

    if isinstance(vecs, dict):
        pooled = {}
        for name, emb in vecs.items():
            if emb is None or emb.ndim != 2:
                continue
            D = emb.shape[1]
            acc = np.zeros((n_patients, D), dtype=np.float64)
            counts = np.zeros(n_patients, dtype=np.float64)
            for i in range(len(inverse)):
                acc[inverse[i]] += emb[i]
                counts[inverse[i]] += 1
            pooled[f"{name}{key_suffix}"] = (acc / counts[:, np.newaxis]).astype(emb.dtype)
        return pooled, unique_ids

    D = vecs.shape[1]
    acc = np.zeros((n_patients, D), dtype=np.float64)
    counts = np.zeros(n_patients, dtype=np.float64)
    for i in range(len(inverse)):
        acc[inverse[i]] += vecs[i]
        counts[inverse[i]] += 1
    return (acc / counts[:, np.newaxis]).astype(vecs.dtype), unique_ids

# =============================================================================
# Metrics
# =============================================================================

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

# =============================================================================
# Results
# =============================================================================

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
    
# =============================================================================
# Temporal Splitting
# =============================================================================

def compute_temporal_split(
    sequences_path: Path,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split samples into train/test by median latest admission date.

    All encounter windows from the same patient are assigned to the same
    split. Patients whose latest admission is strictly before the median
    cutoff go to train; the rest go to test.

    Returns
    -------
    train_mask  : (N,) bool array over subject_ids
    test_mask   : (N,) bool array over subject_ids
    cutoff_iso  : ISO-format string of the cutoff date
    """
    from datetime import datetime
    
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

# =========================================================================
# Patient selection utility
# =========================================================================

def select_interesting_patients(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    labels: np.ndarray,
    n: int = 5,
) -> list[int]:
    """Select patient indices worth visualising across different criteria.

    Criteria
    --------
    biggest_failures  : highest ||z_pred - z_target|| (worst predictions)
    best_predictions  : lowest prediction error norm
    most_dynamic      : z_target farthest from mean (most unusual encounters)
    escalation_cases  : patients with escalation label = 1
    random_sample     : random selection
    """
    from src.utils.seed import get_rng
    rng  = get_rng()

    N = z_pred.shape[0]

    pred_error_norms = np.linalg.norm(z_pred - z_target, axis=-1)
    target_dist = np.linalg.norm(z_target - z_target.mean(axis=0), axis=-1)

    biggest_failures = np.argsort(-pred_error_norms)[:n]
    best_predictions = np.argsort(pred_error_norms)[:n]
    most_dynamic     = np.argsort(-target_dist)[:n]
    escalation_cases = np.where(labels == 1)[0][:n]
    random_sample    = rng.choice(N, size=min(n, N), replace=False)

    # Deduplicate while preserving order
    seen: set[int] = set()
    result: list[int] = []
    for idx_arr in [biggest_failures, best_predictions, most_dynamic,
                    escalation_cases, random_sample]:
        for idx in idx_arr:
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                result.append(idx)

    return result
