"""Vector manipulation for analysis (infra).

Pure-numpy transforms of already-extracted embedding vectors - no model forward,
no disk I/O. Derives analysis vectors (recency, pred-error), reshapes between
sample / encounter / patient levels, and selects patients of interest.
"""
import numpy as np


def compute_derived_vectors(raw_vecs: dict) -> dict:
    """Compute
        pred_error    = z_pred - z_target
        z_enc_recency = z_encs[k-1] - the most-recent context encounter per sample

    The per-sample analysis vector is the recency encounter, not a context mean:
    mean-pooling collapses a patient's trajectory into a running centroid, and the
    encounter encoder is bidirectional over the prefix (so an encounter's vector is
    prefix-length dependent) - the last context slot z_enc[k-1] is the one
    consistent "one point per encounter" and matches the model readout.
    """
    vecs = dict(raw_vecs)

    if "z_pred" in vecs and "z_target" in vecs:
        vecs["pred_error"] = vecs["z_pred"] - vecs["z_target"]

    if "z_encs" in vecs and "mask_pos" in vecs:
        z_encs = vecs["z_encs"]                          # (N, C, D)
        last_idx = (np.asarray(vecs["mask_pos"]) - 1).astype(int)   # (N,) last valid slot
        rows = np.arange(z_encs.shape[0])
        vecs["z_enc_recency"] = z_encs[rows, last_idx]   # (N, D)

    return vecs


def broadcast_to_samples(patient_data, patient_ids, subject_ids) -> np.ndarray:
    """Expand patient-level (P, ..) to sample-level (N, ..) by subject_id lookup."""
    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}
    indices = np.array([pid_to_idx[str(sid)] for sid in subject_ids])
    return patient_data[indices]


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


def select_terminal_by_patient(
    vecs: np.ndarray | dict[str, np.ndarray],
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
    key_suffix: str = ""
) -> tuple:
    """Reduce sample-level vector(s) to one row per patient by selecting each
    patient's terminal sample - the largest ``mask_pos`` (most-informed state) -
    rather than averaging. A mean across a patient's samples smooths over the
    trajectory; the terminal sample is the patient's latest encoded state.

    Accepts a single (N, D) array or a dict of {name: (N, D)} arrays. Returns
    (terminal, unique_subject_ids); for dict input ``terminal`` is a dict with
    the same (optionally suffixed) keys.
    """
    sids = np.asarray(subject_ids, dtype=str)
    mask_pos = np.asarray(mask_pos)
    unique_ids, inverse = np.unique(sids, return_inverse=True)

    # -- index of the largest-mask_pos sample for each patient
    terminal_idx = np.full(len(unique_ids), -1, dtype=int)
    best_mpos = np.full(len(unique_ids), -1)
    for i in range(len(inverse)):
        p = inverse[i]
        if mask_pos[i] > best_mpos[p]:
            best_mpos[p] = mask_pos[i]
            terminal_idx[p] = i

    if isinstance(vecs, dict):
        terminal = {
            f"{name}{key_suffix}": emb[terminal_idx]
            for name, emb in vecs.items()
            if emb is not None and emb.ndim == 2
        }
        return terminal, unique_ids

    return vecs[terminal_idx], unique_ids


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
