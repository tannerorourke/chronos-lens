"""Stateless numpy operations used across multiple analysis modules."""

import numpy as np


def pool_to_patients(
    vecs: np.ndarray | dict[str, np.ndarray],
    subject_ids: np.ndarray,
) -> tuple:
    """Mean-pool sample-level arrays to patient-level. Accepts a single (N, D) 
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
            pooled[name] = (acc / counts[:, np.newaxis]).astype(emb.dtype)
        return pooled, unique_ids

    D = vecs.shape[1]
    acc = np.zeros((n_patients, D), dtype=np.float64)
    counts = np.zeros(n_patients, dtype=np.float64)
    for i in range(len(inverse)):
        acc[inverse[i]] += vecs[i]
        counts[inverse[i]] += 1
    return (acc / counts[:, np.newaxis]).astype(vecs.dtype), unique_ids


def broadcast_to_samples(patient_data, patient_ids, subject_ids) -> np.ndarray:
    """Expand patient-level (P, ..) to sample-level (N, ..) by subject_id lookup."""
    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}
    indices = np.array([pid_to_idx[str(sid)] for sid in subject_ids])
    return patient_data[indices]


def flatten_valid_encounters(z_encs, ctx_pad_masks, subject_ids) -> tuple:
    """Flatten (N, C, D) to (N_valid, D) using pad masks.

    Returns (z_enc_flat, enc_subject_ids, enc_positions).
    enc_positions[i] is the context position index for the i-th valid encounter.
    """
    valid_mask = ~ctx_pad_masks.astype(bool)
    z_enc_flat = z_encs[valid_mask]
    sample_idx, ctx_pos = np.where(valid_mask)
    enc_subject_ids = np.asarray(subject_ids, dtype=str)[sample_idx]
    return z_enc_flat, enc_subject_ids, ctx_pos


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """(N, N) pairwise cosine similarity"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xn = X / norms
    return Xn @ Xn.T


def cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    return 1.0 - cosine_sim_matrix(X)


def is_all_binary(col: np.ndarray) -> bool:
    """True if array contains only values in {0, 1}."""
    unique = np.unique(col[~np.isnan(col)])
    return len(unique) <= 2 and all(v in (0.0, 1.0) for v in unique)


def odds_ratio(freq_group: float, freq_pop: float) -> float:
    """Clamped odds ratio to avoid division by zero."""
    p_g = np.clip(freq_group, 1e-10, 1 - 1e-10)
    p_p = np.clip(freq_pop, 1e-10, 1 - 1e-10)
    return (p_g / (1 - p_g)) / (p_p / (1 - p_p))
