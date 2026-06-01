"""
Patient trajectory extraction and geometric analysis through z_enc space.

Tracks how patient representations evolve across successive encounter
windows.  Provides geometric primitives (velocity, curvature, arc length),
concept drift measurement, prospective probing, and SAE-based feature
trajectory analysis.

Functions
---------
  extract_trajectories          : group samples into (P, T_max, D) trajectories
  trajectory_velocity           : finite-difference velocity vectors
  trajectory_curvature          : cosine angle between successive velocities
  trajectory_arc_length         : cumulative path length
  concept_centroid              : mean + covariance of positive-label samples
  drift_toward_concept          : velocity projection toward a concept cluster
  prospective_trajectory_probe  : causal probe test of trajectory hypothesis
  matched_trajectory_neighbors  : early-prefix nearest neighbors
  neighborhood_outcome_variance : label variance in neighbor sets
  sae_trajectory                : SAE activation traces along trajectories
  feature_flip_before_event     : on/off transitions before positive events
"""

from collections.abc import Callable

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.sae import SparseAutoencoder
from src.utils.seed import SEED


# =============================================================================
# Trajectory extraction
# =============================================================================

def extract_trajectories(
    z_enc_pooled: np.ndarray,
    pat_ids: np.ndarray,
    mask_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Group sample-level embeddings into per-patient temporal trajectories.

    Each sample (subject_id, mask_pos) represents the patient's pooled
    encoder state at time T in the encounter window.
    Grouping by patient and sorting by mask_pos ascending recovers the 
    trajectory through z_enc space.

    Parameters
    ----------
    z_enc_pooled : (N, D) pooled encoder output per sample
    pat_ids  : (N,) patient identifier per sample
    mask_pos     : (N,) encounter index per sample (int-castable)

    Returns
    -------
    trajectories : (P, T_max, D) padded trajectory tensor
    valid_mask   : (P, T_max) bool, True = valid step
    patient_ids  : (P,) unique patient identifiers (sorted)
    """
    pat_ids = np.asarray(pat_ids, dtype=str)
    mask_pos = np.asarray(mask_pos, dtype=int)
    D = z_enc_pooled.shape[1]

    unique_ids = np.unique(pat_ids)
    P = len(unique_ids)

    # Group sample indices by patient, storing (mask_pos, sample_index)
    patient_groups: dict[str, list[tuple[int, int]]] = {}
    for i in range(len(pat_ids)):
        sid = pat_ids[i]
        patient_groups.setdefault(sid, []).append((int(mask_pos[i]), i))

    # Sort within each patient by mask_pos; find T_max
    T_max = 0
    for sid in unique_ids:
        patient_groups[sid].sort(key=lambda x: x[0])
        T_max = max(T_max, len(patient_groups[sid]))

    trajectories = np.zeros((P, T_max, D), dtype=z_enc_pooled.dtype)
    valid_mask = np.zeros((P, T_max), dtype=bool)

    for p_idx, sid in enumerate(unique_ids):
        for t, (_, sample_idx) in enumerate(patient_groups[sid]):
            trajectories[p_idx, t] = z_enc_pooled[sample_idx]
            valid_mask[p_idx, t] = True

    return {
        "trajectories": trajectories,
        "validity_mask": valid_mask,
        "patient_ids": unique_ids,
        "times": mask_pos
    }

    return trajectories, valid_mask, unique_ids


# =============================================================================
# Geometric primitives
# =============================================================================

def trajectory_velocity(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference velocity vectors between successive trajectory steps.

    Parameters
    ----------
    trajectories : (P, T_max, D)
    valid_mask   : (P, T_max) bool

    Returns
    -------
    velocity : (P, T_max - 1, D) difference vectors
    vel_mask : (P, T_max - 1) bool, True where both endpoints are valid
    """
    velocity = np.diff(trajectories, axis=1)               # (P, T_max-1, D)
    vel_mask = valid_mask[:, :-1] & valid_mask[:, 1:]      # (P, T_max-1)
    velocity[~vel_mask] = 0.0
    return velocity, vel_mask


def trajectory_curvature(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of the angle between successive velocity vectors.

    Values near 1.0 = straight-line trajectory, near -1.0 = reversal,
    near 0.0 = right-angle turn.

    Parameters
    ----------
    trajectories : (P, T_max, D)
    valid_mask   : (P, T_max) bool

    Returns
    -------
    curvature : (P, T_max - 2) cosine between consecutive velocity pairs
    curv_mask : (P, T_max - 2) bool, True where three consecutive steps valid
    """
    vel, vel_mask = trajectory_velocity(trajectories, valid_mask)

    v1 = vel[:, :-1]    # (P, T_max-2, D)
    v2 = vel[:, 1:]     # (P, T_max-2, D)
    curv_mask = vel_mask[:, :-1] & vel_mask[:, 1:]  # (P, T_max-2)

    n1 = np.linalg.norm(v1, axis=-1).clip(min=1e-10)  # (P, T_max-2)
    n2 = np.linalg.norm(v2, axis=-1).clip(min=1e-10)

    cos_angle = (v1 * v2).sum(axis=-1) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    cos_angle[~curv_mask] = 0.0

    return cos_angle, curv_mask


def trajectory_arc_length(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Total Euclidean arc length of each patient trajectory.

    Parameters
    ----------
    trajectories : (P, T_max, D)
    valid_mask   : (P, T_max) bool

    Returns
    -------
    arc_lengths : (P,) total path length per patient
    """
    vel, vel_mask = trajectory_velocity(trajectories, valid_mask)
    step_lengths = np.linalg.norm(vel, axis=-1)  # (P, T_max-1)
    step_lengths[~vel_mask] = 0.0
    return step_lengths.sum(axis=1)


# =============================================================================
# Concept geometry
# =============================================================================

def concept_centroid(
    z_enc: np.ndarray,
    label_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of z_enc over positive-label samples.

    Parameters
    ----------
    z_enc        : (N, D) embedding matrix
    label_vector : (N,) binary labels (1 = positive)

    Returns
    -------
    centroid   : (D,) mean of positive samples
    covariance : (D, D) covariance matrix of positive samples
    """
    pos_mask = np.asarray(label_vector, dtype=bool)
    z_pos = z_enc[pos_mask]
    centroid = z_pos.mean(axis=0)
    covariance = np.cov(z_pos, rowvar=False)
    return centroid, covariance


# =============================================================================
# Concept drift
# =============================================================================

def drift_toward_concept(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step projection of velocity onto the direction toward a concept centroid.

    Positive values mean the patient is moving toward the concept cluster;
    negative values mean moving away.  This is the primary "trending toward
    escalation cluster" metric.

    Parameters
    ----------
    trajectories : (P, T_max, D)
    valid_mask   : (P, T_max) bool
    centroid     : (D,) concept centroid

    Returns
    -------
    drift      : (P, T_max - 1) signed projection magnitude
    drift_mask : (P, T_max - 1) bool
    """
    vel, vel_mask = trajectory_velocity(trajectories, valid_mask)

    # Direction from current position to centroid
    pos = trajectories[:, :-1]                              # (P, T_max-1, D)
    to_centroid = centroid[np.newaxis, np.newaxis, :] - pos  # (P, T_max-1, D)
    tc_norm = np.linalg.norm(to_centroid, axis=-1, keepdims=True).clip(min=1e-10)
    tc_unit = to_centroid / tc_norm

    drift = (vel * tc_unit).sum(axis=-1)  # (P, T_max-1)
    drift[~vel_mask] = 0.0

    return drift, vel_mask


# =============================================================================
# Prospective trajectory probe
# =============================================================================

def prospective_trajectory_probe(
    trajectories: np.ndarray,
    labels_at_k: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    baseline_vec: np.ndarray,
    n_splits: int = 5,
    min_samples_per_class: int = 5,
) -> dict:
    """Causal probe: does the trajectory prefix predict the next-step label?

    At each step k, compute trajectory-derived features from steps [0, k-1]
    and fit a logistic regression probe predicting label_at_k.  Report AUROC
    against a baseline probe on z_enc[k-1] alone.

    This is the quantitative test of the trajectory hypothesis: if trajectory
    features (velocity, curvature, drift) predict outcomes better than the
    static embedding at the previous step, the temporal structure carries
    signal beyond what a single snapshot provides.

    Parameters
    ----------
    trajectories : (P, T_max, D) padded trajectory tensor
    labels_at_k  : (P, T_max) int labels at each step (0/1; -1 = ignore)
    feature_fn   : callable(prefix: (T, D)) -> (F,) feature vector computed
                   from trajectory prefix of length T (steps 0..T-1)
    baseline_vec : (P, T_max, D) representation for baseline at each step
                   (typically the trajectory itself; probe uses step k-1)
    n_splits     : number of stratified CV folds
    min_samples_per_class : minimum positive/negative samples to run probe

    Returns
    -------
    dict with:
        per_step : list of dicts per valid step k >= 1, each with
            k, n_samples, n_positive, traj_auroc, baseline_auroc, delta_auroc
        summary  : {mean_delta_auroc, n_valid_steps}
    """
    P, T_max, D = trajectories.shape
    per_step: list[dict] = []

    for k in range(1, T_max):
        valid = labels_at_k[:, k] >= 0
        y = labels_at_k[valid, k].astype(int)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        if n_pos < min_samples_per_class or n_neg < min_samples_per_class:
            continue

        valid_idx = np.where(valid)[0]

        # Trajectory features from prefix [0, k-1]
        X_traj = np.array([feature_fn(trajectories[p, :k]) for p in valid_idx])

        # Baseline: static embedding at step k-1
        X_base = baseline_vec[valid, k - 1]

        traj_auroc = _cv_auroc(X_traj, y, n_splits)
        base_auroc = _cv_auroc(X_base, y, n_splits)

        per_step.append({
            "k": k,
            "n_samples": len(y),
            "n_positive": n_pos,
            "traj_auroc": traj_auroc,
            "baseline_auroc": base_auroc,
            "delta_auroc": traj_auroc - base_auroc,
        })

    deltas = [s["delta_auroc"] for s in per_step]
    return {
        "per_step": per_step,
        "summary": {
            "mean_delta_auroc": float(np.mean(deltas)) if deltas else float("nan"),
            "n_valid_steps": len(per_step),
        },
    }


def _cv_auroc(X: np.ndarray, y: np.ndarray, n_splits: int) -> float:
    """Stratified CV AUROC for a logistic regression probe."""
    n_splits = min(n_splits, int(y.sum()), int((~y.astype(bool)).sum()))
    if n_splits < 2:
        return float("nan")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_aurocs: list[float] = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        if y_te.sum() < 1 or (len(y_te) - y_te.sum()) < 1:
            continue

        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        fold_aurocs.append(float(roc_auc_score(y_te, y_prob)))

    return float(np.mean(fold_aurocs)) if fold_aurocs else float("nan")


# =============================================================================
# Trajectory neighbors
# =============================================================================

def matched_trajectory_neighbors(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest neighbors using fixed-length early trajectory prefixes.

    Computes Euclidean distance on the flattened first k steps of each
    trajectory.  Only patients with at least k valid steps are included.

    Parameters
    ----------
    trajectories : (P, T_max, D)
    valid_mask   : (P, T_max) bool
    k            : prefix length (number of early steps to compare)
    n_neighbors  : number of nearest neighbors per patient

    Returns
    -------
    neighbors : (P_valid, n_neighbors) int indices into the P_valid subset
    eligible  : (P_valid,) original patient indices with >= k valid steps
    """
    valid_counts = valid_mask.sum(axis=1)
    eligible = np.where(valid_counts >= k)[0]
    P_valid = len(eligible)

    if P_valid == 0:
        return np.empty((0, n_neighbors), dtype=int), eligible

    D = trajectories.shape[2]
    prefixes = trajectories[eligible, :k].reshape(P_valid, k * D)

    dists = cdist(prefixes, prefixes, metric="euclidean")  # (P_valid, P_valid)
    np.fill_diagonal(dists, np.inf)

    n_nbrs = min(n_neighbors, P_valid - 1)
    neighbors = np.argsort(dists, axis=1)[:, :n_nbrs]

    return neighbors, eligible


def neighborhood_outcome_variance(
    neighbors: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Outcome variance within each patient's neighborhood.

    High variance identifies decision-boundary patients: similar early
    trajectories leading to divergent outcomes.

    Parameters
    ----------
    neighbors : (P_valid, n_neighbors) int indices into labels
    labels    : (P_valid,) binary outcome labels

    Returns
    -------
    variance : (P_valid,) label variance per neighborhood
    """
    neighbor_labels = labels[neighbors]  # (P_valid, n_neighbors)
    return neighbor_labels.var(axis=1)


# =============================================================================
# SAE trajectory analysis
# =============================================================================

def sae_trajectory(
    trajectories: np.ndarray,
    valid_mask: np.ndarray,
    sae_model: SparseAutoencoder,
) -> np.ndarray:
    """Run a trained SAE on each trajectory step to produce activation traces.

    Parameters
    ----------
    trajectories : (P, T_max, D)
    valid_mask   : (P, T_max) bool
    sae_model    : trained SparseAutoencoder in eval mode

    Returns
    -------
    sae_traj : (P, T_max, n_features) sparse activations at each step
    """
    P, T_max, D = trajectories.shape
    n_features = sae_model.n_features
    sae_traj = np.zeros((P, T_max, n_features), dtype=np.float32)

    flat_idx = np.where(valid_mask)
    if len(flat_idx[0]) == 0:
        return sae_traj

    flat_vecs = trajectories[flat_idx]  # (N_valid, D)

    device = next(sae_model.parameters()).device
    sae_model.eval()
    with torch.no_grad():
        x = torch.tensor(flat_vecs, dtype=torch.float32, device=device)
        _, activations = sae_model(x)
        sae_traj[flat_idx] = activations.cpu().numpy()

    return sae_traj


def feature_flip_before_event(
    sae_traj: np.ndarray,
    event_labels: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
) -> dict:
    """Per-feature on/off transitions in the W steps before positive events.

    For each positive event at step t, examines the SAE activation pattern
    in [t - window, t - 1] and counts how often each feature transitions
    from inactive to active (onset) or active to inactive (offset).

    Parameters
    ----------
    sae_traj     : (P, T_max, n_features) activation traces
    event_labels : (P, T_max) binary event indicators (1 = event at step t)
    valid_mask   : (P, T_max) bool
    window       : number of steps before event to examine

    Returns
    -------
    dict with:
        onset_counts  : (n_features,) 0->active transitions before events
        offset_counts : (n_features,) active->0 transitions before events
        n_events      : total positive events examined
        onset_rate    : (n_features,) onset_counts / n_events
        offset_rate   : (n_features,) offset_counts / n_events
    """
    P, T_max, n_features = sae_traj.shape
    onset_counts = np.zeros(n_features, dtype=int)
    offset_counts = np.zeros(n_features, dtype=int)
    n_events = 0

    active = sae_traj != 0  # (P, T_max, n_features)

    for p in range(P):
        for t in range(window, T_max):
            if not valid_mask[p, t] or event_labels[p, t] != 1:
                continue
            w_start = t - window
            if not valid_mask[p, w_start:t].all():
                continue

            n_events += 1
            win = active[p, w_start:t]  # (window, n_features)

            for s in range(1, window):
                onset_counts  += (~win[s - 1] & win[s]).astype(int)
                offset_counts += (win[s - 1] & ~win[s]).astype(int)

    n_safe = max(n_events, 1)
    return {
        "onset_counts": onset_counts,
        "offset_counts": offset_counts,
        "n_events": n_events,
        "onset_rate": onset_counts / n_safe,
        "offset_rate": offset_counts / n_safe,
    }
