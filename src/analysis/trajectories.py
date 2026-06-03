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
  trajectory_temporal_velocity  : velocity in per-day units (Δz / Δt)
  trajectory_curvature          : cosine angle between successive velocities
  trajectory_arc_length         : cumulative path length
  concept_centroid              : mean + covariance of positive-label samples
  drift_toward_concept          : velocity projection toward a concept cluster
  temporal_drift_rate           : per-day drift rate toward concept
  prospective_trajectory_probe  : causal probe test of trajectory hypothesis
  matched_trajectory_neighbors  : early-prefix nearest neighbors
  neighborhood_outcome_variance : label variance in neighbor sets
  sae_trajectory                : SAE activation traces along trajectories
  feature_flip_before_event     : on/off transitions before positive events
"""

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
    z_sample: np.ndarray,
    subject_ids: np.ndarray,
    mask_pos: np.ndarray,
    times: np.ndarray | None = None,
) -> dict:
    """Group sample-level embeddings into per-patient temporal trajectories.

    Each sample (subject_id, mask_pos) contributes one encounter point (the
    recency vector z_enc[k-1]); grouping by patient and sorting by mask_pos
    ascending recovers the patient's trajectory through z_enc space.

    Parameters
    ----------
    z_sample : (N, D) per-sample encounter vector (recency z_enc[k-1])
    subject_ids  : (N,) patient identifier per sample
    mask_pos     : (N,) encounter index per sample (int-castable)
    times        : (N,) optional float, days-since-first-admit per sample.
                   If provided, aligned into (P, T_max) in the output dict.

    Returns
    -------
    dict with:
        trajectories  : (P, T_max, D) padded trajectory tensor (zeros where invalid)
        validity_mask  : (P, T_max) bool, True = real step
        patient_ids    : (P,) str, unique patient IDs in sorted order
        times          : (P, T_max) float, days-since-first-admit per step
                         (zeros where invalid). None if times not provided.
    """
    subject_ids = np.asarray(subject_ids, dtype=str)
    mask_pos = np.asarray(mask_pos, dtype=int)
    D = z_sample.shape[1]
    has_times = times is not None
    if has_times:
        times = np.asarray(times, dtype=float)

    unique_ids = np.unique(subject_ids)
    P = len(unique_ids)

    # Group sample indices by patient: {sid: [(mask_pos, sample_idx), ...]}
    patient_groups: dict[str, list[tuple[int, int]]] = {}
    for i in range(len(subject_ids)):
        sid = subject_ids[i]
        patient_groups.setdefault(sid, []).append((int(mask_pos[i]), i))

    # Sort within each patient by mask_pos; find T_max
    T_max = 0
    for sid in unique_ids:
        patient_groups[sid].sort(key=lambda x: x[0])
        T_max = max(T_max, len(patient_groups[sid]))

    trajectories = np.zeros((P, T_max, D), dtype=z_sample.dtype)
    validity_mask = np.zeros((P, T_max), dtype=bool)
    time_mat = np.zeros((P, T_max), dtype=np.float64) if has_times else None

    for p_idx, sid in enumerate(unique_ids):
        for t, (_, sample_idx) in enumerate(patient_groups[sid]):
            trajectories[p_idx, t] = z_sample[sample_idx]
            validity_mask[p_idx, t] = True
            if has_times:
                time_mat[p_idx, t] = times[sample_idx]

    return {
        "trajectories": trajectories,
        "validity_mask": validity_mask,
        "patient_ids": unique_ids,
        "times": time_mat,
    }


# =============================================================================
# Geometric primitives
# =============================================================================

def trajectory_velocity(traj_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference velocity vectors between successive trajectory steps.

    Parameters
    ----------
    traj_dict : output of extract_trajectories

    Returns
    -------
    velocity : (P, T_max-1, D) difference vectors (masked to 0 where invalid)
    vel_mask : (P, T_max-1) bool, True where both endpoints are valid
    """
    trajectories = traj_dict["trajectories"]  # (P, T_max, D)
    valid_mask = traj_dict["validity_mask"]    # (P, T_max)

    # First difference along time axis
    velocity = np.diff(trajectories, axis=1)               # (P, T_max-1, D)
    vel_mask = valid_mask[:, :-1] & valid_mask[:, 1:]      # (P, T_max-1)
    velocity[~vel_mask] = 0.0
    return velocity, vel_mask


def trajectory_temporal_velocity(traj_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Velocity in per-day units: Δz / Δt.

    Requires ``times`` in traj_dict. Δt is clipped to a minimum of 1 day
    to avoid inf from same-day encounters.

    Parameters
    ----------
    traj_dict : output of extract_trajectories (must contain non-None times)

    Returns
    -------
    temporal_vel : (P, T_max-1, D) per-day velocity vectors
    vel_mask     : (P, T_max-1) bool
    """
    assert traj_dict["times"] is not None, "times required for temporal velocity"

    velocity, vel_mask = trajectory_velocity(traj_dict)  # (P, T_max-1, D), (P, T_max-1)
    times = traj_dict["times"]  # (P, T_max)

    # Δt between consecutive steps, clipped to min 1 day for inf-safety
    dt = np.diff(times, axis=1)         # (P, T_max-1)
    dt = np.clip(dt, a_min=1.0, a_max=None)

    # Broadcast (P, T_max-1, 1) for element-wise division
    temporal_vel = velocity / dt[:, :, np.newaxis]
    temporal_vel[~vel_mask] = 0.0
    return temporal_vel, vel_mask


def trajectory_curvature(traj_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Cosine of the angle between successive velocity vectors.

    Values near 1.0 = straight-line trajectory, near 0.0 = right-angle turn,
    near -1.0 = reversal.  NaN where insufficient valid steps.

    Parameters
    ----------
    traj_dict : output of extract_trajectories

    Returns
    -------
    curvature : (P, T_max-2) cosine between consecutive velocity pairs.
                NaN where fewer than 3 consecutive valid steps.
    curv_mask : (P, T_max-2) bool, True where three consecutive steps valid
    """
    vel, vel_mask = trajectory_velocity(traj_dict)

    v1 = vel[:, :-1]    # (P, T_max-2, D)
    v2 = vel[:, 1:]     # (P, T_max-2, D)
    curv_mask = vel_mask[:, :-1] & vel_mask[:, 1:]  # (P, T_max-2)

    n1 = np.linalg.norm(v1, axis=-1).clip(min=1e-10)  # (P, T_max-2)
    n2 = np.linalg.norm(v2, axis=-1).clip(min=1e-10)

    cos_angle = (v1 * v2).sum(axis=-1) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # NaN where insufficient valid steps
    cos_angle[~curv_mask] = np.nan

    return cos_angle, curv_mask


def trajectory_arc_length(traj_dict: dict) -> np.ndarray:
    """Total Euclidean arc length of each patient trajectory.

    Cumulative sum of ‖v_t‖ over valid steps.

    Parameters
    ----------
    traj_dict : output of extract_trajectories

    Returns
    -------
    arc_lengths : (P,) total path length per patient
    """
    vel, vel_mask = trajectory_velocity(traj_dict)
    step_lengths = np.linalg.norm(vel, axis=-1)  # (P, T_max-1)
    step_lengths[~vel_mask] = 0.0
    return step_lengths.sum(axis=1)


# =============================================================================
# Concept geometry
# =============================================================================

def concept_centroid(
    z_enc: np.ndarray,
    label_vector: np.ndarray,
) -> dict:
    """Mean and covariance of z_enc over positive-label samples.

    Parameters
    ----------
    z_enc        : (N, D) embedding matrix
    label_vector : (N,) binary labels (1 = positive)

    Returns
    -------
    dict with:
        mean       : (D,) mean of positive samples
        cov        : (D, D) covariance matrix of positive samples
        n_positive : int, number of positive samples
    """
    pos_mask = np.asarray(label_vector, dtype=bool)
    z_pos = z_enc[pos_mask]
    return {
        "mean": z_pos.mean(axis=0),
        "cov": np.cov(z_pos, rowvar=False),
        "n_positive": int(pos_mask.sum()),
    }


# =============================================================================
# Concept drift
# =============================================================================

def drift_toward_concept(
    traj_dict: dict,
    centroid_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step projection of velocity onto the direction toward a concept centroid.

    Positive values mean the patient is moving toward the concept cluster;
    negative values mean moving away.  Primary "trending toward escalation
    cluster" metric.

    Parameters
    ----------
    traj_dict     : output of extract_trajectories
    centroid_mean : (D,) concept centroid vector

    Returns
    -------
    drift      : (P, T_max-1) signed projection magnitude
    drift_mask : (P, T_max-1) bool
    """
    trajectories = traj_dict["trajectories"]  # (P, T_max, D)
    vel, vel_mask = trajectory_velocity(traj_dict)

    # Direction from current position to centroid
    pos = trajectories[:, :-1]                              # (P, T_max-1, D)
    to_centroid = centroid_mean[np.newaxis, np.newaxis, :] - pos  # (P, T_max-1, D)
    tc_norm = np.linalg.norm(to_centroid, axis=-1, keepdims=True).clip(min=1e-10)
    tc_unit = to_centroid / tc_norm

    # Scalar projection of velocity onto unit direction to centroid
    drift = (vel * tc_unit).sum(axis=-1)  # (P, T_max-1)
    drift[~vel_mask] = 0.0

    return drift, vel_mask


def temporal_drift_rate(
    traj_dict: dict,
    centroid_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-day drift rate toward a concept centroid.

    Same as drift_toward_concept divided by Δt. Requires ``times`` in
    traj_dict. Δt clipped to min 1 day for inf-safety.

    Parameters
    ----------
    traj_dict     : output of extract_trajectories (must contain non-None times)
    centroid_mean : (D,) concept centroid vector

    Returns
    -------
    drift_rate : (P, T_max-1) per-day signed drift rate
    drift_mask : (P, T_max-1) bool
    """
    assert traj_dict["times"] is not None, "times required for temporal drift rate"

    drift, drift_mask = drift_toward_concept(traj_dict, centroid_mean)
    times = traj_dict["times"]  # (P, T_max)

    # Δt between consecutive steps, clipped to min 1 day
    dt = np.diff(times, axis=1)         # (P, T_max-1)
    dt = np.clip(dt, a_min=1.0, a_max=None)

    drift_rate = drift / dt
    drift_rate[~drift_mask] = 0.0

    return drift_rate, drift_mask


# =============================================================================
# Prospective trajectory probe
# =============================================================================

def prospective_trajectory_probe(
    traj_dict: dict,
    labels_per_step: np.ndarray,
    baseline_z_enc: np.ndarray,
    centroid_mean: np.ndarray | None = None,
    n_splits: int = 5,
    min_samples_per_class: int = 5,
) -> dict:
    """Causal probe: does the trajectory prefix predict the next-step label?

    At each valid step k, builds a feature vector from steps [0, k-1]:
      - last velocity magnitude
      - last curvature (NaN-safe)
      - last drift toward concept (if centroid_mean provided)
      - temporal drift rate (if times available and centroid_mean provided)
      - Δt to target step (if times available)
      - arc length so far
    Fits a logistic regression probe and compares against a baseline probe
    on z_enc[k-1] alone.

    This is the quantitative test of the trajectory hypothesis: if trajectory
    features predict outcomes better than the static embedding at the previous
    step, the temporal structure carries signal beyond what a single snapshot
    provides.

    Parameters
    ----------
    traj_dict       : output of extract_trajectories
    labels_per_step : (P, T_max) int labels at each step (0/1; -1 = ignore)
    baseline_z_enc  : (P, T_max, D) representation for baseline at each step
                      (typically the trajectories themselves; probe uses step k-1)
    centroid_mean   : (D,) concept centroid for drift features. If None, drift
                      features are omitted.
    n_splits        : number of stratified CV folds
    min_samples_per_class : minimum positive/negative samples to run probe

    Returns
    -------
    dict with:
        traj_auroc     : float, AUROC using trajectory features (pooled across steps)
        baseline_auroc : float, AUROC using z_enc[k-1] alone (pooled across steps)
        delta_auroc    : float, traj - baseline
        n_samples      : int, total samples across all valid steps
        feature_names  : list[str], names of trajectory features used
    """
    trajectories = traj_dict["trajectories"]    # (P, T_max, D)
    validity_mask = traj_dict["validity_mask"]   # (P, T_max)
    times = traj_dict["times"]                   # (P, T_max) or None
    P, T_max, D = trajectories.shape

    # Precompute geometric primitives
    vel, vel_mask = trajectory_velocity(traj_dict)           # (P, T_max-1, D), (P, T_max-1)
    vel_mag = np.linalg.norm(vel, axis=-1)                   # (P, T_max-1)
    curv, curv_mask = trajectory_curvature(traj_dict)        # (P, T_max-2), (P, T_max-2)

    has_centroid = centroid_mean is not None
    has_times = times is not None
    if has_centroid:
        drift, drift_mask = drift_toward_concept(traj_dict, centroid_mean)  # (P, T_max-1)
    if has_centroid and has_times:
        tdrift, _ = temporal_drift_rate(traj_dict, centroid_mean)           # (P, T_max-1)

    # Build feature name list
    feature_names = ["vel_mag", "curvature", "arc_length"]
    if has_centroid:
        feature_names.append("drift_toward_concept")
    if has_centroid and has_times:
        feature_names.append("temporal_drift_rate")
    if has_times:
        feature_names.append("dt_to_target")

    # Collect samples across all valid steps k >= 1
    X_traj_all: list[np.ndarray] = []
    X_base_all: list[np.ndarray] = []
    y_all: list[int] = []

    for k in range(1, T_max):
        for p in range(P):
            # Need valid step at k, valid step at k-1, and a non-ignore label
            if not validity_mask[p, k] or not validity_mask[p, k - 1]:
                continue
            if labels_per_step[p, k] < 0:
                continue

            y_all.append(int(labels_per_step[p, k]))

            # -- Trajectory features from prefix [0, k-1]
            feats = []

            # Last velocity magnitude (step k-1)
            feats.append(vel_mag[p, k - 1] if vel_mask[p, k - 1] else 0.0)

            # Last curvature (step k-2, needs k >= 2)
            if k >= 2 and curv_mask[p, k - 2]:
                feats.append(curv[p, k - 2])
            else:
                feats.append(0.0)

            # Arc length so far: sum of step lengths [0, k-1)
            arc = vel_mag[p, :k]
            arc_valid = vel_mask[p, :k]
            feats.append(float(arc[arc_valid].sum()) if arc_valid.any() else 0.0)

            # Drift toward concept at step k-1
            if has_centroid:
                feats.append(drift[p, k - 1] if drift_mask[p, k - 1] else 0.0)

            # Temporal drift rate at step k-1
            if has_centroid and has_times:
                feats.append(tdrift[p, k - 1] if drift_mask[p, k - 1] else 0.0)

            # Δt from step k-1 to step k
            if has_times:
                dt = max(times[p, k] - times[p, k - 1], 0.0)
                feats.append(dt)

            X_traj_all.append(np.array(feats, dtype=np.float64))
            X_base_all.append(baseline_z_enc[p, k - 1])

    if len(y_all) == 0:
        return {
            "traj_auroc": float("nan"),
            "baseline_auroc": float("nan"),
            "delta_auroc": float("nan"),
            "n_samples": 0,
            "feature_names": feature_names,
        }

    X_traj = np.stack(X_traj_all)   # (N_valid, F)
    X_base = np.stack(X_base_all)   # (N_valid, D)
    y = np.array(y_all, dtype=int)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < min_samples_per_class or n_neg < min_samples_per_class:
        return {
            "traj_auroc": float("nan"),
            "baseline_auroc": float("nan"),
            "delta_auroc": float("nan"),
            "n_samples": len(y),
            "feature_names": feature_names,
        }

    traj_auroc = _cv_auroc(X_traj, y, n_splits)
    base_auroc = _cv_auroc(X_base, y, n_splits)

    return {
        "traj_auroc": traj_auroc,
        "baseline_auroc": base_auroc,
        "delta_auroc": traj_auroc - base_auroc,
        "n_samples": len(y),
        "feature_names": feature_names,
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
    traj_dict: dict,
    prefix_len: int,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest neighbors using fixed-length early trajectory prefixes.

    Computes Euclidean distance on the flattened first ``prefix_len`` steps
    of each trajectory.  Only patients with at least ``prefix_len`` valid
    steps are included.

    Parameters
    ----------
    traj_dict   : output of extract_trajectories
    prefix_len  : number of early steps to compare
    n_neighbors : number of nearest neighbors per patient

    Returns
    -------
    neighbors : (P_valid, n_neighbors) int indices into the P_valid subset
    eligible  : (P_valid,) original patient indices with >= prefix_len valid steps
    """
    trajectories = traj_dict["trajectories"]  # (P, T_max, D)
    valid_mask = traj_dict["validity_mask"]    # (P, T_max)

    valid_counts = valid_mask.sum(axis=1)
    eligible = np.where(valid_counts >= prefix_len)[0]
    P_valid = len(eligible)

    if P_valid == 0:
        return np.empty((0, n_neighbors), dtype=int), eligible

    D = trajectories.shape[2]
    prefixes = trajectories[eligible, :prefix_len].reshape(P_valid, prefix_len * D)

    dists = cdist(prefixes, prefixes, metric="euclidean")  # (P_valid, P_valid)
    np.fill_diagonal(dists, np.inf)

    n_nbrs = min(n_neighbors, P_valid - 1)
    neighbors = np.argsort(dists, axis=1)[:, :n_nbrs]

    return neighbors, eligible


def neighborhood_outcome_variance(
    neighbor_indices: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Outcome variance within each patient's neighborhood.

    High variance identifies decision-boundary patients: similar early
    trajectories leading to divergent outcomes.

    Parameters
    ----------
    neighbor_indices : (P_valid, n_neighbors) int indices into labels
    labels           : (P_valid,) binary outcome labels

    Returns
    -------
    variance : (P_valid,) label variance per neighborhood
    """
    neighbor_labels = labels[neighbor_indices]  # (P_valid, n_neighbors)
    return neighbor_labels.var(axis=1)


# =============================================================================
# SAE trajectory analysis
# =============================================================================

def sae_trajectory(
    traj_dict: dict,
    sae_model: SparseAutoencoder,
) -> np.ndarray:
    """Run a trained SAE on each trajectory step to produce activation traces.

    Parameters
    ----------
    traj_dict : output of extract_trajectories
    sae_model : trained SparseAutoencoder in eval mode

    Returns
    -------
    sae_traj : (P, T_max, n_features) sparse activations at each step
    """
    trajectories = traj_dict["trajectories"]  # (P, T_max, D)
    valid_mask = traj_dict["validity_mask"]    # (P, T_max)
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
    event_mask: np.ndarray,
    window: int = 3,
) -> dict[int, dict]:
    """Per-feature on/off transitions in the W steps before positive events.

    For each positive event at step t, examines the SAE activation pattern
    in [t - window, t - 1] and counts how often each feature transitions
    from inactive to active (onset) or active to inactive (offset).

    Parameters
    ----------
    sae_traj   : (P, T_max, n_features) activation traces
    event_mask : (P, T_max) binary event indicators (1 = event at step t)
    window     : number of steps before event to examine

    Returns
    -------
    dict keyed by feature index, each value:
        flip_on_rate  : fraction of events where feature turned on in window
        flip_off_rate : fraction of events where feature turned off in window
        n_events      : total positive events examined for this feature
    """
    P, T_max, n_features = sae_traj.shape
    active = sae_traj != 0  # (P, T_max, n_features)

    # Count per-feature onset and offset events
    onset_counts = np.zeros(n_features, dtype=int)
    offset_counts = np.zeros(n_features, dtype=int)
    n_events = 0

    for p in range(P):
        for t in range(window, T_max):
            if event_mask[p, t] != 1:
                continue

            n_events += 1
            win = active[p, t - window:t]  # (window, n_features)

            # Check for any on/off flip within the window
            for s in range(1, window):
                onset_counts  += (~win[s - 1] & win[s]).astype(int)
                offset_counts += (win[s - 1] & ~win[s]).astype(int)

    # Build per-feature result dict
    n_safe = max(n_events, 1)
    result: dict[int, dict] = {}
    for f in range(n_features):
        if onset_counts[f] > 0 or offset_counts[f] > 0:
            result[f] = {
                "flip_on_rate": float(onset_counts[f] / n_safe),
                "flip_off_rate": float(offset_counts[f] / n_safe),
                "n_events": n_events,
            }

    return result
