"""
Representation-health checks for a finished run's latent geometry.

A fast, label-free GO/NO-GO gate for the three failure modes that make every
downstream lens meaningless: dimensional collapse (a z_enc axis whose across-sample
std sits at the floor), scale inflation (the terminal LayerNorm affine growing without
bound), and a predictor residual P - T that is a radius gap rather than a directional
error. Each check returns raw metrics plus a status in {pass, warn, fail}; assess()
folds them into one verdict.

Operates on z_enc_recency (N, D) for every architecture, plus the (z_pred, z_target)
pair for JEPA. Numpy and the geometry.py PCA helpers only - no forward, no disk I/O.
"""
import numpy as np

from src.analysis.geometry import fit_pca, get_pca_stats
from src.utils.system import SEED

# -- status levels
PASS = "pass"
WARN = "warn"
FAIL = "fail"

# -- collapse / inflation (per-dim std of z_enc)
STD_MIN_FLOOR         = 0.25
STD_MIN_WARN          = 0.35
STD_MEAN_INFLATE_WARN = 1.5
STD_MEAN_INFLATE_FAIL = 2.0

# -- effective rank (participation ratio as a fraction of D)
EFF_DIM_FRAC_WARN = 0.10
EFF_DIM_FRAC_FAIL = 0.03

# -- covariance redundancy (mean |off-diagonal correlation|)
COV_OFFDIAG_WARN = 0.30

# -- predictor alignment
MAG_FRAC_WARN = 0.50
MAG_FRAC_FAIL = 0.80
COS_DIST_WARN = 0.20
COS_DIST_FAIL = 0.40

# -- time_scale sanity band
TIME_SCALE_LO = 0.10
TIME_SCALE_HI = 50.0

# -- row cap for the O(N D^2) PCA / correlation computations
PCA_MAX_ROWS = 50_000


def subsample(z: np.ndarray, max_rows: int = PCA_MAX_ROWS) -> np.ndarray:
    """Seeded row subsample to bound PCA / correlation cost; no-op if small."""
    if z.shape[0] <= max_rows:
        return z
    rng = np.random.default_rng(SEED)
    idx = rng.choice(z.shape[0], size=max_rows, replace=False)
    return z[idx]


def variance_health(z_enc: np.ndarray) -> dict:
    """Per-dimension std and L2-norm stats; flags collapse via std_min and
    scale inflation via std_mean."""
    z = z_enc.astype(np.float64, copy=False)
    std_per_dim = z.std(axis=0)
    std_min = float(std_per_dim.min())
    std_mean = float(std_per_dim.mean())

    norms = np.linalg.norm(z, axis=1)
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())

    if std_min < STD_MIN_FLOOR or std_mean > STD_MEAN_INFLATE_FAIL:
        status = FAIL
    elif std_min < STD_MIN_WARN or std_mean > STD_MEAN_INFLATE_WARN:
        status = WARN
    else:
        status = PASS

    return {
        "std_min": std_min,
        "std_mean": std_mean,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "concentration": float(norm_std / norm_mean) if norm_mean > 0 else 0.0,
        "status": status,
    }


def effective_rank_health(z_enc: np.ndarray, k: int = 10) -> dict:
    """Participation ratio (effective dimensionality) of z_enc; flags rank
    collapse into a small subspace.

    Gates on effective_dimensionality / D, which is ~1 for an isotropic
    full-rank cloud and small under collapse. The Marchenko-Pastur signal count
    is reported for context but does not gate: it counts anisotropic structure
    above the noise floor, so a rich isotropic representation has few signal
    components by design.

    'z_enc' is assumed already subsampled; n_samples for the MP bound is taken
    from its row count.
    """
    N, D = z_enc.shape
    kk = min(k, D)
    pca, _, _ = fit_pca(z_enc.astype(np.float64, copy=False), k=kk)
    stats = get_pca_stats(pca, k=kk, n_samples=N)

    eff_dim = float(stats["effective_dimensionality"])
    eff_dim_frac = eff_dim / D

    if eff_dim_frac < EFF_DIM_FRAC_FAIL:
        status = FAIL
    elif eff_dim_frac < EFF_DIM_FRAC_WARN:
        status = WARN
    else:
        status = PASS

    return {
        "effective_dimensionality": eff_dim,
        "eff_dim_frac": float(eff_dim_frac),
        "n_signal_components": int(stats["n_signal_components"]),
        "components_for_90pct": int(stats["components_for_90pct"]),
        "components_for_95pct": int(stats["components_for_95pct"]),
        "n_dims": int(D),
        "status": status,
    }


def covariance_health(z_enc: np.ndarray) -> dict:
    """Off-diagonal correlation mass across z_enc dimensions (redundancy).

    'z_enc' is assumed already subsampled.
    """
    z = z_enc.astype(np.float64, copy=False)
    # dead (zero-std) dims yield nan correlations; dropped below
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(z, rowvar=False)
    D = corr.shape[0]
    off = np.abs(corr[~np.eye(D, dtype=bool)])
    off = off[~np.isnan(off)]
    if off.size == 0:
        return {"mean_abs_offdiag": 0.0, "max_abs_offdiag": 0.0,
                "frac_pairs_above_half": 0.0, "status": PASS}

    mean_abs = float(off.mean())
    status = WARN if mean_abs > COV_OFFDIAG_WARN else PASS
    return {
        "mean_abs_offdiag": mean_abs,
        "max_abs_offdiag": float(off.max()),
        "frac_pairs_above_half": float((off > 0.5).mean()),
        "status": status,
    }


def predictor_alignment_health(z_pred: np.ndarray, z_target: np.ndarray) -> dict:
    """Directional (cos_dist) and magnitude (radius-gap fraction of P - T)
    quality of the predictor residual.

    With a = ||z_pred||, b = ||z_target||, r = ||z_pred - z_target||, the
    residual at perfect directional alignment is |a - b|, so |a - b| / r is the
    fraction of the residual that is pure radius gap.
    """
    p = z_pred.astype(np.float64, copy=False)
    t = z_target.astype(np.float64, copy=False)

    a = np.linalg.norm(p, axis=1)
    b = np.linalg.norm(t, axis=1)
    r = np.linalg.norm(p - t, axis=1)

    cos = (p * t).sum(axis=1) / np.clip(a * b, 1e-12, None)
    cos_dist = float(1.0 - cos.mean())

    mag_frac_sample = float(np.clip(np.abs(a - b) / np.clip(r, 1e-12, None), 0.0, 1.0).mean())
    mag_frac_agg = float(abs(a.mean() - b.mean()) / max(r.mean(), 1e-12))

    if mag_frac_agg > MAG_FRAC_FAIL or cos_dist > COS_DIST_FAIL:
        status = FAIL
    elif mag_frac_agg > MAG_FRAC_WARN or cos_dist > COS_DIST_WARN:
        status = WARN
    else:
        status = PASS

    return {
        "cos_dist": cos_dist,
        "mag_frac_aggregate": mag_frac_agg,
        "mag_frac_sample_mean": mag_frac_sample,
        "z_pred_norm_mean": float(a.mean()),
        "z_target_norm_mean": float(b.mean()),
        "norm_ratio_pred_over_target": float(a.mean() / max(b.mean(), 1e-12)),
        "pred_err_l2_mean": float(r.mean()),
        "status": status,
    }


def time_scale_health(time_scale: float | None) -> dict:
    """Sanity band on the learned temporal-encoding scale."""
    if time_scale is None:
        return {"time_scale": None, "status": PASS}
    ts = float(time_scale)
    status = WARN if (ts < TIME_SCALE_LO or ts > TIME_SCALE_HI) else PASS
    return {"time_scale": ts, "status": status}


def overall_verdict(checks: dict) -> tuple[str, list[str]]:
    """GO / GO (marginal) / NO-GO from the per-check statuses."""
    fails = [n for n, c in checks.items() if c.get("status") == FAIL]
    warns = [n for n, c in checks.items() if c.get("status") == WARN]
    if fails:
        return "NO-GO", fails
    if warns:
        return "GO (marginal)", warns
    return "GO", []


def assess(vecs: dict, time_scale: float | None = None) -> dict:
    """Run the full panel on a derived-vector dict and assemble the verdict.

    'vecs' must contain z_enc_recency; z_pred / z_target enable the JEPA-only
    predictor-alignment check.
    """
    z_enc = np.asarray(vecs["z_enc_recency"]).astype(np.float32)
    z_enc_sub = subsample(z_enc)

    checks: dict = {
        "variance":       variance_health(z_enc),
        "effective_rank": effective_rank_health(z_enc_sub),
        "covariance":     covariance_health(z_enc_sub),
    }
    if vecs.get("z_pred") is not None and vecs.get("z_target") is not None:
        checks["predictor_alignment"] = predictor_alignment_health(
            np.asarray(vecs["z_pred"]), np.asarray(vecs["z_target"]))
    checks["time_scale"] = time_scale_health(time_scale)

    verdict, flagged = overall_verdict(checks)
    return {"verdict": verdict, "flagged": flagged, "checks": checks}
