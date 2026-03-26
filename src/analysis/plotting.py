from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

import numpy as np
from scipy import stats


def show_or_savefig(
    fig: Figure,
    show: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 150,
    **savefig_kwargs,
):
    """ I'll save your figure, %$&#, I'll even show it for you! """
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
        print(f"Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()
        
        
def plot_loss_curve(
    loss_history: list[float], run_dir: Path,
    show: bool = True
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("JEPA Training Loss")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    save_path = run_dir / "loss_curve.png"
    show_or_savefig(fig, show, save_path)

# =============================================================================
# Displacement geometry
# =============================================================================

def displacement_hist_mag_v_label(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_path: Path = None,
    vector_name: str = "Δ (P-C)",
):
    delta_norm  = np.linalg.norm(delta, axis=1)
    colors = {0: "steelblue", 1: "tomato"}

    # -- histogram --
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, col in colors.items():
        vals = delta_norm[labels == lbl]
        ax.hist(vals, bins=30, alpha=0.6, color=col, label=f"label={lbl}")
    ax.set_xlabel(f"||{vector_name}||  (L2 norm)")
    ax.set_ylabel("Count")
    ax.set_title(f"{vector_name}  Magnitude by Label")
    ax.legend()
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)
    
    
def displacement_boxp_mw(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_path: Path | None = None,
    vector_name: str = "Δ (P-C)",
):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    delta_norm  = np.linalg.norm(delta, axis=1)
    mw_stat, mw_p = stats.mannwhitneyu(x=delta_norm[labels == 0], y=delta_norm[labels == 1],
                                       alternative="two-sided")
    colors = {0: "steelblue", 1: "tomato"}


    fig, ax = plt.subplots(figsize=(5, 4))
    grouped = [
        (delta_norm[labels == lbl], lbl)
        for lbl in [0, 1] if (labels == lbl).any()
    ]

    bp = ax.boxplot([d for d, _ in grouped],
        label=[f"label={lbl}" for _, lbl in grouped],
        patch_artist=True,
    )
    for patch, (_, lbl) in zip(bp["boxes"], grouped):
        patch.set_facecolor(colors[lbl])
        patch.set_alpha(0.7)
    p_str = f"{mw_p:.3f}" if not np.isnan(mw_p) else "n/a"
    ax.set_ylabel(f"||{vector_name}||")
    ax.set_title(f"{vector_name}  Magnitude by Label\n(Mann-Whitney p = {p_str})")
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)
    
    
def displacement_heatmap(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_path: Path | None = None,
    vector_name: str = "Δ (P-C)",
):
    sort_idx     = np.argsort(labels)
    delta_sorted = delta[sort_idx]
    n_neg        = int((labels[sort_idx] == 0).sum())
    vmax = float(np.percentile(np.abs(delta_sorted), 99)) or 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        delta_sorted.T, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.axvline(n_neg - 0.5, color="black", linewidth=2, linestyle="--",
                label="label boundary")
    ax.set_xlabel("Sample (sorted by label)")
    ax.set_ylabel("Embedding dimension")
    ax.set_title(f"{vector_name} | left=label 0, right=label 1  (red= +, blue= -)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)
        
        
def displacement_dim_profile(
    delta: np.ndarray, labels: np.ndarray,
    show: bool = True, save_path: Path | None = None,
    vector_name: str = "Δ (P−C)",
):
    norms = np.linalg.norm(delta, axis=-1)
    safe_norms        = norms[:, np.newaxis].copy()
    safe_norms[safe_norms < 1e-10] = 1e-10
    delta_normed      = delta / safe_norms # unit vectors
    dim_mean          = delta_normed.mean(axis=0)
    dim_std           = delta_normed.std(axis=0)
    D                 = delta.shape[1]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(D)
    ax.bar(x, dim_mean, color="mediumseagreen", alpha=0.75, label="mean direction", width=1.0)
    ax.errorbar(x, dim_mean, yerr=dim_std, fmt="none", ecolor="black",
                elinewidth=1.0, capsize=2, alpha=0.5, label=r"$\pm$1 std")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Embedding dimension", fontsize=10)
    ax.set_xlim(-0.5, D-0.5)
    ax.set_xticks(np.arange(0, D, 2))
    ax.set_ylabel(f"Mean component of {vector_name} / ||{vector_name}||")
    ax.set_title(f"Per-dimension direction profile  (unit-normalised {vector_name})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ax.grid(True, axis="y", alpha=0.4)

    show_or_savefig(fig, show, save_path)
    
    
# =============================================================================
# PCA
# =============================================================================

def plot_pca_scree_v_mp_upper(
    eigenvalues: np.ndarray, mp_upper: float, n_signal: int, D: int, eff_dim: float,
    show: bool = True, save_path: Path | None = None,
    vector_name: str = "Δ (P−C)",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, D + 1), eigenvalues, "o-", markersize=4, label="Eigenvalues")
    ax.axhline(mp_upper, color="red", linestyle="--", linewidth=1.5,
                label=f"MP upper = {mp_upper:.3f}")
    if n_signal > 0:
        ax.axvline(n_signal + 0.5, color="gray", linestyle=":", linewidth=1,
                    label=f"{n_signal} signal component(s)")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(f"Eigenvalues vs. MP upper bound ({vector_name}) | d_eff = {eff_dim:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)

    
def plot_cum_var(
    evr: np.ndarray, D: int, eff_dim: float,
    show: bool = True, save_path: Path | None = None
):
    cumvar       = np.cumsum(evr)
    
    thresh90 = int(np.searchsorted(cumvar, 0.90)) + 1
    thresh95 = int(np.searchsorted(cumvar, 0.95)) + 1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, D + 1), cumvar, "b-o", markersize=3)
    ax.axhline(0.90, color="orange", linestyle="--", linewidth=1.2, label=f"90% ({thresh90} comp)")
    ax.axhline(0.95, color="red",    linestyle="--", linewidth=1.2, label=f"95% ({thresh95} comp)")
    ax.axvline(thresh90, color="orange", linestyle=":", alpha=0.6)
    ax.axvline(thresh95, color="red",    linestyle=":", alpha=0.6)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(r"Cumulative Variance  |  $d_{eff}$ (participation ratio) $= $" + f"{eff_dim:.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path)
    
    
def plot_pca1_vs_pca2(
    pca_projection: np.ndarray, labels: np.ndarray, evr: np.ndarray,
    show: bool = True, save_path: Path | None = None
):
    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl, col in [(0, "steelblue"), (1, "tomato")]:
        mask = labels == lbl
        if mask.any():
            ax.scatter(pca_projection[mask, 0], pca_projection[mask, 1],
                        c=col, label="readmitted" if lbl == 1 else "not readmitted", alpha=0.65, s=20)
    
    ax.set_xlabel(f"PC1  ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2  ({evr[1]:.1%} var)")
    ax.set_title("PC1 vs PC2 (coloured by label)")
    ax.legend()
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path)
    
    
def plot_pca1_vs_pca2_disp_mag(
    pca_projection: np.ndarray, evr: np.ndarray, delta: np.ndarray,
    show: bool = True, save_path: Path | None = None,
    vector_name: str = "Δ (P−C)",
):
    norms = np.linalg.norm(delta, axis=-1)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pca_projection[:, 0], pca_projection[:, 1],
                    c=norms, cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax, label=f"||{vector_name}||")
    ax.set_xlabel(f"PC1  ({evr[0]:.1%} variance)")
    ax.set_ylabel(f"PC2  ({evr[1]:.1%} variance)")
    ax.set_title(f"PC1 vs PC2  (coloured by ||{vector_name}||)")
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)

# =============================================================================
# UMAP
# =============================================================================

def plot_umap_vs_pca(
    umap_emb: np.ndarray, pca_projection: np.ndarray, labels: np.ndarray,
    show: bool = True, save_path: Path | None = None,
    vector_name: str = "Δ (P−C)",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lbl, col in [(0, "steelblue"), (1, "tomato")]:
        mask = labels == lbl
        if mask.any():
            axes[0].scatter(umap_emb[mask, 0], umap_emb[mask, 1],
                            c=col, label=f"label={lbl}", alpha=0.65, s=20)
            axes[1].scatter(pca_projection[mask, 0], pca_projection[mask, 1],
                            c=col, label=f"label={lbl}", alpha=0.65, s=20)
    axes[0].set_title(f"UMAP  ({vector_name})")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].legend(fontsize=8)
    axes[1].set_title("PCA  (same samples)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"UMAP vs PCA on {vector_name}")
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)
    

def plot_umap_disp_mag(
    umap_emb: np.ndarray, delta: np.ndarray,
    show: bool = True, save_path: Path | None = None,
    vector_name: str = "Δ (P−C)",
):
    norms = np.linalg.norm(delta, axis=-1)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(umap_emb[:, 0], umap_emb[:, 1],
                    c=norms, cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(sc, ax=ax, label=f"||{vector_name}||")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP coloured by ||{vector_name}||")
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)

# =============================================================================
# Three-vector decomposition
# =============================================================================

_VEC_COLORS = {
    "observed_traj": "#6B7280",   # gray  — T-C
    "delta":         "#3B82F6",   # blue  — P-C
    "pred_error":    "#EF4444",   # red   — P-T
}
_VEC_LABELS = {
    "observed_traj": "T−C (observed)",
    "delta":         "P−C (predicted)",
    "pred_error":    "P−T (error)",
}


def plot_shared_basis_decomposition(
    variances_observed: np.ndarray,
    variances_delta: np.ndarray,
    variances_pred_error: np.ndarray,
    explained_variance_ratio: np.ndarray,
    top_k: int | None = None,
    show: bool = True, save_path: Path | None = None,
):
    """Grouped bar chart of per-axis variance in the shared (observed_traj) PCA basis.

    Three bars per axis: observed trajectory (T-C), predicted trajectory (P-C),
    and prediction error (P-T).  Annotated with per-axis capture ratio
    ``1 - var(P-T) / var(T-C)``.

    top_k : axes to show (default: min(10, len))
    """
    k = min(top_k or 10, len(variances_observed))
    vo = variances_observed[:k]
    vd = variances_delta[:k]
    vp = variances_pred_error[:k]
    evr = explained_variance_ratio[:k]

    x = np.arange(k)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, k * 1.1), 5))
    ax.bar(x - w, vo, w, color=_VEC_COLORS["observed_traj"], alpha=0.85,
           label=_VEC_LABELS["observed_traj"])
    ax.bar(x,     vd, w, color=_VEC_COLORS["delta"],         alpha=0.85,
           label=_VEC_LABELS["delta"])
    ax.bar(x + w, vp, w, color=_VEC_COLORS["pred_error"],    alpha=0.85,
           label=_VEC_LABELS["pred_error"])

    # Per-axis capture ratio annotation
    with np.errstate(divide="ignore", invalid="ignore"):
        capture = np.where(vo > 1e-12, 1.0 - vp / vo, np.nan)

    for i in range(k):
        if not np.isnan(capture[i]):
            ax.text(x[i], max(vo[i], vd[i], vp[i]) * 1.04,
                    f"{capture[i]:.0%}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i+1}\n({evr[i]:.1%})" for i in range(k)],
                       fontsize=7)
    ax.set_xlabel("Shared PCA axis  (% variance of T-C)")
    ax.set_ylabel("Projection variance")
    ax.set_title("Shared-Basis Decomposition  |  annotations = capture ratio "
                 "(1 - var(P-T)/var(T-C))")
    ax.legend(fontsize=8)
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)


def plot_three_vector_norm_comparison(
    delta: np.ndarray,
    pred_error: np.ndarray,
    observed_traj: np.ndarray,
    labels: np.ndarray,
    show: bool = True, save_path: Path | None = None,
):
    """Overlaid histograms of L2 norms for all three vectors, faceted by label.

    Two panels (label=0, label=1).  Distinct colours with alpha for each vector.
    """
    norms = {
        "delta":         np.linalg.norm(delta, axis=-1),
        "pred_error":    np.linalg.norm(pred_error, axis=-1),
        "observed_traj": np.linalg.norm(observed_traj, axis=-1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, lbl_val in zip(axes, [0, 1]):
        mask = labels == lbl_val
        for name in ("observed_traj", "delta", "pred_error"):
            ax.hist(norms[name][mask], bins=40, alpha=0.45,
                    color=_VEC_COLORS[name], label=_VEC_LABELS[name])
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("Count")
        ax.set_title(f"Label = {lbl_val}  (n={mask.sum()})")
        ax.legend(fontsize=7)

    fig.suptitle("Three-Vector Norm Comparison by Label", fontsize=11)
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)


def plot_context_umap(
    umap_embedding: np.ndarray,
    labels: np.ndarray,
    cluster_labels: np.ndarray | None = None,
    show: bool = True, save_path: Path | None = None,
):
    """UMAP of z_context coloured by label, optionally also by cluster.

    If cluster_labels provided: two side-by-side panels (label / cluster).
    Otherwise: single panel coloured by readmission label.
    """
    def _draw_label_panel(ax):
        for lbl_val, col in [(0, "steelblue"), (1, "tomato")]:
            mask = labels == lbl_val
            tag = "readmitted" if lbl_val else "not readmitted"
            ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                       c=col, s=10, alpha=0.5, label=f"{tag} (n={mask.sum()})",
                       rasterized=True)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title("Coloured by readmission label")
        ax.legend(fontsize=8)

    if cluster_labels is not None:
        fig, (ax_lbl, ax_cls) = plt.subplots(1, 2, figsize=(13, 5))
        _draw_label_panel(ax_lbl)

        unique = sorted(set(cluster_labels))
        cids   = [c for c in unique if c >= 0]
        n_cls  = len(cids)

        noise = cluster_labels == -1
        if noise.any():
            ax_cls.scatter(umap_embedding[noise, 0], umap_embedding[noise, 1],
                           c="lightgray", s=6, alpha=0.25,
                           label=f"noise (n={noise.sum()})", rasterized=True)

        cmap = plt.get_cmap("tab10" if n_cls <= 10 else "tab20")
        for i, cid in enumerate(cids):
            mask = cluster_labels == cid
            ax_cls.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                           c=[cmap(i % cmap.N)], s=10, alpha=0.55,
                           label=f"C{cid} (n={mask.sum()})", rasterized=True)
        ax_cls.set_xlabel("UMAP-1")
        ax_cls.set_ylabel("UMAP-2")
        ax_cls.set_title(f"Coloured by cluster  ({n_cls} clusters)")
        ax_cls.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        fig, ax_lbl = plt.subplots(figsize=(7, 6))
        _draw_label_panel(ax_lbl)

    fig.suptitle("Patient State Space (z_context)", fontsize=11)
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)


def plot_umap_phate_comparison(
    umap_embedding: np.ndarray,
    phate_embedding: np.ndarray,
    labels: np.ndarray,
    show: bool = True, save_path: Path | None = None,
):
    """Side-by-side UMAP vs PHATE, both coloured by readmission label."""
    fig, (ax_u, ax_p) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, emb, title, ax_prefix in [
        (ax_u, umap_embedding, "UMAP (euclidean)", "UMAP"),
        (ax_p, phate_embedding, "PHATE", "PHATE"),
    ]:
        for lbl_val, col in [(0, "steelblue"), (1, "tomato")]:
            mask = labels == lbl_val
            tag = "readmitted" if lbl_val else "not readmitted"
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=col, s=10, alpha=0.5, label=f"{tag} (n={mask.sum()})",
                       rasterized=True)
        ax.set_xlabel(f"{ax_prefix}-1")
        ax.set_ylabel(f"{ax_prefix}-2")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Patient State Space (z_context)", fontsize=11)
    fig.tight_layout()

    show_or_savefig(fig, show, save_path)
    
    
def plot_phate_eigen_decomp(
    eigs: np.ndarray, phate_nn: int, n_eigs: int,
    show: bool = True, save_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(eigs) + 1), eigs, "o-", markersize=4, color="teal")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Diffusion operator eigenvalue")
    ax.set_title(f"PHATE diffusion operator spectrum (top {n_eigs}, knn={phate_nn})")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path)

# =============================================================================
# Plotting - Divergence
# =============================================================================

def plot_scatter_pairwise_dvg_scatter(
    reg_result: dict,
    context_sims:      np.ndarray,
    pred_dists:        np.ndarray,
    pair_indices:      tuple,
    max_scatter:       int = 50000,
    show: bool = True, save_path: Path | None = None
) -> None:
    """
    Context similarity vs. prediction distance scatter with OLS regression line.
    Divergent pairs highlighted in red.
    """
    slope     = reg_result["slope"]
    intercept = reg_result["intercept"]
    r         = reg_result["r"]
    p         = reg_result["p"]
    x         = reg_result["ctx_sim_flat"]
    y         = reg_result["pred_dist_flat"]
    n_total   = len(x)

    row_idx, col_idx = pair_indices
    
    # Background subsample
    n_dvg = len(row_idx)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(n_total, size=min(max_scatter, n_total), replace=False)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[bg_idx], y[bg_idx],
               alpha=0.2, s=4, color="gray", 
               label=f"all pairs (subsample n={len(bg_idx):,})")
    
    # colored divergent pairs
    if n_dvg > 0:
        div_x = context_sims[row_idx, col_idx]
        div_y = pred_dists[row_idx, col_idx]
        ax.scatter(div_x, div_y,
                   alpha=0.8, s=15, color="tomato", zorder=3,
                   label=f"divergent pairs (n={n_dvg:,})")
    
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    ax.plot(x_line, slope * x_line + intercept, "b-", linewidth=2,
            label=f"OLS  r = {r:.3f},  p = {p:.2e}")
    ax.set_xlabel("Context cosine similarity")
    ax.set_ylabel("Prediction cosine distance")
    ax.set_title("Context Similarity vs. Prediction Divergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path)


def plot_divergence_pc_projections(
    div_pc_var:  np.ndarray,
    rand_pc_var: np.ndarray,
    show: bool = True, save_path: Path | None = None
) -> None:
    """
    Grouped bar chart: divergence vs. random pair projection variance per PC.
    """
    k     = len(div_pc_var)
    x_pos = np.arange(k)
    width = 0.35

    safe_div  = np.where(np.isnan(div_pc_var),  0, div_pc_var)
    safe_rand = np.where(np.isnan(rand_pc_var), 0, rand_pc_var)

    fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 4))
    ax.bar(x_pos - width / 2, safe_div,  width,
           label="Divergent pairs", color="tomato",   alpha=0.8)
    ax.bar(x_pos + width / 2, safe_rand, width,
           label="Random pairs",   color="steelblue", alpha=0.8)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Projection variance")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"PC{i+1}" for i in range(k)], rotation=45)
    ax.set_title("Divergence Vectors: Projection Variance per PC")
    ax.legend()
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path)


def plot_variance_comparison(
    div_pc_var:  np.ndarray,
    rand_pc_var: np.ndarray,
    show: bool = True, save_path: Path | None = None
) -> None:
    """
    Variance ratio (divergent / random) per PC.

    Bars above 1.0 → divergent pairs concentrate more variance along that PC
    than random pairs — indicating genuine geometric structure.
    """
    k     = len(div_pc_var)
    x_pos = np.arange(k)
    ratio = np.where(rand_pc_var > 1e-12,
                     div_pc_var / rand_pc_var,
                     np.nan)

    bar_colors = [
        "tomato"    if (not np.isnan(r) and r > 1.0) else "steelblue"
        for r in ratio
    ]

    fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 4))
    ax.bar(x_pos, np.where(np.isnan(ratio), 0, ratio), color=bar_colors, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
               label="ratio = 1  (no enrichment vs. chance)")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Variance ratio  (divergent / random)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"PC{i+1}" for i in range(k)], rotation=45)
    ax.set_title("Divergent vs. Random Pair Variance Ratio per PC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path)


# =============================================================================
# ICC Stability
# =============================================================================

def plot_icc_bar(
    icc_result: dict,
    show: bool = True, save_path: Path | None = None
) -> None:
    """
    ICC per PC bar chart, colour-coded by trait / state threshold.

    Red    = ICC > TRAIT_THRESHOLD  (0.8) - stable across encounter windows
    Blue   = ICC < STATE_THRESHOLD  (0.2) - varies with encounter context
    Silver = intermediate / ambiguous
    """
    icc_dict   = icc_result["icc_per_pc"]
    n_eligible = icc_result["eligible_patients"]
    trait_t    = icc_result["trait_threshold"]
    state_t    = icc_result["state_threshold"]
    method     = icc_result["method"]
    pc_labels  = list(icc_dict.keys())
    k          = len(pc_labels)
    icc_values = np.array([v if v is not None else np.nan
                            for v in icc_dict.values()])

    bar_colors = [
        "tomato"    if (not np.isnan(v) and v > trait_t) else
        "steelblue" if (not np.isnan(v) and v < state_t) else
        "silver"
        for v in icc_values
    ]

    fig, ax = plt.subplots(figsize=(max(7, k * 0.9), 4))
    ax.bar(np.arange(k), np.where(np.isnan(icc_values), 0, icc_values),
           color=bar_colors, alpha=0.85)
    ax.axhline(trait_t, color="tomato",    linestyle="--", linewidth=1.5,
               label=f"Trait  (ICC > {trait_t})")
    ax.axhline(state_t, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"State  (ICC < {state_t})")
    ax.axhline(0.0,     color="black",     linestyle="-",  linewidth=0.5)
    ax.set_xlabel("PC index")
    ax.set_ylabel(f"ICC  ({method})")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(pc_labels, rotation=45)
    ax.set_title(f"Intraclass Correlation per PC | {n_eligible} patients")
    ax.legend(fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path)


def plot_spaghetti(
    pc_projections: np.ndarray,
    subject_ids:    np.ndarray,
    mask_positions: np.ndarray,
    pc_idx:         int = 0,
    max_patients:   int = 10,
    min_samples:    int = 3,
    show: bool = True, save_path: Path | None = None
) -> None:
    """
    PC score vs. mask_position for selected patients ("spaghetti plot").

    Each line is a patient; each point is one masked encounter window.
    High-ICC PCs -> roughly flat lines.
    Low-ICC PCs -> variable lines.

    Parameters
    ----------
    pc_projections : (N, k) PC score matrix
    subject_ids    : (N,) patient IDs
    mask_positions : (N,) integer encounter indices
    out_dir        : output directory
    pc_idx         : which PC to show (0-indexed)
    max_patients   : patients to overlay (largest sample count wins)
    min_samples    : minimum windows per patient for inclusion
    """
    unq_subject_ids   = np.unique(subject_ids)
    sample_counts = {sid: int((subject_ids == sid).sum()) for sid in unq_subject_ids}
    eligible      = [sid for sid in unq_subject_ids if sample_counts[sid] >= min_samples]
    plot_sids     = sorted(eligible, key=lambda s: -sample_counts[s])[:max_patients]

    if not plot_sids:
        print(f"plot_spaghetti: no patients with >= {min_samples} samples..")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, sid in enumerate(plot_sids):
        m         = subject_ids == sid
        positions = mask_positions[m]
        pc_vals   = pc_projections[m, pc_idx]
        order     = np.argsort(positions)
        ax.plot(positions[order], pc_vals[order], "o-",
                color=cmap(i % 10), alpha=0.75, markersize=5,
                label=f"pid {sid}")

    ax.set_xlabel("Mask position (encounter index)")
    ax.set_ylabel(f"PC{pc_idx + 1} projection score")
    ax.set_title(
        f"PC{pc_idx + 1} Score Across Encounter Windows  "
        f"({len(plot_sids)} patients)"
    )
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path)


# =============================================================================
# Feature Extraction (LASSO)
# =============================================================================

def plot_lasso_heatmap(
    lasso_result: dict,
    show: bool = True, save_path: Path | None = None
) -> None:
    """
    Coefficient heatmap: metadata features x PCs.
    Stability annotations (>50%) shown on significant cells.
    Only features selected in at least one PC (stability > 0.3) are shown.
    """

    coeff      = lasso_result["coeff_matrix"]
    stab       = lasso_result["stability_matrix"]
    feat_names = lasso_result["feature_names"]
    k          = lasso_result["top_k"]

    # Filter to features selected in at least one PC
    max_stab = stab[:, :k].max(axis=1)
    sel_mask = max_stab > 0.3
    if sel_mask.sum() == 0:
        sel_mask = np.ones(len(feat_names), dtype=bool)

    coeff_sel = coeff[sel_mask, :k]
    stab_sel  = stab[sel_mask, :k]
    names_sel = [feat_names[i] for i in range(len(feat_names)) if sel_mask[i]]
    n_feats   = len(names_sel)

    fig_w = max(7, k * 0.85)
    fig_h = max(4, n_feats * 0.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.abs(coeff_sel).max()) or 1.0
    im = ax.imshow(coeff_sel, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    # Annotate stability for significant coefficients
    for i in range(n_feats):
        for j in range(k):
            if abs(coeff_sel[i, j]) > 1e-10 and stab_sel[i, j] > 0.5:
                txt_color = ("white" if abs(coeff_sel[i, j]) > vmax * 0.6
                             else "black")
                ax.text(j, i, f"{stab_sel[i, j]:.0%}",
                        ha="center", va="center", fontsize=6, color=txt_color)

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(k)], rotation=45)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(names_sel, fontsize=7)
    ax.set_xlabel("PC component")
    ax.set_ylabel("Metadata feature")
    ax.set_title(
        r"LASSO Coefficients  (metadata $\rightarrow$ PC score)\n"
        "Annotations = bootstrap stability (shown if > 50%)"
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Coefficient")
    fig.tight_layout()
    
    # lasso_coeff_heatmap.png
    show_or_savefig(fig, show, save_path)


def plot_r2_bar(
    lasso_result: dict,
    show: bool = True, save_path: Path | None = None
) -> None:
    """R^2 bar chart per PC with mean R^2 line and unexplained fraction."""
    r2_dict     = lasso_result["r2_per_pc"]
    k           = len(r2_dict)
    r2_vals     = [r2_dict[f"PC{i + 1}"] for i in range(k)]
    mean_r2     = lasso_result["mean_r2"]
    unexplained = lasso_result["unexplained_variance_fraction"]
    n_feats     = lasso_result["n_features"]
    n_patients  = lasso_result["n_patients"]

    fig_w = max(7, k * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    ax.bar(range(k), r2_vals, color="mediumseagreen", alpha=0.85)
    ax.axhline(mean_r2, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean R² = {mean_r2:.3f}")
    ax.set_xlabel("PC index")
    ax.set_ylabel(r"$R^2$")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(k)], rotation=45)
    ax.set_title(
        rf"LASSO R^2 per PC  |  Mean R^2 = {mean_r2:.3f}  |  "
        f"Unexplained = {unexplained:.1%}\n"
        f"({n_feats} metadata features, {n_patients} patients)"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    
    # r2_bar.png
    show_or_savefig(fig, show, save_path)


# =============================================================================
# Feature Extraction (UMAP/HBDSCAN clustering)
# =============================================================================

def plot_cluster_enrichment(
    cluster_result: dict,
    max_features: int = 30,
    show: bool = True, save_path: Path | None = None
) -> None:
    """
    Enrichment heatmap: features x clusters, coloured by z-score.
    Top features selected by max |z-score| across clusters.
    """
    enrichment  = cluster_result["enrichment_matrix"]
    feat_names  = cluster_result["feature_names"]
    cluster_ids = cluster_result["cluster_ids"]
    n_clusters  = cluster_result["n_clusters"]

    if n_clusters == 0:
        return

    # Select top features by max |z-score| across clusters
    max_z   = np.abs(enrichment).max(axis=0)
    top_idx = np.argsort(max_z)[::-1][:max_features]
    top_idx = sorted(top_idx)

    enrichment_sel = enrichment[:, top_idx]
    names_sel      = [feat_names[i] for i in top_idx]
    n_feats        = len(names_sel)

    fig_w = max(7, n_clusters * 1.2 + 3)
    fig_h = max(6, n_feats * 0.35)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(2.0, float(np.abs(enrichment_sel).max()))
    im = ax.imshow(enrichment_sel.T, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    sizes = cluster_result["cluster_sizes"]
    x_labels = [f"C{cid}\n(n={sizes.get(cid, '?')})" for cid in cluster_ids]

    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(names_sel, fontsize=7)
    ax.set_xlabel("HDBSCAN Cluster")
    ax.set_ylabel("Metadata Feature")

    unlabeled = cluster_result.get("unlabeled_clusters", [])
    title = f"Cluster Enrichment (z-score vs population)  |  {n_clusters} clusters"
    if unlabeled:
        title += f"\nUnlabeled clusters (no |z| > 1): {unlabeled}"
    ax.set_title(title, fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.6, label="z-score")
    fig.tight_layout()
    
    # cluster_enrichment_heatmap.png
    show_or_savefig(fig, show, save_path)


def plot_cluster_umap(
    umap_embedding: np.ndarray,
    cluster_labels: np.ndarray,
    show: bool = True, save_path: Path | None = None
) -> None:
    """UMAP scatter coloured by HDBSCAN cluster assignment."""
    unique     = sorted(set(cluster_labels))
    cids       = [c for c in unique if c >= 0]
    n_clusters = len(cids)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Noise points (gray)
    noise = cluster_labels == -1
    if noise.any():
        ax.scatter(umap_embedding[noise, 0], umap_embedding[noise, 1],
                   c="lightgray", s=8, alpha=0.3,
                   label=f"noise (n={noise.sum()})")

    cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
    for i, cid in enumerate(cids):
        mask = cluster_labels == cid
        ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1],
                   c=[cmap(i % cmap.N)], s=20, alpha=0.7,
                   label=f"C{cid} (n={mask.sum()})")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"HDBSCAN Clusters on UMAP  |  {n_clusters} clusters")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    
    # cluster_umap.png
    show_or_savefig(fig, show, save_path)

# =============================================================================
# SAE Analysis
# =============================================================================

def plot_sae_heatmap_top_features(
    sub: np.ndarray, 
    active_pe: np.ndarray, 
    active_ot: np.ndarray,
    show: bool = True, save_path: Path | None = None
):
    top_pe_idx = np.argsort(sub.max(axis=1))[::-1][:30]
    top_ot_idx = np.argsort(sub.max(axis=0))[::-1][:30]
    heatmap_sub = sub[np.ix_(top_pe_idx, top_ot_idx)]

    fig, ax = plt.subplots(figsize=(max(8, len(top_ot_idx) * 0.4),
                                    max(6, len(top_pe_idx) * 0.35)))
    im = ax.imshow(heatmap_sub, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(top_ot_idx)))
    ax.set_xticklabels([f"OT_F{active_ot[i]}" for i in top_ot_idx],
                        rotation=90, fontsize=6)
    ax.set_yticks(range(len(top_pe_idx)))
    ax.set_yticklabels([f"PE_F{active_pe[i]}" for i in top_pe_idx],
                        fontsize=6)
    ax.set_xlabel("observed_traj SAE feature")
    ax.set_ylabel("pred_error SAE feature")
    ax.set_title("Cross-Target Co-Activation\n"
                    "(what real dynamics does the model mispredict?)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalised co-activation")
    fig.tight_layout()
    show_or_savefig(fig, show, save_path)

# =============================================================================
# Linear Probing
# =============================================================================

def plot_probing_curve(
    sweep_results: dict,
    show: bool = True,
    save_path=None,
):
    """AUC by encoder layer ('probing curve').
        - sweep_results : output from run_probing_sweep
    """
    summary = sweep_results["summary"]  # list of (key, mean_auc, std_auc)
    labels = [s[0] for s in summary]
    aucs = [s[1] for s in summary]
    stds = [s[2] for s in summary]

    # Nicer x-tick labels
    tick_labels = []
    for lbl in labels:
        if lbl.startswith("layer_"):
            tick_labels.append(f"L{lbl.split('_')[1]}")
        else:
            tick_labels.append(lbl.capitalize())

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, aucs, yerr=stds, fmt="o-", capsize=4, linewidth=2,
                markersize=6, color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Linear Probing Curve: AUC by Layer")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()

    # probing_curve.png
    show_or_savefig(fig, show, save_path)


def plot_probing_metrics(
    sweep_results: dict,
    show: bool = True,
    save_path=None,
):
    """Grouped bar chart of AUC, Accuracy, F1 per layer.
        - sweep_results : output from run_probing_sweep
    """
    per_layer = sweep_results["per_layer"]
    keys = [s[0] for s in sweep_results["summary"]]

    tick_labels = []
    for k in keys:
        if k.startswith("layer_"):
            tick_labels.append(f"L{k.split('_')[1]}")
        else:
            tick_labels.append(k.capitalize())

    aucs = [per_layer[k]["mean_auc"] for k in keys]
    accs = [per_layer[k]["mean_accuracy"] for k in keys]
    f1s = [per_layer[k]["mean_f1"] for k in keys]

    x = np.arange(len(keys))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, aucs, w, label="AUC", color="#2563eb")
    ax.bar(x, accs, w, label="Accuracy", color="#16a34a")
    ax.bar(x + w, f1s, w, label="F1", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe Metrics by Layer")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    # probing_metrics.png
    show_or_savefig(fig, show, save_path)


def plot_probing_comparison(
    jepa_sweep: dict,
    softmax_sweep: dict,
    show: bool = True,
    save_path=None,
):
    """Side-by-side probing curves for JEPA vs softmax model.
        - jepa_sweep    : output from run_probing_sweep on JEPA
        - softmax_sweep : output from run_probing_sweep on softmax model
    """
    j_summary = jepa_sweep["summary"]
    s_summary = softmax_sweep["summary"]

    labels = [s[0] for s in j_summary]
    tick_labels = []
    for lbl in labels:
        if lbl.startswith("layer_"):
            tick_labels.append(f"L{lbl.split('_')[1]}")
        else:
            tick_labels.append(lbl.capitalize())

    x = np.arange(len(labels))

    j_aucs = [s[1] for s in j_summary]
    j_stds = [s[2] for s in j_summary]
    s_aucs = [s[1] for s in s_summary]
    s_stds = [s[2] for s in s_summary]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, j_aucs, yerr=j_stds, fmt="o-", capsize=4, linewidth=2,
                markersize=6, color="#2563eb", label="JEPA (no softmax)")
    ax.errorbar(x, s_aucs, yerr=s_stds, fmt="s--", capsize=4, linewidth=2,
                markersize=6, color="#dc2626", label="Softmax baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Probing Curve: JEPA vs Softmax")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()

    # probing_comparison.png
    show_or_savefig(fig, show, save_path)


def plot_f3x_stratified_probing(
    stratified_results: dict,
    show: bool = True,
    save_path=None,
):
    """
        Grouped bar chart of AUC by F3x sub-block at the final layer.
        
        Note: Requires sufficient positive samples per sub-block for stable estimates.

        Parameters
        ----------
        stratified_results : dict mapping sub-block name -> probe result dict
            e.g. {"F31 (bipolar)": {...}, "F32 (depressive)": {...}, ...}
    """
    names = list(stratified_results.keys())
    aucs = [stratified_results[n]["mean_auc"] for n in names]
    stds = [stratified_results[n]["std_auc"] for n in names]
    n_pos = [stratified_results[n]["n_positive"] for n in names]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, aucs, yerr=stds, capsize=4, color="#2563eb", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Probing AUC by F3x Sub-block (Final Layer)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")

    # Annotate with n_positive
    for i, (bar, n) in enumerate(zip(bars, n_pos)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    # f3x_probing.png
    show_or_savefig(fig, show, save_path)