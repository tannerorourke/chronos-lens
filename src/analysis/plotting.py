from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

from src.utils.constants import ESCALATION_CRITERIA
from src.utils.io import EXPERIMENTS_DIR


C_POS = "#F97316"  # escalation positive
C_NEG = "#94A3B8"  # escalation negative
CRITERIA_COLORS = {
    "new_subcategory":   "#3B82F6",   # blue
    "severity_increase": "#EF4444",   # red
    "new_specifier":     "#10B981",   # green
    "f32_to_f33":        "#F59E0B",   # amber
    "med_initiation":    "#8B5CF6",   # purple
    "new_drug_class":    "#EC4899",   # pink
}
VEC_COLORS = {
    "z_pred":     "#3B82F6",   # blue  - predictor output
    "z_target":   "#10B981",   # green - target encoder output
    "z_enc":      "#8B5CF6",   # purple - encoder output
    "pred_error": "#EF4444",   # red   - P-T
}
VEC_LABELS = {
    "z_pred":     "z_pred",
    "z_target":   "z_target",
    "z_enc":      "z_enc",
    "pred_error": "P-T (error)",
}
LABEL_COLORS = {0: "#94A3B8", 1: "#F97316"}          # slate / orange
LABEL_NAMES  = {0: "no escalation", 1: "escalation"}
_TITLE_PT  = 10
_LABEL_PT  = 9
_ANNOT_PT  = 7
_TICK_PT   = 7

ARROW_LABELS = { "pred_error": r"$P \to T$ (error)" }


def show_or_savefig(
    fig: Figure,
    show: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 300,
    **savefig_kwargs,
):
    """ I'll save your figure, %$&#, I'll even show it for you! """
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches="tight", **savefig_kwargs)
        print(f"Saved fig: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()
        
        
def plot_loss_curve(
    loss_history: list[float], 
    show: bool = False, save: bool = True,
    fig_dir: Path = EXPERIMENTS_DIR / "figures",
    fig_name: str = "loss_curve",
    title: str = "Training Loss",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title, fontsize=_TITLE_PT)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path=fig_dir / fig_name if save else None)

# =============================================================================
# Encounter Representation Characteristics
# =============================================================================

def _s1_eigenvalue_spectrum(results, pca_enc, pca_pat,
                            show: bool = True, save: bool = False, 
                            fig_dir: Path = EXPERIMENTS_DIR / "figures",
                            fig_name: str = "1_eigenvalue_spectrum"):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    pca_pat_r = results.get("pca_patient", {})
    pca_enc_r = results.get("pca_encounter", {})

    if pca_pat is not None:
        eigs = pca_pat["eigenvalues"]
        D = len(eigs)
        ax.semilogy(range(1, D + 1), eigs, "o-", markersize=3, linewidth=1.5,
                     color=VEC_COLORS.get("z_enc", "#8B5CF6"),
                     label="Patient-level")
        mp = pca_pat_r.get("mp_upper_bound")
        if mp is not None:
            ax.axhline(mp, color=VEC_COLORS.get("z_enc", "#8B5CF6"),
                       linestyle="--", alpha=0.6, linewidth=1,
                       label=f"MP upper (patient) = {mp:.3f}")

    if pca_enc is not None:
        eigs = pca_enc["eigenvalues"]
        D = len(eigs)
        ax.semilogy(range(1, D + 1), eigs, "s-", markersize=3, linewidth=1.5,
                     color=VEC_COLORS.get("z_target", "#10B981"),
                     label="Encounter-level")
        mp = pca_enc_r.get("mp_upper_bound")
        if mp is not None:
            ax.axhline(mp, color=VEC_COLORS.get("z_target", "#10B981"),
                       linestyle="--", alpha=0.6, linewidth=1,
                       label=f"MP upper (encounter) = {mp:.3f}")

    # Annotations
    parts = []
    for tag, r in [("Patient", pca_pat_r), ("Encounter", pca_enc_r)]:
        d_eff = r.get("effective_dimensionality")
        n_sig = r.get("n_signal_components")
        if d_eff is not None:
            parts.append(f"{tag}: d_eff={d_eff:.1f}, {n_sig} signal")
    if parts:
        ax.text(0.98, 0.98, "\n".join(parts), transform=ax.transAxes,
                ha="right", va="top", fontsize=_ANNOT_PT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CBD5E1", alpha=0.9))

    ax.set_xlabel("Component index", fontsize=_LABEL_PT)
    ax.set_ylabel("Eigenvalue (log scale)", fontsize=_LABEL_PT)
    ax.set_title("Eigenvalue spectrum - encounter vs patient PCA",
                 fontsize=_TITLE_PT)
    ax.legend(fontsize=_ANNOT_PT, loc="upper right")
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s1_icc_heatmap(results, 
                    show: bool = True, save: bool = False, 
                    fig_dir: Path = EXPERIMENTS_DIR / "figures",
                    fig_name: str = "1_icc_heatmap"):
    icc = results.get("icc")
    if not icc:
        return

    icc_per_pc = icc["icc_per_pc"]
    trait_t = icc.get("trait_threshold", 0.8)
    state_t = icc.get("state_threshold", 0.2)
    pc_labels = list(icc_per_pc.keys())
    values = np.array([v if v is not None else np.nan
                       for v in icc_per_pc.values()])
    k = len(values)

    fig, ax = plt.subplots(figsize=(2.2, max(3, k * 0.35)), dpi=300)

    # Custom colormap: red (<0.2) -> gray -> green (>0.8)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "icc", ["#EF4444", "#94A3B8", "#10B981"])

    im = ax.imshow(values.reshape(-1, 1), aspect="auto", cmap=cmap,
                   vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_yticks(range(k))
    ax.set_yticklabels(pc_labels, fontsize=_TICK_PT)
    ax.set_xticks([0])
    ax.set_xticklabels(["ICC"], fontsize=_TICK_PT)

    for i, v in enumerate(values):
        if np.isnan(v):
            txt, col = "n/a", "#64748B"
        elif v > trait_t:
            txt, col = f"{v:.2f} trait", "white"
        elif v < state_t:
            txt, col = f"{v:.2f} state", "white"
        else:
            txt, col = f"{v:.2f}", "#1E293B"
        ax.text(0, i, txt, ha="center", va="center",
                fontsize=_ANNOT_PT, fontweight="bold", color=col)

    ax.set_title("ICC per PC axis", fontsize=_TITLE_PT, pad=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="ICC")
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s1_embedding_panel(emb, cluster_labels, labels, 
                        criteria_labels, mask_pos, method_name,
                        show: bool = True, save: bool = False, 
                        fig_dir: Path = EXPERIMENTS_DIR / "figures",
                        fig_name: str = "1_embedding_panel"):
    """ Embedding 2x2 panel (UMAP or PHATE) """
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300)

    prefix = f"{method_name}-1"
    suffix = f"{method_name}-2"

    # (a) HDBSCAN clusters
    ax = axes[0, 0]
    if cluster_labels is not None:
        unique_cl = sorted(set(cluster_labels))
        cids = [c for c in unique_cl if c >= 0]
        cmap_cl = plt.get_cmap("tab10" if len(cids) <= 10 else "tab20")

        noise = cluster_labels == -1
        if noise.any():
            ax.scatter(emb[noise, 0], emb[noise, 1], s=4, alpha=0.12,
                       color="lightgray", label="noise", rasterized=True)
        for i, cid in enumerate(cids):
            m = cluster_labels == cid
            ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.35,
                       color=cmap_cl(i % cmap_cl.N),
                       label=f"C{cid}", rasterized=True)
        ax.legend(fontsize=5, ncol=2, loc="upper right",
                  framealpha=0.7, markerscale=2)
    else:
        ax.text(0.5, 0.5, "No cluster labels", transform=ax.transAxes,
                ha="center", va="center", fontsize=_LABEL_PT)
    ax.set_xlabel(prefix, fontsize=_LABEL_PT)
    ax.set_ylabel(suffix, fontsize=_LABEL_PT)
    ax.set_title("(a) HDBSCAN clusters", fontsize=_TITLE_PT)
    ax.tick_params(labelsize=_TICK_PT)

    # (b) Escalation label (binary)
    ax = axes[0, 1]
    if labels is not None:
        for lv, col in LABEL_COLORS.items():
            m = labels == lv
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.25,
                           color=col, label=LABEL_NAMES[lv], rasterized=True)
        ax.legend(fontsize=_ANNOT_PT, markerscale=2)
    else:
        ax.text(0.5, 0.5, "Labels not provided", transform=ax.transAxes,
                ha="center", va="center", fontsize=_LABEL_PT)
    ax.set_xlabel(prefix, fontsize=_LABEL_PT)
    ax.set_ylabel(suffix, fontsize=_LABEL_PT)
    ax.set_title("(b) Escalation label", fontsize=_TITLE_PT)
    ax.tick_params(labelsize=_TICK_PT)

    # (c) Escalation criterion (categorical)
    ax = axes[1, 0]
    if criteria_labels is not None and labels is not None:
        # Gray background for escalation=0
        m0 = labels == 0
        if m0.any():
            ax.scatter(emb[m0, 0], emb[m0, 1], s=4, alpha=0.08,
                       color=LABEL_COLORS[0], label="no escalation",
                       rasterized=True)
        for crit in ESCALATION_CRITERIA:
            arr = criteria_labels.get(crit)
            if arr is None:
                continue
            m = arr == 1
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.45,
                           color=CRITERIA_COLORS.get(crit, "#64748B"),
                           label=crit, rasterized=True)
        ax.legend(fontsize=5, ncol=2, loc="upper right",
                  framealpha=0.7, markerscale=2)
    else:
        ax.text(0.5, 0.5, "Criteria not provided", transform=ax.transAxes,
                ha="center", va="center", fontsize=_LABEL_PT)
    ax.set_xlabel(prefix, fontsize=_LABEL_PT)
    ax.set_ylabel(suffix, fontsize=_LABEL_PT)
    ax.set_title("(c) Escalation criterion", fontsize=_TITLE_PT)
    ax.tick_params(labelsize=_TICK_PT)

    # (d) Encounter position (ordinal)
    ax = axes[1, 1]
    if mask_pos is not None:
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.30,
                        c=mask_pos, cmap="viridis", rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="Encounter position")
    else:
        ax.text(0.5, 0.5, "mask_pos not provided", transform=ax.transAxes,
                ha="center", va="center", fontsize=_LABEL_PT)
    ax.set_xlabel(prefix, fontsize=_LABEL_PT)
    ax.set_ylabel(suffix, fontsize=_LABEL_PT)
    ax.set_title("(d) Encounter position", fontsize=_TITLE_PT)
    ax.tick_params(labelsize=_TICK_PT)

    fig.suptitle(f"{method_name} embedding - four views", fontsize=11,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s1_lasso_r2_bar(results,
                 show: bool = True, save: bool = False, 
                 fig_dir: Path = EXPERIMENTS_DIR / "figures",
                 fig_name: str = "1_lasso_r2"):
    lasso_pat = results.get("lasso_patient", {})
    lasso_enc = results.get("lasso_encounter", {})
    if not lasso_pat and not lasso_enc:
        return

    r2_pat = lasso_pat.get("r2_per_pc", {})
    r2_enc = lasso_enc.get("r2_per_pc", {})
    pc_keys = sorted(set(list(r2_pat.keys()) + list(r2_enc.keys())),
                     key=lambda k: int(k.replace("PC", "")))
    k = len(pc_keys)
    if k == 0:
        return

    x = np.arange(k)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7, k * 0.9), 4.5), dpi=300)

    vals_enc = [r2_enc.get(pc, 0.0) for pc in pc_keys]
    vals_pat = [r2_pat.get(pc, 0.0) for pc in pc_keys]

    if r2_enc:
        ax.bar(x - w / 2, vals_enc, w, color=VEC_COLORS.get("z_target", "#10B981"),
               alpha=0.85, label="Encounter-level")
    if r2_pat:
        ax.bar(x + w / 2, vals_pat, w, color=VEC_COLORS.get("z_enc", "#8B5CF6"),
               alpha=0.85, label="Patient-level")

    # Annotate means
    parts = []
    if vals_enc:
        mean_enc = np.mean(vals_enc)
        parts.append(f"Encounter mean R²={mean_enc:.3f}")
    if vals_pat:
        mean_pat = np.mean(vals_pat)
        parts.append(f"Patient mean R²={mean_pat:.3f}")
    if parts:
        ax.text(0.98, 0.98, "\n".join(parts), transform=ax.transAxes,
                ha="right", va="top", fontsize=_ANNOT_PT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CBD5E1", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(pc_keys, fontsize=_TICK_PT)
    ax.set_xlabel("PC axis", fontsize=_LABEL_PT)
    ax.set_ylabel("LASSO R²", fontsize=_LABEL_PT)
    ax.set_title("LASSO R² - encounter vs patient level", fontsize=_TITLE_PT)
    ax.legend(fontsize=_ANNOT_PT)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s1_sae_feature_cards(sae_data, top_n=12,
                          show: bool = True, save: bool = False, 
                          fig_dir: Path = EXPERIMENTS_DIR / "figures",
                          fig_name: str = "1_sae_feature_cards"):
    cards = sae_data.get("feature_cards", [])
    cards = sorted(cards, key=lambda c: c.get("activation_fraction", 0),
                   reverse=True)[:top_n]
    if not cards:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(cards) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 2.8), dpi=300)
    axes = np.atleast_2d(axes)

    cls_colors = {"clinical_match": "#10B981", "mixed": "#F59E0B",
                  "no_match": "#94A3B8"}

    for idx, card in enumerate(cards):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fi = card.get("feature_idx", "?")
        frac = card.get("activation_fraction", 0)

        # Classification
        max_or = 0.0
        for e in card.get("top_enriched_icd", []):
            max_or = max(max_or, e.get("odds_ratio", 0))
        for e in card.get("top_enriched_meds", []):
            max_or = max(max_or, e.get("odds_ratio", 0))
        if max_or > 3.0:
            cls_label = "clinical_match"
        elif max_or > 1.5:
            cls_label = "mixed"
        else:
            cls_label = "no_match"
        cls_col = cls_colors[cls_label]

        # Border
        border = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor=cls_col, facecolor="white")
        ax.add_patch(border)

        # Header
        ax.text(0.5, 0.92, f"Feature {fi}", ha="center", va="top",
                fontsize=8, fontweight="bold")
        ax.text(0.5, 0.82, f"activation {frac:.1%} | {cls_label}",
                ha="center", va="top", fontsize=6, color="#64748B")

        # Top ICD codes
        y = 0.72
        for e in card.get("top_enriched_icd", [])[:3]:
            code = e.get("code", "?")
            od = e.get("odds_ratio", 0)
            ax.text(0.08, y, f"ICD {code}  OR={od:.1f}",
                    fontsize=6, va="top", fontfamily="monospace")
            y -= 0.12

        # Top meds
        for e in card.get("top_enriched_meds", [])[:3]:
            med = e.get("med", "?")
            od = e.get("odds_ratio", 0)
            ax.text(0.08, y, f"Med {med}  OR={od:.1f}",
                    fontsize=6, va="top", fontfamily="monospace",
                    color="#6366F1")
            y -= 0.12

    # Turn off unused axes
    for idx in range(len(cards), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("SAE feature cards (z_enc) - top by activation fraction",
                 fontsize=_TITLE_PT, fontweight="bold", y=1.01)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s1_sae_cluster_crossref(sae_data,
                             show: bool = True, save: bool = False, 
                             fig_dir: Path = EXPERIMENTS_DIR / "figures",
                             fig_name: str = "1_sae_cluster_crossref"):
    """Approximate heatmap from feature-card cluster distributions.

    The full (n_features x n_clusters) mean-activation matrix is not
    persisted to JSON.  This reconstructs what it can from per-card
    ``dominant_cluster`` and ``activation_fraction`` fields, and renders
    a summary figure.
    """
    cards = sae_data.get("feature_cards", [])
    if not cards:
        return

    # Collect features that have cluster information
    feat_ids = []
    cluster_set: set[int] = set()
    for card in cards:
        dom_cl = card.get("dominant_cluster")
        if dom_cl is not None and dom_cl >= 0:
            feat_ids.append(card["feature_idx"])
            cluster_set.add(int(dom_cl))
    if not feat_ids or not cluster_set:
        print("  [skip] SAE-cluster crossref: insufficient cluster data")
        return

    cluster_ids = sorted(cluster_set)
    n_feat = len(feat_ids)
    n_cl = len(cluster_ids)
    cl_idx_map = {c: i for i, c in enumerate(cluster_ids)}

    # Build binary concentration matrix
    mat = np.zeros((n_feat, n_cl))
    for i, fi in enumerate(feat_ids):
        card = next(c for c in cards if c["feature_idx"] == fi)
        dom = card.get("dominant_cluster")
        if dom is not None and dom in cl_idx_map:
            frac = card.get("activation_fraction", 0.5)
            mat[i, cl_idx_map[dom]] = frac

    fig, ax = plt.subplots(
        figsize=(max(5, n_cl * 0.7), max(4, n_feat * 0.35)), dpi=300)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(n_cl))
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=_TICK_PT)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([f"F{fi}" for fi in feat_ids], fontsize=_TICK_PT)
    ax.set_xlabel("HDBSCAN cluster", fontsize=_LABEL_PT)
    ax.set_ylabel("SAE feature", fontsize=_LABEL_PT)
    ax.set_title("SAE feature x cluster (activation fraction)",
                 fontsize=_TITLE_PT)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Activation fraction")

    # Annotate concentrated vs spread
    n_single = sum(1 for row in mat if (row > 0).sum() == 1)
    n_multi = n_feat - n_single
    ax.text(1.0, -0.08, f"{n_single} concentrated / {n_multi} spread",
            transform=ax.transAxes, ha="right", fontsize=_ANNOT_PT,
            color="#64748B")

    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


# =============================================================================
# Predictor
# =============================================================================

def _s2_probe_comparison(results,
                         show: bool = True, save: bool = False, 
                         fig_dir: Path = EXPERIMENTS_DIR / "figures",
                         fig_name: str = "2_probe_comparison"):
    probing = results.get("probing", {})
    if not probing:
        return

    # Collect (label_key, vector_name) -> AUROC
    groups: dict[str, dict[str, tuple[float, float]]] = {}
    for label_key, vec_data in probing.items():
        for vec_name, metrics in vec_data.items():
            auroc = metrics.get("mean_auroc", 0)
            std = metrics.get("std_auroc", 0)
            groups.setdefault(label_key, {})[vec_name] = (auroc, std)

    if not groups:
        return

    label_keys = list(groups.keys())
    # Gather unique vector names in order
    vec_names = []
    for lk in label_keys:
        for vn in groups[lk]:
            if vn not in vec_names:
                vec_names.append(vn)

    n_labels = len(label_keys)
    n_vecs = len(vec_names)
    x = np.arange(n_labels)
    w = 0.8 / max(n_vecs, 1)

    _fallback = ["#3B82F6", "#10B981", "#EF4444", "#F59E0B", "#8B5CF6"]

    fig, ax = plt.subplots(figsize=(max(7, n_labels * 2.5), 5), dpi=300)
    for vi, vn in enumerate(vec_names):
        vals = [groups[lk].get(vn, (0, 0))[0] for lk in label_keys]
        errs = [groups[lk].get(vn, (0, 0))[1] for lk in label_keys]
        offset = (vi - (n_vecs - 1) / 2) * w
        color = VEC_COLORS.get(vn.replace("_pooled", ""),
                                _fallback[vi % len(_fallback)])
        ax.bar(x + offset, vals, w, yerr=errs, capsize=3,
               color=color, alpha=0.85, label=vn, edgecolor="white",
               linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(label_keys, fontsize=_TICK_PT)
    ax.set_ylabel("AUROC", fontsize=_LABEL_PT)
    ax.set_title("Probe comparison - z_pred vs z_target", fontsize=_TITLE_PT)
    ax.legend(fontsize=_ANNOT_PT, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s2_subspace_alignment(results,
                           show: bool = True, save: bool = False, 
                           fig_dir: Path = EXPERIMENTS_DIR / "figures",
                           fig_name: str = "2_subspace_alignment"):
    sa = results.get("subspace_alignment", {})
    if not sa:
        return

    angles_cos = sa.get("cosine_principal_angles", [])
    cka = results.get("cka")
    if not angles_cos:
        return

    k = len(angles_cos)
    x = np.arange(k)

    fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 4.5), dpi=300)
    colors = ["#10B981" if v > 0.8 else "#F59E0B" if v > 0.5 else "#EF4444"
              for v in angles_cos]
    ax.bar(x, angles_cos, color=colors, alpha=0.85, edgecolor="white",
           linewidth=0.4)
    ax.axhline(1.0, color="#10B981", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(0.0, color="#94A3B8", linestyle="-", linewidth=0.5)

    if cka is not None:
        ax.text(0.98, 0.98, f"CKA = {cka:.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CBD5E1", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i+1}" for i in range(k)], fontsize=_TICK_PT)
    ax.set_xlabel("Principal angle index", fontsize=_LABEL_PT)
    ax.set_ylabel("cos(principal angle)", fontsize=_LABEL_PT)
    ax.set_title("PCA subspace alignment - z_pred vs z_target",
                 fontsize=_TITLE_PT)
    ax.set_ylim(-0.05, 1.1)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s2_sae_vocab_overlap(results,
                          show: bool = True, save: bool = False, 
                          fig_dir: Path = EXPERIMENTS_DIR / "figures",
                          fig_name: str = "2_sae_vocab_overlap"):
    sc = results.get("sae_comparison")
    if not sc:
        return

    # We have aggregate stats; render as annotated summary bar
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    labels_bar = [r"z_pred $\rightarrow$ z_target", r"z_target $\rightarrow$ z_pred"]
    fracs = [sc.get("frac_shared_a", 0), sc.get("frac_shared_b", 0)]
    means = [sc.get("mean_best_match_a", 0), sc.get("mean_best_match_b", 0)]

    x = np.arange(len(labels_bar))
    ax.bar(x, fracs, color=[VEC_COLORS.get("z_pred", "#3B82F6"),
                             VEC_COLORS.get("z_target", "#10B981")],
           alpha=0.85, edgecolor="white", linewidth=0.4)

    for i in range(len(labels_bar)):
        ax.text(i, fracs[i] + 0.02, f"{fracs[i]:.1%}\nmean cos={means[i]:.3f}",
                ha="center", va="bottom", fontsize=_ANNOT_PT)

    ax.axhline(0.8, color="#EF4444", linestyle="--", linewidth=1,
               label="0.8 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=_TICK_PT)
    ax.set_ylabel("Fraction above threshold", fontsize=_LABEL_PT)
    ax.set_title("SAE dictionary overlap (cosine > 0.8)", fontsize=_TITLE_PT)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=_ANNOT_PT)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


# =============================================================================
# z_pred - z_target Decomposition
# =============================================================================

def _s3_eigenvalue_overlay(results, error_pca, s1_pca_enc,
                           show: bool = True, save: bool = False, 
                           fig_dir: Path = EXPERIMENTS_DIR / "figures",
                           fig_name: str = "3_eigenvalue_overlay"):
    if error_pca is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    eigs_err = error_pca["eigenvalues"]
    eigs_err_norm = eigs_err / eigs_err.sum()
    D_err = len(eigs_err)
    ax.plot(range(1, D_err + 1), eigs_err_norm, "o-", markersize=3,
            linewidth=1.5, color=VEC_COLORS.get("pred_error", "#EF4444"),
            label="pred_error")

    pca_err_r = results.get("pca_error", {})
    d_eff_err = pca_err_r.get("effective_dimensionality")

    if s1_pca_enc is not None:
        eigs_enc = s1_pca_enc["eigenvalues"]
        eigs_enc_norm = eigs_enc / eigs_enc.sum()
        D_enc = len(eigs_enc)
        ax.plot(range(1, D_enc + 1), eigs_enc_norm, "s-", markersize=3,
                linewidth=1.5, color=VEC_COLORS.get("z_enc", "#8B5CF6"),
                label="z_enc (Stage 1)")

    # Annotations
    parts = []
    if d_eff_err is not None:
        parts.append(f"pred_error d_eff = {d_eff_err:.1f}")
    # Check if error has sharper elbow
    if s1_pca_enc is not None and d_eff_err is not None:
        # Rough proxy: compare d_eff
        # (Stage 1 encounter d_eff would need to be passed or looked up)
        parts.append("(compare elbow sharpness visually)")
    if parts:
        ax.text(0.98, 0.98, "\n".join(parts), transform=ax.transAxes,
                ha="right", va="top", fontsize=_ANNOT_PT,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CBD5E1", alpha=0.9))

    ax.set_xlabel("Component index", fontsize=_LABEL_PT)
    ax.set_ylabel("Normalised eigenvalue (sum-to-1)", fontsize=_LABEL_PT)
    ax.set_title("Eigenvalue spectrum - z_enc vs pred_error", fontsize=_TITLE_PT)
    ax.legend(fontsize=_ANNOT_PT)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s3_magnitude_distribution(results,
                               show: bool = True, save: bool = False, 
                               fig_dir: Path = EXPERIMENTS_DIR / "figures",
                               fig_name: str = "3_magnitude_distribution"):
    mag = results.get("magnitude", {})
    mw_u = mag.get("mann_whitney_U")
    mw_p = mag.get("mann_whitney_p")
    mean_esc1 = mag.get("mean_norm_esc1")
    mean_esc0 = mag.get("mean_norm_esc0")

    if mean_esc1 is None or mean_esc0 is None:
        return

    # We don't have the raw norms in artifacts, so render a summary figure
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    bar_x = [0, 1]
    bar_vals = [mean_esc0, mean_esc1]
    bar_colors = [LABEL_COLORS[0], LABEL_COLORS[1]]
    bar_labels = [LABEL_NAMES[0], LABEL_NAMES[1]]

    ax.bar(bar_x, bar_vals, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(bar_labels, fontsize=_TICK_PT)
    ax.set_ylabel("Mean ||pred_error||", fontsize=_LABEL_PT)

    title = "Prediction error magnitude by escalation"
    if mw_p is not None:
        title += f"\n(Mann-Whitney U={mw_u:.0f}, p={mw_p:.2e})"
    ax.set_title(title, fontsize=_TITLE_PT)

    # Per-criterion panel
    per_crit = mag.get("per_criterion", {})
    if per_crit:
        txt_lines = []
        for crit, info in per_crit.items():
            p_val = info.get("p_value", 1.0)
            m_pos = info.get("mean_norm_pos", 0)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            txt_lines.append(f"{crit}: {m_pos:.3f} ({sig})")
        ax.text(0.98, 0.98, "\n".join(txt_lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=6, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#CBD5E1", alpha=0.9))

    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s3_error_umap(error_umap, labels, criteria_labels,
                   show: bool = True, save: bool = False, 
                   fig_dir: Path = EXPERIMENTS_DIR / "figures",
                   fig_name: str = "3_sae_vocab_overlap"):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    if criteria_labels is not None and labels is not None:
        # Gray background for escalation=0
        m0 = labels == 0
        if m0.any():
            ax.scatter(error_umap[m0, 0], error_umap[m0, 1], s=4, alpha=0.10,
                       color=LABEL_COLORS[0], label="no escalation",
                       rasterized=True)
        for crit in ESCALATION_CRITERIA:
            arr = criteria_labels.get(crit)
            if arr is None:
                continue
            m = arr == 1
            if m.any():
                ax.scatter(error_umap[m, 0], error_umap[m, 1], s=6,
                           alpha=0.45,
                           color=CRITERIA_COLORS.get(crit, "#64748B"),
                           label=crit, rasterized=True)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7,
                  markerscale=2)
    elif labels is not None:
        for lv, col in LABEL_COLORS.items():
            m = labels == lv
            if m.any():
                ax.scatter(error_umap[m, 0], error_umap[m, 1], s=4,
                           alpha=0.25, color=col,
                           label=LABEL_NAMES[lv], rasterized=True)
        ax.legend(fontsize=_ANNOT_PT, markerscale=2)
    else:
        ax.scatter(error_umap[:, 0], error_umap[:, 1], s=4, alpha=0.25,
                   color="#94A3B8", rasterized=True)
        ax.text(0.5, 0.02, "Labels not provided", transform=ax.transAxes,
                ha="center", fontsize=_ANNOT_PT, color="#64748B")

    ax.set_xlabel("UMAP-1", fontsize=_LABEL_PT)
    ax.set_ylabel("UMAP-2", fontsize=_LABEL_PT)
    ax.set_title("Prediction error UMAP - by escalation type",
                 fontsize=_TITLE_PT)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s3_sae_crossref_heatmap(crossref,
                             show: bool = True, save: bool = False, 
                             fig_dir: Path = EXPERIMENTS_DIR / "figures",
                             fig_name: str = "3_sae_crossref_heatmap"):
    """Summary figure from aggregate SAE cross-reference statistics.

    The full cosine-similarity matrix between decoder columns is not
    persisted; this figure renders the saved summary statistics.
    """
    frac_a = crossref.get("frac_shared_a", 0)
    frac_b = crossref.get("frac_shared_b", 0)
    mean_a = crossref.get("mean_best_match_a", 0)
    mean_b = crossref.get("mean_best_match_b", 0)
    n_a = crossref.get("n_features_a", 0)
    n_b = crossref.get("n_features_b", 0)
    n_unique_a = crossref.get("n_unique_a", 0)
    n_unique_b = crossref.get("n_unique_b", 0)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    labels_bar = [
        f"pred_error $\rightarrow$ z_enc\n({n_a} features)",
        fr"z_enc $\rightarrow$ pred_error\n({n_b} features)",
    ]
    fracs = [frac_a, frac_b]
    unique_counts = [n_unique_a, n_unique_b]
    means = [mean_a, mean_b]
    colors = [VEC_COLORS.get("pred_error", "#EF4444"),
              VEC_COLORS.get("z_enc", "#8B5CF6")]

    x = np.arange(len(labels_bar))
    ax.bar(x, fracs, color=colors, alpha=0.85, edgecolor="white",
           linewidth=0.4)

    for i in range(len(labels_bar)):
        ax.text(i, fracs[i] + 0.02,
                f"{fracs[i]:.1%}\nmean cos = {means[i]:.3f}\n"
                f"{unique_counts[i]} unique",
                ha="center", va="bottom", fontsize=_ANNOT_PT)

    ax.axhline(0.8, color="#64748B", linestyle="--", linewidth=1, alpha=0.5,
               label="0.8 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=_TICK_PT)
    ax.set_ylabel("Fraction aligned (cos > 0.8)", fontsize=_LABEL_PT)
    ax.set_title("SAE decoder direction alignment - pred_error vs z_enc",
                 fontsize=_TITLE_PT)

    # Key finding annotation
    pct = frac_a * 100
    ax.text(0.98, 0.98,
            f"{pct:.0f}% of error features\nalign with encoder features",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#CBD5E1", alpha=0.9))

    ax.set_ylim(0, max(fracs) * 1.4 + 0.1)
    ax.legend(fontsize=_ANNOT_PT)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s3_sae_feature_cards(sae_data, top_n=12,
                          show: bool = True, save: bool = False, 
                          fig_dir: Path = EXPERIMENTS_DIR / "figures",
                          fig_name: str = "3_sae_feature_cards"):
    cards = sae_data.get("feature_cards", [])
    cards = sorted(cards, key=lambda c: c.get("activation_fraction", 0),
                   reverse=True)[:top_n]
    if not cards:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(cards) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 2.8), dpi=300)
    axes = np.atleast_2d(axes)

    cls_colors = {"clinical_match": "#10B981", "mixed": "#F59E0B",
                  "no_match": "#94A3B8"}

    for idx, card in enumerate(cards):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fi = card.get("feature_idx", "?")
        frac = card.get("activation_fraction", 0)

        max_or = 0.0
        for e in card.get("top_enriched_icd", []):
            max_or = max(max_or, e.get("odds_ratio", 0))
        for e in card.get("top_enriched_meds", []):
            max_or = max(max_or, e.get("odds_ratio", 0))
        if max_or > 3.0:
            cls_label = "clinical_match"
        elif max_or > 1.5:
            cls_label = "mixed"
        else:
            cls_label = "no_match"
        cls_col = cls_colors[cls_label]

        border = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor=cls_col, facecolor="white")
        ax.add_patch(border)

        ax.text(0.5, 0.92, f"Feature {fi}", ha="center", va="top",
                fontsize=8, fontweight="bold")
        ax.text(0.5, 0.82, f"activation {frac:.1%} | {cls_label}",
                ha="center", va="top", fontsize=6, color="#64748B")

        y = 0.72
        for e in card.get("top_enriched_icd", [])[:3]:
            code = e.get("code", "?")
            od = e.get("odds_ratio", 0)
            ax.text(0.08, y, f"ICD {code}  OR={od:.1f}",
                    fontsize=6, va="top", fontfamily="monospace")
            y -= 0.12
        for e in card.get("top_enriched_meds", [])[:3]:
            med = e.get("med", "?")
            od = e.get("odds_ratio", 0)
            ax.text(0.08, y, f"Med {med}  OR={od:.1f}",
                    fontsize=6, va="top", fontfamily="monospace",
                    color="#6366F1")
            y -= 0.12

    for idx in range(len(cards), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("SAE feature cards (pred_error) - top by activation fraction",
                 fontsize=_TITLE_PT, fontweight="bold", y=1.01)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)

# =============================================================================
# Evaluation & Cross-Layer Synthesis
# =============================================================================

def _s4_info_flow_heatmap(results,
                          show: bool = True, save: bool = False, 
                          fig_dir: Path = EXPERIMENTS_DIR / "figures",
                          fig_name: str = "4_information_flow_hm"):
    info_flow = results.get("info_flow", {})
    if not info_flow:
        return

    # Rows = vectors, Columns = labels
    label_keys = list(info_flow.keys())
    vec_names: list[str] = []
    for lk in label_keys:
        for vn in info_flow[lk]:
            if vn not in vec_names:
                vec_names.append(vn)

    n_vecs = len(vec_names)
    n_labels = len(label_keys)
    if n_vecs == 0 or n_labels == 0:
        return

    mat = np.full((n_vecs, n_labels), np.nan)
    for j, lk in enumerate(label_keys):
        for i, vn in enumerate(vec_names):
            val = info_flow[lk].get(vn)
            if val is not None:
                mat[i, j] = val

    fig, ax = plt.subplots(
        figsize=(max(4, n_labels * 1.8), max(3, n_vecs * 0.7)), dpi=300)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.85,
                   interpolation="nearest")

    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(label_keys, fontsize=_TICK_PT)
    ax.set_yticks(range(n_vecs))
    ax.set_yticklabels(vec_names, fontsize=_TICK_PT)

    # Annotate each cell
    for i in range(n_vecs):
        for j in range(n_labels):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt_col = "white" if v < 0.5 or v > 0.8 else "#1E293B"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=txt_col)

    ax.set_title("Information flow - AUROC by vector x label",
                 fontsize=_TITLE_PT)
    fig.colorbar(im, ax=ax, shrink=0.7, label="AUROC")
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s4_mislabeling_gap(results, s1_results, s3_results,
                        show: bool = True, save: bool = False, 
                        fig_dir: Path = EXPERIMENTS_DIR / "figures",
                        fig_name: str = "4_mislabeling_gap"):
    gap = results.get("mislabeling_gap", {})
    if not gap:
        return

    # Three tiers: LASSO R², cluster unlabeled fraction, SAE no-match fraction
    tiers = ["LASSO R²", "Unlabeled\nclusters", "Unlabeled\nSAE features"]
    z_enc_vals = [None, None, None]
    error_vals = [None, None, None]

    # LASSO R²
    r2_pat = gap.get("lasso_r2_patient")
    if r2_pat is not None:
        z_enc_vals[0] = r2_pat
    # Error LASSO not computed separately; use inverse as proxy
    # (error has no separate LASSO in the pipeline)

    # Cluster unlabeled fraction
    n_unl = gap.get("n_unlabeled_clusters", 0)
    n_tot = gap.get("n_total_clusters", 0)
    if n_tot > 0:
        z_enc_vals[1] = n_unl / n_tot

    # SAE no-match fraction
    sae_frac = gap.get("sae_frac_no_match")
    if sae_frac is not None:
        z_enc_vals[2] = sae_frac
    err_sae_frac = gap.get("error_sae_frac_no_match")
    if err_sae_frac is not None:
        error_vals[2] = err_sae_frac

    # Only plot if we have at least some data
    has_data = any(v is not None for v in z_enc_vals + error_vals)
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(tiers))
    w = 0.35

    z_vals = [v if v is not None else 0 for v in z_enc_vals]
    e_vals = [v if v is not None else 0 for v in error_vals]
    z_mask = [v is not None for v in z_enc_vals]
    e_mask = [v is not None for v in error_vals]

    bars_z = ax.bar(x - w / 2, z_vals, w,
                    color=VEC_COLORS.get("z_enc", "#8B5CF6"), alpha=0.85,
                    label="z_enc", edgecolor="white", linewidth=0.4)
    bars_e = ax.bar(x + w / 2, e_vals, w,
                    color=VEC_COLORS.get("pred_error", "#EF4444"), alpha=0.85,
                    label="pred_error", edgecolor="white", linewidth=0.4)

    # Dim bars where data is missing
    for i, has in enumerate(z_mask):
        if not has:
            bars_z[i].set_alpha(0.15)
    for i, has in enumerate(e_mask):
        if not has:
            bars_e[i].set_alpha(0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=_TICK_PT)
    ax.set_ylabel("Fraction / R²", fontsize=_LABEL_PT)
    ax.set_title("Mislabeling gap - z_enc vs pred_error", fontsize=_TITLE_PT)
    ax.legend(fontsize=_ANNOT_PT)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s4_icd_block(results,
                  show: bool = True, save: bool = False, 
                  fig_dir: Path = EXPERIMENTS_DIR / "figures",
                  fig_name: str = "4_icd_block_reconstruction"):
    clinical = results.get("clinical", {})
    icd_pred = clinical.get("icd_blocks_z_pred", {})
    icd_tgt = clinical.get("icd_blocks_z_target", {})

    # Need per-chapter results; check if stored
    # The pipeline stores macro_auroc, n_chapters; per-chapter not in JSON.
    # Render macro comparison if available.
    macro_pred = icd_pred.get("macro_auroc")
    macro_tgt = icd_tgt.get("macro_auroc")
    if macro_pred is None and macro_tgt is None:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    labels_bar = []
    vals = []
    colors = []
    if macro_pred is not None:
        labels_bar.append(r"z_pred")
        vals.append(macro_pred)
        colors.append(VEC_COLORS.get("z_pred", "#3B82F6"))
    if macro_tgt is not None:
        labels_bar.append(r"z_target")
        vals.append(macro_tgt)
        colors.append(VEC_COLORS.get("z_target", "#10B981"))

    x = np.arange(len(labels_bar))
    ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white",
           linewidth=0.4)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=_TICK_PT)
    ax.set_ylabel("Macro AUROC", fontsize=_LABEL_PT)
    ax.set_title("ICD block reconstruction", fontsize=_TITLE_PT)
    ax.set_ylim(0, max(vals) * 1.2 + 0.05)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s4_escalation_type(results,
                        show: bool = True, save: bool = False, 
                        fig_dir: Path = EXPERIMENTS_DIR / "figures",
                        fig_name: str = "4_escalation_type_decomposition"):
    clinical = results.get("clinical", {})
    crit_probes = clinical.get("escalation_criteria", {})
    if not crit_probes:
        return

    criteria = list(crit_probes.keys())
    aurocs = [crit_probes[c].get("auroc", 0) for c in criteria]
    n_pos = [crit_probes[c].get("n_positive", 0) for c in criteria]
    colors = [CRITERIA_COLORS.get(c, "#64748B") for c in criteria]

    order = np.argsort(aurocs)[::-1]
    criteria = [criteria[i] for i in order]
    aurocs = [aurocs[i] for i in order]
    n_pos = [n_pos[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(7, len(criteria) * 1.2), 4.5),
                           dpi=300)
    x = np.arange(len(criteria))
    ax.bar(x, aurocs, color=colors, alpha=0.85, edgecolor="white",
           linewidth=0.4)

    for i in range(len(criteria)):
        ax.text(i, aurocs[i] + 0.01,
                f"{aurocs[i]:.3f}\n(n={n_pos[i]})",
                ha="center", va="bottom", fontsize=_ANNOT_PT)

    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.8,
               label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=_TICK_PT, rotation=25, ha="right")
    ax.set_ylabel("AUROC (z_pred)", fontsize=_LABEL_PT)
    ax.set_title("Escalation type decomposition - per-criterion AUROC",
                 fontsize=_TITLE_PT)
    ax.set_ylim(0, max(aurocs) * 1.15 + 0.05)
    ax.legend(fontsize=_ANNOT_PT)
    ax.tick_params(labelsize=_TICK_PT)
    fig.tight_layout()
    show_or_savefig(fig, show, save_path= fig_dir / fig_name if save else None)


def _s4_seed_stability(seed_stability,
                       show: bool = True, save: bool = False, 
                       fig_dir: Path = EXPERIMENTS_DIR / "figures",
                       fig_name: str = ""):
    seeds = seed_stability.get("seeds_found", [])
    if len(seeds) < 2:
        return

    n_figs = 0

    # (a) Probe AUROC dot plot
    # Not enough data in seed_stability.json for individual AUROC values
    # per seed - render ARI if available
    ari_keys = [k for k in seed_stability if k.startswith("ari_seed_")]
    if ari_keys:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ari_labels = [k.replace("ari_seed_", "seed ") for k in ari_keys]
        ari_vals = [seed_stability[k] for k in ari_keys]

        x = np.arange(len(ari_labels))
        colors_ari = ["#10B981" if v > 0.7 else "#F59E0B" if v > 0.3
                      else "#EF4444" for v in ari_vals]
        ax.bar(x, ari_vals, color=colors_ari, alpha=0.85,
               edgecolor="white", linewidth=0.4)

        for i, v in enumerate(ari_vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(ari_labels, fontsize=_TICK_PT)
        ax.set_ylabel("Adjusted Rand Index", fontsize=_LABEL_PT)
        ax.set_title("Cluster persistence across seeds", fontsize=_TITLE_PT)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(labelsize=_TICK_PT)
        fig.tight_layout()
        show_or_savefig(fig, show, fig_dir / "4_seed_cluster_persistence")
        n_figs += 1

    # (b) SAE stability
    sae_keys = [k for k in seed_stability
                if k.startswith("sae_") and isinstance(seed_stability[k], dict)]
    if sae_keys:
        fig, ax = plt.subplots(figsize=(max(6, len(sae_keys) * 2), 4),
                               dpi=300)

        sae_labels = [k.replace("sae_", "") for k in sae_keys]
        mean_cos = [seed_stability[k].get("mean_cosine", 0) for k in sae_keys]
        frac_stable = [seed_stability[k].get("frac_stable", 0)
                       for k in sae_keys]

        x = np.arange(len(sae_labels))
        w = 0.35
        ax.bar(x - w / 2, mean_cos, w, color="#3B82F6", alpha=0.85,
               label="Mean cosine", edgecolor="white", linewidth=0.4)
        ax.bar(x + w / 2, frac_stable, w, color="#10B981", alpha=0.85,
               label="Frac stable (>0.8)", edgecolor="white", linewidth=0.4)

        ax.axhline(0.8, color="#EF4444", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="0.8 threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(sae_labels, fontsize=_TICK_PT)
        ax.set_ylabel("Score", fontsize=_LABEL_PT)
        ax.set_title("SAE dictionary stability across seeds",
                     fontsize=_TITLE_PT)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=_ANNOT_PT)
        ax.tick_params(labelsize=_TICK_PT)
        fig.tight_layout()
        show_or_savefig(fig, show, fig_dir / "4_seed_sae_stability")
        n_figs += 1

    if n_figs == 0:
        print("  [skip] Seed stability: no plottable data")


# =============================================================================
# PCA
# =============================================================================
    
def vector_heatmap(
    vec: np.ndarray, labels: np.ndarray,
    show: bool = True, save: bool = False, save_path: Path | None = None,
    vector_name: str = "vec",
    label_name: str = "escalation",
):
    """Heatmap of latent vector dimensions (samples x dims), sorted by label."""
    sort_idx   = np.argsort(labels)
    vec_sorted = vec[sort_idx]
    n_neg      = int((labels[sort_idx] == 0).sum())
    vmax = float(np.percentile(np.abs(vec_sorted), 99)) or 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        vec_sorted.T, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.axvline(n_neg - 0.5, color="black", linewidth=2, linestyle="--",
                label=f"{label_name} boundary")
    ax.set_xlabel(f"Sample (sorted by {label_name})")
    ax.set_ylabel("Embedding dimension")
    ax.set_title(f"{vector_name} | left=negative, right=positive  (red= +, blue= -)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    show_or_savefig(fig, show, save_path=save_path if save else None)
        
        
def vector_dim_profile(
    vec: np.ndarray,
    show: bool = True, save: bool = False, save_path: Path | None = None,
    vector_name: str = "vec",
):
    """Per-dimension direction profile of a unit-normalised latent vector."""
    norms = np.linalg.norm(vec, axis=-1)
    safe_norms        = norms[:, np.newaxis].copy()
    safe_norms[safe_norms < 1e-10] = 1e-10
    vec_normed        = vec / safe_norms
    dim_mean          = vec_normed.mean(axis=0)
    dim_std           = vec_normed.std(axis=0)
    D                 = vec.shape[1]

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

    show_or_savefig(fig, show, save_path=save_path if save else None)
    

def plot_pca_scree_v_mp_upper(
    eigenvalues: np.ndarray, mp_upper: float, n_signal: int, D: int, eff_dim: float,
    show: bool = True, save: bool = False, save_path: Path | None = None,
    vector_name: str = "vec",
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

    show_or_savefig(fig, show, save_path=save_path if save else None)

    
def plot_cum_var(
    evr: np.ndarray, D: int, eff_dim: float,
    show: bool = True, save: bool = False, save_path: Path | None = None
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
    
    show_or_savefig(fig, show, save_path=save_path if save else None)
    
    
def plot_pca1_vs_pca2(
    pca_projection: np.ndarray, labels: np.ndarray, evr: np.ndarray,
    show: bool = True, save: bool = False, save_path: Path | None = None,
    label_name: str = "escalation",
):
    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl, col in [(0, "steelblue"), (1, "tomato")]:
        mask = labels == lbl
        if mask.any():
            tag = f"{label_name}" if lbl else f"no {label_name}"
            ax.scatter(pca_projection[mask, 0], pca_projection[mask, 1],
                        c=col, label=tag, alpha=0.65, s=20)
    
    ax.set_xlabel(f"PC1  ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2  ({evr[1]:.1%} var)")
    ax.set_title("PC1 vs PC2 (coloured by label)")
    ax.legend()
    fig.tight_layout()
    
    show_or_savefig(fig, show, save_path=save_path if save else None)


# =============================================================================
# SAE Analysis
# =============================================================================

def plot_sae_heatmap_top_features(
    sub: np.ndarray, 
    active_pe: np.ndarray, 
    active_ot: np.ndarray,
    show: bool = True, save: bool = False, save_path: Path | None = None
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
    ax.set_xlabel("prediction_error SAE feature")
    ax.set_ylabel("pred_error SAE feature")
    ax.set_title("Cross-Target Co-Activation\n"
                    "(what real dynamics does the model mispredict?)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalised co-activation")
    fig.tight_layout()
    show_or_savefig(fig, show, save_path=save_path if save else None)

# =============================================================================
# Linear Probing
# =============================================================================

def plot_probing_curve(
    sweep_results: dict,
    show: bool = True, save: bool = False,
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
    show_or_savefig(fig, show, save_path=save_path if save else None)


def plot_probing_metrics(
    sweep_results: dict,
    show: bool = True, save: bool = False,
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
    show_or_savefig(fig, show, save_path=save_path if save else None)


def plot_probing_comparison(
    jepa_sweep: dict,
    softmax_sweep: dict,
    show: bool = True, save: bool = False,
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
    show_or_savefig(fig, show, save_path=save_path if save else None)


def plot_f3x_stratified_probing(
    stratified_results: dict,
    show: bool = True, save: bool = False,
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
    show_or_savefig(fig, show, save_path=save_path if save else None)
    
    
    
# =========================================================================
# 2. Multi-panel patient profile figure
# =========================================================================

def build_patient_profile_figure(
    patient_idx: int,
    z_pred_all: np.ndarray,
    z_target_all: np.ndarray,
    decomposition: dict,
    labels: np.ndarray,
    pca_basis, # PCA
    metadata_summary: str | None = None,
    show: bool = True,
    save_path: Path | str | None = None,
) -> Figure:
    """Create a multi-panel thesis figure for one patient.

    Left panel shows z_pred vs z_target in shared PCA space with a P->T
    error arrow.  Right panel shows SAE feature decomposition bars.

    Parameters
    ----------
    patient_idx     : index into the (N, D) arrays
    z_pred_all      : (N, D) predictor outputs
    z_target_all    : (N, D) target encoder outputs
    decomposition   : output of decompose_patient()
    labels          : (N,) binary escalation labels (0/1)
    pca_basis       : sklearn PCA fitted on z_target
    metadata_summary : optional text strip for bottom panel
    show / save_path : forwarded to show_or_savefig

    Returns
    -------
    matplotlib Figure
    """
    has_metadata = metadata_summary is not None
    n_rows = 2 if has_metadata else 1
    height_ratios = [1, 0.08] if has_metadata else [1]

    fig = plt.figure(figsize=(14, 6.5 + (0.6 if has_metadata else 0)), dpi=300)
    gs = fig.add_gridspec(
        n_rows, 2,
        width_ratios=[1, 1.2],
        height_ratios=height_ratios,
        hspace=0.25, wspace=0.35,
    )

    # -- Left panel: P vs T in shared PCA space -------------------------
    ax_traj = fig.add_subplot(gs[0, 0])
    _draw_trajectory_panel(
        ax_traj, patient_idx,
        z_pred_all, z_target_all,
        pca_basis, labels,
    )

    # -- Right panel: SAE decomposition bars ----------------------------
    ax_bar = fig.add_subplot(gs[0, 1])
    _draw_decomposition_panel(ax_bar, decomposition)

    # -- Bottom strip: metadata -----------------------------------------
    if has_metadata:
        ax_meta = fig.add_subplot(gs[1, :])
        ax_meta.axis("off")
        ax_meta.text(
            0.5, 0.5, metadata_summary,
            transform=ax_meta.transAxes,
            ha="center", va="center",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC", ec="#CBD5E1", lw=0.8),
        )

    fig.suptitle(
        f"Patient {patient_idx}  -  prediction flow decomposition",
        fontsize=11, fontweight="bold", y=0.98,
    )

    show_or_savefig(fig, show=show, save_path=save_path, dpi=300, facecolor="white")
    return fig


# -- Left panel helpers -----------------------------------------------

def _draw_trajectory_panel(
    ax, patient_idx,
    z_pred_all, z_target_all,
    pca_basis, labels,
):
    """Project all patients onto top-2 PCA axes of z_target; draw P->T error arrow."""
    mean = pca_basis.mean_
    comps = pca_basis.components_[:2]   # (2, D)

    def _proj(z):
        return (z - mean) @ comps.T     # (N, 2)

    pred_2d = _proj(z_pred_all)
    tgt_2d  = _proj(z_target_all)

    # Population scatter (z_target, colored by escalation label)
    for lab_val, color in LABEL_COLORS.items():
        mask = labels == lab_val
        lbl = "escalation" if lab_val else "no escalation"
        ax.scatter(
            tgt_2d[mask, 0], tgt_2d[mask, 1],
            s=4, alpha=0.12, color=color, label=lbl, rasterized=True,
        )

    # Patient: P and T points with error arrow
    p = pred_2d[patient_idx]
    t = tgt_2d[patient_idx]

    # P -> T  (prediction error)
    ax.annotate("", xy=t, xytext=p,
                arrowprops=dict(arrowstyle="-|>", color=VEC_COLORS["pred_error"],
                                lw=2.0, linestyle="--", mutation_scale=14))

    # Vertex markers
    for pt, marker, lbl, color in [
        (p, "s", "P", VEC_COLORS["z_pred"]),
        (t, "^", "T", VEC_COLORS["z_target"]),
    ]:
        ax.plot(pt[0], pt[1], marker, color=color, markersize=7, zorder=6)
        ax.annotate(lbl, pt, textcoords="offset points", xytext=(6, 6),
                    fontsize=8, fontweight="bold", zorder=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=VEC_COLORS["pred_error"], lw=2, ls="--",
               label=ARROW_LABELS["pred_error"]),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower left",
              framealpha=0.85, edgecolor="#CBD5E1")

    ax.set_xlabel("z_target PC1", fontsize=9)
    ax.set_ylabel("z_target PC2", fontsize=9)
    ax.set_title("Prediction vs target (shared PCA basis)", fontsize=10)
    ax.tick_params(labelsize=7)


# -- Right panel helpers ----------------------------------------------

def _draw_decomposition_panel(ax, decomposition: dict):
    """Grouped horizontal bar chart of SAE feature magnitudes."""
    vector_names = list(decomposition.keys())

    # Collect the union of all active feature indices across vectors
    all_features: dict[int, str | None] = {}
    for name in vector_names:
        for entry in decomposition.get(name, []):
            fi = entry["feature_idx"]
            if fi not in all_features or all_features[fi] is None:
                all_features[fi] = entry["label"]

    if not all_features:
        ax.text(0.5, 0.5, "No SAE decomposition available",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return

    # Sort features by total magnitude across vectors (most important first)
    def _total_mag(fi):
        total = 0.0
        for name in vector_names:
            for e in decomposition.get(name, []):
                if e["feature_idx"] == fi:
                    total += abs(e["magnitude"])
        return total

    sorted_features = sorted(all_features.keys(), key=_total_mag, reverse=True)

    n_feat = len(sorted_features)
    n_vecs = len(vector_names)
    bar_height = 0.22
    y_positions = np.arange(n_feat)

    # Build magnitude lookup: (vector_name, feature_idx) -> magnitude
    mag_lookup: dict[tuple[str, int], float] = {}
    for name in vector_names:
        for entry in decomposition.get(name, []):
            mag_lookup[(name, entry["feature_idx"])] = entry["magnitude"]

    # Color fallback for vectors not in COLORS
    _fallback = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]
    vec_colors = {
        name: VEC_COLORS.get(name, _fallback[i % len(_fallback)])
        for i, name in enumerate(vector_names)
    }

    for vi, name in enumerate(vector_names):
        mags = [mag_lookup.get((name, fi), 0.0) for fi in sorted_features]
        offset = (vi - (n_vecs - 1) / 2) * bar_height
        ax.barh(
            y_positions + offset, mags,
            height=bar_height, color=vec_colors[name], alpha=0.85,
            label=name, edgecolor="white", linewidth=0.4,
        )

    # Y-axis labels: feature index + clinical label
    ylabels = []
    for fi in sorted_features:
        label = all_features[fi]
        if label:
            ylabels.append(f"F{fi}  {label}")
        else:
            ylabels.append(f"F{fi}  (unlabeled)")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Activation magnitude", fontsize=9)
    ax.set_title("SAE feature decomposition", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85, edgecolor="#CBD5E1")
    ax.tick_params(axis="x", labelsize=7)
    ax.axvline(0, color="#94A3B8", lw=0.5, zorder=0)

    # Mark unlabeled features with a subtle indicator
    for i, fi in enumerate(sorted_features):
        if all_features[fi] is None:
            ax.annotate("?", xy=(0, i), fontsize=6, color="#94A3B8",
                        ha="right", va="center",
                        xytext=(-4, 0), textcoords="offset points")