"""
Per-patient interpretability figures for JEPA prediction analysis.

Decomposes a single patient's prediction into sparse SAE features and
visualises the prediction-vs-target geometry alongside the sparse
decomposition.

Designed for thesis case-study figures (publication-ready at 300 dpi).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from src.models.sae import SparseAutoencoder
from src.analysis.plotting import show_or_savefig
from src.utils.seed import SEED


# Consistent palette across all figures
COLORS = {
    "z_pred":     "#3B82F6",   # blue  - model prediction
    "z_target":   "#10B981",   # green - ground truth
    "pred_error": "#EF4444",   # red   - prediction error
}
ARROW_LABELS = {
    "pred_error": r"$P \to T$ (error)",
}
LABEL_COLORS = {0: "#94A3B8", 1: "#F97316"}   # neg / pos escalation


# =========================================================================
# 1. SAE decomposition for a single patient
# =========================================================================

def decompose_patient(
    patient_idx: int,
    vectors_dict: dict[str, np.ndarray],
    sae_models_dict: dict[str, SparseAutoencoder],
    feature_cards_dict: dict[str, list[dict]],
    top_k: int = 8,
) -> dict:
    """Decompose one patient's vectors through their respective SAEs.

    Parameters
    ----------
    patient_idx    : row index into the (N, D) arrays
    vectors_dict   : {name: (N, D)} arrays (e.g. pred_error, z_pred, z_target)
    sae_models_dict    : {name: SparseAutoencoder} - trained, eval-mode
    feature_cards_dict : {name: [card, ...]} - from inspect_sae_features
    top_k          : max active features to return per vector

    Returns
    -------
    dict keyed by vector name, each value a list of dicts:
        {feature_idx, magnitude, label, decoder_direction}
    sorted descending by magnitude.
    """
    def _card_lookup(cards: list[dict]) -> dict[int, str | None]:
        lookup: dict[int, str | None] = {}
        for card in cards:
            idx = card["feature_idx"]
            label = None
            enriched = card.get("top_enriched_icd", [])
            if enriched:
                label = enriched[0].get("code")
            if label is None:
                enriched_meds = card.get("top_enriched_meds", [])
                if enriched_meds:
                    label = enriched_meds[0].get("med")
            lookup[idx] = label
        return lookup

    result = {}
    for name, sae in sae_models_dict.items():
        if name not in vectors_dict:
            continue
        vec = vectors_dict[name][patient_idx]
        cards = feature_cards_dict.get(name, [])
        card_lookup = _card_lookup(cards)

        device = next(sae.parameters()).device
        with torch.no_grad():
            x = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
            activations = sae.encode(x).squeeze(0).cpu().numpy()

        decoder_weight = sae.decoder.weight.detach().cpu().numpy()  # (embed_dim, n_features)

        active_idx = np.nonzero(activations)[0]
        magnitudes = activations[active_idx]
        order = np.argsort(-np.abs(magnitudes))[:top_k]
        active_idx = active_idx[order]
        magnitudes = magnitudes[order]

        features = []
        for fi, mag in zip(active_idx, magnitudes):
            features.append({
                "feature_idx": int(fi),
                "magnitude": float(mag),
                "label": card_lookup.get(int(fi)),
                "decoder_direction": decoder_weight[:, int(fi)].tolist(),
            })
        result[name] = features

    return result


# =========================================================================
# 2. Multi-panel patient profile figure
# =========================================================================

def build_patient_profile_figure(
    patient_idx: int,
    z_pred_all: np.ndarray,
    z_target_all: np.ndarray,
    decomposition: dict,
    labels: np.ndarray,
    pca_basis: PCA,
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
                arrowprops=dict(arrowstyle="-|>", color=COLORS["pred_error"],
                                lw=2.0, linestyle="--", mutation_scale=14))

    # Vertex markers
    for pt, marker, lbl, color in [
        (p, "s", "P", COLORS["z_pred"]),
        (t, "^", "T", COLORS["z_target"]),
    ]:
        ax.plot(pt[0], pt[1], marker, color=color, markersize=7, zorder=6)
        ax.annotate(lbl, pt, textcoords="offset points", xytext=(6, 6),
                    fontsize=8, fontweight="bold", zorder=7)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLORS["pred_error"], lw=2, ls="--",
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
        name: COLORS.get(name, _fallback[i % len(_fallback)])
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


# =========================================================================
# 3. Patient selection utility
# =========================================================================

def select_interesting_patients(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    labels: np.ndarray,
    n: int = 5,
    seed: int = SEED,
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
    rng = np.random.default_rng(seed)
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
