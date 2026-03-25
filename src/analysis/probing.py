"""
Linear probing utilities — layer-wise signal localization and
softmax-free vs. softmax comparison for the thesis MI techniques table.

Purpose 1: Extract intermediate representations from each transformer
           encoder layer to probe WHERE the readmission signal emerges.

Purpose 2: Compare linear separability of JEPA (no softmax) vs.
           supervised (softmax classification head) representations.

Functions
---------
  extract_layer_representations : forward-hook extraction at every layer
  compare_softmax_baseline      : tabular comparison of probing results
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

def extract_layer_representations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Extract mean-pooled representations at every encoder layer.

    Uses forward hooks on each TransformerEncoderLayer in the context
    encoder to capture intermediate outputs, then mean-pools over
    non-padded positions (matching the encoder's own pooling logic).

    Parameters
    ----------
    model  : JEPA model (eval mode, frozen)
    loader : DataLoader yielding batch dicts with ctx_tokens, ctx_tok_mask,
             ctx_pad_mask, labels, subject_ids
    device : torch device

    Returns
    -------
    dict with keys:
        "layer_{i}"     : (N, embed_dim) ndarray for each encoder layer
        "final"         : (N, embed_dim) ndarray — z_context (post-norm, pooled)
        "labels"        : (N,) int ndarray
        "subject_ids"   : (N,) str ndarray
        "n_layers"      : int
    """
    model.eval()
    encoder = model.context_encoder
    n_layers = len(encoder.layers)

    # Storage for hook captures (per-batch, per-layer)
    hook_outputs: dict[int, list] = {i: [] for i in range(n_layers)}
    hooks = []

    def _make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output: (B, T, D) — post-residual sequence tensor
            hook_outputs[layer_idx].append(output.detach())
        return hook_fn

    # Register hooks
    for i, layer in enumerate(encoder.layers):
        h = layer.register_forward_hook(_make_hook(i))
        hooks.append(h)

    # Collect
    all_final = []
    all_labels = []
    all_sids = []
    all_pad_masks = []

    try:
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                z_ctx, _, _ = model(batch)
                all_final.append(z_ctx.cpu())
                all_labels.append(batch["labels"].cpu())
                all_sids.extend(batch["subject_ids"])
                all_pad_masks.append(batch["ctx_pad_mask"].cpu())
    finally:
        for h in hooks:
            h.remove()

    # Mean-pool each layer's sequence outputs using padding masks
    result = {}
    for i in range(n_layers):
        layer_seqs = torch.cat(hook_outputs[i], dim=0)  # (N, T, D)
        pad_masks = torch.cat(all_pad_masks, dim=0)     # (N, T)
        valid = (~pad_masks).float().unsqueeze(-1)       # (N, T, 1)
        pooled = (layer_seqs * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        result[f"layer_{i}"] = pooled.numpy()

    result["final"] = torch.cat(all_final, dim=0).numpy()
    result["labels"] = torch.cat(all_labels, dim=0).numpy().astype(int)
    result["subject_ids"] = np.array(all_sids, dtype=str)
    result["n_layers"] = n_layers

    return result



def compare_softmax_baseline(
    jepa_results: dict,
    softmax_results: dict,
) -> dict:
    """Compare probing results from JEPA vs. softmax-trained model.

    Parameters
    ----------
    jepa_results    : output from run_probing_sweep for JEPA model
    softmax_results : output from run_probing_sweep for softmax model

    Returns
    -------
    dict with per-layer comparison and summary interpretation
    """
    layers = sorted(jepa_results["per_layer"].keys())
    comparison = []

    for layer_key in layers:
        j = jepa_results["per_layer"][layer_key]
        s = softmax_results["per_layer"][layer_key]
        comparison.append({
            "layer": layer_key,
            "jepa_auc": j["mean_auc"],
            "softmax_auc": s["mean_auc"],
            "delta_auc": j["mean_auc"] - s["mean_auc"],
            "jepa_f1": j["mean_f1"],
            "softmax_f1": s["mean_f1"],
        })

    # Summary
    jepa_final = jepa_results["per_layer"].get("final", {})
    soft_final = softmax_results["per_layer"].get("final", {})
    jepa_auc = jepa_final.get("mean_auc", 0)
    soft_auc = soft_final.get("mean_auc", 0)

    if jepa_auc >= soft_auc - 0.02:
        interpretation = (
            f"JEPA final-layer AUC ({jepa_auc:.3f}) matches or exceeds "
            f"softmax ({soft_auc:.3f}). The latent space organizes for the "
            f"task geometry naturally without softmax supervision, supporting "
            f"the thesis claim that geometric analysis is meaningful."
        )
    else:
        interpretation = (
            f"Softmax final-layer AUC ({soft_auc:.3f}) exceeds JEPA "
            f"({jepa_auc:.3f}). The geometric structure in the JEPA "
            f"displacement field is genuinely different from supervised "
            f"representations. SAE features may capture structure that "
            f"softmax training would optimize away."
        )

    return {
        "comparison": comparison,
        "interpretation": interpretation,
    }
