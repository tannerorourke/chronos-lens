"""
Linear probing utilities - layer-wise signal localization and
softmax-free vs. softmax comparison for the thesis MI techniques table.

Purpose 1: Extract intermediate representations from each transformer
           encoder layer to probe WHERE the readmission signal emerges.
Purpose 2: Compare linear separability of JEPA (no softmax) vs.
           supervised (softmax classification head) representations.
Purpose 3: Probe all 6 JEPA latent objects against readmission labels
           with patient-level pooling to prevent data leakage.

Functions
---------
  extract_layer_representations : forward-hook extraction at every layer
  compare_softmax_baseline      : tabular comparison of probing results
  probe_latent_objects          : probe all 6 latent vectors from .npz
  probe_by_layer                : probe each encoder layer + final output
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.supervised_transformer import SupervisedTransformer
from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.jepa_ema import JEPA_EMA
from src.analysis.eval_tasks import evaluate_readmission

LATENT_OBJECTS = [
    "z_context", "z_pred", "z_target",
    "delta", "pred_error", "observed_traj",
]

def extract_layer_representations(
    model: JEPA_EMA | JEPAStopGrad | SupervisedTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    layers = model.transformer_layers
    n_layers = len(layers)

    # Storage for hook captures (per-batch, per-layer)
    hook_outputs: dict[int, list] = {i: [] for i in range(n_layers)}
    hooks = []

    def _make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output: (B, T, D) - post-residual sequence tensor
            hook_outputs[layer_idx].append(output.detach())
        return hook_fn

    # Register hooks
    for i, layer in enumerate(layers):
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


# =============================================================================
# Patient-level pooling
# =============================================================================

def _pool_to_patients(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids, inverse = np.unique(subject_ids, return_inverse=True)
    n_patients = len(unique_ids)
    D = embeddings.shape[1]

    patient_embs = np.zeros((n_patients, D), dtype=embeddings.dtype)
    counts = np.zeros(n_patients, dtype=np.int64)
    patient_labels = np.empty(n_patients, dtype=labels.dtype)

    for i, inv in enumerate(inverse):
        patient_embs[inv] += embeddings[i]
        counts[inv] += 1
        patient_labels[inv] = labels[i]

    patient_embs /= counts[:, None]
    return patient_embs, patient_labels


# =============================================================================
# Probe all 6 JEPA latent objects from .npz
# =============================================================================

def probe_latent_objects(
    embeddings_path: Path | str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    npz = np.load(embeddings_path, allow_pickle=True)
    labels = npz["labels"]
    subject_ids = npz["subject_ids"]

    z_context = npz["z_context"]
    z_pred = npz["z_pred"]
    z_target = npz["z_target"]

    vectors = {
        "z_context":     z_context,
        "z_pred":        z_pred,
        "z_target":      z_target,
        "delta":         npz["delta"] if "delta" in npz else z_pred - z_context,
        "pred_error":    npz["pred_error"] if "pred_error" in npz else z_pred - z_target,
        "observed_traj": npz["observed_traj"] if "observed_traj" in npz else z_target - z_context,
    }

    results = {}
    for name, emb in vectors.items():
        if np.all(emb == 0):
            continue
        emb_pat, labels_pat = _pool_to_patients(emb, subject_ids, labels)
        results[name] = evaluate_readmission(
            emb_pat, labels_pat, n_splits=n_splits, seed=seed)

    return results


def probe_by_layer(
    model: JEPA_EMA | JEPAStopGrad | SupervisedTransformer,
    loader: DataLoader,
    device: torch.device,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    reps = extract_layer_representations(model, loader, device)
    subject_ids = reps["subject_ids"]
    n_layers = reps["n_layers"]

    results = {}
    for key in [f"layer_{i}" for i in range(n_layers)] + ["final"]:
        emb = reps[key]
        emb_pat, labels_pat = _pool_to_patients(emb, subject_ids, labels)
        results[key] = evaluate_readmission(
            emb_pat, labels_pat, n_splits=n_splits, seed=seed)

    return results
