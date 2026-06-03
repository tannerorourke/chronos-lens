#!/usr/bin/env python3
"""
Layer-wise linear probing sweep for signal localization.

Loads a trained JEPA checkpoint, extracts the recency encounter representation
(z_enc[k-1]) at every transformer encoder layer (plus the final z_enc) via
forward hooks, then runs a stratified-CV logistic-regression probe at each layer
against a binary clinical label. Reports where prediction signal emerges through
the encoder.

This is an *analysis* entry point (runs locally over a run's frozen artifacts).
It is the generic counterpart to the per-analysis probes scattered across
diagnostic.py / analyze_*.py: a single, reusable "probe every layer" sweep.

Usage
-----
  python -m scripts.probe --exp ema_42_v01 --checkpoint-name checkpoint_100.pt
  python -m scripts.probe --exp ema_42_v01 --checkpoint-name checkpoint_100.pt --label label_30d
  python -m scripts.probe --exp ema_42_v01 --checkpoint-name checkpoint_100.pt --output /tmp/probing.json

Notes
-----
* The sweep relies on model.transformer_layers and slices the recency encounter
  from the per-layer encoder outputs; the supervised encoder exposes the same
  per-layer geometry as the JEPA context encoder.
* Labels are aligned causally to each sample's target encounter via the
  ``mask_pos`` carried through extraction, matching the recency representations
  the sweep produces.
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import torch

from src.infra.loaders import load_scaffolding
from src.analysis.probing import extract_layer_representations, run_probing_sweep
from src.infra.labels import load_label
from src.utils.io import load_sequences_dict, save_json, RUNS_DIR
from src.utils.seed import set_global_seed, load_exp_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise probing sweep for signal localization")
    parser.add_argument("--exp", type=str, required=True,
                        help="run-id from the runs/ directory")
    parser.add_argument("--checkpoint-name", type=str, required=True,
                        help="file name of .pt checkpoint in the run's checkpoints/ directory")
    parser.add_argument("--label", type=str, default="label_30d",
                        help="binary label key to probe (default: label_30d)")
    parser.add_argument("--cv", type=int, default=5,
                        help="number of stratified CV folds (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for results JSON (default: <run_dir>/results/probing.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed at entry from the run's frozen config (load_scaffolding re-applies the
    # same seed; setting it here makes the run's determinism explicit).
    set_global_seed(load_exp_seed(RUNS_DIR / args.exp))

    # --- Rebuild model + loader from the run's frozen artifacts (inference only).
    model, loader, run_dir, (_ckpt, _config), (_ds, _is_supervised, _label_key, _vocab) = \
        load_scaffolding(args.checkpoint_name, args.exp, device)

    # --- Per-layer representations (forward hooks) -> labels -> sweep.
    print("Extracting per-layer representations...")
    reps = extract_layer_representations(model, loader, device)
    subject_ids = reps["subject_ids"]
    mask_pos = reps["mask_pos"]

    # Causal per-encounter label at each sample's target encounter (mask_pos),
    # aligned to the context-pooled rows. Patient-level fallbacks (no mask_pos)
    # are degenerate for these keys - the last encounter is structurally
    # negative for label_30d - so the mask position is required.
    patients_dict = load_sequences_dict()
    labels = load_label(patients_dict, subject_ids, args.label, mask_pos=mask_pos)

    n_pos = int(labels.sum())
    print(f"Probing {reps['n_layers']} layers + final on '{args.label}' "
          f"(N={len(labels)}, positives={n_pos}):")
    if n_pos < args.cv or (len(labels) - n_pos) < args.cv:
        raise SystemExit(
            f"label '{args.label}' has too few of one class "
            f"(positives={n_pos}, negatives={len(labels) - n_pos}) for {args.cv}-fold "
            f"CV. Try a different --label, a larger run, or fewer --cv folds.")
    result = run_probing_sweep(reps, labels, n_splits=args.cv)

    print(f"\n  Best layer: {result['best_layer']} (AUROC={result['best_auroc']:.4f})")
    print(f"  {result['interpretation']}\n")

    output = {
        "exp": args.exp,
        "checkpoint": args.checkpoint_name,
        "label": args.label,
        "cv": args.cv,
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        **result,
    }

    output_path = Path(args.output) if args.output else run_dir / "results" / "probing.json"
    save_json(output, output_path)


if __name__ == "__main__":
    main()
