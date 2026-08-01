#!/usr/bin/env python3
"""
Layer-wise linear probing sweep: where in the encoder does label signal emerge?

Forward-hooks the recency encounter representation z_enc[k-1] out of every transformer
layer (plus the final z_enc), then runs a stratified-CV logistic probe at each one
against a binary label. Labels are aligned causally to each sample's target encounter
via mask_pos, matching the representations the sweep produces. The supervised encoder
exposes the same per-layer geometry as the JEPA context encoder.

Runs local-first over a run's frozen artifacts; embeddings are never pulled from S3,
and only a genuinely-missing checkpoint or vocab is fetched.
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import torch

import logging
logger = logging.getLogger(__name__)

from src.infra.inference import load_scaffolding
from src.analysis.probing import extract_layer_representations, run_probing_sweep
from src.infra.labels import load_label
from src.utils.io import load_sequences_dict, save_json, resolve_run_dir, data_dir
from src.utils.system import set_global_seed, load_exp_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-wise probing sweep for signal localization")
    parser.add_argument("--exp", type=str, required=True,
                        help="run-id from the runs/ directory")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="file name of .pt checkpoint in the run's data/checkpoints/ directory")
    
    parser.add_argument("--label", type=str, default="label_30d",
                        help="binary label key to probe (default: label_30d)")
    parser.add_argument("--cv", type=int, default=5,
                        help="number of stratified CV folds (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for results JSON (default: <run-dir>/probing.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_id = args.exp
    run_dir = resolve_run_dir(args.exp)

    # Seed at entry from the run's frozen config (load_scaffolding re-applies the
    # same seed; setting it here makes the run's determinism explicit).
    set_global_seed(load_exp_seed(data_dir(run_dir)))

    # --- Rebuild model + loader from the run's frozen artifacts (inference only).
    model, loader, (_, _), (_, _), (_, _) = \
        load_scaffolding(run_id, args.ckpt, device)

    # --- Per-layer representations (forward hooks) -> labels -> sweep.
    logger.info("Extracting per-layer representations...")
    reps = extract_layer_representations(model, loader, device)
    subject_ids = reps["subject_ids"]
    mask_pos = reps["mask_pos"]

    # Causal per-encounter label at each sample's target encounter (mask_pos),
    # aligned to the recency rows. Patient-level fallbacks (no mask_pos)
    # are degenerate for these keys - the last encounter is structurally
    # negative for label_30d - so the mask position is required.
    patients_dict = load_sequences_dict()
    labels = load_label(patients_dict, subject_ids, args.label, mask_pos=mask_pos)

    n_pos = int(labels.sum())
    logger.debug(f"Probing {reps['n_layers']} layers + final on '{args.label}' "
                 f"(N={len(labels)}, positives={n_pos}):")
    if n_pos < args.cv or (len(labels) - n_pos) < args.cv:
        raise SystemExit(
            f"label '{args.label}' has too few of one class "
            f"(positives={n_pos}, negatives={len(labels) - n_pos}) for {args.cv}-fold "
            f"CV. Try a different --label, a larger run, or fewer --cv folds.")
    result = run_probing_sweep(reps, labels, n_splits=args.cv)

    logger.info(f"\n  Best layer: {result['best_layer']} (AUROC={result['best_auroc']:.4f})")
    print(f"  {result['interpretation']}\n")

    output = {
        "exp": args.exp,
        "checkpoint": args.ckpt,
        "label": args.label,
        "cv": args.cv,
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        **result,
    }

    output_path = Path(args.output) if args.output else run_dir / "probing.json"
    save_json(output, output_path)


if __name__ == "__main__":
    main()
