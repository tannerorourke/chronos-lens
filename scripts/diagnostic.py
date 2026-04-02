#!/usr/bin/env python3
"""
Unified evaluation script for all trained models (stopgrad, ema, supervised).

Loads a checkpoint, extracts embeddings, and runs downstream probing tasks
on each available latent vector.  Outputs a structured JSON and a summary
table to stdout.

Tasks
-----
  readmit_30d      : 30-day F-code readmission (patient-level, pooled vectors)
  escalation       : per-encounter escalation binary probe
  icd_block        : ICD-10 chapter prediction for the masked encounter
  escalation_type  : per-criterion escalation probes (6 binary probes)

Usage
-----
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --tasks readmit_30d,escalation
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --split temporal
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --eval-subset fcode
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.training.utils.checkpoint import load_model_notrain
from src.training.utils.datasets import (
    MimicDataset, collate_fn,
    SupervisedDataset, supervised_collate_fn)
from src.analysis.eval_infra import (
    extract_jepa_embeddings, extract_supervised_embeddings,
    load_label, load_escalation_labels, compute_escalation_criterions,
    compute_subset_mask, 
    compute_derived_vectors, pool_to_patients,
    format_binary_result, 
    compute_temporal_split, 
    extract_icd_block_targets)
from src.analysis.probing import (
    evaluate_binary_probe, evaluate_binary_probe_temporal, 
    probe_icd_blocks, probe_icd_blocks_temporal)
from src.utils.constants import ESCALATION_CRITERIA, MODEL_PRED_VECS, ALL_TASKS
from src.utils.io import EXPERIMENTS_DIR, load_json, load_sequences, load_sequences_dict, DATA_DIR
from src.utils.seed import SEED, load_exp_seed, set_global_seed

# =============================================================================
# Core evaluation
# =============================================================================

def run_tasks(
    vecs: dict,
    patients_dict: dict[str, dict],
    sequences_path: Path,
    tasks: list[str],
    is_supervised: bool,
    split: str = "random",
    temporal_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Run evaluation tasks on all available latent vectors.

    Patient-level tasks (readmit_30d) use vectors pooled across mask
    positions per patient to prevent data leakage.  Encounter-level tasks
    (escalation, icd_block, escalation_type) use sample-level vectors.
    """
    subject_ids = vecs["subject_ids"]
    mask_pos = vecs.get("mask_pos")
    results: dict = {}

    # -- Build encounter-level vector dict --
    if is_supervised:
        enc_vectors = {"z_enc_pooled": vecs["z_enc_pooled"]}
    else:
        enc_vectors = {
            name: vecs[name] for name in MODEL_PRED_VECS
            if vecs.get(name) is not None
        }

    # -- Build patient-level vector dict --
    if is_supervised:
        pat_vectors = {"z_enc_pooled": vecs["z_enc_pooled"]}
        pat_ids = subject_ids  # one sample per patient already
    else:
        pat_vectors, pat_ids = pool_to_patients(enc_vectors, subject_ids)

    # Patient-level temporal masks (computed separately from sample-level)
    pat_temporal: tuple[np.ndarray, np.ndarray] | None = None
    if split == "temporal":
        pat_train, pat_test, _ = compute_temporal_split(sequences_path, pat_ids)
        pat_temporal = (pat_train, pat_test)

    # -- readmit_30d (patient-level) --
    if "readmit_30d" in tasks:
        labels_30d = load_label(patients_dict, pat_ids, "label_30d")
        task_results: dict = {}
        for vec_name, emb in pat_vectors.items():
            if split == "temporal":
                pat_train, pat_test, _ = compute_temporal_split(sequences_path, pat_ids)
                pat_temporal = (pat_train, pat_test)
                res = evaluate_binary_probe_temporal(emb, labels_30d, 
                                                     train_mask=pat_temporal[0], 
                                                     test_mask=pat_temporal[1])
            else:
                res = evaluate_binary_probe(emb, labels_30d)
                
            task_results[vec_name] = format_binary_result(res)
        results["readmit_30d"] = task_results

    # -- escalation (encounter-level) --
    if "escalation" in tasks:
        if is_supervised or mask_pos is None:
            print("  [escalation] skipped - requires JEPA model with mask positions")
        else:
            labels_esc = load_escalation_labels(patients_dict, subject_ids, mask_pos)
            task_results = {}
            for vec_name, emb in enc_vectors.items():
                if split == "temporal":
                    assert temporal_masks is not None
                    res = evaluate_binary_probe_temporal(emb, labels_esc,
                                                         train_mask=temporal_masks[0], 
                                                         test_mask=temporal_masks[1])
                else:
                    res = evaluate_binary_probe(emb, labels_esc)
                    
                task_results[vec_name] = format_binary_result(res)
            results["escalation"] = task_results

    # -- icd_block (encounter-level) --
    if "icd_block" in tasks:
        if is_supervised or mask_pos is None:
            print("  [icd_block] skipped - requires JEPA model with mask positions")
        else:
            targets, chapter_names = extract_icd_block_targets(sequences_path, subject_ids, mask_pos)
            task_results = {}
            for vec_name, emb in enc_vectors.items():
                if split == "temporal":
                    assert temporal_masks is not None
                    res = probe_icd_blocks_temporal(emb, targets, chapter_names, 
                                                    train_mask=temporal_masks[0], 
                                                    test_mask=temporal_masks[1])
                else:
                    res = probe_icd_blocks(
                        emb, targets, chapter_names)
                    
                task_results[vec_name] = {
                    "macro_auroc": res["macro_auroc"],
                    "macro_auprc": res["macro_auprc"],
                    "macro_f1":    res["macro_f1"],
                    "n_chapters_evaluated": res["n_chapters_evaluated"],
                }
            results["icd_block"] = task_results

    # -- escalation_type (encounter-level) --
    if "escalation_type" in tasks:
        if is_supervised or mask_pos is None:
            print("  [escalation_type] skipped - requires JEPA model with mask positions")
        else:
            criteria_labels = compute_escalation_criterions(patients_dict, subject_ids, mask_pos)
            task_results = {}
            for vec_name, emb in enc_vectors.items():
                per_criterion: dict = {}
                macro_aurocs: list[float] = []
                macro_auprcs: list[float] = []
                macro_f1s: list[float] = []

                for criterion in ESCALATION_CRITERIA:
                    labels = criteria_labels[criterion]
                    n_pos = int(labels.sum())
                    n_neg = len(labels) - n_pos

                    if split == "temporal":
                        assert temporal_masks is not None
                        y_tr = labels[temporal_masks[0]]
                        y_te = labels[temporal_masks[1]]
                        
                        # Skip if either train or test set has no positive or no negative samples
                        if (y_tr.sum() < 1 or (len(y_tr) - y_tr.sum()) < 1
                                           or y_te.sum() < 1
                                           or (len(y_te) - y_te.sum()) < 1):
                            continue
                        res = evaluate_binary_probe_temporal(emb, labels,
                                                             train_mask=temporal_masks[0],
                                                             test_mask=temporal_masks[1])
                    else:
                        # Skip if number of positive or negative samples is too small for stable evaluation
                        if n_pos < 5 or n_neg < 5:
                            continue
                        res = evaluate_binary_probe(emb, labels)

                    per_criterion[criterion] = format_binary_result(res)
                    macro_aurocs.append(res["mean_auroc"])
                    macro_auprcs.append(res["mean_auprc"])
                    macro_f1s.append(res["mean_f1"])

                task_results[vec_name] = {
                    "per_criterion": per_criterion,
                    "n_criteria_evaluated": len(per_criterion),
                    "macro_auroc": float(np.mean(macro_aurocs)) if macro_aurocs else float("nan"),
                    "macro_auprc": float(np.mean(macro_auprcs)) if macro_auprcs else float("nan"),
                    "macro_f1": float(np.mean(macro_f1s)) if macro_f1s else float("nan"),
                }
            results["escalation_type"] = task_results

    return results


def print_summary(
    architecture: str,
    results: dict,
    split: str = "random",
    eval_subset: str = "all",
) -> None:
    label = f"{architecture} ({split} split"
    if eval_subset != "all":
        label += f", {eval_subset} subset"
    label += ")"
    
    print(f"  Evaluation Summary <{label}>")
    print("=" * 70)

    for task_name, task_results in results.items():
        print(f"\n  {task_name}")
        print(f"  {'-' * 70}")

        if task_name in ("icd_block", "escalation_type"):
            col = "n_ch" if task_name == "icd_block" else "n_crit"
            header = (f"  {'vector':<20s} {'macro_AUROC':>12s} {'macro_AUPRC':>12s} {'macro_F1':>12s} {col:>6s}")
            print(header)
            
            n_key = ("n_chapters_evaluated" if task_name == "icd_block" else "n_criteria_evaluated")
            for vec_name, metrics in task_results.items():
                print(f"  {vec_name:<20s} "
                      f"{metrics['macro_auroc']:>12.4f} "
                      f"{metrics['macro_auprc']:>12.4f} "
                      f"{metrics['macro_f1']:>12.4f} "
                      f"{metrics[n_key]:>6d}")
        else:
            header = (f"  {'vector':<20s} {'AUROC':>12s} {'AUPRC':>12s} {'F1':>12s} {'Brier':>12s}")
            print(header)
            for vec_name, metrics in task_results.items():
                print(f"  {vec_name:<20s} "
                      f"{metrics['auroc']:>12.4f} "
                      f"{metrics['auprc']:>12.4f} "
                      f"{metrics['f1']:>12.4f} "
                      f"{metrics['brier']:>12.4f}")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on downstream tasks")
    parser.add_argument("--model", type=str, required=True,
                        help="model name from 'experiments' directory")
    parser.add_argument("--checkpoint-name", type=str, required=True,
                        help="file name of .pt checkpoint in model's checkpoints/ directory.")
    parser.add_argument("--sequences", type=str,
                        default=str(DATA_DIR / "sequences.jsonl"),
                        help="Path to sequences.jsonl")
    parser.add_argument("--output", type=str,
                        default=None,
                        help="Path for results JSON (default: <run_dir>/eval_results.json)")
    parser.add_argument("--tasks", type=str,
                        default="readmit_30d,escalation,icd_block,escalation_type",
                        help="Comma-separated task list")
    parser.add_argument("--split", type=str, 
                        default="random",
                        choices=["random", "temporal"],
                        help="Split strategy: random (stratified CV) or temporal")
    parser.add_argument("--eval-subset", type=str, 
                        default="all",
                        choices=["all", "fcode", "non_fcode"],
                        help="Evaluate on a patient subset: fcode (F30-F39), non_fcode, or all")
    parser.add_argument("--batch-size", type=int, 
                        default=64,
                        help="Batch size for embedding extraction")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"Unknown task '{t}'. Available: {ALL_TASKS}")

    # -- Resolve experiment directory -----------------------------------------
    run_dir = EXPERIMENTS_DIR / args.model
    sequences_path = Path(args.sequences)
    config_path = run_dir / "config.yaml"
    vocab_path = run_dir / "vocab.json"
    
    loaded_seed = load_exp_seed(run_dir)
    set_global_seed(loaded_seed)
    
    

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_params = config["data"]

    # -- Load model -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    
    model, checkpoint = load_model_notrain(ckpt_path, device, restore_rng=False)
    model_params = checkpoint["model_params"]
    architecture = model_params.get("architecture", "")
    epoch = checkpoint.get("epoch", "")
    assert architecture != "" and epoch != "", \
        ValueError("Checkpoint missing 'architecture' or 'epoch'")
    is_supervised = architecture == "supervised"

    print(f"  Model:        {architecture}")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Epoch:        {epoch}")
    print(f"  Device:       {device}")
    print(f"  Tasks:        {tasks}")
    print(f"  Split:        {args.split}")
    print(f"  Eval subset:  {args.eval_subset}")

    # -- Load vocab & build dataset -------------------------------------------
    n_patients = data_params.get("n_patients", None)
    patients = load_sequences(n=n_patients)

    vocab = load_json(vocab_path)
    assert vocab is not None, f"vocab.json not found at {vocab_path}"

    if is_supervised:
        dataset = SupervisedDataset(patients, vocab, data_params, pad_idx=0)
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=supervised_collate_fn, drop_last=False,
            num_workers=0)
    else:
        dataset = MimicDataset(patients, vocab, data_params, pad_idx=0)
        loader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn, drop_last=False,
            num_workers=0)

    print(f"  Samples:      {len(dataset)}")

    # -- Extract embeddings --
    print("\n  Extracting embeddings...")
    if is_supervised:
        vecs = extract_supervised_embeddings(model, loader, device)
    else:
        vecs = extract_jepa_embeddings(model, loader, device)
        vecs = compute_derived_vectors(vecs)

    if is_supervised:
        print(f"  z_enc_pooled shape: {vecs['z_enc_pooled'].shape}")
    else:
        print(f"  z_pred shape:       {vecs['z_pred'].shape}")
        print(f"  z_enc_pooled shape: {vecs['z_enc_pooled'].shape}")

    # -- Load patient data for label lookups --
    patients_dict = load_sequences_dict(sequences_path)

    # -- Filter by eval subset --
    if args.eval_subset != "all":
        subset_mask = compute_subset_mask(patients_dict, vecs["subject_ids"], args.eval_subset)
        vecs = { k: v[subset_mask] for k, v in vecs.items() 
                 if k in ["z_enc_pooled", "z_encs", "z_pred", "z_target"] and v is not None
               }

    # -- Compute temporal split --
    temporal_masks = None
    if args.split == "temporal":
        print("\n  Computing temporal split...")
        train_mask, test_mask, cutoff = compute_temporal_split(
            sequences_path, vecs["subject_ids"])
        temporal_masks = (train_mask, test_mask)
        print(f"  Cutoff date:  {cutoff}")
        print(f"  Train:        {int(train_mask.sum())} samples")
        print(f"  Test:         {int(test_mask.sum())} samples")

    # -- Run evaluation tasks --
    print("\n  Running evaluation tasks...")
    task_results = run_tasks(
        vecs, patients_dict, sequences_path, tasks, is_supervised,
        split=args.split, temporal_masks=temporal_masks)

    # -- Build output --
    output = {
        "model": architecture,
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "seed": loaded_seed,
        "split": args.split,
        "eval_subset": args.eval_subset,
        "n_samples": len(vecs["subject_ids"]),
        "n_patients": len(np.unique(vecs["subject_ids"])),
        "modality": data_params.get("modality", "all"),
        "tasks": task_results,
    }

    # -- Print summary table --
    print_summary(architecture, task_results,
                  split=args.split, eval_subset=args.eval_subset)

    # -- Save JSON --
    output_path = Path(args.output) if args.output else run_dir / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"  Results saved -> {output_path}")


if __name__ == "__main__":
    main()
