#!/usr/bin/env python3
"""
Unified evaluation script for all trained models (stopgrad, ema, supervised).

Loads a checkpoint, extracts embeddings, and runs downstream probing tasks
on each available latent vector. Outputs a structured JSON and a summary
table to stdout.

Tasks
-----
  readmit_30d      : 30-day F-code readmission (patient-level, terminal sample per patient)
  escalation       : per-encounter escalation binary probe
  icd_block        : ICD-10 chapter prediction for the masked encounter
  escalation_type  : per-criterion escalation probes (6 binary probes)

Usage
-----
  python -m scripts.diagnostic --exp stopg_42_v01 --checkpoint-name checkpoint_100.pt
  python -m scripts.diagnostic --exp stopg_42_v01 --checkpoint-name checkpoint_100.pt --tasks readmit_30d,escalation
  python -m scripts.diagnostic --exp stopg_42_v01 --checkpoint-name checkpoint_100.pt --split temporal
  python -m scripts.diagnostic --exp stopg_42_v01 --checkpoint-name checkpoint_100.pt --eval-subset fcode
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.infra.inference import load_scaffolding, in_mem_extract_embeds
from src.infra.labels import (
    load_label, load_escalation_labels, compute_escalation_criterions,
    compute_subset_mask, compute_temporal_split, extract_icd_block_targets)
from src.infra.vector_computation import compute_derived_vectors, select_terminal_by_patient
from src.infra.metrics import format_binary_result
from src.analysis.probing import (
    evaluate_binary_probe, evaluate_binary_probe_temporal, 
    probe_icd_blocks, probe_icd_blocks_temporal)
from src.utils.constants import ESCALATION_CRITERIA, MODEL_PRED_VECS, ALL_TASKS
from src.utils.io import load_sequences_dict, DATA_DIR, EXPS_DIR
from src.utils.system import set_global_seed, load_exp_seed

# =============================================================================
# Core evaluation
# =============================================================================

def run_tasks(
    vecs: dict,
    patients_dict: dict[str, dict],
    sequences_path: Path,
    tasks: list[str],
    split: str = "random",
    temporal_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Run evaluation tasks on all available latent vectors.

    Patient-level tasks (readmit_30d) represent each patient by their terminal
    sample (largest mask_pos) - the latest encoded state, no mean over samples.
    Encounter-level tasks (escalation, icd_block, escalation_type) use
    sample-level vectors.

    Architecture-agnostic: supervised carries the same per-sample ``z_enc_recency``
    as JEPA (which additionally has z_pred/z_target/pred_error), so every model is
    reduced and probed identically.
    """
    subject_ids = vecs["subject_ids"]
    mask_pos = vecs.get("mask_pos")
    results: dict = {}

    # -- Build encounter-level vector dict (whatever vectors are present) --
    #    supervised -> {z_enc_recency}; JEPA -> {z_pred, z_target, pred_error, z_enc_recency}
    enc_vectors = {
        name: vecs[name] for name in MODEL_PRED_VECS
        if vecs.get(name) is not None
    }

    pat_vectors, pat_ids = (
        select_terminal_by_patient(enc_vectors, subject_ids, mask_pos)
        if mask_pos is not None else (None, None)
    )

    # -- readmit_30d (patient-level) --
    if "readmit_30d" in tasks:
        if pat_vectors is None:
            print("  [readmit_30d] skipped - requires per-sample mask positions")
        else:
            labels_30d = load_label(patients_dict, pat_ids, "label_30d")
            task_results: dict = {}
            pat_temporal: tuple[np.ndarray, np.ndarray] | None = None
            if split == "temporal":
                pat_train, pat_test, _ = compute_temporal_split(sequences_path, pat_ids)
                pat_temporal = (pat_train, pat_test)
            for vec_name, emb in pat_vectors.items():
                if split == "temporal":
                    res = evaluate_binary_probe_temporal(emb, labels_30d,
                                                         train_mask=pat_temporal[0],
                                                         test_mask=pat_temporal[1])
                else:
                    res = evaluate_binary_probe(emb, labels_30d)
                task_results[vec_name] = format_binary_result(res)
            results["readmit_30d"] = task_results

    # -- escalation (encounter-level) --
    if "escalation" in tasks:
        if mask_pos is None:
            print("  [escalation] skipped - requires per-sample mask positions")
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
        if mask_pos is None:
            print("  [icd_block] skipped - requires per-sample mask positions")
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
        if mask_pos is None:
            print("  [escalation_type] skipped - requires per-sample mask positions")
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



def build_diagnostic_ctx(args: argparse.Namespace, tasks: list[str]) -> dict:
    """Step [2]: gather everything run_tasks needs - model, loader, dataset,
    extracted embeddings, label lookups, and (optionally) temporal-split masks.

    Rebuilds the model + loader from the run's frozen artifacts via
    ``load_scaffolding`` (inference only; no training), then extracts embeddings
    and applies the eval-subset filter and temporal split. Returns a context dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Rebuild model + loader from the run's frozen artifacts ---------------
    model, loader, (is_supervised, _label_key), (checkpoint, config), (ds, _vocab) = \
        load_scaffolding(args.exp, args.checkpoint_name, device)
    run_dir = EXPS_DIR / args.exp

    architecture = config["model"]["architecture"]
    epoch = checkpoint.get("epoch", "")
    ckpt_path = run_dir / "checkpoints" / args.checkpoint_name

    print(f"  Model:        {architecture}")
    print(f"  Run dir:      {run_dir}")
    print(f"  Epoch:        {epoch}")
    print(f"  Device:       {device}")
    print(f"  Tasks:        {tasks}")
    print(f"  Split:        {args.split}")
    print(f"  Eval subset:  {args.eval_subset}")
    print(f"  Samples:      {len(ds)}")

    # -- Extract embeddings ---------------------------------------------------
    print("\n  Extracting embeddings...")
    # Single extraction path for every arch: z_encs (N, C, D); JEPA also yields
    # z_pred/z_target. compute_derived_vectors adds z_enc_recency (+ pred_error).
    vecs = in_mem_extract_embeds(model, loader, device, is_supv=is_supervised)
    vecs = compute_derived_vectors(vecs)
    print(f"  z_enc_recency shape: {vecs['z_enc_recency'].shape}")

    # -- Label lookup source --------------------------------------------------
    sequences_path = DATA_DIR / "sequences.jsonl"
    patients_dict = load_sequences_dict(sequences_path)

    # -- Filter by eval subset (apply mask to every per-sample array) ---------
    if args.eval_subset != "all":
        subset_mask = compute_subset_mask(patients_dict, vecs["subject_ids"], args.eval_subset)
        vecs = {
            k: (v[subset_mask]
                if isinstance(v, np.ndarray) and v.shape[:1] == subset_mask.shape
                else v)
            for k, v in vecs.items()
        }

    # -- Temporal split masks -------------------------------------------------
    temporal_masks = None
    if args.split == "temporal":
        print("\n  Computing temporal split...")
        train_mask, test_mask, cutoff = compute_temporal_split(
            sequences_path, vecs["subject_ids"])
        temporal_masks = (train_mask, test_mask)
        print(f"  Cutoff date:  {cutoff}")
        print(f"  Train:        {int(train_mask.sum())} samples")
        print(f"  Test:         {int(test_mask.sum())} samples")

    return {
        "vecs": vecs,
        "patients_dict": patients_dict,
        "sequences_path": sequences_path,
        "architecture": architecture,
        "epoch": epoch,
        "seed": config["meta"]["seed"],
        "ckpt_path": ckpt_path,
        "run_dir": run_dir,
        "temporal_masks": temporal_masks,
        "config": config,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # [1] -- Read args --------------------------------------------------------
    parser = argparse.ArgumentParser(description="Evaluate trained model on downstream tasks")
    parser.add_argument("--exp", type=str, required=True,
                        help="run-id from the runs/ directory")
    parser.add_argument("--checkpoint-name", type=str, required=True,
                        help="file name of .pt checkpoint in the run's checkpoints/ directory.")
    parser.add_argument("--output", type=str, default=None,
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
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"Unknown task '{t}'. Available: {ALL_TASKS}")

    # Seed at entry from the run's frozen config (load_scaffolding re-applies the
    # same seed when it rebuilds the model; setting it here makes the run's
    # determinism explicit and independent of that call order).
    set_global_seed(load_exp_seed(EXPS_DIR / args.exp))

    # [2] -- Gather context (model, loader, embeddings, masks) ----------------
    ctx = build_diagnostic_ctx(args, tasks)

    # [3] -- Run evaluation tasks ---------------------------------------------
    print("\n  Running evaluation tasks...")
    task_results = run_tasks(
        ctx["vecs"], ctx["patients_dict"], ctx["sequences_path"], tasks,
        split=args.split, temporal_masks=ctx["temporal_masks"])

    # [4] -- Print summary + save JSON ----------------------------------------
    print_summary(ctx["architecture"], task_results,
                  split=args.split, eval_subset=args.eval_subset)

    output = {
        "model": ctx["architecture"],
        "checkpoint": str(ctx["ckpt_path"]),
        "epoch": ctx["epoch"],
        "seed": ctx["seed"],
        "split": args.split,
        "eval_subset": args.eval_subset,
        "n_samples": len(ctx["vecs"]["subject_ids"]),
        "n_patients": len(np.unique(ctx["vecs"]["subject_ids"])),
        "modality": ctx["config"]["data"].get("modality", "all"),
        "tasks": task_results,
    }

    output_path = Path(args.output) if args.output else ctx["run_dir"] / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"  Results saved -> {output_path}")


if __name__ == "__main__":
    main()
