#!/usr/bin/env python3
"""
Unified evaluation script for all trained models (stopgrad, ema, supervised).

Loads a checkpoint, extracts embeddings, and runs downstream probing tasks
(readmission prediction, ICD-10 chapter prediction) on each available latent
vector.  Outputs a structured JSON and a summary table to stdout.

Usage
-----
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --tasks readmit_90d,icd_block
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --split temporal
  python -m scripts.evaluate --checkpoint experiments/stopg_42_v01/checkpoints/checkpoint_100.pt --eval-subset fcode
"""
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.training.utils.checkpoint import load_model_notrain
from src.training.utils.datasets import (
    MimicDataset, collate_fn,
    SupervisedDataset, supervised_collate_fn,
)
from src.analysis.displacement import calc_embedding_vecs
from src.analysis.eval_tasks import (
    evaluate_readmission,
    extract_icd_block_targets,
    probe_icd_blocks,
)
from src.utils.io import load_sequences, build_vocab, PROCESSED_DIR


ALL_TASKS = ["readmit_90d", "readmit_30d", "icd_block"]

# Latent vectors available per architecture family
JEPA_VECTORS = ["z_context", "z_pred", "z_target", "delta", "pred_error", "observed_traj"]
SUPERVISED_VECTORS = ["z_context"]


# =============================================================================
# Embedding extraction (supervised)
# =============================================================================

@torch.no_grad()
def extract_supervised_embeddings(model, loader, device) -> dict:
    """Extract z_context from a supervised model. Returns dict matching
    the calc_embedding_vecs output format (JEPA fields set to None)."""
    model.eval()
    all_z, all_sids, all_labels = [], [], []
    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        z_context, _ = model(batch_dev)
        all_z.append(z_context.cpu().numpy())
        all_sids.extend(batch["subject_ids"])
        all_labels.append(batch["labels"].numpy())

    return {
        "z_context":      np.concatenate(all_z),
        "z_pred":         None,
        "z_target":       None,
        "delta":          None,
        "pred_error":     None,
        "observed_traj":  None,
        "subject_ids":    np.array(all_sids),
        "mask_positions": None,
        "labels":         np.concatenate(all_labels),
    }


# =============================================================================
# 30-day readmission label computation
# =============================================================================

def compute_30d_labels(sequences_path: Path, subject_ids: np.ndarray) -> np.ndarray:
    """Compute 30-day F-code readmission labels from encounter timestamps.

    Mirrors the extraction pipeline's labeling logic but with a 30-day window:
    label=1 if the patient has ANY consecutive encounter pair where
    gap <= 30 days and the later encounter contains an F-code.
    """
    patients = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    pid_labels = {}
    for pid, p in patients.items():
        encs = p["encounters"]
        found = False
        for i in range(len(encs) - 1):
            t_disch = datetime.fromisoformat(encs[i]["dischtime"])
            t_admit = datetime.fromisoformat(encs[i + 1]["admittime"])
            gap_days = (t_admit - t_disch).total_seconds() / 86400
            if gap_days <= 30:
                if any(c.upper().startswith("F") for c in encs[i + 1].get("icd_codes", [])):
                    found = True
                    break
        pid_labels[pid] = 1 if found else 0

    return np.array([pid_labels.get(str(sid), 0) for sid in subject_ids])


# =============================================================================
# Temporal train/test split
# =============================================================================

def compute_temporal_split(
    sequences_path: Path,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Split samples into train/test by median latest admission date.

    All encounter windows from the same patient are assigned to the same
    split.  Patients whose latest admission is strictly before the median
    cutoff go to train; the rest go to test.

    Returns
    -------
    train_mask  : (N,) bool array over subject_ids
    test_mask   : (N,) bool array over subject_ids
    cutoff_iso  : ISO-format string of the cutoff date
    """
    patients = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    latest: dict[str, datetime] = {}
    for pid, p in patients.items():
        dates = [datetime.fromisoformat(e["admittime"])
                 for e in p["encounters"] if "admittime" in e]
        if dates:
            latest[pid] = max(dates)

    all_dates = sorted(latest.values())
    cutoff = all_dates[len(all_dates) // 2]

    train_mask = np.array([latest.get(str(sid), cutoff) < cutoff
                           for sid in subject_ids])
    test_mask = ~train_mask

    return train_mask, test_mask, cutoff.isoformat()


def evaluate_readmission_temporal(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int = 42,
) -> dict:
    """Single temporal train/test split readmission probe.

    Returns the same dict structure as evaluate_readmission so that
    downstream formatting code works unchanged (fold lists have length 1,
    std values are 0.0).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embeddings[train_mask])
    X_te = scaler.transform(embeddings[test_mask])
    y_tr, y_te = labels[train_mask], labels[test_mask]

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    auroc = float(roc_auc_score(y_te, y_prob))
    auprc = float(average_precision_score(y_te, y_prob))
    f1_val = float(f1_score(y_te, y_pred))
    brier = float(brier_score_loss(y_te, y_prob))

    n_test = int(test_mask.sum())
    n_pos = int(y_te.sum())
    return {
        "fold_auroc": [auroc], "fold_auprc": [auprc],
        "fold_f1": [f1_val], "fold_brier": [brier],
        "mean_auroc": auroc, "std_auroc": 0.0,
        "mean_auprc": auprc, "std_auprc": 0.0,
        "mean_f1": f1_val, "std_f1": 0.0,
        "mean_brier": brier, "std_brier": 0.0,
        "n_samples": n_test,
        "n_positive": n_pos,
        "positive_rate": round(n_pos / n_test, 4) if n_test > 0 else 0.0,
        "n_train": int(train_mask.sum()),
        "n_test": n_test,
    }


def probe_icd_blocks_temporal(
    embeddings: np.ndarray,
    targets: np.ndarray,
    chapter_names: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seed: int = 42,
) -> dict:
    """Single temporal split ICD chapter probing.

    Returns the same dict structure as probe_icd_blocks.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embeddings[train_mask])
    X_te = scaler.transform(embeddings[test_mask])

    N_te = int(test_mask.sum())
    C = targets.shape[1]
    per_chapter: dict[str, dict] = {}
    macro_aurocs: list[float] = []
    macro_auprcs: list[float] = []
    macro_f1s: list[float] = []

    for c in range(C):
        y_tr = targets[train_mask, c]
        y_te = targets[test_mask, c]

        # Need both classes present in both splits
        if y_tr.sum() < 1 or (len(y_tr) - y_tr.sum()) < 1:
            continue
        if y_te.sum() < 1 or (len(y_te) - y_te.sum()) < 1:
            continue

        key = chapter_names[c] if chapter_names else str(c)
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        auroc = float(roc_auc_score(y_te, y_prob))
        auprc = float(average_precision_score(y_te, y_prob))
        f1_val = float(f1_score(y_te, y_pred))

        n_pos = int(y_te.sum())
        per_chapter[key] = {
            "n_positive": n_pos,
            "prevalence": round(n_pos / N_te, 4),
            "mean_auroc": auroc, "std_auroc": 0.0,
            "mean_auprc": auprc, "std_auprc": 0.0,
            "mean_f1": f1_val, "std_f1": 0.0,
        }
        macro_aurocs.append(auroc)
        macro_auprcs.append(auprc)
        macro_f1s.append(f1_val)

    return {
        "per_chapter": per_chapter,
        "n_chapters_evaluated": len(per_chapter),
        "macro_auroc": float(np.mean(macro_aurocs)) if macro_aurocs else float("nan"),
        "macro_auprc": float(np.mean(macro_auprcs)) if macro_auprcs else float("nan"),
        "macro_f1": float(np.mean(macro_f1s)) if macro_f1s else float("nan"),
    }


# =============================================================================
# Cohort subset filtering
# =============================================================================

def compute_subset_mask(
    sequences_path: Path,
    subject_ids: np.ndarray,
    subset: str,
) -> np.ndarray:
    """Compute boolean mask for --eval-subset filtering.

    Parameters
    ----------
    sequences_path : path to sequences.jsonl
    subject_ids    : (N,) sample-level patient IDs
    subset         : "all", "fcode", or "non_fcode"

    Returns
    -------
    (N,) bool array - True for samples to keep
    """
    if subset == "all":
        return np.ones(len(subject_ids), dtype=bool)

    patients: dict[str, dict] = {}
    with open(sequences_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p

    fcode_pids: set[str] = set()
    for pid, p in patients.items():
        for enc in p["encounters"]:
            if any(c.upper().startswith("F3") for c in enc.get("icd_codes", [])):
                fcode_pids.add(pid)
                break

    is_fcode = np.array([str(sid) in fcode_pids for sid in subject_ids])
    return is_fcode if subset == "fcode" else ~is_fcode


def filter_vecs(vecs: dict, mask: np.ndarray) -> dict:
    """Apply a boolean sample mask to all arrays in an embedding dict."""
    filtered = {}
    for key, val in vecs.items():
        if val is None:
            filtered[key] = None
        elif isinstance(val, np.ndarray):
            filtered[key] = val[mask]
        else:
            filtered[key] = val
    return filtered


# =============================================================================
# Core evaluation
# =============================================================================

def run_tasks(
    vecs: dict,
    sequences_path: Path,
    tasks: list[str],
    is_supervised: bool,
    seed: int,
    split: str = "random",
    temporal_masks: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Run requested evaluation tasks on all available latent vectors."""
    if split == "temporal":
        assert temporal_masks is not None, "--split temporal requires temporal_masks"
    vector_names = SUPERVISED_VECTORS if is_supervised else JEPA_VECTORS
    labels_90d = vecs["labels"]
    subject_ids = vecs["subject_ids"]
    mask_positions = vecs["mask_positions"]
    results = {}

    # --- readmission tasks ---------------------------------------------------
    for task in ["readmit_90d", "readmit_30d"]:
        if task not in tasks:
            continue
        if task == "readmit_90d":
            labels = labels_90d
        else:
            labels = compute_30d_labels(sequences_path, subject_ids)

        task_results = {}
        for vec_name in vector_names:
            emb = vecs[vec_name]
            if emb is None:
                continue
            if split == "temporal":
                res = evaluate_readmission_temporal(
                    emb, labels, temporal_masks[0], temporal_masks[1], seed=seed)
            else:
                res = evaluate_readmission(emb, labels, seed=seed)
            task_results[vec_name] = {
                "auroc": res["mean_auroc"],
                "auprc": res["mean_auprc"],
                "f1":    res["mean_f1"],
                "brier": res["mean_brier"],
                "std_auroc": res["std_auroc"],
                "std_auprc": res["std_auprc"],
                "std_f1":    res["std_f1"],
                "std_brier": res["std_brier"],
                "n_samples":  res["n_samples"],
                "n_positive": res["n_positive"],
                "positive_rate": res["positive_rate"],
            }
        results[task] = task_results

    # --- ICD block prediction ------------------------------------------------
    if "icd_block" in tasks:
        if is_supervised or mask_positions is None:
            print("  [icd_block] skipped - requires JEPA model with mask positions")
        else:
            targets, chapter_names = extract_icd_block_targets(
                sequences_path, subject_ids, mask_positions)
            task_results = {}
            for vec_name in vector_names:
                emb = vecs[vec_name]
                if emb is None:
                    continue
                if split == "temporal":
                    res = probe_icd_blocks_temporal(
                        emb, targets, chapter_names,
                        temporal_masks[0], temporal_masks[1], seed=seed)
                else:
                    res = probe_icd_blocks(emb, targets, chapter_names, seed=seed)
                task_results[vec_name] = {
                    "macro_auroc": res["macro_auroc"],
                    "macro_auprc": res["macro_auprc"],
                    "macro_f1":    res["macro_f1"],
                    "n_chapters_evaluated": res["n_chapters_evaluated"],
                }
            results["icd_block"] = task_results

    return results


# =============================================================================
# Summary table
# =============================================================================

def print_summary(
    architecture: str,
    results: dict,
    split: str = "random",
    eval_subset: str = "all",
) -> None:
    """Print a formatted summary table to stdout."""
    label = f"{architecture} ({split} split"
    if eval_subset != "all":
        label += f", {eval_subset} subset"
    label += ")"
    print("\n" + "=" * 80)
    print(f"  Evaluation Summary - {label}")
    print("=" * 80)

    for task_name, task_results in results.items():
        print(f"\n  {task_name}")
        print(f"  {'-' * 70}")

        if "icd_block" in task_name:
            header = f"  {'vector':<18s} {'macro_AUROC':>12s} {'macro_AUPRC':>12s} {'macro_F1':>12s} {'n_ch':>6s}"
            print(header)
            for vec_name, metrics in task_results.items():
                print(f"  {vec_name:<18s} "
                      f"{metrics['macro_auroc']:>12.4f} "
                      f"{metrics['macro_auprc']:>12.4f} "
                      f"{metrics['macro_f1']:>12.4f} "
                      f"{metrics['n_chapters_evaluated']:>6d}")
        else:
            header = f"  {'vector':<18s} {'AUROC':>12s} {'AUPRC':>12s} {'F1':>12s} {'Brier':>12s}"
            print(header)
            for vec_name, metrics in task_results.items():
                print(f"  {vec_name:<18s} "
                      f"{metrics['auroc']:>12.4f} "
                      f"{metrics['auprc']:>12.4f} "
                      f"{metrics['f1']:>12.4f} "
                      f"{metrics['brier']:>12.4f}")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on downstream tasks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--sequences", type=str,
                        default=str(PROCESSED_DIR / "sequences.jsonl"),
                        help="Path to sequences.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for results JSON (default: <run_dir>/eval_results.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", type=str, default="readmit_90d,readmit_30d,icd_block",
                        help="Comma-separated task list")
    parser.add_argument("--split", type=str, default="random",
                        choices=["random", "temporal"],
                        help="Split strategy: random (stratified CV) or temporal (median date cutoff)")
    parser.add_argument("--eval-subset", type=str, default="all",
                        choices=["all", "fcode", "non_fcode"],
                        help="Evaluate on a patient subset: fcode (F30-F39), non_fcode, or all")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    tasks = [t.strip() for t in args.tasks.split(",")]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"Unknown task '{t}'. Available: {ALL_TASKS}")

    # -- Resolve experiment directory -----------------------------------------
    # Checkpoint lives at <run_dir>/checkpoints/checkpoint_N.pt
    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.yaml"
    vocab_path = run_dir / "vocab.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_params = config["data"]
    sequences_path = Path(args.sequences)

    # -- Load model -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_notrain(ckpt_path, device, restore_rng=False)
    model_params = checkpoint["model_params"]
    architecture = model_params.get("architecture", "unknown")
    epoch = checkpoint.get("epoch", "?")
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

    if vocab_path.exists():
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"  Vocab:        {vocab_path} ({len(vocab)} tokens)")
    else:
        vocab = build_vocab(patients, pad_idx=0, dir=run_dir)

    if is_supervised:
        dataset = SupervisedDataset(patients, vocab, data_params, pad_idx=0)
        loader = DataLoader(
            dataset, batch_size=data_params.get("batch_size", 64),
            shuffle=False, collate_fn=supervised_collate_fn, drop_last=False,
            num_workers=0)
    else:
        dataset = MimicDataset(patients, vocab, data_params, pad_idx=0)
        loader = DataLoader(
            dataset, batch_size=data_params.get("batch_size", 64),
            shuffle=False, collate_fn=collate_fn, drop_last=False,
            num_workers=0)

    print(f"  Samples:      {len(dataset)}")

    # -- Extract embeddings ---------------------------------------------------
    print("\n  Extracting embeddings...")
    if is_supervised:
        vecs = extract_supervised_embeddings(model, loader, device)
    else:
        vecs = calc_embedding_vecs(model, loader, device)

    print(f"  z_context shape: {vecs['z_context'].shape}")

    # -- Filter by eval subset if requested -----------------------------------
    if args.eval_subset != "all":
        subset_mask = compute_subset_mask(
            sequences_path, vecs["subject_ids"], args.eval_subset)
        n_before = len(vecs["subject_ids"])
        vecs = filter_vecs(vecs, subset_mask)
        print(f"\n  Subset filter: {args.eval_subset} - "
              f"{int(subset_mask.sum())}/{n_before} samples kept")

    # -- Compute temporal split if requested ----------------------------------
    temporal_masks = None
    if args.split == "temporal":
        print("\n  Computing temporal split...")
        train_mask, test_mask, cutoff = compute_temporal_split(
            sequences_path, vecs["subject_ids"])
        temporal_masks = (train_mask, test_mask)
        print(f"  Cutoff date:  {cutoff}")
        print(f"  Train:        {int(train_mask.sum())} samples")
        print(f"  Test:         {int(test_mask.sum())} samples")

    # -- Run evaluation tasks -------------------------------------------------
    print("\n  Running evaluation tasks...")
    task_results = run_tasks(
        vecs, sequences_path, tasks, is_supervised, args.seed,
        split=args.split, temporal_masks=temporal_masks)

    # -- Build output ---------------------------------------------------------
    output = {
        "model": architecture,
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "seed": args.seed,
        "split": args.split,
        "eval_subset": args.eval_subset,
        "n_samples": len(vecs["subject_ids"]),
        "modality": data_params.get("modality", "all"),
        "tasks": task_results,
    }

    # -- Print summary table --------------------------------------------------
    print_summary(architecture, task_results,
                  split=args.split, eval_subset=args.eval_subset)

    # -- Save JSON ------------------------------------------------------------
    output_path = Path(args.output) if args.output else run_dir / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"  Results saved -> {output_path}")


if __name__ == "__main__":
    main()
