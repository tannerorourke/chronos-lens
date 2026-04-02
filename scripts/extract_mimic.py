import os
os.environ["GRPC_VERBOSITY"] = "NONE" # suppress gRPC/abseil C++ log spam
os.environ["GRPC_TRACE"] = ""

import json
import argparse
from pathlib import Path

import numpy as np

from src.mimic.mimic import build_patient_sequences, get_clean_encounters, load_tables
from src.mimic.helper import validate_sequences
from src.mimic.metadata import extract_metadata
from src.mimic.baselines import run_baselines
from src.mimic.labels import compute_labels
from src.utils.constants import LABEL_ICD10_PREFIX
from src.utils.io import DATA_DIR, save_json, save_metadata, load_sequences
from src.training.utils.datasets import build_vocab

# =============================================================================
# MIMIC Settings - change as needed
BQ_PROJECT_ID = "aihc-463505"
BQ_PROJECT_NAME = "mimic-aihc"
MIMIC_BQ_DATASET = "physionet-data.mimiciv_3_1_hosp"
# =============================================================================

parser = argparse.ArgumentParser(description="""MIMIC-IV Patient Sequence extraction pipeline
    - Requires a PhysioNet-linked BigQuery project, or cached parquet files in data/parquet/.
    - Use --seq-path to skip extraction and work from an existing sequences.jsonl.
    - Example usage:
    python scripts/extract_mimic.py --baseline --dry-run
    python scripts/extract_mimic.py --seq-path data/processed/sequences.jsonl --baseline
""")
parser.add_argument("--seq-path",           default=None, type=str,
                    help="Path to existing sequences.jsonl. Skips to labeling and metadata")
parser.add_argument("--min-encounters",     default=2, type=int,
                    help="Minimum encounters per patient (default: 2)")
parser.add_argument("--skip-metadata",      default=False, action="store_true",
                    help="Skip metadata extraction (can be run later in metadata.py)")
parser.add_argument("--skip-labeling",      default=False, action="store_true",
                    help="Skip label computation (keep existing labels in sequences.jsonl)")
parser.add_argument("--skip-vocab",         default=False, action="store_true",
                    help="Skip creating vocab.json (done automatically when running training model)")
parser.add_argument("--dry-run",            default=False, action="store_true",
                    help="Skip saving to disk")
parser.add_argument("--baseline",           default=False, action="store_true",
                    help="Run logistic regression and XGBoost baselines")


def save_dataset(sequences: list[dict], out_dir: Path):
    """Save sequences and stats to 
       DATA_DIR/
         sequences.jsonl    - one JSON object per patient
         dataset_stats.json - cohort stats, schema, label distribution
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- sequences  --
    jsonl_path = out_dir / "sequences.jsonl"
    with open(jsonl_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps({
                "subject_id": seq["subject_id"],
                "label_30d": seq.get("label_30d"),
                "label_escalation": seq.get("label_escalation"),
                "label_escalation_per_enc": seq.get("label_escalation_per_enc"),
                "escalation_criteria_fired": seq.get("escalation_criteria_fired"),
                "next_enc_icd_blocks": seq.get("next_enc_icd_blocks"),
                "encounters": [
                    {
                        "hadm_id": enc["hadm_id"],
                        "admittime": enc["admittime"].isoformat(),
                        "dischtime": enc["dischtime"].isoformat(),
                        "icd_codes": enc["icd_codes"],
                        "meds": enc["meds"],
                    }
                    for enc in seq["encounters"]
                ],
            }) + "\n")
    print(f"\nDataset saved to {out_dir}/")
    print(f"  {jsonl_path.name:20s} {jsonl_path.stat().st_size / 1024:.0f} KB")

    # -- dataset_stats.json --
    enc_counts = [len(s["encounters"]) for s in sequences]
    n_pos_30d = sum(1 for s in sequences if s.get("label_30d") == 1)
    n_pos_esc = sum(1 for s in sequences if s.get("label_escalation") == 1)
    all_icd = set()
    all_meds = set()
    for s in sequences:
        for enc in s["encounters"]:
            all_icd.update(enc["icd_codes"])
            all_meds.update(enc["meds"])

    meta_path = out_dir / "dataset_stats.json"
    with open(meta_path, "w") as f:
        json.dump({
            "n_patients": len(sequences),
            "n_positive_30d": n_pos_30d,
            "n_negative_30d": len(sequences) - n_pos_30d,
            "positive_rate_30d": round(n_pos_30d / len(sequences), 4),
            "n_positive_esc": n_pos_esc,
            "n_negative_esc": len(sequences) - n_pos_esc,
            "positive_rate_esc": round(n_pos_esc / len(sequences), 4),
            "encounters_per_patient": {
                "mean": round(np.mean(enc_counts), 2),
                "median": int(np.median(enc_counts)),
                "min": int(min(enc_counts)),
                "max": int(max(enc_counts)),
            },
            "vocab_size_icd": len(all_icd),
            "vocab_size_meds": len(all_meds),
            "schema": {
                "subject_id": "str",
                "label_30d": "int",
                "label_escalation": "int",
                "label_escalation_per_enc": "list[int]",
                "escalation_criteria_fired": "list[str]",
                "next_enc_icd_blocks": "list[list[str]]",
                "encounters[].hadm_id": "int",
                "encounters[].admittime": "ISO datetime string",
                "encounters[].dischtime": "ISO datetime string",
                "encounters[].icd_codes": "list[str]",
                "encounters[].meds": "list[str]",
            },
        }, f, indent=2)
    print(f"  {meta_path.name:20s} {meta_path.stat().st_size / 1024:.0f} KB (stats & schema)")



def main():
    args = parser.parse_args()

    # -- Build or load sequences --
    if args.seq_path:
        print(f"Loading sequences from {args.seq_path}...")
        sequences = load_sequences(path=args.seq_path)
        print(f"  Loaded {len(sequences):,} sequences")
    else:
        admissions, patients, diagnoses, prescriptions = load_tables(MIMIC_BQ_DATASET,
                                                                     BQ_PROJECT_ID)
        encounters = get_clean_encounters(admissions, diagnoses, prescriptions,
                                          label_prefix=LABEL_ICD10_PREFIX)
        sequences = build_patient_sequences(encounters,
                                            min_encounters=args.min_encounters)
        del admissions, patients, diagnoses, prescriptions, encounters

    # -- Compute labels --
    if not args.skip_labeling:
        sequences = compute_labels(sequences, LABEL_ICD10_PREFIX)

    # -- Always validate --
    validate_sequences(sequences, args.min_encounters)
    
    # -- Save --
    if not args.dry_run:
        save_dataset(sequences, DATA_DIR)

    # -- Extract/save metadata (from in-memory) --
    if not args.skip_metadata:
        metadata, feature_names, patient_ids = extract_metadata(sequences, subject_ids=None)
        if not args.dry_run:
            save_metadata(metadata, feature_names, patient_ids)
        
        # -- Baselines (optional), can be done thru src/mimic/baselines.py --
        if args.baseline:
            vocab = build_vocab(sequences, pad_idx=0, dir=DATA_DIR)
            baseline_results = run_baselines(
                sequences, metadata, feature_names, patient_ids, vocab=vocab)
            save_json(baseline_results, DATA_DIR / "baseline_results.json")
            

if __name__ == "__main__":
    main()
