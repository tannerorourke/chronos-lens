import os


os.environ["GRPC_VERBOSITY"] = "NONE" # suppress gRPC/abseil C++ log spam
os.environ["GRPC_TRACE"] = ""

import json
import argparse
from pathlib import Path

from src.mimic.mimic import build_patient_sequences, get_clean_encounters, load_tables
from src.mimic.helper import validate_sequences
from src.mimic.metadata import extract_metadata
from src.mimic.baselines import run_baselines
from src.mimic.labels import compute_labels
from src.utils.constants import LABEL_ICD10_PREFIX
from src.utils.io import DATA_DIR, save_json, save_metadata, load_sequences
from src.training.utils.datasets import build_vocab
from src.analysis.plotting import plot_pat_enc_histogram

try:
    from dotenv import load_dotenv
    load_dotenv()  # optional dotenv
except:
    pass

# =============================================================================
# MIMIC Settings - change as needed
BQ_PROJECT_ID = os.environ["BQ_PROJECT_ID"]
BQ_PROJECT_NAME = os.environ["BQ_PROJECT_NAME"]
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
parser.add_argument("--min-encounters",     default=3, type=int,
                    help="Minimum encounters per patient (default: 3)")
parser.add_argument("--max-encounters",     default=250, type=int,
                    help="Maximum encounters per patient (default: 20)")
parser.add_argument("--dry-run",            default=False, action="store_true",
                    help="Skip saving to disk")
parser.add_argument("--baseline",           default=False, action="store_true",
                    help="Run logistic regression and XGBoost baselines")


def save_dataset(
    sequences: list[dict], 
    label_meta: dict = {}, 
    out_dir: Path = DATA_DIR
):
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
                "label_30d_per_enc": seq.get("label_30d_per_enc"),
                "label_escalation": seq.get("label_escalation"),
                "label_escalation_per_enc": seq.get("label_escalation_per_enc"),
                "escalation_criteria_fired": seq.get("escalation_criteria_fired"),
                "next_enc_icd_blocks": seq.get("next_enc_icd_blocks"),
                "encounters": [
                    {
                        "hadm_id": enc["hadm_id"],
                        "admittime": enc["admittime"].isoformat(),
                        "dischtime": enc["dischtime"].isoformat(),
                        "days_since_first": enc["days_since_first"],
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
    all_icd = set()
    all_meds = set()
    for s in sequences:
        for enc in s["encounters"]:
            all_icd.update(enc["icd_codes"])
            all_meds.update(enc["meds"])

    meta_path = out_dir / "dataset_stats.json"
    with open(meta_path, "w") as f:
        json.dump({
            "vocab_size_icd": len(all_icd),
            "vocab_size_meds": len(all_meds),
            **label_meta,
            "schema": {
                "subject_id": "str",
                "label_30d": "int",
                "label_30d_per_enc": "list[int]",
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
    # No seed is set here: this is a deterministic MIMIC ETL (table joins,
    # cleaning, sequence building, label computation) with no sampling, shuffle,
    # or random split. Downstream model/analysis entries seed themselves.
    args = parser.parse_args()

    print(f"Running data extraction!")
    print(f"  min encounters: {args.min_encounters}")
    print(f"  max encounters: {args.max_encounters}")
    print(f"  dry run: {args.dry_run}")
    print(f"  baseline: {args.baseline}")
    print(f"  output dir: {DATA_DIR}\n")

    # -- Build or load sequences
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
                                            min_encounters=args.min_encounters,
                                            max_encounters=args.max_encounters)
        del admissions, patients, diagnoses, prescriptions, encounters

    # -- Compute labels
    sequences, label_meta = compute_labels(sequences, LABEL_ICD10_PREFIX)
    validate_sequences(sequences, args.min_encounters, args.max_encounters)

    # -- Extract/save metadata (from in-memory)
    metadata, feature_names, patient_ids = extract_metadata(sequences, subject_ids=None,
                                                            label_metadata=label_meta)
    
    plot_pat_enc_histogram(sequences, label_meta["encounters_per_patient"])
    
    # -- Save
    if not args.dry_run:
        save_dataset(sequences, label_meta, DATA_DIR)
        save_metadata(metadata, feature_names, patient_ids)
    
    # -- Baselines (optional), can be done thru src/mimic/baselines.py
    if args.baseline:
        vocab = build_vocab(sequences, pad_idx=0, dir=DATA_DIR)
        baseline_results = run_baselines(sequences, metadata, feature_names, 
                                         patient_ids, vocab)
        save_json(baseline_results, DATA_DIR / "baseline_results.json")
            

if __name__ == "__main__":
    main()
