import os
os.environ["GRPC_VERBOSITY"] = "NONE" # suppress gRPC/abseil C++ log spam
os.environ["GRPC_TRACE"] = ""

import argparse
from src.mimic.mimic import build_patient_sequences, get_clean_encounters, load_tables
from src.mimic.helper import save_dataset, validate_sequences
from src.mimic.metadata import extract_metadata
from src.mimic.baselines import run_logistic, run_xgboost
from src.mimic.labels import compute_labels
from src.utils.io import PROCESSED_DIR, save_metadata, load_sequences

# =============================================================================
# MIMIC Settings - change as needed
BQ_PROJECT_ID = "aihc-463505"
BQ_PROJECT_NAME = "mimic-aihc"
MIMIC_BQ_DATASET = "physionet-data.mimiciv_3_1_hosp"

# =============================================================================
# Extraction Settings
READM_WINDOW_DAYS = 90
LABEL_ICD10_PREFIX = "F"
MIN_ENCOUNTERS = 3
COMPUTE_READM_LABELS = True
COMPUTE_ESCALATION_LABELS = True
# =============================================================================


parser = argparse.ArgumentParser(description="""MIMIC-IV Patient Sequence extraction pipeline
    - Requires a PhysioNet-linked BigQuery project, or cached parquet files in data/parquet/.
    - Use --seq-path to skip extraction and work from an existing sequences.jsonl.
    - Example usage:
    python scripts/extract_mimic.py --val-seq --baseline --dry-run
    python scripts/extract_mimic.py --seq-path data/processed/sequences.jsonl --baseline
""")
parser.add_argument("--seq-path",           default=None, type=str,
                    help="Path to existing sequences.jsonl. Skips data extraction.")
parser.add_argument("--min-encounters",     default=MIN_ENCOUNTERS, type=int,
                    help="Minimum encounters per patient")
parser.add_argument("--readm-window-days",  default=READM_WINDOW_DAYS, type=int,
                    help="Readmission window (days) for the primary label")
parser.add_argument("--skip-labeling",      default=False, action="store_true",
                    help="Skip label computation (use existing labels in sequences.jsonl)")
parser.add_argument("--dry-run",            default=False, action="store_true",
                    help="Skip saving to disk")
parser.add_argument("--baseline",           default=False, action="store_true",
                    help="Run logistic regression and XGBoost baselines")


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
                                          LABEL_ICD10_PREFIX)
        sequences = build_patient_sequences(encounters,
                                            min_encounters=args.min_encounters)
        del admissions, patients, diagnoses, prescriptions, encounters

    # -- Compute all labels --
    if not args.skip_labeling:
        sequences = compute_labels(sequences, LABEL_ICD10_PREFIX, COMPUTE_ESCALATION_LABELS,
                                   COMPUTE_READM_LABELS, args.readm_window_days,)

    # -- Always validate --
    validate_sequences(sequences)

    # -- Always extract metadata (from in-memory) --
    metadata, feature_names, patient_ids = extract_metadata(sequences, subject_ids=None)

    # -- Save --
    if not args.dry_run:
        save_dataset(sequences, PROCESSED_DIR, args.readm_window_days)
        save_metadata(metadata, feature_names, patient_ids)

    # -- Baselines (optional), can be done thru src/mimic/baselines.py --
    if args.baseline:
        labels = metadata[:, feature_names.index("label")].astype(int)
        run_logistic(metadata, labels, feature_names)
        run_xgboost(metadata, labels, feature_names)


if __name__ == "__main__":
    main()
