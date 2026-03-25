import os
os.environ["GRPC_VERBOSITY"] = "NONE" # suppress gRPC/abseil C++ log spam
os.environ["GRPC_TRACE"] = ""

import argparse
from src.mimic.mimic import build_patient_sequences, load_tables
from src.mimic.helper import save_dataset, validate_sequences
from src.mimic.baseline import run_baseline
from src.utils.io import PROCESSED_DIR


BQ_PROJECT_ID = "aihc-463505"
BQ_PROJECT_NAME = "mimic-aihc"
MIMIC_BQ_DATASET = "physionet-data.mimiciv_3_1_hosp"

# Readmission window in days (cohort = any readmission within this window)
READM_WINDOW_DAYS = 90
# ICD-10 prefix for positive label
LABEL_ICD10_PREFIX = "F"
# Minimum encounters per patient
MIN_ENCOUNTERS = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""MIMIC-IV Patient Sequence extraction pipeline
      - Running requires either a PhysioNet-linked BigQuery project set up, or admissions.parquet, 
        diagnoses.partquet, medications.parquet, and prescriptions.parquet downloaded and specified 
        with --data-dir.
      - By default, runs data validation tests and a logistic regression baseline. Use --skip-tests and 
        '--skip-baseline' to bypass these.
      - Specify output directory for sequences and metadata with --output-dir (defaults to src/datasets/mimic-out)
      - Example usage:
        python src/extract.py --val-seq --baseline --dry-run --min-encounters 4 --readm-window-days 60
        python src/extract.py --data-dir data/mimic-parquet-1234 --output-dir data/mimic-out-1234

      Happy hacking!""")
    parser.add_argument("--min-encounters",     default=MIN_ENCOUNTERS, type=int,
                        help=f"Minimum encounters per patient")
    parser.add_argument("--readm-window-days",  default=READM_WINDOW_DAYS, type=int,
                        help=f"Readmission window in days")
    parser.add_argument("--val-seq",            default=True, action="store_false")
    parser.add_argument("--baseline",           default=False, action="store_true")
    parser.add_argument("--dry-run",            default=False, action="store_true")
    args = parser.parse_args()

    admissions, patients, diagnoses, prescriptions = load_tables(MIMIC_BQ_DATASET, BQ_PROJECT_ID)

    sequences = build_patient_sequences(
        admissions, patients, diagnoses, prescriptions,
        min_encounters=args.min_encounters,
        readm_window_days=args.readm_window_days,
        label_prefix=LABEL_ICD10_PREFIX)

    if args.val_seq:
        validate_sequences(sequences, args.readm_window_days)
        
    if args.baseline:
        run_baseline(sequences)

    if not args.dry_run:
        save_dataset(sequences, PROCESSED_DIR, args.min_encounters,
                     args.readm_window_days)