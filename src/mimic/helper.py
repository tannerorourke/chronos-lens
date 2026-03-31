import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import PARQUET_DIR


def save_dataset(
    sequences: list[dict],
    out_dir: Path,
):
    """
    Save sequences and stats:
      PROCESSED_DIR/
        sequences.jsonl    - one JSON object per patient
        dataset_stats.json - cohort stats, schema, label distribution
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSONL ---
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

    # --- dataset_stats.json ---
    enc_counts = [len(s["encounters"]) for s in sequences]
    n_pos = sum(1 for s in sequences if s.get("label_30d") == 1)
    all_icd = set()
    all_meds = set()
    for s in sequences:
        for enc in s["encounters"]:
            all_icd.update(enc["icd_codes"])
            all_meds.update(enc["meds"])

    stats = {
        "n_patients": len(sequences),
        "n_positive_30d": n_pos,
        "n_negative_30d": len(sequences) - n_pos,
        "positive_rate_30d": round(n_pos / len(sequences), 4),
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
    }

    meta_path = out_dir / "dataset_stats.json"
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset saved to {out_dir}/")
    print(f"  {jsonl_path.name:20s} {jsonl_path.stat().st_size / 1024:.0f} KB  (primary - for JEPA dataloader)")
    print(f"  {meta_path.name:20s} {meta_path.stat().st_size / 1024:.0f} KB  (cohort stats & schema)")


def save_parquets(admissions, patients, diagnoses, prescriptions) -> None:
    print(f"[save_parquets] Saving parquet's to cache...")
    
    tables = {
        "admissions":    admissions,
        "patients":      patients,
        "diagnoses":     diagnoses,
        "prescriptions": prescriptions,
    }
    
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = PARQUET_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name:20s} {len(df):>8,} rows -> {path.name}")
        
    print(f"-- parquet's saved.")


def load_parquets(data_dir: Path) -> tuple:
    print(f"\n[load_parquets] Loading parquets from DATA_DIR...")

    file_map = {
        "admissions":    "admissions.parquet",
        "patients":      "patients.parquet",
        "diagnoses":     "diagnoses.parquet",
        "prescriptions": "prescriptions.parquet",
    }

    dfs = {}
    for name, filename in file_map.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Expected files: {list(file_map.values())}")
        dfs[name] = pd.read_parquet(path)
        print(f"   {name:20s} {len(dfs[name]):>8,} rows")

    return dfs["admissions"], dfs["patients"], dfs["diagnoses"], dfs["prescriptions"]



def validate_sequences(sequences: list[dict]):
    print("\nVALIDATING SEQUENCES..")
    print("=" * 60)

    assert len(sequences) > 0, "FAIL: no sequences produced"

    min_enc = min(len(s["encounters"]) for s in sequences)
    assert min_enc >= 3, f"FAIL: found sequence with {min_enc} encounters"

    for seq in sequences:
        times = [enc["admittime"] for enc in seq["encounters"]]
        assert times == sorted(times), f"FAIL: patient {seq['subject_id']} not sorted"

    label_cols = [k for k in sequences[0].keys() if k.startswith("label_")]
    assert len(label_cols) > 0, "FAIL: no label columns found in sequences"
    
    # all labels should be 0 or 1
    for label in label_cols:
        labels = set(s[label] for s in sequences)
        assert sequences[label].dtype in [int, np.int64, np.int32], f"FAIL: label column {label} has non-integer type {sequences[label].dtype}"
        assert labels.issubset({0, 1}), f"FAIL: unexpected labels {labels}"

    for seq in sequences:
        assert isinstance(seq["subject_id"], str)
        assert isinstance(seq["encounters"], list)
        for enc in seq["encounters"]:
            assert isinstance(enc["hadm_id"], int)
            assert isinstance(enc["icd_codes"], list)
            assert isinstance(enc["meds"], list)
            assert hasattr(enc["admittime"], "strftime")

    for seq in sequences[:50]:
        for enc in seq["encounters"]:
            for med in enc["meds"]:
                assert med == med.lower().strip(), f"FAIL: med '{med}' not normalized"
    
    print(f"  {len(sequences)} sequences validated")