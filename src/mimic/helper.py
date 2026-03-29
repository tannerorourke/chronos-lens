import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import PARQUET_DIR


def save_dataset(
    sequences: list[dict], 
    out_dir: Path,
    min_encounters: int,
    readm_window_days: int,
):
    """
    Save sequences in (model-ready format)
      PROCESSED_DIR/
        sequences.jsonl    - one JSON object p/patient (primary format)
        sequences.pkl      - pickle backup (preserves datetime objects)
        metadata.json      - cohort stats, schema, label distribution
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- JSONL ---
    jsonl_path = out_dir / "sequences.jsonl"
    with open(jsonl_path, "w") as f:
        for seq in sequences:
            f.write(json.dumps({
                "subject_id": seq["subject_id"],
                "label": seq["label"],
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

    # --- data is so meta ---
    enc_counts = [len(s["encounters"]) for s in sequences]
    n_pos = sum(1 for s in sequences if s["label"] == 1)
    all_icd = set()
    all_meds = set()
    for s in sequences:
        for enc in s["encounters"]:
            all_icd.update(enc["icd_codes"])
            all_meds.update(enc["meds"])

    metadata = {
        "n_patients": len(sequences),
        "n_positive": n_pos,
        "n_negative": len(sequences) - n_pos,
        "positive_rate": round(n_pos / len(sequences), 4),
        "encounters_per_patient": {
            "mean": round(np.mean(enc_counts), 2),
            "median": int(np.median(enc_counts)),
            "min": int(min(enc_counts)),
            "max": int(max(enc_counts)),
        },
        "vocab_size_icd": len(all_icd),
        "vocab_size_meds": len(all_meds),
        "cohort_filters": {
            "cohort_definition": "readmission within window (no diagnosis requirement)",
            "label_definition": "code = F3x at readmission = 1, else = 0",
            "min_encounters": min_encounters,
            "readm_window_days": readm_window_days,
            "exclude_deceased_admissions": True,
        },
        "schema": {
            "subject_id": "str",
            "label": "int",
            "encounters[].hadm_id": "int",
            "encounters[].admittime": "ISO datetime string",
            "encounters[].dischtime": "ISO datetime string",
            "encounters[].icd_codes": "list[str]",
            "encounters[].meds": "list[str]",
        },
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {out_dir}/")
    print(f"  {jsonl_path.name:20s} {jsonl_path.stat().st_size / 1024:.0f} KB  (primary - for JEPA dataloader)")
    print(f"  {meta_path.name:20s} {meta_path.stat().st_size / 1024:.0f} KB  (cohort stats & schema)")


def save_parquets(admissions, patients, diagnoses, prescriptions) -> None:
    print(f"[save_parquets] Saving parquet's to cache...")
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "admissions":    admissions,
        "patients":      patients,
        "diagnoses":     diagnoses,
        "prescriptions": prescriptions,
    }
    
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



def validate_sequences(sequences: list[dict], readm_window_days: int):
    from datetime import timedelta

    print("\nSEQUENCE VALIDATION")
    print("=" * 60)

    assert len(sequences) > 0, "FAIL: no sequences produced"

    min_enc = min(len(s["encounters"]) for s in sequences)
    assert min_enc >= 3, f"FAIL: found sequence with {min_enc} encounters"

    # every patient must have at least one readmission pair within the window
    for seq in sequences:
        encs = seq["encounters"]
        has_readm = False
        for i in range(len(encs) - 1):
            window_end = encs[i]["dischtime"] + timedelta(days=readm_window_days)
            if encs[i + 1]["admittime"] <= window_end:
                has_readm = True
                break
        assert has_readm, (
            f"FAIL: patient {seq['subject_id']} has no readmission within {readm_window_days}d")

    for seq in sequences:
        times = [enc["admittime"] for enc in seq["encounters"]]
        assert times == sorted(times), f"FAIL: patient {seq['subject_id']} not sorted"

    labels = set(s["label"] for s in sequences)
    assert labels.issubset({0, 1}), f"FAIL: unexpected labels {labels}"

    # all F-codes should be exactly 3 chars
    # Non-F ICD-10 codes should retain dot notation
    for seq in sequences[:100]:
        for enc in seq["encounters"]:
            for code in enc["icd_codes"]:
                code_upper = code.upper()
                if code_upper.startswith("F"):
                    assert len(code_upper) == 3, (
                        f"FAIL: F-code '{code}' not truncated to 3 chars "
                        f"(patient {seq['subject_id']})")

    for seq in sequences:
        assert isinstance(seq["subject_id"], str)
        assert isinstance(seq["label"], int)
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