import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.io import PARQUET_DIR
from src.utils.constants import DRUG_CLASSES, PSYCH_CLASSES


# =============================================================================
# Clinical data helpers (labels.py, metadata.py, baselines.py, sae.py)
# =============================================================================

def parse_dt(t) -> datetime | None:
    """Parse datetime from None, datetime, pandas Timestamp, or ISO string."""
    if t is None:
        return None
    if isinstance(t, datetime):
        return t
    if hasattr(t, "to_pydatetime"):
        return t.to_pydatetime()
    try:
        return datetime.fromisoformat(str(t))
    except (ValueError, TypeError):
        return None


def get_patient_icds(patient: dict) -> list[str]:
    """All ICD codes across all encounters for a patient. Uppercased, deduplicated."""
    seen: set[str] = set()
    result: list[str] = []
    for enc in patient.get("encounters", []):
        for c in enc.get("icd_codes", []):
            s = str(c).upper()
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result


def get_patient_meds(patient: dict) -> list[str]:
    """All medications across all encounters for a patient. Lowercased, deduplicated."""
    seen: set[str] = set()
    result: list[str] = []
    for enc in patient.get("encounters", []):
        for m in enc.get("meds", []):
            s = str(m).lower()
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result


def get_encounter_f_codes(enc: dict, full: bool = False) -> list[str]:
    """F-codes from a single encounter. Uppercased.

    If full=True, uses icd_codes_full (for severity lookup). Otherwise uses icd_codes.
    """
    if full:
        raw = enc.get("icd_codes_full") or enc.get("icd_codes", [])
    else:
        raw = enc.get("icd_codes", [])
    return [str(c).upper() for c in raw if str(c).upper().startswith("F")]


def encounter_has_prefix(enc: dict, prefix: str) -> bool:
    """True if any ICD code in the encounter starts with prefix (case-insensitive)."""
    prefix_upper = prefix.upper()
    return any(str(c).upper().startswith(prefix_upper) for c in enc.get("icd_codes", []))


def has_drug_in_class(meds: list[str], drug_list: list[str]) -> bool:
    """True if any medication name contains a drug substring from drug_list."""
    joined = " ".join(meds).lower()
    return any(d in joined for d in drug_list)


def get_psych_drug_classes(meds: list[str]) -> set[str]:
    """Return set of psychiatric drug class names present in the medication list."""
    joined = " ".join(meds).lower()
    return {
        psy_cls
        for psy_cls in PSYCH_CLASSES
        if any(drug in joined for drug in DRUG_CLASSES[psy_cls])
    }


# =============================================================================
# Sequence data
# =============================================================================

def validate_sequences(sequences: list[dict], min_encounters: int):
    print("\nValidating sequences..")

    assert len(sequences) > 0, "FAIL: no sequences produced"

    min_enc = min(len(s["encounters"]) for s in sequences)
    assert min_enc >= min_encounters, f"FAIL: found sequence with {min_enc} encounters"
    
    nmax_enc = max(len(s["encounters"]) for s in sequences)
    assert nmax_enc <= min_encounters, f"FAIL: found sequence with {nmax_enc} encounters"

    for seq in sequences:
        times = [enc["admittime"] for enc in seq["encounters"]]
        assert times == sorted(times), f"FAIL: patient {seq['subject_id']} not sorted"

    label_cols = [k for k in sequences[0].keys() if k.startswith("label_")]
    assert len(label_cols) > 0, "FAIL: no label columns found in sequences"
    
    # all labels should be 0 or 1
    for label in label_cols:
        for s in sequences:
            val = s[label]
            if isinstance(val, list):
                assert all(v in (0, 1) for v in val), \
                    f"FAIL: {label} has non-binary values in patient {s['subject_id']}"
            else:
                assert val in (0, 1), \
                    f"FAIL: {label} has unexpected value {val} in patient {s['subject_id']}"

    for seq in sequences:
        assert isinstance(seq["subject_id"], (str, int)), \
            f"FAIL: subject_id must be str or int, got {type(seq['subject_id'])}"
        seq["subject_id"] = str(seq["subject_id"])
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
    print(f"{'=' * 60}")