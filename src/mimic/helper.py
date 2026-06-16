from datetime import datetime

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

