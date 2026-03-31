"""
Clinical label computation for Chronos-Lens patient sequences.

compute_labels() is the single entry point. It computes ALL labels and
attaches them to each patient dict, overwriting any pre-existing label fields
while preserving "subject_id" and "encounters".

Labels produced
---------------
label_30d                  : int   1 if any readmission within 30 days carries
                                   an F-code (mood-disorder) diagnosis
label_escalation           : int   1 if any clinical escalation event in the sequence
label_escalation_per_enc   : list[int]  per-encounter flag (first encounter always 0)
escalation_criteria_fired  : list[str]  which escalation criteria triggered
next_enc_icd_blocks        : list[list[str]]  per-encounter ICD-10 chapter letters
                                   (first char of each ICD-10 code) present in the
                                   next encounter. Last encounter is always [].
                                   ICD-9 codes (numeric prefix) are excluded.

Escalation criteria
-------------------
new_subcategory   : An F-code subcategory (3-char) appears with no prior codes in
                    that subcategory (always escalation regardless of severity).
severity_increase : A new code within a known subcategory has positive severity >
                    max positive severity in prior encounters for that subcategory.
new_specifier     : A new code in a known subcategory has severity == -1, is not a
                    remission code, and has not been seen before.
f32_to_f33        : F33.x appears for the first time after F32.x was seen (single ->
                    recurrent depression). Fires even if severity is equal or lower.
med_initiation    : First psychiatric medication when none existed in any prior enc.
new_drug_class    : A psychiatric drug class not seen in any prior encounter appears.

Does NOT fire for:
    - Codes with severity == 0 (unspecified/NOS)
    - Codes with severity > 0 but <= prior max (plateau or improvement)
    - Remission codes: F30.3, F30.4, F31.7x, F32.4, F32.5, F33.40-F33.42
"""

import numpy as np
from collections import Counter
from datetime import datetime, timedelta

from src.utils.constants import (
    ICD10_F_CODES, DRUG_CLASSES, _PSYCH_CLASSES, _REMISSION_CODES
)

# =============================================================================
# Module constants (built once at import)
# =============================================================================

def _build_severity_lookup() -> dict[str, int]:
    """Flatten nested ICD10_F_CODES into {leaf_code: severity}."""
    lookup: dict[str, int] = {}

    def _walk(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v)
            else:
                lookup[k] = v

    _walk(ICD10_F_CODES)
    return lookup

_SEVERITY: dict[str, int] = _build_severity_lookup()


# =============================================================================
# Private helpers
# =============================================================================

def _parse_dt(t) -> datetime | None:
    """Parse a datetime from a Timestamp, datetime, or ISO string."""
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


def _has_label_dx(enc: dict, label_prefix: str) -> bool:
    """True if any icd_codes entry starts with label_prefix (case-insensitive)."""
    prefix = label_prefix.upper()
    return any(str(c).upper().startswith(prefix) for c in enc.get("icd_codes", []))


def _compute_readmission_label(
    encs: list[dict], readmission_days: int, label_prefix: str
) -> int:
    """
    1 if any readmission within readmission_days has a label_prefix ICD code; 0 otherwise.
    Encounters must be sorted by admittime (ascending).
    """
    n = len(encs)
    for i in range(n - 1):
        dt_i = _parse_dt(encs[i].get("dischtime"))
        if dt_i is None:
            continue
        window_end = dt_i + timedelta(days=readmission_days)
        for j in range(i + 1, n):
            dt_j = _parse_dt(encs[j].get("admittime"))
            if dt_j is None or dt_j > window_end:
                break
            if _has_label_dx(encs[j], label_prefix):
                return 1
    return 0


def _get_f_codes_full(enc: dict) -> list[str]:
    """Full uppercased F-codes from an encounter (prefers icd_codes_full)."""
    raw = enc.get("icd_codes_full") or enc.get("icd_codes", [])
    return [c.upper() for c in raw if str(c).upper().startswith("F")]


def _get_psych_classes(meds: list[str]) -> set[str]:
    """Psychiatric drug class names present in the medication list."""
    joined = " ".join(meds).lower()
    return {
        cls
        for cls in _PSYCH_CLASSES
        if any(drug in joined for drug in DRUG_CLASSES[cls])
    }


def _check_escalation(
    f_codes: list[str],
    meds: list[str],
    prior_subcats: dict[str, int],
    prior_f_codes: set[str],
    prior_drug_classes: set[str],
    has_prior_psych_meds: bool,
) -> set[str]:
    """Return the set of escalation criterion names that fire for this encounter."""
    criteria: set[str] = set()

    for code in f_codes:
        subcat = code[:3]
        is_new_subcat = subcat not in prior_subcats

        if is_new_subcat:
            criteria.add("new_subcategory")

        if subcat == "F33" and is_new_subcat and "F32" in prior_subcats:
            criteria.add("f32_to_f33")

        if not is_new_subcat and code not in prior_f_codes:
            if code in _REMISSION_CODES:
                continue
            sev = _SEVERITY.get(code)
            prior_max = prior_subcats[subcat]
            if sev is None or sev == 0:
                pass
            elif sev == -1:
                criteria.add("new_specifier")
            else:
                if sev > prior_max:
                    criteria.add("severity_increase")

    enc_classes = _get_psych_classes(meds)
    if enc_classes and not has_prior_psych_meds:
        criteria.add("med_initiation")
    if has_prior_psych_meds and (enc_classes - prior_drug_classes):
        criteria.add("new_drug_class")

    return criteria


def _update_state(
    f_codes: list[str],
    meds: list[str],
    prior_subcats: dict[str, int],
    prior_f_codes: set[str],
    prior_drug_classes: set[str],
) -> bool:
    """Update prior state; returns True if this encounter had psychiatric meds."""
    prior_f_codes.update(f_codes)
    for code in f_codes:
        subcat = code[:3]
        if subcat not in prior_subcats:
            prior_subcats[subcat] = 0
        sev = _SEVERITY.get(code)
        if sev is not None and sev > 0:
            prior_subcats[subcat] = max(prior_subcats[subcat], sev)

    enc_classes = _get_psych_classes(meds)
    prior_drug_classes.update(enc_classes)
    return bool(enc_classes)


# =============================================================================
# Public API
# =============================================================================

def _compute_next_enc_icd_blocks(encs: list[dict]) -> list[list[str]]:
    """
    For each encounter i, return sorted unique ICD-10 chapter letters present
    in encounter i+1 (using icd_codes). Last encounter returns [].
    ICD-9 codes (numeric first character) are excluded.
    """
    result: list[list[str]] = []
    for i in range(len(encs)):
        if i == len(encs) - 1:
            result.append([])
        else:
            next_codes = encs[i + 1].get("icd_codes", [])
            chapters = sorted({
                c[0].upper()
                for c in next_codes
                if c and c[0].isalpha()
            })
            result.append(chapters)
    return result


def compute_labels(
    sequences: list[dict],
    label_prefix: str = "F",
    compute_escalation_labels: bool = True,
    compute_next_enc_labels: bool = True,
) -> list[dict]:
    """
    Compute all clinical labels and attach them to each patient dict.

    Overwrites any pre-existing label fields. Preserves "subject_id" and
    "encounters" (and any other non-label keys).

    Parameters
    ----------
    sequences               : list of patient dicts, each with an "encounters" list.
                              Encounters need "icd_codes", "admittime", "dischtime",
                              "meds". Uses "icd_codes_full" for escalation if present.
    label_prefix            : ICD-10 prefix for positive readmission (default "F")
    compute_escalation_labels : compute escalation labels
    compute_next_enc_labels : compute next_enc_icd_blocks

    Returns
    -------
    sequences : same list, mutated with label fields.
    """
    print("\nComputing labels...")

    criteria_counter: Counter = Counter()
    first_esc_positions: list[int] = []
    n_pos_30d = 0
    # for next-enc stats
    chapter_counter: Counter = Counter()
    chapters_per_enc: list[int] = []

    for patient in sequences:
        encs = patient["encounters"]

        # Remove stale label keys from old pipeline runs
        patient.pop("label", None)
        for k in [k for k in list(patient) if k.startswith("label_") and k.replace("label_", "").isdigit()]:
            del patient[k]

        # ---- 30-day readmission label ----------------------------------------
        patient["label_30d"] = _compute_readmission_label(encs, 30, label_prefix)
        if patient["label_30d"]:
            n_pos_30d += 1

        # ---- Escalation labels ------------------------------------------------
        if compute_escalation_labels:
            n = len(encs)
            per_enc: list[int] = [0] * n
            all_criteria: set[str] = set()

            prior_subcats: dict[str, int] = {}
            prior_f_codes: set[str] = set()
            prior_drug_classes: set[str] = set()
            has_prior_psych_meds: bool = False

            for i, enc in enumerate(encs):
                f_codes = _get_f_codes_full(enc)
                meds    = [m.lower() for m in enc.get("meds", [])]

                if i == 0:
                    had_meds = _update_state(
                        f_codes, meds, prior_subcats, prior_f_codes, prior_drug_classes)
                    has_prior_psych_meds = had_meds
                    continue

                fired = _check_escalation(
                    f_codes, meds,
                    prior_subcats, prior_f_codes, prior_drug_classes,
                    has_prior_psych_meds,
                )
                if fired:
                    per_enc[i] = 1
                    all_criteria.update(fired)

                had_meds = _update_state(
                    f_codes, meds, prior_subcats, prior_f_codes, prior_drug_classes)
                has_prior_psych_meds = has_prior_psych_meds or had_meds

            patient_esc = 1 if any(per_enc) else 0
            patient["label_escalation"]          = patient_esc
            patient["label_escalation_per_enc"]  = per_enc
            patient["escalation_criteria_fired"] = sorted(all_criteria)

            if patient_esc:
                first_esc = next(i for i, v in enumerate(per_enc) if v)
                first_esc_positions.append(first_esc)
                for c in all_criteria:
                    criteria_counter[c] += 1

        # ---- Next-encounter ICD block labels ----------------------------------
        if compute_next_enc_labels:
            blocks = _compute_next_enc_icd_blocks(encs)
            patient["next_enc_icd_blocks"] = blocks
            # Collect stats (exclude last encounter which is always [])
            for blk in blocks[:-1]:
                chapters_per_enc.append(len(blk))
                for ch in blk:
                    chapter_counter[ch] += 1

    # ---- Summary --------------------------------------------------------------
    n_total = len(sequences)
    n_pos_esc = sum(1 for s in sequences if s.get("label_escalation") == 1)

    print(f"\n{'=' * 60}")
    print(f"Label Summary")
    print(f"  Patients: {n_total:,}")
    print(f"  {'label_30d':<28s} {n_pos_30d:,} ({100 * n_pos_30d / n_total:.1f}%)")
    if compute_escalation_labels:
        print(f"  {'label_escalation':<28s} {n_pos_esc:,} ({100 * n_pos_esc / n_total:.1f}%)")
        if criteria_counter:
            print(f"\n  Escalation criterion fire rates (patient-level, may overlap):")
            for criterion, count in sorted(criteria_counter.items(), key=lambda x: -x[1]):
                print(f"    {criterion:<30s} {count:,} ({100 * count / n_total:.1f}%)")
        if first_esc_positions:
            fep = np.array(first_esc_positions)
            print(f"\n  First-escalation encounter index:")
            print(f"    mean={fep.mean():.1f}, median={np.median(fep):.0f}, "
                  f"min={fep.min()}, max={fep.max()}")
    if compute_next_enc_labels and chapters_per_enc:
        cpe = np.array(chapters_per_enc)
        print(f"\n  next_enc_icd_blocks (transitions only, last enc excluded):")
        print(f"    unique chapters : {len(chapter_counter)}")
        print(f"    chapters/enc    : mean={cpe.mean():.2f}, median={np.median(cpe):.0f}, "
              f"min={cpe.min()}, max={cpe.max()}")
        print(f"    top chapters    :", ", ".join(
            f"{ch}={cnt:,}" for ch, cnt in chapter_counter.most_common(10)
        ))
    print(f"{'=' * 60}")

    return sequences
