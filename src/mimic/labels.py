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
from datetime import timedelta

from src.utils.constants import SEVERITY_LOOKUP, REMISSION_CODES
from src.mimic.helper import (
    parse_dt,
    encounter_has_prefix,
    get_encounter_f_codes,
    get_psych_drug_classes)
    
# =============================================================================
# Readmission
# =============================================================================

def _compute_readmission_label(
    encs: list[dict], readmission_days: int, label_prefix: str
) -> int:
    """
    1 if any readmission within readmission_days has a label_prefix ICD code; 0 otherwise.
    Encounters must be sorted by admittime (ascending).
    """
    n = len(encs)
    for i in range(n - 1):
        dt_i = parse_dt(encs[i].get("dischtime"))
        if dt_i is None:
            continue
        window_end = dt_i + timedelta(days=readmission_days)
        for j in range(i + 1, n):
            dt_j = parse_dt(encs[j].get("admittime"))
            if dt_j is None or dt_j > window_end:
                break
            if encounter_has_prefix(encs[j], label_prefix):
                return 1
    return 0


# =============================================================================
# Escalation
# =============================================================================

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
            if code in REMISSION_CODES:
                continue
            sev = SEVERITY_LOOKUP.get(code)
            prior_max = prior_subcats[subcat]
            if sev is None or sev == 0:
                pass
            elif sev == -1:
                criteria.add("new_specifier")
            else:
                if sev > prior_max:
                    criteria.add("severity_increase")

    enc_classes = get_psych_drug_classes(meds)
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
        sev = SEVERITY_LOOKUP.get(code)
        if sev is not None and sev > 0:
            prior_subcats[subcat] = max(prior_subcats[subcat], sev)

    enc_classes = get_psych_drug_classes(meds)
    prior_drug_classes.update(enc_classes)
    return bool(enc_classes)


# =============================================================================
# Next encounter ICD block
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

# =============================================================================
# =============================================================================

def compute_labels(
    sequences: list[dict],
    label_prefix: str = "F",
    COMPUTE_ESCALATION_LABEL: bool = True,
    COMPUTE_NEXT_ENC_LABEL: bool = True,
    COMPUTE_READMISSION_LABEL: bool = True
) -> list[dict]:
    """
    Compute all clinical labels and attach them to each patient dict.

    Overwrites any pre-existing label fields. Preserves "subject_id" and
    "encounters" (and any other non-label keys).

    Parameters
    ----------
    sequences       : list of patient dicts, each with an "encounters" list.
                      Encounters need "icd_codes", "admittime", "dischtime",
                      "meds". Uses "icd_codes_full" for escalation if present.
    label_prefix    : ICD-10 prefix for positive readmission (default "F")

    Returns
    -------
    sequences : same list, mutated with label fields.
    """
    print("\nComputing labels..")

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
        if COMPUTE_READMISSION_LABEL:
            patient["label_30d"] = _compute_readmission_label(encs, 30, label_prefix)
            if patient["label_30d"]:
                n_pos_30d += 1

        # ---- Escalation labels ------------------------------------------------
        if COMPUTE_ESCALATION_LABEL:
            n = len(encs)
            per_enc: list[int] = [0] * n
            all_criteria: set[str] = set()

            prior_subcats: dict[str, int] = {}
            prior_f_codes: set[str] = set()
            prior_drug_classes: set[str] = set()
            has_prior_psych_meds: bool = False

            for i, enc in enumerate(encs):
                f_codes = get_encounter_f_codes(enc, full=True)
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
        if COMPUTE_NEXT_ENC_LABEL:
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

    print(f"-- Label Summary --")
    print(f"    Patients: {n_total:,}")
    
    if COMPUTE_READMISSION_LABEL:
        print(f"    {'label_30d':<28s} {n_pos_30d:,} ({100 * n_pos_30d / n_total:.1f}% pos)")
    
    if COMPUTE_ESCALATION_LABEL:
        print(f"    {'label_escalation':<28s} {n_pos_esc:,} ({100 * n_pos_esc / n_total:.1f}% pos)")
        if criteria_counter:
            print(f"    - Criterion fire rates (patient-level, may overlap):")
            for criterion, count in sorted(criteria_counter.items(), key=lambda x: -x[1]):
                print(f"      {criterion:<30s} {count:,} ({100 * count / n_total:.1f}%)")
        if first_esc_positions:
            fep = np.array(first_esc_positions)
            print(f"\n    First-escalation encounter index:")
            print(f"      mean={fep.mean():.1f}, median={np.median(fep):.0f}, "
                  f"min={fep.min()}, max={fep.max()}")
    
    if COMPUTE_NEXT_ENC_LABEL and chapters_per_enc:
        cpe = np.array(chapters_per_enc)
        print(f"    next_enc_icd_blocks (transitions only, last enc excluded):")
        print(f"      Unique Chapters : {len(chapter_counter)}")
        print(f"      Chapters/Enc    : mean={cpe.mean():.2f}, median={np.median(cpe):.0f}, "
                                        f"min={cpe.min()}, max={cpe.max()}")
        print(f"      Top Chapters    :", ", ".join(
                                        f"{ch}={cnt:,}" for ch, cnt in chapter_counter.most_common(8)))
    print(f"{'=' * 60}")

    return sequences
