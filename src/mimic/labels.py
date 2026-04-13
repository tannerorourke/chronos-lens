"""
Clinical label computation for Chronos-Lens patient sequences.

compute_labels() is the single entry point. It computes ALL labels and
attaches them to each patient dict, overwriting any pre-existing label fields
while preserving "subject_id" and "encounters".

Labels produced
---------------
label_30d: 1 if any readmission within 30 days carries an F-code (mood-disorder) diagnosis (int)  
label_30d_per_enc: per-encounter flag: 1 if any subsequent encounter within 30 days of this 
                   encounter'sdischarge carries an F-code diagnosis. Last encounter is 
                   always 0 (list[int])
label_escalation: 1 if any clinical escalation event in the sequence (int)
label_escalation_per_enc: per-encounter flag (first encounter always 0) (list[int])
escalation_criteria_fired: which escalation criteria triggered (list[str])
next_enc_icd_blocks: per-encounter ICD-10 chapter letters (first char of each ICD-10 code) 
                     present in the next encounter. Last encounter is always []. ICD-9 codes 
                     (numeric prefix) are excluded. (list[list[str]])

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

def _compute_readmission(
    encs: list[dict], 
    readmission_days: int, 
    label_prefix: str, 
) -> list[int]:
    """ Compute per-encounter n-day readmission label """
    n = len(encs)
    labels = [0] * n
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
                labels[i] = 1
                break
    return labels


# =============================================================================
# Escalation
# =============================================================================

def _check_enc_escalation(
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
    skip_labeling: bool = False
) -> tuple[list[dict], dict]:
    """
    Compute all labels and attach them to each patient dict. Overwrites any 
    pre-existing label fields. Preserves "subject_id" and "encounters" 
    (and any other non-label keys).

    Parameters
    ----------
    sequences: list of patient dicts, each with an "encounters" list.
                Encounters need "icd_codes", "admittime", "dischtime",
                "meds". Uses "icd_codes_full" for escalation if present.
    label_prefix: ICD-10 prefix for positive readmission (default "F")

    Returns
    -------
    sequences : same list mutated with label fields.
    seq_metadata : dict
    """
    if skip_labeling:
        print("\nSkipping labels..")
        return sequences, {}
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

        # ---- 30-day readmission label (any and per enc) ----------------------------------------
        patient["label_30d_per_enc"] = _compute_readmission(encs, 30, label_prefix)
        patient["label_30d"] = any([p == 1 for p in patient["label_30d_per_enc"]])

        # ---- Escalation labels ------------------------------------------------
        per_enc: list[int] = [0] * len(encs)
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

            fired = _check_enc_escalation(
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
        blocks = _compute_next_enc_icd_blocks(encs)
        patient["next_enc_icd_blocks"] = blocks
        # Collect stats (exclude last encounter which is always [])
        for blk in blocks[:-1]:
            chapters_per_enc.append(len(blk))
            for ch in blk:
                chapter_counter[ch] += 1

    # ---- Summary Metadata --------------------------------------------------------------
    print(f"-- Label Summary --")
    
    n_patients = len(sequences)
    enc_counts = [len(s["encounters"]) for s in sequences]
    n_pos_30d = sum(1 for s in sequences if s.get("label_30d") == 1)
    n_pos_esc = sum(1 for s in sequences if s.get("label_escalation") == 1)
    n_encs = sum(enc_counts)
    n_pos_30d_pe = sum( sum(s.get("label_30d_per_enc", [])) for s in sequences )
    n_pos_esc_pe = sum( sum(s.get("label_escalation_per_enc", [])) for s in sequences )
    
    print(f"    Patients: {n_patients:,}")
    print(f"    {'label_30d':<28s} {n_pos_30d:,} ({100 * n_pos_30d / n_patients:.1f}% pos)")
    print(f"    {'label_escalation':<28s} {n_pos_esc:,} ({100 * n_pos_esc / n_patients:.1f}% pos)")
    print(f"    Encounters: {n_encs:,}")
    print(f"    {'label_30d_per_enc':<28s} {n_pos_30d_pe:,} patients with >=1 pos ({100 * n_pos_30d_pe / n_encs:.1f}%)")
    print(f"    {'label_esc_per_enc':<28s} {n_pos_esc_pe:,} patients with >=1 pos ({100 * n_pos_esc_pe / n_encs:.1f}%)")
    
    seq_meta = {
        "n_patients": len(sequences),
        "n_positive_30d": n_pos_30d,
        "positive_rate_30d": round(n_pos_30d / n_patients, 4),
        "n_positive_esc": n_pos_esc,
        "positive_rate_esc": round(n_pos_esc / n_patients, 4),
        "tot_encounters": n_encs,
        "n_positive_30d_per_enc": n_pos_30d_pe,
        "positive_rate_30d_per_enc": round(n_pos_30d_pe / n_encs, 4),
        "n_positive_esc_per_enc": n_pos_esc_pe,
        "positive_rate_esc_per_enc": round(n_pos_esc_pe / n_encs, 4),
        "encounters_per_patient": {
            "mean": round(np.mean(enc_counts), 2),
            "median": int(np.median(enc_counts)),
            "min": int(min(enc_counts)),
            "max": int(max(enc_counts)),
        },
    }
    
    if criteria_counter:
        cr_stat = {}
        for k, c in sorted(criteria_counter.items(), key=lambda x: -x[1]):
            cr_stat[k] = { "count": c, "rate": round(100 * c / n_patients, 3) }
            print(f"      {k:<30s} {c:,} ({cr_stat[k]["rate"]:.1f}%)")
        seq_meta["criteria_counts"] = cr_stat
    
    if first_esc_positions:
        fep = np.array(first_esc_positions)
        fep_stat = f"mean={fep.mean()}, median={np.median(fep)}, min={fep.min()}, max={fep.max()}"
        print(f"\n    First-escalation encounter index:", fep_stat)
        seq_meta["first_esc_position_ix"] = fep_stat
    
    if chapters_per_enc:
        cpe = np.array(chapters_per_enc)
        tot = len(chapter_counter)
        cpe_stat = { "mean": cpe.mean(), "median": int(np.median(cpe)), "min": int(cpe.min()), "max": int(cpe.max()) }
        c_common = { ch: cnt for ch, cnt in chapter_counter.most_common(8) }
        print(f"    next_enc_icd_blocks (transitions only, last enc excluded):")
        print(f"      Unique Chapters : {tot}")
        print(f"      Chapters/Enc    : ", [f"{k}={v}" for k, v in cpe_stat.items()])
        print(f"      Top Chapters    :", [f"{k}={v}" for k, v in c_common.items()])
        seq_meta["icd_blocks"] = { "unique": tot, "chapters_per_enc": cpe_stat, "most_common": c_common }
        
    print(f"{'=' * 60}")

    return sequences, seq_meta
