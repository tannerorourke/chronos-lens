"""
Patient metadata feature extraction.

A baseline interpretability approach would entail utilizing (in the case of
clinical health records) utilizing the raw input tokens (a bag of ICD codes
and medication names) to perform interpretability (see inspect_sae_features()
in src/analysis/sae_analysis.py).

Metadata extraction is the additional layer we add on top. These can be defined as
derived features that summarize higher-order patterns in our data, that we KNOW
to be relevant but are NOT directly present in the data. We utilize these to
both to test whether the geometry captures higher-order clinical patterns beyond
what raw token frequencies explain, and to provide a more robust basis for feature
identification.

These features can be applied to any domain.

Summary
-----
Summarative features    (~8)
F-code indicators       (top_n_f_codes, dynamic)
Medication indicators   (top_n_meds individual + drug classes)
Temporal features       (6)
Escalation features     (10) - requires labels.py to have run first
Trajectory features     (3)
"""

import argparse
from collections import Counter
from typing import Callable

import numpy as np

from src.utils.io import DATA_DIR, save_metadata, load_sequences
from src.utils.constants import DRUG_CLASSES, ESCALATION_CRITERIA, SEVERITY_LOOKUP
from src.mimic.helper import parse_dt, has_drug_in_class


def _preprocess(patient: dict) -> dict:
    """Pre-compute encounter-level aggregates per patient """
    encs = patient["encounters"]
    all_icds = [str(c) for enc in encs for c in enc.get("icd_codes", [])]
    all_meds = [str(m).lower() for enc in encs for m in enc.get("meds", [])]
    f_codes = [c for c in all_icds if c.upper().startswith("F")]
    return {**patient, "all_icds": all_icds, "all_meds": all_meds, "f_codes": f_codes}


# =============================================================================
# =============================================================================

def _pat_summary(patient: dict) -> list[tuple[str, float]]:
    """Summary statistics over the encounter sequence."""
    encs = patient["encounters"]
    all_icds = patient["all_icds"]
    all_meds = patient["all_meds"]
    f_codes = patient["f_codes"]
    n_enc = len(encs)

    icd_counts = [len(enc.get("icd_codes", [])) for enc in encs]
    med_counts = [len(enc.get("meds", [])) for enc in encs]
    encs_with_f = sum(
        1 for enc in encs
        if any(str(c).upper().startswith("F") for c in enc.get("icd_codes", []))
    )

    return [
        ("n_encounters",       float(n_enc)),
        ("n_unique_icd_codes", float(len(set(all_icds)))),
        ("n_unique_meds",      float(len(set(all_meds)))),
        ("n_unique_f_codes",   float(len(set(c.upper() for c in f_codes)))),
        ("mean_icd_count",     float(np.mean(icd_counts)) if icd_counts else 0.0),
        ("mean_med_count",     float(np.mean(med_counts)) if med_counts else 0.0),
        ("max_med_count",      float(max(med_counts)) if med_counts else 0.0),
        ("f_code_ratio",       float(encs_with_f / n_enc) if n_enc > 0 else 0.0),
    ]

# =============================================================================
# Binary Indicators
# =============================================================================

def _pat_fcode_freq(patient: dict, top_f_codes: list[str]) -> list[tuple[str, float]]:
    patient_f_set = set(c.upper() for c in patient["f_codes"])
    return [
        ( f"has_{fc}", 1.0 if fc in patient_f_set else 0.0 ) 
        for fc in top_f_codes
    ]
    
def _pat_med_freq(patient: dict, top_meds: list[str]) -> list[tuple[str, float]]:
    patient_med_set = set(patient["all_meds"])
    return [
        ( f"med_{med}", 1.0 if med in patient_med_set else 0.0 ) 
        for med in top_meds
    ]

def _pat_drug_cls_presence(patient: dict) -> list[tuple[str, float]]:
    all_meds = patient["all_meds"]
    return [
        (f"has_{cls}", 1.0 if has_drug_in_class(all_meds, drug_list) else 0.0)
        for cls, drug_list in DRUG_CLASSES.items()
    ]


# =============================================================================
# =============================================================================

def _pat_temporal_patterns(patient: dict) -> list[tuple[str, float]]:
    """Temporal patterns across the encounter sequence."""
    encs = patient["encounters"]
    n_enc = len(encs)

    times = [parse_dt(enc.get("admittime")) for enc in encs]
    times = sorted(t for t in times if t is not None)

    if len(times) >= 2:
        gaps = [(times[i + 1] - times[i]).total_seconds() / 86400.0
                for i in range(len(times) - 1)]
        span = (times[-1] - times[0]).total_seconds() / 86400.0
        mean_gap = float(np.mean(gaps))
        min_gap = float(np.min(gaps))
        freq = float(n_enc / (span / 365.25)) if span > 0 else 0.0
    else:
        mean_gap = min_gap = span = freq = 0.0

    last_has_f = (
        any(str(c).upper().startswith("F") for c in encs[-1].get("icd_codes", []))
        if encs else False
    )

    # med_count_trend (linear slope via manual OLS)
    med_counts = np.array([len(enc.get("meds", [])) for enc in encs], dtype=float)
    if len(med_counts) >= 2 and med_counts.std() > 0:
        x = np.arange(len(med_counts), dtype=float)
        xm, ym = x.mean(), med_counts.mean()
        slope = float(np.sum((x - xm) * (med_counts - ym)) / np.sum((x - xm) ** 2))
    else:
        slope = 0.0

    return [
        ("mean_days_between_admissions", mean_gap),
        ("min_days_between_admissions",  min_gap),
        ("total_history_span_days",      span),
        ("admission_frequency",          freq),
        ("last_encounter_has_f_code",    1.0 if last_has_f else 0.0),
        ("med_count_trend",              slope),
    ]


# =============================================================================
# =============================================================================

def _pat_escalation(patient: dict) -> list[tuple[str, float]]:
    """Escalation labels and criteria indicators. """
    label_esc = patient.get("label_escalation", 0)
    per_enc = patient.get("label_escalation_per_enc", [])
    criteria = set(patient.get("escalation_criteria_fired", []))
    n_enc = len(patient["encounters"])

    n_esc_events = sum(per_enc)
    first_pos = 0.0
    if n_esc_events > 0 and n_enc > 0:
        first_idx = next(i for i, v in enumerate(per_enc) if v)
        first_pos = first_idx / n_enc

    esc_rate = 0.0
    if n_esc_events > 0 and n_enc > 1:
        first_idx = next(i for i, v in enumerate(per_enc) if v)
        remaining = per_enc[first_idx + 1:]
        if remaining:
            esc_rate = sum(remaining) / len(remaining)

    pairs: list[tuple[str, float]] = [
        ("label_escalation",        float(label_esc)),
        ("n_escalation_events",     float(n_esc_events)),
        ("first_escalation_position", first_pos),
        ("escalation_rate",         esc_rate),
    ]
    for criterion in ESCALATION_CRITERIA:
        pairs.append((f"has_{criterion}", 1.0 if criterion in criteria else 0.0))

    return pairs


# =============================================================================
# =============================================================================

def _pat_trajectory(patient: dict) -> list[tuple[str, float]]:
    """Trajectory features capturing diagnostic broadening over time."""
    encs = patient["encounters"]

    seen_blocks: set[str] = set()
    growth = 0
    max_sev = 0

    for i, enc in enumerate(encs):
        # For subcategory tracking: use icd_codes (truncated to 3-char)
        icds = [str(c).upper() for c in enc.get("icd_codes", [])]
        f_subcats = {c[:3] for c in icds if c.startswith("F")}

        for subcat in f_subcats:
            if i > 0 and subcat not in seen_blocks:
                growth += 1
            seen_blocks.add(subcat)

        # For severity: use icd_codes_full if available, fall back to icd_codes
        full_icds = enc.get("icd_codes_full") or enc.get("icd_codes", [])
        for code in full_icds:
            code_upper = str(code).upper()
            if code_upper.startswith("F"):
                sev = SEVERITY_LOOKUP.get(code_upper, 0)
                if sev > 0:
                    max_sev = max(max_sev, sev)

    return [
        ("n_unique_f_blocks", float(len(seen_blocks))),
        ("f_block_growth",    float(growth)),
        ("max_f_severity",    float(max_sev)),
    ]


# =============================================================================
# Metadata extraction (main)
# =============================================================================

def extract_metadata(
    sequences: list[dict],
    subject_ids: np.ndarray | None = None,
    top_n_f_codes: int = 20,
    top_n_meds: int = 25,
    label_metadata: dict = {}
) -> tuple:
    """
    Build a rich metadata matrix from patient sequences. When subject_ids is
    provided, only patients present in that array are included (and deduplicated).
    When None, ALL patients are included.

    Parameters
    ----------
    sequences     : list[dict] of patient dicts (each with "subject_id" and "encounters")
    subject_ids   : (N,) array of subject IDs (may repeat), or None for all
    top_n_f_codes : number of most frequent F-codes
    top_n_meds    : number of most frequent medications

    Returns
    -------
    metadata      : (n_patients, n_features) float64
    feature_names : list[str]
    patient_ids   : (n_patients,) str array of unique subject IDs (row order)
    """
    print("\nExtracting metadata features...")

    patients = {str(s["subject_id"]): s for s in sequences}

    # Deduplicate subject_ids, preserve first-seen order, skip missing
    if subject_ids is not None:
        seen: set[str] = set()
        ordered_pids: list[str] = []
        for sid in subject_ids:
            s = str(sid)
            if s not in seen and s in patients:
                seen.add(s)
                ordered_pids.append(s)
    else:
        ordered_pids = list(patients.keys())

    # -- First pass: preprocess patients aggregates + count cohort frequencies
    preprocessed: dict[str, dict] = {}
    f_code_counter: Counter = Counter()
    med_counter: Counter = Counter()

    for pid in ordered_pids:
        ctx = _preprocess(patients[pid])
        preprocessed[pid] = ctx
        for fc in set(c.upper() for c in ctx["f_codes"]):
            f_code_counter[fc] += 1
        for med in set(ctx["all_meds"]):
            med_counter[med] += 1

    top_f_codes = [code for code, _ in f_code_counter.most_common(top_n_f_codes)]
    top_meds = [med for med, _ in med_counter.most_common(top_n_meds)]

    # -- Tier function list (cohort-level params bound via closures)
    tiers: list[tuple[str, Callable]] = [
        ("summary",    _pat_summary),
        ("f-code",     lambda ctx: _pat_fcode_freq(ctx, top_f_codes)),
        ("med",        lambda ctx: _pat_med_freq(ctx, top_meds)),
        ("drug-class", _pat_drug_cls_presence),
        ("temporal",   _pat_temporal_patterns),
        ("escalation", _pat_escalation),
        ("trajectory", _pat_trajectory),
    ]

    # -- Feature names (derived from first patient)
    first_ctx = preprocessed[ordered_pids[0]]
    feature_names: list[str] = []
    tier_sizes: list[tuple[str, int]] = []
    for tier_name, fn in tiers:
        pairs = fn(first_ctx)
        tier_sizes.append((tier_name, len(pairs)))
        feature_names.extend(name for name, _ in pairs)

    # -- Second pass: build feature matrix -----------------------------------
    rows: list[list[float]] = []
    for pid in ordered_pids:
        ctx = preprocessed[pid]
        row: list[float] = []
        for _, fn in tiers:
            row.extend(val for _, val in fn(ctx))
        rows.append(row)

    metadata = np.array(rows, dtype=np.float64)
    patient_ids = np.array(ordered_pids, dtype=str)

    assert metadata.shape == (len(ordered_pids), len(feature_names)), \
        f"Shape mismatch: {metadata.shape} vs ({len(ordered_pids)}, {len(feature_names)})"

    # -- Summary --------------------------------------------------------------
    tier_str = " + ".join(f"{n} {name}" for name, n in tier_sizes)
    print(f"    Patients:  {len(patient_ids)}")
    print(f"    Features:  {len(feature_names)} ({tier_str})")

    esc_col = feature_names.index("label_escalation")
    n_esc = int(metadata[:, esc_col].sum())
    print(f"    Escalation: {n_esc} positive "
          f"({100 * n_esc / len(patient_ids):.1f}%)")

    for fname in ["n_unique_f_blocks", "f_block_growth", "max_f_severity"]:
        col = feature_names.index(fname)
        vals = metadata[:, col]
        print(f"    {fname:20s}: mean={vals.mean():.2f}, median={np.median(vals):.1f}")
    
    return metadata, feature_names, patient_ids
