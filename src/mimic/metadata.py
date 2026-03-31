"""
Patient metadata feature extraction (4-tier clinical vocabulary).

Tiers
-----
1. Summary features       (~10)
2. F-code indicators      (top_n_f_codes, dynamic)
3. Medication indicators   (top_n_meds individual + drug classes)
4. Temporal features       (~6)

Moved from src/analysis/clustering.py so that metadata can be computed
once during data extraction and reused by baselines, LASSO bridge,
cluster enrichment, and SAE analysis without recomputation.
"""

import argparse
from collections import Counter
from datetime import datetime

import numpy as np

from src.utils.io import PROCESSED_DIR, save_metadata, load_sequences
from src.utils.constants import DRUG_CLASSES

# =============================================================================
# Utility
# =============================================================================

def _has_drug(meds: list, drug_list: list) -> bool:
    """Return True if any medication name contains a drug substring."""
    joined = " ".join(meds).lower()
    return any(d in joined for d in drug_list)


def _parse_admittime(t):
    """Parse ISO datetime string to datetime object."""
    if isinstance(t, datetime):
        return t
    if isinstance(t, str):
        try:
            return datetime.fromisoformat(t)
        except (ValueError, TypeError):
            return None
    return None


def is_binary(col: np.ndarray) -> bool:
    """Check if a column contains only 0s and 1s."""
    unique = np.unique(col[~np.isnan(col)])
    return len(unique) <= 2 and all(v in (0.0, 1.0) for v in unique)


# =============================================================================
# Metadata extraction
# =============================================================================

def extract_metadata(
    sequences: list[dict],
    subject_ids: np.ndarray | None = None,
    top_n_f_codes: int = 20,
    top_n_meds: int = 25,
) -> tuple:
    """
    Build a rich metadata matrix from patient sequences.

    When subject_ids is provided, only patients present in that array are
    included (and deduplicated).
    When None, ALL patients are included.

    Tiers
    -----
    1. Summary features       (~10)
    2. F-code indicators      (top_n_f_codes, dynamic)
    3. Medication indicators  (top_n_meds individual + drug classes)
    4. Temporal features      (~6)

    Parameters
    ----------
    sequences     : list[dict] of patient dicts (each with "subject_id" and "encounters")
    subject_ids   : (N,) array of subject IDs (may repeat), or None for all
    top_n_f_codes : number of most frequent F-codes for Tier 2
    top_n_meds    : number of most frequent medications for Tier 3

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
        seen = set()
        ordered_pids = []
        for sid in subject_ids:
            s = str(sid)
            if s not in seen and s in patients:
                seen.add(s)
                ordered_pids.append(s)
    else:
        ordered_pids = list(patients.keys())

    # -- First pass: count F-code and med frequencies across cohort -----------
    f_code_counter = Counter()
    med_counter    = Counter()
    patient_raw    = {}

    for pid in ordered_pids:
        p    = patients[pid]
        encs = p["encounters"]
        all_icds = [str(c) for enc in encs for c in enc.get("icd_codes", [])]
        all_meds = [str(m).lower() for enc in encs for m in enc.get("meds", [])]
        f_codes  = [c for c in all_icds if c.upper().startswith("F")]

        # Count unique codes/meds per patient (not per encounter)
        for fc in set(c.upper() for c in f_codes):
            f_code_counter[fc] += 1
        for med in set(all_meds):
            med_counter[med] += 1

        # Support old "label" key and new "label_{N}" format (e.g. "label_90")
        _label = p.get("label")
        if _label is None:
            for _k in p:
                if _k.startswith("label_") and _k.replace("label_", "").isdigit():
                    _label = p[_k]
                    break
        patient_raw[pid] = dict(
            encs=encs, label=0 if _label is None else int(_label),
            all_icds=all_icds, all_meds=all_meds, f_codes=f_codes,
        )

    top_f_codes = [code for code, _ in f_code_counter.most_common(top_n_f_codes)]
    top_meds    = [med  for med,  _ in med_counter.most_common(top_n_meds)]

    # -- Build feature name list ----------------------------------------------
    tier1_names = [
        "label", "n_encounters", "n_unique_icd_codes", "n_unique_meds",
        "n_unique_f_codes", "mean_icd_count", "mean_med_count",
        "max_med_count", "f_code_ratio",
    ]
    tier2_names = [f"has_{fc}" for fc in top_f_codes]
    tier3_med_names   = [f"med_{med}" for med in top_meds]
    tier3_class_names = [f"has_{cls}" for cls in DRUG_CLASSES]
    tier4_names = [
        "mean_days_between_admissions", "min_days_between_admissions",
        "total_history_span_days", "admission_frequency",
        "last_encounter_has_f_code", "med_count_trend",
    ]
    feature_names = (tier1_names + tier2_names + tier3_med_names
                     + tier3_class_names + tier4_names)

    # -- Second pass: build feature matrix ------------------------------------
    rows = []
    for pid in ordered_pids:
        raw      = patient_raw[pid]
        encs     = raw["encs"]
        all_icds = raw["all_icds"]
        all_meds = raw["all_meds"]
        f_codes  = raw["f_codes"]
        n_enc    = len(encs)

        row = []

        # ---- Tier 1: Summary ------------------------------------------------
        icd_counts = [len(enc.get("icd_codes", [])) for enc in encs]
        med_counts = [len(enc.get("meds", []))       for enc in encs]
        encs_with_f = sum(
            1 for enc in encs
            if any(str(c).upper().startswith("F")
                   for c in enc.get("icd_codes", []))
        )
        row.extend([
            float(raw["label"]),
            float(n_enc),
            float(len(set(all_icds))),
            float(len(set(all_meds))),
            float(len(set(c.upper() for c in f_codes))),
            float(np.mean(icd_counts)) if icd_counts else 0.0,
            float(np.mean(med_counts)) if med_counts else 0.0,
            float(max(med_counts))     if med_counts else 0.0,
            float(encs_with_f / n_enc) if n_enc > 0  else 0.0,
        ])

        # ---- Tier 2: F-code indicators (dynamic top-N) ---------------------
        patient_f_set = set(c.upper() for c in f_codes)
        for fc in top_f_codes:
            row.append(1.0 if fc in patient_f_set else 0.0)

        # ---- Tier 3a: Individual med indicators (dynamic top-N) -------------
        patient_med_set = set(all_meds)
        for med in top_meds:
            row.append(1.0 if med in patient_med_set else 0.0)

        # ---- Tier 3b: Drug class indicators ---------------------------------
        for cls_name, drug_list in DRUG_CLASSES.items():
            row.append(1.0 if _has_drug(all_meds, drug_list) else 0.0)

        # ---- Tier 4: Temporal features --------------------------------------
        times = [_parse_admittime(enc.get("admittime")) for enc in encs]
        times = sorted(t for t in times if t is not None)

        if len(times) >= 2:
            gaps = [(times[i + 1] - times[i]).total_seconds() / 86400.0
                    for i in range(len(times) - 1)]
            span = (times[-1] - times[0]).total_seconds() / 86400.0
            row.extend([
                float(np.mean(gaps)),
                float(np.min(gaps)),
                float(span),
                float(n_enc / (span / 365.25)) if span > 0 else 0.0,
            ])
        else:
            row.extend([0.0, 0.0, 0.0, 0.0])

        # last_encounter_has_f_code
        last_has_f = (
            any(str(c).upper().startswith("F")
                for c in encs[-1].get("icd_codes", []))
            if encs else False
        )
        row.append(1.0 if last_has_f else 0.0)

        # med_count_trend (linear slope of med count over encounters)
        # manual OLS slope - avoids np.polyfit (LAPACK dependency issues)
        mc = np.array(med_counts, dtype=float)
        if len(mc) >= 2 and mc.std() > 0:
            x = np.arange(len(mc), dtype=float)
            xm, ym = x.mean(), mc.mean()
            slope = float(np.sum((x - xm) * (mc - ym)) / np.sum((x - xm) ** 2))
        else:
            slope = 0.0
        row.append(slope)

        rows.append(row)

    metadata    = np.array(rows, dtype=np.float64)
    patient_ids = np.array(ordered_pids, dtype=str)

    assert metadata.shape == (len(ordered_pids), len(feature_names)), \
        f"Shape mismatch: {metadata.shape} vs ({len(ordered_pids)}, {len(feature_names)})"

    return metadata, feature_names, patient_ids


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract metadata features from sequences.jsonl")
    parser.add_argument(
        "--sequences", type=str,
        default=str(PROCESSED_DIR / "sequences.jsonl"),
        help="Path to sequences.jsonl")
    parser.add_argument(
        "--output", type=str,
        default=str(PROCESSED_DIR),
        help="Output directory for metadata files")
    args = parser.parse_args()

    print("=" * 60)
    print("Metadata Feature Extraction")
    print("=" * 60)

    sequences = load_sequences(path=args.sequences)
    metadata, feature_names, patient_ids = extract_metadata(sequences, subject_ids=None)

    print(f"  Patients:  {len(patient_ids)}")
    print(f"  Features:  {len(feature_names)}")

    save_metadata(metadata, feature_names, patient_ids, path=args.output)
