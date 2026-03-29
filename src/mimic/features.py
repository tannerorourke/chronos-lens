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
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np

from src.utils.io import PROCESSED_DIR, save_metadata


# =============================================================================
# Drug class definitions  (add new classes or members as needed)
# =============================================================================

DRUG_CLASSES = {
    # =========================================================================
    # OVERDOSE REVERSAL - singular administration (atomic event)
    "reversal": [
        "naloxone",
        "flumazenil",
    ],
    "ssri": [
        "sertraline",         # Zoloft
        "escitalopram",       # Lexapro
        "citalopram",         # Celexa
        "fluoxetine",         # Prozac
        "paroxetine",         # Paxil
        "fluvoxamine",        # Luvox, used for OCD
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : SNRIs
    "snri": [
        "duloxetine",         # Cymbalta
        "venlafaxine",        # Effexor
        "desvenlafaxine",     # Pristiq, active metabolite of venlafaxine
        "milnacipran",        # Savella, fibromyalgia
        "levomilnacipran",    # Fetzima
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : Other mechanisms
    "antidepressant_other": [
        "bupropion",          # Wellbutrin
        "mirtazapine",        # Remeron
        "trazodone",          # Desyrel
        "nefazodone",         # Serzone
        "vilazodone",         # Viibryd
        "vortioxetine",       # Trintellix
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : Tricyclics (TCAs), Common for pain, depression, neuropathy
    "tca": [
        "amitriptyline",      # Elavil
        "nortriptyline",      # Pamelor
        "desipramine",        # Norpramin
        "imipramine",         # Tofranil
        "doxepin",            # Sinequan - also used for sleep/itching
        "clomipramine",       # Anafranil - OCD
    ],
    # =========================================================================
    # ANTIDEPRESSANTS : MAOIs (rare but present)
    "maoi": [
        "phenelzine",         # Nardil
        "tranylcypromine",    # Parnate
        "selegiline",         # Emsam / Eldepryl, also used for Parkinsons
    ],
    # =========================================================================
    # ANXIOLYTICS : Non-Benzodiazepine
    "anxiolytic": [
        "buspirone",          # Buspar
        "hydroxyzine",        # Vistaril/Atarax, very common for anxiety/itch
        "pregabalin",         # Lyrica, anxiety, nerve pain, fibromyalgia
        "gabapentin",         # Neurontin, off-label anxiety
    ],
    # =========================================================================
    # BENZODIAZEPINES
    "benzodiazepine": [
        "diazepam",           # Valium
        "clonazepam",         # Klonopin
        "lorazepam",          # Ativan
        "midazolam",          # Versed, ICU sedation
        "alprazolam",         # Xanax
        "chlordiazepoxide",   # Librium, alcohol withdrawal
        "oxazepam",           # Serax
        "temazepam",          # Restoril, sleep
        "triazolam",          # Halcion
        "clorazepate",        # Tranxene
    ],
    # =========================================================================
    # ANTIPSYCHOTICS : 1st Generation (Typical)
    "antipsychotic_typical": [
        "haloperidol",        # Haldol - extremely common in MIMIC (agitation, delirium)
        "chlorpromazine",     # Thorazine
        "fluphenazine",       # Prolixin
        "perphenazine",       # Trilafon
        "thiothixene",        # Navane
        "loxapine",           # Loxitane
        "pimozide",           # Orap - rare
        "prochlorperazine",   # Compazine - often used as antiemetic
    ],
    # =========================================================================
    # ANTIPSYCHOTICS : 2nd Generation (Atypical)
    "antipsychotic_atypical": [
        "quetiapine",         # Seroquel, most common (psychosis, sleep, bipolar)
        "olanzapine",         # Zyprexa
        "risperidone",        # Risperdal
        "aripiprazole",       # Abilify
        "ziprasidone",        # Geodon
        "clozapine",          # Clozaril, treatment-resistant schizophrenia
        "paliperidone",       # Invega
        "lurasidone",         # Latuda
        "brexpiprazole",      # Rexulti
        "cariprazine",        # Vraylar
        "asenapine",          # Saphris
    ],
    # =========================================================================
    # MOOD STABILIZERS / ANTICONVULSANTS
    "mood_stabilizer": [
        "lithium",            # Lithobid, Eskalith, very common
        "valproic acid",      # Depakote/Depakene, very common
        "valproate",          # alternate naming in prescriptions
        "divalproex",         # Depakote
        "carbamazepine",      # Tegretol
        "oxcarbazepine",      # Trileptal
        "lamotrigine",        # Lamictal
        "topiramate",         # Topamax, off-label mood, migraine
    ],
    # =========================================================================
    # ADHD
    "adhd": [
        "methylphenidate",    # Ritalin, Concerta
        "dexmethylphenidate", # Focalin
        "amphetamine",        # Adderall (mixed amphetamine salts)
        "dextroamphetamine",  # Dexedrine
        "lisdexamfetamine",   # Vyvanse
        "atomoxetine",        # Strattera, non-stimulant
        "guanfacine",         # Intuniv, non-stimulant
        "clonidine",          # Kapvay, also used for ADHD, very common
    ],
    # =========================================================================
    "cognitive": [
        "donepezil",          # Aricept
        "memantine",          # Namenda
        "rivastigmine",       # Exelon
        "galantamine",        # Razadyne
    ],
    # =========================================================================
    # MEDICATION ASSISTED TREATMENT (Alcohol, Opioid Dependence)
    "mat": [
        "buprenorphine",      # Suboxone (w/ naloxone), Subutex
        "methadone",          # Methadone
        "naltrexone",         # Vivitrol, ReVia
    ],
    # =========================================================================
    "smoking_cessation": [
        "nicotine",
        "varenicline",        # Chantix
    ],
    # =========================================================================
    # (pain / SUD risk tracking)
    "opioid": [               
        "hydrocodone",        # Vicodin, Norco
        "oxycodone",          # OxyContin, Percocet
        "codeine",            # Tylenol #3
        "fentanyl",           # Duragesic patch, IV (very common in ICU)
        "tramadol",           # Ultram
        "morphine",           # MS Contin, IV, extremely common
        "hydromorphone",      # Dilaudid, very common
        "meperidine",         # Demerol
        "methadone",          # NOTE: also in MAT; context determines category
        "alfentanil",         # Anesthesia
        "sufentanil",         # Anesthesia
        "remifentanil",       # Anesthesia
        "tapentadol",         # Nucynta
    ],
    # =========================================================================
    "sleep": [
        "zolpidem",           # Ambien
        "eszopiclone",        # Lunesta
        "suvorexant",         # Belsomra
        "ramelteon",          # Rozerem
        "melatonin",          # OTC but frequently appears
        # trazodone also used for sleep - listed under antidepressant_other
        # quetiapine low-dose for sleep - listed under antipsychotic_atypical
    ],
}


# =============================================================================
# Private utilities
# =============================================================================

def _has_drug(meds: list, drug_list: list) -> bool:
    """Return True if any medication name contains a drug substring."""
    joined = " ".join(meds).lower()
    return any(d in joined for d in drug_list)


def _load_sequences(path) -> dict:
    """Load sequences.jsonl -> dict mapping subject_id (str) -> record."""
    patients = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p
    return patients


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
    sequences_path,
    subject_ids: np.ndarray | None = None,
    top_n_f_codes: int = 20,
    top_n_meds: int = 25,
) -> tuple:
    """
    Goal: Build a rich metadata matrix from the data, sequences.jsonl.

    When subject_ids is provided, only patients present in that array are
    included (and deduplicated).  
    When None, ALL patients in the file are included - for use at data
    creation time to build the full cohort.

    Tiers
    -----
    1. Summary features       (~10)
    2. F-code indicators      (top_n_f_codes, dynamic)
    3. Medication indicators  (top_n_meds individual + drug classes)
    4. Temporal features      (~6)

    Parameters
    ----------
    sequences_path : path to sequences.jsonl
    subject_ids    : (N,) array of subject IDs (may repeat), or None for all
    top_n_f_codes  : number of most frequent F-codes for Tier 2
    top_n_meds     : number of most frequent medications for Tier 3

    Returns
    -------
    metadata      : (n_patients, n_features) float64
    feature_names : list[str]
    patient_ids   : (n_patients,) str array of unique subject IDs (row order)
    """
    patients = _load_sequences(sequences_path)

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

        patient_raw[pid] = dict(
            encs=encs, label=p.get("label", 0),
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

    metadata, feature_names, patient_ids = extract_metadata(
        args.sequences, subject_ids=None)

    print(f"  Patients:  {len(patient_ids)}")
    print(f"  Features:  {len(feature_names)}")

    save_metadata(metadata, feature_names, patient_ids, path=args.output)
