"""
Partial labeling bridge — Tiers A (LASSO) and B (UMAP + HDBSCAN cluster
enrichment) for the geometric latent space analysis (thesis §5.5).

Connects the model's geometric structure back to clinical concepts using
a rich metadata vocabulary (~60-80 features across 4 tiers).  The richer
the vocabulary, the stronger the "no clinical match" claim for
unexplained geometric structure.

Tier A (LASSO on PCA axes):
    Linear bridge — regress metadata against PC scores.
    Assumes geometry is organised along linear axes.
    Unexplained variance = 1 − R².

Tier B (UMAP + HDBSCAN cluster enrichment):
    Nonlinear bridge — cluster UMAP embedding with HDBSCAN, compute
    enrichment of metadata features per cluster.  Clusters with no
    clear enrichment are the mislabeling problem made visible.

Downstream consumers
--------------------
  metadata_features.npy           → Step 4d (SAE)
  metadata_feature_names.json     → Step 4d (SAE)
  lasso/                          → Tier A results
  clusters/                       → Tier B results
"""

import json
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN

from src.utils.seed import SEED, get_rng
rng  = get_rng()

# =============================================================================
# Drug class definitions  (extensible — add new classes or members as needed)
# =============================================================================

DRUG_CLASSES = {
    # =========================================================================
    # OVERDOSE REVERSAL — singular administration (atomic event)
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
        "fluvoxamine",        # Luvox — present in MIMIC, used for OCD
    ],
    # =========================================================================
    # ANTIDEPRESSANTS — SNRIs
    "snri": [
        "duloxetine",         # Cymbalta
        "venlafaxine",        # Effexor
        "desvenlafaxine",     # Pristiq — active metabolite of venlafaxine
        "milnacipran",        # Savella — fibromyalgia, rare
        "levomilnacipran",    # Fetzima — rare but possible
    ],
    # =========================================================================
    # ANTIDEPRESSANTS — Other mechanisms
    "antidepressant_other": [
        "bupropion",          # Wellbutrin — NDRI
        "mirtazapine",        # Remeron — NaSSA
        "trazodone",          # Desyrel — commonly used for sleep
        "nefazodone",         # Serzone — rare, hepatotoxicity concerns
        "vilazodone",         # Viibryd — newer, less common
        "vortioxetine",       # Trintellix — newer, less common
    ],
    # =========================================================================
    # ANTIDEPRESSANTS — Tricyclics (TCAs)
    # Common in MIMIC for pain, depression, neuropathy
    "tca": [
        "amitriptyline",      # Elavil
        "nortriptyline",      # Pamelor
        "desipramine",        # Norpramin
        "imipramine",         # Tofranil
        "doxepin",            # Sinequan — also used for sleep/itching
        "clomipramine",       # Anafranil — OCD
    ],
    # =========================================================================
    # ANTIDEPRESSANTS — MAOIs (rare but present in MIMIC)
    "maoi": [
        "phenelzine",         # Nardil
        "tranylcypromine",    # Parnate
        "selegiline",         # Emsam (patch) / Eldepryl — also used for Parkinson's
    ],
    # =========================================================================
    # ANXIOLYTICS — Non-Benzodiazepine
    "anxiolytic": [
        "buspirone",          # Buspar
        "hydroxyzine",        # Vistaril/Atarax — very common in MIMIC for anxiety/itch
        "pregabalin",         # Lyrica — anxiety, nerve pain, fibromyalgia
        "gabapentin",         # Neurontin — off-label anxiety, very common in MIMIC
    ],
    # =========================================================================
    # BENZODIAZEPINES
    "benzodiazepine": [
        "diazepam",           # Valium
        "clonazepam",         # Klonopin
        "lorazepam",          # Ativan — extremely common in MIMIC
        "midazolam",          # Versed — ICU sedation
        "alprazolam",         # Xanax
        "chlordiazepoxide",   # Librium — alcohol withdrawal
        "oxazepam",           # Serax
        "temazepam",          # Restoril — sleep
        "triazolam",          # Halcion — rare
        "clorazepate",        # Tranxene — rare
    ],

    # =========================================================================
    # ANTIPSYCHOTICS — First Generation (Typical)
    # NOT in Synthea config but very common in MIMIC
    # =========================================================================
    "antipsychotic_typical": [
        "haloperidol",        # Haldol — extremely common in MIMIC (agitation, delirium)
        "chlorpromazine",     # Thorazine
        "fluphenazine",       # Prolixin
        "perphenazine",       # Trilafon
        "thiothixene",        # Navane
        "loxapine",           # Loxitane
        "pimozide",           # Orap — rare
        "prochlorperazine",   # Compazine — often used as antiemetic
    ],
    # =========================================================================
    # ANTIPSYCHOTICS — Second Generation (Atypical)
    "antipsychotic_atypical": [
        "quetiapine",         # Seroquel — very common (psychosis, sleep, bipolar)
        "olanzapine",         # Zyprexa — common in MIMIC
        "risperidone",        # Risperdal
        "aripiprazole",       # Abilify
        "ziprasidone",        # Geodon
        "clozapine",          # Clozaril — treatment-resistant schizophrenia
        "paliperidone",       # Invega
        "lurasidone",         # Latuda
        "brexpiprazole",      # Rexulti — newer
        "cariprazine",        # Vraylar — newer
        "asenapine",          # Saphris
    ],
    # =========================================================================
    # MOOD STABILIZERS / ANTICONVULSANTS
    "mood_stabilizer": [
        "lithium",            # Lithobid, Eskalith — very common
        "valproic acid",      # Depakote/Depakene — very common
        "valproate",          # alternate naming in prescriptions
        "divalproex",         # Depakote (divalproex sodium)
        "carbamazepine",      # Tegretol
        "oxcarbazepine",      # Trileptal
        "lamotrigine",        # Lamictal — common
        "topiramate",         # Topamax — off-label mood, migraine
    ],
    # =========================================================================
    # 
    "adhd": [
        "methylphenidate",    # Ritalin, Concerta
        "dexmethylphenidate", # Focalin
        "amphetamine",        # Adderall (mixed amphetamine salts)
        "dextroamphetamine",  # Dexedrine
        "lisdexamfetamine",   # Vyvanse
        "atomoxetine",        # Strattera — non-stimulant
        "guanfacine",         # Intuniv — non-stimulant
        "clonidine",          # Kapvay — also used for ADHD, very common
    ],
    # =========================================================================
    "cognitive": [
        "donepezil",          # Aricept
        "memantine",          # Namenda
        "rivastigmine",       # Exelon
        "galantamine",        # Razadyne
        # tacrine excluded — withdrawn from market, extremely unlikely in MIMIC
    ],
    # =========================================================================
    "mat": [
        "buprenorphine",      # Suboxone (w/ naloxone), Subutex
        "methadone",          # Methadone
        "naltrexone",         # Vivitrol, ReVia
    ],
    # =========================================================================
    "smoking_cessation": [
        "nicotine",           # NRT patches, gum, lozenge
        "varenicline",        # Chantix
    ],
    # =========================================================================
    "opioid": [               # (pain / SUD risk tracking)
        "hydrocodone",        # Vicodin, Norco
        "oxycodone",          # OxyContin, Percocet
        "codeine",            # Tylenol #3
        "fentanyl",           # Duragesic patch, IV (very common in MIMIC ICU)
        "tramadol",           # Ultram
        "morphine",           # MS Contin, IV — extremely common in MIMIC (not in Synthea)
        "hydromorphone",      # Dilaudid — very common in MIMIC (not in Synthea)
        "meperidine",         # Demerol — present in MIMIC
        "methadone",          # NOTE: also in MAT; context determines category
        "alfentanil",         # Anesthesia — present in MIMIC ICU
        "sufentanil",         # Anesthesia — present in MIMIC ICU
        "remifentanil",       # Anesthesia — present in MIMIC ICU
        "tapentadol",         # Nucynta — less common
    ],
    # =========================================================================
    "sleep": [
        "zolpidem",           # Ambien
        "eszopiclone",        # Lunesta
        "suvorexant",         # Belsomra
        "ramelteon",          # Rozerem
        "melatonin",          # OTC but frequently ordered in MIMIC
        # trazodone also used for sleep — listed under antidepressant_other
        # quetiapine low-dose for sleep — listed under antipsychotic_atypical
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
    """Load sequences.jsonl → dict mapping subject_id (str) → record."""
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


def _is_binary(col: np.ndarray) -> bool:
    """Check if a column contains only 0s and 1s."""
    unique = np.unique(col[~np.isnan(col)])
    return len(unique) <= 2 and all(v in (0.0, 1.0) for v in unique)


def _lasso_stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 100,
    subsample_frac: float = 0.5,
) -> tuple:
    """
    Bootstrap stability selection for LASSO.

    Returns
    -------
    sel_probs : (n_features,) selection probability [0, 1]
    ci_low    : (n_features,) 2.5th percentile of bootstrap coefficients
    ci_high   : (n_features,) 97.5th percentile of bootstrap coefficients
    """
    n_samples, n_features = X.shape
    sub_n   = max(4, int(n_samples * subsample_frac))
    counts  = np.zeros(n_features)
    coef_samples = []
    n_valid = 0

    for _ in range(n_iter):
        idx = rng.choice(n_samples, size=sub_n, replace=False)
        Xs, ys = X[idx], y[idx]
        if ys.std() < 1e-8:
            continue
        cv_folds = min(5, sub_n - 1)
        if cv_folds < 2:
            continue
        lasso = LassoCV(cv=cv_folds, random_state=SEED,
                        max_iter=5000, n_jobs=1)
        try:
            lasso.fit(Xs, ys)
            counts  += (np.abs(lasso.coef_) > 1e-10).astype(float)
            coef_samples.append(lasso.coef_.copy())
            n_valid += 1
        except Exception:
            pass

    sel_probs = counts / n_valid if n_valid > 0 else counts

    if coef_samples:
        coef_arr = np.array(coef_samples)
        ci_low  = np.percentile(coef_arr, 2.5, axis=0)
        ci_high = np.percentile(coef_arr, 97.5, axis=0)
    else:
        ci_low  = np.zeros(n_features)
        ci_high = np.zeros(n_features)

    return sel_probs, ci_low, ci_high


# =============================================================================
# Public utilities
# =============================================================================

def pool_to_patients(
    values: np.ndarray,
    subject_ids: np.ndarray,
    patient_ids: np.ndarray,
) -> np.ndarray:
    """Mean-pool sample-level values (N, ...) to patient level (P, ...)."""
    sid_str = np.asarray(subject_ids, dtype=str)
    return np.vstack([
        values[sid_str == pid].mean(axis=0)
        for pid in patient_ids
    ])


def broadcast_to_samples(
    metadata: np.ndarray,
    patient_ids: np.ndarray,
    subject_ids: np.ndarray,
) -> np.ndarray:
    """Broadcast patient-level metadata (P, F) to sample-level (N, F)."""
    pid_to_idx = {str(pid): i for i, pid in enumerate(patient_ids)}
    indices = np.array([pid_to_idx[str(sid)] for sid in subject_ids])
    return metadata[indices]


# =============================================================================
# Step 5a — Extract Metadata Features  (4 tiers)
# =============================================================================

def extract_metadata(
    sequences_path,
    subject_ids: np.ndarray | None = None,
    top_n_f_codes: int = 20,
    top_n_meds: int = 25,
) -> tuple:
    """
    Build rich metadata feature matrix from sequences.jsonl.

    When subject_ids is provided, only patients present in that array are
    included (and deduplicated).  When None, ALL patients in the file are
    included — use this at data-creation time to pre-extract metadata for
    the full cohort.

    Tiers
    -----
    1. Summary features       (~10)
    2. F-code indicators      (top_n_f_codes, dynamic)
    3. Medication indicators   (top_n_meds individual + drug classes)
    4. Temporal features       (~6)

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
        # manual OLS slope — avoids np.polyfit (LAPACK dependency issues)
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
# Step 5b — Tier A: LASSO Bridge
# =============================================================================

def run_lasso_bridge(
    pc_projections: np.ndarray,
    metadata: np.ndarray,
    feature_names: list,
    top_k: int = 10,
    n_bootstrap: int = 100,
) -> dict:
    """
    Tier A — LASSO regression of clinical metadata against PC scores.

    For each top-k PC, regresses PC score ~ metadata features with LassoCV.
    Bootstrap stability selection identifies robustly selected features.

    Both pc_projections and metadata should be at patient level
    (use pool_to_patients() on sample-level PC projections first).

    Parameters
    ----------
    pc_projections : (n_patients, k) patient-level PC scores
    metadata       : (n_patients, n_features) patient-level metadata
    feature_names  : list of n_features feature names
    top_k          : number of PCs to regress (clamped to available)
    n_bootstrap    : iterations for stability selection

    Returns
    -------
    dict with r2_per_pc, mean_r2, unexplained_variance_fraction,
    coeff_matrix, stability_matrix, ci_low, ci_high,
    top_predictors_per_pc, feature_names, n_patients.
    """
    n_patients, k_avail = pc_projections.shape
    n_features = metadata.shape[1]
    k = min(top_k, k_avail)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(metadata)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    r2_values        = np.zeros(k)
    coeff_matrix     = np.zeros((n_features, k))
    stability_matrix = np.zeros((n_features, k))
    ci_low_matrix    = np.zeros((n_features, k))
    ci_high_matrix   = np.zeros((n_features, k))

    for pc_idx in range(k):
        y = pc_projections[:, pc_idx]
        if y.std() < 1e-8:
            continue

        cv_folds = min(5, n_patients - 1)
        if cv_folds < 2:
            continue

        lasso = LassoCV(cv=cv_folds, random_state=SEED,
                        max_iter=5000, n_jobs=1)
        try:
            lasso.fit(X_scaled, y)
        except Exception as exc:
            warnings.warn(f"LassoCV failed for PC{pc_idx + 1}: {exc}")
            continue

        y_pred = lasso.predict(X_scaled)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_values[pc_idx]      = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        coeff_matrix[:, pc_idx] = lasso.coef_

        sel_probs, ci_lo, ci_hi = _lasso_stability_selection(
            X_scaled, y, n_iter=n_bootstrap,
        )
        stability_matrix[:, pc_idx] = sel_probs
        ci_low_matrix[:, pc_idx]    = ci_lo
        ci_high_matrix[:, pc_idx]   = ci_hi

    mean_r2     = float(r2_values.mean())
    unexplained = max(0.0, 1.0 - mean_r2)

    # Top predictors per PC (by |coefficient|, only non-zero)
    top_predictors = {}
    for pc_idx in range(k):
        coefs = coeff_matrix[:, pc_idx]
        order = np.argsort(np.abs(coefs))[::-1]
        top_predictors[f"PC{pc_idx + 1}"] = [
            {
                "feature":     feature_names[i],
                "coefficient": float(coefs[i]),
                "stability":   float(stability_matrix[i, pc_idx]),
                "ci_low":      float(ci_low_matrix[i, pc_idx]),
                "ci_high":     float(ci_high_matrix[i, pc_idx]),
            }
            for i in order[:5]
            if abs(coefs[i]) > 1e-10
        ]

    return {
        "n_patients":                   n_patients,
        "n_features":                   n_features,
        "top_k":                        k,
        "feature_names":                feature_names,
        "r2_per_pc":                    {f"PC{i + 1}": float(v) for i, v in enumerate(r2_values)},
        "mean_r2":                      mean_r2,
        "unexplained_variance_fraction": unexplained,
        "coeff_matrix":                 coeff_matrix,        # (n_features, k)
        "stability_matrix":             stability_matrix,    # (n_features, k)
        "ci_low":                       ci_low_matrix,       # (n_features, k)
        "ci_high":                      ci_high_matrix,      # (n_features, k)
        "top_predictors_per_pc":        top_predictors,
        "n_bootstrap_iterations":       n_bootstrap,
        "interpretation_note": (
            "Unexplained variance fraction is a positive scientific finding: "
            "the model has learned geometric structure that falls outside the "
            "conceptual vocabulary captured by these metadata features."
        ),
    }


# =============================================================================
# Step 5c — Tier B: UMAP + HDBSCAN Cluster Enrichment
# =============================================================================

def run_cluster_enrichment(
    umap_embedding: np.ndarray,
    metadata: np.ndarray,
    feature_names: list,
    min_cluster_size: int = 10,
) -> dict:
    """
    Tier B — UMAP + HDBSCAN cluster enrichment analysis.
    https://hdbscan.readthedocs.io/en/latest/index.html
    
    Clusters the UMAP embedding with HDBSCAN, then computes enrichment of
    metadata features in each cluster relative to the population baseline.

    Both umap_embedding and metadata must be at the same granularity
    (sample-level).  Use broadcast_to_samples() to expand patient-level
    metadata before calling.

    Parameters
    ----------
    umap_embedding   : (N, 2) UMAP coordinates
    metadata         : (N, n_features) sample-level metadata
    feature_names    : list of n_features feature names
    min_cluster_size : HDBSCAN minimum cluster size

    Returns
    -------
    dict with cluster_labels, n_clusters, n_noise, enrichment_matrix,
    cluster_profiles, cluster_sizes, unlabeled_clusters.
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,                  # defaults to min_cluster_size
    )
    cluster_labels = clusterer.fit_predict(umap_embedding)

    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    n_clusters  = len(cluster_ids)
    n_noise     = int((cluster_labels == -1).sum())

    if n_clusters == 0:
        warnings.warn("HDBSCAN found 0 clusters (all noise). "
                       "Try reducing min_cluster_size.")
        return {
            "cluster_labels": cluster_labels,
            "n_clusters": 0, "n_noise": n_noise,
            "cluster_ids": [],
            "enrichment_matrix": np.empty((0, metadata.shape[1])),
            "cluster_profiles": {},
            "cluster_sizes": {},
            "unlabeled_clusters": [],
            "feature_names": feature_names,
        }

    n_features = metadata.shape[1]

    # Population statistics (non-noise samples only)
    non_noise  = cluster_labels >= 0
    pop_data   = metadata[non_noise]
    pop_mean   = pop_data.mean(axis=0)
    pop_std    = pop_data.std(axis=0)
    pop_std[pop_std < 1e-10] = 1e-10

    # Enrichment z-scores  (n_clusters × n_features)
    enrichment_matrix = np.zeros((n_clusters, n_features))
    cluster_sizes     = {}

    for i, cid in enumerate(cluster_ids):
        mask = cluster_labels == cid
        cluster_data = metadata[mask]
        cluster_mean = cluster_data.mean(axis=0)
        enrichment_matrix[i] = (cluster_mean - pop_mean) / pop_std
        cluster_sizes[int(cid)] = int(mask.sum())

    # Cluster profiles: top enriched/depleted features per cluster
    binary_mask = np.array([_is_binary(metadata[:, j])
                            for j in range(n_features)])
    cluster_profiles = {}

    for i, cid in enumerate(cluster_ids):
        z = enrichment_matrix[i]
        order = np.argsort(np.abs(z))[::-1]
        features = []
        for j in order[:7]:
            if abs(z[j]) < 0.3:
                break
            entry = {
                "feature":   feature_names[j],
                "z_score":   round(float(z[j]), 3),
                "direction": "enriched" if z[j] > 0 else "depleted",
            }
            # Odds ratio for binary features
            if binary_mask[j]:
                mask_c = cluster_labels == cid
                p_clust = metadata[mask_c, j].mean()
                p_pop   = pop_data[:, j].mean()
                if p_pop > 0 and p_pop < 1:
                    or_val = (p_clust / (1 - p_clust + 1e-10)) / \
                             (p_pop   / (1 - p_pop   + 1e-10))
                    entry["odds_ratio"] = round(float(or_val), 3)
            features.append(entry)
        cluster_profiles[int(cid)] = features

    # Identify unlabeled clusters (max |z| < 1.0 — no strong enrichment)
    max_z_per_cluster = np.abs(enrichment_matrix).max(axis=1)
    unlabeled_clusters = [
        int(cluster_ids[i]) for i in range(n_clusters)
        if max_z_per_cluster[i] < 1.0
    ]

    return {
        "cluster_labels":      cluster_labels,
        "n_clusters":          n_clusters,
        "n_noise":             n_noise,
        "cluster_ids":         cluster_ids,
        "enrichment_matrix":   enrichment_matrix,
        "cluster_profiles":    cluster_profiles,
        "cluster_sizes":       cluster_sizes,
        "unlabeled_clusters":  unlabeled_clusters,
        "feature_names":       feature_names,
    }