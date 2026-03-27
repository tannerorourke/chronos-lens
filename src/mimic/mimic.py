"""
MIMIC-IV Patient Sequence Feature Extraction Pipeline functions
===================================================
Build patient-level temporal sequences from MIMIC-IV v3.1 (BigQuery)
for mood-disorder readmission prediction.

Schema per patient:
{
  "subject_id": str,
  "encounters": [
    {
      "hadm_id": int,
      "admittime": datetime,
      "dischtime": datetime,
      "icd_codes": ["F32", "I10.1", ...],
      "meds": ["sertraline", ...]
    }, ...
  ],
  "label": int,     # 1 if 90-day readmission has F30-F39 diagnosis; 0 otherwise
  "label_30d": int   # 1 if 30-day readmission has F30-F39 diagnosis; 0 otherwise
}

Cohort filters:
  - All patients readmitted within [readm_window_days] of discharge (no diagnosis requirement)
  - min 3 encounters per patient
  - Exclude deceased during admission (hospital_expire_flag = 0, deathtime is null)

ICD code processing:
  - F-codes truncated to 3-char block level (F32.1 -> F32)
  - Non-F ICD-10 codes retain full dot notation (I10.1 stays I10.1)

BigQuery auth:
  gcloud auth application-default login
  gcloud config set project aihc-463505
"""

from datetime import timedelta
import numpy as np
import pandas as pd
import google.auth
from google.cloud import bigquery

from src.mimic.helper import load_parquets, save_parquets
from src.utils.io import PARQUET_DIR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def _authenticate():
    print(f"Authenticating...")
    try:
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        assert credentials is not None, "No credentials found"
        assert project is not None, "No project found in credentials"
        
        print(f"  Authenticated (project: {project})")
        return credentials, project
    except Exception:
        raise RuntimeError("BigQuery auth failed. Run 'gcloud auth application-default login'")
    

def load_tables(dataset: str, project_id: str = None) -> tuple:
    """ Load MIMIC tables from parquet cache if available, otherwise from BigQuery.
        tables are saved to PARQUET_SUBDIR on BigQuery fetch so subsequent runs skip 
        the call entirely (save $$$).
    """
    _PARQUET_FILES = ["admissions.parquet", "patients.parquet",
                      "diagnoses.parquet", "prescriptions.parquet"]
    
    if all((PARQUET_DIR / f).exists() for f in _PARQUET_FILES):
        return load_parquets(PARQUET_DIR)

    credentials, detected_project = _authenticate()
    if project_id is None:
        project_id = detected_project
    client = bigquery.Client(project=project_id, credentials=credentials)
    print(f"\nLoading from BigQuery ({dataset}) from {project_id}...")
    
    bq_tables = {
        "admissions":    f"{dataset}.admissions",
        "patients":      f"{dataset}.patients",
        "diagnoses":     f"{dataset}.diagnoses_icd",
        "prescriptions": f"{dataset}.prescriptions",
    }
    dfs = {}
    for name, table in bq_tables.items():
        df = client.query(f"SELECT * FROM `{table}`").to_dataframe()
        dfs[name] = df
        print(f"  {name:20s} {len(df):>8,} rows")

    save_parquets(dfs["admissions"], dfs["patients"],
                 dfs["diagnoses"], dfs["prescriptions"])

    return dfs["admissions"], dfs["patients"], dfs["diagnoses"], dfs["prescriptions"]
    

def clean_admissions(admissions: pd.DataFrame) -> pd.DataFrame:
    """ Filter to alive-at-discharge admissions and parse datetimes """
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime",
                       "deathtime", "hospital_expire_flag"]].copy()

    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["dischtime"] = pd.to_datetime(adm["dischtime"])

    n_before = len(adm)
    adm = adm[
        (adm["hospital_expire_flag"] == 0)
        & (adm["deathtime"].isna())
        & (adm["admittime"] < adm["dischtime"])
    ].copy()
    
    adm = adm[["subject_id", "hadm_id", "admittime", "dischtime"]].reset_index(drop=True)

    print(f"  Clean admissions: {len(adm):,} / {n_before:,} ({adm['subject_id'].nunique():,} patients)")
    return adm


def build_admission_icd_codes(diagnoses: pd.DataFrame, label_prefix: str) -> pd.DataFrame:
    """
    Group all ICD codes per admission.

    Returns DataFrame with columns:
        hadm_id, icd_codes (list[str]), has_label_dx (bool)

    ICD-10 F-codes: truncated to 3-char block level (e.g., F32.1 -> F32).
    Non-F ICD-10 codes: retain full dot notation (e.g., I10.1).
    ICD-9 codes: as-is.
    """
    dx = diagnoses[["hadm_id", "icd_code", "icd_version"]].copy()

    # strip whitespace, uppercase, remove dots for matching
    dx["icd_clean"] = (
        dx["icd_code"].astype(str).str.strip()
        .str.replace(".", "", regex=False).str.upper()
    )

    # --- Display format ---
    # Default: keep original
    dx["icd_display"] = dx["icd_code"]

    # ICD-10 F-codes: truncate to block level (3 chars)
    is_fcode = (dx["icd_version"] == 10) & dx["icd_clean"].str.startswith("F")
    dx.loc[is_fcode, "icd_display"] = dx.loc[is_fcode, "icd_clean"].str[:3]

    # ICD-10 non-F codes: insert dot after 3rd char
    is_icd10_long_non_f = ((dx["icd_version"] == 10) &
                           (~is_fcode) &
                           (dx["icd_clean"].str.len() > 3))

    dx.loc[is_icd10_long_non_f, "icd_display"] = (
        dx.loc[is_icd10_long_non_f, "icd_clean"].str[:3] +
        "." +
        dx.loc[is_icd10_long_non_f, "icd_clean"].str[3:])

    # --- Label flag: F30-F39 (mood disorder) ---
    dx["has_label_dx"] = (dx["icd_version"] == 10) & dx["icd_clean"].str.startswith(label_prefix)

    # deduplicate display codes per admission before grouping
    dx = dx.drop_duplicates(subset=["hadm_id", "icd_display"])

    adm_dx = (
        dx.groupby("hadm_id")
        .agg(
            icd_codes=("icd_display", list),
            has_label_dx=("has_label_dx", "any"),
        ).reset_index()
    )

    print(f"  Admissions with diagnoses: {len(adm_dx):,}")
    print(f"  Admissions with {label_prefix}* (mood dx): {adm_dx['has_label_dx'].sum():,}")
    return adm_dx


def build_admission_active_meds(
    prescriptions: pd.DataFrame,
    adm_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each admission, find medications active at admission time:
        starttime <= admittime <= stoptime

    Returns DataFrame with columns:
        hadm_id, meds (list[str] - lwc drug names)
    """
    rx = prescriptions[["hadm_id", "drug", "starttime", "stoptime"]].copy()
    rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
    rx["stoptime"] = pd.to_datetime(rx["stoptime"], errors="coerce")

    # Drop missing times
    rx = rx.dropna(subset=["starttime", "stoptime"])

    # Merge admittime
    admittime_lookup = adm_clean[["hadm_id", "admittime"]].drop_duplicates("hadm_id")
    rx = rx.merge(admittime_lookup, on="hadm_id", how="inner")

    # Filter to active at admission (starttime <= admittime <= stoptime)
    rx = rx[(rx["starttime"] <= rx["admittime"]) & (rx["stoptime"] >= rx["admittime"])]

    # lwc drug names, deduplicate per admission
    rx["drug_clean"] = rx["drug"].str.lower().str.strip()
    rx = rx.drop_duplicates(subset=["hadm_id", "drug_clean"])

    adm_meds = (rx.groupby("hadm_id")["drug_clean"]
        .apply(list).reset_index()
        .rename(columns={"drug_clean": "meds"}))

    print(f"  Admissions with active meds: {len(adm_meds):,}")
    return adm_meds


def get_clean_encounters(
    admissions: pd.DataFrame,
    diagnoses: pd.DataFrame,
    prescriptions: pd.DataFrame,
    label_prefix: str
):
    print("Extracting sequences...\n")
    
    print("[1/5] Cleaning admissions")
    adm_clean = clean_admissions(admissions)

    print("[2/5] Building per-admission ICD codes")
    adm_dx = build_admission_icd_codes(diagnoses, label_prefix)

    print("[3/5] Building per-admission active medications")
    adm_meds = build_admission_active_meds(prescriptions, adm_clean)

    print("[4/5] Building encounters")
    return (
        adm_clean
        .merge(adm_dx, on="hadm_id", how="left")
        .merge(adm_meds, on="hadm_id", how="left")
    )


def _compute_readmission_label(rows: list[dict], readmission_days: int) -> int:
    """Check if any readmission within `readmission_days` has a mood-disorder dx.

    Parameters
    ----------
    rows : sorted (by admittime) list of encounter dicts with 'dischtime',
           'admittime', and 'has_label_dx' keys.
    readmission_days : window in days after discharge.

    Returns
    -------
    1 if a positive readmission exists within the window, 0 otherwise.
    """
    n = len(rows)
    for i in range(n - 1):
        window_end = rows[i]["dischtime"] + timedelta(days=readmission_days)
        for j in range(i + 1, n):
            if rows[j]["admittime"] > window_end:
                break
            if rows[j]["has_label_dx"]:
                return 1
    return 0


def build_patient_sequences(
    encounters: pd.DataFrame,
    min_encounters: int,
    readm_window_days: int
) -> list[dict]:
    """
    Main extraction: builds patient sequences with readmission-based cohort
    and mood-disorder (F30-F39) labels.

    Cohort: all patients with at least one readmission within [readm_window_days]
    days of a prior discharge, plus >= [min_encounters] encounters total.

    Label: 1 if the readmission includes an F30-F39 diagnosis; 0 otherwise.
    """
    print("      applying cohort filters")

    # Fill missing lists (admissions with no diagnoses or no active meds)
    encounters["icd_codes"] = encounters["icd_codes"].apply(
        lambda x: x if isinstance(x, list) else [])
    encounters["meds"] = encounters["meds"].apply(
        lambda x: x if isinstance(x, list) else [])
    encounters["has_label_dx"] = encounters["has_label_dx"].fillna(False)

    # Sort by patient and time
    encounters = encounters.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    print(f"      Total encounters: {len(encounters):,}")

    # filter > minimum encounters per patient
    enc_per_patient = encounters.groupby("subject_id").size()
    qualifying = set(enc_per_patient[enc_per_patient >= min_encounters].index)
    print(f"  Patients with >= {min_encounters} encounters: {len(qualifying):,}")
    encounters = encounters[encounters["subject_id"].isin(qualifying)]

    # filter > readmission-based cohort: patients >= 1 readmission pair 
    # where next admittime <= prior dischtime + window
    readm_patients = set()
    for subject_id, group in encounters.groupby("subject_id"):
        rows = group.sort_values("admittime").to_dict("records")
        for i in range(len(rows) - 1):
            window_end = rows[i]["dischtime"] + timedelta(days=readm_window_days)
            if rows[i + 1]["admittime"] <= window_end:
                readm_patients.add(subject_id)
                break
    print(f"      Patients with readmission within {readm_window_days}d: {len(readm_patients):,}")

    encounters = encounters[encounters["subject_id"].isin(readm_patients)].copy()
    print(f"      Final: {len(encounters):,} encounters, {encounters['subject_id'].nunique():,} patients")

    # --- Compute labels and assemble sequences ---
    print(f"\n[5/5] Computing readmission labels ({readm_window_days}-day and 30-day)...")

    sequences = []
    for subject_id, group in encounters.groupby("subject_id"):
        rows = group.sort_values("admittime").to_dict("records")

        enc_list = [{
            "hadm_id": int(row["hadm_id"]),
            "admittime": row["admittime"],
            "dischtime": row["dischtime"],
            "icd_codes": row["icd_codes"],
            "meds": row["meds"],
        } for row in rows]

        sequences.append({
            "subject_id": str(subject_id),
            "encounters": enc_list,
            "label":      _compute_readmission_label(rows, readm_window_days),
            "label_30d":  _compute_readmission_label(rows, 30),
        })

    # --- Summary ---
    n_pos = sum(1 for s in sequences if s["label"] == 1)
    n_neg = len(sequences) - n_pos
    n_pos_30 = sum(1 for s in sequences if s["label_30d"] == 1)
    enc_counts = [len(s["encounters"]) for s in sequences]

    print(f"\n{'=' * 60}")
    print(f"Sequences Summary")
    print(f"  Patients:              {len(sequences):,}")
    print(f"  Label=1 ({readm_window_days}d mood):  {n_pos:,} ({100 * n_pos / len(sequences):.1f}%)")
    print(f"  Label=1 (30d mood):    {n_pos_30:,} ({100 * n_pos_30 / len(sequences):.1f}%)")
    print(f"  Label=0 ({readm_window_days}d):       {n_neg:,} ({100 * n_neg / len(sequences):.1f}%)")
    print(f"  Encounters/pt:         mean={np.mean(enc_counts):.1f}, "
          f"median={np.median(enc_counts):.0f}, "
          f"min={min(enc_counts)}, max={max(enc_counts)}")
    print(f"{'=' * 60}")

    return sequences