"""
MIMIC-IV Patient Sequence Feature Extraction Pipeline functions
===================================================
Build patient-level temporal sequences from MIMIC-IV v3.1 (BigQuery).

Cohort: 
    - self-supervised
    - all patients with >= min_encounters alive-at-discharge
    - admissions are included regardless of diagnosis. 
Labels (readmission, escalation, next-encounter ICD blocks) are 
computed separately in labels.py.

Schema per patient (pre-labeling):
{
  "subject_id": str,
  "encounters": [
    {
      "hadm_id": int,
      "admittime": datetime,
      "dischtime": datetime,
      "icd_codes": ["F32", "I10.1", ...],
      "meds": ["sertraline", ...]
    }, 
    ...
  ]
}

ICD code processing:
  - F-codes truncated to 3-char block level (F32.1 -> F32)
  - Non-F ICD-10 codes retain full dot notation (I10.1 stays I10.1)
  - icd_codes_full: full dot-notation F-codes kept separately for escalation labels

BigQuery auth:
  gcloud auth application-default login
  gcloud config set project aihc-463505
"""

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

    # --- Full F-code display (dot-notation, not truncation) for label computation ---
    dx["icd_display_full"] = dx["icd_display"]  # default: same as truncated (non-F codes)
    is_fcode_long = is_fcode & (dx["icd_clean"].str.len() > 3)
    dx.loc[is_fcode_long, "icd_display_full"] = (
        dx.loc[is_fcode_long, "icd_clean"].str[:3]
        + "."
        + dx.loc[is_fcode_long, "icd_clean"].str[3:]
    )
    # Short F-codes (exactly 3 chars): use icd_clean directly (no dot)
    is_fcode_short = is_fcode & (dx["icd_clean"].str.len() <= 3)
    dx.loc[is_fcode_short, "icd_display_full"] = dx.loc[is_fcode_short, "icd_clean"]

    # --- Build icd_codes (truncated F-codes) via deduplication on icd_display ---
    dx_trunc = dx.drop_duplicates(subset=["hadm_id", "icd_display"])
    adm_dx = (
        dx_trunc.groupby("hadm_id")
        .agg(
            icd_codes=("icd_display", list),
            has_label_dx=("has_label_dx", "any"),
        ).reset_index()
    )

    # --- Build icd_codes_full (full F-codes, for escalation label computation) ---
    dx_f_full = dx[is_fcode].drop_duplicates(subset=["hadm_id", "icd_display_full"])
    adm_f_full = (
        dx_f_full.groupby("hadm_id")["icd_display_full"]
        .apply(list).reset_index()
        .rename(columns={"icd_display_full": "icd_codes_full"})
    )
    adm_dx = adm_dx.merge(adm_f_full, on="hadm_id", how="left")
    adm_dx["icd_codes_full"] = adm_dx["icd_codes_full"].apply(
        lambda x: x if isinstance(x, list) else []
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
    print("\nExtracting sequences..")
    
    print("Cleaning admissions..")
    adm_clean = clean_admissions(admissions)

    print("Building per-admission ICD codes..")
    adm_dx = build_admission_icd_codes(diagnoses, label_prefix)

    print("Building per-admission active medications..")
    adm_meds = build_admission_active_meds(prescriptions, adm_clean)

    print("Merging into encounters table..")
    return (
        adm_clean
        .merge(adm_dx, on="hadm_id", how="left")
        .merge(adm_meds, on="hadm_id", how="left")
    )


def build_patient_sequences(
    encounters: pd.DataFrame,
    min_encounters: int,
) -> list[dict]:
    print("Building final sequences\nApplying cohort filters...")
    print("   Applying cohort filters...")

    # Fill missing lists (admissions with no diagnoses or no active meds)
    encounters["icd_codes"] = encounters["icd_codes"].apply(
        lambda x: x if isinstance(x, list) else [])
    encounters["meds"] = encounters["meds"].apply(
        lambda x: x if isinstance(x, list) else [])
    encounters["has_label_dx"] = encounters["has_label_dx"].fillna(False)
    if "icd_codes_full" in encounters.columns:
        encounters["icd_codes_full"] = encounters["icd_codes_full"].apply(
            lambda x: x if isinstance(x, list) else [])
    else:
        encounters["icd_codes_full"] = [[] for _ in range(len(encounters))]

    # Sort by patient and time
    encounters = encounters.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    print(f"      Total encounters: {len(encounters):,}")

    # Filter to minimum encounters per patient
    enc_per_patient = encounters.groupby("subject_id").size()
    qualifying = set(enc_per_patient[enc_per_patient >= min_encounters].index)
    print(f"      Patients with >= {min_encounters} encounters: {len(qualifying):,}")
    encounters = encounters[encounters["subject_id"].isin(qualifying)]
    print(f"      Final: {len(encounters):,} encounters, {encounters['subject_id'].nunique():,} patients")

    # --- Assemble sequences ---
    sequences = []
    for subject_id, group in encounters.groupby("subject_id"):
        rows = group.sort_values("admittime").to_dict("records")

        enc_list = [{
            "hadm_id":       int(row["hadm_id"]),
            "admittime":     row["admittime"],
            "dischtime":     row["dischtime"],
            "icd_codes":     row["icd_codes"],
            "icd_codes_full": row["icd_codes_full"],
            "meds":          row["meds"],
        } for row in rows]

        sequences.append({
            "subject_id": str(subject_id),
            "encounters": enc_list,
        })

    enc_counts = [len(s["encounters"]) for s in sequences]
    print(f"\n{'=' * 60}")
    print(f"Sequences built:")
    print(f"  Patients:      {len(sequences):,}")
    print(f"  Encounters/pt: mean={np.mean(enc_counts):.1f}, "
          f"median={np.median(enc_counts):.0f}, "
          f"min={min(enc_counts)}, max={max(enc_counts)}")
    print(f"{'=' * 60}")

    return sequences