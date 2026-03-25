from pathlib import Path
import json
# import pickle

import numpy as np
import pandas as pd

# =============================================================================
# paths
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent.parent

PARQUET_DIR = ROOT / "data/parquet"
PROCESSED_DIR = ROOT / "data/processed"
EXPERIMENTS_DIR = ROOT / "experiments"


# =============================================================================
# config IO
# =============================================================================

def resolve_run_dir(prefix: str) -> Path:
    existing = sorted(EXPERIMENTS_DIR.glob(f"{prefix}_v*"))
    if not existing:
        return EXPERIMENTS_DIR / f"{prefix}_v001"
    last_num = int(existing[-1].name.split("_v")[-1])
    return EXPERIMENTS_DIR / f"{prefix}_v{last_num + 1:03d}"


def init_run_dir(model: str) -> Path:
    base_dir = EXPERIMENTS_DIR / model
    run_dir = base_dir

    if not base_dir.exists() or not (base_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"[create_run] Expected config.yaml in experiments/{model}")

    has_artifacts = any(
        (base_dir / sub).exists() and any((base_dir / sub).iterdir())
        for sub in ["checkpoints", "logs"]
    )
    if has_artifacts:
        run_dir = resolve_run_dir(model)
        run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir

# =============================================================================
# Data IO
# =============================================================================

def load_sequences_dict(path: Path) -> dict:
    """Load sequences from JSONL into dict of dicts"""
    patients = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p
    return patients
    

def load_sequences(n=None) -> list[dict]:
    """Load sequences from JSONL into list of dicts, parse ISO datetime strings back to datetime objects"""
    sequences = []
    try:
        with open(PROCESSED_DIR / "sequences.jsonl") as f:
            for line in f:
                record = json.loads(line)
                for enc in record["encounters"]:
                    enc["admittime"] = pd.Timestamp(enc["admittime"])
                    enc["dischtime"] = pd.Timestamp(enc["dischtime"])
                sequences.append(record)
        
        if n is None or n == 0:
            return sequences
        if n < 0:
            raise ValueError("n must be non-negative")
        return sequences[:n]
    except Exception as e:
        raise FileNotFoundError(f"[load_sequences] Error loading .jsonl sequences from '{PROCESSED_DIR}/sequences.jsonl': {e}")
    # pickle
    # with open(path, "rb") as f:
    #     sequences = pickle.load(f)
    
    
# =============================================================================
# Metadata IO
# =============================================================================

def load_metadata(path: Path = None) -> tuple:
    """Load pre-extracted metadata features and feature names.

    Parameters
    ----------
    path : directory containing metadata_features.npy, metadata_feature_names.json,
           and patient_ids.json.  Defaults to PROCESSED_DIR.

    Returns
    -------
    metadata       : (n_patients, n_features) float64
    feature_names  : list[str]
    patient_ids    : (n_patients,) str array
    """
    d = Path(path) if path else PROCESSED_DIR
    metadata = np.load(d / "metadata_features.npy")
    with open(d / "metadata_feature_names.json") as f:
        feature_names = json.load(f)
    with open(d / "patient_ids.json") as f:
        patient_ids = np.array(json.load(f), dtype=str)
    return metadata, feature_names, patient_ids


def save_metadata(
    metadata: np.ndarray,
    feature_names: list,
    patient_ids: np.ndarray,
    path: Path = None,
) -> None:
    """Persist metadata features, names, and patient IDs to disk.

    Parameters
    ----------
    metadata       : (n_patients, n_features) float64
    feature_names  : list[str]
    patient_ids    : (n_patients,) str array
    path           : output directory (defaults to PROCESSED_DIR)
    """
    d = Path(path) if path else PROCESSED_DIR
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "metadata_features.npy", metadata)
    with open(d / "metadata_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(d / "patient_ids.json", "w") as f:
        json.dump([str(pid) for pid in patient_ids], f, indent=2)
    print(f"  Metadata saved -> {d}  ({metadata.shape[0]} patients x {metadata.shape[1]} features)")


# =============================================================================
# Vocab
# =============================================================================

def build_vocab(patients: list[dict], pad_idx: int, dir: Path) -> dict[str, int]:
    """Map every unique ICD code and med name to a positive integer index.
    """
    print(f"[build_vocab] building vocab ([PAD]: {pad_idx})...")
    tokens: set[str] = set()
    for p in patients:
        for enc in p.get("encounters", []):
            tokens.update(enc.get("icd_codes", []))
            tokens.update(enc.get("meds", []))
    vocab: dict[str, int] = {"[PAD]": pad_idx}
    for i, tok in enumerate(sorted(tokens), start=1):
        vocab[tok] = i
        
    with open(dir / "vocab.json", "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, indent=2)
        
    print(f"   len: {len(vocab)}")
    return vocab
    