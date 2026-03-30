from pathlib import Path
import json
from collections import defaultdict

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
# Data Loaders
# =============================================================================

def load_sequences_dict(path: Path) -> dict:
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


def load_metadata(path: Path = None) -> tuple:
    d = Path(path) if path else PROCESSED_DIR
    metadata = np.load(d / "metadata_features.npy")
    with open(d / "metadata_feature_names.json") as f:
        feature_names = json.load(f)
    with open(d / "patient_ids.json") as f:
        patient_ids = np.array(json.load(f), dtype=str)
    return metadata, feature_names, patient_ids

def load_npz_dict(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    return { k: npz[k] for k in npz }

# =============================================================================
# Data Savers
# =============================================================================

def save_metadata(
    metadata: np.ndarray,
    feature_names: list,
    patient_ids: np.ndarray,
    path: Path = None,
) -> None:
    d = Path(path) if path else PROCESSED_DIR
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "metadata_features.npy", metadata)
    with open(d / "metadata_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(d / "patient_ids.json", "w") as f:
        json.dump([str(pid) for pid in patient_ids], f, indent=2)
    print(f"  Metadata saved -> {d}  ({metadata.shape[0]} patients x {metadata.shape[1]} features)")
    
    
def save_embedding_vecs(
    model_records: defaultdict[str, list[np.ndarray]],
    epoch: int | None = None,
    save_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    records = {}
    for k, v in model_records.items():
        if k == "subject_ids":
            records[k] = np.array(v, dtype=str)
        elif k in ("z_encs", "ctx_pad_masks"):
            max_c = max(arr.shape[1] for arr in v)
            padded = []
            for arr in v:
                pad_width = [(0,0)] * arr.ndim
                pad_width[1] = (0, max_c - arr.shape[1])
                padded.append(np.pad(arr, pad_width))
            records[k] = np.concatenate(padded, axis=0)
        else:
            records[k] = np.concatenate(v, axis=0)
    
    if save_dir is not None:
        ep_str = f"_{epoch}" if epoch is not None else ""
        file = (save_dir / f"embeddings{ep_str}").with_suffix(".npz")
        np.savez(
            file,
            z_encs      =records["z_encs"],
            z_pred      =records["z_pred"],
            z_target    =records["z_target"],
            subject_ids =records["subject_ids"],
            mask_pos    =records["mask_pos"],
            labels      =records["labels"],
        )
        print(f"   Saved embeddings -> {file.name} (epoch {epoch})")

    return records

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