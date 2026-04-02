from pathlib import Path
import json
from typing import Tuple

import numpy as np
import pandas as pd

# =============================================================================
# paths
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent.parent

PARQUET_DIR = ROOT / "data/parquet"
DATA_DIR = ROOT / "data/processed"
EXPERIMENTS_DIR = ROOT / "experiments"


# ============================================================================
# JSON / Serialization
# ============================================================================

def _serialize(obj):
    """Recursively convert numpy types for JSON serialisation"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return str(v)
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialize(data), f, indent=2, default=str)
    print(f"    Saved: {path.name}")
    
    
def load_json(p: Path):
    if not p.exists():
        print(f"    WARNING: {p.name} not found")
        return None
    with open(p) as f:
        return json.load(f)

# ============================================================================
# npz/npy
# ============================================================================

def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"    Saved -> {path.name}")


def load_npz_dict(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    npz = dict(npz)
    return { k: npz[k] for k in npz }


def save_embedding_vecs(
    model_records: dict[str, list[np.ndarray]],
    epoch: int | None = None,
    save_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
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
        save_dict = {
            "z_encs":      records["z_encs"],
            "z_pred":      records["z_pred"],
            "z_target":    records["z_target"],
            "subject_ids": records["subject_ids"],
            "mask_pos":    records["mask_pos"],
        }
        if "ctx_pad_masks" in records:
            save_dict["ctx_pad_masks"] = records["ctx_pad_masks"]
        np.savez(file, **save_dict)  # type: ignore[arg-type]
        print(f"Saved embeddings -> {file.name} (epoch {epoch})")

    return records


def load_embeddings(model_dir, embeddings_arg=None) -> Tuple[dict, Path]:
    """Find embeddings npz in model directory. If not found, choose the last one"""
    def _embedding_epoch(path):
        stem = path.stem  # "embeddings_40"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return -1
    
    if embeddings_arg:
        name = embeddings_arg if embeddings_arg.endswith(".npz") else f"{embeddings_arg}.npz"
        path = model_dir / name
        if not path.exists(): # try recursive search
            matches = list(model_dir.glob(f"**/{name}"))
            if matches:
                return dict(np.load(matches[0], allow_pickle=True)), matches[0]
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        return dict(np.load(path, allow_pickle=True)), path

    candidates = list(model_dir.glob("**/embeddings*.npz"))
    if not candidates:
        candidates = list(model_dir.glob("**/embedding*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No embeddings .npz found in {model_dir}. Run scripts/evaluate.py to extract.")
    # Sort by epoch number (highest last), fall back to name for ties
    candidates.sort(key=lambda p: (_embedding_epoch(p), p.name))
    return dict(np.load(candidates[-1], allow_pickle=True)), candidates[-1]

# =============================================================================
# Sequences
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
    

def load_sequences(n=None, path: Path = None) -> list[dict]:
    src = Path(path) if path else DATA_DIR / "sequences.jsonl"
    sequences = []
    try:
        with open(src) as f:
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
        raise FileNotFoundError(f"[load_sequences] Error loading sequences from '{src}': {e}")


# =============================================================================
# Metadata
# =============================================================================

def load_metadata(path: Path = None) -> tuple:
    d = Path(path) if path else DATA_DIR
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
    p = Path(path) if path else DATA_DIR
    p.mkdir(parents=True, exist_ok=True)
    
    np.save(p / "metadata_features.npy", metadata)
    with open(p / "metadata_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(p / "patient_ids.json", "w") as f:
        json.dump([str(pid) for pid in patient_ids], f, indent=2)
        
    print(f"\nMetadata saved to ../{p.parts[-2]} ({metadata.shape[0]} patients x {metadata.shape[1]} features)")

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