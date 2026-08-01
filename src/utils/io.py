from pathlib import Path
from datetime import date
import os
import re
import json

import yaml
import numpy as np
import pandas as pd

# =============================================================================
# paths
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent.parent

PARQUET_DIR = ROOT / "data/parquet"
DATA_DIR = ROOT / "data/processed"
EXPERIMENTS_DIR = ROOT / "experiments"

# ---------------------------------------------------------------------------
# Run artifacts live OUTSIDE the repo working tree. One run is one directory,
# <mm-dd>_<run-id>_<seed>, syncing to s3://<bucket>/runs/<dir-name>/ key-for-key.
# data/ holds the frozen inputs (config, vocab, checkpoints, embeddings, logs);
# analysis writes flat at the root, so globbing <run>/*.json is the record of what
# has been run. The lookup key is the run-id, not the directory name - resolve_run_dir
# maps it to the dated directory. Training and analysis must never redefine these.
# ---------------------------------------------------------------------------
ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_ROOT", ROOT.parent / "artifacts"))

def run_dir_name(run_id: str, seed: int) -> str:
    """ Directory name for a run started today """
    return f"{date.today():%m-%d}_{run_id}_{seed}"

def resolve_run_dir(exp: str | Path) -> Path:
    """
    Resolve a run-id (or a full run-dir name) to its directory under ARTIFACTS_ROOT.
    A run-id matching two dirs raises rather than picking one, since the two would
    differ by date or seed and are not the same cell.
    """
    exp = str(exp)
    exact = ARTIFACTS_ROOT / exp
    if exact.is_dir():
        return exact

    pat = re.compile(rf"\d{{2}}-\d{{2}}_{re.escape(exp)}(_\d+)?$")
    hits = sorted(d for d in ARTIFACTS_ROOT.iterdir() if d.is_dir() and pat.fullmatch(d.name))
    if not hits:
        raise FileNotFoundError(
            f"No run dir for '{exp}' under {ARTIFACTS_ROOT} "
            f"(looked for '{exp}' and '<mm-dd>_{exp}_<seed>').")
    if len(hits) > 1:
        names = "\n  ".join(d.name for d in hits)
        raise FileExistsError(
            f"run-id '{exp}' matched {len(hits)} run dirs:\n  {names}\n"
            f"Pass the full directory name instead.")
    return hits[0]

def data_dir(run: str | Path) -> Path:
    """ Core run data: frozen config, vocab, checkpoints, embeddings, logs """
    root = run if isinstance(run, Path) else resolve_run_dir(run)
    return root / "data"

def run_data_path(rel: str | Path) -> Path:
    """ Run-relative reference into data/: '<run-id>/checkpoints/last.pt' """
    parts = Path(rel).parts
    return data_dir(parts[0]).joinpath(*parts[1:])

# ============================================================================
# JSON / Serialization / npz helpers
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
    
def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"    Saved -> {path.name}")

def load_npz_dict(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    npz = dict(npz)
    return { k: npz[k] for k in npz }

# =============================================================================
# Sequences
# =============================================================================

def load_sequences_dict(path: Path = DATA_DIR / "sequences.jsonl") -> dict:
    patients = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            patients[str(p["subject_id"])] = p
    return patients
    
def load_sequences(n=None, path: Path = DATA_DIR / "sequences.jsonl") -> list[dict]:
    """Load sequences as iterable list of dicts (for training)"""
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
            print(f"-- Patients: {len(sequences)}") 
            return sequences
        if n < 0:
            raise ValueError("n must be non-negative")
        print(f"-- Patients: {len(sequences[:n])}")
        return sequences[:n]
    
    except Exception as e:
        raise FileNotFoundError(f"[load_sequences] Error loading sequences from '{src}': {e}")

# =============================================================================
# Metadata
# =============================================================================

def load_metadata(dir: Path = None) -> tuple:
    d = Path(dir) if dir else DATA_DIR
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
        
    print(f"\nMetadata saved to ../{'/'.join(p.parts[-2:])} ({metadata.shape[0]} patients x {metadata.shape[1]} features)")

# =============================================================================
# config IO
# =============================================================================

def make_run_id(tag: str | None, fallback: str = "run", parent: Path | None = None) -> str:
    """
    Slugified id. With 'parent', suffixes _v2, _v3, ... until the name is free
    under it; without, the raw slug (uniqueness is the caller's problem).
    """
    def _slugify(s: str) -> str:
        """ keep [A-Za-z0-9._-], collapse the rest to '-' """
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")
    tag = tag if tag is not None else fallback
    id = _slugify(f"{tag}")
    if parent is None:
        return id

    i = 2
    while (parent / id).exists():
        id = _slugify(f"{tag}_v{i}")
        i += 1
    return id

def init_exp_dir(run_dir: Path, params: dict | None = None) -> Path:
    """ mkdir run root + data/, freeze data/config.yaml """
    run_dir = Path(run_dir)
    data_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if params is not None:
        params["meta"]["exp_date"] = date.today().isoformat()
        with open(data_dir(run_dir) / "config.yaml", "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)
    return run_dir

def init_exp_config(
    exp: str | Path,
    command: str,
    target: str = None,
    exists_ok: bool = False,
) -> tuple[Path, dict]:
    """Resolve config + run directory. Returns (exp_dir, params) """
    exp = Path(exp)

    # -- SAE inside existing self-contained run dir
    if command == "sae":
        exp_dir = resolve_run_dir(exp)
        cfg_path = data_dir(exp_dir) / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Frozen config not found for run '{exp}': {cfg_path}. SAE training "
                f"operates on an existing run dir under {ARTIFACTS_ROOT}.")

        with open(cfg_path) as f:
            params = yaml.safe_load(f)

        from src.utils.constants import SAE_TARGETS
        assert params and params.get("sae_config"), "config['sae_config'] not found."
        assert target in SAE_TARGETS, \
            f"sae target must be one of {SAE_TARGETS}"
        sae_cfg = params["sae_config"]
        assert target in sae_cfg, f"config['sae_config']['{target}'] not found."
        
        sae_params = sae_cfg[target]
        
        # -- SAE dicts are analysis output: run root beside the result JSONs, not in data/
        sae_exp_dir = exp_dir / make_run_id(f"sae_{target}", parent=exp_dir)
        sae_exp_dir.mkdir(parents=True, exist_ok=True)
        
        # -- freeze SAE config, recording the target it was trained on and the date
        frozen = dict(sae_params)
        frozen["target"] = target
        frozen["exp_date"] = date.today().isoformat()
        with open(sae_exp_dir / "config.yaml", "w") as f:
            yaml.safe_dump(frozen, f, sort_keys=False)
        
        return sae_exp_dir, sae_cfg[target]
      
    # -- model training
    else:
        cfg_path = EXPERIMENTS_DIR / f"{exp}.yaml"
        if not cfg_path.exists():
            fallback = EXPERIMENTS_DIR / f"{exp.parts[-1]}.yaml"
            if not fallback.exists():
                raise FileNotFoundError(f"Model config not found: experiments/{str(exp)}.yaml")
            cfg_path = fallback

        with open(cfg_path, 'r') as y_file:
            params = yaml.safe_load(y_file)
        if not params:
            raise FileNotFoundError(f"'experiments/{exp}.yaml' is empty / not found.")

        assert params.get("model", {}).get("architecture", "") in ["ema", "stopgrad", "supervised"], \
            f"config['model']['architecture'] must be one of 'ema', 'stopgrad', or 'supervised'"
        assert params.get("meta", {}).get("seed"), \
            f"parameter 'seed' missing in config.yaml['meta']"

        # Resume continues in the original run dir (first path component of
        # resume_from names it); otherwise open a fresh <mm-dd>_<run-id>_<seed>.
        if params.get("resume_from"):
            exp_dir = resolve_run_dir(Path(params["resume_from"]).parts[0])
        else:
            tag = params.get("meta", {}).get("tag", None)
            run_id = make_run_id(tag, fallback=cfg_path.stem)
            seed = params["meta"]["seed"]
            dir_name = make_run_id(run_dir_name(run_id, seed), parent=ARTIFACTS_ROOT)
            exp_dir = ARTIFACTS_ROOT / dir_name

            has_artifacts = any(
                (data_dir(exp_dir) / sub).exists() and any((data_dir(exp_dir) / sub).iterdir())
                for sub in ["checkpoints", "logs"]
            )
            if has_artifacts and not exists_ok:
                raise FileExistsError(
                    f"Run '{run_id}' already has artifacts in {exp_dir}. Set "
                    f"config['resume_from'] to resume, or change meta.tag for a new run.")

            # init run dir + freeze config into data/
            init_exp_dir(exp_dir, params)
      
        return exp_dir, params
    

def find_subdir(dir: Path, name: str) -> Path:
    subddir = dir / name
    if subddir.is_dir():
        return subddir
    raise FileNotFoundError(f"subdirectory {name} not found ioon {dir}")
    