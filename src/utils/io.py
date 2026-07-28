from pathlib import Path
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
# Run artifacts live OUTSIDE the repo working tree (default: ../artifacts,
# beside the repo). ../artifacts/<run-id> IS one run - a clean sync/pull/output 
# target. Override with ARTIFACTS_ROOT. Training and analysis consume
# these and must never redefine them.
# ---------------------------------------------------------------------------
ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_ROOT", ROOT.parent / "artifacts"))
EXPS_DIR = ARTIFACTS_ROOT / "training-runs"
ANALYSIS_DIR = ARTIFACTS_ROOT / "analysis"

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

def make_run_id(tag: str | None, fallback: str = "run") -> str:
    def _slugify(s: str) -> str:
        """ keep [A-Za-z0-9._-], collapse the rest to '-' """
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")
    from datetime import date
    tag = tag if tag is not None else fallback
    id = _slugify(f"{tag}")

    if (EXPS_DIR / id).exists():
        i=2
        while (EXPS_DIR / id).exists():
            id = _slugify(f"{tag}_v{i}")
            i += 1
    return id

def init_exp_dir(run_dir: Path, params: dict | None = None) -> Path:
    """ mkdir, freeze config.yaml """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if params is not None:
        from datetime import date
        params["meta"]["run_date"] = date.today().isoformat()
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)
    return run_dir

def init_exp_config(
    exp: str | Path,
    command: str,
    target: str = None,
    exists_ok: bool = False,
) -> tuple[Path, dict]:
    """Resolve config + run directory. Returns (config_path, exp_dir, params): """
    from datetime import date
    
    exp = Path(exp)
    
    # -- SAE inside existing self-contained run dir
    if command == "sae":
        exp_dir = EXPS_DIR / exp
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Run dir not found for run '{exp}': {exp_dir}. SAE training "
                f"requires an existing trained model under {EXPS_DIR}/<exp>."
            )
            
        cfg_path = exp_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Frozen config not found for run '{exp}': {cfg_path}. SAE training "
                f"operates on an existing run dir under {EXPS_DIR}.")
        
        with open(cfg_path) as f:
            params = yaml.safe_load(f)

        from src.utils.constants import SAE_TARGETS
        assert params and params.get("sae_config"), "config['sae_config'] not found."
        assert target in SAE_TARGETS, \
            f"sae target must be one of {SAE_TARGETS}"
        sae_cfg = params["sae_config"]
        assert target in sae_cfg, f"config['sae_config']['{target}'] not found."
        
        sae_params = sae_cfg[target]
        
        # init experiment subdirectory
        sae_exp_dir = exp_dir / make_run_id(f"sae_{target}")
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

        # Resume continues in the original run dir (run-id = first path component of
        # resume_from); otherwise start a fresh date-stamped run id.
        if params.get("resume_from"):
            run_id = Path(params["resume_from"]).parts[0]
        else:
            tag = params.get("meta", {}).get("tag", None)
            run_id = make_run_id(tag, fallback=cfg_path.stem)
        
        exp_dir = EXPS_DIR / run_id

        if not params.get("resume_from"):
            has_artifacts = any(
                (exp_dir / sub).exists() and any((exp_dir / sub).iterdir())
                for sub in ["checkpoints", "logs"]
            )
            if has_artifacts and not exists_ok:
                raise FileExistsError(
                    f"Run '{run_id}' already has artifacts in {exp_dir}. Set "
                    f"config['resume_from'] to resume, or change meta.tag for a new run.")

            # init experiment directory
            exp_dir.mkdir(parents=True, exist_ok=True)

            # freeze config
            params["meta"]["exp_date"] = date.today().isoformat()
            with open(exp_dir / "config.yaml", "w") as f:
                yaml.safe_dump(params, f, sort_keys=False)
      
        return exp_dir, params
    

def find_subdir(dir: Path, name: str) -> Path:
    subddir = dir / name
    if subddir.is_dir():
        return subddir
    raise FileNotFoundError(f"subdirectory {name} not found ioon {dir}")
    