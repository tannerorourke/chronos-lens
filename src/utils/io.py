from pathlib import Path
import os
import re
import json
from datetime import date
from typing import Tuple

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
# beside the repo) so one directory IS a run: a single clean S3 sync source /
# laptop pull target. Override with CHRONOS_ARTIFACTS_ROOT. This is the single
# source of truth for output paths - training and analysis consume these and
# must never redefine them.
# ---------------------------------------------------------------------------
ARTIFACTS_ROOT = Path(os.environ.get("CHRONOS_ARTIFACTS_ROOT", ROOT.parent / "artifacts"))
RUNS_DIR = ARTIFACTS_ROOT / "training-runs"
ANALYSIS_DIR = ARTIFACTS_ROOT / "analysis"


# =============================================================================
# Run identity & directory scaffolding
# =============================================================================

def _slugify(s: str) -> str:
    """Filesystem-safe slug: keep [A-Za-z0-9._-], collapse the rest to '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


def make_run_id(params: dict, fallback: str = "run") -> str:
    """Stable, human-legible run slug derived from config:
    ``<tag>_<arch>_seed<seed>_<ISO-date>`` - so "what is this run?" is answerable
    from the directory name plus one read of config.yaml.
    """
    meta = params.get("meta", {})
    model = params.get("model", {})
    tag = meta.get("tag") or fallback
    arch = model.get("architecture", "model")
    seed = meta.get("seed", "NA")
    return _slugify(f"{tag}_{arch}_seed{seed}_{date.today().isoformat()}")


def freeze_config(params: dict, run_dir: Path) -> None:
    """Write a frozen snapshot of the resolved config into the run dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)


def init_run_dir(run_dir: Path, params: dict | None = None) -> Path:
    """Create the run-dir skeleton: mkdir, freeze config.yaml, create empty
    notes.md if missing. Idempotent (safe to call on resume)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if params is not None:
        freeze_config(params, run_dir)
    notes = run_dir / "notes.md"
    if not notes.exists():
        notes.write_text("", encoding="utf-8")
    return run_dir


def _maybe_migration_hint(config_dir: Path, run_dir: Path) -> None:
    """Print a one-line hint (do NOT move anything) when old in-repo outputs
    exist but the new out-of-repo run dir is still empty."""
    try:
        legacy = any((config_dir / sub).exists()
                     for sub in ["checkpoints", "logs", "embeddings"])
        fresh = (not run_dir.exists()) or (not any(run_dir.iterdir()))
        if legacy and fresh:
            print(f"[migrate] legacy outputs found in '{config_dir}'. New runs write "
                  f"to '{run_dir}'. Move them manually if you want them reused, e.g. "
                  f"`mv {config_dir / 'checkpoints'} {run_dir / 'checkpoints'}`.")
    except Exception:
        pass


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
            return sequences
        if n < 0:
            raise ValueError("n must be non-negative")
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

def get_model_config(
    exp: str | Path,
    command: str = "model",
    target: str = None,
    exists_ok: bool = False,
) -> tuple[Path, Path, dict]:
    """Resolve config + run directory.

    Returns ``(config_path, run_dir, params)``:

    * ``command='model'`` - ``exp`` names ``experiments/<exp>.yaml`` (the
      in-repo, git-tracked *input spec*); ``config_path`` is that file. A fresh
      out-of-repo ``run_dir`` under
      :data:`RUNS_DIR` is computed for all outputs (or the resumed run's dir when
      ``resume_from`` is set). ``params`` is the full config.
    * ``command='sae'`` - ``exp`` names an existing *run-id* under
      :data:`RUNS_DIR`; config is read from that run's frozen ``config.yaml`` and
      ``params`` is the SAE leaf config for ``target``. ``config_path`` mirrors
      ``run_dir`` (the frozen config lives there).
    """
    exp = Path(exp)
    assert command in ["model", "sae"], f"Invalid command: {command}"

    # ----- SAE: operate inside an existing self-contained run dir -----------
    if command == "sae":
        run_dir = RUNS_DIR / exp
        if not run_dir.exists():
            run_dir = RUNS_DIR / exp.parts[-1]
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Frozen config not found for run '{exp}': {cfg_path}. SAE training "
                f"operates on an existing run dir under {RUNS_DIR}.")
        with open(cfg_path) as f:
            params = yaml.safe_load(f)

        from src.utils.constants import SAE_TARGETS
        assert params and params.get("sae_config"), "config['sae_config'] not found."
        assert target in SAE_TARGETS, \
            f"sae target must be one of {SAE_TARGETS}"
        sae_root = params["sae_config"]
        assert target in sae_root, f"config['sae_config']['{target}'] not found."
        return run_dir, run_dir, sae_root[target]

    # ----- model: in-repo config -> out-of-repo run dir ---------------------
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
    # resume_from); otherwise mint a fresh date-stamped run id.
    if params.get("resume_from"):
        run_id = Path(params["resume_from"]).parts[0]
    else:
        run_id = make_run_id(params, fallback=cfg_path.stem)
    run_dir = RUNS_DIR / run_id

    if not params.get("resume_from"):
        has_artifacts = any(
            (run_dir / sub).exists() and any((run_dir / sub).iterdir())
            for sub in ["checkpoints", "logs"])
        if has_artifacts and not exists_ok:
            raise FileExistsError(
                f"Run '{run_id}' already has artifacts in {run_dir}. Set "
                f"config['resume_from'] to resume, or change meta.tag for a new run.")
        # Legacy in-repo outputs (if any) sat under the old experiments/<run-id>/ dir.
        _maybe_migration_hint(EXPERIMENTS_DIR / cfg_path.stem, run_dir)

    return cfg_path, run_dir, params

# ============================================================================
# Embedding IO
# ============================================================================

def save_embedding_vecs(
    model_records: dict[str, list[np.ndarray]],
    epoch: int | None = None,
    save_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    records: dict[str, np.ndarray] = {}
    for k, v in model_records.items():
        # print(k, [a.shape for a in v[:3]])
        if k == "subject_ids":
            records[k] = np.array(v, dtype=str)
        elif k in ("z_encs", "ctx_pad_mask"):
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
        if "ctx_pad_mask" in records:
            save_dict["ctx_pad_mask"] = records["ctx_pad_mask"]
        np.savez(file, **save_dict)  # type: ignore[arg-type]
        print(f"    Saved embeddings -> {file.name} (epoch {epoch})")

    return records

def _embedding_epoch(name: str) -> int:
    """Epoch number parsed from an ``embeddings_<N>.npz`` filename (else -1)."""
    stem = Path(name).stem  # "embeddings_40"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return -1


def load_embeddings(model_dir, embeddings_arg=None) -> Tuple[dict, Path]:
    """Load an embeddings ``.npz`` for a run, pulling exactly one object from S3
    on demand if it isn't already local.

    ``model_dir`` is the run dir (``RUNS_DIR/<run-id>/``); ``run-id`` is its name.
    Embeddings that are too big to keep locally are fetched one object at a time
    via :func:`src.utils.s3.ensure_local` (``aws s3 cp``), never a bulk sync.
    """
    from src.utils.s3 import ensure_local, s3_list

    model_dir = Path(model_dir)
    run_id = model_dir.name

    # --- specific file requested -------------------------------------------
    if embeddings_arg:
        name = embeddings_arg if embeddings_arg.endswith(".npz") else f"{embeddings_arg}.npz"
        # local first (direct, then recursive), else pull the single object.
        direct = model_dir / "embeddings" / name
        matches = [direct] if direct.exists() else list(model_dir.glob(f"**/{name}"))
        if matches:
            return dict(np.load(matches[0], allow_pickle=True)), matches[0]
        path = ensure_local(f"embeddings/{name}", run_id)  # fetch or clear error
        return dict(np.load(path, allow_pickle=True)), path

    # --- latest epoch (no name given) --------------------------------------
    candidates = list(model_dir.glob("**/embeddings*.npz"))
    if not candidates:
        candidates = list(model_dir.glob("**/embedding*.npz"))
    if candidates:
        candidates.sort(key=lambda p: (_embedding_epoch(p.name), p.name))
        return dict(np.load(candidates[-1], allow_pickle=True)), candidates[-1]

    # Nothing local: discover the latest available object in S3 and pull just it.
    remote = [n for n in s3_list(run_id, "embeddings") if n.endswith(".npz")]
    if remote:
        remote.sort(key=lambda n: (_embedding_epoch(n), n))
        path = ensure_local(f"embeddings/{remote[-1]}", run_id)
        return dict(np.load(path, allow_pickle=True)), path

    raise FileNotFoundError(
        f"No embeddings .npz found locally in {model_dir} or in S3 for run-id "
        f"'{run_id}'. Run `python -m scripts.embeddings extract --exp {run_id} "
        f"--ckpt <checkpoint.pt>` (or `... fetch --exp {run_id}`).")