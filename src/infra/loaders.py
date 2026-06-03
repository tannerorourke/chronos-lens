"""Load / get / save of analysis components (infra).

Everything that stands an analysis up from a run's *frozen artifacts* and moves
embedding vectors across the disk boundary - inference only, no gradients, no
training:

- :func:`load_scaffolding`  - rebuild model + loader from a run dir.
- :func:`extract_embeddings` - run the model once and collect vectors in memory
  (the analysis scripts' workhorse).
- :func:`stream_embeddings` - run the model and stream vectors to a single
  ``embeddings_<epoch>.npz`` via :class:`EmbeddingWriter` (too big for RAM).
- :class:`EmbeddingWriter` / :class:`EmbeddingWriterSupv` - the streaming writers.

This is the shared bridge ``scripts/embeddings.py`` and the analysis scripts use;
it may import model + training *definitions* (datasets, checkpoint) but never the
training loops or ``src.analysis``.
"""
from pathlib import Path
from contextlib import nullcontext

import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.models.jepa_ema import JEPA_EMA
from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.supervised_transformer import SupervisedTransformer
from src.training.utils.datasets import MimicDataset
from src.training.utils.checkpoint import load_model_checkpoint
from src.utils.io import RUNS_DIR, EXPERIMENTS_DIR, load_json, load_sequences
from src.utils.seed import set_global_seed
from src.utils.tensors import set_cuda_precision
from src.infra.s3 import ensure_local


# =============================================================================
# Model + loader reconstruction
# =============================================================================

def load_scaffolding(
    ckpt_name: str,
    exp_name: str,
    device: torch.device,
) -> tuple[JEPA_EMA | JEPAStopGrad | SupervisedTransformer, DataLoader, Path, tuple[dict, dict], tuple]:
    """Rebuild model + loader for an existing run from its self-contained run dir.

    ``exp_name`` is a *run-id* under :data:`RUNS_DIR`; config + vocab are read from
    that run's frozen artifacts, and the (heavy) checkpoint is pulled from S3 on
    demand via :func:`ensure_local` if it isn't already local. Returns the run dir
    as the third element.
    """
    # --- resolve the run dir + frozen config (flat input config as fallback)
    run_dir = RUNS_DIR / exp_name
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        legacy = EXPERIMENTS_DIR / f"{exp_name}.yaml"
        if legacy.exists():
            cfg_path = legacy
        else:
            raise FileNotFoundError(
                f"No config.yaml for run '{exp_name}' (looked in {run_dir} and {legacy})")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    run_id = run_dir.name

    # --- data
    data_params     = config["data"]
    batch_size      = data_params["batch_size"]
    n_patients      = data_params["n_patients"]
    max_encounters  = data_params.get("max_encounters", None)
    pin_memory      = data_params.get("pin_mem", True) and device.type == "cuda"
    num_workers     = data_params.get("num_workers", 0)

    # --- meta
    meta_p          = config["meta"]
    seed            = meta_p["seed"]
    use_bfloat16    = meta_p["use_bfloat16"]
    is_supervised   = config["model"]["architecture"] == "supervised"
    label_key       = (meta_p.get("label_key", "label_escalation_per_enc")
                       if is_supervised else None)

    set_global_seed(seed)
    if device.type == "cuda": set_cuda_precision(use_bfloat16)

    # --- checkpoint (local first, else fetch the single heavy object from S3)
    local_ckpt = run_dir / "checkpoints" / ckpt_name
    ckpt_path = local_ckpt if local_ckpt.exists() else ensure_local(
        f"checkpoints/{ckpt_name}", run_id)
    model, ckpt = load_model_checkpoint(ckpt_path, device, restore_rng=True)
    model.eval()

    patients = load_sequences(n=n_patients)
    vocab = load_json(run_dir / "vocab.json")
    if vocab is None:  # vocab may also live only in S3
        try:
            vocab = load_json(ensure_local("vocab.json", run_id))
        except FileNotFoundError:
            pass
    assert vocab is not None, f"vocab.json not found for run {exp_name}"

    ds = MimicDataset(patients, vocab, data_params, pad_idx=0,
                      max_enc=max_encounters, is_supervised=is_supervised,
                      label_key=label_key)
    loader = DataLoader(
        ds, batch_size,
        shuffle=True, collate_fn=ds.mimic_collate, drop_last=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        pin_memory=pin_memory)

    return model, loader, run_dir, (ckpt, config), (ds, is_supervised, label_key, vocab)


# =============================================================================
# In-memory extraction (analysis)
# =============================================================================

def extract_embeddings(model, loader, device) -> dict:
    """Run model inference and collect per-encounter ``z_encs`` for any arch.

    `z_enc` extraction is uniform - always ``(N, C, D)``: JEPA returns it as the
    first forward output; the supervised model exposes it via ``encode`` (its
    forward returns the recency readout, not the per-encounter sequence). The
    only arch-dependent extras are the JEPA predictor outputs ``z_pred`` /
    ``z_target`` (the supervised model has no predictor). Callers derive
    ``z_enc_recency`` / ``pred_error`` via
    :func:`src.infra.vector_computation.compute_derived_vectors`.

    Returns ``z_encs, ctx_pad_mask, subject_ids, mask_pos`` (+ ``z_pred``,
    ``z_target`` for JEPA).

    (Distinct from :func:`stream_embeddings`, which writes to disk.)
    """
    has_predictor = not isinstance(model, SupervisedTransformer)

    all_z_encs: list[np.ndarray] = []
    all_z_pred: list[np.ndarray] = []
    all_z_target: list[np.ndarray] = []
    all_ctx_pad_mask: list[np.ndarray] = []
    all_subject_ids: list[str] = []
    all_mask_pos: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if has_predictor:
                z_enc, z_pred, z_target = model(batch_dev)   # z_enc (B, C, D)
                all_z_pred.append(z_pred.cpu().numpy())
                all_z_target.append(z_target.cpu().numpy())
            else:
                z_enc = model.encode(batch_dev)              # z_enc (B, C, D)

            all_z_encs.append(z_enc.cpu().numpy())
            all_ctx_pad_mask.append(batch_dev["ctx_pad_mask"].cpu().numpy())
            all_subject_ids.extend(batch["subject_ids"])
            all_mask_pos.append(batch_dev["mask_pos"].cpu().numpy())

    # Pad z_encs and ctx_pad_mask to uniform context length across batches
    max_C = max(arr.shape[1] for arr in all_z_encs)
    D = all_z_encs[0].shape[2]
    padded_z_encs: list[np.ndarray] = []
    padded_masks: list[np.ndarray] = []
    for z, m in zip(all_z_encs, all_ctx_pad_mask):
        B, C = z.shape[0], z.shape[1]
        if C < max_C:
            z = np.concatenate([z, np.zeros((B, max_C - C, D), dtype=z.dtype)], axis=1)
            m = np.concatenate([m, np.ones((B, max_C - C), dtype=m.dtype)], axis=1)
        padded_z_encs.append(z)
        padded_masks.append(m)

    out = {
        "z_encs": np.concatenate(padded_z_encs),        # (N, C_max, D)
        "ctx_pad_mask": np.concatenate(padded_masks),   # (N, C_max)
        "subject_ids": np.array(all_subject_ids),       # (N,)
        "mask_pos": np.concatenate(all_mask_pos),       # (N,)
    }
    if has_predictor:
        out["z_pred"] = np.concatenate(all_z_pred)      # (N, D)
        out["z_target"] = np.concatenate(all_z_target)  # (N, D)
    return out


# =============================================================================
# Streaming extraction (training-side save; too big for RAM)
# =============================================================================

def stream_embeddings(
    model: JEPA_EMA | JEPAStopGrad | SupervisedTransformer,
    loader: DataLoader,
    epoch: int,
    n_total: int,
    max_ctx: int,
    embed_dim: int,
    use_bf16: bool,
    is_supervised: bool,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Run one inference epoch and stream embeddings to ``output_dir`` in batches.

    The z_enc vector alone can be tens of GB, so this writes out incrementally via
    :class:`EmbeddingWriter` / :class:`EmbeddingWriterSupv` (S3-syncable on demand).
    """
    cond_autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext())

    cond_writer = (
        EmbeddingWriter(output_dir, n_total, max_ctx, embed_dim, epoch)
        if not is_supervised else
        EmbeddingWriterSupv(output_dir, n_total, max_ctx, embed_dim, epoch)
    )

    with cond_writer as ew:
        with torch.no_grad():
            for batch in tqdm(loader, desc="extracting"):
                batch_dev = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                with cond_autocast_ctx:
                    if is_supervised:
                        assert type(model) is SupervisedTransformer, "expected Supervised Transformer model"
                        # Per-encounter z_enc (B, C, D) - matches the JEPA z_enc
                        # the EmbeddingWriterSupv expects. (forward returns this same
                        # z_enc and computes the recency readout internally; encode()
                        # gives it directly without the classifier head.)
                        data = (model.encode(batch_dev), None)
                    else:
                        assert type(model) is not SupervisedTransformer, "expected Unsupervised JEPA model"
                        # (z_enc, z_pred, z_target)
                        data = model(batch_dev)

                ew.write_batch(
                    data,
                    mask_pos=batch_dev["mask_pos"],
                    ctx_pad_mask=batch_dev["ctx_pad_mask"],
                    subject_ids=batch["subject_ids"],
                )


# =============================================================================
# Streaming writers (memmap -> single .npz)
# =============================================================================

def _mmap(dir: Path, name: str, epoch: int, shape: tuple, dtype=np.float32):
    path = dir / f"_tmp_{name}_{epoch}.npy"
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


class EmbeddingWriter:
    """ Streams embedding vectors to per-field memmaps during a save
        epoch, then consolidates into a single embeddings_{epoch}.npz at
        close. Only I/O.
    """

    def __init__(
        self,
        emb_dir: Path,
        n_total: int, # = len(dataset)
        max_ctx: int, # max_encs-1 or max(len(s["context"]) for s in dataset.samples)
        embed_dim: int,
        epoch: int,
        active: bool = True,
    ):
        self._emb_dir   = emb_dir
        self._epoch     = epoch
        self._n_total   = n_total
        self._max_ctx   = max_ctx
        self._embed_dim = embed_dim
        self._active    = active
        self._write_idx = 0
        self._subject_ids: list[str] = []
        self._mm_z_enc = self._mm_z_pred = self._mm_z_target = None
        self._mm_mask_pos = self._mm_ctx_pad = None

    def __enter__(self) -> "EmbeddingWriter":
        if not self._active:
            return self
        self._emb_dir.mkdir(parents=True, exist_ok=True)

        self._mm_z_enc    = _mmap(self._emb_dir, "z_encs",       self._epoch, (self._n_total, self._max_ctx, self._embed_dim))
        self._mm_z_pred   = _mmap(self._emb_dir, "z_pred",       self._epoch, (self._n_total, self._embed_dim))
        self._mm_z_target = _mmap(self._emb_dir, "z_target",     self._epoch, (self._n_total, self._embed_dim))
        self._mm_mask_pos = _mmap(self._emb_dir, "mask_pos",     self._epoch, (self._n_total,))
        self._mm_ctx_pad  = _mmap(self._emb_dir, "ctx_pad_mask", self._epoch, (self._n_total, self._max_ctx))
        return self

    @property
    def active(self) -> bool:
        return self._active

    def write_batch(
        self,
        data: tuple,
        mask_pos:     torch.Tensor,   # (B,)
        ctx_pad_mask: torch.Tensor,   # (B, C)
        subject_ids:  list[str],
    ) -> None:
        if not self._active:
            return

        z_enc, z_pred, z_target = data
        z_enc_np    = z_enc.detach().cpu().half().numpy()
        z_pred_np   = z_pred.detach().cpu().half().numpy()
        z_target_np = z_target.detach().cpu().half().numpy()
        mask_np     = mask_pos.detach().cpu().float().numpy()
        pad_np      = ctx_pad_mask.detach().cpu().float().numpy()

        B = z_pred_np.shape[0]
        C = z_enc_np.shape[1]
        lo = self._write_idx
        hi = lo + B

        # Guard against last-batch overflow (drop_last=False)
        if hi > self._n_total:
            hi = self._n_total
            B  = hi - lo
            z_enc_np    = z_enc_np[:B]
            z_pred_np   = z_pred_np[:B]
            z_target_np = z_target_np[:B]
            mask_np     = mask_np[:B]
            pad_np      = pad_np[:B]
            subject_ids = subject_ids[:B]

        self._mm_z_enc[lo:hi, :C, :] = z_enc_np     # type: ignore[index]
        self._mm_z_pred[lo:hi]       = z_pred_np    # type: ignore[index]
        self._mm_z_target[lo:hi]     = z_target_np  # type: ignore[index]
        self._mm_mask_pos[lo:hi]     = mask_np      # type: ignore[index]
        self._mm_ctx_pad[lo:hi, :C]  = pad_np       # type: ignore[index]
        self._subject_ids.extend(subject_ids)
        self._write_idx = hi

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._active:
            return

        # Flush memmaps
        for mm in (self._mm_z_enc, self._mm_z_pred, self._mm_z_target,
                   self._mm_mask_pos, self._mm_ctx_pad):
            if mm is not None:
                mm.flush()

        # Consolidate into canonical single .npz, sliced to actual write count
        n = self._write_idx
        tmp_names = {
            "z_encs":       f"_tmp_z_encs_{self._epoch}.npy",
            "z_pred":       f"_tmp_z_pred_{self._epoch}.npy",
            "z_target":     f"_tmp_z_target_{self._epoch}.npy",
            "mask_pos":     f"_tmp_mask_pos_{self._epoch}.npy",
            "ctx_pad_mask": f"_tmp_ctx_pad_mask_{self._epoch}.npy",
        }
        tmp_paths = {k: self._emb_dir / v for k, v in tmp_names.items()}

        # Release memmap handles before reopening (Windows-safe)
        self._mm_z_enc = self._mm_z_pred = self._mm_z_target = None
        self._mm_mask_pos = self._mm_ctx_pad = None

        # Save each array from memmap without loading fully into RAM.
        # Write individual .npy files, then combine into .npz via zipfile.
        out_path = self._emb_dir / f"embeddings_{self._epoch}.npz"
        npy_paths: list[tuple[str, Path]] = []

        for k, p in tmp_paths.items():
            mm = np.load(p, mmap_mode="r")
            out_dtype = bool if k == "ctx_pad_mask" else mm.dtype
            sliced_path = self._emb_dir / f"_sliced_{k}_{self._epoch}.npy"
            # Write slice to a new file chunk-by-chunk to stay memory-friendly
            out_mm = np.lib.format.open_memmap(sliced_path, mode="w+", dtype=out_dtype, shape=mm[:n].shape)
            chunk = 4096
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                out_mm[i:j] = mm[i:j].astype(out_dtype)
            out_mm.flush()
            del mm, out_mm
            npy_paths.append((k, sliced_path))

        # subject_ids
        sid_path = self._emb_dir / f"_sliced_subject_ids_{self._epoch}.npy"
        np.save(sid_path, np.array(self._subject_ids, dtype=str))
        npy_paths.append(("subject_ids", sid_path))

        import zipfile
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_LZMA) as zf:
            for k, p in npy_paths:
                zf.write(p, arcname=f"{k}.npy")

        # Cleanup sliced intermediates
        for _, p in npy_paths:
            try:
                p.unlink()
            except OSError:
                pass

        # Cleanup intermediates
        for p in tmp_paths.values():
            try:
                p.unlink()
            except OSError:
                pass

        print(f"  [EmbeddingWriter] Saved {out_path.name} ({n}/{self._n_total} samples)")


class EmbeddingWriterSupv:
    """ Streams embedding vectors to per-field memmaps during a save
        epoch, then consolidates into a single embeddings_{epoch}.npz at
        close. Only I/O.
    """

    def __init__(
        self,
        emb_dir: Path,
        n_total: int, # = len(dataset)
        max_ctx: int, # max_encs-1 or max(len(s["context"]) for s in dataset.samples)
        embed_dim: int,
        epoch: int,
        active: bool = True,
    ):
        self._emb_dir   = emb_dir
        self._epoch     = epoch
        self._n_total   = n_total
        self._max_ctx   = max_ctx
        self._embed_dim = embed_dim
        self._active    = active
        self._write_idx = 0
        self._subject_ids: list[str] = []
        self._mm_z_enc = None
        self._mm_mask_pos = self._mm_ctx_pad = None

    def __enter__(self) -> "EmbeddingWriterSupv":
        if not self._active:
            return self
        self._emb_dir.mkdir(parents=True, exist_ok=True)

        self._mm_z_enc    = _mmap(self._emb_dir, "z_encs",       self._epoch, (self._n_total, self._max_ctx, self._embed_dim))
        self._mm_mask_pos = _mmap(self._emb_dir, "mask_pos",     self._epoch, (self._n_total,))
        self._mm_ctx_pad  = _mmap(self._emb_dir, "ctx_pad_mask", self._epoch, (self._n_total, self._max_ctx))
        return self

    @property
    def active(self) -> bool:
        return self._active

    def write_batch(
        self,
        data:         tuple,
        mask_pos:     torch.Tensor,   # (B,)
        ctx_pad_mask: torch.Tensor,   # (B, C)
        subject_ids:  list[str],
    ) -> None:
        if not self._active:
            return

        z_enc, _ = data # z_enc: per-encounter (B, C, D)
        z_enc_np    = z_enc.detach().cpu().half().numpy()
        mask_np     = mask_pos.detach().cpu().float().numpy()
        pad_np      = ctx_pad_mask.detach().cpu().float().numpy()

        B, C, _ = z_enc_np.shape
        lo = self._write_idx
        hi = lo + B

        # Guard against last-batch overflow (drop_last=False)
        if hi > self._n_total:
            hi = self._n_total
            B  = hi - lo
            z_enc_np    = z_enc_np[:B]
            mask_np     = mask_np[:B]
            pad_np      = pad_np[:B]
            subject_ids = subject_ids[:B]

        self._mm_z_enc[lo:hi, :C, :] = z_enc_np     # type: ignore[index]
        self._mm_mask_pos[lo:hi]     = mask_np      # type: ignore[index]
        self._mm_ctx_pad[lo:hi, :C]  = pad_np       # type: ignore[index]
        self._subject_ids.extend(subject_ids)
        self._write_idx = hi

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._active:
            return

        # Flush memmaps
        for mm in (self._mm_z_enc, self._mm_mask_pos, self._mm_ctx_pad):
            if mm is not None:
                mm.flush()

        # Consolidate into canonical single .npz, sliced to actual write count
        n = self._write_idx
        tmp_names = {
            "z_encs":       f"_tmp_z_encs_{self._epoch}.npy",
            "mask_pos":     f"_tmp_mask_pos_{self._epoch}.npy",
            "ctx_pad_mask": f"_tmp_ctx_pad_mask_{self._epoch}.npy",
        }
        tmp_paths = {k: self._emb_dir / v for k, v in tmp_names.items()}

        # Release memmap handles before reopening (Windows-safe)
        self._mm_z_enc = self._mm_mask_pos = self._mm_ctx_pad = None

        # Save each array from memmap without loading fully into RAM.
        # Write individual .npy files, then combine into .npz via zipfile.
        out_path = self._emb_dir / f"embeddings_{self._epoch}.npz"
        npy_paths: list[tuple[str, Path]] = []

        for k, p in tmp_paths.items():
            mm = np.load(p, mmap_mode="r")
            out_dtype = bool if k == "ctx_pad_mask" else mm.dtype
            sliced_path = self._emb_dir / f"_sliced_{k}_{self._epoch}.npy"
            # Write slice to a new file chunk-by-chunk to stay memory-friendly
            out_mm = np.lib.format.open_memmap(sliced_path, mode="w+", dtype=out_dtype, shape=mm[:n].shape)
            chunk = 4096
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                out_mm[i:j] = mm[i:j].astype(out_dtype)
            out_mm.flush()
            del mm, out_mm
            npy_paths.append((k, sliced_path))

        # subject_ids
        sid_path = self._emb_dir / f"_sliced_subject_ids_{self._epoch}.npy"
        np.save(sid_path, np.array(self._subject_ids, dtype=str))
        npy_paths.append(("subject_ids", sid_path))

        import zipfile
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_LZMA) as zf:
            for k, p in npy_paths:
                zf.write(p, arcname=f"{k}.npy")

        # Cleanup sliced intermediates
        for _, p in npy_paths:
            try:
                p.unlink()
            except OSError:
                pass

        # Cleanup intermediates
        for p in tmp_paths.values():
            try:
                p.unlink()
            except OSError:
                pass

        print(f"  [EmbeddingWriter] Saved {out_path.name} ({n}/{self._n_total} samples)")
