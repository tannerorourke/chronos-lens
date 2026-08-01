"""Stand an analysis up from a run's frozen artifacts and move embedding vectors
across the disk boundary. Inference only - no gradients, no training.

load_embeddings_for_analysis is the entry point every analysis script uses: local
.npz, else S3, else extraction from the stem-matched checkpoint. Extraction runs in
memory when the cohort fits and streams to mmap temps when it does not.

Shared bridge between the training loops and the analysis scripts, so it may import
model and training *definitions* but never the training loops or src.analysis.
"""
from contextlib import nullcontext
from pathlib import Path
from typing import ClassVar, Iterator, Callable
import zipfile

import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

from src.models import MODEL_TYPE, SupervisedTransformer, JEPA_EMA, JEPAStopGrad, SparseAutoencoder
from src.training.utils.datasets import MimicDataset, NoisyBucketedSampler
from src.training.utils.checkpoint import load_model_eval
from src.utils.io import DATA_DIR, data_dir, resolve_run_dir, find_subdir, load_json, load_sequences
from src.utils.system import set_global_seed, set_cuda_precision
from src.infra.s3 import S3Client


# =============================================================================
# Embedding mmap streaming
# =============================================================================

class EmbeddingStream:
    """
    Mmap-backed dict-like view over a completed embedding extraction.
 
    Owns the streaming temp .npy files on disk and exposes them as zero-copy
    mmap views via dict-style access ('result["z_encs"]'). Use as a context
    manager to delete temps when analysis is finished, or call '.cleanup()'
    directly. Use '.to_npz(path)' to persist a consolidated archive before
    cleanup.
 
    Lifecycle:
        with JEPAEmbeddingWriter(out_dir, stem, n, C, D) as ew:
            for batch in loader:
                ew.write_batch(...)
        result = ew.result                            # may be None on abort
 
        with result:
            z = result["z_encs"]                      # mmap view, sliced
            if persist_local: result.to_npz(out_path) # optional
        # temps deleted here
    """
 
    def __init__(
        self,
        stem: str,
        n_valid: int,
        n_total: int,
        tmp_paths: dict[str, Path],
        subject_ids: np.ndarray,
    ):
        self._stem = stem
        self._n_valid = n_valid
        self._n_total = n_total
        self._tmp_paths = dict(tmp_paths)
        self._subject_ids = subject_ids
        self._closed = False
 
    # --- dict-like access
    def __getitem__(self, key: str) -> np.ndarray:
        if self._closed:
            raise RuntimeError(f"EmbeddingStream({self._stem!r}) is closed")
        if key == "subject_ids":
            return self._subject_ids
        if key not in self._tmp_paths:
            raise KeyError(key)
        # mmap view sliced to actual write count - no RAM copy
        return np.load(self._tmp_paths[key], mmap_mode="r")[: self._n_valid]
 
    def keys(self) -> list[str]:
        return list(self._tmp_paths) + ["subject_ids"]
 
    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())
 
    def __len__(self) -> int:
        return len(self._tmp_paths) + 1
 
    def __contains__(self, key: object) -> bool:
        return key == "subject_ids" or key in self._tmp_paths
 
    @property
    def n_valid(self) -> int:
        return self._n_valid
 
    @property
    def n_total(self) -> int:
        return self._n_total
 
    @property
    def stem(self) -> str:
        return self._stem
 
    # --- persistence
    def to_npz(self, out_path: Path) -> Path:
        """
        Consolidate temp mmaps into a single .npz at 'out_path'.
 
        Uses ZIP_STORED (no compression): embeddings are dense float16 noise
        where LZMA buys <20% size at the cost of hours of CPU at 70GB scale.
 
        Fast path: if 'n_valid == n_total', temps are zip-written directly.
        Slow path: temps sliced to 'n_valid' via chunked memmap copy.
        """
        if self._closed:
            raise RuntimeError(f"EmbeddingStream({self._stem!r}) is closed")
 
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_size = self._n_valid == self._n_total
 
        sliced_paths: dict[str, Path] = {}
        scratch: list[Path] = []  # files we created and should clean up
 
        try:
            for field, tmp_path in self._tmp_paths.items():
                if full_size:
                    sliced_paths[field] = tmp_path
                    continue
                mm = np.load(tmp_path, mmap_mode="r")
                out_p = tmp_path.with_name(f"_sliced_{field}_{self._stem}.npy")
                out_mm = np.lib.format.open_memmap(
                    out_p, mode="w+", dtype=mm.dtype, shape=mm[: self._n_valid].shape
                )
                chunk = 4096
                for i in range(0, self._n_valid, chunk):
                    j = min(i + chunk, self._n_valid)
                    out_mm[i:j] = mm[i:j]
                out_mm.flush()
                del out_mm, mm  # release handles before zipping (Windows)
                sliced_paths[field] = out_p
                scratch.append(out_p)
 
            # subject_ids: dump in-memory array to a one-off .npy for zipping
            sid_path = out_path.parent / f"_sliced_subject_ids_{self._stem}.npy"
            np.save(sid_path, self._subject_ids)
            sliced_paths["subject_ids"] = sid_path
            scratch.append(sid_path)
 
            with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for field, path in sliced_paths.items():
                    zf.write(path, arcname=f"{field}.npy")
        finally:
            # Only delete files we created here; self._tmp_paths is owned by self
            for p in scratch:
                p.unlink(missing_ok=True)
 
        logger.info(
            "[EmbeddingStream] wrote %s (%d/%d samples)",
            out_path.name, self._n_valid, self._n_total,
        )
        return out_path
 
    # --- cleanup
    def cleanup(self) -> None:
        """Delete backing temp .npy files. Result is unusable after this."""
        if self._closed:
            return
        for path in self._tmp_paths.values():
            path.unlink(missing_ok=True)
        self._tmp_paths.clear()
        self._closed = True
 
    def __enter__(self) -> "EmbeddingStream":
        return self
 
    def __exit__(self, *_exc) -> None:
        self.cleanup()


SchemaEntry = tuple[type, Callable[[int, int, int], tuple]]
Schema      = dict[str, SchemaEntry]
_NP_TO_TORCH: dict[type, torch.dtype] = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32:   torch.int32,
    np.int64:   torch.int64,
    np.bool_:   torch.bool,
}

class EmbeddingWriter:
    """ 
    Streams JEPA embeddings to per-field on-disk memmaps during one inference
    epoch. 
    On exit, hands ownership of the temps to an 'EmbeddingStream' for
    analysis/persistence.
    """
    _SCHEMA: ClassVar[Schema] = {}

    def __init__(
        self,
        out_dir: Path,
        stem: str,
        n_total: int,
        max_ctx: int,
        embed_dim: int,
        schema: Schema | None = None
    ):
        self._out_dir   = out_dir
        self._stem      = stem
        self._ncd       = (n_total, max_ctx, embed_dim)
        self._n_total   = n_total
        self._max_ctx   = max_ctx
        self._embed_dim = embed_dim
        self._schema    = schema or self._SCHEMA
        if not self._schema:
            raise ValueError("EmbeddingWriter requires a schema")
        
        self._write_idx = 0
        self._subject_ids: list[str] = []
        self._mmaps:     dict[str, np.memmap] = {}
        self._tmp_paths: dict[str, Path]      = {}
        self.result: EmbeddingStream | None   = None

    def _tmp_path(self, field: str) -> Path:
        # Stem-suffixed so concurrent extractions in the same dir don't collide
        return self._out_dir / f"_tmp_{field}_{self._stem}.npy"
    
    def _to_numpy(self, tensor: torch.Tensor, np_dtype: type) -> np.ndarray:
        """Convert torch tensor to numpy with the target dtype, on-device if possible."""
        torch_dtype = _NP_TO_TORCH.get(np_dtype, None)
        if torch_dtype is not None and tensor.dtype != torch_dtype:
            tensor = tensor.to(torch_dtype)
        return tensor.detach().cpu().numpy()

    def __enter__(self) -> "EmbeddingWriter":
        self._out_dir.mkdir(parents=True, exist_ok=True)
        n, C, D = self._ncd
        for field, (dtype, shape_fn) in self._SCHEMA.items():
            path = self._tmp_path(field)
            self._tmp_paths[field] = path
            self._mmaps[field] = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape_fn(n, C, D),
            )
        return self

    def write_batch(
        self,
        fields: dict[str, torch.Tensor],
        subject_ids: list[str],
    ) -> None:
        missing = set(self._schema) - set(fields)
        extra   = set(fields) - set(self._schema)
        if missing or extra:
            raise KeyError(f"write_batch field mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        
        # Convert all tensors to their schema dtype
        arrs = {
            name: self._to_numpy(t, self._schema[name][0])
            for name, t in fields.items()
        }
        
        # Batch size from any field (all share axis 0)
        B  = next(iter(arrs.values())).shape[0]
        lo = self._write_idx
        hi = lo + B

        # Clamp the last batch if it spills past n_total (drop_last=False)
        if hi > self._n_total:
            hi = self._n_total
            B  = hi - lo
            if B <= 0:
                return
            arrs        = {k: v[:B] for k, v in arrs.items()}
            subject_ids = subject_ids[:B]

        # slice context axis when present, else flat batch slice
        for name, arr in arrs.items():
            mmap = self._mmaps[name]
            if mmap.ndim >= 2 and mmap.shape[1] == self._max_ctx:
                C = arr.shape[1]
                mmap[lo:hi, :C] = arr
            else:
                mmap[lo:hi] = arr
 
        self._subject_ids.extend(subject_ids)
        self._write_idx = hi

    def __exit__(self, exc_type, exc, tb) -> None:
        # Flush and release handles so Windows can unlink later
        for mm in self._mmaps.values():
            mm.flush()
        self._mmaps.clear()
 
        n = self._write_idx
        
        # Abort path: extraction failed or wrote nothing.
        # An empty consolidated npz silently masquerades as a real output and
        # gets S3-synced, so we discard temps and leave self.result = None.
        if exc_type is not None or n == 0:
            reason = "extraction error" if exc_type is not None else "zero samples written"
            logger.warning("[EmbeddingWriter] discarding %s (%s)", self._stem, reason)
            
            for path in self._tmp_paths.values():
                path.unlink(missing_ok=True)
            self.result = None
            return
        
        self.result = EmbeddingStream(
            stem=self._stem,
            n_valid=n,
            n_total=self._n_total,
            tmp_paths=self._tmp_paths,
            subject_ids=np.array(self._subject_ids, dtype=str),
        )


class JEPAEmbeddingWriter(EmbeddingWriter):
    _SCHEMA = {
        "z_encs":       (np.float16, lambda n, C, D: (n, C, D)),
        "z_pred":       (np.float32, lambda n, C, D: (n, D)),
        "z_target":     (np.float32, lambda n, C, D: (n, D)),
        "mask_pos":     (np.int32, lambda n, C, D: (n,)),
        "ctx_pad_mask": (np.bool_,   lambda n, C, D: (n, C)),
    }
 
 
class SupervisedEmbeddingWriter(EmbeddingWriter):
    _SCHEMA = {
        "z_encs":       (np.float16, lambda n, C, D: (n, C, D)),
        "mask_pos":     (np.int32, lambda n, C, D: (n,)),
        "ctx_pad_mask": (np.bool_,   lambda n, C, D: (n, C)),
    }

# =============================================================================
# Loading model & dataset, vocab, and config
# =============================================================================

def load_scaffolding(
    run_id: str,
    ckpt_name: str,
    device: torch.device,
    s3_ctx: S3Client | None = None
) -> tuple[MODEL_TYPE, DataLoader, tuple[bool, str | None], tuple[dict, dict], tuple]:
    """
    Setup model, loader, and system for inference on frozen model from an existing run dir.
    - Assumes that checkpoint exists on disk and can be loaded
    - 'exp_name' is a *run-id*, resolved to its dated dir under ARTIFACTS_ROOT
    - config + vocab are read from that run's frozen artifacts, checkpoint is 
      found on-demand or pulled from S3
    """
    # --- resolve the run's core data dir + frozen config
    ddir = data_dir(run_id)
    cfg_path = ddir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml for run '{run_id}'")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

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
    is_supervised   = config["model"]["architecture"] == "supervised"
    label_key       = (meta_p.get("label_key", "label_escalation")
                       if is_supervised else None)

    set_global_seed(seed)
    if device.type == "cuda":
        set_cuda_precision(use_bf16=True)
    
    ckpt_name = ckpt_name if ckpt_name.endswith(".pt") else f"{ckpt_name}.pt"
    model, ckpt = load_model_eval(device, run_id=run_id, filename=ckpt_name)
    
    patients, vocab = None, None
    if not (ddir / "vocab.json").exists() and s3_ctx is not None:
        s3_ctx.sync(ddir / "vocab.json")
    if not (ddir / "sequences.jsonl").exists() and s3_ctx is not None:
        s3_ctx.sync(DATA_DIR / "sequences.jsonl")
    
    patients = load_sequences(n=n_patients)
    vocab = load_json(ddir / "vocab.json")
    assert patients is not None and vocab is not None

    ds = MimicDataset(
        patients, vocab, data_params, 
        pad_idx=0,
        max_enc=max_encounters, 
        is_supervised=is_supervised,
        label_key=label_key)
    sampler = NoisyBucketedSampler(
        lengths=ds.sample_lengths,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)
    loader = DataLoader(ds,
        batch_sampler=sampler, 
        collate_fn=ds.mimic_collate,
        num_workers=num_workers, 
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory)

    return model, loader, (is_supervised, label_key), (ckpt, config), (ds, vocab)


def load_sae_info(
    exp_id: str,
    sae_exp_id: str,
    device: torch.device
) -> tuple[SparseAutoencoder, Path, dict, np.ndarray, np.ndarray, str | None]:
    sae_exp_dir = find_subdir(resolve_run_dir(exp_id), sae_exp_id)
    # checkpoint/model
    ckpt_path = sae_exp_dir / "sae.pt"
    if not ckpt_path.exists():
        name = list(sae_exp_dir.glob("*.pt"))
        if not name:
            raise FileNotFoundError(f"SAE checkpoint not found in {sae_exp_dir}")
        ckpt_path = sae_exp_dir / name[0]
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_params = ckpt["model_params"]
    model = SparseAutoencoder(**model_params).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # decoder weights, activations
    dec_weights = np.load(sae_exp_dir / "decoder_weights.npy")
    activations = np.load(sae_exp_dir / "activations.npy")

    # -- target the SAE was trained on, None when the frozen config predates the field
    cfg_path = sae_exp_dir / "config.yaml"
    target = None
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        target = cfg.get("target") if cfg else None

    print(f"Loaded SAE model, activations, and decoder weights from {sae_exp_dir}")

    return model, sae_exp_dir, ckpt, dec_weights, activations, target
    
    
# =============================================================================
# Extracting embeddings and vector data
# =============================================================================
def stream_embeddings(
    model: MODEL_TYPE, loader: DataLoader, device: torch.device,
    emb_shape: tuple[int, int, int],
    out_dir: Path,
    out_file_stem: str,
    use_bf16: bool = True
) -> EmbeddingStream:
    """Run one inference epoch and stream embeddings to 'out_dir' in batches.

    The z_enc vector alone can be tens of GB, so this writes out incrementally via
    'JEPAEmbeddingWriter' / 'SupervisedEmbeddingWriter'.
    """
    n_total, max_ctx, emb_dim = emb_shape
    
    is_ema, is_sg, is_supv = (
        type(model) is JEPA_EMA,
        type(model) is JEPAStopGrad,
        type(model) is SupervisedTransformer)

    writer_ctx = (
        JEPAEmbeddingWriter(out_dir, out_file_stem, n_total, max_ctx, emb_dim)
        if (is_ema or is_sg) else
        SupervisedEmbeddingWriter(out_dir, out_file_stem, n_total, max_ctx, emb_dim)
    )
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )

    result = None
    with writer_ctx as ew:
        with torch.no_grad():
            for batch in tqdm(loader, desc="extracting"):
                batch_dev = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                fields = {}
                with autocast_ctx:
                    if is_supv:
                        assert type(model) is SupervisedTransformer, "expected Supervised Transformer model"
                        # Per-encounter z_enc (batch_size, C, embed_dim)
                        # encode() gives z_enc directly without the classifier head.)
                        z_enc = (model.encode(batch_dev), None)
                        fields = {
                            "z_encs":       z_enc,
                            "mask_pos":     batch["mask_pos"],
                            "ctx_pad_mask": batch["ctx_pad_mask"],
                        }
                    else:
                        assert type(model) is not SupervisedTransformer, "expected Unsupervised JEPA model"
                        # (z_enc, z_pred, z_target[, z_target_nograd]) - SG returns a
                        # 4th detached-target tensor used only by the loss.
                        z_enc, z_pred, z_target = model(batch_dev)[:3]
                        fields={
                            "z_encs":       z_enc,
                            "z_pred":       z_pred,
                            "z_target":     z_target,
                            "mask_pos":     batch["mask_pos"],
                            "ctx_pad_mask": batch["ctx_pad_mask"],
                        }
                ew.write_batch(fields, subject_ids=batch["subject_ids"])        
    result = ew.result
    if result is None:
        raise RuntimeError("extraction failed")
    return result


def in_mem_extract_embeds(
    model: MODEL_TYPE,
    loader: DataLoader,
    device: torch.device,
    is_supv: bool | None = None,
):
    """Run one inference epoch and collect embedding vectors in memory.

    Same field layout as the streaming writers (z_encs, ctx_pad_mask,
    subject_ids, mask_pos; plus z_pred/z_target for JEPA archs), padded to the
    longest context length seen. Holds everything in RAM; use
    'stream_embeddings' for full-cohort extraction. 'is_supv' is inferred
    from the model type when omitted.
    """
    if is_supv is None:
        is_supv = isinstance(model, SupervisedTransformer)
    elif is_supv != isinstance(model, SupervisedTransformer):
        raise ValueError(f"CONFIG MISMATCH: is_supv={is_supv} but model is {type(model).__name__}")
    model.eval()

    all_z_encs: list[np.ndarray] = []
    all_z_pred: list[np.ndarray] = []
    all_z_target: list[np.ndarray] = []
    all_ctx_pad_mask: list[np.ndarray] = []
    all_subject_ids: list[str] = []
    all_mask_pos: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if is_supv:
                    z_enc = model.encode(batch_dev)
                else:
                    # SG returns a 4th detached-target tensor (training-only);
                    # take the first three outputs uniformly across JEPA archs.
                    z_enc, z_pred, z_target = model(batch_dev)[:3]
                    all_z_pred.append(z_pred.float().cpu().numpy())
                    all_z_target.append(z_target.float().cpu().numpy())
                # -- bf16 tensors are not numpy-convertible; cast to fp32 first
                all_z_encs.append(z_enc.float().cpu().numpy())
                all_ctx_pad_mask.append(batch_dev["ctx_pad_mask"].cpu().numpy())
                all_subject_ids.extend(batch["subject_ids"])
                all_mask_pos.append(batch_dev["mask_pos"].cpu().numpy())
            
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
        "z_encs": np.concatenate(padded_z_encs),              # (N, C_max, D)
        "ctx_pad_mask": np.concatenate(padded_masks),         # (N, C_max)
        "subject_ids": np.array(all_subject_ids, dtype=str),  # (N,)
        "mask_pos": np.concatenate(all_mask_pos),             # (N,)
    }
    if not is_supv:
        out["z_pred"] = np.concatenate(all_z_pred)      # (N, D)
        out["z_target"] = np.concatenate(all_z_target)  # (N, D)
    
    return out

# =============================================================================
# ANALYSIS ENTRY POINT
# =============================================================================

def load_embeddings_for_analysis(
    run_id: str,
    name: str,
    device: torch.device,
    sync_ckpts: bool = True,
    write_emb_local: bool = False,
    write_emb_s3: bool = False,
    no_s3: bool = False,
) -> tuple[EmbeddingStream, tuple[MODEL_TYPE | None, dict]]:
    """Resolve a run's embeddings for analysis.

    'name' is the embeddings file name; its stem also names the source
    checkpoint ('data/embeddings/<stem>.npz' pairs with 'data/checkpoints/<stem>.pt').

    Resolution order:
      1. local 'data/embeddings/<stem>.npz'  (mmap'd NpzFile)
      2. S3 'runs/<dir>/data/checkpoints/<stem>.pt'  (fetch, fall through to 4)
      3. S3 'runs/<dir>/data/embeddings/<stem>.npz'  (streamed into memory)
      4. extraction from the local checkpoint via one inference epoch

    Returns '(embeddings, (model, config))'. 'embeddings' is dict-like and a
    context manager (NpzFile on paths 1/3, 'EmbeddingStream' on 4);
    'model' is None unless extraction ran. 'write_emb_local' / 'write_emb_s3'
    persist a freshly obtained .npz; 'no_s3' confines resolution to local disk.
    """
    filename = name if name.endswith(".npz") else f"{name}.npz"
    prefix = Path(filename).stem
    # -- S3 keys mirror the run dir, so the client roots there and every
    #    transferred path stays run-relative (data/checkpoints/..., data/embeddings/...)
    root = resolve_run_dir(run_id)
    ldir = data_dir(root)
    lpath_emb = ldir / "embeddings" / filename
    lpath_pt = ldir / "checkpoints" / f"{prefix}.pt"

    cfg_path = ldir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml for run '{run_id}'")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # -- local .npz: no model rebuild, no S3 round-trip required
    if lpath_emb.exists():
        if sync_ckpts and not no_s3:
            with S3Client(root, s3_subdir="runs", strict=False) as s3:
                s3.sync(lpath_pt)
        return np.load(lpath_emb, mmap_mode="r", allow_pickle=True), (None, config)

    # -- resolve a checkpoint from S3, or stream the remote .npz
    if not lpath_pt.exists():
        if no_s3:
            raise FileNotFoundError(
                f"No local embeddings or checkpoint for {filename!r} in {ldir}")
        with S3Client(root, s3_subdir="runs", strict=True) as s3:
            if s3.exists(lpath_pt):
                s3.fetch(lpath_pt)
            elif s3.exists(lpath_emb):
                buf = s3.stream(lpath_emb)
                if write_emb_local:
                    lpath_emb.parent.mkdir(parents=True, exist_ok=True)
                    lpath_emb.write_bytes(buf.getvalue())
                return np.load(buf, allow_pickle=True), (None, config)
            else:
                raise FileNotFoundError(
                    f"No embeddings or checkpoint for {filename!r} locally or on S3")

    # -- extract from the stem-matched checkpoint
    model, loader, _, (ckpt_data, config), (ds, _) = \
        load_scaffolding(run_id, f"{prefix}.pt", device)

    n_total = len(ds)
    if config["data"].get("max_encounters"):
        max_ctx = config["data"]["max_encounters"] - 1
    else:
        max_ctx = max(len(s["context"]) for s in ds.samples)
    emb_dim = ckpt_data["model_params"]["embed_dim"]

    stream = stream_embeddings(
        model, loader, device,
        (n_total, max_ctx, emb_dim),
        out_dir=ldir / "embeddings", out_file_stem=prefix
    )

    if write_emb_local:
        stream.to_npz(lpath_emb)

    if write_emb_s3 and not no_s3:
        tmp = lpath_emb if write_emb_local else (ldir / "embeddings" / f".scratch_{prefix}.npz")
        if not write_emb_local:
            stream.to_npz(tmp)
        with S3Client(root, s3_subdir="runs", strict=False) as s3:
            s3.upload(tmp)
        if not write_emb_local:
            tmp.unlink()

    # -- caller uses 'with stream: ...' to manage cleanup
    return stream, (model, config)
    
# =============================================================================
# EXTRACT-ONLY ENTRY POINT
# =============================================================================

def save_embeds(
    model: MODEL_TYPE, 
    loader: DataLoader, 
    device: torch.device,
    emb_shape: tuple[int, int, int],
    dir: Path,
    file: Path | str,
    s3_client: S3Client,
    write_local: bool = True,
) -> None:
    # -- 'dir' is the run's data/embeddings; 's3_client' must root at the run dir
    #    so the uploaded key mirrors it
    emb_file = dir / Path(file)
    with stream_embeddings(
        model, loader, device, emb_shape,
        out_dir=dir, out_file_stem=emb_file.stem
    ) as stream:
        stream.to_npz(emb_file)
        s3_client.upload(emb_file, _overwrite=True)

    if not write_local:
        emb_file.unlink(missing_ok=True)