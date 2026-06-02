from pathlib import Path

import numpy as np
import torch


def set_cuda_precision(use_bf16: bool = False) -> None:
    """ Disable tf32 matmul when using bfloat16 to avoid stacking 
        two levels of reduced precision
    """
    torch.backends.cudnn.benchmark = True
    
    if use_bf16:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('high')
        
        
        
def _mmap(dir: Path, name: str, epoch: int,shape: tuple, dtype=np.float32):
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