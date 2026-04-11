import time
import csv
from pathlib import Path
import statistics as stats

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class CsvWriter:
    def __init__(self, logdir: Path, fn: str = "epoch_metrics.csv"):
        self._logdir = logdir
        self._logdir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._fp = self._logdir / fn
        self._file = open(self._fp, "w", newline="")

    def write(self, metrics: dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()
        
    def __del__(self):
        self._file.close()


class EmbeddingTracker:
    """Welford online mean/variance + L2-norm tracker for one embedding
    tensor type. Cheap enough to compute per batch per epoch."""

    def __init__(self):
        self._n: int = 0
        self._mean: np.ndarray | None = None   # (D,) float64
        self._M2:   np.ndarray | None = None    # (D,) float64
        self._norm_sum: float = 0.0
        self._norm_sq_sum: float = 0.0

    def update(self, z: np.ndarray) -> None:
        """z: (B, D) numpy array (already pooled if originally (B, C, D))."""
        B, D = z.shape
        if self._mean is None:
            self._mean = np.zeros(D, dtype=np.float64)
            self._M2   = np.zeros(D, dtype=np.float64)

        # Batch Welford
        z64 = z.astype(np.float64, copy=False)
        batch_mean = z64.mean(axis=0)
        batch_M2   = ((z64 - batch_mean) ** 2).sum(axis=0)
        n_b = B
        if self._n == 0:
            self._mean = batch_mean
            self._M2   = batch_M2
            self._n    = n_b
        else:
            delta = batch_mean - self._mean
            new_n = self._n + n_b
            self._mean = self._mean + delta * (n_b / new_n)
            self._M2   = self._M2 + batch_M2 + (delta ** 2) * (self._n * n_b / new_n)
            self._n    = new_n

        # L2 norm running sums (for mean + std of per-sample norms)
        norms = np.linalg.norm(z64, axis=1)   # (B,)
        self._norm_sum    += float(norms.sum())
        self._norm_sq_sum += float((norms ** 2).sum())

    def get_metrics(self) -> dict:
        """Return 4 scalars. Empty dict if fewer than 2 samples."""
        if self._n < 2 or self._mean is None or self._M2 is None:
            return {}
        std_per_dim = np.sqrt(self._M2 / (self._n - 1))
        norm_mean = self._norm_sum / self._n
        norm_var  = max(self._norm_sq_sum / self._n - norm_mean ** 2, 0.0)
        return {
            "std_mean":  float(std_per_dim.mean()),
            "std_min":   float(std_per_dim.min()),
            "norm_mean": float(norm_mean),
            "norm_std":  float(np.sqrt(norm_var)),
        }

    def reset(self) -> None:
        self._n = 0
        self._mean = None
        self._M2 = None
        self._norm_sum = 0.0
        self._norm_sq_sum = 0.0


class EmbeddingWriter:
    """ Streams embedding vectors to per-field memmaps during a save
        epoch, then consolidates into a single embeddings_{epoch}.npz at
        close. Only I/O.
    """

    def __init__(self, emb_dir: Path, epoch: int, n_total: int,
                 max_ctx: int, embed_dim: int, active: bool = True):
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

        def _mmap(name: str, shape: tuple, dtype=np.float32):
            path = self._emb_dir / f"_tmp_{name}_{self._epoch}.npy"
            return np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape)

        self._mm_z_enc    = _mmap("z_encs",       (self._n_total, self._max_ctx, self._embed_dim), dtype=np.float16) # type: ignore
        self._mm_z_pred   = _mmap("z_pred",       (self._n_total, self._embed_dim), dtype=np.float16) # type: ignore
        self._mm_z_target = _mmap("z_target",     (self._n_total, self._embed_dim), dtype=np.float16) # type: ignore
        self._mm_mask_pos = _mmap("mask_pos",     (self._n_total,))
        self._mm_ctx_pad  = _mmap("ctx_pad_mask", (self._n_total, self._max_ctx))
        return self

    @property
    def active(self) -> bool:
        return self._active

    def write_batch(
        self,
        z_enc:        "torch.Tensor",   # (B, C, D)
        z_pred:       "torch.Tensor",   # (B, D)
        z_target:     "torch.Tensor",   # (B, D)
        mask_pos:     "torch.Tensor",   # (B,)
        ctx_pad_mask: "torch.Tensor",   # (B, C)
        subject_ids:  list[str],
    ) -> None:
        if not self._active:
            return

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

        self._mm_z_enc[lo:hi, :C, :] = z_enc_np       # type: ignore[index]
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
            out_mm = np.lib.format.open_memmap(
                sliced_path, mode="w+", dtype=out_dtype, shape=mm[:n].shape)
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


# ------------------------------------------------------------------
# -- Core Logging Class
class TrainingLogger:
    """ All-in-one CSV, TensorBoard, CLI, and embedding logger for
        loss, grad norms, JEPA embeddings, and custom metrics.
        Minimal memory overhead to fit on a single GPU.
    """
    def __init__(
        self,
        logdir: Path,
        epoch: int = 0,
        global_step: int = 0,
        loss_history: list[float] = [],
        embed_every: int | None = None,
        total_epochs: int | None = None,
        log_csv: bool = True,
        log_tb: bool = True,
        verbose: bool = False,
    ):
        self._closed = False
        self._run_start = time.time()
        self._verbose = verbose
        self._logdir = logdir / "logs"
        self._embdir = logdir / "embeddings"

        self._ep_samples: int = 0
        self.epoch_losses: list[float] = []
        self.loss_history: list[float] = loss_history
        self.epoch = epoch
        self.global_step = global_step
        self.grad_norms: list[float] = []

        # Embedding health trackers (updated every batch, every epoch)
        self.embed_tracker_z_enc    = EmbeddingTracker()
        self.embed_tracker_z_pred   = EmbeddingTracker()
        self.embed_tracker_z_target = EmbeddingTracker()

        # Embedding save schedule
        self._embed_every = embed_every
        self._total_epochs = total_epochs

        # initialized on first log_epoch
        self._log_csv = log_csv
        self._csv_writer = CsvWriter(logdir=self._logdir, fn=f"{self._logdir.parent.name}_metrics.csv")

        # TensorBoard
        self._log_tb = log_tb
        self._tb_writer = SummaryWriter(log_dir=str(logdir / "tb_logs"))

        print(f"[TrainingLogger] Logging it up in {logdir.parent.name}/{logdir.name}")

    # --- Utilities
    def is_embed_epoch(self, epoch: int) -> bool:
        if self._embed_every is None:
            return False
        return (epoch % self._embed_every == 0) or (epoch == self._total_epochs)
    
    def log_step_scalar(self, name: str, value: float) -> None:
        # Log a single scalar at the current global_step to TensorBoard
        if self._log_tb:
            self._tb_writer.add_scalar(name, value, self.global_step)

    def embedding_writer(
        self, 
        epoch: int, 
        n_total: int, 
        max_ctx: int, 
        embed_dim: int,
    ) -> EmbeddingWriter:
        return EmbeddingWriter(
            emb_dir=self._embdir,
            epoch=epoch,
            n_total=n_total,
            max_ctx=max_ctx,
            embed_dim=embed_dim,
            active=self.is_embed_epoch(epoch),
        )
        
    def get_norm_metrics(self) -> dict:
        if not self.grad_norms: return {}

        return {
            "grad_norm_mean": stats.mean(self.grad_norms),
            "grad_norm_max": max(self.grad_norms),
            "grad_norm_min": min(self.grad_norms),
            "grad_norm_std": stats.stdev(self.grad_norms) if len(self.grad_norms) > 1 else 0.0,
        }
        
    # --- step/epoch logging
    def log_batch(self, loss, s):
        self._ep_samples += s
        self.epoch_losses.append(float(loss))
        self.global_step += 1
        if self._log_tb:
            step_loss = float(loss) / s
            self._tb_writer.add_scalar("step/loss", step_loss, self.global_step)

    def log_grad_norm(self, grad_norm: int | float) -> None:
        self.grad_norms.append(float(grad_norm))
        self.log_step_scalar("step/grad_norm", grad_norm)

    def update_embed_health_single(self, z_enc_pooled: "torch.Tensor") -> None:
        # z_enc_pooled is already (B, D), no per-encounter pooling needed
        z_np = z_enc_pooled.detach().cpu().float().numpy()
        self.embed_tracker_z_enc.update(z_np)

    def update_embed_health(
        self,
        z_enc:        "torch.Tensor",  # (B, C, D)
        z_pred:       "torch.Tensor",  # (B, D)
        z_target:     "torch.Tensor",  # (B, D)
        ctx_pad_mask: "torch.Tensor",  # (B, C)
    ) -> None:
        # One device->host sync, shared across all three trackers.
        z_enc_np    = z_enc.detach().cpu().float().numpy()        # (B, C, D)
        z_pred_np   = z_pred.detach().cpu().float().numpy()       # (B, D)
        z_target_np = z_target.detach().cpu().float().numpy()     # (B, D)
        pad_np      = ctx_pad_mask.detach().cpu().float().numpy() # (B, C)

        # Pool z_enc over valid positions
        valid     = (pad_np == 0).astype(np.float32)              # (B, C)
        valid_sum = valid.sum(axis=1, keepdims=True).clip(min=1)  # (B, 1)
        z_enc_pooled = (z_enc_np * valid[..., None]).sum(axis=1) / valid_sum

        self.embed_tracker_z_enc.update(z_enc_pooled)
        self.embed_tracker_z_pred.update(z_pred_np)
        self.embed_tracker_z_target.update(z_target_np)

    # --- Final logging
    def log_epoch(self, lr: float | None = None, **metrics):
        self._logdir.mkdir(parents=True, exist_ok=True)

        # -- collect
        self.epoch += 1
        wall_time = time.time() - self._run_start
        epoch_loss = float(sum(self.epoch_losses) / self._ep_samples)
        self.loss_history.append(epoch_loss)
        self.epoch_losses.clear()

        ep_metrics = {
            "epoch": self.epoch,
            "wall_sec": round(wall_time, 1),
            "lr": self._fmt(lr),
            "loss": self._fmt(epoch_loss),
        }

        cli_pp: dict = {
            "lr": self._fmt(lr),
            "loss": self._fmt(epoch_loss)
        }

        # --- grad norms
        if len(self.grad_norms) > 0:
            for k, v in self.get_norm_metrics().items():
                ep_metrics[k] = self._fmt(v)
                if k == "grad_norm_mean" or self._verbose:
                    cli_pp[k] = self._fmt(v)
        self.grad_norms.clear()

        # -- always-on: embed health --
        embed_health = {}
        for tag, tracker in [
            ("z_enc",    self.embed_tracker_z_enc),
            ("z_pred",   self.embed_tracker_z_pred),
            ("z_target", self.embed_tracker_z_target),
        ]:
            m = tracker.get_metrics()
            for k, v in m.items():
                embed_health[f"embed_{tag}_{k}"] = v
            tracker.reset()
        for k, v in embed_health.items():
            ep_metrics[k] = self._fmt(v)

        # -- extras from caller (always CSV/TB, verbose-gated for CLI)
        ep_metrics["steps"] = self.global_step
        for k, v in metrics.items():
            ep_metrics[k] = self._fmt(v)
        if self._verbose:
            cli_pp["steps"] = self.global_step
            for k, v in metrics.items():
                cli_pp[k] = self._fmt(v)

        # -- log to CLI
        print(" | ".join(
            [f"[{self.epoch}]"] + 
            [f"{k}={self._fmt(v)}" for k, v in cli_pp.items() if k != "wall_sec"] + 
            [f"[{round(wall_time, 1):f}s]"]
        ))

        # -- log to CSV
        if self._log_csv:
            self._csv_writer.write(ep_metrics)

        # -- log to Tensorboard
        if self._log_tb:
            self._tb_writer.add_scalar("mem/allocated", torch.cuda.memory_allocated() / 1e9, self.epoch)
            self._tb_writer.add_scalar("mem/reserved", torch.cuda.memory_reserved() / 1e9, self.epoch)
            self._tb_writer.add_scalar("mem/peak", torch.cuda.max_memory_allocated() / 1e9, self.epoch)
            if lr is not None:
                self._tb_writer.add_scalar("epoch/lr", lr, self.epoch)
            for key, value in ep_metrics.items():
                if key in ("epoch", "wall_sec", "steps") or value in (None, ""):
                    continue
                v = int(value) if isinstance(value, bool) else float(value)
                if key.startswith("embed_"):
                    tb_key = f"embed/{key[len('embed_'):]}"
                else:
                    tb_key = f"epoch/{key}"
                self._tb_writer.add_scalar(tb_key, v, self.epoch)
            self._tb_writer.flush()

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self):
        import json
        from datetime import datetime
        from src.analysis.plotting import plot_loss_curve

        total_time = time.time() - self._run_start

        with open(self._logdir / "run_summary.json", "w") as f:
            json.dump({
                "total_epochs": self.epoch,
                "total_steps": self.global_step,
                "wall_time_sec": round(total_time, 1),
                "wall_time_human": self._pp_time(total_time),
                "finished_at": datetime.now().isoformat(),
            }, f, indent=2, default=str)

        plot_loss_curve(self.loss_history, show=False, save=True, fig_dir=self._logdir)

        self._tb_writer.flush()
        self._tb_writer.close()
        self._closed = True
        print(f"\n[TrainingLogger] Run complete in {self._pp_time(total_time)}")

    # -- Utility
    @staticmethod
    def _fmt(v) -> str:
        if v is None: return ""

        if isinstance(v, float):
            return f"{v:.6f}" if abs(v) < 1 else f"{v:.4f}"
        return str(v)

    @staticmethod
    def _pp_time(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"


class DriftMonitor:
    """Tracks encoder drift between online and target encoders on a fixed probe batch.

    For EMA JEPA, the target encoder f_ξ diverges from the online encoder f_θ.
    This means the prediction residual P - T = predictor(f_θ(ctx)) - f_ξ(x_t)
    conflates (1) true prediction error and (2) encoder drift f_θ(x_t) - f_ξ(x_t).
    This monitor quantifies (2) so the confound can be reported and bounded.
    
    the ratio drift_over_pred tells us what fraction of the prediction residual is 
    actually encoder drift. A single frozen cpu batch is measured to ensure 
    trajectories are are comparable. If this ratio is >0.05, the the P - T confound
    is not bounded and residual based cliams need strong caveats.
    """

    def __init__(self):
        self._probe_batch = None  # stored on CPU

    def set_probe(self, batch: dict) -> None:
        """Store a fixed probe batch (CPU tensors). Called once at start of training."""
        self._probe_batch = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def compute(self, model, device) -> dict:
        """Run one forward pass on the probe and return drift metrics."""
        if self._probe_batch is None:
            return {}
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self._probe_batch.items()
        }
        model.eval()
        
        # Online and target encoder outputs on the same target tokens
        z_online = model.encoder(batch["tgt_tokens"], batch["tgt_tok_mask"])
        z_target = model.target_encoder(batch["tgt_tokens"], batch["tgt_tok_mask"])
        
        # Full forward pass for prediction residual
        _, z_pred, z_t = model(batch)
        model.train()

        # calculate drift
        drift    = z_online - z_target                  # (B, D)
        pred_err = z_pred - z_t                         # (B, D)

        drift_l2    = drift.norm(dim=-1)                # (B,)
        pred_err_l2 = pred_err.norm(dim=-1)             # (B,)

        cos_drift_err = torch.nn.functional.cosine_similarity(drift, pred_err, dim=-1)

        return {
            "drift_l2_mean":     drift_l2.mean().item(),
            "drift_l2_max":      drift_l2.max().item(),
            "pred_err_l2_mean":  pred_err_l2.mean().item(),
            "drift_over_pred":   (drift_l2.mean() / pred_err_l2.mean().clamp_min(1e-8)).item(),
            "drift_cos_pred":    cos_drift_err.mean().item(),
        }


class EMAMonitor:
    """
    Track encoder divergence from context encoder.

    Usage:
        Inside EMA update:
            ema_tracker.update(context_encoder, ema_encoder)
        At epoch end:
            logger.log_epoch(..., **ema_tracker.get_metrics())
            ema_tracker.reset()
    """

    def __init__(self):
        self._param_diffs: list[float] = []

    def update(self, online_model, ema_model):
        # -- Cheap computes mean absolute param divergence.
        total_diff = 0.0
        total_params = 0
        for p_online, p_ema in zip(online_model.parameters(), ema_model.parameters()):
            total_diff += (p_online.data - p_ema.data).abs().sum().item()
            total_params += p_online.numel()
        self._param_diffs.append(total_diff / max(total_params, 1))

    def get_metrics(self) -> dict:
        if not self._param_diffs:
            return {}
        return {
            "ema_param_divergence": sum(self._param_diffs) / len(self._param_diffs),
        }

    def reset(self):
        self._param_diffs.clear()
