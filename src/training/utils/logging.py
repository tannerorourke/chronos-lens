import time
import csv
from pathlib import Path
import statistics as stats
\
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class GradientMonitor:
    def __init__(self, model=None):
        self.model = model
        self._norms: list[float] = []

    def capture(self, mparams):
        params = mparams if self.model is None else self.model.parameters()
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self._norms.append(total_norm ** 0.5)

    def get_metrics(self) -> dict:
        if not self._norms: return {}
        
        return {
            "grad_norm_mean": stats.mean(self._norms),
            "grad_norm_max": max(self._norms),
            "grad_norm_min": min(self._norms),
            "grad_norm_std": stats.stdev(self._norms) if len(self._norms) > 1 else 0.0,
        }

    def reset(self):
        self._norms.clear()
        
        
class CsvWriter:
    def __init__(self, logdir: Path):
        self._closed = False
        self._logpath = logdir / "epoch_metrics.csv"
        self._writer = None
        self._file = open(self._logpath, "w", newline="")
        
    def write(self, metrics: dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, 
                                          fieldnames=list(metrics.keys()))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()
        
    def __del__(self):
        if not self._closed:
            self._file.close()
            
            
class EmbeddingTracker:
    """Accumulates per-dimension statistics incrementally from batches so the
    full embedding matrix never needs to live in RAM simultaneously.
    
    Uses Welford online mean/variance per embedding dimension.
 
    Usage:
        stats = _RunningEmbeddingStats()
        for batch in ...:
            stats.update(z)          # (B, D) numpy float32
        mean_std, min_std = stats.finalize()
    """
 
    def __init__(self):
        self._n     = 0
        self._mean  = None
        self._M2    = None
        self._norms = []       # list of per-sample L2 norms (kept as means per batch)
 
    def update(self, z: np.ndarray) -> None:
        """Update with a batch (B, D)."""
        B, D = z.shape
        if self._mean is None:
            self._mean = np.zeros(D, dtype=np.float64)
            self._M2   = np.zeros(D, dtype=np.float64)
 
        # Welford batch update
        for i in range(B):
            self._n += 1
            delta      = z[i] - self._mean
            self._mean += delta / self._n
            delta2     = z[i] - self._mean
            self._M2  += delta * delta2
 
        # Store mean norm for this batch (cheap)
        self._norms.append(float(np.linalg.norm(z, axis=1).mean()))
 
    def finalize(self) -> dict:
        """Return collapse diagnostic scalars."""
        if self._n < 2:
            return {}
        std_per_dim = stats.sqrt(self._M2 / (self._n - 1))   # (D,)
        norm_mean   = float(stats.mean(self._norms))
        norm_std    = float(stats.std(self._norms))
        return {
            "std_mean": float(std_per_dim.mean()),
            "std_min":  float(std_per_dim.min()),
            "norm_mean": norm_mean,
            "norm_std":  norm_std,
        }
 
    def reset(self) -> None:
        self._n    = 0
        self._mean = None
        self._M2   = None
        self._norms.clear()
        
        
class TrainingLogger:
    """ All-in-one CSV, TensorBoard, CLI, and embedding logger for
        loss, grad norms, JEPA embeddings, and custom metrics
    """
    def __init__(
        self, 
        logdir: Path, 
        epoch: int = 0,
        global_step: int = 0, 
        loss_history: list[float] = None, 
        log_norms: bool = True,
        log_csv: bool = True,
        log_tb: bool = True,
        verbose: bool = False
    ):
        self._closed = False
        self._run_start = time.time()
        self._verbose = verbose
        self._logdir = logdir / "logs"
        self._embdir = logdir / "embeddings"
        
        self._ep_samples: int = 0
        self.epoch_losses: list[float] = []
        self.loss_history: list[float] = loss_history if loss_history is not None else []
        self.epoch = epoch
        self.global_step = global_step
        
        self._log_norms = log_norms
        self.grad_mon = GradientMonitor()
        
        # initialized on first log_epoch
        self._log_csv = log_csv
        self._csv_writer = CsvWriter(logdir=self._logdir)

        # TensorBoard
        self._log_tb = log_tb
        self._tb_writer = SummaryWriter(log_dir=str(logdir / "tb_logs"))

        print(f"[TrainingLogger] Logging it up in {logdir.parent.name}/{logdir.name}")
            
    # -- step/epoch/final logging --
    def log_batch(self, loss, s):
        self._ep_samples += s
        self.epoch_losses.append(float(loss))
        self.global_step += 1
        if self._log_tb:
            step_loss = float(loss) / s
            self._tb_writer.add_scalar("step/loss", step_loss, self.global_step)
    
    
    def log_epoch(self, lr: float | None = None, **metrics):
        self._logdir.mkdir(parents=True, exist_ok=True)
        
        # -- collect --
        self.epoch += 1
        wall_time = time.time() - self._run_start
        epoch_loss = float(sum(self.epoch_losses) / self._ep_samples)
        norm_m = self.grad_mon.get_metrics()
        self.loss_history.append(epoch_loss)
        self.epoch_losses.clear()
        self.grad_mon.reset()
        
        ep_metrics = {
            "epoch": self.epoch,
            "wall_sec": round(wall_time, 1),
            "lr": self._fmt(lr),
            "loss": self._fmt(epoch_loss),
        }
        
        cli_pp = {
            "epoch": self.epoch,
            "lr": self._fmt(lr),
            "loss": self._fmt(epoch_loss),
            "wall_sec": round(wall_time, 1),
        }
        
        # -- grad norms --
        if len(norm_m) > 0 and self._log_norms:
            for k, v in norm_m.items():
                metrics[k] = v
                if k == "grad_norm_mean" or self._verbose:
                    cli_pp[k] = self._fmt(v)
            
        # -- extra metrics --
        if self._verbose:
            cli_pp["steps"] = self.global_step
            ep_metrics["steps"] = self.global_step
            for k, v in metrics.items():
                ep_metrics[k] = self._fmt(v)
                cli_pp[k] = self._fmt(v)
                
        # -- CLI --
        cli = [f"{k}={self._fmt(v)}" for k, v in cli_pp.items() if k != "wall_sec"]
        cli.append(f"[{wall_time:.0f}s]")
        print(" | ".join(cli))

        # -- log to CSV --
        if self._log_csv:
            self._csv_writer.write(ep_metrics)
        
        # -- log to Tensorboard --
        if self._log_tb:
            if lr is not None:
                self._tb_writer.add_scalar("epoch/lr", lr, self.epoch)
            for key, value in ep_metrics.items():
                if key in ("epoch", "wall_sec", "steps") or value in (None, ""):
                    continue
                v = int(value) if isinstance(value, bool) else float(value)
                self._tb_writer.add_scalar(f"epoch/{key}", v, self.epoch)
            self._tb_writer.flush()


    def open_embedding_writer(
        self,
        epoch: int,
        n_total: int,
        max_ctx: int,
        embed_dim: int,
        emb_dir: Path,
    ) -> None:
        """Pre-allocate memmaps for a save epoch.
 
        Parameters
        ----------
        epoch     : current epoch number (used for filenames)
        n_total   : total number of samples in the dataset
        max_ctx   : maximum number of context encounters (C dimension)
        embed_dim : embedding dimension (D)
        emb_dir   : directory to write memmap files into
        """
        emb_dir.mkdir(parents=True, exist_ok=True)
        self._emb_dir   = emb_dir
        self._emb_epoch = epoch
        self._write_idx = 0
        self._n_total   = n_total
        self._subject_ids.clear()
 
        def _mmap(name: str, shape: tuple, dtype=np.float32) -> np.ndarray:
            path = emb_dir / f"{name}_{epoch}.npy"
            return np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape)
 
        self._mm_z_enc    = _mmap("z_encs",       (n_total, max_ctx, embed_dim))
        self._mm_z_pred   = _mmap("z_pred",        (n_total, embed_dim))
        self._mm_z_target = _mmap("z_target",      (n_total, embed_dim))
        self._mm_mask_pos = _mmap("mask_pos",       (n_total,), dtype=np.float32)
        self._mm_ctx_pad  = _mmap("ctx_pad_mask",   (n_total, max_ctx), dtype=np.float32)
 
        # Reset running stats
        self._stats_z_enc.reset()
        self._stats_z_pred.reset()
        self._stats_z_target.reset()
 
        self._emb_active = True
        print(f"[EmbWriter] Opened memmaps for epoch {epoch} "
              f"(n={n_total}, max_ctx={max_ctx}, D={embed_dim})")
 
    def write_embedding_batch(
        self,
        z_enc:       torch.Tensor,   # (B, C, D)
        z_pred:      torch.Tensor,   # (B, D)
        z_target:    torch.Tensor,   # (B, D)
        mask_pos:    torch.Tensor,   # (B,)
        ctx_pad_mask: torch.Tensor,  # (B, C)
        subject_ids: list[str],
    ) -> None:
        """Write one batch to the open memmaps and update running stats.
 
        Safe to call even if open_embedding_writer() was not called — it
        is a no-op when the writer is inactive.
        """
        if not self._emb_active:
            return
 
        # --- to numpy ---
        z_enc_np    = z_enc.detach().cpu().float().numpy()      # (B, C, D)
        z_pred_np   = z_pred.detach().cpu().float().numpy()     # (B, D)
        z_target_np = z_target.detach().cpu().float().numpy()   # (B, D)
        mask_np     = mask_pos.cpu().float().numpy()             # (B,)
        pad_np      = ctx_pad_mask.cpu().float().numpy()         # (B, C)
 
        B  = z_pred_np.shape[0]
        C  = z_enc_np.shape[1]
        lo = self._write_idx
        hi = lo + B
 
        # Guard against dataset size mismatch (e.g. drop_last=False last batch)
        if hi > self._n_total:
            hi = self._n_total
            B  = hi - lo
            z_enc_np    = z_enc_np[:B]
            z_pred_np   = z_pred_np[:B]
            z_target_np = z_target_np[:B]
            mask_np     = mask_np[:B]
            pad_np      = pad_np[:B]
            subject_ids = subject_ids[:B]
 
        # Write to memmaps — C may be < max_ctx; pad columns stay zero
        self._mm_z_enc[lo:hi, :C, :]  = z_enc_np
        self._mm_z_pred[lo:hi]        = z_pred_np
        self._mm_z_target[lo:hi]      = z_target_np
        self._mm_mask_pos[lo:hi]      = mask_np
        self._mm_ctx_pad[lo:hi, :C]   = pad_np
        self._subject_ids.extend(subject_ids)
        self._write_idx = hi
 
        # --- Update running stats ---
        # Pool z_enc over valid positions for collapse stats only
        valid     = (pad_np == 0)                                   # (B, C) True=real
        valid_sum = valid.sum(axis=1, keepdims=True).clip(min=1)    # (B, 1)
        z_enc_pooled = (z_enc_np * valid[..., None]).sum(axis=1) / valid_sum  # (B, D)
 
        self._stats_z_enc.update(z_enc_pooled)
        self._stats_z_pred.update(z_pred_np)
        self._stats_z_target.update(z_target_np)
 
    def _close_embedding_writer(self) -> None:
        """Flush memmaps and save subject_ids. Called internally by log_epoch."""
        if not self._emb_active:
            return
 
        # Flush memmaps
        for mm in (self._mm_z_enc, self._mm_z_pred, self._mm_z_target,
                   self._mm_mask_pos, self._mm_ctx_pad):
            if mm is not None:
                mm.flush()
 
        # Subject IDs as numpy string array
        ids_path = self._emb_dir / f"subject_ids_{self._emb_epoch}.npy"
        np.save(ids_path, np.array(self._subject_ids, dtype=str))
 
        actual = self._write_idx
        print(f"[EmbWriter] Saved embeddings for epoch {self._emb_epoch} "
              f"({actual}/{self._n_total} samples) -> {self._emb_dir.name}/")
 
        # Clean up references (memmaps stay on disk)
        self._mm_z_enc = self._mm_z_pred = self._mm_z_target = None
        self._mm_mask_pos = self._mm_ctx_pad = None
        self._emb_active = False
 
    def _drain_collapse_stats(self) -> dict:
        """Finalize running stats and emit TensorBoard scalars. Returns dict for CSV."""
        out = {}
        for tag, stats in [
            ("z_enc",    self._stats_z_enc),
            ("z_pred",   self._stats_z_pred),
            ("z_target", self._stats_z_target),
        ]:
            s = stats.finalize()
            for metric, value in s.items():
                key = f"collapse_{tag}_{metric}"
                out[key] = value
                self.writer.add_scalar(f"collapse/{tag}/{metric}", value, self.epoch)
        return out
 
    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
 
    def finalize(self) -> None:
        if self._emb_active:
            self._close_embedding_writer()
 
        total_time = time.time() - self.run_start
        self._write_json("run_summary.json", {
            "total_epochs":    self.epoch,
            "total_steps":     self.global_step,
            "wall_time_sec":   round(total_time, 1),
            "wall_time_human": self._pp_time(total_time),
            "finished_at":     datetime.now().isoformat(),
        })
        self.writer.flush()
        self.writer.close()
        self._csv_file.close()
        self._closed = True
        print(f"\n[TrainingLogger] Run complete in {self._pp_time(total_time)}")






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

    def __del__(self):
        if not self._closed:
            self._tb_writer.close()
    

# ===========================================================================
# 2. EMA Tracker
# ===========================================================================

class EMATracker:
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


