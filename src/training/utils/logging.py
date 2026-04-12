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


# ------------------------------------------------------------------
# -- Core Logging Class
class TrainingLogger:
    """ All-in-one CSV, TensorBoard, CLI, and embedding logger for
        loss, grad norms, JEPA embeddings, and other metrics.
        Minimal memory overhead to fit on a single GPU.
    """
    def __init__(
        self,
        logdir: Path,
        epoch: int = 0,
        global_step: int = 0,
        loss_history: list[float] = [],
        total_epochs: int | None = None,
        log_csv: bool = True,
        log_tb: bool = True,
        verbose: bool = False,
    ):
        self._closed = False
        self._verbose = verbose
        self._total_epochs = total_epochs
        self._logdir = logdir / "logs"
        self._embdir = logdir / "embeddings"

        self._ep_samples: int = 0
        self.epoch_losses: list[float] = []
        self.loss_history: list[float] = loss_history
        self.epoch = epoch
        self.global_step = global_step
        self.grad_norms: list[float] = []

        # Embedding health trackers (updated p/batch, p/epoch)
        self.embed_tracker_z_enc    = EmbeddingTracker()
        self.embed_tracker_z_pred   = EmbeddingTracker()
        self.embed_tracker_z_target = EmbeddingTracker()

        # CSV/TB
        self._log_csv = log_csv
        self._csv_writer = CsvWriter(logdir=self._logdir, fn=f"{self._logdir.parent.name}_metrics.csv")
        self._log_tb = log_tb
        self._tb_writer = SummaryWriter(log_dir=str(logdir / "tb_logs"))

        self._ep_start = 0.0
        self._run_start = 0.0
        print(f"[TrainingLogger] Logging it up in {logdir.parent.name}/{logdir.name}")

    # --- Utilities
    def lap(self):
        self._ep_start = time.time()
        if self._run_start == 0.0:
            self._run_start = time.time()
        
    def log_step_scalar(self, name: str, value: float) -> None:
        # Log a single scalar at the current global_step to TensorBoard
        if self._log_tb:
            self._tb_writer.add_scalar(name, value, self.global_step)
        
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

    def update_embed_health_single(self, z_enc_pooled: torch.Tensor) -> None:
        # z_enc_pooled is already (B, D), no per-encounter pooling needed
        z_np = z_enc_pooled.detach().cpu().float().numpy()
        self.embed_tracker_z_enc.update(z_np)

    def update_embed_health(
        self,
        z_enc:        torch.Tensor,  # (B, C, D)
        z_pred:       torch.Tensor,  # (B, D)
        z_target:     torch.Tensor,  # (B, D)
        ctx_pad_mask: torch.Tensor,  # (B, C)
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
        
        # additional step-level collapse early warning
        if self.global_step % 20 == 0:
            self.log_step_scalar("step/z_target_std_min", z_target.std(0).min().item())

    # --- Final logging
    def log_epoch(self, lr: float | None = None, **metrics) -> dict:
        self._logdir.mkdir(parents=True, exist_ok=True)

        self.epoch += 1
        wall_time = time.time() - self._run_start
        ep_time = time.time() - self._ep_start
        epoch_loss = float(sum(self.epoch_losses) / self._ep_samples)
        self.loss_history.append(epoch_loss)
        self.epoch_losses.clear()

        raw_metrics = {
            "epoch": self.epoch,
            "wall_sec": wall_time,
            "lr": lr,
            "loss": epoch_loss,
        }

        cli_pp: dict = {
            "lr": self._fmt(lr),
            "loss": self._fmt(epoch_loss)
        }

        # --- grad norms
        if len(self.grad_norms) > 0:
            for k, v in self.get_norm_metrics().items():
                raw_metrics[k] = v
                if k == "grad_norm_mean" or self._verbose:
                    cli_pp[k] = self._fmt(v)
        self.grad_norms.clear()

        # -- always-on: embed health
        for tag, tracker in [
            ("z_enc",    self.embed_tracker_z_enc),
            ("z_pred",   self.embed_tracker_z_pred),
            ("z_target", self.embed_tracker_z_target),
        ]:
            m = tracker.get_metrics()
            for k, v in m.items():
                raw_metrics[f"embed_{tag}_{k}"] = v
            tracker.reset()

        # -- extras from caller (always CSV/TB, verbose-gated CLI)
        raw_metrics["steps"] = self.global_step
        for k, v in metrics.items():
            raw_metrics[k] = v
        if self._verbose:
            cli_pp["steps"] = self.global_step
            for k, v in metrics.items():
                cli_pp[k] = self._fmt(v)

        # -- formatted for CSV/TB
        log_metrics = {k: self._fmt(v) if k not in ("epoch", "wall_sec", "steps") else v
                      for k, v in raw_metrics.items()}

        # -- log to CLI
        print(" | ".join(
            [f"[{self.epoch}]"] + 
            [f"{k}={self._fmt(v)}" for k, v in cli_pp.items() if k != "wall_sec"] + 
            [f"[{self._pp_time(ep_time)} ({ep_time:.0f})]"]
        ))

        # -- log to CSV/TB
        if self._log_csv:
            self._csv_writer.write(log_metrics)

        if self._log_tb:
            self._tb_writer.add_scalar("mem/allocated", torch.cuda.memory_allocated() / 1e9, self.epoch)
            self._tb_writer.add_scalar("mem/peak", torch.cuda.max_memory_allocated() / 1e9, self.epoch)
            if lr is not None:
                self._tb_writer.add_scalar("epoch/lr", lr, self.epoch)
            for key, value in log_metrics.items():
                if key in ("epoch", "wall_sec", "steps") or value in (None, ""):
                    continue
                v = int(value) if isinstance(value, bool) else float(value)
                if key.startswith("embed_"):
                    tb_key = f"embed/{key[len('embed_'):]}"
                else:
                    tb_key = f"epoch/{key}"
                self._tb_writer.add_scalar(tb_key, v, self.epoch)
            self._tb_writer.flush()
            
        return raw_metrics

    def finalize(self):
        import json
        from datetime import datetime
        from src.analysis.plotting import plot_loss_curve

        total_time = time.time() - self._run_start

        with open(self._logdir / "run_summary.json", "w") as f:
            json.dump({
                "total_epochs": self.epoch,
                "total_steps": self.global_step,
                "wall_time": self._pp_time(total_time),
                "finished_at": datetime.now().isoformat(),
            }, f, indent=2, default=str)

        # plot_loss_curve(self.loss_history, show=False, save=True, fig_dir=self._logdir)
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