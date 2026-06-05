import time
import csv
import json
from pathlib import Path
import statistics as stats

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.utils.constants import ARCHITECTURES


_TITLE_PT = 10


def plot_loss_curve(
    loss_history: list[float],
    save_path: Path | str | None = None,
    title: str = "Training Loss",
    ylabel: str = "MSE Loss",
) -> None:
    """ Self-contained training-side utility for plotting a training loss curve """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=_TITLE_PT)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"Saved fig: {save_path}")
    plt.close(fig)


# ------------------------------------------------------------------
# JSON sink
# ------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return str(o)


class JsonlWriter:
    """Append-only JSON-lines metrics sink - the canonical streamed sink.

    One ``json.dumps(record)`` per line, flushed immediately so it is tail-able
    live under VS Code Remote-SSH::

        tail -f metrics.jsonl | jq 'select(.type=="epoch")'

    Diff-friendly, S3-syncable, durable across instance teardown, and the format
    the analysis tooling parses. Opened in append mode so resumed runs extend the
    same file.
    """

    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = open(path, "a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record, default=_json_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass

# ------------------------------------------------------------------
# Drift Monitor
# ------------------------------------------------------------------

class DriftMonitor:
    """Tracks encoder drift between online and target encoders on a fixed probe batch.

    For EMA JEPA, the target encoder f_xi diverges from the online encoder f_theta.
    This means the prediction residual P - T = predictor(f_theta(ctx)) - f_xi(x_t)
    conflates (1) true prediction error and (2) encoder drift f_theta(x_t) - f_xi(x_t).
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
        pb = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self._probe_batch.items()
        }
        model.eval()
        
        # Online and target encoder outputs on the same target tokens
        z_online = model.encoder(pb["tgt_tokens"], pb["tgt_tok_mask"], pb["tgt_times"], pool=True)
        target_encoder = getattr(model, "target_encoder", None)
        z_target = (target_encoder(pb["tgt_tokens"], pb["tgt_tok_mask"], pb["tgt_times"], pool=True)
                    if target_encoder is not None else z_online)  # SG: drift identically 0
        
        # Full forward pass for prediction residual
        outputs = model(pb)
        z_pred, z_t = outputs[1], outputs[2]
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

# ------------------------------------------------------------------
# Streamed diagnostics
# ------------------------------------------------------------------


def extract_time_scale(model) -> float | None:
    """Return the learnable temporal ``time_scale`` parameter (encoder), or None.

    Robust to nesting: searches ``named_parameters()`` for the ``time_scale``
    leaf rather than hard-coding an attribute path.
    """
    if model is None:
        return None
    try:
        for name, p in model.named_parameters():
            if name.rsplit(".", 1)[-1] == "time_scale":
                return float(p.detach().cpu().reshape(-1)[0])
    except Exception:
        return None
    return None


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
        """z: (M, D) numpy array - per-encounter rows (flattened valid encounters)
        for z_enc, or per-sample (B, D) for the predictor vectors."""
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
        """std_mean, std_min, norm_mean, norm_std: Empty dict if fewer than 2 samples."""
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
    """All-in-one metrics logger for loss, grad norms, JEPA embeddings, etc.

    ``metrics.jsonl`` (run root) is the canonical streamed sink - always on,
    tail-able live, S3-syncable, durable across teardown. TensorBoard, CSV, and
    Weights & Biases are strictly OPTIONAL (default OFF) and degrade gracefully.
    Minimal memory overhead to fit on a single GPU.
    """
    def __init__(
        self,
        logdir: Path,
        epoch: int = 0,
        global_step: int = 0,
        loss_history: list[float] = [],
        total_epochs: int | None = None,
        log_jsonl: bool = True,
        log_csv: bool = False,
        log_tb: bool = False,
        sync_s3: bool = True,
        verbose: bool = False,
        arch: ARCHITECTURES = "ema",
    ):
        self.is_jepa = arch in ["ema", "stopgrad"]
        self._closed = False
        self._verbose = verbose
        self._total_epochs = total_epochs
        self._run_root = Path(logdir)
        self._logdir = self._run_root / "logs"
        self._embdir = self._run_root / "embeddings"

        self._ep_samples: int = 0
        self.epoch_losses: list[float] = []
        self.loss_history: list[float] = loss_history
        self.epoch = epoch
        self.global_step = global_step
        self.grad_norms: list[float] = []

        # last-seen per-step values, stashed for the streamed step record
        self._last_grad_norm: float | None = None
        self._last_pred_err: float | None = None
        self._last_lr: float | None = None

        # Embedding health trackers (updated p/batch, p/epoch)
        if self.is_jepa:
            self.embed_tracker_z_enc    = EmbeddingTracker()
            self.embed_tracker_z_pred   = EmbeddingTracker()
            self.embed_tracker_z_target = EmbeddingTracker()
        else:
            self.embed_tracker_z_enc    = EmbeddingTracker()

        # --- canonical sink: metrics.jsonl at the run root (always on) ------
        self._log_jsonl = log_jsonl
        self._jsonl = JsonlWriter(self._run_root / "metrics.jsonl") if log_jsonl else None

        # --- optional sinks (default OFF) -----------------------------------
        self._log_csv = log_csv
        self._csv_writer = (
            CsvWriter(logdir=self._logdir, fn=f"{self._run_root.name}_metrics.csv")
            if log_csv else None)

        self._log_tb = log_tb
        self._tb_writer = None
        if log_tb:  # only create tb_logs/ when explicitly enabled
            self._tb_writer = SummaryWriter(log_dir=str(logdir / "tb_logs"))

        # --- optional non-blocking S3 archive of the run dir ----------------
        from src.infra.s3 import S3Syncer
        self._syncer = S3Syncer(self._run_root, enabled=sync_s3)

        self._ep_start = 0.0
        self._run_start = 0.0
        print(f"[TrainingLogger] streaming -> {self._run_root.name}/metrics.jsonl"
              + (" +tb" if log_tb else "") + (" +csv" if log_csv else "") + (" +s3" if self._syncer.enabled else ""))

    def sync(self, *, blocking: bool = False) -> None:
        """Trigger a (non-blocking) S3 sync of the run dir. No-op unless enabled."""
        self._syncer.sync(blocking=blocking)

    def push_checkpoint(self, local_path, *, verify: bool = True) -> bool:
        """Blocking, verified S3 push of a single irreplaceable file (checkpoint or
        final embeddings npz) under the run root. No-op unless ``sync_s3``.
        """
        local_path = Path(local_path)
        rel = local_path.relative_to(self._run_root).as_posix()
        return self._syncer.push(rel, verify=verify)

    # --- Utilities
    def lap(self):
        self._ep_start = time.time()
        if self._run_start == 0.0:
            self._run_start = time.time()
        
    def log_step_scalar(self, name: str, value: float) -> None:
        # Log a single scalar at the current global_step to TensorBoard
        if self._log_tb:
            assert self._tb_writer is not None, "TensorBoard writer not initialized in log_step_scalar"
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
        step_loss = float(loss) / s
        if self._log_tb:
            assert self._tb_writer is not None, "TensorBoard writer not initialized in log_batch"
            self._tb_writer.add_scalar("step/loss", step_loss, self.global_step)
        # canonical per-step record (loss, grad_norm, lr, z_pred_err_l2)
        if self._jsonl is not None:
            self._jsonl.write({
                "type": "step",
                "step": self.global_step,
                "epoch": self.epoch,
                "loss": step_loss,
                "grad_norm": self._last_grad_norm,
                "lr": self._last_lr,
                "z_pred_err_l2": self._last_pred_err,
            })


    def log_grad_norm(self, grad_norm: int | float) -> None:
        self.grad_norms.append(float(grad_norm))
        self._last_grad_norm = float(grad_norm)
        self.log_step_scalar("step/grad_norm", grad_norm)

    def update_embed_health(
        self,
        z_enc:        torch.Tensor,         # (B, C, D)
        ctx_pad_mask: torch.Tensor,         # (B, C) True = padding
        z_pred:       torch.Tensor = None,  # (B, D), JEPA only
        z_target:     torch.Tensor = None,  # (B, D), JEPA only
    ) -> None:
        """Track `z_enc` collapse health over the **flattened valid encounters**
        `z_enc[~ctx_pad_mask]` - the same per-encounter population VICReg
        regularizes - identically for every arch. JEPA additionally passes
        `z_pred`/`z_target` for predictor-side health + the pred-error scalar.
        """
        z_enc_np = z_enc.detach().cpu().float().numpy()           # (B, C, D)
        valid    = ~ctx_pad_mask.detach().cpu().bool().numpy()    # (B, C) True = real
        self.embed_tracker_z_enc.update(z_enc_np[valid])          # (N_valid, D)

        if z_pred is None or z_target is None:
            return

        z_pred_np   = z_pred.detach().cpu().float().numpy()       # (B, D)
        z_target_np = z_target.detach().cpu().float().numpy()     # (B, D)
        self.embed_tracker_z_pred.update(z_pred_np)
        self.embed_tracker_z_target.update(z_target_np)

        pred_err_l2 = np.linalg.norm(z_pred_np - z_target_np, axis=-1)
        self._last_pred_err = float(pred_err_l2.mean())
        self.log_step_scalar("step/z_pred_err_l2_mean", self._last_pred_err)

    # --- Final logging
    def log_epoch(self, lr: float | None = None, model=None, **metrics) -> dict:
        if self._log_csv:
            self._logdir.mkdir(parents=True, exist_ok=True)

        self.epoch += 1
        self._last_lr = lr
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

        ts = extract_time_scale(model)
        if ts is not None:
            raw_metrics["time_scale"] = ts

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
        trackers = [("z_enc", self.embed_tracker_z_enc)]
        if self.is_jepa:
            trackers.append(("z_pred", self.embed_tracker_z_pred))
            trackers.append(("z_target", self.embed_tracker_z_target))
        for tag, tracker in trackers:
            m = tracker.get_metrics()
            for k, v in m.items():
                raw_metrics[f"embed_{tag}_{k}"] = v
            tracker.reset()

        # -- extras from caller (always CSV/TB/jsonl, verbose-gated CLI)
        raw_metrics["steps"] = self.global_step
        for k, v in metrics.items():
            raw_metrics[k] = v
        if self._verbose:
            cli_pp["steps"] = self.global_step
            for k, v in metrics.items():
                cli_pp[k] = self._fmt(v)

        # -- GPU memory (gb)
        if torch.cuda.is_available():
            raw_metrics["mem_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            raw_metrics["mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

        # -- formatted for CSV/TB
        log_metrics = {k: self._fmt(v) if k not in ("epoch", "wall_sec", "steps") else v
                      for k, v in raw_metrics.items()}

        # -- log to CLI
        print(" | ".join(
            [f"[{self.epoch}]"] +
            [f"{k}={self._fmt(v)}" for k, v in cli_pp.items() if k != "wall_sec"] +
            [f"[{self._pp_time(ep_time)} ({ep_time:.0f})]"]
        ))

        # -- canonical per-epoch record
        if self._jsonl is not None:
            self._jsonl.write({"type": "epoch", **raw_metrics})

        # -- optional sinks
        if self._log_csv:
            assert self._csv_writer is not None, "CSV writer not initialized in log_epoch"
            self._csv_writer.write(log_metrics)

        if self._log_tb:
            assert self._tb_writer is not None, "TensorBoard writer not initialized in log_epoch"
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
        from datetime import datetime

        total_time = time.time() - self._run_start

        # run_summary.json at the run root. Loss curves stay reconstructable from
        # metrics.jsonl rather than depending on TB event files.
        with open(self._run_root / "run_summary.json", "w") as f:
            json.dump({
                "total_epochs": self.epoch,
                "total_steps": self.global_step,
                "wall_time": self._pp_time(total_time),
                "finished_at": datetime.now().isoformat(),
            }, f, indent=2, default=str)

        if self._jsonl is not None:
            self._jsonl.close()
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()

        # Final full S3 sync so teardown is consequence-free (no-op unless enabled).
        self._syncer.close()
        self._closed = True
        print(f"\n[TrainingLogger] Run complete in {self._pp_time(total_time)}")

    # -- Print utility
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