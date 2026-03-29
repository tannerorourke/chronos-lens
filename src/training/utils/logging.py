import json
import time
import csv
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """ All-in-one CSV, TensorBoard, and cli logger."""
    def __init__(
        self, 
        logdir: Path, 
        epoch: int = 0,
        global_step: int = 0, 
        loss_history: list[float] = None, 
        flush_every: int = 1
    ):
        self._closed = False
        self.flush_every = flush_every
        self.run_start = time.time()
        self.epoch = epoch
        self.global_step = global_step
        self._epoch_losses: list[float] = []
        self._loss_history: list[float] = loss_history if loss_history is not None else []

        self.logdir = logdir
        self.log_dir = logdir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # initialized on first log_epoch
        self._csv_writer = None
        self._csv_path = self.log_dir / "epoch_metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(logdir / "tb_logs"))

        print(f"[TrainingLogger] Logging it up in {logdir.parent.name}/{logdir.name}")
        
    @property
    def epoch_train_loss(self) -> float:
        if not self._epoch_losses:
            return float("nan")
        return sum(self._epoch_losses) / len(self._epoch_losses)

    def log_step(self, loss: float):
        self._epoch_losses.append(loss)
        self.global_step += 1
        self.writer.add_scalar("step/loss", loss, self.global_step)

    def log_epoch(self, loss: float, lr: float | None = None, **metrics):
        self.epoch += 1
        wall_time = time.time() - self.run_start

        if "train_loss" not in metrics and self._epoch_losses:
            metrics["train_loss"] = self.epoch_train_loss

        row = {
            "epoch": self.epoch,
            "wall_sec": round(wall_time, 1),
            "steps": self.global_step,
            "train_loss": self._fmt(loss),
            **{k: self._fmt(v) for k, v in metrics.items()},
        }

        # Initialize CSV header on first epoch
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)

        self._loss_history.append(float(loss))

        if self.epoch % self.flush_every == 0:
            self._csv_file.flush()
        self._epoch_losses.clear()

        # TensorBoard metrics
        for key, value in row.items():
            if key in ("epoch", "wall_sec", "steps") or value in (None, ""):
                continue
            v = int(value) if isinstance(value, bool) else float(value)
            self.writer.add_scalar(f"epoch/{key}", v, self.epoch)
        if lr is not None:
            self.writer.add_scalar("epoch/lr", lr, self.epoch)
        self.writer.flush()

        parts = [f"Epoch {self.epoch:>4d}"]
        for k, v in metrics.items():
            parts.append(f"{k}={self._fmt(v)}")
        parts.append(f"[{wall_time:.0f}s]")
        print(" | ".join(parts))

    def finalize(self):
        total_time = time.time() - self.run_start

        self._write_json("run_summary.json", {
            "total_epochs": self.epoch,
            "total_steps": self.global_step,
            "wall_time_sec": round(total_time, 1),
            "wall_time_human": self._pp_time(total_time),
            "finished_at": datetime.now().isoformat(),
        })

        self.writer.flush()
        self.writer.close()
        self._csv_file.close()
        self._closed = True
        print(f"\n[TrainingLogger] Run complete in {self._pp_time(total_time)}")

    # -- Utility

    def _write_json(self, filename: str, data):
        with open(self.log_dir / filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _fmt(v) -> str:
        if v is None:
            return ""
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
            self._csv_file.close()
            self.writer.close()
    

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


class GradientMonitor:
    """Lightweight grad monitor.."""

    def __init__(self, model):
        self.model = model
        self._grad_norms: list[float] = []

    def capture(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self._grad_norms.append(total_norm ** 0.5)

    def get_metrics(self) -> dict:
        if not self._grad_norms:
            return {}
        import statistics
        norms = self._grad_norms
        return {
            "grad_norm_mean": statistics.mean(norms),
            "grad_norm_max": max(norms),
            "grad_norm_min": min(norms),
            "grad_norm_std": statistics.stdev(norms) if len(norms) > 1 else 0.0,
        }

    def reset(self):
        self._grad_norms.clear()