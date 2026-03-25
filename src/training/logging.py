import json
import time
import csv
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(
        self, 
        run_dir: Path, 
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

        self.run_dir = run_dir
        self.log_dir = run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.log_dir / "epoch_metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = None  # initialized on first log_epoch (dynamic columns)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))

        # Freeze config YAML and log to TensorBoard
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            config_text = config_path.read_text(encoding="utf-8")
            self.writer.add_text("config", f"```yaml\n{config_text}```", 0)
            config_path = Path(config_path).resolve()
            config_path.chmod(0o444)
            print(f"[TrainingLogger] Froze {config_path.name} (read-only)")

        print(f"[TrainingLogger] Logging it up in {run_dir.parent.name}/{run_dir.name}")

    def log_step(self, loss: float):
        self._epoch_losses.append(loss)
        self.global_step += 1
        self.writer.add_scalar("step/loss", loss, self.global_step)

    @property
    def epoch_train_loss(self) -> float:
        if not self._epoch_losses:
            return float("nan")
        return sum(self._epoch_losses) / len(self._epoch_losses)

    def log_epoch(self, loss: float, lr: float | None = None, **metrics):
        """
        Call once per epoch with whatever metrics you want.

        Common keys: train_loss, val_loss, val_auc, val_f1, lr, ema_decay, etc.
        Columns are auto-detected from the first call; new keys in later calls are silently ignored.
        """
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

        # TensorBoard — all metrics
        for key, value in row.items():
            if key in ("epoch", "wall_sec", "steps"):
                continue
            if value == "":
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
        """ Writes summary.json and closes files."""
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

    # ----- Utility -----

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
    

# ---------------------------------------------------------------------------
# 2. EMA Tracker — bolt onto EMA update loop
# ---------------------------------------------------------------------------

class EMATracker:
    """
    Tracks EMA encoder divergence from context encoder.

    Usage:
        ema_tracker = EMATracker()

        # Inside your EMA update:
        ema_tracker.update(context_encoder, ema_encoder)

        # At epoch end:
        logger.log_epoch(..., **ema_tracker.get_metrics())
        ema_tracker.reset()
    """

    def __init__(self):
        self._param_diffs: list[float] = []

    def update(self, online_model, ema_model):
        """Call after each EMA step. Cheaply computes mean absolute param divergence."""
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


# ---------------------------------------------------------------------------
# 3. Gradient Health Monitor — bolt onto backward pass
# ---------------------------------------------------------------------------

class GradientMonitor:
    """
    Lightweight gradient statistics per epoch.

    Usage:
        grad_mon = GradientMonitor(model)

        for batch in loader:
            loss.backward()
            grad_mon.capture()  # call after backward, before optimizer.step()
            optimizer.step()

        logger.log_epoch(..., **grad_mon.get_metrics())
        grad_mon.reset()
    """

    def __init__(self, model):
        self.model = model
        self._grad_norms: list[float] = []

    def capture(self):
        """Capture total gradient norm for this step."""
        import torch
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


# ---------------------------------------------------------------------------
# 4. Embedding Health — bolt onto embedding extraction
# ---------------------------------------------------------------------------

def embedding_health(z_context, z_pred, z_target=None) -> dict:
    """
    Quick diagnostics on embedding tensors. Call at end of epoch or periodically.

    Args:
        z_context: (N, D) tensor — context encoder outputs
        z_pred:    (N, D) tensor — predictor outputs
        z_target:  (N, D) tensor — EMA target outputs (optional)

    Returns:
        Dict of metrics safe to pass into logger.log_epoch(**embedding_health(...))
    """
    import torch
    metrics = {}

    # Displacement field basics
    delta = z_pred - z_context
    metrics["delta_norm_mean"] = delta.norm(dim=1).mean().item()
    metrics["delta_norm_std"] = delta.norm(dim=1).std().item()

    # Cosine similarity distributions
    cos = torch.nn.functional.cosine_similarity
    metrics["cos_ctx_pred_mean"] = cos(z_context, z_pred, dim=1).mean().item()

    if z_target is not None:
        metrics["cos_pred_target_mean"] = cos(z_pred, z_target, dim=1).mean().item()
        metrics["cos_ctx_target_mean"] = cos(z_context, z_target, dim=1).mean().item()

    # Collapse detection: effective rank via singular values
    # Only compute on a sample if N is large
    sample = z_pred[:2048] if z_pred.shape[0] > 2048 else z_pred
    try:
        s = torch.linalg.svdvals(sample.float())
        p = s / s.sum()
        metrics["z_pred_effective_rank"] = torch.exp(-(p * p.log()).sum()).item()
    except Exception:
        pass  # skip if SVD fails on small batches

    # Check for representation collapse
    mean_pairwise_cos = cos(
        z_pred[:256].unsqueeze(1),
        z_pred[:256].unsqueeze(0),
        dim=2
    ).triu(diagonal=1)
    nonzero = mean_pairwise_cos[mean_pairwise_cos != 0]
    if nonzero.numel() > 0:
        metrics["z_pred_pairwise_cos_mean"] = nonzero.mean().item()

    return metrics