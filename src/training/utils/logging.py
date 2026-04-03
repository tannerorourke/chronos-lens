import time
import csv
from pathlib import Path
import statistics as stats

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
        
        import statistics
        return {
            "grad_norm_mean": statistics.mean(self._norms),
            "grad_norm_max": max(self._norms),
            "grad_norm_min": min(self._norms),
            "grad_norm_std": statistics.stdev(self._norms) if len(self._norms) > 1 else 0.0,
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
        
        

class TrainingLogger:
    """ All-in-one CSV, TensorBoard, and CLI logger for
        loss, grad norms, and custom metrics
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
        
        self._ep_samples: int = 0
        self.epoch_losses: list[float] = []
        self.loss_history: list[float] = loss_history if loss_history is not None else []
        self.epoch = epoch
        self.global_step = global_step
        
        self._log_norms = log_norms
        self.grad_mon = GradientMonitor()
        
        
        self._logdir = logdir / "logs"
        self._logdir.mkdir(parents=True, exist_ok=True)

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
        self.epoch += 1
        
        # -- collect --
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
            
        plot_loss_curve(self.loss_history, self._logdir, show=False, save=True)

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


