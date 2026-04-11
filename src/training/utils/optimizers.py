import math
import torch.optim as optim
from torch.amp import GradScaler # type: ignore


def LinearDecay(
    optimizer, 
    num_epochs: int, 
    ipe: int,
    min_lr_ratio: float = 1e-5,
):
    """ Linear decay from base_lr to (min_lr_ratio * base_lr) over total_steps """
    total_steps = ipe*num_epochs
    return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: max(min_lr_ratio, 
                                       1.0 - (1.0 - min_lr_ratio) * step / total_steps))


class WarmupCosineAnnealing:
    """ Linear warmup from min_lr to base_lr, then cosine decay back to min_lr."""
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int, 
        total_steps: int, 
        min_lr: float = 1e-5, 
        log=False
    ):
        self.optimizer    = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps  = total_steps
        self.min_lr       = float(min_lr)
        self.base_lr      = float(optimizer.param_groups[0]["lr"])
        self._step        = 0
        self._last_lr     = [self.min_lr]
        
        # derived.logging
        self.log = log
        self.non_warmup_steps = max(0, total_steps - warmup_steps)
        self.T_max = max(1, self.non_warmup_steps)
        self._apply(self.min_lr)
        
        if log:
            print(f"[WCA] initialized with: "
                  f"   warmup_steps={self.warmup_steps} total_steps={self.total_steps}"
                  f"   min_lr={self.min_lr} base_lr={self.base_lr}")

    def _compute_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (step / max(1, self.warmup_steps))
        t = min(step - self.warmup_steps, self.non_warmup_steps)
        
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t / self.T_max))

    def _apply(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> None:
        self._step += 1
        lr = self._compute_lr(self._step)
        self._last_lr = [lr]
        self._apply(lr)
        
        if self.log:
            print(f"[WCA] new step={self._step} new lr={lr}")

    def get_last_lr(self) -> list[float]:
        return self._last_lr

    def state_dict(self) -> dict:
        return {"_step": self._step, "_last_lr": self._last_lr}

    def load_state_dict(self, sd: dict) -> None:
        self._step    = sd["_step"]
        self._last_lr = sd["_last_lr"]
        self._apply(self._last_lr[0])

    @property
    def last_epoch(self) -> int:
        return self._step


def init_optimizers(
    model, opt_params,
    ipe: int,
    num_epochs: int,
):
    base_lr = float(opt_params.get("base_lr", 0.0))
    wd = float(opt_params.get("weight_decay", 0.05))
    betas = opt_params.get("betas", [0.9, 0.95])
    b1, b2 = betas[0], betas[1]
    
    # Exclude biases and LayerNorm params from weight decay using param-group split
    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() < 2 or "norm" in name.lower() or "bias" in name:
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=base_lr, betas=(b1, b2))
    
    sched_type = opt_params.get("schedule", "")
    if sched_type == "warmup_cosine":
        scheduler = WarmupCosineAnnealing(
            optimizer,
            warmup_steps=ipe*opt_params.get("warmup_epochs", 0),
            total_steps=ipe*num_epochs,
            min_lr=opt_params.get("min_lr", 0.0)
        )
    elif sched_type == "linear_decay":
        min_lr_ratio = opt_params.get("min_lr_ratio", 0.0)
        
        scheduler = LinearDecay(
            optimizer,
            num_epochs=num_epochs,
            ipe=ipe,
            min_lr_ratio=min_lr_ratio
        )
    else:
        scheduler = None
    
    # No scaler is needed for bf16/fp32
    
    return optimizer, scheduler