"""Opt-in extension of a restored cosine schedule without an LR jump."""
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

def extend_resumed_cosine(scheduler, global_step, target_step):
    scheduler = getattr(scheduler, "scheduler", scheduler)
    warmup = 0
    cosine = scheduler
    if isinstance(scheduler, SequentialLR):
        if len(scheduler._schedulers) != 2 or len(scheduler._milestones) != 1:
            raise ValueError("Expected warmup followed by cosine")
        warmup = int(scheduler._milestones[0])
        cosine = scheduler._schedulers[-1]
    if not isinstance(cosine, CosineAnnealingLR):
        raise ValueError("Resume extension requires a cosine scheduler")
    if not warmup <= global_step < target_step:
        raise ValueError("Resume must be after warmup and before target")
    if cosine.last_epoch != global_step - warmup:
        raise ValueError("Restored scheduler/global step mismatch")
    if target_step - warmup < cosine.T_max:
        raise ValueError("Cannot shorten schedule with resume extension")
    current = [g["lr"] for g in cosine.optimizer.param_groups]
    cosine.T_max = int(target_step - warmup)
    factor = (1 + math.cos(math.pi * cosine.last_epoch / cosine.T_max)) / 2
    cosine.base_lrs = [cosine.eta_min + (lr - cosine.eta_min) / factor for lr in current]
    cosine._last_lr = list(current)
    return current
