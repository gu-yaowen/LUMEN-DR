from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, StepLR


def build_scheduler(optimizer: Optimizer, args: Any):
    scheduler_name = getattr(args, "scheduler", "none")
    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        t_max = args.scheduler_t_max if args.scheduler_t_max > 0 else args.epochs
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.scheduler_min_lr)

    if scheduler_name == "step":
        step_size = args.scheduler_step_size if args.scheduler_step_size > 0 else max(1, args.epochs // 4)
        return StepLR(optimizer, step_size=step_size, gamma=args.scheduler_gamma)

    if scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_gamma,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )

    if scheduler_name == "linear":
        base_lr = optimizer.param_groups[0]["lr"]
        min_lr = args.scheduler_min_lr
        warmup_epochs = max(0, int(args.warmup_epochs))
        total_epochs = max(1, int(args.epochs))
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(epoch_index: int) -> float:
            current_epoch = epoch_index + 1
            if warmup_epochs > 0 and current_epoch <= warmup_epochs:
                return max(min_factor, current_epoch / warmup_epochs)
            decay_steps = max(1, total_epochs - warmup_epochs)
            progress = min(max(current_epoch - warmup_epochs, 0), decay_steps) / decay_steps
            return max(min_factor, 1.0 - (1.0 - min_factor) * progress)

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def scheduler_step(scheduler, args: Any, val_aupr: float) -> None:
    if scheduler is None:
        return
    if getattr(args, "scheduler", "none") == "plateau":
        scheduler.step(val_aupr)
        return
    scheduler.step()

