from __future__ import annotations
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.

    During warmup: lr increases linearly from 0 to base_lr.
    After warmup:  lr follows cosine decay from base_lr to min_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * scale for base_lr in self.base_lrs]

        # Cosine annealing
        progress = (epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]


class CyclicalLRScheduler(_LRScheduler):
    """
    Cyclical learning rate scheduler (Smith, 2017).
    Oscillates between base_lr and max_lr over a fixed cycle length.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int = 2000,
        mode: str = "triangular2",
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        scale = max(0, 1 - x)

        if self.mode == "triangular2":
            scale /= 2 ** (cycle - 1)
        elif self.mode == "exp_range":
            scale *= 0.99994 ** self.last_epoch

        lr = self.base_lr + (self.max_lr - self.base_lr) * scale
        return [lr for _ in self.base_lrs]


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
) -> _LRScheduler:
    """Factory for learning rate schedulers."""
    if scheduler_type == "cosine":
        return WarmupCosineScheduler(optimizer, warmup_epochs, total_epochs, min_lr)
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")