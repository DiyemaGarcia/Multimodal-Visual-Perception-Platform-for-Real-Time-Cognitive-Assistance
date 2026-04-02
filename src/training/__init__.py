"""
src.training
------------
Training infrastructure for the morphosyntactic VAE.

Modules:
    elbo       — ELBO loss, KL divergence, KL annealing schedule
    trainer    — Main training and validation loop
    scheduler  — Learning rate schedulers (warmup-cosine, cyclical)
    callbacks  — Early stopping and model checkpointing
"""

from src.training.elbo import ELBOLoss, KLAnnealer
from src.training.trainer import Trainer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.scheduler import build_scheduler

__all__ = [
    "ELBOLoss",
    "KLAnnealer",
    "Trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "build_scheduler",
]