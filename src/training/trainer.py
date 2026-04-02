from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from typing import Dict, Optional

from src.models.vae import MorphoSyntaxVAE
from src.training.elbo import ELBOLoss, KLAnnealer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logger import get_logger, EpochLogger
from src.utils.cuda_utils import autocast_context, clip_gradients, move_to_device


class Trainer:
    """
    Training loop for the MorphoSyntaxVAE.

    Handles:
    - Forward/backward passes with mixed precision
    - KL annealing via KLAnnealer
    - Gradient clipping
    - Epoch-level validation
    - Checkpointing and early stopping
    - TensorBoard logging
    """

    def __init__(
        self,
        model: MorphoSyntaxVAE,
        optimizer: torch.optim.Optimizer,
        scheduler,
        elbo_loss: ELBOLoss,
        kl_annealer: KLAnnealer,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        device: torch.device,
        checkpoint: ModelCheckpoint,
        early_stopping: EarlyStopping,
        mixed_precision: bool = True,
        gradient_clip: float = 5.0,
        log_every: int = 100,
        use_tensorboard: bool = True,
        tensorboard_dir: str = "outputs/logs",
        word_dropout_rate: float = 0.3,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.elbo_loss = elbo_loss
        self.kl_annealer = kl_annealer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.gradient_clip = gradient_clip
        self.log_every = log_every
        self.mixed_precision = mixed_precision
        self.word_dropout_rate = word_dropout_rate
        self.scaler = GradScaler(enabled=mixed_precision)

        self.logger = get_logger("trainer")
        self.epoch_logger = EpochLogger(self.logger)

        self.writer = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        self.global_step: int = 0

    # ------------------------------------------------------------------
    # Word dropout
    # ------------------------------------------------------------------

    def _apply_word_dropout(
        self,
        token_ids: torch.Tensor,
        unk_idx: int = 1,
    ) -> torch.Tensor:
        """
        Randomly replace input tokens with <UNK> during training.
        Forces the decoder to rely on z rather than local token context,
        preventing posterior collapse (Bowman et al., 2016).

        Args:
            token_ids: (batch, seq_len) decoder input tokens.
            unk_idx:   Index of the <UNK> token.

        Returns:
            token_ids with random positions replaced by unk_idx.
        """
        if self.word_dropout_rate == 0.0:
            return token_ids
        mask = torch.bernoulli(
            torch.full(
                token_ids.shape,
                self.word_dropout_rate,
                device=token_ids.device
            )
        ).bool()
        token_ids_dropped = token_ids.clone()
        token_ids_dropped[mask] = unk_idx
        return token_ids_dropped

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader):
            batch = move_to_device(batch, self.device)
            token_ids = batch["token_ids"]
            lengths = batch["lengths"]

            # Target: tokens shifted by 1 (teacher forcing)
            targets = token_ids[:, 1:]

            beta = self.kl_annealer.get_beta(epoch, step, n_batches)

            self.optimizer.zero_grad()

            # Apply word dropout to decoder input to prevent posterior collapse
            decoder_input = self._apply_word_dropout(token_ids[:, :-1])

            with autocast_context(enabled=self.mixed_precision):
                # output = self.model(token_ids, lengths)
                # Encode normally, decode with word-dropped input
                mu, logvar = self.model.encode(token_ids, lengths)
                from src.models.latent_space import reparameterise
                z = reparameterise(mu, logvar)
                z_morpho, z_syntax = self.model.latent_space.split(z)
                logits = self.model.decode(decoder_input, z)

                output = {
                    "logits": logits,
                    "mu": mu,
                    "logvar": logvar,
                    "z": z,
                    "z_morpho": z_morpho,
                    "z_syntax": z_syntax,
                }
                loss_dict = self.elbo_loss(output, targets, beta=beta)

            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_gradients(self.model, self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss_dict["loss"].item()
            total_recon += loss_dict["reconstruction_loss"].item()
            total_kl += loss_dict["kl_loss"].item()

            if self.global_step % self.log_every == 0:
                self._log_step(loss_dict, grad_norm, beta)

            self.global_step += 1

        return {
            "train_loss": total_loss / n_batches,
            "train_recon": total_recon / n_batches,
            "train_kl": total_kl / n_batches,
        }

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = len(self.dev_loader)

        for batch in self.dev_loader:
            batch = move_to_device(batch, self.device)
            token_ids = batch["token_ids"]
            lengths = batch["lengths"]
            targets = token_ids[:, 1:]

            output = self.model(token_ids, lengths)
            loss_dict = self.elbo_loss(output, targets, beta=1.0)

            total_loss += loss_dict["loss"].item()
            total_recon += loss_dict["reconstruction_loss"].item()
            total_kl += loss_dict["kl_loss"].item()

        return {
            "val_loss": total_loss / n_batches,
            "val_recon": total_recon / n_batches,
            "val_kl": total_kl / n_batches,
            "val_elbo": (total_recon + total_kl) / n_batches,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(
        self,
        loss_dict: Dict[str, torch.Tensor],
        grad_norm: float,
        beta: float,
    ) -> None:
        if self.writer is not None:
            for key, val in loss_dict.items():
                self.writer.add_scalar(
                    f"train/{key}", val.item() if hasattr(val, "item") else val,
                    self.global_step
                )
            self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
            self.writer.add_scalar("train/beta", beta, self.global_step)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        merged = {**train_metrics, **val_metrics}
        self.epoch_logger.log_epoch(epoch, merged)
        if self.writer is not None:
            for key, val in merged.items():
                self.writer.add_scalar(f"epoch/{key}", val, epoch)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, n_epochs: int) -> Dict[str, list]:
        self.logger.info(f"Starting training for {n_epochs} epochs.")
        history: Dict[str, list] = {}

        for epoch in range(n_epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            elapsed = time.time() - t0

            self._log_epoch(epoch, train_metrics, val_metrics)
            self.logger.info(f"Epoch {epoch} completed in {elapsed:.1f}s")

            # Scheduler step
            if hasattr(self.scheduler, "step"):
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["val_elbo"])
                else:
                    self.scheduler.step()

            # Checkpoint
            self.checkpoint.save(epoch, self.model, self.optimizer, val_metrics)

            # Early stopping
            if self.early_stopping.step(val_metrics):
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

            # Accumulate history
            for key, val in {**train_metrics, **val_metrics}.items():
                history.setdefault(key, []).append(val)

        if self.writer is not None:
            self.writer.close()

        return history