from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ELBOLoss(nn.Module):
    """
    Evidence Lower BOund (ELBO) loss for the morphosyntactic VAE.

    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
         + latent_constraint_penalties

    Supports β-VAE weighting and KL annealing via an external beta schedule.
    """

    def __init__(
        self,
        pad_idx: int = 0,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1.0,
        morpho_constraint_weight: float = 0.5,
        syntax_constraint_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.morpho_constraint_weight = morpho_constraint_weight
        self.syntax_constraint_weight = syntax_constraint_weight
        self.label_smoothing = label_smoothing

    def reconstruction_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token-level cross-entropy reconstruction loss.

        Args:
            logits:  (batch, seq_len, vocab_size)
            targets: (batch, seq_len) — token_ids shifted by 1.

        Returns:
            Scalar mean reconstruction loss.
        """
        batch, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.pad_idx,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
        return loss

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed-form KL divergence: KL(N(mu, sigma^2) || N(0, I)).
        Returns the mean over the batch.
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full ELBO loss.

        Args:
            model_output: Output dict from MorphoSyntaxVAE.forward().
            targets:      (batch, seq_len-1) gold token ids (token_ids[:, 1:]).
            beta:         Current KL annealing coefficient.

        Returns:
            Dictionary with individual loss components and total loss.
        """
        logits = model_output["logits"]
        mu = model_output["mu"]
        logvar = model_output["logvar"]
        z = model_output["z"]

        # Reconstruction
        recon_loss = self.reconstruction_loss(logits, targets)

        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)

        # Latent space constraints (independence between morpho and syntax)
        from src.models.latent_space import LatentSpace
        z_morpho, z_syntax = z[:, :z.size(1)//2], z[:, z.size(1)//2:]

        morpho_penalty = z_morpho.pow(2).mean()
        syntax_penalty = z_syntax.pow(2).mean()

        total_loss = (
            self.reconstruction_weight * recon_loss
            + self.kl_weight * beta * kl_loss
            + self.morpho_constraint_weight * morpho_penalty
            + self.syntax_constraint_weight * syntax_penalty
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "morpho_penalty": morpho_penalty,
            "syntax_penalty": syntax_penalty,
            "beta": torch.tensor(beta),
        }


class KLAnnealer:
    """
    Computes the KL annealing schedule (beta coefficient).

    Supports:
    - linear:    Linearly increases beta from start to stop.
    - cyclical:  Cyclically anneals beta (Fu et al., 2019).
    - monotonic: Same as linear but never resets.
    """

    def __init__(
        self,
        strategy: str = "cyclical",
        start: float = 0.0,
        stop: float = 1.0,
        n_cycles: int = 4,
        beta_max: float = 4.0,
        n_epochs: int = 100,
    ):
        self.strategy = strategy
        self.start = start
        self.stop = stop
        self.n_cycles = n_cycles
        self.beta_max = beta_max
        self.n_epochs = n_epochs

    def get_beta(self, epoch: int, step: int, total_steps: int) -> float:
        """
        Compute the current beta value.

        Args:
            epoch:       Current epoch (0-indexed).
            step:        Current step within the epoch.
            total_steps: Total steps per epoch.

        Returns:
            beta: Float in [start, beta_max].
        """
        if self.strategy == "linear" or self.strategy == "monotonic":
            ratio = min(epoch / max(self.n_epochs, 1), 1.0)
            beta = self.start + ratio * (self.stop - self.start)
            return beta * self.beta_max

        elif self.strategy == "cyclical":
            # Global step position
            period = self.n_epochs / self.n_cycles
            cycle_pos = epoch % period
            ratio = min(cycle_pos / (period * 0.5), 1.0)
            beta = self.start + ratio * (self.stop - self.start)
            return beta * self.beta_max

        else:
            raise ValueError(f"Unknown KL annealing strategy: {self.strategy}")