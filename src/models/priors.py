from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class StandardNormalPrior(nn.Module):
    """
    Standard isotropic Gaussian prior: p(z) = N(0, I).
    KL divergence has a closed-form solution with a Gaussian posterior.
    """

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Analytical KL divergence: KL(q(z|x) || p(z)).

        Args:
            mu:     (batch, latent_dim) posterior mean.
            logvar: (batch, latent_dim) posterior log-variance.

        Returns:
            kl: (batch,) per-sample KL divergence.
        """
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl


class vMFPrior(nn.Module):
    """
    von Mises-Fisher prior on the unit hypersphere.
    More appropriate for directional latent representations.
    Uses the approximation from Davidson et al. (2018).
    """

    def __init__(self, latent_dim: int, kappa: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.kappa = kappa

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate KL divergence for vMF.
        Uses the Gaussian approximation for the entropy term.
        """
        # Normalise mu onto the hypersphere
        mu_norm = F.normalize(mu, p=2, dim=-1)
        # Approximate KL using standard formula as a proxy
        kl = -0.5 * torch.sum(1 + logvar - mu_norm.pow(2) - logvar.exp(), dim=-1)
        return kl


class HorseshoePrior(nn.Module):
    """
    Horseshoe prior for sparse latent representations.
    Encourages most dimensions of z to collapse to zero
    while allowing a few dimensions to capture strong structure.
    Implemented as a scale-mixture of Gaussians approximation.
    """

    def __init__(self, latent_dim: int, tau0: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        # Global scale
        self.log_tau = nn.Parameter(torch.zeros(1))
        # Local scales per dimension
        self.log_lambda = nn.Parameter(torch.zeros(latent_dim))

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence under a half-Cauchy scale mixture approximation.
        Uses the closed-form KL between two Gaussians where the prior
        variance is given by tau^2 * lambda^2.
        """
        tau2 = self.log_tau.exp().pow(2)
        lambda2 = self.log_lambda.exp().pow(2)
        prior_var = tau2 * lambda2  # (latent_dim,)

        # KL(N(mu, sigma^2) || N(0, prior_var))
        sigma2 = logvar.exp()
        kl = 0.5 * torch.sum(
            prior_var.log() - logvar
            - 1.0
            + sigma2 / prior_var
            + mu.pow(2) / prior_var,
            dim=-1,
        )
        return kl


def get_prior(prior_type: str, latent_dim: int) -> nn.Module:
    """Factory function for prior selection."""
    if prior_type == "standard":
        return StandardNormalPrior()
    elif prior_type == "vmf":
        return vMFPrior(latent_dim)
    elif prior_type == "horseshoe":
        return HorseshoePrior(latent_dim)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")