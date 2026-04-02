from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LatentSpace(nn.Module):
    """
    Structured latent space with dedicated subspaces for
    morphological and syntactic information.

    The full latent vector z is partitioned as:
        z = [z_morpho | z_syntax]
    where z_morpho captures morphological regularities and
    z_syntax captures syntactic structure.
    """

    def __init__(
        self,
        z_dim: int,
        morpho_dim: int,
        syntax_dim: int,
    ):
        super().__init__()
        assert morpho_dim + syntax_dim == z_dim, (
            f"morpho_dim ({morpho_dim}) + syntax_dim ({syntax_dim}) "
            f"must equal z_dim ({z_dim})"
        )
        self.z_dim = z_dim
        self.morpho_dim = morpho_dim
        self.syntax_dim = syntax_dim

    def split(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split z into morphological and syntactic subspaces.

        Args:
            z: (batch, z_dim)

        Returns:
            z_morpho: (batch, morpho_dim)
            z_syntax: (batch, syntax_dim)
        """
        z_morpho = z[:, : self.morpho_dim]
        z_syntax = z[:, self.morpho_dim :]
        return z_morpho, z_syntax

    def independence_penalty(self, z: torch.Tensor) -> torch.Tensor:
        """
        Penalise linear dependence between z_morpho and z_syntax
        using the total correlation approximation (covariance off-diagonal).

        Args:
            z: (batch, z_dim)

        Returns:
            Scalar penalty term.
        """
        z_morpho, z_syntax = self.split(z)
        # Cross-covariance matrix between the two subspaces
        z_m_centered = z_morpho - z_morpho.mean(dim=0, keepdim=True)
        z_s_centered = z_syntax - z_syntax.mean(dim=0, keepdim=True)
        cross_cov = (z_m_centered.T @ z_s_centered) / (z.size(0) - 1)
        return cross_cov.pow(2).sum()

    def orthogonality_constraint(self, z: torch.Tensor) -> torch.Tensor:
        """
        Enforce approximate orthogonality between z_morpho and z_syntax
        via cosine similarity penalty.
        """
        z_morpho, z_syntax = self.split(z)
        # Pad the smaller subspace to the same dim for cosine similarity
        if self.morpho_dim != self.syntax_dim:
            target_dim = max(self.morpho_dim, self.syntax_dim)
            if z_morpho.size(1) < target_dim:
                z_morpho = F.pad(z_morpho, (0, target_dim - z_morpho.size(1)))
            else:
                z_syntax = F.pad(z_syntax, (0, target_dim - z_syntax.size(1)))
        cos_sim = F.cosine_similarity(z_morpho, z_syntax, dim=-1)
        return cos_sim.pow(2).mean()


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterisation trick: z = mu + eps * std.
    Allows gradients to flow through the sampling operation.

    Args:
        mu:     (batch, latent_dim) posterior mean.
        logvar: (batch, latent_dim) posterior log-variance.

    Returns:
        z: (batch, latent_dim) sampled latent vector.
    """
    if not torch.is_grad_enabled():
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std