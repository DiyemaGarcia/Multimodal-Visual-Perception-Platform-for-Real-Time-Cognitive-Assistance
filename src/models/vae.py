from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.models.encoder import BiLSTMEncoder, TransformerEncoder
from src.models.decoder import LSTMDecoder
from src.models.priors import get_prior
from src.models.latent_space import LatentSpace, reparameterise


class MorphoSyntaxVAE(nn.Module):
    """
    Full Variational Autoencoder for unsupervised morphosyntactic induction.

    Architecture:
        Encoder  →  q(z|x) [mu, logvar]
        Sample   →  z via reparameterisation
        Decoder  →  p(x|z) [reconstruction logits]

    The latent space is structured into morphological and syntactic subspaces.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        z_dim: int,
        morpho_dim: int,
        syntax_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        encoder_type: str = "bilstm",
        prior_type: str = "standard",
        num_heads: int = 8,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.pad_idx = pad_idx

        # Encoder
        if encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                latent_dim=z_dim,
                num_layers=num_layers,
                dropout=dropout,
                pad_idx=pad_idx,
                pretrained_embeddings=pretrained_embeddings,
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                latent_dim=z_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                pad_idx=pad_idx,
                pretrained_embeddings=pretrained_embeddings,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Decoder
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=z_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # Structured latent space
        self.latent_space = LatentSpace(z_dim, morpho_dim, syntax_dim)

        # Prior
        self.prior = get_prior(prior_type, z_dim)

    def encode(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to posterior parameters."""
        mu, logvar, _ = self.encoder(token_ids, lengths)
        return mu, logvar

    def decode(
        self,
        decoder_input: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from latent z, returns logits."""
        logits, _ = self.decoder(decoder_input, z)
        return logits

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full VAE forward pass.

        Args:
            token_ids: (batch, seq_len) input token indices.
            lengths:   (batch,) actual sequence lengths.

        Returns:
            Dictionary containing:
                - logits:    (batch, seq_len-1, vocab_size) reconstruction logits.
                - mu:        (batch, z_dim) posterior mean.
                - logvar:    (batch, z_dim) posterior log-variance.
                - z:         (batch, z_dim) sampled latent vector.
                - z_morpho:  (batch, morpho_dim) morphological subspace.
                - z_syntax:  (batch, syntax_dim) syntactic subspace.
        """
        # Encode
        mu, logvar = self.encode(token_ids, lengths)

        # Sample via reparameterisation
        z = reparameterise(mu, logvar)

        # Split latent space
        z_morpho, z_syntax = self.latent_space.split(z)

        # Decode (teacher forcing: input is tokens[:-1], target is tokens[1:])
        decoder_input = token_ids[:, :-1]
        logits = self.decode(decoder_input, z)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_morpho": z_morpho,
            "z_syntax": z_syntax,
        }

    def generate(
        self,
        z: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate sequences from latent vectors."""
        return self.decoder.generate(z, bos_idx, eos_idx, max_len, temperature)

    def sample_prior(
        self,
        n_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample z directly from the prior p(z) = N(0, I)."""
        return torch.randn(n_samples, self.z_dim, device=device)