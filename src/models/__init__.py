"""
src.models
----------
VAE model components for morphosyntactic structure induction.

Modules:
    encoder      — BiLSTM and Transformer encoders
    decoder      — Autoregressive LSTM decoder
    priors       — Standard Gaussian, vMF, and Horseshoe priors
    latent_space — Structured latent space with morpho/syntax subspaces
    vae          — Full MorphoSyntaxVAE model
"""

from src.models.vae import MorphoSyntaxVAE
from src.models.encoder import BiLSTMEncoder, TransformerEncoder
from src.models.decoder import LSTMDecoder
from src.models.priors import StandardNormalPrior, vMFPrior, HorseshoePrior, get_prior
from src.models.latent_space import LatentSpace, reparameterise

__all__ = [
    "MorphoSyntaxVAE",
    "BiLSTMEncoder",
    "TransformerEncoder",
    "LSTMDecoder",
    "StandardNormalPrior",
    "vMFPrior",
    "HorseshoePrior",
    "get_prior",
    "LatentSpace",
    "reparameterise",
]