"""
vae_morphosyntax
================
Unsupervised Morphological and Syntactic Structure Induction via VAE.

Package structure:
    src.data        — Data loading, parsing, tokenisation, vocabulary
    src.models      — VAE encoder, decoder, priors, latent space
    src.training    — ELBO loss, trainer, scheduler, callbacks
    src.analysis    — Probing, clustering, correlation, visualisation
    src.evaluation  — Morphological, syntactic, reconstruction metrics
    src.utils       — Config, logging, seeding, CUDA utilities
"""

__version__ = "1.0.0"
__author__  = "vae_morphosyntax"