import pytest
import torch
from src.training.elbo import ELBOLoss, KLAnnealer


@pytest.fixture
def elbo():
    return ELBOLoss(pad_idx=0)


def test_reconstruction_loss_shape(elbo):
    logits = torch.randn(4, 9, 200)
    targets = torch.randint(0, 200, (4, 9))
    loss = elbo.reconstruction_loss(logits, targets)
    assert loss.ndim == 0  # scalar


def test_kl_divergence_positive(elbo):
    mu = torch.randn(4, 16)
    logvar = torch.randn(4, 16)
    kl = elbo.kl_divergence(mu, logvar)
    assert kl.item() >= 0.0


def test_elbo_forward_keys(elbo):
    model_output = {
        "logits": torch.randn(4, 9, 200),
        "mu": torch.randn(4, 16),
        "logvar": torch.randn(4, 16),
        "z": torch.randn(4, 16),
        "z_morpho": torch.randn(4, 8),
        "z_syntax": torch.randn(4, 8),
    }
    targets = torch.randint(0, 200, (4, 9))
    result = elbo(model_output, targets, beta=1.0)
    assert "loss" in result
    assert "reconstruction_loss" in result
    assert "kl_loss" in result


def test_kl_annealer_cyclical():
    annealer = KLAnnealer(
        strategy="cyclical", start=0.0, stop=1.0,
        n_cycles=4, beta_max=4.0, n_epochs=100
    )
    beta_0 = annealer.get_beta(0, 0, 100)
    beta_50 = annealer.get_beta(50, 0, 100)
    assert 0.0 <= beta_0 <= 4.0
    assert 0.0 <= beta_50 <= 4.0


def test_kl_annealer_linear():
    annealer = KLAnnealer(
        strategy="linear", start=0.0, stop=1.0,
        n_cycles=1, beta_max=1.0, n_epochs=100
    )
    beta_0 = annealer.get_beta(0, 0, 100)
    beta_100 = annealer.get_beta(100, 0, 100)
    assert beta_0 < beta_100