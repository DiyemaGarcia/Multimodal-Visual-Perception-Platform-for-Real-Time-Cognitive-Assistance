import pytest
import torch
from src.models.vae import MorphoSyntaxVAE


@pytest.fixture
def small_vae():
    return MorphoSyntaxVAE(
        vocab_size=200,
        embedding_dim=32,
        hidden_dim=64,
        z_dim=16,
        morpho_dim=8,
        syntax_dim=8,
        num_layers=1,
        dropout=0.0,
        pad_idx=0,
        encoder_type="bilstm",
        prior_type="standard",
    )


def test_vae_output_keys(small_vae):
    token_ids = torch.randint(1, 200, (4, 10))
    lengths = torch.tensor([10, 8, 6, 4])
    output = small_vae(token_ids, lengths)
    required_keys = {"logits", "mu", "logvar", "z", "z_morpho", "z_syntax"}
    assert required_keys.issubset(set(output.keys()))


def test_vae_output_shapes(small_vae):
    batch, seq_len = 4, 10
    token_ids = torch.randint(1, 200, (batch, seq_len))
    lengths = torch.tensor([seq_len] * batch)
    output = small_vae(token_ids, lengths)
    assert output["logits"].shape == (batch, seq_len - 1, 200)
    assert output["mu"].shape == (batch, 16)
    assert output["z_morpho"].shape == (batch, 8)
    assert output["z_syntax"].shape == (batch, 8)


def test_vae_generate(small_vae):
    z = torch.randn(3, 16)
    generated = small_vae.generate(z, bos_idx=2, eos_idx=3, max_len=15)
    assert generated.shape[0] == 3
    assert generated.shape[1] <= 15


def test_vae_sample_prior(small_vae):
    device = torch.device("cpu")
    z = small_vae.sample_prior(n_samples=10, device=device)
    assert z.shape == (10, 16)