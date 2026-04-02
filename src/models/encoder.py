from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Tuple


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder.
    Maps a sequence of token embeddings to a fixed-size context vector
    used to parameterise the VAE posterior q(z|x).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Project concatenated bidirectional hidden state to latent parameters
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (batch, seq_len) token indices.
            lengths:   (batch,) actual sequence lengths.

        Returns:
            mu:      (batch, latent_dim) posterior mean.
            logvar:  (batch, latent_dim) posterior log-variance.
            hidden:  Last hidden state for decoder initialisation.
        """
        embedded = self.dropout(self.embedding(token_ids))

        # Pack for efficiency
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, _) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Concatenate final forward and backward hidden states
        # hidden: (num_layers * 2, batch, hidden_dim)
        forward_hidden = hidden[-2]   # Last forward layer
        backward_hidden = hidden[-1]  # Last backward layer
        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        combined = self.dropout(combined)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)

        return mu, logvar, combined


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder using multi-head self-attention.
    Uses the [CLS] token representation to parameterise q(z|x).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        latent_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        if pretrained_embeddings is not None:
            self.token_embedding.weight.data.copy_(pretrained_embeddings)

        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim, latent_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (batch, seq_len)
            lengths:   (batch,) — used to build the key padding mask.

        Returns:
            mu, logvar, cls_repr
        """
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (batch, seq_len+1, embed_dim)

        # Build padding mask (True = position to ignore)
        max_len = x.size(1)
        mask = torch.ones(batch, max_len, dtype=torch.bool, device=token_ids.device)
        for i, length in enumerate(lengths):
            mask[i, : length.item() + 1] = False  # +1 for CLS

        x = self.transformer(x, src_key_padding_mask=mask)
        cls_repr = x[:, 0, :]  # [CLS] position

        mu = self.fc_mu(cls_repr)
        logvar = self.fc_logvar(cls_repr)

        return mu, logvar, cls_repr