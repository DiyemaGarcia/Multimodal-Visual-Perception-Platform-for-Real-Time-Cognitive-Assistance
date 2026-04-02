from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMDecoder(nn.Module):
    """
    Autoregressive LSTM decoder.
    Reconstructs the input sequence from the latent vector z.
    At each step, z is concatenated to the token embedding (input feeding).
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # Input: embedding + z (concatenated)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Project z to decoder initial hidden state
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

    def _init_hidden(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise the LSTM hidden and cell states from z.

        Args:
            z: (batch, latent_dim)

        Returns:
            h0: (num_layers, batch, hidden_dim)
            c0: (num_layers, batch, hidden_dim)
        """
        batch = z.size(0)
        h0 = torch.tanh(self.z_to_hidden(z))
        c0 = torch.tanh(self.z_to_cell(z))
        h0 = h0.view(batch, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = c0.view(batch, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h0, c0

    def forward(
        self,
        token_ids: torch.Tensor,
        z: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Teacher-forced forward pass.

        Args:
            token_ids: (batch, seq_len) — shifted input tokens (BOS + tokens).
            z:         (batch, latent_dim) — latent vector.
            hidden:    Optional initial LSTM state.

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: Updated LSTM state.
        """
        if hidden is None:
            hidden = self._init_hidden(z)

        embedded = self.dropout(self.embedding(token_ids))

        # Expand z along the sequence dimension and concatenate
        z_expanded = z.unsqueeze(1).expand(-1, token_ids.size(1), -1)
        lstm_input = torch.cat([embedded, z_expanded], dim=-1)

        output, hidden = self.lstm(lstm_input, hidden)
        output = self.dropout(output)
        logits = self.output_projection(output)

        return logits, hidden

    def generate(
        self,
        z: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation from a latent vector z.

        Args:
            z:           (batch, latent_dim)
            bos_idx:     Begin-of-sequence token index.
            eos_idx:     End-of-sequence token index.
            max_len:     Maximum generation length.
            temperature: Sampling temperature (1.0 = no scaling).

        Returns:
            generated: (batch, max_len) token indices.
        """
        batch = z.size(0)
        device = z.device
        hidden = self._init_hidden(z)

        token = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
        generated = []
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, hidden = self.forward(token, z, hidden)
            logits = logits[:, -1, :] / temperature
            token = logits.argmax(dim=-1, keepdim=True)
            generated.append(token)
            finished = finished | (token.squeeze(-1) == eos_idx)
            if finished.all():
                break

        return torch.cat(generated, dim=1)