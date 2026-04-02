from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]


class Vocabulary:
    """
    Bidirectional mapping between tokens and integer indices.
    Supports building from a corpus and serialisation to/from JSON.
    """

    def __init__(
        self,
        max_size: int = 50000,
        min_freq: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or SPECIAL_TOKENS

        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._counter: Counter = Counter()
        self._built = False

        # Reserve indices for special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[UNK_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS_TOKEN]

    @property
    def sep_idx(self) -> int:
        return self.token2idx[SEP_TOKEN]

    def __len__(self) -> int:
        return len(self.token2idx)

    def __contains__(self, token: str) -> bool:
        return token in self.token2idx

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def update(self, tokens: List[str]) -> None:
        """Add token counts from a list of tokens."""
        self._counter.update(tokens)

    def build(self) -> None:
        """
        Finalize the vocabulary by selecting the most frequent tokens
        up to max_size, filtering those below min_freq.
        """
        next_idx = len(self.special_tokens)
        for token, count in self._counter.most_common():
            if len(self.token2idx) >= self.max_size:
                break
            if count < self.min_freq:
                break
            if token not in self.token2idx:
                self.token2idx[token] = next_idx
                self.idx2token[next_idx] = token
                next_idx += 1
        self._built = True

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, tokens: List[str], add_special: bool = False) -> List[int]:
        """
        Map a list of tokens to indices.

        Args:
            tokens:      Input token list.
            add_special: Wrap with BOS and EOS if True.

        Returns:
            List of integer indices.
        """
        indices = [self.token2idx.get(tok, self.unk_idx) for tok in tokens]
        if add_special:
            indices = [self.bos_idx] + indices + [self.eos_idx]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """
        Map a list of indices back to tokens.

        Args:
            indices:        Input index list.
            remove_special: Strip special tokens from output if True.

        Returns:
            List of token strings.
        """
        tokens = [self.idx2token.get(idx, UNK_TOKEN) for idx in indices]
        if remove_special:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return tokens

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "special_tokens": self.special_tokens,
            "token2idx": self.token2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(
            max_size=data["max_size"],
            min_freq=data["min_freq"],
            special_tokens=data["special_tokens"],
        )
        vocab.token2idx = {k: int(v) for k, v in data["token2idx"].items()}
        vocab.idx2token = {int(v): k for k, v in data["token2idx"].items()}
        vocab._built = True
        return vocab