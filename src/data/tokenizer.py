from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from src.data.vocabulary import Vocabulary


class CharTokenizer:
    """
    Character-level tokenizer.
    Splits every token into its individual characters.
    """

    def tokenize(self, word: str) -> List[str]:
        return list(word)

    def tokenize_sentence(self, words: List[str], sep: str = " ") -> List[str]:
        chars = []
        for i, word in enumerate(words):
            chars.extend(self.tokenize(word))
            if i < len(words) - 1:
                chars.append(sep)
        return chars


class WordTokenizer:
    """
    Word-level tokenizer.
    Optionally lowercases the input.
    """

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def tokenize(self, sentence: List[str]) -> List[str]:
        if self.lowercase:
            return [w.lower() for w in sentence]
        return list(sentence)


class BPETokenizer:
    """
    Subword tokenizer using Byte Pair Encoding (BPE) via HuggingFace tokenizers.
    Trains on a corpus and saves/loads the model.
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self._tokenizer: Optional[HFTokenizer] = None

    def train(self, corpus: List[str], save_path: Optional[str | Path] = None) -> None:
        """
        Train BPE on a list of sentences (strings).

        Args:
            corpus:    List of whitespace-separated sentences.
            save_path: If provided, saves the trained tokenizer.
        """
        tokenizer = HFTokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>")),
            ],
        )
        self._tokenizer = tokenizer
        if save_path is not None:
            self._tokenizer.save(str(save_path))

    def load(self, path: str | Path) -> None:
        """Load a previously saved BPE tokenizer."""
        self._tokenizer = HFTokenizer.from_file(str(path))

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        assert self._tokenizer is not None, "Tokenizer not trained or loaded."
        return self._tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs into a string."""
        assert self._tokenizer is not None, "Tokenizer not trained or loaded."
        return self._tokenizer.decode(ids)

    def get_vocab_size(self) -> int:
        assert self._tokenizer is not None
        return self._tokenizer.get_vocab_size()