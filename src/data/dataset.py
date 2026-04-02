from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.conllu_parser import Sentence, load_conllu_sentences
from src.data.vocabulary import Vocabulary


class MorphoSyntaxDataset(Dataset):
    """
    PyTorch Dataset for morphosyntactic VAE training.

    Each item contains:
    - token_ids:    Encoded token sequence (with BOS/EOS)
    - length:       Actual sequence length (without padding)
    - upos_ids:     Encoded UPOS tag sequence (for evaluation probing)
    - deprel_ids:   Encoded dependency relation sequence
    - morph_feats:  Morphological feature strings per token
    """

    def __init__(
        self,
        sentences: List[Sentence],
        vocab: Vocabulary,
        upos_vocab: Vocabulary,
        deprel_vocab: Vocabulary,
        max_len: int = 50,
        min_len: int = 2,
    ):
        self.vocab = vocab
        self.upos_vocab = upos_vocab
        self.deprel_vocab = deprel_vocab
        self.max_len = max_len
        self.min_len = min_len
        self.data = self._process(sentences)

    def _process(self, sentences: List[Sentence]) -> List[Dict]:
        processed = []
        for sent in sentences:
            forms = sent.forms
            upos = sent.upos_tags
            deprels = sent.deprels

            # Filter by length
            if not (self.min_len <= len(forms) <= self.max_len):
                continue

            token_ids = self.vocab.encode(forms, add_special=True)
            upos_ids = self.upos_vocab.encode(upos, add_special=False)
            deprel_ids = self.deprel_vocab.encode(deprels, add_special=False)

            processed.append({
                "token_ids": torch.tensor(token_ids, dtype=torch.long),
                "upos_ids": torch.tensor(upos_ids, dtype=torch.long),
                "deprel_ids": torch.tensor(deprel_ids, dtype=torch.long),
                "length": len(forms),
                "raw_forms": forms,
                "morph_feats": sent.morph_feats,
            })
        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def collate_fn(batch: List[Dict], pad_idx: int) -> Dict:
    """
    Custom collate function for variable-length sequences.
    Pads token_ids, upos_ids, and deprel_ids to the max length in the batch.
    """
    token_ids = pad_sequence(
        [item["token_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_idx,
    )
    upos_ids = pad_sequence(
        [item["upos_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    deprel_ids = pad_sequence(
        [item["deprel_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    return {
        "token_ids": token_ids,
        "upos_ids": upos_ids,
        "deprel_ids": deprel_ids,
        "lengths": lengths,
        "raw_forms": [item["raw_forms"] for item in batch],
        "morph_feats": [item["morph_feats"] for item in batch],
    }


def build_dataloaders(
    train_sentences: List[Sentence],
    dev_sentences: List[Sentence],
    test_sentences: List[Sentence],
    vocab: Vocabulary,
    upos_vocab: Vocabulary,
    deprel_vocab: Vocabulary,
    batch_size: int = 64,
    max_len: int = 50,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, dev, and test DataLoaders.

    Returns:
        (train_loader, dev_loader, test_loader)
    """
    from functools import partial

    _collate = partial(collate_fn, pad_idx=vocab.pad_idx)

    train_ds = MorphoSyntaxDataset(train_sentences, vocab, upos_vocab, deprel_vocab, max_len)
    dev_ds = MorphoSyntaxDataset(dev_sentences, vocab, upos_vocab, deprel_vocab, max_len)
    test_ds = MorphoSyntaxDataset(test_sentences, vocab, upos_vocab, deprel_vocab, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, dev_loader, test_loader


def save_processed(dataset: MorphoSyntaxDataset, path: str | Path) -> None:
    """Serialize a processed dataset to disk as a .pt file."""
    torch.save(dataset.data, path)


def load_processed(path: str | Path) -> List[Dict]:
    """Load a serialized dataset from a .pt file."""
    return torch.load(path)