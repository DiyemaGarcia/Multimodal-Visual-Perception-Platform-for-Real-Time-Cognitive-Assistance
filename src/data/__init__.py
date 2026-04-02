"""
src.data
--------
Data pipeline for the morphosyntactic VAE.

Modules:
    conllu_parser  — Parse Universal Dependencies CoNLL-U files
    morpho_parser  — Parse Morpho Challenge and UniMorph lexicons
    tokenizer      — Character, word, and BPE tokenisers
    vocabulary     — Token-to-index mapping with serialisation
    dataset        — PyTorch Dataset and DataLoader construction
"""

from src.data.vocabulary import Vocabulary, PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from src.data.conllu_parser import parse_conllu_file, load_conllu_sentences, Sentence, Token
from src.data.dataset import MorphoSyntaxDataset, build_dataloaders

__all__ = [
    "Vocabulary",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "parse_conllu_file",
    "load_conllu_sentences",
    "Sentence",
    "Token",
    "MorphoSyntaxDataset",
    "build_dataloaders",
]