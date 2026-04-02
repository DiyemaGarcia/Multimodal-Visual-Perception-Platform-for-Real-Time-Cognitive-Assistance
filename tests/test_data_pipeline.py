import pytest
import torch
from src.data.vocabulary import Vocabulary
from src.data.conllu_parser import Token, Sentence


def make_dummy_sentence():
    tokens = [
        Token("1", "The", "the", "DET", "DT", {}, 2, "det", "_", "_"),
        Token("2", "cat", "cat", "NOUN", "NN", {"Number": "Sing"}, 0, "root", "_", "_"),
        Token("3", "runs", "run", "VERB", "VBZ", {"Tense": "Pres"}, 2, "nsubj", "_", "_"),
    ]
    return Sentence(sent_id="test_1", text="The cat runs", tokens=tokens)


def test_vocabulary_build():
    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.update(["the", "cat", "runs", "the", "cat"])
    vocab.build()
    assert "the" in vocab
    assert "cat" in vocab
    assert len(vocab) > 5  # special tokens + content


def test_vocabulary_encode_decode():
    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.update(["hello", "world"])
    vocab.build()
    encoded = vocab.encode(["hello", "world"], add_special=True)
    decoded = vocab.decode(encoded, remove_special=True)
    assert decoded == ["hello", "world"]


def test_vocabulary_unk():
    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.update(["known"])
    vocab.build()
    encoded = vocab.encode(["unknown_word"])
    assert encoded[0] == vocab.unk_idx


def test_sentence_properties():
    sent = make_dummy_sentence()
    assert sent.forms == ["The", "cat", "runs"]
    assert sent.upos_tags == ["DET", "NOUN", "VERB"]
    assert sent.deprels == ["det", "root", "nsubj"]


def test_vocabulary_save_load(tmp_path):
    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.update(["hello", "world", "foo"])
    vocab.build()
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    assert len(vocab) == len(loaded)
    assert vocab.token2idx == loaded.token2idx