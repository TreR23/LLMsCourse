import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


import pytest

from src.tokenizer import SimpleTokenizer, build_vocab


@pytest.fixture()
def vocab():
    # Small, deterministic toy "dataset" of tokens
    preprocessed = ["Hello", ",", "world", "!", "I", "am", "Tre"]
    return build_vocab(preprocessed)


@pytest.fixture()
def tokenizer(vocab):
    return SimpleTokenizer(vocab)


def test_build_vocab_contains_special_tokens(vocab):
    assert "<|unk|>" in vocab
    assert "<|endoftext|>" in vocab


def test_encode_known_tokens(tokenizer, vocab):
    ids = tokenizer.encode("Hello, world!")
    expected = [vocab["Hello"], vocab[","], vocab["world"], vocab["!"]]
    assert ids == expected


def test_encode_unknown_token_becomes_unk(tokenizer, vocab):
    ids = tokenizer.encode("Hello, Mars!")
    expected = [vocab["Hello"], vocab[","], vocab["<|unk|>"], vocab["!"]]
    assert ids == expected


def test_decode_round_trip_known_tokens(tokenizer):
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_decode_spacing_rules(tokenizer, vocab):
    ids = [vocab["Hello"], vocab[","], vocab["world"], vocab["!"]]
    assert tokenizer.decode(ids) == "Hello, world!"
