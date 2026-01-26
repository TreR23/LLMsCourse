from pathlib import Path

import pytest

from src.tokenizer import SimpleTokenizer, build_vocab, load_star_wars_dataset


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


def test_tokenizer_runs_on_star_wars_dataset():
    dataset_path = Path("SW_EpisodeIV_VI.json")
    texts = load_star_wars_dataset(dataset_path)

    assert len(texts) > 0, f"No readable dataset text extracted from: {dataset_path.resolve()}"

    # Build vocab from dataset (simple whitespace tokenization for vocab building)
    preprocessed: list[str] = []
    for text in texts:
        preprocessed.extend(text.split())

    vocab = build_vocab(preprocessed)
    tok = SimpleTokenizer(vocab)

    # Encode/decode a real sample from the dataset
    sample = texts[0][:200]
    ids = tok.encode(sample)
    decoded = tok.decode(ids)

    assert isinstance(ids, list)
    assert isinstance(decoded, str)
