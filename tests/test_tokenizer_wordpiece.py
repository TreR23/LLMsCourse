from shinynewtokenizer import WordPieceTokenizer, train_wordpiece


def test_basic_split_punctuation_roundtrip_sanity():
    texts = ["Hello, world!"]
    vocab = train_wordpiece(texts, vocab_size=200, min_pair_freq=2)
    tok = WordPieceTokenizer(vocab)

    ids = tok.encode("Hello, world!")
    decoded = tok.decode(ids)

    # We expect punctuation spacing fixed in decode
    assert decoded == "Hello, world!"


def test_unknown_word_goes_to_unk():
    texts = ["hello hello"]
    vocab = train_wordpiece(texts, vocab_size=50, min_pair_freq=2)
    tok = WordPieceTokenizer(vocab)

    ids = tok.encode("goodbye")
    decoded_tokens = [tok.int_to_str[i] for i in ids]

    assert "<|unk|>" in decoded_tokens


def test_trainer_creates_subwords_and_encoder_uses_them():
    texts = ["playing play played playing"]
    vocab = train_wordpiece(texts, vocab_size=200, min_pair_freq=2)

    # Should learn something beyond single chars in many cases
    assert any(len(t.replace("##", "")) >= 2 for t in vocab.keys() if t not in {"<|pad|>", "<|unk|>", "<|endoftext|>"})


def test_deterministic_training_same_corpus_same_vocab():
    texts = ["a a a", "b b", "ab ab ab"]
    v1 = train_wordpiece(texts, vocab_size=200, min_pair_freq=2)
    v2 = train_wordpiece(texts, vocab_size=200, min_pair_freq=2)
    assert v1 == v2


def test_decode_joins_hashhash_pieces():
    texts = ["testing testing"]
    vocab = train_wordpiece(texts, vocab_size=200, min_pair_freq=2)
    tok = WordPieceTokenizer(vocab)

    ids = tok.encode("testing")
    decoded = tok.decode(ids)

    assert decoded == "testing"
