from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|endoftext|>"]


def normalize_text(text: str) -> str:
    # Keep this minimal for now; you can expand later (unicode normalization, etc.)
    return text.strip()


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# This splits text into "words" and standalone punctuation tokens.
# Example: "Hello, world!" -> ["Hello", ",", "world", "!"]


def basic_split(text: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(text))


def word_to_initial_pieces(word: str) -> Tuple[str, ...]:
    """
    WordPiece-style initial segmentation:
      'cat' -> ('c', '##a', '##t')
      'I'   -> ('I',)
    """
    if not word:
        return tuple()
    chars = list(word)
    pieces = [chars[0]] + [f"##{c}" for c in chars[1:]]
    return tuple(pieces)


def merge_pair_in_word(pieces: Tuple[str, ...], a: str, b: str, merged: str) -> Tuple[str, ...]:
    """
    Replace occurrences of adjacent pair (a, b) with merged inside a word's piece sequence.
    """
    out: List[str] = []
    i = 0
    while i < len(pieces):
        if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return tuple(out)


def compute_pair_counts(word_pieces: Dict[Tuple[str, ...], int]) -> Counter[Tuple[str, str]]:
    """
    Count adjacent piece pairs across the corpus, weighted by word frequency.
    """
    pair_counts: Counter[Tuple[str, str]] = Counter()
    for pieces, freq in word_pieces.items():
        for i in range(len(pieces) - 1):
            pair_counts[(pieces[i], pieces[i + 1])] += freq
    return pair_counts


def train_wordpiece(
    texts: Iterable[str],
    vocab_size: int = 2000,
    min_pair_freq: int = 2,
) -> Dict[str, int]:
    """
    Train a small WordPiece-like vocabulary with merge operations (BPE-ish),
    using '##' prefix for non-initial subword pieces.

    Returns token->id vocab dict (including SPECIAL_TOKENS).
    """
    # 1) Build word frequency table
    word_freq: Counter[str] = Counter()
    for text in texts:
        for tok in basic_split(text):
            # Only train subwords on "word tokens"; keep punctuation as atomic tokens
            if tok.isalnum() or "_" in tok:
                word_freq[tok] += 1

    # Edge case: empty corpus
    vocab = set(SPECIAL_TOKENS)
    if not word_freq:
        return {t: i for i, t in enumerate(sorted(vocab))}

    # 2) Initialize each word as char pieces
    word_pieces: Dict[Tuple[str, ...], int] = {}
    for w, f in word_freq.items():
        pieces = word_to_initial_pieces(w)
        if pieces:
            word_pieces[pieces] = word_pieces.get(pieces, 0) + f

    # Add initial character pieces to vocab
    for pieces in word_pieces.keys():
        for p in pieces:
            vocab.add(p)

    # 3) Iteratively merge frequent adjacent pairs
    while len(vocab) < vocab_size:
        pair_counts = compute_pair_counts(word_pieces)
        if not pair_counts:
            break

        (a, b), freq = pair_counts.most_common(1)[0]
        if freq < min_pair_freq:
            break

        # Create merged token name
        # WordPiece convention: if b starts with ##, merge removes the boundary.
        if b.startswith("##"):
            merged = a + b[2:]
        else:
            # This case happens rarely here, but keep it safe.
            merged = a + b

        if merged in vocab:
            # If already present, stop to avoid infinite loops
            break

        # Apply merge to all word piece sequences
        new_word_pieces: Dict[Tuple[str, ...], int] = {}
        for pieces, wf in word_pieces.items():
            new_seq = merge_pair_in_word(pieces, a, b, merged)
            new_word_pieces[new_seq] = new_word_pieces.get(new_seq, 0) + wf

        word_pieces = new_word_pieces
        vocab.add(merged)

    # 4) Also include punctuation tokens seen in the corpus (atomic)
    for text in texts:
        for tok in basic_split(text):
            if not (tok.isalnum() or "_" in tok):
                vocab.add(tok)

    # 5) Build stable ids (special tokens first, then the rest sorted)
    ordered = list(SPECIAL_TOKENS) + sorted(t for t in vocab if t not in SPECIAL_TOKENS)
    return {t: i for i, t in enumerate(ordered)}


@dataclass
class WordPieceTokenizer:
    vocab: Dict[str, int]

    def __post_init__(self) -> None:
        self.str_to_int = self.vocab
        self.int_to_str = {i: s for s, i in self.vocab.items()}
        self.unk = "<|unk|>"

    def encode(self, text: str) -> List[int]:
        tokens = basic_split(text)
        ids: List[int] = []
        for tok in tokens:
            if not (tok.isalnum() or "_" in tok):
                # punctuation: atomic
                ids.append(self.str_to_int.get(tok, self.str_to_int[self.unk]))
                continue

            pieces = self._encode_word(tok)
            for p in pieces:
                ids.append(self.str_to_int.get(p, self.str_to_int[self.unk]))
        return ids

    def _encode_word(self, word: str) -> List[str]:
        """
        Greedy longest-match WordPiece encoding.
        """
        if not word:
            return [self.unk]

        pieces: List[str] = []
        i = 0
        while i < len(word):
            matched = None
            # Try longest substring from current position
            for j in range(len(word), i, -1):
                substr = word[i:j]
                candidate = substr if i == 0 else f"##{substr}"
                if candidate in self.str_to_int:
                    matched = candidate
                    i = j
                    break

            if matched is None:
                return [self.unk]
            pieces.append(matched)

        return pieces

    def decode(self, ids: Sequence[int]) -> str:
        toks = [self.int_to_str.get(i, self.unk) for i in ids]
        out: List[str] = []
        for t in toks:
            if t in SPECIAL_TOKENS:
                # Keep special tokens as standalone (or you can skip them)
                out.append(t)
                continue

            if t.startswith("##") and out:
                out[-1] = out[-1] + t[2:]
            else:
                out.append(t)

        # Fix spacing before punctuation
        text = " ".join(out)
        text = re.sub(r"\s+([,.:;?!\"()\'])", r"\1", text)
        return text
