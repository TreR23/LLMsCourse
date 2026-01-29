import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


def build_vocab(preprocessed: Iterable[str]) -> Dict[str, int]:
    """Build a vocab dict mapping token -> integer id."""
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {token: integer for integer, token in enumerate(all_tokens)}


class SimpleTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        preprocessed = re.split(r"([,.:;?_!\"()\']|--|\s)", text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: List[int]) -> str:
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r"\s+([,.:;?!" r"()\'])", r"\1", text)
        return text


def load_star_wars_dataset(path: Path) -> list[str]:
    """
    Load Star Wars dataset lines from a JSON file.

    Expected format:
      [
        {"Character": "...", "Line": "..."},
        ...
      ]

    Returns a list of the "Line" strings (non-empty).
    """
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".json":
        return []

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8-sig")

    data = json.loads(raw)
    if not isinstance(data, list):
        return []

    lines: list[str] = []
    for item in data:
        if isinstance(item, dict):
            line = item.get("Line")
            if isinstance(line, str):
                line = line.strip()
                if line:
                    lines.append(line)

    return lines
