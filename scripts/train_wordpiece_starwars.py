from pathlib import Path
import json

from src.tokenizer import WordPieceTokenizer, train_wordpiece
from src.data import load_star_wars_dataset  # adjust if needed


def main() -> None:
    data_path = Path("data/star_wars.json")

    lines = load_star_wars_dataset(data_path)

    vocab = train_wordpiece(
        lines,
        vocab_size=2000,
        min_pair_freq=2,
    )

    tokenizer = WordPieceTokenizer(vocab)

    # Optional: save vocab for reuse
    out_path = Path("artifacts/wordpiece_vocab.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    # Quick sanity check
    sample = "I have a bad feeling about this."
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)

    print(sample)
    print(ids)
    print(decoded)


if __name__ == "__main__":
    main()
