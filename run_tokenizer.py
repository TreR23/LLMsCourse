from pathlib import Path
from tokenizer import SimpleTokenizer, build_vocab

DATA_DIR = Path("SW_EpisodeIV_VI")

# Read all text files
texts = []
for file in DATA_DIR.glob("*.txt"):
    texts.append(file.read_text(encoding="utf-8"))

# Very naive preprocessing (just split on whitespace for vocab)
preprocessed = []
for text in texts:
    preprocessed.extend(text.split())

vocab = build_vocab(preprocessed)
tokenizer = SimpleTokenizer(vocab)

# Encode a sample
sample_text = texts[0][:500]
ids = tokenizer.encode(sample_text)
decoded = tokenizer.decode(ids)

print("Sample original:")
print(sample_text[:200])
print("\nDecoded:")
print(decoded[:200])
