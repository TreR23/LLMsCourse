# The Star Wars file should be a list of {"Character": ..., "Line": ...} dicts.
# Validate early so failures are clear during testing/CI.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class StarWarsExample:
    """One training example from the Star Wars dialogue dataset."""

    character: str
    line: str


class StarWarsDataset(Dataset[StarWarsExample]):
    """
    A minimal PyTorch Dataset for the HW1 Star Wars JSON.

    Expected JSON format: a list of objects like:
      {"Character": "LUKE", "Line": "I'm Luke Skywalker..."}
    """

    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        raw = json.loads(self.json_path.read_text(encoding="utf-8"))

        if not isinstance(raw, list):
            raise ValueError("StarWars JSON must be a list of objects.")

        examples: List[StarWarsExample] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not an object/dict.")

            # Match your HW1 keys exactly
            character = item.get("Character")
            line = item.get("Line")

            if not isinstance(character, str) or not isinstance(line, str):
                raise ValueError(f"Item {i} must contain string fields 'Character' and 'Line'.")

            examples.append(StarWarsExample(character=character, line=line))

        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> StarWarsExample:
        return self._examples[index]


def collate_starwars(batch: List[StarWarsExample]) -> Dict[str, Any]:
    """
    Collate a list of StarWarsExample objects into a batch.

    We keep text as Python lists (not tokenized) because:
    - This homework is about DataLoader mechanics, not modeling.
    - Tokenization/padding choices vary; tests stay simple and robust.
    """
    characters = [ex.character for ex in batch]
    lines = [ex.line for ex in batch]

    # Helpful to include indices or lengths for later use/testing
    line_lengths = torch.tensor([len(s) for s in lines], dtype=torch.int64)

    return {
        "character": characters,
        "line": lines,
        "line_length": line_lengths,
    }


def make_starwars_dataloader(
    json_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Convenience helper: create a Dataset + DataLoader in one call.
    """
    dataset = StarWarsDataset(json_path=json_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_starwars,
    )
