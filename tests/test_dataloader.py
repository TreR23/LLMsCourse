from __future__ import annotations

from pathlib import Path

import torch

from src.dataloader import StarWarsDataset, make_starwars_dataloader



def test_starwars_dataset_len_and_item() -> None:
    dataset = StarWarsDataset("SW_EpisodeIV_VI.json")

    # Basic sanity checks
    assert len(dataset) > 0

    ex0 = dataset[0]
    assert isinstance(ex0.character, str)
    assert isinstance(ex0.line, str)
    assert ex0.character != ""
    assert ex0.line != ""


def test_starwars_dataloader_batch_shape() -> None:
    dl = make_starwars_dataloader(
        json_path="SW_EpisodeIV_VI.json",
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dl))

    assert set(batch.keys()) == {"character", "line", "line_length"}

    assert isinstance(batch["character"], list)
    assert isinstance(batch["line"], list)
    assert len(batch["character"]) == 4
    assert len(batch["line"]) == 4

    assert isinstance(batch["line_length"], torch.Tensor)
    assert batch["line_length"].shape == (4,)
    assert batch["line_length"].dtype == torch.int64


def test_invalid_json_raises(tmp_path: Path) -> None:
    # Dataset expects a JSON list, so a dict should raise ValueError.
    bad = tmp_path / "bad.json"
    bad.write_text('{"Character": "LUKE", "Line": "Hi"}', encoding="utf-8")

    try:
        StarWarsDataset(bad)
        raise AssertionError("Expected ValueError for invalid JSON shape")
    except ValueError:
        pass
