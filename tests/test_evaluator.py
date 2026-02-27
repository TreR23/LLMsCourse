from __future__ import annotations

from typing import Any

import pytest

import src.evaluator as evaluator_mod
from src.evaluator import TextClassificationEvaluator


class DummyDataset:
    def __init__(self, sentences: list[str], labels: list[int]) -> None:
        self._sentences = sentences
        self._labels = labels
        self.column_names = ["sentence", "label"]

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, key: str) -> Any:
        if key == "sentence":
            return self._sentences
        if key == "label":
            return self._labels
        raise KeyError(key)


class DummyMetric:
    def compute(self, predictions: list[int], references: list[int]) -> dict[str, float]:
        correct = sum(int(p == r) for p, r in zip(predictions, references))
        return {"accuracy": correct / max(1, len(references))}


class DummyPipeline:
    # Return HF-like outputs: [{"label": "POSITIVE"}, {"label": "NEGATIVE"}, ...]
    def __call__(self, batch: list[str], truncation: bool = True) -> list[dict[str, str]]:
        # simple deterministic rule so test is stable:
        # even index => POSITIVE, odd index => NEGATIVE
        out: list[dict[str, str]] = []
        for i, _txt in enumerate(batch):
            out.append({"label": "POSITIVE" if i % 2 == 0 else "NEGATIVE"})
        return out


def test_evaluator_returns_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    # 8 examples
    sentences = [f"s{i}" for i in range(8)]
    # match DummyPipeline rule for perfect accuracy
    labels = [1 if i % 2 == 0 else 0 for i in range(8)]
    ds = DummyDataset(sentences, labels)

    monkeypatch.setattr(evaluator_mod, "load_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(evaluator_mod, "pipeline", lambda *args, **kwargs: DummyPipeline())
    monkeypatch.setattr(evaluator_mod.evaluate, "load", lambda *args, **kwargs: DummyMetric())

    ev = TextClassificationEvaluator(split="validation[:8]", batch_size=4)
    results = ev.run()
    assert "accuracy" in results
    assert results["accuracy"] == 1.0


def test_evaluator_empty_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = DummyDataset([], [])

    monkeypatch.setattr(evaluator_mod, "load_dataset", lambda *args, **kwargs: ds)
    monkeypatch.setattr(evaluator_mod, "pipeline", lambda *args, **kwargs: DummyPipeline())
    monkeypatch.setattr(evaluator_mod.evaluate, "load", lambda *args, **kwargs: DummyMetric())

    ev = TextClassificationEvaluator(split="validation[:0]")
    with pytest.raises(ValueError):
        ev.run()