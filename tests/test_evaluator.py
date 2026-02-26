import pytest

from src.evaluator import TextClassificationEvaluator


def test_evaluator_returns_accuracy():
    ev = TextClassificationEvaluator(split="validation[:8]", batch_size=4)
    results = ev.run()

    assert isinstance(results, dict)
    assert "accuracy" in results
    assert isinstance(results["accuracy"], float)
    assert 0.0 <= results["accuracy"] <= 1.0


def test_evaluator_empty_dataset_raises():
    ev = TextClassificationEvaluator(split="validation[:0]")
    with pytest.raises(ValueError):
        ev.run()
