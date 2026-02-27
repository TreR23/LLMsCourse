from __future__ import annotations

import importlib
from typing import Any

import pytest


def test_trainer_module_imports() -> None:
    # Just importing exercises module-level code paths (if any) and counts for coverage.
    importlib.import_module("src.trainer")


def test_trainer_config_defaults_exist() -> None:
    from src.trainer import TrainerConfig

    cfg = TrainerConfig()
    assert isinstance(cfg.model_name, str)
    assert cfg.model_name
    assert "sst2" in cfg.dataset_config.lower()


def test_trainer_tokenize_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Cover ModelTrainer.__init__ and _tokenize without downloading real models.
    We monkeypatch AutoTokenizer.from_pretrained to a tiny stub.
    """
    from src import trainer as trainer_mod

    class DummyTokenizer:
        def __call__(self, texts: Any, truncation: bool, padding: str):
            # Return the keys Trainer expects later
            if isinstance(texts, list):
                n = len(texts)
            else:
                n = 1
            return {
                "input_ids": [[1, 2, 3]] * n,
                "attention_mask": [[1, 1, 1]] * n,
            }

    monkeypatch.setattr(
        trainer_mod.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    cfg = trainer_mod.TrainerConfig(train_split="train[:2]", eval_split="validation[:2]")
    tr = trainer_mod.ModelTrainer(cfg)

    out = tr._tokenize({"sentence": ["hi", "there"]})
    assert "input_ids" in out
    assert "attention_mask" in out
