from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import evaluate
from datasets import load_dataset
from transformers import pipeline


@dataclass(frozen=True)
class TextClassificationEvaluator:
    """
    Custom evaluator (minimal):
    - Loads a labeled dataset split (GLUE/SST-2)
    - Runs a transformers text-classification pipeline for predictions
    - Computes accuracy using the evaluate library
    """

    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    split: str = "validation[:64]"  # small slice to keep runtime reasonable
    text_column: str = "sentence"
    label_column: str = "label"
    metric_name: str = "accuracy"
    batch_size: int = 16

    def _validate_dataset(self, ds: Any) -> None:
        if len(ds) == 0:
            raise ValueError(f"Dataset split produced 0 rows: split={self.split!r}")
        if self.text_column not in ds.column_names:
            raise ValueError(f"Missing text column {self.text_column!r}. Available: {ds.column_names}")
        if self.label_column not in ds.column_names:
            raise ValueError(f"Missing label column {self.label_column!r}. Available: {ds.column_names}")

    def _batched(self, items: list[str], batch_size: int) -> Iterable[list[str]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def run(self) -> dict[str, Any]:
        ds = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
        self._validate_dataset(ds)

        texts: list[str] = list(ds[self.text_column])
        labels: list[int] = list(ds[self.label_column])

        clf = pipeline("text-classification", model=self.model_name)

        preds: list[int] = []
        for batch in self._batched(texts, self.batch_size):
            outputs = clf(batch, truncation=True)
            for out in outputs:
                label_str = str(out["label"]).upper()
                if label_str in {"POSITIVE", "LABEL_1"}:
                    preds.append(1)
                elif label_str in {"NEGATIVE", "LABEL_0"}:
                    preds.append(0)
                else:
                    raise ValueError(f"Unexpected model label: {out['label']!r}")

        if len(preds) != len(labels):
            raise RuntimeError("Prediction count did not match label count.")

        metric = evaluate.load(self.metric_name)
        results = metric.compute(predictions=preds, references=labels)
        return dict(results)


def main() -> None:
    ev = TextClassificationEvaluator()
    results = ev.run()
    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
