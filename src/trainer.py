from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.evaluator import TextClassificationEvaluator


@dataclass
class TrainerConfig:
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    train_split: str = "train[:128]"
    eval_split: str = "validation[:128]"
    output_dir: str = "./results"
    batch_size: int = 8
    num_train_epochs: int = 1


class ModelTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
        )

    def load_datasets(self):
        train_ds = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.train_split,
        )

        eval_ds = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.eval_split,
        )

        train_ds = train_ds.map(self._tokenize, batched=True)
        eval_ds = eval_ds.map(self._tokenize, batched=True)

        train_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        eval_ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        return train_ds, eval_ds

    def train(self) -> Dict[str, Any]:
        train_ds, eval_ds = self.load_datasets()

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
        )

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=10,
            save_strategy="no",
            evaluation_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )

        trainer.train()

        return {"status": "training_complete"}

    def evaluate(self) -> Dict[str, Any]:
        evaluator = TextClassificationEvaluator()
        return evaluator.run()
