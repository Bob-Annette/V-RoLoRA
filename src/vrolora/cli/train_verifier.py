"""Train the lightweight multi-label discriminator used by RLVR."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from vrolora.cli.common import configure_utf8_stdout
from vrolora.verifier import (
    build_verifier_text,
    condition_labels,
    multilabel_metrics,
    positive_class_weights,
)


@dataclass
class VerifierModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B-Base")
    cache_dir: str = field(default=".cache/huggingface")
    trust_remote_code: bool = field(default=False)
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)


@dataclass
class VerifierDataArguments:
    train_file: str = field(metadata={"help": "Relative JSON or JSONL training file."})
    validation_file: str = field(metadata={"help": "Relative JSON or JSONL validation file."})
    task: str = field(default="value", metadata={"help": "value or mic"})
    prompt_column: str = field(default="input")
    response_column: str = field(default="response")
    condition_column: str = field(default="value_ids")
    max_length: int = field(default=768)
    threshold: float = field(default=0.5)


class MultiLabelCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.float32)
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


class MultiLabelTrainer(Trainer):
    def __init__(self, *args, positive_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positive_weights = positive_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.positive_weights.to(outputs.logits.device, outputs.logits.dtype)
        )
        loss = loss_function(outputs.logits, labels.to(outputs.logits.device, outputs.logits.dtype))
        return (loss, outputs) if return_outputs else loss


def _tokenize_batch(
    examples: dict[str, list[Any]],
    *,
    tokenizer,
    data_args: VerifierDataArguments,
    expected_labels: int,
) -> dict[str, list[Any]]:
    texts = []
    labels = []
    for prompt, response, condition in zip(
        examples[data_args.prompt_column],
        examples[data_args.response_column],
        examples[data_args.condition_column],
    ):
        if len(condition) != expected_labels:
            raise ValueError(
                f"Expected {expected_labels} labels for {data_args.task}, received {len(condition)}"
            )
        texts.append(build_verifier_text(prompt, response, data_args.task))
        labels.append([float(value) for value in condition])
    tokenized = tokenizer(texts, truncation=True, max_length=data_args.max_length)
    tokenized["labels"] = labels
    return tokenized


def main() -> None:
    configure_utf8_stdout()
    parser = HfArgumentParser(
        (VerifierModelArguments, VerifierDataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    labels = condition_labels(data_args.task)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(
        "json",
        data_files={"train": data_args.train_file, "validation": data_args.validation_file},
        cache_dir=model_args.cache_dir,
    )
    preprocess = partial(
        _tokenize_batch,
        tokenizer=tokenizer,
        data_args=data_args,
        expected_labels=len(labels),
    )
    processed = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc=f"Tokenizing the {data_args.task} verifier data",
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        dtype="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    adapter_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=[
            module.strip() for module in model_args.target_modules.split(",") if module.strip()
        ],
    )
    model = get_peft_model(base_model, adapter_config)
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    positive_weights = positive_class_weights(processed["train"]["labels"])

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        return multilabel_metrics(
            prediction.predictions,
            prediction.label_ids,
            threshold=data_args.threshold,
        )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        data_collator=MultiLabelCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        positive_weights=positive_weights,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
