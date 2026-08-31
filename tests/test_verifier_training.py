import tempfile

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForSequenceClassification, TrainingArguments

from vrolora.cli.train_verifier import MultiLabelTrainer


def _tiny_classifier() -> LlamaForSequenceClassification:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        num_labels=2,
        pad_token_id=0,
        problem_type="multi_label_classification",
    )
    return LlamaForSequenceClassification(config)


def _collate(features: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([feature["input_ids"] for feature in features]),
        "attention_mask": torch.tensor([feature["attention_mask"] for feature in features]),
        "labels": torch.tensor([feature["labels"] for feature in features]),
    }


def test_verifier_trains_saves_and_reloads() -> None:
    model = get_peft_model(
        _tiny_classifier(),
        LoraConfig(task_type=TaskType.SEQ_CLS, r=2, lora_alpha=2, target_modules=["q_proj"]),
    )
    dataset = Dataset.from_list(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1.0, 0.0]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1], "labels": [0.0, 1.0]},
        ]
    )

    with tempfile.TemporaryDirectory() as output_dir:
        trainer = MultiLabelTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                max_steps=1,
                per_device_train_batch_size=2,
                save_strategy="no",
                report_to="none",
                disable_tqdm=True,
                use_cpu=True,
            ),
            train_dataset=dataset,
            data_collator=_collate,
            positive_weights=torch.ones(2),
        )
        trainer.train()
        trainer.save_model(output_dir)
        reloaded = PeftModel.from_pretrained(_tiny_classifier(), output_dir)
        output = reloaded(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )
        assert output.logits.shape == (1, 2)
