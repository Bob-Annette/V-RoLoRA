"""Dataset formatting utilities for condition-controlled SFT and GRPO."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


def build_prompt(query: str, history: Sequence[Sequence[str]] | None = None, prefix: str = "") -> str:
    if not history:
        return prefix + query
    turns = []
    for turn_idx, (old_query, response) in enumerate(history):
        turns.append(
            f"[Round {turn_idx}]\n<|im_start|>user\n{old_query}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
    turns.append(
        f"[Round {len(history)}]\n<|im_start|>user\n{query}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prefix + "\n".join(turns)


def tokenize_sft_batch(
    examples: dict[str, list[Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    prompt_column: str,
    response_column: str,
    condition_column: str,
    history_column: str | None,
    prefix: str,
    max_source_length: int,
    max_target_length: int,
) -> dict[str, list[Any]]:
    output: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": [], "value_ids": []}
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id")

    for index, query in enumerate(examples[prompt_column]):
        answer = examples[response_column][index]
        if not query or not answer:
            continue
        history = examples[history_column][index] if history_column and history_column in examples else None
        prompt = build_prompt(str(query), history, prefix)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[-max_source_length:]
        answer_ids = tokenizer.encode(str(answer), add_special_tokens=False)[: max_target_length - 1]
        input_ids = prompt_ids + answer_ids + [eos_token_id]
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids + [eos_token_id]
        output["input_ids"].append(input_ids)
        output["attention_mask"].append([1] * len(input_ids))
        output["labels"].append(labels)
        output["value_ids"].append(examples[condition_column][index])
    return output


def format_grpo_batch(
    examples: dict[str, list[Any]],
    *,
    prompt_column: str,
    response_column: str | None,
    condition_column: str,
    history_column: str | None,
    prefix: str,
) -> dict[str, list[Any]]:
    output: dict[str, list[Any]] = {"prompt": [], "value_ids": []}
    if response_column:
        output["response"] = []
    for index, query in enumerate(examples[prompt_column]):
        history = examples[history_column][index] if history_column and history_column in examples else None
        output["prompt"].append(build_prompt(str(query), history, prefix))
        output["value_ids"].append(examples[condition_column][index])
        if response_column:
            output["response"].append(examples[response_column][index])
    return output


@dataclass
class ConditionedCausalCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [
            {"input_ids": item["input_ids"], "attention_mask": item.get("attention_mask")}
            for item in instances
        ]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        labels = torch.full((len(instances), max_length), IGNORE_INDEX, dtype=torch.long)
        for row, item in enumerate(instances):
            item_labels = torch.as_tensor(item["labels"], dtype=torch.long)
            labels[row, : item_labels.numel()] = item_labels
        batch["labels"] = labels
        batch["value_ids"] = torch.stack(
            [torch.as_tensor(item["value_ids"], dtype=torch.float32) for item in instances]
        )
        return batch
