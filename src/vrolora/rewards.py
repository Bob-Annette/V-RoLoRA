"""Model-based conditional consistency reward used by V-RoLoRA RLVR."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vrolora.verifier import MORAL_DIMENSIONS, VALUE_NAMES, build_verifier_text

VALUE_EVAL_PATTERN = re.compile(
    r'(?P<stance>in favor of|against)\s+with that because\s+(?P<premise>.+?)"\s+'
    r"about\s+(?P<conclusion>.+?)\.?$",
    re.IGNORECASE | re.DOTALL,
)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(value, ensure_ascii=False)


@dataclass
class VerifierRewardConfig:
    model_name_or_path: str
    adapter_path: str | None = None
    task: str = "value"
    device: str | None = None
    threshold: float = 0.5
    max_length: int = 768
    consistency_weight: float = 1.0
    format_weight: float = 1.0
    stance_weight: float = 1.0
    cache_dir: str = ".cache/huggingface"
    trust_remote_code: bool = False


class VerifierReward:
    """Score generated responses with a multi-label condition discriminator."""

    def __init__(self, config: VerifierRewardConfig) -> None:
        task = config.task.lower()
        if task not in {"value", "mic"}:
            raise ValueError("task must be either 'value' or 'mic'")
        self.config = config
        self.labels = VALUE_NAMES if task == "value" else MORAL_DIMENSIONS
        self.device = self._select_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
            num_labels=len(self.labels),
            problem_type="multi_label_classification",
        )
        self.model = (
            PeftModel.from_pretrained(base_model, config.adapter_path)
            if config.adapter_path
            else base_model
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        **kwargs: Any,
    ) -> list[float]:
        targets = kwargs.get("value_ids")
        references = kwargs.get("response")
        if targets is None:
            raise ValueError("The verifier reward requires a value_ids condition vector")

        rewards = []
        for index, (prompt, completion, target) in enumerate(zip(prompts, completions, targets)):
            prompt_text, completion_text = _text(prompt), _text(completion)
            predicted = self._predict(prompt_text, completion_text)
            target_tensor = torch.as_tensor(target, dtype=torch.int64).flatten()
            if target_tensor.numel() != len(self.labels):
                raise ValueError(
                    f"Expected {len(self.labels)} condition labels for {self.config.task}, "
                    f"received {target_tensor.numel()}"
                )
            consistency = (predicted == target_tensor).float().mean().item()
            score = self.config.consistency_weight * consistency

            if self.config.task == "value":
                format_ok = float(VALUE_EVAL_PATTERN.search(completion_text.strip()) is not None)
                score += self.config.format_weight * format_ok
                if references is not None:
                    score += self.config.stance_weight * self._stance_match(
                        completion_text, _text(references[index])
                    )
            rewards.append(float(score))
        return rewards

    def _predict(self, prompt: str, completion: str) -> torch.Tensor:
        classifier_text = build_verifier_text(prompt, completion, self.config.task)
        encoded = self.tokenizer(
            classifier_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(**encoded).logits[0])
        return (probabilities >= self.config.threshold).to(torch.int64).cpu()

    @staticmethod
    def _stance_match(completion: str, reference: str) -> float:
        predicted = VALUE_EVAL_PATTERN.search(completion.strip())
        expected = VALUE_EVAL_PATTERN.search(reference.strip())
        if predicted is None or expected is None:
            return 0.0
        return float(predicted.group("stance").lower() == expected.group("stance").lower())

    @staticmethod
    def _select_device(requested: str | None) -> torch.device:
        if requested:
            device = torch.device(requested)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("A CUDA verifier device was requested, but CUDA is unavailable")
            return device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
