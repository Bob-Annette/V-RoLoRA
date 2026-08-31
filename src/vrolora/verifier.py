"""Shared preprocessing and metrics for the external condition verifier."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch

VALUE_NAMES = (
    "Achievement",
    "Benevolence",
    "Conformity",
    "Hedonism",
    "Power",
    "Security",
    "Self-Direction",
    "Stimulation",
    "Tradition",
    "Universalism",
)
MORAL_DIMENSIONS = ("Care", "Fairness", "Liberty", "Loyalty", "Authority", "Sanctity")

MIC_QUESTION_PATTERN = re.compile(r"(Q\s*:.*)", re.IGNORECASE | re.DOTALL)


def condition_labels(task: str) -> tuple[str, ...]:
    task_name = task.strip().lower()
    if task_name == "value":
        return VALUE_NAMES
    if task_name == "mic":
        return MORAL_DIMENSIONS
    raise ValueError("task must be either 'value' or 'mic'")


def build_verifier_text(prompt: str, response: str, task: str) -> str:
    """Match the verifier input construction used in the paper experiments."""

    task_name = task.strip().lower()
    prompt_text = str(prompt).strip()
    response_text = str(response).strip()
    if task_name == "mic":
        match = MIC_QUESTION_PATTERN.search(prompt_text)
        question = match.group(1).strip() if match else prompt_text
        return f"{question}{response_text}"
    if task_name == "value":
        if response_text.lower().startswith("i would say"):
            return response_text
        return f'I would say, "I {response_text}'
    raise ValueError("task must be either 'value' or 'mic'")


def positive_class_weights(label_rows: Sequence[Sequence[float]]) -> torch.Tensor:
    labels = np.asarray(label_rows, dtype=np.float32)
    if labels.ndim != 2 or labels.shape[0] == 0:
        raise ValueError("Expected a non-empty two-dimensional label matrix")
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    return torch.tensor(negatives / np.clip(positives, 1, None), dtype=torch.float32)


def multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    predictions = (probabilities >= threshold).astype(np.int32)
    targets = labels.astype(np.int32)

    true_positive = (predictions & targets).sum()
    false_positive = (predictions & (1 - targets)).sum()
    false_negative = ((1 - predictions) & targets).sum()
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    per_label_true_positive = (predictions & targets).sum(axis=0)
    per_label_false_positive = (predictions & (1 - targets)).sum(axis=0)
    per_label_false_negative = ((1 - predictions) & targets).sum(axis=0)
    per_label_precision = np.divide(
        per_label_true_positive,
        per_label_true_positive + per_label_false_positive,
        out=np.zeros_like(per_label_true_positive, dtype=np.float64),
        where=(per_label_true_positive + per_label_false_positive) > 0,
    )
    per_label_recall = np.divide(
        per_label_true_positive,
        per_label_true_positive + per_label_false_negative,
        out=np.zeros_like(per_label_true_positive, dtype=np.float64),
        where=(per_label_true_positive + per_label_false_negative) > 0,
    )
    per_label_f1 = np.divide(
        2 * per_label_precision * per_label_recall,
        per_label_precision + per_label_recall,
        out=np.zeros_like(per_label_true_positive, dtype=np.float64),
        where=(per_label_precision + per_label_recall) > 0,
    )
    intersection = (predictions & targets).sum()
    union = (predictions | targets).sum()

    return {
        "micro_f1": float(micro_f1),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_accuracy": float((predictions == targets).mean()),
        "macro_f1": float(per_label_f1.mean()) if per_label_f1.size else 0.0,
        "jaccard": float(intersection / union) if union > 0 else 0.0,
    }
