import numpy as np
import torch

from vrolora.verifier import (
    build_verifier_text,
    condition_labels,
    multilabel_metrics,
    positive_class_weights,
)


def test_verifier_text_matches_training_format() -> None:
    value_text = build_verifier_text("unused", "against with that because ...", "value")
    assert value_text == 'I would say, "I against with that because ...'

    mic_text = build_verifier_text("System prefix\nQ: Is this fair?\nA:", "Yes.", "mic")
    assert mic_text == "Q: Is this fair?\nA:Yes."


def test_task_dimensions() -> None:
    assert len(condition_labels("value")) == 10
    assert len(condition_labels("mic")) == 6


def test_positive_weights_and_metrics() -> None:
    weights = positive_class_weights([[1, 0], [1, 1], [0, 0]])
    assert torch.allclose(weights, torch.tensor([0.5, 2.0]))

    logits = np.array([[5.0, -5.0], [-5.0, 5.0]])
    labels = np.array([[1, 0], [0, 1]])
    metrics = multilabel_metrics(logits, labels)
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["jaccard"] == 1.0
