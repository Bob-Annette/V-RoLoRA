import torch
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING
from torch import nn

import vrolora
from vrolora.peft.layer import MOELoraLinear, ValueScoreEmbedding


def test_registry_is_available() -> None:
    assert PEFT_TYPE_TO_CONFIG_MAPPING["MOELORA"] is vrolora.MOELoraConfig
    assert PEFT_TYPE_TO_TUNER_MAPPING["MOELORA"] is vrolora.MOELoraModel


def test_projection_is_frozen_by_default() -> None:
    projection = ValueScoreEmbedding(6, 16)
    assert all(not parameter.requires_grad for parameter in projection.parameters())


def test_explicit_routing_selects_different_experts() -> None:
    base = nn.Linear(2, 1, bias=False)
    nn.init.zeros_(base.weight)
    layer = MOELoraLinear(
        "default",
        base_layer=base,
        in_features=2,
        out_features=1,
        r=2,
        lora_alpha=2,
        expert_num=2,
    )
    with torch.no_grad():
        layer.lora_A["default"].loraA[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        layer.lora_A["default"].loraA[1].weight.copy_(torch.tensor([[0.0, 1.0]]))
        layer.lora_B["default"].loraB[0].weight.fill_(1.0)
        layer.lora_B["default"].loraB[1].weight.fill_(1.0)

    inputs = torch.tensor([[2.0, 5.0], [2.0, 5.0]])
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    outputs = layer(inputs, expert_weight=weights).squeeze(-1)
    assert torch.allclose(outputs, torch.tensor([2.0, 5.0]))
