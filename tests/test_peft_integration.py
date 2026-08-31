import tempfile

import torch
from peft import PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from vrolora import MOELoraConfig, register_moelora


def _tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(config)


def test_peft_wrap_forward_save_and_reload() -> None:
    register_moelora()
    config = MOELoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        expert_num=2,
        task_num=3,
        task_embedding_dim=8,
    )
    model = get_peft_model(_tiny_llama(), config)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    value_ids = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    output = model(input_ids=input_ids, value_ids=value_ids)
    assert output.logits.shape == (2, 3, 64)
    output.logits.mean().backward()
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )

    with tempfile.TemporaryDirectory() as adapter_dir:
        model.save_pretrained(adapter_dir)
        reloaded = PeftModel.from_pretrained(_tiny_llama(), adapter_dir)
        reloaded_output = reloaded(input_ids=input_ids, value_ids=value_ids)
        assert reloaded_output.logits.shape == output.logits.shape
        expected = {
            name: parameter.detach()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }
        actual = {
            name: parameter.detach()
            for name, parameter in reloaded.named_parameters()
            if "lora_" in name
        }
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[name], actual[name]) for name in expected)
