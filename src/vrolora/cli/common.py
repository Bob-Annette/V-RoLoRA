"""Shared command-line configuration and model loading."""

import sys
from dataclasses import dataclass, field

from peft import PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from vrolora import MOELoraConfig


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Hugging Face model ID or a relative model path."})
    peft_path: str | None = field(default=None, metadata={"help": "Relative cold-start adapter path."})
    cache_dir: str = field(default=".cache/huggingface")
    trust_remote_code: bool = field(default=False)
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    lora_rank: int = field(default=32)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    expert_num: int = field(default=8)
    condition_dim: int = field(default=10)
    projection_dim: int = field(default=64)
    projection_std: float = field(default=0.01)
    projection_sparsity: float = field(default=0.5)


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Relative JSON or JSONL training file."})
    validation_file: str | None = field(default=None)
    prompt_column: str = field(default="input")
    response_column: str = field(default="response")
    condition_column: str = field(default="value_ids")
    history_column: str | None = field(default=None)
    source_prefix: str = field(default="")
    max_source_length: int = field(default=400)
    max_target_length: int = field(default=200)


def configure_utf8_stdout() -> None:
    """Allow dependency help text to render on non-UTF-8 Windows consoles."""

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def load_tokenizer(model_args: ModelArguments):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy(model_args: ModelArguments, *, is_trainable: bool = True):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        dtype="auto",
    )
    model.config.use_cache = False
    if model_args.peft_path:
        return PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=is_trainable)
    config = MOELoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[item.strip() for item in model_args.target_modules.split(",") if item.strip()],
        inference_mode=not is_trainable,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        expert_num=model_args.expert_num,
        task_num=model_args.condition_dim,
        task_embedding_dim=model_args.projection_dim,
        projection_std=model_args.projection_std,
        projection_sparsity=model_args.projection_sparsity,
        freeze_projection=True,
    )
    return get_peft_model(model, config)
