"""PEFT model wrapper for V-RoLoRA."""

import re
import warnings

import torch
from peft.tuners.lora import LoraLayer, LoraModel
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
)
from torch import nn
from transformers.pytorch_utils import Conv1D

from .layer import MOELoraLayer, MOELoraLinear, ValueScoreEmbedding


def mark_only_adapter_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name
    if bias == "none":
        return
    if bias == "all":
        for name, parameter in model.named_parameters():
            if "bias" in name:
                parameter.requires_grad = True
        return
    if bias == "lora_only":
        for module in model.modules():
            if isinstance(module, LoraLayer) and getattr(module, "bias", None) is not None:
                module.bias.requires_grad = True
        return
    raise NotImplementedError(f"Unsupported bias mode: {bias}")


class Router(nn.Module):
    def __init__(self, input_size: int, expert_num: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, expert_num, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.proj(inputs), dim=-1)


class MOELoraModel(LoraModel):
    """Wrap a Transformers model with condition-routed LoRA expert pools."""

    prefix = "lora_"

    def __init__(self, model: nn.Module, config: dict, adapter_name: str) -> None:
        nn.Module.__init__(self)
        self.model = model
        self.peft_config = config
        self.adapter_name = adapter_name
        self.add_adapter(adapter_name, config[adapter_name])

        adapter_config = self.peft_config[adapter_name]
        self.task_num = adapter_config.task_num
        self.expert_num = adapter_config.expert_num
        projection_dim = adapter_config.task_embedding_dim
        self.lora_task_embedding = nn.ModuleDict(
            {adapter_name: nn.Embedding(self.task_num + 1, projection_dim)}
        )
        self.lora_value_embedding = nn.ModuleDict(
            {
                adapter_name: ValueScoreEmbedding(
                    self.task_num,
                    projection_dim,
                    std=adapter_config.projection_std,
                    sparsity=adapter_config.projection_sparsity,
                    frozen=adapter_config.freeze_projection,
                )
            }
        )
        self.lora_share_gate = nn.ModuleDict(
            {adapter_name: Router(projection_dim, self.expert_num)}
        )

    def _compute_expert_weights(
        self,
        *,
        expert_weight=None,
        task_weight=None,
        task_id=None,
        value_ids=None,
    ) -> torch.Tensor | None:
        router = self.lora_share_gate[self.adapter_name]
        device = next(router.parameters()).device
        if expert_weight is not None:
            return torch.as_tensor(expert_weight, device=device)
        if task_weight is not None:
            return router(
                torch.as_tensor(
                    task_weight,
                    device=device,
                    dtype=next(router.parameters()).dtype,
                )
            )
        if value_ids is not None:
            values = torch.as_tensor(value_ids, device=device)
            projected = self.lora_value_embedding[self.adapter_name](values)
            return router(projected)
        if task_id is not None:
            task_ids = torch.as_tensor(task_id, dtype=torch.long, device=device)
            embedded = self.lora_task_embedding[self.adapter_name](task_ids)
            return router(embedded)
        return None

    def _set_cache_on_layers(
        self,
        expert_weight=None,
        task_weight=None,
        task_id=None,
        value_ids=None,
    ) -> None:
        weights = self._compute_expert_weights(
            expert_weight=expert_weight,
            task_weight=task_weight,
            task_id=task_id,
            value_ids=value_ids,
        )
        for module in self.model.modules():
            if isinstance(module, MOELoraLinear):
                module._expert_weight_cache = (
                    None if weights is None else weights.to(next(module.parameters()).device)
                )

    def forward(
        self,
        *args,
        expert_weight=None,
        task_weight=None,
        task_id=None,
        value_ids=None,
        **kwargs,
    ):
        self._set_cache_on_layers(
            expert_weight=expert_weight,
            task_weight=task_weight,
            task_id=task_id,
            value_ids=value_ids,
        )
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self._set_cache_on_layers(
            expert_weight=kwargs.pop("expert_weight", None),
            task_weight=kwargs.pop("task_weight", None),
            task_id=kwargs.pop("task_id", None),
            value_ids=kwargs.pop("value_ids", None),
        )
        return self.model.generate(*args, **kwargs)

    def add_adapter(self, adapter_name: str, config=None) -> None:
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_moelora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("V-RoLoRA supports only one adapter with a trainable bias")
        mark_only_adapter_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name: str) -> None:
        config = self.peft_config[adapter_name]
        target_found = False
        layer_kwargs = {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "fan_in_fan_out": config.fan_in_fan_out,
            "init_lora_weights": config.init_lora_weights,
            "expert_num": config.expert_num,
        }
        for key, _ in list(self.model.named_modules()):
            if isinstance(config.target_modules, str):
                matches = re.fullmatch(config.target_modules, key)
            else:
                matches = any(key.endswith(name) for name in config.target_modules)
            if not matches:
                continue
            target_found = True
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, MOELoraLayer):
                target.update_layer(
                    adapter_name,
                    config.r,
                    config.lora_alpha,
                    config.lora_dropout,
                    config.init_lora_weights,
                )
                continue
            if isinstance(target, nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if layer_kwargs["fan_in_fan_out"]:
                    warnings.warn("Disabling fan_in_fan_out for torch.nn.Linear", stacklevel=2)
                    layer_kwargs["fan_in_fan_out"] = config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = target.weight.shape
                if not layer_kwargs["fan_in_fan_out"]:
                    warnings.warn("Enabling fan_in_fan_out for transformers.Conv1D", stacklevel=2)
                    layer_kwargs["fan_in_fan_out"] = config.fan_in_fan_out = True
            else:
                raise TypeError(
                    f"Unsupported target module {type(target).__name__}; use Linear or Conv1D modules"
                )
            replacement = MOELoraLinear(
                adapter_name=adapter_name,
                base_layer=target,
                in_features=in_features,
                out_features=out_features,
                **layer_kwargs,
            )
            self._replace_module(parent, target_name, replacement, target)
        if not target_found:
            raise ValueError(f"Target modules {config.target_modules} were not found in the base model")

    @staticmethod
    def _prepare_moelora_config(config, model_config):
        if config.target_modules is None:
            model_type = model_config["model_type"]
            if model_type not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify target_modules for this model architecture")
            config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
        return config

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
