"""Value-conditioned mixture-of-experts LoRA layers."""

import warnings

import torch
from peft.tuners.lora import LoraLayer
from torch import nn


class ValueScoreEmbedding(nn.Module):
    """Project a multi-hot condition vector through a fixed sparse Gaussian map."""

    def __init__(
        self,
        input_size: int,
        output_dim: int,
        *,
        std: float = 0.01,
        sparsity: float = 0.5,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_dim)
        with torch.no_grad():
            nn.init.normal_(self.linear.weight, mean=0.0, std=std)
            if sparsity > 0:
                keep = torch.rand_like(self.linear.weight) >= sparsity
                self.linear.weight.mul_(keep)
            nn.init.zeros_(self.linear.bias)
        if frozen:
            self.linear.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(dtype=self.linear.weight.dtype)
        max_value = values.amax().clamp_min(1.0)
        return self.linear(values / max_value)


class Expert(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)

    @property
    def weight(self) -> nn.Parameter:
        return self.mlp.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class MOELinearA(nn.Module):
    def __init__(self, in_features: int, total_rank: int, expert_num: int) -> None:
        super().__init__()
        if total_rank % expert_num != 0:
            raise ValueError("LoRA rank must be divisible by the number of experts")
        self.expert_num = expert_num
        self.rank_per_expert = total_rank // expert_num
        self.loraA = nn.ModuleList(
            [Expert(in_features, self.rank_per_expert) for _ in range(expert_num)]
        )


class MOELinearB(nn.Module):
    def __init__(self, total_rank: int, out_features: int, expert_num: int) -> None:
        super().__init__()
        if total_rank % expert_num != 0:
            raise ValueError("LoRA rank must be divisible by the number of experts")
        self.expert_num = expert_num
        self.rank_per_expert = total_rank // expert_num
        self.loraB = nn.ModuleList(
            [Expert(self.rank_per_expert, out_features) for _ in range(expert_num)]
        )


class MOELoraLayer(LoraLayer):
    def __init__(self, base_layer: nn.Module, in_features: int, out_features: int, expert_num: int) -> None:
        super().__init__(base_layer=base_layer, in_features=in_features, out_features=out_features)
        self.expert_num = expert_num

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool,
    ) -> None:
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: dropout}))
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict({adapter_name: MOELinearA(self.in_features, r, self.expert_num)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: MOELinearB(r, self.out_features, self.expert_num)})
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.get_base_layer().weight.device)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: bool = True) -> None:
        if adapter_name not in self.lora_A:
            return
        for expert_idx in range(self.expert_num):
            nn.init.normal_(
                self.lora_A[adapter_name].loraA[expert_idx].weight,
                mean=0.0,
                std=0.01,
            )
            nn.init.zeros_(self.lora_B[adapter_name].loraB[expert_idx].weight)


class MOELoraLinear(MOELoraLayer, nn.Module):
    """Linear layer augmented with a sample-level mixture of LoRA experts."""

    def __init__(
        self,
        adapter_name: str,
        *,
        base_layer: nn.Module,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ) -> None:
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = int(kwargs.pop("expert_num", 1))
        nn.Module.__init__(self)
        self.base_layer = base_layer
        MOELoraLayer.__init__(
            self,
            base_layer=base_layer,
            in_features=in_features,
            out_features=out_features,
            expert_num=self.expert_num,
        )
        if hasattr(base_layer, "weight"):
            base_layer.weight.requires_grad = False
        if getattr(base_layer, "bias", None) is not None:
            base_layer.bias.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.set_adapter(adapter_name)
        self._expert_weight_cache: torch.Tensor | None = None

    def merge(self, *args, **kwargs) -> None:
        warnings.warn("V-RoLoRA routing is sample-dependent; adapter merging is disabled.", stacklevel=2)

    def unmerge(self, *args, **kwargs) -> None:
        warnings.warn("V-RoLoRA routing is sample-dependent; adapter unmerging is disabled.", stacklevel=2)

    def _active_adapter_name(self) -> str:
        active = self.active_adapter
        return active[0] if isinstance(active, list) else active

    def _routing_weights(self, inputs: torch.Tensor, explicit: torch.Tensor | None) -> torch.Tensor:
        weights = explicit if explicit is not None else self._expert_weight_cache
        batch_size = inputs.shape[0]
        if weights is None:
            return torch.full(
                (batch_size, self.expert_num),
                1.0 / self.expert_num,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        weights = weights.to(device=inputs.device)
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        if weights.shape[-1] != self.expert_num:
            raise ValueError(
                f"Expected {self.expert_num} expert weights, received {weights.shape[-1]}"
            )
        if weights.shape[0] == 1 and batch_size > 1:
            weights = weights.expand(batch_size, -1)
        elif weights.shape[0] != batch_size:
            if batch_size % weights.shape[0] != 0:
                raise ValueError(
                    f"Routing batch size {weights.shape[0]} cannot be expanded to {batch_size}"
                )
            weights = weights.repeat_interleave(batch_size // weights.shape[0], dim=0)
        return weights

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        previous_dtype = inputs.dtype
        result = self.base_layer(inputs)
        active = self._active_adapter_name()
        if self.disable_adapters or active not in self.lora_A or self.r[active] <= 0:
            return result.to(previous_dtype)

        weights = self._routing_weights(inputs, kwargs.get("expert_weight"))
        adapter_dtype = self.lora_A[active].loraA[0].weight.dtype
        adapter_inputs = self.lora_dropout[active](inputs.to(adapter_dtype))
        for expert_idx in range(self.expert_num):
            hidden = self.lora_A[active].loraA[expert_idx](adapter_inputs)
            update = self.lora_B[active].loraB[expert_idx](hidden)
            shape = [weights.shape[0]] + [1] * (update.ndim - 1)
            mixture_weight = weights[:, expert_idx].view(*shape).to(update.dtype)
            result = result + update * mixture_weight * self.scaling[active]
        return result.to(previous_dtype)
