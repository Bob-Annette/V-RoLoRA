"""Compatibility hooks for serializing a PEFT extension adapter."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from functools import wraps
from typing import Any

import peft
import peft.peft_model as peft_model_module
import peft.utils as peft_utils
import peft.utils.save_and_load as save_and_load_module
from peft.utils.peft_types import PeftType

PEFT_TYPE = "MOELORA"
_PATCH_MARKER = "_vrolora_serialization_compat"


def _with_lora_serialization(original: Callable[..., Any]) -> Callable[..., Any]:
    """Route MOELORA state dictionaries through PEFT's LoRA serializer."""

    @wraps(original)
    def wrapped(model, *args, **kwargs):
        adapter_name = kwargs.get("adapter_name", args[1] if len(args) > 1 else "default")
        config = model.peft_config[adapter_name]
        if config.peft_type != PEFT_TYPE:
            return original(model, *args, **kwargs)

        compatibility_config = copy(config)
        compatibility_config.peft_type = PeftType.LORA
        model.peft_config[adapter_name] = compatibility_config
        try:
            return original(model, *args, **kwargs)
        finally:
            model.peft_config[adapter_name] = config

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def install_serialization_compat() -> None:
    """Install idempotent PEFT save/load hooks for the registered adapter."""

    current_get = peft_model_module.get_peft_model_state_dict
    current_set = peft_model_module.set_peft_model_state_dict
    if getattr(current_get, _PATCH_MARKER, False):
        return

    wrapped_get = _with_lora_serialization(current_get)
    wrapped_set = _with_lora_serialization(current_set)

    peft_model_module.get_peft_model_state_dict = wrapped_get
    peft_model_module.set_peft_model_state_dict = wrapped_set
    save_and_load_module.get_peft_model_state_dict = wrapped_get
    save_and_load_module.set_peft_model_state_dict = wrapped_set
    peft_utils.get_peft_model_state_dict = wrapped_get
    peft_utils.set_peft_model_state_dict = wrapped_set
    peft.get_peft_model_state_dict = wrapped_get
    peft.set_peft_model_state_dict = wrapped_set
