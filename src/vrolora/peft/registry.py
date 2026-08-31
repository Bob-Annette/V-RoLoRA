"""Runtime registration of V-RoLoRA with an unmodified PEFT installation."""

from peft.mapping import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
    PEFT_TYPE_TO_PREFIX_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
)

from .compat import install_serialization_compat
from .config import MOELoraConfig
from .model import MOELoraModel

PEFT_TYPE = "MOELORA"


def register_moelora() -> None:
    """Register V-RoLoRA without vendoring or modifying the PEFT package."""

    install_serialization_compat()
    existing = PEFT_TYPE_TO_TUNER_MAPPING.get(PEFT_TYPE)
    if existing is not None:
        if existing is not MOELoraModel:
            raise RuntimeError("A different PEFT tuner is already registered as MOELORA")
        return

    PEFT_TYPE_TO_CONFIG_MAPPING[PEFT_TYPE] = MOELoraConfig
    PEFT_TYPE_TO_TUNER_MAPPING[PEFT_TYPE] = MOELoraModel
    PEFT_TYPE_TO_MIXED_MODEL_MAPPING[PEFT_TYPE] = MOELoraModel
    PEFT_TYPE_TO_PREFIX_MAPPING[PEFT_TYPE] = "lora_"
