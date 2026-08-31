"""Configuration for the V-RoLoRA PEFT adapter."""

from dataclasses import dataclass, field

from peft import LoraConfig


@dataclass
class MOELoraConfig(LoraConfig):
    """Configuration for value-conditioned routed LoRA.

    ``task_num`` is the width of the discrete condition vector. The historical
    name is retained for checkpoint compatibility.
    """

    task_num: int = field(default=2, metadata={"help": "Number of condition dimensions."})
    task_embedding_dim: int = field(default=64, metadata={"help": "Router projection width."})
    expert_num: int = field(default=4, metadata={"help": "Number of LoRA experts."})
    projection_std: float = field(default=0.01, metadata={"help": "Gaussian projection standard deviation."})
    projection_sparsity: float = field(
        default=0.5,
        metadata={"help": "Fraction of entries set to zero in the fixed projection."},
    )
    freeze_projection: bool = field(default=True, metadata={"help": "Keep the condition projection fixed."})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.task_num <= 0:
            raise ValueError("task_num must be positive")
        if self.task_embedding_dim <= 0:
            raise ValueError("task_embedding_dim must be positive")
        if self.expert_num <= 0:
            raise ValueError("expert_num must be positive")
        if self.r <= 0 or self.r % self.expert_num != 0:
            raise ValueError("r must be positive and divisible by expert_num")
        if not 0.0 <= self.projection_sparsity < 1.0:
            raise ValueError("projection_sparsity must be in [0, 1)")
        self.peft_type = "MOELORA"
