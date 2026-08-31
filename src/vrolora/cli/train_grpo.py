"""RLVR post-training with GRPO and a condition discriminator."""

from dataclasses import dataclass, field
from functools import partial

from datasets import load_dataset
from transformers import HfArgumentParser, set_seed
from trl import GRPOConfig

from vrolora.cli.common import (
    DataArguments,
    ModelArguments,
    configure_utf8_stdout,
    load_policy,
    load_tokenizer,
)
from vrolora.data import format_grpo_batch
from vrolora.rewards import VerifierReward, VerifierRewardConfig
from vrolora.trainer_grpo import CustomGRPOTrainer


@dataclass
class RewardArguments:
    verifier_model_name_or_path: str = field(
        metadata={"help": "Hugging Face model ID or relative path for the condition discriminator."}
    )
    verifier_adapter_path: str | None = field(default=None)
    reward_task: str = field(default="value", metadata={"help": "value or mic"})
    verifier_device: str | None = field(default=None)
    verifier_threshold: float = field(default=0.5)
    consistency_weight: float = field(default=1.0)
    format_weight: float = field(default=1.0)
    stance_weight: float = field(default=1.0)


def main() -> None:
    configure_utf8_stdout()
    parser = HfArgumentParser((ModelArguments, DataArguments, RewardArguments, GRPOConfig))
    model_args, data_args, reward_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    tokenizer = load_tokenizer(model_args)
    tokenizer.padding_side = "left"
    model = load_policy(model_args, is_trainable=True)
    model.print_trainable_parameters()

    data_files = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
    raw = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    preprocess = partial(
        format_grpo_batch,
        prompt_column=data_args.prompt_column,
        response_column=data_args.response_column,
        condition_column=data_args.condition_column,
        history_column=data_args.history_column,
        prefix=data_args.source_prefix,
    )
    processed = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Formatting condition-controlled GRPO data",
    )

    reward = VerifierReward(
        VerifierRewardConfig(
            model_name_or_path=reward_args.verifier_model_name_or_path,
            adapter_path=reward_args.verifier_adapter_path,
            task=reward_args.reward_task,
            device=reward_args.verifier_device,
            threshold=reward_args.verifier_threshold,
            consistency_weight=reward_args.consistency_weight,
            format_weight=reward_args.format_weight,
            stance_weight=reward_args.stance_weight,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    )
    trainer = CustomGRPOTrainer(
        model=model,
        reward_funcs=[reward],
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed.get("validation"),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
