"""Supervised cold-start training for V-RoLoRA."""

from functools import partial

from datasets import load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

from vrolora.cli.common import (
    DataArguments,
    ModelArguments,
    configure_utf8_stdout,
    load_policy,
    load_tokenizer,
)
from vrolora.data import ConditionedCausalCollator, tokenize_sft_batch


def main() -> None:
    configure_utf8_stdout()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    tokenizer = load_tokenizer(model_args)
    tokenizer.padding_side = "right"
    model = load_policy(model_args, is_trainable=True)
    model.print_trainable_parameters()

    data_files = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file
    raw = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    preprocess = partial(
        tokenize_sft_batch,
        tokenizer=tokenizer,
        prompt_column=data_args.prompt_column,
        response_column=data_args.response_column,
        condition_column=data_args.condition_column,
        history_column=data_args.history_column,
        prefix=data_args.source_prefix,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
    )
    processed = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing condition-controlled SFT data",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed.get("validation"),
        data_collator=ConditionedCausalCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
