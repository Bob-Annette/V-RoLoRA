#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=2 -m vrolora.cli.train_sft \
  --model_name_or_path Qwen/Qwen3-8B \
  --train_file data/valueeval/train.jsonl \
  --validation_file data/valueeval/validation.jsonl \
  --condition_dim 10 \
  --expert_num 8 \
  --projection_dim 64 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --max_source_length 400 \
  --max_target_length 200 \
  --output_dir outputs/valueeval/sft \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --max_steps 3000 \
  --learning_rate 9e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --weight_decay 0.1 \
  --logging_steps 10 \
  --save_steps 1000 \
  --bf16 true \
  --remove_unused_columns false \
  --deepspeed configs/deepspeed_zero3.json \
  --report_to none
