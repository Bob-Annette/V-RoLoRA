#!/usr/bin/env bash
set -euo pipefail

python -m vrolora.cli.train_verifier \
  --model_name_or_path Qwen/Qwen3-0.6B-Base \
  --task value \
  --train_file data/valueeval/train.jsonl \
  --validation_file data/valueeval/validation.jsonl \
  --output_dir models/value-verifier \
  --max_length 768 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_steps 200 \
  --logging_steps 50 \
  --gradient_checkpointing true \
  --bf16 true \
  --report_to none
