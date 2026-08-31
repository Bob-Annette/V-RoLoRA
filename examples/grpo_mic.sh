#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=2 -m vrolora.cli.train_grpo \
  --model_name_or_path Qwen/Qwen3-8B \
  --peft_path outputs/mic/sft \
  --train_file data/mic/train.jsonl \
  --condition_dim 6 \
  --expert_num 8 \
  --projection_dim 64 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --verifier_model_name_or_path Qwen/Qwen3-0.6B \
  --verifier_adapter_path models/mic-verifier \
  --reward_task mic \
  --format_weight 0 \
  --stance_weight 0 \
  --output_dir outputs/mic/grpo \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_generations 8 \
  --max_prompt_length 200 \
  --max_completion_length 200 \
  --temperature 1.0 \
  --top_p 1.0 \
  --beta 0.0 \
  --num_iterations 1 \
  --epsilon 0.2 \
  --learning_rate 9e-6 \
  --max_steps 300 \
  --logging_steps 10 \
  --save_steps 100 \
  --bf16 true \
  --remove_unused_columns false \
  --deepspeed configs/deepspeed_zero3.json \
  --report_to none
