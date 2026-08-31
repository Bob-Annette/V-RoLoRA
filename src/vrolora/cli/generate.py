"""Generate text from a V-RoLoRA checkpoint and a condition vector."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import vrolora  # noqa: F401 -- registers the custom PEFT adapter before loading.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--condition", required=True, help="Comma-separated binary condition vector.")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        dtype="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path).to(args.device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    condition = torch.tensor(
        [[float(item.strip()) for item in args.condition.split(",")]],
        device=args.device,
    )
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            value_ids=condition,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = generated[0, encoded["input_ids"].shape[1] :]
    print(tokenizer.decode(completion, skip_special_tokens=True))


if __name__ == "__main__":
    main()
