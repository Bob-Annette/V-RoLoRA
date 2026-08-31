"""Custom GRPO trainer with MoELoRA support."""

from __future__ import annotations

import copy
import re
from contextlib import nullcontext
from typing import Any

import torch
from accelerate.utils import broadcast_object_list, is_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import (
    gather_object,
    is_conversational,
    is_flash_attn_2_available,
    maybe_apply_chat_template,
    pad,
    prepare_multimodal_messages,
    profiling_context,
    truncate_with_protected_tokens,
    unwrap_model_for_generation,
)
from trl.trainer.utils import entropy_from_logits, nanmax, nanmin, nanstd, selective_log_softmax


class CustomGRPOTrainer(GRPOTrainer):
    """Extend ``trl.GRPOTrainer`` to forward V-RoLoRA routing hints.

    The default GRPO trainer ignores extra routing keys such as ``expert_weight``,
    ``task_weight`` and ``task_id`` that are required by the MoELoRA adapter. This
    subclass keeps the original logic while making sure those values are propagated
    through generation and log-probability computation steps.
    """

    _MOELORA_KEYS = ("expert_weight", "task_weight", "task_id", "value_ids")

    def _extract_moelora_kwargs(self, inputs: list[dict[str, Any]] | dict[str, Any]):
        """Collect MoELoRA-specific kwargs from the given inputs."""

        if isinstance(inputs, dict):
            iterable = [inputs]
        else:
            iterable = inputs

        kwargs: dict[str, torch.Tensor] = {}
        if not iterable:
            return kwargs

        for key in self._MOELORA_KEYS:
            if key in iterable[0]:
                values = []
                for example in iterable:
                    value = example[key]
                    if not torch.is_tensor(value):
                        value = torch.tensor(value)
                    values.append(value)
                kwargs[key] = torch.stack(values)
        return kwargs

    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
        **moelora_kwargs,
    ):
        # Prime V-RoLoRA caches before bypassing the PEFT wrapper for hidden states.
        if is_peft_model(unwrapped_model):
            cache_setter = getattr(unwrapped_model, "_set_cache_on_layers", None)
            if cache_setter is not None:
                cache_setter(**moelora_kwargs)
            target_model = unwrapped_model.base_model.model
        else:
            target_model = getattr(unwrapped_model, "model", unwrapped_model)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False

        # last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        last_hidden_state = target_model(**model_inputs).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]
        return last_hidden_state

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
        **moelora_kwargs,
    ):
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

            if image_grid_thw is not None and pixel_values is not None:
                model_inputs["image_grid_thw"] = image_grid_thw[start : start + batch_size]
                start_pixel_idx = image_grid_thw[:start].prod(-1).sum().item()
                end_pixel_idx = image_grid_thw[: start + batch_size].prod(-1).sum().item()
                model_inputs["pixel_values"] = pixel_values[start_pixel_idx:end_pixel_idx]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]

            for key, value in moelora_kwargs.items():
                model_inputs[key] = value[start : start + batch_size]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        if self.use_vllm:
            raise NotImplementedError(
                "V-RoLoRA uses per-sample dynamic routing and does not support vLLM generation"
            )
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]

        # Extract MoELoRA routing hints early so they can be forwarded during generation
        moelora_kwargs = {k: v.to(device) for k, v in self._extract_moelora_kwargs(inputs).items()}

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=1)

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        # Call the ``Trainer`` implementation directly to avoid re-entering this
        # class's ``_prepare_inputs`` (which would trigger another round of GRPO
        # generation and recurse).
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_inputs.update(moelora_kwargs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            # The chat template sometimes inserts a single image token into the prompt text. However, when this text is
            # later tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original chat template in case it originally
            # applies it. Otherwise, it assumes that the chat template uses only vision_start_token_id to indicate images
            # (e.g. Gemma 3) and removes all image_token instances and vision_end_token_id as well, leaving only
            # the vision_start_token_id (e.g. <start_of_image>).
            if self.image_token is not None:
                escaped_img_token = re.escape(self.image_token)
                # Search for the image token in the chat template
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    prompts_text = [re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text]
                else:
                    # If the chat template doesn't use the image token, we remove all instances of it + vision_end_token_id
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(self.processing_class.tokenizer.decode([self.vision_end_token_id]))
                        prompts_text = [re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text]
                    else:
                        # If vision_end_token_id is None, just remove the image tokens
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            from vllm import SamplingParams
            from vllm.sampling_params import GuidedDecodingParams

            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                completion_ids, all_logprobs = obj_list[0]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
                all_logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]
                    all_logprobs = all_logprobs[tp_slice]

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            sampling_per_token_logps = [
                torch.tensor(logprobs, device=device, dtype=torch.float32) for logprobs in all_logprobs
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # In some rare cases, the model may generate a single token (e.g., when the prompt already contains EOS)
            if completion_ids.numel() == 0:
                completion_ids = torch.full((prompt_ids.size(0), 1), self.eos_token_id, device=device, dtype=torch.long)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            # Compute per-token log probabilities of the generated tokens. Only meaningful if generating using transformers
            # directly in the GRPOTrainer.
            with (
                profiling_context(self, "transformers.logp"),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
                torch.no_grad(),
            ):
                # Compute the per-token log probabilities. Not needed for vLLM because it already does it, and not
                # possible for paged attention because paged attention does not return the full hidden states.
                sampling_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    prompt_completion_ids.ne(self.pad_token_id),
                    completion_ids.size(1),
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                    **moelora_kwargs,
                )

        # Compute the importance sampling correction (if needed), and calculate advantage estimates
        num_items_in_batch = prompt_ids.size(0)
        completion_mask = completion_ids.ne(self.pad_token_id)
        batch_size = prompt_inputs["input_ids"].size(0)

        # Preserve the local tensors for forward/backward. We only use gathered copies for
        # reward computation and logging to avoid duplicating gradients across ranks.
        local_prompt_completion_ids = prompt_completion_ids
        local_completion_ids = completion_ids
        local_completion_mask = completion_mask
        local_attention_mask = local_prompt_completion_ids.ne(self.pad_token_id)
        local_logits_to_keep = local_completion_ids.size(1)

        prompt_completion_ids = self.accelerator.pad_across_processes(
            prompt_completion_ids, dim=1, pad_index=self.pad_token_id
        )
        completion_ids = self.accelerator.pad_across_processes(completion_ids, dim=1, pad_index=self.pad_token_id)
        completion_mask = self.accelerator.pad_across_processes(completion_mask, dim=1, pad_index=0)
        prompt_completion_ids = self.accelerator.gather(prompt_completion_ids)
        completion_ids = self.accelerator.gather(completion_ids)
        if isinstance(completion_mask, torch.Tensor):
            completion_mask = completion_mask.long()
        completion_mask = self.accelerator.gather(completion_mask)
        gathered_logits_to_keep = completion_ids.size(1)
        # Add batch dimension for the number of generations
        completion_ids_list = completion_ids
        num_items_in_batch = torch.tensor(num_items_in_batch, device=device)
        num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

        # Compute the completion lengths and masks
        completion_lengths = torch.tensor(
            [sum((completion != self.pad_token_id).int()) for completion in completion_ids],
            device=device,
        )
        agg_completion_lengths = self.accelerator.gather(completion_lengths)

        # Compute completion endings (i.e., whether the completion ended with EOS). last_non_pad will be equivalent to last non-zero
        last_non_pad = (completion_mask.cumsum(dim=1).argmax(dim=1) - 1).clamp(min=0)
        is_eos = completion_ids.gather(1, last_non_pad.unsqueeze(1)) == self.eos_token_id

        # Re-create prompt and completion ids in their original shape (ungrouped) for decoding later
        # These tensors do not contain pixel grids, so no unsplitting is needed.

        if self.use_vllm:
            # When vLLM is used, sampling_per_token_logps are directly obtained during generation
            sampling_per_token_logps = sampling_per_token_logps.to(self.accelerator.device)
            completion_mask = completion_mask.to(self.accelerator.device)
            completion_lengths = completion_lengths.to(self.accelerator.device)
            prompt_completion_ids = prompt_completion_ids.to(self.accelerator.device)

        gathered_completion_ids = completion_ids_list[..., -gathered_logits_to_keep:]
        completion_ids = gathered_completion_ids
        completion_mask = completion_mask[..., -gathered_logits_to_keep:]

        # Use the local tensors for loss computation to keep each rank operating on its own data.
        prompt_completion_ids = local_prompt_completion_ids
        completion_ids = local_completion_ids
        completion_mask = local_completion_mask
        attention_mask = local_attention_mask
        logits_to_keep = local_logits_to_keep

        # Decode completions locally for logging and use the gathered versions later for rewards.
        local_completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute the per-token log probabilities for the model (only if not using transformers_paged)
        if self.use_transformers_paged:
            old_per_token_logps = None
        else:
            if self.use_vllm:
                # sampling_per_token_logps is already computed during vLLM generation
                with torch.no_grad():
                    old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                        **moelora_kwargs,
                    )
            else:
                old_per_token_logps = sampling_per_token_logps.detach()

        # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
        if self.use_vllm and self.vllm_importance_sampling_correction:
            importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
            importance_sampling_ratio = torch.clamp(importance_sampling_ratio, max=self.vllm_importance_sampling_cap)

        # Compute the per-token log probabilities for the reference model
        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                    **moelora_kwargs,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                        **moelora_kwargs,
                    )
        else:
            ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Gather prompts and inputs across processes so reward functions receive tensors that match the gathered completions.
        local_batch_size = len(prompts)
        gathered_prompts = gather_object(prompts)
        gathered_original_prompts = gather_object(original_prompts)
        gathered_inputs = gather_object(inputs)

        completions_text = self.processing_class.batch_decode(gathered_completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(gathered_prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            gathered_inputs, gathered_original_prompts, completions, completion_ids_list
        )

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * local_batch_size,
            (self.accelerator.process_index + 1) * local_batch_size,
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(local_completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if has_images:
            self._logs["image"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(self.accelerator.gather(mean_delta).mean().item())
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(self.accelerator.gather(max_delta).max().item())

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            mean_importance_sampling_ratio = torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            max_importance_sampling_ratio = torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item())
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item())
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            **moelora_kwargs,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        return output

    def _compute_loss(self, model, inputs):
        moelora_kwargs = {key: inputs[key] for key in self._MOELORA_KEYS if key in inputs}

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            **moelora_kwargs,
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        else:
            old_per_token_logps = old_per_token_logps.detach()

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def compute_liger_loss(self, unwrapped_model, inputs):
        moelora_kwargs = {key: inputs[key] for key in self._MOELORA_KEYS if key in inputs}

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
            **moelora_kwargs,
        )

        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )

        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps
