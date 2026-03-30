from typing import Literal
import random
import torch
from transformers import GenerationConfig, LogitsProcessorList, NoBadWordsLogitsProcessor, Pipeline

from rewardbench.models.basic_sft_model.collator import (
    SimpleDataCollatorForPreference,
    tokenize_example,
    _REWARDBENCH_JUDGE_SYSTEM,
    _rewardbench_pair_user,
    # process_example,
)


def process_example_rewardbench(
    example,
    add_thinking: bool = False,
    thinking_key: str = "global_analysis",
):

    chosen = example["text_chosen"]
    rejected = example["text_rejected"]
    prompt = chosen[: len(chosen) - 1]
    if prompt != rejected[: len(rejected) - 1]:
        raise ValueError("Prompt input IDs do not match the rejected input IDs")
    chosen = chosen[-1]["content"]
    rejected = rejected[-1]["content"]
    question = prompt[-1]["content"]
    result = {}
    result["overall"] = [
        {"role": "system", "content": _REWARDBENCH_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _rewardbench_pair_user(question, chosen, rejected),
        },
        {"role": "assistant", "content": "A"},
    ]
    # Must match sft_stage1/data_utils/basic_sft.py `process_example`: swap A/B blocks so the
    # better answer maps to "B" when the assistant ends with "B".
    result["overall_reversed"] = [
        {"role": "system", "content": _REWARDBENCH_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _rewardbench_pair_user(question, rejected, chosen),
        },
        {"role": "assistant", "content": "B"},
    ]
    if add_thinking:
        result["overall"][-1]["reasoning_content"] = example[thinking_key]
        result["overall_reversed"][-1]["reasoning_content"] = example[thinking_key]
    return result


def process_example_with_template_rewardbench(
    example,
    add_thinking: bool = False,
    thinking_key: str = "global_analysis",
    first_placeholder: str = "A",
    second_placeholder: str = "B",
):
    chosen = example["text_chosen"]
    rejected = example["text_rejected"]
    prompt = chosen[: len(chosen) - 1]
    if prompt != rejected[: len(rejected) - 1]:
        raise ValueError("Prompt input IDs do not match the rejected input IDs")
    chosen = chosen[-1]["content"]
    rejected = rejected[-1]["content"]
    question = prompt[-1]["content"]
    result = {}
    overall = [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": chosen,
            "identifier": first_placeholder,
        },
        {
            "role": "assistant",
            "content": rejected,
            "identifier": second_placeholder,
        },
        {"role": "judge", "content": first_placeholder},
    ]

    overall_reversed = [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": rejected,
            "identifier": first_placeholder,
        },
        {
            "role": "assistant",
            "content": chosen,
            "identifier": second_placeholder,
        },
        {"role": "judge", "content": second_placeholder},
    ]

    if add_thinking:
        result["overall"][-1]["reasoning_content"] = example[thinking_key]
        result["overall_reversed"][-1]["reasoning_content"] = example[thinking_key]

    result["overall"] = overall
    result["overall_reversed"] = overall_reversed
    result["prompt"] = [
        {"role": "user", "content": example["prompt"]},
    ]

    return result


class BasicSFTJudgePipeline(Pipeline):
    """
    This class outputs a delta rather than a score for each.
    """

    def __init__(
        self,
        paradigm: Literal["generative", "discriminative"],
        enable_thinking: bool,
        advanced_template: bool = False,
        max_length: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model.eval().requires_grad_(False)
        self.enable_thinking = enable_thinking
        self.paradigm = paradigm
        self.advanced_template = advanced_template
        self.max_length = max_length
        self.eot_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.bot_id = self.tokenizer.convert_tokens_to_ids("<think>")

        self.judge_token = "<|judgement|>"
        self.judge_token_id = self.tokenizer.convert_tokens_to_ids(self.judge_token)

        self.first_choice_token = "A"
        self.second_choice_token = "B"

        self.first_choice_token_id = self.tokenizer.convert_tokens_to_ids(self.first_choice_token)
        self.second_choice_token_id = self.tokenizer.convert_tokens_to_ids(self.second_choice_token)

        self.collator = SimpleDataCollatorForPreference(
            tokenizer=self.tokenizer,
            paradigm=paradigm,
            add_thinking=False,
            judge_token=self.judge_token,
        )

    def _logits_processor_no_judge_while_generating(self) -> LogitsProcessorList:
        """Ban <|judgement|> during `generate`; judge slots are appended after generation."""
        return LogitsProcessorList(
            [
                NoBadWordsLogitsProcessor(
                    bad_words_ids=[[int(self.judge_token_id)]],
                    eos_token_id=self.eot_id,
                )
            ]
        )

    def _move_tensors(self, inputs: dict):
        for k in inputs.keys():
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)
                if v.dtype != torch.long:
                    inputs[k] = inputs[k].to(self.model.dtype)
        return inputs

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        generation_config = kwargs.get("generation_config", None)
        if generation_config is not None:
            forward_kwargs["generation_config"] = self.model.generation_config
        return preprocess_kwargs, forward_kwargs, {}

    def postprocess(self, outputs: dict):
        # we need to modify the following to support when batch size is >=1
        first_logits = outputs["judge_logits"][:, 0]
        second_logits = outputs["judge_logits"][:, 1]
        shuffle = outputs["shuffle"]
        rewards_chosen = torch.where(shuffle, second_logits, first_logits)
        rewards_rejected = torch.where(shuffle, first_logits, second_logits)
        results = (rewards_chosen > rewards_rejected).int()
        return {
            "results": results.tolist(),
            "rewards_chosen": rewards_chosen.tolist(),
            "rewards_rejected": rewards_rejected.tolist(),
        }

    def preprocess(self, inputs: list[dict]):
        all_results = []
        for _input in inputs:
            _input["global_analysis"] = "abracadabra"
            _func = (
                process_example_with_template_rewardbench
                if self.advanced_template
                else process_example_rewardbench
            )
            result = _func(
                example=_input,
                add_thinking=self.enable_thinking,
                thinking_key="global_analysis",
            )
            result = tokenize_example(result, self.tokenizer)
            if self.max_length is not None:
                if _input_len := len(result["input_ids"]) > self.max_length:
                    raise NotImplementedError(
                        f"Input length {_input_len} is greater than max length {self.max_length}, 目前不支持这种truncation"
                    )
            # remove every thing after <think>
            if self.enable_thinking:
                _position = result["input_ids"].index(self.bot_id)
                result["input_ids"] = result["input_ids"][: _position + 1]
                _position = result["reversed_input_ids"].index(self.bot_id)
                result["reversed_input_ids"] = result["reversed_input_ids"][: _position + 1]
            result["shuffle"] = random.random() < 0.5
            all_results.append(result)

        all_results = self.collator(all_results)
        all_results = self._move_tensors(all_results)
        return all_results

    def _forward_with_thinking(self, model_inputs, generation_config: GenerationConfig | None = None):
        batch_size = model_inputs["input_ids"].shape[0]
        shuffle = torch.as_tensor(
            model_inputs["shuffles"], dtype=torch.bool, device=model_inputs["input_ids"].device
        ).view(batch_size)
        device = self.model.device
        gen_outputs = self.model.generate(
            **model_inputs,  # include the attention mask as well.
            use_cache=True,
            eos_token_id=self.eot_id,
            generation_config=generation_config,
            logits_processor=self._logits_processor_no_judge_while_generating(),
            return_dict_in_generate=True,
        )
        final_kv = gen_outputs.past_key_values
        new_content = "\n\n"
        if self.paradigm == "discriminative":
            new_content += self.judge_token * 2
        new_content_ids = self.tokenizer(new_content, return_tensors="pt").input_ids
        final_input_ids = gen_outputs.sequences
        # we will forward one-by-one for now

        judge_logits = []
        output_ids = []
        for i in range(batch_size):
            _input_ids = final_input_ids[i]
            _kv_cache = final_kv.copy()
            indices = torch.tensor([i], device=device, dtype=torch.long)
            _kv_cache.batch_select_indices(indices)
            # check where is </think> and remove the pad tokens after that
            _position = (_input_ids == self.eot_id).nonzero().view(-1)[-1].item()
            _input_ids = _input_ids[: _position + 1]
            # add the judge tokens
            _input_ids = torch.cat(
                [
                    _input_ids,
                    torch.tensor([self.judge_token_id, self.judge_token_id], device=_input_ids.device),
                ],
                dim=-1,
            )
            # add the new content
            _input_ids = torch.cat([_input_ids, new_content_ids], dim=-1)
            # forward
            _final_outputs = self.model.forward(
                input_ids=_input_ids, use_cache=True, past_key_values=_kv_cache
            )
            output_ids.append(_final_outputs.sequences[i])
            if self.paradigm == "discriminative":
                _judge_logits = _final_outputs.judge_logits.view(2)
            else:
                first_choice_token_logits = _final_outputs.logits[i, -1, self.first_choice_token_id]
                second_choice_token_logits = _final_outputs.logits[i, -1, self.second_choice_token_id]
                _judge_logits = torch.stack([first_choice_token_logits, second_choice_token_logits], dim=1)
            judge_logits.append(_judge_logits)

        return {
            "judge_logits": torch.stack(judge_logits, dim=0),
            "output_ids": output_ids,
            "shuffle": shuffle,
        }

    def _forward(self, model_inputs, generation_config: GenerationConfig | None = None):
        if not self.enable_thinking:
            return self._forward_no_think(model_inputs)
        else:
            return self._forward_with_thinking(model_inputs, generation_config)

    def _forward_no_think(
        self,
        model_inputs,
    ):
        B = model_inputs["input_ids"].shape[0]

        shuffle = model_inputs.pop("shuffles")
        shuffle = torch.as_tensor(shuffle, dtype=torch.bool, device=model_inputs["input_ids"].device).view(B)

        result = self.model.forward(**model_inputs)
        if self.paradigm == "discriminative":
            judge_logits = result.judge_logits.view(B, 2)
        else:
            batch_idx = torch.arange(B, device=model_inputs["input_ids"].device)
            eos_token_id = self.tokenizer.eos_token_id
            eos_mask = model_inputs["input_ids"] == eos_token_id
            seq_len = model_inputs["input_ids"].shape[1]
            pos = torch.arange(seq_len, device=model_inputs["input_ids"].device).unsqueeze(0).expand(B, -1)
            masked_pos = torch.where(eos_mask, pos, torch.full_like(pos, -1))
            last_eos_token = masked_pos.max(dim=1).values
            if (last_eos_token < 0).any():
                bad = (last_eos_token < 0).nonzero(as_tuple=True)[0]
                raise ValueError(f"No EOS token found for batch row(s): {bad.tolist()}.")
            # get the logits of A/B
            first_choice_token_logits = result.logits[
                batch_idx, last_eos_token - 2, self.first_choice_token_id
            ]  # (B,)
            second_choice_token_logits = result.logits[
                batch_idx, last_eos_token - 2, self.second_choice_token_id
            ]  # (B,)
            judge_logits = torch.stack(
                [first_choice_token_logits, second_choice_token_logits], dim=1
            )  # (B, 2)

        return {
            "judge_logits": judge_logits,
            "shuffle": shuffle,
        }
