from typing import Any
import torch
from rewardbench.models.urm import Qwen3ForGenerativeRewarding


def apply_template(
    example,
    use_ver_role: bool = False,
    use_ver_token: bool = False,
    use_judge_role: bool = False,
    use_judge_token: bool = False,
    with_thinking: bool = False,
    chosen_id_placeholder_token: str | None = None,
    rejected_id_placeholder_token: str | None = None,
):
    chosen = example["text_chosen"]
    rejected = example["text_rejected"]

    if use_ver_token:
        assert use_ver_role, "use_ver_role must be True when use_ver_token is True"
    if use_judge_token:
        assert use_judge_role, "use_judge_role must be True when use_judge_token is True"
    if with_thinking:
        assert (
            use_ver_role or use_judge_role
        ), "use_ver_role or use_judge_role must be True when with_thinking is True"
    assert not (
        use_ver_role and use_judge_role
    ), "use_ver_role and use_judge_role cannot be True at the same time"
    prompt = chosen[: len(chosen) - 1]
    if prompt != rejected[: len(rejected) - 1]:
        raise ValueError("Prompt input IDs do not match the rejected input IDs")
    result = {}
    if use_ver_role:
        content = "" if not use_ver_token else "<|verification|>"
        chosen_thinking = None
        rejected_thinking = None
        if with_thinking:
            chosen_thinking = example["text_chosen_thinking"]
            rejected_thinking = example["text_rejected_thinking"]
            assert "<think>" not in chosen_thinking, "chosen_thinking should not contain <think>"
            assert "</think>" not in chosen_thinking, "chosen_thinking should not contain </think>"
            assert "<think>" not in rejected_thinking, "rejected_thinking should not contain <think>"
            assert "</think>" not in rejected_thinking, "rejected_thinking should not contain </think>"
        chosen += [{"role": "verifier", "content": content, "reasoning_content": chosen_thinking}]
        rejected += [{"role": "verifier", "content": content, "reasoning_content": rejected_thinking}]

    elif use_judge_role:
        assert (
            chosen_id_placeholder_token is not None and rejected_id_placeholder_token is not None
        ), "chosen_id_placeholder_token and rejected_id_placeholder_token are required when use_judge_role is True"
        assert use_judge_token, "use_judge_token must be True when use_judge_role is True"
        special_token = "<|judgement|>"
        content = special_token * 2
        rejected[-1]["identifier"] = rejected_id_placeholder_token
        chosen[-1]["identifier"] = chosen_id_placeholder_token
        overall = prompt + [chosen[-1]] + [rejected[-1]] + [{"role": "judge", "content": content}]
        overall_reversed = prompt + [rejected[-1]] + [chosen[-1]] + [{"role": "judge", "content": content}]
        if with_thinking:
            # TODO: the key is text_thinking
            overall[-1]["reasoning_content"] = example["text_thinking"]
            overall_reversed[-1]["reasoning_content"] = example["text_thinking"]
        result["overall"] = overall
        result["overall_reversed"] = overall_reversed
    result["chosen"] = chosen
    result["rejected"] = rejected
    result["prompt"] = prompt
    return result


def apply_template_multiple(
    example,
    use_ver_role: bool = False,
    use_ver_token: bool = False,
    use_judge_role: bool = False,
    use_judge_token: bool = False,
    with_thinking: bool = False,
    chosen_id_placeholder_token: str | None = None,
    rejected_id_placeholder_token: str | None = None,
):
    chosens = example["texts_chosen"]
    rejecteds = example["texts_rejected"]
    prompt = chosens[0][: len(chosens[0]) - 1]

    if use_ver_token:
        assert use_ver_role, "use_ver_role must be True when use_ver_token is True"
    if use_judge_token:
        assert use_judge_role, "use_judge_role must be True when use_judge_token is True"
    if with_thinking:
        assert (
            use_ver_role or use_judge_role
        ), "use_ver_role or use_judge_role must be True when with_thinking is True"
    assert not (
        use_ver_role and use_judge_role
    ), "use_ver_role and use_judge_role cannot be True at the same time"

    for chosen in chosens:
        if prompt != chosen[: len(chosen) - 1]:
            raise ValueError("Prompt input IDs do not match the chosen input IDs")
    for rejected in rejecteds:
        if prompt != rejected[: len(rejected) - 1]:
            raise ValueError("Prompt input IDs do not match the rejected input IDs")

    result = {}
    if use_ver_role:
        content = "" if not use_ver_token else "<|verification|>"
        chosen_thinkings = [None] * len(chosens)
        rejected_thinkings = [None] * len(rejecteds)
        if with_thinking:
            chosen_thinkings = example["text_chosen_thinking"]
            rejected_thinkings = example["text_rejected_thinking"]
            for i in len(chosen_thinkings):
                for s in ["<think>", "</think>"]:
                    if s in chosen_thinkings[i]:
                        raise ValueError(f"found {s} in chosen_thinking: {chosen_thinkings[i]}")
            for i in len(rejected_thinkings):
                for s in ["<think>", "</think>"]:
                    if s in rejected_thinkings[i]:
                        raise ValueError(f"found {s} in rejected_thinking: {rejected_thinkings[i]}")

        for i in len(chosens):
            chosens[i] = chosens[i] + [
                {"role": "verifier", "content": content, "reasoning_content": chosen_thinkings[i]}
            ]
        for i in len(rejecteds):
            rejecteds[i] = rejecteds[i] + [
                {"role": "verifier", "content": content, "reasoning_content": rejected_thinkings[i]}
            ]
    elif use_judge_role:
        assert (
            chosen_id_placeholder_token is not None and rejected_id_placeholder_token is not None
        ), "chosen_id_placeholder_token and rejected_id_placeholder_token are required when use_judge_role is True"
        assert use_judge_token, "use_judge_token must be True when use_judge_role is True"
        special_token = "<|judgement|>"
        content = special_token * 2
        for i in len(chosens):
            chosens[i][-1]["identifier"] = chosen_id_placeholder_token
        for i in len(rejecteds):
            rejecteds[i][-1]["identifier"] = rejected_id_placeholder_token
        overall = (
            prompt
            + [chosen[-1] for chosen in chosens]
            + [rejected[-1] for rejected in rejecteds]
            + [{"role": "judge", "content": content}]
        )
        if with_thinking:
            # TODO: the key is text_thinking
            overall[-1]["reasoning_content"] = example["text_thinking"]
        result["overall"] = overall
    result["chosen"] = chosens
    result["rejected"] = rejecteds
    result["prompt"] = prompt
    return result


def formatting_fn(
    example,
    processing_class,
    thinking_prompt: bool = True,
    **chat_template_kwargs,
):
    """
    This is used for formatting the input for vllm thinking generation (without tokenization)
    """
    chosen = example["text_chosen"]
    rejected = example["text_rejected"]
    prompt = chosen[: len(chosen) - 1]
    if prompt != rejected[: len(rejected) - 1]:
        raise ValueError("Prompt input IDs do not match the rejected input IDs")
    chosen_input = processing_class.apply_chat_template(
        chosen,
        tokenize=False,
        add_generation_prompt=True,
        generation_prompt_role="verifier",
        **chat_template_kwargs,
    )
    rejected_input = processing_class.apply_chat_template(
        rejected,
        tokenize=False,
        add_generation_prompt=True,
        generation_prompt_role="verifier",
        **chat_template_kwargs,
    )
    prompt_input = processing_class.apply_chat_template(
        prompt,
        tokenize=False,
        **chat_template_kwargs,
    )
    if thinking_prompt:
        thinking_prompt = "<think>\n"
        chosen_input += thinking_prompt
        rejected_input += thinking_prompt

    _len = len(prompt_input)
    if not rejected_input[:_len] == prompt_input:
        raise ValueError("Prompt input IDs do not match the rejected input IDs")
    if not chosen_input[:_len] == prompt_input:
        raise ValueError("Prompt input IDs do not match the chosen input IDs")

    output = {
        "chosen_input": chosen_input,
        "rejected_input": rejected_input,
        "prompt_input": prompt_input,
    }
    return output


class ParallelRMRewardBenchPipeline:
    """
    This class outputs a delta rather than a score for each.
    """

    def __init__(
        self,
        task,
        model,
        tokenizer,
        standard: bool = False,
        parallel_context: bool = True,
        use_ver_role: bool = False,
        use_ver_token: bool = False,
        use_judge_role: bool = False,
        use_judge_token: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval().requires_grad_(False)
        self.standard = standard
        self.parallel_context = parallel_context

        self.use_ver_role = use_ver_role
        self.use_ver_token = use_ver_token

        self.use_judge_role = use_judge_role
        self.use_judge_token = use_judge_token

        self.use_scorer = isinstance(self.model, Qwen3ForGenerativeRewarding)

    def calculate_verifier_rewards(
        self,
        inputs: dict[str, torch.Tensor | Any],
        outputs,
        shuffled: torch.Tensor | None = None,
    ):
        B = inputs["input_ids"].shape[0]
        batch_idx = torch.arange(B, device=inputs["input_ids"].device)

        if self.use_scorer:
            rewards_first = outputs.verify_logits[:, 0].view(B)
            rewards_second = outputs.verify_logits[:, 1].view(B)
        else:
            # each sequence is [prompt | comp1 | comp2] with 3 EOS (prompt end, comp1 end, comp2 end).
            # shuffled[i] True => order was [prompt, rejected, chosen], so comp1=rejected, comp2=chosen.
            if self.use_ver_token:
                eos_token_id = self.model.config.task_specific_params["ver_token_id"]
            else:
                eos_token_id = self.tokenizer.eos_token_id

            eos_nonzero = (inputs["input_ids"] == eos_token_id).nonzero(as_tuple=True)
            eos_seq_idx = eos_nonzero[1]  # (3*B,)
            if self.use_ver_role:
                if self.use_ver_token:
                    num_eos = 2  # chosen and rejected
                    idx1, idx2 = 0, 1
                else:
                    num_eos = 5  # prompt, chosen, chosen verifier, rejected, rejected verifier
                    idx1, idx2 = 2, 4
            else:
                num_eos = 3  # prompt, chosen, rejected
                idx1, idx2 = 1, 2
            if len(eos_seq_idx) != num_eos * B:
                raise ValueError(
                    f"Expected {num_eos} EOS per sequence, got {len(eos_seq_idx)} for batch size {B}."
                )

            # output.logits: [B, seq_len, 1]; take reward at 2nd and 3rd EOS per sequence
            logits = outputs.logits.squeeze(-1)  # (B, seq_len)
            seq_2d = eos_seq_idx.reshape(B, num_eos)  # [b, 0/1/2] = prompt end, first comp, second comp

            rewards_first = logits[batch_idx, seq_2d[:, idx1]]
            rewards_second = logits[batch_idx, seq_2d[:, idx2]]  # (B,)

            # shuffled[i] True => first=rejected, second=chosen; False => first=chosen, second=rejected
            rewards_chosen = torch.where(shuffled, rewards_second, rewards_first)
            rewards_rejected = torch.where(shuffled, rewards_first, rewards_second)
        return rewards_chosen, rewards_rejected

    def calculate_judge_rewards(
        self,
        inputs: dict[str, torch.Tensor | Any],
        outputs,
        shuffled: torch.Tensor | None = None,
    ):
        B = inputs["input_ids"].shape[0]
        judge_logits = outputs.judge_logits  # (B, 2)
        rewards_first = judge_logits[:, 0].view(B)
        rewards_second = judge_logits[:, 1].view(B)

        rewards_chosen = torch.where(shuffled, rewards_second, rewards_first)
        rewards_rejected = torch.where(shuffled, rewards_first, rewards_second)
        return rewards_chosen, rewards_rejected

    def __call__(self, inputs: dict):
        B = inputs["input_ids"].shape[0]
        if "shuffled" in inputs:
            shuffled = inputs.pop("shuffled")
            shuffled = torch.as_tensor(
                shuffled, dtype=torch.bool, device=inputs["input_ids"].device
            ).flatten()
        for k in inputs.keys():
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                # convert if is float (not long)
                inputs[k] = v.to(self.model.device)
                if v.dtype != torch.long:
                    inputs[k] = inputs[k].to(self.model.dtype)
        outputs = self.model(**inputs)

        if self.standard:
            B = B // 2
            rewards_chosen, rewards_rejected = torch.chunk(outputs.logits.squeeze(-1), chunks=2)
        elif self.use_judge_role:
            rewards_chosen, rewards_rejected = self.calculate_judge_rewards(inputs, outputs, shuffled)
        else:
            rewards_chosen, rewards_rejected = self.calculate_verifier_rewards(inputs, outputs, shuffled)

        result = (rewards_chosen > rewards_rejected).view(B).int().tolist()
        rewards_chosen = rewards_chosen.view(B).tolist()
        rewards_rejected = rewards_rejected.view(B).tolist()
        return {
            "results": result,
            "rewards_chosen": rewards_chosen,
            "rewards_rejected": rewards_rejected,
        }


class ParallelRMRewardBenchMultiplePipeline:
    def __init__(
        self,
        task,
        model,
        tokenizer,
        standard: bool = False,
        parallel_context: bool = True,
        use_ver_role: bool = False,
        use_ver_token: bool = False,
        use_judge_role: bool = False,
        use_judge_token: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval().requires_grad_(False)
        self.standard = standard
        self.parallel_context = parallel_context

        self.use_ver_role = use_ver_role
        self.use_ver_token = use_ver_token

        self.use_judge_role = use_judge_role
        self.use_judge_token = use_judge_token

        self.use_scorer = isinstance(self.model, Qwen3ForGenerativeRewarding)

    def calculate_verifier_rewards(
        self,
        inputs: dict[str, torch.Tensor | Any],
        outputs,
        num_candidates: int,
    ):
        B = inputs["input_ids"].shape[0]
        batch_idx = torch.arange(B, device=inputs["input_ids"].device)

        if self.use_scorer:
            rewards = outputs.verify_logits.view(B, num_candidates)
        else:
            # each sequence is [prompt | comp1 | comp2 | ... | compN] with N+1 EOS (prompt end, comp1 end, comp2 end, ... , compN end).
            if self.use_ver_token:
                eos_token_id = self.model.config.task_specific_params["ver_token_id"]
            else:
                eos_token_id = self.tokenizer.eos_token_id

            eos_nonzero = (inputs["input_ids"] == eos_token_id).nonzero(as_tuple=True)
            eos_seq_idx = eos_nonzero[1]  # (3*B,)
            dev = inputs["input_ids"].device
            if self.use_ver_role:
                if self.use_ver_token:
                    num_eos = num_candidates  # chosen and rejected
                    idx_eos = torch.arange(num_candidates, device=dev)
                else:
                    num_eos = 1 + num_candidates * 2  # prompt, comp1 ver1, comp2 ver2, ... , compN verN

                    idx_eos = torch.arange(2, 1 + num_candidates * 2, 2, device=dev)
            else:
                num_eos = 1 + num_candidates  # prompt, comp1, comp2, ... , compN
                idx_eos = torch.arange(1, 1 + num_candidates, device=dev)
            if eos_seq_idx.shape[0] != num_eos * B:
                raise ValueError(
                    f"Expected {num_eos} EOS per sequence, got {eos_seq_idx.shape[0]} for batch size {B}."
                )

            # output.logits: [B, seq_len, 1]; take reward at 2nd and 3rd EOS per sequence
            logits = outputs.logits.squeeze(-1)  # (B, seq_len)
            seq_2d = eos_seq_idx.reshape(
                B, num_eos
            )  # [b, 0/1/2] = prompt end, comp1 end, comp2 end, ... , compN end
            actual_pos_indices = seq_2d[:, idx_eos]  # (B, N)
            rewards = logits[batch_idx.unsqueeze(1), actual_pos_indices]

        return rewards

    def __call__(self, inputs: dict):
        B = inputs["input_ids"].shape[0]
        num_candidates = inputs.pop("num_candidates")
        for k in inputs.keys():
            v = inputs[k]
            if isinstance(v, torch.Tensor):
                # convert if is float (not long)
                inputs[k] = v.to(self.model.device)
                if v.dtype != torch.long:
                    inputs[k] = inputs[k].to(self.model.dtype)
        outputs = self.model(**inputs)

        if self.standard:
            # rewards = torch.chunk(outputs.logits.squeeze(-1), chunks=num_candidates)  # LIST of (B,)
            # rewards = torch.stack(rewards, dim=1)  # (B, N)
            B = B // num_candidates
            rewards = outputs.logits.squeeze(-1).view(B, num_candidates)
        elif self.use_judge_role:
            rewards = outputs.judge_logits
        else:
            rewards = self.calculate_verifier_rewards(inputs, outputs, num_candidates)  # (B, N)

        return rewards
