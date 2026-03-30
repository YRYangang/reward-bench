import random
from typing import Any, Literal

import torch
from trl.trainer.utils import pad


# RewardBench pair-v2 (evaluations/reward-bench/rewardbench/generative.py: MTBENCH_V2 / prompt_v2)
_REWARDBENCH_JUDGE_SYSTEM = (
    "You are an impartial judge and evaluate the quality of the responses provided by two AI assistants. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. "
    "Output your final verdict by strictly following this format: "
    '"A" if assistant A is better, "B" if assistant B is better.'
)


def _rewardbench_pair_user(question: str, answer_a: str, answer_b: str) -> str:
    """MTBENCH_V2 prompt_template: A vs B block layout."""
    return (
        f"[User Question]\n{question}\n\n"
        f"[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
        f"[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
    )


def tokenize_example(example, processing_class):

    input_ids = processing_class.apply_chat_template(
        example["overall"],
        tools=example.get("tools"),
        return_dict=True,
        **example.get("chat_template_kwargs", {}),
    )["input_ids"]
    _len = len(input_ids)
    reversed_input_ids = processing_class.apply_chat_template(
        example["overall_reversed"],
        tools=example.get("tools"),
        return_dict=True,
        **example.get("chat_template_kwargs", {}),
    )["input_ids"]
    output = {}
    output["input_ids"] = input_ids
    output["reversed_input_ids"] = reversed_input_ids

    return output


def process_example(
    example: dict[str, Any],
    add_thinking: bool = False,
) -> dict[str, Any]:
    """
    LLM-as-judge in RewardBench pair-v2 style: system = impartial judge instructions,
    user = [Question] + Assistant A/B answers; label is [[A]] or [[B]] depending on order.
    Ground truth: chosen is better → [[A]] when A=chosen, [[B]] when A=rejected (B=chosen).
    """
    question = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    result = {}
    result["overall"] = [
        {"role": "system", "content": _REWARDBENCH_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _rewardbench_pair_user(question, chosen, rejected),
        },
        {"role": "assistant", "content": "A"},
    ]
    result["overall_reversed"] = [
        {"role": "system", "content": _REWARDBENCH_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _rewardbench_pair_user(question, rejected, chosen),
        },
        {"role": "assistant", "content": "B"},
    ]
    if add_thinking:
        # raise NotImplementedError("Thinking is not supported for judge mode")
        global_analysis = example["global_analysis"]
        global_analysis = global_analysis.replace("<<chosen>>", "A")
        global_analysis = global_analysis.replace("<<rejected>>", "B")
        result["overall"][-1]["reasoning_content"] = global_analysis
        result["overall_reversed"][-1]["reasoning_content"] = global_analysis
    return result


class SimpleDataCollatorForPreference:
    tokenizer = None
    pad_to_multiple_of: int | None = None
    add_thinking: bool = False
    paradigm: Literal["generative", "discriminative "] = "generative"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.add_thinking:
            self.thinking_begin_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
            self.thinking_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")

        if self.paradigm == "discriminative":
            judge_token_id = kwargs.get("judge_token_id", None)
            if judge_token_id is None:
                judge_token = kwargs.get("judge_token", None)
                if judge_token is None:
                    raise ValueError(
                        "judge_token or judge_token_id is required when paradigm is discriminative"
                    )
                judge_token_id = self.tokenizer.convert_tokens_to_ids(judge_token)
            self.judge_token_id = judge_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "margin" in examples[0]:
            margins = torch.tensor([example["margin"] for example in examples], dtype=torch.float)
        output = (
            self._generative(examples) if self.paradigm == "generative" else self._discriminative(examples)
        )
        if "margin" in examples[0]:
            output["margin"] = margins
        return output

    def _discriminative(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        shuffles = []
        labels = []
        input_ids = []
        for example in examples:
            shuffle = example.get("shuffle", random.random() < 0.5)
            _input_ids = example["input_ids"] if not shuffle else example["reversed_input_ids"]
            _input_ids = torch.as_tensor(_input_ids, dtype=torch.long)

            # find the last eos token
            last_eos_token = (_input_ids == self.tokenizer.eos_token_id).nonzero().view(-1)[-1].item()
            # asserrt the token before that is either A or B
            A_token_id = self.tokenizer.convert_tokens_to_ids("A")
            B_token_id = self.tokenizer.convert_tokens_to_ids("B")
            assert _input_ids[last_eos_token - 1] in [
                A_token_id,
                B_token_id,
            ], f"The token before the last eos token should be either A or B, but got  Token: {self.tokenizer.decode(_input_ids[last_eos_token - 1])}, ID: {_input_ids[last_eos_token - 1]}"
            # replace the A/B token with 2 judge tokens
            _len_before = _input_ids.shape[0]
            _input_ids = torch.cat(
                [
                    _input_ids[: last_eos_token - 1],
                    torch.tensor([self.judge_token_id, self.judge_token_id]),
                    _input_ids[last_eos_token:],
                ]
            )
            _len_after = _input_ids.shape[0]
            assert (
                _len_after == _len_before + 1
            ), f"The length of the input_ids should be increased by 1, but got {_len_after} - {_len_before} = {_len_after - _len_before}"

            if self.add_thinking:
                _labels = torch.zeros_like(_input_ids) - 100
                bot_id = self.thinking_begin_token_id
                eot_id = self.thinking_end_token_id
                _begin_think = (_input_ids == bot_id).nonzero().view(-1).item()
                _end_think = (_input_ids == eot_id).nonzero().view(-1).item()
                _labels[_begin_think + 1 : _end_think + 1] = _input_ids[_begin_think + 1 : _end_think + 1]
                labels.append(_labels)
            input_ids.append(_input_ids)
            shuffles.append(shuffle)
        output = {}
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if len(labels) > 0:
            output["labels"] = pad(
                labels,
                padding_value=-100,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        output["attention_mask"] = pad(
            [torch.ones_like(ids) for ids in input_ids],
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["shuffles"] = shuffles
        return output

    def _generative(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        shuffles = []
        labels = []
        input_ids = []
        for example in examples:
            shuffle = example.get("shuffle", random.random() < 0.5)
            _input_ids = example["input_ids"] if not shuffle else example["reversed_input_ids"]
            _input_ids = torch.as_tensor(_input_ids, dtype=torch.long)
            _labels = torch.zeros_like(_input_ids) - 100
            if self.add_thinking:
                bot_id = self.thinking_begin_token_id
                eot_id = self.thinking_end_token_id
                _begin_think = (_input_ids == bot_id).nonzero().view(-1).item()
                _end_think = (_input_ids == eot_id).nonzero().view(-1).item()

                # add 1 to exclude the <think> token (but include the  </think> token)
                # model does not need to know whether to think, but it needs to know when to stop
                _labels[_begin_think + 1 : _end_think + 1] = _input_ids[_begin_think + 1 : _end_think + 1]
            # find the last eos token
            last_eos_token = (_input_ids == self.tokenizer.eos_token_id).nonzero().view(-1)[-1].item()
            # asserrt the token before that is either A or B
            A_token_id = self.tokenizer.convert_tokens_to_ids("A")
            B_token_id = self.tokenizer.convert_tokens_to_ids("B")
            assert _input_ids[last_eos_token - 1] in [
                A_token_id,
                B_token_id,
            ], f"The token before the last eos token should be either A or B, but got {_input_ids[last_eos_token - 1]}"
            _labels[last_eos_token - 1] = _input_ids[last_eos_token - 1]

            labels.append(_labels)
            input_ids.append(_input_ids)
            shuffles.append(shuffle)

        output = {}

        output["input_ids"] = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        output["labels"] = pad(
            labels,
            padding_value=-100,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            [torch.ones_like(ids) for ids in input_ids],
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["shuffles"] = shuffles

        return output
