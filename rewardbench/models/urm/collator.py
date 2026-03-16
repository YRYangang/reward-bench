import random
from typing import Any, Literal

import torch
from trl.trainer.utils import pad

from rewardbench.models.urm.utils import (
    construct_attn_mask,
    construct_pos_ids,
    pad_2d_attn_masks,
    sample_identifiers,
)


class ParallelDataCollatorForPreference:
    tokenizer = None
    pad_to_multiple_of: int | None = None
    parallel_context: bool = False
    standard: bool = True
    add_thinking: bool = False
    mode: Literal["verifier", "judge"] = "verifier"
    chosen_placeholder_token_id: int = -1
    rejected_placeholder_token_id: int = -1
    sample_id_include_alphabet: bool = True
    sample_id_include_number: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.add_thinking:
            self.thinking_begin_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
            self.thinking_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        if self.mode == "judge":
            judge_token_id = kwargs.get("judge_token_id", None)
            if judge_token_id is None:
                judge_token = kwargs.get("judge_token", None)
                if judge_token is None:
                    raise ValueError("judge_token or judge_token_id is required when mode is judge")
                judge_token_id = self.tokenizer.convert_tokens_to_ids(judge_token)
            self.judge_token_id = judge_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "margin" in examples[0]:
            margins = torch.tensor([example["margin"] for example in examples], dtype=torch.float)
        if not self.standard:
            assert self.mode is not None, "mode is required when standard is False"
            if self.mode == "verifier":
                output = self._verify(examples)
            elif self.mode == "judge":
                output = self._judge(examples)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            output = self._standard(examples)
        if "margin" in examples[0]:
            output["margin"] = margins
        return output

    def _standard(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]

        input_ids = chosen_input_ids + rejected_input_ids
        attention_mask = [torch.ones_like(ids) for ids in input_ids]

        output = {}

        # Pad
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_mask,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return output

    def _verify(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        all_input_ids = []
        shuffled = []
        attn_masks = []
        position_ids = []

        add_thinking = self.add_thinking
        if add_thinking:
            labels = []
            bot_id = self.thinking_begin_token_id
            eot_id = self.thinking_end_token_id
        for example in examples:
            prompt_input_ids = example["prompt_input_ids"]
            prompt_len = len(prompt_input_ids)
            chosen_input_ids = example["chosen_input_ids"][prompt_len:]
            chosen_len = len(chosen_input_ids)
            rejected_input_ids = example["rejected_input_ids"][prompt_len:]
            rejected_len = len(rejected_input_ids)
            prompt_input_ids = torch.tensor(prompt_input_ids)
            chosen_input_ids = torch.tensor(chosen_input_ids)
            rejected_input_ids = torch.tensor(rejected_input_ids)
            if add_thinking:
                chosen_begin_think = (chosen_input_ids == bot_id).nonzero().view(-1).item()
                chosen_end_think = (chosen_input_ids == eot_id).nonzero().view(-1).item()
                rejected_begin_think = (rejected_input_ids == bot_id).nonzero().view(-1).item()
                rejected_end_think = (rejected_input_ids == eot_id).nonzero().view(-1).item()
                chosen_thinking_labels = torch.zeros_like(chosen_input_ids) - 100
                rejected_thinking_labels = torch.zeros_like(rejected_input_ids) - 100
                prompt_thinking_labels = torch.zeros_like(prompt_input_ids) - 100
                # add 1 to exclude the <think> token (but include the  </think> token)
                # model does not need to know whether to think, but it needs to know when to stop
                chosen_thinking_labels[chosen_begin_think + 1 : chosen_end_think + 1] = chosen_input_ids[
                    chosen_begin_think + 1 : chosen_end_think + 1
                ]
                rejected_thinking_labels[rejected_begin_think + 1 : rejected_end_think + 1] = (
                    rejected_input_ids[rejected_begin_think + 1 : rejected_end_think + 1]
                )

            shuffle = example.get("shuffle", random.random() < 0.5)
            if shuffle:
                input_ids = torch.cat(
                    [
                        prompt_input_ids,
                        rejected_input_ids,
                        chosen_input_ids,
                    ],
                    dim=0,
                )
                if add_thinking:
                    _labels = torch.cat(
                        [
                            prompt_thinking_labels,
                            rejected_thinking_labels,
                            chosen_thinking_labels,
                        ],
                        dim=0,
                    )
            else:
                input_ids = torch.cat(
                    [
                        prompt_input_ids,
                        chosen_input_ids,
                        rejected_input_ids,
                    ],
                    dim=0,
                )
                if add_thinking:
                    _labels = torch.cat(
                        [
                            prompt_thinking_labels,
                            chosen_thinking_labels,
                            rejected_thinking_labels,
                        ],
                        dim=0,
                    )
            if add_thinking:
                labels.append(_labels)
            all_input_ids.append(input_ids)
            shuffled.append(shuffle)
            if self.parallel_context:
                if shuffle:
                    complesion_mask = torch.cat(
                        [torch.zeros(prompt_len), torch.zeros(rejected_len) + 1, torch.zeros(chosen_len) + 2],
                        dim=0,
                    )
                else:
                    complesion_mask = torch.cat(
                        [torch.zeros(prompt_len), torch.zeros(chosen_len) + 1, torch.zeros(rejected_len) + 2],
                        dim=0,
                    )
                attention_mask = construct_attn_mask(seq_len=input_ids.shape[0], local_mask=complesion_mask)
                position_id = construct_pos_ids(prompt_len=prompt_len, local_lens=[chosen_len, rejected_len])
                assert (
                    len(position_id) == input_ids.shape[0]
                ), f"position_id length {len(position_id)} does not match input_ids length {input_ids.shape[0]}"
                assert (
                    attention_mask.shape[0] == input_ids.shape[0]
                ), f"attention_mask length {len(attention_mask)} does not match input_ids length {input_ids.shape[0]}"
                attn_masks.append(attention_mask)
                position_ids.append(position_id)

        output = {}
        output["input_ids"] = pad(
            all_input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if add_thinking:
            output["labels"] = pad(
                labels,
                padding_value=-100,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

        if self.parallel_context:
            output["attention_mask"] = pad_2d_attn_masks(attn_masks, self.pad_to_multiple_of).unsqueeze(
                1
            )  # (B, 1, seq, seq)
            output["position_ids"] = pad(
                position_ids,
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            output["attention_mask"] = pad(
                [torch.ones_like(ids) for ids in all_input_ids],
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        output["shuffled"] = shuffled
        return output

    def _judge(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        all_input_ids = []
        shuffled = []
        attn_masks = []
        position_ids = []
        id_token_ids = []

        chosen_id = self.chosen_placeholder_token_id
        rejected_id = self.rejected_placeholder_token_id

        add_thinking = self.add_thinking
        if add_thinking:
            labels = []
            bot_id = self.thinking_begin_token_id
            eot_id = self.thinking_end_token_id
        for example in examples:
            shuffle = random.random() < 0.5
            shuffled.append(shuffle)
            overall_key = "overall" if not shuffle else "overall_reversed"
            prompt_input_ids = example["prompt_input_ids"]
            prompt_len = len(prompt_input_ids)
            overall_input_ids = example[overall_key + "_input_ids"]
            overall_len = len(overall_input_ids)
            chosen_input_ids = example["chosen_input_ids"][prompt_len:]
            chosen_len = len(chosen_input_ids)
            rejected_input_ids = example["rejected_input_ids"][prompt_len:]
            rejected_len = len(rejected_input_ids)

            judge_len = overall_len - prompt_len - chosen_len - rejected_len
            prompt_input_ids = torch.tensor(prompt_input_ids)
            chosen_input_ids = torch.tensor(chosen_input_ids)
            rejected_input_ids = torch.tensor(rejected_input_ids)
            overall_input_ids = torch.tensor(overall_input_ids)

            # sample and replace identifier tokens
            str_ids = sample_identifiers(
                2,
                include_alphabet=self.sample_id_include_alphabet,
                include_number=self.sample_id_include_number,
            )  # list[str]
            token_ids = self.tokenizer.convert_tokens_to_ids(str_ids)
            id_token_ids.append(token_ids)
            overall_input_ids[overall_input_ids == chosen_id] = token_ids[0]
            overall_input_ids[overall_input_ids == rejected_id] = token_ids[1]
            # the parts below should not be necessary but just in case
            chosen_input_ids[chosen_input_ids == chosen_id] = token_ids[0]
            chosen_input_ids[chosen_input_ids == rejected_id] = token_ids[1]
            rejected_input_ids[rejected_input_ids == chosen_id] = token_ids[0]
            rejected_input_ids[rejected_input_ids == rejected_id] = token_ids[1]

            # haddle thinking
            if add_thinking:
                _begin_think = (overall_input_ids == bot_id).nonzero().view(-1).item()
                _end_think = (overall_input_ids == eot_id).nonzero().view(-1).item()
                _thinking_labels = torch.zeros_like(overall_input_ids) - 100
                # add 1 to exclude the <think> token (but include the </think> token)
                # model does not need to know whether to think, but it needs to know when to stop
                _thinking_labels[_begin_think + 1 : _end_think + 1] = overall_input_ids[
                    _begin_think + 1 : _end_think + 1
                ]
                labels.append(_thinking_labels)
            all_input_ids.append(overall_input_ids)

            if self.parallel_context:
                if not shuffle:
                    complesion_mask = torch.cat(
                        [
                            torch.zeros(prompt_len),
                            torch.zeros(chosen_len) + 1,
                            torch.zeros(rejected_len) + 2,
                            # torch.zeros(judge_len),
                        ],
                        dim=0,
                    )
                else:
                    complesion_mask = torch.cat(
                        [
                            torch.zeros(prompt_len),
                            torch.zeros(rejected_len) + 1,
                            torch.zeros(chosen_len) + 2,
                            # torch.zeros(judge_len),
                        ],
                        dim=0,
                    )
                # see if there is any judge tokens in the input_ids
                # has_judge_tokens = (overall_input_ids == self.judge_token_id).any().item()
                # if has_judge_tokens:
                #     # additionally set the judge tokens complesion mask
                #     complesion_mask[overall_input_ids == self.judge_token_id] = torch.tensor(
                #         [1, 2], dtype=torch.float, device=complesion_mask.device
                #     )

                attention_mask = construct_attn_mask(
                    seq_len=overall_input_ids.shape[0], local_mask=complesion_mask
                )
                if not shuffle:
                    position_id = construct_pos_ids(
                        prompt_len=prompt_len, local_lens=[chosen_len, rejected_len], global_len=judge_len
                    )
                else:
                    position_id = construct_pos_ids(
                        prompt_len=prompt_len, local_lens=[rejected_len, chosen_len], global_len=judge_len
                    )

                # if has_judge_tokens:
                #     _nonzero_list = (overall_input_ids == self.judge_token_id).nonzero().view(-1)
                #     # set the judge tokens to **identical position ids** ( the pos of the first judge token)
                #     position_id[_nonzero_list] = position_id[_nonzero_list[0]].item()
                #     # then set all the ids afterwards - 1
                #     position_id[_nonzero_list[-1] + 1 :] = position_id[_nonzero_list[-1] + 1 :] - 1

                assert (
                    len(position_id) == overall_input_ids.shape[0]
                ), f"position_id length {len(position_id)} does not match input_ids length {overall_input_ids.shape[0]}"
                assert (
                    attention_mask.shape[0] == overall_input_ids.shape[0]
                ), f"attention_mask length {len(attention_mask)} does not match input_ids length {overall_input_ids.shape[0]}"

                attn_masks.append(attention_mask)
                position_ids.append(position_id)

        output = {}
        output["input_ids"] = pad(
            all_input_ids,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if add_thinking:
            output["labels"] = pad(
                labels,
                padding_value=-100,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

        if self.parallel_context:
            output["attention_mask"] = pad_2d_attn_masks(attn_masks, self.pad_to_multiple_of).unsqueeze(
                1
            )  # (B, 1, seq, seq)
            output["position_ids"] = pad(
                position_ids,
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            output["attention_mask"] = pad(
                [torch.ones_like(ids) for ids in all_input_ids],
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        output["shuffled"] = shuffled
        output["identifier_token_ids"] = torch.tensor(id_token_ids)  # (B, 2)
        return output


class ParallelDataCollatorForMultiplePreference(ParallelDataCollatorForPreference):
    # pad_to_multiple_of: int | None = None
    # parallel_context: bool = False
    # standard: bool = True
    # add_thinking: bool = False
    # mode: Literal["verifier", "judge"] = "verifier"
    # chosen_placeholder_token_id: int = -1
    # rejected_placeholder_token_id: int = -1
    # tokenizer = None
    # sample_id_include_alphabet: bool = True
    # sample_id_include_number: bool = False

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        num_info = self._get_num_candidates(examples)
        output = super().__call__(examples)
        return output | num_info

    def _get_num_candidates(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        num_chosens = []
        num_rejecteds = []
        num_candidates = []
        for example in examples:
            _num_chosen = len(example["chosen_input_ids"])
            _num_rejected = len(example["rejected_input_ids"])
            num_chosens.append(_num_chosen)
            num_rejecteds.append(_num_rejected)
            num_candidates.append(_num_chosen + _num_rejected)
            if len(set(num_candidates)) != 1:
                raise ValueError(
                    f"Number of candidates should be the same for all examples within a batch, but got {num_candidates}"
                )
        num_chosens = torch.tensor(num_chosens, dtype=torch.long)
        num_rejecteds = torch.tensor(num_rejecteds, dtype=torch.long)
        num_candidates = num_candidates[0]
        result = {
            "num_chosens": num_chosens,
            "num_rejecteds": num_rejecteds,
            "num_candidates": num_candidates,
        }
        return result

    def _standard(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        # chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        # rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        input_ids = []
        attention_masks = []
        is_chosen = []
        for example in examples:
            for chosen in example["chosen_input_ids"]:
                chosen = torch.tensor(chosen)
                input_ids.append(chosen)
                attention_masks.append(torch.ones_like(chosen))
                is_chosen.append(True)
            for rejected in example["rejected_input_ids"]:
                rejected = torch.tensor(rejected)
                input_ids.append(rejected)
                attention_masks.append(torch.ones_like(rejected))
                is_chosen.append(False)

        output = {}

        # Pad
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_masks,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["is_chosen"] = torch.tensor(is_chosen)

        return output

    def _verify(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        all_input_ids = []
        shuffled = []
        attn_masks = []
        position_ids = []
        for example in examples:
            prompt_input_ids = example["prompt_input_ids"]
            prompt_len = len(prompt_input_ids)
            chosen_input_ids = [chosen[prompt_len:] for chosen in example["chosen_input_ids"]]
            chosen_lens = [len(chosen) for chosen in chosen_input_ids]
            chosen_input_tensors = [torch.tensor(chosen) for chosen in chosen_input_ids]
            rejected_input_ids = [rejected[prompt_len:] for rejected in example["rejected_input_ids"]]
            rejected_lens = [len(rejected) for rejected in rejected_input_ids]
            rejected_input_tensors = [torch.tensor(rejected) for rejected in rejected_input_ids]
            # shuffle = example.get("shuffle", random.random() < 0.5)
            # TODO: add shuffle, currently not supported
            input_ids = torch.cat(
                [torch.tensor(prompt_input_ids)] + chosen_input_tensors + rejected_input_tensors,
                dim=0,
            )
            all_input_ids.append(input_ids)
            shuffled.append(False)
            if self.parallel_context:
                complesion_mask = [torch.zeros(prompt_len)]
                for chosen_len in chosen_lens:
                    num_complesion = len(complesion_mask)
                    complesion_mask.append(torch.zeros(chosen_len) + num_complesion)
                for rejected_len in rejected_lens:
                    num_complesion = len(complesion_mask)
                    complesion_mask.append(torch.zeros(rejected_len) + num_complesion)
                complesion_mask = torch.cat(complesion_mask, dim=0)

                attention_mask = construct_attn_mask(
                    seq_len=input_ids.shape[0],
                    local_mask=complesion_mask,
                )
                position_id = construct_pos_ids(
                    prompt_len=prompt_len,
                    local_lens=chosen_lens + rejected_lens,
                )
                assert (
                    len(position_id) == input_ids.shape[0]
                ), f"position_id length {len(position_id)} does not match input_ids length {input_ids.shape[0]}"
                assert (
                    attention_mask.shape[0] == input_ids.shape[0]
                ), f"attention_mask length {len(attention_mask)} does not match input_ids length {input_ids.shape[0]}"
                attn_masks.append(attention_mask)
                position_ids.append(position_id)

        output = {}
        output["input_ids"] = pad(
            all_input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        if self.parallel_context:
            output["attention_mask"] = pad_2d_attn_masks(attn_masks, self.pad_to_multiple_of).unsqueeze(
                1
            )  # (B, 1, seq, seq)
            output["position_ids"] = pad(
                position_ids,
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            output["attention_mask"] = pad(
                [torch.ones_like(ids) for ids in all_input_ids],
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        output["shuffled"] = shuffled
        return output

    def _judge(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        all_input_ids = []
        attn_masks = []
        position_ids = []
        id_token_ids = []

        chosen_id = self.chosen_placeholder_token_id
        rejected_id = self.rejected_placeholder_token_id

        for example in examples:
            prompt_input_ids = example["prompt_input_ids"]
            prompt_len = len(prompt_input_ids)

            chosen_input_ids = [chosen[prompt_len:] for chosen in example["chosen_input_ids"]]
            chosen_lens = [len(chosen) for chosen in chosen_input_ids]
            # chosen_input_tensors = [torch.tensor(chosen) for chosen in chosen_input_ids]
            num_chosen = len(chosen_input_ids)
            rejected_input_ids = [rejected[prompt_len:] for rejected in example["rejected_input_ids"]]
            rejected_lens = [len(rejected) for rejected in rejected_input_ids]
            # rejected_input_tensors = [torch.tensor(rejected) for rejected in rejected_input_ids]
            num_rejected = len(rejected_input_ids)

            overall_input_ids = example["overall_input_ids"]
            overall_len = len(overall_input_ids)
            judge_len = overall_len - prompt_len - sum(chosen_lens) - sum(rejected_lens)

            str_ids = sample_identifiers(
                num_chosen + num_rejected,
                include_alphabet=self.sample_id_include_alphabet,
                include_number=self.sample_id_include_number,
            )  # list[str]
            token_ids = self.tokenizer.convert_tokens_to_ids(str_ids)
            id_token_ids.append(token_ids)
            for i in range(num_chosen):
                overall_input_ids[overall_input_ids == chosen_id] = token_ids[i]
            for i in range(num_rejected):
                overall_input_ids[overall_input_ids == rejected_id] = token_ids[num_chosen + i]

            all_input_ids.append(overall_input_ids)
            if self.parallel_context:
                complesion_mask = [torch.zeros(prompt_len)]
                for chosen_len in chosen_lens:
                    num_complesion = len(complesion_mask)
                    complesion_mask.append(torch.zeros(chosen_len) + num_complesion)
                for rejected_len in rejected_lens:
                    num_complesion = len(complesion_mask)
                    complesion_mask.append(torch.zeros(rejected_len) + num_complesion)
                complesion_mask.append(torch.zeros(judge_len))
                complesion_mask = torch.cat(complesion_mask, dim=0)

                attention_mask = construct_attn_mask(
                    seq_len=overall_input_ids.shape[0],
                    local_mask=complesion_mask,
                )
                position_id = construct_pos_ids(
                    prompt_len=prompt_len,
                    local_lens=chosen_lens + rejected_lens,
                    global_len=judge_len,
                )
                assert (
                    len(position_id) == overall_input_ids.shape[0]
                ), f"position_id length {len(position_id)} does not match input_ids length {overall_input_ids.shape[0]}"
                assert (
                    attention_mask.shape[0] == overall_input_ids.shape[0]
                ), f"attention_mask length {len(attention_mask)} does not match input_ids length {overall_input_ids.shape[0]}"
                attn_masks.append(attention_mask)
                position_ids.append(position_id)

        output = {}
        output["input_ids"] = pad(
            all_input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        if self.parallel_context:
            output["attention_mask"] = pad_2d_attn_masks(attn_masks, self.pad_to_multiple_of).unsqueeze(
                1
            )  # (B, 1, seq, seq)
            output["position_ids"] = pad(
                position_ids,
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            output["attention_mask"] = pad(
                [torch.ones_like(ids) for ids in all_input_ids],
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        output["identifier_token_ids"] = torch.tensor(id_token_ids)  # (B, 2)

        return output
