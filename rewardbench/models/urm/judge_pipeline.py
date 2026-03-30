from types import MethodType
import random
import torch
from transformers import GenerationConfig, LogitsProcessorList, NoBadWordsLogitsProcessor, Pipeline

from rewardbench.models.urm.collator import ParallelDataCollatorForPreference
from rewardbench.models.urm.utils import prepare_inputs_for_generation, tokenize_fn
from rewardbench.models.urm.pipeline import apply_template


class ParallelRMRewardBenchJudgePipeline(Pipeline):
    """
    This class outputs a delta rather than a score for each.
    """

    def __init__(self, parallel_context: bool, enable_thinking: bool, **kwargs):
        super().__init__(**kwargs)
        self.model.eval().requires_grad_(False)
        self.parallel_context = parallel_context
        self.enable_thinking = enable_thinking
        self.eot_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.bot_id = self.tokenizer.convert_tokens_to_ids("<think>")

        self.judge_token = "<|judgement|>"
        if enable_thinking and parallel_context:
            self.model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.model)

        self.chosen_id_placeholder_token = "<|object_ref_start|>"
        self.rejected_id_placeholder_token = "<|object_ref_end|>"
        self.chosen_id_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            self.chosen_id_placeholder_token
        )
        self.rejected_id_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            self.rejected_id_placeholder_token
        )
        self.collator = ParallelDataCollatorForPreference(
            tokenizer=self.tokenizer,
            parallel_context=parallel_context,
            standard=False,
            mode="judge",
            add_thinking=False,
            chosen_placeholder_token_id=self.chosen_id_placeholder_token_id,
            rejected_placeholder_token_id=self.rejected_id_placeholder_token_id,
            judge_token=self.judge_token,
        )

    def _logits_processor_no_judge_while_generating(self) -> LogitsProcessorList:
        """Ban <|judgement|> during `generate`; judge slots are appended after generation."""
        return LogitsProcessorList(
            [
                NoBadWordsLogitsProcessor(
                    bad_words_ids=[[int(self.collator.judge_token_id)]],
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
        # make it like an output of batchsize = 1
        rewards_chosen = outputs["judge_logits"][0]
        rewards_rejected = outputs["judge_logits"][1]
        shuffle = outputs["shuffle"]
        if shuffle:
            rewards_chosen, rewards_rejected = rewards_rejected, rewards_chosen
        return {
            "results": [rewards_chosen > rewards_rejected],
            "rewards_chosen": [rewards_chosen],
            "rewards_rejected": [rewards_rejected],
        }

    def preprocess(self, inputs: dict):
        inputs["text_thinking"] = "abracadabra"

        result = apply_template(
            example=inputs,
            use_judge_token=True,
            use_judge_role=True,
            with_thinking=self.enable_thinking,
            chosen_id_placeholder_token=self.chosen_id_placeholder_token,
            rejected_id_placeholder_token=self.rejected_id_placeholder_token,
        )
        result = tokenize_fn(result, self.tokenizer, True)
        # remove every thing after <think>
        if self.enable_thinking:
            _position = result["overall_input_ids"].index(self.bot_id)
            result["overall_input_ids"] = result["overall_input_ids"][: _position + 1]
            _position = result["overall_reversed_input_ids"].index(self.bot_id)
            result["overall_reversed_input_ids"] = result["overall_reversed_input_ids"][: _position + 1]
        result["shuffle"] = random.random() < 0.5
        self.tmp_tokenize_results = result

        result = self.collator([result])
        result = self._move_tensors(result)
        return result

    def _re_collate(self, new_input_ids: torch.Tensor, cache_position: int | None = None):
        result = self.tmp_tokenize_results
        shuffle = result["shuffle"]
        if shuffle:
            result["overall_reversed_input_ids"] = new_input_ids.view(-1).tolist()
            del result["overall_input_ids"]
        else:
            result["overall_input_ids"] = new_input_ids.view(-1).tolist()
            del result["overall_reversed_input_ids"]
        assert not (new_input_ids == self.tokenizer.pad_token_id).any().item()

        result = self.collator([result])
        result = self._move_tensors(result)
        if cache_position is not None:
            result["input_ids"] = result["input_ids"][:, cache_position:]
            if "position_ids" in result:
                result["position_ids"] = result["position_ids"][:, cache_position:]
            if "attention_mask" in result:
                _dim = result["attention_mask"].dim()
                if _dim == 2:
                    result["attention_mask"] = result["attention_mask"][:, cache_position:]
                elif _dim == 4:
                    result["attention_mask"] = result["attention_mask"][:, :, cache_position:]
                else:
                    raise ValueError(f"Unexpected dimension of attention_mask: {_dim}")
        return result

    def _forward_naive(self, model_inputs, generation_config: GenerationConfig | None = None):
        shuffle = model_inputs["shuffled"][0]
        device = self.model.device
        gen_outputs = self.model.generate(
            input_ids=model_inputs["input_ids"],
            use_cache=True,
            eos_token_id=self.eot_id,
            generation_config=generation_config,
            logits_processor=self._logits_processor_no_judge_while_generating(),
            return_dict_in_generate=True,
        )
        final_kv = gen_outputs.past_key_values
        new_content = "\n\n" + self.judge_token * 2
        final_input_ids = gen_outputs.sequences

        new_input_ids = self.tokenizer(new_content, return_tensors="pt").input_ids
        new_input_ids = new_input_ids.to(device).view(1, -1)
        new_full_input_ids = torch.cat([final_input_ids, new_input_ids], dim=1)
        # cache_position = final_kv.get_seq_length()
        # new_full_inputs = self._re_collate(new_full_input_ids, cache_position, shuffle)

        inputs = self.model.forward(
            input_ids=new_full_input_ids,
            # past_key_values=final_kv,
            use_cache=True,
        )
        return {
            "judge_logits": inputs.judge_logits.flatten().tolist(),
            "output_ids": new_full_input_ids,
            "shuffle": shuffle,
        }

    def _forward(self, model_inputs, generation_config: GenerationConfig | None = None):
        if not self.enable_thinking:
            return self._forward_no_think(model_inputs)
        if self.parallel_context:
            return self._forward_parallel(model_inputs, generation_config)
        else:
            return self._forward_naive(model_inputs, generation_config)

    def _forward_no_think(
        self,
        model_inputs,
    ):
        if "shuffled" in model_inputs:
            shuffle = model_inputs.pop("shuffled")[0]
        else:
            shuffle = False
        result = self.model.forward(**model_inputs)
        return {
            "judge_logits": result.judge_logits.flatten().tolist(),
            "shuffle": shuffle,
        }

    def _forward_parallel(
        self,
        model_inputs,
        generation_config: GenerationConfig | None = None,
    ):
        if "shuffled" in model_inputs:
            shuffle = model_inputs.pop("shuffled")[0]
        else:
            shuffle = False
        outputs = self.model.forward(**model_inputs, use_cache=True, logits_to_keep=1)
        kv_cache = outputs.past_key_values
        next_token = outputs.logits[0, -1, :].argmax().view(1, 1)

        new_full_input_ids = torch.cat([model_inputs["input_ids"], next_token], dim=-1)
        position_ids = model_inputs["position_ids"].max(dim=-1).values.view(1, 1)
        # we do not +1 here because the _prepare_inputs_for_generation will do this

        device = self.model.device

        gen_outputs = self.model.generate(
            input_ids=new_full_input_ids,
            attention_mask=torch.ones_like(new_full_input_ids, dtype=torch.bool),
            position_ids=position_ids,
            past_key_values=kv_cache,
            use_cache=True,
            eos_token_id=self.eot_id,
            generation_config=generation_config,
            logits_processor=self._logits_processor_no_judge_while_generating(),
            return_dict_in_generate=True,
        )
        final_kv = gen_outputs.past_key_values
        new_content = "\n\n" + self.judge_token * 2
        final_input_ids = gen_outputs.sequences

        new_input_ids = self.tokenizer(new_content, return_tensors="pt").input_ids
        new_input_ids = new_input_ids.to(device).view(1, -1)
        new_full_input_ids = torch.cat([final_input_ids, new_input_ids], dim=1)
        cache_position = final_kv.get_seq_length()
        new_full_inputs = self._re_collate(new_full_input_ids, cache_position)

        inputs = self.model.forward(
            **new_full_inputs,
            past_key_values=final_kv,
            use_cache=True,
        )
        return {
            "judge_logits": inputs.judge_logits.flatten().tolist(),
            "output_ids": new_full_input_ids,
            "shuffle": shuffle,
        }
