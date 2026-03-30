import inspect
from typing import Optional

import torch
import string
import random

from transformers.cache_utils import Cache
from transformers.masking_utils import create_masks_for_generate


def tokenize_fn(example, processing_class, must_has_prompt: bool = False):
    chosen_input_ids = processing_class.apply_chat_template(
        example["chosen"],
        tools=example.get("tools"),
        return_dict=True,
        **example.get("chat_template_kwargs", {}),
    )["input_ids"]
    rejected_input_ids = processing_class.apply_chat_template(
        example["rejected"],
        tools=example.get("tools"),
        return_dict=True,
        **example.get("chat_template_kwargs", {}),
    )["input_ids"]
    if must_has_prompt and "prompt" not in example:
        raise ValueError("prompt is required when standard is False")
    if "prompt" in example:  # explicit prompt case
        prompt_input_ids = processing_class.apply_chat_template(
            example["prompt"],
            tools=example.get("tools"),
            return_dict=True,
            **example.get("chat_template_kwargs", {}),
        )["input_ids"]
        _len = len(prompt_input_ids)
        # if not rejected_input_ids[:_len] == prompt_input_ids:
        #     raise ValueError("Prompt input IDs do not match the rejected input IDs")
        # if not chosen_input_ids[:_len] == prompt_input_ids:
        #     raise ValueError("Prompt input IDs do not match the chosen input IDs")
    if "overall" in example:
        overall_input_ids = processing_class.apply_chat_template(
            example["overall"],
            tools=example.get("tools"),
            return_dict=True,
            **example.get("chat_template_kwargs", {}),
        )["input_ids"]
        overall_reversed_input_ids = processing_class.apply_chat_template(
            example["overall_reversed"],
            tools=example.get("tools"),
            return_dict=True,
            **example.get("chat_template_kwargs", {}),
        )["input_ids"]
        _len = len(overall_input_ids)

    output = {
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }
    if "prompt" in example:
        output["prompt_input_ids"] = prompt_input_ids
    if "overall" in example:
        output["overall_input_ids"] = overall_input_ids
        output["overall_reversed_input_ids"] = overall_reversed_input_ids
    return output



def sample_identifiers(num: int, include_alphabet: bool = True, include_number: bool = True) -> list[str]:
    options = []
    if include_alphabet:
        options.extend(string.ascii_uppercase)
    if include_number:
        options.extend("0123456789")
    assert len(options) >= num, f"Not enough options to sample {num} identifiers"
    identifiers = random.sample(options, num)
    return identifiers


def pad_2d_attn_masks(attention_masks: list[torch.Tensor], pad_to_multiple_of) -> torch.Tensor:
    """
    Pad a list of 2D attention masks to the same size.

    Args:
        attention_masks: List of 2D attention masks, each of shape [seq_len, seq_len]

    Returns:
        Padded attention masks stacked into a batch tensor of shape [batch_size, max_seq_len, max_seq_len]
    """
    # Find maximum sequence length
    max_seq_len = max(mask.shape[0] for mask in attention_masks)

    # Apply pad_to_multiple_of if specified
    if pad_to_multiple_of is not None:
        max_seq_len = ((max_seq_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    padded_attention_masks = []
    for mask in attention_masks:
        seq_len = mask.shape[0]
        if seq_len < max_seq_len:
            # Pad with -inf (masked positions) on the right and bottom
            padded_mask = torch.full(
                (max_seq_len, max_seq_len), -torch.inf, dtype=mask.dtype, device=mask.device
            )
            padded_mask[:seq_len, :seq_len] = mask
        else:
            padded_mask = mask
        padded_attention_masks.append(padded_mask)

    return torch.stack(padded_attention_masks, dim=0)  # (B, max_seq_len, max_seq_len)


def construct_pos_ids(
    prompt_len: int,
    local_lens: list[int],
    global_len: int | None = None,
) -> list[torch.Tensor]:
    """
    Construct position ids for parallel context, each tensor is of shape [seq_len].
    The position ids are 1-indexed.
    Args:
        prompt_len: Length of the prompt
        local_lens: Lengths of the locals
        global_len: Length of the global, if None, only the prompt and locals are considered

    Returns:
        Position ids of shape [seq_len]
    """

    position_ids = [torch.arange(prompt_len)]

    for i in range(len(local_lens)):
        prompt_comp_len = prompt_len + local_lens[i]
        position_ids += [torch.arange(prompt_len, prompt_comp_len)]
    if global_len is not None and global_len > 0:
        max_full_len = prompt_len + max(local_lens)
        position_ids += [torch.arange(max_full_len, max_full_len + global_len)]

    position_ids = torch.cat(position_ids)
    position_ids += 1  # position ids are 1-indexed
    return position_ids


def construct_attn_mask(
    seq_len: int,
    local_mask: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate attention mask (shape: [seq_len, seq_len]) for multiverse-style SFT.

    Visibility rules:
    - Different locals cannot see each other
    - Global can see all locals
    """
    # Start with a lower triangular matrix (causal mask)
    bool_attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Convert masks to tensors for easier indexing
    if isinstance(local_mask, list):
        local_mask = torch.tensor(local_mask, device=device)

    # Get unique local indices (excluding 0 which is prompt/no mask)
    num_locals = int(local_mask.max().item())

    # 1. Mask between different locals (locals cannot see each other)
    for comp_a in range(1, num_locals + 1):
        indices_a = (local_mask == comp_a).nonzero(as_tuple=True)[0]
        if len(indices_a) == 0:
            continue
        for comp_b in range(comp_a + 1, num_locals + 1):
            indices_b = (local_mask == comp_b).nonzero(as_tuple=True)[0]
            if len(indices_b) == 0:
                continue
            # Neither can see the other
            grid_i, grid_j = torch.meshgrid(indices_a, indices_b, indexing="ij")
            bool_attention_mask[grid_i.flatten(), grid_j.flatten()] = False
            bool_attention_mask[grid_j.flatten(), grid_i.flatten()] = False

    # Convert the final boolean mask to float format (0.0 for True, -inf for False)
    float_attention_mask = torch.full_like(bool_attention_mask, -torch.inf, dtype=torch.float)
    float_attention_mask = float_attention_mask.masked_fill(bool_attention_mask, 0.0)

    return float_attention_mask


def prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Prepare the model inputs for generation. Notable steps include selecting the correct input key and cloning when appropriate,
    creating position_ids from the attention_mask when missing, slicing inputs and converting 2D attention masks to 4D for
    compilable caches, and finally forwarding all additional keyword arguments unchanged to the model's forward pass.

    See the forward pass in the model documentation for expected arguments (different models might have different
    requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
    """

    # 1. Handle BC:
    model_inputs = {}
    model_inputs["cache_position"] = cache_position

    # 2. Generic cache-dependent input preparation
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        # TODO (joao): handle the case where cache length == input_ids length. The function below results in an
        # exception because we get empty input_ids after slicing. In essence, we need to roll back the cache 1
        # token to recompute the logits for the first token to be generated (but not all caches support roll backs)
        inputs_embeds, input_ids = self._cache_dependant_input_preparation(
            input_ids, inputs_embeds, cache_position
        )

    # 3. Prepare base model inputs
    input_ids_key = "input_ids"
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.

    if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
        model_inputs[input_ids_key] = None
        model_inputs["inputs_embeds"] = inputs_embeds
    else:
        # `clone` calls in this function ensure a consistent stride. See #32227
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
        model_inputs["inputs_embeds"] = None

    # 4. Create missing `position_ids` on the fly
    encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None

    attention_mask_key = "attention_mask"
    position_ids_key = "position_ids"
    if (
        attention_mask is not None
        and kwargs.get(position_ids_key) is None
        and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

    # 5. Slice model inputs if it's an input that should have the same length as `input_ids`

    model_input = kwargs.get("position_ids")
    if model_input is not None:
        # add one for every sequence if this is not prefill (seq_len = 1)

        if past_key_values is not None:
            current_input_length = (
                model_inputs["inputs_embeds"].shape[1]
                if model_inputs.get("inputs_embeds") is not None
                else model_inputs[input_ids_key].shape[1]
            )
            if current_input_length == 1:
                model_input += 1
            model_input = model_input[:, -current_input_length:]
            model_input = model_input.clone(memory_format=torch.contiguous_format)
        model_inputs["position_ids"] = model_input

    # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
    # pass)
    if (
        isinstance(past_key_values, Cache)
        and past_key_values.is_compileable
        and attention_mask is not None
        and attention_mask.ndim == 2
    ):
        if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

        # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
        # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
        base_model = getattr(self, self.base_model_prefix, self)
        decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
        causal_mask_creation_function = getattr(
            base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
        )
        if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
            causal_mask_creation_function = getattr(
                decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )

        # If it's not defined, it means the model uses the new general mask API
        if causal_mask_creation_function is None:  # can't be found
            token_type_ids = model_inputs.get("token_type_ids")
            position_ids = model_inputs.get(position_ids_key)
            # Some models may overwrite the general one
            causal_mask_creation_function = getattr(
                self, "create_masks_for_generate", create_masks_for_generate
            )
            attention_mask = causal_mask_creation_function(
                config=self.config,
                # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
                input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask

    if encoder_attention_mask is not None:
        model_inputs["attention_mask"] = encoder_attention_mask

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs
