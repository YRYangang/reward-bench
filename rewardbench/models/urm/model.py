from typing import Optional, Union, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel, Qwen3PreTrainedModel, GenerationMixin, Qwen3Config
from transformers.modeling_layers import Cache, TransformersKwargs, Unpack
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import can_return_tuple


@dataclass
class GenerativeRewardingOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    judge_logits: Optional[torch.FloatTensor] = None
    verify_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


def get_hidden_states_logits(
    hidden_states: torch.FloatTensor,
    input_ids: torch.LongTensor,
    token_id: int,
    batch_index: torch.LongTensor | None = None,
    scorer: Callable[[torch.FloatTensor], torch.FloatTensor] | None = None,
) -> torch.Tensor:
    if batch_index is None:
        batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
    batch_size = input_ids.shape[0]

    # Find all verify token positions
    verify_mask = input_ids == token_id  # (B, seq_len)
    # if no verify token in the input_ids, return None
    if not verify_mask.any():
        return None

    # Get all verify token positions: (N, 2) where each row is [batch_idx, pos_idx]
    verify_positions_all = torch.argwhere(verify_mask)

    if verify_positions_all.shape[0] == 0:
        return None

    # Group verify positions by batch index
    batch_indices = verify_positions_all[:, 0]
    position_indices = verify_positions_all[:, 1]

    # Count verify tokens per batch and find maximum
    unique_batch_indices, counts = torch.unique(batch_indices, return_counts=True)
    max_num_verify = counts.max().item() if len(counts) > 0 else 0

    # Check if all sequences have the same number of verify tokens
    all_same_count = len(unique_batch_indices) == batch_size and (counts == max_num_verify).all()

    hidden_size = hidden_states.shape[-1]

    if not all_same_count:
        # Handle variable number of verify tokens per sequence
        if scorer is None:
            # Pad with zeros for sequences with fewer verify tokens
            verify_hidden_states = torch.zeros(
                (batch_size, max_num_verify, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # Fill in hidden states for sequences that have verify tokens
            for batch_idx in unique_batch_indices:
                batch_idx_int = batch_idx.item()
                mask = batch_indices == batch_idx
                positions = position_indices[mask]
                # Sort positions to ensure consistent ordering
                positions, _ = torch.sort(positions)
                num_tokens = positions.shape[0]
                verify_hidden_states[batch_idx_int, :num_tokens] = hidden_states[
                    batch_index[batch_idx_int],
                    positions,
                    :,
                ]
            return verify_hidden_states
        else:
            # For scorer, only pass real hidden states (no padding)
            # First, determine the output shape from scorer by scoring a dummy tensor
            dummy_hidden = torch.zeros(
                (1, 1, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
            )
            dummy_scored = scorer(dummy_hidden)  # (1, 1, ...)
            output_shape = dummy_scored.shape[2:]  # Shape after (batch, num_verify) dimensions
            # print(f"output_shape: {output_shape}")

            # Extract real hidden states for each sequence and score them
            padded_results = torch.zeros(
                (batch_size, max_num_verify) + output_shape,
                dtype=dummy_scored.dtype,
                device=dummy_scored.device,
            )

            for batch_idx in range(batch_size):
                mask = batch_indices == batch_idx
                if mask.any():
                    # This sequence has verify tokens - extract and score only real hidden states
                    positions = position_indices[mask]
                    # Sort positions to ensure consistent ordering
                    positions, _ = torch.sort(positions)
                    sequence_hidden_states = hidden_states[
                        batch_index[batch_idx],
                        positions,
                        :,
                    ]  # (num_verify_tokens, hidden_size)
                    # Score this sequence's hidden states (add batch dimension for scorer)
                    scored = scorer(sequence_hidden_states.unsqueeze(0))  # (1, num_verify_tokens, ...)
                    num_tokens = scored.shape[1]
                    padded_results[batch_idx, :num_tokens] = scored.squeeze(0)  # (num_verify_tokens, ...)
                # If no verify tokens, leave as zeros (already initialized)

            return padded_results
    else:
        # All sequences have the same number of verify tokens (original behavior)
        # Sort by batch index, then by position to ensure correct ordering
        seq_len = input_ids.shape[1]
        sorted_indices = torch.argsort(verify_positions_all[:, 0] * seq_len + verify_positions_all[:, 1])
        verify_positions_sorted = verify_positions_all[sorted_indices]
        num_verify_per_seq = verify_positions_sorted.shape[0] // batch_size
        verify_position = verify_positions_sorted.view(batch_size, num_verify_per_seq, 2)[
            ..., -1
        ]  # (B, num_identifiers)

        verify_hidden_states = hidden_states[
            batch_index.unsqueeze(1),
            verify_position,
        ]  # (B, num_identifiers, hidden_size)
        if scorer is None:
            return verify_hidden_states
        else:
            return scorer(verify_hidden_states)


class GenerativeRewarding:
    def __init__(self, config):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        identifier_token_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> GenerativeRewardingOutputWithPast:
        # assert (
        #     self.config.task_specific_params["ver_token_id"] != -1
        #     and self.config.task_specific_params["judge_token_id"] != -1
        # ), "verification and judgement token ids should be set"
        batch_size = input_ids.shape[0]
        batch_index = torch.arange(batch_size)
        outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # (B, seq_len, vocab_size)
        try:
            ver_token_id = self.config.task_specific_params.get("ver_token_id", -1)
            judge_token_id = self.config.task_specific_params.get("judge_token_id", -1)
        except:
            ver_token_id = -1
            judge_token_id = -1

        verify_logits = None
        if ver_token_id != -1:
            verify_logits = get_hidden_states_logits(
                hidden_states=hidden_states,
                input_ids=input_ids,
                token_id=ver_token_id,
                batch_index=batch_index,
                scorer=lambda x: self.score(self.dropout(x)).squeeze(-1),
            )

        judge_logits = None
        if judge_token_id != -1:
            judge_logits = get_hidden_states_logits(
                hidden_states=hidden_states,
                input_ids=input_ids,
                token_id=judge_token_id,
                batch_index=batch_index,
                scorer=lambda x: self.score(self.dropout(x)).squeeze(-1),
            )

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        return GenerativeRewardingOutputWithPast(
            loss=loss,
            logits=logits,
            judge_logits=judge_logits,
            verify_logits=verify_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen3ForGenerativeRewarding(GenerativeRewarding, Qwen3PreTrainedModel, GenerationMixin):
    config_class = Qwen3Config
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
