# Copyright 2023 AllenAI. All rights reserved.
#
# Debug helper: tokenizer + dataset + collator only (no model / no inference).
# Use this to see why SimpleDataCollatorForPreference asserts on A/B before EOS.

from __future__ import annotations

import argparse
import logging
import os
import sys

from accelerate.logging import get_logger
from tqdm import tqdm
from transformers import AutoTokenizer

from rewardbench import load_eval_dataset
from rewardbench.models.basic_sft_model.collator import SimpleDataCollatorForPreference, tokenize_example
from rewardbench.models.basic_sft_model.pipeline import process_example_rewardbench
from rewardbench.script_args import add_common_generative_args


HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run RewardBench rows through the basic-SFT collator without loading a model.",
    )
    add_common_generative_args(parser, dataset=True, score_w_ratings=True)
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Truncate tokenized sequences (same as run_basic_sft_model pipeline).",
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        default="discriminative",
        choices=["generative", "discriminative"],
        help="Must match the collator path you use in eval.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Match run_basic_sft_model --enable_thinking (uses dummy global_analysis).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First dataset index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last dataset index (exclusive). Default: len(dataset).",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on first collator failure.",
    )
    parser.add_argument(
        "--show_ok",
        action="store_true",
        help="Also print a one-line tail summary for rows that pass (verbose).",
    )
    args = parser.parse_args()
    if isinstance(args.model, list):
        if len(args.model) != 1:
            raise ValueError("Pass a single --model path.")
        args.model = args.model[0]
    return args


def _tail_debug(tokenizer, input_ids: list[int] | torch.Tensor, window: int = 24) -> str:
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.tolist()
    else:
        ids = list(input_ids)
    eos_id = tokenizer.eos_token_id
    eos_idx = [i for i, t in enumerate(ids) if t == eos_id]
    tail = ids[-window:] if len(ids) >= window else ids
    tail_txt = tokenizer.decode(tail, skip_special_tokens=False)
    last_eos = eos_idx[-1] if eos_idx else None
    before = None
    if last_eos is not None and last_eos > 0:
        before = (ids[last_eos - 1], tokenizer.decode([ids[last_eos - 1]]))
    return (
        f"len={len(ids)} last_eos_idx={last_eos} n_eos={len(eos_idx)} "
        f"token_before_last_eos={before} | tail_ids={tail} | tail_decode={tail_txt!r}"
    )


def _build_one(
    raw: dict,
    tokenizer,
    max_length: int | None,
    enable_thinking: bool,
    shuffle: bool,
) -> dict:
    """Mirror BasicSFTJudgePipeline.preprocess for a single row (no random shuffle)."""
    raw = dict(raw)
    raw["global_analysis"] = "abracadabra"
    result = process_example_rewardbench(
        example=raw,
        add_thinking=enable_thinking,
        thinking_key="global_analysis",
    )
    result = tokenize_example(result, tokenizer)
    if max_length is not None:
        result["input_ids"] = result["input_ids"][:max_length]
        result["reversed_input_ids"] = result["reversed_input_ids"][:max_length]
    if enable_thinking:
        bot_id = tokenizer.convert_tokens_to_ids("<think>")
        _position = result["input_ids"].index(bot_id)
        result["input_ids"] = result["input_ids"][: _position + 1]
        _position = result["reversed_input_ids"].index(bot_id)
        result["reversed_input_ids"] = result["reversed_input_ids"][: _position + 1]
    result["shuffle"] = shuffle
    return result


def main():
    args = get_args()
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=None,
        custom_dialogue_formatting=True,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    end = len(dataset) if args.end is None else min(args.end, len(dataset))
    start = max(0, args.start)
    if start >= end:
        raise ValueError(f"Invalid range: start={start} end={end}")

    collator = SimpleDataCollatorForPreference(
        tokenizer=tokenizer,
        paradigm=args.paradigm,
        add_thinking=args.enable_thinking,
        judge_token="<|judgement|>",
    )

    failures: list[dict] = []
    for i in tqdm(range(start, end), desc="collator check"):
        row = dataset[i]
        example_id = ids[i]
        subset = subsets[i]
        for shuffle in (False, True):
            branch = "reversed" if shuffle else "forward"
            item = _build_one(
                row,
                tokenizer,
                args.max_length,
                args.enable_thinking,
                shuffle=shuffle,
            )
            key = "reversed_input_ids" if shuffle else "input_ids"
            try:
                collator([item])
            except AssertionError as e:
                seq = item[key]
                rec = {
                    "index": i,
                    "id": example_id,
                    "subset": subset,
                    "shuffle": shuffle,
                    "branch": branch,
                    "error": str(e),
                }
                failures.append(rec)
                print("\n" + "=" * 80)
                print(
                    f"FAIL index={i} id={example_id!r} subset={subset!r} "
                    f"shuffle={shuffle} ({branch}) paradigm={args.paradigm}"
                )
                print(f"Assertion: {e}")
                print(f"[{branch}] {_tail_debug(tokenizer, seq)}")
                # Extra: show last 8 tokens with individual decodes
                s = seq if isinstance(seq, list) else seq.tolist()
                tail_n = 8
                tail_slice = s[-tail_n:]
                print(f"Last {tail_n} token ids: {tail_slice}")
                for j, tid in enumerate(tail_slice):
                    print(f"  [{len(s) - tail_n + j}] id={tid} decode={tokenizer.decode([tid])!r}")
                if args.fail_fast:
                    print("\n--fail_fast: stopping.")
                    return
            else:
                if args.show_ok:
                    seq = item[key]
                    print(f"OK index={i} {branch}: {_tail_debug(tokenizer, seq)}")

    print("\n" + "=" * 80)
    print(f"Checked indices [{start}, {end}) x2 shuffles. Failures: {len(failures)}")
    if failures:
        print("Summary (index, id, subset, shuffle):")
        for r in failures:
            print(f"  {r['index']}, {r['id']!r}, {r['subset']!r}, shuffle={r['shuffle']}")
    else:
        print("No collator assertion failures in this range.")


if __name__ == "__main__":
    main()
