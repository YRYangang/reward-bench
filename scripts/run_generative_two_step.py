# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Two-step generative RM: (1) rate each candidate's answer(s) on 0-5; (2) given candidates + analyses, output [[A]] or [[B]].
# Examples:
# python scripts/run_generative_two_step.py --model gpt-3.5-turbo
# python scripts/run_generative_two_step.py --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer

from rewardbench import load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.script_args import add_common_generative_args
from rewardbench.generative import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_from_analyses,
    get_rating_0_5_user_prompt,
    process_judgement,
    run_judge_two_step,
)
from rewardbench.utils import calculate_scores_per_section

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    parser = argparse.ArgumentParser()
    add_common_generative_args(parser, dataset=False, score_w_ratings=False)
    return parser.parse_args()


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Running two-step reward model on {args.model} with chat template {args.chat_template}")

    model_type = "Generative RM (two-step)"

    # Two-step does not support ensemble; use single model
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        logger.warning("Two-step judge does not support ensemble; using first model only.")
        args.model = args.model[0]

    # define variable if is API or local
    if args.force_local:
        is_api_models = False
    else:
        is_api_models = args.model in API_MODEL_LIST

    # if model isn't API, load via vllm
    if not is_api_models:
        from vllm import LLM, SamplingParams

        # if multi gpu, set multiproc method to spawn
        if args.num_gpus > 1:
            # Set the environment variable
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # load model
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.vllm_gpu_util,
            disable_custom_all_reduce=True,
            # max_seq_length=args.vllm_max_seq_length,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "Llama-3" in args.model or "llama3-8b" in args.model and "3.1" not in args.model:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1,
            max_tokens=2048,
            stop_token_ids=stop_token_ids,
        )

    # Two-step uses default [[A]]/[[B]] parsing; no model-specific modifier
    model_modifier = None

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=get_conv_template("raw"),  # not used in this script (handled later)
        custom_dialogue_formatting=True,  # handle formatting later
        tokenizer=None,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
        max_turns=4,
    )

    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    if is_api_models:
        ############################
        # Run inference via API (two-step: rate each 0-5, then [[A]]/[[B]])
        ############################
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_judgement(batch, debug=args.debug):
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
                winner, request, judgement = run_judge_two_step(
                    prompt, answer_a, answer_b, args.model, multi_turn=mult_turn, model_modifier=model_modifier
                )
                if debug:
                    print(f"Request (step2): {request.get('step2_user_prompt', '')[:200]}...")
                    print(f"Judgement: {judgement}")

                if winner == winner_text:
                    return 1
                elif winner == loser_text:
                    return 0
                else:  # if "error"
                    return 0.5  # effectively a tie
            else:
                return 0.5

        # if debug, do not multi-thread
        if args.debug:
            num_threads = 1
        else:
            num_threads = args.num_threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(get_judgement, x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()
    else:
        ############################
        # Run model weights with vllm (two-step: 3 batched passes over dataset)
        ############################
        def _messages_to_prompt(system_content, user_content):
            messages = [
                {"role": "system", "content": system_content or ""},
                {"role": "user", "content": user_content},
            ]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

        def _batch_generate(prompts_list):
            """Run one batched generate; prompts_list is list of prompt strings."""
            if model_modifier == "Atla":
                prompt_token_ids = [
                    tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts_list
                ]
                outputs = model.generate(
                    prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
                )
            else:
                outputs = model.generate(prompts_list, sampling_params=sampling_params)
            return [o.outputs[0].text for o in outputs]

        # Precompute per-example: prompt, answer_a, answer_b, is_shuffled
        rows = []
        for i in range(len(dataset)):
            batch = dataset[i]
            mult_turn = len(batch["text_chosen"]) > 2
            prompt = batch["text_chosen"][0]["content"]
            answer_a = list(batch["text_chosen"])
            answer_b = list(batch["text_rejected"])
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
            rows.append({
                "prompt": prompt,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "is_shuffled": is_shuffled,
                "mult_turn": mult_turn,
            })

        # Pass 1: batch rate assistant A (0–5) for all examples
        logger.info("*** Run inference: pass 1/3 (rate A) ***")
        prompts_rating_a = [
            get_rating_0_5_user_prompt(r["prompt"], r["answer_a"], r["mult_turn"]) for r in rows
        ]
        prompts_a_full = [_messages_to_prompt("", u) for u in prompts_rating_a]
        analyses_a = _batch_generate(prompts_a_full)

        # Pass 2: batch rate assistant B (0–5) for all examples
        logger.info("*** Run inference: pass 2/3 (rate B) ***")
        prompts_rating_b = [
            get_rating_0_5_user_prompt(r["prompt"], r["answer_b"], r["mult_turn"]) for r in rows
        ]
        prompts_b_full = [_messages_to_prompt("", u) for u in prompts_rating_b]
        analyses_b = _batch_generate(prompts_b_full)

        # Pass 3: batch step-2 verdict ([[A]] / [[B]]) using analyses
        logger.info("*** Run inference: pass 3/3 (verdict) ***")
        step2_prompts = []
        for r, ana_a, ana_b in zip(rows, analyses_a, analyses_b):
            sys_s, user_s = format_judge_from_analyses(
                r["prompt"], r["answer_a"], r["answer_b"], ana_a, ana_b, r["mult_turn"]
            )
            step2_prompts.append(_messages_to_prompt(sys_s or "", user_s))
        judgments = _batch_generate(step2_prompts)

        def process_shuffled(win, shuffle):
            winner_text = "B" if shuffle else "A"
            loser_text = "A" if shuffle else "B"
            if win == winner_text:
                return 1
            if win == loser_text:
                return 0
            return 0.5

        winners = [process_judgement(j, None) for j in judgments]
        results = [process_shuffled(w, r["is_shuffled"]) for w, r in zip(winners, rows)]
        logger.info("*** Inference done ***")

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    model_name = args.model
    # if model in openai or Anthropic list, append org to model name
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

    ############################
    # Upload results to hub
    #############################
    # Two-step results not comparable to default; save locally unless --do_not_save
    do_not_save = args.do_not_save

    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    results_url = save_to_hub(
        results_grouped,
        model_name,
        sub_path,
        args.debug,
        local_only=do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
        save_postfix=getattr(args, "save_postfix", ""),
    )
    if not do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    logger.info("Two-step judge: individual 0-5 analyses + pairwise [[A]]/[[B]] verdict")

    ############################
    # Save per-prompt results to hub
    ############################
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = model_name
    scores_dict["model_type"] = model_type

    sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    scores_url = save_to_hub(
        scores_dict, model_name, sub_path_scores, args.debug, local_only=args.do_not_save,
        save_postfix=getattr(args, "save_postfix", ""),
    )
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()
