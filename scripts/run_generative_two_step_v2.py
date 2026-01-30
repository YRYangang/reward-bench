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

# Two-step generative RM for RewardBench 2: (1) rate each candidate 0-5; (2) verdict [[A]]/[[B]]/[[C]]/[[D]].
# Non-Ties: 4-way two-step. Ties: ratings-based (same as run_generative_v2).
# Examples:
# python scripts/run_generative_two_step_v2.py --model gpt-3.5-turbo
# python scripts/run_generative_two_step_v2.py --model=claude-3-haiku-20240307

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
from datasets import concatenate_datasets
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rewardbench import load_eval_dataset_multi, process_single_model, save_to_hub
from rewardbench.script_args import add_common_generative_args
from rewardbench.generative import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_from_analyses_four,
    get_rating_0_5_user_prompt,
    process_judgement_four,
    run_judge_two_step_four,
)
from rewardbench.generative_v2 import (
    get_single_rating,
    run_judge_ratings_multi,
)

# get token from HF_TOKEN env variable
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login
    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    parser = argparse.ArgumentParser()
    add_common_generative_args(parser, dataset=True, score_w_ratings=False)
    parser.add_argument(
        "--step1-thinking-only",
        action="store_true",
        default=False,
        help="Only use thinking for step 1.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    logger.info(f"Running two-step reward model (v2) on {args.model} with chat template {args.chat_template}")

    model_type = "Generative RM (two-step v2)"
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        logger.warning("Two-step does not support ensemble; using first model only.")
        args.model = args.model[0]

    if args.force_local:
        is_api_models = False
    else:
        is_api_models = args.model in API_MODEL_LIST

    model_modifier = None
    if not is_api_models:
        if args.num_gpus > 1:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.vllm_gpu_util,
            disable_custom_all_reduce=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        stop_token_ids = [128009] if ("Llama-3" in args.model or "llama3-8b" in args.model) and "3.1" not in args.model else None
        
        default_sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=2048, stop_token_ids=stop_token_ids)

        sampling_params_step1 = default_sampling_params
        if args.enable_thinking and args.step1_thinking_only:
            eot_token_id = tokenizer.convert_tokens_to_ids("</think>")
            stop_token_ids = [eot_token_id]
            sampling_params_step1 = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=2048, stop_token_ids=stop_token_ids)

    ############################
    # Load dataset (RB2: Ties + non-Ties)
    ############################
    logger.info("*** Load dataset ***")
    dataset = load_eval_dataset_multi(
        core_set=not args.pref_sets,
        dataset=args.dataset,
        conv=get_conv_template("raw"),
        custom_dialogue_formatting=True,
        tokenizer=None,
        logger=logger,
        keep_columns=["texts_chosen", "texts_rejected", "id", "subset", "num_correct"],
        max_turns=4,
    )
    ties_dataset = dataset.filter(lambda example: example["subset"] == "Ties")
    dataset = dataset.filter(lambda example: example["subset"] != "Ties")
    nonties_ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    if args.debug:
        dataset = dataset.select(range(10))
        ties_dataset = ties_dataset.select(range(10))
        nonties_ids = nonties_ids[:10]

    if is_api_models:
        ############################
        # API: non-Ties (4-way two-step)
        ############################
        def update_progress_bar(done, total):
            progress = int(50 * done / total)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_judgement_non_ties(batch, debug=args.debug):
            mult_turn = len(batch["texts_chosen"][0]) > 2
            prompt = batch["texts_chosen"][0][0]["content"]
            answer_a = batch["texts_chosen"][0]
            answer_b = batch["texts_rejected"][0]
            answer_c = batch["texts_rejected"][1]
            answer_d = batch["texts_rejected"][2]
            shuffle_option = np.random.randint(0, 4)
            if shuffle_option == 1:
                answer_a, answer_b = answer_b, answer_a
            elif shuffle_option == 2:
                answer_a, answer_c = answer_c, answer_a
            elif shuffle_option == 3:
                answer_a, answer_d = answer_d, answer_a
            options = ["A", "B", "C", "D"]
            winner_text = options.pop(shuffle_option)
            loser_texts = options

            if len(batch["texts_chosen"][0]) <= 4:
                winner, request, judgement = run_judge_two_step_four(
                    prompt, answer_a, answer_b, answer_c, answer_d,
                    args.model, multi_turn=mult_turn, model_modifier=model_modifier,
                )
                if debug:
                    print(f"Judgement: {judgement}")
                if winner == winner_text:
                    return 1
                if winner in loser_texts:
                    return 0
            return 0.25

        logger.info("*** Run inference on non-ties (4-way two-step) ***")
        results = [None] * len(dataset)
        done_tasks = 0
        with ThreadPoolExecutor(max_workers=1 if args.debug else args.num_threads) as executor:
            future_to_index = {executor.submit(get_judgement_non_ties, x): i for i, x in enumerate(dataset)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                done_tasks += 1
                update_progress_bar(done_tasks, len(dataset))
        print()

        ############################
        # API: Ties (ratings)
        ############################
        logger.info("*** Run inference on Ties subset (ratings) ***")
        results_ties = [None] * len(ties_dataset)
        done_tasks = 0
        def get_judgement_ties(batch):
            prompt = batch["texts_chosen"][0][0]["content"]
            answers = batch["texts_chosen"] + batch["texts_rejected"]
            winners, requests, info = run_judge_ratings_multi(
                prompt, answers, args.model, multi_turn=len(batch["texts_chosen"][0]) > 2,
                model_modifier=model_modifier, is_ties=True,
            )
            return info["ratings"]

        with ThreadPoolExecutor(max_workers=1 if args.debug else args.num_threads) as executor:
            future_to_index = {executor.submit(get_judgement_ties, x): i for i, x in enumerate(ties_dataset)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results_ties[index] = future.result()
                done_tasks += 1
                update_progress_bar(done_tasks, len(ties_dataset))
        print()
    else:
        ############################
        # vLLM: non-Ties (4-way two-step, batched)
        ############################
        def _messages_to_prompt(system_content, user_content):
            messages = [
                {"role": "system", "content": system_content or ""},
                {"role": "user", "content": user_content},
            ]
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking,
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        def _batch_generate(prompts_list, sampling_params):
            if model_modifier == "Atla":
                prompt_token_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts_list]
                outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
            else:
                outputs = model.generate(prompts_list, sampling_params=sampling_params)
            return [o.outputs[0].text for o in outputs]

        rows = []
        for i in range(len(dataset)):
            batch = dataset[i]
            mult_turn = len(batch["texts_chosen"][0]) > 2
            prompt = batch["texts_chosen"][0][0]["content"]
            answer_a = list(batch["texts_chosen"][0])
            answer_b = list(batch["texts_rejected"][0])
            answer_c = list(batch["texts_rejected"][1])
            answer_d = list(batch["texts_rejected"][2])
            shuffle_option = np.random.randint(0, 4)
            if shuffle_option == 1:
                answer_a, answer_b = answer_b, answer_a
            elif shuffle_option == 2:
                answer_a, answer_c = answer_c, answer_a
            elif shuffle_option == 3:
                answer_a, answer_d = answer_d, answer_a
            rows.append({
                "prompt": prompt, "answer_a": answer_a, "answer_b": answer_b, "answer_c": answer_c, "answer_d": answer_d,
                "shuffle_option": shuffle_option, "mult_turn": mult_turn,
            })

        # Pass 1: rate A,B,C,D for all examples (4*N prompts)
        logger.info("*** Run inference: pass 1/2 (rate A,B,C,D) ***")
        all_rating_prompts = []
        for r in rows:
            for key in ("answer_a", "answer_b", "answer_c", "answer_d"):
                u = get_rating_0_5_user_prompt(r["prompt"], r[key], r["mult_turn"])
                all_rating_prompts.append(_messages_to_prompt("", u))
        all_analyses = _batch_generate(all_rating_prompts, sampling_params_step1)
        n = len(rows)
        analyses_a = [all_analyses[i * 4 + 0] for i in range(n)]
        analyses_b = [all_analyses[i * 4 + 1] for i in range(n)]
        analyses_c = [all_analyses[i * 4 + 2] for i in range(n)]
        analyses_d = [all_analyses[i * 4 + 3] for i in range(n)]

        # Pass 2: step-2 verdict ([[A]]/[[B]]/[[C]]/[[D]])
        logger.info("*** Run inference: pass 2/2 (verdict) ***")
        step2_prompts = []
        for r, ana_a, ana_b, ana_c, ana_d in zip(rows, analyses_a, analyses_b, analyses_c, analyses_d):
            sys_s, user_s = format_judge_from_analyses_four(
                r["prompt"], r["answer_a"], r["answer_b"], r["answer_c"], r["answer_d"],
                ana_a, ana_b, ana_c, ana_d, r["mult_turn"],
            )
            step2_prompts.append(_messages_to_prompt(sys_s or "", user_s))
        judgments = _batch_generate(step2_prompts, default_sampling_params)

        def process_shuffled(win, shuffle_option):
            options = ["A", "B", "C", "D"]
            winner_text = options.pop(shuffle_option)
            loser_texts = options
            if win == winner_text:
                return 1
            if win in loser_texts:
                return 0
            return 0.25

        winners = [process_judgement_four(j) for j in judgments]
        results = [process_shuffled(w, r["shuffle_option"]) for w, r in zip(winners, rows)]
        logger.info("*** Inference done (non-ties) ***")

        ############################
        # vLLM: Ties (ratings per answer)
        ############################
        def format_ratings(batch):
            mult_turn = len(batch["texts_chosen"][0]) > 2
            prompt = batch["texts_chosen"][0][0]["content"]
            all_answers = batch["texts_chosen"] + batch["texts_rejected"]
            batch["prompt"] = prompt
            batch["answers"] = [a[1]["content"] for a in all_answers]
            batch["mult_turn"] = mult_turn
            return batch

        vllm_model_dict = {
            "model": model,
            "tokenizer": tokenizer,
            "sampling_params": default_sampling_params,
            "chat_template": get_conv_template(args.chat_template) if args.chat_template else None,
        }
        logger.info("*** Run inference on Ties subset (ratings) ***")
        ties_dataset_formatted = ties_dataset.map(format_ratings)
        results_ties = []
        for batch_idx, batch in enumerate(ties_dataset_formatted):
            prompt = batch["prompt"]
            mult_turn = batch["mult_turn"]
            ratings = []
            for ans_idx, answer_text in enumerate(batch["answers"]):
                if (prompt or "").strip() == "" or (answer_text or "").strip() == "":
                    logger.warning(
                        f"Ties batch {batch_idx} answer {ans_idx}: empty prompt or answer, skipping vLLM call (rating=-1)"
                    )
                    ratings.append(-1)
                    continue
                rating, _ = get_single_rating(
                    question_text=prompt,
                    answer_text=answer_text,
                    model=args.model,
                    model_modifier=model_modifier,
                    is_ties=True,
                    vllm_model=vllm_model_dict,
                )
                ratings.append(rating)
            results_ties.append(ratings)

    ############################
    # Print & process results
    ############################
    out_dataset = dataset.add_column("results", results)
    out_dataset = out_dataset.add_column("id", nonties_ids)
    out_dataset_ties = ties_dataset.add_column("scores", results_ties)
    out_dataset_ties, ties_score = process_single_model(out_dataset_ties)
    out_dataset = concatenate_datasets([out_dataset, out_dataset_ties], axis=0)

    model_name = args.model
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name

    results_grouped = {}
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template
    present_subsets = np.unique(out_dataset["subset"])
    logger.info(f"Present subsets: {present_subsets}")
    for subset in present_subsets:
        if subset.lower() == "ties":
            print(f"{subset}: Ties score: {ties_score}")
            results_grouped[subset] = ties_score
        else:
            subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

    sub_path = "eval-set/"
    results_url = save_to_hub(
        results_grouped,
        model_name,
        sub_path,
        args.debug,
        local_only=args.do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
        best_of_n=True,
        save_postfix=getattr(args, "save_postfix", ""),
    )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = model_name
    scores_dict["model_type"] = model_type
    sub_path_scores = "eval-set-scores/"
    scores_url = save_to_hub(
        scores_dict, model_name, sub_path_scores, args.debug, local_only=args.do_not_save, best_of_n=True,
        save_postfix=getattr(args, "save_postfix", ""),
    )
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()
