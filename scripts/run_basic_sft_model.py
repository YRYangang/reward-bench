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

import argparse
import logging
import os
import sys

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewardbench import (
    REWARD_MODEL_CONFIG,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.models.basic_sft_model import (
    BasicSFTJudgePipeline,
    Qwen3ForGenerativeRewarding,
)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from rewardbench.script_args import add_common_generative_args

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    add_common_generative_args(parser, dataset=True, score_w_ratings=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max length of RM inputs (passed to pipeline); omit for no truncation",
    )
    parser.add_argument(
        "--not_quantized",
        action="store_true",
        help="disable quantization for models that are quantized by default",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: float16)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use (default: None)",
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        default="generative",
        choices=["generative", "discriminative"],
        help="The paradigm to use for training.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Whether to enable thinking",
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()
    # --model has nargs="+" in common args; normalize to single path for this script
    if isinstance(args.model, list):
        if len(args.model) != 1:
            raise ValueError("run_rm.py expects a single model path; got multiple.")
        args.model = args.model[0]
    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # Generative basic-SFT training uses AutoModelForCausalLM; discriminative uses Qwen3ForGenerativeRewarding.
    model_cls = Qwen3ForGenerativeRewarding if args.paradigm == "discriminative" else AutoModelForCausalLM
    model_builder = model_cls.from_pretrained
    pipeline_builder = BasicSFTJudgePipeline

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=None,
        custom_dialogue_formatting=True,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )

    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    ############################
    # Load reward model pipeline
    ############################
    batch_size = args.batch_size
    logger.info("*** Load reward model ***")

    model_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": args.trust_remote_code,
    }

    model = model_builder(args.model, **model_kwargs)
    if args.debug:
        model.generation_config.max_new_tokens = 10
    # model.generation_config.max_new_tokens = 500
    model.generation_config.do_sample = False
    reward_pipe = pipeline_builder(
        model=model,
        tokenizer=tokenizer,
        paradigm=args.paradigm,
        enable_thinking=args.enable_thinking,
        max_length=args.max_length,
    )

    ############################
    # Tokenization settings & dataset preparation
    ############################
    # set pad token to eos token if not set
    if reward_pipe.tokenizer.pad_token_id is None:
        logger.info(f"Setting pad token to eos token: {reward_pipe.tokenizer.eos_token_id}")
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    # For models whose config did not contains `pad_token_id`
    if reward_pipe.model.config.pad_token_id is None:
        logger.info(f"Setting pad token to {reward_pipe.tokenizer.pad_token_id}")
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    model = accelerator.prepare(model)

    results = []
    scores_chosen = []
    scores_rejected = []
    for step, start in enumerate(tqdm(range(0, len(dataset), batch_size), desc="RM batch steps")):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        results_sub = reward_pipe(inputs=batch)
        results.extend(results_sub["results"])
        scores_chosen.extend(results_sub["rewards_chosen"])
        scores_rejected.extend(results_sub["rewards_rejected"])

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = args.model

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    results_leaderboard = None
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

    ############################
    # Upload results to hub
    ############################
    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    results_url = save_to_hub(
        results_grouped,
        args.model,
        sub_path,
        args.debug,
        local_only=args.do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
        save_postfix=getattr(args, "save_postfix", ""),
    )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    if results_leaderboard is not None:
        sub_path_leaderboard = "eval-set-leaderboard/"
        leaderboard_url = save_to_hub(
            results_leaderboard,
            args.model,
            sub_path_leaderboard,
            args.debug,
            local_only=args.do_not_save,
            save_metrics_for_beaker=not args.disable_beaker_save,
            save_postfix=getattr(args, "save_postfix", ""),
        )
        if not args.do_not_save:
            logger.info(f"Uploaded leaderboard results to {leaderboard_url}")

    # upload chosen-rejected with scores
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = args.model

    sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    scores_url = save_to_hub(
        scores_dict,
        args.model,
        sub_path_scores,
        args.debug,
        local_only=args.do_not_save,
        save_postfix=getattr(args, "save_postfix", ""),
    )
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()
