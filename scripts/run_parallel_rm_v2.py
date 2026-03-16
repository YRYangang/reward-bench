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
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

from rewardbench import (
    REWARD_MODEL_CONFIG,
    load_eval_dataset_multi,
    process_single_model,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.models.urm import (
    apply_template_multiple,
    tokenize_fn,
    ParallelDataCollatorForMultiplePreference,
    ParallelRMRewardBenchMultiplePipeline,
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
        default=2048,
        help="Max length of RM inputs (passed to pipeline)",
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
        "--parallel_context",
        action="store_true",
        help="Whether to use parallel context",
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Whether to share the prompt",
    )
    parser.add_argument(
        "--use_ver_token",
        action="store_true",
        help="Whether to use verifier token",
    )
    parser.add_argument(
        "--use_ver_role",
        action="store_true",
        help="Whether to use verifier role",
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

    config = REWARD_MODEL_CONFIG["parallelRM"]
    logger.info(f"Using reward model config: {config}")

    quantized = config["quantized"]  # only Starling isn't quantized for now
    # if llama-3 in name, switch quantized to False (severely degrades performance)
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
        or args.not_quantized
    ):
        quantized = False
        logger.info(
            f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})"
        )

    model_type = config["model_type"]
    if args.standard:
        model_builder = AutoModelForSequenceClassification.from_pretrained
    else:
        model_builder = AutoModelForTokenClassification.from_pretrained
    pipeline_builder = ParallelRMRewardBenchMultiplePipeline
    torch_dtype = config.get("torch_dtype", None)

    # if not datatype in config (default), check args
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dataset = load_eval_dataset_multi(
        dataset="allenai/reward-bench-2",
        core_set=not args.pref_sets,
        conv=None,
        custom_dialogue_formatting=True,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["texts_chosen", "texts_rejected", "id", "subset", "num_correct"],
        # max_turns=4,
    )

    def _func(example):
        return tokenize_fn(
            apply_template_multiple(
                example, use_ver_role=args.use_ver_role, use_ver_token=args.use_ver_token
            ),
            tokenizer,
        )

    dataset = dataset.map(_func, desc="Tokenizing dataset")
    # copy id for saving, then remove
    ties_ids = dataset.filter(lambda example: example["subset"] == "Ties")["id"]

    # separate dataset into dataset for non-ties and ties_dataset for ties based on "subset" == "Ties"
    ties_dataset = dataset.filter(lambda example: example["subset"] == "Ties")
    dataset = dataset.filter(lambda example: example["subset"] != "Ties")
    nonties_ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        ties_dataset = ties_dataset.select(range(10))
        ties_ids = ties_ids[:10]  # add ties ids to ties_ids
        nonties_ids = nonties_ids[:10]  # add ties ids to ids

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    if quantized:
        model_kwargs = {
            "load_in_8bit": True,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
    else:
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch_dtype,
        }

    # if attn_implementation is not specified, this falls back to Hugging Face's default
    # strategy (which chooses between sdpa and eager depending on pytorch version)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        standard=args.standard,
        parallel_context=args.parallel_context,
        use_ver_role=args.use_ver_role,
        use_ver_token=args.use_ver_token,
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

    logger.info("*** Running dataloader on non-ties dataset ***")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=ParallelDataCollatorForMultiplePreference(
            pad_token_id=tokenizer.pad_token_id,
            parallel_context=args.parallel_context,
            standard=args.standard,
        ),
        shuffle=False,
        drop_last=False,
    )

    dataloader, model = accelerator.prepare(dataloader, model)

    results = []
    scores_chosen = []
    scores_rejected = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        _result = reward_pipe(batch)
        max_reward_indice = torch.argmax(_result, dim=1)  # (B,)
        correct = (max_reward_indice == 0).int().tolist()
        results.extend(correct)
        # results.extend(results_sub["results"])
        # scores_chosen.extend(results_sub["rewards_chosen"])
        # scores_rejected.extend(results_sub["rewards_rejected"])

    logger.info("*** Running dataloader on ties dataset ***")
    dataloader = torch.utils.data.DataLoader(
        ties_dataset,
        batch_size=1,
        collate_fn=ParallelDataCollatorForMultiplePreference(
            pad_token_id=tokenizer.pad_token_id,
            parallel_context=args.parallel_context,
            standard=args.standard,
        ),
        shuffle=False,
        drop_last=False,
    )
    dataloader, model = accelerator.prepare(dataloader, model)

    results_ties = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        _result = reward_pipe(batch)
        results_ties.extend(_result.tolist())

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)
    # out_dataset = out_dataset.add_column("id", nonties_ids)

    # process results for ties, then merge datasets
    out_dataset_ties = ties_dataset.add_column("scores", results_ties)
    # out_dataset_ties = out_dataset_ties.add_column("id", ties_ids)
    out_dataset_ties, ties_score = process_single_model(out_dataset_ties)

    out_dataset = concatenate_datasets([out_dataset, out_dataset_ties], axis=0)

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = args.model
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template

    # print per subset and log into results_grouped file
    present_subsets = np.unique(out_dataset["subset"])
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
        best_of_n=True,
        save_postfix=getattr(args, "save_postfix", ""),
    )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    logger.info("Not uploading chosen-rejected text with scores due to model compatibility")

    # upload chosen-rejected with scores

    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = args.model
    scores_dict["model_type"] = model_type

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
