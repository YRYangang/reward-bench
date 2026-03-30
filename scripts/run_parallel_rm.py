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
from transformers import AutoTokenizer

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    save_to_hub,
    torch_dtype_mapping,
)
from rewardbench.models.urm import (
    apply_template,
    tokenize_fn,
    ParallelDataCollatorForPreference,
    Qwen3ForGenerativeRewarding,
)
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
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
    parser.add_argument(
        "--use_judge_role",
        action="store_true",
        help="Whether to use judge role",
    )
    parser.add_argument(
        "--use_judge_token",
        action="store_true",
        help="Whether to use judge token",
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

    # load chat template
    if args.chat_template is not None:
        logger.info(f"Loading chat template {args.chat_template}")
        conv = get_conv_template(args.chat_template)
    else:
        logger.info("No chat template provided, using default")
        conv = None

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

    custom_dialogue = config["custom_dialogue"]
    model_type = config["model_type"]
    if args.standard:
        model_builder = AutoModelForSequenceClassification
    else:
        model_builder = (
            Qwen3ForGenerativeRewarding if args.use_judge_role else AutoModelForTokenClassification
        )
    model_builder = model_builder.from_pretrained
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)

    # if not datatype in config (default), check args
    # if torch_dtype is None:
    #     # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
    #     if args.torch_dtype == torch.bfloat16:
    #         quantized = False
    #         logger.info("Disabling quantization for bfloat16 datatype")
    #     torch_dtype = args.torch_dtype

    # not included in config to make user explicitly understand they are passing this
    trust_remote_code = args.trust_remote_code

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        tokenizer=tokenizer,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
    )

    def _func(example):

        return tokenize_fn(
            apply_template(
                example,
                use_ver_role=args.use_ver_role,
                use_ver_token=args.use_ver_token,
                use_judge_role=args.use_judge_role,
                use_judge_token=args.use_judge_token,
                chosen_id_placeholder_token=tokenizer.convert_tokens_to_ids("<|object_ref_start|>"),
                rejected_id_placeholder_token=tokenizer.convert_tokens_to_ids("<|object_ref_end|>"),
            ),
            tokenizer,
        )

    dataset = dataset.map(
        _func,
        desc="Tokenizing dataset",
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
        use_judge_role=args.use_judge_role,
        use_judge_token=args.use_judge_token,
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

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    else:
        logger.info("*** Running dataloader to collect results ***")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=ParallelDataCollatorForPreference(
                tokenizer=tokenizer,
                parallel_context=args.parallel_context,
                standard=args.standard,
                mode="judge" if args.use_judge_role else "verifier",
                chosen_id_placeholder_token="<|object_ref_start|>",
                rejected_id_placeholder_token="<|object_ref_end|>",
            ),
            shuffle=False,
            drop_last=False,
        )

        dataloader, model = accelerator.prepare(dataloader, model)

        results = []
        scores_chosen = []
        scores_rejected = []
        for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
            results_sub = reward_pipe(batch)
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
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = (
        args.chat_template if not check_tokenizer_chat_template(tokenizer) else "tokenizer"
    )

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
    if not model_type == "Custom Classifier":  # custom classifiers do not return scores
        # create new json with scores and upload
        scores_dict = out_dataset.to_dict()
        scores_dict["model"] = args.model
        scores_dict["model_type"] = model_type
        scores_dict["chat_template"] = args.chat_template

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
    else:
        logger.info("Not uploading chosen-rejected text with scores due to model compatibility")


if __name__ == "__main__":
    main()
