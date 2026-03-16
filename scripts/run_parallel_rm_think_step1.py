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
import shutil

from transformers import AutoTokenizer

from rewardbench import load_eval_dataset
from rewardbench.models.urm import formatting_fn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model to use",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="number of gpus to use, for multi-node vllm",
    )
    parser.add_argument(
        "--vllm_gpu_util",
        type=float,
        default=0.9,
        help="gpu utilization for vllm",
    )
    parser.add_argument(
        "--pref_sets",
        action="store_true",
        help="run on common preference sets",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode (e.g. smaller subset)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=10,
        help="number of threads for parallel processing of examples",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max length of RM inputs (passed to pipeline)",
    )
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

    logger.info(f"Running reward model {args.model}")

    # if model isn't API, load via vllm

    from vllm import LLM, SamplingParams

    # if multi gpu, set multiproc method to spawn
    if args.num_gpus > 1:
        # Set the environment variable
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # load model
    model = LLM(
        args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.vllm_gpu_util,
        disable_custom_all_reduce=True,
        # max_seq_length=args.vllm_max_seq_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    eot_token_id = tokenizer.convert_tokens_to_ids("</think>")
    default_sampling_params = SamplingParams(
        n=1,
        temperature=0,
        top_p=1,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[eot_token_id],
    )

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=None,  # not used in this script
        custom_dialogue_formatting=True,  # handle formatting later
        tokenizer=None,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
        # max_turns=4,
    )
    dataset = dataset.map(
        formatting_fn,
        fn_kwargs={"processing_class": tokenizer, "thinking_prompt": True},
        desc="Tokenizing dataset",
    )
    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.add_column("subset", subsets)

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    ############################
    # Run model weights with vllm (two-step: 3 batched passes over dataset)
    ############################

    # Pass 1: batch rate assistant A (0–5) for all examples
    _root_path = "./results/cache"
    os.makedirs(_root_path, exist_ok=True)
    target_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    _path = f"{_root_path}/{target_path}{args.model}"
    # if not empty, raise warning and add a postfix to the path
    if os.path.exists(_path) and os.listdir(_path):
        raise Warning(f"Path {_path} is not empty, saving to {_path}_1 instead")
        _path = f"{_path}_1"
    os.makedirs(_path, exist_ok=True)
    logger.info("*** Run inference on chosen inputs***")
    chosen_outputs = model.generate(dataset["chosen_input"], sampling_params=default_sampling_params)
    thinking = [o.outputs[0].text for o in chosen_outputs]
    dataset = dataset.add_column("chosen_thinking", thinking)
    dataset.save_to_disk(_path)
    logger.info("*** Run inference on rejected inputs***")
    rejected_outputs = model.generate(dataset["rejected_input"], sampling_params=default_sampling_params)
    thinking = [o.outputs[0].text for o in rejected_outputs]
    dataset = dataset.add_column("rejected_thinking", thinking)
    # remove the previously saved file (use shutil to remove)
    shutil.rmtree(_path)
    dataset.save_to_disk(_path)
    logger.info(f"Saved results to {_path}")
    logger.info("*** Done ***")
    if args.debug:
        import rich

        rich.print(dataset[-1])


if __name__ == "__main__":
    main()
