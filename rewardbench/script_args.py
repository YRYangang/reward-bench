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

"""Shared argument parsers for reward-bench generative evaluation scripts."""

import argparse


def add_common_generative_args(
    parser: argparse.ArgumentParser,
    *,
    dataset: bool = False,
    score_w_ratings: bool = False,
) -> None:
    """
    Add arguments common to generative RM scripts (run_generative*.py, run_generative_two_step*.py).

    Args:
        parser: ArgumentParser to add arguments to.
        dataset: If True, add --dataset (for RewardBench 2 / multi-dataset scripts).
        score_w_ratings: If True, add --score_w_ratings (for single-step generative scripts).
    """
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="model to use",
    )
    if dataset:
        parser.add_argument(
            "--dataset",
            type=str,
            default="allenai/reward-bench-2",
            help="path to huggingface dataset",
        )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="fastchat chat template (optional)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="directly load model instead of pipeline",
    )
    if score_w_ratings:
        parser.add_argument(
            "--score_w_ratings",
            action="store_true",
            default=False,
            help="score with ratings instead of pairwise ranking",
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
        "--do_not_save",
        action="store_true",
        help="do not save results to hub",
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
        "--disable_beaker_save",
        action="store_true",
        help="disable saving to /output for AI2 Beaker",
    )
    parser.add_argument(
        "--force_local",
        action="store_true",
        default=False,
        help="force local run, even if model is on API provider list",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Pass enable_thinking=True to tokenizer.apply_chat_template (if supported).",
    )
    parser.add_argument(
        "--save-postfix",
        type=str,
        default="",
        help="postfix to add to the save directory",
    )
