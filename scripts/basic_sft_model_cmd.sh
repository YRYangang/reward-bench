#!/usr/bin/env zsh



MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-basic-sft-discriminative-no-thinking"
EXTRA=(--disable_beaker_save --do_not_save --paradigm="discriminative" --batch_size=1)
srun --gres=gpu:1 --qos=interactive python run_basic_sft_model.py --model="$MODEL_PATH" "${EXTRA[@]}"


MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-basic-sft-generative-no-thinking-1w"
EXTRA=(--disable_beaker_save --do_not_save --paradigm="generative" --batch_size=1)
# CUDA_VISIBLE_DEVICES=5 python run_basic_sft_model.py --model="$MODEL_PATH" "${EXTRA[@]}"
srun --gres=gpu:1 --qos=interactive python run_basic_sft_model.py --model="$MODEL_PATH" "${EXTRA[@]}"