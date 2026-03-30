#!/usr/bin/env zsh


###### 2-STEPS with base ######
MODEL_PATH="/data/models/Qwen3-4B"
EXTRA=(--disable_beaker_save --enable-thinking --do_not_save --num_gpus=2)
python run_generative_two_step_v2.py --model="$MODEL_PATH" "${EXTRA[@]}" --save-postfix="-v2-not-score"
python run_generative_two_step.py --model="$MODEL_PATH" "${EXTRA[@]}" --save-postfix="-v1-not-score"

###### base #######
MODEL_PATH="/data/models/Qwen3-4B"
EXTRA=(--disable_beaker_save --do_not_save --num_gpus=2)
CUDA_VISIBLE_DEVICES=0,1 python run_generative_v2.py --model="$MODEL_PATH" "${EXTRA[@]}" --save-postfix="-v2"


MODEL_PATH="/data/models/Qwen3-4B"
EXTRA=(--disable_beaker_save --do_not_save --num_gpus=2)
CUDA_VISIBLE_DEVICES=3,5 python run_generative.py --model="$MODEL_PATH" "${EXTRA[@]}"



###### RM ######

MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-rm-retry"
EXTRA=(--disable_beaker_save --do_not_save)
srun --gres=gpu:1 --qos=interactive python run_parallel_rm.py --model="$MODEL_PATH" "${EXTRA[@]}" --standard --batch_size=32


MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-rm-ver-noscorebias"
EXTRA=(--disable_beaker_save --do_not_save --batch_size=32 --use_ver_role --use_ver_token --parallel_context)
srun --gres=gpu:1 --qos=interactive python run_parallel_rm.py --model="$MODEL_PATH" "${EXTRA[@]}" 


MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-parallel"
EXTRA=(--disable_beaker_save --do_not_save)
CUDA_VISIBLE_DEVICES=4 python run_parallel_rm.py --model="$MODEL_PATH" "${EXTRA[@]}" --batch_size=1 --parallel_context



MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-rm-ver-2gpu"
EXTRA=(--disable_beaker_save --do_not_save --use_ver_token --use_ver_role --parallel_context)
srun --gres=gpu:1 --qos=interactive python run_parallel_rm.py --model="$MODEL_PATH" "${EXTRA[@]}" --standard --batch_size=32

###### Generative RM with VLLM ######
MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-var-think-causal"
EXTRA=(--max_new_tokens=4096  --num_gpus=2)
PYTHONPATH=/data/runyang/URM/evaluations/reward-bench CUDA_VISIBLE_DEVICES=3,4,5 python run_parallel_rm_think_vllm.py --model="$MODEL_PATH" "${EXTRA[@]}"

###### Generative RM ######
MODEL_PATH="/data_new/runyang/URM_checkpoints/qwen3-4b-var-think"
EXTRA=(--disable_beaker_save --do_not_save --use_ver_token --batch_size=4)
CUDA_VISIBLE_DEVICES=3 python run_parallel_rm_think.py --model="$MODEL_PATH" "${EXTRA[@]}"


# MODEL_PATH="/data_new/runyang/URM_checkpoints/qwen3-4b-rm-judge-fixed"
# EXTRA=(--disable_beaker_save --do_not_save --use_judge_role --batch_size=8)
# CUDA_VISIBLE_DEVICES=4 python run_parallel_rm.py --model="$MODEL_PATH" "${EXTRA[@]}"

# MODEL_PATH="/data_new/runyang/URM_checkpoints/qwen3-4b-rm-judge-token"
# EXTRA=(--disable_beaker_save --do_not_save --use_judge_role --use_judge_token --batch_size=8)
# CUDA_VISIBLE_DEVICES=5 python run_parallel_rm.py --model="$MODEL_PATH" "${EXTRA[@]}"


###### Judge RM ######
MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-rm-judge-v2-newid-bs4"
EXTRA=(--disable_beaker_save --do_not_save --parallel_context)
CUDA_VISIBLE_DEVICES=0 python run_judge_rm.py --model="$MODEL_PATH" "${EXTRA[@]}"


MODEL_PATH="/data/runyang/URM_checkpoints/qwen3-4b-rm-judge-v2-nomask"
EXTRA=(--disable_beaker_save --do_not_save)
srun --gres=gpu:1 --qos=interactive python run_judge_rm.py --model="$MODEL_PATH" "${EXTRA[@]}"

# bug: remove everything after <think> to enable thinking (now it's generating after nabracadabra <\think>)