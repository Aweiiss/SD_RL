#!/bin/bash

PROJECT_DIR=/home/k00914150/new_verl/verl
MODEL_PATH=/home/k00914150/Temp/Spec-RL/model
SAVE_PATH=/home/k00914150/new_verl/verl
PROJECT_NAME=test_baseline_first_run
CUDA_VISIBLE_DEVICES=6,7

# for vllm 0.9.1: uncomment to fix oov issue
# sudo bash ${PROJECT_DIR}/training_scripts/fix_vllm_oov.sh ${PROJECT_DIR}

cd ${PROJECT_DIR}
export WORKING_DIR=${PROJECT_DIR}


bash ${PROJECT_DIR}/training_scripts/train_grpo.sh \
    --dataset_name deepmath \
    --train_file_name train_sample_6144 \
    --model_name Qwen3-1.7B-base \
    --model_path ${MODEL_PATH} \
    --max_response_length 4096 \
    --train_batch_size 1024 \
    --rollout_n 8 \
    --rollout_gpu_memory_util 0.4 \
    --rollout_tp 1 \
    --total_epochs 20 \
    --total_steps 100 \
    --rollout_name vllm \
    --save_freq 10 \
    --checkpoint_path ${SAVE_PATH} \
    --project_name ${PROJECT_NAME}