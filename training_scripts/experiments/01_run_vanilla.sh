#!/bin/bash
# =============================================================================
# Config 1: Vanilla GRPO (no speculative decoding) - OOM Optimized
# =============================================================================
set -xeuo pipefail

PROJECT_DIR="/home/lingquh1xx/L2598/Temp/verl"
MODEL_PATH="/home/lingquh1xx/L2598/Temp/verl/model/Qwen3-1.7B-base"
SAVE_PATH="/home/lingquh1xx/L2598/Temp/verl/checkpoints"
DATA_PATH="/home/lingquh1xx/L2598/Temp/verl/data"
NUM_GPU=4
PROJECT_NAME="spec_rl_compare"
EXPERIMENT_NAME="vanilla"
ROLLOUT_DATA_DIR="${PROJECT_DIR}/rollouts/${EXPERIMENT_NAME}"

export CUDA_VISIBLE_DEVICES=0,2,5,6
export WORKING_DIR=${PROJECT_DIR}

cd ${PROJECT_DIR}
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="${DATA_PATH}/deepmath/train_sample_6144.parquet" \
    data.val_files="${DATA_PATH}/deepmath/test.parquet" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NUM_GPU} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=0 \
    trainer.val_before_train=False \
    trainer.total_epochs=2 \
    +trainer.total_training_steps=96 \
    trainer.default_local_dir="${SAVE_PATH}/${EXPERIMENT_NAME}" \
    +trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    +trainer.spec_decoding=False \
    +trainer.spec_bias=0.0 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_penalty=kl \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    2>&1 | tee "${PROJECT_DIR}/logs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"