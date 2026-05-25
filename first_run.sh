CUDA_VISIBLE_DEVICES=0,1,2,7\
NNODES=1 NGPUS_PER_NODE=4 \
INFER_BACKEND=vllm ROLLOUT_N=8 TRAIN_BATCH_SIZE=512 \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
# MODEL_PATH=/home/k00914150/Temp/Spec-RL/model \