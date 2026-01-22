#!/bin/bash
eval "$(conda shell.bash hook)"
env_name="LLM-inference"
conda activate $env_name


# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# vllm serve /data/hf_cache/models/Qwen/Qwen3-32B --host=127.0.0.1 --port=8030 --api-key="token-abc123"  --tensor-parallel-size 8 --max-model-len 4096 --dtype auto --gpu-memory-utilization 0.7
# export CUDA_VISIBLE_DEVICES=0,1


# vllm serve /data/hf_cache/models/Qwen/Qwen3-32B \
#   --host=127.0.0.1 --port=8085 \
#   --api-key="token-abc123" \
#   --tensor-parallel-size 4 \
#   --max-model-len 32768 \
#   --max-num-batched-tokens 32768 \
#   --gpu-memory-utilization 0.9

# vllm serve /data/hf_cache/models/Qwen/Qwen3-32B --host=127.0.0.1 --port=8085 --api-key="token-abc123"  --tensor-parallel-size 4 --max-model-len 40000 --dtype auto --gpu-memory-utilization 0.9 --max-num-seqs 128


vllm serve /data/hf_cache/models/Qwen/Qwen3-30B-A3B-Thinking-2507 --host=127.0.0.1 --port=8085 --api-key="token-abc123"  --tensor-parallel-size 4 --max-model-len 65536 --dtype auto --gpu-memory-utilization 0.9 --max-num-seqs 128

# vllm serve /data/hf_cache/models/Qwen/Qwen3-30B-A3B-Instruct-2507 --host=127.0.0.1 --port=8085 --api-key="token-abc123"  --tensor-parallel-size 4 --max-model-len 65536 --dtype auto --gpu-memory-utilization 0.9

# /data/hf_cache/models/Qwen/Qwen3-235B-A22B-Instruct-2507
# vllm serve /data/hf_cache/models/Qwen/Qwen3-235B-A22B-Instruct-2507 --host=127.0.0.1 --port=8085 --api-key="token-abc123"  --tensor-parallel-size 8 --max-model-len 40000 --dtype auto --gpu-memory-utilization 0.8