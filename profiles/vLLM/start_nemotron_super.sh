#!/bin/bash
# Name: Nemotron-3 Super 120B (NVFP4)
# Description: 69.5G weights + bounded KV at util 0.75 -- native 262K ctx, tool calling
# VRAM: 95
#
# GB10 profile verified 2026-07-14: vLLM 0.20.0, marlin NVFP4 backend.
# Serves "nvidia/nemotron-3-super", "nemotron-3-super", and "vllm-active" on :8000.
set -euo pipefail

docker rm -f vllm_node 2>/dev/null || true

exec docker run --name vllm_node --restart unless-stopped --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  vllm/vllm-openai:v0.20.0 \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --served-model-name nvidia/nemotron-3-super nemotron-3-super vllm-active \
  --host 0.0.0.0 --port 8000 \
  --async-scheduling --dtype auto --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --trust-remote-code --gpu-memory-utilization 0.75 \
  --enable-chunked-prefill --max-num-seqs 2 --max-model-len 262144 \
  --moe-backend marlin --mamba_ssm_cache_dtype float16 \
  --quantization fp4 \
  --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
  --reasoning-parser super_v3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --generation-config vllm
