#!/bin/bash
# Name: Nemotron-3 Nano 30B (NVFP4)
# Description: 18G weights + 21G KV cache @ util 0.35 — 256K ctx, tool calling
# VRAM: 42
#
# Proven config (2026-07-10): cu130-nightly, marlin NVFP4 backend.
# Serves as "nemotron-3-nano" on :8000. Weights from HF cache (offline).
set -euo pipefail

docker rm -f vllm_node 2>/dev/null || true

exec docker run --name vllm_node --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:cu130-nightly \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --served-model-name nemotron-3-nano vllm-active \
  --host 0.0.0.0 --port 8000 \
  --async-scheduling --dtype auto --kv-cache-dtype fp8 \
  --trust-remote-code --gpu-memory-utilization 0.35 \
  --enable-chunked-prefill --max-num-seqs 4 --max-model-len 262144 \
  --moe-backend marlin --mamba_ssm_cache_dtype float32 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
