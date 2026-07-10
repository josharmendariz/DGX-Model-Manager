#!/bin/bash
# Name: Qwen3-Next 80B A3B (NVFP4)
# Description: 48G weights @ util 0.55 — 32K ctx, prefix caching, /mnt/models
# VRAM: 67
#
# Flags ported from proven k3s deployment (dgx-stack/k3s/vllm-q80.yaml):
# NVFP4 auto-detected from config.json — do NOT pass --quantization.
set -euo pipefail

docker rm -f vllm_node 2>/dev/null || true

exec docker run --name vllm_node --gpus all -p 8000:8000 \
  -v /mnt/models:/models:ro \
  -e HF_HUB_OFFLINE=1 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  vllm/vllm-openai:cu130-nightly \
  --model /models/qwen3-next-80b-a3b-nvfp4 \
  --served-model-name qwen3-next-80b vllm-active \
  --host 0.0.0.0 --port 8000 \
  --moe-backend flashinfer_cutlass \
  --gpu-memory-utilization 0.55 \
  --max-model-len 32768 --max-num-seqs 4 \
  --kv-cache-dtype fp8 --enable-prefix-caching
