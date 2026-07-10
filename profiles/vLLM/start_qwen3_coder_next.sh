#!/bin/bash
# Name: Qwen3-Coder-Next (NVFP4)
# Description: 43G weights @ util 0.55 — 32K ctx, qwen3_coder tools, /mnt/models
# VRAM: 67
#
# Flags ported from proven k3s deployment (dgx-stack/k3s/vllm-codernext.yaml).
# k3s ran util 0.40 to co-run with qwen-14b; standalone on :8000 we can use 0.55.
# NVFP4 via compressed-tensors, auto-detected — do NOT pass --quantization.
set -euo pipefail

docker rm -f vllm_node 2>/dev/null || true

exec docker run --name vllm_node --gpus all -p 8000:8000 \
  -v /mnt/models:/models:ro \
  -e HF_HUB_OFFLINE=1 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  vllm/vllm-openai:cu130-nightly \
  --model /models/qwen3-coder-next-nvfp4 \
  --served-model-name qwen3-coder-next \
  --host 0.0.0.0 --port 8000 \
  --moe-backend flashinfer_cutlass \
  --gpu-memory-utilization 0.55 \
  --max-model-len 32768 --max-num-seqs 4 \
  --kv-cache-dtype fp8 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
