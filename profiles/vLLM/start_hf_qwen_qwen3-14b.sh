#!/bin/bash
# Name: HF Qwen/Qwen3-14B
# Description: Qwen3-14B dense BF16 via vLLM (helix test model, ~30 GB on disk)
# VRAM: 34
#
# helix candidate. eugr/spark-vllm:latest (SM121-correct vLLM 0.23.1). Dense BF16
# does not touch the Marlin MXFP4 path, but this is the known-good GB10 image.
# Snapshot hash resolved at launch from the HF cache.
set -euo pipefail

SNAP_HOST=$(ls -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/"*/ 2>/dev/null | head -1)
if [ -z "${SNAP_HOST}" ]; then
  echo "ERROR: Qwen3-14B snapshot not found in HF cache — run ~/download-qwen3.sh first" >&2
  exit 1
fi
HASH=$(basename "${SNAP_HOST}")
MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/${HASH}"

docker rm -f vllm_node 2>/dev/null || true

exec docker run -d --name vllm_node --restart unless-stopped --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  --label "dgx.profile=hf_qwen_qwen3-14b" \
  --label "dgx.engine=vllm" \
  eugr/spark-vllm:latest \
  vllm serve \
  --model "${MODEL}" \
  --served-model-name "Qwen/Qwen3-14B" "Qwen--Qwen3-14B" vllm-active \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --dtype auto \
  --gpu-memory-utilization 0.75 \
  --max-model-len 32768 --max-num-seqs 8 \
  --enable-chunked-prefill
