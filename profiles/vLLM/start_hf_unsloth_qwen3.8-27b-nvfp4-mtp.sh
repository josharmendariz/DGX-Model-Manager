#!/bin/bash
# Name: HF unsloth/Qwen3.8-27B-NVFP4 (MTP)
# Description: NVFP4 Qwen3.8-27B with MTP speculation + prefix caching — fastest at long context
# VRAM: 85
#
# Same launch shape as the FP8 profile, different weights. Measured head to head
# on gb10 2026-09-08, both with prefix caching and depth-4 MTP:
#
#                        FP8      NVFP4
#   4k   single-stream   16.44    21.81   tok/s
#   32k  single-stream   16.49    28.17
#   128k single-stream    9.77    21.54
#   aggregate @ 8 seqs   83.31    87.67
#   TTFT 32k / 128k      10.8s / 116.0s   10.2s / 116.9s   (no difference)
#   KV pool              769k     1,680k  tokens
#
# The advantage grows with context — 1.33x at 4k, 2.20x at 128k — which is the
# shape that matters here, because observed KV usage has a p90 of 131k tokens.
# It buys nothing on TTFT: prefill is bound by the Triton/FLA GDN kernel, which
# runs above FP4 regardless of how the weights are stored.
#
# Quality vs the FP8 checkpoint, on the checks that have ground truth:
#   verifiable answers  10/10 both      executable code   5/5 both
#   well-formed tool calls  5/5 both, with identical call shapes
# Output drift could NOT be measured: this engine does not reproduce its own
# output on long prose at temperature 0 (NVFP4 against itself scored 1/8 exact
# and 0.128 prefix agreement), because continuous batching and speculative
# decoding change reduction order between runs. Any FP8-vs-FP4 text difference is
# dominated by that nondeterminism, so it is not evidence either way. If you need
# stronger assurance than 20 ground-truth tasks, run a shadow period rather than
# trusting a text-similarity score.
set -euo pipefail

SNAP_HOST=$(ls -d "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/"*/ 2>/dev/null | head -1)
if [ -z "${SNAP_HOST}" ]; then
  echo "ERROR: Qwen3.8-27B-NVFP4 snapshot not found in HF cache" >&2
  echo "       hf download unsloth/Qwen3.8-27B-NVFP4" >&2
  exit 1
fi
HASH=$(basename "${SNAP_HOST}")
MODEL="/root/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/${HASH}"

# This repo carries its own MTP head as model_mtp.safetensors; no extra download.
#
# Depth 4 is the top of the ladder, confirmed 2026-09-08 by measuring past it:
#           4k      32k     128k    agg@8   tokens/step @32k
#   depth 4 21.81   28.17   21.54   87.67   3.88
#   depth 5 19.92   30.47   19.13   74.86   4.49
#   depth 6 21.42   25.76   19.19   76.06   4.06
# The speculator itself keeps improving — tokens per step rises monotonically —
# but past depth 4 the extra draft compute costs more than the extra accepted
# tokens return, and 128k TTFT degrades (116.9s -> 139.5s) as the larger draft
# slot count eats the same 2048-token scheduling budget that
# --max-num-batched-tokens collides with. Depth 5 wins only at 32k single-stream.
SPEC_DEPTH="${SPEC_DEPTH:-4}"

# Same draft-slot constraint as the FP8 profile, and NVFP4 is not exempt: at
# seqs 16 with depth 4 speculation vanished and per-stream fell to 1.43 tok/s,
# an 8x collapse against the same config at seqs 8. Depth 2 at seqs 16 survives
# (71.3% acceptance), so it is the product that binds, not the model.
MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
if [ "${SPEC_DEPTH}" != "0" ] && [ $(( MAX_SEQS * (SPEC_DEPTH + 1) )) -gt 48 ]; then
  echo "WARN: ${MAX_SEQS} seqs x depth ${SPEC_DEPTH} = $(( MAX_SEQS * (SPEC_DEPTH + 1) )) draft slots." >&2
  echo "      Above ~48 vLLM silently drops the MTP draft model. Verify after launch:" >&2
  echo "        curl -s localhost:8000/metrics | grep spec_decode_num_accepted" >&2
fi

UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
CTX="${VLLM_MAX_MODEL_LEN:-262144}"

docker rm -f vllm_node 2>/dev/null || true

if [[ "${SKIP_DROP_CACHES:-0}" != "1" ]]; then
  sync && (echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null) || \
    echo "WARN: could not drop page cache; KV budget may be short" >&2
fi

# Serve the name of the model that is actually loaded. An earlier revision of this
# profile advertised the FP8 ids for "compatibility", which was wrong twice over:
# nothing depends on them (litellm routes only on vllm-active, and no file in
# agent-control, opencode.json or this repo mentions the FP8 string), and it made
# /v1/models report a checkpoint the box was not running — exactly the drift the
# nightly image checker exists to catch.
ARGS=(
  --model "${MODEL}"
  --served-model-name "unsloth/Qwen3.8-27B-NVFP4" "unsloth--Qwen3.8-27B-NVFP4" vllm-active
  --host 0.0.0.0 --port 8000
  --trust-remote-code --dtype auto
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
  --generation-config vllm
  --max-model-len "${CTX}"
  --gpu-memory-utilization "${UTIL}"
  --max-num-seqs "${MAX_SEQS}"
  --enable-prefix-caching
  --enable-chunked-prefill
)
[ "${SPEC_DEPTH}" != "0" ] && ARGS+=( --speculative-config \
  "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${SPEC_DEPTH}}" )

# Do NOT add --max-num-batched-tokens here. vLLM warns that its 2048 clamp is
# suboptimal and suggests raising it; on this box that advice is wrong in both
# quantisations. See the notes in the FP8 profile for the numbers.
exec docker run -d --name vllm_node --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  --label "dgx.profile=hf_unsloth_qwen3.8-27b-nvfp4-mtp" \
  --label "dgx.engine=vllm" \
  eugr/spark-vllm:latest \
  vllm serve "${ARGS[@]}"
