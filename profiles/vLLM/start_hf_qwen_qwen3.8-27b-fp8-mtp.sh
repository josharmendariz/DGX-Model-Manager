#!/bin/bash
# Name: HF Qwen/Qwen3.8-27B-FP8 (MTP)
# Description: Qwen3.8-27B-FP8 with the checkpoint's own MTP head + prefix caching
# VRAM: 92
#
# Supersedes the plain FP8 launch. Measured on gb10 2026-09-08 over a 15-config
# exclusive sweep (harness and raw JSON in ~/vllm-sweep/ on the box):
#
#                     per-stream 4k   per-stream 32k   aggregate @8   TTFT 32k
#   previous config        7.85            7.34            11.44        19.18s
#   this config           16.44           16.49            83.31        10.78s
#
# Decode on this box is memory-bandwidth-bound, not compute-bound: the model is
# dense 27B, so ~27.4 GB of weights is read per decoded token and GB10 has
# 273 GB/s. That caps a conventional decode at ~10 tok/s, and the previous
# config was already achieving 79% of peak bandwidth. The only way past the wall
# is to emit more than one token per weight read, which is what MTP does.
set -euo pipefail

SNAP_HOST=$(ls -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/"*/ 2>/dev/null | head -1)
if [ -z "${SNAP_HOST}" ]; then
  echo "ERROR: Qwen3.8-27B-FP8 snapshot not found in HF cache" >&2
  exit 1
fi
HASH=$(basename "${SNAP_HOST}")
MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/${HASH}"

# The MTP head ships inside this repo as mtp.safetensors and needs no separate
# download. vLLM 0.23.1 lists qwen3_5_mtp in MTPModelTypes, so the draft model is
# loaded from the same snapshot. Depth 4 measured 3.22 tokens per engine step at
# a 56.1% acceptance rate; depth 3 gave 2.93 and depth 2 gave 2.21, so the ladder
# had not saturated at the depth this settled on.
SPEC_DEPTH="${SPEC_DEPTH:-4}"

# Concurrency and draft depth cannot both be raised. What binds is the draft-slot
# product, MAX_SEQS x (SPEC_DEPTH + 1), against the 2048-token scheduling budget
# vLLM clamps to when speculation is on. Measured 2026-09-08:
#   seqs  8, depth 4  -> 40 slots, 56.1% acceptance, 16.44 tok/s   (this config)
#   seqs 16, depth 2  -> 48 slots, 71.3% acceptance, 20.90 tok/s   (survives)
#   seqs 16, depth 4  -> 80 slots, speculation GONE,  1.43 tok/s   (collapses)
#   seqs 16/32, depth 4 on FP8 -> speculation GONE, 1.01 tokens per step
# When it fails, vLLM logs no error at all: the draft model is simply never
# loaded, and the only symptoms are tokens-per-step falling to 1.0 and a KV pool
# larger by exactly the draft model's footprint.
MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
if [ "${SPEC_DEPTH}" != "0" ] && [ $(( MAX_SEQS * (SPEC_DEPTH + 1) )) -gt 48 ]; then
  echo "WARN: ${MAX_SEQS} seqs x depth ${SPEC_DEPTH} = $(( MAX_SEQS * (SPEC_DEPTH + 1) )) draft slots." >&2
  echo "      Above ~48 vLLM silently drops the MTP draft model and throughput" >&2
  echo "      collapses. Verify tokens-per-step > 1 after launch:" >&2
  echo "        curl -s localhost:8000/metrics | grep spec_decode_num_accepted" >&2
fi

# 0.49 (the previous value) left ~60 GB of the 128 GB idle while the host sat at
# 2 GB free with 50 GB in page cache. 0.90 went too far the other way: the KV
# pool reached 1.22M tokens, the box went 6.4 GB into swap and every measurement
# aborted. 0.72 gives ~776k tokens of KV, far more than 8 streams can use.
UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"

# max-model-len is NOT a memory lever here — in vLLM V1 the KV pool is sized by
# gpu-memory-utilization, so shortening the context frees nothing. Kept at the
# full 256K because observed KV usage has a p90 of 131k tokens and a p99 of 188k;
# cutting it would start rejecting real agent-control traffic.
CTX="${VLLM_MAX_MODEL_LEN:-262144}"

docker rm -f vllm_node 2>/dev/null || true

# Same page-cache reasoning as the llama.cpp profile: GB10 unified memory bills
# the host page cache against the GPU budget.
if [[ "${SKIP_DROP_CACHES:-0}" != "1" ]]; then
  sync && (echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null) || \
    echo "WARN: could not drop page cache; KV budget may be short" >&2
fi

ARGS=(
  --model "${MODEL}"
  --served-model-name "Qwen/Qwen3.8-27B-FP8" "Qwen--Qwen3.8-27B-FP8" vllm-active
  --host 0.0.0.0 --port 8000
  --trust-remote-code --dtype auto
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
  --generation-config vllm
  --max-model-len "${CTX}"
  --gpu-memory-utilization "${UTIL}"
  --max-num-seqs "${MAX_SEQS}"
  # Was off, at a 0.0% hit rate for the whole week before this was measured.
  # Agent prompts share a large role preamble, and prefill dominates the request
  # at this box's real context lengths: TTFT at 32k fell 20.3s -> 10.2s.
  --enable-prefix-caching
  --enable-chunked-prefill
)
[ "${SPEC_DEPTH}" != "0" ] && ARGS+=( --speculative-config \
  "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${SPEC_DEPTH}}" )

# Detached and deliberately without --restart, for the same reason as every other
# profile here: a restart policy on a launch that may be wrong turns a bad script
# into a persistent outage.
exec docker run -d --name vllm_node --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  --label "dgx.profile=hf_qwen_qwen3.8-27b-fp8-mtp" \
  --label "dgx.engine=vllm" \
  eugr/spark-vllm:latest \
  vllm serve "${ARGS[@]}"

# ── Notes ─────────────────────────────────────────────────────────────────────
# Correctness: speculative decoding is output-identical under greedy sampling, and
# that was checked rather than assumed — depths 2, 3 and 4 each reproduced the
# non-speculative baseline exactly on five deterministic prompts (5/5). A mismatch
# would have meant a broken kernel on this dev build, not a speed/quality trade.
#
# Ruled out on 2026-09-08, so they don't get re-tried:
#   --max-num-batched-tokens  vLLM warns that its 2048 clamp is "suboptimal" and
#                      recommends raising it. On this box that advice is wrong:
#                      8192/16384/32768 made TTFT worse monotonically (32k:
#                      10.8s -> 19.5/19.3/22.5s; 128k: 116s -> 187/185/214s) and
#                      cut aggregate throughput from 83.3 to 63.8/56.7/31.2.
#                      Two causes: big prefill chunks head-of-line block the
#                      decode queue, and the Triton/FLA GDN kernel is less
#                      efficient at large chunks. Leave it clamped.
#   ngram speculation  1.07 tokens/step (accepts nothing here) and hurts concurrency
#   suffix decoding    unsupported on this build; the container exits immediately
#   FLASHINFER backend marginally slower than the default FLASH_ATTN on every cell
#
# Not yet adopted: unsloth/Qwen3.8-27B-NVFP4, already in the HF cache. Benchmarked
# properly on 2026-09-08 with this same prefix-caching + depth-4 configuration:
#     4k 21.81 tok/s (vs 16.44 here) · 32k 28.17 (vs 16.49) · 128k 21.54 (vs 9.77)
#     aggregate @8 87.67 (vs 83.31) · KV pool 1.68M tokens (vs 769k)
# The advantage grows with context, which matters because observed KV usage has a
# p90 of 131k tokens. It gives nothing on TTFT — prefill is bound by the GDN
# kernel, not by weight precision.
# Quality: identical to FP8 on every check with ground truth — 10/10 verifiable,
# 5/5 executable code, 5/5 well-formed tool calls, same tool-call shapes. Output
# drift against FP8 could NOT be measured: this engine does not reproduce its own
# output on long prose at temperature 0 (FP4-vs-itself scored 1/8 exact, 0.128
# prefix agreement), so any FP8-vs-FP4 divergence is dominated by continuous-batch
# nondeterminism rather than by quantisation.
