#!/bin/bash
# Name: HF nvidia/Qwen3.8-Flash-Next-NVFP4 (hybrid, MTP)
# Description: Qwen3.8-Flash-Next MoE (125B total / 6B active) via blazux/qwen3.8-Flash-DGX,
#   hybrid fp8 side-layers + NVFP4 experts, MTP=2, 500k context via YaRN
# VRAM: 97
#
# Replaces the dense Qwen3.8-27B-NVFP4-MTP profile as the box's default. Benchmarked
# head-to-head on gb10 2026-09-17, same corpus/harness as that profile's own sweep:
#
#                      dense Qwen3.8-27B NVFP4+MTP4   this (Flash-Next hybrid+MTP2)
#   4k decode                  22.2 tok/s                    39.7 tok/s   (1.8x)
#   32k decode                 21.6 tok/s                    34.0 tok/s   (1.6x)
#   128k decode                22.5 tok/s                    39.3 tok/s   (1.75x)
#   128k TTFT (cold)          110.1 s                         67.2 s      (1.6x faster)
#   aggregate @8 (4k)          62.5 tok/s                    100.9 tok/s  (1.6x)
#   native ctx                262144                          262144, extended to 500k
#                                                              via YaRN (needle-in-haystack
#                                                              validated to 414k upstream)
#
# Every measurement here is at MTP on, prefix caching on, greedy/deterministic — same
# methodology as the dense profile. The first 128k TTFT measurement looked absurdly fast
# (1.5s) because a prior failed test run had already prefix-cache-warmed the exact same
# deterministic corpus; re-run with a fresh seed gave the 67.2s figure above, which is
# the one to trust.
#
# THE ARCHITECTURE IS WHY THIS WINS: hybrid Gated DeltaNet (linear attention) on 3 of
# every 4 layers + Qwen Sparse Attention, a 512-expert MoE (10 active/token, only 6B
# active params vs the dense model's 27B), plus a separate 51B-parameter n-gram
# embedding table. Fewer active bytes read per decoded token is the whole story on this
# bandwidth-bound box (GB10: 273 GB/s). Quality per NVIDIA's own published eval table
# (NVFP4 vs FP8 baseline, essentially tied): GPQA Diamond 91.5, HLE 35.4, Terminal-Bench
# 2.1 82.9. Also vision+video capable (untested here beyond architecture inspection).
#
# THIS DOES NOT RUN ON STOCK vLLM. The architecture (Qwen4ExpForConditionalGeneration /
# qwen4_exp) is not in the model registry of any vLLM build available as of 2026-09-17,
# including this box's own eugr/spark-vllm:latest and its newest nightly (checked: 0 of
# 355 registered archs match). NVIDIA's own reference command additionally calls for
# --tensor-parallel-size 8 on B200/B300 — an 8-GPU datacenter config, not this single-GPU
# GB10. What makes this profile possible is a third-party community project,
# github.com/blazux/qwen3.8-Flash-DGX (272 stars at review time, Apache-2.0, actively
# maintained, patches individually attributed and traceable to upstream vLLM PR/issue
# numbers), which:
#   - builds on an OFFICIAL preview image (vllm/vllm-openai:qwen38-flash-next, pinned by
#     digest) that DOES register qwen4_exp — the box's generic image just isn't that one
#   - solves the memory problem: the 51B-param n-gram ("PLE") table is served from NVMe
#     via mmap instead of held resident, dropping weights from ~125 GiB to ~75 GiB so it
#     fits next to a usable KV cache in the 128 GB unified pool
#   - fixes a prefix-caching bug specific to this architecture on GB10 (a block_size
#     mismatch silently restored an all-zero Mamba state on cache hits) and a
#     non-deterministic top-k in the sparse-attention kernel
#   - runs at TP=1 on a single GB10 despite NVIDIA's 8-GPU reference command — this
#     project is the only reason that works at all
# Reviewed before use: Apache-2.0 license, all external fetches pinned by commit SHA
# AND verified by sha256 checksum in the Dockerfile, every patch gated behind an opt-in
# env var (a no-op with the flag off) and self-verifies at build time. Cloned to
# ~/qwen3.8-Flash-DGX on this box; this script is a thin wrapper around its own
# scripts/serve.sh (the "default" profile: MODE=hybrid YARN=1 CTX=500000 MTP=2), not a
# reimplementation — the serve script's own patch-application, YaRN math, MTP+YaRN
# interaction fix, and effort-alias chat-template rewrite are all nontrivial and already
# validated; hand-rolling them here would just be a second, untested copy.
#
# Local edits to that repo's scripts/serve.sh (both one-line, both additive, no-op for
# upstream users): --served-model-name gained "vllm-active" (the alias litellm forwards
# to; nothing else in agent-control or this repo names the upstream "qwen3.8-flash-next"
# id directly) and the docker run gained dgx.profile / dgx.engine labels, matching every
# other profile in this directory.
set -euo pipefail

FLASH_REPO="$HOME/qwen3.8-Flash-DGX"
if [ ! -x "$FLASH_REPO/scripts/serve.sh" ]; then
  echo "ERROR: $FLASH_REPO not found (clone github.com/blazux/qwen3.8-Flash-DGX there)" >&2
  exit 1
fi
if ! docker image inspect qwen38-flash-dgx >/dev/null 2>&1; then
  echo "ERROR: image 'qwen38-flash-dgx' not built. Run: (cd $FLASH_REPO && docker build -t qwen38-flash-dgx .)" >&2
  exit 1
fi
HYBRID_SNAP="$(ls -d "$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"*-fp8hybrid 2>/dev/null | head -1)"
if [ -z "$HYBRID_SNAP" ]; then
  echo "ERROR: hybrid checkpoint not prepared. Run: (cd $FLASH_REPO && ./scripts/prepare-hybrid.sh)" >&2
  exit 1
fi

# Same page-cache reasoning as every other profile here: GB10 unified memory bills the
# host page cache against the GPU budget.
if [[ "${SKIP_DROP_CACHES:-0}" != "1" ]]; then
  sync && (echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null) || \
    echo "WARN: could not drop page cache; KV budget may be short" >&2
fi

# serve.sh does its own `docker rm -f "$NAME"`; no need to duplicate it here.
# --restart is deliberately stripped after launch (see the two dense-model profiles'
# notes): a restart policy on a launch that may be wrong turns a bad script into a
# persistent outage. vllm-default-model.service owns boot recovery instead.
cd "$FLASH_REPO"
NAME=vllm_node PORT=8000 IMAGE=qwen38-flash-dgx MODE=hybrid YARN=1 CTX=500000 MTP=2 \
  GPU_MEM=0.80 ./scripts/serve.sh
docker update --restart no vllm_node
