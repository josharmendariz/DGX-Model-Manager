#!/bin/bash
# Name: HF Qwen/Qwen3.6-35B-A3B-FP8
# Description: Tuned solo recipe via run-recipe.sh (FP8, util 0.55, 256K ctx)
# VRAM: 67
#
# RECIPE-BACKED PROFILE — deliberately not an auto-generated `docker run`.
#
# This model has a hand-measured recipe at
# ~/spark-vllm-docker/recipes/qwen3.6-35b-a3b-fp8-solo.yaml, and the generated
# form cannot reproduce it. The recipe applies mods/fix-qwen3.6-chat-template
# *inside* the container and passes flags the generator does not emit
# (--load-format fastsafetensors, --attention-backend flashinfer,
# --reasoning-parser qwen3, --enable-prefix-caching, --max-num-batched-tokens).
# Replicating those inline would fork the truth and silently drift. So this
# profile delegates, and the recipe stays the single source of truth.
#
# Why util 0.55 and not the generator's 0.75: utilization is a hard reservation
# of a fraction of ALL 121 GB of unified memory, and it is the only lever —
# lowering --max-model-len frees nothing. 0.8 measured 92.3 G allocated and left
# the host at 4 GB available with swap engaged, while k3s kept scheduling burst
# limits against memory it cannot see this container holding. See the recipe
# header for the full measurement.
#
# The `docker rm -f` below is load-bearing, not boilerplate: run-recipe.sh checks
# for a container named vllm_node and prints "Cluster containers are already
# running. Skipping launch." if it finds one — including a *crash-looping* one.
# That is exactly how a broken profile took the box's default model offline
# across a reboot on 2026-08-04.
set -euo pipefail

RECIPE_DIR="$HOME/spark-vllm-docker"
RECIPE="qwen3.6-35b-a3b-fp8-solo"

docker rm -f vllm_node 2>/dev/null || true

cd "$RECIPE_DIR"
# -d is required: without it run-recipe stays attached streaming container logs.
# Progress is read from `docker logs -f vllm_node` instead, which works
# identically for recipe-backed and generated profiles.
exec ./run-recipe.sh "$RECIPE" -d
