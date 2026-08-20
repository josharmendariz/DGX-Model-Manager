#!/bin/bash
# Name: GGUF HauhauCS/Qwen3.8-27B-Uncensored-Aggressive-MTP
# Description: llama.cpp GGUF + embedded MTP speculative decoding + BF16 vision projector
# VRAM: 31
#
# Why llama.cpp and not vLLM: this repo ships GGUF only (no HF safetensors/config.json).
# vLLM's GGUF path is experimental, single-file, needs --tokenizer pointed at a separate
# HF repo, has no IQ-quant kernels, no mmproj/vision, and no MTP head support — i.e. it
# would drop the three things this release exists for. See the notes at the bottom.
#
# One script, whole quant ladder. Settings come from config.json llamacpp.recipes, so the
# ladder is data the UI can enumerate rather than a script per quant:
#   RECIPE=quality ./start_gguf_hauhaucs_qwen3.8-27b-aggressive-mtp.sh
#   ./start_gguf_hauhaucs_qwen3.8-27b-aggressive-mtp.sh --list-recipes
# Individual env vars still win over the recipe, for one-off experiments:
#   RECIPE=balanced CTX=8192 ./start_...sh
set -euo pipefail

# Path is expressed relative to the HF cache root so the same string resolves on both
# sides of the mount: the host copy is checked for existence, the container copy is what
# llama-server is told to open.
REL="hub/models--HauhauCS--Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/snapshots/993a5971fda8f30dd1b7eb2654792ba4415c7460"
HOST_CACHE="$HOME/.cache/huggingface"
CT_CACHE="/root/.cache/huggingface"
HOST_SNAP="$HOST_CACHE/$REL"
CT_SNAP="$CT_CACHE/$REL"
BASE="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive"
CONFIG="${DMM_CONFIG:-$HOME/DGX-Model-Manager/config.json}"

# One reader for both --list-recipes and the launch path, so the listing can never drift
# from what actually gets used.
_recipes() { python3 - "$CONFIG" "$@" <<'PY'
import json, os, sys
cfg = json.load(open(os.path.expanduser(sys.argv[1])))
lc = cfg.get("llamacpp", {})
recipes = lc.get("recipes", {})
mode = sys.argv[2]
if mode == "list":
    width = max((len(k) for k in recipes), default=0)
    for k, r in recipes.items():
        print(f"  {k.ljust(width)}  {r['quant']:<8} ctx={r['ctx']:<7} spec={r['spec']:<8} "
              f"vision={int(bool(r.get('vision', True)))}  ~{r['vram_gb']} GB")
        print(f"  {' '*width}  {r.get('desc','')}")
    sys.exit(0)
name = sys.argv[3] or lc.get("default_recipe", "balanced")
if name not in recipes:
    sys.exit(f"ERROR: unknown recipe {name!r}; have: {', '.join(recipes)}")
r = recipes[name]
print(f"R_NAME={name}")
print(f"R_QUANT={r['quant']}")
print(f"R_CTX={r['ctx']}")
print(f"R_SPEC={r['spec']}")
print(f"R_VISION={int(bool(r.get('vision', True)))}")
# fastmtp needs the patched build; the recipe choice, not the operator, selects the image.
key = "fastmtp_image" if r["spec"] == "fastmtp" else "image"
print(f"R_IMAGE={lc.get(key, 'ghcr.io/ggml-org/llama.cpp:server-cuda')}")
print(f"R_NAME_CT={lc.get('container_name', 'llamacpp_node')}")
PY
}

if [[ "${1:-}" == "--list-recipes" ]]; then
  echo "llamacpp recipes in $CONFIG:"; _recipes list; exit 0
fi

# Capture before eval: `eval "$(cmd)"` discards cmd's exit status, so a bad recipe name
# would otherwise surface as a bewildering "R_QUANT: unbound variable".
_RESOLVED="$(_recipes use "${RECIPE:-}")" || exit 1
eval "$_RESOLVED"

QUANT="${QUANT:-$R_QUANT}"
CTX="${CTX:-$R_CTX}"
# embedded  = NextN head baked into every target GGUF, works on the stock image.
# fastmtp   = the 32K sidecar; REQUIRES an image built from HauhauCS-FastMTP-llama.cpp.patch,
#             otherwise the draft load fails with
#             "expected 5120, 248320, got 5120, 32768". Confirmed 2026-08-20 — see notes.
SPEC="${SPEC:-$R_SPEC}"
# Vision costs ~0.9 GB and the projector is BF16 regardless of target quant.
VISION="${VISION:-$R_VISION}"
NGL="${NGL:-all}"
PORT="${PORT:-8081}"
# The image's HEALTHCHECK is hardcoded to `curl -f http://localhost:8080/health`, so the
# container must serve on 8080 internally or docker reports it unhealthy forever while it
# is in fact serving. Publish on $PORT, listen on 8080.
CT_PORT=8080
DEPTH="${DEPTH:-3}"
IMAGE="${LLAMACPP_IMAGE:-$R_IMAGE}"
CT_NAME="${CT_NAME:-$R_NAME_CT}"

[[ -f "$HOST_SNAP/$BASE-$QUANT.gguf" ]] || {
  echo "ERROR: no such quant: $HOST_SNAP/$BASE-$QUANT.gguf" >&2; exit 1; }

# A CPU-only image accepts -ngl and silently ignores it, serving ~100x slow with no error
# — exactly what the first bare-metal dry run on 2026-08-18 hit. Fail loudly instead.
# Costs one throwaway container (~2s) and has already caught this once.
if [[ "$NGL" != "0" ]] && ! docker run --rm --gpus all --entrypoint /app/llama-server \
     "$IMAGE" --list-devices 2>&1 | grep -qi "cuda\|gpu"; then
  echo "ERROR: $IMAGE reports no GPU backend." >&2
  echo "       Verify the image is the CUDA variant and that --gpus all works:" >&2
  echo "         docker run --rm --gpus all --entrypoint /app/llama-server \\" >&2
  echo "           $IMAGE --list-devices" >&2
  echo "       Or set NGL=0 to accept CPU-only inference deliberately." >&2
  exit 1
fi

echo "recipe=$R_NAME quant=$QUANT ctx=$CTX spec=$SPEC vision=$VISION image=$IMAGE" >&2

docker rm -f "$CT_NAME" 2>/dev/null || true

# Same page-cache reasoning as the vLLM profiles: GB10 unified memory bills the page
# cache against the GPU budget, so a warm cache from the previous model steals KV.
# SKIP_DROP_CACHES=1 for side-by-side runs, where evicting the *other* engine's cache is
# a needless tax.
if [[ "${SKIP_DROP_CACHES:-0}" != "1" ]]; then
  sync && (echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null) || \
    echo "WARN: could not drop page cache; KV budget may be short" >&2
fi

ARGS=(
  --model "$CT_SNAP/$BASE-$QUANT.gguf"
  --alias "HauhauCS/Qwen3.8-27B-Uncensored-Aggressive-$QUANT"
  --host 0.0.0.0 --port "$CT_PORT"
  --ctx-size "$CTX"
  --n-gpu-layers "$NGL"
  --split-mode none
  --parallel 1
  --batch-size 2048 --ubatch-size 512
  --flash-attn on
  --jinja
  --no-mmap            # weights resident in unified memory; mmap double-counts as cache
  --spec-draft-n-max "$DEPTH"
  --spec-draft-p-min 0
)

case "$SPEC" in
  embedded) ARGS+=( --spec-type draft-mtp ) ;;
  fastmtp)  ARGS+=( --spec-type draft-mtp
                    --spec-draft-model "$CT_SNAP/$BASE-FastMTP-32K.gguf"
                    --spec-draft-ngl all ) ;;
  none)     ARGS+=( --spec-type none ) ;;
  *) echo "ERROR: SPEC must be embedded|fastmtp|none" >&2; exit 1 ;;
esac

[[ "$VISION" == "1" ]] && ARGS+=( --mmproj "$CT_SNAP/mmproj-$BASE-BF16.gguf" )

# Detached, and deliberately no --restart policy — same reasoning as the vLLM profiles.
# A generated profile carrying `--restart unless-stopped` resurrected itself at boot on
# 2026-08-04 and crash-looped under a name the boot recipe checks for, so the box's
# default model never came up. A restart policy on a launch that may be wrong converts a
# bad script into a persistent outage.
#
# Running detached also means the engine is not in DMM's cgroup, so `systemctl --user
# restart dgx-model-manager` cannot kill it (the 2026-08-18 incident) without needing the
# systemd-run --scope wrapper that _launch_argv applies to bare-metal launches.
#
# The dgx.profile label is how reclaim identifies what to stop: substring-matching a
# served model name is ambiguous across profiles that all mention the same repo.
exec docker run -d --name "$CT_NAME" --gpus all -p "$PORT:$CT_PORT" \
  --label "dgx.profile=gguf_hauhaucs_qwen3.8-27b-aggressive-mtp" \
  --label "dgx.engine=llamacpp" \
  -v "$HOST_CACHE:$CT_CACHE:ro" \
  --entrypoint /app/llama-server \
  "$IMAGE" "${ARGS[@]}"

# ── Notes ─────────────────────────────────────────────────────────────────────
# Thinking is on by default (Qwen3.8 template). To default it off for every request:
#   --chat-template-kwargs '{"enable_thinking":false}'
#
# Image choice: ghcr.io/ggml-org/llama.cpp:server-cuda publishes linux/arm64 and, verified
# 2026-08-20 on this box, carries working GB10 kernels — 17x23 answered correctly and TG
# held 8.3 tok/s across 3 reps at Q6_K_P/4096 with no speculation. That is at the
# bandwidth ceiling for ~22 GB of weights on ~273 GB/s, so these are real CUDA kernels,
# not a correct-but-100x-slow generic fallback. No custom sm_121a build is needed for the
# stock path, which is why `binary`/`fastmtp_binary` in config.json became
# `image`/`fastmtp_image`.
#
# FastMTP: TESTED 2026-08-20 — THE FORK IS REQUIRED. SPEC=fastmtp on mainline 555881e died
# exactly as the model card predicted:
#   check_tensor_dims: tensor 'output.weight' has wrong shape;
#   expected 5120, 248320, got 5120, 32768
# on loading the FastMTP-32K sidecar, then "[spec] failed to measure draft model memory".
# So although upstream has the draft-mtp *mechanism* — draft-mtp (common/speculative.cpp:35),
# the d2t tensor (src/llama-arch.cpp:608), a separate draft context
# (common/speculative.cpp:2343) — it does not accept this sidecar's 32K-vocab output head.
# The gap is the vocab shape, not the mechanism. To use SPEC=fastmtp, build the patch into
# an image and push it to fastmtp_image; until then RECIPE=fast fails on the missing image,
# which is correct — it fails before any weights load.
