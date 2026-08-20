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

SNAP="$HOME/.cache/huggingface/hub/models--HauhauCS--Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/snapshots/993a5971fda8f30dd1b7eb2654792ba4415c7460"
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
# fastmtp needs the patched binary; the recipe choice, not the operator, selects it.
key = "fastmtp_binary" if r["spec"] == "fastmtp" else "binary"
print(f"R_BIN={os.path.expanduser(lc.get(key, '~/llama.cpp/build/bin/llama-server'))}")
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
# embedded  = NextN head baked into every target GGUF, works on stock llama.cpp.
# fastmtp   = the 32K sidecar; REQUIRES a build with HauhauCS-FastMTP-llama.cpp.patch
#             applied, otherwise the draft load fails with
#             "expected 5120, 248320, got 5120, 32768".
SPEC="${SPEC:-$R_SPEC}"
# Vision costs ~0.9 GB and the projector is BF16 regardless of target quant.
VISION="${VISION:-$R_VISION}"
NGL="${NGL:-all}"
PORT="${PORT:-8080}"
DEPTH="${DEPTH:-3}"

MODEL="$SNAP/$BASE-$QUANT.gguf"
[[ -f "$MODEL" ]] || { echo "ERROR: no such quant: $MODEL" >&2; exit 1; }

BIN="${LLAMA_SERVER:-$R_BIN}"
[[ -x "$BIN" ]] || { echo "ERROR: llama-server not built at $BIN" >&2; exit 1; }

# A CPU-only build accepts -ngl and silently ignores it, serving ~100x slow with no error
# — exactly what the first dry run on 2026-08-18 hit. Fail loudly instead.
if [[ "$NGL" != "0" ]] && ! "$BIN" --list-devices 2>&1 | grep -qi "cuda\|gpu"; then
  echo "ERROR: $BIN has no GPU backend (built with GGML_CUDA=OFF)." >&2
  echo "       Rebuild:" >&2
  echo "         cmake -S ~/llama.cpp -B ~/llama.cpp/build-cuda -DCMAKE_BUILD_TYPE=Release \\" >&2
  echo "           -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real \\" >&2
  echo "           -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_COMPRESSION_MODE=speed \\" >&2
  echo "           -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF" >&2
  echo "         cmake --build ~/llama.cpp/build-cuda -j4   # -j16 only with vLLM stopped" >&2
  echo "       121a-real: GB10 is compute capability 12.1, and ggml rewrites 12X->12Xa" >&2
  echo "       anyway (Blackwell FP4 instructions are not forward-compatible); -real drops" >&2
  echo "       the unused PTX. Or set NGL=0 to accept CPU-only inference deliberately." >&2
  exit 1
fi

echo "recipe=$R_NAME quant=$QUANT ctx=$CTX spec=$SPEC vision=$VISION bin=$BIN" >&2

# Run bare-metal, not in Docker: llama.cpp here is a local aarch64 CUDA build, and the
# FastMTP variant needs a *patched* binary that no published image carries.
pkill -f "llama-server .*--port $PORT" 2>/dev/null || true

# Same page-cache reasoning as the vLLM profiles: GB10 unified memory bills the page
# cache against the GPU budget, so a warm cache from the previous model steals KV.
# SKIP_DROP_CACHES=1 for side-by-side dry runs, where evicting the *other* engine's
# cache is a needless tax.
if [[ "${SKIP_DROP_CACHES:-0}" != "1" ]]; then
  sync && (echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null) || \
    echo "WARN: could not drop page cache; KV budget may be short" >&2
fi

ARGS=(
  --model "$MODEL"
  --alias "HauhauCS/Qwen3.8-27B-Uncensored-Aggressive-$QUANT"
  --host 0.0.0.0 --port "$PORT"
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
                    --spec-draft-model "$SNAP/$BASE-FastMTP-32K.gguf"
                    --spec-draft-ngl all ) ;;
  none)     ARGS+=( --spec-type none ) ;;
  *) echo "ERROR: SPEC must be embedded|fastmtp|none" >&2; exit 1 ;;
esac

[[ "$VISION" == "1" ]] && ARGS+=( --mmproj "$SNAP/mmproj-$BASE-BF16.gguf" )

exec "$BIN" "${ARGS[@]}"

# ── Notes ─────────────────────────────────────────────────────────────────────
# Thinking is on by default (Qwen3.8 template). To default it off for every request:
#   --chat-template-kwargs '{"enable_thinking":false}'
# FastMTP: TESTED 2026-08-20 — THE FORK IS REQUIRED. Answering the open question below.
# SPEC=fastmtp on the mainline 555881e build-cuda died exactly as the model card predicted:
#   check_tensor_dims: tensor 'output.weight' has wrong shape;
#   expected 5120, 248320, got 5120, 32768
# on loading the FastMTP-32K sidecar, then "[spec] failed to measure draft model memory".
# So although upstream has the draft-mtp *mechanism*, it does not accept this sidecar's
# 32K-vocab output head. To use SPEC=fastmtp, build the fork into a git worktree at
# ~/llama.cpp-fastmtp (see fastmtp_binary in config.json), never as a rewind of the main
# checkout other profiles depend on. Until then SPEC=fastmtp / RECIPE=fast will fail fast,
# which is correct: the recipe's binary check catches it before any weights load.
#
# Original reasoning, kept because it explains why the test was worth running: at 555881e
# upstream already has draft-mtp (common/speculative.cpp:35), the d2t tensor
# (src/llama-arch.cpp:608), and a separate draft context (common/speculative.cpp:2343) —
# i.e. the mechanism the patch describes adding. The gap turned out to be the vocab shape,
# not the mechanism.
