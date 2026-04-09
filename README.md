# 🔰DGX Spark Model Manager

A lightweight web UI for managing AI models on the **NVIDIA DGX Spark / HP ZGX Nano G1n** (GB10, 128 GB unified memory). Pull Ollama models, download from HuggingFace, manage LiteLLM routing, and control SGLang or vLLM — all from one browser tab.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-aarch64%20Ubuntu-orange)

---

## Features

- **Ollama Models** — pull, list, and delete models with live download progress
- **LiteLLM Routing** — one-click wildcard routing so every Ollama model is auto-exposed to all your apps
- **SGLang Engine** — start/stop the SGLang Docker container via configurable launch profiles
- **vLLM Engine** — start/stop the vLLM Docker container via configurable launch profiles
- **HuggingFace Download** — download any model from HF Hub and manage them directly on the device
- **Live Status Bar** — real-time health indicators for SGLang, vLLM, Ollama, and LiteLLM
- **Built-in Help Page** — documentation at `/help` or Docs button (upper-right corner) 

Both SGLang and vLLM are fully supported as inference engines. Use whichever you prefer — or both, on different ports with different models. Each engine has its own tab, profiles, and status indicator.

---

## Requirements

| Component | Required | Notes |
|-----------|----------|-------|
| Python 3.10+ | ✅ | Pre-installed on DGX Spark |
| [Ollama](https://ollama.com) | ✅ | Core model management |
| [LiteLLM](https://github.com/BerriAI/litellm) | Optional | Unified API routing |
| [SGLang](https://github.com/sgl-project/sglang) | Optional | Large model inference |
| [vLLM](https://github.com/vllm-project/vllm) | Optional | Large model inference (alternative to SGLang) |
| Docker | Optional | Required for SGLang and vLLM start/stop |

The app works with just Ollama installed. LiteLLM, SGLang, and vLLM tabs gracefully show offline status if those services aren't running.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/calico88x/DGX-Model-Manager
cd DGX-Model-Manager

# 2. Configure (edit before running)
nano config.json

# 3. Run setup
bash setup.sh

# 4. Open in browser
http://<your-dgx-ip>:8090
```

---

## Configuration

Edit `config.json` before running setup. All fields have sensible defaults.

```json
{
  "app": {
    "host": "0.0.0.0",
    "port": 8090,
    "display_name": "DGX Spark"
  },
  "services": {
    "ollama_base":  "http://127.0.0.1:11434",
    "litellm_base": "http://127.0.0.1:4000",
    "sglang_base":  "http://127.0.0.1:30000",
    "vllm_base":    "http://127.0.0.1:8000"
  },
  "paths": {
    "litellm_config": "~/litellm/litellm_config.yaml",
    "hf_cache":       "~/.cache/huggingface/hub"
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `app.port` | `8090` | Port the app listens on |
| `app.display_name` | `DGX Spark` | Name shown in the UI header |
| `services.ollama_base` | `http://127.0.0.1:11434` | Ollama API URL |
| `services.litellm_base` | `http://127.0.0.1:4000` | LiteLLM proxy URL |
| `services.sglang_base` | `http://127.0.0.1:30000` | SGLang API URL |
| `services.vllm_base` | `http://127.0.0.1:8000` | vLLM API URL |
| `paths.litellm_config` | `~/litellm/litellm_config.yaml` | Path to your LiteLLM config file |
| `paths.hf_cache` | `~/.cache/huggingface/hub` | HuggingFace model cache directory |

---

## SGLang Scripts

Place your SGLang startup scripts in `~/SGLang/`. Any file named `start_*.sh` is automatically discovered and listed as a profile in the SGLang tab — no JSON editing required.

```
~/SGLang/
  start_mistral_small4.sh
  start_qwen3_70b.sh
  ...
```

Example (working) SGLang startup script for Mistral Small 4 NVFP4 w/ 256K context window:
Add optional header comments to control what the profile card displays:

```bash
#!/usr/bin/env bash
# Name: Mistral Small 4
# Description: 119B NVFP4 · ~15 min warm-up
# VRAM: 100 GB

sudo docker run --gpus all -d --rm \
  --name sglang \
  -p 30000:30000 \
  -v /tmp:/tmp \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  -e SGLANG_ENABLE_SPEC_V2=True \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  lmsysorg/sglang:nightly-dev-cu13-20260325-37420dce \
  sglang serve \
    --model-path mistralai/Mistral-Small-4-119B-2603-NVFP4 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1 \
    --attention-backend triton \
    --moe-runner-backend flashinfer_cutlass \
    --reasoning-parser mistral \
    --tool-call-parser mistral \
    --mem-fraction-static 0.75 \
    --context-length 262144 \
    --served-model-name Mistral-Small-4 \
    --enable-metrics
```

The `setup.sh` script creates `~/SGLang/` and drops an annotated example script there to use as a starting point.

> **GB10 / SM121A note:** The GB10 uses the `sm_121a` architecture which older bundled `ptxas` versions don't recognise. Two workarounds are available:
>
> **Option A — recommended:** Set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` in your Docker environment. This tells Triton to use the system CUDA `ptxas` which natively supports SM121A, and allows you to use the FlashInfer attention backend:
> ```bash
> docker run ... \
>   -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
>   ...
> ```
>
> **Option B — fallback:** Add `--attention-backend triton` to your SGLang launch flags. This bypasses the broken `ptxas` path entirely. Do not use `--quantization modelopt_fp4` or `--fp4-gemm-backend` alongside this option.
>
> See [triton-lang/triton#8539](https://github.com/triton-lang/triton/issues/8539) for upstream tracking.

---

## vLLM Scripts

Place your vLLM startup scripts in `~/vLLM/`. Any file named `start_*.sh` is automatically discovered and listed as a profile in the vLLM tab. The container can be named anything — the app identifies it by port.

```
~/vLLM/
  start_nemotron3super_vllm.sh
  start_llama3_70b.sh
  ...
```

Example (working) vLLM startup script for Nemotron 3 Super NVFP4 w/ 256K context window:
Same header comment format as SGLang:

```bash
#!/usr/bin/env bash
# Start Nemotron-3-Super-120B-A12B-NVFP4 via vLLM on DGX Spark
# Port: 8000 (vLLM default, matches Model Manager config)
# IMPORTANT: Cannot run simultaneously with SGLang (combined VRAM > 128 GB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARSER_PATH="$SCRIPT_DIR/super_v3_reasoning_parser.py"

# Download reasoning parser if not present
if [ ! -f "$PARSER_PATH" ]; then
  echo "[nemotron] Downloading reasoning parser..."
  wget -q -O "$PARSER_PATH" \
    https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
fi

echo "[nemotron] Starting vLLM with Nemotron-3-Super (NVFP4)..."

docker run --gpus all -d --rm \
  --name vllm-nemotron \
  -p 8000:8000 \
  --ipc=host \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PARSER_PATH":/app/super_v3_reasoning_parser.py \
  vllm/vllm-openai:cu130-nightly \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name Nemotron-3-Super \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --max-model-len 1000000 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

echo "[nemotron] Container started. Waiting for model load..."
echo "[nemotron] Monitor with: docker logs -f vllm-nemotron"
```

The `setup.sh` script creates `~/vLLM/` and drops an annotated example script there to use as a starting point.

### Common vLLM flags

| Flag | Description |
|------|-------------|
| `--model` | Path to model weights or HuggingFace repo ID |
| `--tensor-parallel-size` | Number of GPUs for tensor parallelism (1 for DGX Spark) |
| `--max-model-len` | Maximum context length (reduce if hitting OOM) |
| `--gpu-memory-utilization` | Fraction of GPU memory to use (default 0.9) |
| `--dtype` | Data type: `auto`, `float16`, `bfloat16` |
| `--quantization` | Quantization method: `awq`, `gptq`, `squeezellm`, etc. |
| `--tool-call-parser` | Tool calling format: `mistral`, `hermes`, `llama3_json`, etc. |
| `--enable-auto-tool-choice` | Auto-detect tool calls from model output |

> **Tip:** vLLM's default port is `8000`. If you're running both SGLang and vLLM simultaneously on different models, the default ports (`30000` and `8000`) won't conflict. Just make sure you have enough memory for both.

### Using vLLM with LiteLLM

To route vLLM models through LiteLLM at `:4000`, add an entry to your `litellm_config.yaml`:

```yaml
- model_name: my-vllm-model
  litellm_params:
    model: openai/my-model-name
    api_base: http://127.0.0.1:8000/v1
    api_key: token-abc123  # vLLM default, or set with --api-key
```

Then restart LiteLLM from the LiteLLM tab or via `sudo systemctl restart litellm`.

---

## LiteLLM Wildcard Routing

The **Apply Wildcard** button in the LiteLLM tab adds this single entry to your `litellm_config.yaml`:

```yaml
- model_name: ollama/*
  litellm_params:
    model: ollama/*
    api_base: http://127.0.0.1:11434
```

After this one-time change, every model you pull into Ollama is automatically available to all apps connected to LiteLLM at `:4000` — no further config edits required.

The button also restarts the LiteLLM service automatically. This requires passwordless sudo for `systemctl restart litellm` — the setup script will offer to configure this for you.

> **Note:** The wildcard only covers Ollama models. SGLang and vLLM models need explicit entries in the LiteLLM config — see the sections above for examples.

---

## Running Without systemd

```bash
# Activate the venv and run directly
source venv/bin/activate
python3 app.py
```

Or with uvicorn for more control:

```bash
venv/bin/uvicorn app:app --host 0.0.0.0 --port 8090 --reload
```

---

## Service Management

```bash
# Status
sudo systemctl status model-manager

# Restart
sudo systemctl restart model-manager

# Logs
sudo journalctl -u model-manager -f

# Disable autostart
sudo systemctl disable model-manager
```

---

## Stack Architecture

```
Your Apps (Open WebUI, scripts, any OpenAI client, etc.)
         │
         ▼
   LiteLLM :4000  ──────────────────────────────┬──────────────────┐
         │                                      │                  │
         ▼                                      ▼                  ▼
  SGLang :30000                          Ollama :11434        vLLM :8000
  (large models,                         (small/medium,       (large models,
   NVFP4, MoE)                            hot-swap)            alternative engine)
```

SGLang and vLLM serve the same role — high-performance inference for large models. Use whichever fits your workflow. Both are controlled via Docker and managed through their respective tabs in the UI.

---

## SGLang vs vLLM — Which Should I Use?

Both engines are excellent. Here's a quick comparison to help you decide:

| | SGLang | vLLM |
|---|---|---|
| **Strengths** | RadixAttention (prefix caching), fast structured output, strong MoE support | Broad model/quantization support, PagedAttention, mature ecosystem |
| **Docker image** | `lmsysorg/sglang:*` | `vllm/vllm-openai:*` |
| **Default port** | 30000 | 8000 |
| **API format** | OpenAI-compatible (`/v1/`) | OpenAI-compatible (`/v1/`) |
| **Tool calling** | `--tool-call-parser` | `--tool-call-parser` + `--enable-auto-tool-choice` |
| **Container name** | Any (detected by port :30000) | Any (detected by port :8000) |

Both expose the same OpenAI-compatible API, so your apps don't need to change when switching between them. LiteLLM routes to either one the same way.

---

## License

MIT
