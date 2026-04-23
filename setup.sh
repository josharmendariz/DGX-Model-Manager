#!/usr/bin/env bash
# DGX Model Manager — setup script
# Run once: bash setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv"

echo "==> DGX Model Manager Setup"
echo "    Directory: $SCRIPT_DIR"
echo ""

# ── Python venv ───────────────────────────────────────────────────────────────
echo "==> Creating Python venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
echo "    Done."

# ── UFW (optional) ────────────────────────────────────────────────────────────
if command -v ufw &>/dev/null; then
  PORT=$(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/config.json')); print(d.get('app',{}).get('port',8090))" 2>/dev/null || echo 8090)
  echo ""
  echo "==> UFW detected — adding rule for port $PORT"
  echo "    Edit the subnet below to match your network before confirming."
  read -r -p "    Allow port $PORT from subnet [192.168.1.0/24]: " SUBNET
  SUBNET="${SUBNET:-192.168.1.0/24}"
  sudo ufw allow from "$SUBNET" to any port "$PORT" proto tcp
  echo "    UFW rule added."
fi

# ── Sudoers (optional — for LiteLLM restart button) ───────────────────────────
echo ""
read -r -p "==> Add passwordless sudo for 'systemctl restart litellm'? [y/N]: " ADD_SUDO
if [[ "$ADD_SUDO" =~ ^[Yy]$ ]]; then
  USER_NAME="$(whoami)"
  SUDOERS_LINE="$USER_NAME ALL=(ALL) NOPASSWD: /bin/systemctl restart litellm, /bin/systemctl restart litellm.service"
  echo "$SUDOERS_LINE" | sudo tee /etc/sudoers.d/model-manager-litellm > /dev/null
  sudo chmod 440 /etc/sudoers.d/model-manager-litellm
  echo "    Sudoers entry created."
fi

# ── Systemd service ───────────────────────────────────────────────────────────
echo ""
read -r -p "==> Install as systemd service (starts on boot)? [y/N]: " ADD_SERVICE
if [[ "$ADD_SERVICE" =~ ^[Yy]$ ]]; then
  USER_NAME="$(whoami)"
  cat <<UNIT | sudo tee /etc/systemd/system/model-manager.service > /dev/null
[Unit]
Description=DGX Model Manager
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV/bin/python3 $SCRIPT_DIR/app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

  sudo systemctl daemon-reload
  sudo systemctl enable model-manager
  sudo systemctl start model-manager
  echo "    Service installed and started."
  echo ""
  sudo systemctl status model-manager --no-pager
else
  echo ""
  echo "==> To run manually:"
  echo "    $VENV/bin/python3 $SCRIPT_DIR/app.py"
fi

# ── Engine script directories ─────────────────────────────────────────────────
echo ""
echo "==> Creating engine script directories"

SGLANG_DIR="$HOME/SGLang"
VLLM_DIR="$HOME/vLLM"
LLAMACPP_DIR="$HOME/llama.cpp"
LOCALAI_DIR="$HOME/LocalAI"
COMFYUI_DIR="$HOME/ComfyUI"

mkdir -p "$SGLANG_DIR"
mkdir -p "$VLLM_DIR"
mkdir -p "$LLAMACPP_DIR"
mkdir -p "$LOCALAI_DIR"
mkdir -p "$COMFYUI_DIR"

# Write SGLang example script only if none exist yet
if ! ls "$SGLANG_DIR"/start_*.sh &>/dev/null; then
  cat > "$SGLANG_DIR/start_example_sglang.sh" <<'SGLANG_SCRIPT'
#!/usr/bin/env bash
# Name: Example SGLang Model
# Description: Replace with your model description and VRAM requirements
# VRAM: 97
#
# ── How this file works ───────────────────────────────────────────────────────
# DGX Model Manager scans ~/SGLang/ for files named start_*.sh and lists them
# as selectable profiles on the SGLang tab. The three header comments above
# (Name, Description, VRAM) control what the UI displays. Remove this file and
# add your own start_<modelname>.sh scripts to this folder.
#
# ── Example: Mistral Small 4 NVFP4 via SGLang ─────────────────────────────────
# sudo docker run --rm --gpus all --ipc=host \
#   --name my-sglang-container \
#   -v ~/.cache/huggingface:/root/.cache/huggingface \
#   -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
#   -p 30000:30000 \
#   lmsysorg/sglang:nightly-dev-cu13 \
#   python3 -m sglang.launch_server \
#     --model-path /root/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/main \
#     --quantization modelopt_fp4 \
#     --host 0.0.0.0 \
#     --port 30000 \
#     --tool-call-parser mistral
#
# ── GB10 / SM121A note ────────────────────────────────────────────────────────
# The GB10 chip requires a system ptxas that supports SM121A. Set the env var:
#   -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# This lets you use --quantization modelopt_fp4 and the FlashInfer backend.
# Without it, use --attention-backend triton as a fallback.

echo "Replace this script with your actual SGLang launch command."
echo "See the comments above for an example."
SGLANG_SCRIPT
  chmod +x "$SGLANG_DIR/start_example_sglang.sh"
  echo "    Created $SGLANG_DIR/start_example_sglang.sh"
else
  echo "    $SGLANG_DIR already has start_*.sh scripts — skipping example"
fi

# Write vLLM example script only if none exist yet
if ! ls "$VLLM_DIR"/start_*.sh &>/dev/null; then
  cat > "$VLLM_DIR/start_example_vllm.sh" <<'VLLM_SCRIPT'
#!/usr/bin/env bash
# Name: Example vLLM Model
# Description: Replace with your model description and VRAM requirements
# VRAM: 97
#
# ── How this file works ───────────────────────────────────────────────────────
# DGX Model Manager scans ~/vLLM/ for files named start_*.sh and lists them
# as selectable profiles on the vLLM tab. The three header comments above
# (Name, Description, VRAM) control what the UI displays. Remove this file and
# add your own start_<modelname>.sh scripts to this folder.
#
# ── Example: Nemotron 3 Super via vLLM ────────────────────────────────────────
# sudo docker run --rm --gpus all --ipc=host \
#   --name my-vllm-container \
#   -v ~/.cache/huggingface:/root/.cache/huggingface \
#   -p 8000:8000 \
#   vllm/vllm-openai:latest \
#   --model /root/.cache/huggingface/hub/models--nvidia--Nemotron-Super-49B-v1/snapshots/main \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --tool-call-parser mistral \
#   --enable-auto-tool-choice
#
# ── Common vLLM flags ─────────────────────────────────────────────────────────
# --max-model-len      Maximum context length (reduce if hitting OOM)
# --gpu-memory-utilization  Fraction of GPU memory to use (default 0.9)
# --dtype              Data type: auto, float16, bfloat16
# --quantization       awq, gptq, squeezellm, etc.

echo "Replace this script with your actual vLLM launch command."
echo "See the comments above for an example."
VLLM_SCRIPT
  chmod +x "$VLLM_DIR/start_example_vllm.sh"
  echo "    Created $VLLM_DIR/start_example_vllm.sh"
else
  echo "    $VLLM_DIR already has start_*.sh scripts — skipping example"
fi

# Write llama.cpp example script only if none exist yet
if ! ls "$LLAMACPP_DIR"/start_*.sh &>/dev/null; then
  cat > "$LLAMACPP_DIR/start_example_llamacpp.sh" <<'LLAMACPP_SCRIPT'
#!/usr/bin/env bash
# Name: Example llama.cpp Model
# Description: Replace with your GGUF model description
# VRAM: 8
#
# ── How this file works ───────────────────────────────────────────────────────
# DGX Model Manager scans ~/llama.cpp/ for files named start_*.sh and lists them
# as selectable profiles on the llama.cpp tab. The three header comments above
# (Name, Description, VRAM) control what the UI displays. Remove this file and
# add your own start_<modelname>.sh scripts to this folder.
#
# ── Example: llama-server with a GGUF model ───────────────────────────────────
# ./llama-server \
#   --model ~/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf \
#   --host 0.0.0.0 \
#   --port 8080 \
#   --n-gpu-layers 99 \
#   --ctx-size 4096
#
# ── Notes ─────────────────────────────────────────────────────────────────────
# llama.cpp uses llama-server (formerly server) for OpenAI-compatible inference.
# Health endpoint: /health    Models endpoint: /v1/models
# Build from source: cmake -B build && cmake --build build --config Release

echo "Replace this script with your actual llama.cpp launch command."
echo "See the comments above for an example."
LLAMACPP_SCRIPT
  chmod +x "$LLAMACPP_DIR/start_example_llamacpp.sh"
  echo "    Created $LLAMACPP_DIR/start_example_llamacpp.sh"
else
  echo "    $LLAMACPP_DIR already has start_*.sh scripts — skipping example"
fi

# Write LocalAI example script only if none exist yet
if ! ls "$LOCALAI_DIR"/start_*.sh &>/dev/null; then
  cat > "$LOCALAI_DIR/start_example_localai.sh" <<'LOCALAI_SCRIPT'
#!/usr/bin/env bash
# Name: Example LocalAI
# Description: Replace with your LocalAI configuration
# VRAM: 16
#
# ── How this file works ───────────────────────────────────────────────────────
# DGX Model Manager scans ~/LocalAI/ for files named start_*.sh and lists them
# as selectable profiles on the LocalAI tab. The three header comments above
# (Name, Description, VRAM) control what the UI displays. Remove this file and
# add your own start_<modelname>.sh scripts to this folder.
#
# ── Example: LocalAI via Docker ───────────────────────────────────────────────
# sudo docker run --rm --gpus all \
#   --name localai \
#   -v ~/LocalAI/models:/models \
#   -p 9090:8080 \
#   localai/localai:latest-gpu-nvidia-cuda-12
#
# ── Notes ─────────────────────────────────────────────────────────────────────
# LocalAI supports LLM, TTS, STT, and image generation in one service.
# Health endpoint: /readyz    Models endpoint: /v1/models

echo "Replace this script with your actual LocalAI launch command."
echo "See the comments above for an example."
LOCALAI_SCRIPT
  chmod +x "$LOCALAI_DIR/start_example_localai.sh"
  echo "    Created $LOCALAI_DIR/start_example_localai.sh"
else
  echo "    $LOCALAI_DIR already has start_*.sh scripts — skipping example"
fi

# Write ComfyUI example script only if none exist yet
if ! ls "$COMFYUI_DIR"/start_*.sh &>/dev/null; then
  cat > "$COMFYUI_DIR/start_example_comfyui.sh" <<'COMFYUI_SCRIPT'
#!/usr/bin/env bash
# Name: Example ComfyUI
# Description: Replace with your ComfyUI configuration
# VRAM: 8
#
# ── How this file works ───────────────────────────────────────────────────────
# DGX Model Manager scans ~/ComfyUI/ for files named start_*.sh and lists them
# as selectable profiles on the ComfyUI tab. The three header comments above
# (Name, Description, VRAM) control what the UI displays. Remove this file and
# add your own start_<modelname>.sh scripts to this folder.
#
# ── Example: ComfyUI via Docker ───────────────────────────────────────────────
# sudo docker run --rm --gpus all \
#   --name comfyui \
#   -v ~/ComfyUI/models:/comfyui/models \
#   -v ~/ComfyUI/output:/comfyui/output \
#   -p 8188:8188 \
#   comfyanonymous/comfyui:latest
#
# ── Notes ─────────────────────────────────────────────────────────────────────
# ComfyUI provides its own web UI at port 8188. When running, the "Open UI"
# button on the ComfyUI tab links directly to the ComfyUI interface.

echo "Replace this script with your actual ComfyUI launch command."
echo "See the comments above for an example."
COMFYUI_SCRIPT
  chmod +x "$COMFYUI_DIR/start_example_comfyui.sh"
  echo "    Created $COMFYUI_DIR/start_example_comfyui.sh"
else
  echo "    $COMFYUI_DIR already has start_*.sh scripts — skipping example"
fi

echo "    SGLang   scripts → $SGLANG_DIR/"
echo "    vLLM     scripts → $VLLM_DIR/"
echo "    llama.cpp scripts → $LLAMACPP_DIR/"
echo "    LocalAI  scripts → $LOCALAI_DIR/"
echo "    ComfyUI  scripts → $COMFYUI_DIR/"

# ── Done ──────────────────────────────────────────────────────────────────────
PORT=$(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/config.json')); print(d.get('app',{}).get('port',8090))" 2>/dev/null || echo 8090)
echo ""
echo "==> Setup complete."
echo "    Open: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
