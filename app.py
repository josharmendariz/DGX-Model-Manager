#!/usr/bin/env python3
"""
DGX Model Manager
Unified web UI for managing models across Ollama, SGLang, vLLM, and LiteLLM.
Run via systemd: model-manager.service
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import platform
import re as _re
import socket
import subprocess
import sys
import shutil
import time as _time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional
from urllib.parse import urlparse

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# ─── Logging ─────────────────────────────────────────────────────────────────

class _MemoryHandler(logging.Handler):
    """Ring-buffer log handler that stores last N entries in memory."""
    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.buffer: deque[dict] = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def emit(self, record: logging.LogRecord):
        self.buffer.append({
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "func": record.funcName or "",
            "msg": self.format(record),
        })

    def get_entries(self, level: str = None, search: str = None, limit: int = 200) -> list[dict]:
        _levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        min_level = _levels.get(level, 0) if level else 0
        entries = list(self.buffer)
        if min_level:
            entries = [e for e in entries if _levels.get(e["level"], 0) >= min_level]
        if search:
            s = search.lower()
            entries = [e for e in entries if s in e["msg"].lower() or s in e["logger"].lower()]
        return entries[-limit:]

    def clear(self):
        self.buffer.clear()

_log_handler = _MemoryHandler(maxlen=500)
_log_handler.setFormatter(logging.Formatter("%(message)s"))
_logger = logging.getLogger("dgx")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
for _uv in ("uvicorn", "uvicorn.error"):
    logging.getLogger(_uv).addHandler(_log_handler)

_APP_START = _time.monotonic()
_APP_START_UTC = datetime.now(timezone.utc).isoformat()

# ─── Config ───────────────────────────────────────────────────────────────────

HOME             = Path.home()
_APP_DIR         = Path(__file__).resolve().parent
CUSTOM_DIRS_FILE = _APP_DIR / "custom_dirs.json"

# Load app config — resolved relative to app.py's directory
_CONFIG_FILE = _APP_DIR / "config.json"
_app_config: dict = {}
if _CONFIG_FILE.exists():
    try:
        _app_config = json.loads(_CONFIG_FILE.read_text())
    except Exception:
        pass

APP_PORT = _app_config.get("app", {}).get("port", 8090)


def _hash_key(key: str) -> str:
    """SHA-256 hash of an API key."""
    return hashlib.sha256(key.encode()).hexdigest()


# API key stored as hash — never plaintext
_raw_key = _app_config.get("app", {}).get("api_key", "")
if _raw_key and len(_raw_key) != 64:
    # Legacy plaintext key found — hash it on first load
    _API_KEY_HASH = _hash_key(_raw_key)
else:
    _API_KEY_HASH = _raw_key  # already a hash (or empty)

# Service URLs — loaded from config.json with sensible defaults
_svc = _app_config.get("services", {})
OLLAMA_BASE    = _svc.get("ollama_base",  "http://127.0.0.1:11434")
LITELLM_BASE   = _svc.get("litellm_base", "http://127.0.0.1:4000")

# Paths — loaded from config.json with sensible defaults
_paths = _app_config.get("paths", {})
LITELLM_CONFIG    = Path(os.path.expanduser(_paths.get("litellm_config", "~/litellm/litellm_config.yaml")))
HF_CACHE_DIR      = Path(os.path.expanduser(_paths.get("hf_cache", "~/.cache/huggingface/hub")))

# ─── Engine Registry ─────────────────────────────────────────────────────────
# Data-driven engine definitions — add a new engine by adding an entry here.
# Each engine gets: /api/{key}/profiles, /api/{key}/status, /api/{key}/start,
# /api/{key}/stop routes auto-generated, plus a tab, sidebar item, status pill,
# and settings card in the frontend.

_ENGINES = {
    "sglang": {
        "name": "SGLang",
        "description": "High-performance LLM inference engine (Docker)",
        "icon": "\U0001f680",
        "default_base": "http://127.0.0.1:30000",
        "config_key": "sglang_base",
        "script_dir_default": "SGLang",
        "script_dir_config_key": "sglang_scripts",
        "health_path": "/health",
        "models_path": "/v1/models",
        "docker_filter": "sglang",
    },
    "vllm": {
        "name": "vLLM",
        "description": "Production LLM inference engine (Docker)",
        "icon": "\u26a1",
        "default_base": "http://127.0.0.1:8000",
        "config_key": "vllm_base",
        "script_dir_default": "vLLM",
        "script_dir_config_key": "vllm_scripts",
        "health_path": "/health",
        "models_path": "/v1/models",
        "docker_filter": "vllm",
    },
    "llamacpp": {
        "name": "llama.cpp",
        "description": "GGUF model inference engine",
        "icon": "\U0001f999",
        "default_base": "http://127.0.0.1:8080",
        "config_key": "llamacpp_base",
        "script_dir_default": "llama.cpp",
        "script_dir_config_key": "llamacpp_scripts",
        "health_path": "/health",
        "models_path": "/v1/models",
        "docker_filter": "llamacpp",
    },
    "localai": {
        "name": "LocalAI",
        "description": "Multi-modal AI engine \u2014 LLM, TTS, STT, image gen (Docker)",
        "icon": "\U0001f916",
        "default_base": "http://127.0.0.1:9090",
        "config_key": "localai_base",
        "script_dir_default": "LocalAI",
        "script_dir_config_key": "localai_scripts",
        "health_path": "/readyz",
        "models_path": "/v1/models",
        "docker_filter": "local-ai",
    },
    "comfyui": {
        "name": "ComfyUI",
        "description": "Image generation workflow engine (Docker)",
        "icon": "\U0001f3a8",
        "default_base": "http://127.0.0.1:8188",
        "config_key": "comfyui_base",
        "script_dir_default": "ComfyUI",
        "script_dir_config_key": "comfyui_scripts",
        "health_path": "/",
        "models_path": None,
        "docker_filter": "comfyui",
        "webui": True,
    },
}

# Build derived state from registry + config
_engine_bases: dict[str, str] = {}
_engine_dirs: dict[str, Path] = {}
for _ek, _ev in _ENGINES.items():
    _engine_bases[_ek] = _svc.get(_ev["config_key"], _ev["default_base"])
    _engine_dirs[_ek] = HOME / _paths.get(_ev["script_dir_config_key"], _ev["script_dir_default"])


# ─── Auth ─────────────────────────────────────────────────────────────────────

async def verify_auth(request: Request):
    """Check API key on mutating endpoints. No-op when no key is configured."""
    if not _API_KEY_HASH:
        return
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        incoming_hash = _hash_key(auth[7:])
        if hmac.compare_digest(incoming_hash, _API_KEY_HASH):
            return
    _logger.warning("Auth rejected: %s %s", request.method, request.url.path)
    raise HTTPException(401, "Invalid or missing API key")


def _get_local_ip() -> str:
    """Best-effort LAN IP detection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _get_total_memory_gb() -> int:
    """Total system memory in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024)
    except Exception:
        pass
    return 0

# ─── Models ───────────────────────────────────────────────────────────────────

class PullRequest(BaseModel):
    name: str

class EngineStartRequest(BaseModel):
    profile: str

class HFDownloadRequest(BaseModel):
    repo_id: str
    local_dir: Optional[str] = None

# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _run(*cmd: str, timeout: float = 30) -> subprocess.CompletedProcess:
    """Run a subprocess without blocking the event loop."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return subprocess.CompletedProcess(cmd, 1, b"", b"timed out")
    return subprocess.CompletedProcess(
        cmd, proc.returncode or 0,
        stdout.decode() if stdout else "",
        stderr.decode() if stderr else "",
    )


async def service_ok(base: str, path: str = "/health") -> bool:
    try:
        r = await _http.get(base + path, timeout=3.0)
        return r.status_code < 400
    except Exception:
        return False


def load_litellm_config() -> dict:
    if LITELLM_CONFIG.exists():
        with open(LITELLM_CONFIG) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_litellm_config(cfg: dict):
    with open(LITELLM_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _parse_script_meta(script_path: Path) -> dict:
    """Derive profile metadata from a start_*.sh script.

    Optional header comments (in the first 20 lines) override defaults:
        # Name: Mistral Small 4
        # Description: 119B NVFP4 quantized
        # VRAM: 119
    Falls back to a human-readable name derived from the filename.
    """
    name = description = None
    vram_gb = None
    try:
        for line in script_path.read_text().splitlines()[:20]:
            line = line.strip()
            if line.startswith("# Name:"):
                name = line[7:].strip()
            elif line.startswith("# Description:"):
                description = line[14:].strip()
            elif line.startswith("# VRAM:"):
                try:
                    vram_gb = int(line[7:].strip().upper().rstrip("GB").strip())
                except Exception:
                    pass
    except Exception:
        pass

    if not name:
        stem = script_path.stem  # e.g. "start_mistral_small4"
        if stem.startswith("start_"):
            stem = stem[6:]
        name = stem.replace("_", " ").replace("-", " ").title()

    return {
        "id":          script_path.stem,
        "name":        name,
        "script":      str(script_path),
        "description": description or f"Script: {script_path.name}",
        "vram_gb":     vram_gb,
    }


def _scan_profiles(engine_key: str) -> list:
    """Scan ~/{engine_dir}/start_*.sh and return profile list."""
    d = _engine_dirs.get(engine_key)
    if not d or not d.exists():
        return []
    return [_parse_script_meta(s) for s in sorted(d.glob("start_*.sh"))]

# ─── HF Inventory helpers ──────────────────────────────────────────────────────

def _load_custom_dirs() -> list:
    if CUSTOM_DIRS_FILE.exists():
        try:
            return json.loads(CUSTOM_DIRS_FILE.read_text())
        except Exception:
            pass
    return []

def _save_custom_dirs(dirs: list) -> None:
    CUSTOM_DIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_DIRS_FILE.write_text(json.dumps(dirs))

def _dir_size_gb(path: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1e9, 2)
    except Exception:
        return 0.0

# ── Name-token tables ─────────────────────────────────────────────────────────
# Checked in order; most-specific entries must come first.
_NAME_DTYPE_TOKENS: list[tuple[str, str]] = [
    # Exact tokens after splitting on [-_. ]
    ("nvfp4",    "FP4"),  ("fp4",      "FP4"),
    ("fp8e4m3",  "FP8"),  ("fp8e5m2",  "FP8"),  ("fp8",  "FP8"),
    ("bfloat16", "BF16"), ("bf16",     "BF16"),
    ("float16",  "FP16"), ("fp16",     "FP16"),  ("f16",  "FP16"),
    ("float32",  "FP32"), ("fp32",     "FP32"),  ("f32",  "FP32"),
    ("awq",      "INT4"), ("gptq",     "INT4"),  ("bnb4", "INT4"),
    ("int4",     "INT4"), ("q4",       "INT4"),
    ("int8",     "INT8"), ("q8",       "INT8"),
    ("gguf",     "GGUF"),
]

_NAME_REASONING_TOKENS: frozenset[str] = frozenset({
    "r1", "qwq", "thinking", "cot", "reasoning",
    "reflect", "deepthink", "thinker", "o1",
})

_NAME_VISION_TOKENS: frozenset[str] = frozenset({
    "vl", "vision", "visual", "pixtral", "llava", "cogvlm",
    "idefics", "flamingo", "qwenvl", "internvl", "phi4mm",
})
# Substrings that don't tokenise cleanly but signal vision
_NAME_VISION_SUBSTR: tuple[str, ...] = (
    "llava", "qwen-vl", "cogvlm", "phi-4-mm", "internvl",
)

_NAME_AUDIO_TOKENS: frozenset[str] = frozenset({
    "audio", "whisper", "speech", "asr", "voice", "tts",
    "hubert", "wav2vec", "wav2vec2", "wavlm",
})

_NAME_EMBED_TOKENS: frozenset[str] = frozenset({
    "embed", "embedding", "embeddings",
    "e5", "bge", "gte", "nomic", "mxbai",
    "minilm", "sbert",
})

_NAME_MOE_TOKENS: frozenset[str] = frozenset({"moe", "mixture"})

_KNOWN_MOE_MODEL_TYPES: frozenset[str] = frozenset({
    "qwen2_moe", "mixtral", "deepseek_v2", "deepseek_v3",
    "olmoe", "phimoe", "jetmoe",
})


def _tokenize(name: str) -> frozenset[str]:
    """Split model name into lowercase tokens on -, _, ., space."""
    return frozenset(_re.split(r"[-_.\s]+", name.lower()))


def _infer_from_name(model_name: str) -> dict:
    """
    Last-resort inference from model name tokens and substrings.

    Returns a dict with keys:
      dtype            str | None   — e.g. "FP4", "BF16"
      is_moe           bool | None  — True if MoE signal found; None = no signal
      is_reasoning     bool | None
      extra_modalities list[str]    — e.g. ["Image", "Embedding"]
      params_b         float | None — parameter count in billions
    """
    tokens = _tokenize(model_name)
    name_lower = model_name.lower()

    # ── Dtype ────────────────────────────────────────────────────────────────
    dtype = None
    for tok, d in _NAME_DTYPE_TOKENS:
        if tok in tokens:
            dtype = d
            break
    # GGUF-style quant suffixes:  Q4_K_M, Q5_K_S, IQ3_XXS …
    if dtype is None:
        m = _re.search(r"\bq(\d)_k_[a-z]+\b", name_lower)
        if m:
            dtype = "INT4" if int(m.group(1)) <= 4 else "INT8"
        elif _re.search(r"\biq\d", name_lower):
            dtype = "INT4"

    # ── MoE ──────────────────────────────────────────────────────────────────
    is_moe: Optional[bool] = None
    if tokens & _NAME_MOE_TOKENS:
        is_moe = True
    # "235B-A22B" style (total params - active params) notation
    elif _re.search(r"\d+b-?a\d+b", name_lower):
        is_moe = True

    # ── Reasoning ────────────────────────────────────────────────────────────
    is_reasoning: Optional[bool] = None
    if tokens & _NAME_REASONING_TOKENS:
        is_reasoning = True

    # ── Modalities ───────────────────────────────────────────────────────────
    extra_modalities: list[str] = []
    if (tokens & _NAME_VISION_TOKENS
            or any(s in name_lower for s in _NAME_VISION_SUBSTR)):
        extra_modalities.append("Image")
    if tokens & _NAME_AUDIO_TOKENS:
        extra_modalities.append("Audio")
    if tokens & _NAME_EMBED_TOKENS:
        extra_modalities.append("Embedding")

    # ── Params ───────────────────────────────────────────────────────────────
    params_b: Optional[float] = None
    m2 = _re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?:[^a-z]|$)", model_name)
    if m2:
        params_b = float(m2.group(1))

    return {
        "dtype":            dtype,
        "is_moe":           is_moe,
        "is_reasoning":     is_reasoning,
        "extra_modalities": extra_modalities,
        "params_b":         params_b,
    }


_script_content_cache: dict[str, str] = {}

def _check_script_xref(model_name: str, all_profiles: list) -> tuple[bool, Optional[str]]:
    """Check if any engine script references this model name."""
    name_lower = model_name.lower()
    search_term = name_lower.replace("-", "_")
    for p, engine_label in all_profiles:
        script_path = p["script"]
        if script_path not in _script_content_cache:
            try:
                _script_content_cache[script_path] = Path(script_path).read_text().lower()
            except Exception:
                _script_content_cache[script_path] = ""
        content = _script_content_cache[script_path]
        if name_lower in content or search_term in content:
            return True, engine_label
    return False, None


_DTYPE_MAP = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16",
              "float8":  "FP8",  "float4":  "FP4"}
_BYTES_PER_DTYPE = {"FP32": 4, "FP16": 2, "BF16": 2, "FP8": 1,
                    "FP4": 0.5, "INT4": 0.5, "INT8": 1}

_PIPELINE_TO_TASK = {
    "text-generation": "Text Gen", "text2text-generation": "Text Gen",
    "image-text-to-text": "Vision LLM", "visual-question-answering": "Vision LLM",
    "feature-extraction": "Embedding", "sentence-similarity": "Embedding",
    "automatic-speech-recognition": "STT", "text-to-speech": "TTS",
    "text-to-image": "Image Gen", "image-to-image": "Image Gen",
    "text-to-video": "Video Gen", "text-to-audio": "Audio Gen",
    "image-classification": "Image Class.", "audio-classification": "Audio Class.",
    "translation": "Translation", "summarization": "Summarization",
    "fill-mask": "Fill Mask", "zero-shot-classification": "Classification",
    "object-detection": "Object Detection", "image-segmentation": "Segmentation",
}

def _task_from_modalities(modalities: list[str]) -> str:
    """Derive a task label from modality list when no pipeline_tag is available."""
    if "Embedding" in modalities:
        return "Embedding"
    if "Audio" in modalities and "Image" not in modalities:
        return "Audio"
    if "Image" in modalities:
        return "Vision LLM"
    return "Text Gen"

def _detect_format(dir_path: Path, is_hf_cache: bool = False) -> str:
    """Detect model file format from directory contents."""
    scan_dir = dir_path
    if is_hf_cache:
        snaps = dir_path / "snapshots"
        if snaps.exists():
            for s in sorted(snaps.iterdir()):
                if s.is_dir():
                    scan_dir = s
                    break
    try:
        for f in scan_dir.iterdir():
            n = f.name.lower()
            if n.endswith(".safetensors") or n.endswith(".safetensors.index.json"):
                return "safetensors"
            if n.endswith(".gguf"):
                return "gguf"
        for f in scan_dir.iterdir():
            if f.name.lower().endswith(".bin"):
                return "pytorch"
    except Exception:
        pass
    return "unknown"


def _infer_from_config(config: dict, name_hints: dict) -> dict:
    """Infer dtype, MoE, reasoning, and modalities from a model's config.json + name hints.

    Returns {dtype, is_moe, is_reasoning, modalities}.
    """
    # ── Dtype ────────────────────────────────────────────────────────────────
    raw_dtype = config.get("torch_dtype", "")
    dtype = _DTYPE_MAP.get(raw_dtype, raw_dtype.upper() if raw_dtype else None)

    quant = config.get("quantization_config", {}) or {}
    qt = str(quant.get("quant_type", quant.get("quant_method", ""))).lower()
    bits = quant.get("bits", 0) or quant.get("num_bits", 0)
    if "fp4" in qt or "nvfp4" in qt:
        dtype = "FP4"
    elif "fp8" in qt:
        dtype = "FP8"
    elif "int4" in qt or bits == 4 or quant.get("load_in_4bit"):
        dtype = "INT4"
    elif "int8" in qt or bits == 8 or quant.get("load_in_8bit"):
        dtype = "INT8"

    if not dtype or dtype == "Unknown":
        dtype = name_hints["dtype"] or "Unknown"
    elif name_hints["dtype"] in ("FP4", "INT4", "FP8", "INT8") and dtype in ("FP32", "FP16", "BF16"):
        dtype = name_hints["dtype"]

    # ── Architecture / MoE ───────────────────────────────────────────────────
    archs = config.get("architectures", [])
    arch_str = " ".join(archs).lower()
    is_moe = (
        config.get("num_experts") is not None
        or config.get("num_local_experts") is not None
        or config.get("num_experts_per_tok") is not None
        or "moe" in arch_str
        or config.get("model_type", "").lower() in _KNOWN_MOE_MODEL_TYPES
        or bool(name_hints["is_moe"])
    )

    # ── Reasoning ────────────────────────────────────────────────────────────
    is_reasoning = config.get("is_thinking", False) or bool(name_hints["is_reasoning"])

    # ── Modalities ───────────────────────────────────────────────────────────
    modalities: list[str] = ["Text"]
    if (config.get("vision_config") is not None
            or "vision" in arch_str or "llava" in arch_str
            or "Image" in name_hints["extra_modalities"]):
        modalities.append("Image")
    if (config.get("audio_config") is not None
            or "audio" in arch_str or "whisper" in arch_str
            or "Audio" in name_hints["extra_modalities"]):
        modalities.append("Audio")
    if "Embedding" in name_hints["extra_modalities"]:
        modalities.append("Embedding")

    return {"dtype": dtype, "is_moe": is_moe, "is_reasoning": is_reasoning,
            "modalities": modalities, "arch_str": arch_str}


def _parse_hf_model_dir(model_dir: Path, all_profiles: list = None) -> dict:
    """Parse a single HF cache model directory (models--owner--name)."""
    stem = model_dir.name
    if stem.startswith("models--"):
        tail = stem[8:]
        parts = tail.split("--", 1)
        owner = parts[0] if len(parts) > 1 else ""
        model_name = parts[1] if len(parts) > 1 else parts[0]
    else:
        owner = ""
        model_name = stem
    full_name = f"{owner}/{model_name}" if owner else model_name
    name_hints = _infer_from_name(model_name)

    # Find config.json inside snapshots/
    config: dict = {}
    snapshots_dir = model_dir / "snapshots"
    snapshot_used: Optional[Path] = None
    if snapshots_dir.exists():
        for snap in sorted(snapshots_dir.iterdir()):
            cfg_path = snap / "config.json"
            if cfg_path.exists():
                try:
                    config = json.loads(cfg_path.read_text())
                    snapshot_used = snap
                    break
                except Exception:
                    pass

    info = _infer_from_config(config, name_hints)
    dtype = info["dtype"]

    # ── Parameter count (HF cache has index files for this) ──────────────────
    params_b: Optional[float] = None
    if name_hints["params_b"] and dtype in ("FP4", "INT4", "FP8", "INT8"):
        params_b = name_hints["params_b"]
    elif snapshot_used:
        for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            idx_path = snapshot_used / idx_name
            if idx_path.exists():
                try:
                    idx = json.loads(idx_path.read_text())
                    total_bytes = idx.get("metadata", {}).get("total_size", 0)
                    if total_bytes:
                        bytes_per = _BYTES_PER_DTYPE.get(dtype, 2)
                        params_b = round(total_bytes / bytes_per / 1e9, 1)
                        break
                except Exception:
                    pass
    if params_b is None:
        params_b = name_hints["params_b"]

    # ── Size on disk (blobs dir avoids symlink double-counting) ───────────────
    blobs_dir = model_dir / "blobs"
    if blobs_dir.exists():
        try:
            size_gb = round(
                sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file()) / 1e9, 1
            )
        except Exception:
            size_gb = _dir_size_gb(model_dir)
    else:
        size_gb = _dir_size_gb(model_dir)

    has_script, script_engine = _check_script_xref(model_name, all_profiles or [])
    fmt = _detect_format(model_dir, is_hf_cache=True)

    return {
        "name":          model_name,
        "owner":         owner,
        "full_name":     full_name,
        "dir_path":      str(model_dir),
        "dtype":         dtype,
        "params_b":      params_b,
        "model_arch":    "MoE" if info["is_moe"] else "Dense",
        "size_gb":       size_gb,
        "is_reasoning":  info["is_reasoning"],
        "has_script":    has_script,
        "script_engine": script_engine,
        "modalities":    info["modalities"],
        "source":        "hf_cache",
        "format":        fmt,
        "pipeline_tag":  None,
        "task_label":    _task_from_modalities(info["modalities"]),
        "hf_downloads":  None,
        "hf_likes":      None,
    }


def _parse_flat_model_dir(model_dir: Path, all_profiles: list = None) -> dict:
    """Parse a flat model directory (not HF cache format) that contains config.json."""
    stem = model_dir.name
    if "--" in stem:
        parts = stem.split("--", 1)
        owner = parts[0]
        model_name = parts[1]
    else:
        owner = ""
        model_name = stem
    full_name = f"{owner}/{model_name}" if owner else model_name
    name_hints = _infer_from_name(model_name)

    config: dict = {}
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        try:
            config = json.loads(cfg_path.read_text())
        except Exception:
            pass

    info = _infer_from_config(config, name_hints)
    has_script, script_engine = _check_script_xref(model_name, all_profiles or [])
    fmt = _detect_format(model_dir)

    return {
        "name":          model_name,
        "owner":         owner,
        "full_name":     full_name,
        "dir_path":      str(model_dir),
        "dtype":         info["dtype"],
        "params_b":      name_hints["params_b"],
        "model_arch":    "MoE" if info["is_moe"] else "Dense",
        "size_gb":       _dir_size_gb(model_dir),
        "is_reasoning":  info["is_reasoning"],
        "has_script":    has_script,
        "script_engine": script_engine,
        "modalities":    info["modalities"],
        "source":        "custom_dir",
        "format":        fmt,
        "pipeline_tag":  None,
        "task_label":    _task_from_modalities(info["modalities"]),
        "hf_downloads":  None,
        "hf_likes":      None,
    }


def _scan_directory(directory: Path, all_profiles: list = None) -> dict:
    """Scan a directory for models. Returns {path, is_hf_cache, models}."""
    models = []
    is_hf_cache = False

    if not directory.exists():
        return {"path": str(directory), "is_hf_cache": False, "models": [], "error": "Directory not found"}

    # HF cache format: contains models--* subdirs
    hf_dirs = [d for d in sorted(directory.iterdir()) if d.is_dir() and d.name.startswith("models--")]
    if hf_dirs:
        is_hf_cache = True
        for d in hf_dirs:
            try:
                models.append(_parse_hf_model_dir(d, all_profiles))
            except Exception:
                pass
    # Also scan flat model dirs (subdirs with config.json) even alongside HF cache dirs
    for d in sorted(directory.iterdir()):
        if d.is_dir() and not d.name.startswith("models--") and (d / "config.json").exists():
            try:
                models.append(_parse_flat_model_dir(d, all_profiles))
            except Exception:
                pass

    # Deduplicate: if same full_name appears from both HF cache and flat dir, keep HF cache version
    seen: dict[str, int] = {}
    deduped: list[dict] = []
    for m in models:
        key = m.get("full_name", m["name"]).lower()
        if key in seen:
            # Keep the one with more data (size_gb > 0 preferred)
            existing = deduped[seen[key]]
            if not existing.get("size_gb") and m.get("size_gb"):
                deduped[seen[key]] = m
        else:
            seen[key] = len(deduped)
            deduped.append(m)
    models = deduped

    return {"path": str(directory), "is_hf_cache": is_hf_cache, "models": models}

# ─── HF Metadata cache ──────────────────────────────────────────────────────

HF_META_CACHE_FILE = _APP_DIR / "hf_meta_cache.json"
_HF_META_TTL = 7 * 24 * 3600  # 7 days

def _load_hf_meta_cache() -> dict:
    if HF_META_CACHE_FILE.exists():
        try:
            return json.loads(HF_META_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}

def _save_hf_meta_cache(cache: dict) -> None:
    HF_META_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    HF_META_CACHE_FILE.write_text(json.dumps(cache, indent=1))

async def _fetch_hf_model_meta(owner: str, name: str) -> dict | None:
    """Fetch model metadata from HuggingFace API. Returns cached result or fetches fresh."""
    cache = _load_hf_meta_cache()
    key = f"{owner}/{name}"
    entry = cache.get(key)
    if entry and (_time.time() - entry.get("fetched_at", 0)) < _HF_META_TTL:
        return entry
    try:
        r = await _http.get(f"https://huggingface.co/api/models/{owner}/{name}", timeout=15.0)
        if r.status_code != 200:
            return entry  # return stale cache if available
        d = r.json()
        result = {
            "pipeline_tag": d.get("pipeline_tag"),
            "tags": d.get("tags", [])[:20],
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "library_name": d.get("library_name"),
            "fetched_at": _time.time(),
        }
        cache[key] = result
        _save_hf_meta_cache(cache)
        return result
    except Exception:
        return entry

# ─── App ──────────────────────────────────────────────────────────────────────

_http: httpx.AsyncClient = None  # type: ignore[assignment]

@asynccontextmanager
async def _lifespan(app):
    global _http
    _http = httpx.AsyncClient(timeout=10.0)
    _logger.info("App started on port %s", APP_PORT)
    yield
    _logger.info("App shutting down")
    await _http.aclose()

app = FastAPI(title="DGX Model Manager", lifespan=_lifespan)

# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    # Build parallel health checks for all engines + special services
    check_keys = []
    check_coros = []
    for key, eng in _ENGINES.items():
        check_keys.append(key)
        check_coros.append(service_ok(_engine_bases[key], eng.get("health_path", "/health")))
    check_keys += ["ollama", "litellm"]
    check_coros += [service_ok(OLLAMA_BASE, "/api/tags"), service_ok(LITELLM_BASE, "/health")]
    results = await asyncio.gather(*check_coros)

    status = {}
    for key, ok in zip(check_keys, results):
        entry = {"ok": ok}
        if ok and key in _ENGINES:
            mp = _ENGINES[key].get("models_path")
            if mp:
                try:
                    r = await _http.get(_engine_bases[key] + mp, timeout=3.0)
                    d = r.json().get("data", [])
                    if d:
                        entry["model"] = d[0]["id"]
                except Exception:
                    pass
        status[key] = entry
    return status

@app.get("/api/nodeinfo")
async def get_nodeinfo():
    hostname = socket.gethostname()
    ip = _get_local_ip()
    arch = platform.machine()
    mem_gb = _get_total_memory_gb()
    ollama_port = OLLAMA_BASE.rsplit(":", 1)[-1]
    litellm_port = LITELLM_BASE.rsplit(":", 1)[-1]
    # Build services dict from engines + special services
    services = {"ollama": OLLAMA_BASE, "litellm": LITELLM_BASE}
    engine_ports = {}
    for key in _ENGINES:
        services[key] = _engine_bases[key]
        engine_ports[key + "_port"] = _engine_bases[key].rsplit(":", 1)[-1]
    return {
        "hostname": hostname,
        "ip": ip,
        "port": APP_PORT,
        "arch": arch,
        "memory_gb": mem_gb,
        # Legacy per-engine port keys for backward compat
        "sglang_port": engine_ports.get("sglang_port", ""),
        "vllm_port": engine_ports.get("vllm_port", ""),
        "ollama_port": ollama_port,
        "litellm_port": litellm_port,
        "ollama_base": OLLAMA_BASE,
        "services": services,
        "engine_ports": engine_ports,
    }

# ── Ollama ────────────────────────────────────────────────────────────────────

@app.get("/api/scriptdirs")
async def get_scriptdirs():
    return {key: str(_engine_dirs[key]) for key in _ENGINES}


@app.get("/api/ollama/models")
async def list_ollama_models():
    try:
        r = await _http.get(OLLAMA_BASE + "/api/tags")
        return r.json()
    except Exception as e:
        raise HTTPException(502, f"Ollama unreachable: {e}")


@app.post("/api/ollama/pull", dependencies=[Depends(verify_auth)])
async def pull_ollama_model(req: PullRequest):
    async def stream() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=None) as c:
                async with c.stream(
                    "POST", OLLAMA_BASE + "/api/pull",
                    json={"name": req.name, "stream": True}
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield 'data: {"done":true}\n\n'

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/ollama/models/{name:path}", dependencies=[Depends(verify_auth)])
async def delete_ollama_model(name: str):
    try:
        r = await _http.request("DELETE", OLLAMA_BASE + "/api/delete", json={"name": name}, timeout=60.0)
        if r.status_code == 404:
            raise HTTPException(404, f"Model '{name}' not found in Ollama")
        if r.status_code not in (200, 204):
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise HTTPException(r.status_code, detail)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, str(e))

# ── LiteLLM ───────────────────────────────────────────────────────────────────

@app.get("/api/litellm/models")
async def list_litellm_models():
    try:
        r = await _http.get(LITELLM_BASE + "/v1/models", timeout=5.0)
        return r.json()
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/litellm/config")
async def get_litellm_config():
    if not LITELLM_CONFIG.exists():
        return {"model_list": [], "_raw": "# config file not found"}
    raw = LITELLM_CONFIG.read_text()
    cfg = yaml.safe_load(raw) or {}
    cfg["_raw"] = raw
    return cfg


@app.post("/api/litellm/apply-wildcard", dependencies=[Depends(verify_auth)])
async def apply_litellm_wildcard():
    cfg = load_litellm_config()
    model_list = cfg.get("model_list", [])

    if any(m.get("model_name") == "ollama/*" for m in model_list):
        return {"ok": True, "message": "Wildcard already present"}

    # Drop any explicit ollama/ entries to avoid duplicates
    model_list = [
        m for m in model_list
        if not str(m.get("litellm_params", {}).get("model", "")).startswith("ollama/")
    ]
    model_list.append({
        "model_name": "ollama/*",
        "litellm_params": {
            "model": "ollama/*",
            "api_base": OLLAMA_BASE,
        },
    })
    cfg["model_list"] = model_list
    try:
        save_litellm_config(cfg)
    except OSError as e:
        _logger.error("Failed to write LiteLLM config: %s", e)
        raise HTTPException(500, f"Failed to save LiteLLM config: {e}")
    _logger.info("LiteLLM wildcard applied, restarting service")

    result = await _run("sudo", "systemctl", "restart", "litellm", timeout=15)
    if result.returncode != 0:
        _logger.error("LiteLLM restart failed after wildcard: %s", result.stderr.strip())
        hint = " — configure passwordless sudo (see Settings or the banner on this tab)" if "password" in result.stderr.lower() else ""
        raise HTTPException(500, f"Config saved but restart failed: {result.stderr.strip()}{hint}")
    _logger.info("LiteLLM restarted successfully")
    return {"ok": True, "message": "Wildcard applied — LiteLLM restarted"}


@app.post("/api/litellm/restart", dependencies=[Depends(verify_auth)])
async def restart_litellm():
    _logger.info("LiteLLM restart requested")
    result = await _run("sudo", "systemctl", "restart", "litellm", timeout=15)
    if result.returncode != 0:
        _logger.error("LiteLLM restart failed: %s", result.stderr.strip())
        hint = " — configure passwordless sudo (see Settings or the banner on this tab)" if "password" in result.stderr.lower() else ""
        raise HTTPException(500, f"Restart failed: {result.stderr.strip()}{hint}")
    _logger.info("LiteLLM restarted successfully")
    return {"ok": True}

# ── Shared engine helpers ─────────────────────────────────────────────────────

async def _find_container_by_port(port: int) -> Optional[str]:
    """Return the container ID listening on the given host port, or None."""
    result = await _run("docker", "ps", "--filter", f"publish={port}", "--format", "{{.ID}}", timeout=5)
    lines = result.stdout.strip().splitlines() if result.stdout.strip() else []
    return lines[0].strip() if lines else None


async def _docker_stop(container_id: str) -> tuple[bool, str]:
    """Stop a container by ID, falling back to sudo if needed."""
    if not _CONTAINER_ID_RE.match(container_id):
        return False, "Invalid container ID"
    r = await _run("docker", "stop", container_id, timeout=60)
    if r.returncode == 0:
        _logger.info("Docker container %s stopped", container_id[:12])
        return True, (r.stdout + r.stderr).strip()
    r2 = await _run("sudo", "docker", "stop", container_id, timeout=60)
    if r2.returncode == 0:
        _logger.info("Docker container %s stopped (sudo)", container_id[:12])
    else:
        _logger.error("Docker stop failed for %s: %s", container_id[:12], (r2.stdout + r2.stderr).strip())
    return r2.returncode == 0, (r2.stdout + r2.stderr).strip()


async def _engine_status(base_url: str, docker_name: str,
                         health_path: str = "/health",
                         models_path: str | None = "/v1/models") -> dict:
    """Get running status, loaded model, and container info for an engine."""
    running = await service_ok(base_url, health_path)
    model = None
    if running and models_path:
        try:
            r = await _http.get(base_url + models_path, timeout=3.0)
            d = r.json().get("data", [])
            if d:
                model = d[0]["id"]
        except Exception:
            pass
    result = await _run("docker", "ps", "--filter", f"name={docker_name}", "--format", "{{.Names}}\t{{.Status}}", timeout=5)
    return {"running": running, "model": model, "container_info": result.stdout.strip()}


def _extract_port(url: str) -> int:
    """Extract port number from a URL like http://host:port or http://host:port/path."""
    try:

        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
    except Exception:
        pass
    raise ValueError(f"Cannot extract port from URL: {url}")


async def _engine_stop(base_url: str, engine_name: str) -> dict:
    """Stop the Docker container for an engine by its configured port."""
    try:
        port = _extract_port(base_url)
    except ValueError:
        raise HTTPException(400, f"Invalid {engine_name} URL — cannot determine port from '{base_url}'")
    cid = await _find_container_by_port(port)
    if not cid:
        raise HTTPException(404, f"No container found listening on {engine_name} port — already stopped?")
    ok, output = await _docker_stop(cid)
    return {"ok": ok, "output": output}


async def _engine_start(req_profile: str, scan_fn, engine_name: str) -> dict:
    """Start a Docker engine by launching the selected profile script."""
    profiles = scan_fn()
    profile = next((p for p in profiles if p["id"] == req_profile), None)
    if not profile:
        raise HTTPException(404, f"Profile '{req_profile}' not found")
    script = os.path.expanduser(profile.get("script", ""))
    if not Path(script).exists():
        raise HTTPException(400, f"Script not found: {script}")
    safe_id = _re.sub(r"[^a-zA-Z0-9._-]", "_", req_profile)
    log_path = f"/tmp/{engine_name.lower()}_{safe_id}.log"
    _logger.info("%s starting profile '%s' — script: %s", engine_name, profile["name"], script)
    with open(log_path, "w") as logf:
        subprocess.Popen(
            ["bash", script],
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _logger.info("%s launched — logs at %s", engine_name, log_path)
    return {"ok": True, "message": f"Launched {profile['name']} — logs at {log_path}"}


# ── Dynamic engine routes ─────────────────────────────────────────────────────
# Auto-generate /api/{key}/profiles, status, stop, start for every engine.

for _ek, _ev in _ENGINES.items():
    def _make_engine_routes(key: str, eng: dict):
        @app.get(f"/api/{key}/profiles", name=f"{key}_profiles")
        async def profiles(k=key):
            return _scan_profiles(k)

        @app.get(f"/api/{key}/status", name=f"{key}_status")
        async def status(k=key, e=eng):
            return await _engine_status(
                _engine_bases[k], e.get("docker_filter", k),
                e.get("health_path", "/health"), e.get("models_path"))

        @app.post(f"/api/{key}/stop", name=f"{key}_stop",
                  dependencies=[Depends(verify_auth)])
        async def stop(k=key, e=eng):
            return await _engine_stop(_engine_bases[k], e["name"])

        @app.post(f"/api/{key}/start", name=f"{key}_start",
                  dependencies=[Depends(verify_auth)])
        async def start(req: EngineStartRequest, k=key):
            return await _engine_start(req.profile, lambda kk=k: _scan_profiles(kk), k)

    _make_engine_routes(_ek, _ev)

# ── HuggingFace Download ───────────────────────────────────────────────────────

_HF_DOWNLOAD_SCRIPT = """
import sys, json, os, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import list_repo_tree, hf_hub_download
from pathlib import Path

repo = os.environ["HF_REPO_ID"]
local_dir = os.environ.get("HF_LOCAL_DIR") or None
J = lambda **kw: print(json.dumps(kw), flush=True)
J(status="starting", repo=repo)

try:
    entries = [e for e in list_repo_tree(repo, recursive=True)
               if hasattr(e, 'size') and not e.path.startswith('.')]
    total_files = len(entries)
    total_bytes = sum(e.size or 0 for e in entries)
    J(status=f"Found {total_files} files ({total_bytes/1024**3:.1f} GB)")
    done_bytes = 0
    dl_start = time.time()
    errors = []
    result_path = None
    for i, entry in enumerate(entries, 1):
        fname = entry.path
        fsize = entry.size or 0
        sz_str = f"{fsize/1024**2:.0f} MB" if fsize > 1024**2 else f"{fsize/1024:.0f} KB" if fsize > 1024 else f"{fsize} B"
        J(file_start=dict(idx=i, total=total_files, name=fname, size_str=sz_str))
        t0 = time.time()
        try:
            dl_kw = dict(repo_id=repo, filename=fname)
            if local_dir:
                dl_kw["local_dir"] = local_dir
            fpath = hf_hub_download(**dl_kw)
            if result_path is None:
                result_path = str(Path(fpath).parent)
        except Exception as exc:
            errors.append(fname)
            J(file_error=dict(idx=i, name=fname, error=str(exc)))
            continue
        done_bytes += fsize
        elapsed = max(time.time() - t0, 0.001)
        total_elapsed = max(time.time() - dl_start, 0.001)
        speed = fsize / elapsed
        pct = done_bytes / total_bytes * 100 if total_bytes else 100
        if speed >= 1024**2:    spd = f"{speed/1024**2:.0f} MiB/s"
        elif speed >= 1024:     spd = f"{speed/1024:.0f} KiB/s"
        else:                   spd = f"{speed:.0f} B/s"
        J(progress=dict(pct=round(pct,1), done_mb=round(done_bytes/1024**2,1),
                        total_mb=round(total_bytes/1024**2,1), speed=spd,
                        idx=i, total_files=total_files, file=fname))
    total_elapsed = time.time() - dl_start
    avg = done_bytes / max(total_elapsed, 0.001)
    avg_str = f"{avg/1024**2:.0f} MiB/s" if avg >= 1024**2 else f"{avg/1024:.0f} KiB/s"
    out_path = local_dir or result_path or "HF cache"
    J(status="complete", path=out_path, avg_speed=avg_str,
      elapsed=f"{total_elapsed/60:.1f} min" if total_elapsed > 60 else f"{total_elapsed:.0f}s",
      errors=len(errors))
except Exception as e:
    J(status="error", error=str(e))
"""

_HF_REPO_RE = _re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")
_VALID_URL_RE = _re.compile(r"^https?://[a-zA-Z0-9._-]+(:\d{1,5})?(/.*)?$")


def _validate_service_url(url: str, label: str = "URL"):
    """Validate a service URL has valid format and port range."""
    if not _VALID_URL_RE.match(url):
        raise HTTPException(400, f"Invalid {label} — must be http://host:port or https://host:port")
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.port is not None and not (1 <= parsed.port <= 65535):
        raise HTTPException(400, f"Invalid {label} — port must be between 1 and 65535")
_CONTAINER_ID_RE = _re.compile(r"^[a-f0-9]{12,64}$")


@app.post("/api/hf/download", dependencies=[Depends(verify_auth)])
async def hf_download(req: HFDownloadRequest):
    repo_id = req.repo_id.strip()
    _logger.info("HF download requested: %s", repo_id)
    if not _HF_REPO_RE.match(repo_id):
        raise HTTPException(400, "Invalid repo ID format. Expected: owner/model-name")

    local_dir = (req.local_dir or "").strip()
    if local_dir and ("\0" in local_dir or "\n" in local_dir):
        raise HTTPException(400, "Invalid characters in local directory path")
    if local_dir and ".." in Path(local_dir).parts:
        raise HTTPException(400, "Path traversal not allowed in local directory")

    # Track custom dir so inventory can scan it later
    if local_dir:
        custom_dirs = _load_custom_dirs()
        expanded = os.path.expanduser(local_dir)
        parent = str(Path(expanded).parent)
        if parent not in custom_dirs and parent != str(HF_CACHE_DIR):
            custom_dirs.append(parent)
            _save_custom_dirs(custom_dirs)

    # Pass user input via environment variables — never interpolate into script
    sub_env = {**os.environ, "HF_REPO_ID": repo_id}
    if local_dir:
        sub_env["HF_LOCAL_DIR"] = local_dir

    async def stream() -> AsyncGenerator[str, None]:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _HF_DOWNLOAD_SCRIPT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=sub_env,
            )
            assert proc.stdout
            async for raw in proc.stdout:
                line = raw.decode().strip()
                if line:
                    yield f"data: {line}\n\n"
            stderr_data = await proc.stderr.read()  # type: ignore[union-attr]
            for line in stderr_data.decode().split("\n"):
                stripped = line.strip()
                if stripped and "%" not in stripped and "it/s" not in stripped:
                    yield f"data: {json.dumps({'log': stripped})}\n\n"
            await proc.wait()
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/hf/inventory")
async def hf_inventory():
    """Scan HF cache + custom dirs and return model inventory."""
    # Build profile list once for all models (avoids re-scanning per model)
    _script_content_cache.clear()
    all_profiles = []
    for ek, ev in _ENGINES.items():
        all_profiles += [(p, ev["name"]) for p in _scan_profiles(ek)]

    custom_dirs = _load_custom_dirs()
    directories = []

    # Always include the default HF cache
    directories.append(_scan_directory(HF_CACHE_DIR, all_profiles))

    # Custom dirs (skip if same as default)
    default_str = str(HF_CACHE_DIR)
    for d in custom_dirs:
        d_expanded = os.path.expanduser(d)
        if d_expanded != default_str:
            directories.append(_scan_directory(Path(d_expanded), all_profiles))

    return {"directories": directories}


@app.get("/api/inventory")
async def unified_inventory(include_ollama: bool = True):
    """Unified inventory: HF cache + custom dirs + optionally Ollama models."""
    _script_content_cache.clear()
    all_profiles = []
    for ek, ev in _ENGINES.items():
        all_profiles += [(p, ev["name"]) for p in _scan_profiles(ek)]

    custom_dirs = _load_custom_dirs()
    directories = []
    directories.append(_scan_directory(HF_CACHE_DIR, all_profiles))
    default_str = str(HF_CACHE_DIR)
    for d in custom_dirs:
        d_expanded = os.path.expanduser(d)
        if d_expanded != default_str:
            directories.append(_scan_directory(Path(d_expanded), all_profiles))

    # Include Ollama models as a virtual directory
    ollama_models = []
    if include_ollama:
        try:
            r = await _http.get(f"{OLLAMA_BASE}/api/tags", timeout=5.0)
            if r.status_code == 200:
                for m in r.json().get("models", []):
                    name = m.get("name", "")
                    size_bytes = m.get("size", 0)
                    details = m.get("details", {})
                    param_str = details.get("parameter_size", "")
                    params_b = None
                    if param_str:
                        try:
                            params_b = float(param_str.replace("B", "").strip())
                        except ValueError:
                            pass
                    quant = details.get("quantization_level", "")
                    ollama_models.append({
                        "name": name.split(":")[0] if ":" in name else name,
                        "owner": "",
                        "full_name": name,
                        "dir_path": "",
                        "dtype": quant.upper() if quant else "Unknown",
                        "params_b": params_b,
                        "model_arch": "Dense",
                        "size_gb": round(size_bytes / 1e9, 1) if size_bytes else 0,
                        "is_reasoning": False,
                        "has_script": False,
                        "script_engine": None,
                        "modalities": ["Text"],
                        "source": "ollama",
                        "format": "ollama",
                        "pipeline_tag": None,
                        "task_label": "Text Gen",
                        "hf_downloads": None,
                        "hf_likes": None,
                    })
        except Exception:
            pass  # Ollama offline — skip silently

    if ollama_models:
        directories.append({
            "path": "Ollama",
            "is_hf_cache": False,
            "models": ollama_models,
        })

    return {"directories": directories}


@app.get("/api/hf/inventory/dirs")
async def list_inventory_dirs():
    """Return the list of custom directories (lightweight, no scan)."""
    custom = _load_custom_dirs()
    dirs = [{"path": str(HF_CACHE_DIR), "default": True}]
    default_str = str(HF_CACHE_DIR)
    for d in custom:
        expanded = os.path.expanduser(d)
        if expanded != default_str:
            dirs.append({"path": expanded, "default": False})
    return {"dirs": dirs}


class AddDirRequest(BaseModel):
    path: str

_BLOCKED_ROOTS = frozenset({"/", "/etc", "/usr", "/bin", "/sbin", "/var", "/boot", "/dev", "/proc", "/sys", "/root"})

@app.post("/api/hf/inventory/dirs", dependencies=[Depends(verify_auth)])
async def add_inventory_dir(req: AddDirRequest):
    """Add a custom directory to the inventory scan list."""
    expanded = os.path.expanduser(req.path.strip())
    resolved = str(Path(expanded).resolve())
    if resolved in _BLOCKED_ROOTS or resolved == "/":
        raise HTTPException(400, "Cannot add a system root directory")
    if not Path(expanded).is_dir():
        raise HTTPException(400, "Directory does not exist")
    dirs = _load_custom_dirs()
    if expanded not in dirs:
        dirs.append(expanded)
        _save_custom_dirs(dirs)
    return {"ok": True, "dirs": dirs}

@app.delete("/api/hf/inventory/dirs", dependencies=[Depends(verify_auth)])
async def remove_inventory_dir(path: str):
    """Remove a custom directory from the inventory scan list."""
    expanded = os.path.expanduser(path.strip())
    dirs = [d for d in _load_custom_dirs() if os.path.expanduser(d) != expanded]
    _save_custom_dirs(dirs)
    return {"ok": True, "dirs": dirs}


class DeleteModelRequest(BaseModel):
    path: str


@app.post("/api/hf/inventory/delete", dependencies=[Depends(verify_auth)])
async def delete_inventory_model(req: DeleteModelRequest):
    """Delete a downloaded model directory from disk."""
    target = Path(os.path.expanduser(req.path.strip())).resolve()

    # Safety: only allow deletion under HF cache or known custom dirs
    allowed_roots = [HF_CACHE_DIR.resolve()]
    for d in _load_custom_dirs():
        allowed_roots.append(Path(os.path.expanduser(d)).resolve())
    allowed = False
    for root in allowed_roots:
        try:
            target.relative_to(root)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise HTTPException(400, "Path is not under a known model directory")
    if not target.exists():
        raise HTTPException(404, "Directory not found")
    if not target.is_dir():
        raise HTTPException(400, "Path is not a directory")

    try:
        shutil.rmtree(target)
        return {"ok": True, "deleted": str(target)}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete: {e}")

# ── HF Metadata & Search ────────────────────────────────────────────────────

@app.get("/api/hf/meta/{owner}/{name}")
async def hf_model_meta(owner: str, name: str):
    """Fetch/return cached HF metadata for a single model."""
    meta = await _fetch_hf_model_meta(owner, name)
    if not meta:
        raise HTTPException(404, "Could not fetch metadata")
    return {
        "pipeline_tag": meta.get("pipeline_tag"),
        "task_label": _PIPELINE_TO_TASK.get(meta.get("pipeline_tag", ""), "Unknown"),
        "downloads": meta.get("downloads", 0),
        "likes": meta.get("likes", 0),
        "tags": meta.get("tags", []),
        "library_name": meta.get("library_name"),
    }

class EnrichRequest(BaseModel):
    models: list[dict]  # [{owner, name}, ...]

@app.post("/api/hf/meta/enrich", dependencies=[Depends(verify_auth)])
async def hf_meta_enrich(req: EnrichRequest):
    """Bulk enrich models with HF metadata. Max 50 per call."""
    results = {}
    for entry in req.models[:50]:
        owner = entry.get("owner", "")
        name = entry.get("name", "")
        if not owner or not name:
            continue
        meta = await _fetch_hf_model_meta(owner, name)
        if meta:
            key = f"{owner}/{name}"
            results[key] = {
                "pipeline_tag": meta.get("pipeline_tag"),
                "task_label": _PIPELINE_TO_TASK.get(meta.get("pipeline_tag", ""), "Unknown"),
                "downloads": meta.get("downloads", 0),
                "likes": meta.get("likes", 0),
            }
        await asyncio.sleep(0.2)  # rate-limit HF API calls
    return {"results": results}

@app.get("/api/hf/search")
async def hf_search(q: str, sort: str = "downloads", limit: int = 20, pipeline_tag: str = None):
    """Proxy search to HuggingFace Hub API."""
    params = {"search": q, "sort": sort, "limit": min(limit, 50), "full": "true"}
    if pipeline_tag:
        params["filter"] = pipeline_tag
    try:
        r = await _http.get("https://huggingface.co/api/models", params=params, timeout=15.0)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        raise HTTPException(502, f"HuggingFace API error: {e}")
    models = []
    for m in raw:
        tags = m.get("tags", [])
        ptag = m.get("pipeline_tag", "")
        models.append({
            "id": m.get("modelId") or m.get("id", ""),
            "pipeline_tag": ptag,
            "task_label": _PIPELINE_TO_TASK.get(ptag, ptag or "Unknown"),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "tags": tags[:15],
            "library_name": m.get("library_name", ""),
            "last_modified": m.get("lastModified", ""),
            "has_gguf": "gguf" in tags,
            "has_safetensors": "safetensors" in tags,
        })
    return {"models": models}

@app.get("/api/hf/search/variants")
async def hf_search_variants(model_id: str):
    """Find quantized variants (GGUF, GPTQ, AWQ) of a model."""
    parts = model_id.split("/", 1)
    base_name = parts[1] if len(parts) > 1 else parts[0]
    # strip common suffixes to get base model name
    for suffix in ("-Instruct", "-Chat", "-it", "-hf"):
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    variants = []
    for tag in ("gguf", "gptq", "awq"):
        try:
            r = await _http.get("https://huggingface.co/api/models",
                                params={"search": base_name, "filter": tag, "sort": "downloads", "limit": "5"},
                                timeout=15.0)
            if r.status_code == 200:
                for m in r.json():
                    mid = m.get("modelId") or m.get("id", "")
                    if mid != model_id:
                        variants.append({
                            "id": mid,
                            "format": tag.upper(),
                            "downloads": m.get("downloads", 0),
                        })
        except Exception:
            pass
    # deduplicate by id
    seen = set()
    deduped = []
    for v in variants:
        if v["id"] not in seen:
            seen.add(v["id"])
            deduped.append(v)
    return {"variants": deduped}

@app.get("/api/hf/model/{owner}/{name}/files")
async def hf_model_files(owner: str, name: str):
    """List files in a HuggingFace repo with sizes."""
    try:
        r = await _http.get(f"https://huggingface.co/api/models/{owner}/{name}",
                            params={"full": "true"}, timeout=15.0)
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        raise HTTPException(502, f"HuggingFace API error: {e}")
    siblings = d.get("siblings", [])
    files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if fname.startswith("."):
            continue
        files.append({"name": fname, "size": s.get("size")})
    return {"files": files, "total": len(files)}

# ── Config Management ─────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    """Return current running configuration. API key is masked."""
    services = {"ollama_base": OLLAMA_BASE, "litellm_base": LITELLM_BASE}
    for key, eng in _ENGINES.items():
        services[eng["config_key"]] = _engine_bases[key]
    return {
        "app": {"host": "0.0.0.0", "port": APP_PORT, "api_key_set": bool(_API_KEY_HASH)},
        "services": services,
        "paths": {
            "litellm_config": str(LITELLM_CONFIG),
            "hf_cache":       str(HF_CACHE_DIR),
        },
    }


@app.post("/api/auth/check")
async def auth_check(request: Request):
    """Verify an API key is correct. Returns ok:true if valid or if no key is set."""
    if not _API_KEY_HASH:
        return {"ok": True, "auth_required": False}
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        incoming_hash = _hash_key(auth[7:])
        if hmac.compare_digest(incoming_hash, _API_KEY_HASH):
            return {"ok": True, "auth_required": True}
    return {"ok": False, "auth_required": True}


@app.get("/api/sudo/check")
async def sudo_check():
    """Check if passwordless sudo works for the commands the app needs."""
    checks = {}
    # LiteLLM restart
    try:
        r = await _run("sudo", "-n", "systemctl", "restart", "--dry-run", "litellm", timeout=5)
        if r.returncode != 0:
            r = await _run("sudo", "-n", "true", timeout=5)
        checks["systemctl"] = r.returncode == 0
    except Exception:
        checks["systemctl"] = False
    # Docker (without sudo — user may be in docker group)
    try:
        r = await _run("docker", "ps", "--format", "{{.ID}}", timeout=5)
        checks["docker"] = r.returncode == 0
    except Exception:
        checks["docker"] = False
    return checks


class ConfigUpdate(BaseModel):
    services: Optional[dict] = None
    api_key: Optional[str] = None


@app.put("/api/config", dependencies=[Depends(verify_auth)])
async def update_config(req: ConfigUpdate):
    """Update service URLs and/or API key, save to config.json, and apply in-memory."""
    global OLLAMA_BASE, LITELLM_BASE, _API_KEY_HASH

    svc = req.services or {}
    # Validate all URL values
    all_url_keys = {"ollama_base", "litellm_base"} | {eng["config_key"] for eng in _ENGINES.values()}
    for key in svc:
        if key in all_url_keys:
            _validate_service_url(svc[key], key)
    # Apply special service URLs
    if "ollama_base" in svc:
        OLLAMA_BASE = svc["ollama_base"].rstrip("/")
    if "litellm_base" in svc:
        LITELLM_BASE = svc["litellm_base"].rstrip("/")
    # Apply engine URLs from registry
    for key, eng in _ENGINES.items():
        ck = eng["config_key"]
        if ck in svc:
            _engine_bases[key] = svc[ck].rstrip("/")

    # Persist to config.json
    cfg = {}
    if _CONFIG_FILE.exists():
        try:
            cfg = json.loads(_CONFIG_FILE.read_text())
        except Exception:
            pass
    cfg.setdefault("services", {})
    cfg["services"]["ollama_base"] = OLLAMA_BASE
    cfg["services"]["litellm_base"] = LITELLM_BASE
    for key, eng in _ENGINES.items():
        cfg["services"][eng["config_key"]] = _engine_bases[key]

    # Update API key if provided (empty string clears it)
    if req.api_key is not None:
        if req.api_key:
            _API_KEY_HASH = _hash_key(req.api_key)
        else:
            _API_KEY_HASH = ""
        cfg.setdefault("app", {})
        cfg["app"]["api_key"] = _API_KEY_HASH  # store hash, never plaintext

    try:
        _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except OSError as e:
        _logger.error("Failed to write config: %s", e)
        raise HTTPException(500, f"Config applied in memory but failed to save to disk: {e}")
    _logger.info("Config updated: %s", ", ".join(list(svc.keys()) + (["api_key"] if req.api_key is not None else [])))

    return {"ok": True, "services": cfg["services"], "api_key_set": bool(_API_KEY_HASH)}


class TestServiceRequest(BaseModel):
    url: str
    type: str  # ollama, litellm, sglang, vllm


@app.post("/api/test-service", dependencies=[Depends(verify_auth)])
async def test_service(req: TestServiceRequest):
    """Test connectivity to a service endpoint."""
    health_paths = {"ollama": "/api/tags", "litellm": "/health"}
    for key, eng in _ENGINES.items():
        health_paths[key] = eng.get("health_path", "/health")
    _validate_service_url(req.url, req.type)
    path = health_paths.get(req.type, "/health")
    url = req.url.rstrip("/") + path
    try:
        t0 = _time.monotonic()
        r = await _http.get(url, timeout=5.0)
        latency_ms = round((_time.monotonic() - t0) * 1000)
        if r.status_code < 400:
            return {"ok": True, "latency_ms": latency_ms}
        return {"ok": False, "latency_ms": latency_ms, "error": f"HTTP {r.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "error": "Connection refused"}
    except httpx.ConnectTimeout:
        return {"ok": False, "error": "Connection timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── Debug & Logs ─────────────────────────────────────────────────────────────

@app.get("/api/debug/system")
async def debug_system():
    """Comprehensive system overview for diagnostics."""
    # Service health checks with response time (parallel)
    async def _check(name, base, path):
        try:
            t0 = _time.monotonic()
            r = await _http.get(base + path, timeout=3.0)
            ms = round((_time.monotonic() - t0) * 1000)
            return name, {"url": base, "healthy": r.status_code < 400, "response_ms": ms}
        except Exception:
            return name, {"url": base, "healthy": False, "response_ms": None}

    check_coros = [
        _check("ollama", OLLAMA_BASE, "/api/tags"),
        _check("litellm", LITELLM_BASE, "/health"),
    ]
    for key, eng in _ENGINES.items():
        check_coros.append(_check(key, _engine_bases[key], eng.get("health_path", "/health")))
    checks = await asyncio.gather(*check_coros)
    services = {name: info for name, info in checks}

    # Disk usage for HF cache
    disk = {}
    for label, path in [("hf_cache", HF_CACHE_DIR)]:
        try:
            usage = shutil.disk_usage(str(path))
            disk[label] = {
                "path": str(path),
                "total_gb": round(usage.total / 1e9, 1),
                "free_gb": round(usage.free / 1e9, 1),
                "used_pct": round((usage.used / usage.total) * 100, 1),
            }
        except Exception:
            disk[label] = {"path": str(path), "error": "unavailable"}

    # Sudo/docker permissions
    perms = {"systemctl": False, "docker": False}
    try:
        r = await _run("sudo", "-n", "systemctl", "restart", "--dry-run", "litellm", timeout=5)
        perms["systemctl"] = r.returncode == 0
    except Exception:
        pass
    try:
        r = await _run("docker", "ps", "--format", "{{.ID}}", timeout=5)
        perms["docker"] = r.returncode == 0
    except Exception:
        pass

    return {
        "hostname": socket.gethostname(),
        "ip": _get_local_ip(),
        "arch": platform.machine(),
        "memory_gb": _get_total_memory_gb(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "app_port": APP_PORT,
        "app_start_utc": _APP_START_UTC,
        "uptime_seconds": int(_time.monotonic() - _APP_START),
        "api_key_set": bool(_API_KEY_HASH),
        "disk": disk,
        "services": services,
        "permissions": perms,
    }


@app.get("/api/debug/config")
async def debug_config():
    """Return organized running configuration."""
    litellm_cfg = None
    litellm_raw = ""
    try:
        litellm_cfg = load_litellm_config()
        litellm_raw = Path(os.path.expanduser(str(LITELLM_CONFIG))).read_text()
    except Exception:
        pass
    services = {"ollama_base": OLLAMA_BASE, "litellm_base": LITELLM_BASE}
    paths = {"litellm_config": str(LITELLM_CONFIG), "hf_cache": str(HF_CACHE_DIR)}
    engine_profiles = {}
    for key, eng in _ENGINES.items():
        services[eng["config_key"]] = _engine_bases[key]
        paths[key + "_scripts"] = str(_engine_dirs[key])
        engine_profiles[key] = _scan_profiles(key)
    return {
        "app": {
            "port": APP_PORT,
            "api_key_set": bool(_API_KEY_HASH),
            "config_file": str(_CONFIG_FILE),
            "start_utc": _APP_START_UTC,
        },
        "services": services,
        "paths": paths,
        "litellm": {"parsed": litellm_cfg, "raw": litellm_raw},
        "engine_profiles": engine_profiles,
    }


@app.get("/api/logs/app")
async def get_app_logs(level: str = None, search: str = None, limit: int = 200):
    """Return recent application log entries from the in-memory ring buffer."""
    entries = _log_handler.get_entries(level=level, search=search, limit=limit)
    return {"entries": entries, "total": len(_log_handler.buffer), "buffer_size": _log_handler.maxlen}


@app.delete("/api/logs/app", dependencies=[Depends(verify_auth)])
async def clear_app_logs():
    """Clear the in-memory log buffer."""
    _log_handler.clear()
    _logger.info("Log buffer cleared")
    return {"ok": True}


@app.get("/api/logs/engine/{engine}")
async def get_engine_logs(engine: str, lines: int = 150, search: str = None):
    """Read log files for SGLang or vLLM engine."""
    if engine not in _ENGINES:
        raise HTTPException(400, f"Unknown engine '{engine}'")
    import glob
    log_files = sorted(glob.glob(f"/tmp/{engine}_*.log"), key=lambda f: os.path.getmtime(f), reverse=True)
    if not log_files:
        return {"file": None, "lines": [], "total_lines": 0, "available_files": []}
    target = log_files[0]
    try:
        with open(target, "r", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return {"file": target, "lines": [str(e)], "total_lines": 0, "available_files": log_files}
    if search:
        s = search.lower()
        all_lines = [l for l in all_lines if s in l.lower()]
    result = [l.rstrip("\n") for l in all_lines[-lines:]]
    return {"file": target, "lines": result, "total_lines": len(all_lines), "available_files": log_files}


@app.get("/api/logs/litellm")
async def get_litellm_logs(lines: int = 100, search: str = None):
    """Read LiteLLM service logs from journalctl."""
    for cmd in (
        ["journalctl", "-u", "litellm", "--no-pager", "-n", str(lines), "--output=short-iso"],
        ["sudo", "-n", "journalctl", "-u", "litellm", "--no-pager", "-n", str(lines), "--output=short-iso"],
    ):
        r = await _run(*cmd, timeout=10)
        if r.returncode == 0:
            result = r.stdout.strip().split("\n") if r.stdout.strip() else []
            if search:
                s = search.lower()
                result = [l for l in result if s in l.lower()]
            return {"lines": result, "available": True, "error": None}
    return {"lines": [], "available": False, "error": "journalctl access denied — add user to systemd-journal group or configure sudo"}


@app.get("/api/debug/docker")
async def debug_docker():
    """Return running Docker container state."""
    r = await _run("docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.CreatedAt}}", timeout=10)
    if r.returncode != 0:
        return {"containers": [], "available": False, "error": (r.stdout + r.stderr).strip()}
    containers = []
    for line in r.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 6:
            containers.append({
                "id": parts[0][:12],
                "name": parts[1],
                "image": parts[2],
                "status": parts[3],
                "ports": parts[4],
                "created": parts[5],
            })
    return {"containers": containers, "available": True}

# ─── Frontend ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DGX · Model Manager</title>
<link rel="icon" type="image/png" href="/favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #08080c;
  --s1:       #0f0f14;
  --s2:       #16161e;
  --s3:       #1e1e28;
  --border:   #252535;
  --border2:  #30304a;
  --text:     #d4d4e8;
  --muted:    #6a6a90;
  --amber:    #f0a034;
  --amber2:   #c07020;
  --amber-bg: #1a120400;
  --green:    #3dba78;
  --red:      #e05050;
  --blue:     #5a9af5;
  --purple:   #9a6af5;
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'Space Grotesk', sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:var(--bg);
  color:var(--text);
  font-family:var(--sans);
  font-size:14px;
  line-height:1.5;
  display:flex;
  flex-direction:column;
  height:100vh;
  overflow:hidden;
}

/* ── Scanline texture ── */
body::before{
  content:'';
  position:fixed;inset:0;
  background:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,.08) 2px,
    rgba(0,0,0,.08) 4px
  );
  pointer-events:none;
  z-index:1000;
  opacity:.4;
}

/* ── Header ── */
.header{
  display:flex;align-items:center;gap:20px;
  padding:0 20px;height:52px;
  border-bottom:1px solid var(--border);
  background:var(--s1);
  flex-shrink:0;
  position:relative;
  z-index:10;
}
.hdr-logo{
  display:flex;align-items:center;gap:10px;
}
.hdr-sigil{
  width:28px;height:28px;
  background:var(--amber);
  clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:700;color:#000;
  font-family:var(--mono);
  flex-shrink:0;
}
.hdr-name{
  font-family:var(--mono);
  font-size:12px;font-weight:600;
  letter-spacing:.12em;
  color:var(--amber);
  text-transform:uppercase;
}
.hdr-node{
  font-family:var(--mono);
  font-size:10px;color:var(--muted);
  letter-spacing:.06em;
}
.hdr-sep{flex:1}

/* ── Status pills ── */
.status-cluster{display:flex;gap:6px;align-items:center}
.pill{
  display:flex;align-items:center;gap:5px;
  padding:3px 10px 3px 7px;
  border-radius:20px;
  border:1px solid var(--border);
  background:var(--s2);
  font-family:var(--mono);
  font-size:10px;
  color:var(--muted);
  transition:border-color .2s,color .2s;
  cursor:default;
  white-space:nowrap;
}
.pill.ok{border-color:#1e3a28;color:#8dd4a8}
.pill.err{border-color:#3a1818;color:#e08888}
.dot{width:5px;height:5px;border-radius:50%;background:var(--muted);transition:background .3s,box-shadow .3s}
.pill.ok .dot{background:var(--green);box-shadow:0 0 5px var(--green)}
.pill.err .dot{background:var(--red)}
.refresh-btn{
  width:26px;height:26px;
  border-radius:6px;
  border:1px solid var(--border);
  background:var(--s2);
  color:var(--muted);
  cursor:pointer;
  font-size:13px;
  display:flex;align-items:center;justify-content:center;
  transition:all .15s;
}
.refresh-btn:hover{border-color:var(--amber);color:var(--amber)}

/* ── Layout ── */
.body-wrap{display:flex;flex:1;overflow:hidden}
.sidebar{
  width:192px;flex-shrink:0;
  border-right:1px solid var(--border);
  background:var(--s1);
  display:flex;flex-direction:column;
  padding:12px 0;
  overflow-y:auto;
}
.nav-section-label{
  font-family:var(--mono);
  font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;
  color:var(--muted);
  padding:12px 16px 6px;
  opacity:.6;
}
.nav-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 16px;
  font-size:13px;font-weight:500;
  color:var(--muted);
  cursor:pointer;
  border-left:2px solid transparent;
  transition:all .12s;
  user-select:none;
}
.nav-item:hover{color:var(--text);background:var(--s2)}
.nav-item.active{
  color:var(--amber);
  border-left-color:var(--amber);
  background:linear-gradient(90deg,rgba(240,160,52,.07),transparent);
}
.nav-icon{font-size:14px;width:16px;text-align:center;flex-shrink:0}
.nav-badge{
  margin-left:auto;
  background:var(--s3);border:1px solid var(--border);
  border-radius:10px;padding:1px 7px;
  font-family:var(--mono);font-size:10px;
  color:var(--muted);
}

/* ── Main ── */
.main{flex:1;overflow-y:auto;padding:24px}
.tab{display:none}
.tab.active{display:block;animation:fadein .15s ease}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}

/* ── Page header ── */
.page-hdr{margin-bottom:20px}
.page-title{font-size:18px;font-weight:700;letter-spacing:-.01em}
.page-sub{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.6}
.page-sub code{font-family:var(--mono);color:var(--amber);font-size:11px}

/* ── Section label ── */
.sec-label{
  font-family:var(--mono);
  font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;
  color:var(--muted);
  margin:20px 0 10px;
  display:flex;align-items:center;gap:10px;
}
.sec-label::after{content:'';flex:1;height:1px;background:var(--border)}

/* ── Cards ── */
.card{
  background:var(--s1);
  border:1px solid var(--border);
  border-radius:8px;
  padding:16px 18px;
  margin-bottom:10px;
}
.card-row{display:flex;align-items:flex-start;gap:12px}
.card-icon{
  width:32px;height:32px;flex-shrink:0;
  background:var(--s2);border:1px solid var(--border);
  border-radius:7px;
  display:flex;align-items:center;justify-content:center;
  font-size:14px;
}
.card-info{flex:1;min-width:0}
.card-name{font-size:13px;font-weight:600}
.card-meta{font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:2px}
.card-actions{margin-left:auto;display:flex;gap:6px;align-items:center;flex-shrink:0}
.card-desc{font-size:12px;color:var(--muted);line-height:1.6;margin-top:10px}
.card-desc code{font-family:var(--mono);color:var(--amber);font-size:11px}

/* ── Model grid ── */
.model-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  gap:8px;
  margin-bottom:8px;
}
.model-card{
  background:var(--s1);border:1px solid var(--border);
  border-radius:8px;padding:12px 14px;
  display:flex;align-items:center;gap:10px;
  transition:border-color .15s;
}
.model-card:hover{border-color:var(--border2)}
.model-card-info{flex:1;min-width:0}
.model-card-name{
  font-family:var(--mono);font-size:12px;font-weight:500;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.model-card-meta{font-size:11px;color:var(--muted);margin-top:2px}
.model-card-right{display:flex;align-items:center;gap:6px;flex-shrink:0}

/* ── Tags ── */
.tag{
  display:inline-block;padding:2px 7px;
  border-radius:4px;font-size:9px;
  font-family:var(--mono);font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;
}
.tag-ollama{background:#0e2018;color:#5cc480;border:1px solid #1a3a28}
.tag-sglang{background:#101828;color:#6898e8;border:1px solid #1a2a40}
.tag-amber{background:#1a1000;color:var(--amber);border:1px solid #2a2000}

/* ── Buttons ── */
.btn{
  display:inline-flex;align-items:center;gap:6px;
  padding:7px 14px;border-radius:6px;
  font-size:12px;font-weight:600;font-family:var(--sans);
  cursor:pointer;border:1px solid var(--border);
  background:var(--s2);color:var(--text);
  transition:all .12s;white-space:nowrap;
  line-height:1;
}
.btn:hover{border-color:var(--border2);background:var(--s3)}
.btn:active{transform:scale(.97)}
.btn:disabled{opacity:.35;cursor:not-allowed;pointer-events:none}
.btn-primary{background:var(--amber);color:#000;border-color:var(--amber)}
.btn-primary:hover{background:var(--amber2);border-color:var(--amber2);color:#000}
.btn-danger{background:#180808;color:#e08888;border-color:#2a1010}
.btn-danger:hover{border-color:var(--red);color:var(--red)}
.btn-sm{padding:4px 10px;font-size:11px}
.btn-ghost{background:transparent;border-color:transparent;color:var(--muted)}
.btn-ghost:hover{color:var(--text);background:var(--s2);border-color:var(--border)}

/* ── Input ── */
.input-row{display:flex;gap:8px;align-items:stretch;margin-bottom:14px}
.input{
  flex:1;
  background:var(--s2);border:1px solid var(--border);
  border-radius:6px;padding:8px 12px;
  color:var(--text);font-size:13px;font-family:var(--mono);
  outline:none;transition:border-color .15s;
}
.input:focus{border-color:var(--amber)}
.input::placeholder{color:var(--muted)}

/* ── Progress ── */
.progress-wrap{margin-top:10px;display:none}
.progress-wrap.show{display:block}
.prog-bar-outer{height:3px;background:var(--s3);border-radius:2px;overflow:hidden;margin-bottom:8px}
.prog-bar{height:100%;background:var(--amber);border-radius:2px;transition:width .3s;width:0}
.prog-bar.spin{width:35%!important;animation:pgslide 1.2s ease-in-out infinite}
@keyframes pgslide{0%{transform:translateX(-200%)}100%{transform:translateX(500%)}}
.prog-log{
  font-family:var(--mono);font-size:11px;color:var(--muted);
  background:#04040a;border:1px solid var(--border);
  border-radius:5px;padding:8px 10px;
  max-height:110px;overflow-y:auto;
  line-height:1.7;
  white-space:pre-wrap;
}

/* ── Engine card ── */
.engine-card{
  background:var(--s1);border:1px solid var(--border);
  border-radius:10px;padding:20px 22px;margin-bottom:14px;
  position:relative;overflow:hidden;
}
.engine-card::before{
  content:'';position:absolute;
  top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--amber),transparent);
  opacity:0;transition:opacity .3s;
}
.engine-card.online::before{opacity:1}
.engine-status-row{display:flex;align-items:center;gap:14px;margin-bottom:12px}
.engine-led{
  width:10px;height:10px;border-radius:50%;
  background:var(--red);flex-shrink:0;
  transition:background .3s,box-shadow .3s;
}
.engine-led.on{background:var(--green);box-shadow:0 0 8px var(--green)}
.engine-title{font-size:15px;font-weight:700}
.engine-model{font-size:11px;color:var(--amber);font-family:var(--mono);margin-top:2px}
.engine-footer{font-size:11px;color:var(--muted);font-family:var(--mono)}
.engine-actions{margin-left:auto;display:flex;gap:6px}

/* ── Profile list ── */
.profile-list{display:flex;flex-direction:column;gap:6px}
.profile-item{
  display:flex;align-items:center;gap:12px;
  padding:12px 14px;
  background:var(--s1);border:1px solid var(--border);
  border-radius:8px;cursor:pointer;
  transition:border-color .15s;
}
.profile-item:hover{border-color:var(--border2)}
.profile-item.selected{border-color:var(--amber);background:linear-gradient(90deg,rgba(240,160,52,.05),transparent)}
.p-radio{
  width:14px;height:14px;border-radius:50%;
  border:2px solid var(--border);flex-shrink:0;
  transition:all .15s;
}
.profile-item.selected .p-radio{border-color:var(--amber);background:var(--amber);box-shadow:0 0 6px var(--amber)}
.p-info{flex:1;min-width:0}
.p-name{font-size:13px;font-weight:600}
.p-desc{font-size:11px;color:var(--muted);margin-top:2px}
.p-vram{font-family:var(--mono);font-size:11px;color:var(--amber);flex-shrink:0}

/* ── Config block ── */
.config-block{
  font-family:var(--mono);font-size:11px;
  background:#04040a;border:1px solid var(--border);
  border-radius:6px;padding:14px;
  overflow:auto;max-height:280px;
  color:#a0a0c0;line-height:1.8;
  white-space:pre;
}

/* ── Wildcard status ── */
.wc-active{
  display:flex;align-items:center;gap:7px;
  font-size:12px;color:#5cc480;
  margin-top:8px;
}
.wc-inactive{font-size:12px;color:var(--muted);margin-top:8px}

/* ── Empty state ── */
.empty{
  text-align:center;padding:40px 20px;
  color:var(--muted);
}
.empty-icon{font-size:28px;margin-bottom:10px;opacity:.5}
.empty-text{font-size:13px}

/* ── Spinner ── */
.spin-icon{
  width:13px;height:13px;
  border:2px solid rgba(255,255,255,.15);
  border-top-color:currentColor;
  border-radius:50%;
  animation:spin .6s linear infinite;
  flex-shrink:0;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Toast ── */
#toast-root{
  position:fixed;bottom:20px;right:20px;
  display:flex;flex-direction:column;gap:6px;
  z-index:9999;pointer-events:none;
}
.toast{
  background:var(--s2);border:1px solid var(--border);
  border-radius:8px;padding:10px 14px;
  font-size:13px;max-width:320px;
  pointer-events:auto;
  animation:toast-in .2s ease;
}
@keyframes toast-in{from{transform:translateX(100%);opacity:0}to{opacity:1;transform:none}}
.toast.ok{border-color:#1e3a28}
.toast.err{border-color:#3a1818}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* ── Inventory table ── */
.inv-dir-block{margin-bottom:24px}
.inv-dir-header{
  display:flex;align-items:center;gap:10px;
  padding:10px 14px;
  background:var(--s2);border:1px solid var(--border);
  border-radius:8px 8px 0 0;
  border-bottom:1px solid var(--border2);
}
.inv-dir-path{
  font-family:var(--mono);font-size:11px;color:var(--amber);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.inv-dir-badge{
  font-family:var(--mono);font-size:9px;
  padding:2px 7px;border-radius:3px;
  background:rgba(240,160,52,.12);border:1px solid rgba(240,160,52,.25);
  color:var(--amber);white-space:nowrap;
}
.inv-dir-badge.custom{
  background:rgba(90,154,245,.1);border-color:rgba(90,154,245,.25);color:var(--blue);
}
.inv-table-wrap{
  border:1px solid var(--border);border-top:none;
  border-radius:0 0 8px 8px;overflow:hidden;
}
.inv-table{width:100%;border-collapse:collapse;font-size:12px;}
.inv-table th{
  font-family:var(--mono);font-size:9px;font-weight:600;
  letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);background:var(--s2);
  padding:7px 10px;text-align:left;
  border-bottom:1px solid var(--border);
  white-space:nowrap;
}
.inv-table td{
  padding:9px 10px;border-bottom:1px solid var(--border);
  vertical-align:middle;
  color:var(--text);
}
.inv-table tr:last-child td{border-bottom:none}
.inv-table tr:hover td{background:rgba(255,255,255,.02)}
.inv-model-name{font-family:var(--mono);font-size:11px;font-weight:500;color:var(--text)}
.inv-owner{font-size:10px;color:var(--muted);margin-top:1px}
.inv-badge{
  display:inline-block;padding:2px 6px;border-radius:3px;
  font-family:var(--mono);font-size:9px;font-weight:600;
  letter-spacing:.05em;white-space:nowrap;
  margin:1px;
}
.inv-fp32{background:#0a1830;color:#6090e0;border:1px solid #1030508}
.inv-fp16{background:#0e1a30;color:#5a9af5;border:1px solid #1a2a50}
.inv-bf16{background:#0e1a30;color:#5a9af5;border:1px solid #1a2a50}
.inv-fp8{background:#12200e;color:#5cc480;border:1px solid #1a3a18}
.inv-fp4{background:#1a1200;color:var(--amber);border:1px solid #2a2000}
.inv-int4{background:#1a1200;color:#d09030;border:1px solid #2a1800}
.inv-int8{background:#0e1818;color:#40b0b0;border:1px solid #183030}
.inv-unknown{background:var(--s3);color:var(--muted);border:1px solid var(--border)}
.inv-moe{background:#1a0a28;color:#a06af5;border:1px solid #2a1040}
.inv-dense{background:var(--s3);color:var(--muted);border:1px solid var(--border)}
.inv-yes{color:var(--green)}
.inv-no{color:var(--muted)}
.inv-engine{
  font-family:var(--mono);font-size:10px;
  padding:2px 6px;border-radius:3px;
}
.inv-engine-sg{background:#101828;color:#6898e8;border:1px solid #1a2a40}
.inv-engine-vl{background:#0e1a14;color:#5cc480;border:1px solid #1a3020}
.inv-modality{
  display:inline-block;padding:1px 5px;border-radius:3px;
  font-family:var(--mono);font-size:9px;
  background:var(--s3);color:var(--muted);border:1px solid var(--border);
  margin:1px;
}
.inv-modality.embed{
  background:#10102a;color:#8888e0;border-color:#20205a;
}
.inv-modality.audio{
  background:#0e1a1a;color:#50c8b8;border-color:#183838;
}
.inv-empty{
  text-align:center;padding:28px;color:var(--muted);
  font-size:12px;
  background:var(--s1);border:1px solid var(--border);
  border-top:none;border-radius:0 0 8px 8px;
}
.btn-icon-del{
  background:none;border:1px solid transparent;border-radius:4px;
  color:var(--muted);font-size:13px;cursor:pointer;
  width:26px;height:26px;display:flex;align-items:center;justify-content:center;
  transition:all .12s;padding:0;
}
.btn-icon-del:hover{color:var(--red);border-color:var(--red);background:#180808}
.inv-custom-dirs{
  display:flex;flex-direction:column;gap:6px;
  margin-bottom:14px;
}
.inv-custom-dir-row{
  display:flex;align-items:center;gap:8px;
  padding:8px 12px;
  background:var(--s2);border:1px solid var(--border);
  border-radius:6px;
}
.inv-custom-dir-path{font-family:var(--mono);font-size:11px;color:var(--blue);flex:1}
.inv-remove-btn{
  background:none;border:1px solid var(--border);
  color:var(--muted);font-size:11px;
  border-radius:4px;padding:2px 8px;cursor:pointer;
  transition:all .12s;
}
.inv-remove-btn:hover{color:var(--red);border-color:var(--red)}

/* ── Inventory toolbar ── */
.inv-toolbar{
  display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  margin-bottom:12px;
}
.inv-search{
  flex:1;min-width:160px;max-width:280px;
  font-size:12px !important;padding:6px 10px !important;
}
.inv-filter{
  background:var(--s2);color:var(--text);border:1px solid var(--border);
  border-radius:6px;padding:5px 8px;font-size:11px;font-family:var(--sans);
  cursor:pointer;outline:none;
}
.inv-filter:focus{border-color:var(--amber)}
.inv-stats{
  font-family:var(--mono);font-size:11px;color:var(--muted);
  margin-bottom:14px;padding:6px 0;
  display:flex;gap:16px;flex-wrap:wrap;
}
.inv-stats span{color:var(--amber)}
.inv-dirs-section{
  margin-bottom:14px;border:1px solid var(--border);border-radius:8px;
  background:var(--s1);
}
.inv-dirs-section summary{
  padding:10px 14px;cursor:pointer;font-family:var(--mono);
  font-size:11px;color:var(--muted);letter-spacing:.08em;
  text-transform:uppercase;user-select:none;
}
.inv-dirs-section summary:hover{color:var(--text)}
.inv-dirs-section[open] > summary{border-bottom:1px solid var(--border)}
.inv-dirs-section > div,.inv-dirs-section > .input-row{padding:10px 14px}
.inv-source-badge{
  display:inline-block;padding:2px 6px;border-radius:3px;
  font-family:var(--mono);font-size:9px;font-weight:600;
  letter-spacing:.05em;white-space:nowrap;margin:1px;
}
.inv-src-hf{background:#1a1200;color:var(--amber);border:1px solid #2a2000}
.inv-src-custom{background:rgba(90,154,245,.1);color:var(--blue);border:1px solid rgba(90,154,245,.25)}
.inv-src-ollama{background:#0e1a14;color:#5cc480;border:1px solid #1a3020}
.inv-format-badge{
  display:inline-block;padding:2px 6px;border-radius:3px;
  font-family:var(--mono);font-size:9px;font-weight:600;
  letter-spacing:.04em;white-space:nowrap;margin:1px;
  background:var(--s3);color:var(--muted);border:1px solid var(--border);
}
.inv-fmt-safe{background:#101828;color:#6898e8;border:1px solid #1a2a40}
.inv-fmt-gguf{background:#1a0a28;color:#a06af5;border:1px solid #2a1040}
.inv-fmt-pt{background:#1a1200;color:#d09030;border:1px solid #2a1800}
.inv-fmt-ollama{background:#0e1a14;color:#5cc480;border:1px solid #1a3020}
.inv-task-badge{
  display:inline-block;padding:2px 6px;border-radius:3px;
  font-family:var(--mono);font-size:9px;font-weight:600;
  letter-spacing:.04em;white-space:nowrap;margin:1px;
  background:rgba(90,154,245,.08);color:#5a9af5;border:1px solid rgba(90,154,245,.2);
}

/* ── HF Browse ── */
.hfb-search-bar{
  display:flex;gap:8px;align-items:center;flex-wrap:wrap;
  margin-bottom:18px;
}
.hfb-card{
  background:var(--s1);border:1px solid var(--border);border-radius:8px;
  padding:14px 16px;margin-bottom:10px;transition:border-color .15s;
}
.hfb-card:hover{border-color:var(--amber)}
.hfb-card-hdr{display:flex;align-items:flex-start;gap:10px;margin-bottom:8px}
.hfb-card-name{
  font-family:var(--mono);font-size:13px;font-weight:600;color:var(--text);
  flex:1;word-break:break-word;
}
.hfb-card-meta{
  display:flex;gap:12px;align-items:center;font-size:11px;color:var(--muted);
  margin-bottom:8px;flex-wrap:wrap;
}
.hfb-card-meta .dl{color:#5cc480}
.hfb-card-meta .lk{color:#e060a0}
.hfb-tags{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}
.hfb-tag{
  display:inline-block;padding:1px 6px;border-radius:3px;
  font-family:var(--mono);font-size:9px;
  background:var(--s3);color:var(--muted);border:1px solid var(--border);
}
.hfb-tag.fmt{background:rgba(90,154,245,.08);color:#5a9af5;border-color:rgba(90,154,245,.2)}
.hfb-card-actions{display:flex;gap:8px;align-items:center;margin-top:10px}
.hfb-expand{
  margin-top:10px;padding-top:10px;border-top:1px solid var(--border);
  font-size:12px;
}
.hfb-expand-toggle{
  background:none;border:none;color:var(--muted);font-size:11px;
  cursor:pointer;font-family:var(--mono);padding:0;
}
.hfb-expand-toggle:hover{color:var(--amber)}
.hfb-file-list{
  max-height:250px;overflow-y:auto;margin-top:8px;
  font-family:var(--mono);font-size:10px;
}
.hfb-file-row{
  display:flex;gap:8px;padding:3px 0;border-bottom:1px solid var(--border);
  color:var(--text);
}
.hfb-file-row .size{color:var(--muted);margin-left:auto;white-space:nowrap}
.hfb-variants{margin-top:10px}
.hfb-variant-row{
  display:flex;align-items:center;gap:8px;padding:4px 0;
  font-family:var(--mono);font-size:11px;
}
.hfb-variant-row .fmt{
  padding:2px 6px;border-radius:3px;font-size:9px;font-weight:600;
  background:#1a0a28;color:#a06af5;border:1px solid #2a1040;
}
.hfb-loading{text-align:center;padding:20px;color:var(--muted);font-size:12px}

/* ── Debug / Logs ── */
.debug-grid{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:10px;margin-bottom:14px;
}
.debug-stat{
  background:var(--s2);border:1px solid var(--border);border-radius:6px;
  padding:10px 12px;
}
.debug-stat-label{
  font-family:var(--mono);font-size:9px;text-transform:uppercase;
  letter-spacing:.1em;color:var(--muted);margin-bottom:4px;
}
.debug-stat-value{
  font-family:var(--mono);font-size:13px;font-weight:600;color:var(--text);
}
.debug-stat-value.ok{color:var(--green)}
.debug-stat-value.err{color:var(--red)}
.debug-stat-value.warn{color:var(--amber)}
.debug-section-hdr{
  font-family:var(--mono);font-size:12px;font-weight:600;cursor:pointer;
  color:var(--text);list-style:none;display:flex;align-items:center;gap:8px;
}
.debug-section-hdr::before{content:'▸';color:var(--muted);transition:transform .15s;font-size:10px}
details[open]>.debug-section-hdr::before{transform:rotate(90deg)}
.config-block{
  font-family:var(--mono);font-size:11px;line-height:1.6;
  background:#04040a;border:1px solid var(--border);border-radius:6px;
  padding:10px 12px;white-space:pre-wrap;word-break:break-all;
  max-height:350px;overflow-y:auto;color:var(--text);
}
.log-toolbar{
  display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;
}
.log-level-select{width:120px;flex:none}
.log-auto-label{
  font-family:var(--mono);font-size:11px;color:var(--muted);
  display:flex;align-items:center;gap:4px;cursor:pointer;white-space:nowrap;
}
.log-auto-label input[type="checkbox"]{accent-color:var(--amber)}
.log-pane{
  font-family:var(--mono);font-size:11px;background:#04040a;
  border:1px solid var(--border);border-radius:6px;padding:8px 10px;
  max-height:400px;overflow-y:auto;line-height:1.7;
  white-space:pre-wrap;word-break:break-all;
}
.log-footer{
  font-family:var(--mono);font-size:10px;color:var(--muted);
  margin-top:6px;text-align:right;
}
.log-entry{padding:1px 0}
.log-ts{color:var(--muted)}
.log-src{color:var(--blue)}
.log-level-DEBUG{color:#6a6a90}
.log-level-INFO{color:#8dd4a8}
.log-level-WARNING{color:#f0c050}
.log-level-ERROR{color:#e05050;font-weight:600}
.log-tab-bar{display:flex;gap:4px}
.log-tab-btn.active{background:var(--amber);color:#000;border-color:var(--amber)}
.btn-danger{background:#2a0808;color:#e05050;border:1px solid #401010}
.btn-danger:hover{background:#3a0a0a;border-color:#e05050}
.docker-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}
.docker-table th{
  text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);
  color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.1em;
}
.docker-table td{padding:6px 8px;border-bottom:1px solid var(--border);color:var(--text)}
.docker-table tr:last-child td{border-bottom:none}
.docker-table tr:hover td{background:var(--s2)}

/* ── Settings ── */
.svc-status{
  font-family:var(--mono);font-size:11px;
  display:flex;align-items:center;gap:6px;
  min-width:90px;justify-content:flex-end;
}
.svc-status.ok{color:var(--green)}
.svc-status.err{color:var(--red)}
.svc-status.testing{color:var(--muted)}</style>
</head>
<body>

<header class="header">
  <div class="hdr-logo">
    <div class="hdr-sigil">D</div>
    <div>
      <div class="hdr-name">Model Manager</div>
      <div class="hdr-node" id="hdr-node">loading…</div>
    </div>
  </div>
  <div class="hdr-sep"></div>
  <div class="status-cluster">
    <div class="pill" id="pill-ollama"><div class="dot"></div><span>Ollama</span></div>
    <div class="pill" id="pill-litellm"><div class="dot"></div><span>LiteLLM</span></div>
""" + "".join(f'    <div class="pill" id="pill-{k}"><div class="dot"></div><span>{e["name"]}</span></div>\n' for k, e in _ENGINES.items()) + r"""
    <button class="refresh-btn" onclick="pollStatus()" title="Refresh status">↻</button>
    <a href="/help" target="_blank" style="font-family:var(--mono);font-size:10px;color:var(--muted);text-decoration:none;padding:4px 10px;border:1px solid var(--border);border-radius:5px;transition:all .15s;" onmouseover="this.style.color='var(--amber)';this.style.borderColor='var(--amber)'" onmouseout="this.style.color='var(--muted)';this.style.borderColor='var(--border)'">? Docs</a>
  </div>
</header>

<div class="body-wrap">
  <nav class="sidebar">
    <div class="nav-section-label">Models</div>
    <div class="nav-item active" id="nav-ollama" onclick="switchTab('ollama')">
      <span class="nav-icon">🦙</span>Ollama
      <span class="nav-badge" id="badge-ollama">—</span>
    </div>
    <div class="nav-item" id="nav-inventory" onclick="switchTab('inventory')">
      <span class="nav-icon">📦</span>Inventory
      <span class="nav-badge" id="badge-inventory">—</span>
    </div>
    <div class="nav-item" id="nav-hfbrowse" onclick="switchTab('hfbrowse')">
      <span class="nav-icon">🔍</span>HF Browse
    </div>
    <div class="nav-item" id="nav-hf" onclick="switchTab('hf')">
      <span class="nav-icon">🤗</span>HF Download
    </div>
    <div class="nav-section-label">Routing</div>
    <div class="nav-item" id="nav-litellm" onclick="switchTab('litellm')">
      <span class="nav-icon">⚡</span>LiteLLM
      <span class="nav-badge" id="badge-litellm">—</span>
    </div>
    <div class="nav-section-label">Engines</div>
""" + "".join(f'    <div class="nav-item" id="nav-{k}" onclick="switchTab(\'{k}\')">\n      <span class="nav-icon">{e["icon"]}</span>{e["name"]}\n    </div>\n' for k, e in _ENGINES.items()) + r"""
    <div class="nav-section-label">System</div>
    <div class="nav-item" id="nav-settings" onclick="switchTab('settings')">
      <span class="nav-icon">&#9881;</span>Settings
    </div>
    <div class="nav-section-label">Diagnostics</div>
    <div class="nav-item" id="nav-debug" onclick="switchTab('debug')">
      <span class="nav-icon">&#128269;</span>Logs &amp; Debug
    </div>
  </nav>

  <main class="main">

    <!-- ─── OLLAMA ─── -->
    <div class="tab active" id="tab-ollama">
      <div class="page-hdr">
        <div class="page-title">Ollama Models</div>
        <div class="page-sub">Pull models from the Ollama library. With wildcard routing enabled, every pulled model is instantly available at <code id="ollama-litellm-port">LiteLLM</code>.</div>
      </div>

      <div class="input-row">
        <input class="input" id="pull-input"
          placeholder="Model name — e.g. llama3.2  qwen2.5:7b  phi4  gemma3:4b  deepseek-r1:7b"
          onkeydown="if(event.key==='Enter')pullModel()">
        <button class="btn btn-primary" id="pull-btn" onclick="pullModel()">⬇ Pull</button>
      </div>

      <div class="progress-wrap" id="pull-progress">
        <div class="prog-bar-outer"><div class="prog-bar spin" id="pull-bar"></div></div>
        <div class="prog-log" id="pull-log"></div>
      </div>

      <div class="sec-label">Installed <span id="badge-ollama-inline"></span></div>
      <div id="ollama-list"><div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div></div></div>
    </div>

    <!-- ─── HF DOWNLOAD ─── -->
    <div class="tab" id="tab-hf">
      <div class="page-hdr">
        <div class="page-title">HuggingFace Download</div>
        <div class="page-sub">Download any model from HuggingFace Hub directly to your device. Large models land in <code>~/.cache/huggingface/hub/</code> — ready for SGLang or vLLM.</div>
      </div>

      <div class="card">
        <div class="sec-label" style="margin-top:0;margin-bottom:8px">Repository ID</div>
        <div class="input-row">
          <input class="input" id="hf-repo"
            placeholder="e.g. mistralai/Mistral-7B-Instruct-v0.3">
        </div>
        <div class="sec-label" style="margin-bottom:8px">Local Directory <span style="color:var(--muted);font-size:10px">(optional — leave blank for HF cache default)</span></div>
        <div class="input-row" style="margin-bottom:0">
          <input class="input" id="hf-dir" placeholder="/home/user/models/my-model">
          <button class="btn btn-primary" id="hf-btn" onclick="hfDownload()">⬇ Download</button>
        </div>
      </div>

      <div class="progress-wrap" id="hf-progress">
        <div class="prog-bar-outer"><div class="prog-bar spin" id="hf-bar"></div></div>
        <div class="prog-log" id="hf-log"></div>
      </div>

    </div>

    <!-- ─── INVENTORY ─── -->
    <div class="tab" id="tab-inventory">
      <div class="page-hdr">
        <div class="page-title">Model Inventory</div>
        <div class="page-sub">All models across HuggingFace cache, custom directories, and Ollama.</div>
      </div>

      <div class="inv-toolbar">
        <input class="input inv-search" id="inv-search" placeholder="Search models..." oninput="filterInventory()">
        <select class="inv-filter" id="inv-filter-source" onchange="filterInventory()">
          <option value="">All Sources</option>
          <option value="hf_cache">HF Cache</option>
          <option value="custom_dir">Custom Dir</option>
          <option value="ollama">Ollama</option>
        </select>
        <select class="inv-filter" id="inv-filter-format" onchange="filterInventory()">
          <option value="">All Formats</option>
          <option value="safetensors">Safetensors</option>
          <option value="gguf">GGUF</option>
          <option value="pytorch">PyTorch</option>
          <option value="ollama">Ollama</option>
        </select>
        <select class="inv-filter" id="inv-filter-task" onchange="filterInventory()">
          <option value="">All Tasks</option>
          <option value="Text Gen">Text Gen</option>
          <option value="Vision LLM">Vision LLM</option>
          <option value="Embedding">Embedding</option>
          <option value="STT">STT</option>
          <option value="TTS">TTS</option>
          <option value="Image Gen">Image Gen</option>
          <option value="Audio">Audio</option>
        </select>
        <select class="inv-filter" id="inv-sort" onchange="sortAndRender()">
          <option value="name">Sort: Name</option>
          <option value="size">Sort: Size</option>
          <option value="params">Sort: Params</option>
        </select>
        <button class="btn btn-sm btn-ghost" onclick="loadUnifiedInventory()">&#8635; Refresh</button>
        <button class="btn btn-sm" onclick="enrichInventoryMeta()">Fetch HF Info</button>
      </div>

      <div class="inv-stats" id="inv-stats"></div>

      <details class="inv-dirs-section">
        <summary>Scan Directories</summary>
        <div id="inv-custom-dirs"></div>
        <div class="input-row" style="margin-top:8px">
          <input class="input" id="inv-add-dir" placeholder="/home/user/models  or  ~/models" style="font-size:12px"
            onkeydown="if(event.key==='Enter')addInventoryDir()">
          <button class="btn btn-sm" onclick="addInventoryDir()">+ Add</button>
        </div>
      </details>

      <div id="inv-root">
        <div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div><div style="font-size:12px;color:var(--muted)">Loading inventory...</div></div>
      </div>
    </div>

    <!-- ─── HF BROWSE ─── -->
    <div class="tab" id="tab-hfbrowse">
      <div class="page-hdr">
        <div class="page-title">Browse HuggingFace</div>
        <div class="page-sub">Search and discover models on HuggingFace Hub. Find quant variants, preview files, and download directly.</div>
      </div>

      <div class="hfb-search-bar">
        <input class="input" id="hfb-query" placeholder="Search models... e.g. llama 3.1, whisper, stable diffusion"
          onkeydown="if(event.key==='Enter')hfbSearch()" style="flex:1">
        <select class="inv-filter" id="hfb-pipeline">
          <option value="">All Types</option>
          <option value="text-generation">Text Generation</option>
          <option value="image-text-to-text">Vision LLM</option>
          <option value="feature-extraction">Embeddings</option>
          <option value="automatic-speech-recognition">Speech-to-Text</option>
          <option value="text-to-speech">Text-to-Speech</option>
          <option value="text-to-image">Image Generation</option>
          <option value="text-to-video">Video Generation</option>
        </select>
        <select class="inv-filter" id="hfb-sort">
          <option value="downloads">Most Downloads</option>
          <option value="likes">Most Likes</option>
          <option value="lastModified">Recently Updated</option>
          <option value="trending">Trending</option>
        </select>
        <button class="btn btn-primary" onclick="hfbSearch()">Search</button>
      </div>

      <div id="hfb-results">
        <div class="empty"><div class="empty-icon" style="font-size:32px">&#129303;</div>
        <div class="empty-text">Search HuggingFace to discover models</div></div>
      </div>
    </div>

    <!-- ─── LITELLM ─── -->
    <div class="tab" id="tab-litellm">
      <div class="page-hdr">
        <div class="page-title">LiteLLM Routing</div>
        <div class="page-sub">Unified gateway at <code id="litellm-port-display">LiteLLM</code>. All apps — Open WebUI, scripts, agents — connect here. This config controls which models they can see.</div>
      </div>

      <div id="sudo-banner-litellm" style="display:none;margin-bottom:12px;padding:12px 16px;border-radius:8px;font-size:12px;line-height:1.8"></div>

      <div class="card" id="wildcard-card">
        <div class="card-row">
          <div class="card-icon">🃏</div>
          <div class="card-info">
            <div class="card-name">Ollama Wildcard Routing</div>
            <div class="card-meta" id="wc-meta">ollama/* → Ollama</div>
          </div>
          <div class="card-actions">
            <button class="btn btn-primary" id="wc-btn" onclick="applyWildcard()">Apply Wildcard</button>
          </div>
        </div>
        <div class="card-desc">
          Adds a single <code>ollama/*</code> entry to your config. After this one change, any model you pull into Ollama is automatically available at <code id="wc-litellm-port">LiteLLM</code> — no YAML edits, no restarts.
        </div>
        <div id="wc-status"></div>
      </div>

      <div class="sec-label">Active Routes <span class="nav-badge" id="litellm-route-count">—</span></div>
      <div id="litellm-list"><div class="empty"><div class="spin-icon" style="margin:0 auto"></div></div></div>

      <div class="sec-label">Config File</div>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
          <span style="font-family:var(--mono);font-size:11px;color:var(--muted)">~/litellm/litellm_config.yaml</span>
          <div style="display:flex;gap:6px">
            <button class="btn btn-sm btn-ghost" onclick="loadLiteLLMConfig()">↻ Refresh</button>
            <button class="btn btn-sm" onclick="restartLiteLLM()">⟳ Restart</button>
          </div>
        </div>
        <div class="config-block" id="config-block">Loading…</div>
      </div>
    </div>

    <!-- ─── ENGINE TABS (generated) ─── -->
""" + "".join(f'''    <div class="tab" id="tab-{k}">
      <div class="page-hdr">
        <div class="page-title">{e["name"]} Engine</div>
        <div class="page-sub">{e["description"]}. Profiles auto-detected from
          <code style="font-family:var(--mono);color:var(--amber);font-size:12px">~/{e["script_dir_default"]}/</code> &mdash;
          add a <code style="font-family:var(--mono);color:var(--amber);font-size:12px">start_*.sh</code> script to create profiles.</div>
      </div>
      <div id="sudo-banner-{k}" style="display:none;margin-bottom:12px;padding:12px 16px;border-radius:8px;font-size:12px;line-height:1.8"></div>
      <div class="engine-card" id="{k}-engine-card">
        <div class="engine-status-row">
          <div class="engine-led" id="{k}-engine-led"></div>
          <div>
            <div class="engine-title" id="{k}-engine-title">Checking\u2026</div>
            <div class="engine-model" id="{k}-engine-model"></div>
          </div>
          <div class="engine-actions">
            {"<a class=&quot;btn btn-sm&quot; id=&quot;" + k + "-webui-btn&quot; target=&quot;_blank&quot; style=&quot;display:none&quot;>Open UI \u2197</a>" if e.get("webui") else ""}
            <button class="btn btn-danger" id="{k}-stop-btn" onclick="stopEngine(engines.{k})" disabled>\u25a0 Stop</button>
          </div>
        </div>
        <div class="engine-footer" id="{k}-engine-footer">loading\u2026</div>
      </div>
      <div style="margin-bottom:12px;padding:12px 16px;background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.22);border-radius:8px;font-size:12px;color:var(--muted);line-height:1.8">
        <div style="color:var(--amber);font-weight:700;font-size:13px;margin-bottom:4px">\U0001f4c1 Script directory: <span id="{k}-script-dir-banner">~/{e["script_dir_default"]}/</span></div>
        Place scripts named <code style="font-family:var(--mono);color:var(--amber);font-size:11px">start_*.sh</code> in this folder &mdash; each one becomes a profile card below.
        Name, description, and VRAM are read from optional header comments in the script:<br>
        <code style="font-family:var(--mono);font-size:11px;color:var(--dim)"># Name: My Model &nbsp;\u00b7&nbsp; # Description: ... &nbsp;\u00b7&nbsp; # VRAM: 119</code>
      </div>
      <div class="sec-label">Profiles</div>
      <div class="profile-list" id="{k}-profile-list">
        <div class="empty"><div class="spin-icon" style="margin:0 auto"></div></div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:14px">
        <button class="btn btn-primary" id="{k}-start-btn" onclick="startEngine(engines.{k})">\u25b6 Start Selected</button>
        <span style="font-size:12px;color:var(--muted)">Runs start script in background \u00b7 check status pill</span>
      </div>
      <div class="progress-wrap" id="{k}-progress" style="margin-top:14px">
        <div class="prog-bar-outer"><div class="prog-bar spin"></div></div>
        <div class="prog-log" id="{k}-log"></div>
      </div>
    </div>
''' for k, e in _ENGINES.items()) + r"""
    <!-- ─── SETTINGS ─── -->
    <div class="tab" id="tab-settings">
      <div class="page-hdr">
        <div class="page-title">Service Configuration</div>
        <div class="page-sub">Configure the address and port for each service. Changes take effect immediately and are saved to <code>config.json</code>.</div>
      </div>

      <div class="card" id="svc-ollama-card" style="margin-bottom:8px">
        <div class="card-row" style="align-items:center">
          <div class="card-icon">🦙</div>
          <div class="card-info" style="flex:1">
            <div class="card-name">Ollama</div>
            <div class="card-meta">Model pulling, listing, and deletion</div>
          </div>
          <div class="svc-status" id="svc-ollama-status"></div>
        </div>
        <div style="display:flex;gap:8px;margin-top:10px;align-items:center">
          <input class="input" id="svc-ollama-url" placeholder="http://127.0.0.1:11434" style="flex:1;font-size:12px">
          <button class="btn btn-sm" onclick="testService('ollama')">Test</button>
        </div>
      </div>

      <div class="card" id="svc-litellm-card" style="margin-bottom:8px">
        <div class="card-row" style="align-items:center">
          <div class="card-icon">⚡</div>
          <div class="card-info" style="flex:1">
            <div class="card-name">LiteLLM</div>
            <div class="card-meta">Unified API gateway and model routing</div>
          </div>
          <div class="svc-status" id="svc-litellm-status"></div>
        </div>
        <div style="display:flex;gap:8px;margin-top:10px;align-items:center">
          <input class="input" id="svc-litellm-url" placeholder="http://127.0.0.1:4000" style="flex:1;font-size:12px">
          <button class="btn btn-sm" onclick="testService('litellm')">Test</button>
        </div>
      </div>

""" + "".join(f'''      <div class="card" id="svc-{k}-card" style="margin-bottom:8px">
        <div class="card-row" style="align-items:center">
          <div class="card-icon">{e["icon"]}</div>
          <div class="card-info" style="flex:1">
            <div class="card-name">{e["name"]}</div>
            <div class="card-meta">{e["description"]}</div>
          </div>
          <div class="svc-status" id="svc-{k}-status"></div>
        </div>
        <div style="display:flex;gap:8px;margin-top:10px;align-items:center">
          <input class="input" id="svc-{k}-url" placeholder="{e["default_base"]}" style="flex:1;font-size:12px">
          <button class="btn btn-sm" onclick="testService('{k}')">Test</button>
        </div>
      </div>
''' for k, e in _ENGINES.items()) + r"""

      <div style="display:flex;align-items:center;gap:12px;margin-top:16px">
        <button class="btn btn-primary" onclick="saveConfig()">Save Configuration</button>
        <button class="btn btn-sm btn-ghost" onclick="testAllServices()">Test All</button>
        <span id="settings-msg" style="font-size:12px;color:var(--muted)"></span>
      </div>

      <div class="sec-label" style="margin-top:24px">Security</div>

      <div class="card" style="margin-bottom:8px">
        <div class="card-row" style="align-items:center">
          <div class="card-icon">🔒</div>
          <div class="card-info" style="flex:1">
            <div class="card-name">API Key</div>
            <div class="card-meta">When set, all actions (pull, delete, start, stop, config changes) require this key. Leave blank for open access.</div>
          </div>
          <div class="svc-status" id="auth-status"></div>
        </div>
        <div style="display:flex;gap:8px;margin-top:10px;align-items:center">
          <input class="input" id="svc-api-key" type="password" placeholder="Enter a key to protect this instance (optional)" style="flex:1;font-size:12px">
          <button class="btn btn-sm" onclick="saveApiKey()">Set Key</button>
          <button class="btn btn-sm btn-danger" onclick="clearApiKey()">Clear</button>
        </div>
      </div>
    </div>

    <!-- ─── LOGS & DEBUG ─── -->
    <div class="tab" id="tab-debug">
      <div class="page-hdr">
        <div class="page-title">Logs &amp; Debug</div>
        <div class="page-sub">System diagnostics, running configuration, and log viewer.</div>
      </div>

      <div class="sec-label">System Overview</div>
      <div id="debug-system" style="margin-bottom:18px">
        <div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div><div style="font-size:12px;color:var(--muted)">Loading system info...</div></div>
      </div>

      <div class="sec-label">Running Configuration</div>
      <div class="card" style="margin-bottom:8px;padding:12px 14px">
        <details><summary class="debug-section-hdr">App Configuration</summary>
        <div id="debug-cfg-app" class="config-block" style="margin-top:10px">Loading...</div>
        </details>
      </div>
      <div class="card" style="margin-bottom:8px;padding:12px 14px">
        <details><summary class="debug-section-hdr">LiteLLM Configuration</summary>
        <div id="debug-cfg-litellm" class="config-block" style="margin-top:10px">Loading...</div>
        </details>
      </div>
""" + "".join(f'''      <div class="card" style="margin-bottom:8px;padding:12px 14px">
        <details><summary class="debug-section-hdr">{e["name"]} Profiles</summary>
        <div id="debug-cfg-{k}" class="config-block" style="margin-top:10px">Loading...</div>
        </details>
      </div>
''' for k, e in _ENGINES.items()) + r"""

      <div class="sec-label">Application Logs</div>
      <div class="card" style="margin-bottom:18px">
        <div class="log-toolbar">
          <select class="input log-level-select" id="log-level-filter" onchange="loadAppLogs()">
            <option value="">All Levels</option>
            <option value="DEBUG">DEBUG</option>
            <option value="INFO" selected>INFO+</option>
            <option value="WARNING">WARNING+</option>
            <option value="ERROR">ERROR</option>
          </select>
          <input class="input" id="log-search" placeholder="Search logs..." style="flex:1" onkeydown="if(event.key==='Enter')loadAppLogs()">
          <label class="log-auto-label"><input type="checkbox" id="log-auto-refresh"> Auto</label>
          <button class="btn btn-sm" onclick="loadAppLogs()">Refresh</button>
          <button class="btn btn-sm btn-danger" onclick="clearAppLogs()">Clear</button>
        </div>
        <div class="log-pane" id="app-log-pane">
          <div class="empty"><div class="empty-text">No log entries yet.</div></div>
        </div>
        <div class="log-footer" id="app-log-footer"></div>
      </div>

      <div class="sec-label">Engine Logs</div>
      <div class="card" style="margin-bottom:18px">
        <div class="log-toolbar">
          <div class="log-tab-bar">
""" + "".join(f'            <button class="btn btn-sm log-tab-btn{" active" if i == 0 else ""}" id="eng-tab-{k}" onclick="switchEngineLog(\'{k}\')">{e["name"]}</button>\n' for i, (k, e) in enumerate(_ENGINES.items())) + r"""          </div>
          <input class="input" id="engine-log-search" placeholder="Search..." style="flex:1" onkeydown="if(event.key==='Enter')loadEngineLog()">
          <label class="log-auto-label"><input type="checkbox" id="engine-auto-refresh"> Auto</label>
          <button class="btn btn-sm" onclick="loadEngineLog()">Refresh</button>
        </div>
        <div class="log-pane" id="engine-log-pane">
          <div class="empty"><div class="empty-text">Select an engine and click Refresh.</div></div>
        </div>
        <div class="log-footer" id="engine-log-footer"></div>
      </div>

      <div class="sec-label">LiteLLM Service Logs</div>
      <div class="card" style="margin-bottom:18px">
        <div class="log-toolbar">
          <input class="input" id="litellm-log-search" placeholder="Search..." style="flex:1" onkeydown="if(event.key==='Enter')loadLiteLLMLogs()">
          <label class="log-auto-label"><input type="checkbox" id="litellm-auto-refresh"> Auto</label>
          <button class="btn btn-sm" onclick="loadLiteLLMLogs()">Refresh</button>
        </div>
        <div class="log-pane" id="litellm-log-pane">
          <div class="empty"><div class="empty-text">Click Refresh to load journalctl output.</div></div>
        </div>
        <div class="log-footer" id="litellm-log-footer"></div>
      </div>

      <div class="sec-label">Docker Containers</div>
      <div class="card">
        <div style="display:flex;justify-content:flex-end;margin-bottom:8px">
          <button class="btn btn-sm" onclick="loadDockerState()">Refresh</button>
        </div>
        <div id="docker-state-content">
          <div class="empty"><div class="empty-text">Click Refresh to load Docker state.</div></div>
        </div>
      </div>
    </div>

  </main>
</div>

<div id="toast-root"></div>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

let activeTab = 'ollama';
let selectedProfile = null;
let selectedVLLMProfile = null;
let statusTimer = null;
let litellmPort = '';
let ollamaBase = '';

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  pollStatus();
  loadOllamaModels();
  loadNodeInfo();
  loadScriptDirs();
  checkSudo();
  statusTimer = setInterval(pollStatus, 12000);
});

let _nodeServices = {};
function _engineBaseUrl(key) { return _nodeServices[key] || ''; }

async function loadScriptDirs() {
  try {
    const d = await apiFetch('/api/scriptdirs');
    for (const [key, path] of Object.entries(d)) {
      const el = document.getElementById(key + '-script-dir-banner');
      if (el) el.textContent = path + '/';
    }
  } catch(e) {}
}

async function checkSudo() {
  try {
    const r = await fetch('/api/sudo/check');
    const d = await r.json();
    const liteBanner = document.getElementById('sudo-banner-litellm');
    if (!d.systemctl) {
      liteBanner.style.display = 'block';
      liteBanner.style.background = 'rgba(239,68,68,0.08)';
      liteBanner.style.border = '1px solid rgba(239,68,68,0.25)';
      liteBanner.innerHTML = '<div style="color:var(--red);font-weight:700;margin-bottom:4px">\u26a0 Passwordless sudo not configured</div>' +
        'Restarting LiteLLM requires <code>sudo systemctl restart litellm</code>. To enable this without a password prompt:<br>' +
        '<code style="font-size:11px;display:block;margin-top:6px;padding:8px 10px;background:rgba(0,0,0,.15);border-radius:4px">echo \\"$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart litellm\\" | sudo tee /etc/sudoers.d/model-manager</code>';
    } else {
      liteBanner.style.display = 'block';
      liteBanner.style.background = 'rgba(34,197,94,0.08)';
      liteBanner.style.border = '1px solid rgba(34,197,94,0.25)';
      liteBanner.innerHTML = '<span style="color:var(--green)">\u2713</span> Passwordless sudo is configured \u2014 LiteLLM restart will work.';
    }
    const dockerOk = d.docker;
    Object.keys(engines).forEach(key => {
      const el = document.getElementById('sudo-banner-' + key);
      if (!el) return;
      if (!dockerOk) {
        el.style.display = 'block';
        el.style.background = 'rgba(239,68,68,0.08)';
        el.style.border = '1px solid rgba(239,68,68,0.25)';
        el.innerHTML = '<div style="color:var(--red);font-weight:700;margin-bottom:4px">\u26a0 Docker access issue</div>' +
          'Cannot list containers. Make sure Docker is installed and your user is in the <code>docker</code> group:<br>' +
          '<code style="font-size:11px;display:block;margin-top:6px;padding:8px 10px;background:rgba(0,0,0,.15);border-radius:4px">sudo usermod -aG docker $USER && newgrp docker</code>';
      } else {
        el.style.display = 'block';
        el.style.background = 'rgba(34,197,94,0.08)';
        el.style.border = '1px solid rgba(34,197,94,0.25)';
        el.innerHTML = '<span style="color:var(--green)">\u2713</span> Docker access confirmed \u2014 container management will work.';
      }
    });
  } catch(e) {}
}

async function loadNodeInfo() {
  try {
    const r = await fetch('/api/nodeinfo');
    const d = await r.json();
    document.getElementById('hdr-node').textContent =
      d.hostname + ' \u00b7 ' + d.ip + ' \u00b7 :' + d.port;
    litellmPort = d.litellm_port || '';
    ollamaBase = d.ollama_base || '';
    // Store service URLs for engine webui links
    _nodeServices = d.services || {};
    // Populate engine footers dynamically
    for (const [key, eng] of Object.entries(engines)) {
      const footer = document.getElementById(eng.ids.footer);
      if (footer && d.engine_ports && d.engine_ports[key + '_port']) {
        const parts = ['Port :' + d.engine_ports[key + '_port']];
        if (d.arch) parts.push(d.arch);
        if (d.memory_gb) parts.push(d.memory_gb + ' GB memory');
        footer.textContent = parts.join(' \u00b7 ');
      }
    }
    // Populate dynamic port displays
    const lp = ':' + litellmPort;
    const setTxt = (id, txt) => { const e = document.getElementById(id); if (e) e.textContent = txt; };
    setTxt('ollama-litellm-port', lp);
    setTxt('litellm-port-display', lp);
    setTxt('wc-litellm-port', lp);
    setTxt('wc-meta', 'ollama/* \u2192 ' + ollamaBase);
  } catch(e) {
    document.getElementById('hdr-node').textContent = 'could not detect';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────────────────

function switchTab(name) {
  // Clear debug auto-refresh timers when navigating away
  Object.keys(_debugTimers).forEach(k => { clearInterval(_debugTimers[k]); delete _debugTimers[k]; });
  ['log-auto-refresh','engine-auto-refresh','litellm-auto-refresh'].forEach(id => {
    const cb = document.getElementById(id); if (cb) cb.checked = false;
  });
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  activeTab = name;
  if (name === 'litellm') { loadLiteLLMModels(); loadLiteLLMConfig(); checkWildcard(); }
  else if (engines[name]) { loadEngineStatus(engines[name]); loadEngineProfiles(engines[name]); }
  else if (name === 'ollama') { loadOllamaModels(); }
  else if (name === 'inventory') { loadUnifiedInventory(); loadCustomDirs(); }
  else if (name === 'settings') { loadConfig(); }
  else if (name === 'debug') { loadDebugTab(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Status
// ─────────────────────────────────────────────────────────────────────────────

async function pollStatus() {
  try {
    const d = await apiFetch('/api/status');
    setPill('pill-ollama', d.ollama ? d.ollama.ok : false, 'Ollama');
    setPill('pill-litellm', d.litellm ? d.litellm.ok : false, 'LiteLLM');
    for (const [key, eng] of Object.entries(engines)) {
      if (d[key]) {
        setPill('pill-' + key, d[key].ok,
          d[key].model ? eng.name + ' \u00b7 ' + d[key].model.split('/').pop().slice(0,18) : eng.name);
      }
    }
  } catch(e) {}
}

function setPill(id, ok, label) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'pill ' + (ok ? 'ok' : 'err');
  const sp = el.querySelector('span');
  if (sp) sp.textContent = label;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ollama
// ─────────────────────────────────────────────────────────────────────────────

async function loadOllamaModels() {
  const el = document.getElementById('ollama-list');
  try {
    const d = await apiFetch('/api/ollama/models');
    const models = d.models || [];
    const n = models.length;
    document.getElementById('badge-ollama').textContent = n;
    document.getElementById('badge-ollama-inline').textContent = n + ' model' + (n !== 1 ? 's' : '');

    if (!n) {
      el.innerHTML = '<div class="empty"><div class="empty-icon">🦙</div><div class="empty-text">No models installed. Pull one above.</div></div>';
      return;
    }

    el.innerHTML = '<div class="model-grid">' + models.map(m => {
      const gb = m.size ? (m.size / 1e9).toFixed(1) + ' GB' : '?';
      const date = m.modified_at ? new Date(m.modified_at).toLocaleDateString() : '';
      const safeName = m.name.replace(/'/g, "\\'");
      return `<div class="model-card">
        <div class="model-card-info">
          <div class="model-card-name">${m.name}</div>
          <div class="model-card-meta">${gb}${date ? ' · ' + date : ''}</div>
        </div>
        <div class="model-card-right">
          <span class="tag tag-ollama">ollama</span>
          <button class="btn btn-sm btn-danger" onclick="deleteModel('${safeName}', this)">✕</button>
        </div>
      </div>`;
    }).join('') + '</div>';
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">⚠</div><div class="empty-text">Ollama unreachable · ' + e.message + '</div></div>';
  }
}

async function pullModel() {
  const input = document.getElementById('pull-input');
  const name = input.value.trim();
  if (!name) { input.focus(); return; }

  const btn  = document.getElementById('pull-btn');
  const prog = document.getElementById('pull-progress');
  const bar  = document.getElementById('pull-bar');
  const log  = document.getElementById('pull-log');

  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Pulling…';
  prog.classList.add('show');
  log.textContent = 'Connecting…';
  bar.className = 'prog-bar spin';

  try {
    const resp = await fetch('/api/ollama/pull', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({name}),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.done) break;
          if (ev.error) { toast('Error: ' + ev.error, 'err'); break; }
          if (ev.total && ev.completed) {
            const pct = Math.round(ev.completed / ev.total * 100);
            bar.className = 'prog-bar';
            bar.style.width = pct + '%';
            log.textContent = (ev.status || '') + ' — ' + pct + '% (' +
              (ev.completed/1e6).toFixed(0) + ' / ' + (ev.total/1e6).toFixed(0) + ' MB)';
          } else if (ev.status) {
            log.textContent = ev.status;
          }
        } catch(e) {}
      }
    }

    bar.className = 'prog-bar';
    bar.style.width = '100%';
    toast('✓ ' + name + ' ready', 'ok');
    input.value = '';
    await loadOllamaModels();
    setTimeout(() => { prog.classList.remove('show'); bar.style.width = '0'; }, 2000);

  } catch(e) {
    toast('Pull failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '⬇ Pull';
  }
}

async function deleteModel(name, btn) {
  if (!confirm('Delete ' + name + '?\nThis cannot be undone.')) return;
  btn.disabled = true;
  btn.innerHTML = '…';
  try {
    const r = await fetch('/api/ollama/models/' + encodeURIComponent(name), {method:'DELETE', headers: authHeaders()});
    if (!r.ok) throw new Error(await r.text());
    toast('✓ Deleted ' + name, 'ok');
    loadOllamaModels();
  } catch(e) {
    toast('Delete failed: ' + e.message, 'err');
    btn.disabled = false;
    btn.innerHTML = '✕';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LiteLLM
// ─────────────────────────────────────────────────────────────────────────────

async function loadLiteLLMModels() {
  const el = document.getElementById('litellm-list');
  try {
    const d = await apiFetch('/api/litellm/models');
    const models = d.data || [];
    document.getElementById('badge-litellm').textContent = models.length;
    document.getElementById('litellm-route-count').textContent = models.length;

    if (!models.length) {
      el.innerHTML = '<div class="empty"><div class="empty-text">No routes active</div></div>';
      return;
    }

    el.innerHTML = '<div class="model-grid">' + models.map(m => {
      const isOllama = m.id.toLowerCase().includes('ollama') || m.id.toLowerCase().includes(':');
      return `<div class="model-card">
        <div class="model-card-info">
          <div class="model-card-name">${m.id}</div>
        </div>
        <span class="tag ${isOllama ? 'tag-ollama' : 'tag-sglang'}">${isOllama ? 'ollama' : 'sglang'}</span>
      </div>`;
    }).join('') + '</div>';
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-text">LiteLLM unreachable</div></div>';
  }
}

async function loadLiteLLMConfig() {
  const el = document.getElementById('config-block');
  try {
    const d = await apiFetch('/api/litellm/config');
    el.textContent = d._raw || JSON.stringify(d, null, 2);
  } catch(e) { el.textContent = 'Could not load config'; }
}

async function checkWildcard() {
  try {
    const d = await apiFetch('/api/litellm/config');
    const models = d.model_list || [];
    const has = models.some(m => m.model_name === 'ollama/*');
    const status = document.getElementById('wc-status');
    const btn = document.getElementById('wc-btn');
    if (has) {
      status.innerHTML = '<div class="wc-active">✓ Wildcard active — all Ollama models auto-exposed at :' + litellmPort + '</div>';
      btn.textContent = '✓ Applied';
      btn.disabled = true;
    } else {
      status.innerHTML = '<div class="wc-inactive">Not yet applied — each Ollama model requires a manual config entry</div>';
      btn.textContent = 'Apply Wildcard';
      btn.disabled = false;
    }
  } catch(e) {}
}

async function applyWildcard() {
  const btn = document.getElementById('wc-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Applying…';
  try {
    const d = await apiFetch('/api/litellm/apply-wildcard', 'POST');
    toast('✓ Wildcard applied — LiteLLM restarted', 'ok');
    await checkWildcard();
    await loadLiteLLMConfig();
    setTimeout(loadLiteLLMModels, 3500);
  } catch(e) {
    toast('Failed: ' + e.message, 'err');
    btn.disabled = false;
    btn.textContent = 'Apply Wildcard';
  }
}

async function restartLiteLLM() {
  toast('Restarting LiteLLM…', null);
  try {
    await apiFetch('/api/litellm/restart', 'POST');
    toast('✓ LiteLLM restarted', 'ok');
    setTimeout(loadLiteLLMModels, 3500);
  } catch(e) { toast('Failed: ' + e.message, 'err'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// SGLang
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Engine helpers (shared by SGLang + vLLM)
// ─────────────────────────────────────────────────────────────────────────────

const engines = {
""" + ",\n".join(f'''  {k}: {{
    name: '{e["name"]}', api: '/api/{k}', selectedProfile: null,{" webui: true," if e.get("webui") else ""}
    key: '{k}',
    ids: {{ led: '{k}-engine-led', title: '{k}-engine-title', model: '{k}-engine-model',
           card: '{k}-engine-card', stop: '{k}-stop-btn', start: '{k}-start-btn',
           profiles: '{k}-profile-list', prog: '{k}-progress', log: '{k}-log',
           footer: '{k}-engine-footer'{(", webui: '" + k + "-webui-btn'") if e.get("webui") else ""} }}
  }}''' for k, e in _ENGINES.items()) + r"""
};

async function loadEngineStatus(eng) {
  try {
    const d = await apiFetch(eng.api + '/status');
    const led   = document.getElementById(eng.ids.led);
    const title = document.getElementById(eng.ids.title);
    const model = document.getElementById(eng.ids.model);
    const card  = document.getElementById(eng.ids.card);
    const stop  = document.getElementById(eng.ids.stop);
    if (!led || !title || !card) return;
    if (d.running) {
      led.className = 'engine-led on';
      title.textContent = eng.name + ' \u2014 Running';
      model.textContent = d.model || (eng.webui ? '' : 'Model loading\u2026');
      card.classList.add('online');
      stop.disabled = false;
      if (eng.webui && eng.ids.webui) {
        const wb = document.getElementById(eng.ids.webui);
        if (wb) { wb.style.display = ''; wb.href = _engineBaseUrl(eng.key); }
      }
    } else {
      led.className = 'engine-led';
      title.textContent = eng.name + ' \u2014 Stopped';
      model.textContent = '';
      card.classList.remove('online');
      stop.disabled = true;
      if (eng.webui && eng.ids.webui) {
        const wb = document.getElementById(eng.ids.webui);
        if (wb) wb.style.display = 'none';
      }
    }
  } catch(e) {}
}

async function loadEngineProfiles(eng) {
  const el = document.getElementById(eng.ids.profiles);
  try {
    const profiles = await apiFetch(eng.api + '/profiles');
    if (!profiles.length) {
      el.innerHTML = '<div class="empty"><div class="empty-text">No profiles defined</div></div>';
      return;
    }
    if (!eng.selectedProfile) eng.selectedProfile = profiles[0].id;
    el.innerHTML = profiles.map(p => `
      <div class="profile-item ${eng.selectedProfile === p.id ? 'selected' : ''}"
           onclick="selectEngineProfile('${eng.key}', '${p.id}', this)">
        <div class="p-radio"></div>
        <div class="p-info">
          <div class="p-name">${p.name}</div>
          <div class="p-desc">${p.description}</div>
        </div>
        <div class="p-vram">${p.vram_gb != null ? p.vram_gb + ' GB' : '\u2014'}</div>
      </div>
    `).join('');
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-text">Could not load profiles</div></div>';
  }
}

function selectEngineProfile(key, id, el) {
  engines[key].selectedProfile = id;
  const container = document.getElementById(engines[key].ids.profiles);
  container.querySelectorAll('.profile-item').forEach(p => p.classList.remove('selected'));
  el.classList.add('selected');
}

async function stopEngine(eng) {
  if (!confirm('Stop ' + eng.name + '? This will interrupt any active inference requests.')) return;
  const btn = document.getElementById(eng.ids.stop);
  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div>';
  try {
    const d = await apiFetch(eng.api + '/stop', 'POST');
    toast(d.ok ? '\u2713 ' + eng.name + ' stopped' : 'Stop may have failed: ' + d.output, d.ok ? 'ok' : 'err');
    setTimeout(() => { loadEngineStatus(eng); btn.innerHTML = '\u25a0 Stop'; }, 1500);
  } catch(e) {
    toast('Error: ' + e.message, 'err');
    btn.innerHTML = '\u25a0 Stop';
  }
}

async function startEngine(eng) {
  if (!eng.selectedProfile) { toast('Select a profile first', 'err'); return; }
  const btn  = document.getElementById(eng.ids.start);
  const prog = document.getElementById(eng.ids.prog);
  const log  = document.getElementById(eng.ids.log);

  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Launching\u2026';
  prog.classList.add('show');
  log.textContent = 'Sending start command\u2026';

  try {
    const d = await apiFetch(eng.api + '/start', 'POST', {profile: eng.selectedProfile});
    toast('\u2713 ' + eng.name + ' starting', 'ok');
    log.textContent = d.message + '\n\nPolling status every 20 seconds\u2026';

    let pollCount = 0;
    const poll = setInterval(async () => {
      pollCount++;
      await loadEngineStatus(eng);
      const led = document.getElementById(eng.ids.led);
      if (led.classList.contains('on')) {
        const modelEl = document.getElementById(eng.ids.model);
        if (modelEl.textContent && modelEl.textContent !== 'Model loading\u2026') {
          clearInterval(poll);
          toast('\u2713 ' + eng.name + ' is ready!', 'ok');
          prog.classList.remove('show');
        } else {
          log.textContent = d.message + '\n\nContainer running \u2014 model still loading\u2026';
        }
      } else if (pollCount >= 30) {
        clearInterval(poll);
        log.textContent += '\n\n\u26a0 Timed out after 10 minutes \u2014 check logs';
        toast(eng.name + ' did not start within 10 minutes', 'err');
      }
    }, 20000);
  } catch(e) {
    toast('Start failed: ' + e.message, 'err');
    prog.classList.remove('show');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '\u25b6 Start Selected';
  }
}

// Legacy wrappers (kept for backward compat with docs page references)
function selectProfile(id, el)     { selectEngineProfile('sglang', id, el); }
function selectVLLMProfile(id, el) { selectEngineProfile('vllm', id, el); }

// ─────────────────────────────────────────────────────────────────────────────
// HF Download
// ─────────────────────────────────────────────────────────────────────────────

async function hfDownload() {
  const repo = document.getElementById('hf-repo').value.trim();
  const dir  = document.getElementById('hf-dir').value.trim();
  if (!repo) { document.getElementById('hf-repo').focus(); return; }

  const btn  = document.getElementById('hf-btn');
  const prog = document.getElementById('hf-progress');
  const bar  = document.getElementById('hf-bar');
  const log  = document.getElementById('hf-log');

  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Downloading…';
  prog.classList.add('show');
  bar.className = 'prog-bar spin';
  bar.style.width = '';
  const lines = ['Starting download: ' + repo];
  log.textContent = lines[0];

  try {
    const resp = await fetch('/api/hf/download', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({repo_id: repo, local_dir: dir || undefined}),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.status === 'complete') {
            bar.className = 'prog-bar';
            bar.style.width = '100%';
            const parts = ['✓ Complete → ' + ev.path];
            if (ev.avg_speed) parts.push('Avg: ' + ev.avg_speed);
            if (ev.elapsed) parts.push('Time: ' + ev.elapsed);
            if (ev.errors > 0) parts.push('⚠ ' + ev.errors + ' error(s)');
            lines.push(parts.join('  ·  '));
            toast('✓ Downloaded: ' + repo, 'ok');
          } else if (ev.status === 'error') {
            bar.className = 'prog-bar';
            bar.style.width = '0';
            toast('Error: ' + ev.error, 'err');
            lines.push('✗ ' + ev.error);
          } else if (ev.progress) {
            const p = ev.progress;
            bar.className = 'prog-bar';
            bar.style.width = p.pct + '%';
            lines[lines.length - 1] = '[' + p.idx + '/' + p.total_files + '] ✓ ' + p.file
              + '  ·  ' + p.pct + '%  ·  '
              + p.done_mb.toFixed(0) + ' / ' + p.total_mb.toFixed(0) + ' MB  ·  ' + p.speed;
          } else if (ev.file_start) {
            const f = ev.file_start;
            lines.push('[' + f.idx + '/' + f.total + '] ' + f.name + ' (' + f.size_str + ')');
          } else if (ev.file_error) {
            lines[lines.length - 1] = '⚠ Failed: ' + ev.file_error.name + ' — ' + ev.file_error.error;
          } else if (ev.status) {
            lines.push(ev.status);
          }
          log.textContent = lines.join('\n');
          log.scrollTop = log.scrollHeight;
        } catch(e) {}
      }
    }
  } catch(e) {
    toast('Download failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '⬇ Download';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified Inventory
// ─────────────────────────────────────────────────────────────────────────────

let _inventoryData = [];

function dtypeClass(dtype) {
  const m = {
    'FP32':'inv-fp32','FP16':'inv-fp16','BF16':'inv-bf16',
    'FP8':'inv-fp8','FP4':'inv-fp4','INT4':'inv-int4','INT8':'inv-int8',
    'GGUF':'inv-int4',
  };
  return m[dtype] || 'inv-unknown';
}
function sourceClass(src) {
  return {'hf_cache':'inv-src-hf','custom_dir':'inv-src-custom','ollama':'inv-src-ollama'}[src] || '';
}
function sourceLabel(src) {
  return {'hf_cache':'HF Cache','custom_dir':'Custom','ollama':'Ollama'}[src] || src;
}
function formatClass(fmt) {
  return {'safetensors':'inv-fmt-safe','gguf':'inv-fmt-gguf','pytorch':'inv-fmt-pt','ollama':'inv-fmt-ollama'}[fmt] || '';
}
function formatLabel(fmt) {
  return {'safetensors':'Safetensors','gguf':'GGUF','pytorch':'PyTorch','ollama':'Ollama','unknown':'—'}[fmt] || fmt;
}

async function loadUnifiedInventory() {
  const root = document.getElementById('inv-root');
  if (!root) return;
  root.innerHTML = '<div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div><div style="font-size:12px;color:var(--muted)">Scanning model directories...</div></div>';
  try {
    const d = await apiFetch('/api/inventory?include_ollama=true');
    const dirs = d.directories || [];
    _inventoryData = [];
    for (const dir of dirs) {
      for (const m of (dir.models || [])) {
        _inventoryData.push(m);
      }
    }
    const badge = document.getElementById('badge-inventory');
    if (badge) badge.textContent = _inventoryData.length || '—';
    sortAndRender();
  } catch(e) {
    root.innerHTML = '<div class="empty"><div class="empty-icon">&#9888;</div><div class="empty-text">Could not load inventory: ' + e.message + '</div></div>';
  }
}

function getFilteredInventory() {
  const q = (document.getElementById('inv-search')?.value || '').toLowerCase();
  const src = document.getElementById('inv-filter-source')?.value || '';
  const fmt = document.getElementById('inv-filter-format')?.value || '';
  const task = document.getElementById('inv-filter-task')?.value || '';
  return _inventoryData.filter(m => {
    if (q && !m.name.toLowerCase().includes(q) && !(m.owner||'').toLowerCase().includes(q) && !(m.full_name||'').toLowerCase().includes(q)) return false;
    if (src && m.source !== src) return false;
    if (fmt && m.format !== fmt) return false;
    if (task && m.task_label !== task) return false;
    return true;
  });
}

function sortAndRender() {
  const key = document.getElementById('inv-sort')?.value || 'name';
  _inventoryData.sort((a, b) => {
    if (key === 'size') return (b.size_gb || 0) - (a.size_gb || 0);
    if (key === 'params') return (b.params_b || 0) - (a.params_b || 0);
    return (a.name || '').localeCompare(b.name || '');
  });
  filterInventory();
}

function filterInventory() {
  const models = getFilteredInventory();
  renderInventoryTable(models);
}

function renderInventoryTable(models) {
  const root = document.getElementById('inv-root');
  if (!root) return;

  // Stats bar
  const stats = document.getElementById('inv-stats');
  if (stats) {
    const totalSize = models.reduce((s, m) => s + (m.size_gb || 0), 0);
    const sources = new Set(models.map(m => m.source));
    stats.innerHTML = '<span>' + models.length + '</span> models &middot; <span>' + totalSize.toFixed(1) + '</span> GB &middot; <span>' + sources.size + '</span> source' + (sources.size !== 1 ? 's' : '');
  }

  if (!models.length) {
    root.innerHTML = '<div class="empty" style="border-radius:8px"><div class="empty-icon" style="font-size:24px">&#128237;</div><div class="empty-text">No models match your filters</div></div>';
    return;
  }

  let html = '<div class="inv-table-wrap" style="border-radius:8px;border:1px solid var(--border)"><table class="inv-table"><thead><tr>';
  html += '<th>Model</th><th>Task</th><th>Format</th><th>Dtype</th><th>Params</th><th>Size</th><th>Source</th><th>Script</th><th style="width:36px"></th>';
  html += '</tr></thead><tbody>';

  for (const m of models) {
    const params = m.params_b != null ? m.params_b + 'B' : '\u2014';
    const size = m.size_gb ? m.size_gb + ' GB' : '\u2014';
    const dc = dtypeClass(m.dtype);
    let scriptBadge = '<span class="inv-no">\u2014</span>';
    if (m.has_script && m.script_engine) {
      const cls = m.script_engine === 'SGLang' ? 'inv-engine-sg' : 'inv-engine-vl';
      scriptBadge = '<span class="inv-engine ' + cls + '">' + m.script_engine + '</span>';
    }
    const delBtn = m.source === 'ollama'
      ? ''
      : '<button class="btn-icon-del" title="Delete model" onclick="deleteInventoryModel(\'' + m.dir_path.replace(/'/g,"\\'") + "','" + (m.full_name || m.name).replace(/'/g,"\\'") + '\')">&#10005;</button>';

    html += '<tr>';
    html += '<td><div class="inv-model-name">' + m.name + '</div>' + (m.owner ? '<div class="inv-owner">' + m.owner + '</div>' : '') + '</td>';
    html += '<td><span class="inv-task-badge">' + (m.task_label || '\u2014') + '</span></td>';
    html += '<td><span class="inv-format-badge ' + formatClass(m.format) + '">' + formatLabel(m.format) + '</span></td>';
    html += '<td><span class="inv-badge ' + dc + '">' + m.dtype + '</span></td>';
    html += '<td style="font-family:var(--mono);font-size:11px">' + params + '</td>';
    html += '<td style="font-family:var(--mono);font-size:11px;white-space:nowrap">' + size + '</td>';
    html += '<td><span class="inv-source-badge ' + sourceClass(m.source) + '">' + sourceLabel(m.source) + '</span></td>';
    html += '<td>' + scriptBadge + '</td>';
    html += '<td>' + delBtn + '</td>';
    html += '</tr>';
  }

  html += '</tbody></table></div>';
  root.innerHTML = html;
}

async function enrichInventoryMeta() {
  const toEnrich = _inventoryData.filter(m => m.owner && m.source !== 'ollama');
  if (!toEnrich.length) { toast('No HF models to enrich', 'err'); return; }
  toast('Fetching HF metadata for ' + toEnrich.length + ' models...', 'ok');
  try {
    const payload = toEnrich.map(m => ({owner: m.owner, name: m.name}));
    const d = await apiFetch('/api/hf/meta/enrich', 'POST', {models: payload});
    const results = d.results || {};
    let count = 0;
    for (const m of _inventoryData) {
      const key = m.full_name || (m.owner + '/' + m.name);
      if (results[key]) {
        m.pipeline_tag = results[key].pipeline_tag;
        m.task_label = results[key].task_label || m.task_label;
        m.hf_downloads = results[key].downloads;
        m.hf_likes = results[key].likes;
        count++;
      }
    }
    filterInventory();
    toast('Enriched ' + count + ' models with HF metadata', 'ok');
  } catch(e) {
    toast('Enrich failed: ' + e.message, 'err');
  }
}

async function loadCustomDirs() {
  try {
    const d = await apiFetch('/api/hf/inventory/dirs');
    const customDirsEl = document.getElementById('inv-custom-dirs');
    if (!customDirsEl) return;
    const customDirs = (d.dirs || []).filter(x => !x.default);
    if (!customDirs.length) {
      customDirsEl.innerHTML = '<div style="font-size:12px;color:var(--muted);padding:2px 0">No additional directories added</div>';
      return;
    }
    customDirsEl.innerHTML = customDirs.map(dir => {
      return '<div class="inv-custom-dir-row">'
        + '<span class="inv-custom-dir-path">' + dir.path.replace(/^\/home\/[^/]+/, '~') + '</span>'
        + '<button class="inv-remove-btn" onclick="removeInventoryDir(\'' + dir.path.replace(/'/g,"\\'") + '\')">&#10005; Remove</button>'
        + '</div>';
    }).join('');
  } catch(e) {}
}

async function addInventoryDir() {
  const input = document.getElementById('inv-add-dir');
  const path = input.value.trim();
  if (!path) { input.focus(); return; }
  try {
    await apiFetch('/api/hf/inventory/dirs', 'POST', {path});
    input.value = '';
    toast('Directory added', 'ok');
    await loadCustomDirs();
    await loadUnifiedInventory();
  } catch(e) {
    toast('Failed: ' + e.message, 'err');
  }
}

async function removeInventoryDir(path) {
  try {
    await fetch('/api/hf/inventory/dirs?' + new URLSearchParams({path}), {method:'DELETE', headers: authHeaders()});
    toast('Directory removed', 'ok');
    await loadCustomDirs();
    await loadUnifiedInventory();
  } catch(e) {
    toast('Failed: ' + e.message, 'err');
  }
}

async function deleteInventoryModel(dirPath, modelName) {
  if (!confirm('Delete "' + modelName + '" from disk?\n\nThis will permanently remove all files in:\n' + dirPath)) return;
  try {
    await apiFetch('/api/hf/inventory/delete', 'POST', {path: dirPath});
    toast('Deleted: ' + modelName, 'ok');
    await loadUnifiedInventory();
  } catch(e) {
    toast('Delete failed: ' + e.message, 'err');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HF Browse
// ─────────────────────────────────────────────────────────────────────────────

function fmtNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function fmtSize(bytes) {
  if (!bytes) return '';
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
  return bytes + ' B';
}

async function hfbSearch() {
  const q = document.getElementById('hfb-query')?.value.trim();
  if (!q) return;
  const pipeline = document.getElementById('hfb-pipeline')?.value || '';
  const sort = document.getElementById('hfb-sort')?.value || 'downloads';
  const root = document.getElementById('hfb-results');
  root.innerHTML = '<div class="hfb-loading"><div class="spin-icon" style="margin:0 auto 8px"></div>Searching HuggingFace...</div>';
  try {
    let url = '/api/hf/search?q=' + encodeURIComponent(q) + '&sort=' + sort + '&limit=20';
    if (pipeline) url += '&pipeline_tag=' + encodeURIComponent(pipeline);
    const d = await apiFetch(url);
    const models = d.models || [];
    if (!models.length) {
      root.innerHTML = '<div class="empty"><div class="empty-text">No results found for "' + q + '"</div></div>';
      return;
    }
    root.innerHTML = models.map(renderHFBCard).join('');
  } catch(e) {
    root.innerHTML = '<div class="empty"><div class="empty-icon">&#9888;</div><div class="empty-text">Search failed: ' + e.message + '</div></div>';
  }
}

function renderHFBCard(m) {
  const taskBadge = m.task_label && m.task_label !== 'Unknown'
    ? '<span class="inv-task-badge">' + m.task_label + '</span>' : '';
  const fmtTags = [];
  if (m.has_safetensors) fmtTags.push('<span class="hfb-tag fmt">safetensors</span>');
  if (m.has_gguf) fmtTags.push('<span class="hfb-tag fmt">gguf</span>');
  const tags = (m.tags || []).filter(t => t !== 'safetensors' && t !== 'gguf').slice(0, 8)
    .map(t => '<span class="hfb-tag">' + t + '</span>').join('');
  const safeId = m.id.replace(/'/g, "\\'");

  return '<div class="hfb-card" id="hfb-card-' + m.id.replace(/\//g, '--') + '">'
    + '<div class="hfb-card-hdr"><div class="hfb-card-name">' + m.id + '</div>' + taskBadge + '</div>'
    + '<div class="hfb-card-meta">'
    + '<span class="dl">&#11015; ' + fmtNum(m.downloads) + '</span>'
    + '<span class="lk">&#9829; ' + fmtNum(m.likes) + '</span>'
    + (m.library_name ? '<span>' + m.library_name + '</span>' : '')
    + '</div>'
    + '<div class="hfb-tags">' + fmtTags.join('') + tags + '</div>'
    + '<div class="hfb-card-actions">'
    + '<button class="btn btn-sm btn-primary" onclick="hfbDownload(\'' + safeId + '\')">Download</button>'
    + '<button class="hfb-expand-toggle" onclick="hfbToggleExpand(\'' + safeId + '\')">&#9660; Files &amp; Variants</button>'
    + '</div>'
    + '<div class="hfb-expand" id="hfb-exp-' + m.id.replace(/\//g, '--') + '" style="display:none"></div>'
    + '</div>';
}

async function hfbToggleExpand(modelId) {
  const elId = 'hfb-exp-' + modelId.replace(/\//g, '--');
  const el = document.getElementById(elId);
  if (!el) return;
  if (el.style.display !== 'none') { el.style.display = 'none'; return; }
  el.style.display = 'block';
  if (el.dataset.loaded) return;
  el.innerHTML = '<div class="hfb-loading">Loading...</div>';

  const parts = modelId.split('/');
  if (parts.length < 2) { el.innerHTML = '<div class="inv-no">Invalid model ID</div>'; return; }
  const [owner, name] = parts;

  try {
    const [filesRes, varRes] = await Promise.all([
      apiFetch('/api/hf/model/' + owner + '/' + name + '/files'),
      apiFetch('/api/hf/search/variants?model_id=' + encodeURIComponent(modelId)),
    ]);

    let html = '<div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.1em">Files (' + (filesRes.total || 0) + ')</div>';
    html += '<div class="hfb-file-list">';
    for (const f of (filesRes.files || []).slice(0, 50)) {
      html += '<div class="hfb-file-row"><span>' + f.name + '</span><span class="size">' + fmtSize(f.size) + '</span></div>';
    }
    if ((filesRes.total || 0) > 50) html += '<div style="padding:4px 0;color:var(--muted);font-size:10px">...and ' + (filesRes.total - 50) + ' more files</div>';
    html += '</div>';

    const variants = varRes.variants || [];
    if (variants.length) {
      html += '<div class="hfb-variants"><div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.1em">Quantized Variants</div>';
      for (const v of variants) {
        const vSafe = v.id.replace(/'/g, "\\'");
        html += '<div class="hfb-variant-row">'
          + '<span class="fmt">' + v.format + '</span>'
          + '<span style="flex:1;color:var(--text)">' + v.id + '</span>'
          + '<span style="color:var(--muted);font-size:10px">&#11015; ' + fmtNum(v.downloads) + '</span>'
          + '<button class="btn btn-sm" style="padding:2px 8px;font-size:10px" onclick="hfbDownload(\'' + vSafe + '\')">Download</button>'
          + '</div>';
      }
      html += '</div>';
    }

    el.innerHTML = html;
    el.dataset.loaded = '1';
  } catch(e) {
    el.innerHTML = '<div style="color:var(--red);font-size:12px">Failed to load: ' + e.message + '</div>';
  }
}

function hfbDownload(repoId) {
  document.getElementById('hf-repo').value = repoId;
  switchTab('hf');
  toast('Repo pre-filled: ' + repoId + '. Click Download to start.', 'ok');
}

// ─────────────────────────────────────────────────────────────────────────────
// Debug / Logs
// ─────────────────────────────────────────────────────────────────────────────

let _debugEngineTab = 'sglang';
const _debugTimers = {};

function _escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function _fmtUptime(sec) {
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60);
  if (d > 0) return d + 'd ' + h + 'h ' + m + 'm';
  if (h > 0) return h + 'h ' + m + 'm';
  return m + 'm ' + Math.floor(sec % 60) + 's';
}

function loadDebugTab() {
  loadSystemOverview();
  loadDebugConfig();
  loadAppLogs();
}

async function loadSystemOverview() {
  const el = document.getElementById('debug-system');
  if (!el) return;
  try {
    const d = await apiFetch('/api/debug/system');
    let html = '<div class="debug-grid">';
    html += _statCard('Hostname', d.hostname);
    html += _statCard('IP Address', d.ip);
    html += _statCard('Architecture', d.arch);
    html += _statCard('Memory', d.memory_gb + ' GB');
    html += _statCard('Python', d.python_version);
    html += _statCard('Uptime', _fmtUptime(d.uptime_seconds));
    html += _statCard('App Port', ':' + d.app_port);
    html += _statCard('API Key', d.api_key_set ? 'Active' : 'Not set', d.api_key_set ? 'warn' : '');
    html += '</div>';

    // Disk
    const hfDisk = d.disk?.hf_cache;
    if (hfDisk && !hfDisk.error) {
      html += '<div style="font-family:var(--mono);font-size:11px;color:var(--muted);margin-bottom:10px">'
        + 'Disk: <span style="color:var(--text)">' + hfDisk.path.replace(/^\/home\/[^/]+/,'~') + '</span>'
        + ' &mdash; <span style="color:var(--amber)">' + (hfDisk.total_gb - hfDisk.free_gb).toFixed(1) + '</span> / ' + hfDisk.total_gb + ' GB'
        + ' (' + hfDisk.used_pct + '%)</div>';
    }

    // Services
    html += '<div class="debug-grid" style="margin-bottom:10px">';
    for (const [name, info] of Object.entries(d.services || {})) {
      const cls = info.healthy ? 'ok' : 'err';
      const ms = info.response_ms != null ? info.response_ms + 'ms' : 'timeout';
      html += '<div class="debug-stat">'
        + '<div class="debug-stat-label">' + name.toUpperCase() + '</div>'
        + '<div class="debug-stat-value ' + cls + '">' + (info.healthy ? '\u25CF ' + ms : '\u25CB Offline') + '</div>'
        + '<div style="font-size:10px;color:var(--muted);margin-top:2px">' + _escHtml(info.url) + '</div>'
        + '</div>';
    }
    html += '</div>';

    // Permissions
    const p = d.permissions || {};
    html += '<div style="font-family:var(--mono);font-size:11px;color:var(--muted)">'
      + 'Sudo: <span class="' + (p.systemctl ? 'debug-stat-value ok' : 'debug-stat-value err') + '" style="font-size:11px">'
      + (p.systemctl ? '\u2713' : '\u2717') + ' systemctl</span>'
      + ' &nbsp; <span class="' + (p.docker ? 'debug-stat-value ok' : 'debug-stat-value err') + '" style="font-size:11px">'
      + (p.docker ? '\u2713' : '\u2717') + ' docker</span></div>';

    el.innerHTML = html;
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-text">Failed to load system info: ' + _escHtml(e.message) + '</div></div>';
  }
}

function _statCard(label, value, cls) {
  return '<div class="debug-stat"><div class="debug-stat-label">' + label + '</div>'
    + '<div class="debug-stat-value' + (cls ? ' ' + cls : '') + '">' + _escHtml(value) + '</div></div>';
}

async function loadDebugConfig() {
  try {
    const d = await apiFetch('/api/debug/config');
    // App config
    const appEl = document.getElementById('debug-cfg-app');
    if (appEl) {
      let t = 'Port:           ' + d.app.port + '\n'
        + 'API Key:        ' + (d.app.api_key_set ? 'Active' : 'Not set') + '\n'
        + 'Config File:    ' + d.app.config_file + '\n'
        + 'Started:        ' + d.app.start_utc + '\n\n'
        + '--- Service URLs ---\n';
      for (const [k, v] of Object.entries(d.services)) {
        t += (k + ':').padEnd(16) + v + '\n';
      }
      t += '\n--- Paths ---\n';
      for (const [k, v] of Object.entries(d.paths)) {
        t += (k + ':').padEnd(16) + v + '\n';
      }
      appEl.textContent = t;
    }
    // LiteLLM config
    const litEl = document.getElementById('debug-cfg-litellm');
    if (litEl) litEl.textContent = d.litellm?.raw || 'No LiteLLM config found';
    // Engine profiles (dynamic)
    const ep = d.engine_profiles || {};
    for (const [key, profiles] of Object.entries(ep)) {
      const el = document.getElementById('debug-cfg-' + key);
      if (el) el.textContent = profiles?.length ? JSON.stringify(profiles, null, 2) : 'No profiles found';
    }
  } catch(e) {}
}

async function loadAppLogs() {
  if (activeTab !== 'debug') return;
  const pane = document.getElementById('app-log-pane');
  const footer = document.getElementById('app-log-footer');
  if (!pane) return;
  const level = document.getElementById('log-level-filter')?.value || '';
  const search = document.getElementById('log-search')?.value || '';
  try {
    let url = '/api/logs/app?limit=200';
    if (level) url += '&level=' + encodeURIComponent(level);
    if (search) url += '&search=' + encodeURIComponent(search);
    const d = await apiFetch(url);
    const entries = d.entries || [];
    if (!entries.length) {
      pane.innerHTML = '<span style="color:var(--muted)">No log entries match your filters.</span>';
      if (footer) footer.textContent = '0 / ' + d.total + ' entries (buffer: ' + d.buffer_size + ')';
      return;
    }
    // Smart scroll: only auto-scroll if already at bottom
    const atBottom = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 30;
    let html = '';
    for (const e of entries) {
      const ts = e.ts.substring(11, 19);
      html += '<div class="log-entry">'
        + '<span class="log-ts">' + ts + '</span> '
        + '<span class="log-level-' + e.level + '">' + e.level.padEnd(7) + '</span> '
        + '<span class="log-src">' + _escHtml(e.logger) + '</span> '
        + _escHtml(e.msg)
        + '</div>';
    }
    pane.innerHTML = html;
    if (atBottom) pane.scrollTop = pane.scrollHeight;
    if (footer) footer.textContent = entries.length + ' / ' + d.total + ' entries (buffer: ' + d.buffer_size + ')';
  } catch(e) {
    pane.innerHTML = '<span style="color:var(--red)">Failed to load logs: ' + _escHtml(e.message) + '</span>';
  }
}

async function clearAppLogs() {
  try {
    await apiFetch('/api/logs/app', 'DELETE');
    toast('Log buffer cleared', 'ok');
    loadAppLogs();
  } catch(e) {
    toast('Failed: ' + e.message, 'err');
  }
}

function _setupAutoRefresh(cbId, fn, ms) {
  const cb = document.getElementById(cbId);
  if (!cb) return;
  if (cb.checked) {
    if (!_debugTimers[cbId]) _debugTimers[cbId] = setInterval(fn, ms);
  } else {
    clearInterval(_debugTimers[cbId]);
    delete _debugTimers[cbId];
  }
}

async function loadEngineLog() {
  if (activeTab !== 'debug') return;
  const pane = document.getElementById('engine-log-pane');
  const footer = document.getElementById('engine-log-footer');
  if (!pane) return;
  const search = document.getElementById('engine-log-search')?.value || '';
  try {
    let url = '/api/logs/engine/' + _debugEngineTab + '?lines=150';
    if (search) url += '&search=' + encodeURIComponent(search);
    const d = await apiFetch(url);
    if (!d.file) {
      pane.innerHTML = '<span style="color:var(--muted)">No log files found for ' + _debugEngineTab + '.</span>';
      if (footer) footer.textContent = '';
      return;
    }
    const atBottom = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 30;
    pane.innerHTML = d.lines.map(l => '<div class="log-entry">' + _escHtml(l) + '</div>').join('');
    if (atBottom) pane.scrollTop = pane.scrollHeight;
    if (footer) {
      const shortFile = d.file.replace(/^\/tmp\//, '/tmp/');
      footer.textContent = shortFile + ' \u00B7 ' + d.total_lines + ' total lines'
        + (d.available_files.length > 1 ? ' \u00B7 ' + d.available_files.length + ' log files' : '');
    }
  } catch(e) {
    pane.innerHTML = '<span style="color:var(--red)">Failed: ' + _escHtml(e.message) + '</span>';
  }
}

function switchEngineLog(engine) {
  _debugEngineTab = engine;
  for (const key of Object.keys(engines)) {
    const tab = document.getElementById('eng-tab-' + key);
    if (tab) tab.classList.toggle('active', engine === key);
  }
  loadEngineLog();
}

async function loadLiteLLMLogs() {
  if (activeTab !== 'debug') return;
  const pane = document.getElementById('litellm-log-pane');
  const footer = document.getElementById('litellm-log-footer');
  if (!pane) return;
  const search = document.getElementById('litellm-log-search')?.value || '';
  try {
    let url = '/api/logs/litellm?lines=100';
    if (search) url += '&search=' + encodeURIComponent(search);
    const d = await apiFetch(url);
    if (!d.available) {
      pane.innerHTML = '<span style="color:var(--amber)">' + _escHtml(d.error || 'journalctl not available') + '</span>';
      if (footer) footer.textContent = '';
      return;
    }
    if (!d.lines.length) {
      pane.innerHTML = '<span style="color:var(--muted)">No log entries found.</span>';
      if (footer) footer.textContent = '';
      return;
    }
    const atBottom = pane.scrollTop + pane.clientHeight >= pane.scrollHeight - 30;
    pane.innerHTML = d.lines.map(l => '<div class="log-entry">' + _escHtml(l) + '</div>').join('');
    if (atBottom) pane.scrollTop = pane.scrollHeight;
    if (footer) footer.textContent = d.lines.length + ' lines';
  } catch(e) {
    pane.innerHTML = '<span style="color:var(--red)">Failed: ' + _escHtml(e.message) + '</span>';
  }
}

async function loadDockerState() {
  const el = document.getElementById('docker-state-content');
  if (!el) return;
  try {
    const d = await apiFetch('/api/debug/docker');
    if (!d.available) {
      el.innerHTML = '<div class="empty"><div class="empty-text">Docker not available: ' + _escHtml(d.error || '') + '</div></div>';
      return;
    }
    if (!d.containers.length) {
      el.innerHTML = '<div class="empty"><div class="empty-text">No running containers.</div></div>';
      return;
    }
    let html = '<table class="docker-table"><thead><tr>'
      + '<th>ID</th><th>Name</th><th>Image</th><th>Status</th><th>Ports</th>'
      + '</tr></thead><tbody>';
    for (const c of d.containers) {
      html += '<tr><td>' + _escHtml(c.id) + '</td><td>' + _escHtml(c.name) + '</td>'
        + '<td>' + _escHtml(c.image) + '</td><td>' + _escHtml(c.status) + '</td>'
        + '<td style="font-size:10px">' + _escHtml(c.ports) + '</td></tr>';
    }
    html += '</tbody></table>';
    el.innerHTML = html;
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-text">Failed: ' + _escHtml(e.message) + '</div></div>';
  }
}

// Wire up auto-refresh checkboxes
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('log-auto-refresh')?.addEventListener('change', function() {
    _setupAutoRefresh('log-auto-refresh', loadAppLogs, 3000);
  });
  document.getElementById('engine-auto-refresh')?.addEventListener('change', function() {
    _setupAutoRefresh('engine-auto-refresh', loadEngineLog, 3000);
  });
  document.getElementById('litellm-auto-refresh')?.addEventListener('change', function() {
    _setupAutoRefresh('litellm-auto-refresh', loadLiteLLMLogs, 5000);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Utils
// ─────────────────────────────────────────────────────────────────────────────

function authHeaders() {
  const h = {'Content-Type': 'application/json'};
  const k = localStorage.getItem('dgx_api_key');
  if (k) h['Authorization'] = 'Bearer ' + k;
  return h;
}

async function apiFetch(url, method = 'GET', body = null) {
  const opts = {method, headers: authHeaders()};
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(url, opts);
  if (r.status === 401) {
    showAuthModal();
    throw new Error('Authentication required');
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { const d = await r.json(); msg = d.detail || JSON.stringify(d); } catch(e) {}
    throw new Error(msg);
  }
  const ct = r.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    const txt = await r.text();
    try { return JSON.parse(txt); } catch(e) { throw new Error('Server returned non-JSON response'); }
  }
  return r.json();
}

function showAuthModal() {
  if (document.getElementById('auth-modal')) return;
  const overlay = document.createElement('div');
  overlay.id = 'auth-modal';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:9999;display:flex;align-items:center;justify-content:center';
  overlay.innerHTML = `
    <div style="background:var(--s1);border:1px solid var(--border);border-radius:10px;padding:24px;width:380px;max-width:90vw">
      <div style="font-size:15px;font-weight:700;margin-bottom:4px">API Key Required</div>
      <div style="font-size:12px;color:var(--muted);margin-bottom:16px">This instance has an API key configured. Enter it to continue.</div>
      <input class="input" id="auth-key-input" type="password" placeholder="Enter API key" style="width:100%;margin-bottom:12px"
        onkeydown="if(event.key==='Enter')submitAuthKey()">
      <div style="display:flex;gap:8px;justify-content:flex-end">
        <button class="btn btn-sm" onclick="document.getElementById('auth-modal').remove()">Cancel</button>
        <button class="btn btn-primary btn-sm" onclick="submitAuthKey()">Unlock</button>
      </div>
      <div id="auth-error" style="font-size:11px;color:var(--red);margin-top:8px"></div>
    </div>`;
  document.body.appendChild(overlay);
  document.getElementById('auth-key-input').focus();
}

async function submitAuthKey() {
  const input = document.getElementById('auth-key-input');
  const key = input.value.trim();
  if (!key) return;
  try {
    const r = await fetch('/api/auth/check', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
    });
    const d = await r.json();
    if (d.ok) {
      localStorage.setItem('dgx_api_key', key);
      document.getElementById('auth-modal').remove();
      toast('Authenticated');
    } else {
      document.getElementById('auth-error').textContent = 'Invalid key';
    }
  } catch(e) {
    document.getElementById('auth-error').textContent = 'Connection error';
  }
}

function toast(msg, type) {
  const root = document.getElementById('toast-root');
  const el = document.createElement('div');
  el.className = 'toast ' + (type || '');
  el.textContent = msg;
  root.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .3s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 320);
  }, 3500);
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings
// ─────────────────────────────────────────────────────────────────────────────

async function loadConfig() {
  try {
    const d = await apiFetch('/api/config');
    document.getElementById('svc-ollama-url').value = d.services.ollama_base || '';
    document.getElementById('svc-litellm-url').value = d.services.litellm_base || '';
    // Populate engine URL inputs dynamically
    for (const [key, eng] of Object.entries(engines)) {
      const el = document.getElementById('svc-' + key + '-url');
      if (el) {
        // Find config key by looking for matching key in services
        const cfgKey = Object.keys(d.services).find(k => k === key + '_base') || key + '_base';
        el.value = d.services[cfgKey] || '';
      }
    }
    const authSt = document.getElementById('auth-status');
    if (d.app && d.app.api_key_set) {
      authSt.className = 'svc-status ok';
      authSt.textContent = 'Key active';
    } else {
      authSt.className = 'svc-status';
      authSt.textContent = 'Open (no key)';
    }
  } catch(e) {}
}

async function testService(type) {
  const input = document.getElementById('svc-' + type + '-url');
  const status = document.getElementById('svc-' + type + '-status');
  const url = input.value.trim();
  if (!url) { status.className = 'svc-status err'; status.textContent = 'No URL'; return; }
  status.className = 'svc-status testing';
  status.textContent = 'Testing\u2026';
  try {
    const r = await fetch('/api/test-service', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({url, type})
    });
    if (r.status === 401) { showAuthModal(); status.className = 'svc-status err'; status.textContent = '\u2717 Auth required'; return; }
    const d = await r.json();
    if (d.ok) {
      status.className = 'svc-status ok';
      status.textContent = '\u2713 ' + d.latency_ms + 'ms';
    } else {
      status.className = 'svc-status err';
      status.textContent = '\u2717 ' + (d.error || 'Failed');
    }
  } catch(e) {
    status.className = 'svc-status err';
    status.textContent = '\u2717 Error';
  }
}

async function testAllServices() {
  const types = ['ollama', 'litellm', ...Object.keys(engines)];
  await Promise.all(types.map(s => testService(s)));
}

async function saveConfig() {
  const services = {
    ollama_base:  document.getElementById('svc-ollama-url').value.trim(),
    litellm_base: document.getElementById('svc-litellm-url').value.trim(),
  };
  // Collect engine URLs dynamically
  for (const key of Object.keys(engines)) {
    const el = document.getElementById('svc-' + key + '-url');
    if (el) services[key + '_base'] = el.value.trim();
  }
  const msg = document.getElementById('settings-msg');
  try {
    const r = await fetch('/api/config', {
      method: 'PUT',
      headers: authHeaders(),
      body: JSON.stringify({services})
    });
    if (r.status === 401) { showAuthModal(); return; }
    const d = await r.json();
    if (d.ok) {
      toast('Configuration saved');
      msg.style.color = 'var(--green)';
      msg.textContent = 'Saved \u2014 changes are live';
      setTimeout(() => { msg.textContent = ''; }, 3000);
      // Refresh status and nodeinfo with new URLs
      pollStatus();
      loadNodeInfo();
    } else {
      msg.style.color = 'var(--red)';
      msg.textContent = 'Save failed';
    }
  } catch(e) {
    msg.style.color = 'var(--red)';
    msg.textContent = 'Save failed: ' + e.message;
  }
}

async function saveApiKey() {
  const input = document.getElementById('svc-api-key');
  const key = input.value.trim();
  if (!key) { toast('Enter a key first', 'err'); return; }
  try {
    await apiFetch('/api/config', 'PUT', {api_key: key});
    localStorage.setItem('dgx_api_key', key);
    input.value = '';
    toast('API key set');
    loadConfig();
  } catch(e) {
    toast('Failed to set key: ' + e.message, 'err');
  }
}

async function clearApiKey() {
  try {
    await apiFetch('/api/config', 'PUT', {api_key: ''});
    localStorage.removeItem('dgx_api_key');
    document.getElementById('svc-api-key').value = '';
    toast('API key cleared \u2014 open access');
    loadConfig();
  } catch(e) {
    toast('Failed to clear key: ' + e.message, 'err');
  }
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(HTML)


@app.get("/favicon.png")
async def favicon():
    path = Path(os.path.expanduser("~/model-manager/favicon.png"))
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.get("/help", response_class=HTMLResponse)
async def help_page():
    docs_path = Path(os.path.expanduser("~/model-manager/docs.html"))
    if not docs_path.exists():
        raise HTTPException(404, "docs.html not found in ~/model-manager/")
    return HTMLResponse(docs_path.read_text())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")