#!/usr/bin/env python3
"""
ZGX Model Manager
Unified web UI for managing models across Ollama, SGLang, and LiteLLM.
Port: 8090  |  Run via systemd: model-manager.service
"""

import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
import shutil
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_BASE    = "http://127.0.0.1:11434"
LITELLM_BASE   = "http://127.0.0.1:4000"
SGLANG_BASE    = "http://127.0.0.1:30000"
VLLM_BASE      = "http://127.0.0.1:8000"
LITELLM_CONFIG   = Path(os.path.expanduser("~/litellm/litellm_config.yaml"))
HOME             = Path.home()
SGLANG_SCRIPT_DIR = HOME / "SGLang"
VLLM_SCRIPT_DIR   = HOME / "vLLM"
HF_CACHE_DIR      = HOME / ".cache" / "huggingface" / "hub"
CUSTOM_DIRS_FILE  = Path(os.path.expanduser("~/model-manager/custom_dirs.json"))

# Load app config
_CONFIG_FILE = Path(os.path.expanduser("~/model-manager/config.json"))
_app_config: dict = {}
if _CONFIG_FILE.exists():
    try:
        _app_config = json.loads(_CONFIG_FILE.read_text())
    except Exception:
        pass
APP_PORT = _app_config.get("app", {}).get("port", 8090)


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

class SGLangStartRequest(BaseModel):
    profile: str

class HFDownloadRequest(BaseModel):
    repo_id: str
    local_dir: Optional[str] = None

class VLLMStartRequest(BaseModel):
    profile: str

# ─── Helpers ──────────────────────────────────────────────────────────────────

async def service_ok(base: str, path: str = "/health") -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(base + path)
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


def scan_sglang_profiles() -> list:
    """Scan ~/SGLang/start_*.sh and return profile list."""
    if not SGLANG_SCRIPT_DIR.exists():
        return []
    return [_parse_script_meta(s) for s in sorted(SGLANG_SCRIPT_DIR.glob("start_*.sh"))]


def scan_vllm_profiles() -> list:
    """Scan ~/vLLM/start_*.sh and return profile list."""
    if not VLLM_SCRIPT_DIR.exists():
        return []
    return [_parse_script_meta(s) for s in sorted(VLLM_SCRIPT_DIR.glob("start_*.sh"))]

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

import re as _re

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


def _parse_hf_model_dir(model_dir: Path) -> dict:
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

    # Pre-compute name-based inference — used as fallback throughout
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

    # ── Dtype ────────────────────────────────────────────────────────────────
    raw_dtype = config.get("torch_dtype", "")
    _dtype_map = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16",
                  "float8":  "FP8",  "float4":  "FP4"}
    dtype = _dtype_map.get(raw_dtype, raw_dtype.upper() if raw_dtype else None)

    quant = config.get("quantization_config", {}) or {}
    qt = str(quant.get("quant_type", quant.get("quant_method", ""))).lower()
    bits = quant.get("bits", 0) or quant.get("num_bits", 0)
    if "fp4" in qt or "nvfp4" in qt:
        dtype = "FP4"
    elif "fp8" in qt:
        dtype = "FP8"
    elif "int4" in qt or bits == 4:
        dtype = "INT4"
    elif "int8" in qt or bits == 8:
        dtype = "INT8"
    elif quant.get("load_in_4bit"):
        dtype = "INT4"
    elif quant.get("load_in_8bit"):
        dtype = "INT8"

    # Fallback: name-based dtype
    if not dtype or dtype == "Unknown":
        dtype = name_hints["dtype"] or "Unknown"
    # Name-based quantized dtype overrides config's unquantized torch_dtype
    # (common when torch_dtype reports original precision but model is quantized)
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
        # Fallback: name-based MoE signal
        or bool(name_hints["is_moe"])
    )

    # ── Parameter count ───────────────────────────────────────────────────────
    params_b: Optional[float] = None
    # For quantized dtypes, index file total_size is unreliable (includes scales/overhead)
    # Prefer name-based params when available
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
                        bytes_per = {"FP32": 4, "FP16": 2, "BF16": 2, "FP8": 1,
                                     "FP4": 0.5, "INT4": 0.5, "INT8": 1}.get(dtype, 2)
                        params_b = round(total_bytes / bytes_per / 1e9, 1)
                        break
                except Exception:
                    pass
    # Fallback: name-based params
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

    # ── Reasoning ────────────────────────────────────────────────────────────
    is_reasoning = (
        # Config signals (rare but some models set this)
        config.get("is_thinking", False)
        # Fallback: name-based reasoning signal
        or bool(name_hints["is_reasoning"])
    )

    # ── Modalities ───────────────────────────────────────────────────────────
    modalities: list[str] = ["Text"]
    has_vision = (
        config.get("vision_config") is not None
        or "vision" in arch_str
        or "llava" in arch_str
        or "Image" in name_hints["extra_modalities"]
    )
    has_audio = (
        config.get("audio_config") is not None
        or "audio" in arch_str
        or "whisper" in arch_str
        or "Audio" in name_hints["extra_modalities"]
    )
    has_embed = "Embedding" in name_hints["extra_modalities"]
    if has_vision:
        modalities.append("Image")
    if has_audio:
        modalities.append("Audio")
    if has_embed:
        modalities.append("Embedding")

    # ── Script cross-reference ────────────────────────────────────────────────
    name_lower = model_name.lower()
    has_script = False
    script_engine: Optional[str] = None
    search_term = name_lower.replace("-", "_")
    for profiles, engine_label in (
        (scan_sglang_profiles(), "SGLang"),
        (scan_vllm_profiles(), "vLLM"),
    ):
        for p in profiles:
            try:
                content = Path(p["script"]).read_text().lower()
                if name_lower in content or search_term in content:
                    has_script = True
                    script_engine = engine_label
                    break
            except Exception:
                pass
        if has_script:
            break

    return {
        "name":          model_name,
        "owner":         owner,
        "full_name":     full_name,
        "dir_path":      str(model_dir),
        "dtype":         dtype,
        "params_b":      params_b,
        "model_arch":    "MoE" if is_moe else "Dense",
        "size_gb":       size_gb,
        "is_reasoning":  is_reasoning,
        "has_script":    has_script,
        "script_engine": script_engine,
        "modalities":    modalities,
    }


def _parse_flat_model_dir(model_dir: Path) -> dict:
    """Parse a flat model directory (not HF cache format) that contains config.json."""
    stem = model_dir.name
    # Handle owner--name format (from old-style downloads)
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

    # ── Dtype ────────────────────────────────────────────────────────────────
    raw_dtype = config.get("torch_dtype", "")
    _dtype_map = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
    dtype = _dtype_map.get(raw_dtype, raw_dtype.upper() if raw_dtype else None)
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
        or "moe" in arch_str
        or config.get("model_type", "").lower() in _KNOWN_MOE_MODEL_TYPES
        or bool(name_hints["is_moe"])
    )

    # ── Params, size ─────────────────────────────────────────────────────────
    params_b = name_hints["params_b"]
    size_gb = _dir_size_gb(model_dir)

    # ── Reasoning ────────────────────────────────────────────────────────────
    is_reasoning = config.get("is_thinking", False) or bool(name_hints["is_reasoning"])

    # ── Modalities ───────────────────────────────────────────────────────────
    modalities: list[str] = ["Text"]
    if (config.get("vision_config") or "vision" in arch_str
            or "Image" in name_hints["extra_modalities"]):
        modalities.append("Image")
    if (config.get("audio_config") or "audio" in arch_str
            or "Audio" in name_hints["extra_modalities"]):
        modalities.append("Audio")
    if "Embedding" in name_hints["extra_modalities"]:
        modalities.append("Embedding")

    # ── Script cross-reference ────────────────────────────────────────────────
    name_lower = model_name.lower()
    has_script = False
    script_engine = None
    for profiles, engine_label in (
        (scan_sglang_profiles(), "SGLang"),
        (scan_vllm_profiles(), "vLLM"),
    ):
        for p in profiles:
            try:
                content = Path(p["script"]).read_text().lower()
                if name_lower in content or name_lower.replace("-", "_") in content:
                    has_script = True
                    script_engine = engine_label
                    break
            except Exception:
                pass
        if has_script:
            break

    return {
        "name":          model_name,
        "owner":         owner,
        "full_name":     full_name,
        "dir_path":      str(model_dir),
        "dtype":         dtype,
        "params_b":      params_b,
        "model_arch":    "MoE" if is_moe else "Dense",
        "size_gb":       size_gb,
        "is_reasoning":  is_reasoning,
        "has_script":    has_script,
        "script_engine": script_engine,
        "modalities":    modalities,
    }


def _scan_directory(directory: Path) -> dict:
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
                models.append(_parse_hf_model_dir(d))
            except Exception:
                pass
    # Also scan flat model dirs (subdirs with config.json) even alongside HF cache dirs
    for d in sorted(directory.iterdir()):
        if d.is_dir() and not d.name.startswith("models--") and (d / "config.json").exists():
            try:
                models.append(_parse_flat_model_dir(d))
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

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ZGX Model Manager")

# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    sglang_ok, ollama_ok, litellm_ok, vllm_ok = await asyncio.gather(
        service_ok(SGLANG_BASE, "/health"),
        service_ok(OLLAMA_BASE, "/api/tags"),
        service_ok(LITELLM_BASE, "/health"),
        service_ok(VLLM_BASE, "/health"),
    )
    sglang_model = None
    if sglang_ok:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(SGLANG_BASE + "/v1/models")
                d = r.json().get("data", [])
                if d:
                    sglang_model = d[0]["id"]
        except Exception:
            pass
    vllm_model = None
    if vllm_ok:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(VLLM_BASE + "/v1/models")
                d = r.json().get("data", [])
                if d:
                    vllm_model = d[0]["id"]
        except Exception:
            pass
    return {
        "sglang":  {"ok": sglang_ok,  "model": sglang_model},
        "ollama":  {"ok": ollama_ok},
        "litellm": {"ok": litellm_ok},
        "vllm":    {"ok": vllm_ok, "model": vllm_model},
    }

@app.get("/api/nodeinfo")
async def get_nodeinfo():
    hostname = socket.gethostname()
    ip = _get_local_ip()
    arch = platform.machine()
    mem_gb = _get_total_memory_gb()
    sglang_port = SGLANG_BASE.rsplit(":", 1)[-1]
    vllm_port = VLLM_BASE.rsplit(":", 1)[-1]
    return {
        "hostname": hostname,
        "ip": ip,
        "port": APP_PORT,
        "arch": arch,
        "memory_gb": mem_gb,
        "sglang_port": sglang_port,
        "vllm_port": vllm_port,
    }

# ── Ollama ────────────────────────────────────────────────────────────────────

@app.get("/api/scriptdirs")
async def get_scriptdirs():
    return {
        "sglang": str(SGLANG_SCRIPT_DIR),
        "vllm":   str(VLLM_SCRIPT_DIR),
    }


@app.get("/api/ollama/models")
async def list_ollama_models():
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(OLLAMA_BASE + "/api/tags")
            return r.json()
    except Exception as e:
        raise HTTPException(502, f"Ollama unreachable: {e}")


@app.post("/api/ollama/pull")
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


@app.delete("/api/ollama/models/{name:path}")
async def delete_ollama_model(name: str):
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.request("DELETE", OLLAMA_BASE + "/api/delete", json={"name": name})
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
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                LITELLM_BASE + "/v1/models",
                headers={"Authorization": "Bearer sk-sglang"},
            )
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


@app.post("/api/litellm/apply-wildcard")
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
            "api_base": "http://127.0.0.1:11434",
        },
    })
    cfg["model_list"] = model_list
    save_litellm_config(cfg)

    result = subprocess.run(
        ["sudo", "systemctl", "restart", "litellm"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise HTTPException(500, f"Config saved but restart failed: {result.stderr}")
    return {"ok": True, "message": "Wildcard applied — LiteLLM restarted"}


@app.post("/api/litellm/restart")
async def restart_litellm():
    result = subprocess.run(
        ["sudo", "systemctl", "restart", "litellm"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise HTTPException(500, result.stderr)
    return {"ok": True}

# ── SGLang ────────────────────────────────────────────────────────────────────

@app.get("/api/sglang/profiles")
async def get_sglang_profiles():
    return scan_sglang_profiles()


@app.get("/api/sglang/status")
async def sglang_status():
    # Primary: HTTP health check — reliable regardless of container name
    running = await service_ok(SGLANG_BASE, "/health")
    model = None
    if running:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(SGLANG_BASE + "/v1/models")
                d = r.json().get("data", [])
                if d:
                    model = d[0]["id"]
        except Exception:
            pass
    # Secondary: docker ps for container_info display only
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=sglang", "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True, text=True, timeout=5,
    )
    return {"running": running, "model": model, "container_info": result.stdout.strip()}


def _find_container_by_port(port: int) -> Optional[str]:
    """Return the container ID listening on the given host port, or None."""
    result = subprocess.run(
        ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.ID}}"],
        capture_output=True, text=True, timeout=5,
    )
    cid = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else None
    return cid or None


def _docker_stop(container_id: str) -> tuple[bool, str]:
    """Stop a container by ID, falling back to sudo if needed."""
    r = subprocess.run(["docker", "stop", container_id],
                       capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        return True, (r.stdout + r.stderr).strip()
    r2 = subprocess.run(["sudo", "docker", "stop", container_id],
                        capture_output=True, text=True, timeout=60)
    return r2.returncode == 0, (r2.stdout + r2.stderr).strip()


@app.post("/api/sglang/stop")
async def stop_sglang():
    port = int(SGLANG_BASE.rsplit(":", 1)[-1])
    cid = _find_container_by_port(port)
    if not cid:
        raise HTTPException(404, "No container found listening on SGLang port — already stopped?")
    ok, output = _docker_stop(cid)
    return {"ok": ok, "output": output}


@app.post("/api/sglang/start")
async def start_sglang(req: SGLangStartRequest):
    profiles = scan_sglang_profiles()
    profile = next((p for p in profiles if p["id"] == req.profile), None)
    if not profile:
        raise HTTPException(404, f"Profile '{req.profile}' not found")
    script = os.path.expanduser(profile.get("script", ""))
    if not Path(script).exists():
        raise HTTPException(400, f"Script not found: {script}")

    log_path = f"/tmp/sglang_{req.profile}.log"
    with open(log_path, "w") as logf:
        subprocess.Popen(
            ["bash", script],
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"ok": True, "message": f"Launched {profile['name']} — logs at {log_path}"}

# ── vLLM ──────────────────────────────────────────────────────────────────────

@app.get("/api/vllm/profiles")
async def get_vllm_profiles():
    return scan_vllm_profiles()


@app.get("/api/vllm/status")
async def vllm_status():
    # Primary: HTTP health check — reliable regardless of container name
    running = await service_ok(VLLM_BASE, "/health")
    model = None
    if running:
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(VLLM_BASE + "/v1/models")
                d = r.json().get("data", [])
                if d:
                    model = d[0]["id"]
        except Exception:
            pass
    # Secondary: docker ps for container_info display only
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=vllm", "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True, text=True, timeout=5,
    )
    return {"running": running, "model": model, "container_info": result.stdout.strip()}


@app.post("/api/vllm/stop")
async def stop_vllm():
    port = int(VLLM_BASE.rsplit(":", 1)[-1])
    cid = _find_container_by_port(port)
    if not cid:
        raise HTTPException(404, "No container found listening on vLLM port — already stopped?")
    ok, output = _docker_stop(cid)
    return {"ok": ok, "output": output}


@app.post("/api/vllm/start")
async def start_vllm(req: VLLMStartRequest):
    profiles = scan_vllm_profiles()
    profile = next((p for p in profiles if p["id"] == req.profile), None)
    if not profile:
        raise HTTPException(404, f"Profile '{req.profile}' not found")
    script = os.path.expanduser(profile.get("script", ""))
    if not Path(script).exists():
        raise HTTPException(400, f"Script not found: {script}")

    log_path = f"/tmp/vllm_{req.profile}.log"
    with open(log_path, "w") as logf:
        subprocess.Popen(
            ["bash", script],
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {"ok": True, "message": f"Launched {profile['name']} — logs at {log_path}"}

# ── HuggingFace Download ───────────────────────────────────────────────────────

@app.post("/api/hf/download")
async def hf_download(req: HFDownloadRequest):
    safe_repo = req.repo_id.replace("'", "").replace('"', "")
    use_custom_dir = bool(req.local_dir)
    local_dir = req.local_dir or ""

    # Track custom dir so inventory can scan it later
    if use_custom_dir:
        custom_dirs = _load_custom_dirs()
        expanded = os.path.expanduser(req.local_dir)
        parent = str(Path(expanded).parent)
        if parent not in custom_dirs and parent != str(HF_CACHE_DIR):
            custom_dirs.append(parent)
            _save_custom_dirs(custom_dirs)

    script = f"""
import sys, json, os, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from huggingface_hub import list_repo_tree, hf_hub_download
from pathlib import Path

repo = "{safe_repo}"
use_custom_dir = {use_custom_dir}
local_dir = "{local_dir}" if use_custom_dir else None
J = lambda **kw: print(json.dumps(kw), flush=True)
J(status="starting", repo=repo)

try:
    entries = [e for e in list_repo_tree(repo, recursive=True)
               if hasattr(e, 'size') and not e.path.startswith('.')]
    total_files = len(entries)
    total_bytes = sum(e.size or 0 for e in entries)
    J(status=f"Found {{total_files}} files ({{total_bytes/1024**3:.1f}} GB)")
    done_bytes = 0
    dl_start = time.time()
    errors = []
    result_path = None
    for i, entry in enumerate(entries, 1):
        fname = entry.path
        fsize = entry.size or 0
        sz_str = f"{{fsize/1024**2:.0f}} MB" if fsize > 1024**2 else f"{{fsize/1024:.0f}} KB" if fsize > 1024 else f"{{fsize}} B"
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
        avg_speed = done_bytes / total_elapsed
        pct = done_bytes / total_bytes * 100 if total_bytes else 100
        if speed >= 1024**2:    spd = f"{{speed/1024**2:.0f}} MiB/s"
        elif speed >= 1024:     spd = f"{{speed/1024:.0f}} KiB/s"
        else:                   spd = f"{{speed:.0f}} B/s"
        J(progress=dict(pct=round(pct,1), done_mb=round(done_bytes/1024**2,1),
                        total_mb=round(total_bytes/1024**2,1), speed=spd,
                        idx=i, total_files=total_files, file=fname))
    total_elapsed = time.time() - dl_start
    avg = done_bytes / max(total_elapsed, 0.001)
    avg_str = f"{{avg/1024**2:.0f}} MiB/s" if avg >= 1024**2 else f"{{avg/1024:.0f}} KiB/s"
    out_path = local_dir or result_path or "HF cache"
    J(status="complete", path=out_path, avg_speed=avg_str,
      elapsed=f"{{total_elapsed/60:.1f}} min" if total_elapsed > 60 else f"{{total_elapsed:.0f}}s",
      errors=len(errors))
except Exception as e:
    J(status="error", error=str(e))
"""

    async def stream() -> AsyncGenerator[str, None]:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
    custom_dirs = _load_custom_dirs()
    directories = []

    # Always include the default HF cache
    directories.append(_scan_directory(HF_CACHE_DIR))

    # Custom dirs (skip if same as default)
    default_str = str(HF_CACHE_DIR)
    for d in custom_dirs:
        d_expanded = os.path.expanduser(d)
        if d_expanded != default_str:
            directories.append(_scan_directory(Path(d_expanded)))

    return {"directories": directories}


class AddDirRequest(BaseModel):
    path: str

@app.post("/api/hf/inventory/dirs")
async def add_inventory_dir(req: AddDirRequest):
    """Add a custom directory to the inventory scan list."""
    expanded = os.path.expanduser(req.path.strip())
    dirs = _load_custom_dirs()
    if expanded not in dirs:
        dirs.append(expanded)
        _save_custom_dirs(dirs)
    return {"ok": True, "dirs": dirs}

@app.delete("/api/hf/inventory/dirs")
async def remove_inventory_dir(path: str):
    """Remove a custom directory from the inventory scan list."""
    expanded = os.path.expanduser(path.strip())
    dirs = [d for d in _load_custom_dirs() if os.path.expanduser(d) != expanded]
    _save_custom_dirs(dirs)
    return {"ok": True, "dirs": dirs}


class DeleteModelRequest(BaseModel):
    path: str


@app.post("/api/hf/inventory/delete")
async def delete_inventory_model(req: DeleteModelRequest):
    """Delete a downloaded model directory from disk."""
    target = Path(os.path.expanduser(req.path.strip())).resolve()

    # Safety: only allow deletion under HF cache or known custom dirs
    allowed_roots = [HF_CACHE_DIR.resolve()]
    for d in _load_custom_dirs():
        allowed_roots.append(Path(os.path.expanduser(d)).resolve())
    if not any(str(target).startswith(str(r)) for r in allowed_roots):
        raise HTTPException(400, f"Path is not under a known model directory")
    if not target.exists():
        raise HTTPException(404, "Directory not found")
    if not target.is_dir():
        raise HTTPException(400, "Path is not a directory")

    try:
        shutil.rmtree(target)
        return {"ok": True, "deleted": str(target)}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete: {e}")

# ─── Frontend ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZGX · Model Manager</title>
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
.inv-remove-btn:hover{color:var(--red);border-color:var(--red)}</style>
</head>
<body>

<header class="header">
  <div class="hdr-logo">
    <div class="hdr-sigil">Z</div>
    <div>
      <div class="hdr-name">Model Manager</div>
      <div class="hdr-node" id="hdr-node">loading…</div>
    </div>
  </div>
  <div class="hdr-sep"></div>
  <div class="status-cluster">
    <div class="pill" id="pill-sglang"><div class="dot"></div><span>SGLang</span></div>
    <div class="pill" id="pill-ollama"><div class="dot"></div><span>Ollama</span></div>
    <div class="pill" id="pill-litellm"><div class="dot"></div><span>LiteLLM</span></div>
    <div class="pill" id="pill-vllm"><div class="dot"></div><span>vLLM</span></div>
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
    <div class="nav-item" id="nav-hf" onclick="switchTab('hf')">
      <span class="nav-icon">🤗</span>HF Download
    </div>
    <div class="nav-section-label">Routing</div>
    <div class="nav-item" id="nav-litellm" onclick="switchTab('litellm')">
      <span class="nav-icon">⚡</span>LiteLLM
      <span class="nav-badge" id="badge-litellm">—</span>
    </div>
    <div class="nav-section-label">Engine</div>
    <div class="nav-item" id="nav-sglang" onclick="switchTab('sglang')">
      <span class="nav-icon">🚀</span>SGLang
    </div>
    <div class="nav-item" id="nav-vllm" onclick="switchTab('vllm')">
      <span class="nav-icon">⚡</span>vLLM
    </div>
  </nav>

  <main class="main">

    <!-- ─── OLLAMA ─── -->
    <div class="tab active" id="tab-ollama">
      <div class="page-hdr">
        <div class="page-title">Ollama Models</div>
        <div class="page-sub">Pull models from the Ollama library. With wildcard routing enabled, every pulled model is instantly available at <code>:4000</code>.</div>
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
        <div class="page-sub">Download any model from HuggingFace Hub directly to the ZGX. Large models land in <code>~/.cache/huggingface/hub/</code> — ready for SGLang or vLLM.</div>
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

      <!-- ── Model Inventory ── -->
      <div style="display:flex;align-items:center;gap:10px;margin:24px 0 14px">
        <div class="sec-label" style="margin:0;flex:1">Model Inventory</div>
        <button class="btn btn-sm btn-ghost" id="inv-refresh-btn" onclick="loadInventory()">↻ Refresh</button>
      </div>

      <!-- Custom directory management -->
      <div class="card" style="margin-bottom:14px;padding:14px 16px">
        <div style="font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px">Additional Scan Directories</div>
        <div class="inv-custom-dirs" id="inv-custom-dirs"></div>
        <div class="input-row" style="margin-bottom:0">
          <input class="input" id="inv-add-dir" placeholder="/home/user/models  or  ~/models" style="font-size:12px"
            onkeydown="if(event.key==='Enter')addInventoryDir()">
          <button class="btn btn-sm" onclick="addInventoryDir()">+ Add Directory</button>
        </div>
      </div>

      <div id="inv-root">
        <div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div><div style="font-size:12px;color:var(--muted)">Loading inventory…</div></div>
      </div>
    </div>

    <!-- ─── LITELLM ─── -->
    <div class="tab" id="tab-litellm">
      <div class="page-hdr">
        <div class="page-title">LiteLLM Routing</div>
        <div class="page-sub">Unified gateway at <code>:4000</code>. All apps — Open WebUI, scripts, agents — connect here. This config controls which models they can see.</div>
      </div>

      <div class="card" id="wildcard-card">
        <div class="card-row">
          <div class="card-icon">🃏</div>
          <div class="card-info">
            <div class="card-name">Ollama Wildcard Routing</div>
            <div class="card-meta">ollama/* → http://127.0.0.1:11434</div>
          </div>
          <div class="card-actions">
            <button class="btn btn-primary" id="wc-btn" onclick="applyWildcard()">Apply Wildcard</button>
          </div>
        </div>
        <div class="card-desc">
          Adds a single <code>ollama/*</code> entry to your config. After this one change, any model you pull into Ollama is automatically available at <code>:4000</code> — no YAML edits, no restarts.
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

    <!-- ─── SGLANG ─── -->
    <div class="tab" id="tab-sglang">
      <div class="page-hdr">
        <div class="page-title">SGLang Engine</div>
        <div class="page-sub">SGLang loads one large model at a time. ~5 min warm-up. Profiles are auto-detected from <code style="font-family:var(--mono);color:var(--amber);font-size:12px" id="sglang-script-dir">~/SGLang/</code> — add a <code style="font-family:var(--mono);color:var(--amber);font-size:12px">start_*.sh</code> script there to make it appear below.</div>
      </div>

      <div class="engine-card" id="engine-card">
        <div class="engine-status-row">
          <div class="engine-led" id="engine-led"></div>
          <div>
            <div class="engine-title" id="engine-title">Checking…</div>
            <div class="engine-model" id="engine-model"></div>
          </div>
          <div class="engine-actions">
            <button class="btn btn-danger" id="stop-btn" onclick="stopSGLang()" disabled>■ Stop</button>
          </div>
        </div>
        <div class="engine-footer" id="engine-footer">loading…</div>
      </div>

      <div style="margin-bottom:12px;padding:12px 16px;background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.22);border-radius:8px;font-size:12px;color:var(--muted);line-height:1.8">
        <div style="color:var(--amber);font-weight:700;font-size:13px;margin-bottom:4px">📁 Script directory: <span id="sglang-script-dir-banner">~/SGLang/</span></div>
        Place scripts named <code style="font-family:var(--mono);color:var(--amber);font-size:11px">start_*.sh</code> in this folder — each one becomes a profile card below.
        Name, description, and VRAM are read from optional header comments in the script:<br>
        <code style="font-family:var(--mono);font-size:11px;color:var(--dim)"># Name: My Model &nbsp;·&nbsp; # Description: 119B NVFP4 &nbsp;·&nbsp; # VRAM: 119</code>
      </div>

      <div class="sec-label">Profiles</div>
      <div class="profile-list" id="profile-list">
        <div class="empty"><div class="spin-icon" style="margin:0 auto"></div></div>
      </div>

      <div style="display:flex;align-items:center;gap:12px;margin-top:14px">
        <button class="btn btn-primary" id="start-btn" onclick="startSGLang()">▶ Start Selected</button>
        <span style="font-size:12px;color:var(--muted)">Runs start script in background · check status pill</span>
      </div>

      <div class="progress-wrap" id="sglang-progress" style="margin-top:14px">
        <div class="prog-bar-outer"><div class="prog-bar spin"></div></div>
        <div class="prog-log" id="sglang-log"></div>
      </div>
    </div>

    <!-- ─── VLLM ─── -->
    <div class="tab" id="tab-vllm">
      <div class="page-hdr">
        <div class="page-title">vLLM Engine</div>
        <div class="page-sub">vLLM loads one large model at a time via Docker. Profiles are auto-detected from <code style="font-family:var(--mono);color:var(--amber);font-size:12px">~/vLLM/</code> — add a <code style="font-family:var(--mono);color:var(--amber);font-size:12px">start_*.sh</code> script there to make it appear below.</div>
      </div>

      <div class="engine-card" id="vllm-engine-card">
        <div class="engine-status-row">
          <div class="engine-led" id="vllm-engine-led"></div>
          <div>
            <div class="engine-title" id="vllm-engine-title">Checking…</div>
            <div class="engine-model" id="vllm-engine-model"></div>
          </div>
          <div class="engine-actions">
            <button class="btn btn-danger" id="vllm-stop-btn" onclick="stopVLLM()" disabled>■ Stop</button>
          </div>
        </div>
        <div class="engine-footer" id="vllm-engine-footer">loading…</div>
      </div>

      <div style="margin-bottom:12px;padding:12px 16px;background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.22);border-radius:8px;font-size:12px;color:var(--muted);line-height:1.8">
        <div style="color:var(--amber);font-weight:700;font-size:13px;margin-bottom:4px">📁 Script directory: <span id="vllm-script-dir-banner">~/vLLM/</span></div>
        Place scripts named <code style="font-family:var(--mono);color:var(--amber);font-size:11px">start_*.sh</code> in this folder — each one becomes a profile card below.
        Name, description, and VRAM are read from optional header comments in the script:<br>
        <code style="font-family:var(--mono);font-size:11px;color:var(--dim)"># Name: My Model &nbsp;·&nbsp; # Description: 49B BF16 &nbsp;·&nbsp; # VRAM: 97</code>
      </div>

      <div class="sec-label">Profiles</div>
      <div class="profile-list" id="vllm-profile-list">
        <div class="empty"><div class="spin-icon" style="margin:0 auto"></div></div>
      </div>

      <div style="display:flex;align-items:center;gap:12px;margin-top:14px">
        <button class="btn btn-primary" id="vllm-start-btn" onclick="startVLLM()">▶ Start Selected</button>
        <span style="font-size:12px;color:var(--muted)">Runs start script in background · check status pill</span>
      </div>

      <div class="progress-wrap" id="vllm-progress" style="margin-top:14px">
        <div class="prog-bar-outer"><div class="prog-bar spin"></div></div>
        <div class="prog-log" id="vllm-log"></div>
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

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  pollStatus();
  loadOllamaModels();
  loadNodeInfo();
  loadScriptDirs();
  statusTimer = setInterval(pollStatus, 12000);
});

async function loadScriptDirs() {
  try {
    const d = await apiFetch('/api/scriptdirs');
    ['sglang-script-dir-banner'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = d.sglang + '/';
    });
    ['vllm-script-dir-banner'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = d.vllm + '/';
    });
  } catch(e) {}
}

async function loadNodeInfo() {
  try {
    const r = await fetch('/api/nodeinfo');
    const d = await r.json();
    document.getElementById('hdr-node').textContent =
      d.hostname + ' \u00b7 ' + d.ip + ' \u00b7 :' + d.port;
    const parts = ['Port :' + d.sglang_port, 'Docker'];
    if (d.arch) parts.push(d.arch);
    if (d.memory_gb) parts.push(d.memory_gb + ' GB memory');
    document.getElementById('engine-footer').textContent = parts.join(' \u00b7 ');
    const vparts = ['Port :' + d.vllm_port, 'Docker'];
    if (d.arch) vparts.push(d.arch);
    if (d.memory_gb) vparts.push(d.memory_gb + ' GB memory');
    document.getElementById('vllm-engine-footer').textContent = vparts.join(' \u00b7 ');
  } catch(e) {
    document.getElementById('hdr-node').textContent = 'could not detect';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tab switching
// ─────────────────────────────────────────────────────────────────────────────

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  activeTab = name;
  if (name === 'litellm') { loadLiteLLMModels(); loadLiteLLMConfig(); checkWildcard(); }
  if (name === 'sglang')  { loadSGLangStatus(); loadProfiles(); }
  if (name === 'vllm')    { loadVLLMStatus(); loadVLLMProfiles(); }
  if (name === 'ollama')  { loadOllamaModels(); }
  if (name === 'hf')      { loadInventory(); loadCustomDirs(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Status
// ─────────────────────────────────────────────────────────────────────────────

async function pollStatus() {
  try {
    const d = await apiFetch('/api/status');
    setPill('pill-sglang', d.sglang.ok,
      d.sglang.model ? 'SGLang · ' + d.sglang.model.split('/').pop().slice(0,18) : 'SGLang');
    setPill('pill-ollama', d.ollama.ok, 'Ollama');
    setPill('pill-litellm', d.litellm.ok, 'LiteLLM');
    setPill('pill-vllm', d.vllm.ok,
      d.vllm.model ? 'vLLM · ' + d.vllm.model.split('/').pop().slice(0,18) : 'vLLM');
  } catch(e) {}
}

function setPill(id, ok, label) {
  const el = document.getElementById(id);
  el.className = 'pill ' + (ok ? 'ok' : 'err');
  el.querySelector('span').textContent = label;
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
      headers: {'Content-Type': 'application/json'},
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
    const r = await fetch('/api/ollama/models/' + encodeURIComponent(name), {method:'DELETE'});
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
      status.innerHTML = '<div class="wc-active">✓ Wildcard active — all Ollama models auto-exposed at :4000</div>';
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

async function loadSGLangStatus() {
  try {
    const d = await apiFetch('/api/sglang/status');
    const led   = document.getElementById('engine-led');
    const title = document.getElementById('engine-title');
    const model = document.getElementById('engine-model');
    const card  = document.getElementById('engine-card');
    const stop  = document.getElementById('stop-btn');

    if (d.running) {
      led.className = 'engine-led on';
      title.textContent = 'SGLang — Running';
      model.textContent = d.model || 'Model loading…';
      card.classList.add('online');
      stop.disabled = false;
    } else {
      led.className = 'engine-led';
      title.textContent = 'SGLang — Stopped';
      model.textContent = '';
      card.classList.remove('online');
      stop.disabled = true;
    }
  } catch(e) {}
}

async function loadProfiles() {
  const el = document.getElementById('profile-list');
  try {
    const profiles = await apiFetch('/api/sglang/profiles');
    if (!profiles.length) {
      el.innerHTML = '<div class="empty"><div class="empty-text">No profiles defined</div></div>';
      return;
    }
    if (!selectedProfile) selectedProfile = profiles[0].id;
    renderProfiles(profiles);
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-text">Could not load profiles</div></div>';
  }
}

function renderProfiles(profiles) {
  document.getElementById('profile-list').innerHTML = profiles.map(p => `
    <div class="profile-item ${selectedProfile === p.id ? 'selected' : ''}"
         onclick="selectProfile('${p.id}', this, ${JSON.stringify(profiles).replace(/"/g,"'")})">
      <div class="p-radio"></div>
      <div class="p-info">
        <div class="p-name">${p.name}</div>
        <div class="p-desc">${p.description}</div>
      </div>
      <div class="p-vram">${p.vram_gb != null ? p.vram_gb + ' GB' : '—'}</div>
    </div>
  `).join('');
}

function selectProfile(id, el, profiles) {
  selectedProfile = id;
  document.querySelectorAll('.profile-item').forEach(p => p.classList.remove('selected'));
  el.classList.add('selected');
}

async function stopSGLang() {
  if (!confirm('Stop SGLang? This will interrupt any active inference requests.')) return;
  const btn = document.getElementById('stop-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div>';
  try {
    const d = await apiFetch('/api/sglang/stop', 'POST');
    toast(d.ok ? '✓ SGLang stopped' : 'Stop may have failed: ' + d.output, d.ok ? 'ok' : 'err');
    setTimeout(() => { loadSGLangStatus(); btn.innerHTML = '■ Stop'; }, 1500);
  } catch(e) {
    toast('Error: ' + e.message, 'err');
    btn.innerHTML = '■ Stop';
  }
}

async function startSGLang() {
  if (!selectedProfile) { toast('Select a profile first', 'err'); return; }
  const btn  = document.getElementById('start-btn');
  const prog = document.getElementById('sglang-progress');
  const log  = document.getElementById('sglang-log');

  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Launching…';
  prog.classList.add('show');
  log.textContent = 'Sending start command…';

  try {
    const d = await apiFetch('/api/sglang/start', 'POST', {profile: selectedProfile});
    toast('✓ SGLang starting — takes ~5 min', 'ok');
    log.textContent = d.message + '\n\nPolling status every 20 seconds…';

    const poll = setInterval(async () => {
      await loadSGLangStatus();
      const led = document.getElementById('engine-led');
      if (led.classList.contains('on')) {
        clearInterval(poll);
        toast('✓ SGLang is ready!', 'ok');
        prog.classList.remove('show');
      }
    }, 20000);
  } catch(e) {
    toast('Start failed: ' + e.message, 'err');
    prog.classList.remove('show');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ Start Selected';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// vLLM
// ─────────────────────────────────────────────────────────────────────────────

async function loadVLLMStatus() {
  try {
    const d = await apiFetch('/api/vllm/status');
    const led   = document.getElementById('vllm-engine-led');
    const title = document.getElementById('vllm-engine-title');
    const model = document.getElementById('vllm-engine-model');
    const card  = document.getElementById('vllm-engine-card');
    const stop  = document.getElementById('vllm-stop-btn');

    if (d.running) {
      led.className = 'engine-led on';
      title.textContent = 'vLLM — Running';
      model.textContent = d.model || 'Model loading…';
      card.classList.add('online');
      stop.disabled = false;
    } else {
      led.className = 'engine-led';
      title.textContent = 'vLLM — Stopped';
      model.textContent = '';
      card.classList.remove('online');
      stop.disabled = true;
    }
  } catch(e) {}
}

async function loadVLLMProfiles() {
  const el = document.getElementById('vllm-profile-list');
  try {
    const profiles = await apiFetch('/api/vllm/profiles');
    if (!profiles.length) {
      el.innerHTML = '<div class="empty"><div class="empty-text">No vLLM start scripts found — add <code style="font-family:var(--mono);color:var(--amber);font-size:11px">start_*.sh</code> scripts to <code style="font-family:var(--mono);color:var(--amber);font-size:11px">~/vLLM/</code></div></div>';
      return;
    }
    if (!selectedVLLMProfile) selectedVLLMProfile = profiles[0].id;
    renderVLLMProfiles(profiles);
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-text">Could not load profiles</div></div>';
  }
}

function renderVLLMProfiles(profiles) {
  document.getElementById('vllm-profile-list').innerHTML = profiles.map(p => `
    <div class="profile-item ${selectedVLLMProfile === p.id ? 'selected' : ''}"
         onclick="selectVLLMProfile('${p.id}', this)">
      <div class="p-radio"></div>
      <div class="p-info">
        <div class="p-name">${p.name}</div>
        <div class="p-desc">${p.description}</div>
      </div>
      <div class="p-vram">${p.vram_gb != null ? p.vram_gb + ' GB' : '—'}</div>
    </div>
  `).join('');
}

function selectVLLMProfile(id, el) {
  selectedVLLMProfile = id;
  document.querySelectorAll('#vllm-profile-list .profile-item').forEach(p => p.classList.remove('selected'));
  el.classList.add('selected');
}

async function stopVLLM() {
  if (!confirm('Stop vLLM? This will interrupt any active inference requests.')) return;
  const btn = document.getElementById('vllm-stop-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div>';
  try {
    const d = await apiFetch('/api/vllm/stop', 'POST');
    toast(d.ok ? '✓ vLLM stopped' : 'Stop may have failed: ' + d.output, d.ok ? 'ok' : 'err');
    setTimeout(() => { loadVLLMStatus(); btn.innerHTML = '■ Stop'; }, 1500);
  } catch(e) {
    toast('Error: ' + e.message, 'err');
    btn.innerHTML = '■ Stop';
  }
}

async function startVLLM() {
  if (!selectedVLLMProfile) { toast('Select a profile first', 'err'); return; }
  const btn  = document.getElementById('vllm-start-btn');
  const prog = document.getElementById('vllm-progress');
  const log  = document.getElementById('vllm-log');

  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Launching…';
  prog.classList.add('show');
  log.textContent = 'Sending start command…';

  try {
    const d = await apiFetch('/api/vllm/start', 'POST', {profile: selectedVLLMProfile});
    toast('✓ vLLM starting — may take a few minutes', 'ok');
    log.textContent = d.message + '\n\nPolling status every 20 seconds…';

    const poll = setInterval(async () => {
      await loadVLLMStatus();
      const led = document.getElementById('vllm-engine-led');
      if (led.classList.contains('on')) {
        const model = document.getElementById('vllm-engine-model');
        if (model.textContent && model.textContent !== 'Model loading…') {
          clearInterval(poll);
          toast('✓ vLLM is ready!', 'ok');
          prog.classList.remove('show');
        } else {
          log.textContent = d.message + '\n\nContainer running — model still loading…';
        }
      }
    }, 20000);
  } catch(e) {
    toast('Start failed: ' + e.message, 'err');
    prog.classList.remove('show');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ Start Selected';
  }
}

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
      headers: {'Content-Type': 'application/json'},
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
// HF Inventory
// ─────────────────────────────────────────────────────────────────────────────

function dtypeClass(dtype) {
  const m = {
    'FP32':'inv-fp32','FP16':'inv-fp16','BF16':'inv-bf16',
    'FP8':'inv-fp8','FP4':'inv-fp4','INT4':'inv-int4','INT8':'inv-int8',
    'GGUF':'inv-int4',
  };
  return m[dtype] || 'inv-unknown';
}

function renderInventoryDir(dirData, isFirst) {
  const { path, is_hf_cache, models, error } = dirData;
  const shortPath = path.replace(/^\/home\/[^/]+/, '~');

  let html = `<div class="inv-dir-block">
    <div class="inv-dir-header">
      <span class="inv-dir-path" title="${path}">${shortPath}</span>`;

  if (isFirst) {
    html += `<span class="inv-dir-badge">HF Cache · Default</span>`;
  } else {
    html += `<span class="inv-dir-badge custom">Custom</span>`;
  }
  html += `<span style="font-family:var(--mono);font-size:10px;color:var(--muted)">${models.length} model${models.length !== 1 ? 's' : ''}</span>`;
  html += `</div>`;

  if (error) {
    html += `<div class="inv-empty">⚠ ${error}</div></div>`;
    return html;
  }
  if (!models.length) {
    html += `<div class="inv-empty">No models found in this directory</div></div>`;
    return html;
  }

  html += `<div class="inv-table-wrap"><table class="inv-table">
    <thead><tr>
      <th>Model</th>
      <th>Dtype</th>
      <th>Params</th>
      <th>Arch</th>
      <th>Size</th>
      <th>Reasoning</th>
      <th>Script</th>
      <th>Modalities</th>
      <th style="width:36px"></th>
    </tr></thead>
    <tbody>`;

  for (const m of models) {
    const params = m.params_b != null ? m.params_b + 'B' : '—';
    const size   = m.size_gb  ? m.size_gb + ' GB' : '—';
    const dc     = dtypeClass(m.dtype);
    const archBadge = m.model_arch === 'MoE'
      ? `<span class="inv-badge inv-moe">MoE</span>`
      : `<span class="inv-badge inv-dense">Dense</span>`;
    const reasonBadge = m.is_reasoning
      ? `<span class="inv-yes">✓ Yes</span>`
      : `<span class="inv-no">—</span>`;
    let scriptBadge = `<span class="inv-no">—</span>`;
    if (m.has_script && m.script_engine) {
      const cls = m.script_engine === 'SGLang' ? 'inv-engine-sg' : 'inv-engine-vl';
      scriptBadge = `<span class="inv-engine ${cls}">${m.script_engine}</span>`;
    }
    const modBadges = m.modalities.map(mod => {
      const cls = mod === 'Embedding' ? ' embed' : mod === 'Audio' ? ' audio' : '';
      return `<span class="inv-modality${cls}">${mod}</span>`;
    }).join('');

    html += `<tr>
      <td>
        <div class="inv-model-name">${m.name}</div>
        ${m.owner ? `<div class="inv-owner">${m.owner}</div>` : ''}
      </td>
      <td><span class="inv-badge ${dc}">${m.dtype}</span></td>
      <td style="font-family:var(--mono);font-size:11px">${params}</td>
      <td>${archBadge}</td>
      <td style="font-family:var(--mono);font-size:11px;white-space:nowrap">${size}</td>
      <td>${reasonBadge}</td>
      <td>${scriptBadge}</td>
      <td>${modBadges}</td>
      <td><button class="btn-icon-del" title="Delete model" onclick="deleteInventoryModel('${m.dir_path}','${m.full_name || m.name}')">✕</button></td>
    </tr>`;
  }

  html += `</tbody></table></div></div>`;
  return html;
}

async function loadInventory() {
  const root = document.getElementById('inv-root');
  if (!root) return;
  root.innerHTML = '<div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div><div style="font-size:12px;color:var(--muted)">Scanning model directories…</div></div>';
  try {
    const d = await apiFetch('/api/hf/inventory');
    const dirs = d.directories || [];
    if (!dirs.length || dirs.every(dir => !dir.models.length && !dir.error)) {
      root.innerHTML = '<div class="empty"><div class="empty-icon">📭</div><div class="empty-text">No models found. Pull or download a model to get started.</div></div>';
      return;
    }
    root.innerHTML = dirs.map((dir, i) => renderInventoryDir(dir, i === 0)).join('');
  } catch(e) {
    root.innerHTML = `<div class="empty"><div class="empty-icon">⚠</div><div class="empty-text">Could not load inventory: ${e.message}</div></div>`;
  }
}

async function loadCustomDirs() {
  // Reload custom dirs display by fetching inventory and reading dirs
  try {
    const d = await apiFetch('/api/hf/inventory');
    const customDirsEl = document.getElementById('inv-custom-dirs');
    if (!customDirsEl) return;
    const customDirs = (d.directories || []).slice(1); // skip default
    if (!customDirs.length) {
      customDirsEl.innerHTML = '<div style="font-size:12px;color:var(--muted);padding:2px 0">No additional directories added</div>';
      return;
    }
    customDirsEl.innerHTML = customDirs.map(dir => {
      const safe = encodeURIComponent(dir.path);
      return `<div class="inv-custom-dir-row">
        <span class="inv-custom-dir-path">${dir.path.replace(/^\/home\/[^/]+/, '~')}</span>
        <button class="inv-remove-btn" onclick="removeInventoryDir('${dir.path.replace(/'/g,"\\'")}')">✕ Remove</button>
      </div>`;
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
    toast('✓ Directory added', 'ok');
    await loadCustomDirs();
    await loadInventory();
  } catch(e) {
    toast('Failed: ' + e.message, 'err');
  }
}

async function removeInventoryDir(path) {
  try {
    await fetch('/api/hf/inventory/dirs?' + new URLSearchParams({path}), {method:'DELETE'});
    toast('✓ Directory removed', 'ok');
    await loadCustomDirs();
    await loadInventory();
  } catch(e) {
    toast('Failed: ' + e.message, 'err');
  }
}

async function deleteInventoryModel(dirPath, modelName) {
  if (!confirm('Delete "' + modelName + '" from disk?\n\nThis will permanently remove all files in:\n' + dirPath)) return;
  try {
    await apiFetch('/api/hf/inventory/delete', 'POST', {path: dirPath});
    toast('✓ Deleted: ' + modelName, 'ok');
    await loadInventory();
  } catch(e) {
    toast('Delete failed: ' + e.message, 'err');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utils
// ─────────────────────────────────────────────────────────────────────────────

async function apiFetch(url, method = 'GET', body = null) {
  const opts = {method, headers: {'Content-Type': 'application/json'}};
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const d = await r.json(); msg = d.detail || JSON.stringify(d); } catch(e) {}
    throw new Error(msg);
  }
  return r.json();
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
