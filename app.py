#!/usr/bin/env python3
"""
DGX Model Manager
Unified web UI for managing models across Ollama, SGLang, vLLM, and LiteLLM.
Run via systemd: model-manager.service
"""

import asyncio
import difflib
import fnmatch
import hashlib
import hmac
import json
import logging
import os
import platform
import re as _re
import shlex
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

from discord_notify import send_discord_alert

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
    except Exception as e:
        print(f"WARNING: failed to parse {_CONFIG_FILE}: {e} — starting with defaults", flush=True)

APP_PORT = _app_config.get("app", {}).get("port", 8090)
APP_HOST = _app_config.get("app", {}).get("host", "0.0.0.0")


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

# LiteLLM backend mode — when litellm runs inside Kubernetes (fed by a
# configmap) instead of a systemd unit, restarts must sync the config file
# into the configmap and roll the deployment. Opt in via config.json:
#   "litellm_k8s": {"enabled": true, "namespace": "llm-inference",
#                   "configmap": "litellm-config", "configmap_key": "config.yaml",
#                   "deployment": "litellm"}
_litellm_k8s = _app_config.get("litellm_k8s", {})

# Dashboards — the tab lists every web UI actually running on this box, found
# by discovery (host TCP listeners + Kubernetes NodePorts, see _discover_sites).
# The "sites" array in config.json is an *overlay*, not the list: entries with a
# "port" rename/describe/group a discovered port, and entries with a verbatim
# "url" pin something discovery cannot see (a UI on another host).
#   {"name": "...", "desc": "...", "group": "...", "port": 3000}  — resolved
#   against app.sites_base (falls back to app.host, then the request host), or
#   {"name": "...", "url": "http://other-host:1234"}              — verbatim.
# Optional "scheme" (default "http") applies to port-based entries. A missing
# or empty array is normal — discovery still fills the tab.
_SITES = _app_config.get("sites") or []
_SITES_BASE = _app_config.get("app", {}).get("sites_base", "")

# Discovery knobs, all optional (config.json "sites_discovery"):
#   enabled       turn auto-discovery off and fall back to the "sites" array
#   kubernetes    probe NodePort services via kubectl (skipped if it fails)
#   ttl_s         cache lifetime; the tab re-probes ~40 ports per refresh
#   probe_timeout_s / exclude_ports / include_ports
_SITES_DISCOVERY = _app_config.get("sites_discovery") or {}

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

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")


def _auth_open_unauthenticated() -> bool:
    """True when no key is set AND we are bound somewhere other than loopback.

    Reads the module globals at call time rather than caching: PUT /api/config mutates
    _API_KEY_HASH through a `global` statement, and the next request must see it.
    """
    return not _API_KEY_HASH and APP_HOST not in _LOOPBACK_HOSTS


async def verify_auth(request: Request):
    """Check the API key on mutating endpoints. Four states:

      1. Key set + valid Bearer header      → allow.
      2. Key set + missing/wrong header     → 401.
      3. No key, bound to loopback          → allow (local-only deployments are fine).
      4. No key, bound to a real interface  → 503, unless MODEL_MANAGER_ALLOW_UNAUTH is
         set, in which case allow but log every unauthenticated mutation. Never silent.

    Bootstrap consequence of state 4: with the 503 active, PUT /api/config cannot install
    the first API key from a remote host. That is intentional — an endpoint that lets an
    unauthenticated caller set the credential is not a fix. Bootstrap by setting
    app.api_key in config.json, or by reaching the UI over loopback.
    """
    if not _API_KEY_HASH:
        if not _auth_open_unauthenticated():
            return  # loopback-only deployment
        if not os.environ.get("MODEL_MANAGER_ALLOW_UNAUTH"):
            raise HTTPException(
                503,
                f"Refusing to serve a mutating request: bound to {APP_HOST} with no API "
                "key configured. Set app.api_key in config.json (or via the UI over "
                "loopback), or set MODEL_MANAGER_ALLOW_UNAUTH=1 to accept the risk.")
        _logger.warning("UNAUTHENTICATED mutating request allowed by "
                        "MODEL_MANAGER_ALLOW_UNAUTH: %s %s",
                        request.method, request.url.path)
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


def _get_available_memory_gb() -> float:
    """Currently-available system memory in GB, from /proc/meminfo MemAvailable.

    On the GB10 (DGX Spark) the GPU and system RAM are ONE ~121 GB unified pool,
    and nvidia-smi reports memory as N/A on this hardware — so MemAvailable is
    the only trustworthy signal of how much room a new model actually has. Note
    that vLLM's --gpu-memory-utilization reserves a fraction of this whole pool,
    yet that reservation never appears in the container's RSS or docker stats, so
    a running engine's real footprint is unmeasurable and must be estimated from
    profile metadata (see _vram_admission_check).
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0

# ─── Models ───────────────────────────────────────────────────────────────────

class PullRequest(BaseModel):
    name: str

class EngineStartRequest(BaseModel):
    profile: str
    force: bool = False
    # llama.cpp only: names an entry in config.json llamacpp.recipes. One profile script
    # covers the whole quant ladder, so the quant/ctx/spec choice is a request parameter
    # rather than a script per combination.
    recipe: Optional[str] = None
    # vLLM only: per-launch tuning knobs. Transient by design — nothing is written to
    # the script on disk, so an override cannot outlive the launch that asked for it.
    overrides: Optional[dict] = None

class OllamaStopRequest(BaseModel):
    name: str

class CreateVLLMProfileRequest(BaseModel):
    path: str
    model_name: Optional[str] = None

class HFDownloadRequest(BaseModel):
    repo_id: str
    local_dir: Optional[str] = None
    ignore_patterns: Optional[list[str]] = None
    allow_patterns: Optional[list[str]] = None

class ApplyRecRequest(BaseModel):
    id: str
    profile: str
    confirm: bool = False

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
        return r.status_code < 400 or r.status_code in (401, 403)
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


async def _restart_litellm_backend() -> tuple[bool, str]:
    """Restart LiteLLM after a config change.

    k8s mode: sync LITELLM_CONFIG into the configmap, then roll the deployment
    (the pod mounts the configmap read-only, so a file write alone is invisible
    to the cluster). Default mode: restart the systemd unit (upstream behavior).
    """
    if _litellm_k8s.get("enabled"):
        ns  = _litellm_k8s.get("namespace", "default")
        cm  = _litellm_k8s.get("configmap", "litellm-config")
        key = _litellm_k8s.get("configmap_key", "config.yaml")
        dep = _litellm_k8s.get("deployment", "litellm")
        r = await _run("kubectl", "create", "configmap", cm, "-n", ns,
                       f"--from-file={key}={LITELLM_CONFIG}",
                       "--dry-run=client", "-o", "yaml", timeout=15)
        if r.returncode != 0:
            return False, f"configmap render failed: {r.stderr.strip()}"
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "apply", "-f", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate(r.stdout.encode())
        if proc.returncode != 0:
            return False, f"configmap apply failed: {err.decode().strip()}"
        r2 = await _run("kubectl", "rollout", "restart", f"deployment/{dep}", "-n", ns, timeout=15)
        if r2.returncode != 0:
            return False, f"rollout restart failed: {r2.stderr.strip()}"
        return True, "configmap synced + deployment rolled"
    result = await _run("sudo", "systemctl", "restart", "litellm", timeout=15)
    if result.returncode != 0:
        hint = " — configure passwordless sudo (see Settings or the banner on this tab)" if "password" in result.stderr.lower() else ""
        return False, result.stderr.strip() + hint
    return True, "systemd unit restarted"


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
    derived = recommended = generated_at = None
    warnings: list = []
    meta_error = None

    def _json_header(prefix: str, raw: str):
        """Parse one JSON header. An unparseable header is a DEFECT, not a fallback
        (04-CONTEXT.md locked decision) — it must be visible, never swallowed."""
        nonlocal meta_error
        try:
            return json.loads(raw)
        except Exception as exc:
            if meta_error is None:
                meta_error = f"{prefix} header is not valid JSON: {exc}"
            return None

    text = ""
    try:
        text = script_path.read_text()
    except Exception:
        pass

    try:
        for line in text.splitlines()[:20]:
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
            elif line.startswith("# Derived:"):
                parsed = _json_header("# Derived:", line[10:].strip())
                derived = parsed if isinstance(parsed, dict) else derived
                if parsed is not None and not isinstance(parsed, dict) and meta_error is None:
                    meta_error = "# Derived: header is not a JSON object"
            elif line.startswith("# Recommended:"):
                parsed = _json_header("# Recommended:", line[14:].strip())
                recommended = parsed if isinstance(parsed, dict) else recommended
            elif line.startswith("# Warnings:"):
                parsed = _json_header("# Warnings:", line[11:].strip())
                if isinstance(parsed, list):
                    warnings = [str(w) for w in parsed]
                elif parsed is not None and meta_error is None:
                    meta_error = "# Warnings: header is not a JSON array"
            elif line.startswith("# Generated:"):
                generated_at = line[12:].strip() or None
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
        # 04-02 (Option A): everything the profile card needs comes from the script
        # text itself. Deliberately NO filesystem or config.json read here — this
        # function sits on `_scan_profiles`, the hot list path (threat T-04-08).
        "derived":       derived,
        "recommended":   recommended,
        "warnings":      warnings,
        "generated_at":  generated_at,
        "meta_error":    meta_error,
        # 04-03: the card picks a state from `classification`; `flags` gives a
        # legacy script parsed values to render read-only (or `unparseable`).
        "classification": _classify_script(text),
        "flags":          _parse_script_flags(text),
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


def _script_model_dir(script_path: str, model_dirs: list[Path]) -> Optional[Path]:
    """Inverse of _check_script_xref: which model directory does this script launch?

    Matches on the cache directory name (models--owner--name) and on the bare
    owner/name, since hand-written profiles reference a resolved snapshot path
    rather than a repo id. Returns the longest match: `Qwen3-8B` is a substring
    of `Qwen3-8B-FP8`, and the shorter name would otherwise win arbitrarily.
    """
    try:
        content = _script_content_cache.get(script_path)
        if content is None:
            content = Path(script_path).read_text().lower()
            _script_content_cache[script_path] = content
    except Exception:
        return None

    best: Optional[Path] = None
    for d in model_dirs:
        stem = d.name
        # A short name is not evidence: generic path components match everything.
        if len(stem) < 8:
            continue
        needles = [stem.lower()]
        if stem.startswith("models--"):
            needles.append(stem[8:].replace("--", "/").lower())
        if any(n in content for n in needles):
            if best is None or len(d.name) > len(best.name):
                best = d
    return best


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


# ── Derived launch spec: attention topology and KV sizing ────────────────────
# Layer types that hold NO KV cache at all: their state is a fixed-size recurrent tensor,
# independent of context length. Mis-filing one of these as attention is what produces the
# 4-11x overestimate the naive num_hidden_layers count gives on every hybrid model here.
_KV_STATELESS_LAYER_TYPES = frozenset({"linear_attention", "mamba", "recurrent"})
# Nemotron's hybrid_override_pattern alphabet: M = Mamba, E = MLP/expert, * = attention.
# Only '*' carries KV, so len(pattern) - count('*') is a stateless count, NOT a sliding one.
_PATTERN_FULL_CHAR = "*"
_PATTERN_STATELESS_CHARS = frozenset({"M", "E"})


def _as_int(value, default: int = 0) -> int:
    """Coerce a vendor-authored config value to int, falling back instead of raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_attention_topology(config: dict) -> dict:
    """Classify a model's layers into full / bounded / stateless KV from its parsed config.

    WHY this exists: a naive `num_hidden_layers` KV estimate overestimates by 4-11x on every
    hybrid model on this box, because most of their layers hold no KV cache at all. A
    recommender without this classification refuses context that is actually free — an
    88-layer Nemotron carries only 8 attention layers, and a 40-layer Qwen3.6 only 10.

    There are THREE KV classes, not two:
      full      — grows with context: bytes/token x max_model_len
      bounded   — capped by a sliding window: bytes/token x sliding_window
      stateless — zero, a fixed recurrent state (linear_attention, Mamba/expert layers)

    Precedence chain, first hit wins, verified against all 17 configs present here:
      layer_types -> hybrid_override_pattern -> full_attention_interval ->
      sliding_window AND use_sliding_window -> dense.
    The order matters and is not cosmetic: Qwen3.6 and qwen3-next declare BOTH `layer_types`
    and `full_attention_interval`, and only agree by arithmetic luck (48 // 4 == 12).

    Two traps this deliberately guards:
      1. An unrecognized layer type is counted as FULL attention and reported in `warnings`,
         never silently dropped to zero. Over-reserving memory is recoverable; under-reserving
         OOMs at model load, and it is a silent wrong answer rather than a crash.
      2. `sliding_window` is only honoured when `use_sliding_window` is also truthy. Five of
         the seventeen models here declare a window while having the flag false; reading the
         window alone would collapse a 17.2 GB dense KV estimate to near zero.

    Pure: dict in, dict out. No filesystem, network, kernel meminfo or vLLM. Pool size and
    weights are the caller's parameters, not lookups — the whole phase must be verifiable
    with vLLM down.

    Returns {num_hidden_layers, full_attention_layers, bounded_kv_layers, stateless_layers,
    sliding_window, source_field, warnings}. `sliding_window` is the window that applies to
    the bounded layers, or None when no layer is bounded — a dense model's vestigial window is
    deliberately not reported, so a caller cannot multiply by it.
    """
    cfg = config if isinstance(config, dict) else {}
    # VL and nested-text models carry the transformer fields under text_config; read there.
    inner = cfg.get("text_config")
    t = inner if isinstance(inner, dict) and inner else cfg

    declared = _as_int(t.get("num_hidden_layers"))
    warns: list[str] = []
    seen_unknown: set[str] = set()

    def _warn_unknown(label: str) -> None:
        # One warning per distinct unknown type: a 40-layer model of unknown layers should
        # produce one actionable line, not forty.
        if label in seen_unknown:
            return
        seen_unknown.add(label)
        warns.append(f"unknown layer type: {label!r} — counted as full attention")

    def _result(source: str, total: int, full: int, bounded: int, stateless: int,
                window) -> dict:
        return {
            "num_hidden_layers": total,
            "full_attention_layers": full,
            "bounded_kv_layers": bounded,
            "stateless_layers": stateless,
            "sliding_window": window if bounded > 0 else None,
            "source_field": source,
            "warnings": warns,
        }

    # 1. Explicit per-layer list (Qwen3.6, qwen3-next, gpt-oss).
    layer_types = t.get("layer_types")
    if isinstance(layer_types, (list, tuple)) and layer_types:
        full = bounded = stateless = 0
        for entry in layer_types:
            name = str(entry)
            if name == "full_attention":
                full += 1
            elif "sliding" in name:
                bounded += 1
            elif name in _KV_STATELESS_LAYER_TYPES:
                stateless += 1
            else:
                _warn_unknown(name)
                full += 1
        return _result("layer_types", declared or len(layer_types), full, bounded, stateless,
                       _as_int(t.get("sliding_window")) or None)

    # 2. Nemotron-style pattern string.
    pattern = t.get("hybrid_override_pattern")
    if isinstance(pattern, str) and pattern:
        full = stateless = 0
        for ch in pattern:
            if ch == _PATTERN_FULL_CHAR:
                full += 1
            elif ch in _PATTERN_STATELESS_CHARS:
                stateless += 1
            else:
                _warn_unknown(ch)
                full += 1
        return _result("hybrid_override_pattern", declared or len(pattern), full, 0, stateless,
                       None)

    # 3. Every Nth layer is attention; the rest hold no KV.
    interval = _as_int(t.get("full_attention_interval"))
    if interval > 0:
        full = declared // interval
        return _result("full_attention_interval", declared, full, 0, declared - full, None)

    # 4. Uniformly windowed — only when the model actually enables the window.
    window = _as_int(t.get("sliding_window"))
    if window > 0 and t.get("use_sliding_window"):
        return _result("sliding_window", declared, 0, declared, 0, window)

    # 5. Dense fallback: every layer is full attention.
    return _result("dense", declared, declared, 0, 0, None)


def _kv_bytes_per_token(config: dict, kv_dtype_bytes: int = 1) -> dict:
    """Size one model's KV cache from its parsed config, split by how it scales with context.

    per_layer_bytes      = 2 (K and V) x num_key_value_heads x head_dim x kv_dtype_bytes
    full_bytes_per_token = full_attention_layers x per_layer_bytes
    bounded_bytes_total  = bounded_kv_layers x per_layer_bytes x sliding_window

    Note the deliberate asymmetry in those last two, because it is easy to misuse: the first
    is a RATE and the caller multiplies it by max_model_len; the second is already a TOTAL and
    must not be. A windowed layer holds `sliding_window` tokens whether the context is 8k or
    262k, so growing the context does not grow its cost. Stateless layers contribute nothing
    at all — they are excluded by construction in `_resolve_attention_topology`, not by
    falling through an unmatched branch, which is the bug that made the prototype accidentally
    correct.

    `head_dim` falls back to `hidden_size // num_attention_heads` (13 of the 17 models on this
    box derive it that way; only the hybrid Qwen/Nemotron families declare it). The division
    is guarded: a config with zero or no attention heads yields a head_dim of 0 rather than a
    ZeroDivisionError, because these dicts come from vendor-authored files that this codebase
    does not control (T-02-04).

    Pure: dict in, dict out. No filesystem, network, kernel meminfo or vLLM — pool size,
    weight size and context length are the caller's parameters.

    Returns {topology, num_key_value_heads, head_dim, head_dim_source, per_layer_bytes,
    full_bytes_per_token, bounded_bytes_total}.
    """
    cfg = config if isinstance(config, dict) else {}
    inner = cfg.get("text_config")
    t = inner if isinstance(inner, dict) and inner else cfg

    topology = _resolve_attention_topology(cfg)

    attention_heads = _as_int(t.get("num_attention_heads"))
    # Grouped-query attention shrinks the KV width; without it the KV head count is the
    # attention head count.
    kv_heads = _as_int(t.get("num_key_value_heads")) or attention_heads

    head_dim = _as_int(t.get("head_dim"))
    if head_dim > 0:
        head_dim_source = "explicit"
    else:
        head_dim_source = "hidden_size//num_attention_heads"
        head_dim = _as_int(t.get("hidden_size")) // attention_heads if attention_heads > 0 else 0

    per_layer_bytes = 2 * kv_heads * head_dim * _as_int(kv_dtype_bytes)

    return {
        "topology": topology,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "head_dim_source": head_dim_source,
        "per_layer_bytes": per_layer_bytes,
        "full_bytes_per_token": topology["full_attention_layers"] * per_layer_bytes,
        "bounded_bytes_total": (topology["bounded_kv_layers"] * per_layer_bytes
                                * (topology["sliding_window"] or 0)),
    }


def _kv_budget_gb(util: float, pool_gb: float, weights_gb: float,
                  overhead_gb: float = 6.0, resident_gb: float = 0.0) -> float:
    """How many GB of KV cache actually fit at a given --gpu-memory-utilization.

    WHY `resident_gb` exists — do not delete it as padding: on this GB10 the driver reports
    MemFree, not MemAvailable, so every byte the page cache is holding is charged against
    `--gpu-memory-utilization` even though the kernel would evict it on demand. This was
    measured, not assumed: at an unchanged, hand-validated 0.55 on Qwen3.6, the engine had
    25.97 GiB of KV after dropping caches and only 0.19 GiB with ~25 GB still cached, and it
    refused to start while blaming max_model_len. A budget that assumes the whole pool share
    is free will therefore recommend a utilization that works on a freshly-booted box and
    fails on a working one.

    The result is clamped at 0.0: a budget smaller than the model is "no room", never a
    negative number a caller might add to something.

    Pure: scalars in, scalar out. Pool size, weight size and resident bytes are the caller's
    parameters — reading them here would make the whole phase unverifiable with vLLM down.
    """
    return max(0.0, pool_gb * util - resident_gb - weights_gb - overhead_gb)


def _derive_launch_spec(config: dict, weights_gb: float = 0.0, pool_gb: float = 121.0,
                        kv_dtype_bytes: int = 1, overhead_gb: float = 6.0,
                        util_margin: float = 0.04, resident_gb: float = 0.0,
                        requested_context: Optional[int] = None,
                        util_floor: float = 0.10, util_cap: float = 0.95) -> dict:
    """Turn one parsed config.json into the launch numbers vLLM should be started with.

    This is the public entry point of the derived-spec work: layer classification, KV bytes
    per token, the largest context that fits, and the `--gpu-memory-utilization` to request.

        recommended_util = (weights + KV + overhead + resident) / pool + margin

    calibrated against a number a human already measured by hand: qwen3-next-80b-a3b-nvfp4 was
    tuned to 0.55 on this box, and this arithmetic independently derives 0.54. That agreement
    is the evidence the whole approach is sound — if it ever breaks, the formula is wrong, not
    the hand-measured recipe.

    The margin (default 0.04) covers allocator fragmentation and the fact that the pool is
    unified with the OS, so the derived need is a floor rather than an exact requirement.
    The 6 GB default overhead covers CUDA context, captured graphs and activations.

    Every vendor-authored and caller-supplied number is treated as untrusted: non-numeric and
    negative scalars are coerced to 0.0 and NAMED in `warnings` rather than silently zeroed,
    and both divisions are guarded, so a malformed config yields a clamped answer with a trail
    instead of a traceback. `recommended_util` is clamped into [util_floor, util_cap] because
    its consumer hands it to a real launch: an absurd config must not be able to request more
    memory than the box has.

    Warnings collected by the topology resolver are propagated rather than dropped — an
    unrecognized layer type over-reserves memory, and that decision has to stay visible at the
    only layer a caller sees.

    Pure: a dict and scalars in, a dict out. No filesystem, network, kernel meminfo or vLLM.

    Returns the sixteen keys documented in the phase interface, notably `max_model_len` (never
    above the model's declared maximum nor above what fits), `kv_gb` at that length, and
    `recommended_util`.
    """
    cfg = config if isinstance(config, dict) else {}
    inner = cfg.get("text_config")
    t = inner if isinstance(inner, dict) and inner else cfg

    sizing = _kv_bytes_per_token(cfg, kv_dtype_bytes)
    topology = sizing["topology"]
    warnings: list[str] = list(topology.get("warnings") or [])

    def _scalar(value, field: str) -> float:
        """Coerce one caller-supplied GB figure, naming a bad value instead of hiding it."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            warnings.append(f"non-numeric {field}: {value!r} — treated as 0.0")
            return 0.0
        if number != number or number in (float("inf"), float("-inf")):
            warnings.append(f"non-finite {field}: {value!r} — treated as 0.0")
            return 0.0
        if number < 0:
            warnings.append(f"negative {field}: {number} — floored at 0.0")
            return 0.0
        return number

    weights_gb = _scalar(weights_gb, "weights_gb")
    pool_gb = _scalar(pool_gb, "pool_gb")
    overhead_gb = _scalar(overhead_gb, "overhead_gb")
    resident_gb = _scalar(resident_gb, "resident_gb")

    kv_rate = sizing["full_bytes_per_token"]      # a RATE: multiply by context
    bounded_total = sizing["bounded_bytes_total"]  # already a TOTAL: do not
    declared_max_context = _as_int(t.get("max_position_embeddings"))

    # ── Largest context that fits, at the most aggressive utilization allowed ──
    if pool_gb <= 0:
        warnings.append(f"invalid pool_gb: {pool_gb} — no budget can be sized")
        max_fitting_context = 0
    elif kv_rate <= 0:
        # Nothing grows with context: an all-stateless (or empty) model is limited by what it
        # was trained for, not by memory.
        warnings.append("no full-attention layers; context is not KV-bounded")
        max_fitting_context = declared_max_context
    else:
        budget_bytes = _kv_budget_gb(util_cap, pool_gb, weights_gb, overhead_gb,
                                     resident_gb) * 1e9
        max_fitting_context = max(0, int((budget_bytes - bounded_total) // kv_rate))

    max_model_len = min(declared_max_context or max_fitting_context, max_fitting_context)

    if declared_max_context and max_fitting_context < declared_max_context:
        warnings.append(
            f"budget-limited context: {max_fitting_context} tokens fit, "
            f"model declares {declared_max_context}"
        )

    if requested_context is not None:
        requested = max(0, _as_int(requested_context))
        if requested > max_model_len:
            warnings.append(
                f"requested context {requested} exceeds the usable {max_model_len}"
            )
        elif requested > 0:
            max_model_len = requested

    kv_gb = (kv_rate * max_model_len + bounded_total) / 1e9

    if pool_gb <= 0:
        recommended_util = util_cap
    else:
        raw = (weights_gb + kv_gb + overhead_gb + resident_gb) / pool_gb + util_margin
        recommended_util = round(min(util_cap, max(util_floor, round(raw, 2))), 2)

    return {
        "topology": topology,
        "num_hidden_layers": topology["num_hidden_layers"],
        "full_attention_layers": topology["full_attention_layers"],
        "bounded_kv_layers": topology["bounded_kv_layers"],
        "stateless_layers": topology["stateless_layers"],
        "source_field": topology["source_field"],
        "kv_bytes_per_token": kv_rate,
        "bounded_bytes_total": bounded_total,
        "declared_max_context": declared_max_context,
        "max_fitting_context": max_fitting_context,
        "max_model_len": max_model_len,
        "kv_gb": kv_gb,
        "weights_gb": weights_gb,
        "overhead_gb": overhead_gb,
        "recommended_util": recommended_util,
        "warnings": warnings,
    }


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
    if APP_HOST not in ("127.0.0.1", "::1", "localhost") and not _API_KEY_HASH:
        if not os.environ.get("MODEL_MANAGER_ALLOW_UNAUTH"):
            msg = (f"SECURITY WARNING: binding to {APP_HOST} with no API key set — "
                   "the management UI is open to anyone on this network. "
                   "Set an API key in Settings, or set MODEL_MANAGER_ALLOW_UNAUTH=1 to suppress this warning. "
                   "Waiting 10 seconds before accepting connections...")
            _logger.warning(msg)
            print(msg, flush=True)
            await asyncio.sleep(10)
        else:
            msg = "SECURITY NOTICE: no API key set and MODEL_MANAGER_ALLOW_UNAUTH=1 — open access acknowledged."
            _logger.warning(msg)
            print(msg, flush=True)
    alert_task = asyncio.create_task(_alert_loop())
    _logger.info("App started on port %s", APP_PORT)
    yield
    _logger.info("App shutting down")
    alert_task.cancel()
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
    check_coros += [service_ok(OLLAMA_BASE, "/api/tags"), service_ok(LITELLM_BASE, "/v1/models")]
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
    # Unified-memory snapshot for the header gauge — cheap /proc/meminfo read.
    status["memory"] = _meminfo_snapshot()
    return status


def _meminfo_snapshot() -> dict:
    vals = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                vals[k] = int(v.split()[0])
    except Exception as e:
        return {"error": str(e)}

    total = vals.get("MemTotal", 0) / 1024 / 1024
    available = vals.get("MemAvailable", 0) / 1024 / 1024
    used = max(total - available, 0)
    return {
        "total_gb": round(total, 1),
        "available_gb": round(available, 1),
        "used_gb": round(used, 1),
        "used_pct": round((used / total) * 100, 1) if total else 0,
        "swap_total_gb": round(vals.get("SwapTotal", 0) / 1024 / 1024, 1),
        "swap_free_gb": round(vals.get("SwapFree", 0) / 1024 / 1024, 1),
    }


async def _nvidia_compute_apps() -> dict:
    result = await _run(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
        timeout=5,
    )
    if result.returncode != 0:
        return {"ok": False, "apps": [], "error": (result.stderr or result.stdout).strip()}

    apps = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            used_mib = int(parts[2])
        except ValueError:
            used_mib = None
        cmdline = ""
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_text(errors="ignore").replace("\0", " ").strip()
        except Exception:
            pass
        apps.append({
            "pid": pid,
            "process": parts[1],
            "used_mib": used_mib,
            "used_gb": round(used_mib / 1024, 1) if used_mib is not None else None,
            "cmd": cmdline[:280],
        })
    apps.sort(key=lambda x: x.get("used_mib") or 0, reverse=True)
    return {"ok": True, "apps": apps, "total_mib": sum(a.get("used_mib") or 0 for a in apps)}


async def _docker_model_containers() -> dict:
    result = await _run("docker", "ps", "--format", "{{json .}}", timeout=5)
    if result.returncode != 0:
        return {"ok": False, "containers": [], "error": (result.stderr or result.stdout).strip()}
    keywords = ("vllm", "ollama", "llama", "sglang", "localai", "local-ai", "comfy", "litellm", "triton", "tgi")
    containers = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        hay = " ".join(str(row.get(k, "")) for k in ("Names", "Image", "Command", "Ports")).lower()
        if not any(k in hay for k in keywords):
            continue
        containers.append({
            "id": row.get("ID", ""),
            "name": row.get("Names", ""),
            "image": row.get("Image", ""),
            "status": row.get("Status", ""),
            "ports": row.get("Ports", ""),
            "command": row.get("Command", ""),
        })
    return {"ok": True, "containers": containers}


def _parse_ollama_ps(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines()[1:]:
        if not line.strip():
            continue
        parts = _re.split(r"\s{2,}", line.strip())
        if len(parts) >= 6:
            rows.append({
                "name": parts[0],
                "id": parts[1],
                "size": parts[2],
                "processor": parts[3],
                "context": parts[4],
                "until": parts[5],
            })
        elif parts:
            rows.append({"name": parts[0], "raw": line.strip()})
    return rows


async def _ollama_warm_models() -> dict:
    result = await _run("ollama", "ps", timeout=5)
    if result.returncode != 0:
        return {"ok": False, "models": [], "raw": "", "error": (result.stderr or result.stdout).strip()}
    return {"ok": True, "models": _parse_ollama_ps(result.stdout), "raw": result.stdout}


async def _k8s_vllm_deployments() -> dict:
    result = await _run("kubectl", "get", "deploy", "-n", "llm-inference", "-o", "json", timeout=8)
    if result.returncode != 0:
        return {"ok": False, "deployments": [], "error": (result.stderr or result.stdout).strip()}
    try:
        data = json.loads(result.stdout)
    except Exception as e:
        return {"ok": False, "deployments": [], "error": str(e)}
    deployments = []
    for item in data.get("items", []):
        spec = item.get("spec", {})
        status = item.get("status", {})
        tmpl = spec.get("template", {}).get("spec", {})
        containers = tmpl.get("containers", [])
        images = [c.get("image", "") for c in containers]
        name = item.get("metadata", {}).get("name", "")
        if name.startswith("prometheus-"):
            continue
        hay = (name + " " + " ".join(images)).lower()
        if "vllm" not in hay:
            continue
        deployments.append({
            "name": name,
            "replicas": spec.get("replicas", 0),
            "available": status.get("availableReplicas", 0),
            "ready": status.get("readyReplicas", 0),
            "updated": status.get("updatedReplicas", 0),
            "images": images,
        })
    return {"ok": True, "deployments": deployments}


def _identify_active_profile(status: dict, profiles: list[dict]) -> Optional[str]:
    served = status.get("model") or ""
    if not served:
        return None
    served_l = served.lower()
    for p in profiles:
        try:
            content = Path(os.path.expanduser(p.get("script", ""))).read_text(errors="ignore")
        except Exception:
            content = ""
        if served in content:
            return p.get("id")
    for p in profiles:
        pid = p.get("id", "").lower().removeprefix("start_")
        pname = p.get("name", "").lower()
        if pid and (pid in served_l or served_l in pid or served_l in pname):
            return p.get("id")
    return None


@app.get("/api/warm-models")
async def get_warm_models():
    """Resource-oriented view of currently loaded/warm model runtimes."""
    vllm_profiles = _scan_profiles("vllm")
    vllm_status = await _engine_status(
        _engine_bases["vllm"], _ENGINES["vllm"].get("docker_filter", "vllm"),
        _ENGINES["vllm"].get("health_path", "/health"), _ENGINES["vllm"].get("models_path"))
    active_profile = _identify_active_profile(vllm_status, vllm_profiles)
    nvidia, docker, ollama, k8s = await asyncio.gather(
        _nvidia_compute_apps(),
        _docker_model_containers(),
        _ollama_warm_models(),
        _k8s_vllm_deployments(),
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "memory": _meminfo_snapshot(),
        "nvidia": nvidia,
        "docker": docker,
        "ollama": ollama,
        "vllm": {
            "status": vllm_status,
            "profiles": vllm_profiles,
            "active_profile": active_profile,
        },
        "kubernetes": k8s,
    }

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
        # Optional deep-link target for the header memory gauge (Settings-free:
        # set app.grafana_url in config.json; empty string = plain gauge).
        "grafana_url": _app_config.get("app", {}).get("grafana_url", ""),
    }

# ── Dashboards ────────────────────────────────────────────────────────────────
# Live inventory of the web UIs running on this box, consumed by the Dashboards
# tab. Unauthenticated by design, like /api/status.
#
# Discovery is two cheap listings — `ss` for host TCP listeners and `kubectl`
# for NodePort services — followed by one HTTP GET per candidate. The GET is
# what does the real filtering: a port earns a card only by answering with an
# HTML page. That separates dashboards from the many API/metrics ports on this
# host (vLLM, Ollama, node-exporter, traefik) without maintaining a port list,
# and the page's <title> supplies a name for anything the config doesn't cover.
# Ports that answer non-HTML are kept as kind="api" so the UI can offer them
# behind a toggle rather than hiding a running service outright.

_TITLE_RE = _re.compile(r"<title[^>]*>(.*?)</title>", _re.I | _re.S)

# Listeners that are never a dashboard and would waste a probe. Everything else
# has to prove itself by responding — this stays short on purpose.
_DISCOVERY_SKIP_PORTS = {22, 53, 111, 631, 6443, 10250}

# Titles too generic to name a card by — the framework's default, not the app's.
# When one of these comes back, fall through to the k8s service / process name.
_GENERIC_TITLES = {"streamlit", "dashboard", "home", "index", "login", "sign in",
                   "react app", "vite app", "document", "untitled"}

_SITES_CACHE: dict = {"at": 0.0, "payload": None}
_SITES_LOCK = asyncio.Lock()


def _resolve_site_url(site: dict, request_host: str) -> str:
    """Resolve one sites entry to a full URL — verbatim "url" wins, otherwise
    scheme://base:port where base is app.sites_base → app.host → request host."""
    if site.get("url"):
        return site["url"]
    host = _SITES_BASE or APP_HOST
    if not host or host == "0.0.0.0":
        host = request_host or "127.0.0.1"
    return f"{site.get('scheme', 'http')}://{host}:{site.get('port')}"


def _is_loopback(addr: str) -> bool:
    a = addr.split("%")[0]
    return a.startswith("127.") or a == "::1"


def _is_wildcard(addr: str) -> bool:
    return addr in ("0.0.0.0", "::", "*", "")


def _parse_ss_listeners(text: str) -> list[dict]:
    """Parse `ss -tlnpH` rows into {addr, port, proc}. Local address is field 4;
    IPv6 arrives bracketed ([::]:8123) and wildcard binds as *:9090, both of
    which normalise to a wildcard addr. Process names are only visible for our
    own UID — root-owned listeners legitimately come back with proc="".
    """
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        addr, _, port = parts[3].rpartition(":")
        if not port.isdigit():
            continue
        addr = addr.strip("[]")
        m = _re.search(r'\(\("([^"]+)"', line)
        rows.append({"addr": "0.0.0.0" if _is_wildcard(addr) else addr,
                     "port": int(port), "proc": m.group(1) if m else ""})
    return rows


def _parse_nodeport_services(payload: dict) -> list[dict]:
    """Flatten `kubectl get svc -A -o json` down to NodePort exposures."""
    out = []
    for item in payload.get("items", []) or []:
        meta = item.get("metadata", {})
        for port in item.get("spec", {}).get("ports", []) or []:
            if port.get("nodePort"):
                out.append({"port": int(port["nodePort"]),
                            "svc": meta.get("name", ""),
                            "namespace": meta.get("namespace", "")})
    return out


async def _discover_listeners() -> dict:
    r = await _run("ss", "-tlnpH", timeout=5)
    if r.returncode != 0:
        return {"ok": False, "rows": [], "error": (r.stderr or r.stdout).strip() or "ss failed"}
    return {"ok": True, "rows": _parse_ss_listeners(r.stdout), "error": ""}


async def _discover_nodeports() -> dict:
    r = await _run("kubectl", "get", "svc", "-A", "-o", "json", timeout=8)
    if r.returncode != 0:
        return {"ok": False, "rows": [], "error": (r.stderr or r.stdout).strip() or "kubectl failed"}
    try:
        return {"ok": True, "rows": _parse_nodeport_services(json.loads(r.stdout)), "error": ""}
    except Exception as e:
        return {"ok": False, "rows": [], "error": str(e)}


def _build_candidates(listeners: list[dict], nodeports: list[dict]) -> dict:
    """Collapse both discovery sources into one candidate per port.

    probe_host is where *we* reach it (a wildcard bind is probed on loopback);
    bind_addr is what the browser must use — a service bound only to the tailnet
    address, like this app itself, is unreachable at 127.0.0.1 and its card has
    to link to that address rather than to the configured site base.
    """
    excluded = _DISCOVERY_SKIP_PORTS | set(_SITES_DISCOVERY.get("exclude_ports") or [])
    forced = set(_SITES_DISCOVERY.get("include_ports") or [])
    include_loopback = bool(_SITES_DISCOVERY.get("include_loopback"))
    cands: dict[int, dict] = {}
    for row in listeners:
        port, addr = row["port"], row["addr"]
        if port in excluded and port not in forced:
            continue
        if _is_loopback(addr) and not include_loopback and port not in forced:
            continue
        c = cands.setdefault(port, {"port": port, "source": "host", "proc": "",
                                    "svc": "", "namespace": "", "bind_addr": ""})
        c["proc"] = c["proc"] or row["proc"]
        # A wildcard bind is the more permissive one — prefer it over an
        # address-specific row for the same port.
        if _is_wildcard(addr):
            c["bind_addr"] = ""
        elif not c["bind_addr"]:
            c["bind_addr"] = addr
    for row in nodeports:
        port = row["port"]
        if port in excluded and port not in forced:
            continue
        c = cands.setdefault(port, {"port": port, "source": "k8s", "proc": "",
                                    "svc": "", "namespace": "", "bind_addr": ""})
        c["source"] = "k8s"
        c["svc"], c["namespace"] = row["svc"], row["namespace"]
    for c in cands.values():
        c["probe_host"] = c["bind_addr"] or "127.0.0.1"
    return cands


async def _probe_site(url: str, timeout: float) -> dict:
    """One classified GET. "ui" = answered with HTML, i.e. a page a human can
    open; "api" = answered with JSON/plain text; "down" = nothing there.
    Redirects are followed (Grafana lands on /login) and auth walls still count
    as a UI — Headlamp and Hermes both gate their pages behind one.
    """
    if _http is None:
        return {"kind": "down", "status": 0, "title": "", "url": url}
    try:
        r = await _http.get(url, timeout=timeout, follow_redirects=True)
    except Exception:
        # A plain-HTTP GET against a TLS port fails at the protocol level; the
        # one retry is what keeps HTTPS-only UIs from reading as dead.
        if url.startswith("http://"):
            try:
                r = await _http.get("https://" + url[7:], timeout=timeout, follow_redirects=True)
                url = "https://" + url[7:]
            except Exception:
                return {"kind": "down", "status": 0, "title": "", "url": url}
        else:
            return {"kind": "down", "status": 0, "title": "", "url": url}
    if r.status_code >= 400 and r.status_code not in (401, 403):
        return {"kind": "down", "status": r.status_code, "title": "", "url": url}
    ctype = r.headers.get("content-type", "").lower()
    if "html" not in ctype:
        return {"kind": "api", "status": r.status_code, "title": "", "url": url}
    title = ""
    try:
        m = _TITLE_RE.search(r.text[:8192])
        if m:
            title = _re.sub(r"\s+", " ", m.group(1)).strip()[:60]
    except Exception:
        pass
    return {"kind": "ui", "status": r.status_code, "title": title, "url": url}


def _discovered_name(cand: dict, title: str) -> str:
    """Name a card from the best evidence available: the page's own <title>,
    else the k8s service, else the listening process, else the bare port."""
    clean = title.strip()
    if clean and clean.lower() not in _GENERIC_TITLES \
            and not clean.lower().startswith("directory listing"):
        return clean
    if cand.get("svc"):
        return _re.sub(r"-(svc|service)$", "", cand["svc"])
    if cand.get("proc"):
        return cand["proc"]
    return f"Port {cand['port']}"


async def _discover_sites(request_host: str) -> dict:
    """Full inventory: discovered UIs merged with the config.json overlay.

    Overlay entries matched by port win on name/desc/group (curation beats a
    <title>); config entries with a verbatim url, or whose port is not listening,
    are still listed so a known-but-down dashboard stays visible instead of
    silently vanishing.
    """
    overlay = {}
    pinned = []
    for s in _SITES:
        if not isinstance(s, dict) or not s.get("name"):
            continue
        if s.get("port"):
            overlay[int(s["port"])] = s
        elif s.get("url"):
            pinned.append(s)

    listeners = {"ok": False, "rows": [], "error": "discovery disabled"}
    nodeports = {"ok": False, "rows": [], "error": "disabled"}
    if _SITES_DISCOVERY.get("enabled", True):
        listeners = await _discover_listeners()
        if _SITES_DISCOVERY.get("kubernetes", True):
            nodeports = await _discover_nodeports()

    cands = _build_candidates(listeners["rows"], nodeports["rows"])
    base_host = _SITES_BASE or APP_HOST
    if not base_host or base_host == "0.0.0.0":
        base_host = request_host or "127.0.0.1"
    timeout = float(_SITES_DISCOVERY.get("probe_timeout_s", 2.0))

    ordered = sorted(cands.values(), key=lambda c: c["port"])
    probes = await asyncio.gather(*(
        _probe_site(f"http://{c['probe_host']}:{c['port']}/", timeout) for c in ordered
    ))

    sites, seen_ports = [], set()
    for cand, probe in zip(ordered, probes):
        if probe["kind"] == "down":
            continue
        port = cand["port"]
        seen_ports.add(port)
        conf = overlay.get(port, {})
        link_host = cand["bind_addr"] or base_host
        scheme = "https" if probe["url"].startswith("https://") else conf.get("scheme", "http")
        detail = (f"{cand['namespace']}/{cand['svc']}" if cand.get("svc")
                  else (cand.get("proc") or ""))
        sites.append({
            "name": conf.get("name") or _discovered_name(cand, probe["title"]),
            "desc": conf.get("desc", "") or detail,
            "group": conf.get("group", "") or ("Kubernetes" if cand["source"] == "k8s" else "Host"),
            "url": conf.get("url") or f"{scheme}://{link_host}:{port}",
            "reachable": True,
            "port": port,
            "kind": probe["kind"],
            "source": "config" if conf else cand["source"],
            "detail": detail,
        })

    # Tallied before the overlay leftovers are appended — these numbers describe
    # the sweep, and a pinned remote UI was never part of it.
    counts = {"ui": sum(1 for s in sites if s["kind"] == "ui"),
              "api": sum(1 for s in sites if s["kind"] == "api")}

    # Configured entries discovery couldn't confirm — a remote pin, or a
    # dashboard that is currently down. Probed individually so the dot is honest.
    leftovers = pinned + [s for p, s in overlay.items() if p not in seen_ports]
    if leftovers:
        urls = [_resolve_site_url(s, request_host) for s in leftovers]
        checks = await asyncio.gather(*(_site_reachable(u) for u in urls))
        for s, url, ok in zip(leftovers, urls, checks):
            sites.append({
                "name": s["name"], "desc": s.get("desc", ""),
                "group": s.get("group", "") or "Other", "url": url,
                "reachable": ok, "port": s.get("port"), "kind": "ui",
                "source": "config", "detail": "",
            })

    return {
        "sites": sites,
        "discovery": {
            "enabled": bool(_SITES_DISCOVERY.get("enabled", True)),
            "host": {"ok": listeners["ok"], "error": listeners["error"]},
            "kubernetes": {"ok": nodeports["ok"], "error": nodeports["error"]},
            "candidates": len(cands),
            **counts,
        },
    }


async def _site_reachable(url: str) -> bool:
    """Best-effort probe — mirrors service_ok, but with a shorter timeout so a
    page of dead links can't slow the endpoint. Auth walls still count as up."""
    if _http is None:
        return False
    try:
        r = await _http.get(url, timeout=1.5)
        return r.status_code < 400 or r.status_code in (401, 403)
    except Exception:
        return False


@app.get("/api/sites")
async def get_sites(request: Request, refresh: int = 0):
    """Cached because a full sweep is ~40 HTTP probes; switching tabs shouldn't
    pay for that. The lock collapses concurrent callers onto one sweep."""
    ttl = float(_SITES_DISCOVERY.get("ttl_s", 30))
    async with _SITES_LOCK:
        age = _time.monotonic() - _SITES_CACHE["at"]
        if refresh or _SITES_CACHE["payload"] is None or age > ttl:
            _SITES_CACHE["payload"] = await _discover_sites(request.url.hostname or "")
            _SITES_CACHE["at"] = _time.monotonic()
            age = 0.0
    payload = dict(_SITES_CACHE["payload"])
    payload["cached_age_s"] = round(age, 1)
    return payload


# ── Recommendations ─────────────────────────────────────────────────────────
# Layer-2 recommender: diffs the curated Spark-specific KB (recommendations.json)
# against installed vLLM profiles + live memory state. Each KB entry carries a
# `match` rule; only fired recommendations are returned, ranked by severity.

_RECOMMENDATIONS_FILE = _APP_DIR / "recommendations.json"
_MODEL_CAPABILITIES_FILE = _APP_DIR / "model_capabilities.json"
_REC_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _load_recommendations() -> dict:
    try:
        return json.loads(_RECOMMENDATIONS_FILE.read_text())
    except Exception as e:
        _logger.warning("Could not load recommendations.json: %s", e)
        return {"meta": {}, "recommendations": []}


def _load_model_capabilities() -> dict:
    try:
        return json.loads(_MODEL_CAPABILITIES_FILE.read_text())
    except Exception as e:
        _logger.warning("Could not load model_capabilities.json: %s", e)
        return {"meta": {}, "models": []}


_PARSER_NAME_RE = _re.compile(r"^[a-z0-9_.-]{1,64}$")
_PARSER_EMISSION_CONFIDENCE = {
    "recipe-proven", "template-identical-to-recipe-proven",
}


def _capability_entry(info: dict) -> dict | None:
    capabilities = _load_model_capabilities()
    rows = capabilities.get("models", []) if isinstance(capabilities, dict) else []
    if not isinstance(rows, list):
        return None

    name = info.get("name")
    name_key = name.casefold() if isinstance(name, str) else ""
    for row in rows:
        matches = row.get("matches", []) if isinstance(row, dict) else []
        if (name_key and isinstance(matches, list)
                and any(isinstance(match, str) and match.casefold() == name_key
                        for match in matches)):
            return row

    architectures = info.get("architectures", [])
    architecture = (architectures[0]
                    if isinstance(architectures, list) and architectures else None)
    architecture_key = architecture.casefold() if isinstance(architecture, str) else ""
    for row in rows:
        matches = row.get("matches", []) if isinstance(row, dict) else []
        if (architecture_key and isinstance(matches, list)
                and any(isinstance(match, str) and match.casefold() == architecture_key
                        for match in matches)):
            return row
    return None


def _capability_emission_details(
        entry: dict | None) -> tuple[str | None, str | None, tuple[str, ...]]:
    if not isinstance(entry, dict):
        return None, None, ()

    warnings = []
    values = []
    for field in ("tool_call_parser", "reasoning_parser"):
        value = entry.get(field)
        if value is not None and (not isinstance(value, str)
                                  or not _PARSER_NAME_RE.match(value)):
            warnings.append(
                f"capability map {field} is invalid; expected "
                "lowercase letters, digits, dot, underscore or hyphen (max 64)")
            value = None
        values.append(value)

    explicit_emit = entry.get("emit")
    emittable = (explicit_emit if isinstance(explicit_emit, bool)
                 else entry.get("confidence") in _PARSER_EMISSION_CONFIDENCE)
    if not emittable:
        return None, None, tuple(warnings)
    return values[0], values[1], tuple(warnings)


def _capability_emission(entry: dict | None) -> tuple[str | None, str | None]:
    tool_parser, reasoning_parser, _warnings = _capability_emission_details(entry)
    return tool_parser, reasoning_parser


def _profile_script_text(profile: dict) -> str:
    try:
        return Path(os.path.expanduser(profile.get("script", ""))).read_text()
    except Exception:
        return ""


def _eval_profile_match(match: dict, text: str) -> Optional[str]:
    """Return a per-profile 'why it fired' detail, or None if it doesn't fire."""
    mt = match.get("type")
    if mt == "script_contains_any":
        for p in match.get("patterns", []):
            if p in text:
                return f"matches '{p}'"
        return None
    if mt == "script_missing_flag":
        pats = match.get("patterns", [])
        if pats and not any(p in text for p in pats):
            return f"no {pats[0]}"
        return None
    if mt == "moe_backend_not_marlin":
        m = _re.search(r"--moe-backend\s+(\S+)", text)
        if not m:
            return None  # not a MoE profile / no explicit backend
        if any(sig in text for sig in match.get("marlin_signals", [])) or m.group(1) == "marlin":
            return None
        return f"uses --moe-backend {m.group(1)} (not marlin)"
    if mt == "script_low_util":
        m = _re.search(r"--gpu-memory-utilization\s+([0-9.]+)", text)
        if not m:
            return None
        util = float(m.group(1))
        thr = match.get("threshold", 0.6)
        return f"gpu-memory-utilization {util:g} < {thr:g}" if util < thr else None
    if mt == "script_high_num_seqs":
        m = _re.search(r"--max-num-seqs\s+(\d+)", text)
        if not m:
            return None
        seqs = int(m.group(1))
        thr = match.get("threshold", 16)
        return f"max-num-seqs {seqs} > {thr}" if seqs > thr else None
    if mt == "image_tag_suspect":
        for p in match.get("patterns", []):
            if p in text:
                return f"image references '{p}'"
        return None
    return None


@app.get("/api/recommendations")
async def get_recommendations():
    kb = _load_recommendations()
    profiles = _scan_profiles("vllm")
    scripts = {p["id"]: _profile_script_text(p) for p in profiles}
    haystack = " ".join(
        [t.lower() for t in scripts.values()]
        + [f"{p.get('id','')} {p.get('name','')}".lower() for p in profiles])

    fired = []
    for rec in kb.get("recommendations", []):
        match = rec.get("match", {})
        base = {k: rec[k] for k in
                ("id", "kind", "severity", "title", "summary", "action",
                 "sources", "confidence", "download", "apply") if k in rec}
        if match.get("type") == "model_absent":
            signals = match.get("signals", [])
            if not any(s.lower() in haystack for s in signals):
                fired.append({**base, "scope": "global",
                              "fired_for": [],
                              "detail": f"no profile serves {' / '.join(signals)}"})
            continue
        hits = [{"profile": pid, "detail": detail}
                for pid, text in scripts.items()
                if (detail := _eval_profile_match(match, text))]
        if hits:
            fired.append({**base, "scope": "profile", "fired_for": hits})

    fired.sort(key=lambda r: _REC_SEVERITY_ORDER.get(r.get("severity"), 9))
    mem = _meminfo_snapshot()
    return {
        "meta": kb.get("meta", {}),
        "state": {"available_gb": mem.get("available_gb"),
                  "total_gb": mem.get("total_gb")},
        "profiles_checked": len(profiles),
        "recommendations": fired,
    }


# Research-refresh (KB layer 3). The job fetches curated sources, runs a
# synthesis engine (Codex, ~1-2 min), and writes PROPOSED updates for review.
# It never touches the live KB — approval stays a deliberate CLI step
# (research_refresh.py --apply, then formalize each manual match rule). The UI
# only *runs research* and *shows the proposals*; it intentionally cannot apply.
_PROPOSED_FILE = _APP_DIR / "recommendations.proposed.json"
_refresh_lock = asyncio.Lock()


@app.get("/api/recommendations/proposed")
async def get_proposed_recommendations():
    """Return the last proposals (recommendations.proposed.json), if any."""
    try:
        data = json.loads(_PROPOSED_FILE.read_text())
    except Exception:
        return {"proposed": [], "summary": "", "exists": False}
    return {**data, "exists": True}


@app.post("/api/recommendations/refresh", dependencies=[Depends(verify_auth)])
async def refresh_recommendations():
    """Run the research-refresh job and return the proposed updates for review."""
    if _refresh_lock.locked():
        raise HTTPException(409, "A refresh is already running")
    async with _refresh_lock:
        import research_refresh
        _logger.info("Research-refresh requested (engine=%s)", research_refresh.ENGINE)
        try:
            result = await asyncio.to_thread(research_refresh.research)
        except Exception as e:
            _logger.error("Research-refresh failed: %s", e)
            raise HTTPException(500, f"Refresh failed: {e}")
        props = result.get("proposed", [])
        _logger.info("Research-refresh produced %d proposal(s)", len(props))
        return {"ok": True, "engine": research_refresh.ENGINE,
                "summary": result.get("summary", ""), "proposed": props}


def _apply_rec_edit(spec: dict, text: str) -> tuple[str, bool, str]:
    """Apply a recommendation's `apply` spec to a profile script's text.

    Returns (new_text, changed, note). Edits anchor on the `--model` line
    (present in every vLLM start script and always continued with a trailing
    backslash), so inserts stay inside the docker-run argument block.
    """
    typ = spec.get("type")
    trailing_nl = "\n" if text.endswith("\n") else ""
    if typ == "add_flag":
        flag = str(spec.get("flag", "")).strip()
        if not flag:
            raise HTTPException(400, "apply.add_flag requires 'flag'")
        token = flag.split("=", 1)[0]
        if token in text:
            return text, False, f"{token} already present"
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if _re.match(r"\s*--model\b", ln):
                indent = ln[:len(ln) - len(ln.lstrip())]
                lines.insert(i + 1, f"{indent}{flag} \\")
                return "\n".join(lines) + trailing_nl, True, f"added {flag}"
        raise HTTPException(422, "No --model line to anchor the flag")
    if typ == "remove_flag":
        flag = str(spec.get("flag", "")).strip()
        if not flag or flag not in text:
            return text, False, f"{flag or 'flag'} not present"
        out = []
        for ln in text.splitlines():
            if ln.strip() in (flag, flag + " \\"):
                continue  # flag owned the whole line — drop it
            if flag in ln:
                ln = _re.sub(r"\s*" + _re.escape(flag) + r"\b", "", ln)
            out.append(ln)
        return "\n".join(out) + trailing_nl, True, f"removed {flag}"
    if typ == "set_flag":
        flag = str(spec.get("flag", "")).strip()
        value = str(spec.get("value", "")).strip()
        if not flag or not value:
            raise HTTPException(400, "apply.set_flag requires 'flag' and 'value'")
        pat = _re.compile(_re.escape(flag) + r"\s+(\S+)")
        m = pat.search(text)
        if not m:
            raise HTTPException(422, f"{flag} not found in profile")
        if m.group(1) == value:
            return text, False, f"{flag} already {value}"
        return pat.sub(f"{flag} {value}", text, count=1), True, f"{flag} -> {value}"
    raise HTTPException(400, f"Unknown apply.type '{typ}'")


@app.post("/api/recommendations/apply", dependencies=[Depends(verify_auth)])
async def apply_recommendation(req: ApplyRecRequest):
    """Apply a config/tuning rec's edit to a flagged profile script.

    Two-phase: without `confirm` it returns a unified diff for preview; with
    `confirm` it writes the script atomically (tmp + os.replace, mode 0755)."""
    kb = _load_recommendations()
    rec = next((r for r in kb.get("recommendations", []) if r.get("id") == req.id), None)
    if not rec:
        raise HTTPException(404, f"No recommendation '{req.id}'")
    spec = rec.get("apply")
    if not spec:
        raise HTTPException(400, f"Recommendation '{req.id}' has no apply action")
    prof = next((p for p in _scan_profiles("vllm") if p["id"] == req.profile), None)
    if not prof:
        raise HTTPException(404, f"No vLLM profile '{req.profile}'")
    path = Path(prof["script"])
    old = path.read_text()
    new, changed, note = _apply_rec_edit(spec, old)
    if not changed:
        return {"changed": False, "note": note, "profile": req.profile}
    diff = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=path.name, tofile=path.name + " (proposed)"))
    if not req.confirm:
        return {"changed": True, "confirm_required": True,
                "note": note, "diff": diff, "profile": req.profile}
    tmp = path.with_suffix(path.suffix + ".apply.tmp")
    tmp.write_text(new)
    os.chmod(tmp, 0o755)
    os.replace(tmp, path)
    _logger.info("Applied rec %s to %s (%s)", req.id, path.name, note)
    return {"changed": True, "applied": True, "note": note,
            "diff": diff, "profile": req.profile}


# ── Alerting ──────────────────────────────────────────────────────────────────
# Threshold alerts, salvaged from Hermes SysEng. State comes from this app's
# own health/memory helpers — no external processes. GPU thresholds are
# deliberately skipped (too noisy on the GB10 unified pool).
#
# Configured via the optional "alerts" section of config.json (documented in
# config.example.json). Every key is optional — a missing/empty section keeps
# the previous hardcoded/env-driven behavior exactly.

_alerts_cfg: dict = _app_config.get("alerts") or {}

_ALERT_ENABLED = bool(_alerts_cfg.get("enabled", True))
_ALERT_THRESHOLDS = {
    "memory_percent": 90,    # unified-pool usage — the only trustworthy VRAM signal here
    "endpoint_failures": 2,  # core serving endpoints unhealthy at once
}
_ALERT_THRESHOLDS.update(_alerts_cfg.get("thresholds") or {})
# Only the core serving path — other registered engines are usually off by design.
_ALERT_ENDPOINTS = tuple(_alerts_cfg.get("endpoints") or ("vllm", "litellm"))
# Config wins; env var is the migration-era fallback (systemd EnvironmentFile).
_ALERT_INTERVAL_S = int(_alerts_cfg.get("interval_s")
                        or os.environ.get("ALERT_CHECK_INTERVAL", "300"))
_ALERT_COOLDOWN_S = int(_alerts_cfg.get("cooldown_s") or 1800)

# Cooldown state — wall-clock epoch timestamps persisted across restarts so a
# restart mid-incident doesn't re-fire every alert. Gitignored, best-effort I/O:
# a corrupt or missing state file must never break alerting.
_ALERT_STATE_FILE = _APP_DIR / "alert_state.json"


def _load_alert_state() -> dict[str, float]:
    try:
        data = json.loads(_ALERT_STATE_FILE.read_text())
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


def _save_alert_state() -> None:
    try:
        _ALERT_STATE_FILE.write_text(json.dumps(_last_alert_sent))
    except Exception as e:
        _logger.warning("Failed to persist alert state: %s", e)


_last_alert_sent: dict[str, float] = _load_alert_state()

# Channel configs — discord is on by default (previous behavior); extra
# channels are opt-in via config.json alerts.channels.
_ALERT_CHANNELS: dict[str, dict] = {"discord": {"enabled": True}}
for _ch_name, _ch_cfg in (_alerts_cfg.get("channels") or {}).items():
    _ALERT_CHANNELS[_ch_name] = {**_ALERT_CHANNELS.get(_ch_name, {}), **(_ch_cfg or {})}


def _resolve_discord_webhook(cfg: dict) -> str:
    """Webhook resolution order: config.json → DISCORD_WEBHOOK_URL env fallback."""
    return cfg.get("webhook_url") or os.environ.get("DISCORD_WEBHOOK_URL", "")


def _notify_discord(alert: dict, cfg: dict) -> bool:
    color = 0xFF0000 if alert.get("severity") == "critical" else 0xFFA500
    title = f"{alert['type'].upper()} - {alert.get('severity', 'warning').upper()}"
    return send_discord_alert(_resolve_discord_webhook(cfg), title, alert["message"], color)


def _notify_log(alert: dict, cfg: dict) -> bool:
    """Log-only channel — always available, useful when no webhook is set."""
    _logger.warning("ALERT [%s] %s: %s", alert.get("severity", "warning"),
                    alert["type"], alert["message"])
    return True


# Notifier registry — adding a channel (email, ntfy, …) is one entry here plus
# an optional readiness predicate below. Signature: (alert, channel_cfg) -> bool.
_ALERT_NOTIFIERS: dict = {
    "discord": _notify_discord,
    "log": _notify_log,
}

# Channels that need external config before they can deliver. Missing entry
# means "always ready" (e.g. the log channel).
_ALERT_CHANNEL_READY: dict = {
    "discord": lambda cfg: bool(_resolve_discord_webhook(cfg)),
}


def _configured_alert_channels() -> dict[str, dict]:
    """Enabled channels whose notifier is registered and ready to deliver."""
    out: dict[str, dict] = {}
    for name, cfg in _ALERT_CHANNELS.items():
        if name not in _ALERT_NOTIFIERS or not cfg.get("enabled", True):
            continue
        if not _ALERT_CHANNEL_READY.get(name, lambda c: True)(cfg):
            continue
        out[name] = cfg
    return out


async def _alert_endpoint_health() -> dict[str, bool]:
    """Health of the endpoints that matter for alerting, keyed by service name."""
    keys, coros = [], []
    for key in _ALERT_ENDPOINTS:
        if key in _ENGINES:
            keys.append(key)
            coros.append(service_ok(_engine_bases[key], _ENGINES[key].get("health_path", "/health")))
        elif key == "litellm":
            keys.append(key)
            coros.append(service_ok(LITELLM_BASE, "/v1/models"))
        elif key == "ollama":
            keys.append(key)
            coros.append(service_ok(OLLAMA_BASE, "/api/tags"))
    results = await asyncio.gather(*coros)
    return dict(zip(keys, results))


async def _collect_alerts() -> list[dict]:
    """Run threshold checks against in-process state and return active alerts."""
    alerts = []

    mem = _meminfo_snapshot()
    used_pct = mem.get("used_pct", 0)
    if used_pct > _ALERT_THRESHOLDS["memory_percent"]:
        alerts.append({
            "type": "memory_high_usage",
            "severity": "critical",
            "message": (f"Memory usage is {used_pct}% "
                        f"({mem.get('used_gb', 0)} / {mem.get('total_gb', 0)} GB)"),
        })

    health = await _alert_endpoint_health()
    failed = [k for k, ok in health.items() if not ok]
    if len(failed) >= _ALERT_THRESHOLDS["endpoint_failures"]:
        alerts.append({
            "type": "endpoint_failures",
            "severity": "critical",
            "message": f"{len(failed)} serving endpoint(s) unhealthy: {', '.join(failed)}",
        })

    return alerts


def _send_alerts(alerts: list[dict], force: bool = False) -> list[str]:
    """Route alerts to every configured notifier channel, honoring the cooldown.

    Blocking (urllib in the discord notifier) — call via asyncio.to_thread
    from async contexts.
    """
    channels = _configured_alert_channels()
    sent: list[str] = []
    if not channels:
        if alerts:
            _logger.warning("Alerts active but no alert channel is configured: %s",
                            [a["type"] for a in alerts])
        return sent
    now = _time.time()  # wall-clock so persisted cooldowns survive restarts
    for alert in alerts:
        last = _last_alert_sent.get(alert["type"], 0.0)
        if not force and last and now - last < _ALERT_COOLDOWN_S:
            continue
        delivered = False
        for name, cfg in channels.items():
            if _ALERT_NOTIFIERS[name](alert, cfg):
                delivered = True
            else:
                _logger.warning("Failed to send %s alert: %s", name, alert["type"])
        if delivered:
            _last_alert_sent[alert["type"]] = now
            sent.append(alert["type"])
            _save_alert_state()
    return sent


async def _alert_loop():
    """Periodic in-process alert check, started from _lifespan."""
    if not _ALERT_ENABLED:
        _logger.info("Alerting disabled via config (alerts.enabled=false)")
        return
    _logger.info("Alerting loop started (interval %ss, endpoints %s, channels %s)",
                 _ALERT_INTERVAL_S, ",".join(_ALERT_ENDPOINTS),
                 ",".join(_configured_alert_channels()) or "none")
    while True:
        await asyncio.sleep(_ALERT_INTERVAL_S)
        try:
            alerts = await _collect_alerts()
            if alerts:
                _logger.warning("%d alert(s) active: %s",
                                len(alerts), [a["type"] for a in alerts])
                await asyncio.to_thread(_send_alerts, alerts)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _logger.error("Alert check failed: %s", e)


@app.post("/api/alerts/check", dependencies=[Depends(verify_auth)])
async def run_alert_check():
    """Manual alert check — returns active alerts; sends to Discord if configured."""
    alerts = await _collect_alerts()
    sent = await asyncio.to_thread(_send_alerts, alerts, True)
    channels = _configured_alert_channels()
    return {
        "alerts": alerts,
        "sent": sent,
        "channels": sorted(channels),
        "webhook_configured": "discord" in channels,
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


@app.post("/api/ollama/stop", dependencies=[Depends(verify_auth)])
async def stop_ollama_model(req: OllamaStopRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Model name is required")
    result = await _run("ollama", "stop", name, timeout=30)
    if result.returncode != 0:
        raise HTTPException(500, (result.stderr or result.stdout or "ollama stop failed").strip())
    _logger.info("Ollama warm model stopped: %s", name)
    return {"ok": True, "output": (result.stdout + result.stderr).strip()}

# ── LiteLLM ───────────────────────────────────────────────────────────────────

@app.get("/api/litellm/models")
async def list_litellm_models():
    try:
        r = await _http.get(LITELLM_BASE + "/v1/models", timeout=5.0)
        return r.json()
    except Exception as e:
        raise HTTPException(502, str(e))


@app.get("/api/litellm/config", dependencies=[Depends(verify_auth)])
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

    ok, detail = await _restart_litellm_backend()
    if not ok:
        _logger.error("LiteLLM restart failed after wildcard: %s", detail)
        raise HTTPException(500, f"Config saved but restart failed: {detail}")
    _logger.info("LiteLLM restarted successfully (%s)", detail)
    return {"ok": True, "message": f"Wildcard applied — LiteLLM restarted ({detail})"}


@app.post("/api/litellm/restart", dependencies=[Depends(verify_auth)])
async def restart_litellm():
    _logger.info("LiteLLM restart requested")
    ok, detail = await _restart_litellm_backend()
    if not ok:
        _logger.error("LiteLLM restart failed: %s", detail)
        raise HTTPException(500, f"Restart failed: {detail}")
    _logger.info("LiteLLM restarted successfully (%s)", detail)
    return {"ok": True, "message": detail}

# ── Shared engine helpers ─────────────────────────────────────────────────────

async def _find_container_by_port(port: int, docker_filter: str | None = None) -> Optional[str]:
    """Return the container ID listening on the given host port, or None.

    `--filter publish=` only matches *published* port mappings, so a container
    run with `--network host` (which is how vLLM runs on the GB10) is invisible
    to it despite genuinely owning the port. Fall back to a name filter.
    """
    result = await _run("docker", "ps", "--filter", f"publish={port}", "--format", "{{.ID}}", timeout=5)
    lines = result.stdout.strip().splitlines() if result.stdout.strip() else []
    if lines:
        return lines[0].strip()
    if docker_filter:
        result = await _run("docker", "ps", "--filter", f"name={docker_filter}",
                            "--format", "{{.ID}}", timeout=5)
        lines = result.stdout.strip().splitlines() if result.stdout.strip() else []
        if lines:
            return lines[0].strip()
    return None


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
    """Get running status, loaded model, and container info for an engine.

    Discovers all Docker containers matching docker_name and health-checks
    each on its actual published port — handles multi-instance setups
    (e.g., dual vLLM on ports 8000 + 8001).
    """
    import re as _re

    # Discover all matching containers with their published ports
    result = await _run(
        "docker", "ps", "--filter", f"name={docker_name}",
        "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}", timeout=5)

    instances = []
    for line in (result.stdout.strip().split("\n") if result.stdout.strip() else []):
        parts = line.split("\t")
        name = parts[0]
        status = parts[1] if len(parts) > 1 else ""
        ports_str = parts[2] if len(parts) > 2 else ""

        # Extract first host port from "0.0.0.0:8001->8001/tcp, ..."
        port_match = _re.search(r"0\.0\.0\.0:(\d+)->", ports_str)
        host_port = int(port_match.group(1)) if port_match else None

        inst_url = f"http://127.0.0.1:{host_port}" if host_port else base_url
        inst_running = await service_ok(inst_url, health_path)
        inst_model = None
        if inst_running and models_path:
            try:
                r = await _http.get(inst_url + models_path, timeout=3.0)
                d = r.json().get("data", [])
                if d:
                    inst_model = d[0]["id"]
            except Exception:
                pass

        instances.append({
            "name": name, "status": status, "port": host_port,
            "running": inst_running, "model": inst_model,
        })

    # No containers found — fall back to direct health check (non-Docker engine)
    if not instances:
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
        if not running:
            state = "stopped"
        elif models_path is None or model:
            state = "serving"
        else:
            state = "loading"
        return {"running": running, "model": model, "state": state,
                "container_info": "", "instances": []}

    # Aggregate: running if ANY instance is healthy
    any_running = any(i["running"] for i in instances)
    primary_model = next((i["model"] for i in instances if i["running"] and i["model"]), None)
    legacy_info = "\n".join(f"{i['name']}\t{i['status']}" for i in instances)

    # Readiness: a container being up ("running") is not the same as the model
    # being loaded — vLLM weight-load takes minutes, during which /v1/models is
    # empty. Surface a tri-state so the UI can show "Loading…" vs "Serving".
    if models_path is None:      # engines w/o a models endpoint: health == ready
        state = "serving" if any_running else "loading"
    elif primary_model:
        state = "serving"
    else:
        state = "loading"

    return {
        "running": any_running,
        "model": primary_model,
        "state": state,
        "container_info": legacy_info,
        "instances": instances,
    }


def _extract_port(url: str) -> int:
    """Extract port number from a URL like http://host:port or http://host:port/path."""
    try:

        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
    except Exception:
        pass
    raise ValueError(f"Cannot extract port from URL: {url}")


async def _engine_stop(base_url: str, engine_name: str, docker_filter: str | None = None) -> dict:
    """Stop the Docker container for an engine by its configured port."""
    try:
        port = _extract_port(base_url)
    except ValueError:
        raise HTTPException(400, f"Invalid {engine_name} URL — cannot determine port from '{base_url}'")
    cid = await _find_container_by_port(port, docker_filter)
    if not cid:
        raise HTTPException(404, f"No container found listening on {engine_name} port — already stopped?")
    ok, output = await _docker_stop(cid)
    return {"ok": ok, "output": output}


async def _running_profile_vram_credit(engine_key: str, profiles: list) -> tuple[float, str]:
    """VRAM (GB) to credit back for the profile currently running on an engine.

    A new profile's start script does `docker rm -f <container>`, so whatever is
    running now will be torn down and its unified memory freed before the new one
    loads. Because a running engine's real footprint is unmeasurable on the GB10
    (see _get_available_memory_gb), we identify WHICH profile is live by matching
    the engine's reported served-model-name against each profile's start script
    (scripts embed their own --served-model-name), and credit that profile's
    declared vram_gb. Returns (0.0, "") if nothing is running or the running
    profile cannot be identified — a deliberately conservative credit.
    """
    eng = _ENGINES.get(engine_key)
    if not eng:
        return 0.0, ""
    try:
        status = await _engine_status(
            _engine_bases[engine_key], eng.get("docker_filter", engine_key),
            eng.get("health_path", "/health"), eng.get("models_path"))
    except Exception:
        return 0.0, ""
    if not status.get("running"):
        return 0.0, ""
    served = status.get("model")
    if not served:
        return 0.0, ""
    # Primary match: served name appears verbatim in a profile's start script.
    for p in profiles:
        if p.get("vram_gb") is None:
            continue
        try:
            content = Path(os.path.expanduser(p.get("script", ""))).read_text()
        except Exception:
            content = ""
        if served in content:
            return float(p["vram_gb"]), p["id"]
    # Fallback heuristic: fuzzy token match on served name vs profile id/name.
    sv = served.lower()
    for p in profiles:
        if p.get("vram_gb") is None:
            continue
        pid = p["id"].lower()
        if pid.startswith("start_"):
            pid = pid[6:]
        if sv in pid or pid in sv or sv in p.get("name", "").lower():
            return float(p["vram_gb"]), p["id"]
    return 0.0, ""


# Safety margin (GB) reserved for the OS and other services on the unified pool.
_VRAM_SAFETY_MARGIN_GB = 8

# Minimum projected headroom (GB, on top of the safety margin) required to admit a
# profile whose VRAM need is unknown (no `# VRAM:` header). We cannot math an
# unmetered launch, so instead of skipping the check we refuse to launch onto a
# nearly-full pool — the case that actually hangs the box.
_VRAM_UNKNOWN_MIN_HEADROOM_GB = 24


async def _other_running_engines(exclude_key: str) -> list[str]:
    """Names of OTHER engines currently holding the unified pool.

    Their footprint is already reflected in MemAvailable (so the admission math
    stays correct without crediting them), but naming them tells the user which
    engine to stop — the admission failure often means a *different* engine is
    occupying memory, not that this engine's model is too big. Checked only on
    the rejection path; probes run concurrently.
    """
    async def _running(key: str) -> Optional[str]:
        eng = _ENGINES.get(key, {})
        try:
            status = await _engine_status(
                _engine_bases[key], eng.get("docker_filter", key),
                eng.get("health_path", "/health"), eng.get("models_path"))
        except Exception:
            return None
        return eng.get("name", key) if status.get("running") else None

    others = [k for k in _ENGINES if k != exclude_key]
    names = await asyncio.gather(*(_running(k) for k in others))
    return [n for n in names if n]


async def _other_engines_note(engine_key: str) -> str:
    """A trailing sentence naming other running engines, or '' if none."""
    try:
        running = await _other_running_engines(engine_key)
    except Exception:
        return ""
    if not running:
        return ""
    return (f" Note: {', '.join(running)} {'is' if len(running) == 1 else 'are'} "
            f"also running and holding the shared pool — stopping it frees memory.")


async def _vram_admission_check(engine_key: str, profile: dict, force: bool,
                                scan_fn=None) -> None:
    """Reject a profile launch that would overcommit the GB10 unified-memory pool.

    GB10 reasoning: GPU and system RAM share one ~121 GB pool and nvidia-smi
    can't report memory here, so we admit based on /proc/meminfo MemAvailable
    plus a "reclaim credit" for the profile about to be torn down and replaced
    by this one (its start script runs `docker rm -f` first). Engine-generic:
    works for any engine in _ENGINES. Profiles without a declared vram_gb, and
    force=true launches, skip the check.
    """
    if force:
        _logger.warning("VRAM admission check SKIPPED (force=true) for profile '%s'",
                        profile.get("id"))
        return
    vram = profile.get("vram_gb")
    profiles = scan_fn() if scan_fn else _scan_profiles(engine_key)
    available = _get_available_memory_gb()
    credit, credit_id = await _running_profile_vram_credit(engine_key, profiles)
    projected = available + credit
    margin = _VRAM_SAFETY_MARGIN_GB
    if vram is None:
        # Unknown need: can't math it, but still refuse to launch onto a
        # nearly-full pool. Require a healthy headroom floor above the margin.
        headroom = projected - margin
        if headroom < _VRAM_UNKNOWN_MIN_HEADROOM_GB:
            credit_note = (f"{credit:.0f} GB reclaimed from running profile '{credit_id}'"
                           if credit else "no reclaimable running profile identified")
            raise HTTPException(
                409,
                f"Refusing to start '{profile.get('id')}': it declares no VRAM "
                f"footprint (add a '# VRAM: NN' header to its start script), and only "
                f"{headroom:.0f} GB of unified memory headroom is projected available "
                f"({available:.0f} GB free + {credit_note}, minus a {margin} GB "
                f"safety margin) — below the {_VRAM_UNKNOWN_MIN_HEADROOM_GB} GB floor "
                f"required for an unmetered launch."
                f"{await _other_engines_note(engine_key)} Pass force=true to override.")
        _logger.warning(
            "Profile '%s' has no VRAM metadata — admitted on %.0f GB headroom floor; "
            "add a '# VRAM: NN' header for a precise check",
            profile.get("id"), headroom)
        return
    if vram > projected - margin:
        credit_note = (f"{credit:.0f} GB reclaimed from running profile '{credit_id}'"
                       if credit else
                       "no running profile could be identified, so a conservative "
                       "0 GB reclaim credit was used")
        raise HTTPException(
            409,
            f"Insufficient unified memory to start '{profile.get('id')}': needs "
            f"{vram} GB but only {projected:.0f} GB is projected available "
            f"({available:.0f} GB free + {credit_note}), and {margin} GB is held "
            f"back as an OS/services safety margin."
            f"{await _other_engines_note(engine_key)} Pass force=true to override.")


def _launch_argv(script: str, safe_id: str) -> list[str]:
    """Wrap a profile launch so it OUTLIVES this service.

    DMM runs as a systemd --user unit, and a profile script spawned with plain Popen
    lands in dgx-model-manager.service's cgroup. systemd's default
    KillMode=control-group then takes every one of those children down on `systemctl
    --user restart dgx-model-manager` — including a `docker run` client in the
    foreground, which stops the container it is attached to. Observed 2026-08-18: a
    routine restart to pick up new code stopped a vllm_node that had been serving for
    25 hours (clean exit 0, ~4 min outage). start_new_session=True does NOT help; it
    detaches the session, not the cgroup.

    `systemd-run --user --scope` puts the launch in its own transient scope, a sibling
    of this service rather than a child, so a restart or crash-loop of DMM cannot reach
    it. Verified: the scope's cgroup is .../app.slice/<unit>.scope.

    Falls back to a bare `bash` when systemd-run is unavailable (non-systemd host, no
    session bus). That restores the old fragile behavior rather than failing the launch
    — a manager that cannot start a model is worse than one whose restarts are unsafe.
    """
    if not shutil.which("systemd-run"):
        _logger.warning("systemd-run not found — launching in DMM's own cgroup; a DMM "
                        "restart will kill this engine")
        return ["bash", script]
    # Unit names must be unique per launch: a lingering scope from a previous start of
    # the same profile would otherwise collide and fail the launch.
    unit = f"dmm-{safe_id}-{int(_time.time())}"[:200]
    return [
        "systemd-run", "--user", "--scope", "--quiet",
        # --collect reaps the scope if the script exits non-zero; without it a failed
        # launch leaves a dead unit that blocks nothing but clutters `systemctl --user`.
        "--collect", f"--unit={unit}",
        "bash", script,
    ]


def _llamacpp_recipes() -> dict:
    """The llama.cpp quant ladder from config.json, or {} if unconfigured.

    Read live rather than cached at import: editing config.json is how the ladder is
    tuned, and a restart-to-see-it loop is how stale recipes get launched by mistake.
    """
    try:
        cfg = json.loads(_CONFIG_FILE.read_text())
    except Exception:
        return {}
    return cfg.get("llamacpp", {}).get("recipes", {}) or {}


def _override_int(lo: int, hi: int | None = None):
    def _check(value):
        if isinstance(value, bool) or not isinstance(value, (int, str, float)):
            raise HTTPException(400, f"expected an integer in {lo}..{hi or '∞'}")
        try:
            ival = int(str(value).strip())
        except (TypeError, ValueError):
            raise HTTPException(400, f"expected an integer in {lo}..{hi or '∞'}, "
                                     f"got {value!r}")
        if ival < lo or (hi is not None and ival > hi):
            raise HTTPException(400, f"expected an integer in {lo}..{hi or '∞'}, got {ival}")
        return str(ival)
    return _check


def _override_float(lo: float, hi: float):
    def _check(value):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise HTTPException(400, f"expected a number in {lo}..{hi}")
        try:
            fval = float(str(value).strip())
        except (TypeError, ValueError):
            raise HTTPException(400, f"expected a number in {lo}..{hi}, got {value!r}")
        if not (lo <= fval <= hi):
            raise HTTPException(400, f"expected a number in {lo}..{hi}, got {fval}")
        return repr(fval)
    return _check


# Allow-list of NAMES, never a prefix match and never pass-through: these values arrive in
# an HTTP body and land in a bash script's environment, then in docker argv. The bounds
# reuse `_derive_launch_spec`'s own util_floor/util_cap rather than inventing new numbers —
# two sources of truth for the same clamp is how they drift apart.
_UTIL_FLOOR = 0.10
_UTIL_CAP = 0.95
_OVERRIDE_ENV: dict[str, tuple[str, object]] = {
    "max_model_len": ("VLLM_MAX_MODEL_LEN", _override_int(1)),
    "gpu_memory_utilization": ("VLLM_GPU_MEMORY_UTILIZATION",
                               _override_float(_UTIL_FLOOR, _UTIL_CAP)),
    "max_num_seqs": ("VLLM_MAX_NUM_SEQS", _override_int(1, 256)),
}


def _collect_overrides(raw: dict | None) -> dict:
    """Drop the fields the user left blank. The pure request-body builder.

    The profile card sends one input per override; an empty or whitespace-only box means
    "use the derived default", which must be expressed as the key being ABSENT, not as an
    empty string (an empty string is a validation error, and the derived default lives in
    the script's own `${VAR:-N}` placeholder). The browser applies the same rule before
    sending; this is the server-side half of the pair, so a hand-rolled client cannot make
    a blank field mean something different.
    """
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[key] = value.strip() if isinstance(value, str) else value
    return out


def _resolve_overrides(overrides: dict | None) -> dict[str, str]:
    """Validate an untrusted override map into env-ready strings, or raise 400."""
    if isinstance(overrides, dict):
        overrides = _collect_overrides(overrides)
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise HTTPException(400, "overrides must be an object")
    accepted = ", ".join(sorted(_OVERRIDE_ENV))
    resolved: dict[str, str] = {}
    for key, value in overrides.items():
        if key not in _OVERRIDE_ENV:
            raise HTTPException(400, f"Unknown override '{key}'; accepted: {accepted}")
        if value is None:
            continue
        env_name, validator = _OVERRIDE_ENV[key]
        try:
            resolved[env_name] = validator(value)
        except HTTPException as exc:
            raise HTTPException(400, f"Invalid override '{key}': {exc.detail}")
    return resolved


_GENERATED_FROM_MARKER = "# Auto-generated by DGX Model Manager from:"

# ─── Legacy-script flag parsing (04-03) ───────────────────────────────────────
# Deliberately NOT a bash parser (04-RESEARCH §"Don't Hand-Roll"): a regex that
# only recognises a literal numeric token, and reports `unparseable` for
# everything else. Guessing at `--max-model-len "$CTX"` would render a number the
# script does not actually use, which is worse than admitting we cannot tell.
UNPARSEABLE = "unparseable"

_FLAG_KEYS = {
    "--max-model-len": "max_model_len",
    "--gpu-memory-utilization": "util",
    "--max-num-seqs": "max_num_seqs",
}

# `--flag N`, `--flag=N` or `--flag "N"`. The value group is captured loosely and
# validated afterwards so a non-numeric token is *seen* (and marked unparseable)
# rather than silently skipped, which would look identical to "flag absent".
_FLAG_RE = _re.compile(
    r"(?<![\w-])(--max-model-len|--gpu-memory-utilization|--max-num-seqs)"
    r"(?:\s*=\s*|\s+)(\S+)"
)
_NUM_RE = _re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def _parse_script_flags(text: str) -> dict:
    """Extract the three launch flags from arbitrary script text.

    Returns `{max_model_len, util, max_num_seqs}` where each value is an int, a
    float, or the string `UNPARSEABLE`. A flag is unparseable when it is absent,
    when its token is not a bare literal number (`$VAR`, `$(...)`, an array
    element, a quoted expansion), or when it appears twice with different values.
    """
    seen: dict[str, list] = {key: [] for key in _FLAG_KEYS.values()}
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            continue
        for flag, token in _FLAG_RE.findall(line):
            key = _FLAG_KEYS[flag]
            token = token.strip().strip("\"'")
            if not _NUM_RE.match(token):
                seen[key].append(UNPARSEABLE)
                continue
            seen[key].append(float(token) if "." in token else int(token))

    out = {}
    for key, values in seen.items():
        if not values or UNPARSEABLE in values or len(set(values)) > 1:
            out[key] = UNPARSEABLE
        else:
            out[key] = values[0]
    return out


_PLACEHOLDER_MARKER = "${VLLM_MAX_MODEL_LEN"


def _classify_script(text: str) -> str:
    """One of `parameterized`, `generated`, `recipe`, `legacy`.

    Order matters: a parameterized script is also a generated one, and a
    recipe-backed generated script carries the marker too — so the placeholder
    test runs first and the recipe test is only reached by non-placeholder text.
    """
    text = text or ""
    has_marker = _GENERATED_FROM_MARKER in text
    if has_marker and _PLACEHOLDER_MARKER in text:
        return "parameterized"
    if _PF_RECIPE_RE.search(text):
        return "recipe"
    if has_marker:
        return "generated"
    return "legacy"


def _profile_source_dir(script: str) -> Path | None:
    """The launch dir a generated profile was built from, per its own header.

    The header is the only link back to config.json: profiles are scanned as scripts,
    not as models. Hand-written scripts carry no marker and return None.
    """
    try:
        lines = Path(script).read_text().splitlines()[:20]
    except Exception:
        return None
    for idx, line in enumerate(lines):
        if line.strip().startswith(_GENERATED_FROM_MARKER):
            if idx + 1 < len(lines) and lines[idx + 1].startswith("# "):
                candidate = Path(os.path.expanduser(lines[idx + 1][2:].strip()))
                return candidate if candidate.is_dir() else None
    return None


def _overridden_vram_gb(profile: dict, override_env: dict[str, str],
                        force: bool) -> float | None:
    """Footprint of an OVERRIDDEN launch, or None when nothing footprint-affecting changed.

    Admitting an overridden launch against the script header's static `# VRAM:` comment
    is the same trap the llama.cpp recipe ladder already sprang: the header describes the
    launch the generator intended, not the one being asked for.
    """
    ctx = override_env.get("VLLM_MAX_MODEL_LEN")
    util = override_env.get("VLLM_GPU_MEMORY_UTILIZATION")
    if ctx is None and util is None:
        return None

    launch_dir = _profile_source_dir(profile.get("script", ""))
    config = None
    if launch_dir is not None:
        try:
            parsed = json.loads((launch_dir / "config.json").read_text())
            config = parsed if isinstance(parsed, dict) else None
        except Exception:
            config = None
    if config is None:
        # Never silently fall back to the stale header: an unverifiable override is
        # exactly the case where admission matters most.
        if not force:
            raise HTTPException(
                400, "Cannot verify this override against the model config "
                     f"({profile.get('id')}): no readable config.json for the profile. "
                     "Re-send with force=true to launch unchecked.")
        _logger.warning("override footprint underived for '%s' — forced launch",
                        profile.get("id"))
        return None

    pool_gb = 121.0
    if util is not None:
        # vLLM reserves this share of the pool up front, whatever the KV math says.
        return round(pool_gb * float(util), 1)

    info = None
    try:
        info = _profile_model_info(launch_dir, None)
    except Exception:
        info = None
    weights_gb = float((info or {}).get("size_gb") or 0.0)
    kv_dtype = _kv_dtype_from_config(config)
    spec = _derive_launch_spec(
        config, weights_gb=weights_gb, pool_gb=pool_gb,
        kv_dtype_bytes=1 if kv_dtype == "fp8" else 2,
        requested_context=int(ctx))
    for warning in spec.get("warnings") or []:
        _logger.info("override derive warning for '%s': %s", profile.get("id"), warning)
    return round(spec["weights_gb"] + spec["kv_gb"] + spec["overhead_gb"], 1)


async def _engine_start(req_profile: str, scan_fn, engine_name: str,
                        engine_key: str | None = None, force: bool = False,
                        recipe: str | None = None,
                        overrides: dict | None = None) -> dict:
    """Start a Docker engine by launching the selected profile script."""
    profiles = scan_fn()
    profile = next((p for p in profiles if p["id"] == req_profile), None)
    if not profile:
        raise HTTPException(404, f"Profile '{req_profile}' not found")
    # Resolve the recipe BEFORE admission: the script header declares one nominal
    # footprint, but the ladder spans 17 GB (tiny) to 60 GB (longctx). Admitting on the
    # header would wave through a launch three times its declared size.
    known_recipes = _llamacpp_recipes() if recipe else {}
    if recipe:
        if recipe not in known_recipes:
            raise HTTPException(400, f"Unknown recipe '{recipe}'; "
                                     f"have: {', '.join(known_recipes) or 'none'}")
        r_vram = known_recipes[recipe].get("vram_gb")
        if r_vram is not None:
            profile = {**profile, "vram_gb": r_vram}

    # Same ordering rule as the recipe block: an override that raises the real footprint
    # must be admitted against that footprint, not the script header's static comment.
    override_env = _resolve_overrides(overrides)
    if override_env:
        _over_vram = _overridden_vram_gb(profile, override_env, force)
        if _over_vram is not None:
            profile = {**profile, "vram_gb": _over_vram}

    await _vram_admission_check(engine_key or engine_name, profile, force, scan_fn)
    script = os.path.expanduser(profile.get("script", ""))
    if not Path(script).exists():
        raise HTTPException(400, f"Script not found: {script}")
    safe_id = _re.sub(r"[^a-zA-Z0-9._-]", "_", req_profile)
    log_path = f"/tmp/{engine_name.lower()}_{safe_id}.log"

    # Recipe reaches the script as an env var. Already allow-listed against the configured
    # map above — it arrives in an HTTP body and lands in a bash script's environment, so
    # a known-names check is the only acceptable filter.
    env = os.environ.copy()
    if recipe:
        env["RECIPE"] = recipe
        _logger.info("%s recipe '%s' -> %s", engine_name, recipe, known_recipes[recipe])
    if override_env:
        # `systemd-run --user --scope` inherits Popen(env=), so there is one transport
        # here, not two; no --setenv= enumeration in _launch_argv.
        env.update(override_env)
        _logger.info("%s launch overrides: %s", engine_name, override_env)

    _logger.info("%s starting profile '%s' — script: %s", engine_name, profile["name"], script)
    try:
        _fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        os.unlink(log_path)
        _fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(_fd, "w") as logf:
        subprocess.Popen(
            _launch_argv(script, safe_id),
            stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    _logger.info("%s launched — logs at %s", engine_name, log_path)
    label = f"{profile['name']} [{recipe}]" if recipe else profile["name"]
    return {"ok": True, "message": f"Launched {label} — logs at {log_path}"}


# ── Unified-memory visibility ─────────────────────────────────────────────────
# On GB10 the GPU and system RAM are one pool, and CUDA's mem_get_info reports
# *free* memory — NOT MemAvailable. Page cache is reclaimable by the kernel but
# CUDA counts it as unavailable, so vLLM's budget is
#
#     usable = MemTotal * gpu_memory_utilization - (MemTotal - MemFree)
#
# Reading a 37 GB safetensors set fills the page cache, and that cache is then
# charged against the very budget that has to hold those weights. Observed
# 2026-08-04 on Qwen3.6-35B-A3B-FP8 at an unchanged util of 0.55:
#
#     buff/cache ~25 GB  ->  Available KV cache memory  0.19 GiB  (refused to start)
#     buff/cache  ~2 GB  ->  Available KV cache memory 25.97 GiB  (ready in 165s)
#
# This is why admission based on MemAvailable (see _get_available_memory_gb) can
# pass while the launch still dies in engine init: MemAvailable counts the page
# cache as free, and CUDA does not.

_MEM_RECLAIM_WARN_GIB = 8      # page cache above this measurably shrinks the budget
_MEM_RECLAIM_FAIL_GIB = 20     # at this point a full-context launch will very likely fail
_DROP_CACHES_CMD = "sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"


def _cuda_visible_memory() -> dict:
    """MemTotal/MemFree/reclaimable in GiB, from CUDA's point of view.

    Deliberately reports MemFree rather than MemAvailable: the gap between them
    is exactly the page cache that vLLM cannot use but is still billed for.
    """
    vals = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                vals[k] = int(v.split()[0])
    except Exception as e:
        return {"error": str(e)}
    gib = lambda kb: kb / 1024 / 1024
    total = gib(vals.get("MemTotal", 0))
    free = gib(vals.get("MemFree", 0))
    reclaimable = gib(vals.get("Cached", 0) + vals.get("Buffers", 0)
                      - vals.get("Shmem", 0))
    return {
        "total_gib": round(total, 1),
        "free_gib": round(free, 1),
        "available_gib": round(gib(vals.get("MemAvailable", 0)), 1),
        "reclaimable_gib": round(max(reclaimable, 0.0), 1),
        "cuda_unavailable_gib": round(max(total - free, 0.0), 1),
    }


def _vllm_budget_gib(util: float, mem: dict) -> dict:
    """What vLLM will actually have to work with at `util`, and what a cache drop buys."""
    total = mem.get("total_gib", 0.0)
    unavailable = mem.get("cuda_unavailable_gib", 0.0)
    reclaimable = mem.get("reclaimable_gib", 0.0)
    budget = total * util
    return {
        "util": util,
        "budget_gib": round(budget, 1),
        "charged_gib": round(unavailable, 1),
        "usable_gib": round(budget - unavailable, 1),
        "usable_after_reclaim_gib": round(budget - max(unavailable - reclaimable, 0.0), 1),
    }


# ── vLLM load-progress parsing ────────────────────────────────────────────────
# Phase weights are wall-clock share of a measured cold start, not equal slices:
# on this box weight load is ~30s of a ~165s start while torch.compile plus graph
# capture is ~65s, so an evenly-weighted bar would sit at "loading weights" and
# then jump. Percentages are the value at which the phase BEGINS.

_LOAD_PHASES = [
    ("starting",         3,  "Container starting"),
    ("engine_init",     10,  "Initializing engine"),
    ("loading_weights", 30,  "Loading weights"),
    ("compiling",       45,  "Compiling model (torch.compile)"),
    ("profiling",       62,  "Profiling memory"),
    ("kv_cache",        78,  "Sizing KV cache"),
    ("capturing",       88,  "Capturing CUDA graphs"),
    ("ready",          100,  "Ready"),
]
_PHASE_PCT = {p: pct for p, pct, _ in _LOAD_PHASES}
_PHASE_LABEL = {p: label for p, _, label in _LOAD_PHASES}

# Ordered longest-lived first: a single line can match several, and the LAST
# matching rule wins so progress only ever moves forward.
_LOAD_PATTERNS = [
    ("engine_init",     _re.compile(r"Initializing a V1 LLM engine|api_utils.*non-default args")),
    # "Model loading took" belongs here, not to profiling: it is emitted when the
    # weights finish, and treating it as the start of profiling skipped the whole
    # compile phase and parked the bar at 65% for a minute.
    ("loading_weights", _re.compile(r"Loading weights|default_loader|Loading safetensors"
                                    r"|Model loading took")),
    ("compiling",       _re.compile(r"torch\.compile|Compiling a graph|Dynamo bytecode"
                                    r"|Directly load the compiled graph|torch_compile_cache"
                                    r"|backend='inductor'")),
    ("profiling",       _re.compile(r"Memory profiling|Profiling CUDA graph memory")),
    # Measured order on this box: estimated graph memory and the KV verdict are
    # both logged ~10s BEFORE capture actually starts, so they precede capturing.
    ("kv_cache",        _re.compile(r"Estimated CUDA graph memory|Available KV cache memory"
                                    r"|GPU KV cache size|maximum concurrency")),
    ("capturing",       _re.compile(r"Capturing CUDA graph|Graph capturing finished"
                                    r"|CuTeDSL warmup")),
    ("ready",           _re.compile(r"Application startup complete|Starting vLLM API server")),
]

# A container that is `Up` proves nothing: the recipe path keeps the container
# alive and runs vLLM as an exec inside it, so engine death leaves a healthy-
# looking container with a dead server. Death must be read from the log.
_LOAD_FAILED_RE = _re.compile(
    r"Engine core initialization failed"
    r"|EngineDeadError"
    r"|raise (ValueError|RuntimeError)\("
    r"|torch\.OutOfMemoryError"
    r"|CUDA out of memory"
    r"|Error response from daemon"
    r"|invalid option"
)

# The line worth putting in front of a human, extracted from a traceback storm.
# Searched, not anchored: vLLM prefixes every line with "(EngineCore pid=217) ERROR
# 08-04 03:51:46 [core.py:1231] " and the prefix shape varies by subsystem, so the
# exception has to be found mid-line. Requiring the colon excludes the `raise
# ValueError(` frames that appear in the traceback body above the real message.
_LOAD_CAUSE_RE = _re.compile(
    r"((?:torch\.)?(?:ValueError|RuntimeError|OutOfMemoryError|AssertionError|OSError|"
    r"ImportError|MemoryError)\s*:\s*\S.*)$")


def _classify_vllm_log_line(line: str) -> Optional[str]:
    """Map one container log line to a load phase, or None if it says nothing new."""
    for phase, pat in _LOAD_PATTERNS:
        if pat.search(line):
            return phase
    return None


def _load_failure_reason(lines: list) -> str:
    """Pull the one explanatory line out of a failed startup's log tail.

    vLLM reports the same error three times (EngineCore, its re-raise, then the
    APIServer's RuntimeError wrapper) wrapped in ~90 lines of traceback. The
    useful one is the first concrete exception message; the RuntimeError wrapper
    ("See root cause above") is the least useful and is only a fallback.
    """
    fallback = ""
    for line in lines:
        m = _LOAD_CAUSE_RE.search(line.strip())
        if not m:
            continue
        msg = " ".join(m.group(1).split())
        if "See root cause above" in msg or "Engine core initialization failed" in msg:
            fallback = fallback or msg
            continue
        return msg
    if fallback:
        return fallback
    for line in reversed(lines):
        if line.strip():
            # Label it honestly. Presenting an arbitrary trailing line as "the
            # error" is how a routine access-log entry got reported as the cause
            # of a failure that had not actually happened.
            return ("No exception was logged. Last output: "
                    + " ".join(line.split())[:300])
    return "Container exited without logging a cause."


# ── Launch preflight ("dry load") ─────────────────────────────────────────────
# Every real launch failure on this box so far was knowable before committing
# ~100 GB of unified memory and 3 minutes of weight load:
#
#   2026-07-30  mount scope   — snapshots/<sha> mounted without blobs/, every
#                               weight file a dangling symlink inside the container
#   2026-08-04  entrypoint    — `docker run <image> --model ...` against an image
#                               whose entrypoint execs its arguments; dead in 2s
#   2026-08-04  memory budget — page cache charged against gpu-memory-utilization
#
# So preflight is static-first: parse the script, check the things that are true
# before anything runs, and only then spend a couple of seconds in a container.

_PF_IMAGE_RE   = _re.compile(r"^\s*(?:--\S+\s+)*([a-z0-9][\w./-]*(?::[\w.-]+)?)\s*\\\s*$", _re.M)
_PF_MOUNT_RE   = _re.compile(r"-v\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
_PF_MODEL_RE   = _re.compile(r"--model[= ]\s*(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
_PF_UTIL_RE    = _re.compile(r"--gpu-memory-utilization[= ]\s*([0-9.]+)")
_PF_RESTART_RE = _re.compile(r"--restart[= ]\s*(\S+)")
_PF_RECIPE_RE  = _re.compile(r"run-recipe\.sh\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))")


def _pf(level: str, check: str, title: str, detail: str, fix: str = "") -> dict:
    return {"level": level, "check": check, "title": title, "detail": detail, "fix": fix}


def _first_group(m) -> str:
    return next((g for g in m.groups() if g), "") if m else ""


_PF_ASSIGN_RE = _re.compile(r"^\s*([A-Za-z_]\w*)=(?:\"([^\"]*)\"|'([^']*)'|(\S+))\s*$", _re.M)
_PF_VAR_RE = _re.compile(r"\$\{(\w+)\}|\$(\w+)")


# Only these come from the process environment. Scripts legitimately use $HOME in
# mount paths, but the preflight result is returned over HTTP, so expansion is an
# allowlist rather than a blanket os.environ lookup — nothing else gets echoed back.
_PF_ENV_ALLOWED = ("HOME", "USER")


def _expand_script_vars(value: str, text: str) -> str:
    """Resolve `$VAR` against literal assignments in the same script.

    Launch scripts name things once at the top (`RECIPE="qwen3.6-…-solo"`,
    `-v "$HOME/.cache/huggingface:…"`) and use the variable below, so a parser
    that reads the use site literally comes away with "$RECIPE" or a mount docker
    rejects as "invalid characters for a local volume name" — and every check
    downstream of it degrades into a false failure.
    """
    if "$" not in value:
        return value
    env = {k: os.environ[k] for k in _PF_ENV_ALLOWED if k in os.environ}
    for m in _PF_ASSIGN_RE.finditer(text):
        env[m.group(1)] = next((g for g in m.groups()[1:] if g is not None), "")
    return _PF_VAR_RE.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), value)


def _script_code(text: str) -> str:
    """The script with whole-line comments removed.

    Load scripts carry long rationale headers that quote the very commands being
    looked for — the Qwen3.6 profile's own comments mention `run-recipe.sh` and
    `docker rm -f vllm_node`. Parsing the raw text makes preflight read the
    documentation instead of the code, so every fact is taken from here.
    """
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def _parse_launch_script(text: str) -> dict:
    """Extract the launch facts preflight reasons about, for either script shape."""
    code = _script_code(text)
    recipe = _expand_script_vars(_first_group(_PF_RECIPE_RE.search(code)), code)
    image = ""
    for m in _PF_IMAGE_RE.finditer(code):
        cand = m.group(1)
        if "/" in cand or ":" in cand:
            image = cand
            break
    mounts = [_expand_script_vars(_first_group(m), code)
              for m in _PF_MOUNT_RE.finditer(code)]
    return {
        "code": code,
        "recipe": recipe,
        "recipe_backed": bool(recipe),
        "image": image,
        "mounts": mounts,
        "model": _expand_script_vars(_first_group(_PF_MODEL_RE.search(code)), code),
        "util": float(_first_group(_PF_UTIL_RE.search(code)) or 0) or None,
        "restart": _first_group(_PF_RESTART_RE.search(code)),
        "has_serve": bool(_re.search(r"^\s*vllm serve\b", code, _re.M)),
        "clears_container": "docker rm -f vllm_node" in code,
    }


def _preflight_static(text: str, cfg: dict) -> list:
    """Checks that need nothing but the script text. Pure — unit-testable."""
    facts = _parse_launch_script(text)
    out = []

    if facts["recipe_backed"]:
        out.append(_pf("ok", "shape", "Recipe-backed profile",
                       f"Delegates to run-recipe.sh recipe '{facts['recipe']}'. "
                       "Launch flags are owned by the recipe YAML, not by this script."))
    else:
        expected = _vllm_serve_command(facts["image"], cfg)
        if expected and not facts["has_serve"]:
            out.append(_pf(
                "fail", "entrypoint", "Missing `vllm serve` subcommand",
                f"Image '{facts['image'] or '(unparsed)'}' execs its arguments, so the "
                f"container will try to exec `--model` and exit 2 within seconds.",
                f"Insert `{expected} \\` immediately after the image line."))
        elif not expected and facts["has_serve"]:
            out.append(_pf(
                "fail", "entrypoint", "Unexpected `vllm serve` subcommand",
                f"Image '{facts['image']}' already starts the API server, so `vllm serve` "
                f"is passed to it as a positional model argument.",
                "Remove the `vllm serve` line."))
        else:
            out.append(_pf("ok", "entrypoint", "Entrypoint contract matches image",
                           f"'{facts['image'] or 'image'}' "
                           f"{'needs' if expected else 'does not need'} an explicit "
                           f"`vllm serve`, and the script "
                           f"{'has' if facts['has_serve'] else 'omits'} one."))

    if facts["restart"]:
        out.append(_pf(
            "warn", "restart_policy", f"Script sets --restart {facts['restart']}",
            "A failed launch will be resurrected across reboot under the name "
            "vllm_node. run-recipe.sh treats any existing vllm_node as 'already "
            "running' and skips its own launch, so a broken profile can keep the "
            "box's default model down indefinitely.",
            "Remove --restart; vllm-default-model.service owns boot recovery."))

    if not facts["clears_container"]:
        out.append(_pf(
            "warn", "collision", "Script does not clear the existing container",
            "Without `docker rm -f vllm_node` a previous container keeps the name and "
            "the port, and the launch either fails or is silently skipped.",
            "Add `docker rm -f vllm_node 2>/dev/null || true` before launching."))

    if facts["model"] and not facts["recipe_backed"]:
        under_mount = any(facts["model"].startswith(m.split(":", 1)[-1].rstrip("/"))
                          for m in facts["mounts"] if ":" in m)
        if not under_mount:
            out.append(_pf(
                "warn", "mount_scope", "Model path is not under any bind mount",
                f"--model points at {facts['model']} but no -v maps a container path "
                f"containing it. HF snapshot dirs are symlinks into ../../blobs/, so a "
                f"mount scoped to snapshots/<sha> leaves every weight file dangling.",
                "Mount the models--*/ root, not the snapshot subdirectory."))
    return out


def _preflight_memory(util: Optional[float]) -> list:
    """The budget check. `util` None means the script did not declare one."""
    mem = _cuda_visible_memory()
    if "error" in mem:
        return [_pf("warn", "memory", "Could not read /proc/meminfo", mem["error"])]
    reclaim = mem["reclaimable_gib"]
    budget = _vllm_budget_gib(util, mem) if util else None
    arith = ""
    if budget:
        arith = (f" At util {util}, vLLM's budget is {budget['budget_gib']} GiB, of which "
                 f"{budget['charged_gib']} GiB is already charged as unavailable — leaving "
                 f"{budget['usable_gib']} GiB. Reclaiming the cache would raise that to "
                 f"{budget['usable_after_reclaim_gib']} GiB.")
    detail = (f"{reclaim} GiB of page cache is held. CUDA reports free memory, not "
              f"MemAvailable, so cached pages are billed against "
              f"--gpu-memory-utilization even though the kernel would happily drop "
              f"them.{arith}")
    if reclaim >= _MEM_RECLAIM_FAIL_GIB:
        level = "fail"
    elif reclaim >= _MEM_RECLAIM_WARN_GIB:
        level = "warn"
    else:
        return [_pf("ok", "memory", "Page cache is not eating the budget",
                    f"{reclaim} GiB cached; {mem['free_gib']} GiB genuinely free.{arith}")]
    return [_pf(level, "memory", f"{reclaim} GiB of page cache will be charged to vLLM",
                detail, _DROP_CACHES_CMD)]


# Long enough for `import vllm` (~15s cold) plus argparse, short enough that the
# button never feels hung. A timeout is reported as `skip`, never as `fail`: a
# slow probe is not evidence of a bad script.
_PF_SMOKE_TIMEOUT_S = 90

# Validates flags without allocating a single byte of KV cache. Kept tolerant of
# vLLM's module reshuffles — an ImportError here means "cannot check", not "bad".
_PF_ARGPARSE_PROBE = (
    "import sys\n"
    "try:\n"
    "    from vllm.utils.argparse_utils import FlexibleArgumentParser\n"
    "except Exception:\n"
    "    from vllm.utils import FlexibleArgumentParser\n"
    "from vllm.entrypoints.openai.cli_args import make_arg_parser\n"
    "make_arg_parser(FlexibleArgumentParser()).parse_args(sys.argv[1:])\n"
    "print('ARGS_OK')\n"
)


async def _preflight_runtime(facts: dict, script: str) -> list:
    """Checks that need docker. Each degrades to `skip` rather than a false failure."""
    out = []

    name = await _run("docker", "ps", "-a", "--filter", "name=^vllm_node$",
                      "--format", "{{.Status}}", timeout=15)
    existing = (name.stdout or "").strip()
    if existing:
        out.append(_pf(
            "warn", "container_exists", f"A container named vllm_node exists ({existing})",
            "It holds the name and port 8000. The script's `docker rm -f` clears it, but "
            "run-recipe.sh would instead report 'already running' and skip launching.",
            "docker rm -f vllm_node"))

    image = facts.get("image")
    if image:
        insp = await _run("docker", "image", "inspect", image, "--format", "{{.Id}}",
                          timeout=20)
        if insp.returncode != 0:
            out.append(_pf(
                "warn", "image", f"Image '{image}' is not present locally",
                "The launch will pull it first, which can take several minutes and will "
                "look like a hung load.", f"docker pull {image}"))
        else:
            out.append(_pf("ok", "image", f"Image '{image}' present", insp.stdout.strip()[:19]))

    # The mount-scope test from 2026-07-30, run for real: read the model's own
    # config.json through the exact bind mounts the launch will use. A dangling
    # symlink fails here in about a second instead of after a 3-minute load.
    if "$" in (facts.get("model") or "") or any("$" in m for m in facts.get("mounts", [])):
        # Scripts compute paths at runtime (`HASH=$(basename "$SNAP_HOST")`), and
        # command substitution cannot be resolved without executing the script.
        # An unresolved path is "unknown", never "broken" — reporting it as a
        # failure would train the reader to ignore this check.
        out.append(_pf(
            "skip", "mount_readable", "Model path is computed at runtime",
            f"{facts.get('model')} contains a shell substitution, so the files could "
            f"not be read ahead of the launch. Static checks still apply."))
    elif facts.get("model") and facts.get("mounts") and not facts["recipe_backed"]:
        args = ["docker", "run", "--rm", "--entrypoint", "/bin/sh"]
        for m in facts["mounts"]:
            args += ["-v", m]
        target = facts["model"].rstrip("/") + "/config.json"
        args += [image or "busybox", "-c", f"cat {shlex.quote(target)} >/dev/null"]
        probe = await _run(*args, timeout=60)
        if probe.returncode == 0:
            out.append(_pf("ok", "mount_readable", "Model files readable inside the container",
                           f"Read {target} through the configured bind mounts."))
        else:
            out.append(_pf(
                "fail", "mount_readable", "Model files are NOT readable inside the container",
                f"Reading {target} through the configured mounts failed: "
                f"{(probe.stderr or probe.stdout or '').strip()[:300]}. This reads like a "
                f"corrupt download but is almost always mount scope — HF snapshot entries "
                f"are relative symlinks into ../../blobs/.",
                "Mount the models--*/ root so blobs/ and snapshots/ are both in scope."))
    return out


async def _preflight_smoke(facts: dict, script: str) -> list:
    """Spend a couple of seconds proving the arguments actually parse."""
    if facts["recipe_backed"]:
        _, recipe_root = _recipe_dirs()   # runner dir = parent of the recipe yaml dir
        runner = recipe_root / "run-recipe.sh"
        if not runner.exists():
            return [_pf("skip", "smoke", "Recipe runner not found",
                        f"{runner} does not exist, so the recipe could not be dry-run.")]
        proc = await asyncio.create_subprocess_exec(
            str(runner), facts["recipe"], "--dry-run",
            cwd=str(recipe_root),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(),
                                               timeout=_PF_SMOKE_TIMEOUT_S)
        except asyncio.TimeoutError:
            proc.kill(); await proc.wait()
            return [_pf("skip", "smoke", "Recipe dry-run timed out", "")]
        text = (stdout or b"").decode(errors="replace")
        if proc.returncode == 0:
            return [_pf("ok", "smoke", "Recipe dry-run succeeded",
                        text.strip()[-600:] or "run-recipe.sh --dry-run exited 0")]
        return [_pf("fail", "smoke", "Recipe dry-run failed", text.strip()[-600:])]

    if not facts.get("image"):
        return [_pf("skip", "smoke", "No image parsed from the script", "")]
    # Everything after the `vllm serve` line is a server flag; feed exactly those
    # to vLLM's own parser inside the image.
    code = facts.get("code") or _script_code(script)
    body = code.split("vllm serve", 1)[1] if facts["has_serve"] else ""
    flags = [_expand_script_vars(a, code)
             for a in shlex.split(body.replace("\\\n", " "))] if body else []
    if not flags:
        return [_pf("skip", "smoke", "No server flags parsed from the script", "")]
    args = ["docker", "run", "--rm", "--entrypoint", "python3"]
    for m in facts["mounts"]:
        args += ["-v", m]
    args += [facts["image"], "-c", _PF_ARGPARSE_PROBE] + flags
    probe = await _run(*args, timeout=_PF_SMOKE_TIMEOUT_S)
    combined = ((probe.stdout or "") + (probe.stderr or "")).strip()
    if "ARGS_OK" in combined:
        return [_pf("ok", "smoke", "vLLM accepted every flag",
                    f"{len(flags)} arguments parsed by vLLM's own argument parser "
                    f"inside {facts['image']}, without loading weights.")]
    # Only argparse's own vocabulary counts as the script being wrong. Everything
    # else — an import that needs CUDA, a moved module, a slow pull — means the
    # probe could not run, and saying "rejected" there would be a false alarm.
    # vLLM builds pydantic config objects during parsing, and some of them touch
    # the device, which this deliberately GPU-less probe cannot provide.
    if _re.search(r"error: (unrecognized arguments|argument |invalid choice|"
                  r"the following arguments are required|expected)", combined):
        return [_pf("fail", "smoke", "vLLM rejected the launch arguments",
                    combined[-600:])]
    return [_pf("skip", "smoke", "Could not run the argument probe",
                (combined[-400:] or "no output") +
                "\n\nThe probe runs without a GPU, so vLLM config objects that touch "
                "the device cannot be constructed. This says nothing about the script.")]


@app.post("/api/vllm/preflight", dependencies=[Depends(verify_auth)])
async def vllm_preflight(req: EngineStartRequest):
    """Dry-load a profile: everything a real launch would hit, minus the weights."""
    profiles = _scan_profiles("vllm")
    profile = next((p for p in profiles if p["id"] == req.profile), None)
    if not profile:
        raise HTTPException(404, f"Profile '{req.profile}' not found")
    script_path = Path(os.path.expanduser(profile.get("script", "")))
    if not script_path.exists():
        raise HTTPException(400, f"Script not found: {script_path}")
    text = script_path.read_text(errors="ignore")
    facts = _parse_launch_script(text)

    checks = _preflight_static(text, _app_config.get("vllm", {}) or {})
    util = facts["util"]
    if util is None and facts["recipe_backed"]:
        util = _recipe_util(facts["recipe"])
    checks += _preflight_memory(util)
    checks += await _preflight_runtime(facts, text)
    checks += await _preflight_smoke(facts, text)

    verdict = ("fail" if any(c["level"] == "fail" for c in checks)
               else "warn" if any(c["level"] == "warn" for c in checks) else "ok")
    return {
        "profile": req.profile,
        "verdict": verdict,
        "checks": checks,
        "facts": facts,
        "memory": _cuda_visible_memory(),
        "budget": _vllm_budget_gib(util, _cuda_visible_memory()) if util else None,
    }


# The one owner of "where the recipe YAMLs and run-recipe.sh live." Generator, Dry-Run
# smoke, and preflight memory math must all agree on this, or a configured recipe_dir
# would launch from one place while validating another. Config -> default (no env
# override: no other vllm block key has one either).
def _live_vllm_cfg() -> dict:
    """The `vllm` config block with env overrides applied — config -> env -> default,
    the same precedence the `alerts` block uses.

    The env layer lands HERE rather than inside the generator because
    `_build_vllm_profile_script` is deliberately a pure function of the `vllm_cfg`
    it is handed. Resolving the override at the single point where the live block
    is read keeps the generator and the preflight helpers (`_recipe_dirs`,
    `_recipe_util`) from disagreeing about which recipe directory is in force.
    """
    cfg = dict(_app_config.get("vllm", {}) or {})
    if env_dir := os.environ.get("MODEL_MANAGER_VLLM_RECIPE_DIR"):
        cfg["recipe_dir"] = cfg.get("recipe_dir") or env_dir
    return cfg


def _vllm_recipe_dir() -> str:
    return _live_vllm_cfg().get("recipe_dir") or "~/spark-vllm-docker/recipes"


def _recipe_dirs() -> tuple[Path, Path]:
    """(recipe_yaml_dir, runner_dir) from the live config. Pre-flight-only use
    (_recipe_util, _preflight_smoke): the generator stays a pure function of its
    vllm_cfg argument. run-recipe.sh sits in the checkout root, sibling of the
    recipes/ subdir, so the runner dir is the yaml dir's parent — the same
    assumption the generated wrapper's `cd "$RECIPE_DIR/.."` makes."""
    ydir = Path(os.path.expanduser(_vllm_recipe_dir()))
    return ydir, ydir.parent


def _read_recipe(recipe_dir, name: str) -> tuple[Optional[dict], list[str]]:
    """Render and normalize one run-recipe YAML; malformed input is no opinion."""
    if not isinstance(name, str) or not _re.fullmatch(r"[\w.-]+", name):
        return None, [f"invalid recipe name {name!r}"]

    try:
        path = Path(os.path.expanduser(os.fspath(recipe_dir))) / f"{name}.yaml"
        recipe = yaml.safe_load(path.read_text())
        if not isinstance(recipe, dict):
            return None, [f"recipe {name!r} is not a YAML mapping"]
        if "command" not in recipe:
            return None, [f"recipe {name!r} has no command"]
        defaults = recipe.get("defaults", {})
        params = {**defaults, **{}}
        try:
            rendered = recipe["command"].format(**params)
        except (KeyError, IndexError, ValueError) as exc:
            return None, [f"recipe {name!r} command placeholder error: {exc}"]
        tokens = shlex.split(rendered)

        flags = {}
        wanted = {
            "--gpu-memory-utilization", "--gpu-memory-utilization-gb",
            "--max-model-len", "--kv-cache-dtype", "--tool-call-parser",
            "--reasoning-parser", "--port",
        }
        for index, token in enumerate(tokens):
            flag, separator, inline = token.partition("=")
            if flag not in wanted:
                continue
            if separator:
                flags[flag] = inline
            elif index + 1 < len(tokens):
                flags[flag] = tokens[index + 1]

        def _number(flag: str, cast):
            value = flags.get(flag)
            return cast(value) if value is not None else None

        return {
            "name": recipe.get("name") or None,
            "description": recipe.get("description") or None,
            "model": recipe.get("model") or None,
            "gpu_memory_utilization": _number("--gpu-memory-utilization", float),
            "gpu_memory_utilization_gb": _number("--gpu-memory-utilization-gb", float),
            "max_model_len": _number("--max-model-len", int),
            "kv_cache_dtype": flags.get("--kv-cache-dtype") or None,
            "tool_call_parser": flags.get("--tool-call-parser") or None,
            "reasoning_parser": flags.get("--reasoning-parser") or None,
            "port": _number("--port", int),
            "solo_only": recipe.get("solo_only") if "solo_only" in recipe else None,
            "cluster_only": recipe.get("cluster_only") if "cluster_only" in recipe else None,
            "mods": recipe.get("mods") if "mods" in recipe else None,
        }, []
    except Exception as exc:
        location = str(path) if "path" in locals() else repr(recipe_dir)
        return None, [f"could not read recipe {name!r} from {location}: {exc}"]


def _recipe_util(recipe: str) -> Optional[float]:
    """Return only a fractional recipe utilization for preflight memory math."""
    record, _ = _read_recipe(_recipe_dirs()[0], recipe)
    return record.get("gpu_memory_utilization") if record is not None else None


@app.post("/api/vllm/reclaim-cache", dependencies=[Depends(verify_auth)])
async def vllm_reclaim_cache():
    """Drop the page cache so it stops being charged against the launch budget.

    Non-destructive: the kernel re-reads from disk on demand. Requires passwordless
    sudo; a box without it gets a clear 501 rather than a silent no-op. This grants
    no privilege the profile scripts did not already have — they run as the same
    user — but it is behind verify_auth because it is a system-wide side effect.
    """
    before = _cuda_visible_memory()
    probe = await _run("sudo", "-n", "true", timeout=10)
    if probe.returncode != 0:
        raise HTTPException(501, "Passwordless sudo is unavailable, so the page cache "
                                 f"cannot be dropped from here. Run manually: {_DROP_CACHES_CMD}")
    os.sync()
    res = await _run("sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches", timeout=60)
    if res.returncode != 0:
        raise HTTPException(500, f"drop_caches failed: {(res.stderr or '').strip()[:300]}")
    after = _cuda_visible_memory()
    freed = round(after.get("free_gib", 0) - before.get("free_gib", 0), 1)
    return {"ok": True, "freed_gib": freed, "before": before, "after": after,
            "message": f"Reclaimed {freed} GiB — now visible to CUDA."}


# ── Load progress stream ──────────────────────────────────────────────────────
# Sourced from `docker logs -f`, NOT from the /tmp launch log, because the /tmp
# log only ever captures the *first* foreground attempt: a detached or recipe-
# backed launch writes almost nothing there, and after a reboot /tmp is gone
# entirely. `docker logs` is the one source that works for every profile shape
# and survives a manager restart.

_PROGRESS_TAIL = 400          # enough to catch a load already in flight
_PROGRESS_CAUSE_WINDOW = 120  # log lines kept for root-cause extraction


async def _container_state(name: str) -> dict:
    res = await _run("docker", "inspect", name,
                     "--format", "{{.Id}}|{{.State.Status}}|{{.State.ExitCode}}",
                     timeout=15)
    if res.returncode != 0:
        return {"exists": False, "status": "absent", "exit_code": None, "id": ""}
    cid, _, rest = (res.stdout or "").strip().partition("|")
    status, _, code = rest.partition("|")
    return {"exists": True, "id": cid, "status": status,
            "exit_code": int(code) if code.strip().lstrip("-").isdigit() else None}


_PROGRESS_APPEAR_TIMEOUT_S = 45   # docker rm -f + docker run, plus an image pull check


async def _vllm_health_ok() -> bool:
    try:
        r = await _http.get(_engine_bases["vllm"] + "/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


async def _load_progress_events(container: str, fresh: bool = False) -> AsyncGenerator[str, None]:
    """Stream load phases until the model is ready or the load provably failed.

    Contract, mirroring _hf_download_events: exactly one terminal event (`ready`
    or `failed`). The pre-existing UI polled /status every 20s for ten minutes and
    called that "Model loading…", which is indistinguishable from a container that
    died in two seconds — the exact reason a broken profile looked like nothing
    happening at all.
    """
    started = _time.monotonic()
    phase, pct = "starting", _PHASE_PCT["starting"]
    recent: list = []
    facts: dict = {}
    sent_terminal = False

    def frame(**kw) -> str:
        kw.setdefault("elapsed_s", round(_time.monotonic() - started, 1))
        return f"data: {json.dumps(kw)}\n\n"

    # An already-serving model must report ready, not "starting": the log tail of a
    # long-running container is full of request lines and the startup milestones
    # have long since scrolled out of it. Skipped for a fresh launch, where the
    # *previous* model may still be answering /health for another second or two.
    if not fresh and await _vllm_health_ok():
        yield frame(status="ready", phase="ready", percent=100,
                    label=_PHASE_LABEL["ready"], line="Already serving")
        return

    # The launch script runs `docker rm -f` before `docker run`, so right after
    # POST /start the name still resolves to the OUTGOING container. Attaching to
    # that one and watching it get removed looks exactly like a crash — the first
    # end-to-end test of this stream reported "failed" 3.6s in, quoting a stray
    # access-log line from the model being replaced. So a fresh launch waits for a
    # container with a different ID, and identity is the container ID, not the name.
    state = await _container_state(container)
    prior_id = state.get("id", "") if fresh else ""
    if not state["exists"] or (fresh and state.get("id") == prior_id and prior_id):
        deadline = _time.monotonic() + _PROGRESS_APPEAR_TIMEOUT_S
        while _time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            state = await _container_state(container)
            if state["exists"] and state.get("id") != prior_id:
                break
            yield frame(status="loading", phase="starting", percent=2,
                        label="Waiting for container", line="")
    if not state["exists"] or (prior_id and state.get("id") == prior_id):
        yield frame(status="failed", phase="absent", percent=0,
                    error=f"No new container named {container} appeared within "
                          f"{_PROGRESS_APPEAR_TIMEOUT_S}s. The launch script exited "
                          f"without starting it — check the script's own log.")
        return
    container_id = state["id"]

    proc = await asyncio.create_subprocess_exec(
        "docker", "logs", "-f", "--tail", str(_PROGRESS_TAIL), container_id,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    yield frame(status="loading", phase=phase, percent=pct,
                label=_PHASE_LABEL[phase], line="Attached to container log")
    try:
        while True:
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=15)
            except asyncio.TimeoutError:
                # Silence is normal during torch.compile. Use it to re-check that
                # the container is still alive, then emit a heartbeat so the bar
                # keeps showing elapsed time.
                state = await _container_state(container_id)
                if state["status"] in ("exited", "dead"):
                    break
                yield frame(status="loading", phase=phase, percent=pct,
                            label=_PHASE_LABEL[phase], line="")
                continue
            if not raw:
                break
            line = raw.decode(errors="replace").rstrip()
            recent.append(line)
            del recent[:-_PROGRESS_CAUSE_WINDOW]

            # Opportunistic telemetry — the two numbers worth seeing mid-load.
            m = _re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", line)
            if m:
                facts["kv_cache_gib"] = float(m.group(1))
            m = _re.search(r"Model loading took\s*([\d.]+)\s*GiB", line)
            if m:
                facts["weights_gib"] = float(m.group(1))

            if _LOAD_FAILED_RE.search(line):
                reason = _load_failure_reason(recent)
                hint = ""
                if "KV cache" in reason or "out of memory" in reason.lower():
                    hint = (f"On GB10 the page cache is charged against "
                            f"--gpu-memory-utilization. Reclaim it and retry: "
                            f"{_DROP_CACHES_CMD}")
                yield frame(status="failed", phase="failed", percent=pct,
                            error=reason, hint=hint, line=line, **facts)
                sent_terminal = True
                break

            new = _classify_vllm_log_line(line)
            if new and _PHASE_PCT[new] > pct:
                phase, pct = new, _PHASE_PCT[new]
            if new == "ready":
                yield frame(status="ready", phase="ready", percent=100,
                            label=_PHASE_LABEL["ready"], line=line, **facts)
                sent_terminal = True
                break
            yield frame(status="loading", phase=phase, percent=pct,
                        label=_PHASE_LABEL[phase], line=line[-300:], **facts)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()

    if not sent_terminal:
        # The stream ended without a verdict: the container died, or the log closed.
        state = await _container_state(container_id)
        if state["status"] in ("exited", "dead"):
            yield frame(status="failed", phase="failed", percent=pct,
                        error=_load_failure_reason(recent),
                        exit_code=state.get("exit_code"), **facts)
        else:
            # Container is Up but the log ended — on the recipe path the container
            # outlives a dead engine, so `Up` is not proof of health. Ask the API.
            if await _vllm_health_ok():
                yield frame(status="ready", phase="ready", percent=100,
                            label=_PHASE_LABEL["ready"], **facts)
            else:
                yield frame(status="failed", phase="failed", percent=pct,
                            error=_load_failure_reason(recent),
                            hint="The container is running but the API is not "
                                 "answering /health — on the recipe path vLLM runs as "
                                 "an exec inside a container that survives its death.",
                            **facts)


@app.get("/api/vllm/progress")
async def vllm_progress(container: str = "vllm_node", fresh: bool = False):
    if not _re.fullmatch(r"[A-Za-z0-9][\w.-]{0,63}", container):
        raise HTTPException(400, "Invalid container name")
    return StreamingResponse(
        _load_progress_events(container, fresh=fresh), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Dynamic engine routes ─────────────────────────────────────────────────────
# Auto-generate /api/{key}/profiles, status, stop, start for every engine.

for _ek, _ev in _ENGINES.items():
    def _make_engine_routes(key: str, eng: dict):
        @app.get(f"/api/{key}/profiles", name=f"{key}_profiles")
        async def profiles(k=key):
            return _profiles_with_models(k)

        @app.delete(f"/api/{key}/profiles/{{profile_id}}", name=f"{key}_profile_delete",
                    dependencies=[Depends(verify_auth)])
        async def delete_profile(profile_id: str, k=key):
            """Delete a profile's start script. Leaves model weights untouched."""
            return _delete_profile_script(k, profile_id)

        @app.get(f"/api/{key}/status", name=f"{key}_status")
        async def status(k=key, e=eng):
            return await _engine_status(
                _engine_bases[k], e.get("docker_filter", k),
                e.get("health_path", "/health"), e.get("models_path"))

        @app.post(f"/api/{key}/stop", name=f"{key}_stop",
                  dependencies=[Depends(verify_auth)])
        async def stop(k=key, e=eng):
            return await _engine_stop(_engine_bases[k], e["name"],
                                      e.get("docker_filter", k))

        @app.post(f"/api/{key}/start", name=f"{key}_start",
                  dependencies=[Depends(verify_auth)])
        async def start(req: EngineStartRequest, k=key):
            return await _engine_start(req.profile, lambda kk=k: _scan_profiles(kk), k,
                                       engine_key=k, force=req.force,
                                       recipe=req.recipe if k == "llamacpp" else None,
                                       overrides=req.overrides if k == "vllm" else None)

    _make_engine_routes(_ek, _ev)


@app.get("/api/llamacpp/recipes")
async def llamacpp_recipes():
    """The quant ladder the llama.cpp profile scripts read.

    llama.cpp differs from vLLM here: one GGUF repo ships ten quants of the same weights,
    so a script-per-quant would be ten near-identical files that drift. The ladder is data.
    """
    try:
        cfg = json.loads(_CONFIG_FILE.read_text())
    except Exception as e:
        return {"recipes": {}, "default": None, "error": str(e)}
    lc = cfg.get("llamacpp", {})
    return {"recipes": lc.get("recipes", {}) or {},
            "default": lc.get("default_recipe")}

# ── HuggingFace Download ───────────────────────────────────────────────────────

_HF_DOWNLOAD_SCRIPT = """
import sys, json, os, time, fnmatch, importlib.util
sys.stdout.reconfigure(line_buffering=True)
# Define the event reporter FIRST. Anything that raises before J exists dies as a bare
# traceback on stderr, which the parent turns into `log` events the UI discards — that is
# exactly how an orphaned hf_transfer NameError here stayed invisible while every download
# failed. Keep initialization inside the guarded block below so it can never regress.
J = lambda **kw: print(json.dumps(kw), flush=True)

try:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # huggingface_hub 1.x downloads Xet-backed repos through hf_xet automatically; the
    # perf knob is HF_XET_HIGH_PERFORMANCE (raises hf_xet concurrency/throughput on big
    # shards). The old HF_HUB_ENABLE_HF_TRANSFER/hf_transfer path is superseded by Xet.
    # Must be set before importing huggingface_hub; guarded so downloads still work if
    # hf_xet is absent (falls back to the default HTTP path).
    _has_xet = importlib.util.find_spec("hf_xet") is not None
    if _has_xet:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import list_repo_tree, hf_hub_download
    from pathlib import Path

    repo = os.environ.get("HF_REPO_ID")
    if not repo:
        raise RuntimeError("HF_REPO_ID is not set")
    local_dir = os.environ.get("HF_LOCAL_DIR") or None
    ignore = json.loads(os.environ.get("HF_IGNORE_PATTERNS") or "[]")
    allow = json.loads(os.environ.get("HF_ALLOW_PATTERNS") or "[]")
    J(status="starting", repo=repo)
    if _has_xet:
        J(status="Xet acceleration enabled")
except Exception as e:
    J(status="error", error=f"startup failed: {e}")
    raise SystemExit(0)

def _keep(p):
    if allow and not any(fnmatch.fnmatch(p, g) for g in allow):
        return False
    if ignore and any(fnmatch.fnmatch(p, g) for g in ignore):
        return False
    return True

try:
    entries = [e for e in list_repo_tree(repo, recursive=True)
               if hasattr(e, 'size') and not e.path.startswith('.')]
    if ignore or allow:
        kept = [e for e in entries if _keep(e.path)]
        skipped = len(entries) - len(kept)
        skipped_gb = sum(e.size or 0 for e in entries if not _keep(e.path)) / 1024**3
        if skipped:
            J(status=f"Skipping {skipped} file(s) ({skipped_gb:.1f} GB) by pattern")
        entries = kept
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


def _safe_profile_slug(name: str) -> str:
    slug = _re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip().lower())
    slug = _re.sub(r"_+", "_", slug).strip("._-")
    return slug[:96] or "hf_model"


def _path_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _allowed_model_roots() -> list[Path]:
    roots = [HF_CACHE_DIR]
    for d in _load_custom_dirs():
        try:
            roots.append(Path(os.path.expanduser(d)))
        except Exception:
            pass
    return roots


def _looks_like_model_dir(d: Path) -> bool:
    """True only for a directory that is itself one model.

    Deliberately strict. `/opt/models/hub` is a 282 GB *cache root* full of
    models--* entries; it is a directory under an allowed root, and the token
    'hub' appears in every HF path, so a laxer test made it a delete candidate
    for a profile whose real model directory had been removed.
    """
    if d.name.startswith(".") or d.name in ("hub", "blobs", "snapshots", "refs"):
        return False
    if (d / "config.json").exists():
        return True
    if (d / "snapshots").is_dir() and not any(
            c.name.startswith("models--") for c in d.iterdir() if c.is_dir()):
        return True
    return False


def _candidate_model_dirs() -> list[Path]:
    """Every top-level model directory under the allowed roots."""
    dirs = []
    for root in _allowed_model_roots():
        try:
            dirs += [d for d in root.iterdir() if d.is_dir() and _looks_like_model_dir(d)]
        except Exception:
            continue
    return dirs


def _profiles_with_models(engine_key: str) -> list:
    """Profiles for an engine, annotated with the model dir each one launches.

    The UI needs `model_dir` to offer a weights delete next to a profile delete;
    `model_size_gb` so the confirm can state what is actually being freed.
    """
    model_dirs = _candidate_model_dirs()
    out = []
    for p in _scan_profiles(engine_key):
        d = _script_model_dir(p["script"], model_dirs)
        p["model_dir"] = str(d) if d else None
        p["model_missing"] = bool(d is None)
        if d:
            blobs = d / "blobs"
            src = blobs if blobs.exists() else d
            try:
                p["model_size_gb"] = round(
                    sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e9, 1)
            except Exception:
                p["model_size_gb"] = None
        else:
            p["model_size_gb"] = None
        out.append(p)
    return out


def _delete_profile_script(engine_key: str, profile_id: str) -> dict:
    """Remove a start_*.sh for an engine, by profile id (the script stem)."""
    d = _engine_dirs.get(engine_key)
    if not d:
        raise HTTPException(404, "Unknown engine")
    # Reject traversal: the id is a stem, never a path.
    if "/" in profile_id or "\\" in profile_id or profile_id.startswith("."):
        raise HTTPException(400, "Invalid profile id")
    script = (d / f"{profile_id}.sh").resolve()
    try:
        script.relative_to(d.resolve())
    except ValueError:
        raise HTTPException(400, "Profile is outside the engine directory")
    if not script.exists():
        raise HTTPException(404, "Profile not found")
    script.unlink()
    _script_content_cache.pop(str(script), None)
    _logger.info("Deleted %s profile script %s", engine_key, script.name)
    return {"ok": True, "deleted": str(script)}


def _find_launch_dir(path: Path) -> Path:
    """Accept an HF model dir, snapshot dir, or flat model dir and return the launch dir."""
    if (path / "config.json").exists():
        return path
    snaps = path / "snapshots"
    if snaps.exists():
        candidates = [s for s in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                      if s.is_dir() and (s / "config.json").exists()]
        if candidates:
            return candidates[0]
    raise HTTPException(400, "Could not find a launchable snapshot with config.json")


def _profile_model_info(launch_dir: Path, requested_name: str | None = None) -> dict:
    config = {}
    try:
        config = json.loads((launch_dir / "config.json").read_text())
    except Exception:
        pass
    model_name = requested_name or launch_dir.name
    hf_root = launch_dir
    for parent in launch_dir.parents:
        if parent.name.startswith("models--"):
            hf_root = parent
            tail = parent.name[8:]
            parts = tail.split("--", 1)
            if len(parts) == 2:
                model_name = f"{parts[0]}/{parts[1]}"
            break
    fmt = _detect_format(hf_root, is_hf_cache=hf_root.name.startswith("models--"))
    hints = _infer_from_name(model_name.split("/")[-1])
    inferred = _infer_from_config(config, hints)
    task_label = _task_from_modalities(inferred.get("modalities", ["Text"]))
    if hf_root.name.startswith("models--") and (hf_root / "blobs").exists():
        try:
            size_gb = round(sum(f.stat().st_size for f in (hf_root / "blobs").iterdir() if f.is_file()) / 1e9, 1)
        except Exception:
            size_gb = _dir_size_gb(hf_root)
    else:
        size_gb = _dir_size_gb(launch_dir)
    # VRAM estimate = weights (≈ on-disk size, scaled by how much the format
    # expands when loaded) + a fixed overhead for CUDA context, activations and
    # the fp8 KV cache. Already-quantized weights load ~1:1, so the old flat
    # 1.35× over-inflated every 4-/8-bit model by ~35% and tripped admission on
    # a model that actually fits (e.g. gpt-oss-120b FP4: 100→81 GB).
    _dtype = (inferred.get("dtype") or "").upper()
    _weight_mult = (
        1.05 if _dtype in ("FP4", "INT4") else
        1.15 if _dtype in ("FP8", "INT8") else
        1.35  # BF16/FP16/Unknown: fp16 weights + workspace headroom
    )
    vram_gb = int(min(112, max(16, round((size_gb * _weight_mult) + 12))))
    return {
        "name": model_name,
        "served": model_name.replace("/", "--"),
        "fmt": fmt,
        "dtype": inferred.get("dtype") or "Unknown",
        "is_moe": inferred.get("is_moe"),
        "modalities": inferred.get("modalities", ["Text"]),
        "task_label": task_label,
        "size_gb": size_gb,
        "vram_gb": vram_gb,
        "architectures": (config.get("architectures", [])
                          if isinstance(config.get("architectures", []), list) else []),
    }


def _container_model_mount(launch_dir: Path, slug: str) -> tuple[list[str], str]:
    hf_cache_parent = HF_CACHE_DIR.parent.resolve()  # ~/.cache/huggingface
    if _path_under(launch_dir, hf_cache_parent):
        rel = launch_dir.resolve().relative_to(hf_cache_parent)
        # Deliberately NOT shlex.quote'd: this line is a constant, and it relies on the
        # shell expanding $HOME at launch time. Quoting would make $HOME a literal and
        # break every cache-backed profile. No untrusted value is interpolated here.
        return [
            f'  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\',
        ], "/root/.cache/huggingface/" + str(rel).replace(os.sep, "/")
    # HF-layout repo outside HF_CACHE_DIR (i.e. under a custom dir such as
    # /opt/models or /mnt/models). Entries in snapshots/<rev>/ are relative
    # symlinks to ../../blobs/<hash>, so mounting the snapshot dir alone leaves
    # every file dangling inside the container and vLLM aborts with
    # "Invalid repository ID or local directory specified". Mount the whole
    # models--* repo (blobs/ + snapshots/) and point --model at the subpath.
    resolved = launch_dir.resolve()
    for repo_root in (resolved, *resolved.parents):
        if (repo_root.name.startswith("models--")
                and (repo_root / "blobs").is_dir()
                and (repo_root / "snapshots").is_dir()):
            rel = resolved.relative_to(repo_root)
            container_root = f"/models/{slug}"
            container_model = container_root
            if rel.parts:
                container_model += "/" + rel.as_posix()
            return [
                f'  -v {shlex.quote(f"{repo_root}:{container_root}:ro")} \\',
            ], container_model
    # Flat model dir: files live directly inside, safe to mount on its own.
    return [
        f'  -v {shlex.quote(f"{launch_dir}:/models/{slug}:ro")} \\',
    ], f"/models/{slug}"


# A served model name ends up as a shell word in a generated start script AND in the
# `# Name:` comment header that _parse_script_meta reads back. Anything outside this set is
# rejected outright: grammar first, shlex.quote second, because defence-in-depth here is
# cheap and the blast radius is arbitrary code execution as the service user.
_SERVED_NAME_RE = _re.compile(r"^[A-Za-z0-9._/-]{1,128}$")
_MOE_BACKEND_RE = _re.compile(r"^[a-z0-9_]{1,64}$")


def _validate_served_name(name: str) -> str:
    """Reject any model name that could break out of a shell word or a comment line."""
    if not isinstance(name, str) or not _SERVED_NAME_RE.match(name):
        raise HTTPException(
            400,
            "Invalid model name. Allowed characters: letters, digits, dot, underscore, "
            "hyphen and forward slash (max 128 chars).")
    return name


def _one_line(value: str) -> str:
    """Flatten a value destined for a single-line `# Header:` comment."""
    return _re.sub(r"[\r\n]+", " ", str(value)).strip()


def _vllm_serve_command(image: str, cfg: dict) -> str:
    """Return the serve subcommand this image needs, or "" if its entrypoint serves already.

    Entrypoint semantics belong to the image, not to the model. Two shapes verified on this
    box:

      eugr/spark-vllm:latest → Entrypoint ["/opt/nvidia/nvidia_entrypoint.sh"], Cmd null.
                               That script execs its arguments, so without an explicit
                               `vllm serve` the container tries to exec `--model` and dies.
      vllm/vllm-openai:*     → its entrypoint already starts the OpenAI API server, so
                               arguments are server flags and `vllm serve` must NOT be added.

    The known-good hand-written profiles/vLLM/start_hf_qwen_qwen3-8b.sh uses exactly this
    form on the eugr image. Deliberately NOT using `docker run --entrypoint vllm`: that
    bypasses nvidia_entrypoint.sh, which does CUDA environment setup inside the image.

    Unknown images default to the explicit form, which is the safe direction — `vllm serve`
    also works when passed as the container command to an image that execs its arguments.
    """
    if "serve_command" in (cfg or {}):
        # Explicit wins, including an empty string meaning "add nothing" (same idiom as
        # vllm.moe_backend).
        val = (cfg or {}).get("serve_command")
        return "" if val is None else str(val).strip()
    ref = (image or "").split("@")[0]  # drop any @sha256: digest before matching
    return "" if "vllm-openai" in ref else "vllm serve"


def _resolve_recipe_model(model_name: str, vllm_cfg: dict) -> tuple[Optional[str], list[str]]:
    """Choose the deterministic most-specific configured recipe glob."""
    recipes = vllm_cfg.get("recipes") if isinstance(vllm_cfg, dict) else None
    if not isinstance(recipes, dict) or not recipes:
        return None, []
    folded_name = str(model_name).casefold()
    matches = []
    for pattern, recipe_name in recipes.items():
        if not isinstance(pattern, str):
            continue
        if fnmatch.fnmatchcase(folded_name, pattern.casefold()):
            matches.append((pattern, recipe_name))
    if not matches:
        return None, []

    def _specificity(item):
        pattern = item[0]
        wildcard_count = sum(pattern.count(char) for char in "*?[")
        literal_length = sum(char not in "*?[]" for char in pattern)
        return wildcard_count, -literal_length, pattern.casefold(), pattern

    pattern, recipe_name = min(matches, key=_specificity)
    warnings = []
    if len(matches) > 1:
        warnings.append(
            f"multiple recipe patterns match {model_name!r}; chose {pattern!r}")
    if not isinstance(recipe_name, str) or not _re.fullmatch(r"[\w.-]+", recipe_name):
        warnings.append(f"invalid recipe name {recipe_name!r} for pattern {pattern!r}")
        return None, warnings
    return recipe_name, warnings


def _kv_dtype_from_config(config: dict) -> Optional[str]:
    """Return fp8 only for an explicit model-owned floating 8-bit KV declaration."""
    if not isinstance(config, dict):
        return None
    candidates = [config]
    if isinstance(config.get("text_config"), dict):
        candidates.append(config["text_config"])
    for candidate in candidates:
        quant = candidate.get("quantization_config")
        if not isinstance(quant, dict):
            continue
        if str(quant.get("kv_cache_dtype", "")).casefold() == "fp8":
            return "fp8"
        scheme = quant.get("kv_cache_scheme")
        if not isinstance(scheme, dict):
            continue
        try:
            eight_bit = int(scheme.get("num_bits")) == 8
        except (TypeError, ValueError):
            eight_bit = False
        kind = " ".join(str(scheme.get(key, "")).casefold()
                        for key in ("type", "dtype"))
        if eight_bit and ("float" in kind or "fp8" in kind):
            return "fp8"
    return None


def _resolve_launch(config: dict, info: dict, vllm_cfg: dict) -> dict:
    """Resolve recipe delegation or derived Docker flags for one model."""
    name = str(info.get("name", ""))
    is_gpt_oss = "gpt-oss" in name.lower() or "gpt_oss" in name.lower()
    if is_gpt_oss:
        return {"shape": "docker", "gpt_oss": True}

    recipe_name, warnings = _resolve_recipe_model(name, vllm_cfg)
    if recipe_name is not None:
        res_dir = vllm_cfg.get("recipe_dir") or "~/spark-vllm-docker/recipes"
        record, reader_warnings = _read_recipe(res_dir, recipe_name)
        warnings.extend(reader_warnings)
        if record is not None:
            if record.get("cluster_only") is True:
                warnings.append(f"recipe {recipe_name!r} is cluster_only")
            if record.get("port") is not None and record["port"] != 8000:
                warnings.append(
                    f"recipe {recipe_name!r} uses port {record['port']}, not 8000")
            return {
                "shape": "recipe", "recipe": recipe_name,
                "record": record, "warnings": warnings,
            }

    kv_dtype = _kv_dtype_from_config(config)
    spec = _derive_launch_spec(
        config, weights_gb=info.get("size_gb", 0.0), pool_gb=121.0,
        kv_dtype_bytes=1 if kv_dtype == "fp8" else 2)
    warnings.extend(spec.get("warnings") or [])
    max_model_len = spec.get("max_model_len")
    if not isinstance(max_model_len, (int, float)) or max_model_len <= 0:
        max_model_len = None
    else:
        max_model_len = int(max_model_len)
    util = spec.get("recommended_util")
    if not isinstance(util, (int, float)) or util <= 0.10:
        util = None
    else:
        util = float(util)
    return {
        "shape": "docker", "kv_dtype": kv_dtype, "util": util,
        "max_model_len": max_model_len, "warnings": warnings,
        # The full spec is threaded through so `_vllm_script_preamble` can emit the
        # `# Derived:` header without re-deriving. Historically this function dropped
        # fourteen of the spec's keys on the floor and the card had nothing to show.
        "spec": spec,
    }


# The `# Derived:` header is a versioned data format read back by `_parse_script_meta`.
# Bump when the shape changes; readers must tolerate an unknown version.
_SCRIPT_META_VERSION = 1

# Placeholder fallbacks used when the spec could not derive a number. These are the same
# constants the `${VAR:-N}` emission used inline before they were hoisted here — hoisting
# is what makes "one source of truth, two renderings" literally true.
_GPT_OSS_MAX_MODEL_LEN = 65536
_FALLBACK_MAX_MODEL_LEN = 32768
_FALLBACK_UTIL = 0.75
_DEFAULT_MAX_NUM_SEQS = 2


def _launch_defaults(resolved: dict) -> dict:
    """The three numbers that become the `${VAR:-N}` placeholder defaults.

    Single source of truth: the script argv and the `# Derived:` header both read this,
    so the header can never advertise a context the script would not actually use.
    """
    if resolved.get("gpt_oss"):
        # gpt-oss takes the full-precision KV path with a fixed context; `_resolve_launch`
        # short-circuits before deriving a spec for it.
        return {"max_model_len": _GPT_OSS_MAX_MODEL_LEN, "util": _FALLBACK_UTIL,
                "max_num_seqs": _DEFAULT_MAX_NUM_SEQS}
    ml = resolved.get("max_model_len")
    util = resolved.get("util")
    return {
        "max_model_len": int(ml) if ml is not None else _FALLBACK_MAX_MODEL_LEN,
        "util": float(util) if util is not None else _FALLBACK_UTIL,
        "max_num_seqs": _DEFAULT_MAX_NUM_SEQS,
    }


def _json_num(value):
    """Coerce to a JSON-representable number, or None. NaN/Inf are not valid JSON."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if value != value or value in (float("inf"), float("-inf")):
            return None
    except Exception:
        return None
    return int(value) if isinstance(value, int) else float(value)


def _script_meta_headers(info: dict, resolved: Optional[dict]) -> str:
    """Emit the four machine-readable header comments `_parse_script_meta` reads back.

    Each is a single line of compact JSON after the colon so the parser has exactly one
    shape to handle. Warning strings embed `{value!r}` of vendor-supplied config fields,
    so they are flattened AND json-escaped — a raw newline would break the contract.
    """
    if not resolved:
        return ""
    if resolved.get("shape") == "recipe":
        derived = {
            "v": _SCRIPT_META_VERSION,
            "editable": False,
            "reason": "the recipe YAML owns these flags",
        }
        recommended = {}
    else:
        defaults = _launch_defaults(resolved)
        spec = resolved.get("spec") or {}
        derived = {
            "v": _SCRIPT_META_VERSION,
            "editable": True,
            "max_model_len": defaults["max_model_len"],
            "max_fitting_context": _json_num(spec.get("max_fitting_context")),
            "declared_max_context": _json_num(spec.get("declared_max_context")),
            "max_num_seqs": defaults["max_num_seqs"],
            "util": defaults["util"],
        }
        recommended = {
            "gpu_memory_utilization": _json_num(spec.get("recommended_util")),
            "max_model_len": _json_num(spec.get("max_model_len")),
        }
    warnings = [_one_line(w) for w in (info.get("warnings") or [])]
    dump = lambda obj: json.dumps(obj, separators=(",", ":"))
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return (
        f"# Derived: {dump(derived)}\n"
        f"# Recommended: {dump(recommended)}\n"
        f"# Warnings: {dump(warnings)}\n"
        f"# Generated: {stamp}\n"
    )


def _vllm_script_preamble(info: dict, launch_dir: Path, *, recipe_backed: bool,
                          resolved: Optional[dict] = None) -> str:
    """The single shared owner of generated profile metadata and collision cleanup."""
    if recipe_backed:
        rationale = (
            "# Recipe-backed: the YAML remains the source of truth for measured flags and\n"
            "# in-container mods that a generated docker command cannot reproduce.\n")
    else:
        rationale = ""
    return f"""#!/bin/bash
# Name: HF {_one_line(info['name'])}
# Description: Local HF snapshot via vLLM ({_one_line(info['dtype'])}, {info['size_gb']:.1f} GB on disk)
# VRAM: {info['vram_gb']}
#
# Auto-generated by DGX Model Manager from:
# {_one_line(launch_dir)}
{_script_meta_headers(info, resolved)}{rationale}set -euo pipefail

docker rm -f vllm_node 2>/dev/null || true

"""


# Defence in depth behind `_OVERRIDE_ENV`'s allow-list: the placeholders expand host-side
# straight into docker argv, so a value that reached the environment by any other route
# (a stray `export`, a hand-edited unit) must abort before `docker run`, not after.
_VLLM_OVERRIDE_GUARD = r"""# Override guard — these expand into docker argv.
for _dmm_var in VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS; do
  _dmm_val="${!_dmm_var:-}"
  if [[ -n "$_dmm_val" && ! "$_dmm_val" =~ ^[0-9]+$ ]]; then
    echo "Invalid $_dmm_var: expected a positive integer, got '$_dmm_val'" >&2
    exit 2
  fi
done
if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" \
   && ! "${VLLM_GPU_MEMORY_UTILIZATION}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Invalid VLLM_GPU_MEMORY_UTILIZATION: expected a decimal, got" \
       "'${VLLM_GPU_MEMORY_UTILIZATION}'" >&2
  exit 2
fi
unset _dmm_var _dmm_val

"""


def _recipe_profile_body(recipe_dir, recipe_name: str) -> str:
    recipe_path = Path(os.path.expanduser(os.fspath(recipe_dir)))
    return f"""RECIPE_DIR={shlex.quote(str(recipe_path))}
RECIPE={shlex.quote(recipe_name)}

# recipe_dir owns the YAMLs; run-recipe.sh is its sibling in the checkout root.
cd "$RECIPE_DIR/.."
# Detached mode lets DMM read progress uniformly from docker logs.
exec ./run-recipe.sh "$RECIPE" -d
"""


def _build_vllm_profile_script(launch_dir: Path, model_name: str | None = None) -> tuple[str, str, dict]:
    info = _profile_model_info(launch_dir, model_name)
    if info["fmt"] not in ("safetensors", "pytorch"):
        raise HTTPException(400, f"Only safetensors/PyTorch snapshots can be used with vLLM; found {info['fmt']}")
    if info.get("task_label") not in ("Text Gen", "Vision LLM"):
        raise HTTPException(400, f"Only text/vision LLM snapshots can be added to vLLM; detected {info.get('task_label')}")
    # Validate the name that actually reaches the script — including one recovered from a
    # models--* ancestor, not just the caller-supplied one.
    _validate_served_name(info["name"])
    _validate_served_name(info["served"])
    slug = _safe_profile_slug(info["name"])
    script_name = f"start_hf_{slug}.sh"
    _vllm_cfg = _live_vllm_cfg()
    try:
        config = json.loads((launch_dir / "config.json").read_text())
        if not isinstance(config, dict):
            config = {}
    except Exception:
        config = {}
    resolved = _resolve_launch(config, info, _vllm_cfg)
    info["warnings"] = list(resolved.get("warnings") or [])
    if resolved["shape"] == "recipe":
        preamble = _vllm_script_preamble(
            info, launch_dir, recipe_backed=True, resolved=resolved)
        res_dir = _vllm_cfg.get("recipe_dir") or "~/spark-vllm-docker/recipes"
        return (script_name, preamble + _recipe_profile_body(
            res_dir, resolved["recipe"]), info)
    # The docker preamble is deliberately built LAST (just before script assembly):
    # capability warnings are appended to info["warnings"] further down, and the
    # `# Warnings:` header must carry the complete list, not a prefix of it.

    mounts, container_model = _container_model_mount(launch_dir, slug)
    dtype = info["dtype"]
    is_fp4 = dtype in ("FP4", "INT4") or "fp4" in info["name"].lower() or "nvfp4" in info["name"].lower()
    is_moe = bool(info["is_moe"]) or "moe" in info["name"].lower() or "a3b" in info["name"].lower()
    # gpt-oss family uses the OpenAI harmony chat format + MXFP4 weights. On GB10
    # (sm_121) the stock vllm-openai Marlin MXFP4 kernel miscomputes the first decode
    # token (vLLM #37030) → harmony parser desync → OpenWebUI token-soup. The
    # SM121-patched eugr/spark-vllm image (vLLM 0.23.1) fixes it. See
    # profiles/vLLM/start_hf_openai_gpt-oss-120b.sh for the full root-cause writeup.
    is_gpt_oss = "gpt-oss" in info["name"].lower() or "gpt_oss" in info["name"].lower()

    env_lines = [
        "  -e HF_HUB_OFFLINE=1 \\",
        "  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \\",
    ]
    if is_gpt_oss:
        # Harmony/MXFP4 on sm_121: seed the o200k vocab for offline openai_harmony,
        # force Marlin MoE with the SM121 split-K atomic-add race fix. NOT
        # VLLM_NVFP4_GEMM_BACKEND (that's for NVFP4 models and is a no-op here).
        env_lines += [
            "  -e TIKTOKEN_ENCODINGS_BASE=/root/.cache/huggingface/harmony-encodings \\",
            "  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \\",
            "  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \\",
            "  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \\",
        ]
    elif is_fp4:
        env_lines += [
            "  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \\",
            "  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \\",
            "  -e VLLM_NVFP4_GEMM_BACKEND=marlin \\",
        ]

    # shlex.quote every dynamic atom (defence-in-depth behind _validate_served_name). It
    # supplies its own quoting, so these must NOT also be wrapped in double quotes —
    # double-quoting its output would reintroduce $(...) evaluation. `vllm-active` is a
    # literal and needs none.
    # One source of truth for the three placeholder defaults; the `# Derived:` header
    # renders the same dict.
    _defaults = _launch_defaults(resolved)
    arg_lines = [
        f'  --model {shlex.quote(container_model)} \\',
        f'  --served-model-name {shlex.quote(info["name"])} {shlex.quote(info["served"])} vllm-active \\',
        "  --host 0.0.0.0 --port 8000 \\",
        "  --trust-remote-code --dtype auto \\",
        # Placeholders are deliberately NOT shlex.quote'd: the comment above concerns
        # dynamic atoms, and these must stay bash-expandable. The derived value is the
        # default, so an unset environment reproduces today's script exactly.
        f"  --gpu-memory-utilization ${{VLLM_GPU_MEMORY_UTILIZATION:-"
        f"{_defaults['util']}}} \\",
    ]
    if is_gpt_oss:
        # Full-precision KV (fp8 KV is unneeded on the 128 GB unified pool and adds
        # sampling-tail noise on the harmony path); 65536 = ~324K token KV capacity.
        arg_lines += [
            f"  --max-model-len ${{VLLM_MAX_MODEL_LEN:-{_defaults['max_model_len']}}} \\",
            f"  --max-num-seqs ${{VLLM_MAX_NUM_SEQS:-{_defaults['max_num_seqs']}}} \\",
            "  --enable-chunked-prefill \\",
        ]
    else:
        arg_lines += [
            f"  --max-model-len ${{VLLM_MAX_MODEL_LEN:-{_defaults['max_model_len']}}} \\",
            f"  --max-num-seqs ${{VLLM_MAX_NUM_SEQS:-{_defaults['max_num_seqs']}}} \\",
        ]
        if resolved.get("kv_dtype") == "fp8":
            arg_lines.append("  --kv-cache-dtype fp8 --enable-chunked-prefill \\")
        else:
            arg_lines.append("  --enable-chunked-prefill \\")
    # On GB10 (sm_121) Marlin MoE miscomputes for some architectures; setting
    # vllm.moe_backend to "" in config.json omits the flag and lets vLLM autoselect.
    _moe_backend = _vllm_cfg.get("moe_backend", "marlin")
    _moe_backend = "" if _moe_backend is None else str(_moe_backend).strip()
    if (is_moe or is_gpt_oss) and _moe_backend:
        if not _MOE_BACKEND_RE.match(_moe_backend):
            raise HTTPException(400, "Invalid vllm.moe_backend: expected lowercase letters, "
                                     "digits and underscores only.")
        arg_lines.append(f"  --moe-backend {_moe_backend} \\")
    if not is_gpt_oss:
        tool_parser, reasoning_parser, capability_warnings = (
            _capability_emission_details(_capability_entry(info)))
        info["warnings"].extend(capability_warnings)
        if tool_parser is not None:
            arg_lines += [
                "  --enable-auto-tool-choice \\",
                f"  --tool-call-parser {shlex.quote(tool_parser)} \\",
            ]
        if reasoning_parser is not None:
            arg_lines.append(
                f"  --reasoning-parser {shlex.quote(reasoning_parser)} \\")
    arg_lines.append("  --generation-config vllm")

    # Whether an explicit `vllm serve` is needed is a property of the IMAGE, not of the
    # model family. Treating it as model-specific meant every generated non-gpt-oss script
    # was unlaunchable whenever config.json pointed `vllm.image` at an exec-args image.
    if is_gpt_oss:
        image = _vllm_cfg.get("image_gpt_oss") or "eugr/spark-vllm:latest"
    else:
        image = _vllm_cfg.get("image") or "vllm/vllm-openai:v0.20.0"
    _serve_cmd = _vllm_serve_command(image, _vllm_cfg)
    if _serve_cmd:
        arg_lines.insert(0, f"  {_serve_cmd} \\")
    image = shlex.quote(str(image))

    preamble = _vllm_script_preamble(
        info, launch_dir, recipe_backed=False, resolved=resolved)
    script = preamble + _VLLM_OVERRIDE_GUARD + f"""docker run -d --name vllm_node --gpus all -p 8000:8000 \\
{chr(10).join(mounts)}
{chr(10).join(env_lines)}
  {image} \\
{chr(10).join(arg_lines)}
"""
    return script_name, script, info


def _create_vllm_profile_from_path(path: str, model_name: str | None = None) -> dict:
    raw = Path(os.path.expanduser(path.strip())).resolve()
    if not raw.exists() or not raw.is_dir():
        raise HTTPException(404, "Model directory not found")
    if not any(_path_under(raw, root) for root in _allowed_model_roots()):
        raise HTTPException(403, "Model path must be under the HF cache or a registered inventory directory")
    launch_dir = _find_launch_dir(raw)
    script_name, script, info = _build_vllm_profile_script(launch_dir, model_name)
    target_dir = _engine_dirs["vllm"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / script_name
    if target.exists():
        existing = target.read_text(errors="ignore")
        if str(launch_dir) not in existing:
            raise HTTPException(409, f"Profile script already exists with different contents: {target.name}")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(script)
    os.chmod(tmp, 0o755)
    os.replace(tmp, target)
    _logger.info("Created vLLM HF profile %s for %s", target, launch_dir)
    return {"ok": True, "profile": _parse_script_meta(target), "path": str(target), "model": info}


@app.post("/api/vllm/profiles/from-hf", dependencies=[Depends(verify_auth)])
async def create_vllm_profile_from_hf(req: CreateVLLMProfileRequest):
    return _create_vllm_profile_from_path(req.path, req.model_name)


# ─── Parameterize a legacy script (04-03) ─────────────────────────────────────
# Single source of truth for flag → placeholder variable: the same allow-list the
# override API validates against, so a script can never grow a placeholder the
# override path would reject as unknown.
_FLAG_TO_ENV = {
    "--max-model-len": _OVERRIDE_ENV["max_model_len"][0],
    "--gpu-memory-utilization": _OVERRIDE_ENV["gpu_memory_utilization"][0],
    "--max-num-seqs": _OVERRIDE_ENV["max_num_seqs"][0],
}


def _placeholder_re(env_name: str):
    return _re.compile(r"^\$\{" + _re.escape(env_name) + r":-[0-9]+(?:\.[0-9]+)?\}$")


class ParameterizeRefused(Exception):
    """A rewrite that would be partial. Refusing whole is the only safe answer:
    the hand-tuned scripts carry root-cause writeups and are not reconstructible."""


def _parameterize_script_text(text: str) -> tuple[str, list[str]]:
    """Rewrite literal flag values into `${VLLM_*:-<literal>}` placeholders.

    Pure and total: the original literal becomes the default, so an unset
    environment reproduces the current script's behaviour byte-for-byte in argv.
    Idempotent — a token that is already the exact placeholder is left alone.
    Raises `ParameterizeRefused` if any target flag is present-but-not-literal or
    absent entirely; a half-rewritten hand-tuned script is worse than none.
    """
    notes: list[str] = []
    rewrote: dict[str, str] = {}
    already: set = set()

    lines = (text or "").splitlines(keepends=True)
    out_lines = []
    for raw in lines:
        if raw.strip().startswith("#"):
            out_lines.append(raw)
            continue

        def _sub(match):
            flag, token = match.group(1), match.group(2)
            env = _FLAG_TO_ENV[flag]
            if _placeholder_re(env).match(token):
                already.add(flag)
                return match.group(0)
            if not _NUM_RE.match(token):
                raise ParameterizeRefused(
                    f"{flag} value {token!r} is not a literal number — refusing to "
                    f"rewrite this script (it would be a partial rewrite)")
            if flag in rewrote and rewrote[flag] != token:
                raise ParameterizeRefused(
                    f"{flag} appears twice with different values "
                    f"({rewrote[flag]!r} and {token!r}) — refusing")
            rewrote[flag] = token
            return f"{flag} ${{{env}:-{token}}}"

        out_lines.append(_FLAG_RE.sub(_sub, raw))

    missing = [f for f in _FLAG_TO_ENV if f not in rewrote and f not in already]
    if missing:
        raise ParameterizeRefused(
            "script does not set " + ", ".join(sorted(missing)) +
            " as a literal value — refusing rather than guessing a default")

    for flag in sorted(rewrote):
        notes.append(f"{flag} {rewrote[flag]} → ${{{_FLAG_TO_ENV[flag]}:-{rewrote[flag]}}}"
                     " (same value; only the default changes source)")
    for flag in sorted(already):
        notes.append(f"{flag} is already parameterized — left unchanged")
    return "".join(out_lines), notes


_PROFILE_ID_RE = _re.compile(r"^start_[A-Za-z0-9._-]+$")


def _resolve_profile_script(profile_id: str) -> Path:
    """profile_id → the script Path, with no way out of the profile directory."""
    if not _PROFILE_ID_RE.match(profile_id or ""):
        raise HTTPException(400, "Invalid profile id")
    target = (_engine_dirs["vllm"] / f"{profile_id}.sh")
    if not _path_under(target, _engine_dirs["vllm"]) or not target.is_file():
        raise HTTPException(404, "Profile script not found")
    return target


def _parameterize_preview(profile_id: str) -> dict:
    target = _resolve_profile_script(profile_id)
    current = target.read_text()
    try:
        proposed, notes = _parameterize_script_text(current)
    except ParameterizeRefused as exc:
        raise HTTPException(422, str(exc))
    # Built server-side on purpose: the client never holds or resends script text,
    # so the apply path cannot be fed attacker-authored content (T-04-13).
    diff = "".join(difflib.unified_diff(
        current.splitlines(keepends=True), proposed.splitlines(keepends=True),
        fromfile=f"a/{target.name}", tofile=f"b/{target.name}"))
    return {
        "profile_id": profile_id,
        "current": current,
        "proposed": proposed,
        "diff": diff,
        "changed": proposed != current,
        "notes": notes,
        "sha256": hashlib.sha256(current.encode()).hexdigest(),
    }


class ParameterizeApplyRequest(BaseModel):
    sha256: str


def _parameterize_apply(profile_id: str, expect_sha: str) -> dict:
    target = _resolve_profile_script(profile_id)
    current = target.read_text()
    actual = hashlib.sha256(current.encode()).hexdigest()
    if not expect_sha or not hmac.compare_digest(actual, expect_sha.strip()):
        # TOCTOU guard (T-04-12): ~3 concurrent sessions share this directory, so
        # "the preview I showed you" and "the file on disk" are not the same claim.
        raise HTTPException(409, (
            "Profile script changed since the preview was generated "
            f"(expected sha256 {expect_sha[:12]}…, on disk {actual[:12]}…). "
            "Nothing was written — re-run the preview."))
    try:
        proposed, notes = _parameterize_script_text(current)
    except ParameterizeRefused as exc:
        raise HTTPException(422, str(exc))

    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(current)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(proposed)
    os.chmod(tmp, 0o755)
    os.replace(tmp, target)
    _logger.info("Parameterized profile %s (backup %s)", target, backup.name)
    return {"ok": True, "profile_id": profile_id, "backup": str(backup),
            "notes": notes, "sha256": hashlib.sha256(proposed.encode()).hexdigest(),
            "profile": _parse_script_meta(target)}


@app.post("/api/vllm/profiles/{profile_id}/parameterize/preview",
          dependencies=[Depends(verify_auth)])
async def vllm_parameterize_preview(profile_id: str):
    return _parameterize_preview(profile_id)


@app.post("/api/vllm/profiles/{profile_id}/parameterize/apply",
          dependencies=[Depends(verify_auth)])
async def vllm_parameterize_apply(profile_id: str, req: ParameterizeApplyRequest):
    return _parameterize_apply(profile_id, req.sha256)


_active_downloads: set = set()  # (repo_id, local_dir) of in-progress HF downloads


def _is_terminal_hf_event(ev: dict) -> bool:
    """A download stream is finished once the worker reports complete or error."""
    return isinstance(ev, dict) and ev.get("status") in ("complete", "error")


# Cap on the stderr tail echoed back to the client. A traceback storm must not blow up the
# SSE frame, and only the worker's own stderr is ever echoed — never sub_env, which carries
# HF tokens and the rest of the process environment.
_HF_STDERR_TAIL_CHARS = 2000


async def _hf_download_events(sub_env: dict, repo_id: str,
                              dl_key: tuple) -> AsyncGenerator[str, None]:
    """Run the HF download worker and yield SSE frames, exactly one of them terminal.

    Module-level (rather than nested in the route) so the terminal-event contract is
    directly testable without a live HTTP request. The contract: every stream ends with
    exactly one `complete` or `error` event. A worker that dies before emitting one — the
    failure mode that hid the orphaned hf_transfer NameError — gets one synthesized here
    from its return code and stderr tail, so a dead download can never look in-progress.
    """
    proc = None
    saw_terminal = False
    stderr_tail = ""
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
            if not line:
                continue
            # Parse first, so a malformed line can't masquerade as an auto-profile failure.
            try:
                ev = json.loads(line)
            except Exception:
                yield f"data: {line}\n\n"
                continue
            if _is_terminal_hf_event(ev):
                if saw_terminal:
                    # Never emit a second terminal event; demote it to a log line.
                    yield f"data: {json.dumps({'log': line})}\n\n"
                    continue
                saw_terminal = True
            yield f"data: {line}\n\n"
            if (ev.get("status") == "complete"
                    and int(ev.get("errors", 0) or 0) == 0 and ev.get("path")):
                try:
                    profile_result = _create_vllm_profile_from_path(str(ev["path"]), repo_id)
                    yield f"data: {json.dumps({'auto_profile': profile_result})}\n\n"
                except Exception as profile_exc:
                    yield f"data: {json.dumps({'auto_profile_error': str(profile_exc)})}\n\n"
        stderr_data = await proc.stderr.read()  # type: ignore[union-attr]
        decoded = stderr_data.decode(errors="replace")
        stderr_tail = "\n".join(
            ln.strip() for ln in decoded.split("\n") if ln.strip()
        )[-_HF_STDERR_TAIL_CHARS:]
        for line in decoded.split("\n"):
            stripped = line.strip()
            if stripped and "%" not in stripped and "it/s" not in stripped:
                yield f"data: {json.dumps({'log': stripped})}\n\n"
        rc = await proc.wait()
        if not saw_terminal:
            # A silent worker is a failed worker, whatever its return code says.
            msg = f"Download worker exited with code {rc} without reporting a result"
            if stderr_tail:
                msg += f": {stderr_tail}"
            saw_terminal = True
            yield f"data: {json.dumps({'status': 'error', 'error': msg, 'returncode': rc})}\n\n"
    except Exception as e:
        if not saw_terminal:
            saw_terminal = True
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
    finally:
        # Client disconnect (GeneratorExit) also lands here: terminate the
        # child so it can't orphan and block on a full stdout pipe, and free
        # the repo so a later download can start. A terminal event cannot be
        # yielded from here after GeneratorExit, which is why the synthesized
        # one belongs in the normal path above.
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        _active_downloads.discard(dl_key)


@app.post("/api/hf/download", dependencies=[Depends(verify_auth)])
async def hf_download(req: HFDownloadRequest):
    repo_id = req.repo_id.strip()
    _logger.info("HF download requested: %s", repo_id)
    if not _HF_REPO_RE.match(repo_id):
        raise HTTPException(400, "Invalid repo ID format. Expected: owner/model-name")

    local_dir = (req.local_dir or "").strip()
    if local_dir and ("\0" in local_dir or "\n" in local_dir):
        raise HTTPException(400, "Invalid characters in local directory path")
    if local_dir:
        _target = Path(os.path.expanduser(local_dir)).resolve()
        _allowed = [HF_CACHE_DIR.resolve()] + [
            Path(os.path.expanduser(d)).resolve() for d in _load_custom_dirs()
        ]
        if not any(_target == r or _target.is_relative_to(r) for r in _allowed):
            raise HTTPException(403, "Download directory must be the HF cache or a registered custom directory")

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
    if req.ignore_patterns:
        sub_env["HF_IGNORE_PATTERNS"] = json.dumps(req.ignore_patterns)
    if req.allow_patterns:
        sub_env["HF_ALLOW_PATTERNS"] = json.dumps(req.allow_patterns)

    # Reject a duplicate download of the same target while one is in progress.
    # Two writers to the same HF cache blob corrupt/stall each other — this is
    # easy to trigger by clicking Approve twice or reloading the stream.
    dl_key = (repo_id, local_dir)
    if dl_key in _active_downloads:
        raise HTTPException(409, f"A download of {repo_id} is already in progress")
    _active_downloads.add(dl_key)

    return StreamingResponse(
        _hf_download_events(sub_env, repo_id, dl_key), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _all_profiles() -> list:
    """(profile, engine_name) pairs across every engine, for script cross-refs."""
    _script_content_cache.clear()
    pairs = []
    for ek, ev in _ENGINES.items():
        pairs += [(p, ev["name"]) for p in _scan_profiles(ek)]
    return pairs


@app.get("/api/hf/inventory")
async def hf_inventory():
    """Scan HF cache + custom dirs and return model inventory."""
    # Build profile list once for all models (avoids re-scanning per model)
    all_profiles = _all_profiles()

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
    all_profiles = _all_profiles()

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
    force: bool = False


async def _model_dir_in_use(target: Path) -> Optional[str]:
    """Return the name of a running container serving files under `target`, or None.

    A delete can pull weights out from under a live engine: it keeps serving from
    page cache and then fails at the next load with an error pointing nowhere near
    this endpoint.

    Docker metadata alone is not a reliable signal. The GB10 vLLM container runs
    `sleep infinity` with the whole HF cache bind-mounted and the model launched
    inside it, so neither Cmd nor Mounts names the model. Ask each engine what it
    is actually serving.
    """
    # models--owner--name → "owner/name"
    stem = target.name
    if stem.startswith("models--"):
        parts = stem[8:].split("--", 1)
        served_candidates = {"/".join(parts).lower(), parts[-1].lower()}
    else:
        served_candidates = {stem.lower()}

    for key, eng in _ENGINES.items():
        models_path = eng.get("models_path")
        if not models_path:
            continue
        try:
            r = await _http.get(_engine_bases[key] + models_path, timeout=3.0)
            ids = [d.get("id", "") for d in r.json().get("data", [])]
        except Exception:
            continue
        for mid in ids:
            if mid.lower() in served_candidates or mid.lower().endswith("/" + stem.lower()):
                return f"{eng['name']} (serving {mid})"

    # Secondary: an explicit path in a container's argv (non-`sleep` launches).
    result = await _run("docker", "ps", "--format", "{{.Names}}", timeout=5)
    for name in [n.strip() for n in result.stdout.strip().splitlines() if n.strip()]:
        insp = await _run("docker", "inspect", name,
                          "--format", "{{json .Config.Cmd}}", timeout=5)
        if insp.returncode == 0 and target.name in insp.stdout:
            return name
    return None


@app.post("/api/hf/inventory/delete", dependencies=[Depends(verify_auth)])
async def delete_inventory_model(req: DeleteModelRequest):
    """Delete a downloaded model directory from disk."""
    target = Path(os.path.expanduser(req.path.strip())).resolve()

    # Safety: only allow deletion under HF cache or known custom dirs
    allowed_roots = [HF_CACHE_DIR.resolve()]
    for d in _load_custom_dirs():
        allowed_roots.append(Path(os.path.expanduser(d)).resolve())
    # A root is not a model. `relative_to(root)` succeeds for root itself, so
    # without this an allowed root passes every check below and rmtree takes the
    # entire cache — 282 GB in the case of /opt/models/hub.
    if target in allowed_roots:
        raise HTTPException(400, "Refusing to delete a model root directory")

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
    if not _looks_like_model_dir(target):
        raise HTTPException(400, f"'{target.name}' is not a single model directory")
    if not target.exists():
        raise HTTPException(404, "Directory not found")
    if not target.is_dir():
        raise HTTPException(400, "Path is not a directory")

    # Never delete weights a running engine is serving, even with force.
    in_use = await _model_dir_in_use(target)
    if in_use:
        raise HTTPException(
            409, f"Model is in use by running container '{in_use}' — stop it first")

    # A profile script pointing at deleted weights fails only at next launch, so
    # surface the cross-reference here and require an explicit override.
    if not req.force:
        stem = target.name
        if stem.startswith("models--"):
            stem = stem[8:].split("--", 1)[-1]
        has_script, engine = _check_script_xref(stem, _all_profiles())
        if has_script:
            raise HTTPException(
                409, f"A {engine} profile script references this model — "
                     f"delete the profile first, or re-send with force")

    try:
        shutil.rmtree(target)
        return {"ok": True, "deleted": str(target)}
    except PermissionError:
        pass
    except Exception as e:
        raise HTTPException(500, f"Failed to delete: {e}")

    # Models pulled by a root-run downloader land root-owned (everything under
    # /opt/models is), and the service runs as the login user. Every safety check
    # above has already passed by this point; `_run` takes argv, so no shell.
    r = await _run("sudo", "-n", "rm", "-rf", "--", str(target), timeout=300)
    if r.returncode != 0:
        raise HTTPException(
            500, f"'{target.name}' is owned by another user and passwordless sudo "
                 f"is unavailable: {(r.stdout + r.stderr).strip()[:200]}")
    if target.exists():
        raise HTTPException(500, f"Delete reported success but {target} still exists")
    _logger.info("Deleted model dir %s (via sudo)", target)
    return {"ok": True, "deleted": str(target), "sudo": True}

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
    # LiteLLM restart — use `sudo -ln` to check permission without executing.
    # Matches the exact command setup.sh grants in /etc/sudoers.d/model-manager-litellm.
    try:
        r = await _run("sudo", "-ln", "/bin/systemctl", "restart", "litellm", timeout=5)
        if r.returncode != 0:
            # Fall back to blanket passwordless check
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
        _tmp = _CONFIG_FILE.with_suffix(".json.tmp")
        _tmp.write_text(json.dumps(cfg, indent=2))
        os.replace(_tmp, _CONFIG_FILE)
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
    health_paths = {"ollama": "/api/tags", "litellm": "/v1/models"}
    for key, eng in _ENGINES.items():
        health_paths[key] = eng.get("health_path", "/health")
    _validate_service_url(req.url, req.type)
    path = health_paths.get(req.type, "/health")
    url = req.url.rstrip("/") + path
    try:
        t0 = _time.monotonic()
        r = await _http.get(url, timeout=5.0)
        latency_ms = round((_time.monotonic() - t0) * 1000)
        if r.status_code < 400 or r.status_code in (401, 403):
            return {"ok": True, "latency_ms": latency_ms, "auth_required": r.status_code in (401, 403)}
        return {"ok": False, "latency_ms": latency_ms, "error": f"HTTP {r.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "error": "Connection refused"}
    except httpx.ConnectTimeout:
        return {"ok": False, "error": "Connection timed out"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── Debug & Logs ─────────────────────────────────────────────────────────────

@app.get("/api/debug/system", dependencies=[Depends(verify_auth)])
async def debug_system():
    """Comprehensive system overview for diagnostics."""
    # Service health checks with response time (parallel)
    async def _check(name, base, path):
        try:
            t0 = _time.monotonic()
            r = await _http.get(base + path, timeout=3.0)
            ms = round((_time.monotonic() - t0) * 1000)
            return name, {"url": base, "healthy": r.status_code < 400 or r.status_code in (401, 403),
                          "response_ms": ms, "auth_required": r.status_code in (401, 403)}
        except Exception:
            return name, {"url": base, "healthy": False, "response_ms": None}

    check_coros = [
        _check("ollama", OLLAMA_BASE, "/api/tags"),
        _check("litellm", LITELLM_BASE, "/v1/models"),
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


@app.get("/api/debug/config", dependencies=[Depends(verify_auth)])
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


@app.get("/api/logs/app", dependencies=[Depends(verify_auth)])
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


@app.get("/api/logs/engine/{engine}", dependencies=[Depends(verify_auth)])
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


@app.get("/api/logs/litellm", dependencies=[Depends(verify_auth)])
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


@app.get("/api/debug/docker", dependencies=[Depends(verify_auth)])
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

/* ── Header memory gauge ── */
.hdr-mem{
  display:none;align-items:center;gap:8px;
  padding:3px 10px;
  border-radius:6px;
  border:1px solid var(--border);
  background:var(--s2);
  font-family:var(--mono);
  font-size:10px;
  color:var(--text);
  text-decoration:none;
  white-space:nowrap;
  transition:border-color .2s;
}
a.hdr-mem[href]:hover{border-color:var(--amber)}
.hdr-mem-pct{color:var(--muted);margin-left:6px}
.hdr-mem-pct.crit{color:var(--red)}
.hdr-mem svg{display:block;flex:none;width:72px;height:20px}
.mem-spark-area{fill:var(--amber);opacity:.16;stroke:none}
.mem-spark-line{fill:none;stroke:var(--amber);stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round;opacity:.9}
.mem-spark-threshold{stroke:var(--red);stroke-width:1;stroke-dasharray:2 3;opacity:.6}
.mem-spark-dot{fill:var(--amber)}

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

/* ── Dashboards (site link cards reuse model-card) ── */
a.model-card{text-decoration:none;color:inherit;cursor:pointer}
a.model-card:hover{border-color:var(--amber)}
.site-dot{
  width:6px;height:6px;border-radius:50%;flex-shrink:0;
  background:var(--s2);border:1px solid var(--border2);
  transition:background .3s,box-shadow .3s;
}
.site-dot.ok{background:var(--green);border-color:var(--green);box-shadow:0 0 5px var(--green)}
.site-dot.err{background:var(--red);border-color:var(--red)}

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
/* Phase line above the bar: the bar alone cannot distinguish a 30s weight load
   from a 65s torch.compile, and that ambiguity is what made a dead container
   look like a slow one. */
.prog-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:6px}
.prog-phase{font-size:12px;font-weight:600;color:var(--text)}
.prog-meta{font-family:var(--mono);font-size:11px;color:var(--muted)}
.preflight-wrap{margin-top:12px}
.preflight-wrap:empty{display:none}
.preflight-head{
  display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  font-size:12px;font-weight:600;padding:8px 10px;border-radius:5px 5px 0 0;
  border:1px solid var(--border);border-bottom:none;background:var(--s2);
}
.preflight-head.pf-ok{color:var(--green)}
.preflight-head.pf-warn{color:var(--amber)}
.preflight-head.pf-fail{color:var(--red)}
.preflight-budget{font-family:var(--mono);font-size:11px;font-weight:400;color:var(--muted)}
.preflight-row{
  border:1px solid var(--border);border-top:none;padding:8px 10px;
  border-left:3px solid var(--border);background:#04040a;
}
.preflight-row:last-child{border-radius:0 0 5px 5px}
.preflight-row.pf-ok{border-left-color:var(--green)}
.preflight-row.pf-warn{border-left-color:var(--amber)}
.preflight-row.pf-fail{border-left-color:var(--red)}
.preflight-row.pf-skip{border-left-color:var(--muted);opacity:.65}
.pf-title{font-size:12px;font-weight:600;color:var(--text)}
.pf-detail{font-size:11px;color:var(--muted);line-height:1.6;margin-top:3px}
.pf-fix{
  font-family:var(--mono);font-size:11px;color:var(--amber);
  margin-top:5px;white-space:pre-wrap;word-break:break-word;
}
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
.engine-led.loading{background:var(--amber);box-shadow:0 0 8px var(--amber);animation:led-pulse 1.1s ease-in-out infinite}
@keyframes led-pulse{0%,100%{opacity:1}50%{opacity:.35}}
.engine-title{font-size:15px;font-weight:700}
.engine-model{font-size:11px;color:var(--amber);font-family:var(--mono);margin-top:2px}
.engine-footer{font-size:11px;color:var(--muted);font-family:var(--mono)}
.engine-actions{margin-left:auto;display:flex;gap:6px}

/* ── Profile list ── */
.recipe-bar{display:flex;align-items:center;gap:10px;margin-bottom:10px;padding:8px 10px;
  border:1px solid var(--border);border-radius:8px;background:var(--s2)}
.recipe-label{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.recipe-select{font-family:var(--mono);font-size:12px;color:var(--text);background:var(--s1);
  border:1px solid var(--border2);border-radius:6px;padding:5px 8px;min-width:230px}
.recipe-select:focus{outline:none;border-color:var(--amber)}
.recipe-desc{font-size:11px;color:var(--muted);line-height:1.4;flex:1}
.profile-list{display:flex;flex-direction:column;gap:6px}
/* 04-02 launch-settings panel: only the selected card shows it, so the list stays
   scannable and the inputs visibly reset when the selection moves. */
.p-settings{flex-basis:100%;display:none;margin-top:10px;padding-top:10px;border-top:1px solid var(--border);cursor:default}
.profile-item.selected .p-settings{display:block}
.p-set-row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end}
.p-set-field{display:flex;flex-direction:column;gap:3px}
.p-set-field label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
.p-set-field input{
  width:110px;background:var(--s0);border:1px solid var(--border);border-radius:5px;
  color:var(--fg);font-family:var(--mono);font-size:11px;padding:4px 6px;
}
.p-set-field input:disabled{opacity:.45}
.p-set-rec{font-size:10px;color:var(--amber);font-family:var(--mono)}
.p-set-note{font-size:10px;color:var(--dim);margin-top:6px}
.p-set-warn{font-size:10px;color:var(--amber);margin-top:6px;font-family:var(--mono)}
.p-set-error{font-size:11px;color:var(--red,#e05252);margin-top:6px;font-weight:600}
.profile-item{
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;
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
/* Hidden until row hover: destructive controls should not sit under the cursor
   on a list whose primary action is selecting a profile to launch. */
.subtabs{display:flex;align-items:center;gap:4px;margin:0 0 10px;border-bottom:1px solid var(--border)}
.subtab{
  background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);
  font-family:inherit;font-size:12px;font-weight:600;padding:7px 12px;cursor:pointer;
}
.subtab:hover{color:var(--fg)}
.subtab.active{color:var(--amber);border-bottom-color:var(--amber)}
.subtab-note{margin-left:auto;font-size:11px;color:var(--dim);font-family:var(--mono)}
.p-actions{display:flex;gap:4px;flex-shrink:0;margin-left:8px;opacity:0;transition:opacity .12s}
.profile-item:hover .p-actions,.profile-item.selected .p-actions{opacity:1}

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

/* ── Warm Models ── */
.warm-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.warm-panel{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:14px}
.warm-panel.full{grid-column:1/-1}
.warm-title{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
.warm-title-main{font-size:13px;font-weight:700;color:var(--text)}
.warm-sub{font-size:11px;color:var(--muted);line-height:1.5}
.warm-metric{font-family:var(--mono);font-size:22px;font-weight:700;color:var(--text);margin-bottom:6px}
.warm-bar{height:10px;border-radius:999px;background:var(--s3);overflow:hidden;border:1px solid var(--border)}
.warm-bar-fill{height:100%;background:linear-gradient(90deg,var(--green),var(--amber));width:0%}
.warm-row{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:9px 0;border-top:1px solid rgba(51,65,85,.55)}
.warm-row:first-child{border-top:0}
.warm-name{font-size:12px;font-weight:700;color:var(--text);word-break:break-word}
.warm-meta{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:2px;word-break:break-word}
.warm-pill{font-family:var(--mono);font-size:10px;border:1px solid var(--border);border-radius:999px;padding:3px 8px;color:var(--muted);white-space:nowrap}
.warm-pill.ok{color:var(--green);border-color:rgba(34,197,94,.35);background:rgba(34,197,94,.08)}
.warm-pill.warn{color:var(--amber);border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.08)}
.warm-pill.err{color:var(--red);border-color:rgba(239,68,68,.35);background:rgba(239,68,68,.08)}
.warm-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.warm-profile-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:8px}
.warm-profile{border:1px solid var(--border);background:var(--s2);border-radius:8px;padding:10px;cursor:pointer;transition:all .12s}
.warm-profile:hover{border-color:rgba(251,191,36,.45)}
.warm-profile.active{border-color:var(--green);background:rgba(34,197,94,.08)}
.warm-profile.selected{border-color:var(--amber);background:rgba(251,191,36,.08)}
.warm-profile-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
.warm-profile-name{font-size:12px;font-weight:700;color:var(--text)}
.warm-profile-desc{font-size:11px;color:var(--muted);line-height:1.45;margin-top:5px}
@media (max-width: 900px){.warm-grid{grid-template-columns:1fr}.warm-panel.full{grid-column:auto}}

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
  <a class="hdr-mem" id="hdr-mem" target="_blank" rel="noopener" title="Unified memory">
    <svg width="72" height="20" viewBox="0 0 72 20" aria-hidden="true">
      <polygon class="mem-spark-area" id="mem-spark-area" points=""></polygon>
      <line class="mem-spark-threshold" id="mem-spark-threshold" x1="1" y1="2.8" x2="71" y2="2.8"></line>
      <polyline class="mem-spark-line" id="mem-spark-line" points=""></polyline>
      <circle class="mem-spark-dot" id="mem-spark-dot" r="2" cx="-5" cy="-5"></circle>
    </svg>
    <span><span id="mem-used">--</span><span class="hdr-mem-pct" id="mem-pct">--%</span></span>
  </a>
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
    <div class="nav-section-label">Dashboards</div>
    <div class="nav-item" id="nav-sites" onclick="switchTab('sites')">
      <span class="nav-icon">&#128202;</span>Dashboards
    </div>
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
    <div class="nav-section-label">Advisor</div>
    <div class="nav-item" id="nav-recs" onclick="switchTab('recs')">
      <span class="nav-icon">💡</span>Recommendations
      <span class="nav-badge" id="badge-recs">—</span>
    </div>
    <div class="nav-section-label">Routing</div>
    <div class="nav-item" id="nav-litellm" onclick="switchTab('litellm')">
      <span class="nav-icon">⚡</span>LiteLLM
      <span class="nav-badge" id="badge-litellm">—</span>
    </div>
    <div class="nav-section-label">Engines</div>
""" + "".join(f'    <div class="nav-item" id="nav-{k}" onclick="switchTab(\'{k}\')">\n      <span class="nav-icon">{e["icon"]}</span>{e["name"]}\n    </div>\n' for k, e in _ENGINES.items()) + r"""
    <div class="nav-section-label">System</div>
    <div class="nav-item" id="nav-warm" onclick="switchTab('warm')">
      <span class="nav-icon">&#9729;</span>Warm Models
    </div>
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
      <div class="subtabs" id="{k}-subtabs">
        <button class="subtab active" id="{k}-subtab-models" onclick="setProfileView('{k}','models')">Models</button>
        <button class="subtab" id="{k}-subtab-profiles" onclick="setProfileView('{k}','profiles')">Profiles</button>
        <span class="subtab-note" id="{k}-subtab-note"></span>
      </div>
      <div class="recipe-bar" id="{k}-recipe-bar" style="display:none">
        <label class="recipe-label" for="{k}-recipe">Recipe</label>
        <select class="recipe-select" id="{k}-recipe" onchange="selectRecipe('{k}', this.value)"></select>
        <span class="recipe-desc" id="{k}-recipe-desc"></span>
      </div>
      <div class="profile-list" id="{k}-profile-list">
        <div class="empty"><div class="spin-icon" style="margin:0 auto"></div></div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:14px">
        <button class="btn btn-primary" id="{k}-start-btn" onclick="startEngine(engines.{k})">\u25b6 Start Selected</button>
        <button class="btn" id="{k}-dryrun-btn" onclick="dryRunProfile(engines.{k})">\u2697 Dry Run</button>
        <span style="font-size:12px;color:var(--muted)">Dry Run checks the launch without loading weights</span>
      </div>
      <div class="progress-wrap" id="{k}-progress" style="margin-top:14px">
        <div class="prog-head" id="{k}-prog-head">
          <span class="prog-phase" id="{k}-prog-phase">Starting\u2026</span>
          <span class="prog-meta" id="{k}-prog-meta"></span>
        </div>
        <div class="prog-bar-outer"><div class="prog-bar spin" id="{k}-prog-bar"></div></div>
        <div class="prog-log" id="{k}-log"></div>
      </div>
      <div id="{k}-preflight" class="preflight-wrap"></div>
    </div>
''' for k, e in _ENGINES.items()) + r"""
    <!-- ─── WARM MODELS ─── -->
    <div class="tab" id="tab-warm">
      <div class="page-hdr">
        <div class="page-title">Warm Model Resources</div>
        <div class="page-sub">See what is loaded, which processes are holding unified memory, and switch the active vLLM profile.</div>
      </div>

      <div class="warm-actions" style="margin-bottom:14px">
        <button class="btn btn-sm btn-ghost" onclick="loadWarmModels()">&#8635; Refresh</button>
        <button class="btn btn-sm btn-danger" onclick="warmStopVLLM()">Stop vLLM</button>
        <span class="warm-sub" id="warm-updated">Not loaded yet</span>
      </div>

      <div class="warm-grid">
        <div class="warm-panel">
          <div class="warm-title">
            <div>
              <div class="warm-title-main">Unified Memory</div>
              <div class="warm-sub">GB10 RAM and GPU share this pool</div>
            </div>
            <span class="warm-pill" id="warm-memory-pill">--</span>
          </div>
          <div class="warm-metric" id="warm-memory-metric">--</div>
          <div class="warm-bar"><div class="warm-bar-fill" id="warm-memory-bar"></div></div>
          <div class="warm-sub" id="warm-memory-sub" style="margin-top:8px"></div>
        </div>

        <div class="warm-panel">
          <div class="warm-title">
            <div>
              <div class="warm-title-main">vLLM Active</div>
              <div class="warm-sub">Container, served model, and matched profile</div>
            </div>
            <span class="warm-pill" id="warm-vllm-pill">--</span>
          </div>
          <div id="warm-vllm-active"></div>
        </div>

        <div class="warm-panel full">
          <div class="warm-title">
            <div>
              <div class="warm-title-main">Switch vLLM Profile</div>
              <div class="warm-sub">Launching a profile replaces the current vLLM container.</div>
            </div>
            <button class="btn btn-primary btn-sm" onclick="warmStartSelected()">Start Selected</button>
          </div>
          <div class="warm-profile-grid" id="warm-vllm-profiles"></div>
        </div>

        <div class="warm-panel">
          <div class="warm-title">
            <div>
              <div class="warm-title-main">GPU Compute Apps</div>
              <div class="warm-sub">Resident GPU processes from nvidia-smi</div>
            </div>
            <span class="warm-pill" id="warm-gpu-total">--</span>
          </div>
          <div id="warm-gpu-apps"></div>
        </div>

        <div class="warm-panel">
          <div class="warm-title">
            <div>
              <div class="warm-title-main">Ollama Warm Models</div>
              <div class="warm-sub">Models currently kept alive by Ollama</div>
            </div>
            <span class="warm-pill" id="warm-ollama-total">--</span>
          </div>
          <div id="warm-ollama-models"></div>
        </div>

        <div class="warm-panel">
          <div class="warm-title">
            <div>
              <div class="warm-title-main">Model Containers</div>
              <div class="warm-sub">Docker runtimes related to model serving</div>
            </div>
          </div>
          <div id="warm-containers"></div>
        </div>

        <div class="warm-panel">
          <div class="warm-title">
            <div>
              <div class="warm-title-main">Kubernetes vLLM</div>
              <div class="warm-sub">Scaled deployments in llm-inference</div>
            </div>
          </div>
          <div id="warm-k8s"></div>
        </div>
      </div>
    </div>

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

    <!-- ─── DASHBOARDS ─── -->
    <div class="tab" id="tab-sites">
      <div class="page-hdr" style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">
        <div>
          <div class="page-title">Dashboards</div>
          <div class="page-sub">Web UIs running on this box right now &mdash; discovered from listening ports and Kubernetes NodePorts. Names and groups can be curated via the <code>sites</code> array in <code>config.json</code>.</div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;white-space:nowrap">
          <label class="page-sub" style="display:flex;align-items:center;gap:6px;margin:0;cursor:pointer"
                 title="Also show ports that answered with JSON or plain text instead of a page (APIs, exporters).">
            <input type="checkbox" id="sites-show-api" onchange="renderSites()"> APIs
          </label>
          <button class="btn btn-sm" onclick="loadSites(true)" title="Re-probe every listening port">&#8635; Rescan</button>
        </div>
      </div>
      <div id="sites-meta" class="page-sub" style="margin-bottom:14px"></div>
      <div id="sites-root">
        <div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div></div>
      </div>
    </div>

    <!-- ─── RECOMMENDATIONS ─── -->
    <div class="tab" id="tab-recs">
      <div class="page-hdr" style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">
        <div>
          <div class="page-title">Recommendations</div>
          <div class="page-sub">Spark-specific tuning &amp; model advice, diffed against your vLLM profiles and live memory. Curated in <code>recommendations.json</code>.</div>
        </div>
        <button class="btn btn-sm" id="recs-refresh-btn" onclick="refreshKB()"
                title="Fetch curated sources and ask the research engine for proposed KB updates (~1-2 min). Proposals are for review only — apply via research_refresh.py.">&#8635; Refresh KB</button>
      </div>
      <div id="recs-meta" class="page-sub" style="margin-bottom:14px"></div>
      <div id="recs-proposed" style="margin-bottom:18px"></div>
      <div id="recs-root">
        <div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div></div>
      </div>
    </div>

  </main>
</div>

<div id="toast-root"></div>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

let activeTab = 'vllm';
let selectedProfile = null;
let selectedVLLMProfile = null;
let statusTimer = null;
let litellmPort = '';
let ollamaBase = '';
let warmSelectedProfile = null;

// Header memory gauge — alert threshold spliced in from server config
const MEM_ALERT_PCT = """ + str(_ALERT_THRESHOLDS["memory_percent"]) + r""";
let memHistory = [];  // last ~60 used_pct samples (12s poll ≈ 12 min window)

// ─────────────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  switchTab(activeTab);  // land on the vLLM switcher (static markup defaults to ollama)
  pollStatus();
  loadOllamaModels();    // still needed for the sidebar badge
  loadNodeInfo();
  loadScriptDirs();
  loadSites();           // pre-render the Dashboards pane
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
    // Memory gauge deep-links to Grafana when app.grafana_url is configured
    if (d.grafana_url) document.getElementById('hdr-mem').href = d.grafana_url;
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
  else if (name === 'warm') { loadWarmModels(); }
  else if (name === 'settings') { loadConfig(); }
  else if (name === 'debug') { loadDebugTab(); }
  else if (name === 'sites') { loadSites(); }
  else if (name === 'recs') { loadRecommendations(); loadProposed(); }
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
    updateMemGauge(d.memory);
  } catch(e) {}
}

function updateMemGauge(mem) {
  if (!mem || mem.error || !mem.total_gb) return;
  memHistory.push(mem.used_pct);
  if (memHistory.length > 60) memHistory.shift();
  const el = document.getElementById('hdr-mem');
  el.style.display = 'flex';
  document.getElementById('mem-used').textContent =
    Math.round(mem.used_gb) + ' / ' + Math.round(mem.total_gb) + ' GB';
  const pctEl = document.getElementById('mem-pct');
  pctEl.textContent = Math.round(mem.used_pct) + '%';
  pctEl.classList.toggle('crit', mem.used_pct >= MEM_ALERT_PCT);
  el.title = 'Unified memory \u2014 ' + mem.used_gb + ' GB used, ' + mem.available_gb +
    ' GB available of ' + mem.total_gb + ' GB \u00b7 alert threshold ' + MEM_ALERT_PCT +
    '% (dashed line) \u00b7 ~12 min history';
  // Inline-SVG sparkline on a fixed 0-100% scale so the threshold line is stable.
  // A single sample is drawn as a full-width level so the gauge reads as a
  // filled meter immediately, before ~12 min of history accumulates.
  const W = 72, H = 20, n = memHistory.length, base = H - 1;
  const y = p => base - (Math.min(Math.max(p, 0), 100) / 100) * (H - 2);
  const x = i => n > 1 ? 1 + (i / (n - 1)) * (W - 2) : 71;
  const linePts = n === 1
    ? ['1,' + y(memHistory[0]).toFixed(1), '71,' + y(memHistory[0]).toFixed(1)]
    : memHistory.map((p, i) => x(i).toFixed(1) + ',' + y(p).toFixed(1));
  document.getElementById('mem-spark-line').setAttribute('points', linePts.join(' '));
  // Fill the area under the line down to the baseline so the "level" is visible.
  document.getElementById('mem-spark-area').setAttribute('points',
    ['1,' + base, ...linePts, '71,' + base].join(' '));
  const dot = document.getElementById('mem-spark-dot');
  dot.setAttribute('cx', '71');
  dot.setAttribute('cy', y(memHistory[n - 1]).toFixed(1));
  const th = document.getElementById('mem-spark-threshold');
  th.setAttribute('y1', y(MEM_ALERT_PCT).toFixed(1));
  th.setAttribute('y2', y(MEM_ALERT_PCT).toFixed(1));
}

function setPill(id, ok, label) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'pill ' + (ok ? 'ok' : 'err');
  const sp = el.querySelector('span');
  if (sp) sp.textContent = label;
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboards
// ─────────────────────────────────────────────────────────────────────────────

let _lastSites = null;

async function loadSites(rescan) {
  const root = document.getElementById('sites-root');
  const meta = document.getElementById('sites-meta');
  if (rescan) root.innerHTML = '<div class="empty"><div class="spin-icon" style="margin:0 auto 8px"></div></div>';
  try {
    _lastSites = await apiFetch('/api/sites' + (rescan ? '?refresh=1' : ''));
    const dd = _lastSites.discovery || {};
    const bits = [];
    if (dd.enabled === false) bits.push('discovery disabled &mdash; showing config only');
    else bits.push(dd.candidates + ' listening ports probed &middot; ' + dd.ui + ' UIs, ' + dd.api + ' APIs');
    if (dd.host && dd.host.ok === false) bits.push('host scan failed: ' + _escHtml(dd.host.error || ''));
    if (dd.kubernetes && dd.kubernetes.ok === false) bits.push('kubectl unavailable');
    if (_lastSites.cached_age_s) bits.push('cached ' + _lastSites.cached_age_s + 's ago');
    meta.innerHTML = bits.join(' &middot; ');
    renderSites();
  } catch(e) {
    root.innerHTML = '<div class="empty"><div class="empty-icon">&#9888;</div>' +
      '<div class="empty-text">Could not load dashboards &middot; ' + _escHtml(e.message) + '</div></div>';
  }
}

function renderSites() {
  const root = document.getElementById('sites-root');
  if (!_lastSites) return;
  const showApi = document.getElementById('sites-show-api').checked;
  const sites = (_lastSites.sites || []).filter(s => showApi || s.kind !== 'api');
  if (!sites.length) {
    root.innerHTML = '<div class="empty"><div class="empty-icon">&#128202;</div>' +
      '<div class="empty-text">No web UIs found on this box</div></div>';
    return;
  }
  // Two services can share a <title> ("Node Exporter" on :9100 and :9101) — a
  // repeated name makes the pair unreadable, so it earns its port back.
  const nameCount = {};
  sites.forEach(s => { nameCount[s.name] = (nameCount[s.name] || 0) + 1; });
  const label = s => nameCount[s.name] > 1 && s.port ? s.name + ' :' + s.port : s.name;

  // Group cards under sec-label headings. Curated groups keep their meaning and
  // sort first; the auto-assigned buckets fall to the bottom of the page.
  const TAIL = ['Kubernetes', 'Host', 'Other'];
  const groups = new Map();
  sites.forEach(s => {
    const g = s.group || 'Other';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(s);
  });
  const order = [...groups.keys()].sort((a, b) => {
    const ai = TAIL.indexOf(a), bi = TAIL.indexOf(b);
    if (ai !== bi) return (ai < 0 ? -1 : ai) - (bi < 0 ? -1 : bi);
    return a.localeCompare(b);
  });
  let html = '';
  for (const group of order) {
    html += '<div class="sec-label">' + _escHtml(group) + '</div>';
    html += '<div class="model-grid">' + groups.get(group).map(s => {
      let hostLabel = s.url;
      try { hostLabel = new URL(s.url).host; } catch(e) {}
      const dotCls = s.reachable === true ? ' ok' : (s.reachable === false ? ' err' : '');
      const dotTitle = s.reachable === true ? 'reachable' : (s.reachable === false ? 'unreachable' : 'unknown');
      const href = _escHtml(s.url).replace(/"/g, '&quot;');
      const tag = s.kind === 'api'
        ? '<span class="tag">api</span>'
        : '<span class="tag tag-amber">open ↗</span>';
      return '<a class="model-card" href="' + href + '" target="_blank" rel="noopener">' +
        '<span class="site-dot' + dotCls + '" title="' + dotTitle + '"></span>' +
        '<div class="model-card-info">' +
          '<div class="model-card-name">' + _escHtml(label(s)) + '</div>' +
          '<div class="model-card-meta">' + _escHtml(hostLabel) +
            (s.desc ? ' · ' + _escHtml(s.desc) : '') + '</div>' +
        '</div>' +
        '<div class="model-card-right">' + tag + '</div>' +
      '</a>';
    }).join('') + '</div>';
  }
  root.innerHTML = html;
}

// ─────────────────────────────────────────────────────────────────────────────
// Recommendations
// ─────────────────────────────────────────────────────────────────────────────

const _SEV = {
  high:   { label: 'HIGH',   bg: 'rgba(239,68,68,.15)',  fg: '#f87171' },
  medium: { label: 'MEDIUM', bg: 'rgba(245,158,11,.15)', fg: '#fbbf24' },
  low:    { label: 'INFO',   bg: 'rgba(148,163,184,.15)', fg: '#94a3b8' },
};
let _lastRecs = [];  // last fired recommendations, for approve/apply lookups by id

async function loadRecommendations() {
  const root = document.getElementById('recs-root');
  const meta = document.getElementById('recs-meta');
  try {
    const d = await apiFetch('/api/recommendations');
    const recs = d.recommendations || [];
    _lastRecs = recs;
    document.getElementById('badge-recs').textContent = recs.length;
    const st = d.state || {};
    meta.innerHTML = _escHtml((d.meta && d.meta.hardware) || '') +
      (st.total_gb ? ' · ' + Math.round(st.available_gb) + ' / ' + Math.round(st.total_gb) +
        ' GB free · ' + d.profiles_checked + ' profiles checked' : '') +
      (d.meta && d.meta.last_updated ? ' · KB ' + _escHtml(d.meta.last_updated) : '');

    if (!recs.length) {
      root.innerHTML = '<div class="empty"><div class="empty-icon">✅</div>' +
        '<div class="empty-text">No recommendations — your profiles match current Spark best practice.</div></div>';
      return;
    }
    root.innerHTML = recs.map(r => {
      const sev = _SEV[r.severity] || _SEV.low;
      const chip = (txt, extra) => '<span style="display:inline-block;padding:2px 8px;border-radius:6px;' +
        'font-size:11px;font-weight:600;' + (extra || '') + '">' + txt + '</span>';
      const fired = r.scope === 'global'
        ? '<div class="model-card-meta" style="margin-top:8px">' + _escHtml(r.detail) + '</div>'
        : '<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:6px">' +
            r.fired_for.map(h => chip(_escHtml(h.profile.replace(/^start_/, '')) +
              ' · ' + _escHtml(h.detail),
              'background:var(--s2);color:var(--muted);font-weight:500')).join('') +
          '</div>';
      const src = (r.sources || []).map(s =>
        '<a href="' + _escHtml(s.url).replace(/"/g, '&quot;') + '" target="_blank" rel="noopener" ' +
        'style="color:var(--blue);text-decoration:none">' +
        _escHtml((s.note || s.url)) + (s.date ? ' (' + _escHtml(s.date) + ')' : '') + '</a>'
      ).join(' · ');
      const rid = (r.id || '').replace(/'/g, "\\'");
      const actions = [];
      if (r.kind === 'model' && r.download && r.download.repo_id) {
        actions.push('<button class="btn btn-sm btn-primary" onclick="approveModel(\'' + rid + '\')">' +
          '⬇ Approve &amp; download</button>');
      }
      if (r.apply && (r.fired_for || []).length) {
        r.fired_for.forEach(h => {
          const pid = (h.profile || '').replace(/'/g, "\\'");
          actions.push('<button class="btn btn-sm" onclick="applyRec(\'' + rid + '\',\'' + pid + '\')">' +
            '✎ Apply to ' + _escHtml(h.profile.replace(/^start_/, '').replace(/\.sh$/, '')) + '</button>');
        });
      }
      const actionBar = actions.length
        ? '<div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:8px">' + actions.join('') + '</div>' : '';
      return '<div class="model-card" style="display:block;cursor:default;margin-bottom:12px">' +
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">' +
          chip(sev.label, 'background:' + sev.bg + ';color:' + sev.fg) +
          '<span class="model-card-name">' + _escHtml(r.title) + '</span>' +
          chip(_escHtml(r.kind) + ' · ' + _escHtml(r.confidence || '') + ' conf',
               'background:var(--s2);color:var(--muted);font-weight:500') +
        '</div>' +
        '<div class="model-card-meta" style="margin-top:8px;line-height:1.5">' + _escHtml(r.summary) + '</div>' +
        '<div style="margin-top:8px;font-size:13px"><strong style="color:var(--text)">Do:</strong> ' +
          _escHtml(r.action) + '</div>' +
        fired +
        (src ? '<div class="model-card-meta" style="margin-top:10px;font-size:12px">Sources: ' + src + '</div>' : '') +
        actionBar +
      '</div>';
    }).join('');
  } catch(e) {
    root.innerHTML = '<div class="empty"><div class="empty-icon">⚠</div>' +
      '<div class="empty-text">Could not load recommendations · ' + _escHtml(e.message) + '</div></div>';
  }
}

// Proposed KB updates from the research-refresh job. Review-only: applying stays
// a deliberate CLI step (research_refresh.py --apply, then formalize matches).
function renderProposed(summary, proposed) {
  const panel = document.getElementById('recs-proposed');
  if (!proposed || !proposed.length) { panel.innerHTML = ''; return; }
  const rows = proposed.map(p => {
    const isUpd = p.status === 'update';
    const tag = (txt, bg, fg) => '<span style="display:inline-block;padding:1px 7px;border-radius:6px;' +
      'font-size:10px;font-weight:700;background:' + bg + ';color:' + fg + '">' + txt + '</span>';
    const src = (p.sources || []).map(s =>
      '<a href="' + _escHtml(s.url).replace(/"/g, '&quot;') + '" target="_blank" rel="noopener" ' +
      'style="color:var(--blue);text-decoration:none">' + _escHtml(s.note || s.url) + '</a>').join(' · ');
    return '<div style="padding:10px 0;border-top:1px solid var(--border)">' +
      '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
        tag(isUpd ? 'UPDATE' : 'NEW', isUpd ? 'rgba(59,130,246,.15)' : 'rgba(16,185,129,.15)',
            isUpd ? '#60a5fa' : '#34d399') +
        tag(_escHtml(p.severity || '?'), 'var(--s2)', 'var(--muted)') +
        '<span style="font-weight:600;color:var(--text)">' + _escHtml(p.title || p.id) + '</span>' +
      '</div>' +
      '<div class="model-card-meta" style="margin-top:6px;line-height:1.5">' + _escHtml(p.summary || '') + '</div>' +
      '<div style="margin-top:6px;font-size:12px"><strong style="color:var(--text)">Match intent:</strong> ' +
        _escHtml(p.suggested_match || '') + '</div>' +
      (src ? '<div class="model-card-meta" style="margin-top:6px;font-size:11px">Sources: ' + src + '</div>' : '') +
    '</div>';
  }).join('');
  panel.innerHTML = '<div class="model-card" style="display:block;cursor:default;' +
    'border-color:var(--blue);background:rgba(59,130,246,.04)">' +
    '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">' +
      '<span class="model-card-name">🧪 Proposed KB updates · ' + proposed.length + '</span>' +
      '<span class="model-card-meta" style="font-size:12px">review-only</span>' +
    '</div>' +
    (summary ? '<div class="model-card-meta" style="margin-top:6px;line-height:1.5">' + _escHtml(summary) + '</div>' : '') +
    rows +
    '<div class="model-card-meta" style="margin-top:12px;font-size:12px">Apply deliberately: ' +
      '<code>python3 research_refresh.py --apply</code>, then formalize each <code>match.type=="manual"</code> rule.</div>' +
  '</div>';
}

async function loadProposed() {
  try {
    const d = await apiFetch('/api/recommendations/proposed');
    renderProposed(d.summary, d.proposed);
  } catch(e) { /* non-fatal: panel just stays empty */ }
}

async function refreshKB() {
  const btn = document.getElementById('recs-refresh-btn');
  const orig = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spin-icon" style="width:12px;height:12px;vertical-align:-1px"></span> Researching…';
  toast('Running research refresh — this takes ~1-2 min', '');
  try {
    const d = await apiFetch('/api/recommendations/refresh', 'POST');
    renderProposed(d.summary, d.proposed);
    const n = (d.proposed || []).length;
    toast(n ? '✓ ' + n + ' proposal(s) — review below' : 'No new proposals', n ? 'ok' : '');
  } catch(e) {
    toast('Refresh failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.innerHTML = orig;
  }
}

// Approve a model recommendation → confirm the HF repo, then run the existing
// download flow (SSE), which auto-wires a vLLM profile on completion.
function closeRecModal() { const m = document.getElementById('rec-modal'); if (m) m.remove(); }

function approveModel(id) {
  const r = _lastRecs.find(x => x.id === id);
  if (!r || !r.download) { toast('No download info for this recommendation', 'err'); return; }
  window._approveRecId = id;
  const repo = r.download.repo_id || '';
  const dir = r.download.local_dir || '';
  closeRecModal();
  const overlay = document.createElement('div');
  overlay.id = 'rec-modal';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:9999;display:flex;align-items:center;justify-content:center';
  overlay.innerHTML =
    '<div style="background:var(--s1);border:1px solid var(--border);border-radius:10px;padding:24px;width:520px;max-width:92vw">' +
      '<div style="font-size:15px;font-weight:700;margin-bottom:4px">Approve &amp; download</div>' +
      '<div style="font-size:12px;color:var(--muted);margin-bottom:16px">' + _escHtml(r.title) +
        '. This pulls the model from Hugging Face and auto-creates a vLLM profile when the download finishes.</div>' +
      '<label style="font-size:11px;color:var(--muted)">HF repo</label>' +
      '<input class="input" id="rec-dl-repo" value="' + _escHtml(repo).replace(/"/g, '&quot;') + '" ' +
        'placeholder="owner/model-name" style="width:100%;margin:4px 0 12px">' +
      '<label style="font-size:11px;color:var(--muted)">Target dir (blank = HF cache)</label>' +
      '<input class="input" id="rec-dl-dir" value="' + _escHtml(dir).replace(/"/g, '&quot;') + '" ' +
        'placeholder="~/.cache/huggingface" style="width:100%;margin:4px 0 12px">' +
      '<div id="rec-dl-progress" style="display:none;margin:8px 0 4px">' +
        '<div class="prog" style="height:6px;background:var(--s2);border-radius:4px;overflow:hidden">' +
          '<div id="rec-dl-bar" class="prog-bar" style="height:100%;width:0"></div></div>' +
        '<pre id="rec-dl-log" style="font-size:11px;color:var(--muted);margin:8px 0 0;white-space:pre-wrap;' +
          'max-height:160px;overflow:auto"></pre>' +
      '</div>' +
      '<div style="display:flex;gap:8px;justify-content:flex-end;margin-top:8px">' +
        '<button class="btn btn-sm" id="rec-dl-close" onclick="closeRecModal()">Cancel</button>' +
        '<button class="btn btn-primary btn-sm" id="rec-dl-go" onclick="startApproveDownload(\'' +
          id.replace(/'/g, "\\'") + '\')">Download</button>' +
      '</div>' +
    '</div>';
  document.body.appendChild(overlay);
  document.getElementById('rec-dl-repo').focus();
}

async function startApproveDownload(id) {
  const repo = document.getElementById('rec-dl-repo').value.trim();
  const dir = document.getElementById('rec-dl-dir').value.trim();
  if (!repo) { document.getElementById('rec-dl-repo').focus(); return; }
  const go = document.getElementById('rec-dl-go');
  const cancel = document.getElementById('rec-dl-close');
  const prog = document.getElementById('rec-dl-progress');
  const bar = document.getElementById('rec-dl-bar');
  const log = document.getElementById('rec-dl-log');
  go.disabled = true;
  go.innerHTML = '<span class="spin-icon" style="width:12px;height:12px;vertical-align:-1px"></span> Downloading…';
  prog.style.display = 'block';
  bar.className = 'prog-bar spin';
  const lines = ['Starting download: ' + repo];
  log.textContent = lines[0];
  let wired = false;
  try {
    const r = _lastRecs.find(x => x.id === (window._approveRecId || ''));
    const dl = (r && r.download) || {};
    const resp = await fetch('/api/hf/download', {
      method: 'POST', headers: authHeaders(),
      body: JSON.stringify({repo_id: repo, local_dir: dir || undefined,
        ignore_patterns: dl.ignore_patterns, allow_patterns: dl.allow_patterns}),
    });
    if (!resp.ok) {
      let msg = resp.statusText;
      try { const d = await resp.json(); msg = d.detail || JSON.stringify(d); } catch(e) {}
      throw new Error(msg);
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split('\n')) {
        if (!line.startsWith('data: ')) continue;
        let ev; try { ev = JSON.parse(line.slice(6)); } catch(e) { continue; }
        if (ev.file_start) {
          const f = ev.file_start;
          bar.className = 'prog-bar spin';
          lines.push('[' + f.idx + '/' + f.total + '] ⤓ ' + f.name + ' (' + f.size_str + ')…');
        } else if (ev.progress) {
          const p = ev.progress;
          bar.className = 'prog-bar'; bar.style.width = p.pct + '%';
          lines[lines.length - 1] = '[' + p.idx + '/' + p.total_files + '] ✓ ' + p.file + '  ·  ' +
            p.pct + '%  ·  ' + p.done_mb.toFixed(0) + ' / ' + p.total_mb.toFixed(0) + ' MB  ·  ' + p.speed;
        } else if (ev.status && typeof ev.status === 'string' && ev.status !== 'complete' && ev.status !== 'error') {
          lines.push(ev.status);  // "Skipping N files…", "Found N files…"
        } else if (ev.status === 'complete') {
          bar.className = 'prog-bar'; bar.style.width = '100%';
          lines.push('✓ Downloaded → ' + ev.path + (ev.errors > 0 ? '  ⚠ ' + ev.errors + ' error(s)' : ''));
          toast('✓ Downloaded: ' + repo, 'ok');
        } else if (ev.auto_profile) {
          wired = true;
          const p = ev.auto_profile.profile || {};
          lines.push('✓ vLLM profile wired: ' + (p.name || p.id || 'profile'));
          toast('✓ Profile wired — recommendation satisfied', 'ok');
        } else if (ev.auto_profile_error) {
          lines.push('⚠ Profile not auto-created: ' + ev.auto_profile_error);
        } else if (ev.status === 'error') {
          throw new Error(ev.error || 'download failed');
        } else if (ev.log) {
          lines.push(ev.log);
        }
        log.textContent = lines.join('\n');
        log.scrollTop = log.scrollHeight;
      }
    }
    go.innerHTML = wired ? '✓ Done' : 'Finished';
    cancel.textContent = 'Close';
    loadRecommendations();          // fired model rec should now clear
    loadEngineProfiles(engines.vllm);
    if (typeof loadWarmModels === 'function') loadWarmModels();
  } catch(e) {
    bar.className = 'prog-bar'; bar.style.width = '0';
    lines.push('✗ ' + e.message);
    log.textContent = lines.join('\n');
    toast('Download failed: ' + e.message, 'err');
    go.disabled = false;
    go.innerHTML = 'Retry';
  }
}

// Apply a config/tuning rec to a flagged profile: fetch a diff preview, then
// confirm to write the edited start script.
function _diffHtml(diff) {
  return (diff || '').split('\n').map(l => {
    let c = 'var(--muted)';
    if (l.startsWith('+') && !l.startsWith('+++')) c = '#34d399';
    else if (l.startsWith('-') && !l.startsWith('---')) c = '#f87171';
    else if (l.startsWith('@@')) c = 'var(--blue)';
    return '<span style="color:' + c + '">' + _escHtml(l) + '</span>';
  }).join('\n');
}

async function applyRec(id, profile) {
  toast('Computing diff…', '');
  let d;
  try {
    d = await apiFetch('/api/recommendations/apply', 'POST', {id: id, profile: profile});
  } catch(e) { toast('Apply failed: ' + e.message, 'err'); return; }
  if (!d.changed) { toast('Already applied: ' + (d.note || 'no change'), 'ok'); return; }
  const short = profile.replace(/^start_/, '').replace(/\.sh$/, '');
  closeRecModal();
  const overlay = document.createElement('div');
  overlay.id = 'rec-modal';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:9999;display:flex;align-items:center;justify-content:center';
  overlay.innerHTML =
    '<div style="background:var(--s1);border:1px solid var(--border);border-radius:10px;padding:24px;width:640px;max-width:94vw">' +
      '<div style="font-size:15px;font-weight:700;margin-bottom:4px">Apply to ' + _escHtml(short) + '</div>' +
      '<div style="font-size:12px;color:var(--muted);margin-bottom:12px">' + _escHtml(d.note || '') +
        ' — review the change to the start script, then confirm.</div>' +
      '<pre style="font-size:11px;line-height:1.45;background:var(--s2);border:1px solid var(--border);' +
        'border-radius:8px;padding:12px;max-height:340px;overflow:auto;white-space:pre">' + _diffHtml(d.diff) + '</pre>' +
      '<div id="rec-apply-note" style="font-size:11px;color:var(--muted);margin-top:8px">' +
        'The profile takes effect the next time you start it.</div>' +
      '<div style="display:flex;gap:8px;justify-content:flex-end;margin-top:12px">' +
        '<button class="btn btn-sm" id="rec-apply-cancel" onclick="closeRecModal()">Cancel</button>' +
        '<button class="btn btn-primary btn-sm" id="rec-apply-go" onclick="confirmApplyRec(\'' +
          id.replace(/'/g, "\\'") + '\',\'' + profile.replace(/'/g, "\\'") + '\')">Apply change</button>' +
      '</div>' +
    '</div>';
  document.body.appendChild(overlay);
}

async function confirmApplyRec(id, profile) {
  const go = document.getElementById('rec-apply-go');
  go.disabled = true;
  go.innerHTML = '<span class="spin-icon" style="width:12px;height:12px;vertical-align:-1px"></span> Applying…';
  try {
    const d = await apiFetch('/api/recommendations/apply', 'POST', {id: id, profile: profile, confirm: true});
    toast('✓ ' + (d.note || 'applied'), 'ok');
    closeRecModal();
    loadRecommendations();
    loadEngineProfiles(engines.vllm);
  } catch(e) {
    toast('Apply failed: ' + e.message, 'err');
    go.disabled = false;
    go.innerHTML = 'Apply change';
  }
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
// Warm model resources
// ─────────────────────────────────────────────────────────────────────────────

function _gb(v) {
  return (v == null || isNaN(v)) ? '--' : Number(v).toFixed(1) + ' GB';
}

function _warmEmpty(msg) {
  return '<div class="empty" style="padding:16px"><div class="empty-text">' + _escHtml(msg) + '</div></div>';
}

function _warmRow(name, meta, rightHtml) {
  return '<div class="warm-row"><div><div class="warm-name">' + _escHtml(name) + '</div>'
    + '<div class="warm-meta">' + _escHtml(meta || '') + '</div></div>'
    + '<div class="warm-actions">' + (rightHtml || '') + '</div></div>';
}

async function loadWarmModels() {
  const setHtml = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };
  try {
    const d = await apiFetch('/api/warm-models');
    const updated = document.getElementById('warm-updated');
    if (updated) updated.textContent = 'Updated ' + new Date(d.timestamp).toLocaleTimeString();

    const mem = d.memory || {};
    const usedPct = Math.max(0, Math.min(100, mem.used_pct || 0));
    document.getElementById('warm-memory-metric').textContent = _gb(mem.used_gb) + ' used';
    document.getElementById('warm-memory-bar').style.width = usedPct + '%';
    document.getElementById('warm-memory-sub').textContent = _gb(mem.available_gb) + ' available of ' + _gb(mem.total_gb);
    const memPill = document.getElementById('warm-memory-pill');
    memPill.textContent = usedPct.toFixed(1) + '%';
    memPill.className = 'warm-pill ' + (mem.available_gb < 10 ? 'err' : mem.available_gb < 25 ? 'warn' : 'ok');

    const v = d.vllm || {};
    const status = v.status || {};
    const profiles = v.profiles || [];
    const activeProfile = v.active_profile || null;
    if (!warmSelectedProfile) warmSelectedProfile = activeProfile || (profiles[0] && profiles[0].id) || null;
    const activeProfileInfo = profiles.find(p => p.id === activeProfile);
    const state = status.state || (status.running ? 'serving' : 'stopped');
    const vPill = document.getElementById('warm-vllm-pill');
    vPill.textContent = state;
    vPill.className = 'warm-pill ' + (state === 'serving' ? 'ok' : state === 'loading' ? 'warn' : 'err');
    const inst = (status.instances || [])[0] || {};
    setHtml('warm-vllm-active',
      _warmRow(status.model || 'No served model',
        [
          activeProfileInfo ? activeProfileInfo.name : (activeProfile ? activeProfile : 'no matched profile'),
          inst.name ? inst.name : '',
          inst.port ? ':' + inst.port : ''
        ].filter(Boolean).join(' · '),
        status.running ? '<button class="btn btn-sm btn-danger" onclick="warmStopVLLM()">Stop</button>' : '')
    );

    setHtml('warm-vllm-profiles', profiles.length ? profiles.map(p => {
      const isActive = p.id === activeProfile;
      const isSelected = p.id === warmSelectedProfile;
      const cls = 'warm-profile' + (isActive ? ' active' : '') + (isSelected ? ' selected' : '');
      const vram = p.vram_gb != null ? p.vram_gb + ' GB' : '--';
      return '<div class="' + cls + '" onclick="warmSelectProfile(\'' + p.id.replace(/'/g, "\\'") + '\')">'
        + '<div class="warm-profile-head"><div class="warm-profile-name">' + _escHtml(p.name || p.id) + '</div>'
        + '<span class="warm-pill ' + (isActive ? 'ok' : '') + '">' + _escHtml(vram) + '</span></div>'
        + '<div class="warm-profile-desc">' + _escHtml(p.description || p.id) + '</div>'
        + (isActive ? '<div class="warm-meta" style="color:var(--green);margin-top:6px">active now</div>' : '')
        + '</div>';
    }).join('') : _warmEmpty('No vLLM profiles found'));

    const gpu = d.nvidia || {};
    const apps = gpu.apps || [];
    const gpuTotal = document.getElementById('warm-gpu-total');
    gpuTotal.textContent = apps.length ? _gb((gpu.total_mib || 0) / 1024) : 'none';
    gpuTotal.className = 'warm-pill ' + (apps.length ? 'warn' : 'ok');
    setHtml('warm-gpu-apps', apps.length ? apps.map(a =>
      _warmRow(a.process || ('PID ' + a.pid), 'pid ' + a.pid + (a.cmd ? ' · ' + a.cmd : ''),
        '<span class="warm-pill warn">' + _escHtml(_gb(a.used_gb)) + '</span>')
    ).join('') : _warmEmpty(gpu.ok === false ? (gpu.error || 'nvidia-smi unavailable') : 'No GPU compute apps'));

    const ollama = d.ollama || {};
    const warmModels = ollama.models || [];
    const ollamaTotal = document.getElementById('warm-ollama-total');
    ollamaTotal.textContent = warmModels.length + ' warm';
    ollamaTotal.className = 'warm-pill ' + (warmModels.length ? 'warn' : 'ok');
    setHtml('warm-ollama-models', warmModels.length ? warmModels.map(m =>
      _warmRow(m.name || 'unknown', [m.size, m.processor, m.until].filter(Boolean).join(' · '),
        '<button class="btn btn-sm btn-danger" onclick="warmStopOllama(\'' + (m.name || '').replace(/'/g, "\\'") + '\')">Unload</button>')
    ).join('') : _warmEmpty(ollama.ok === false ? (ollama.error || 'Ollama unavailable') : 'No warm Ollama models'));

    const containers = (d.docker || {}).containers || [];
    setHtml('warm-containers', containers.length ? containers.map(c =>
      _warmRow(c.name || c.id, [c.image, c.status, c.ports].filter(Boolean).join(' · '), '')
    ).join('') : _warmEmpty((d.docker || {}).ok === false ? ((d.docker || {}).error || 'Docker unavailable') : 'No model containers'));

    const deployments = (d.kubernetes || {}).deployments || [];
    setHtml('warm-k8s', deployments.length ? deployments.map(dep => {
      const available = dep.available || 0;
      const replicas = dep.replicas || 0;
      const cls = replicas === 0 ? '' : available >= replicas ? 'ok' : 'warn';
      return _warmRow(dep.name, (dep.images || []).join(' · '),
        '<span class="warm-pill ' + cls + '">' + available + '/' + replicas + '</span>');
    }).join('') : _warmEmpty((d.kubernetes || {}).ok === false ? ((d.kubernetes || {}).error || 'kubectl unavailable') : 'No scaled vLLM deployments'));
  } catch(e) {
    setHtml('warm-vllm-active', _warmEmpty('Failed to load resources: ' + e.message));
  }
}

function warmSelectProfile(id) {
  warmSelectedProfile = id;
  document.querySelectorAll('.warm-profile').forEach(el => el.classList.remove('selected'));
  loadWarmModels();
}

async function warmStartSelected() {
  if (!warmSelectedProfile) { toast('Select a vLLM profile first', 'err'); return; }
  try {
    await apiFetch('/api/vllm/start', 'POST', {profile: warmSelectedProfile});
    toast('vLLM profile starting', 'ok');
    setTimeout(loadWarmModels, 2500);
  } catch(e) {
    if (String(e.message || '').includes('unified memory') &&
        confirm(e.message + '\n\nForce start anyway?')) {
      try {
        await apiFetch('/api/vllm/start', 'POST', {profile: warmSelectedProfile, force: true});
        toast('vLLM profile force-started', 'ok');
        setTimeout(loadWarmModels, 2500);
      } catch(forceErr) {
        toast('Force start failed: ' + forceErr.message, 'err');
      }
    } else {
      toast('Start failed: ' + e.message, 'err');
    }
  }
}

async function warmStopVLLM() {
  if (!confirm('Stop vLLM? This will interrupt active inference requests.')) return;
  try {
    await apiFetch('/api/vllm/stop', 'POST');
    toast('vLLM stopped', 'ok');
    setTimeout(loadWarmModels, 1500);
  } catch(e) {
    toast('Stop failed: ' + e.message, 'err');
  }
}

async function warmStopOllama(name) {
  if (!name || !confirm('Unload ' + name + ' from Ollama?')) return;
  try {
    await apiFetch('/api/ollama/stop', 'POST', {name});
    toast('Ollama model unloaded', 'ok');
    setTimeout(loadWarmModels, 1000);
  } catch(e) {
    toast('Unload failed: ' + e.message, 'err');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SGLang
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Engine helpers (shared by SGLang + vLLM)
// ─────────────────────────────────────────────────────────────────────────────

const engines = {
""" + ",\n".join(f'''  {k}: {{
    name: '{e["name"]}', api: '/api/{k}', selectedProfile: null, selectedRecipe: null, recipes: null,{" webui: true," if e.get("webui") else ""}
    key: '{k}',
    ids: {{ led: '{k}-engine-led', title: '{k}-engine-title', model: '{k}-engine-model',
           card: '{k}-engine-card', stop: '{k}-stop-btn', start: '{k}-start-btn',
           profiles: '{k}-profile-list', prog: '{k}-progress', log: '{k}-log',
           bar: '{k}-prog-bar', phase: '{k}-prog-phase', meta: '{k}-prog-meta',
           dryrun: '{k}-dryrun-btn', preflight: '{k}-preflight',
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
    // Tri-state: a live container ("loading") is distinct from a ready model
    // ("serving") \u2014 vLLM takes minutes to load weights. Fall back to running.
    const state = d.state || (d.running ? 'serving' : 'stopped');
    if (state === 'serving') {
      led.className = 'engine-led on';
      title.textContent = eng.name + ' \u2014 Serving';
      model.textContent = d.model || (eng.webui ? '' : 'Ready');
      card.classList.add('online');
      stop.disabled = false;
      if (eng.webui && eng.ids.webui) {
        const wb = document.getElementById(eng.ids.webui);
        if (wb) { wb.style.display = ''; wb.href = _engineBaseUrl(eng.key); }
      }
    } else if (state === 'loading') {
      led.className = 'engine-led loading';
      title.textContent = eng.name + ' \u2014 Loading\u2026';
      model.textContent = 'Container up \u2014 loading model weights\u2026';
      card.classList.add('online');
      stop.disabled = false;
      if (eng.webui && eng.ids.webui) {
        const wb = document.getElementById(eng.ids.webui);
        if (wb) wb.style.display = 'none';
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
    eng.profiles = await apiFetch(eng.api + '/profiles');
    renderEngineProfiles(eng);
  } catch(e) {
    el.innerHTML = '<div class="empty"><div class="empty-text">Could not load profiles</div></div>';
  }
  if (eng.key === 'llamacpp') loadRecipes(eng);
}

// llama.cpp only. A GGUF repo ships one set of weights at ten quantizations, so the
// quant/context/speculation choice is a launch *parameter*, not a separate profile —
// otherwise the profile list becomes ten near-identical rows that drift apart.
async function loadRecipes(eng) {
  const bar = document.getElementById(eng.key + '-recipe-bar');
  const sel = document.getElementById(eng.key + '-recipe');
  if (!bar || !sel) return;
  let d;
  try { d = await apiFetch('/api/llamacpp/recipes'); }
  catch(e) { bar.style.display = 'none'; return; }

  const names = Object.keys(d.recipes || {});
  // No recipes configured means the scripts use their own defaults. Showing an empty
  // dropdown would imply a choice that does not exist.
  if (!names.length) { bar.style.display = 'none'; eng.recipes = null; return; }

  eng.recipes = d.recipes;
  if (!eng.selectedRecipe || !eng.recipes[eng.selectedRecipe]) {
    eng.selectedRecipe = (d.default && eng.recipes[d.default]) ? d.default : names[0];
  }
  sel.innerHTML = names.map(n => {
    const r = eng.recipes[n];
    return '<option value="' + _escHtml(n) + '"' +
           (n === eng.selectedRecipe ? ' selected' : '') + '>' +
           _escHtml(n + '  —  ' + r.quant + ', ' + (r.ctx / 1024) + 'K ctx, ~' + r.vram_gb + ' GB') +
           '</option>';
  }).join('');
  bar.style.display = 'flex';
  selectRecipe(eng.key, eng.selectedRecipe);
}

function selectRecipe(key, name) {
  const eng = engines[key];
  if (!eng || !eng.recipes || !eng.recipes[name]) return;
  eng.selectedRecipe = name;
  const r = eng.recipes[name];
  const d = document.getElementById(key + '-recipe-desc');
  if (d) {
    d.textContent = (r.desc || '') +
      (r.spec && r.spec !== 'none' ? '  ·  spec: ' + r.spec : '') +
      (r.vision ? '  ·  vision' : '');
  }
}

// Two views over the same list, so the launch list stays uncluttered:
//   models   — profiles whose weights are on disk; the things you can start.
//   profiles — every start_*.sh, including ones whose weights are gone. This is
//              the housekeeping view, and the only place orphans are visible.
function setProfileView(key, view) {
  const eng = engines[key];
  eng.profileView = view;
  ['models','profiles'].forEach(v => {
    const b = document.getElementById(key + '-subtab-' + v);
    if (b) b.classList.toggle('active', v === view);
  });
  renderEngineProfiles(eng);
}

function renderEngineProfiles(eng) {
  const el = document.getElementById(eng.ids.profiles);
  const all = eng.profiles || [];
  const view = eng.profileView || 'models';
  const profiles = view === 'models' ? all.filter(p => !p.model_missing) : all;

  const note = document.getElementById(eng.key + '-subtab-note');
  if (note) {
    const onDisk = {};
    all.forEach(p => { if (p.model_dir) onDisk[p.model_dir] = p.model_size_gb || 0; });
    const gb = Object.values(onDisk).reduce((s,v) => s+v, 0);
    const orphans = all.filter(p => p.model_missing).length;
    note.textContent = all.length + ' profiles · ' + gb.toFixed(1) + ' GB on disk'
      + (orphans ? ' · ' + orphans + ' orphaned' : '');
  }

  if (!all.length) {
    el.innerHTML = '<div class="empty"><div class="empty-text">No profiles defined</div></div>';
    return;
  }
  if (!profiles.length) {
    el.innerHTML = '<div class="empty"><div class="empty-text">No profiles with weights on disk</div></div>';
    return;
  }
  // Selection must stay on a visible row, or Start launches something unseen.
  if (!eng.selectedProfile || !profiles.some(p => p.id === eng.selectedProfile)) {
    eng.selectedProfile = profiles[0].id;
  }
  {
    const q = s => String(s == null ? '' : s).replace(/'/g, "\\'");
    el.innerHTML = profiles.map(p => `
      <div class="profile-item ${eng.selectedProfile === p.id ? 'selected' : ''}"
           onclick="selectEngineProfile('${eng.key}', '${p.id}', this)">
        <div class="p-radio"></div>
        <div class="p-info">
          <div class="p-name">${p.name}${p.model_missing
            ? ' <span class="inv-no" title="No matching model directory on disk">weights missing</span>' : ''}</div>
          <div class="p-desc">${p.description}</div>
        </div>
        <div class="p-vram">${p.vram_gb != null ? p.vram_gb + ' GB' : '\u2014'}</div>
        <div class="p-actions" onclick="event.stopPropagation()">
          ${p.model_dir
            ? `<button class="btn-icon-del" title="Delete model weights from disk${
                 p.model_size_gb ? ' (' + p.model_size_gb + ' GB)' : ''}"
                 onclick="deleteProfileWeights('${eng.key}','${q(p.model_dir)}','${q(p.name)}',${p.model_size_gb || 0})">&#9679;</button>`
            : ''}
          <button class="btn-icon-del" title="Delete this profile script"
                  onclick="deleteEngineProfile('${eng.key}','${q(p.id)}','${q(p.name)}')">&#10005;</button>
        </div>
        ${eng.key === 'vllm' ? renderProfileSettings(p) : ''}
      </div>
    `).join('');
    // Warning and meta-error text is vendor-supplied (it embeds {value!r} of config
    // fields), so it is written with textContent, never interpolated into the HTML
    // above. T-04-07.
    el.querySelectorAll('.profile-item').forEach((card, i) => {
      const p = profiles[i];
      if (!p) return;
      const errEl = card.querySelector('.p-set-error');
      if (errEl) errEl.textContent =
        'Profile header is unreadable: ' + (p.meta_error || 'unknown parse failure');
      const warnEl = card.querySelector('.p-set-warn');
      if (warnEl && (p.warnings || []).length) {
        warnEl.textContent = '⚠ ' + p.warnings.join(' · ');
      }
    });
  }
}

// Five card states: header-error, recipe-backed, editable, read-only (04-03) and
// unparseable (04-03). meta_error is checked FIRST and on its own: a corrupt header is a
// defect, not a fallback (04-CONTEXT.md), so it must never fall through to the legacy
// rendering that would silently show a launch as ordinary.
function renderProfileSettings(p) {
  if (p.meta_error) {
    return `<div class="p-settings" data-state="header-error" onclick="event.stopPropagation()">
      <div class="p-set-error"></div>
      <div class="p-set-note">The script itself is still valid and can be launched, but its
        generated metadata cannot be read. Regenerate the profile to repair it.</div>
    </div>`;
  }
  const d = p.derived;
  if (d && d.editable === false) {
    return `<div class="p-settings" data-state="recipe-backed" onclick="event.stopPropagation()">
      <div class="p-set-row">
        <div class="p-set-field"><label>Context</label><input disabled placeholder="—"></div>
        <div class="p-set-field"><label>GPU memory util</label><input disabled placeholder="—"></div>
        <div class="p-set-field"><label>Max num seqs</label><input disabled placeholder="—"></div>
      </div>
      <div class="p-set-note">Not adjustable here — ${d.reason || 'the recipe YAML owns these flags'}.</div>
    </div>`;
  }
  if (!d) return '';  // legacy / unparseable: 04-03 owns those states.
  const rec = p.recommended || {};
  const recUtil = rec.gpu_memory_utilization != null
    ? `<span class="p-set-rec">recommended ${rec.gpu_memory_utilization}</span>` : '';
  const recCtx = rec.max_model_len != null
    ? `<span class="p-set-rec">recommended ${rec.max_model_len}</span>` : '';
  const ceiling = d.declared_max_context != null
    ? ` Over-requesting is clamped to the KV ceiling (${d.max_fitting_context}) and the
        vendor ceiling (${d.declared_max_context}); the request is discarded, not applied,
        so a value above those will silently give you less context than you typed.` : '';
  return `<div class="p-settings" data-state="editable" onclick="event.stopPropagation()">
    <div class="p-set-row">
      <div class="p-set-field">
        <label>Context ${recCtx}</label>
        <input type="number" min="1" step="1" data-override="max_model_len"
               placeholder="${d.max_model_len}">
      </div>
      <div class="p-set-field">
        <label>GPU memory util ${recUtil}</label>
        <input type="number" min="0.10" max="0.95" step="0.01"
               data-override="gpu_memory_utilization" placeholder="${d.util}">
      </div>
      <div class="p-set-field">
        <label>Max num seqs</label>
        <input type="number" min="1" max="256" step="1" data-override="max_num_seqs"
               placeholder="${d.max_num_seqs}">
      </div>
      <button class="btn btn-sm" onclick="reclaimPageCache(this)">Reclaim page cache</button>
    </div>
    <div class="p-set-note">Per-launch only — nothing is written to the script on disk.
      Leave a box empty to use the derived default shown in grey. Linux page cache is billed
      against the same budget as gpu-memory-utilization, so reclaim it before judging a
      utilization recommendation.${ceiling}</div>
    <div class="p-set-warn"></div>
  </div>`;
}

// Page cache counts against the util budget, so a correct recommendation looks broken
// until it is reclaimed. Same endpoint the Dry Run memory check offers.
async function reclaimPageCache(btn) {
  btn.disabled = true;
  try {
    const res = await apiFetch('/api/vllm/reclaim-cache', 'POST', {});
    toast('✓ ' + res.message, 'ok');
  } catch (e) {
    toast('Reclaim failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
  }
}

// The request-body builder. Mirrors `_collect_overrides` in app.py: a blank or
// whitespace-only box means "use the derived default", expressed as the key being ABSENT.
function collectProfileOverrides() {
  const panel = document.querySelector('#vllm-profile-list .profile-item.selected .p-settings[data-state="editable"]');
  if (!panel) return undefined;
  const out = {};
  panel.querySelectorAll('input[data-override]').forEach(inp => {
    const raw = String(inp.value == null ? '' : inp.value).trim();
    if (!raw) return;
    const num = Number(raw);
    out[inp.dataset.override] = Number.isFinite(num) ? num : raw;
  });
  return Object.keys(out).length ? out : undefined;
}

async function deleteEngineProfile(key, id, name) {
  if (!confirm('Delete the profile "' + name + '"?\n\nThis removes the start script only. Model weights on disk are untouched.')) return;
  try {
    await apiFetch(engines[key].api + '/profiles/' + encodeURIComponent(id), 'DELETE');
    toast('Profile deleted: ' + name, 'ok');
    if (engines[key].selectedProfile === id) engines[key].selectedProfile = null;
    await loadEngineProfiles(engines[key]);
  } catch(e) {
    toast('Delete failed: ' + e.message, 'err');
  }
}

// Weights deletion goes through the guarded inventory endpoint, so the in-use
// and profile-cross-reference checks apply here too. The cross-reference will
// always fire from this page — this profile references the model by definition.
async function deleteProfileWeights(key, dirPath, name, sizeGb) {
  const size = sizeGb ? ' (' + sizeGb + ' GB)' : '';
  if (!confirm('Delete the model weights for "' + name + '"' + size + '?\n\nPermanently removes:\n' + dirPath + '\n\nThe profile is kept and will fail to launch until the weights are re-downloaded.')) return;
  try {
    await apiFetch('/api/hf/inventory/delete', 'POST', {path: dirPath, force: true});
    toast('Weights deleted: ' + name + size, 'ok');
    await loadEngineProfiles(engines[key]);
  } catch(e) {
    toast('Delete failed: ' + e.message, 'err');
  }
}

function selectEngineProfile(key, id, el) {
  engines[key].selectedProfile = id;
  const container = document.getElementById(engines[key].ids.profiles);
  container.querySelectorAll('.profile-item').forEach(p => p.classList.remove('selected'));
  // Overrides are per-launch and per-profile: carrying a typed context across a selection
  // change would launch a different model with numbers the user meant for the old one.
  container.querySelectorAll('.p-settings input[data-override]').forEach(i => { i.value = ''; });
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
  // A FAILED PREFLIGHT is launch evidence, not a preflight-endpoint error. Only when this
  // engine's last Dry Run came back verdict=fail do we hold Start for a confirm, and the
  // block is force-overridable (mirrors the 'unified memory' force path below). The recheck
  // below throws? We fall through and start unforced — a dead preflight endpoint must not be
  // able to block the only GPU. Runs before any button state is touched, so declining leaves
  // the UI exactly as it was.
  let forceFromGate = false;
  if (eng.key === 'vllm' && eng._preflightVerdict === 'fail') {
    try {
      const r = await apiFetch('/api/vllm/preflight', 'POST', {profile: eng.selectedProfile});
      if (r && r.verdict === 'fail') {
        const blocking = (r.checks || []).filter(c => c.level === 'fail')
          .map(c => c.title + (c.detail ? ': ' + c.detail : '')).join('; ');
        if (!confirm('Preflight found blocking problems:\n\n' + blocking +
                     '\n\nRun Dry Run to see the details. Force start anyway?')) {
          toast('Start blocked by preflight: ' + blocking, 'err');
          return;
        }
        forceFromGate = true;
      }
    } catch (e) {
      toast('Preflight recheck failed (' + e.message + '); starting ungated', 'err');
    }
  }
  const btn  = document.getElementById(eng.ids.start);
  const prog = document.getElementById(eng.ids.prog);
  const log  = document.getElementById(eng.ids.log);

  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Launching\u2026';
  prog.classList.add('show');
  log.textContent = 'Sending start command\u2026';

  // Live load progress, streamed from `docker logs -f` server-side. Replaces a
  // 20s status poll that reported "Model loading\u2026" for ten minutes whether the
  // model was loading or the container had died two seconds in.
  const beginPoll = (d) => {
    toast('\u2713 ' + eng.name + ' starting', 'ok');
    log.textContent = d.message;
    if (eng.key !== 'vllm') { return legacyPoll(eng, d); }
    followLoadProgress(eng, d);
  };

  const startWith = (force) =>
    apiFetch(eng.api + '/start', 'POST',
             {profile: eng.selectedProfile, force,
              overrides: eng.key === 'vllm' ? collectProfileOverrides() : undefined,
              recipe: eng.key === 'llamacpp' ? eng.selectedRecipe : undefined});


  try {
    beginPoll(await startWith(forceFromGate));
  } catch(e) {
    if (String(e.message || '').includes('unified memory') &&
        confirm(e.message + '\n\nForce start anyway?')) {
      try {
        beginPoll(await startWith(true));
      } catch(forceErr) {
        toast('Force start failed: ' + forceErr.message, 'err');
        prog.classList.remove('show');
      }
    } else {
      toast('Start failed: ' + e.message, 'err');
      prog.classList.remove('show');
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = '\u25b6 Start Selected';
  }
}

// The pre-existing status poll, still used by engines with no progress stream.
function legacyPoll(eng, d) {
  const prog = document.getElementById(eng.ids.prog);
  const log  = document.getElementById(eng.ids.log);
  let pollCount = 0;
  const poll = setInterval(async () => {
    pollCount++;
    await loadEngineStatus(eng);
    const led = document.getElementById(eng.ids.led);
    if (led.classList.contains('on')) {
      const modelEl = document.getElementById(eng.ids.model);
      if (modelEl.textContent && modelEl.textContent !== 'Model loading…') {
        clearInterval(poll);
        toast('✓ ' + eng.name + ' is ready!', 'ok');
        prog.classList.remove('show');
      } else {
        log.textContent = d.message + '\n\nContainer running — model still loading…';
      }
    } else if (pollCount >= 30) {
      clearInterval(poll);
      log.textContent += '\n\n⚠ Timed out after 10 minutes — check logs';
      toast(eng.name + ' did not start within 10 minutes', 'err');
    }
  }, 20000);
}

function setProgress(eng, pct, phaseLabel, meta) {
  const bar   = document.getElementById(eng.ids.bar);
  const phase = document.getElementById(eng.ids.phase);
  const metaEl = document.getElementById(eng.ids.meta);
  if (bar) {
    // A real percentage means a determinate bar; drop the indeterminate sweep.
    if (typeof pct === 'number') { bar.classList.remove('spin'); bar.style.width = pct + '%'; }
    else { bar.classList.add('spin'); bar.style.width = ''; }
  }
  if (phase && phaseLabel) phase.textContent = phaseLabel;
  if (metaEl) metaEl.textContent = meta || '';
}

function followLoadProgress(eng, d) {
  const prog = document.getElementById(eng.ids.prog);
  const log  = document.getElementById(eng.ids.log);
  prog.classList.add('show');
  setProgress(eng, 0, 'Attaching to container log…', '');

  // fresh=1: a launch was just issued, so do not short-circuit on the *previous*
  // model's /health, and tolerate the container not existing for a moment.
  const es = new EventSource('/api/vllm/progress?fresh=1');
  const finish = (ok, msg) => {
    es.close();
    loadEngineStatus(eng);
    toast((ok ? '✓ ' : '✗ ') + msg, ok ? 'ok' : 'err');
    if (ok) setTimeout(() => prog.classList.remove('show'), 4000);
  };

  es.onmessage = (ev) => {
    let e; try { e = JSON.parse(ev.data); } catch { return; }
    const mins = e.elapsed_s != null
      ? Math.floor(e.elapsed_s / 60) + 'm' + String(Math.round(e.elapsed_s % 60)).padStart(2, '0') + 's'
      : '';
    const bits = [mins];
    if (e.weights_gib)   bits.push('weights ' + e.weights_gib + ' GiB');
    if (e.kv_cache_gib)  bits.push('KV ' + e.kv_cache_gib + ' GiB');

    if (e.status === 'ready') {
      setProgress(eng, 100, 'Ready', bits.join(' · '));
      finish(true, eng.name + ' is ready in ' + mins);
      return;
    }
    if (e.status === 'failed') {
      setProgress(eng, e.percent, 'Load failed', bits.join(' · '));
      // textContent, never innerHTML — this string is container output.
      log.textContent = 'LOAD FAILED\n\n' + (e.error || 'unknown cause')
        + (e.exit_code != null ? '\n\nContainer exit code: ' + e.exit_code : '')
        + (e.hint ? '\n\n→ ' + e.hint : '');
      finish(false, eng.name + ' failed to load');
      return;
    }
    setProgress(eng, e.percent, e.label || 'Loading…', bits.join(' · '));
    if (e.line) log.textContent = e.line;
  };
  es.onerror = () => {
    es.close();
    log.textContent += '\n\n⚠ Progress stream dropped — falling back to status polling.';
    legacyPoll(eng, d);
  };
}

async function dryRunProfile(eng) {
  if (!eng.selectedProfile) { toast('Select a profile first', 'err'); return; }
  const btn = document.getElementById(eng.ids.dryrun);
  const out = document.getElementById(eng.ids.preflight);
  btn.disabled = true;
  btn.innerHTML = '<div class="spin-icon"></div> Checking…';
  out.textContent = '';
  eng._preflightVerdict = null;
  try {
    const r = await apiFetch('/api/vllm/preflight', 'POST', {profile: eng.selectedProfile});
    renderPreflight(out, r);
    const t = {ok: 'Dry run clean', warn: 'Dry run passed with warnings', fail: 'Dry run found blocking problems'}[r.verdict];
    eng._preflightVerdict = r.verdict;
    toast(t, r.verdict === 'fail' ? 'err' : 'ok');
  } catch (e) {
    toast('Dry run failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '⚗ Dry Run';
  }
}

function renderPreflight(out, r) {
  const icons = {ok: '✓', warn: '⚠', fail: '✗', skip: '–'};
  out.textContent = '';
  const head = document.createElement('div');
  head.className = 'preflight-head pf-' + r.verdict;
  head.textContent = icons[r.verdict] + ' ' + r.profile;
  if (r.budget) {
    const b = document.createElement('span');
    b.className = 'preflight-budget';
    b.textContent = 'util ' + r.budget.util + ' → ' + r.budget.usable_gib
      + ' GiB usable of a ' + r.budget.budget_gib + ' GiB budget';
    head.appendChild(b);
  }
  out.appendChild(head);

  for (const c of r.checks) {
    const row = document.createElement('div');
    row.className = 'preflight-row pf-' + c.level;
    const t = document.createElement('div');
    t.className = 'pf-title';
    t.textContent = (icons[c.level] || '·') + ' ' + c.title;
    row.appendChild(t);
    if (c.detail) {
      const d = document.createElement('div');
      d.className = 'pf-detail';
      d.textContent = c.detail;
      row.appendChild(d);
    }
    if (c.fix) {
      const f = document.createElement('div');
      f.className = 'pf-fix';
      f.textContent = c.fix;
      row.appendChild(f);
      if (c.check === 'memory') {
        const btn = document.createElement('button');
        btn.className = 'btn btn-sm';
        btn.textContent = 'Reclaim page cache';
        btn.onclick = async () => {
          btn.disabled = true;
          try {
            const res = await apiFetch('/api/vllm/reclaim-cache', 'POST', {});
            toast('✓ ' + res.message, 'ok');
            dryRunProfile(engines.vllm);
          } catch (e) { toast('Reclaim failed: ' + e.message, 'err'); btn.disabled = false; }
        };
        row.appendChild(btn);
      }
    }
    out.appendChild(row);
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
    if (!resp.ok) {
      let msg = resp.statusText;
      try { const d = await resp.json(); msg = d.detail || JSON.stringify(d); } catch(e) {}
      throw new Error(msg);
    }
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
          } else if (ev.auto_profile) {
            const p = ev.auto_profile.profile || {};
            lines.push('✓ vLLM profile added: ' + (p.name || p.id || 'profile'));
            toast('✓ vLLM profile added', 'ok');
            loadUnifiedInventory();
            loadEngineProfiles(engines.vllm);
            loadWarmModels();
          } else if (ev.auto_profile_error) {
            lines.push('vLLM profile not auto-created: ' + ev.auto_profile_error);
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
    prog.classList.remove('show');
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
  html += '<th>Model</th><th>Task</th><th>Format</th><th>Dtype</th><th>Params</th><th>Size</th><th>Source</th><th>Script</th><th style="width:150px"></th>';
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
      : '<button class="btn-icon-del" title="Delete model" onclick="deleteInventoryModel(\'' + m.dir_path.replace(/'/g,"\\'") + "','" + (m.full_name || m.name).replace(/'/g,"\\'") + "'," + (m.size_gb || 0) + ')">&#10005;</button>';
    const canCreateVllm = m.source !== 'ollama' && !m.has_script
      && (m.format === 'safetensors' || m.format === 'pytorch')
      && (m.task_label === 'Text Gen' || m.task_label === 'Vision LLM');
    const createBtn = canCreateVllm
      ? '<button class="btn btn-sm" style="font-size:10px;padding:3px 8px" onclick="createVLLMProfileFromInventory(\'' + m.dir_path.replace(/'/g,"\\'") + "','" + (m.full_name || m.name).replace(/'/g,"\\'") + '\')">Create vLLM</button>'
      : '';

    html += '<tr>';
    html += '<td><div class="inv-model-name">' + m.name + '</div>' + (m.owner ? '<div class="inv-owner">' + m.owner + '</div>' : '') + '</td>';
    html += '<td><span class="inv-task-badge">' + (m.task_label || '\u2014') + '</span></td>';
    html += '<td><span class="inv-format-badge ' + formatClass(m.format) + '">' + formatLabel(m.format) + '</span></td>';
    html += '<td><span class="inv-badge ' + dc + '">' + m.dtype + '</span></td>';
    html += '<td style="font-family:var(--mono);font-size:11px">' + params + '</td>';
    html += '<td style="font-family:var(--mono);font-size:11px;white-space:nowrap">' + size + '</td>';
    html += '<td><span class="inv-source-badge ' + sourceClass(m.source) + '">' + sourceLabel(m.source) + '</span></td>';
    html += '<td>' + scriptBadge + '</td>';
    html += '<td><div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">' + createBtn + delBtn + '</div></td>';
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

async function createVLLMProfileFromInventory(dirPath, modelName) {
  try {
    const d = await apiFetch('/api/vllm/profiles/from-hf', 'POST', {path: dirPath, model_name: modelName});
    const p = d.profile || {};
    toast('vLLM profile added: ' + (p.name || p.id || modelName), 'ok');
    await loadUnifiedInventory();
    if (engines.vllm) {
      await loadEngineProfiles(engines.vllm);
    }
    await loadWarmModels();
  } catch(e) {
    toast('Profile creation failed: ' + e.message, 'err');
  }
}

async function deleteInventoryModel(dirPath, modelName, sizeGb) {
  const size = sizeGb ? ' (' + sizeGb + ' GB)' : '';
  if (!confirm('Delete "' + modelName + '"' + size + ' from disk?\n\nThis will permanently remove all files in:\n' + dirPath)) return;
  try {
    await apiFetch('/api/hf/inventory/delete', 'POST', {path: dirPath});
    toast('Deleted: ' + modelName + size, 'ok');
    await loadUnifiedInventory();
  } catch(e) {
    // A profile cross-reference is overridable; an in-use container is not.
    if (/profile script references/.test(e.message)) {
      if (!confirm(modelName + ' is referenced by a profile script.\n\nDeleting the weights will make that profile fail at next launch.\n\nDelete anyway?')) return;
      try {
        await apiFetch('/api/hf/inventory/delete', 'POST', {path: dirPath, force: true});
        toast('Deleted: ' + modelName + size, 'ok');
        await loadUnifiedInventory();
      } catch(e2) {
        toast('Delete failed: ' + e2.message, 'err');
      }
      return;
    }
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
    path = _APP_DIR / "favicon.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")

@app.get("/help", response_class=HTMLResponse)
async def help_page():
    docs_path = _APP_DIR / "docs.html"
    if not docs_path.exists():
        raise HTTPException(404, "docs.html not found")
    return HTMLResponse(docs_path.read_text())


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
