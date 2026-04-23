# Changelog

All notable changes to DGX Model Manager are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] - 2026-04-23

### Added

- **Engine Registry System** — data-driven `_ENGINES` registry replaces hardcoded per-engine code. All engine routes, HTML, JavaScript, sidebar items, status pills, settings cards, and debug sections are generated dynamically from the registry. Adding a future engine requires only a registry entry (~12 lines) with zero code duplication.

- **llama.cpp Engine** — GGUF model inference engine support. Profiles auto-discovered from `~/llama.cpp/start_*.sh`. Default port 8080, health at `/health`, OpenAI-compatible at `/v1/models`.

- **LocalAI Engine** — multi-modal AI engine (LLM, TTS, STT, image generation). Profiles auto-discovered from `~/LocalAI/start_*.sh`. Default port 9090, health at `/readyz`, OpenAI-compatible at `/v1/models`.

- **ComfyUI Engine** — image generation workflow engine with its own web UI. Profiles auto-discovered from `~/ComfyUI/start_*.sh`. Default port 8188. "Open UI" button appears when running, linking directly to the ComfyUI interface.

- **Model Inventory Tab** — dedicated tab with unified view of all local models across HF cache, custom directories, and Ollama. Includes real-time search, filter by source/format/task, sort by name/size/params, stats bar, and bulk HF metadata enrichment.

- **HuggingFace Browse Tab** — built-in HF Hub search with pipeline type and sort filters. Result cards show task badges, download/like counts, and format tags. Expandable details load file lists with sizes and discover quantized variants (GGUF/GPTQ/AWQ). One-click download pre-fills the HF Download tab.

- **Logs & Debug Tab** — centralized diagnostics panel with six sections:
  - System Overview: hostname, IP, architecture, memory, Python version, uptime, disk usage, service health checks with latency, sudo/docker permissions
  - Running Configuration: collapsible sections for app config, LiteLLM YAML, and all engine profiles
  - Application Logs: in-memory ring buffer (500 entries) with level filtering, text search, 3-second auto-refresh, color-coded severity
  - Engine Logs: tabbed viewer for all engine log files from `/tmp/`
  - LiteLLM Logs: journalctl integration with search and auto-refresh
  - Docker Containers: live container state table

- **API Key Authentication** — optional API key protection for all mutating endpoints (POST/PUT/DELETE). Set via Settings tab or config. Key stored hashed in config file. All read-only endpoints remain accessible without auth.

- **Settings Tab** — centralized configuration UI for display name, service URLs (all 7 services), API key management, and service connectivity testing with one-click Test All.

- **HF Metadata Cache** — 7-day TTL cache at `~/model-manager/hf_meta_cache.json` for HuggingFace model metadata (pipeline_tag, downloads, likes). Avoids repeated API calls during inventory enrichment.

- **Format Detection** — automatic detection of model format (safetensors, GGUF, PyTorch, Ollama) from file extensions in model directories.

- **Task Classification** — maps HuggingFace `pipeline_tag` to human-readable labels (Text Gen, Vision LLM, Embedding, STT, TTS, Image Gen, etc.) with modality-based fallback inference.

- **Unified Inventory Endpoint** — `GET /api/inventory` combines HF cache scan, custom directory scan, and Ollama model list in one response. Ollama inclusion is optional via query parameter.

- **Lightweight Dirs Endpoint** — `GET /api/hf/inventory/dirs` returns just directory names without triggering a full model scan.

- **In-Memory Logging** — custom `_MemoryHandler` with `deque(maxlen=500)` ring buffer. Zero disk I/O, ~50KB memory. Captures app events, uvicorn access logs, and startup/shutdown lifecycle.

- **Engine Start Timeout** — polling exits after 30 iterations (10 minutes) with a timeout toast if the container never becomes healthy.

### Changed

- **Configurable Service URLs** — all service URLs (Ollama, LiteLLM, SGLang, vLLM, llama.cpp, LocalAI, ComfyUI) are now configurable via Settings tab and `config.json`. Previously hardcoded to localhost defaults.

- **Shared httpx Client** — single lifespan-managed `httpx.AsyncClient` replaces per-request client creation. Eliminates ~6+ client create/destroy cycles every 12 seconds from status polling.

- **Async Subprocess** — all `subprocess.run()` calls replaced with `asyncio.create_subprocess_exec()` via `_run()` helper with configurable timeout. No longer blocks the event loop during Docker/systemctl operations.

- **Profile Scan Caching** — script file contents cached per inventory request. Eliminates O(models x scripts) filesystem reads during cross-referencing.

- **Deduplicated Model Parsing** — shared `_infer_from_config()` extracts dtype, MoE, reasoning, and modality inference. Removed ~80 lines of duplicated logic between HF cache and flat directory parsers.

- **Deduplicated Engine Code** — generic `_engine_status()`, `_engine_stop()`, `_engine_start()` helpers replace per-engine implementations. Frontend uses single `engines` config object with generic handler functions.

- **Dynamic Route Generation** — all per-engine API routes (`/api/{engine}/profiles`, `/status`, `/start`, `/stop`) generated in a loop from the engine registry.

- **SGLang Element IDs Normalized** — SGLang HTML IDs changed from unprefixed (`engine-led`, `engine-card`) to prefixed (`sglang-engine-led`, `sglang-engine-card`) for consistency with all other engines.

- **Status Polling** — `pollStatus()` loops over the `engines` config object instead of hardcoded `setPill` calls. Supports any number of engines without JS changes.

- **Config Model Flexibility** — `ConfigUpdate` Pydantic model `services` field changed from required to optional, allowing API key operations without a dummy services object.

### Fixed

- **4 Missing Auth Headers** — `pullModel()`, `deleteModel()`, `hfDownload()`, and `removeInventoryDir()` were missing `authHeaders()` in fetch calls. Would fail silently when API key auth was enabled.

- **Config Key Check** — `loadConfig()` was checking `d.app.api_key` (never returned by backend). Fixed to `d.app.api_key_set`.

- **Stray Imports** — `import re` and `import time` were defined mid-file. Moved to the import block at top of file.

### Security

- **PII Removal** — removed all hardcoded hostnames, IP addresses, usernames, and device-specific references from the codebase. All identifying information is now derived from runtime config or system queries.

- **API Key Hashing** — API keys stored as SHA-256 hashes in `config.json`, never in plaintext. Verification uses `hmac.compare_digest()` on the hashes for timing-attack resistance. Legacy plaintext keys from older configs are auto-upgraded to hashes on first load.

### Removed

- **Hardcoded Service URLs** — `SGLANG_BASE`, `VLLM_BASE` and similar globals removed. Replaced by `_engine_bases` dict derived from registry + config.

- **Per-Engine Scan Functions** — `scan_sglang_profiles()` and `scan_vllm_profiles()` replaced by generic `_scan_profiles(engine_key)`.

- **Per-Engine Request Models** — `SGLangStartRequest` and `VLLMStartRequest` replaced by single `EngineStartRequest`.

- **Thin JS Wrappers** — `loadSGLangStatus()`, `loadVLLMStatus()`, `stopSGLang()`, `stopVLLM()`, etc. removed. HTML calls generic functions directly.
