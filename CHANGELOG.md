# Changelog

All notable changes to DGX Model Manager are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

Summarised from `git log v0.2.0..main` (2026-05-14 → 2026-10-13). Planning-only
(`.planning/`) and test-only commits are omitted.

### Security

- **Authenticated service default** — the shipped user unit no longer enables
  `MODEL_MANAGER_ALLOW_UNAUTH`. An authentication drop-in is provided for existing
  customized units. Server-side key hashes and browser-side plaintext token storage
  are now documented separately.
- **Mutating endpoints fail closed on an open bind** — with no API key and a
  non-loopback bind, `verify_auth` now returns 503 for every mutating request
  instead of allowing it. `MODEL_MANAGER_ALLOW_UNAUTH=1` restores open access and
  logs each unauthenticated mutation. The 10-second startup warning remains.
- **Shell-quoted profile generation** — every dynamic value written into a
  generated start script is shell-quoted, so a model name cannot inject commands.
- **Allow-listed launch overrides** — `max_model_len`, `gpu_memory_utilization`
  and `max_num_seqs` are the only overrides accepted, range-checked before they
  reach the script environment.
- **Escaped UI rendering** — one `esc()` helper is applied at every enumerated
  DOM sink (e.g. HF-browse results).

### Added

- **Read-only LiteLLM startup drift check** — Kubernetes mode compares the local
  router YAML with the ConfigMap and logs a mismatch or unavailable comparison
  without exposing config values, syncing, or restarting LiteLLM.
- **llama.cpp engine** — GGUF profiles driven by a config-owned quant ladder
  (`llamacpp.recipes`), run in the upstream `server-cuda` container.
- **Derived launch spec** — `_derive_launch_spec` computes attention topology,
  KV bytes/token, fitting context and recommended `--gpu-memory-utilization`,
  hybrid-attention aware; curated `vllm.recipes` override derived values.
- **Explicit capability map** (`model_capabilities.json`) for tool/reasoning
  parsers instead of substring guessing.
- **Parameterized profile scripts** — generated scripts read `VLLM_*` env
  placeholders; profile cards expose context/utilization/seqs overrides with
  derived defaults; legacy scripts can be parameterized via preview/apply with a
  sha256 confirm token.
- **Launch preflight** and `POST /api/vllm/reclaim-cache`; a live SSE
  load-progress stream with a determinate bar; a load-log tail view.
- **Pinned vLLM model** — a pinned profile is launched once per boot when nothing
  else holds vLLM.
- **VRAM admission control** before every engine launch, with reclaim credit for
  the profile being replaced and a headroom floor for profiles with no
  `# VRAM:` header.
- **Advisor tab** — recommendations knowledge base, research-refresh job, and
  approve/apply actions for model and tuning recommendations.
- **Dashboards tab** (discovers running web UIs; named hosts, k8s dedup) and a
  **Homelab** tab listing nodes with local and Tailscale IPs.
- **Unified-memory gauge** with sparkline in the header.
- **Config-driven alerting** with a pluggable notifier registry (Discord) and
  persisted cooldown state.
- **Unified background-job runner** with progress and history.
- **Nightly container image drift checker** (`image_updates.py`).
- **k8s-mode LiteLLM restart** — sync the ConfigMap and roll the deployment.
- Stable `vllm-active` alias served by every profile; systemd unit for the default
  model; new GB10 vLLM start profiles.
- pytest suite and a CI workflow running it on push/PR.

### Changed

- **Engines launch outside DMM's cgroup** (`systemd-run --user --scope`), so
  restarting the manager no longer stops a running model. Generated profiles use
  `docker run -d` and carry no `--restart` policy.
- **Active profile is identified by the `dgx.profile` docker label**, falling back
  to served-name matching only for unlabeled containers.
- **Overridden launches are admitted against the overridden footprint**, not the
  script header's static `# VRAM:` value; llama.cpp recipes resolve before
  admission for the same reason.
- Dashboard frontend extracted from `app.py` into `dashboard.html`.
- HF downloads use `hf_transfer` and filter out formats GB10 cannot serve.
- Profile tuning: Qwen3.6 util lowered to reclaim stranded KV; host headroom kept
  for Qwen Flash; the `lyf` profile retired.

### Fixed

- HF download worker crash (`NameError: _HF_XFER`); every download stream now
  emits exactly one terminal event.
- `vllm serve` subcommand treated as an image property; DeepSeek profiles no
  longer mount only `snapshots/<rev>`; HF repos in custom dirs generate working
  profiles.
- gpt-oss-120b Harmony token-soup on GB10 (sm_121).
- llama.cpp serves on the port its image healthcheck probes; `llamacpp_base`
  probes host port 8081.
- systemd start-limit settings placed in the correct unit section.
- Model directories that fail to parse are surfaced in the inventory.
- vLLM tab hook moved out of `switchTab`'s else-if chain.
- Discord 403 (explicit User-Agent); header memory-gauge rendering.

### Removed

- Two stale `.sh.bak` profile backups from the repository; `*.bak` is now
  gitignored.

## [0.2.0] - 2026-05-14

### Security

Thanks to [@DavRodSwede](https://github.com/DavRodSwede) for a thorough static
security audit that drove all changes in this release.
See [issue #3](https://github.com/calico88x/DGX-Model-Manager/issues/3) for the full report.

- **Authenticated debug and log endpoints** — `GET /api/debug/config`,
  `/api/debug/system`, `/api/debug/docker`, `/api/logs/app`,
  `/api/logs/litellm`, `/api/logs/engine/{engine}`, and `/api/litellm/config`
  now require auth. These endpoints previously exposed LiteLLM YAML (including
  upstream API keys), journalctl output, engine launch parameters, and host
  fingerprint data to anyone reaching port 8090.

- **Startup warning on open LAN** — app now emits a loud warning to the
  systemd journal and waits 10 seconds before accepting connections if binding
  to a non-loopback address with no API key configured. Suppressed by setting
  `MODEL_MANAGER_ALLOW_UNAUTH=1` in the service environment.

- **Configurable bind host** — `host` is now read from `config.json`
  (`app.host`) and passed to uvicorn, replacing the hardcoded `0.0.0.0`.

- **Atomic config write** — `config.json` is now written to a `.tmp` file and
  renamed atomically via `os.replace()`. A crash or power loss mid-write
  previously produced a truncated config that silently dropped the API key hash
  on next boot, re-opening unauthenticated access.

- **Config load errors now logged** — a corrupted `config.json` at startup
  previously silently fell back to empty defaults. The error is now printed to
  stdout and visible in the systemd journal.

- **HF download path containment** — the `local_dir` parameter is now resolved
  via `Path.resolve()` and validated against the HF cache and registered custom
  directories, mirroring the protection already on the delete endpoint. Symlink-
  based traversal is no longer possible.

- **Engine log symlink hardening** — `/tmp/{engine}_*.log` files are now
  created with `O_EXCL | O_NOFOLLOW`, preventing a symlink attack where a
  pre-created symlink could cause the log write to clobber an arbitrary file.

- **Relative paths for static files** — `favicon.png` and `docs.html` are now
  resolved relative to `app.py` via `_APP_DIR`, removing the hardcoded
  `~/DGX-Model-Manager/` folder name assumption.

### Changed

- **HF download destination validation** — downloads to a custom directory now
  require the directory to be pre-registered under Inventory → Scan Directories.
  Unregistered paths are rejected with a 403. Documented in the help page.

### Fixed

- **HF download error handling** — a 403 or other HTTP error response from the
  download endpoint previously caused the progress UI to spin silently with no
  feedback. Errors are now surfaced as a toast and the progress bar is hidden.

### Documentation

- Security section updated to reflect the expanded auth scope and first-boot
  warning behaviour.
- HF Download section documents the custom directory pre-registration
  requirement.
- Troubleshooting section adds entries for HF download 403 errors and silent
  download failures.
- Removed non-existent Display Name setting from the Settings section.

---

## [0.1.2a] - 2026-05-10

### Fixed

- Update paths for favicon and docs in app.py

---

## [0.1.2] - 2026-05-02

### Fixed

- **Multi-instance vLLM detection** — `_engine_status` now discovers all running vLLM containers dynamically via `docker ps`, parses each container's actual published port, and health-checks independently. Previously only checked the single configured base URL (port 8000), so containers on alternate ports (e.g., dual-mode Coder-30B on port 8001) showed `running: false` despite being healthy.

### Changed

- Status response includes a new `instances` array with per-container name, port, running state, and loaded model. Top-level `running`, `model`, and `container_info` fields remain backward-compatible.

---

## [0.1.1] - 2026-04-23

### Fixed

- **Passwordless sudo check false negative** — the LiteLLM tab showed a "Passwordless sudo not configured" warning even when the sudoers rule from `setup.sh` was correctly installed. The check ran `sudo -n systemctl restart --dry-run litellm`, which didn't match the exact command granted in `/etc/sudoers.d/model-manager-litellm` (sudo matches arguments strictly, and `--dry-run` isn't part of the grant). Replaced with `sudo -ln /bin/systemctl restart litellm`, which asks sudo whether the command would be permitted without actually executing anything and matches the installed rule exactly.

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
