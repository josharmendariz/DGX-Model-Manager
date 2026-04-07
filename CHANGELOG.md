# Changelog

All notable changes to DGX Model Manager are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.0.5] - 2026-04-07

### Added

- **vLLM Engine tab** — full start/stop/profiles support mirroring the SGLang tab. Docker-based, with its own profile system (`vllm_profiles.json`), status LED, 20-second polling, and ready detection requiring both container running + model serving. Status pill in the header bar shows live vLLM health alongside the other services.

- **vLLM status pill** — header bar now shows four service indicators: SGLang, Ollama, LiteLLM, and vLLM. The vLLM pill displays the active model name when running.

### Fixed

- **`~/` paths in profile scripts now resolve correctly** — both SGLang and vLLM start endpoints now call `os.path.expanduser()` on the script path before checking existence or executing. Previously, profiles using `~/sglang/start.sh` style paths would always fail with "Script not found".

- **Docker container detection is now exact-match** — status checks for SGLang and vLLM now use `name=^sglang$` / `name=^vllm$` filters instead of substring matching, preventing false positives from unrelated containers with similar names.

### Changed

- **config.json** — added `services.vllm_base` field (default `http://127.0.0.1:8000`).
- **docs.html** — added vLLM Engine tab guide (section 05), vLLM Profiles reference (section 09), vLLM troubleshooting entries. Architecture diagram updated to three backends. Model recommendation badges made engine-agnostic.
- **`/api/status`** — now includes `vllm` key with `ok` and `model` fields.
- **`/api/nodeinfo`** — now includes `vllm_port`.

---

## [0.0.4] - 2026-04-02

### Fixed

- **Ollama model delete** — frontend error handler now correctly parses FastAPI's `{"detail": "..."}` JSON error response instead of displaying the raw JSON string. Backend timeout increased from 30s to 60s and now handles 404 (model not found) distinctly from other errors, with a clear user-facing message in both cases.

- **SGLang ready toast fires too early** — the "SGLang is ready" toast previously fired as soon as the Docker container appeared in `docker ps`, which happens within seconds of launch. The model itself takes ~5 minutes to load. The poll now checks both `status.running` (container up) AND `status.model` (SGLang is serving a model at `/v1/models`) before firing the ready toast. The log message also updates mid-poll to show "Container running — model still loading…" so the user has visibility into the two-stage startup.

### Added

- **Favicon support** — added `/favicon.png` route to serve `favicon.png` from the app directory. Place `favicon.png` in `~/model-manager/` (same directory as `app.py`). Both the main app and the `/help` docs page now include `<link rel="icon" type="image/png" href="/favicon.png">`.

### Changed

- **README — PTXAS workaround updated** — documented `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` as the recommended Docker environment variable workaround for the SM121A PTXAS error on GB10. This allows use of the FlashInfer attention backend. The previous `--attention-backend triton` flag is retained as a fallback option. References upstream fix tracking at [triton-lang/triton#8539](https://github.com/triton-lang/triton/issues/8539).

---

## [0.0.3] - 2026-04-01

### Added

- Initial release
- Ollama tab — pull models with live progress, list installed models, delete models
- LiteLLM tab — one-click wildcard routing (`ollama/*`), live route list, config viewer, restart button
- SGLang tab — start/stop via configurable launch profiles, live status LED, 20-second polling
- HuggingFace Download tab — stream download any HF Hub model to local cache
- Live status bar — polls SGLang, Ollama, and LiteLLM health every 12 seconds
- Built-in `/help` documentation page
- `config.json` — single config file for all service URLs, paths, and app settings
- `setup.sh` — interactive setup: venv, UFW rules, sudoers entry, systemd service
- `sglang_profiles.json` — per-model SGLang launch profile system
