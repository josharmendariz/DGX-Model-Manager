# app.py backend modularization — dependency notes

`app.py` is still a single-file FastAPI app (~6,400 lines after the frontend was
extracted to [`dashboard.html`](../dashboard.html) — see git history for that change).
This doc records what two research passes found about how the backend code would split
into modules, so a future session doesn't have to re-derive the call graph. **No backend
code has been moved yet** — this is analysis only.

## Proposed buckets

Line ranges below are from the pre-extraction file and still apply — all of these buckets
sit before the old HTML block, so line numbers are unchanged.

| Bucket | Range | Content |
|---|---|---|
| A. auth | ~1–330 | `_hash_key`, `_auth_open_unauthenticated`, `verify_auth`, `_MemoryHandler`, config globals |
| B. inference/parsing | ~330–1470 | Pydantic request models, `_infer_from_config`, `_derive_launch_spec`, KV-cache math, `_parse_hf_model_dir`, `_scan_directory` |
| C. status/sites | ~1470–2043 | `/api/status`, `/api/nodeinfo`, site discovery/probing |
| D. recommendations | ~2043–2396 | `/api/recommendations` + apply logic |
| E. alerts | ~2396–2645 | Discord/log notify, alert loop, image drift check |
| F. agents | ~2645–2820 | background agent runner (`/api/agents/*`) |
| G. ollama/litellm proxy | ~2820–2954 | `/api/ollama/*`, `/api/litellm/*` |
| H. engines | ~2954–4558 | engine start/stop/status, VRAM admission, preflight, recipe handling |
| I. profiles | ~4558–5610 | vLLM profile creation/parameterization from HF paths, script templating |
| J. hf inventory | ~5610–6052 | unified inventory, HF download/search/meta, inventory dirs |
| K. admin/debug | ~6052–6292 | config CRUD, debug/system, logs endpoints |

## Most depended-upon (extract last, or into a shared `core.py` first)

- **`_scan_profiles`** — the single most central function. Called from C, D, H, I, J, K
  (6 of the 11 buckets).
- **Config-derived module globals** — `_ENGINES`, `_engine_bases`, `_engine_dirs`,
  `_SITES*` (populated ~lines 92–234). Read by nearly every bucket.
- **B's inference/parsing helpers** (`_infer_from_config`, `_scan_directory`, etc.) —
  consumed by H, I, and J. High fan-in, low fan-out: a true leaf-inward utility bucket,
  safe to extract early despite being widely used, since it calls out to almost nothing.

## Leaf buckets (call out but aren't called into — safe to extract first)

- **K (admin/debug)** — only aggregates from other buckets' globals; nothing calls back in.
- **G (ollama/litellm proxy)** — only touches config globals, no calls into H/I.
- **D (recommendations)** and **E (alerts)** — leaf-ish except where F (agents) calls them.
- **A (auth)** — called by everyone via `verify_auth` dependency injection, but calls
  nothing else itself; extract essentially first, alongside the config globals.

## Module-level shared mutable state (candidates for a `state.py`)

| Variable | Bucket | Notes |
|---|---|---|
| `_app_config`, `_ENGINES`, `_engine_bases`, `_engine_dirs`, `_SITES*` | A | read everywhere |
| `_http` (AsyncClient), set in `_lifespan` | global | read by C, E, H, J |
| `_script_content_cache` | B | written by B, cleared by I and J |
| `_SITES_CACHE`, `_SITES_LOCK` | C | C only |
| `_refresh_lock` | D | shared `asyncio.Lock`, also used by F |
| `_last_alert_sent` | E | in-memory dict, loaded once from `_load_alert_state()`, mutated in `_send_alerts`, persisted via `_save_alert_state` |
| `_image_check_lock` | E | shared `asyncio.Lock`, also used by F |
| `_agent_history`, `_AGENTS`, `_agent_tasks` | F | in-memory, persisted via `_agent_history_save` |

Note: the HF-meta cache is **not** in-memory — `_fetch_hf_model_meta` reloads it fresh from
disk on every call via `_load_hf_meta_cache()`/`_save_hf_meta_cache()`, unlike alerts and
agents which keep a live module-level object. No `app.state` usage exists anywhere; all
shared state is plain module globals.

## Circular dependencies (block a clean linear split — resolve first)

- **B ↔ H**: the script-parsing trio (`_parse_script_meta`, `_classify_script`,
  `_parse_script_flags`, `_parse_templated_defaults`) is defined across B and H but called
  from both sides — B's `_scan_profiles` calls into H's script parsers, and H's
  `_engine_start` route calls B's `_scan_profiles`. These three functions should move into
  a shared module rather than staying in either B or H.
- **H ↔ I**: engine-lifecycle and profile-creation code call into each other repeatedly
  (`_preflight_static`→`_vllm_serve_command`, `_build_vllm_profile_script`→`_live_vllm_cfg`,
  `_overridden_vram_gb`→`_kv_dtype_from_config`, etc.). H and I are mutually entangled
  enough that they likely need to merge into one module (e.g. `vllm_launch_core.py`)
  rather than split cleanly.
- **F → D/E** (one-directional, not circular, but still a blocker): `_agent_core`
  dispatches directly into D's and E's core check functions, and F's `_AGENTS` dict embeds
  D's `_refresh_lock` and E's `_image_check_lock` as shared `asyncio.Lock` objects. F
  cannot be extracted before D and E's locks live in shared state that F can import.

## Suggested extraction order

1. A (auth) + the config-derived globals → shared `core.py`/`config.py`
2. B (inference/parsing), minus the H-coupled script-parsing trio
3. K (admin/debug), G (ollama/litellm proxy) — leaves, extract in either order
4. D (recommendations), E (alerts) — with their locks made importable from shared state
5. F (agents) — now safe, since D/E's locks are already shared
6. H + I merged (engines + profiles) — the largest, riskiest cut; the script-parsing trio
   from step 2 belongs here or in the shared core, not duplicated
7. J (hf inventory) — last, depends on B's helpers and lightly on I's profile listing

## Out of scope here

This doc is reference material only. It does not itself move any backend code, and does
not resolve the B↔H / H↔I circular dependencies — both are called out above as work a
future split needs to do explicitly.
