# DGX-Model-Manager (GB10 fork)

## What This Is

A single-file FastAPI web UI (`app.py`, ~6700 lines, embedded HTML/JS) for managing local
LLM inference engines on an NVIDIA GB10 DGX Spark. Fork of `calico88x/DGX-Model-Manager`,
kept deliberately upstream-mergeable. Runs as `dgx-model-manager.service` (systemd --user)
on Tailscale at `:8090`. Its primary job is switching which vLLM model occupies the box's
single unified-memory pool, plus HF model download/inventory, a recommendations advisor,
Discord alerting, and a dashboards index.

## Core Value

Switching the active vLLM model must be safe and correct: the right weights get mounted,
the box does not get pushed into swap, and litellm keeps routing through `vllm-active`.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] `vllm-active` served-name alias so model switches never break litellm routing
- [x] Discord alerting driven by config.json with persisted cooldown state
- [x] Advisor tab: curated KB diffed against installed profiles, one-click apply
- [x] Custom-dir HF repo mounting (`models--*` root, not the snapshot) — commit `6ffc899`

### Active

<!-- Current scope. Building toward these. -->

- [ ] REQ-01: HF model downloads complete successfully and report terminal status
- [ ] REQ-02: Auto-generated vLLM profiles actually launch on the configured image
- [ ] REQ-03: No HTTP-reachable path can inject shell into a generated start script
- [ ] REQ-04: Launch settings (context length, memory utilization, concurrency) are derived
      from the model's own config rather than one hardcoded default for every model
- [ ] REQ-05: Hand-measured recipes override derived defaults instead of being clobbered
- [ ] REQ-06: The UI exposes context/utilization options with the recommended value visible
- [ ] REQ-07: Memory admission compares like with like (executor budget, not a weights guess)

### Out of Scope

- Extracting the frontend out of `app.py` — the embedded-HTML shape is what keeps this fork
  mergeable with upstream; a split would fork it permanently.
- Replacing `start_*.sh` profiles with a database — "drop a script in the folder and it
  becomes a profile" is the app's core UX contract and upstream's design.
- Multi-model concurrent serving — the GB10's single unified pool makes it impossible.
- Grafana Cloud / hosted metrics — explicitly rejected for this box.

## Context

- **GB10 unified memory**: CPU and GPU share one ~121 GiB pool. `nvidia-smi` cannot report
  memory here, so admission is based on `/proc/meminfo` MemAvailable plus a reclaim credit
  for the profile about to be torn down.
- `--gpu-memory-utilization` is a fraction of that *whole* pool, so `0.75` claims ~91 GiB.
- Every start script does `docker rm -f vllm_node` first: starting a profile evicts the
  previous one. Only one large vLLM runs at a time.
- **Most models on this box are hybrid-attention.** Nemotron-Super is 8 attention layers of
  88 (`hybrid_override_pattern`), Qwen3.6 is 10/40, qwen3-next 12/48
  (`full_attention_interval`), gpt-oss 18 full + 18 sliding(128) of 36 (`layer_types`).
  A naive `num_hidden_layers` KV estimate overestimates these by 4–11x.
- KV cost at full context is therefore tiny for the MoE/hybrid models: Qwen3.6 needs only
  ~2.7 GB of fp8 KV at its full 262144 context. Utilization is the real memory lever,
  not `--max-model-len`.
- Blessed launch recipes live outside this repo and are hand-measured; DMM profiles are
  for quick switching and currently carry generic defaults that contradict them.

## Constraints

- **Tech stack**: Single-file FastAPI + embedded HTML/JS. No build step, no framework.
- **Compatibility**: Stay upstream-mergeable — no file splits, no restructuring `app.py`.
- **Hardware**: One GB10, 121 GiB unified pool, sm_121. Marlin MoE kernels are unsafe here
  for some architectures (see the gpt-oss token-soup root cause in the profile header).
- **Security**: Bound to Tailscale, currently with no API key set, so `verify_auth` is a
  no-op on every mutating endpoint. Docker access is root-equivalent.
- **Testing**: pytest suite in `tests/` (59 tests). vLLM is frequently down by design when
  the box is doing other work (e.g. Helix training), so tests must not require a live vLLM.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep frontend embedded in `app.py` | Upstream mergeability | ✓ Good |
| `vllm-active` alias for litellm | Model switch shouldn't require a litellm restart | ✓ Good |
| `moe_backend: ""` in config.json | Hardcoded Marlin caused gpt-oss token soup on sm_121 | ✓ Good |
| Profiles remain shell scripts | Core UX contract; readable, hand-editable | ✓ Good |
| Env-var overrides instead of rewriting scripts at launch | Avoids TOCTOU and clobbering hand-tuned scripts | — Pending |

---
*Last updated: 2026-08-02 after the vLLM-surface code audit (Codex xhigh + live verification)*
