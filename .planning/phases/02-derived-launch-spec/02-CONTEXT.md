# Phase 2: Derived launch spec - Context

**Gathered:** 2026-08-03
**Status:** Ready for planning
**Source:** Carried forward from the 2026-08-02 vLLM-surface audit and the Phase 1 handoff
(`.planning/.continue-here.md`, `.planning/HANDOFF.json`), not from a fresh discuss-phase.

> **Why no discuss-phase:** the audit plus a working, hand-validated prototype already
> settled every open question this phase would have asked — the precedence chain, the
> formula, the calibration target, and the scope fence. This mirrors the Phase 1 decision to
> bootstrap `.planning/` by hand rather than re-deriving established context with research
> agents.

<domain>
## Phase Boundary

Add a **pure function** to `app.py` that turns one model's parsed `config.json` into correct
launch numbers: how many layers actually hold a KV cache, how many bytes per token that
costs, the largest context that fits the GB10's unified pool, and the
`--gpu-memory-utilization` that should be requested.

**In scope:**
- Attention-topology resolution (hybrid-aware) from an already-parsed config dict.
- KV bytes/token, max fitting context, recommended utilization.
- A table test over every model config currently on the box.

**Explicitly OUT of scope — do not let the phase grow into these:**
- The `vllm.recipes` table in `config.json` and any hand-measured override precedence → **Phase 3**.
- The tool-parser / capability map replacing `qwen3_coder` substring guessing → **Phase 3**.
- Emitting the derived numbers into generated scripts, `${VLLM_MAX_MODEL_LEN:-…}`
  placeholders, or any UI control → **Phase 4**.
- Admission / executor-budget math → **Phase 5**.

Phase 2 ends when the function exists, is correct, and is tested. **Wiring it into
`_build_vllm_profile_script` is Phase 4's job, not this one.** The flat `0.75` / `32768` /
`65536` literals at `app.py:2560-2573` stay exactly as they are in this phase.
</domain>

<decisions>
## Implementation Decisions

### Locked — attention topology
- Resolution follows a strict precedence chain, first hit wins:
  `layer_types` → `hybrid_override_pattern` (count `*`) → `full_attention_interval`
  (`num_hidden_layers // interval`) → `num_hidden_layers` (dense fallback).
- **This is the load-bearing decision of the phase.** Measured on this box, a naive
  `num_hidden_layers` estimate overestimates KV by 4–11x on every hybrid model present. A
  recommender without it refuses context that is actually free.
- VL models nest their transformer fields under `text_config`; read from there when present.
- Sliding-window layers are counted separately from full-attention layers and cost
  `sliding_window` tokens of KV, not the full context.

### Locked — the formula
- `kv_bytes_per_token = 2 × attn_layers × num_key_value_heads × head_dim × dtype_bytes`
- `head_dim` falls back to `hidden_size // num_attention_heads` when not declared.
- `recommended_util = (weights + KV + ~6 GB overhead) / 121 + 0.04`, where 6 GB covers CUDA
  context, graphs and activations, and 121 is the GB10 unified pool in GB.
- Utilization is clamped; it is a fraction of the whole unified pool, per the note at
  `app.py:295-305`.

### Locked — purity
- `_derive_launch_spec(config)` takes an already-parsed dict and returns a dict. **No
  filesystem access, no network, no vLLM, no `/proc` reads inside the function.** The pool
  size and weight size are parameters, not lookups.
- Rationale: vLLM is intentionally down and the whole phase must be verifiable offline. A
  function that reads the box cannot be table-tested.

### Locked — the calibration check
- The derived utilization for `qwen3-next-80b` must land within 0.02 of the hand-measured
  **0.55**. The prototype independently produced **0.54**.
- This agreement *is* the evidence the approach is sound. Encode it as a test. If a later
  refactor breaks it, the formula is wrong — not the hand-measured recipe.

### Claude's Discretion
- Function/parameter naming beyond `_derive_launch_spec`, the exact returned dict keys, and
  how the topology resolver is factored (one function or a helper pair).
- Whether the fixture table is inlined in the test module or lives as a JSON fixture.
- Where in `app.py` the function is placed.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### The validated prototype
- `.planning/phases/02-derived-launch-spec/kvcalc-prototype.py` — scratch-quality working
  implementation of the precedence chain and the utilization formula. **Re-derive it into
  `app.py` as a pure function; do not import it.** Note its impurity (globs the filesystem)
  is exactly what must not survive the port.

### Closest existing analog in `app.py`
- `app.py:677` `_infer_from_config(config, name_hints) -> dict` — the house idiom for a pure
  config-dict-in / dict-out inference helper. Match its shape, docstring style, and
  `.get()`-with-fallback defensiveness.
- `app.py:2478` `_vllm_serve_command(image, cfg) -> str` — Phase 1's example of the
  documented-rationale docstring standard expected in this codebase.
- `app.py:280-305` `_get_total_memory_gb` / `_get_available_memory_gb` — where the 121 GB
  unified-pool reality is documented. Do **not** call these from the pure function.

### Consumers this phase must not touch yet
- `app.py:2506` `_build_vllm_profile_script` — Phase 4 wires derived values in here.
- `app.py:2555-2588` — the flat `0.75` / `32768` / `65536` / `fp8` literals being replaced later.

### Test conventions
- `tests/test_vllm_profile_generation.py` — the nearest test module; follow its structure.
- `tests/conftest.py` — shared fixtures.
</canonical_refs>

<specifics>
## Specific Ideas

Ground-truth expectations measured on this box (these are the table-test rows):

| model | attention layers / total | source field | KV fp8 @ max ctx | naive estimate |
|---|---|---|---|---|
| Nemotron-3-Super-120B | 8 / 88 | `hybrid_override_pattern` | 1.1 GB @ 262k | 11.8 GB |
| Qwen3.6-35B-A3B-FP8 | 10 / 40 | `hybrid_override_pattern` | 2.7 GB @ 262k | 10.7 GB |
| qwen3-next-80b | 12 / 48 | `full_attention_interval` | 3.2 GB @ 262k | 12.9 GB |
| gpt-oss-120b | 18 full + 18 sliding(128) / 36 | `layer_types` | 2.4 GB @ 131k | 4.8 GB |
| R1-Distill-Qwen-32B | 64 / 64 (dense) | — | 17.2 GB @ 131k | 17.2 GB |

A machine-generated inventory of every config on the box — including the exact field values
and any anomalies — is at `02-MODEL-FIXTURES.json`, produced and spot-verified for this phase.

**Consequence worth keeping in view (informs Phases 3–4, not this one):**
`--max-model-len 32768` is leaving nearly-free context on the floor. On the hybrid/MoE
models here, **utilization is the real memory lever, not context length** — Qwen3.6 needs
only ~2.7 GB of fp8 KV at its full 262144 context.
</specifics>

<deferred>
## Deferred Ideas

- Curated recipe overrides and the `vllm.recipes` config block — Phase 3.
- Explicit tool-parser capability map — Phase 3.
- Parameterized scripts, env-var overrides, UI context/utilization controls — Phase 4.
- Executor-budget admission, docker-label reclaim, launch lock — Phase 5.
- One live launch of a regenerated DeepSeek profile to close out Phase 1 — blocked on Helix
  training finishing; unrelated to this phase's verification.
</deferred>

<constraints>
## Environment Constraints

- **vLLM is intentionally DOWN** (GB10 running local Helix training). Every Phase 2 check
  must be `pytest` — no model load, no container start. This is not a hardship: the
  deliverable is a pure function and has no excuse to need a model.
- The GB10 is a **noisy benchmark host**; nothing in this phase should assert on timings.
- ~6 concurrent Claude sessions run in this repo. `profiles/vLLM/start_hf_openai_gpt-oss-120b.sh`
  (modified) and `start_hf_qwen_qwen3-{8b,14b}.sh` (untracked) are **someone else's dirty
  files** — commit by explicit path, never `git add -A`.
- Branch is `rec-approve-actions`. Do not create, rename, or switch branches.
</constraints>

---

*Phase: 02-derived-launch-spec*
*Context carried forward 2026-08-03 from the Phase 1 handoff*
