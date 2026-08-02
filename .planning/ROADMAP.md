# Roadmap: DGX-Model-Manager vLLM Loading Hardening

## Overview

A code audit of the vLLM surface (2026-08-02) found the download → auto-profile → launch
path broken end-to-end, plus an unauthenticated shell-injection route and launch defaults
that contradict the box's measured recipes. This milestone unbreaks that path, then
replaces the one-size-fits-all launch defaults with settings derived from each model's own
config — hybrid-attention-aware — with hand-measured recipes taking precedence. Phases 1
and 2 are independent and testable without a running vLLM; phases 3–5 build the recommender
into the generator and the UI.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Unbreak the load path** - Fix the three confirmed outages and the injection hole
- [ ] **Phase 2: Derived launch spec** - Hybrid-aware KV/context/utilization solver, pure and testable
- [ ] **Phase 3: Curated recipe overrides** - config.json recipe table that wins over derived values
- [ ] **Phase 4: Parameterized scripts + UI settings** - Env-var overrides and the context/util controls
- [ ] **Phase 5: Admission truth** - Admit on executor budget; identify reclaim target by docker label

## Phase Details

### Phase 1: Unbreak the load path
**Goal**: The download → auto-profile → launch path works end to end again, and no HTTP
request can put shell metacharacters into an executable start script.
**Depends on**: Nothing (first phase)
**Requirements**: REQ-01, REQ-02, REQ-03
**Success Criteria** (what must be TRUE):
  1. A `POST /api/hf/download` for a small repo runs to completion and emits exactly one
     terminal `complete` or `error` event (currently dies with `NameError: _HF_XFER`).
  2. A generated non-gpt-oss profile script invokes vLLM correctly for the configured
     image, verifiable by inspecting the script without launching it.
  3. `_build_vllm_profile_script` with a `model_name` containing `$(...)`, backticks, `"`,
     or a newline produces a script where those characters cannot execute.
  4. Starting the app bound to a non-loopback host with no API key configured either
     refuses to start or loudly degrades — `verify_auth` is never a silent no-op.
  5. The two committed DeepSeek profiles no longer mount only `snapshots/<rev>`.
  6. `pytest` passes with new regression tests for each of the above.
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Fix the HF download worker (`_HF_XFER`); guarantee exactly one terminal event per download stream (wave 1)
- [x] 01-02-PLAN.md — Make `vllm serve` an image property; regenerate the stale DeepSeek profiles (wave 2)
- [x] 01-03-PLAN.md — Shell-quote every dynamic atom; close the non-loopback no-key auth hole (wave 3)

### Phase 2: Derived launch spec
**Goal**: A pure function turns a model's `config.json` into correct context/memory numbers,
with hybrid attention handled properly.
**Depends on**: Nothing (independent of Phase 1)
**Requirements**: REQ-04
**Success Criteria** (what must be TRUE):
  1. `_derive_launch_spec(config)` returns attention-layer count, KV bytes/token, a maximum
     fitting context, and a recommended `--gpu-memory-utilization`.
  2. Hybrid models resolve correctly via `layer_types` → `hybrid_override_pattern` →
     `full_attention_interval` → `num_hidden_layers`, in that precedence.
  3. Nemotron-Super resolves to 8 attention layers (not 88), Qwen3.6 to 10 (not 40),
     qwen3-next to 12 (not 48), gpt-oss to 18 full + 18 sliding.
  4. VL models read from `text_config`.
  5. The derived utilization for qwen3-next-80b lands within 0.02 of the hand-measured
     0.55, which is the calibration check that the formula is sound.
  6. The function is pure — no filesystem, no network, no vLLM — and table-tested against
     every model config currently on the box.
**Plans**: 2 plans

Plans:
- [ ] 02-01: Attention-topology + KV-bytes-per-token solver with precedence chain
- [ ] 02-02: Context/utilization fitting and the calibration test table

### Phase 3: Curated recipe overrides
**Goal**: Hand-measured recipes beat derived defaults, and model-family flags stop being
guessed from substrings.
**Depends on**: Phase 2
**Requirements**: REQ-05
**Success Criteria** (what must be TRUE):
  1. A `vllm.recipes` block in config.json, keyed by model-name pattern, overrides derived
     values, following the same config→env→default precedence idiom as the `alerts` block.
  2. Qwen3.6 resolves to the measured 0.55 / 262144 / `qwen3_xml`.
  3. gpt-oss keeps 65536 and full-precision KV.
  4. Tool-parser selection comes from an explicit capability map, so Qwen2.5, Qwen-VL and
     DeepSeek-R1-Distill-Qwen no longer receive `qwen3_coder`.
  5. `--kv-cache-dtype fp8` is applied only where declared, not blanket.
**Plans**: 2 plans

Plans:
- [ ] 03-01: Recipe table, precedence resolution, and merge into the generator
- [ ] 03-02: Explicit tool-parser/capability map replacing substring guessing

### Phase 4: Parameterized scripts + UI settings
**Goal**: Launch settings are adjustable per-launch from the UI without rewriting or
clobbering the script on disk.
**Depends on**: Phase 3
**Requirements**: REQ-06
**Success Criteria** (what must be TRUE):
  1. Generated scripts use `${VLLM_MAX_MODEL_LEN:-<derived>}`-style placeholders.
  2. `_engine_start` passes overrides as environment to the subprocess — the script file is
     never mutated at launch, so there is no preview/confirm TOCTOU.
  3. The profile card shows editable context / utilization / max-num-seqs with the derived
     value as placeholder and the recommended value labelled.
  4. Hand-written legacy scripts render their parsed values read-only with an opt-in
     "parameterize" action that shows a diff before writing.
  5. Untrusted strings reach the DOM via `textContent`, not raw `innerHTML`.
**Plans**: 3 plans

Plans:
- [ ] 04-01: Emit parameterized scripts; env-override plumbing through `_engine_start`
- [ ] 04-02: Profile-card settings controls with recommended-value hints
- [ ] 04-03: Legacy-script parameterize action with diff preview; escape the raw-HTML paths

### Phase 5: Admission truth
**Goal**: Admission compares the same quantity vLLM will actually reserve, and credits the
right profile.
**Depends on**: Phase 4
**Requirements**: REQ-07
**Success Criteria** (what must be TRUE):
  1. Profiles declare an executor budget (`util × MemTotal`), not a weights estimate.
  2. Admission uses consistent binary GiB units throughout.
  3. The reclaim target is identified by a docker label (`dgx.profile=<id>`), not by
     substring-matching a served name that every script contains.
  4. Two concurrent starts cannot both pass admission against the same MemAvailable.
  5. A container that dies during weight load surfaces its exit code / OOM status in the
     UI rather than spinning for ten minutes.
**Plans**: 2 plans

Plans:
- [ ] 05-01: Executor-budget metadata + unit-consistent admission math
- [ ] 05-02: Label-based reclaim identification, launch lock, and failure surfacing

## Requirements Traceability

| Requirement | Phase | Source |
|---|---|---|
| REQ-01 downloads complete | 1 | Audit H2 — reproduced live (`NameError: _HF_XFER`) |
| REQ-02 generated profiles launch | 1 | Audit H4 — confirmed via `docker image inspect` |
| REQ-03 no shell injection | 1 | Audit C2 — confirmed reachable, auth is a no-op |
| REQ-04 derived settings | 2 | Audit H6 + hybrid-attention KV analysis |
| REQ-05 recipes win | 3 | Generated defaults contradict measured recipes |
| REQ-06 UI options | 4 | User request: context options + recommended settings |
| REQ-07 admission truth | 5 | Audit C1, H1, H5 |
