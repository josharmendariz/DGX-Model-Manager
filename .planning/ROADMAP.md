# Roadmap: DGX-Model-Manager vLLM Loading Hardening

## Overview

A code audit of the vLLM surface (2026-08-02) found the download → auto-profile → launch
path broken end-to-end, plus an unauthenticated shell-injection route and launch defaults
that contradict the box's measured recipes. This milestone unbreaks that path, then
replaces the one-size-fits-all launch defaults with settings derived from each model's own
config — hybrid-attention-aware — with hand-measured recipes taking precedence. Phases 1
and 2 are independent and testable without a running vLLM; phases 3–5 build the recommender
into the generator and the UI. Phase 6 hardens the other half of the recommender — the
research job that feeds its knowledge base — so a finding cannot cite a number no source
printed, and so synthesis runs on this box rather than an external service.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Unbreak the load path** - Fix the three confirmed outages and the injection hole
- [x] **Phase 1.1: Launch feedback** - INSERTED - Repair the Qwen3.6 profile; preflight + live load progress
- [x] **Phase 2: Derived launch spec** - Hybrid-aware KV/context/utilization solver, pure and testable
- [x] **Phase 3: Curated recipe overrides** - config.json recipe table that wins over derived values
- [ ] **Phase 4: Parameterized scripts + UI settings** - Env-var overrides and the context/util controls
- [ ] **Phase 5: Admission truth** - Admit on executor budget; identify reclaim target by docker label
- [ ] **Phase 6: Verifiable research capture** - Findings must cite a quote that exists; rules stay human-owned

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

### Phase 1.1: Launch feedback (INSERTED 2026-08-04)
**Goal**: A launch that will fail is caught before it costs ~100 GB and three minutes, and
one that is running shows real progress instead of an indeterminate ten-minute spinner.
**Trigger**: Loading `HF Qwen/Qwen3.6-35B-A3B-FP8` from the UI appeared to do nothing.
**Depends on**: Phase 1
**Success Criteria** (what must be TRUE):
  1. The Qwen3.6 profile launches and serves — verified by a live 178s cold start. ✓
  2. Generated profiles carry no `--restart` policy, so a bad script cannot survive a
     reboot and shadow `vllm-default-model.service`. ✓
  3. Preflight reports the entrypoint contract, restart policy, mount scope, container
     collision, image presence and memory budget without launching anything. ✓
  4. Preflight never reports a false failure: unresolvable paths and an unrunnable probe
     degrade to `skip`. ✓
  5. A container that dies during load surfaces its cause within seconds, not after ten
     minutes of "Model loading…". ✓
  6. `pytest` passes with regression tests for each of the three root causes. ✓ (130)
**Plans**: none — reactive work, executed without a PLAN.md.

**Not covered here, deliberately**: preflight's memory verdict does not credit the
memory a running engine will release, so it reads `fail` during a live switch. That is
Phase 5's admission-truth work (criterion 3, docker-label reclaim).

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
- [x] 02-01: Attention-topology + KV-bytes-per-token solver with precedence chain
- [x] 02-02: Context/utilization fitting and the calibration test table

**Verified** 2026-08-05 — VERIFICATION.md: 6/6 criteria PASS, verified against the real
`config.json` files on the box rather than the committed fixtures. Suite 310 passing.
Carried forward by design: `warnings` is threaded but unconsumed (Phase 4 surfaces it),
Qwen3.6's derived 0.42 vs hand-measured 0.55 (Phase 3 recipe precedence), and
`_derive_launch_spec` has no application call site yet (Phase 3 plan 03-01 wires it).

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
- [x] 03-01: Recipe table, precedence resolution, and merge into the generator
- [x] 03-02: Explicit tool-parser/capability map replacing substring guessing

**Executed without PLAN.md files** — planning paused before `gsd-planner` (2b6723d) and the
work was implemented directly against 03-CONTEXT.md and 03-RESEARCH.md, then committed as
716ca98. The plan bullets above record what was built, not a plan that was followed.

**Verified** 2026-08-17 — criteria checked against the implementation, not a VERIFICATION.md
run: 5/5 PASS. Suite 331 passing.
  1. `vllm.recipes` resolves via `_resolve_recipe_model`; `recipe_dir` follows the `alerts`
     config→env→default idiom. The env layer was the one real gap found at close-out and was
     added in `_live_vllm_cfg` — applied where the live block is read, so the generator stays
     a pure function of its `vllm_cfg` and cannot disagree with `_recipe_dirs`/`_recipe_util`.
  2. Qwen3.6 matches `Qwen/Qwen3.6-*` → `qwen3.6-35b-a3b-fp8-solo`, whose recipe carries
     0.55 / 262144; `qwen3_xml` + `qwen3` come from the capability map (was `qwen3_coder`).
  3. `_resolve_launch` short-circuits gpt-oss to its own template: 65536, no `--kv-cache-dtype`.
  4. `model_capabilities.json` gates emission on confidence (`recipe-proven` /
     `template-identical-to-recipe-proven`). Ambiguous architectures — `Qwen2ForCausalLM`,
     `Qwen3NextForCausalLM`, `NemotronHForCausalLM` — are deliberately not match keys, so an
     unknown model of that family gets no tool flags rather than a first-row guess.
  5. `_kv_dtype_from_config` returns fp8 only on a model-owned 8-bit floating KV declaration.

Carried forward by design: `warnings` is threaded through `_resolve_launch` but still
unconsumed (Phase 4 surfaces it).

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

### Phase 6: Verifiable research capture
**Goal**: A research run produces findings that cannot assert a number no source actually
printed, runs on this box's own stack rather than an external Codex call, and never writes
the recommendation rules a human tuned by hand.
**Depends on**: Nothing (independent of Phases 3–5; touches `research_refresh.py` and the
KB, not the launch path)
**Requirements**: REQ-08, REQ-09
**Success Criteria** (what must be TRUE):
  1. A source that 404s, times out, or returns an empty body aborts the run naming the dead
     URL. `_fetch`'s `[fetch failed: ...]` string can no longer reach the synthesis prompt,
     so a run against five dead pages produces zero findings instead of confident ones.
  2. Every finding carries a verbatim `quote`; a finding whose `quote` is not a substring of
     the fetched text for its own `source_url` is rejected before it is written, and the
     rejection is reported with the offending claim. Fabricated figures fail mechanically,
     with no reviewer judgment involved.
  3. Every finding's `source_url` is one of the URLs in `research_sources.json`. A URL the
     engine invented is rejected by the same gate.
  4. `RESEARCH_ENGINE=litellm` against the local router (`:30400`, `vllm-active`) completes a
     full run and its output passes the same gates as the Codex path — verified by running
     both engines over one identical fetched-source set and diffing accept/reject counts.
     The local path is schema-constrained via the router's structured-output mode, not
     regex-scraped out of prose.
  5. Local evidence is a first-class source: a benchmark run on this box is captured as a
     finding of kind `local` whose `quote` is its own recorded output, and a `local` finding
     outranks a `web` finding making a competing claim.
  6. Findings are append-only and dated at capture. A re-run can add a finding or supersede
     one by id; it cannot silently rewrite or delete one, so re-running research is
     non-destructive and therefore cheap to do often.
  7. The research job has no write path to `recommendations.json`. Rules reference finding
     ids; `apply()`'s `{**prior, **entry}` clobber of hand-edited `title` / `summary` /
     `action` / `severity` is gone.
  8. `GET /api/recommendations` reports the formalization backlog — how many KB rules are
     inert (`match.type == "manual"`, which `_eval_profile_match` can never fire) and how
     many rules rest only on superseded findings.
  9. `pytest` passes with regression tests for each gate, including a fabricated-quote
     fixture and a dead-source fixture.
**Plans**: 4 plans

Plans:
- [ ] 06-01: Fail-loud fetch with content-hash caching; a fetched-source manifest carrying
      retrieval dates that the verification gates read
- [ ] 06-02: Findings store — schema, quote + source-URL verification, append-only writes
      and supersede semantics
- [ ] 06-03: Local synthesis engine — schema-constrained `litellm` path against `:30400`,
      engine-parity test, `local`-kind findings from bench output
- [ ] 06-04: Rule/finding separation — rules reference finding ids, research loses KB write
      access, stale-and-inert reporting on the recommendations API

**Not covered here, deliberately**: `_eval_profile_match` still greps raw script text, so a
commented-out flag can fire a rule and `model_absent` is suppressed by an incidental mention
in a profile id. Matching should resolve through `_derive_launch_spec`'s parsed values
instead — that lands after Phase 4 parameterizes the scripts, and is scoped as its own phase.

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
| REQ-08 findings are verifiable | 6 | Research review 2026-08-17 — a discarded proposal asserted `33.53 tok/s`, `862.84 aggregate`, `115–120 GB` and source dates, none checkable against the fetched text |
| REQ-09 research runs locally | 6 | User request: synthesize on the box's own stack, not an external Codex call |
