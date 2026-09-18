---
phase: 04-parameterized-scripts-ui-settings
plan: 01
subsystem: vllm-launch
tags: [overrides, script-generation, admission, security]
requires: []
provides: [VLLM_env_placeholders, _OVERRIDE_ENV, _resolve_overrides, _overridden_vram_gb]
affects: [app.py, profiles/vLLM/*]
tech-stack:
  added: []
  patterns: [name-allow-list-at-http-boundary, defence-in-depth-shell-guard]
key-files:
  created: [tests/test_engine_start.py]
  modified: [app.py, tests/test_vllm_profile_generation.py, tests/test_vram_admission.py]
key-decisions: []
requirements-completed: [REQ-06]
duration: ~35 min
completed: 2026-08-20
---

# Phase 4 Plan 01: Parameterized vLLM scripts + override plumbing Summary

Generated vLLM scripts now carry `${VLLM_MAX_MODEL_LEN:-derived}` /
`${VLLM_GPU_MEMORY_UTILIZATION:-derived}` / `${VLLM_MAX_NUM_SEQS:-2}` placeholders, and
`_engine_start` delivers an allow-listed, range-validated override map into the launched
process environment without ever rewriting the script on disk.

## What Was Built

**Task 1 — parameterized emission + generator drift fix** (`775c1fd`)
- Split the welded `--max-model-len N --max-num-seqs 2` f-string on BOTH branches into
  separate `arg_lines` entries (the prerequisite refactor).
- Emitted the three placeholders unquoted (deliberately not through `shlex.quote`; the
  existing comment at the quoting block concerns dynamic atoms, placeholders are the
  opposite case). Defaults are today's derived values — an unset environment reproduces
  the previous script byte-for-byte in meaning.
- `_VLLM_OVERRIDE_GUARD`: a numeric guard block emitted between the preamble's
  `docker rm -f` and `docker run`, exiting 2 and naming the offending variable.
- `shape="recipe"` left entirely untouched (delegates to `run-recipe.sh` in another repo).
- Locked drift fix: generator now emits `docker run -d --name vllm_node`, not
  `exec docker run`. No `--restart` added.

**Task 2 — allow-listed overrides into process env** (`4c4a14e`)
- `_OVERRIDE_ENV`: request key → (env var, validator). Names allow-listed; not a prefix
  match, not pass-through.
- `_resolve_overrides()` returns env-ready strings or raises 400 naming the key and the
  accepted set. Util bounds reuse `_derive_launch_spec`'s `util_floor`/`util_cap`, and a
  test asserts the two stay equal by introspecting the signature.
- `overrides` threaded through `EngineStartRequest` → `_engine_start` → `env`, applied
  only for `k == "vllm"`. No `--setenv=` added to `_launch_argv`: `--scope` inherits.

**Task 3 — admission sees the override** (`8639fe0`)
- `_profile_source_dir()` reads the generator's `# Auto-generated ... from:` header — the
  only link from a scanned script back to its `config.json`.
- `_overridden_vram_gb()` re-derives via `_derive_launch_spec(requested_context=...)` and
  substitutes `vram_gb` into the profile dict, in the same `{**profile, ...}` shape the
  recipe block uses, before the `_vram_admission_check` await.

## Deviations from Plan

**1. [Rule 2 — missing critical] `gpu_memory_utilization` footprint is `pool_gb * util`,
not a `_derive_launch_spec` re-derivation**
- Found during: Task 3.
- Issue: the plan specified re-deriving the footprint via `requested_context=<override>`,
  which covers a `max_model_len` override but says nothing about a utilization override.
  `_derive_launch_spec` has no `requested_util` input, so there is nothing to feed it.
- Fix: a util override admits against `121.0 * util` — literally the share of the pool
  vLLM reserves up front. One multiplication, documented at the site.
- Files: `app.py` (`_overridden_vram_gb`). Commit `8639fe0`.

**2. [Rule 2 — missing critical] `_profile_source_dir()` was not in the plan**
- Found during: Task 3.
- Issue: `_engine_start` receives a *scanned script* profile dict with no path back to
  `config.json`, so `_derive_launch_spec(config, ...)` had no `config` to be called with.
  The plan's interface list did not surface this gap.
- Fix: parse the generator's own `# Auto-generated ... from:` header. Hand-written scripts
  have no such header and hit the force-required path the plan already mandates.
- Files: `app.py`. Commit `8639fe0`.

**3. [Rule 1 — bug in a new test] derived-default test used a config too thin to derive**
- Found during: Task 1. The existing `_write_model_config` fixture has no
  `max_position_embeddings`, so `_resolve_launch` nulled `max_model_len` and the test
  compared the fallback constant to itself (`32768 == 0` failure). Replaced with a
  real-shaped config so the assertion exercises the derived path.

**Total deviations:** 3 auto-fixed (2 missing-critical, 1 test bug).
**Impact:** no scope change; two small helpers exist that the plan did not name.

## Verification

| Check | Result |
|---|---|
| `python3 -m pytest tests/ -q` | **360 passed**, 0 failed (baseline 337 → +23) |
| `grep -v '^#' app.py \| grep -c 'exec docker run'` | 0 |
| `grep -v '^#' app.py \| grep -c 'docker run -d --name vllm_node'` | 1 |
| `bash -n` over every `profiles/*/*.sh` | all clean |
| New imports beyond stdlib | none |
| Shell-injection smoke: `VLLM_MAX_MODEL_LEN='4096; rm -rf /'` | guard exits rc=2 before `docker run` |

The `systemd-run --user --scope` env-transport test **ran** on this box (not skipped) and
passed — the phase's core premise is verified against the real wrapper, not a mocked
`Popen` call-args tuple.

## Known Stubs

None.

## Threat Flags

None. T-04-01 / T-04-02 / T-04-03 are each mitigated and covered by tests; T-04-04 is
covered by `test_script_not_mutated_by_a_launch_with_overrides`.

## Issues Encountered

None blocking. Note for 04-02/03: `_derive_launch_spec` **clamps** `requested_context` —
an over-request is warned about and discarded, not applied. A UI that offers a context
slider must surface that warning or users will silently get less context than they asked
for.

## Next

Ready for 04-02.

## Self-Check: PASSED
- `tests/test_engine_start.py` exists on disk.
- Commits `775c1fd`, `4c4a14e`, `8639fe0` all present in `git log`.
- All task acceptance criteria re-run and passing.
