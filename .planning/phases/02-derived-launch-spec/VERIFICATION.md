---
phase: 02-derived-launch-spec
verified: 2026-08-05T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: n/a
deferred:
  - truth: "`warnings` reaches a consumer (UI or launch log)"
    addressed_in: "Phase 4"
    evidence: "Phase 4 goal: 'Launch settings are adjustable per-launch from the UI'; SC3 'profile card shows editable context / utilization ... with the recommended value labelled'. Accepted carry-forward #1."
  - truth: "Qwen3.6 resolves to the hand-measured 0.55 rather than the derived 0.42"
    addressed_in: "Phase 3"
    evidence: "Phase 3 SC2 verbatim: 'Qwen3.6 resolves to the measured 0.55 / 262144 / qwen3_xml'. Accepted carry-forward #2."
  - truth: "Preflight memory verdict is accurate during a live switch"
    addressed_in: "Phase 5"
    evidence: "Phase 5 'Admission truth'. Accepted carry-forward #3."
  - truth: "`_derive_launch_spec` is called by application code"
    addressed_in: "Phase 3 / Phase 4"
    evidence: "Phase 3 plan 03-01 'merge into the generator'; Phase 4 SC1 'Generated scripts use ${VLLM_MAX_MODEL_LEN:-<derived>}-style placeholders'. Phase 2's goal is the pure function itself; wiring is explicitly fenced out."
---

# Phase 2: Derived launch spec — Verification Report

**Phase Goal:** A pure function turns a model's `config.json` into correct context/memory
numbers, with hybrid attention handled properly.
**Verified:** 2026-08-05
**Status:** passed
**Re-verification:** No — initial verification
**Method:** Every success criterion was re-derived by calling the shipped helpers directly
against the **real `config.json` files on this box**, not against the committed fixture
snapshot. This defeats the self-fulfilling-fixture risk the plan itself flagged.

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `_derive_launch_spec(config)` returns attention-layer count, KV bytes/token, max fitting context, recommended util | VERIFIED | `app.py:948`. Independent call returns 16 keys including `full_attention_layers`, `kv_bytes_per_token`, `max_fitting_context`, `recommended_util`. Synthetic 32-layer config → 32 / 65536 / 1418304 / 0.24. |
| 2 | Hybrid models resolve via `layer_types` → `hybrid_override_pattern` → `full_attention_interval` → `num_hidden_layers`, in that precedence | VERIFIED | `app.py:754-864`. Adversarial synthetic configs declaring **all** branches simultaneously select the correct one at each rung: `layer_types` (12 full) → `hybrid_override_pattern` (8 full) → `full_attention_interval` (12 full) → `sliding_window` (48 bounded) → `dense` (48 full). See note below on the extra rung. |
| 3 | Nemotron-Super → 8 attention layers (not 88), Qwen3.6 → 10 (not 40), qwen3-next → 12 (not 48), gpt-oss → 18 full + 18 sliding | VERIFIED | Run against real on-disk configs: Nemotron-Super-120B `full=8 stateless=80 tot=88`; Qwen3.6-35B `full=10 stateless=30 tot=40`; qwen3-next-80b `full=12 stateless=36 tot=48`; gpt-oss-120b `full=18 bounded=18 win=128`. All four exact. |
| 4 | VL models read from `text_config` | VERIFIED | `/mnt/models/qwen3-vl-4b-fp8` config has **no top-level `num_hidden_layers`** — only `text_config.num_hidden_layers=36`. Resolver returns `tot=36 full=36`; `_kv_bytes_per_token` picks up `kv_heads=8`, `head_dim=128` (explicit, from `text_config`) → `per_layer=2048`. |
| 5 | Derived utilization for qwen3-next-80b within 0.02 of the hand-measured 0.55 | VERIFIED | Real config + **weight size measured from the on-disk safetensors (50.758 GB)**, not the fixture: `recommended_util = 0.54`, delta 0.010 ≤ 0.02. Negative control (dense 48-layer count) → 0.62, correctly outside the band. |
| 6 | Function is pure (no filesystem, network, vLLM) and table-tested against every model config on the box | VERIFIED | Source-token scan over all four helpers (docstrings stripped) finds zero forbidden tokens; all four execute with `builtins.open` patched to raise. Exactly **17 LLM configs** exist on the box (`/mnt/models`, `/opt/models`, HF hub) and the fixture snapshot contains exactly those 17. All 17 derive a spec without raising. |

**Score:** 6/6 truths verified

### Note on SC2's precedence chain

The implementation inserts one rung the roadmap wording does not name: after
`full_attention_interval` and before the dense `num_hidden_layers` fallback, it checks
`sliding_window` **AND** `use_sliding_window`. This is an addition, not a reduction — the
roadmap's four rungs remain in the stated order. It exists for a measured reason: 5 of the 17
configs on this box declare a vestigial `sliding_window` with `use_sliding_window=False`, and
honouring the window alone would collapse a 17.2 GB dense KV estimate to near zero. Verified
by the two-case check `window branch (flag True) → src=sliding_window` vs
`window ignored (flag False) → src=dense`. Recorded as an accepted enhancement.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app.py:_as_int` | Untrusted-scalar coercion | VERIFIED | :746, used throughout both resolvers |
| `app.py:_resolve_attention_topology` | 5-branch precedence classifier | VERIFIED | :754-864, 111 lines, `text_config`-first, warns on unknown types |
| `app.py:_kv_bytes_per_token` | Rate/total KV sizing | VERIFIED | :867-923, `head_dim` fallback guarded against zero attention heads |
| `app.py:_kv_budget_gb` | Page-cache-corrected KV headroom | VERIFIED | :926-945, `resident_gb` term present, clamped at 0.0 |
| `app.py:_derive_launch_spec` | Public 16-key entry point | VERIFIED | :948-1074, warnings propagated from topology, util clamped to [0.10, 0.95] |
| `tests/test_launch_spec.py` | V1-V9 + 17-model table | VERIFIED | 1017 lines, 180 tests, **0 skips** (`grep -c pytest.mark.skip` → 0) |
| `tests/conftest.py` | Fixture loaders | VERIFIED | Session-scoped `model_fixtures` / `fixture_models` |
| `.planning/.../02-MODEL-FIXTURES.json` | 17-model snapshot | VERIFIED | 17 rows, matches the 17 real on-box configs exactly |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_derive_launch_spec` | `_kv_bytes_per_token` | direct call | WIRED | :990 |
| `_kv_bytes_per_token` | `_resolve_attention_topology` | direct call | WIRED | :898 |
| `_derive_launch_spec` | `_kv_budget_gb` | direct call | WIRED | :1028 |
| topology `warnings` | spec `warnings` | list seed at :992 | WIRED | Verified: unknown layer types surface in `spec["warnings"]` |
| `_derive_launch_spec` | application code | (none) | DEFERRED | Zero call sites outside tests — by design; wiring is Phase 3/4 scope. See `deferred`. |

### Data-Flow Trace (Level 4)

| Artifact | Data variable | Source | Produces real data | Status |
|----------|---------------|--------|--------------------|--------|
| `_derive_launch_spec` | `kv_rate` / `bounded_total` | `_kv_bytes_per_token` ← real config fields | Yes — 17/17 real configs produce non-degenerate, per-model-distinct numbers | FLOWING |
| `_resolve_attention_topology` | layer counts | real `layer_types` / `hybrid_override_pattern` on disk | Yes — verified against the actual files, not the snapshot | FLOWING |
| calibration path | `recommended_util` | real config + on-disk safetensors byte count | Yes — 0.54 reproduced from files, no fixture involved | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full suite green | `python3 -m pytest tests/ -q` | `310 passed in 0.51s` | PASS |
| Launch-spec module green, no skips | `python3 -m pytest tests/test_launch_spec.py -q` | `180 passed in 0.08s` | PASS |
| Topology over real configs | direct call, 15 hub/opt configs | all resolve, 0 warnings | PASS |
| Calibration from disk | real config + measured 50.758 GB weights | `util=0.54`, delta 0.010 | PASS |
| Negative control discriminates | strip `layer_types`, re-derive | `util=0.62` (outside band) | PASS |
| Purity under patched `open` | all four helpers with `builtins.open` raising | no exception | PASS |
| Degenerate inputs | empty dict / negative / string layers / `weights_gb="banana"` | 16 keys, clamped util, warning trail, no traceback | PASS |
| Full 17-model derived table | all real on-box configs | 17/17 launchable specs | PASS |
| Phase 1 literals untouched | `grep -c "gpu-memory-utilization 0.75" app.py` | `1` (unchanged) | PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` exist in this repo and neither plan declares a probe.
Pytest is this phase's declared verification vehicle and was executed above.

### Requirements Coverage

| Requirement | Source plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| REQ-04 | 02-01, 02-02 | Launch settings derived rather than guessed | SATISFIED | `_derive_launch_spec` derives context/util/KV from config alone; calibration proves the arithmetic. Consumption of the derived values is Phase 3/4 (PROJECT.md REQ-04 stays unchecked until then). |

No orphaned requirements: ROADMAP maps only REQ-04 to Phase 2, and both plans claim it.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | None. `TBD` / `FIXME` / `XXX` / `TODO` / `HACK` / `PLACEHOLDER` scan over `app.py`, `tests/test_launch_spec.py`, `tests/conftest.py` returns zero matches. |

Notable non-defects checked and cleared:
- `return None` on `sliding_window` is deliberate (a dense model's vestigial window must not be
  multiplicable by a caller), documented, and tested.
- `_kv_bytes_per_token({})` returning zeros is a clamped answer, not a silent stub — the empty
  config produces a `warnings` entry at the `_derive_launch_spec` level.

### Observations (informational, not gaps)

1. **Summary test-count is off by one.** `02-02-SUMMARY.md` self-check claims
   `tests/test_launch_spec.py -q → 181 passed`; the actual count is **180**. `git log` shows
   the file is unmodified since `ce57024`, and `git diff --stat ce57024 HEAD` is empty, so
   nothing was lost — the summary simply miscounted. The suite-wide figure it claims (310) is
   exact.
2. **Several fixture rows carry implausible `weight_bytes`** (Qwen2.5-0.5B 11.5 MB,
   Qwen2.5-Coder-14B 11.5 MB, R1-Llama-70B 9.1 MB) because those models are not fully
   downloaded on this box. This does **not** affect any success criterion: the 17-row table
   asserts invariants (util in range, `max_model_len <= declared`, partition sums,
   `source_field` match), never a per-model util value, and the calibration model's 50.758 GB
   of weights are genuinely present on disk and were re-measured during this verification.
   Worth knowing before anyone reads the table's `recommended_util` column as ground truth for
   those three rows.
3. **R1-Distill-Llama-70B derives `max_model_len=0`** with a `budget-limited context` warning
   when scored against its real (partial) weight size. The behaviour is correct — a model that
   does not fit yields no context rather than a negative number — and it is the case the
   `warnings` propagation exists to make visible, which reinforces carry-forward #1's priority.

### Accepted Carry-Forwards (recorded, not counted as failures)

| # | Item | Owner phase | Confirmed in codebase |
|---|------|-------------|-----------------------|
| 1 | `warnings` propagated but unconsumed | Phase 4 | `spec["warnings"]` populated at :992/:1073; zero readers outside tests |
| 2 | Qwen3.6 derives 0.42 vs hand-measured 0.55 | Phase 3 | Reproduced independently (0.42); pinned by `test_calibration_qwen36_gap_is_left_for_phase_3` with a docstring forbidding a formula edit. Phase 3 SC2 names 0.55 explicitly. |
| 3 | Preflight memory verdict overstates during a live switch | Phase 5 | Out of Phase 2's surface; `_kv_budget_gb` supplies the `resident_gb` term Phase 5 needs |

### Human Verification Required

None. This phase's entire deliverable is a pure, deterministic function; every success
criterion was re-derived programmatically from real on-disk model configs during this
verification, including the calibration number. No visual, real-time or external-service
behaviour is in scope, and vLLM was correctly not required.

### Gaps Summary

No gaps. All six roadmap success criteria are satisfied by code that exists, is substantive,
is internally wired, and produces real data traced from real `config.json` files rather than
from the committed snapshot. The one architecturally significant absence — `_derive_launch_spec`
having no application call site — is an explicit scope fence, not an omission: Phase 3 merges
it into the generator and Phase 4 wires it to the UI, both stated in ROADMAP.md.

The strongest evidence that this phase achieved its goal rather than merely completing tasks:
the calibration is reproducible **outside** the test suite. Feeding the real
`/mnt/models/qwen3-next-80b-a3b-nvfp4/config.json` and the actual byte size of its safetensors
into `_derive_launch_spec` yields 0.54 against a 0.55 a human measured by hand months earlier,
while the naive dense count yields 0.62. The formula is doing real work, and the test that
guards it discriminates.

---

*Verified: 2026-08-05*
*Verifier: Claude (gsd-verifier)*
