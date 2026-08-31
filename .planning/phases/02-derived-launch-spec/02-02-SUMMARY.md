---
phase: 02-derived-launch-spec
plan: 02
subsystem: infra
tags: [vllm, kv-cache, gpu-memory-utilization, calibration, pure-functions, pytest]

# Dependency graph
requires:
  - phase: 02-derived-launch-spec
    provides: "plan 02-01's _resolve_attention_topology and _kv_bytes_per_token, the committed 17-model fixture snapshot, and the V1-V9 test module"
provides:
  - "app._derive_launch_spec(config, ...) -> the sixteen-key public launch spec: layer counts, KV bytes/token, max fitting context, max_model_len, kv_gb, recommended_util, warnings"
  - "app._kv_budget_gb(util, pool_gb, weights_gb, overhead_gb, resident_gb) -> page-cache-corrected KV headroom in GB, clamped at 0.0"
  - "tests/test_launch_spec.py — V7 calibration, V8 purity, V9 degenerate and the 17-model derived-spec table; zero skips remain"
affects: [03-recipes-and-capabilities, 04-script-wiring, 05-admission-budget]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Calibration-as-test: an automated check against a number a human measured by hand, with a negative control proving the check discriminates"
    - "Untrusted-scalar coercion that NAMES the bad field in a returned warnings list rather than silently zeroing it"
    - "Purity enforced two ways: an inspect.getsource token scan over a named helper tuple, plus a call under a patched builtins.open"

key-files:
  created: []
  modified:
    - app.py
    - tests/test_launch_spec.py

key-decisions:
  - "resident_gb is a first-class budget term, not padding — CUDA reports MemFree not MemAvailable on this box"
  - "recommended_util is clamped into [0.10, 0.95] before it can reach a launch, so an absurd config saturates instead of over-asking"
  - "topology warnings are propagated into the spec's warnings list (the Wave 1 open question), answered yes"
  - "V8's purity scan covers all four helpers, which forced two Wave 1 docstrings off the '/proc' token"

requirements-completed: [REQ-04]

# Metrics
duration: 25min
completed: 2026-08-05
---

# Phase 2 Plan 02: Derived launch spec and the utilization solver Summary

**`_derive_launch_spec` turns one parsed `config.json` plus a weight size into a launchable
`--gpu-memory-utilization` and `--max-model-len`, and independently reproduces the
hand-measured 0.55 for qwen3-next-80b as 0.54 — the evidence the whole derived-spec approach
is sound.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-08-05T00:35Z
- **Completed:** 2026-08-05T01:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- **Calibration closed (V7).** `qwen3-next-80b-a3b-nvfp4`, rebuilt from the committed fixture
  row with nothing but its config fields and `weight_bytes / 1e9`, derives `0.54` against the
  hand-measured `0.55`. The tolerance bound is asserted explicitly and the hand-measured
  number appears in the failure message.
- **The calibration is proven to discriminate.** The same row scored with the naive
  `num_hidden_layers` count (48 attention layers instead of 12 — the 4x KV overestimate) lands
  at `0.62`, outside the ±0.02 band. Without this negative control the calibration test would
  still pass if hybrid classification regressed entirely.
- **`_kv_budget_gb` encodes the measured GB10 page-cache constraint.** At the unchanged 0.55,
  Qwen3.6 has 23.09 GB of KV headroom on an idle box and exactly `0.0` with ~25 GB resident —
  reproducing the Phase 1.1 failure where the engine had 0.19 GiB instead of 25.97 GiB and
  refused to start while blaming `max_model_len`. Derived utilization rises monotonically with
  `resident_gb` (0.54 → 0.62 → 0.70 at 0 / 10 / 20 GB).
- **Purity is proven, not asserted (V8).** A source-token scan runs over all **four** helpers,
  including the two ported from the prototype whose documented impurity was scanning the model
  cache — plus a behavioural half that runs every helper with `builtins.open` patched to raise.
- **Nine degenerate configs return clamped answers (V9)**, all sixteen keys, no traceback:
  empty dict, negative and string layer counts, zero attention heads, a 10⁹-layer config,
  `weights_gb="banana"`, negative weights, `pool_gb=0`, and a 1e12 overhead.
- Suite went 237 → 310 passing (+73), 3 skipped → **0 skipped**, still sub-second.

## Task Commits

1. **Task 1: `_kv_budget_gb` + `_derive_launch_spec`** — `d072e78` (test, RED: 23 failures) →
   `b5e3664` (feat, GREEN)
2. **Task 2: V7 / V8 / V9 + the 17-model spec table** — `ce57024` (test)

## TDD Gate Compliance

Task 1 shows the full RED → GREEN sequence: `d072e78` committed 23 failing tests, `b5e3664`
turned them green. RED was verified by running the suite between commits, not assumed. No
REFACTOR commit — neither helper needed cleanup.

Task 2 is marked `tdd="true"` in the plan but its own `<action>` states it "changes no
production code", so it has no implementation step and therefore no RED gate: the tests were
written against the code Task 1 had already shipped and passed on first run. That is the
plan's intent, not a skipped gate — Task 2 is the *verification* half of Task 1's RED/GREEN
pair, expressed against the fixture snapshot rather than against synthetic configs.

## Files Created/Modified

- `app.py` — `_kv_budget_gb` (:926) and `_derive_launch_spec` (:948), inserted between
  `_kv_bytes_per_token` and `_parse_hf_model_dir`. `_vllm_serve_command` (:3553) and
  `_build_vllm_profile_script` (:3581) are untouched, and
  `grep -c "gpu-memory-utilization 0.75"` is unchanged at `1`.
- `tests/test_launch_spec.py` — 130 → 181 tests: `test_budget_*` and `test_fitting_*` for the
  solver, then V7/V8/V9 and the 17-model derived-spec table replacing the three stubs.

## Decisions Made

- **`warnings` is propagated** (the explicit Wave 1 handoff question). `_derive_launch_spec`
  seeds its list from `topology["warnings"]` and appends its own. A silently-collected warning
  is only half the T-02-01 mitigation, and the entry point is the only layer a caller sees. No
  UI or logging was built for it — that stays Phase 4.
- **`sliding_window` is consumed only through `bounded_bytes_total`.** Per the Wave 1 handoff,
  the raw config value is never read around the `None` that `_resolve_attention_topology`
  returns when nothing is bounded.
- **Utilization is clamped, then rounded.** `round(min(cap, max(floor, round(raw, 2))), 2)` —
  the inner round reproduces the prototype's numbers exactly, the clamp makes an absurd config
  saturate at 0.95 rather than request memory the box does not have (T-02-08). Qwen2.5-0.5B's
  raw 0.091 becomes exactly `0.10`.
- **Negative scalars warn as well as floor.** The plan required `max(0.0, float(x))` with a
  warning on non-numeric input; a negative `weights_gb` is equally a caller bug, so it also
  names the field. Costs nothing and makes T-02-06's trail complete.
- **`requested_context` caps but never inflates `max_model_len`**, and a request above what
  fits is refused with a warning rather than honoured. The interface's `max_model_len` formula
  is unchanged for the default `None` case.
- **Qwen3.6's 0.42 is asserted as-is**, with a test docstring stating that closing the gap by
  editing the formula would break the only number validated against a measurement. That gap is
  Phase 3's recipe precedence.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Two Wave 1 docstrings reworded off the `/proc` token**

- **Found during:** Task 1 (pre-emptively, to keep Task 2 free of production edits)
- **Issue:** V8 requires scanning all four helpers for a forbidden-token list that includes
  `/proc`, but `_resolve_attention_topology` and `_kv_bytes_per_token` each said "No
  filesystem, network, /proc or vLLM" in prose. The scan would have failed on a docstring
  describing the very property it asserts.
- **Fix:** Reworded both to "No filesystem, network, kernel meminfo or vLLM", preserving the
  meaning. Done in Task 1 because the plan requires Task 2 to introduce no `app.py` hunk —
  verified: `git diff -U0 app.py` shows nothing from Task 2.
- **Files modified:** `app.py`
- **Commit:** `b5e3664`

### Documented Non-Fixes

**2. The plan says 6 non-dense fixture rows; there are 7**

- **Found during:** Task 2, writing the hybrid-benefit assertion
- **Detail:** The snapshot carries 5 `layer_types` rows and 2 `hybrid_override_pattern` rows.
  The test parameterizes over the actual non-dense set and asserts `len(HYBRID_ROWS) >= 6`
  rather than hardcoding either number, so a future snapshot regeneration that adds a hybrid
  model does not need a test edit. No behavioural difference.

**3. `FORBIDDEN_SOURCE_TOKENS` contains the literal strings `glob` and `expanduser`**

- **Detail:** Plan 02-01 carried an acceptance grep asserting those words were absent from the
  test module. This plan explicitly mandates them as members of the V8 token list, so they now
  appear — as data being searched *for*, not as an operation. The semantic rule that matters
  (VALIDATION's Wave 0 fixture rule) still holds: no test in this module reads a model config
  off the filesystem; every one loads the committed snapshot.

---

**Total deviations:** 1 auto-fixed (blocking), 2 documented non-fixes.
**Impact on plan:** No scope creep, no behavioural change. Every functional acceptance
criterion in both tasks passes verbatim.

## Known Stubs

None. No skipped tests remain in `tests/test_launch_spec.py`
(`grep -c "pytest.mark.skip"` → `0`), and both new helpers are fully implemented.

Both helpers remain unreferenced by application code **by design** — the plan's scope fence
puts wiring into `_build_vllm_profile_script` in Phase 4. This is not a stub: the deliverable
of this phase is a correct, tested pure function, and its consumer is a later phase.

## Issues Encountered

- The plan's example for the budget-binding warning (`weights_gb=110`) drives
  `max_fitting_context` to 0, which makes the "budget binds *below* the declared max" case
  indistinguishable from the "nothing fits at all" case. The test uses `weights_gb=107.0`
  instead, which yields a non-zero fitting context of 158,691 tokens against a declared
  262,144 — the regime the warning actually exists to describe.
- `max_model_len = min(declared or fitting, fitting)` means a config with no
  `max_position_embeddings` inherits the fitting context. That is deliberate and matches the
  interface, but it means an empty config returns `max_model_len == 0` rather than a default —
  correct, since there is no context to guess.

## User Setup Required

None. vLLM stayed down for the entire plan, as intended: nothing here starts a container,
touches a GPU, or opens a socket.

## Next Phase Readiness

- Phase 3 owns recipe precedence. The Qwen3.6 0.42-vs-0.55 gap is now pinned by a test with a
  docstring explaining why it must be closed by a curated recipe winning over the derived
  value, never by editing the formula.
- Phase 4 wiring has everything it needs: `_derive_launch_spec` is the single call site, and
  `resident_gb` is the parameter the caller should fill from the box's current memory reading.
  The `0.75` / `32768` / `65536` literals and `_build_vllm_profile_script` are untouched.
- `warnings` now reaches the public entry point but still has no consumer. Phase 4 should
  surface it in the UI or the launch log — an unrecognized layer type over-reserving memory is
  a decision the user should be able to see.
- Phase 5's admission math should reuse `_kv_budget_gb` rather than re-deriving the pool
  arithmetic; the `resident_gb` term is the part most likely to be dropped by accident.

## Self-Check: PASSED

- `app.py` — FOUND (contains `def _kv_budget_gb` and `def _derive_launch_spec`)
- `tests/test_launch_spec.py` — FOUND (contains `def test_calibration_`)
- Commits `d072e78`, `b5e3664`, `ce57024` — all FOUND in `git log`
- `python3 -m pytest tests/ -q` → 310 passed, 0 skipped
- `python3 -m pytest tests/test_launch_spec.py -q` → 181 passed, 0 skipped

---
*Phase: 02-derived-launch-spec*
*Completed: 2026-08-05*
