---
phase: 02-derived-launch-spec
plan: 01
subsystem: infra
tags: [vllm, kv-cache, attention-topology, hybrid-models, pytest, pure-functions]

# Dependency graph
requires:
  - phase: 01-launch-integrity
    provides: "the documented-rationale docstring standard (_vllm_serve_command) and the pure-function test idiom (tests/test_launch_preflight.py)"
provides:
  - "app._resolve_attention_topology(config) -> full / bounded / stateless layer classification via a five-branch precedence chain"
  - "app._kv_bytes_per_token(config, kv_dtype_bytes) -> per-layer, per-token and bounded-total KV byte sizing"
  - "tests/test_launch_spec.py — the module the whole phase verifies through (V1-V6 green, V7-V9 stubbed for 02-02)"
  - "tests/conftest.py model_fixtures / fixture_models loaders for the committed 17-model snapshot"
affects: [02-02-derived-spec-and-utilization, 03-recipes-and-capabilities, 04-script-wiring, 05-admission-budget]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Committed JSON fixture snapshot instead of a live scan of the model cache"
    - "Three-class KV accounting (full / bounded / stateless) replacing the prototype's two buckets"
    - "Unknown-input-is-loud: unrecognized layer types over-reserve and warn rather than scoring zero"

key-files:
  created:
    - tests/test_launch_spec.py
  modified:
    - app.py
    - tests/conftest.py

key-decisions:
  - "topology['sliding_window'] reports None when no layer is bounded, so a dense model's vestigial window cannot be multiplied by a caller"
  - "Unknown layer types warn once per distinct type, not once per layer"
  - "num_hidden_layers falls back to the length of layer_types / hybrid_override_pattern when the config omits it, keeping the three classes a strict partition"
  - "full_attention_interval is covered by a synthetic config only — no model on this box selects that branch"

patterns-established:
  - "Pure config-dict-in / dict-out helpers matching the _infer_from_config house idiom, with .get()-and-coerce defensiveness throughout"
  - "Vendor-authored config values coerced through _as_int rather than trusted or asserted"

requirements-completed: [REQ-04]

# Metrics
duration: 30min
completed: 2026-08-05
---

# Phase 2 Plan 01: Attention topology and KV sizing Summary

**Hybrid-aware layer classifier and KV-bytes solver as pure helpers in `app.py`, reproducing all 17 committed model fixtures and correcting the prototype's silent-zero bug for unrecognized layer types.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-08-05T00:19Z
- **Completed:** 2026-08-05T00:49Z
- **Tasks:** 3
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- `_resolve_attention_topology` implements the five-branch precedence chain
  (`layer_types` → `hybrid_override_pattern` → `full_attention_interval` →
  `sliding_window`+`use_sliding_window` → dense), `text_config`-first, and classifies layers
  into **three** KV classes rather than the prototype's two. All 17 fixtures reproduce their
  recorded `full` / `bounded` / `stateless` counts *and* their `precedence_field`.
- The prototype's accidental-correctness bug is closed: an unrecognized `layer_types` entry or
  `hybrid_override_pattern` character is counted as **full** attention and reported in
  `warnings` (T-02-01). Over-reserving is recoverable; under-reserving OOMs at model load.
- The `use_sliding_window` trap has a dedicated regression over the 5 real configs that spring
  it (T-02-02) — without the guard, a 64-layer 17.2 GB dense KV estimate collapses to near zero.
- `_kv_bytes_per_token` separates the context-scaling rate (`full_bytes_per_token`) from the
  already-total windowed cost (`bounded_bytes_total`); the three hand-computed byte anchors
  (gpt-oss 18432 / 2359296, qwen3-next 12288, R1-32B 131072) match exactly.
- Suite went 130 → 237 passing (+107), still 0.48s. Zero regressions.

## Task Commits

1. **Task 1 (Wave 0): Fixture loader + V1-V9 test skeleton** — `026723c` (test)
2. **Task 2: `_resolve_attention_topology`** — `1719e4b` (test, RED) → `39bf277` (feat, GREEN)
3. **Task 3: `_kv_bytes_per_token` + 17-model table** — `744aaa8` (test, RED) → `0894b3c` (feat, GREEN)

No REFACTOR commits — neither helper needed cleanup after going green.

## TDD Gate Compliance

Both TDD tasks show the required gate sequence in `git log`: a `test(...)` commit with the
tests failing, then a `feat(...)` commit turning them green. RED was verified by running the
suite between commits (23 failures for Task 2, 28 for Task 3), not assumed.

## Files Created/Modified

- `app.py` — `_as_int`, `_KV_STATELESS_LAYER_TYPES`, `_PATTERN_*` constants,
  `_resolve_attention_topology` (:754) and `_kv_bytes_per_token` (:866), inserted between
  `_infer_from_config` (:677) and `_parse_hf_model_dir` (:925). Single diff hunk region;
  nothing else in the file touched.
- `tests/test_launch_spec.py` — 107 tests across all nine VALIDATION selectors, plus the
  `config_from_fixture` rebuilder.
- `tests/conftest.py` — session-scoped `model_fixtures` / `fixture_models` reading the
  committed snapshot; `isolate_alert_state` untouched.

## Decisions Made

- **`sliding_window` is reported only when a layer is actually bounded.** Five models carry a
  vestigial `sliding_window` they do not use; echoing it back would let a downstream caller in
  02-02 multiply by a window that applies to nothing. `bounded_kv_layers == 0` ⇒ `None`.
- **Warnings are deduplicated per distinct unknown type.** A 40-layer model of an unknown type
  should produce one actionable line, not forty.
- **`num_hidden_layers` falls back to the length of the topology field** when the config omits
  it, so `full + bounded + stateless` always partitions the reported total.
- **The self-fulfilling-fixture risk is handled by `source_field`, not by the counts.** No
  snapshot row carries a verbatim `layer_types` list, so the rebuilt config could in principle
  agree with itself; `precedence_field` is the independent signal, because the rebuild cannot
  fake which branch the real config selected. This is documented in `config_from_fixture`'s
  docstring for whoever regenerates the snapshot.

## Deviations from Plan

### Acceptance criterion not met literally (documented, not fixed)

**1. [Rule 4-adjacent — reported, not silently changed] `_resolve_attention_topology` lands at app.py:754, outside the plan's stated 690-740 window**

- **Found during:** Task 2 verification
- **Issue:** The criterion `grep -n "def _resolve_attention_topology" app.py` reports a line
  number between 690 and 740 assumed the helper would follow `_infer_from_config` (which ends
  at :732) with no preamble. The implementation needs ~20 lines of module-level preamble first:
  the stateless-layer-type and pattern-alphabet constants, and the `_as_int` coercion helper.
- **Resolution:** Left as-is. The criterion's *intent* — "inserted between `_infer_from_config`
  and `_parse_hf_model_dir`" — holds exactly (`677 < 754 < 866 < 925`), and the plan itself
  notes that raw line numbers in `app.py` drift because the file carries concurrent work. The
  alternatives were worse: defining constants after their user, or rebuilding two frozensets on
  every call, both to satisfy a number the planner estimated rather than measured.
- **Verification:** `git diff -U0 app.py | grep '^@@'` shows one hunk region, at :735 and :866;
  `_build_vllm_profile_script` is at :3429, far outside it.

### Wording adjustments to satisfy the purity grep

**2. [Rule 3 - Blocking] Two docstrings reworded to pass the "no filesystem access" grep**

- **Found during:** Task 1 verification
- **Issue:** The acceptance check
  `grep -v '^#' tests/test_launch_spec.py | grep -cE "glob|expanduser|huggingface|/mnt/models"`
  must print `0`, and it matched two *prose* mentions in docstrings that were describing what
  the tests must not do.
- **Fix:** Reworded to "scan the model cache directories" and "no filesystem, network or vLLM
  dependency", preserving the meaning.
- **Verification:** The grep now prints `0`.
- **Committed in:** `026723c`

---

**Total deviations:** 1 documented non-fix (cosmetic line-number criterion), 1 auto-fixed (blocking grep).
**Impact on plan:** No scope creep, no behavioral change. Every functional acceptance criterion
in all three tasks passes verbatim.

## Known Stubs

Three tests are intentionally `pytest.mark.skip(reason="plan 02-02")`, as the plan directs —
they belong to plan 02-02, not to this one:

| Test | File:line intent | Reason |
|------|------------------|--------|
| `test_calibration_qwen3_next_matches_hand_measured_util` | tests/test_launch_spec.py | V7 needs the utilization formula, which 02-02 builds |
| `test_purity_no_filesystem_or_network_in_helper_bodies` | tests/test_launch_spec.py | V8 source-level purity scan, scoped to 02-02 |
| `test_degenerate_configs_return_defined_values` | tests/test_launch_spec.py | V9, scoped to 02-02 |

The *behaviors* V8 and V9 assert are already true of this plan's helpers and are covered
incidentally (`_kv_bytes_per_token({})` returns zeros without raising; neither body touches the
filesystem) — 02-02 adds the explicit, source-scanning versions.

No hardcoded empty values flow to any UI: this plan adds no UI, no route and no data path.
Both helpers are currently unreferenced by application code *by design* — the plan's scope
fence puts wiring in Phase 4.

## Issues Encountered

- The fixture snapshot records `use_sliding_window` as JSON `null` for several models where the
  real config simply omits the key. `config_from_fixture` omits the field entirely rather than
  planting `None`, because conflating "absent" with "false" would have made the V4 guard test
  pass for the wrong reason.
- `pytest.mark.parametrize` cannot consume a session fixture, so the 17-row table is
  parameterized from a module-level load of the same committed file the conftest fixtures read
  (imported via `from conftest import MODEL_FIXTURES_PATH`, keeping one source of truth for the
  path).

## User Setup Required

None — no external service configuration required. vLLM stayed down for the entire plan, as
intended: nothing here starts a container, touches a GPU, or opens a socket.

## Next Phase Readiness

- The `<interfaces>` contract 02-02 consumes is implemented exactly as specified: both return
  dicts carry only the documented keys, verified by dedicated tests.
- 02-02 should note the rate/total asymmetry: multiply **only** `full_bytes_per_token` by
  context; `bounded_bytes_total` is already a total.
- The `0.75` / `32768` / `65536` literals and `_build_vllm_profile_script` are untouched, so
  Phase 4's wiring surface is exactly where it was.
- Not a blocker, but worth carrying: `_resolve_attention_topology` returns `warnings` that
  nothing currently surfaces. When 02-02 or Phase 4 wires the derived spec in, those warnings
  should reach the UI or the log — a silently-collected warning is only half the T-02-01
  mitigation.

## Self-Check: PASSED

- `tests/test_launch_spec.py` — FOUND
- `tests/conftest.py` — FOUND (contains `02-MODEL-FIXTURES.json`)
- `app.py` — FOUND (contains `def _resolve_attention_topology` and `def _kv_bytes_per_token`)
- Commits `026723c`, `1719e4b`, `39bf277`, `744aaa8`, `0894b3c` — all FOUND in `git log`
- `python3 -m pytest tests/ -q` → 237 passed, 3 skipped

---
*Phase: 02-derived-launch-spec*
*Completed: 2026-08-05*
