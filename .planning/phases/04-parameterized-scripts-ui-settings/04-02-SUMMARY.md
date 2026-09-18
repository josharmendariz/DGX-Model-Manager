---
phase: 04-parameterized-scripts-ui-settings
plan: 02
subsystem: vllm-launch
tags: [script-header, profile-card, warnings, ui]
requires: [VLLM_env_placeholders, _OVERRIDE_ENV, _resolve_overrides]
provides: [_script_meta_headers, _launch_defaults, _collect_overrides, renderProfileSettings, profile_meta_derived_keys]
affects: [app.py, profiles/vLLM/*]
tech-stack:
  added: []
  patterns: [versioned-header-as-data-format, one-source-of-truth-two-renderings, defect-not-fallback]
key-files:
  created: [tests/test_ui_render.py]
  modified: [app.py, tests/test_vllm_profile_generation.py, tests/test_profile_meta.py, tests/test_launch_spec.py]
key-decisions: []
requirements-completed: [REQ-06]
duration: ~40 min
completed: 2026-08-20
---

# Phase 4 Plan 02: Derived/recommended surfacing + profile-card settings Summary

Generated vLLM scripts now carry `# Derived:` / `# Recommended:` / `# Warnings:` /
`# Generated:` as single-line JSON, `_parse_script_meta` reads them back with no
filesystem access, and the profile card renders editable context / utilization /
max-num-seqs controls whose placeholders are the derived values — giving the `warnings`
list its first UI reader since Phase 2.

## What Was Built

**Task 1 — header persistence** (`b5507bf`)
- `_script_meta_headers(info, resolved)` emits four header comments inside the first 20
  lines (the window `_parse_script_meta` scans). Each value is compact JSON; `# Derived:`
  carries `"v": 1` because the header is now a versioned data format.
- `_launch_defaults(resolved)` is the single source for the three `${VAR:-N}` placeholder
  defaults. The docker argv and the `# Derived:` object both read it, so the header cannot
  advertise a context the script would not use — asserted directly by a test.
- `_resolve_launch` now threads the full `_derive_launch_spec` result through under a
  `spec` key instead of dropping fourteen of them, which is what let the header carry
  `max_fitting_context` / `declared_max_context` / `recommended_util` without re-deriving.
- Recipe-backed scripts emit `# Derived: {"v":1,"editable":false,"reason":...}`.
- Warning strings are flattened with `_one_line` *and* json-escaped: they embed
  `{value!r}` of vendor config fields, and a raw newline would split the header line.

**Task 2 — reading it back** (`e175043`)
- `_parse_script_meta` gains `derived`, `recommended`, `warnings`, `generated_at` and
  `meta_error`. Missing headers → `None`/`[]` (the legacy case). A header that is present
  but unparseable sets `meta_error` instead of being swallowed by the pre-existing broad
  `except Exception: pass` — the locked decision is that a corrupt header is a defect.
- No filesystem or config reads added. A source-inspection test asserts `read_text()`
  appears exactly once and that `config.json`, `glob(`, `open(`, `_derive_launch_spec`
  and `_profile_source_dir` appear nowhere in the function body (T-04-08).

**Task 3 — the card** (`793cbcc`)
- `renderProfileSettings(p)` implements three of the five card states: **header-error**
  (checked first, before any classification, so a corrupt header cannot fall through),
  **recipe-backed** (three disabled inputs plus the reason), **editable**. `!p.derived`
  returns empty — read-only and unparseable belong to 04-03.
- Editable state: one `<input data-override=...>` per field, `placeholder` from
  `derived`, and a `recommended N` label inside the utilization/context `<label>`.
- Warning and meta-error text is written with `textContent` after `innerHTML`, never
  interpolated (T-04-07). `renderProfileSettings` never mentions `p.warnings`.
- `POST /api/vllm/reclaim-cache` is offered next to the utilization control with the
  page-cache note (Pitfall 4).
- `_collect_overrides(raw)` is the pure request-body builder: blank / whitespace-only /
  `None` → key absent; anything else passes through trimmed to `_resolve_overrides`.
  `collectProfileOverrides()` in the browser applies the same rule, and
  `_resolve_overrides` now calls the Python half so a hand-rolled client sending `""`
  gets the derived default rather than a 400.
- Inputs clear in `selectEngineProfile`, so overrides stay per-launch and per-profile.

## Deviations from Plan

**1. [Rule 3 — blocker] The docker preamble had to move to the end of `_build_vllm_profile_script`**
- Found during: Task 1.
- Issue: the preamble was built immediately after `_resolve_launch`, but capability
  warnings (`_capability_emission_details`) are appended to `info["warnings"]` ~80 lines
  later. Emitting `# Warnings:` at the old site would have shipped a truncated list that
  looked complete.
- Fix: the recipe branch keeps its early preamble (it returns immediately); the docker
  branch builds its preamble just before script assembly. Commented at both sites.
- Files: `app.py`. Commit `b5507bf`.

**2. [Rule 2 — missing critical] `_collect_overrides` applied server-side, not only in the browser**
- Found during: Task 3.
- Issue: the plan's rule is "blank input sends nothing". Implemented only in JS, an empty
  string from any other client would reach `_override_int`/`_override_float` and 400 —
  turning "I left the box alone" into a launch failure.
- Fix: `_resolve_overrides` calls `_collect_overrides` first. The browser applies the same
  rule so the request body genuinely omits the key; the server half is defence in depth.
- Files: `app.py`, `tests/test_ui_render.py`. Commit `793cbcc`.

**3. [Rule 2 — missing critical] the clamping carry-over is surfaced on the card**
- Found during: Task 3, from 04-01's handoff note.
- Issue: `_derive_launch_spec` *clamps* an over-requested context — it warns and discards
  rather than applying. A bare context input would silently give less than typed.
- Fix: the editable panel names both ceilings (`max_fitting_context`,
  `declared_max_context`) and states that the request is discarded, not applied. Asserted
  by `test_clamping_warning_is_surfaced_next_to_the_context_control`.
- Files: `app.py`, `tests/test_ui_render.py`. Commit `793cbcc`.

**4. [scope note] `tests/test_ui_render.py` is a new file**
- The plan's Task 3 `<files>` names it but the frontmatter `files_modified` does not. It
  did not exist; created rather than folded into an existing module.

**Total deviations:** 3 auto-fixed (1 blocker, 2 missing-critical) + 1 scope note.
**Impact:** no scope change. One new public helper (`_collect_overrides`) beyond the plan.

## Verification

| Check | Result |
|---|---|
| `python3 -m pytest tests/ -q` | **404 passed**, 0 failed (04-01 baseline 360 → +44) |
| `pytest tests/test_profile_meta.py -k derived_roundtrip` | 2 passed |
| `pytest tests/test_launch_spec.py -k fixture_bounds` | 17 passed (every committed fixture model) |
| Four header lines parse with `json.loads` | asserted per generated script |
| Four header lines inside the first 20 lines | asserted |
| `# Derived: max_model_len` == emitted `${VLLM_MAX_MODEL_LEN:-N}` | asserted (also util and max_num_seqs) |
| Warning round-trip, no literal newline | asserted with a `we\nird_type` layer fixture |
| `bash -n` over generated scripts (incl. quote-hostile warnings) | clean |
| `node --check` over the new JS (render + builder + map template) | clean |
| `grep -v '^#' app.py \| grep -c 'reclaim-cache'` | 3 (≥2 required) |
| `git status --porcelain profiles/` | empty |
| New imports beyond stdlib | none |

Task 3's automated acceptance criteria — the primary evidence for criterion 3 — all pass
with no GPU, no docker and no service restart.

## Known Stubs

`renderProfileSettings` returns an empty string when `p.derived` is `null`. That is the
**legacy / unparseable** case, which 04-03 owns by design (the plan assigns read-only and
unparseable to that plan). It is not a silent fallback: `meta_error` is branched on first
and always renders a visible banner.

## Threat Flags

None. T-04-06 (header values are display-only; the launch still resolves through
`_OVERRIDE_ENV`), T-04-07 (`textContent`) and T-04-08 (no filesystem reads in
`_parse_script_meta`, asserted) are each mitigated and covered by tests. T-04-09 is
`accept` with `# Generated:` making staleness visible.

## Issues Encountered

**The human-verify checkpoint (Task 3 steps 1-6) has NOT been performed.** Executing them
requires `systemctl --user restart dgx-model-manager.service` and a real container launch,
both explicitly out of bounds for this run (code + tests only, no live vLLM). Outstanding:
- Step 2/3: visual confirmation of the three controls and the recipe-backed reason.
- Step 4: `docker inspect vllm_node` showing `--gpu-memory-utilization 0.60` after typing
  0.60 and clicking Start. This is the only end-to-end proof that the browser's
  `overrides` object reaches docker argv; every layer of it is unit-tested, but the
  browser→HTTP hop is asserted only as source text.
- Step 6: clearing the input and confirming the derived default returns.

Step 5 (`git status --porcelain profiles/` empty) is already verified above.

## Next

Ready for 04-03 (esc() helper, read-only + unparseable card states, parameterize action).

## Self-Check: PASSED
- `tests/test_ui_render.py` exists on disk.
- Commits `b5507bf`, `e175043`, `793cbcc` all present in `git log`.
- All task acceptance criteria re-run and passing; full suite 404 passed.
- Human-verify steps 1-6 deferred (see Issues Encountered) — not a code failure.
