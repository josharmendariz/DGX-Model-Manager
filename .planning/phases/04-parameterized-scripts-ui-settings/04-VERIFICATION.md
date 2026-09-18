---
phase: 04-parameterized-scripts-ui-settings
verified: 2026-08-21T00:00:00Z
status: gaps_found
score: 4/5 must-haves verified (1 blocked by a live test-suite regression)
overrides_applied: 0
gaps:
  - truth: "python3 -m pytest tests -q is green at HEAD (required by every plan's <verification> block)"
    status: failed
    reason: "43fee2e (adopting the two live-directory rewrites into git) reclassifies start_hf_qwen_qwen3-8b.sh from legacy to parameterized (5652b57's classifier fix no longer requires the generated-by marker). tests/test_profile_meta.py's _EXPECTED_CLASSES oracle was never updated to match, so the committed working tree fails its own regression test."
    artifacts:
      - path: tests/test_profile_meta.py
        issue: "_EXPECTED_CLASSES['start_hf_qwen_qwen3-8b.sh'] = 'legacy', but the committed blob (post-43fee2e) now classifies 'parameterized'"
    missing:
      - "Update _EXPECTED_CLASSES for start_hf_qwen_qwen3-8b.sh (and verify start_hf_qwen3-vl-4b-fp8.sh, also rewritten by 43fee2e, still matches its 'generated'/'parameterized' allowance) so the suite is green at HEAD."
---

# Phase 4: Parameterized scripts + UI settings — Verification Report

**Phase Goal:** Launch settings are adjustable per-launch from the UI without rewriting or
clobbering the script on disk.
**Verified:** 2026-08-21
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Generated scripts use `${VLLM_MAX_MODEL_LEN:-<derived>}`-style placeholders | ✓ VERIFIED | `app.py:4991` emits `docker run -d --name vllm_node` (not `exec`); placeholders present in generator; `grep -c 'exec docker run'` = 0 confirmed live |
| 2 | `_engine_start` passes overrides as env; script never mutated at launch | ✓ VERIFIED | `_resolve_overrides` (app.py:3084) allow-lists by name, raises 400 on unknown key/out-of-range value; `test_script_not_mutated_by_a_launch_with_overrides` and env-transport tests pass in the current 464-test run |
| 3 | Profile card shows editable context/util/max-num-seqs, derived placeholder, recommended label | ✓ VERIFIED (code) / ? UNCERTAIN (visual) | `renderProfileSettings` editable branch confirmed in app.py; visual placeholder/label rendering and the `docker inspect` proof of util=0.60 reaching the container were established only by the live checkpoint recorded in 04-03-SUMMARY (Issues Encountered / 04-02 Task 3 — CLOSED), not by an automated DOM test |
| 4 | Legacy scripts render parsed values read-only, opt-in parameterize shows a diff before writing | ✓ VERIFIED | `_parse_script_flags`, `_classify_script`, preview/apply endpoints with `difflib.unified_diff`, sha256 confirm-token (`hmac.compare_digest`), 409-on-mismatch, `.bak`-then-`os.replace` all present in app.py and covered by passing tests; the 409-under-concurrent-edit and "diff appears before any write" behavior were additionally proven live per the 04-03-SUMMARY checkpoint |
| 5 | Untrusted strings reach the DOM via `textContent`, not raw `innerHTML` | ✓ VERIFIED (documented deviation) | `const esc` (app.py:7510) is the single helper, applied at every enumerated sink incl. hf.co card and concatenation sinks (8652/8657); 04-03-PLAN.md explicitly documents keeping `innerHTML`+`esc()` instead of a `textContent` rewrite as a locked, rationale-backed deviation — scored on "cannot become markup," per the plan's own verifier guidance |

**Score:** 5/5 criteria have supporting code; **but the automated regression gate all five criteria rely on ("pytest tests -q green") currently fails at HEAD.**

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app.py` `_OVERRIDE_ENV`/`_resolve_overrides` | allow-list + validators | ✓ VERIFIED | app.py:3084-3104 |
| `app.py` `_vllm_script_preamble`/header emission | `# Derived:`/`# Recommended:`/`# Warnings:` | ✓ VERIFIED | present, JSON-parseable |
| `app.py` `_parse_script_flags`/`_classify_script` | regex parser, 4-state classifier | ✓ VERIFIED | present; classifier no longer requires marker (5652b57) |
| `app.py` parameterize preview/apply endpoints | diff + sha256 confirm-token | ✓ VERIFIED | `unified_diff`, `hmac.compare_digest`, `.bak`, `os.replace` all present |
| `app.py` `const esc` | one helper, all sinks | ✓ VERIFIED | `grep -c "const esc"` = 1 |
| `app.py` `regen_path`/Regenerate metadata button | c78bfa9 | ✓ VERIFIED | app.py:544, 8795-8809 |
| `app.py` `script_defaults`/`_parse_templated_defaults` | e794918 | ✓ VERIFIED | app.py:544-547, 3166+, 8809 |
| `tests/test_profile_meta.py::test_classify_covers_every_committed_profile` | passes at HEAD | ✗ FAILED | oracle not updated after 43fee2e adopted the rewrite of `start_hf_qwen_qwen3-8b.sh` into git |

### Test Suite

`python3 -m pytest -q` at HEAD (43fee2e): **463 passed, 1 failed** (464 total).

The failure is `tests/test_profile_meta.py::test_classify_covers_every_committed_profile`:
`start_hf_qwen_qwen3-8b.sh: parameterized not in {'legacy'}`. This is a direct consequence
of the last commit in the phase (43fee2e, "adopt the two live-directory rewrites") — it
changed the committed script's classification but nobody updated the test oracle that pins
per-file expected classes. 04-03-SUMMARY's self-check ("454 passed" / later "464" after
e794918) predates this commit and is therefore stale; it does not describe the current
HEAD.

### Human Verification Required

These roadmap-criterion-supporting behaviors were established only by the live checkpoint
documented in 04-03-SUMMARY.md (2026-08-21, run with Josh), not by any automated test in
this repo. Re-running them is optional but they are not independently re-verifiable from
the codebase alone:

1. **Override reaches the running container** — `docker inspect vllm_node` showing
   `--gpu-memory-utilization 0.3` / `--max-model-len 8192` after typing those values and
   clicking Start (04-03-SUMMARY, "04-02 Task 3 — CLOSED").
2. **Diff-before-write and stale-hash 409** — confirmed live against the real profile
   directory, including a concurrent-edit 409 (04-03-SUMMARY, Task 4 steps 3-4).
3. **HF-browse XSS payload renders as literal text, not markup** — confirmed live in a
   browser (04-03-SUMMARY, Task 4 step 6); the source guard test is automated, but the
   rendered-DOM behavior itself was eyeballed.

## Gaps Summary

Four of the five roadmap criteria are fully supported by code and passing tests. The fifth
(criterion 5) is a documented, rationale-backed deviation rather than a gap. The blocking
issue is process, not architecture: the final commit on this branch (43fee2e) adopted a
real, value-preserving script rewrite into git but didn't update the test that asserts the
classification of every committed profile script, so `pytest -q` — the gate named in all
three plans' `<verification>` blocks — is red at HEAD. This is a one-line fix (update
`_EXPECTED_CLASSES['start_hf_qwen_qwen3-8b.sh']` to `'parameterized'`, and confirm
`start_hf_qwen3-vl-4b-fp8.sh` still matches), but it must land before the phase is closed —
"tests pass" is a load-bearing claim in every plan and the SUMMARY's final self-check count
no longer describes HEAD.

---

_Verified: 2026-08-21_
_Verifier: Claude (gsd-verifier)_
