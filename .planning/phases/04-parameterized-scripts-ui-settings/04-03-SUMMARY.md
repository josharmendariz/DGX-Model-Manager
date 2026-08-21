---
phase: 04-parameterized-scripts-ui-settings
plan: 03
subsystem: vllm-launch
tags: [flag-parser, classifier, parameterize, xss, escaping, toctou]
requires: [_OVERRIDE_ENV, _launch_defaults, _parse_script_meta, renderProfileSettings]
provides: [_parse_script_flags, _classify_script, _parameterize_script_text, parameterize_preview_apply, esc]
affects: [app.py, profiles/vLLM/*]
tech-stack:
  added: []
  patterns: [honest-unparseable-state, refuse-whole-not-partial, confirm-token-against-toctou, escape-at-the-sink-exactly-once]
key-files:
  created: [tests/test_ui_escaping.py]
  modified: [app.py, tests/test_profile_meta.py, tests/test_ui_render.py]
key-decisions: []
requirements-completed: [REQ-06]
duration: ~50 min
completed: 2026-08-21
---

# Phase 4 Plan 03: Legacy flag parser, parameterize action, escaping pass Summary

Hand-written scripts now get a literal-only regex flag parser with an explicit
`unparseable` state, an opt-in parameterize action gated by a server-built unified diff
and a sha256 confirm-token, and every enumerated untrusted string reaches the DOM through
one `esc()` helper — including the hf.co repo names, the reflected search term and
`e.message`.

## What Was Built

**Task 1 — parser + classifier** (`f30337e`)
- `_parse_script_flags(text)` — one anchored regex over non-comment lines, matching
  `--max-model-len` / `--gpu-memory-utilization` / `--max-num-seqs` in both the
  `--flag N` and `--flag=N` forms. The value group is captured loosely on purpose and
  validated afterwards, so a non-literal token is *seen* and reported `unparseable`
  rather than skipped — skipping would be indistinguishable from "flag absent".
- Five things produce `unparseable`: absent, `$VAR`, `${VAR:-N}`, `$(...)`, and the same
  flag twice with different values. Twice with identical values resolves to the number.
  The llama.cpp `ARGS=( --flag "$CTX" )` shape therefore fails safe by construction.
- No bash grammar, no `bashlex`, no new import — `re` only, asserted by a test.
- `_classify_script(text)` returns `parameterized` / `generated` / `recipe` / `legacy`.
  Order is load-bearing: placeholder-and-marker first, then the `run-recipe.sh` test
  (reusing the existing `_PF_RECIPE_RE`), then marker-only, then legacy.
- Both are pure text functions. `_parse_script_meta` now reads the script text once and
  surfaces `classification` and `flags` — the existing "exactly one `read_text()`"
  guard (T-04-08) still passes.

**Task 2 — parameterize preview/apply** (`e60491d`)
- `_parameterize_script_text(text) -> (text, notes)` rewrites each literal into
  `${VLLM_*:-<the same literal>}`, so the rewrite changes no behaviour: an unset
  environment reproduces the current argv exactly. Flag→variable comes from
  `_OVERRIDE_ENV`, so a script can never grow a placeholder the override API would
  reject as unknown.
- Idempotent: a token that is *already* the exact `${VAR:-N}` placeholder is recognised
  and left alone rather than treated as unparseable — that is what makes applying the
  function to its own output byte-identical.
- Refuses whole (`ParameterizeRefused` → 422) if any target flag is non-literal,
  conflicting, or absent. A partial rewrite of the hand-tuned scripts is irreversible
  and they carry root-cause writeups (T-04-14).
- Preview returns `{current, proposed, diff, changed, notes, sha256}` with the diff built
  server-side via `difflib.unified_diff`. The client never holds or resends script text,
  so the apply path cannot be fed attacker-authored content (T-04-13).
- Apply requires that sha256 back, compared with `hmac.compare_digest`, and 409s naming
  both hashes without writing anything on mismatch (T-04-12). On match: `.bak` sibling
  first, then the existing `tmp.write_text` → `os.chmod(0o755)` → `os.replace` idiom.
  Never `target.write_text`, asserted by source inspection.
- `_resolve_profile_script` pins the id to `^start_[A-Za-z0-9._-]+$` and re-checks
  `_path_under` the profile dir. Both endpoints are behind `Depends(verify_auth)`,
  asserted by walking `app.routes`.

**Task 3 — one `esc()`, applied** (`c3de100`)
- `const esc` sits at the top of the `<script>` block with a five-character table
  (`& < > " '`), null-safe and stringifying. The pre-existing `_escHtml` (which only
  covered three characters) is now a one-line alias, so there is exactly one
  implementation and `grep -c "const esc" app.py` is 1 (T-04-16).
- HF browse (`renderHFBCard`): `m.id` as text, `m.library_name`, each `m.tags` entry,
  `m.task_label`, and a single escaped `domId` feeding both `id=` attributes. The
  download / expand buttons use `esc(q(...))` — `q` closes the JS-string context, `esc`
  closes the attribute context, and neither alone is sufficient (T-04-10).
- The two concatenation sinks: the reflected search term and `e.message` (T-04-10b).
- Profile cards: `p.name`, `p.description`, and `esc(q(...))` on the
  `selectEngineProfile` / `deleteProfileWeights` / `deleteEngineProfile` attributes
  (T-04-11). Ollama and LiteLLM model-card names too.
- `renderProfileSettings`: `d.reason`, both `recommended` values, the two ceiling numbers
  and all three placeholders. Warning and meta-error text still goes through
  `textContent` (T-04-07, T-04-15) — the 04-02 carry-over guard is re-asserted in the
  new test file.
- **Read-only and unparseable card states** (`renderLegacyProfileSettings`) landed at the
  `if (!d) return ''` insertion point the carry-over notes named. The `meta_error` branch
  is untouched and still first. Read-only shows the parsed numbers in disabled inputs
  plus a "Parameterize…" button; unparseable shows the literal word and *withholds* the
  button, because the server would refuse anyway. `parameterizeProfile()` previews first,
  shows the diff and notes, then applies with the returned hash.

## Deviations from Plan

**1. [Rule 1 — stale acceptance criterion] `grep -c 'unified_diff' app.py` is 2, not 1**
- **Found during:** Task 2.
- **Issue:** the plan asserts the count is exactly 1. `difflib.unified_diff` was already
  used once (app.py, the config-diff path) before this plan started, so the criterion was
  false the moment it was written.
- **Fix:** no test encoding the false claim was added. `os.replace` is now 2 as specified.
  The intent — one server-side diff builder, reused, not a JS reimplementation — holds.

**2. [Rule 2 — missing critical functionality] read-only / unparseable card states**
- **Found during:** Task 3.
- **Issue:** must_haves truth #1 ("parsed and shown read-only") had no owning task action;
  Task 1 stops at the data and Task 3 nominally only escapes. `if (!d) return ''` would
  have shipped a legacy profile with a blank settings panel.
- **Fix:** `renderLegacyProfileSettings` + `parameterizeProfile` added in the Task 3
  commit, at the exact insertion point the carry-over notes reserved.

**3. [deliberate test change] two 04-02 assertions updated**
- `test_each_control_takes_its_placeholder_from_the_derived_object` and
  `test_recommended_util_label_sits_next_to_the_utilization_input` asserted the bare
  `${d.max_model_len}` / `${rec.gpu_memory_utilization}` forms. They now assert the
  escaped form, so a bare interpolation is itself a regression. This is the "change the
  test deliberately" branch the carry-over notes anticipated.

**4. [Rule 1 — flaky oracle] the classifier oracle reads `git show HEAD:`, not the worktree**
- **Found during:** Task 1. `profiles/vLLM/start_hf_qwen3-vl-4b-fp8.sh` changed under the
  test mid-run (a concurrent session, or the running app regenerating it) and flipped from
  `generated` to `parameterized`. A working-tree read makes this oracle flap for reasons
  unrelated to the classifier, so it reads the committed blob and accepts either class for
  marker-bearing files — the distinction being pinned is marker vs recipe vs neither.

## Verification

- `python3 -m pytest tests -q` → **449 passed** (404 at 04-02, +45).
- `bash -n` clean over every script in `profiles/`.
- `grep -c "const esc" app.py` → 1. No new third-party dependency (`re`, `difflib`,
  `hashlib`, `hmac` are stdlib and all four were already imported).
- Both source guards were mutation-tested: reverting `esc()` on `p.name` fails
  `test_no_unescaped_interpolation`, and reverting it on the search term / `e.message`
  fails `test_no_unescaped_concatenation`. The character table runs under `node`
  (present on this host) against the helper text extracted from `app.py` itself, so a
  copy cannot drift, plus an always-run Python reimplementation.

## Threat Flags

None new. T-04-10, T-04-10b, T-04-11, T-04-12, T-04-13, T-04-14, T-04-16 are each
mitigated with a test; T-04-15 is unchanged from 04-02 (`textContent`) and re-guarded.
T-04-SC is vacuous — no packages installed.

## Known Stubs

None.

## Issues Encountered

**Task 4 (the blocking human-verify checkpoint) has NOT been performed.** This run was
constrained to code and tests: no writes under `profiles/vLLM/`, no service restart, no
container action, and `vllm_node` is serving production traffic. Outstanding, in order:
1. `systemctl --user restart dgx-model-manager.service`.
2. Legacy profile card renders parsed values read-only with a Parameterize action.
3. Diff appears before any write; `git status --porcelain profiles/vLLM/` still empty.
4. Concurrent edit → Apply → 409, file unchanged apart from the edit.
5. Apply for real → diff applied, `.bak` exists, `bash -n` clean.
6. HF browse renders a repo name as text, not markup; `# Name:` payload test.

Steps 3, 4 and 5 are unit-tested against a `monkeypatch`ed profile directory; what is
untested is the browser→HTTP hop and a real write to the live directory. **Note that
applying the rewrite to any live profile is a diff-then-adopt decision for Josh, not the
executor's** — the Qwen3.6 recipe wrapper serves production.

Also still outstanding from 04-02: its Task 3 human-verify steps (util 0.60 →
`docker inspect vllm_node`).

## Next

Phase verification (`gsd-verifier`), after Task 4 and the 04-02 checkpoint are run
against the live service.

## Self-Check: PASSED
- `tests/test_ui_escaping.py` exists on disk.
- Commits `f30337e`, `e60491d`, `c3de100` all present in `git log`.
- Full suite re-run at HEAD: 449 passed, above the 404 floor.
- Task 4 deferred by hard constraint (see Issues Encountered) — not a code failure.
