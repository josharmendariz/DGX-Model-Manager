# Phase 4: Parameterized scripts + UI settings - Context

**Gathered:** 2026-08-20
**Status:** Ready for planning
**Source:** Answers to the four open questions in 04-RESEARCH.md (Josh, 2026-08-20)

<domain>
## Phase Boundary

Launch settings become adjustable per-launch from the UI without rewriting or clobbering
the script on disk. Covers generated vLLM profiles and hand-written legacy scripts.

Explicitly OUT of scope: `shape="recipe"` scripts, which delegate to `run-recipe.sh` in
another repo and cannot be parameterized from here (04-RESEARCH.md §Criterion 1). They
render as a fourth card state, not as an editable form.

</domain>

<decisions>
## Implementation Decisions

### Override lifetime
- **Per-launch only.** Overrides are transient. Nothing is persisted to disk, and no new
  state file is introduced. Criterion 2 already forbids mutating the script at launch, so
  transience is the consistent choice. Persistence is deferred (see Deferred Ideas).

### Override transport — premise CONFIRMED, criterion unchanged
- The roadmap's criterion 2 stands as written. `systemd-run --user --scope` **does**
  inherit `Popen(env=)`; verified on this box (`FOO=hello systemd-run --user --scope` →
  `scope_FOO=[hello]`), and corroborated in production by `env["RECIPE"]` at `app.py:2971`
  already reaching the llama.cpp engine through the same wrapper.
- There is **one** transport mechanism, not two. The `docker run -d` path is unaffected:
  `${VAR:-derived}` expands host-side into docker **argv**, and `--max-model-len` and
  friends are CLI flags, not env vLLM reads. No `-e` plumbing needed.
- Overrides MUST resolve **before** `_vram_admission_check` (`app.py:2961`) — same ordering
  trap as the llama.cpp recipe.

### Generator drift (blocking side-finding)
- **Fix in 04-01.** `079917b` edited the 12 on-disk profiles but not the generator;
  `app.py:4492` still emits `exec docker run` with no `-d`, so regenerating any profile
  silently reverts half the cgroup fix. Verified in this tree. The 04-01 generator block is
  open anyway.

### Derived / recommended surfacing
- **Option A — header comments.** Generated scripts carry derived and recommended values as
  header comments, read back via `_parse_script_meta`. Accepted tradeoff: the script header
  becomes a versioned data format and must stay parseable. Plan accordingly — a header the
  parser cannot read is a defect, not a fallback.
- The so-far-unconsumed `warnings` field must reach the card. Today it has exactly one
  reader (`app.py:4521`), never on the list path.

### XSS scope
- **`app.py:8655` (hf.co repo names in the browse card) is IN scope, fixed in 04-03.** It is
  the highest-severity enumerated site and the same one-line fix as the profile-card sites;
  the `esc()` helper lands in 04-03 regardless. There are currently zero escaping helpers in
  the codebase — `q()` (`app.py:7970`) escapes single quotes for JS-string context only and
  is not a substitute.

### Legacy scripts — under-scoped in the roadmap
- Criterion 4 says "render their parsed values read-only", but **no flag parser exists**.
  `_parse_script_meta` reads three header comments and nothing else. This phase must build a
  real regex flag parser with an explicit `unparseable` card state.
- The parameterize action needs a sha256 confirm-token between preview and apply, or the
  diff preview is a TOCTOU.

### Claude's Discretion
- Plan/task decomposition within the 3-plan split, wave assignment, exact regex shapes,
  helper naming, and test file organization.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase research
- `.planning/phases/04-parameterized-scripts-ui-settings/04-RESEARCH.md` — 627 lines;
  criterion-by-criterion analysis with `app.py` line numbers, the Validation Architecture
  section, Don't Hand-Roll, and Common Pitfalls. Authoritative for this phase.

### Situation
- `.planning/HANDOFF.json` — live engine/litellm state, what landed 2026-08-18..20, and the
  three still-open items (Qwen3.8 dense parser gap, FastMTP fork image, model provenance).

### Test conventions
- `tests/conftest.py` — `MODEL_FIXTURES_PATH` now resolves to `tests/fixtures/model-fixtures.json`.
  Ground truth is that committed snapshot; tests must never scan model-cache directories.

</canonical_refs>

<specifics>
## Specific Ideas

- Placeholder style is `${VLLM_MAX_MODEL_LEN:-<derived>}`.
- Emission sites: `app.py:4440` (utilization), `4451` (max-model-len), `4446/4451`
  (max-num-seqs — welded to the same f-string line, so splitting that line is a
  prerequisite, not a nicety).
- Three script shapes today; four card states once `unparseable` is added.
- No new dependencies: `difflib`, `hashlib`, `re` are stdlib.

</specifics>

<deferred>
## Deferred Ideas

- Persisting last-used overrides per profile — needs a state file and a stale-override
  story when the derived value changes underneath it.
- Parameterizing `shape="recipe"` scripts — blocked on the other repo owning `run-recipe.sh`.

</deferred>

---

*Phase: 04-parameterized-scripts-ui-settings*
*Context gathered: 2026-08-20 from 04-RESEARCH.md open questions*
