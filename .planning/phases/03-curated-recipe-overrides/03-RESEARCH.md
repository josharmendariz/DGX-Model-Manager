# Phase 3: Curated recipe overrides — Research

**Researched:** 2026-08-05
**Domain:** Launch-script generation; YAML recipe ingestion; config-driven capability tables
**Confidence:** HIGH (nearly everything here was executed against this box, not recalled)

> This document is the **implementation-approach** layer. The factual survey lives in
> `03-CODEX-SURVEY.md` and is not restated. Where this document depends on a survey fact it
> cites the survey section rather than copying the table.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

1. **Recipe source of truth — LOCKED (Josh, 2026-08-05).** The recipe YAMLs at
   `~/spark-vllm-docker/recipes/` are the source of truth. `config.json`'s `vllm.recipes`
   holds only a **model-name pattern → recipe name** mapping, plus `recipe_dir`. Re-typing
   `0.55 / 262144 / qwen3_xml` into `config.json` would fork the source of truth.
2. **Use `yaml.safe_load`.** `pyyaml==6.0.2` is already a declared *runtime* dependency.
   The hand-rolled regex parser at `app.py:3049` was solving a constraint that does not
   exist. Fold it into the new reader; do not grow a second ad-hoc parser.
3. **Do NOT close the Qwen3.6 0.42-vs-0.55 gap by editing the Phase 2 formula — LOCKED.**
   The gap closes because the recipe wins. `tests/test_launch_spec.py::
   test_calibration_qwen36_gap_is_left_for_phase_3` pins this and its docstring forbids a
   formula edit.
4. **`vllm.recipes` is an opt-in allowlist.** Only models Josh explicitly maps become
   recipe-backed. gpt-oss is simply not mapped, so it keeps its known-good generated path
   with full-precision KV and 65536 (roadmap SC3), despite its recipe declaring fp8 KV.
5. **A recipe-backed model gets a DELEGATING WRAPPER**, not a reconstructed `docker run`.
   Reconstruction would silently drop the recipe's in-container `mods/` and the flags the
   generator does not emit. The existing preflight already understands both shapes
   (`_PF_RECIPE_RE`); Phase 3 fits that contract rather than changing it.
6. **Tool-parser selection comes from an explicit capability map**, keyed on something
   durable (`architectures[]` and/or model id), never a display-name substring. It must be
   able to say "this model does not support tool calling". A model with no entry gets **no
   tool flags** rather than a guessed parser.
7. **`--kv-cache-dtype fp8` becomes opt-in** — applied when the recipe's command block
   declares it, or when the model's own config evidences it.
8. **Do not validate parsers against a closed enum.** The host has vLLM 0.21.0 but the
   containers run 0.20.0 / 0.23.1 / cu130-nightly; `nano_v3` and `super_v3` are
   plugin-provided. The map carries values, it does not police them.
9. **Precedence idiom:** config → env → default, the same shape as the existing `alerts`
   block. Recipe wins over derived; derived wins over generic default. A missing key in a
   recipe falls through to derived — **"absent" must not be read as "zero"**.

### Claude's Discretion

- Pattern-matching mechanism for `vllm.recipes` keys (glob vs prefix vs regex) and how
  ambiguity between two matching patterns is resolved — but it must be deterministic and
  tested, since a silent wrong-recipe match is worse than no match.
- Where the capability map physically lives (module constant vs `config.json` vs a tracked
  JSON file beside `recommendations.json`).
- Internal function decomposition and naming.
- Whether the recipe reader returns a normalized dataclass/dict or raw YAML.

### Deferred Ideas (OUT OF SCOPE)

- `${VLLM_MAX_MODEL_LEN:-…}` parameterization and UI context/util controls — Phase 4.
- Consuming `spec["warnings"]` in the UI — Phase 4 (carry-forward #1).
- Executor-budget admission, docker-label reclaim, launch lock — Phase 5.
- The unresolved `VLLM_USE_FLASHINFER_MOE_FP4` disagreement — not Phase 3 scope, but do not
  entrench the env var further while it is disputed.
- Optional per-field inline override on top of a YAML-backed recipe — set aside as more
  precedence surface than this phase needs.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-05 | Hand-measured recipes override derived defaults instead of being clobbered | §1 gives the decomposition that lets a recipe-backed model bypass flag emission entirely; §2 gives the verified recipe reader; §4 gives the per-field precedence chain; §5 proves all five success criteria without a launch. |

`.planning/PROJECT.md:37` carries REQ-05 unchecked. `.planning/ROADMAP.md:106` maps REQ-05
to this phase and nothing else, so there is no requirement-sharing to reconcile.
</phase_requirements>

## Project Constraints (from CLAUDE.md)

| Directive | Consequence for this phase |
|-----------|----------------------------|
| Automation-first; patterns over patches | The capability map and recipe mapping should be **data**, not new branching code. Prefer an existing repo pattern (`_load_recommendations`, `app.py:1649`) over a novel one. |
| Look for existing conventions before adding new code | `yaml.safe_load` is already used at `app.py:373` and `app.py:2160`; `yaml` is already imported at `app.py:31`. The precedence idiom already exists at `app.py:1878-1891`. Reuse all three shapes. |
| Small, reviewable commits; commit by explicit path, never `git add -A` | ~3 concurrent Claude sessions run in this repo. `profiles/vLLM/start_hf_openai_gpt-oss-120b.sh` is **currently modified and uncommitted** by a peer session, and `start_hf_qwen_qwen3-8b.sh` / `start_hf_qwen_qwen3-14b.sh` are **untracked**. Verified via `git status` 2026-08-05. |
| Explain WHY not WHAT; terse | Generated scripts already carry rationale comments; keep that habit for any new emitted comment. |
| Codex for well-specified mechanical subtasks; plan/verify on Claude | The recipe reader and the capability-map table are good Codex candidates; the precedence wiring inside `_build_vllm_profile_script` is not (it is judgment-heavy and touches live-serving behavior). |

---

## Summary

Phase 3 is not really "add a config block". It is **three separable changes that share one
function**, and the whole risk of the phase is concentrated in the order those three changes
compose inside `_build_vllm_profile_script` (`app.py:3581-3702`).

The three changes are: (a) a recipe reader that replaces `_recipe_util` (`app.py:3048`) and
serves both the generator and preflight; (b) a resolution chain that lets recipe values beat
`_derive_launch_spec` values which beat today's literals; and (c) a capability map that
replaces `if "qwen" in info["name"].lower()` (`app.py:3658`). Only (b) touches the generator's
control flow, and it does so at exactly one place: the `arg_lines` list built at
`app.py:3630-3663`.

The single most important finding is a **coupling the roadmap does not name**: SC5 (fp8 KV
only where declared) changes an *input* to Phase 2's solver, not just an output flag.
`_derive_launch_spec(kv_dtype_bytes=…)` defaults to `1` (fp8). If a model stops getting
`--kv-cache-dtype fp8`, its KV cache doubles and the derived numbers must be recomputed at
`kv_dtype_bytes=2`. Verified on the real Qwen3.6 config: util 0.42 at fp8 vs 0.44 at
full precision. Any plan that emits the KV flag *after* deriving the spec is wrong by
construction.

The second most important finding is that **the current test suite will not catch a wiring
mistake**. A naive wiring emits `--max-model-len 0 --gpu-memory-utilization 0.1` for every
existing generation test (verified — the minimal fixture configs at
`tests/test_vllm_profile_generation.py:14` have no `num_hidden_layers`), and all 310 tests
still pass, because no test asserts on a numeric flag value and `bash -n` accepts
`--max-model-len 0`. The Validation Architecture in §5 exists mainly to close that hole.

**Primary recommendation:** decompose `_build_vllm_profile_script` into a shared *preamble*
builder plus two *body* builders (recipe-delegating, flag-emitting), resolve the KV dtype
before deriving the spec, put both the recipe map and the capability map in tracked JSON
using the `_load_recommendations` pattern, and add a `03-RECIPE-FIXTURES/` directory of
byte-copied real recipe YAMLs so the suite proves resolution without depending on
`~/spark-vllm-docker` existing.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Recipe YAML read + normalize | Backend helper (pure-ish, filesystem read) | — | Filesystem is unavoidable; keep it in ONE function so tests can point it at a fixture dir. |
| Pattern → recipe-name lookup | Backend helper (pure) | — | Pure string matching over a config dict. Must be pure so collision determinism is testable. |
| Capability map lookup | Backend helper (pure) | — | Dict lookup keyed on `architectures[]` / model id. |
| Precedence resolution (recipe → derived → default) | Backend helper (pure) | — | The one place the phase's contract lives. Purity is what makes SC1-SC5 provable statically. |
| Script text assembly | `_build_vllm_profile_script` (`app.py:3581`) | — | Already the sole owner of script text; do not add a second emitter. |
| Recipe-backed launch execution | `run-recipe.sh` (external) | `_engine_start` (`app.py:2501`) | Locked decision #5 — the generator delegates, it does not reconstruct. |
| Preflight recipe introspection | `_preflight_smoke` / `_recipe_util` (`app.py:2956`, `:3048`) | — | Already exists; Phase 3 subsumes `_recipe_util` into the new reader rather than forking it. |
| Profile identity / VRAM credit | `_parse_script_meta` (`app.py:417`), `_identify_active_profile` (`:1511`), `_running_profile_vram_credit` (`:2350`) | — | All three read the `#` header block or the served-name string. A recipe wrapper must keep both or these silently degrade. See §6. |

## Package Legitimacy Audit

**This phase installs no new packages.** `pyyaml==6.0.2` is already declared in
`requirements.txt` and already imported at `app.py:31`, with `yaml.safe_load` already in
production use at `app.py:373` and `app.py:2160` [VERIFIED: read from disk]. `fnmatch` (if
chosen for pattern matching in §4) is Python stdlib. No slopcheck run is required because
no package is added; if a plan later proposes one, the gate applies then.

---

## §1 — Integration seam

### 1.1 What `_build_vllm_profile_script` does today

`app.py:3581-3702`, in order:

| Lines | Stage | Recipe path needs it? |
|-------|-------|----------------------|
| 3582-3590 | `_profile_model_info` + format/task gate + `_validate_served_name` | **Yes** |
| 3591-3592 | slug → `script_name` | **Yes** |
| 3593-3594 | vllm cfg read + `_container_model_mount` | mount: **no** |
| 3595-3603 | family flags `is_fp4` / `is_moe` / `is_gpt_oss` | **no** |
| 3605-3624 | `env_lines` (`-e …`) | **no** — recipe `env:` owns this |
| 3630-3663 | `arg_lines` (every vLLM flag) | **no** — this is the whole point |
| 3665-3675 | image + `_vllm_serve_command` + quote | **no** |
| 3679-3701 | header block + `set -euo pipefail` + `docker rm -f` + `exec docker run …` | header/preamble **yes**, `docker run` **no** |

So the recipe path reuses stages 1, 2 and *part of* 8. Everything else is derived-path-only.

### 1.2 Recommended decomposition

The naive split ("`if recipe: build_recipe_script() else: build_derived_script()`") is exactly
the copy-paste fork the question asks to avoid, because stage 8's header is the part that
**other subsystems parse** — `_parse_script_meta` (`app.py:417-455`) reads `# Name:`,
`# Description:`, `# VRAM:` from the first 20 lines; `_identify_active_profile`
(`app.py:1511-1528`) and `_running_profile_vram_credit` (`app.py:2350-2396`) both do a
substring match of the served model name against the **whole script text**, which for the
live recipe wrapper succeeds only because `# Name: HF Qwen/Qwen3.6-35B-A3B-FP8` is present
[VERIFIED: read `profiles/vLLM/start_hf_qwen_qwen3.6-35b-a3b-fp8.sh`]. Forking the header
into two builders is how that quietly breaks.

**Split by *body*, share the *preamble*:**

```
_build_vllm_profile_script(launch_dir, model_name)
  ├─ info      = _profile_model_info(...)          # unchanged, stages 1-2
  ├─ resolved  = <resolve recipe/derived/default>  # NEW: pure, returns a decision object
  ├─ preamble  = <header + set -euo + docker rm -f> # shared by BOTH shapes
  └─ body      = recipe_body(resolved) | docker_run_body(info, resolved)
```

Three properties make this work and should be stated as plan constraints:

1. **The preamble builder is the only writer of `# Name:` / `# Description:` / `# VRAM:`.**
   Both shapes therefore keep working with `_parse_script_meta`, `_identify_active_profile`
   and `_running_profile_vram_credit`.
2. **`docker rm -f vllm_node` lives in the preamble, not the body.** `_preflight_static`
   (`app.py:2830-2835`) warns when it is missing, for *both* shapes, and the live wrapper's
   own comment explains it is load-bearing (run-recipe.sh reports "already running" for a
   crash-looping container). Putting it in the shared preamble makes it structurally
   impossible to omit from one shape.
3. **Neither shape emits `--restart`.** Locked in Phase 1.1 (`HANDOFF.json` decisions). The
   preamble is where a future regression would sneak in, so it is where a test should point.

The `resolved` object is the seam. It should be a plain dict/dataclass produced by a **pure**
function so that §5's tests can assert on resolution without touching the filesystem —
mirroring exactly why Phase 2 made `_derive_launch_spec` pure.

### 1.3 Where `_derive_launch_spec`'s 16 keys go

`_derive_launch_spec` returns 16 keys (`app.py:1057-1074`). **Only two become flags.**

| Key | Destination | Notes |
|-----|-------------|-------|
| `max_model_len` | `--max-model-len` | Replaces the literal `32768` at `app.py:3646`. **Must be floored** — see §1.5. |
| `recommended_util` | `--gpu-memory-utilization` | Replaces the literal `0.75` at `app.py:3635`. |
| `kv_gb` | comment / diagnostics only | Could refine `# VRAM:`, but `vram_gb` is already computed at `app.py:3481` and admission depends on it — **do not change it in this phase**, that is Phase 5's surface. |
| `declared_max_context`, `max_fitting_context` | comment only | Useful as a "why this number" rationale line. |
| `topology`, `num_hidden_layers`, `full_attention_layers`, `bounded_kv_layers`, `stateless_layers`, `source_field`, `kv_bytes_per_token`, `bounded_bytes_total` | diagnostics only | Never flags. |
| `weights_gb`, `overhead_gb` | echo of inputs | Never flags. |
| `warnings` | **Phase 4** (carry-forward #1) | If a plan chooses to emit them as `#` comments now, they are attacker-influenced strings derived from a vendor `config.json` and MUST go through `_one_line` (`app.py:3548`) and a length cap, exactly like `_one_line(info['name'])` at `app.py:3680`. Recommend *not* emitting them in Phase 3 — it widens the script-text diff for zero SC coverage. |

Nothing else in the 16-key output has a flag home. That is a useful sanity check for the
plan-checker: a plan that maps a third key to a flag is inventing scope.

### 1.4 The inputs `_derive_launch_spec` needs, and where they come from

This is the plumbing work the phase actually requires, and none of it exists yet
(`_derive_launch_spec` has **zero call sites outside tests** — `grep` confirms).

| Parameter | Source available at generation time | Caveat |
|-----------|-------------------------------------|--------|
| `config` (positional) | `_profile_model_info` parses `launch_dir/config.json` at `app.py:3446` and **throws it away** | Needs plumbing — see below |
| `weights_gb` | `info["size_gb"]` (`app.py:3490`) | For HF-cache repos this sums **all blobs** (`app.py:3465`), i.e. every revision plus non-weight files, and is rounded to 1 dp. It over-states for multi-revision caches. Acceptable (it is the same number admission already trusts) but should be stated in the plan, not discovered later. |
| `pool_gb` | Phase 2 default `121.0`, or `_get_total_memory_gb()` (`app.py:282`) | **Recommend keeping the 121.0 default.** Reading meminfo makes every generation test host-dependent and destroys the "testable with vLLM up and untouched" property. |
| `kv_dtype_bytes` | **Must be decided before this call** — see §1.5 | The trap |
| `resident_gb` | Leave at `0.0` | Page cache at *generation* time says nothing about *launch* time. `_preflight_memory` (`app.py:2850`) is the right place for live memory, and it already does it. |
| `requested_context` | `None` | UI-supplied context is Phase 4. |

**Plumbing the config:** `_profile_model_info` returns 9 keys and none of them is the parsed
config or `architectures[]` (`app.py:3482-3492`). `_infer_from_config` does compute
`arch_str` (a lowercased join, `app.py:733`) but that is lossy for map lookup. Two options:

- *(recommended)* add `"architectures": config.get("architectures", [])` to the
  `_profile_model_info` return, and have `_build_vllm_profile_script` read `config.json`
  once more for the solver. Additive to the dict, so the `{"model": info}` HTTP response at
  `app.py:3725` gains one small list — no leak of the whole config.
- return `(info, config)` internally and keep the public dict unchanged. Cleaner data flow,
  but changes a two-caller signature (`app.py:3582`, and nothing else) — also fine.

Either is acceptable; the plan should pick one and say so, because a task that "just reads
config.json again inside `_build_vllm_profile_script`" produces a third parse of the same
file and is the kind of patch CLAUDE.md's "patterns over patches" rule is aimed at.

### 1.5 THE ORDERING TRAP (highest-value finding in this document)

`_derive_launch_spec`'s signature takes `kv_dtype_bytes: int = 1` (`app.py:949`) — i.e. it
currently assumes fp8 KV, which is true today because `app.py:3647` blanket-emits
`--kv-cache-dtype fp8` for every non-gpt-oss model.

SC5 removes that blanket. The moment it does, **the KV dtype becomes an input to the solver,
not an output of it.** Measured on the real Qwen3.6 config with `weights_gb=37.0`
[VERIFIED: executed 2026-08-05]:

| `kv_dtype_bytes` | `max_model_len` | `recommended_util` | `kv_gb` |
|---|---|---|---|
| 1 (fp8) | 262144 | **0.42** | 2.68 |
| 2 (full) | 262144 | **0.44** | 5.37 |

Here only util moves, because the model's declared 262144 cap binds before the budget does.
For a model where the *budget* binds (Phase 2's verification records
`DeepSeek-R1-Distill-Llama-70B` deriving `max_model_len=0` under its real weight size),
doubling KV bytes per token **halves the derivable context**. Getting the order wrong
produces a script that looks plausible and reserves the wrong amount of memory.

**Required order in the plan:**

```
1. resolve recipe (if any)
2. decide kv_cache_dtype        ← recipe command block > model config evidence > none
3. derive spec with kv_dtype_bytes = 1 if fp8 else 2
4. apply recipe overrides on top of the derived spec
5. emit flags
```

Note step 2 precedes step 3 but step 4 follows it — the recipe is consulted twice, for
different things. A plan that collapses this into one pass will get either the dtype or the
precedence wrong.

### 1.6 What a recipe-backed model does NOT get

For a recipe-backed model, steps 3-5 above are skipped entirely: the wrapper emits no flags,
so there is nothing for the derived spec to fill in. `_derive_launch_spec` should not be
called at all on that path (or if called, only to populate a rationale comment). This is
worth stating because it means **SC2 is satisfied by delegation, not by resolution** — the
0.55 reaches vLLM because `run-recipe.sh` reads the YAML at launch, not because the
generator copied `0.55` into a flag. The plan's SC2 test must therefore assert on the
*wrapper naming the right recipe* plus *the reader resolving that recipe to 0.55/262144/
qwen3_xml*, as two separate assertions. Asserting `"0.55" in script` would be asserting a
value the script correctly does not contain.

---

## §2 — Recipe reading

### 2.1 The minimal robust read: render, then tokenize

The survey established that `gpu_memory_utilization` is a **GB count** in two recipes
(survey §1, §2). The fix is not a special case — it is reading the flag from the rendered
`command` block instead of lifting the key from `defaults`, because in those two recipes the
value feeds `--gpu-memory-utilization-gb`, a *different flag name*.

`run-recipe.py` itself defines the rendering semantics at `/home/josh/spark-vllm-docker/
run-recipe.py:484-503`:

```
params  = {**recipe.get("defaults", {}), **overrides}
command = recipe["command"].format(**params)
```

So the reader should do **exactly that** — `str.format` with `defaults` as the mapping — and
then `shlex.split` the result to pull flag values. This is the single most important design
choice in the phase, because it means the reader's answer is *the same string vLLM will
receive*, not a re-derivation of it.

**This was executed against all 27 recipes on disk** [VERIFIED: run 2026-08-05]:

- `command.format(**defaults)` succeeds for **27/27**. No recipe needs a value that is not
  in its own `defaults`.
- Render-then-`shlex.split` extraction yields, for the two GB-count recipes
  (`qwen3.5-397b-int4-autoround`, `step-3.7-flash-fp8`), the key
  `--gpu-memory-utilization-gb: 108` and **no** `--gpu-memory-utilization` key at all. The
  trap is avoided structurally rather than by a magnitude heuristic like "if > 1.0 it must
  be GB".
- `qwen3.6-35b-a3b-fp8-solo` yields exactly `--gpu-memory-utilization 0.55`,
  `--max-model-len 262144`, `--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`,
  `--kv-cache-dtype fp8` — i.e. the whole of SC2 plus its KV declaration, from one read.

**Do not use a magnitude heuristic.** A reader that says "0.55 is a fraction, 108 is GB"
would silently mis-read a future recipe using `--gpu-memory-utilization-gb 0.9` or a patched
flag that takes a fraction above 1. The flag name is the ground truth; the value is not.

### 2.2 Two escaping hazards a regex reader would hit

Both are real, both are in the recipe set today:

1. **`str.format` brace escaping.** `qwen3.6-35b-a3b-fp8-dflash.yaml` contains
   `--speculative-config '{{"method": "dflash", …}}'` — doubled braces, which `str.format`
   collapses to single braces [VERIFIED: read from disk]. A regex reader that scans the raw
   YAML text sees `{{"method"…}}`; only the rendered form is correct. Another reason to
   render first.
2. **Shell quoting.** That same value is a single-quoted shell word containing spaces and
   colons. `shlex.split` handles it; a whitespace `.split()` would shred it and misalign
   every subsequent flag/value pair.

### 2.3 Fields worth normalizing out of a recipe

The reader should return a small normalized record, not raw YAML, so that callers cannot
accidentally reach for `defaults["gpu_memory_utilization"]`:

| Field | Source | Used for |
|-------|--------|----------|
| `name`, `description` | top-level keys | wrapper header text |
| `model` | top-level `model:` | sanity check against the model being profiled (**not** a lookup key — survey §3 proves three recipes share one `model:`) |
| `gpu_memory_utilization` | rendered `--gpu-memory-utilization` | preflight budget; SC2 |
| `gpu_memory_utilization_gb` | rendered `--gpu-memory-utilization-gb` | preflight must **not** treat this as a fraction |
| `max_model_len` | rendered `--max-model-len` | SC2 |
| `kv_cache_dtype` | rendered `--kv-cache-dtype` | SC5 evidence |
| `tool_call_parser`, `reasoning_parser` | rendered flags | SC2/SC4 evidence; also promotes a map entry to "recipe-proven" |
| `port` | rendered `--port` | see §2.5 |
| `solo_only`, `cluster_only` | top-level booleans | see §2.5 |
| `mods` | top-level list | rationale comment in the wrapper: "this recipe applies N in-container mods the generator cannot reproduce" — the *reason* delegation exists |

### 2.4 Failure modes — every one must degrade, not raise

`_build_vllm_profile_script` is reached from an authenticated HTTP route
(`app.py:3728-3730`) **and** from the auto-profile step of a completed HF download
(`app.py:3788`). The download path wraps profile creation in a bare `except Exception` and
emits `auto_profile_error`, so a raising reader turns into a confusing download-time error
message rather than a launch failure — but it still loses the profile.

| Failure | Correct behavior | Why |
|---------|------------------|-----|
| `recipe_dir` does not exist | Fall through to derived path; no recipe fields | The mapping is a *preference*, not a requirement. A box without `~/spark-vllm-docker` must still generate profiles. |
| Recipe file missing for a mapped name | Fall through to derived + surface a warning | Typo in `config.json` is the likely cause; silent derivation is safer than a traceback but must be visible. |
| YAML malformed (`yaml.YAMLError`) | Fall through to derived + warning | `yaml.safe_load` only; never `yaml.load`. |
| YAML parses to a non-dict (e.g. a bare string) | Treat as malformed | `yaml.safe_load("just text")` returns a `str`, and `.get` would `AttributeError`. |
| `command:` missing | Treat as unusable | `run-recipe.py:194` lists `command` as required, so such a file could never launch either. |
| `command.format(**defaults)` raises `KeyError` | Fall through to derived + warning naming the missing placeholder | Mirrors `run-recipe.py:502-503`, which exits with the same diagnosis. 0/27 recipes hit this today, but a future hand-edit will. |
| `shlex.split` raises `ValueError` (unbalanced quote) | Fall through | Same class as malformed. |
| Recipe name fails a safe-name grammar | Reject before touching the filesystem | `_recipe_util` already does this at `app.py:3050` with `_re.fullmatch(r"[\w.-]+", recipe)`. **Keep that guard** — the recipe name comes from `config.json` and is interpolated into a path *and* into a generated shell script. Reuse the same grammar or tighter. |

The unifying rule to state in the plan: **the reader returns "no opinion" for anything it
cannot read, and "no opinion" is distinguishable from "declared zero".** Concretely, the
normalized record should use `None` for absent, never `0` / `0.0` / `""`. This is the same
"absent != zero" discipline `config_from_fixture` documents at
`tests/test_launch_spec.py:71-73` and that `_resolve_attention_topology` observes for
`use_sliding_window`.

### 2.5 Two cheap validations worth adding while the file is open

Neither is required by an SC, but both prevent a class of silent wrong-launch that this
phase newly makes possible (before Phase 3, only one hand-written recipe wrapper existed):

1. **`cluster_only: true`.** 12 of 27 recipes are cluster-only (survey §1). Mapping one
   produces a wrapper that `run-recipe.py:1057` will refuse at launch time. Detectable
   statically at generation time, so detect it there.
2. **Rendered `--port` != 8000.** `_engine_bases["vllm"]` points at `127.0.0.1:8000`; a
   recipe on another port launches successfully and then reads as "not running" everywhere
   in the UI. All 27 recipes default to 8000 today, so this is pure future-proofing — but it
   costs one comparison.

Both should be surfaced as warnings on the generation result and/or as `_pf("warn", …)`
checks, not as hard failures.

### 2.6 Subsuming `_recipe_util` fixes a live latent bug

`_recipe_util` (`app.py:3048-3057`) regexes `^\s*gpu_memory_utilization:\s*([0-9.]+)` out of
the raw YAML — i.e. it lifts the `defaults` key by name, which is precisely the failure mode
the survey warned about. Executed against the real files [VERIFIED 2026-08-05]:

```
_recipe_util("qwen3.6-35b-a3b-fp8-solo")     -> 0.55   (correct)
_recipe_util("step-3.7-flash-fp8")           -> 108.0  (WRONG — that is GiB)
_recipe_util("qwen3.5-397b-int4-autoround")  -> 108.0  (WRONG — that is GiB)
```

That value flows into `_preflight_memory(util)` (`app.py:3029-3032`) and then
`_vllm_budget_gib(108.0, mem)` (`app.py:2581`), which computes `budget = total * util` — a
budget of ~13 TiB, reported to the user as a green "usable" number. Neither recipe is mapped
to a profile today, so it is latent, not active. **Replacing `_recipe_util` with the §2.1
reader closes it as a side effect**, and that is a legitimate regression test to write
(`_preflight_memory` must decline to report a fractional budget when the recipe declares GB).

The plan should keep `_recipe_util`'s *call site* contract — preflight wants "a fraction or
`None`" — and change only its implementation, so `app.py:3029-3031` needs no edit.

---

## §3 — Capability map shape

### 3.1 What the repo already does for curated knowledge

Two existing precedents, and they point in different directions:

| Structure | Location | Shape | Reload | Fits? |
|-----------|----------|-------|--------|-------|
| `alerts` block | `config.json`, read at `app.py:1878-1891` | Machine/deployment knobs, read **once at import** into module constants | Requires service restart | Poor fit — a capability map is curated knowledge, not a deployment knob, and `_ALERT_*` constants are frozen at import |
| `recommendations.json` | tracked file beside `app.py`, loaded by `_load_recommendations` (`app.py:1649-1655`) | Curated KB with `meta.schema_version`, per-entry `id` / `severity` / `sources[]` / `confidence`, loaded **on every call** with a `try/except` that logs and returns an empty skeleton | Live — no restart | **Strong fit** |

`recommendations.json` already carries a per-entry `confidence` field and a `sources[]` list
with URLs and dates — exactly the "recipe-proven vs template-inferred" distinction the
survey (§6) demands, in a shape Josh already maintains (and that `research_refresh.py`
already proposes updates against).

**Recommendation:** a new tracked `model_capabilities.json` beside `recommendations.json`,
loaded by a `_load_model_capabilities()` modeled line-for-line on `_load_recommendations`.
Rationale: (a) it is the established repo pattern, which CLAUDE.md's "patterns over patches"
rule prefers; (b) it keeps `app.py` churn small, which matters with ~3 concurrent sessions
editing it; (c) it is live-reloading, so correcting a wrong parser does not require
`systemctl --user restart`; (d) it keeps curated model knowledge out of `config.json`, which
is deployment configuration.

Counter-argument worth recording: a module constant in `app.py` cannot be desynced from the
code that reads it and needs no I/O error path. If the planner prefers that, it is defensible
— but it forfeits live reload and adds ~60 lines to an 8134-line file two other sessions are
editing.

### 3.2 Keying

**Key on both, checked in this order: exact model id → `architectures[]` → nothing.**

Evidence for needing both:

- `architectures[]` alone is insufficient. Survey §4 shows `Qwen/Qwen3.6-35B-A3B-FP8` and
  `lyf/Qwen3.6-…-NVFP4` share `Qwen3_5MoeForConditionalGeneration`, which is fine — but
  `qwen2.5-14b-instruct-gptq-int8` and both DeepSeek-R1-Distill-Qwen models all share
  `Qwen2ForCausalLM`, and they need **different** answers (`hermes` vs "no tool parser").
  A pure-architecture map cannot express SC4.
- Model id alone is insufficient for generalization: a newly downloaded Qwen3 dense model
  would have no entry at all, whereas `Qwen3ForCausalLM` covers the family.
- `architectures[]` is genuinely available: every LLM config on this box declares it except
  the two `faster-whisper` models (survey §4), and those are excluded upstream anyway by the
  `task_label` gate at `app.py:3585-3586`.

Note the model-id key must match the same string the generator has, i.e. `info["name"]`
(`app.py:3483`), which is `owner/repo` for HF-cache repos and a **bare directory name** for
flat dirs (`qwen3-vl-4b-fp8`, `qwen3-next-80b-a3b-nvfp4`). Both forms appear in survey §4's
victim list, so the map must contain both shapes verbatim. Match case-insensitively, or the
map becomes a trap.

### 3.3 Entry shape

Five things must be expressible; today's code can express one.

| Concept | Field | Values |
|---------|-------|--------|
| Tool parser | `tool_call_parser` | string \| `null` |
| Reasoning parser | `reasoning_parser` | string \| `null` |
| "Does not support tool calling" | `supports_tool_calling` | `true` \| `false` \| `null` (= unknown) |
| Provenance | `confidence` | `"recipe-proven"` \| `"template-inferred"` \| `"unproven"` |
| Why | `evidence` | free text + optional path, mirroring `recommendations.json`'s `sources[]` |

`supports_tool_calling: false` and `tool_call_parser: null` are **not** the same statement —
the first is a positive finding (whisper models, `supports_tool_calling: false`), the second
is an absence. Collapsing them loses the survey's distinction between "no tool support" and
"tool support probable but no parser confirmed" (the DeepSeek-R1 distills).

**Emission rule (needs a decision — see Open Question OQ-1):** the CONTEXT locks
"emitting no flag degrades to a working server, emitting the wrong parser produces silently
corrupted tool calls", and the survey (§6) says unproven rows should emit no tool flags. The
conservative reading — emit only `confidence: recipe-proven` — satisfies SC2 and SC4 in full
and is strictly an improvement on today for every affected model (they currently receive a
*wrong* parser, so emitting nothing is a net gain, not a regression). The permissive reading
would also emit `template-inferred` `hermes` for the Qwen3 dense / Qwen2.5 / Qwen3-Next /
Qwen3-VL rows, gaining them tool calling they do not have today at the cost of an unverified
guess. **Recommendation: make it data, not code** — a per-entry `emit: true|false` derived
from `confidence` by a documented default, overridable per row. That way promoting `hermes`
after Josh tests it is a JSON edit, not a code change (CLAUDE.md automation-first).

The `lyf/Qwen3.6-…` row is the interesting middle case: its `chat_template.jinja` is
md5-identical to the recipe-backed Qwen3.6 model's (survey §4, `52b6d51a…`). That is
stronger evidence than "template-inferred" and weaker than "a recipe for this exact model".
Recommend a distinct `confidence: "template-identical-to-recipe-proven"` with `emit: true`,
so the reasoning is recorded rather than laundered into one of the other two buckets.

### 3.4 No enum validation — but do validate *shape*

Locked decision #8 forbids checking parser names against a registry. That does **not** mean
accepting arbitrary strings into a shell script. The value is interpolated into a generated
`start_*.sh`, so it needs the same treatment `_MOE_BACKEND_RE` (`app.py:3535`,
`^[a-z0-9_]{1,64}$`) already gives `--moe-backend`: a **grammar** check, not a **membership**
check. `qwen3_xml`, `nano_v3`, `super_v3`, `openai_gptoss`, `granite-20b-fc` and `step3p5`
all pass a `^[a-z0-9_.-]{1,64}$` grammar; nothing shell-dangerous does. `shlex.quote` on top,
per the Phase 1 defence-in-depth decision recorded in `HANDOFF.json`.

### 3.5 The Nemotron-Super conflict is not this phase's to resolve

Survey §8: `start_nemotron_super.sh` uses `--reasoning-parser super_v3` (local plugin at
`/home/josh/super_v3_reasoning_parser.py`) while `nemotron-3-super-nvfp4.yaml` uses
`nemotron_v3`. Both profiles are **hand-written and not regenerated by this phase**. The map
should carry one value with an `evidence` note recording the disagreement, and the plan
should not touch either artifact. Flagged as OQ-3.

---

## §4 — Precedence

### 4.1 The chain, per field

"Recipe wins, then derived, then default" is true but under-specified, because the three
layers do not all have opinions about the same fields. Per-field:

| Flag | Recipe layer | Derived layer | Generic default | Notes |
|------|-------------|---------------|-----------------|-------|
| `--gpu-memory-utilization` | rendered flag value | `spec["recommended_util"]` | `0.75` (`app.py:3635`) | Recipe-backed models never reach the flag emitter at all (§1.6). |
| `--max-model-len` | rendered flag value | `spec["max_model_len"]` **if > 0** | `32768`, or `65536` for gpt-oss (`app.py:3641`) | The `> 0` floor is mandatory — see §4.3. |
| `--kv-cache-dtype` | rendered flag value | model config evidence only | **omit** | Decided *before* derivation (§1.5). |
| `--tool-call-parser` / `--reasoning-parser` | rendered flag values (also promote the map entry to recipe-proven) | capability map | **omit** | The map is not a "derived" layer in the Phase 2 sense; it is a third independent source. |
| `--max-num-seqs`, `--enable-chunked-prefill`, `--moe-backend`, `--trust-remote-code`, `--dtype auto`, `--generation-config vllm` | — | — | unchanged literals | **Out of scope.** No SC names them; leave `app.py:3630-3663` alone for these. |

### 4.2 What "absent" means at each layer

| Layer | "Absent" means | Representation |
|-------|----------------|----------------|
| Recipe | the flag is not in the rendered command | `None` in the normalized record — **never** `0`/`0.0`/`""` |
| Derived | the solver could not produce a usable number | `max_model_len == 0`, or a `warnings` entry naming the reason |
| Capability map | no entry for this model id or architecture | key absent from the dict; distinct from an entry with `tool_call_parser: null` |
| Generic default | n/a — the default layer is total by construction | it always answers |

The rule that makes this safe to implement: **the resolver must test `is None`, never
truthiness.** `if recipe_util:` silently discards a legitimate `0.0`; `if recipe_max_len:`
silently discards `0`. Neither value is plausible in a real recipe today, but the failure is
invisible and the guard is free.

### 4.3 The zero-derivation floor (a real, verified landmine)

`_derive_launch_spec` returns `max_model_len: 0` for a config with no attention-layer
information. Executed against the exact minimal config the existing generation tests use
(`{"model_type": "qwen3", "torch_dtype": "bfloat16"}`, `tests/test_vllm_profile_generation.py:14`)
[VERIFIED 2026-08-05]:

```
max_model_len 0 | max_fitting_context 0 | declared_max_context 0
recommended_util 0.1 | kv_bytes_per_token 0
warnings ['no full-attention layers; context is not KV-bounded']
```

A naive wiring therefore emits `--max-model-len 0 --gpu-memory-utilization 0.1` — a script
that is syntactically valid bash, passes `_bash_syntax_ok`
(`tests/test_vllm_profile_generation.py:116`), and cannot serve. **This is the generic-default
layer's reason to exist.** The resolver must treat a non-positive derived `max_model_len`
(and, arguably, a `recommended_util` that landed on the `util_floor` of 0.10) as "the
derived layer has no opinion" and fall through to the literal default.

Concretely: the third precedence tier is not decoration. Any plan that implements only
`recipe → derived` will ship this bug, and the current suite will not fail.

### 4.4 Pattern matching for `vllm.recipes` keys

Discretionary (CONTEXT), but the constraint is "deterministic and tested".

**Recommendation: `fnmatch.fnmatchcase` over a casefolded pair.** Rationale:

- The CONTEXT's own example already uses glob syntax (`"Qwen/Qwen3.6-*"`), so glob is the
  shape Josh has in mind.
- `fnmatch.fnmatch` normalizes case via `os.path.normcase`, which on Linux is the identity
  function — so `fnmatch` is case-*sensitive* here and would be case-*insensitive* on a
  hypothetical Windows run. `fnmatchcase(name.lower(), pattern.lower())` makes the behavior
  explicit and platform-independent. [VERIFIED: stdlib semantics; `os.path.normcase` is
  identity on POSIX]
- Regex would let a `config.json` typo become a catastrophic-backtracking DoS on an
  authenticated route. Prefix matching cannot express `*-NVFP4`.

**Collision resolution.** With `Qwen/Qwen3.6-*` and `Qwen/*` both present, two models of
determinism are available:

1. *First match in file order.* `json.load` preserves object key order (Python 3.7+), so
   this is deterministic and human-controllable — but it is invisible in the file (nothing
   tells a reader that order matters) and a reformatter that sorts keys silently changes
   behavior.
2. *(recommended)* **Most-specific wins, with a total order.** Rank candidates by
   (a) fewer wildcard characters, then (b) longer literal length, then (c) lexicographic
   pattern — the last rung guarantees totality so two equally-specific patterns still resolve
   deterministically rather than by dict order. Emit a warning when more than one pattern
   matches, so the ambiguity is visible even though it is resolved.

Whichever is chosen, the plan must include a test with **two deliberately overlapping
patterns** asserting the exact winner, and a test that shuffling the dict's insertion order
does not change the answer (which option 1 would fail, by design — that is the trade-off to
make explicitly rather than implicitly).

### 4.5 Why the allowlist keeps SC3 true

Locked decision #4 makes `vllm.recipes` opt-in, and SC3 (gpt-oss keeps 65536 +
full-precision KV) then holds trivially: `openai/gpt-oss-*` is simply not a key, so gpt-oss
never enters the recipe path and keeps the `is_gpt_oss` branch at `app.py:3637-3643`
verbatim. Two consequences the plan must respect:

- **The `is_gpt_oss` branch must survive the refactor unchanged.** SC3 says the behavior
  "already exists and must survive rather than being re-derived" (CONTEXT). A plan that
  replaces `--max-model-len 65536` with a derived number for gpt-oss breaks SC3 even if the
  number happens to come out at 65536.
- **`vllm.recipes` being absent or empty must be a no-op**, restoring exactly today's
  behavior for every model. `test_defaults_unchanged_when_no_vllm_config`
  (`tests/test_vllm_profile_generation.py:97`) already encodes that habit for `vllm.image` /
  `moe_backend`; the same test shape should be extended, not replaced.

---

## Validation Architecture

> Required by Nyquist validation. `.planning/config.json` does not exist in this repo, so
> `workflow.nyquist_validation` is absent — treated as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.1.0 (+ pytest-asyncio), from `requirements-dev.txt` [VERIFIED: `python3 -m pytest --version`] |
| Config file | **none** — no `pytest.ini`, `pyproject.toml`, `setup.cfg` or `tox.ini` in the repo. Discovery is default; `tests/conftest.py:7-8` does the `sys.path` insert. |
| Quick run command | `python3 -m pytest tests/test_vllm_profile_generation.py tests/test_launch_preflight.py -q` |
| Full suite command | `python3 -m pytest tests/ -q` |
| Current baseline | `310 passed in 0.74s` [VERIFIED 2026-08-05] |

The whole suite runs in under a second and touches no network, no docker and no GPU. There is
no reason for a per-task command narrower than the full suite; use the full suite everywhere.

### The core question: does script-text assertion cover SC1-SC5?

**Yes for SC1, SC3, SC4, SC5. Partially for SC2 — and the gap is by design, not by weakness.**

Existing generation tests assert on emitted script text
(`tests/test_vllm_profile_generation.py`, 285 lines). That works for every criterion whose
observable is a flag in a generated script. SC2 is different: for a recipe-backed model the
generated script **correctly contains no flags at all** (§1.6). Its numbers live in the YAML
that `run-recipe.sh` reads at launch. So SC2 needs two assertions in different places:

- *script text*: the wrapper names recipe `qwen3.6-35b-a3b-fp8-solo` and delegates to
  `run-recipe.sh`.
- *reader output*: that recipe name resolves to `0.55 / 262144 / qwen3_xml`.

A single `assert "0.55" in script` would be asserting a value the correct script does not
contain, and would push an implementer toward reconstructing the recipe inline — the exact
thing locked decision #5 forbids. This is worth stating in the plan explicitly.

### Phase Requirements → Test Map

| Req | Behavior (roadmap SC) | Test type | Automated command | File exists? |
|-----|----------------------|-----------|-------------------|--------------|
| REQ-05 / SC1 | `vllm.recipes` pattern map overrides derived values; absent/empty block is a no-op | unit | `pytest tests/test_recipe_resolution.py -x` | ❌ Wave 0 |
| REQ-05 / SC1 | Two overlapping patterns resolve deterministically; insertion-order shuffle does not change the winner | unit | `pytest tests/test_recipe_resolution.py -k collision -x` | ❌ Wave 0 |
| REQ-05 / SC2 | `qwen3.6-35b-a3b-fp8-solo` reads as `0.55 / 262144 / qwen3_xml / qwen3 / fp8` | unit (fixture YAML) | `pytest tests/test_recipe_reader.py -k qwen36 -x` | ❌ Wave 0 |
| REQ-05 / SC2 | A mapped model emits a delegating wrapper naming that recipe, with `docker rm -f`, no `--restart`, no `--model` | unit (script text) | `pytest tests/test_vllm_profile_generation.py -k recipe_backed -x` | ⚠️ file exists, tests new |
| REQ-05 / SC2 | The wrapper is recognized by the existing preflight contract (`_parse_launch_script` → `recipe_backed=True`, correct recipe name) | unit | `pytest tests/test_launch_preflight.py -k generated_recipe_wrapper -x` | ⚠️ file exists, test new |
| REQ-05 / SC3 | gpt-oss emits `--max-model-len 65536` and **no** `--kv-cache-dtype` | unit (script text) | `pytest tests/test_vllm_profile_generation.py -k gpt_oss -x` | ⚠️ file exists, test new |
| REQ-05 / SC3 | gpt-oss is not matched by any `vllm.recipes` entry in the shipped `config.example.json` | unit | `pytest tests/test_recipe_resolution.py -k gpt_oss_not_mapped -x` | ❌ Wave 0 |
| REQ-05 / SC4 | The five substring victims (survey §5) receive **no** `qwen3_coder` | unit, parametrized over 5 model ids | `pytest tests/test_capability_map.py -k victims -x` | ❌ Wave 0 |
| REQ-05 / SC4 | A model with no map entry emits neither `--tool-call-parser` nor `--enable-auto-tool-choice` | unit | `pytest tests/test_capability_map.py -k unmapped -x` | ❌ Wave 0 |
| REQ-05 / SC4 | `supports_tool_calling: false` is expressible and suppresses flags | unit | `pytest tests/test_capability_map.py -k unsupported -x` | ❌ Wave 0 |
| REQ-05 / SC5 | `--kv-cache-dtype fp8` appears only for models with recipe or config evidence; parametrized negative cases over the profile-only rows in survey §5 | unit | `pytest tests/test_capability_map.py -k kv_dtype -x` | ❌ Wave 0 |
| REQ-05 / SC5 | KV dtype is decided **before** derivation — same config derives different util at `kv_dtype_bytes` 1 vs 2, and the emitted util matches the emitted dtype | unit | `pytest tests/test_recipe_resolution.py -k kv_dtype_ordering -x` | ❌ Wave 0 |
| REQ-05 (guard) | A config with no attention info does **not** emit `--max-model-len 0` or `--gpu-memory-utilization 0.1` | unit | `pytest tests/test_vllm_profile_generation.py -k degenerate -x` | ❌ Wave 0 |
| REQ-05 (guard) | `_recipe_util` returns `None`, not `108.0`, for the two GB-count recipes | unit | `pytest tests/test_recipe_reader.py -k gb_not_fraction -x` | ❌ Wave 0 |
| REQ-05 (guard) | Malformed / missing / non-dict / unformattable recipe degrades to derived, never raises | unit, parametrized | `pytest tests/test_recipe_reader.py -k degrades -x` | ❌ Wave 0 |
| REQ-05 (guard) | Every generated script still passes `bash -n` on both paths | unit | `pytest tests/test_vllm_profile_generation.py -q` | ✅ helper exists (`:116`) |

### Fixtures — what the tests read

Three kinds are needed, and the choice matters because `tests/conftest.py:15-22` records a
deliberate repo rule: *committed snapshot files, never a live glob*, so that adding or
deleting a model on the box cannot break unrelated tests.

1. **Recipe YAML fixtures — commit byte-identical copies.** Copy at minimum
   `qwen3.6-35b-a3b-fp8-solo.yaml` (SC2), `openai-gpt-oss-120b.yaml` (SC3's
   recipe-vs-profile conflict), `qwen3.6-35b-a3b-fp8-dflash.yaml` (the `{{…}}` brace-escape
   case), `step-3.7-flash-fp8.yaml` (the GB-count trap), and one `cluster_only` recipe.
   Point the reader at a `tmp_path`/fixture dir via the `recipe_dir` parameter — which is
   why `recipe_dir` must be a *parameter with a default*, not a module constant. This is the
   single most important testability constraint in the phase.
2. **A drift alarm, not a live dependency.** Add one test that, **if**
   `~/spark-vllm-docker/recipes/` exists, asserts each committed fixture is byte-identical to
   its live counterpart, and `pytest.skip`s otherwise. That gets the fixture-stability
   benefit without making the suite depend on a directory outside the repo.
3. **Model configs — reuse `02-MODEL-FIXTURES.json`.** `tests/conftest.py:33-43` already
   exposes `model_fixtures` / `fixture_models` session fixtures over 17 real on-box configs,
   including all five SC4 victims and gpt-oss. **Caveat, verified:** the snapshot rows carry
   no `architectures` field (keys are `name`, `model_type`, `precedence_field`, layer counts,
   …). Since §3.2 keys the capability map on `architectures[]`, either the snapshot must be
   regenerated with an `architectures` column (a deliberate act per the conftest rule) or the
   capability-map tests must supply architectures inline. **Recommend regenerating the
   snapshot to add `architectures`** — the survey §4 already lists the correct value for all
   17 models, so it is a mechanical, verifiable addition, and it avoids a second parallel
   fixture source. Flagged as a Wave 0 gap because it must happen before the map tests.

### Sampling Rate

- **Per task commit:** `python3 -m pytest tests/ -q` (0.74s — no reason to sample narrower)
- **Per wave merge:** `python3 -m pytest tests/ -q` plus `bash -n` over every file in
  `profiles/vLLM/` (catches a wrapper that is valid in a test but not on disk)
- **Phase gate:** full suite green, plus the two live checks in the next subsection

### Wave 0 Gaps

- [ ] `tests/test_recipe_reader.py` — reader semantics, GB trap, degradation matrix
- [ ] `tests/test_recipe_resolution.py` — pattern matching, collision determinism, precedence, kv-dtype ordering
- [ ] `tests/test_capability_map.py` — SC4 victims, unmapped, unsupported, SC5 kv evidence
- [ ] Recipe YAML fixtures (5 files) + the byte-identity drift test
- [ ] `02-MODEL-FIXTURES.json` regenerated with an `architectures` column (deliberate act; conftest rule)
- [ ] No framework install needed — pytest is present and the suite is green

### What genuinely CANNOT be proven statically

Stated honestly, with what would prove each:

| Claim | Why static proof is impossible | What would prove it |
|-------|-------------------------------|---------------------|
| The container's vLLM accepts `qwen3_xml` / `nano_v3` / `super_v3` | Survey §4: the host has vLLM 0.21.0 but containers run 0.20.0 / 0.23.1 / cu130-nightly, and their parser registries are not readable from the host. `nano_v3` is downloaded at mod-apply time and is not on disk at all. | `vllm serve --help` inside each exact image, or a real launch. **Deliberately not attempted** — this is precisely why locked decision #8 forbids enum validation. |
| The derived util for an unmapped model actually fits | Memory behavior under load is not a property of script text | A launch. Out of scope; Phase 5 owns admission truth. |
| `hermes` is the right parser for the Qwen3 dense / Qwen2.5 / Qwen3-Next / Qwen3-VL rows | Survey §6 marks these template-inferred, not recipe-confirmed | A tool-calling integration test against a live server, or a hand-measured recipe. §3.3's conservative emission rule means Phase 3 does not depend on the answer. |
| The recipe wrapper actually launches | Requires running it | **Partially available without a launch:** `run-recipe.sh <recipe> --dry-run` is read-only — `run-recipe.py:1205` skips the image check under `--dry-run` and each build/download phase is guarded (`:1132`, `:1174`, `:1271`). `_preflight_smoke` (`app.py:2958-2978`) already invokes it in production. Safe to run with vLLM up. |

### Live checks that are safe with vLLM UP (phase gate, not suite)

vLLM is currently serving Qwen3.6 at util 0.55 and must not be disturbed. These two are
read-only and touch nothing the running container owns:

1. `POST /api/vllm/preflight` against the *existing* live Qwen3.6 profile — exercises
   `_recipe_util`'s replacement through its real call site and confirms the budget number is
   still 0.55-derived, not 108.
2. `cd ~/spark-vllm-docker && ./run-recipe.sh qwen3.6-35b-a3b-fp8-solo --dry-run` — proves
   the mapped recipe is still launchable, without launching.

Neither starts, stops or inspects the running container. Do **not** add a phase-gate step
that regenerates the live Qwen3.6 profile; see §6.

---

## §6 — Regression risk

### 6.1 Which existing tests break, and which SHOULD

Baseline: **310 passed**. Searched the whole suite for assertions on the values this phase
changes [VERIFIED: `grep -rn "0.75\|32768\|qwen3_coder\|kv-cache-dtype\|65536\|max-model-len" tests/`]:

| Test | Contains | Breaks? | Should it? |
|------|----------|---------|------------|
| `tests/test_launch_preflight.py:33-34,62` | `--gpu-memory-utilization 0.75`, `--max-model-len 32768`, `assert facts["util"] == 0.75` | **No** — these live in a hand-written *sample script string* inside the test, not in generator output | No. Leave untouched. |
| `tests/test_vram_admission.py:40-45` | `start_qwen3_coder` | **No** — a profile *filename*, unrelated to the parser flag | No |
| `tests/test_profile_meta.py:41-44` | `start_qwen3_coder.sh` | **No** — same | No |
| `tests/test_vllm_profile_generation.py` (all 285 lines) | mounts, image, moe-backend, serve-command, injection | **No** | — |
| `tests/test_launch_spec.py::test_calibration_qwen36_gap_is_left_for_phase_3` (`:822-832`) | asserts derived util stays `0.42` | **No**, and must not | **Explicitly no.** Locked decision #3. If a plan makes this fail, the plan edited the formula. |

**Conclusion: zero existing tests encode the old hardcoded values, so zero will break.** That
sounds like good news and is actually the phase's biggest validation hazard: it means the
suite gives **no signal at all** on whether the new numbers are right. Verified concretely —
a wiring that emits `--max-model-len 0 --gpu-memory-utilization 0.1` keeps all 310 green
(§4.3). Every guarantee in this phase comes from *new* tests; none comes from *not breaking*
old ones.

One test *shape* that should be extended rather than left alone:
`test_defaults_unchanged_when_no_vllm_config` (`:97-110`) is the existing "absent config =
previous behavior" guard. Extending it to also assert no recipe/capability behavior fires
when the blocks are absent is the cheapest possible SC1 no-op proof.

### 6.2 Live profiles on disk — what a regenerate would clobber

**The live Qwen3.6 wrapper is protected today, and Phase 3 can accidentally remove that
protection.**

`_create_vllm_profile_from_path` (`app.py:3716-3719`):

```
if target.exists():
    existing = target.read_text(errors="ignore")
    if str(launch_dir) not in existing:
        raise HTTPException(409, "Profile script already exists with different contents")
```

So an existing script is overwritten **iff it contains the launch-dir path string**. The live
`start_hf_qwen_qwen3.6-35b-a3b-fp8.sh` is hand-written and contains no
`/home/josh/.cache/...` path [VERIFIED: read from disk] — so re-adding Qwen3.6 through the UI
today returns 409 and the live wrapper survives. That safety is **incidental**.

If Phase 3's generated wrapper carries an `# Auto-generated by DGX Model Manager from:
<launch_dir>` provenance line (as the derived path does at `app.py:3684-3685`), then the
*next* regeneration will silently overwrite it. For a generated wrapper that is correct and
desirable. For the **currently live, hand-written, production-serving** one it is not, until
a human has diffed the two.

**Plan requirements:**

- Do not regenerate `profiles/vLLM/start_hf_qwen_qwen3.6-35b-a3b-fp8.sh` as part of
  implementation. Generate to a scratch path and **diff** against the live file; adopting it
  is a separate, explicitly reversible task (`cp` the original aside first). The CONTEXT
  already demands this ("plans must not regenerate or overwrite the live Qwen3.6 profile
  without an explicit, reversible task").
- The generated wrapper should be *semantically equivalent* to the hand-written one, and the
  diff is the evidence. Expect legitimate differences: the hand-written file uses
  `RECIPE_DIR`/`RECIPE` shell variables (which `_expand_script_vars`, `app.py:2737`, exists
  to resolve) while a generator would more likely inline the values. Both parse correctly —
  `tests/test_launch_preflight.py:284-287` already covers the variable form. Inlining is
  simpler and avoids depending on `_expand_script_vars`; either is acceptable, but the plan
  should choose deliberately, not by accident.

Other on-disk profiles worth knowing about:

| File | State | Risk |
|------|-------|------|
| `start_hf_lyf_…-nvfp4.sh` | Auto-generated but **stale** — carries `--restart unless-stopped` and no `vllm serve`, i.e. pre-Phase-1 output [VERIFIED: read] | Regenerating it would *fix* those and apply SC4's `qwen3_xml`. Good, but it is a behavior change to a file on disk — make it an explicit task, not a side effect. |
| `start_hf_openai_gpt-oss-120b.sh` | **Modified, uncommitted** by a peer session (removes `VLLM_USE_FLASHINFER_MOE_FP4`) [VERIFIED: `git diff`] | Do not touch. Regenerating it would revert a peer's uncommitted work and re-entrench the disputed env var — which the CONTEXT's deferred list explicitly warns against. |
| `start_hf_qwen_qwen3-8b.sh`, `start_hf_qwen_qwen3-14b.sh` | **Untracked** [VERIFIED: `git status`] | Hand-written, use runtime `$(ls -d …)` snapshot resolution. Not generator output; leave alone. |
| `start_nemotron_super.sh` | Hand-written, `super_v3` vs the recipe's `nemotron_v3` | Survey §8 — note, do not pick a side. |

### 6.3 Deployment

Editing `app.py` does not restart the service; `systemctl --user restart
dgx-model-manager.service` is required (STATE.md, CLAUDE.md). Since `_load_recommendations`-style
JSON files are read per-call, a capability map in JSON needs no restart — but the code that
*reads* it does, on first deploy.

---

## Don't Hand-Roll

| Problem | Don't build | Use instead | Why |
|---------|-------------|-------------|-----|
| Parsing recipe YAML | A regex scanner (the `_recipe_util` approach) | `yaml.safe_load` | Already a declared runtime dep, already imported (`app.py:31`), already used twice. The regex version is *demonstrably wrong* on 2/27 recipes (§2.6). |
| Rendering `{placeholders}` | Manual `str.replace` per key | `command.format(**defaults)` | Matches `run-recipe.py:501` exactly, so the reader's answer is the launcher's answer. Handles `{{` escaping for free. Verified on 27/27. |
| Splitting the command into flags | `.split()` / regex per flag | `shlex.split` | The dflash recipe contains a quoted JSON blob with spaces; whitespace splitting misaligns every subsequent pair. |
| Deciding fraction-vs-GB | A magnitude heuristic (`> 1.0 → GB`) | Read the flag *name* from the rendered command | The name is ground truth; the value is not (§2.1). |
| Glob matching | Hand-written `*`/`?` walker or a regex translation | `fnmatch.fnmatchcase` | stdlib, no ReDoS surface, matches the CONTEXT's example syntax. |
| Validating parser names | An enum/registry check | A shell-safety grammar + `shlex.quote` | Locked decision #8; the host registry is not authoritative for the container. `_MOE_BACKEND_RE` (`app.py:3535`) is the existing precedent. |
| Re-deriving max context per KV dtype | Custom arithmetic in the generator | `_derive_launch_spec(kv_dtype_bytes=…)` | It already takes the parameter and is table-tested against 17 real configs. |
| A second curated-KB loader | New file format / loader | `_load_recommendations` shape (`app.py:1649`) | Live reload, logged failure, empty-skeleton fallback — all three already solved. |

**Key insight:** every trap in this phase is a *reading* trap, not a computing trap. The
values are correct on disk; the only way to get them wrong is to read them from the wrong
place — the `defaults` key instead of the rendered command, the raw YAML instead of the
formatted string, the model id instead of the explicit mapping, the display name instead of
the architecture.

## Common Pitfalls

### P1 — Emitting `--kv-cache-dtype` after deriving the spec
**What goes wrong:** util and context are computed at fp8 KV sizing but the script omits the
fp8 flag, so vLLM allocates 2× the KV the budget assumed.
**Why:** `kv_dtype_bytes` defaults to `1` (`app.py:949`) and the current blanket flag makes
that silently true today.
**Avoid:** the five-step order in §1.5. Decide dtype → derive → override → emit.
**Warning sign:** the emitted `--gpu-memory-utilization` is identical for two models that
differ only in whether they got the fp8 flag.

### P2 — Treating `0` / `None` / missing as the same
**What goes wrong:** `if recipe_util:` discards a real `0.0`; `if spec["max_model_len"]:`
correctly discards a degenerate `0` **by accident** and then someone "fixes" it.
**Avoid:** `is None` for absence, an explicit `> 0` guard for usability. They are different
checks with different meanings (§4.2, §4.3).

### P3 — Forking the script header between the two shapes
**What goes wrong:** the recipe wrapper drifts from the derived script's header, and
`_parse_script_meta` / `_identify_active_profile` / `_running_profile_vram_credit` silently
degrade — the UI stops showing which profile is active, and admission stops crediting the
running model's VRAM.
**Avoid:** shared preamble builder (§1.2). The served model name must appear in the wrapper
text; today it does only via `# Name: HF <model id>`.
**Warning sign:** `/api/warm-models` returns `active_profile: null` while vLLM is serving.

### P4 — Assuming the green suite means the numbers are right
**Covered in §6.1.** 310 tests pass with `--max-model-len 0` emitted.

### P5 — Auto-discovering recipes by their `model:` field
**What goes wrong:** three recipes declare `model: Qwen/Qwen3.6-35B-A3B-FP8` and are
materially different launches (survey §3). Auto-discovery picks one arbitrarily.
**Avoid:** the mapping is an explicit human choice in `config.json`; `model:` is a *sanity
check*, never a key (§2.3).

### P6 — Mapping a `cluster_only` recipe
**What goes wrong:** the wrapper generates fine and fails at launch (`run-recipe.py:1057`).
12 of 27 recipes are cluster-only.
**Avoid:** the §2.5 validation.

## Open Questions

**OQ-1 — Emission policy for `template-inferred` capability entries.**
- Known: the CONTEXT locks "no map entry → no tool flags", and the survey says unproven rows
  should emit nothing.
- Unclear: whether `template-inferred` (the `hermes` rows) counts as "unproven". SC2 and SC4
  hold either way.
- Recommendation: conservative default (emit only recipe-proven), expressed as a per-entry
  `emit` field so promotion is a JSON edit. Worth one sentence of confirmation from Josh
  before implementation, since it decides whether Qwen3-8B gains or simply loses tool calling.

**OQ-2 — Does the `lyf/Qwen3.6-…` md5-identical template justify emitting `qwen3_xml`?**
- Known: byte-identical chat template to the recipe-backed Qwen3.6 (survey §4).
- Unclear: whether "identical template" is sufficient evidence for Josh.
- Recommendation: a distinct `confidence` value with `emit: true`, so the reasoning is
  recorded. SC4 is satisfied regardless (it only requires *not* `qwen3_coder`).

**OQ-3 — Nemotron-Super `super_v3` vs `nemotron_v3`.**
- Survey §8 says note it, do not pick a side. Neither profile is regenerated by this phase.
- Recommendation: record both in the map's `evidence` and leave both artifacts alone.

**OQ-4 — Regenerate `02-MODEL-FIXTURES.json` to add `architectures`?**
- Known: the snapshot lacks the field; `tests/conftest.py:15-22` makes regeneration a
  deliberate act.
- Recommendation: yes, regenerate (Wave 0), because the alternative is a second parallel
  fixture source for the same 17 models.

**OQ-5 — Inline the recipe name in the wrapper, or use a `RECIPE=` shell variable?**
- Both are preflight-compatible (`tests/test_launch_preflight.py:284-287` covers the
  variable form). Inlining avoids depending on `_expand_script_vars`.
- Recommendation: inline, but this is genuinely discretionary.

## Environment Availability

| Dependency | Required by | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python 3 + pytest | the whole validation architecture | ✓ | pytest 9.1.0 | — |
| `pyyaml` | recipe reader | ✓ | 6.0.2, imported at `app.py:31` | — |
| `~/spark-vllm-docker/recipes/` | real recipe values | ✓ | 27 YAMLs | Committed fixtures (§Validation) — the suite must not require it |
| `~/spark-vllm-docker/run-recipe.sh` | wrapper execution + preflight dry-run | ✓ | — | `_preflight_smoke` already degrades to `skip` if absent (`app.py:2961-2963`) |
| docker | not needed by any Phase 3 test | n/a | — | — |
| Running vLLM | not needed; must stay untouched | ✓ (serving Qwen3.6 @ 0.55) | — | — |

**Missing dependencies with no fallback:** none.

## Assumptions Log

| # | Claim | Section | Risk if wrong |
|---|-------|---------|---------------|
| A1 | `fnmatch.fnmatch` is case-sensitive on Linux because `os.path.normcase` is the identity on POSIX | §4.4 | Low — the recommendation is `fnmatchcase` on a casefolded pair, which is correct either way |
| A2 | A tracked JSON capability map is preferable to a module constant | §3.1 | Low — both work; the JSON choice is a maintainability judgment, and CONTEXT marks it discretionary |
| A3 | `_derive_launch_spec` should be skipped entirely for recipe-backed models | §1.6 | Low — worst case it is computed and unused |
| A4 | `info["size_gb"]` is an acceptable `weights_gb` despite summing all blobs | §1.4 | Medium — it can overstate for multi-revision caches, inflating derived util. Admission already trusts the same number, so this is consistent, not novel. Worth one sentence in the plan. |
| A5 | Emitting `spec["warnings"]` as script comments is better deferred to Phase 4 | §1.3 | Low — deferral is the CONTEXT's own position |

## Sources

### Primary (HIGH — executed or read on this box, 2026-08-05)
- `app.py` — all cited line numbers read directly (8134 lines)
- `/home/josh/spark-vllm-docker/run-recipe.py` (1442 lines) — `:194` required fields,
  `:484-503` render semantics, `:519-522` solo/Ray normalization, `:1039-1077` solo gating,
  `:1096`/`:1132`/`:1174`/`:1205`/`:1271` dry-run guards
- `/home/josh/spark-vllm-docker/recipes/*.yaml` — all 27 rendered and tokenized
- `tests/` — all 10 files; `310 passed in 0.74s`
- `profiles/vLLM/*.sh` — live profile shapes; `git status` / `git diff` for working-tree state
- `.planning/phases/03-curated-recipe-overrides/03-CODEX-SURVEY.md` — cited, not restated
- `.planning/phases/02-derived-launch-spec/VERIFICATION.md`, `02-MODEL-FIXTURES.json`
- `.planning/ROADMAP.md:102-119`, `.planning/PROJECT.md:37`, `.planning/STATE.md`,
  `.planning/HANDOFF.json`, `./CLAUDE.md`

### Executed verifications (reproducible)
- `command.format(**defaults)` over 27 recipes → 27/27 OK
- render + `shlex.split` flag extraction over 27 recipes → GB recipes yield only
  `--gpu-memory-utilization-gb`
- `_recipe_util("step-3.7-flash-fp8")` → `108.0`
- `_derive_launch_spec({"model_type":"qwen3","torch_dtype":"bfloat16"})` →
  `max_model_len=0`, `recommended_util=0.1`
- `_derive_launch_spec(<real Qwen3.6 config>, weights_gb=37.0, kv_dtype_bytes=1|2)` →
  util `0.42` / `0.44`

### Secondary (MEDIUM)
- None. No web search was performed: this phase is entirely about code and files on this
  machine, and no external library behavior is in question.

### Tertiary (LOW)
- None.

## Metadata

**Confidence breakdown:**
- Integration seam: HIGH — every line cited was read; the ordering trap was measured
- Recipe reading: HIGH — the proposed algorithm was executed against all 27 real recipes
- Capability map shape: MEDIUM — the *shape* is well-grounded in repo precedent, but the
  emission policy (OQ-1) is a product decision, not a technical finding
- Precedence: HIGH — the zero-derivation landmine was reproduced
- Validation architecture: HIGH — baseline measured; the "green suite proves nothing" claim
  was demonstrated, not asserted
- Regression risk: HIGH — grep-verified across the suite; working-tree state verified via git

**Research date:** 2026-08-05
**Valid until:** ~2026-09-05 for the code findings (they track `app.py`, which two other
sessions are editing — re-check line numbers before implementing). The recipe-file findings
are valid until `~/spark-vllm-docker/recipes/` changes; the committed-fixture drift test in
§Validation is the mechanism for noticing.
