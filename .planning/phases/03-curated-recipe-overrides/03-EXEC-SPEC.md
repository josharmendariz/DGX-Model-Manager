# Phase 3 — Execution Spec (2026-08-15)

Supersedes line numbers and open questions in 03-RESEARCH.md (research was written
against an 8134-line app.py; the live file is 8448 lines). All anchors below were
re-verified against the live worktree on 2026-08-15.

## Goal

Deliver REQ-05 / Roadmap SC1–SC5:
1. `vllm.recipes` block in config.json (pattern -> recipe name) overrides derived
   values; absent/empty block is a no-op (SC1).
2. Qwen3.6 resolves to measured 0.55 / 262144 / qwen3_xml (SC2, via delegating
   wrapper + reader resolution — two separate assertions).
3. gpt-oss keeps 65536 + full-precision KV (SC3; `is_gpt_oss` branch survives verbatim).
4. Tool parser from an explicit capability map; the five "qwen" substring victims
   (Qwen2.5-14B-GPTQ, qwen3-vl-4b, DeepSeek-R1-Distill-Qwen 14B/32B, lyf NVFP4) no
   longer receive qwen3_coder (SC4).
5. `--kv-cache-dtype fp8` only where recipe or model-config evidence says so (SC5).

## Open questions — resolved on research recommendations (log for the report)

- OQ-1: conservative emission. `emit` defaults false; only `confidence` in
  {"recipe-proven","template-identical-to-recipe-proven"} emit, overridable per entry
  via explicit `"emit": true|false`. Template-inferred rows (hermes) carry values but
  emit nothing until Josh promotes them by JSON edit.
- OQ-2: distinct `confidence: "template-identical-to-recipe-proven"`, emit true,
  evidence records the md5 match.
- OQ-3: Nemotron super_v3 vs nemotron_v3 — carry both in evidence; no side picked;
  touch neither artifact.
- OQ-4: regenerate 02-MODEL-FIXTURES.json with an `architectures` column (Wave 0).
- OQ-5: generated wrapper is SELF-CONTAINED — inlines recipe dir + name as a top-of-file literal (`RECIPE_DIR=… RECIPE=…`, matching the live hand-written wrapper and what _parse_launch_script/_PF_RECIPE_RE expect); NO external env dependency. (Clarification: the earlier 'no RECIPE= shell vars' was loose wording.)

## Locked (inherited, non-negotiable)

Lock L1..L9 = 03-RESEARCH.md "Locked Decisions" 1–9, all still in force.
Key: recipe YAML = source of truth; yaml.safe_load; no formula edits (the pinned
test test_calibration_qwen36_gap_is_left_for_phase_3 MUST stay green); recipes are
an opt-in allowlist; recipe-backed = delegating wrapper; capability map keyed on
durable keys, no enum validation (grammar + shlex.quote only); absent != zero
(resolver tests `is None`, never truthiness).

## Live anchors (app.py, 8448 lines, verified 2026-08-15)

- `_derive_launch_spec`          : 958
- `_load_recommendations`        : 1918  (mirror this for `_load_model_capabilities`)
- `_vllm_budget_gib`             : 2850
- `_preflight_memory`            : 3119  (call site for recipe util, ~:3300 region)
- `_recipe_util`                 : 3317  (to be REPLACED, keep call-site contract)
- `_safe_profile_slug`           : 3675
- `_profile_model_info`          : 3712  (additive: expose `architectures`)
- `_container_model_mount`       : 3764
- `_MOE_BACKEND_RE` (grammar)    : 3804  (precedent for parser-name grammar)
- `_one_line`                    : 3817
- `_vllm_serve_command`          : 3822
- `_build_vllm_profile_script`   : 3850  (THE function being decomposed)
- `_create_vllm_profile_from_path`: 3974 (the 409 overwrite guard lives here)
- `tests/conftest.py` MODEL_FIXTURES_PATH -> .planning/phases/02-derived-launch-spec/02-MODEL-FIXTURES.json
- recipe reader's ground truth: /home/josh/spark-vllm-docker/run-recipe.py:484-503
  (params = {**defaults, **overrides}; command = command.format(**params)); :502-503
  KeyError handling; :194 required fields; :1057 cluster_only gating.

## Structural plan (from RESEARCH §1, restated as build order)

The whole risk is the composition order inside `_build_vllm_profile_script`.
Decompose:

    _build_vllm_profile_script(launch_dir, model_name)
      ├─ info     = _profile_model_info(...)               # unchanged + architectures
      ├─ resolved = _resolve_launch(config, info, vllm_cfg)# NEW, PURE -> decision dict
      ├─ preamble = _script_preamble(info, ...)            # shared: header + set -euo + docker rm -f
      └─ body     = _recipe_body(resolved) | _docker_run_body(info, resolved)

Constraints (tests must point exactly here):
  C1. preamble is the ONLY writer of `# Name:` / `# Description:` / `# VRAM:`.
  C2. `docker rm -f vllm_node` lives in the preamble (both shapes).
  C3. Neither shape emits `--restart`.
  C4. `_resolve_launch` is pure (no fs/network) so SCs are provable statically.

### The five-step order (THE ordering trap, RESEARCH §1.5) — _resolve_launch must do:
  1. resolve recipe (if the model maps to one)          -> if recipe body, STOP (return wrapper)
  2. decide kv_cache_dtype: rendered recipe flag > model-config evidence > omit
  3. _derive_launch_spec(..., kv_dtype_bytes = 1 if fp8 else 2)
  4. apply recipe-derived overrides on top of the spec (absent -> fall through to derived)
  5. emit flags (per-field precedence: recipe > derived[if usable] > default)

  KV dtype BEFORE derivation is the single most important line in this phase.

### Zero-derivation floor (RESEARCH §4.3)
  If spec.max_model_len <= 0 -> treat as "no derived opinion" -> use 32768 (65536 gpt-oss).
  If spec.recommended_util <= util_floor(0.10) -> use 0.75. These protect the
  degenerate-config path from emitting `--max-model-len 0 --gpu-memory-utilization 0.1`.

### gpt-oss branch (SC3)
  The `is_gpt_oss` branch (65536 + full-precision KV + its own env block) must survive
  VERBATIM. It is never recipe-backed in the shipped config, so it simply never enters
  the recipe path. Do not re-derive its numbers.

## Recipe reader (W0/B: new function `_read_recipe(recipe_dir, name)` in app.py)

Algorithm (RESEARCH §2.1 — the single most important design choice):
  1. guard: `re.fullmatch(r"[\w.-]+", name)` (keep `_recipe_util`'s grammar; name comes
     from config.json and is interpolated into a path AND a generated shell script).
  2. path = Path(recipe_dir) / f"{name}.yaml"; missing dir/file -> return None (+ warning).
  3. `yaml.safe_load` only; non-dict result -> None (e.g. bare string).
  4. `command:` missing -> None (unusable; run-recipe.py:194 requires it).
  5. render EXACTLY as run-recipe.py:484-503:
         params = {**recipe.get("defaults", {}), **{}}   # no CLI overrides at read time
         try:   rendered = recipe["command"].format(**params)
         except (KeyError, IndexError, ValueError):  # missing/stray placeholder
             -> None + warning naming the placeholder (mirrors run-recipe.py:502-503)
  6. `shlex.split(rendered)` -> flag/value pairs. Extract by FLAG NAME:
         --gpu-memory-utilization   -> gpu_memory_utilization   (float, may be None)
         --gpu-memory-utilization-gb-> gpu_memory_utilization_gb (float; preflight must
                                       NOT read it as a fraction)
         --max-model-len, --kv-cache-dtype, --tool-call-parser,
         --reasoning-parser, --port
  7. NO magnitude heuristic. The flag name is ground truth; 0.55 is a fraction, 108 is GB.

Normalized record (all absent fields = None, never 0/""):
  {name, description, model, gpu_memory_utilization, gpu_memory_utilization_gb,
   max_model_len, kv_cache_dtype, tool_call_parser, reasoning_parser, port,
   solo_only, cluster_only, mods}
Every failure degrades to "no opinion" (None record + warning list), never raises —
the reader is reached from an HTTP route AND the auto-profile step of an HF download
(RESEARCH §2.4).

Subsume `_recipe_util` (3317): keep the publicly visible call-site contract
(preflight wants "a fraction or None"), reimplement as
`return _read_recipe(...).get("gpu_memory_utilization")`. This closes the latent
bug: _recipe_util("step-3.7-flash-fp8") currently returns 108.0 (WRONG — it is GiB).
New regression test asserts None-or-fraction, never 108.0-as-fraction.

## vllm.recipes config (SC1)

config.json shape (add to the existing `vllm` block; read ONCE at import like the rest
of the vllm block, OR follow the per-call idiom — follow what `_vllm_serve_command`'s
`cfg` arg already does: it is passed down, so plumbing vllm_cfg into the resolver is
the existing pattern):
  "vllm": {
    "recipe_dir": "~/spark-vllm-docker/recipes",
    "recipes": { "Qwen/Qwen3.6-*": "qwen3.6-35b-a3b-fp8-solo", ... }
  }
precedence: config.json `vllm.recipe_dir` -> default `~/spark-vllm-docker/recipes`. (Correction: no env override — no other `vllm` block key has one, so adding it here would be unexplained surface. The alerts env idiom named here does not apply to the vllm block.)
MATCHING: fnmatch.fnmatchcase(model_name.lower(), pattern.lower()); most-specific wins
(fewer wildcards, then longer literal, then lexicographic) + WARNING when >1 pattern
matches. Shipped config.example.json maps ONLY Qwen3.6 (gpt-oss NOT mapped — SC3).
ABSENT/EMPTY vllm.recipes must be a byte-for-byte no-op for every existing model.

## Capability map (03-02)

- FILE: `model_capabilities.json` beside recommendations.json (repo root). Loader
  `_load_model_capabilities()` mirrors `_load_recommendations` (1918) line-for-line:
  load per call, try/except log + empty-skeleton fallback, no restart needed.
- META: meta.schema_version, plus a top-level `emission_policy` documenting the rule:
  emit iff confidence in {"recipe-proven","template-identical-to-recipe-proven"}
  UNLESS the entry carries explicit "emit": true|false.
- ENTRY KEY GROUNDING = info["name"] (owner/repo for HF repos, bare dir for flat)
  AND architectures[] (lowercased). Lookup order:
  exact model id (casefold) -> architectures[0] (casefold) -> no entry.
- ENTRY FIELDS: matches (list of ids/architectures), tool_call_parser, reasoning_parser,
  supports_tool_calling (true|false|null), confidence, evidence, emit (optional bool).
- EMISSION (in _build_vllm_profile_script, REPLACES `if "qwen" in name.lower()`):
    entry = lookup(info)
    if entry and entry emittable:
        emit --enable-auto-tool-choice (always when tool_call_parser present)
        emit --tool-call-parser <val>  if val
        emit --reasoning-parser <val>  if val
    -> nothing. No entry / not emittable -> NO tool flags (strictly better than today's
       wrong qwen3_coder for all five SC4 victims).
- NAME GRAMMAR: parser values must match `^[a-z0-9_.-]{1,64}$` (LOOSER than
  _MOE_BACKEND_RE at 3804 to admit "granite-20b-fc"/"step3p5"/"nano_v3" — all
  legit) then shlex.quote. Never an enum check.
- SEMANTICS: emit --tool-call-parser only when the entry provides it;
  supports_tool_calling:false is a positive finding, distinct from absent.

### Ship rows (from 03-CODEX-SURVEY.md §4, verbatim evidence):
1. Qwen/Qwen3.6-35B-A3B-FP8              arch Qwen3_5MoeForConditionalGeneration
   tool qwen3_xml, reasoning qwen3, supports true, confidence recipe-proven, emit true
   (3 exact-model recipes set both parsers).
2. lyf/Qwen3.6-35B-A3B-Uncensored-…-NVFP4  (both name forms: "lyf/…" and bare
   "lyf_qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive-nvfp4")
   arch Qwen3_5MoeForConditionalGeneration, qwen3_xml/qwen3, supports true,
   confidence template-identical-to-recipe-proven (chat_template.jinja md5-identical
   to row 1: 52b6d51a…), emit true.  <- the 5th SC4 victim
3. Qwen/Qwen3-1.7B | Qwen/Qwen3-8B | Qwen/Qwen3-14B  arch Qwen3ForCausalLM
   tool hermes, reasoning qwen3, supports true, confidence template-inferred,
   emit FALSE (OQ-1 conservative; promotion = JSON edit).
4. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4  arch NemotronHForCausalLM
   tool qwen3_coder, reasoning nano_v3, supports true, recipe-proven, emit true.
5. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  arch NemotronHForCausalLM
   tool qwen3_coder, reasoning nemotron_v3, recipe-proven, emit true,
   evidence: "profile start_nemotron_super.sh uses super_v3 via local plugin
   super_v3_reasoning_parser.py — conflict recorded, not resolved (OQ-3)".
6. openai/gpt-oss-120b  arch GptOssForCausalLM  tool openai, reasoning openai_gptoss,
   recipe-proven, emit true. (Harmless: gpt-oss never maps to a recipe; if a
   gpt-oss snapshot is ever profiled generated-style, it gets the right parsers.)
7. qwen3-coder-next-nvfp4  (flat dir name)  arch Qwen3NextForCausalLM
   tool qwen3_coder, reasoning null, recipe-proven, emit true.
8. qwen2.5-14b-instruct-gptq-int8  (flat)  arch Qwen2ForCausalLM  tool hermes,
   reasoning null, template-inferred, emit FALSE.   <- SC4 victim: no flags
9. qwen3-next-80b-a3b-nvfp4  (flat)  arch Qwen3NextForCausalLM  tool hermes,
   reasoning null, template-inferred, emit FALSE.
10. qwen3-vl-4b-fp8  (flat; also "Qwen/qwen3-vl-4b-fp8" form if it appears as
    owner/repo)  arch Qwen3VLForConditionalGeneration  tool hermes, reasoning null,
    template-inferred, emit FALSE.   <- SC4 victim
11. deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  arch Qwen2ForCausalLM  tool null,
    reasoning deepseek_r1, supports null (uncertain), template-inferred, emit FALSE.
12. deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  (as 11)   <- SC4 victims: NO flags
13. Systran/faster-whisper-base | Systran/faster-whisper-medium  arch null
    supports_tool_calling FALSE, everything null, emit FALSE — the
    "positive finding: no tool support" row.

  NOTE for the subagent: the emitted name for flat-dir models is the bare dir name
  exactly as it appears on this box; include BOTH bare and owner/repo forms where
  both appear in the survey (casefold match).

## Test map (contract — every row becomes a test; RESEARCH §Validation)

TEST SUITE INVARIANT: zero existing tests encode the old hardcoded values, so a
WIRED-BUT-WRONG implementation (e.g. `--max-model-len 0 --gpu-memory-utilization 0.1`
for every model) keeps the whole suite green. The new tests ARE the phase's signal;
no old test may be deleted or weakened, and `test_calibration_qwen36_gap_is_left_for_phase_3`
is sacred.

W0 fixtures (before any coverage test):
  F1. tests/fixtures/recipes/ — byte-identical copies of 5 recipe YAMLs:
      qwen3.6-35b-a3b-fp8-solo.yaml (SC2), openai-gpt-oss-120b.yaml (SC3
      recipe-vs-profile conflict), qwen3.6-35b-a3b-fp8-dflash.yaml ({{brace}}
      escape), step-3.7-flash-fp8.yaml (GB trap), deepseek-v4-flash.yaml
      (cluster_only=true example). Copy via `cp`, verify with md5sum; the reader
      must take recipe_dir as a PARAMETER so tests point at fixtures (a required arg; a hidden default would just relocate the magic path).
  F2. tests/fixtures/recipes/DRIFT test: if ~/spark-vllm-docker/recipes exists,
      assert each committed fixture md5sums equal its live counterpart, else skip.
  F3. REGENERATE .planning/phases/02-derived-launch-spec/02-MODEL-FIXTURES.json:
      add an `architectures` column (array) to every row from the exact survey §4
      values (both whisper rows: null). Keep all existing columns/rows; the
      conftest fixture `model_fixtures`/`fixture_models` must still pass. This is
      the deliberate act the conftest comment requires; do NOT glob live caches.

SC1  tests/test_recipe_resolution.py (new):
  t1  vllm.recipes present + matching model -> resolved recipe name.
  t2  absent block AND empty block -> resolver output identical to today's
      generator output for 3+ representative models (extend
      test_defaults_unchanged_when_no_vllm_config, don't replace it).
  t3  collision: two overlapping patterns ("Qwen/*" vs "Qwen/Qwen3.6-*") -> the
      more specific wins; shuffling dict insertion order does not change winner;
      a warning is emitted on multi-match.
  t4  gpt-oss style model not in map -> nothing matches.
  t5  recipe name failing [\w.-]+ grammar -> rejected, no fs touched, no raise.
  t6  cluster_only mapped recipe -> generate-to-memory returns warning
      (not an exception), wrapper still produced.
  t7  rendered --port != 8000 -> warning same treatment.

SC2  tests/test_recipe_reader.py (new) + extensions:
  t8  qwen3.6-35b-a3b-fp8-solo fixture -> record has util 0.55, max_model_len
      262144, tool qwen3_xml, reasoning qwen3, kv fp8 (assert each field == value,
      no substring games).
  t9  step-3.7-flash-fp8 / qwen3.5-397b-int4-autoround fixtures ->
      gpu_memory_utilization_gb == 108, gpu_memory_utilization is None.
  t10 GB trap at the call site: the `_recipe_util`-successor returns None (or a
      fraction), never 108.0, for the two GB recipes.
  t11 dflash fixture renders its speculative-config JSON single-quoted blob intact
      (shlex gives one token).
  t12 degradation matrix (parametrized, none may raise): missing dir, missing file,
      malformed yaml, yaml->str, missing command:, format KeyError, shlex ValueError.
  t13 generation: model mapped to qwen3.6 recipe -> emitted script text names the
      recipe, contains `run-recipe.sh`, `docker rm -f vllm_node`, and does NOT
      contain `--restart`, `--model`, `--max-model-len`, `--gpu-memory-utilization`.
  t14 the generated wrapper passes the existing preflight parser:
      _parse_launch_script(script) -> recipe_backed True + right recipe name.
  t15 preflight on the generated wrapper gives verdict ok or warn (never fail) and
      its memory budget (if util known) derives from 0.55.

SC3  tests/test_vllm_profile_generation.py (extend):
  t16 gpt-oss fixture config -> emitted script has `--max-model-len 65536` and NO
      `--kv-cache-dtype` line, and the env block (TIKTOKEN/VLLM_FLASHINFER_
      ALLREDUCE/VLLM_MARLIN_USE_ATOMIC_ADD) is unchanged.

SC4  tests/test_capability_map.py (new):
  t17 parametrized over the 5 victims' info dicts (from the regenerated fixtures):
      NONE of the emitted scripts contains `qwen3_coder`; DeepSeek pair and
      qwen2.5/qwen3-vl contain no --tool-call-parser at all; lyf NVFP4 contains
      qwen3_xml (promotion case, emit true).
  t18 model with no map entry AND no recipe -> no --tool-call-parser, no
      --enable-auto-tool-choice, no --reasoning-parser.
  t19 supports_tool_calling:false row (whisper) -> no tool flags and the NO-tool-
      support distinction is preserved in the record (not collapsed to null).
  t20 map lookup: exact-id hit beats arch hit beats nothing; casefolded.

SC5  (in test_capability_map.py + test_recipe_resolution.py):
  t21 kv dtype decided BEFORE derivation: same model config, kv_dtype_bytes=1 vs 2
      produces different recommended_util (use real Qwen3.6 config from fixtures:
      0.42 vs 0.44 with weights_gb=37.0) and the EMITTED util matches the EMITTED
      dtype (if script has no --kv-cache-dtype, flag values come from the
      kv_dtype_bytes=2 derivation).
  t22 negative: a non-recipe, no-kv-evidence model (e.g. qwen2.5 fixture) emits
      NO --kv-cache-dtype line at all.

GUARDS:
  t23 degenerate config ({"model_type":"qwen3","torch_dtype":"bfloat16"}) ->
      emitted max-model-len is the 32768 default, util is 0.75 default; NEVER 0 / 0.1.
  t24 every generated script (both shapes) passes bash -n (helper exists at
      tests/test_vllm_profile_generation.py:~116).
  t25 NO test in the whole suite may now emit/ assert old blanket behaviors:
      grep test files after the fact for `--kv-cache-dtype fp8` and confirm each
      occurrence is either the SC5 positive case (t15 scenario / recipe record) or
      the preflight sample-script string family (test_launch_preflight.py:33-34,62 —
      which is a hand-written sample string and stays untouched).

## Untouchable list (regression minefield, RESEARCH §6.2)

- Do NOT regenerate any file under profiles/vLLM/ as a side effect. All generation
  tests run against in-memory / tmp_path scripts.
- start_hf_openai_gpt-oss-120b.sh is modified-uncommitted in the USER's tree (that
  tree only — the worktree is clean); do not touch profiles/ at all in the worktree.
- Live file start_hf_qwen_qwen3.6-35b-a3b-fp8.sh: LATER, separately, generate the
  wrapper to /tmp and diff; adoption is an explicitly-reversible human task.
- start_nemotron_super.sh: leave alone (OQ-3).

## Waves

  W0 (subagent A, Codex): F1-F3 fixtures only + their tests (drift, conftest compat).
  W1 (subagent B, Codex): reader + resolution + generator wiring + t1-t16,t21-t25
      (03-01). TDD: write failing tests first where they don't exist.
  W2 (subagent C, Codex): model_capabilities.json + loader + emission + t17-t24 map
      rows (03-02).
  W3 (me): preflight start-gate JS (block on verdict=fail, force override, keep
      Dry Run button) + full suite + bash -n over all profiles + Codex adversarial
      review of the diff (send code+tests+writeup; require rank-ordered findings and
      an explicit disagreements section).

ACCEPTANCE GATE (phase complete): full suite green (baseline 328 + new tests),
bash -n clean on all profile scripts, SC1-SC5 each with named passing tests,
zero changes under profiles/, the pinned calibration test green, and the two
read-only live checks passed: POST /api/vllm/preflight on the live Qwen3.6
profile (service restarted under approval) and run-recipe.sh
qwen3.6-35b-a3b-fp8-solo --dry-run.
