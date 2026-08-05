# Phase 3: Curated recipe overrides - Context

**Gathered:** 2026-08-05
**Status:** Ready for planning
**Source:** Roadmap success criteria + Phase 2 carry-forwards + Phase 1.1 deferred decision,
with one gray area resolved directly with Josh (recipe source of truth).

<domain>
## Phase Boundary

Phase 2 delivered `_derive_launch_spec` — a pure, table-tested solver that turns a model's
`config.json` into correct context/KV/utilization numbers. It has **zero call sites outside
tests**, by design; Phase 2's VERIFICATION.md fences wiring out explicitly as Phase 3 scope.

Phase 3 does two things:

1. **Gives the generator a resolution chain.** Hand-measured recipe → derived spec →
   generic default, replacing today's hardcoded `--gpu-memory-utilization 0.75 /
   --max-model-len 32768 / --kv-cache-dtype fp8` for every non-gpt-oss model
   (`app.py` `_build_vllm_profile_script`, ~:3634-3665). This is `_derive_launch_spec`'s
   first application call site.
2. **Replaces substring guessing with an explicit capability map.** `if "qwen" in
   info["name"].lower()` currently hands `--tool-call-parser qwen3_coder` to
   Qwen2.5-14B-GPTQ, Qwen3-VL-4B and both DeepSeek-R1-Distill-Qwen profiles, none of which
   are Qwen3 Coder.

**In scope:** the recipe resolution layer, the capability map, conditional `--kv-cache-dtype`,
and generalizing the one-off recipe-backed profile shape into a real profile type.

**Out of scope (named so the planner does not drift into them):**
- Parameterized `${VLLM_MAX_MODEL_LEN:-…}` scripts and UI controls — Phase 4.
- Surfacing `spec["warnings"]` to a human — Phase 4 (Phase 2 carry-forward #1).
- Executor-budget admission and docker-label reclaim — Phase 5 (carry-forward #3).
- Editing the Phase 2 formula. See the locked decision below.
</domain>

<decisions>
## Implementation Decisions

### Recipe source of truth — LOCKED (Josh, 2026-08-05)

**The recipe YAMLs are the source of truth. `config.json` holds only the mapping.**

`/home/josh/spark-vllm-docker/recipes/` holds 27 hand-measured, known-good recipes for this
exact machine, each carrying its rationale in comments. The roadmap's original wording ("a
`vllm.recipes` block in config.json") was read as a literal value table; that was rejected
because re-typing `0.55 / 262144 / qwen3_xml` into `config.json` forks the source of truth —
the precise failure the Phase 1.1 decision already called out when it refused to inline the
recipe's flags into the Qwen3.6 profile.

So `vllm.recipes` maps **model-name pattern → recipe name**, and the values are parsed from
the YAML at generation time:

```json
"vllm": {
  "recipe_dir": "~/spark-vllm-docker/recipes",
  "recipes": {
    "Qwen/Qwen3.6-*": "qwen3.6-35b-a3b-fp8-solo",
    "openai/gpt-oss-*": "openai-gpt-oss-120b"
  }
}
```

Precedent exists: `app.py:3049` already parses `gpu_memory_utilization` out of a recipe YAML.

**Corollary — use `yaml.safe_load`.** `requirements.txt` already declares `pyyaml==6.0.2` as a
*runtime* dependency, so the hand-rolled regex parser at `app.py:3049` ("without a YAML
dependency") was solving a constraint that does not exist. Phase 3 should parse properly and
fold that function into the new reader rather than growing a second ad-hoc parser.

**Precedence idiom:** config → env → default, the same shape as the existing `alerts` block.
Recipe values win over derived values; derived values win over generic defaults. A missing
key in a recipe falls through to derived — "absent" must not be read as "zero".

### Do NOT close the Qwen3.6 0.42-vs-0.55 gap by editing the formula — LOCKED

Phase 2 derives 0.42 for Qwen3.6; the hand-measured recipe says 0.55. Roadmap SC2 requires
0.55. That gap closes **because the recipe wins**, not because the solver is retuned.
`tests/test_launch_spec.py::test_calibration_qwen36_gap_is_left_for_phase_3` pins this and its
docstring forbids a formula edit. The qwen3-next-80b calibration (derived 0.54 vs measured
0.55) is the evidence the formula is sound; changing it to chase Qwen3.6 would break the one
number that validates the whole approach.

### Tool-parser selection comes from an explicit capability map

Keyed on something durable — `architectures[]` from `config.json` and/or the model id — not on
a substring of a display name. The map must state, per entry, both `--tool-call-parser` and
`--reasoning-parser`, and must be able to say **"this model does not support tool calling"**,
which the current code cannot express. A model with no map entry gets no tool flags rather
than a guessed parser: emitting no flag degrades to a working server, emitting the wrong
parser produces silently corrupted tool calls.

### `--kv-cache-dtype fp8` only where declared

Today it is blanket for every non-gpt-oss model. It becomes opt-in: applied when the recipe's
command block declares it, or when the model's own config evidences it. gpt-oss keeps
full-precision KV and 65536 (roadmap SC3) — that behavior already exists and must survive the
refactor rather than being re-derived.

### Generalize the recipe-backed profile type (deferred here from Phase 1.1)

`start_hf_qwen_qwen3.6-35b-a3b-fp8.sh` is currently a hand-written one-off that `docker rm -f`s
and delegates to `run-recipe.sh <recipe> -d`. The Phase 1.1 decision record explicitly says
"Generalize into a recipe-backed profile type in Phase 3."

The generator cannot reproduce a recipe's in-container `mods/` (e.g.
`fix-qwen3.6-chat-template`) or flags it does not emit (`--load-format fastsafetensors`,
`--attention-backend flashinfer`, `--reasoning-parser qwen3`). **Therefore: when a model maps
to a recipe, the correct generated artifact is a delegating wrapper, not a reconstructed
`docker run`.** Reconstruction would silently drop the mods. The derived path stays for models
with no recipe.

The existing preflight already understands both shapes (`_PF_RECIPE_RE`, `run-recipe.py
--dry-run` for recipe profiles vs. the argparse probe for generated ones) — Phase 3 should fit
that contract, not change it.

### Constraints the Codex survey established (all spot-checked against disk)

These are findings, not opinions; each was independently verified before being written here.

**1. `--kv-cache-dtype fp8` — recipe-wins and roadmap SC3 collide on gpt-oss.**
`openai-gpt-oss-120b.yaml` sets `--kv-cache-dtype fp8`, but the hand-corrected, known-good
`start_hf_openai_gpt-oss-120b.sh` deliberately uses full-precision KV and documents why.
Roadmap SC3 requires gpt-oss keep full-precision KV. So "the recipe always wins" would
**regress SC3**.

*Resolution to plan against:* `vllm.recipes` is an **opt-in allowlist**. Only models Josh
explicitly maps become recipe-backed; gpt-oss is simply not mapped and keeps its known-good
generated path. This satisfies SC1 and SC3 without inventing per-field exception machinery.
If the planner sees a better resolution, it must state why SC3 still holds.

**2. A recipe's `model:` field cannot be the lookup key.** All three of
`qwen3.6-35b-a3b-fp8.yaml`, `-solo.yaml` and `-dflash.yaml` declare the identical
`model: Qwen/Qwen3.6-35B-A3B-FP8` (verified), yet are materially different launches (solo vs
clustered vs speculative-decode). Auto-discovery by `model:` is ambiguous by construction —
which is exactly why the mapping must be an explicit human choice in `config.json`. The
solo recipe is the correct one for this box.

**3. `gpu_memory_utilization` is not always a utilization fraction.** In 25 recipes it is a
fraction; in two it is a **GB count** (e.g. `108`) feeding a patched
`--gpu-memory-utilization-gb`. A reader that lifts the key and passes it to
`--gpu-memory-utilization` would request 108× the pool. **Read the flag from the `command`
block, not the key name from `defaults`.**

**4. Do not validate parsers against a closed enum.** The host has vLLM `0.21.0`, but the
containers run `v0.20.0`, `0.23.1` and `cu130-nightly` — the host registry is **not
authoritative** for what a container accepts. Further, `nano_v3` and `super_v3` are
plugin-provided (`--reasoning-parser-plugin`), so they are legitimately absent from any
built-in registry. The capability map must carry values, not police them.

**5. SC4's victim list is incomplete — there are five, not three.** Beyond Qwen2.5, Qwen-VL
and the DeepSeek-R1-Distill-Qwen pair, `lyf/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4`
also receives `qwen3_coder` today. Its `chat_template.jinja` is **md5-identical** to the
recipe-backed Qwen3.6 FP8 model's (`52b6d51a…`, verified), so it should get `qwen3_xml` /
`qwen3`. Plans must cover it.

**6. Honest uncertainty is required in the map.** `hermes` for the Qwen3 dense models,
Qwen2.5, Qwen3-Next and Qwen3-VL is *inferred from chat-template protocol*, not confirmed by
any known-good recipe. Tool-calling support for the DeepSeek-R1 distills is unproven — their
templates render tool results but never inject tool definitions. The map should distinguish
recipe-proven entries from template-inferred ones, and unproven rows should emit **no tool
flags** rather than an inferred parser.

**7. Blanket fp8 KV is unproven far more widely than SC5 implies.** Only
`nemotron-3-super` and `qwen3-next-80b-a3b-nvfp4` declare a KV scheme in their own config.
Every other fp8-KV flag on the box traces to a recipe or to the generator's own blanket
default — including Qwen2.5-14B-GPTQ, qwen3-vl-4b, qwen3-coder-next and both DeepSeek
distills, where it is profile-only and unproven.

**8. A pre-existing conflict, not Phase 3's to resolve.** `start_nemotron_super.sh` uses
`--reasoning-parser super_v3` via a local plugin; `nemotron-3-super-nvfp4.yaml` uses
`nemotron_v3`. Note it, do not silently pick a side.

### Claude's Discretion

- Pattern-matching mechanism for `vllm.recipes` keys (glob vs prefix vs regex) and how
  ambiguity between two matching patterns is resolved — but it must be deterministic and
  tested, since a silent wrong-recipe match is worse than no match.
- Where the capability map physically lives (module constant vs `config.json` vs a tracked
  JSON file beside `recommendations.json`).
- Internal function decomposition and naming.
- Whether the recipe reader returns a normalized dataclass/dict or raw YAML.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase inputs
- `.planning/ROADMAP.md` — Phase 3 section; the five success criteria are the contract.
- `.planning/phases/02-derived-launch-spec/VERIFICATION.md` — carry-forwards #1/#2/#4 and the
  explicit statement that wiring is Phase 3 scope.
- `.planning/phases/02-derived-launch-spec/02-RESEARCH.md`,
  `02-MODEL-FIXTURES.json` — the model configs the solver was table-tested against.
- `.planning/HANDOFF.json` — the Phase 1.1 decision records, especially the recipe-delegation
  and no-`--restart` rulings.

### Code under change
- `app.py` `_build_vllm_profile_script` (~:3560-3700) — the hardcoded flags and the
  `"qwen" in name` substring test.
- `app.py` `_derive_launch_spec` (:948-1074) — Phase 2's 16-key public entry point.
- `app.py` `_recipe_util` (~:3049) — the existing ad-hoc recipe YAML parser to subsume.
- `app.py` `_PF_RECIPE_RE` / preflight (~:2716-2990) — the recipe-profile contract to fit.

### External source of truth
- `/home/josh/spark-vllm-docker/recipes/*.yaml` — 27 hand-measured recipes.
- `/home/josh/spark-vllm-docker/run-recipe.py` — how recipe fields are actually consumed.

### Survey
- `.planning/phases/03-curated-recipe-overrides/03-CODEX-SURVEY.md` — Codex read-only survey
  of recipe fields, recipe↔model coverage, parser capability evidence, and kv-dtype ground
  truth. **Facts only; it was explicitly forbidden from proposing a design.**
</canonical_refs>

<specifics>
## Specific Ideas

**Roadmap success criteria, verbatim — these are the acceptance contract:**
1. A `vllm.recipes` block in config.json, keyed by model-name pattern, overrides derived
   values, following the same config→env→default precedence idiom as the `alerts` block.
2. Qwen3.6 resolves to the measured 0.55 / 262144 / `qwen3_xml`.
3. gpt-oss keeps 65536 and full-precision KV.
4. Tool-parser selection comes from an explicit capability map, so Qwen2.5, Qwen-VL and
   DeepSeek-R1-Distill-Qwen no longer receive `qwen3_coder`.
5. `--kv-cache-dtype fp8` is applied only where declared, not blanket.

**The four profiles that prove SC4** (all currently match `"qwen" in name.lower()`):
- `start_hf_qwen2.5-14b-instruct-gptq-int8.sh`
- `start_hf_qwen3-vl-4b-fp8.sh`
- `start_hf_deepseek-ai_deepseek-r1-distill-qwen-14b.sh`
- `start_hf_deepseek-ai_deepseek-r1-distill-qwen-32b.sh`

**Testing.** 310 tests currently pass. `tests/test_vllm_profile_generation.py` and
`tests/test_launch_spec.py` are the closest analogs for new tests. Generation tests assert on
the *script text*, so recipe resolution can be verified without launching anything — keep it
that way; Phase 3 should remain testable with vLLM up and untouched.

**Deployment reality.** Editing `app.py` does not restart the running service;
`systemctl --user restart dgx-model-manager.service` is required. vLLM is currently UP serving
Qwen3.6 at util 0.55 — plans must not regenerate or overwrite the live Qwen3.6 profile
without an explicit, reversible task.

**Repo hygiene.** ~3 concurrent Claude sessions run in this repo. Commit by explicit path;
never `git add -A`.
</specifics>

<deferred>
## Deferred Ideas

- `${VLLM_MAX_MODEL_LEN:-…}` parameterization and UI context/util controls — Phase 4.
- Consuming `spec["warnings"]` in the UI — Phase 4 (carry-forward #1).
- Executor-budget admission, docker-label reclaim, launch lock — Phase 5.
- The unresolved `VLLM_USE_FLASHINFER_MOE_FP4` disagreement between the generator and a peer
  session's edit to `start_hf_openai_gpt-oss-120b.sh` (vLLM 0.23.1 logs it as unknown). Listed
  in HANDOFF.json `human_actions_pending`; **not** Phase 3 scope, but the planner should avoid
  entrenching the env var further while it is disputed.
- Optional per-field inline override on top of a YAML-backed recipe — considered and set aside
  as more precedence surface than this phase needs.

</deferred>

---

*Phase: 03-curated-recipe-overrides*
*Context gathered: 2026-08-05 — roadmap + carry-forwards, one gray area resolved with Josh*
