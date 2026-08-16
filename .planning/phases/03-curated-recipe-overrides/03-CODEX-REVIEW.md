# 03 — Codex adversarial review of 5214fb7..fc8f084 (2026-08-16, gpt-5.6-sol medium)

The review prompt (`/tmp/opencode/w3/prompt.md`, read-only sandbox) named 11 headline
claims and demanded each be falsified with `file:line`. Verdict: **11/11 HOLDS**, each
cited. It then named 5 "disagreements" + 1 "risk". Adjudication (checked against the
code and the real on-disk configs, not the reviewer's framing):

## Findings and what happened to each

| # | Claim | Adjudicated as | Disposition |
|---|-------|----------------|-------------|
| D1 | preflight readers hardcode `~/spark-vllm-docker`, ignoring configured `vllm.recipe_dir` — a custom dir would launch one recipe while Dry Run validates another | **REAL** (introduced by W1 exposing the knob to the generator only; base 5214fb7 hardcoded this and had no `recipe_dir` concept at all — app.py:3226-3235) | Fixed: `_vllm_recipe_dir()`/`_recipe_dirs()` single resolver drives `_recipe_util` + `_preflight_smoke`; generator stays pure on its `vllm_cfg` arg. Invariant pinned (`test_recipe_dir_is_one_truth_for_generator_and_preflight`). |
| D2 | `_resolve_launch` is not "pure" (reads YAML) | **Spec wording wrong, code right** — the same spec section requires the recipe record to come back in the result | Spec line corrected. |
| D3 | OQ-5 "no `RECIPE=` shell vars" but the wrapper emits them | **Spec wording wrong, code right** — the live hand-written wrapper and `_PF_RECIPE_RE`/`_expand_script_vars` expect top-of-file literals; intent = no external env dependency, which holds | Spec line corrected. |
| D4 | `_read_recipe` should default `recipe_dir` | **Rejected** — a required param is the cleaner design; a default would relocate the magic path (the whole point of D1's fix) | Spec line corrected; no code change. |
| D5 | config→env→default override idiom as in alerts | **Rejected** — no other `vllm` block key has an env override; unexplained surface. (Reviewer was citing *my* spec's borrowed idiom.) | Spec line corrected; no code change. |
| R | `kv_cache_dtype: "fp8_e4m3"` not recognized by exact-match | **Refuted for this fleet** — parked as an open note | See below. |

## What I'd do differently (reviewer-quality notes, kept per "record what a
review changed, including what was rejected and why")

- **The 11/11 did not find my own later bug**: the second review round (0aa7be0) is
  not from Codex — it's from *running the new pipeline in-process against the live
  wrapper profile*, which caught `NameError: recipe_dir` in `_preflight_smoke`
  (a renamed local I left at the subprocess `cwd=`). **The 378-test suite was
  green the whole time.** Lesson: the preflight suite is pure-function by design,
  so the recipe-smoke subprocess branch had *zero* coverage. Two tmp-checkout fake-
  runner tests now exist and were mutated to fail on the exact `NameError` before
  being kept. **This is the strongest argument in the portfolio for making
  "run the new code against the real profile in-process" a standing pre-PR step
  for preflight-touching changes, not an optional live check.
  Dry-rehearsal in `/tmp/opencode/deploy-rehearsal` confirmed the deploy shape: load
  app.py + model_capabilities.json + the **production-shaped** `vllm.recipes` map, then
  (a) Qwen3.6 resolves to the recipe wrapper, (b) the LIVE wrapper profile runs the full
  preflight chain -> verdict **warn** (page-cache + existing-container only; smoke ok,
  i.e. the real `run-recipe.sh --dry-run` fired), (c) an unmapped model still derives plain
  docker (opt-in holds). `pyflakes` on the whole file: **zero undefined names** — the bug
  class behind the `NameError` is closed file-wide, not just at that one spot. (The two
  pyflakes style hints, unused import + f-string, also exist at base 5214fb7 — pre-existing,**

## RISK item, settled on this box (parked for the fleet)

Ran `_kv_dtype_from_config` on every on-disk config with a row in the fixtures:

| model | kv_cache_dtype (top) | q.kv_cache_dtype | q.kv_cache_scheme | -> |
|-------|----------------------|------------------|-------------------|----|
| Nemotron-Super NVFP4 | None | None | `{num_bits:8, type:float}` | **fp8** |
| qwen3-next-80b NVFP4 | None | None | `{num_bits:8, type:float}` | **fp8** |
| Nemotron-Nano NVFP4 | None | None | None | None |
| Qwen3.6-35B FP8 (weights only) | None | None | None | None |
| gpt-oss-120b (MXFP4) | None | None | None | None |
| qwen2.5-14b GPTQ / qwen3-vl FP8 / Qwen3-1.7B | None | None | none/FP8-weights | None |

Matches research §5 exactly: no fleet model uses a literal `kv_cache_dtype`; the
two 8-bit-KV models declare it via `quantization_config.kv_cache_scheme`, which the
reader handles (app.py `_kv_dtype_from_config`, scheme branch). A *future* model
shipping literal `kv_cache_dtype: fp8_e4m3`/`fp8_e5m2` would derive as 2-byte KV —
a conservative miss, never an unsafe `--kv-cache-dtype fp8`. If the fleet ever runs
one, exact-match is too narrow; the fix is a 3-line set-membership
(`{"fp8","fp8_e4m3","fp8_e5m2"}`), parked here rather than speculative.

## Verification the review does NOT replace

- JS gate: `node --check` on the extracted inline script + 6-case Node harness
  (decline / accept→force / cleared / sglang-skip / recheck-500 fail-open).
- End-to-end generation from real model dirs (5 SC4 victims + GPTQ no-KV +
  Nemotron-Super exact-id map hit).
- In-process preflight on the live Qwen3.6 wrapper: verdict **warn**
  (container_exists 31h, page-cache warn), smoke ok (real `--dry-run`), util 0.55.
