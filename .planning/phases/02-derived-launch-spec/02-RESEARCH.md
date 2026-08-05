# Phase 2 Research: Derived launch spec

**Date:** 2026-08-03
**Method:** Every claim below was computed from the actual `config.json` files on this box.
A Codex pass enumerated the raw fields; **every value it reported was re-verified against
the raw configs before being used, and its semantic classification was found wrong** (see
"Corrections to the handoff"). Ground truth is committed as `02-MODEL-FIXTURES.json`.

---

## 1. Corrections to the handoff table

The Phase 1 handoff's KV table is **numerically right and attributionally wrong**. Both
corrections change what the code must do.

### Correction A — `layer_types` is present on Qwen3.6 and qwen3-next

The handoff lists Qwen3.6's source field as `hybrid_override_pattern` and qwen3-next's as
`full_attention_interval`. Neither is what the precedence chain actually selects:

| model | `layer_types`? | `hybrid_override_pattern`? | `full_attention_interval` | chain selects |
|---|---|---|---|---|
| Qwen3.6-35B-A3B-FP8 | **yes** (40 entries) | no | 4 | `layer_types` |
| qwen3-next-80b | **yes** (48 entries) | no | 4 | `layer_types` |

The counts coincide (40÷4 = 10, 48÷4 = 12) so the *numbers* in the handoff are correct —
but only by arithmetic luck. **A test asserting "qwen3-next resolves via
`full_attention_interval`" would fail against the real config**, and ROADMAP success
criterion 2 is worded in a way that invites exactly that test. Assert the resolved *counts*
and the *selected source field as actually observed*, not the handoff's attribution.

### Correction B — non-full layers are not "sliding"; most are stateless

This is the one that matters for correctness. There are **three** KV classes, not two:

| class | KV cost | layer types seen on this box |
|---|---|---|
| **full** | grows with context: `bytes/token × max_model_len` | `full_attention`, `*` in a Nemotron pattern |
| **bounded** | capped: `bytes/token × sliding_window` | `sliding_attention` (gpt-oss only, window 128) |
| **stateless** | **zero** — fixed recurrent state, context-independent | `linear_attention` (Qwen3.6, qwen3-next), `M`/`E` in Nemotron patterns |

Measured distributions:

```
Qwen3.6-35B-A3B-FP8      layer_types = {full_attention: 10, linear_attention: 30}
qwen3-next-80b           layer_types = {full_attention: 12, linear_attention: 36}
gpt-oss-120b             layer_types = {full_attention: 18, sliding_attention: 18}, sliding_window=128
Nemotron-Super-120B      hybrid_override_pattern (88 chars) = {M: 40, E: 40, *: 8}
Nemotron-Nano-30B        hybrid_override_pattern (52 chars) = {M: ..., E: ..., *: 6}
```

Nemotron's pattern alphabet is `M` = Mamba, `E` = MLP/expert, `*` = attention. Only `*`
carries KV. `pat.count("*")` is correct; `len(pat) - count("*")` is **not** a sliding count.

**The prototype gets the right answer for the wrong reason.**
`kvcalc-prototype.py:13-15` counts `x == "full_attention"` and `"sliding" in x`, so
`linear_attention` matches neither bucket and silently contributes zero. That is accidentally
correct today and **actively dangerous tomorrow**: any future layer type that *does* carry KV
would also silently score zero, and the recommender would confidently under-reserve memory
and OOM at load. **The port must classify explicitly and raise/flag on an unrecognized layer
type rather than defaulting it to zero.** This is the single most important behavioral
difference between the prototype and the shipped function.

---

## 2. The trap: `sliding_window` without `use_sliding_window`

**Five of the seventeen models on this box** declare a `sliding_window` value while having
`use_sliding_window: false`:

```
Qwen2.5-0.5B-Instruct              sliding_window=32768   use_sliding_window=False
Qwen2.5-Coder-14B-Instruct         sliding_window=131072  use_sliding_window=False
qwen2.5-14b-instruct-gptq-int8     sliding_window=131072  use_sliding_window=False
DeepSeek-R1-Distill-Qwen-14B       sliding_window=131072  use_sliding_window=False
DeepSeek-R1-Distill-Qwen-32B       sliding_window=131072  use_sliding_window=False
```

These are **dense** models. Reading `sliding_window` without checking `use_sliding_window`
reclassifies every one of them as fully-sliding and collapses their KV estimate to near
zero — a 64-layer, 17.2 GB-at-131k model would be scored as almost free. The prototype's
`if sw and t.get("use_sliding_window")` guard at line 22-24 is load-bearing. **This deserves
a dedicated regression test**, because it is a silent wrong-answer failure, not a crash.

---

## 3. Verified precedence chain

First hit wins. Verified against all 17 configs:

1. `layer_types` (list) — classify each entry into full / bounded / stateless.
2. `hybrid_override_pattern` (str) — `count("*")` = full; every other char is stateless.
3. `full_attention_interval` (int) — full = `num_hidden_layers // interval`; rest stateless.
4. `sliding_window` **and** `use_sliding_window` truthy — all layers bounded.
5. else dense — full = `num_hidden_layers`.

`text_config` nesting: read transformer fields from `config["text_config"]` when present.
Two models on this box nest: `Qwen3.6-35B-A3B-FP8` and `qwen3-vl-4b-fp8`. Note the VL model
is the criterion-4 fixture, and it is **dense** (36/36) — so nesting and hybridity are
independent axes and need independent tests.

---

## 4. Verified formula and calibration

```
per_layer_bytes   = 2 × num_key_value_heads × head_dim × kv_dtype_bytes
kv_bytes(ctx)     = full_layers × per_layer_bytes × ctx
                  + bounded_layers × per_layer_bytes × sliding_window
                  + stateless_layers × 0
recommended_util  = min(0.95, (weights_gb + kv_gb + 6.0) / 121.0 + 0.04)
```

`head_dim` falls back to `hidden_size // num_attention_heads` (13 of 17 models here derive
it this way; only the hybrid Qwen/Nemotron families declare it explicitly).

**Calibration reproduced on the corrected classification:**

| model | ctx | weights | KV fp8 | naive KV | derived util | measured |
|---|---|---|---|---|---|---|
| qwen3-next-80b-a3b-nvfp4 | 262144 | 50.8 GB | 3.2 GB | 12.9 GB | **0.54** | **0.55** ✓ |
| Qwen3.6-35B-A3B-FP8 | 262144 | 37.5 GB | 2.7 GB | 10.7 GB | 0.42 | 0.55 (recipe) |
| Nemotron-Super-120B | 262144 | 80.3 GB | 1.1 GB | 11.8 GB | 0.76 | — |
| gpt-oss-120b | 131072 | 65.2 GB | 2.4 GB | 4.8 GB | 0.65 | — |
| R1-Distill-Qwen-32B | 131072 | 65.5 GB | 17.2 GB | 17.2 GB | 0.77 | — |

Within 0.02 of the hand-measured value — ROADMAP criterion 5 is satisfiable as specified.

> **Note for Phase 3, not this phase:** the derived 0.42 for Qwen3.6 is *below* its measured
> recipe of 0.55. That is not a formula error — the recipe reserves headroom the formula does
> not model. It is precisely why curated recipes must win over derived values in Phase 3.
> Do **not** "fix" the formula to chase 0.55 here.

---

## 5. Integration surface in `app.py`

| location | role in this phase |
|---|---|
| `app.py:677` `_infer_from_config(config, name_hints) -> dict` | **Closest analog.** Pure, dict-in/dict-out, `.get()` fallbacks throughout. Match this shape and its section-comment style. |
| `app.py:2478` `_vllm_serve_command` | Phase 1's documented-rationale docstring standard. |
| `app.py:280-305` | Documents the 121 GB unified pool. **Do not call from the pure function** — pass the pool in. |
| `app.py:2506` `_build_vllm_profile_script` | **Do not modify.** Phase 4 consumes the derived spec here. |
| `app.py:2555-2588` | The flat `0.75` / `32768` / `65536` literals. **Leave in place this phase.** |

`_parse_hf_model_dir` (`app.py:735`) already locates and parses `config.json` from
`snapshots/`, so the eventual caller has a parsed dict available — reinforcing that the new
function should take a dict, never a path.

## Validation Architecture

Every check is `pytest`. **Nothing in this phase may require vLLM, a container, a GPU, or a
network call** — vLLM is intentionally down for Helix training, and a pure function has no
excuse to need any of them.

| # | Property under test | Signal | Failure mode it catches |
|---|---|---|---|
| V1 | Precedence order | Synthetic configs with 2+ topology fields present; assert the higher-precedence one wins | Chain reordered by a refactor |
| V2 | Hybrid counts | Table test over all 17 fixtures: Nemotron-Super→8, Qwen3.6→10, qwen3-next→12, gpt-oss→18 full + 18 bounded, R1-32B→64 | The 4–11x naive overestimate returning |
| V3 | **Unknown layer type is loud** | Config with `layer_types: ["quantum_attention"]` → raises/flags, does **not** silently score zero | The prototype's accidental-correctness bug shipping |
| V4 | **`use_sliding_window` guard** | The 5 dense models with `sliding_window` set → classified dense, KV unchanged | Silent 10x KV underestimate on dense models |
| V5 | `text_config` nesting | Qwen3.6 + qwen3-vl-4b resolve from nested fields | VL models reading top-level zeros |
| V6 | `head_dim` fallback | Model without explicit `head_dim` → `hidden_size // num_attention_heads` | Division-by-zero / wrong KV width |
| V7 | **Calibration** | qwen3-next-80b derived util within 0.02 of 0.55 | Formula drift — the phase's core evidence |
| V8 | Purity | Function called with only a dict; no `open`/`glob`/`os.path`/`requests` in its body | Untestable filesystem dependency creeping in |
| V9 | Degenerate configs | Empty dict, missing `num_hidden_layers`, zero heads → defined behavior, no traceback | Crash on a malformed snapshot |

**Fixture source:** `02-MODEL-FIXTURES.json` (17 models, generated and verified 2026-08-03).
It is a committed snapshot, not a live read — tests must not glob the filesystem, or they
break the moment a model is added or removed from the box.

---

## 6. Risks

| risk | mitigation |
|---|---|
| Scope creep into the recipe table | CONTEXT.md fences it to Phase 3; the flat literals stay untouched this phase |
| Test fixtures drift from the box | Fixture JSON is committed; regeneration is a deliberate act, not a test-time glob |
| Formula "corrected" to chase Qwen3.6's 0.55 | Documented above as a Phase 3 concern; only qwen3-next is a calibration target |
| Someone else's dirty files swept into the commit | ~6 concurrent Claude sessions here — commit by explicit path, never `git add -A` |
