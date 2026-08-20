# Phase 4: Parameterized scripts + UI settings — Research

**Researched:** 2026-08-20
**Domain:** Bash parameter expansion in generated launch scripts; systemd-run scope env
inheritance; FastAPI dict-shaping; vanilla-JS DOM safety
**Confidence:** HIGH (every load-bearing claim verified against this box or this tree)

## Summary

The phase's riskiest premise — that criterion 2 was invalidated by commit `079917b` — is
**FALSE**. `systemd-run --user --scope` executes the command in a process forked from the
caller, so `Popen(env=...)` propagates verbatim. Measured on this box. The `docker run -d`
change is likewise a non-issue, because overrides are consumed by **bash on the host** as
`${VAR:-derived}` expansions that become `docker run` *argv*, not container env.

The pattern this phase needs already exists and already ships: the llama.cpp profile
(`profiles/llama.cpp/start_gguf_hauhaucs_qwen3.8-27b-aggressive-mtp.sh`) is a working,
in-production instance of exactly criteria 1+2 — `CTX="${CTX:-$R_CTX}"`, delivered as
`RECIPE=`/`CTX=` env through `_engine_start` → `_launch_argv` → scope → bash. Phase 4 is
mostly "port that idiom to the vLLM generator, then surface it."

The real work is not the transport. It is (a) `warnings` and the derived/recommended split
are computed at *generation* time and thrown away — nothing on the profile-list path
recomputes or persists them; (b) there is **no flag parser at all** for existing scripts
(`_parse_script_meta` reads three header comments and nothing else), so criterion 4 needs a
new parser built from scratch; (c) the UI has **zero** HTML escaping helpers across 86
`innerHTML` sites.

**Primary recommendation:** Keep the roadmap's 3-plan split. Plan 04-01 (generator emits
`${VLLM_*:-derived}` + `_engine_start` accepts an allow-listed override map, modelled
literally on the existing `RECIPE` code path) is small and low-risk. Plan 04-02 must first
*persist or recompute* the spec so the card has something to show. Plan 04-03 (legacy
parse + diff + escaping) is the largest and should be split if it grows.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-06 | Launch settings adjustable per-launch from the UI without rewriting or clobbering the script on disk | Transport verified end-to-end (§Criterion 2 Verdict); emission point identified at `app.py:4435-4497`; override allow-list precedent at `app.py:2968-2974`; UI data gap and escaping gap enumerated (§Criterion 3, §Criterion 5) |
</phase_requirements>

## Criterion 2 Verdict — the premise holds, unchanged

**VERDICT: criterion 2 as written is still implementable. Do not rewrite it.**

### 1. Bare-metal path: `systemd-run --user --scope` DOES inherit the caller's env

`--scope` is not `--service`. In scope mode systemd-run registers a transient cgroup with
the user manager and then **forks/execs the command itself**, inheriting its own
environment. In service mode the *user manager* (PID of `systemd --user`) spawns the
process, which is why service mode needs `--setenv=`. `_launch_argv` (`app.py:2919-2925`)
uses `--scope`. [VERIFIED: measured on this box, systemd 255 (255.4-1ubuntu8.16)]

```
$ FOO=hello RECIPE=tiny systemd-run --user --scope --quiet --collect --unit=envtest-$$ bash /tmp/envt.sh
RECIPE=[tiny] FOO=[hello]                      # ← inherited, no --setenv needed

$ FOO=hello systemd-run --user --quiet --collect --unit=... --pipe bash /tmp/envt.sh
RECIPE=[] FOO=[]                               # ← service mode: NOT inherited
```

`--setenv=` is therefore **not required** and per-var enumeration is **not needed**. It is
also *harmful* to add: passing `--setenv=FOO=hello` and nothing else does not filter the
inherited set, so it buys nothing and creates the illusion of an explicit contract.

**Corroborating evidence from production code, not just the probe:** `_engine_start`
already ships an env-borne override — `env["RECIPE"] = recipe` at `app.py:2971-2973` — and
the llama.cpp script consumes it at `_RESOLVED="$(_recipes use "${RECIPE:-}")"`. That path
is live and working today, through the exact same `--scope` wrapper. If scope mode dropped
env, the entire llama.cpp quant ladder (commit `cfb634f`) would be silently stuck on
`default_recipe` — it is not.

**Residual caveat (must be in the plan):** the `shutil.which("systemd-run")` fallback at
`app.py:2912-2915` returns `["bash", script]`, which also inherits env. **Both branches of
`_launch_argv` transport env correctly.** No branch-specific handling is needed.

### 2. Container path: overrides never cross the docker boundary at all

The correct design consumes overrides **host-side, in bash**, before `docker run` is
invoked. `${VLLM_MAX_MODEL_LEN:-32768}` expands in the script's own shell and the *result*
becomes an argv token of `docker run … --max-model-len 262144`. Nothing needs `-e`.

This matters because the vLLM flags in question (`--max-model-len`,
`--gpu-memory-utilization`, `--max-num-seqs`) are **CLI arguments to `vllm serve`**, not
env vars vLLM reads. Passing them as container env would do nothing. The llama.cpp script
demonstrates the correct shape at its `ARGS=(… --ctx-size "$CTX" …)` array.

`-e` inside the generated script (`app.py:4410-4429`) is a *separate, unrelated* concern —
it carries `HF_HUB_OFFLINE`, `VLLM_MARLIN_USE_ATOMIC_ADD`, etc. Phase 4 should not touch it.

### 3. Are there two distinct transport mechanisms? No — one.

| Path | Transport | Extra work needed |
|------|-----------|-------------------|
| bare-metal / `systemd-run --scope` | process env inheritance | none |
| `bash` fallback (no systemd-run) | process env inheritance | none |
| generated vLLM `docker run -d` | bash `${VAR:-default}` → argv | none |
| llama.cpp `docker run -d` | bash `${VAR:-default}` → argv | already done |

**There is exactly ONE mechanism: env in, bash expansion, argv out.** The phase does not
need extra plans on transport grounds. The roadmap's 3-plan split stands.

## Blocking side-finding: the generator and the on-disk profiles have drifted

Commit `079917b` edited the twelve files in `profiles/vLLM/` **but not the generator.**

- `app.py:4492` still emits `exec docker run --name vllm_node --gpus all …` — no `-d`.
- `git show 079917b -- app.py | grep "docker run"` returns only doc-comment lines.
- All 12 on-disk scripts now have `-d`; ten of them still carry
  `--restart unless-stopped`, which `716ca98`-era notes say was deliberately dropped.

Consequence: **regenerating any vLLM profile silently reverts the 2026-08-18 cgroup fix**
for that profile. The `--scope` wrapper still protects it, so it is not an outage today,
but the two halves of `079917b` disagree.

Phase 4 rewrites this exact code block. **Fold `-d` into the generator in plan 04-01** —
it is one token in the string at `app.py:4492`, and it is cheaper to fix while the block is
already open than to leave a documented divergence for Phase 5.

[VERIFIED: `git show 079917b --stat`, `grep -rn "docker run" profiles/vLLM/`]

## Criterion 1 — where placeholders get emitted

### Generation path (single call chain)

```
POST /api/vllm/profiles/from-hf            app.py:4523
  └─ _create_vllm_profile_from_path        app.py:4501   (path validation, atomic write)
      └─ _build_vllm_profile_script        app.py:4371   ← returns (name, script, info)
          ├─ _profile_model_info
          ├─ _resolve_launch               app.py:4293   ← recipe | gpt_oss | derived
          │    └─ _derive_launch_spec      app.py:994
          ├─ _vllm_script_preamble         app.py:4337   ← header comments + docker rm -f
          ├─ _recipe_profile_body          app.py:4359   ← recipe shape: exec run-recipe.sh
          └─ arg_lines / env_lines / f-string  app.py:4410-4497
```

### The three emission sites to parameterize

| Flag | Current literal | Line | Proposed |
|------|-----------------|------|----------|
| `--gpu-memory-utilization` | `{resolved['util'] … else 0.75}` | 4440 | `${VLLM_GPU_MEMORY_UTILIZATION:-<derived>}` |
| `--max-model-len` | `{resolved['max_model_len'] … else 32768}` | 4451 | `${VLLM_MAX_MODEL_LEN:-<derived>}` |
| `--max-num-seqs` | hardcoded `2` (both branches, 4446/4451) | 4446, 4451 | `${VLLM_MAX_NUM_SEQS:-2}` |

Note `--max-num-seqs 2` is **glued to `--max-model-len` in the same f-string** on the
non-gpt-oss branch and to `--max-model-len 65536` on the gpt-oss branch. Splitting those
into separate `arg_lines` entries is a prerequisite edit, not an incidental one.

### Three shapes, not one — the plan must say which are parameterized

`_resolve_launch` returns three shapes (`app.py:4293-4334`):

1. `shape="recipe"` → body is `exec ./run-recipe.sh "$RECIPE" -d` (`app.py:4359-4368`).
   **The YAML owns the flags.** Placeholders cannot be emitted here without teaching
   `run-recipe.sh` (a *different repo*, `~/spark-vllm-docker`) about overrides. **Recommend
   explicitly out of scope; the card must render recipe-backed profiles as non-editable
   with the reason shown.** This is the single most likely scope trap in the phase.
2. `shape="docker", gpt_oss=True` → hardcoded 65536 / full-precision KV. Parameterizable,
   but the 65536 is a deliberate measured constant; emit it as the default.
3. `shape="docker"` derived → the main case.

### Quoting

`_build_vllm_profile_script` `shlex.quote`s every dynamic atom (`app.py:4431-4437`), with an
explicit comment that quoted output must NOT be re-wrapped in double quotes. Placeholders
are the opposite case: `${VLLM_MAX_MODEL_LEN:-32768}` must be emitted **literally and
unquoted** so bash expands it. Do not run it through `shlex.quote`. The *value* is
untrusted at runtime — validate it in `_engine_start` (see below), and additionally emit a
numeric guard in the script so a bad env var fails before `docker run`, e.g.
`[[ "$MML" =~ ^[0-9]+$ ]] || { echo "…"; exit 1; }`. Defence in depth matches the repo's
existing `_validate_served_name` + `shlex.quote` idiom.

### config.example.json

`config.json` is gitignored; `config.example.json` carries `app / services / vllm /
llamacpp / paths / alerts / sites_discovery / sites`. The `vllm` block already holds
`image`, `image_gpt_oss`, `moe_backend`, `serve_command`, `recipes`, `recipe_dir`, each with
a `_comment*` sibling. If Phase 4 adds config-level override defaults, follow that idiom
exactly (leading `_comment`, config→env→default precedence per `_live_vllm_cfg`).

## Criterion 2 — the override-plumbing edit

`_engine_start` (`app.py:2941-2991`) already has the shape. The `recipe` parameter is the
template to copy:

```python
env = os.environ.copy()
if recipe:
    env["RECIPE"] = recipe          # app.py:2971-2974
```

with the comment that names the security contract exactly: *"it arrives in an HTTP body and
lands in a bash script's environment, so a known-names check is the only acceptable
filter."* Overrides must obey the same rule — an **allow-list of names** (not a prefix
match, not a pass-through dict) plus **type/range validation of values**.

Recommended shape:

```python
_OVERRIDE_ENV = {                    # name -> (env var, validator)
    "max_model_len":  ("VLLM_MAX_MODEL_LEN", _int_in(1, 10_000_000)),
    "gpu_memory_utilization": ("VLLM_GPU_MEMORY_UTILIZATION", _float_in(0.10, 0.95)),
    "max_num_seqs":   ("VLLM_MAX_NUM_SEQS", _int_in(1, 256)),
}
```

The util bounds should reuse `_derive_launch_spec`'s own `util_floor=0.10` /
`util_cap=0.95` (`app.py:998`) rather than inventing new constants — the docstring at
`app.py:1018-1020` states the reason those exist ("its consumer hands it to a real launch").

### Admission ordering — a real trap

`_vram_admission_check` runs at `app.py:2961`, and the llama.cpp `recipe` block deliberately
resolves **before** it (`app.py:2949-2959`, "the ladder spans 17 GB to 60 GB; admitting on
the header would wave through a launch three times its declared size").

A `max_model_len` or `util` override changes the real footprint the same way. **Overrides
must be applied before `_vram_admission_check`, not after**, or the phase re-introduces the
exact bug the recipe code comments about. Since the profile header's `# VRAM:` is a static
comment, the plan needs a decision: either recompute a footprint from the override (needs
`_derive_launch_spec` with `requested_context=`, which it already supports at `app.py:997`)
or refuse to admit and require `force=true`. **Recommend the former** — `requested_context`
exists precisely for this and already emits a warning when the request exceeds what fits
(`app.py:1088-1091`).

### TOCTOU

Criterion 2's "never mutated at launch" claim is satisfied by construction: the script file
is written only by `_create_vllm_profile_from_path` (`app.py:4516-4519`, tmp + `os.replace`,
atomic) and read-only at launch. Nothing in `_engine_start` writes. Preserve that.

## Criterion 3 — the derived/recommended data is computed then discarded

This is the phase's largest hidden cost, and it is a **data-availability** problem, not a UI
problem.

**What exists.** `_derive_launch_spec` returns 16 keys (`app.py:1103-1120`): `topology`,
`num_hidden_layers`, `full_attention_layers`, `bounded_kv_layers`, `stateless_layers`,
`source_field`, `kv_bytes_per_token`, `bounded_bytes_total`, `declared_max_context`,
`max_fitting_context`, `max_model_len`, `kv_gb`, `weights_gb`, `overhead_gb`,
`recommended_util`, `warnings`.

**Derived vs recommended, precisely:**

| Concept | Key | Meaning |
|---------|-----|---------|
| derived (the placeholder) | `max_model_len` | min(declared max, what fits) — what the script will actually use |
| ceiling context | `max_fitting_context` | largest context the KV budget allows |
| model's own ceiling | `declared_max_context` | vendor `max_position_embeddings` |
| recommended (the label) | `recommended_util` | `(weights+kv+overhead+resident)/pool + margin`, clamped [0.10, 0.95] |

Note `_resolve_launch` maps `spec["recommended_util"]` onto its own key `"util"`
(`app.py:4326-4332`) and drops the other fourteen keys. So by the time a script is written,
only two numbers survive — and both survive **only as literals inside the .sh text**.

**Where it is lost.** The profile-list path is `_scan_profiles` → `_parse_script_meta`
(`app.py:~440-473`), which reads exactly three header comments (`# Name:`,
`# Description:`, `# VRAM:`) and returns `{id, name, script, description, vram_gb}`. It
never opens a model config, never calls `_derive_launch_spec`, and knows nothing of
`warnings`.

**The `warnings` field.** `info["warnings"]` is assembled at `app.py:4391` and extended at
`app.py:4320` (spec warnings), `4304/4307/4310` (recipe warnings), `4469` (capability
warnings). It is returned **once**, in the `model` key of the create-profile response
(`app.py:4521`), and never again. `grep` confirms no other reader. Sources of warning text:

- topology: unrecognized layer type counted as full attention (`app.py:846-862`)
- scalar hygiene: `non-numeric`/`non-finite`/`negative <field>` (`app.py:1045-1051`)
- budget: `invalid pool_gb`, `no full-attention layers` (`app.py:1066-1071`)
- requested context exceeds usable (`app.py:1088-1091`)
- recipe: `cluster_only`, non-8000 port, invalid recipe name (`app.py:4258-4310`)
- capability map ambiguity (`app.py:2010-2027`)

**Two viable designs — the plan must pick one:**

| Option | How | Cost | Risk |
|--------|-----|------|------|
| A. Persist at generation | Emit `# Derived: {json}` / `# Warnings: {json}` header comments; extend `_parse_script_meta` to read them | small, stays pure-function, testable with zero GPU | stale if the model or config changes after generation |
| B. Recompute on read | New endpoint recomputes `_derive_launch_spec` per profile from its `--model` path | always fresh | `_scan_profiles` gains filesystem+JSON reads on a hot list path; needs a model-dir back-reference the profile does not currently carry |

**Recommend A**, with a `# Generated:` timestamp so staleness is visible, and a separate
on-demand endpoint for B-style refresh if wanted later. A keeps the whole thing inside
`_parse_script_meta`, which is already the repo's parsing seam and already 100% unit-tested
(`tests/test_profile_meta.py`). Header comments are also exactly how `# VRAM:` already works.

## Criterion 4 — "parsed values read-only" against a parser that does not exist

**There is no flag parser today.** `_parse_script_meta` (`app.py:~440`) reads only
`# Name:`, `# Description:`, `# VRAM:` from the first 20 lines. It does not read
`--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, the image, or the model
path. Criterion 4 therefore requires **new code**, and its size was likely underestimated
when the roadmap was written.

### Detecting generated vs legacy

The generator's preamble emits a stable marker (`app.py:4345-4356`):

```
# Auto-generated by DGX Model Manager from:
# <launch_dir>
```

`_create_vllm_profile_from_path` already uses a weaker form of this for collision detection
— `if str(launch_dir) not in existing: raise 409` (`app.py:4512-4515`). So:

- **generated** = contains `Auto-generated by DGX Model Manager`
- **parameterized** = generated AND contains `${VLLM_MAX_MODEL_LEN`
- **legacy** = everything else (all 12 current `profiles/vLLM/*.sh` that predate this phase,
  plus the four hand-written ones: `start_nemotron_nano.sh`, `start_nemotron_super.sh`,
  `start_qwen3_coder_next.sh`, `start_qwen3_next_80b.sh`)

That is a 3-state model, not 2. The card needs three renderings: editable, read-only +
"parameterize" action, and recipe-backed (non-editable, per Criterion 1 above) — **four
states total**.

### What "parse" can honestly mean

Do **not** attempt a bash parser. The realistic contract, and the one the plan should lock:

> Match `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs` followed by a
> literal numeric token, via anchored regex, on a line-by-line scan. If the token is not a
> plain number (a `$VAR`, a `$(…)`, a bash array element, absent, or present twice with
> different values), report the flag as **unparseable** rather than guessing.

All 12 on-disk scripts satisfy the literal-numeric form today, so this is sufficient in
practice while failing safe on the llama.cpp-style `ARGS=( --ctx-size "$CTX" )` shape.
Unparseable ⇒ the "parameterize" action is offered but disabled, with the reason shown.

### The diff preview

- Compute the rewrite server-side and return `{current, proposed, diff}` — do **not** build
  the diff in JS, which would need the full script text in the browser.
- `difflib.unified_diff` is stdlib and keeps the function pure/testable.
- The write must reuse the existing atomic idiom (`tmp.write_text` → `os.chmod 0o755` →
  `os.replace`, `app.py:4516-4519`), not a naive `write_text`.
- **Confirm-token TOCTOU:** the preview and the apply are two HTTP calls. Return a hash of
  the current file content with the preview and require it on apply, rejecting with 409 if
  the file changed. This repo runs ~3 concurrent Claude sessions and the profile dir is
  shared; the 409-on-different-contents precedent already exists at `app.py:4512`.
- Keep a `.bak` or refuse to parameterize a script whose parse was partial. Rewriting a
  hand-tuned script (e.g. the 60-line gpt-oss one with its root-cause writeup) is
  destructive and irreversible.

## Criterion 5 — XSS surface, enumerated

**There is no HTML-escaping helper anywhere in the codebase.** `grep -n "escapeHtml"` →
zero hits. 86 `innerHTML` assignments total. The only sanitizer present is
`const q = s => String(s ?? '').replace(/'/g, "\\'")` (`app.py:7970`), which escapes single
quotes for a **JS string literal inside an attribute** — it does not escape `<`, `>`, `&`,
or `"`, so it does not make the surrounding HTML safe and it can be broken out of with a
double quote.

### Sites taking untrusted strings

| Line | Sink | Untrusted input | Provenance of the string |
|------|------|-----------------|--------------------------|
| `app.py:7971-7990` | `el.innerHTML = profiles.map(...)` | `${p.name}`, `${p.description}` | `# Name:` / `# Description:` header comments of any `start_*.sh`; also filename-derived |
| `app.py:7985-7988` | attribute + `onclick` | `q(p.model_dir)`, `q(p.name)`, `q(p.id)` | same, plus a filesystem path |
| `app.py:7903` | `sel.innerHTML = names.map(...)` | model/profile names | litellm/profile names |
| `app.py:7467` | `el.innerHTML = '<div class="model-grid">' + models.map(...)` | ollama model names | remote registry-supplied |
| `app.py:7582` | same shape | litellm model names | config/ConfigMap-supplied |
| `app.py:8655` | `root.innerHTML = models.map(renderHFBCard)` | HF repo id / author / tags | **hf.co API responses — fully attacker-controlled** |
| `app.py:7227` | `panel.innerHTML = '<div class="model-card"…'` | model card fields | inventory scan |
| `app.py:8463` | `stats.innerHTML` | counts only | numeric — low risk |

`app.py:8655` (Hugging Face browse results) is the highest-severity of these: a third party
can name a repo, and that name reaches `innerHTML` unescaped. It is *not* in Phase 4's
nominal scope (Phase 4 is about profile cards) but it is the same one-line fix.

### Prescription

1. Add one helper next to the existing `q`:
   `const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));`
2. Wrap **every** interpolation of a name/description/path/warning in `esc(...)`.
3. For `onclick="fn('${q(x)}')"` attributes, `q` alone is insufficient — apply `esc(q(x))`,
   or better, migrate those to `dataset` attributes + a delegated listener, which removes
   the nested-quoting problem entirely.
4. Warnings text (new in this phase) must go through `textContent` or `esc` — warning
   strings embed `{value!r}` of vendor-supplied config fields (`app.py:1045`), so they carry
   attacker-influenced content by construction.

**Anti-pattern to avoid:** "escape at the source in Python." Escaping server-side then
rendering into `textContent` double-escapes; escaping server-side for one sink breaks JSON
consumers. Escape at the DOM sink, in JS, exactly once.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (no config file; plain `tests/` + `conftest.py`) |
| Config file | none — `tests/conftest.py` does `sys.path.insert(0, REPO_ROOT)` |
| Quick run command | `python3 -m pytest tests -q` |
| Full suite command | `python3 -m pytest tests -q` |
| Current baseline | **337 passed in 0.58s** [VERIFIED: run 2026-08-20] |

The suite is pure-function by design and runs in under a second with no GPU, no docker, no
network. **Every Phase 4 criterion is testable inside that constraint** — see below. The
ground-truth model snapshot is `tests/fixtures/model-fixtures.json`; `conftest.py:15-22`
states the rule explicitly ("Deliberately a *file*, not a live glob of
~/.cache/huggingface or /mnt/models"). **No new test may glob a model-cache directory.**

Existing convention to follow: `tests/test_vllm_profile_generation.py` builds a fake HF repo
under `tmp_path` (`_make_hf_repo`, `_write_model_config`) and asserts on the **returned
script text**, never on a launched process. Phase 4's script-emission tests are the same
shape.

### Phase Requirements → Test Map

| Req | Criterion | Behavior | Type | Automated command | Exists? |
|-----|-----------|----------|------|-------------------|---------|
| REQ-06 | 1 | Generated script text contains `${VLLM_MAX_MODEL_LEN:-<derived>}` etc., and the derived default equals `_derive_launch_spec`'s value | unit | `pytest tests/test_vllm_profile_generation.py -k placeholder -x` | ❌ Wave 0 |
| REQ-06 | 1 | Placeholder is emitted unquoted (not `shlex.quote`d) and the script still passes `bash -n` | unit | `pytest tests/test_vllm_profile_generation.py -k bash_syntax -x` | ❌ Wave 0 |
| REQ-06 | 1 | Recipe-shape and gpt-oss-shape scripts emit the documented (different) forms | unit | `pytest tests/test_vllm_profile_generation.py -k shape -x` | ❌ Wave 0 |
| REQ-06 | 2 | `_launch_argv` output, when executed with a patched env, reaches the script | unit | `pytest tests/test_engine_start.py -k env_transport -x` | ❌ Wave 0 |
| REQ-06 | 2 | Override map → env dict: allow-list rejects unknown names; validators reject out-of-range util / non-integer context | unit | `pytest tests/test_engine_start.py -k override_allowlist -x` | ❌ Wave 0 |
| REQ-06 | 2 | Script file mtime + sha256 unchanged across a launch (no mutation / no TOCTOU) | unit | `pytest tests/test_engine_start.py -k script_not_mutated -x` | ❌ Wave 0 |
| REQ-06 | 2 | Overrides are applied **before** `_vram_admission_check` (ordering regression) | unit | `pytest tests/test_vram_admission.py -k override_before_admission -x` | ❌ Wave 0 |
| REQ-06 | 3 | Derived + recommended + warnings survive round-trip: generate → `_parse_script_meta` → same values | unit | `pytest tests/test_profile_meta.py -k derived_roundtrip -x` | ❌ Wave 0 |
| REQ-06 | 3 | Every model in `model-fixtures.json` yields a spec whose `recommended_util` is in [0.10, 0.95] and `max_model_len` ≤ `declared_max_context` | unit (fixture-driven) | `pytest tests/test_launch_spec.py -k fixture_bounds -x` | partial — extend |
| REQ-06 | 4 | 3-state classifier: generated / parameterized / legacy, over the 12 committed `profiles/vLLM/*.sh` read as text fixtures | unit | `pytest tests/test_profile_meta.py -k classify -x` | ❌ Wave 0 |
| REQ-06 | 4 | Flag parser returns numeric values for literal forms and `unparseable` for `$VAR`/`$(…)`/absent/duplicate | unit | `pytest tests/test_profile_meta.py -k parse_flags -x` | ❌ Wave 0 |
| REQ-06 | 4 | Parameterize rewrite is idempotent (apply twice = same text) and `bash -n` clean | unit | `pytest tests/test_profile_meta.py -k parameterize_idempotent -x` | ❌ Wave 0 |
| REQ-06 | 4 | Apply with a stale content hash returns 409 and leaves the file byte-identical | unit | `pytest tests/test_profile_meta.py -k stale_hash -x` | ❌ Wave 0 |
| REQ-06 | 5 | `esc()` escapes all five of `& < > " '`; property-style table of hostile inputs | unit (JS extracted) | `pytest tests/test_ui_escaping.py -x` | ❌ Wave 0 |
| REQ-06 | 5 | No `innerHTML` template in the profile-card render path interpolates a bare `${p.…}` without `esc(` | **static assertion over `app.py` source text** | `pytest tests/test_ui_escaping.py -k no_unescaped_interpolation -x` | ❌ Wave 0 |

### How the GPU-free claims are made honest

- **Criterion 2 without launching an engine.** Do not mock `subprocess.Popen` and assert on
  call args — that tests the mock. Instead call the real `_launch_argv(script, safe_id)` and
  `subprocess.run` it against a throwaway `tmp_path` script that just `echo`es the env vars,
  with a patched `env`. That exercises the *actual* systemd-run/bash boundary in <1s with no
  GPU, no docker, no model. Guard with `pytest.mark.skipif(not shutil.which("systemd-run"))`
  and keep a second, always-run test for the bash-fallback branch.
- **Criterion 1 without vLLM.** `bash -n <script>` is a syntax check that never executes;
  it catches the realistic failure (a malformed `${…:-}` or an unbalanced quote from the
  f-string surgery) at ~2 ms per script. Add it to the generation test.
- **Criterion 4 without touching real profiles.** Copy the 12 committed
  `profiles/vLLM/*.sh` into `tmp_path` per test. They are in-repo, so they are already a
  committed snapshot in the `model-fixtures.json` spirit — no cache glob needed.
- **Criterion 5 without a browser.** The UI is a JS string inside `app.py`. A source-text
  assertion (regex for `innerHTML` templates containing `${p.` not preceded by `esc(`) is
  crude but catches the actual regression — a future edit adding an unescaped field — and
  costs nothing. Pair it with a direct unit test of the `esc` logic reimplemented in Python
  from the same character table, or extracted and run under `node` if available
  (`skipif(not shutil.which("node"))`).

### Sampling Rate

- **Per task commit:** `python3 -m pytest tests -q` (whole suite; it is 0.58s — there is no
  reason to sample a subset)
- **Per wave merge:** same, plus `bash -n` over every file in `profiles/`
- **Phase gate:** full suite green, then `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_engine_start.py` — new file; env transport, allow-list, no-mutation (criterion 2)
- [ ] `tests/test_ui_escaping.py` — new file; `esc` behavior + source-text guard (criterion 5)
- [ ] Extend `tests/test_vllm_profile_generation.py` — placeholder emission, `bash -n`, shape coverage (criterion 1)
- [ ] Extend `tests/test_profile_meta.py` — classifier, flag parser, parameterize round-trip, stale-hash 409 (criteria 3, 4)
- [ ] Extend `tests/test_vram_admission.py` — override-before-admission ordering (criterion 2)
- [ ] No framework install needed; no new dependency (difflib, hashlib, re are stdlib)

## Don't Hand-Roll

| Problem | Don't build | Use instead | Why |
|---------|-------------|-------------|-----|
| Diff for the preview | a line-differ | `difflib.unified_diff` (stdlib) | correct, pure, already available |
| HTML escaping | per-site ad-hoc `.replace('<','&lt;')` | one `esc()` helper, applied at every sink | 86 sinks; per-site escaping is how the current gap happened |
| Atomic script write | `path.write_text(...)` | the existing tmp → `chmod` → `os.replace` idiom at `app.py:4516-4519` | ~3 concurrent sessions share this repo |
| Env transport across systemd | `--setenv=` enumeration | nothing — `--scope` inherits | measured; enumeration is dead code |
| Parsing bash | a bash grammar / `bashlex` | anchored regex + explicit `unparseable` state | no new dep; fails safe |
| Deciding "does it fit" | new arithmetic in the override path | `_derive_launch_spec(requested_context=…)` (`app.py:997`) | already implemented and already warns on over-request |
| Content-change detection | mtime | `hashlib.sha256` of the text | mtime is not reliable under `os.replace` + concurrent edits |

**Key insight:** every primitive this phase needs is already in the tree. The phase is
almost entirely *wiring and surfacing*, plus one genuinely new component (the legacy flag
parser). Any plan that introduces a dependency is off-track.

## Common Pitfalls

### Pitfall 1: Assuming the `-e` docker flags are the override channel
**What goes wrong:** `-e VLLM_MAX_MODEL_LEN=…` is added to the container. vLLM ignores it;
the CLI flag still carries the old literal. The launch silently uses the wrong value.
**Why:** the generator already has an `env_lines` list (`app.py:4410`) and it looks like the
right place. It is not — those are vLLM *runtime* env vars, a different concept.
**Avoid:** overrides expand in host bash and become argv. Test asserts on the rendered
`--max-model-len` token, not on the presence of an env var.

### Pitfall 2: Parameterizing the recipe shape
**What goes wrong:** `shape="recipe"` bodies delegate to `run-recipe.sh` in a *different*
repo. Emitting placeholders there does nothing, or worse, looks like it works.
**Avoid:** three-way branch in the generator; card renders recipe-backed as non-editable.

### Pitfall 3: Overriding after admission
**What goes wrong:** a user raises context to 262144, admission passes on the header's
stale `# VRAM:`, the launch dies in engine init. This is the same class as the bug the
llama.cpp recipe code comments about at `app.py:2949-2951`.
**Avoid:** resolve overrides before `_vram_admission_check` (`app.py:2961`).

### Pitfall 4: The page-cache tax makes any util number look wrong
**What goes wrong:** the same util yields wildly different KV depending on `buff/cache`
(0.19 GiB vs 25.97 GiB at util 0.55 — `app.py:2994-3010`). A UI that shows "recommended
0.55" without that context invites the user to conclude the recommendation is broken.
**Avoid:** the card should surface the reclaim action / note alongside the util control.
`POST /api/vllm/reclaim-cache` already exists.

### Pitfall 5: Editing `app.py` does not deploy
`systemctl --user restart dgx-model-manager.service` is required. And per STATE.md, ~3
concurrent Claude sessions run in this repo — **commit by explicit path, never `git add -A`**.

### Pitfall 6: `--max-num-seqs` is welded to `--max-model-len`
Both f-strings at `app.py:4446` and `app.py:4451` emit the two flags on one line. Splitting
them is a prerequisite refactor; skipping it makes the placeholder substitution look
correct in a diff while producing a malformed line.

## Architectural Responsibility Map

| Capability | Primary tier | Secondary | Rationale |
|------------|-------------|-----------|-----------|
| Derive context/util/KV numbers | pure Python (`_derive_launch_spec`) | — | already pure, already fixture-tested |
| Emit placeholders into script text | generator (`_build_vllm_profile_script`) | — | script text is the only artifact that outlives the request |
| Expand override → concrete value | **bash, in the script, at launch** | — | the value must become docker argv; only bash sits at that seam |
| Validate override name/range | FastAPI `_engine_start` | bash numeric guard | untrusted HTTP body reaching a shell — allow-list at the boundary, defence in depth in the script |
| Transport override to the script | process env via `Popen(env=)` | — | `--scope` inherits; no other channel needed |
| Persist derived/recommended/warnings | script header comments | — | `_parse_script_meta` is the existing read seam |
| Escape untrusted text | browser JS, at the DOM sink | — | escaping server-side breaks JSON consumers and double-escapes |
| Diff + confirm | FastAPI (server-side `difflib` + sha256 token) | — | client must not hold or resend script text |

## Project Constraints (from CLAUDE.md)

- **Route to Codex before facts get written down.** The criterion-2 verdict in this document
  and any measured claim in the eventual VERIFICATION should go to
  `codex exec --sandbox read-only --skip-git-repo-check --ephemeral` for falsification, and
  the "where do you disagree" section must be asked for explicitly.
- **Well-specified coding subtasks → Codex.** Plan 04-01 and the Wave 0 test files are
  exactly this shape.
- Plan/verify stays on Claude; implementation and review can be offloaded.
- Subagent output ceiling: any PLAN.md must be built across many small `Edit` calls, one
  section per call. A single large `Write` dies at 8192 tokens having written nothing.
- Cap `max_tokens` at 2048.
- Present times in America/Chicago.
- Small, reviewable commits; commit by explicit path (repo has concurrent sessions).
- Patterns over patches — the llama.cpp script and the `alerts` config block are the
  patterns this phase must follow rather than inventing new ones.

## Environment Availability

| Dependency | Required by | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| systemd (user) | `_launch_argv` scope wrapper; criterion-2 test | ✓ | systemd 255 (255.4-1ubuntu8.16) | bare `bash` branch (already coded) |
| bash | placeholder expansion; `bash -n` tests | ✓ | system bash | none needed |
| python3 + pytest | whole suite | ✓ | suite runs, 337 passed / 0.58s | — |
| docker | runtime only; **not needed by any test** | ✓ | — | tests never invoke it |
| GPU / live vLLM | runtime only | (vLLM currently up, Qwen3.6-35B-A3B-FP8) | — | **no test may depend on it** |
| node | optional, for running the extracted `esc` under a real JS engine | unverified | — | reimplement the character table in Python |
| difflib / hashlib / re | diff, hash, parse | ✓ (stdlib) | — | — |

**Missing with no fallback:** none.
**No external packages are installed by this phase**, so the Package Legitimacy Audit is
not applicable — nothing to slopcheck.

## Assumptions Log

| # | Claim | Section | Risk if wrong |
|---|-------|---------|---------------|
| A1 | All 12 committed `profiles/vLLM/*.sh` carry literal numeric values for the three target flags, so the regex parser suffices | Criterion 4 | spot-checked one file fully and grepped the rest for `docker run`; a script with a `$VAR` value would be reported unparseable, which is the safe outcome — low risk |
| A2 | `node` is present for the optional JS-engine escaping test | Environment | falls back to the Python reimplementation |
| A3 | Emitting `-d` in the generator is desirable (matches what the on-disk profiles were hand-edited to) | Side-finding | if `-d` was deliberately kept out of the generator, this is a scope addition rather than a fix — confirm with Josh |
| A4 | `run-recipe.sh` in `~/spark-vllm-docker` has no override mechanism of its own | Criterion 1 | not inspected (out of this repo); if it does, recipe-backed profiles could also be parameterized and the phase grows |

## Open Questions

1. **Should overrides persist, or be per-launch only?**
   - Known: criterion 2 says the script is never mutated at launch, so overrides are
     inherently transient.
   - Unclear: whether the UI should remember the last-used override per profile.
   - Recommend: per-launch only in Phase 4; persistence is a separate decision that pulls in
     a new state file. Raise it in `/gsd-discuss-phase`.

2. **Does the generator's missing `-d` get fixed here (A3)?**
   - Recommend: yes, in 04-01, one token, block already open. Needs a yes/no from Josh.

3. **Is the hf.co browse-card XSS (`app.py:8655`) in scope for criterion 5?**
   - Known: it is the highest-severity of the enumerated sites and the same one-line fix.
   - Recommend: fix it in 04-03 — the `esc()` helper lands there anyway and leaving the
     worst site unfixed while fixing lesser ones is indefensible.

4. **Option A vs B for surfacing derived values (§Criterion 3).**
   - Recommend A (header comments). Needs confirmation because it makes the header a
     versioned data format.

## Sources

### Primary (HIGH)
- `app.py` in this tree at `911bd14` — line references throughout, read directly
- `profiles/llama.cpp/start_gguf_hauhaucs_qwen3.8-27b-aggressive-mtp.sh` — the working precedent
- `profiles/vLLM/*.sh` (12 files) — current on-disk state
- `tests/conftest.py`, `tests/test_vllm_profile_generation.py` — suite conventions
- `git show 079917b`, `git log --oneline -12` — the commit that motivated the premise check
- Live probe on this box: systemd-run scope vs service env inheritance (transcript in §Criterion 2)
- Live run: `python3 -m pytest tests -q` → 337 passed in 0.58s
- `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/HANDOFF.json`, `config.example.json`

### Notes on gaps
- `.planning/REQUIREMENTS.md` **does not exist** in this tree. REQ-06's text above is taken
  from the ROADMAP Phase 4 goal line, not from a requirements file. [CITED: ROADMAP.md]
- No CONTEXT.md exists for Phase 4 yet, so nothing constrains scope — the four Open
  Questions are the natural agenda for `/gsd-discuss-phase 4`.
- No `./CLAUDE.md` in the repo; constraints above come from `/home/josh/CLAUDE.md`.

## Metadata

**Confidence breakdown:**
- Criterion-2 transport verdict: **HIGH** — measured on this box, plus a live production code path that would already be broken if it were false
- Emission points / call graph: **HIGH** — read directly, line-referenced
- Criterion 3 data-loss analysis: **HIGH** — `grep` confirms `warnings` has exactly one reader
- Criterion 4 sizing: **HIGH** on "no parser exists"; **MEDIUM** on the regex being sufficient (A1)
- XSS enumeration: **HIGH** for the listed sites; **MEDIUM** that the list is exhaustive across all 86 `innerHTML` sites (filtered by keyword, not audited one-by-one)

**Research date:** 2026-08-20
**Valid until:** ~2026-09-19, or the next commit touching `_build_vllm_profile_script`,
`_engine_start`, or `_launch_argv` — whichever is first.
