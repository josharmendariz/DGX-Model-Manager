# Research: Phase 1 — Unbreak the load path

**Source:** Codex `gpt-5.6-sol` xhigh audit of the vLLM surface (2026-08-02, 307k tokens),
with every finding below independently re-verified against the live box. Claims that could
not be verified are marked as such.

## Finding 1 — `_HF_XFER` is undefined: every HF download fails (REQ-01)

**Location:** [app.py:2202](../../app.py) inside the `_HF_DOWNLOAD_SCRIPT` string.

**Verified.** Running the embedded worker directly:

```
{"status": "starting", "repo": "Qwen/Qwen2.5-0.5B-Instruct"}
NameError: name '_HF_XFER' is not defined     rc=1
```

**Root cause.** Commit `ac9b2e7` ("Enable hf_transfer for much faster model downloads")
introduced `_HF_XFER`. A later rewrite replaced hf_transfer with the huggingface_hub 1.x
Xet path (`importlib.util.find_spec("hf_xet")` → `HF_XET_HIGH_PERFORMANCE`) and removed the
definition but left the `if _HF_XFER:` use at line 2202. The only surviving references are
the comment block at 2186–2191 and the orphaned use.

**Why it stayed hidden.** The `NameError` fires *before* the worker enters its `try`, so the
worker's own error handler never runs. The parent never checks `proc.returncode` and turns
stderr into `log` events, which the main HF UI ignores — the progress area just spins.

**Implementation notes.**
- Delete the `if _HF_XFER:` branch (the Xet env var above it is the actual mechanism), or
  bind a correctly-named `has_xet` from the `find_spec` result and report that.
- Move all worker initialization inside the error handler.
- Check `returncode` in the parent and always emit exactly one terminal `complete`/`error`.
- Test by executing `app._HF_DOWNLOAD_SCRIPT` in a subprocess — no network needed to prove
  startup succeeds past the `starting` event.

## Finding 2 — image/entrypoint mismatch: generated profiles cannot launch (REQ-02)

**Location:** [app.py:2477-2483](../../app.py).

**Verified:**
```
$ docker image inspect eugr/spark-vllm:latest --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
["/opt/nvidia/nvidia_entrypoint.sh"] null
```

`nvidia_entrypoint.sh` execs its arguments. The generator prepends `vllm serve` **only when
`is_gpt_oss`**, treating it as a model-family property. But `config.json` sets
`vllm.image: eugr/spark-vllm:latest` for *all* models, so every generated non-gpt-oss script
passes `--model` as the container command with nothing to exec it.

Confirmed in the committed artifacts: `start_hf_qwen_qwen3.6-35b-a3b-fp8.sh` (generated) goes
straight from the image to `--model`, while the hand-written `start_hf_qwen_qwen3-8b.sh`
correctly includes `vllm serve`.

**Implementation notes.**
- Entrypoint semantics belong to the *image*, not the model. Robust form:
  `docker run --entrypoint vllm <image> serve ...`, which works for both image shapes.
- Alternatively a per-image `needs_serve_subcommand` flag in the config `vllm` block.
- Pin the image by digest or immutable tag; `:latest` can silently change entrypoint shape.
- Testable offline: assert the generated script text, no launch required.

## Finding 3 — unauthenticated shell injection into an executed script (REQ-03)

**Location:** [app.py:221-231](../../app.py) (auth), [app.py:2445-2500](../../app.py) (generation).

**Verified, both halves.**

Auth: `config.json` has no `api_key` and `app.host` is `100.115.54.83` (tailnet). `verify_auth`
opens with `if not _API_KEY_HASH: return` — its own docstring says "No-op when no key is
configured." So every `Depends(verify_auth)` endpoint is unauthenticated on the tailnet.

Injection: `model_name` from `POST /api/vllm/profiles/from-hf` reaches
`--served-model-name "{info['name']}"` unquoted. Generated with
`model_name='evil$(id > /tmp/dgx-pwn-proof)'`:

```
--served-model-name "evil$(id > /tmp/dgx-pwn-proof)" "evil$(id > --tmp--dgx-pwn-proof)" vllm-active \
```

The substitution survives verbatim into a file later run via `bash <script>`, and `$(...)`
executes inside double quotes. (String generation only — nothing was executed.)

**Why the existing guards don't cover it.** `_safe_profile_slug` sanitizes the *filename* and
the mount path; `_path_under` proves directory containment. Neither is shell escaping. Note
`_profile_model_info` overwrites `model_name` for repos with a `models--*` ancestor, so the
reachable path is a **flat/custom dir** (e.g. `/mnt/models/...`) where the caller-supplied
name survives.

**Implementation notes.**
- `shlex.quote` every dynamic atom interpolated into the script; reject control characters.
- Validate served names against a strict alias grammar.
- Refuse non-loopback startup when no API key is set (or force-generate one and log it).
- Longer term, argv/Docker-API execution removes the class entirely — but that conflicts with
  the "scripts are the user-facing contract" constraint, so quoting is the Phase 1 fix.

## Finding 4 — committed DeepSeek profiles predate the mount fix

`start_hf_deepseek-ai_deepseek-r1-distill-qwen-{14b,32b}.sh` still mount only
`snapshots/<rev>`, whose entries are relative symlinks into `../../blobs/`. Inside the
container every weight file dangles and vLLM aborts with `Invalid repository ID or local
directory specified`. Commit `6ffc899` fixed the generator but did not migrate existing
scripts, and there is no regenerate action for profiles already marked as having scripts.

**Diagnostic idiom for this class:** `docker run --rm -v <mount> busybox cat <file>` separates
mount-scope from download corruption instantly.

## Constraints for this phase

- **vLLM is intentionally down** (box is running Helix training). Every success criterion must
  be checkable by generating/inspecting script text, running the download worker, or unit
  tests — never by launching a model.
- Keep changes inside `app.py` and `profiles/`; no file splits (upstream mergeability).
- Existing suite is 59 tests across `tests/`; `test_vllm_profile_generation.py` is the natural
  home for generator assertions.
- Editing `app.py` does not restart the running service.

## Unverified / open

- Whether `/v1/models` returns `vllm-active` first, which decides how badly Finding H1
  (reclaim-credit mismatching) bites. Cannot check with vLLM down. Deferred to Phase 5.
- Live end-to-end launch of a regenerated profile. Deferred until vLLM is back up.
