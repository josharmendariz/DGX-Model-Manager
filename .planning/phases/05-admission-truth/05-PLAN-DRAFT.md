# Phase 5: Admission truth — implementation plan (DRAFT)

**Status:** draft for Josh's review. Nothing in this file has been implemented.
**Written:** 2026-10-18, against `origin/main` 56bdfb5. Line numbers refer to that commit;
they will move, so match on function names.
**Requirement:** REQ-07 (ROADMAP.md, Phase 5 success criteria 1–5)

## 0. What is actually still open

Each item was verified against the code. Criteria are numbered as in ROADMAP.md.

| # | Criterion | State | Evidence |
|---|-----------|-------|----------|
| 1 | Profiles declare an executor budget (`util × MemTotal`) | **open** | Generator writes `# VRAM:` = `size_gb × {1.05,1.15,1.35} + 12` (`_profile_model_info`, app.py ~5160; emitted ~5469). Admission reads that header (`_parse_script_meta` ~499). |
| 2 | Consistent binary GiB | **open** | `_get_available_memory_gb` is GiB (`kB/1024/1024`, ~327). Weights/KV in `_derive_launch_spec` (~1159, ~1179) and `_profile_model_info` (~1252, ~1264) are decimal GB (`/1e9`). `_overridden_vram_gb` mixes them with a hard-coded `pool_gb = 121.0` (~3626). |
| 3 | Reclaim target by `dgx.profile` label | **mostly done** (9410656) | `_running_profile_vram_credit` (~3159) reads the label first, then falls back to substring/fuzzy name matching. `start_hf_qwen_qwen3.6-35b-a3b-fp8.sh` (recipe-backed) still sets no label. |
| 4 | Two concurrent starts cannot both pass admission | **open** | `_engine_start` (~3647) runs admission → `Popen` with no lock and no in-flight record. The only asyncio locks in app.py are `_SITES_LOCK`, `_refresh_lock` and `_image_check_lock`. |
| 5 | Death during load surfaces exit code / OOM | **partial** | `_container_state` (~4366) reads `State.ExitCode`. `_load_progress_events` puts it on the terminal SSE frame (~4508), and `dashboard.html` (~3434) shows it. `State.OOMKilled` is never read. The verdict exists only while a client watches the stream. No test covers the exit-code path. |

### Why criterion 1 matters now (measured from the committed scripts)

The table compares each script's effective util, read with `_parse_script_flags` and
`_parse_templated_defaults`, to its `# VRAM:` header. MemTotal on gb10 is 121.6 GiB.

| profile | `# VRAM:` | util | util × 121.6 GiB |
|---|---|---|---|
| start_hf_qwen_qwen3-8b | 20 | 0.75 | **91.2** |
| start_hf_qwen3-vl-4b-fp8 | 20 | 0.75 | **91.2** |
| start_hf_qwen_qwen3-14b | 34 | 0.75 | **91.2** |
| start_hf_qwen2.5-14b-instruct-gptq-int8 | 34 | 0.75 | **91.2** |
| start_hf_deepseek-…-qwen-14b | 52 | 0.75 | **91.2** |
| start_hf_openai_gpt-oss-120b | 81 | 0.75 | 91.2 |
| start_hf_qwen_qwen3.8-27b-fp8 | 48 | 0.49 | 59.6 |
| start_hf_deepseek-…-qwen-32b | 100 | 0.75 | 91.2 (header overstates) |
| start_hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid | 92 | 0.76 | 92.4 (ok) |
| start_qwen3_coder_next / qwen3_next_80b | 67 | 0.55 | 66.9 (ok) |

Today, admission would pass `qwen3-8b` with about 28 GiB available, but vLLM then tries to
reserve 91 GiB. Hand-written profiles happen to be right because their authors used
`util × 121`. Generated profiles are wrong.

## 05-01: Executor-budget metadata and unit-consistent admission

### Design

1. **One unit.** Admission works only in GiB.
   - Add `_GIB = 1024 ** 3` and `_mem_total_gib()`, which reads MemTotal from `/proc/meminfo`.
     Reuse the read in `_cuda_visible_memory`; do not add a third parser.
   - Rename `_get_available_memory_gb` → `_get_available_memory_gib` and keep a thin alias
     for one release, since tests monkeypatch it.
   - Rename `_VRAM_SAFETY_MARGIN_GB` and `_VRAM_UNKNOWN_MIN_HEADROOM_GB` to `…_GIB`
     suffixes. The values are unchanged; they were always compared against a GiB number.
   - User-facing 409 messages say "GiB".
2. **Footprint = what vLLM reserves.** Add a pure
   `_profile_footprint_gib(profile, script_text, override_env, mem_total_gib) -> (gib, source)`
   with this precedence:
   1. `VLLM_GPU_MEMORY_UTILIZATION` override → `util × mem_total_gib`, source
      `override-util`
   2. the script's effective util: a literal `--gpu-memory-utilization`, else the
      `${VLLM_GPU_MEMORY_UTILIZATION:-N}` default (both parsers already exist) →
      `util × mem_total_gib`, source `script-util`
   3. the `# VRAM:` header, now documented as GiB → source `header`
   4. none of the above → `None`, which keeps today's unknown-need headroom floor

   A context-only override (`VLLM_MAX_MODEL_LEN`) **does not change the footprint**,
   because vLLM reserves `util × total` whatever the context. `_overridden_vram_gb`'s
   `_derive_launch_spec` branch therefore leaves admission. Its "no readable config.json →
   400 unless force" rule stays only if the util path needs it (it does not), so that
   rule can be relaxed. `_derive_launch_spec` stays pure and keeps `pool_gb` as a
   parameter; callers pass `_mem_total_gib()` instead of the literal `121.0` (~3626, ~5346).
3. **llama.cpp.** There is no util knob. Recipe `vram_gb` from `config.json` and the
   `# VRAM:` header remain the footprint and are treated as GiB. The plan does not try to
   derive a footprint for GGUF.
4. **Reclaim credit uses the same function.** `_running_profile_vram_credit` currently
   credits the running profile's header. It should credit
   `_profile_footprint_gib(running_profile, …)` so the credit and the need use the same
   quantity. The long-term version reads the running container's actual
   `--gpu-memory-utilization` from `docker inspect .Config.Cmd`; that is optional and can
   be left out.
5. **Generator.** `_vllm_script_preamble` writes `# VRAM: <round(util × MemTotal_gib)>` when
   a util is known, and keeps the weights estimate only when util is `None` (recipe/legacy).
   Existing scripts are **not** rewritten: admission prefers the script's util (step 2.2),
   so their stale headers stop mattering. Preflight gains a `warn` check when
   `|header − util × total| > 5 GiB`, so the drift is visible.

### Files and functions

- `app.py`:
  - new: `_GIB`, `_mem_total_gib`, `_profile_footprint_gib`
  - changed: `_vram_admission_check`, `_running_profile_vram_credit`, `_engine_start`
    (calls the footprint function instead of `_overridden_vram_gb`), `_overridden_vram_gb`
    (removed or reduced to the util path), the `pool_gb` call sites, `_profile_model_info`
    / `_vllm_script_preamble` header emission, and `_preflight_static` (new drift warn)
- `tests/test_vram_admission.py`: update the GB → GiB message assertions and the
  override-footprint tests (`test_override_before_admission_raises_the_seen_footprint`,
  `test_underivable_override_requires_force_instead_of_stale_header` — the second
  changes meaning).
- `tests/test_engine_start.py`, `tests/test_vllm_profile_generation.py` (header value).

### Tests to add

- Table test over every committed `profiles/*/start_*.sh`: `_profile_footprint_gib`
  returns `util × total` where util parses, and `header` otherwise. Qwen3.6 recipe and
  llama.cpp go to `header`.
- `qwen3-8b`-shaped profile (header 20, util 0.75) with MemAvailable 40 GiB is **rejected**.
  This is the regression the phase exists for.
- Unit consistency: for each source, the footprint and MemAvailable come from one GiB
  conversion, checked with a fake `/proc/meminfo`.
- A context-only override does not change the footprint. A util override does.
- Generated header equals `round(util × total)` when util is derived.

## 05-02: Launch lock, label reclaim, and failure surfacing

### A. Launch lock and in-flight reservation (criterion 4)

A mutex around `admission → Popen` is **not enough**. `_engine_start` returns right after
`Popen`, and the new engine's memory appears over the next 1–5 minutes, so a second start
during that window sees MemAvailable that has not dropped yet.

- `_LAUNCH_LOCK = asyncio.Lock()` (module level). DMM runs one uvicorn worker
  (`uvicorn.run(app, …)`, no `workers=`), so an in-process lock is sufficient. Note this
  assumption in the code.
- `_INFLIGHT: dict[engine_key, Launch]`, where `Launch` holds profile id, footprint GiB,
  container name, `started` (monotonic) and the container id once known.
- In `_engine_start`, under the lock:
  1. Prune settled entries. An entry is settled when its engine health is OK, its
     container has exited, or it is older than `_INFLIGHT_TTL_S` (15 min).
  2. If the **same** engine has an unsettled launch, return 409 "launch already in
     progress" (`force` overrides). Two scripts racing on `docker rm -f vllm_node` +
     `docker run --name vllm_node` is the failure this prevents.
  3. Run admission with `available − Σ(unsettled reservations of other engines)`.
  4. `Popen`, then record the reservation.
- Pinned autostart (`_pinned_autostart_once`) goes through `_engine_start`, so it is
  covered with no extra code.
- Expose `_INFLIGHT` read-only on `/api/{key}/status` (`launching: {...}`) so the UI can
  disable the Start button.

### B. Label reclaim to completion (criterion 3)

- Fix a latent bug in `_running_profile_vram_credit`: when the label resolves to a profile
  with no footprint, the loop `continue`s and falls through to name matching, which can
  credit a **different** profile. A label hit must be final: return that profile's
  footprint, or `(0, id)`.
- Decide the fallback (**Josh**):
  - Option (a): keep the served-name fallback, but only for containers with no
    `dgx.profile` label, and log a warning.
  - Option (b): remove the fallback entirely. Unlabeled → 0 credit, which is
    conservative and may refuse swaps away from Qwen3.6.

  Recommendation: (a) now, (b) once the Qwen3.6 recipe profile carries the label. That
  needs `run-recipe.sh` (in `~/spark-vllm-docker`, outside this repo) to accept extra
  `docker run` labels, or the profile to wrap the recipe's container creation.
- Add a test: every committed vLLM/llama.cpp profile except a named allowlist sets
  `--label "dgx.profile=<id without start_>"`, and the value matches its filename.

### C. Failure surfacing (criterion 5)

- `_container_state`: read `{{.State.OOMKilled}}` as well (format
  `Id|Status|ExitCode|OOMKilled`).
- Terminal `failed` frame: add `oom_killed` and a plain-language `cause`:
  - `OOMKilled=true` → "killed by the kernel OOM killer"
  - exit 137 without OOMKilled → "SIGKILL (likely memory pressure or docker kill)"
  - other non-zero exit codes → the existing log-derived reason
- `dashboard.html` (~3434): render `cause` and `oom_killed` next to the exit code.
- Persist the last terminal verdict per engine in memory (`_LAST_LAUNCH[engine_key]`),
  written by the in-flight pruning in A. That way the verdict reaches `/status` even when
  no client watched the stream. Persisting across restarts is optional and can be left out.

### Tests to add

- Concurrency: two `_engine_start` coroutines under `asyncio.gather`, with fake
  MemAvailable that fits exactly one launch. Exactly one succeeds and the other gets 409.
  Repeat with different engines (the reservation subtracts) and with the same engine
  (in-progress 409).
- A reservation is released when health turns OK, when the container exits, or at TTL.
- `force=true` bypasses the in-progress check but still records a reservation.
- `_container_state` parses the 4-field format; a malformed `OOMKilled` → `None`.
- The progress stream on a container that exits 137 with `OOMKilled=true` emits one
  `failed` frame with `exit_code=137`, `oom_killed=true` and a cause.
- A label hit on a profile with no footprint does not fall through to name matching.

## Risks

- **Concurrent edits to app.py.** Other sessions edit app.py, so:
  - Branch from fresh `origin/main` for each plan.
  - Commit by explicit path (`git add app.py tests/…`), **never `git add -A`**.
  - Rebase before pushing, and keep 05-01 and 05-02 as separate PRs.
- **Admission becomes much stricter.** With criterion 1 fixed, several generated
  profiles (util 0.75) will be refused when any other engine is up. That is correct, but
  it is a behaviour change: llama.cpp + small-vLLM co-residency that "worked" by
  under-counting will now need lower util in those scripts, or `force`. List the affected
  profiles in the PR. Their util values are an operator decision.
- **MemAvailable vs CUDA-free.** Admission still uses MemAvailable, which counts page
  cache as free, while vLLM's budget is charged for page cache. That gap is already
  handled by preflight's memory check and `reclaim-cache`, not by admission. Do not merge
  the two in this phase.
- **Unmerged branch `agent/dgx-mm/cd3deb561c86`** (b6d1f00, "require profile preflight
  before launch", also on origin) wraps `/start` in a preflight gate. If it lands too,
  preflight's page-cache `fail` (≥ 20 GiB reclaimable) would block ordinary starts. Josh
  needs to decide its fate before 05-02 touches the same route.
- **OOMKilled semantics on GB10.** It is not established whether docker sets
  `OOMKilled=true` when the **host-global** OOM killer (not a cgroup limit) kills a
  process inside the container. The exit-137 fallback covers either outcome.

## Live checks and approvals needed (gb10)

Read-only, any time:
- Run the footprint table above against gb10's actual scripts and live MemTotal (a
  Python one-liner over `profiles/`; no writes).
- `docker version`, and confirm cgroup v2 (`stat -fc %T /sys/fs/cgroup`), for the
  OOMKilled question.
- Inspect a known OOM-killed container's `State`, if any exists in `docker ps -a`.

Needs Josh's approval:
- **Deploying** each PR. Run `git pull --ff-only` in `/home/josh/DGX-Model-Manager`, then
  `systemctl --user restart dgx-model-manager.service`. Engines survive the restart
  (`systemd-run --scope`), but the UI blips.
- **Post-deploy smoke.** A start request that admission should refuse (e.g. qwen3-8b while
  flash-next is up) must return 409 **without** launching. Do not run any start that
  would actually replace the serving model unless Josh asks.
- **Any util change** to the committed profiles that the stricter admission exposes.

## Out of scope

Phase 6, the app.py split, the litellm ConfigMap drift guard (HANDOFF task 21), and
llamacpp_node supervision.
