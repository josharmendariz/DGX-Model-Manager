---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-18T00:00:00-05:00"
last_activity: 2026-10-18 -- approved authentication and drift-check rollout deployed and verified on gb10
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 8
  completed_plans: 7
  percent: 86
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-02)

**Core value:** Switching the active vLLM model must be safe and correct.
**Current focus:** Authentication hardening and startup drift detection are deployed and verified. Profile tuning needs a separate model-switching window. Phase 6 remains optional.

## Current Position

Updated 2026-10-18 (America/Chicago) against GitHub `main` c005e60 and live checks
of gb10. PRs #26, #27 and #28 are merged. The dated Claude handoff is historical evidence.

- Phases 1, 1.1, 2, 3, 4 and 5 are complete.
- **Phase 5 (admission truth)** shipped in #23 and #24 (follow-ups #22, #25) and is
  included in gb10's release c005e60. Per-criterion status and the still-open items (util
  measurement for five profiles, refusal smoke test, Qwen3.6 label check) are in ROADMAP.md.
- **Phase 6 (verifiable research capture)** has not started. It is independent of the launch
  path and low value unless the Advisor's recommendations are relied on.
- Released implementation (#27) and startup-journal follow-up (#28):
  **692 passed, 4 skipped**, Ruff 0.14.0 passed, and systemd accepted the template
  plus authentication drop-in. The merged tree matches the tested candidate.
- CI (`.github/workflows/tests.yml`) runs pytest plus a ruff E9/F correctness lint (#19).

## Off-roadmap work since Phase 4 (on main)

Image drift checker (cb30594), unified background-job runner (b1ba906), dashboards/homelab
tabs (c08019b, 75090de), dashboard frontend extracted to `dashboard.html` (e8ab05e), pinned
vLLM model on boot + load-log tail (db32241), Qwen Flash host-headroom fix (6730286).

## Open items carried forward

- **Authentication is active.** Josh installed a key through Settings. The live API
  reports `api_key_set: true`; the persisted key hash is present. Protected reads reject
  both missing and invalid tokens with 401. The token/hash was not printed. Josh confirmed
  System Overview works in his browser before the approved restart. Refresh the browser
  after rollout to confirm the operator session still works.
- **Unauthenticated exception removed.** The reviewed `require-auth.conf` drop-in is
  installed; the running process has no `MODEL_MANAGER_ALLOW_UNAUTH` variable. The
  customized unit and saved configuration are byte-for-byte unchanged. DMM was
  restarted once with Josh's explicit approval; no engine was restarted.
- **LiteLLM startup drift check is deployed.** The existing
  `_restart_litellm_backend` explicitly syncs the file into the ConfigMap. The release
  adds a read-only startup comparison with bounded Kubernetes lookup and redacted
  diagnostics. The new invocation journal reports `LiteLLM config drift startup result: match`.
  No ConfigMap sync or LiteLLM rollout was performed.
- Next: profile tuning in an approved model-switching window. Refusal smoke,
  modularization, and Phase 6 remain outside this session's implementation scope.

## Environment Notes

- **Live state, 2026-10-18 (gb10):**
  - **Service:** `dgx-model-manager.service` is active at deployed code c005e60.
    Git fast-forwarded from 1c2d785; only DMM was restarted after validation.
  - **Inference:** direct `/v1/models` responds with `qwen3.8-flash-next`,
    `nvidia--Qwen3.8-Flash-Next-NVFP4`, and `vllm-active`. These are observed API IDs;
    the handoff's `qwen38-flash-dgx` label was not used as current model proof.
  - **Work:** zero DMM background jobs running; vLLM `launch.in_flight` is null.
    Refresh both immediately before any approved restart.
  - **LiteLLM:** Kubernetes mode enabled. Source
    `/home/josh/dgx-stack/litellm-config-prod.yaml` is YAML-equivalent to
    `llm-inference/litellm-config`, key `config.yaml`. No contents were printed.
  - **Unit:** the customized installed unit is unchanged. The new authentication
    drop-in removes the exception. DMM still listens only on `100.115.54.83:8090`.
  - **Inference continuity:** container `4c4f68ed11cd` retained its full identity and
    `2026-10-16T03:18:53.41314325Z` start time. Direct inference returned 200 both on
    GB10 and from controlplane after the DMM restart.
  - **Recovery:** private configuration/unit backups and nonsecret verification
    evidence are under GB10's
    `~/.local/state/dgx-model-manager/rollouts/20261019T022354Z-auth-and-drift`.

Historical hardware sizing/profile settings in the Claude handoff were not remeasured
this session. Do not use them as current memory or capacity measurements.

- **Page cache is charged against `--gpu-memory-utilization` on this box.** CUDA reports
  MemFree, not MemAvailable, so `usable = MemTotal * util - (MemTotal - MemFree)`. Drop
  caches before a large launch (`POST /api/vllm/reclaim-cache`).

- Editing `app.py` does NOT restart the running service. Deploying requires
  `systemctl --user restart dgx-model-manager.service` and Josh's approval. Engines run in
  their own `systemd-run --scope`, so a restart does not stop them.

- **Several Claude sessions may work in this repo concurrently.** Commit by explicit path
  and never use `git add -A`.
