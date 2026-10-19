---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-18T00:00:00-05:00"
last_activity: 2026-10-18 -- API key verified live; authentication cleanup and drift check prepared locally
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
**Current focus:** Finish authentication hardening and deploy the verified startup drift check after approval. Profile tuning needs a separate model-switching window. Phase 6 remains optional.

## Current Position

Updated 2026-10-18 (America/Chicago) against GitHub `main` 2315f4f and live read-only
checks of gb10. PR #26 (`docs/phase5-status`, 9d213c8) is still open; this follow-up
builds on its Phase 5 reconciliation. The dated Claude handoff is historical evidence.

- Phases 1, 1.1, 2, 3, 4 and 5 are complete.
- **Phase 5 (admission truth)** shipped in #23 and #24 (follow-ups #22, #25) and is
  deployed to gb10 at 1c2d785. Per-criterion status and the still-open items (util
  measurement for five profiles, refusal smoke test, Qwen3.6 label check) are in ROADMAP.md.
- **Phase 6 (verifiable research capture)** has not started. It is independent of the launch
  path and low value unless the Advisor's recommendations are relied on.
- Suite: 673 passed, 4 skipped (`.venv/bin/python -m pytest -q`, about 2.5 s). The skips
  are host-dependent tests (litellm config, which also skip in CI).
- Candidate implementation `30a5c1d` on `fix/auth-default-and-litellm-drift`:
  **692 passed, 4 skipped**, Ruff 0.14.0 passed, and systemd accepted the template
  plus authentication drop-in. This candidate has not been published or deployed.
- CI (`.github/workflows/tests.yml`) runs pytest plus a ruff E9/F correctness lint (#19).

## Off-roadmap work since Phase 4 (on main)

Image drift checker (cb30594), unified background-job runner (b1ba906), dashboards/homelab
tabs (c08019b, 75090de), dashboard frontend extracted to `dashboard.html` (e8ab05e), pinned
vLLM model on boot + load-log tail (db32241), Qwen Flash host-headroom fix (6730286).

## Open items carried forward

- **Authentication is active.** Josh installed a key through Settings. The live API
  reports `api_key_set: true`; the persisted key hash is present. Protected reads reject
  both missing and invalid tokens with 401. The token/hash was not printed. A successful
  authenticated browser read still needs operator confirmation.
- **Remove the unauthenticated exception after approval.** The running process still has
  `MODEL_MANAGER_ALLOW_UNAUTH=1`, but it does not bypass a configured key. Candidate
  `deploy/require-auth.conf` removes it without replacing the customized installed unit.
  Installing the drop-in, reloading systemd, and restarting DMM require approval.
- **LiteLLM startup drift check is prepared, not deployed.** The existing
  `_restart_litellm_backend` explicitly syncs the file into the ConfigMap. The candidate
  adds a read-only startup comparison with bounded Kubernetes lookup and redacted
  diagnostics. The live source and ConfigMap currently match; no sync was performed.
- Next: approve the authentication-only rollout; publish/review the candidate after
  #26; approve its merge and deployment separately. Profile tuning, refusal smoke,
  modularization, and Phase 6 remain outside this session's implementation scope.

## Environment Notes

- **Live state, 2026-10-18 (gb10):**
  - **Service:** `dgx-model-manager.service` is active; checkout is clean at 1c2d785.
    This session did not deploy, restart, or launch/stop an engine.
  - **Inference:** direct `/v1/models` responds with `qwen3.8-flash-next`,
    `nvidia--Qwen3.8-Flash-Next-NVFP4`, and `vllm-active`. These are observed API IDs;
    the handoff's `qwen38-flash-dgx` label was not used as current model proof.
  - **Work:** zero DMM background jobs running; vLLM `launch.in_flight` is null.
    Refresh both immediately before any approved restart.
  - **LiteLLM:** Kubernetes mode enabled. Source
    `/home/josh/dgx-stack/litellm-config-prod.yaml` is YAML-equivalent to
    `llm-inference/litellm-config`, key `config.yaml`. No contents were printed.
  - **Unit:** installed user unit differs from the repository template; no drop-ins
    were installed. Preserve it and use the prepared minimal authentication drop-in.

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
