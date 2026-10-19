---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-18T00:00:00.000Z"
last_activity: 2026-10-18 -- state reconciled against code and live gb10 (read-only); Phase 5 plan drafted
progress:
  total_phases: 7
  completed_phases: 5
  total_plans: 8
  completed_plans: 5
  percent: 63
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-02)

**Core value:** Switching the active vLLM model must be safe and correct.
**Current focus:** Phase 5 (admission truth). A plan draft exists; it has not been executed.

## Current Position

Reconciled 2026-10-18 against `origin/main` 56bdfb5. Only the facts below were verified;
older narrative is in git history (`git log -- .planning/STATE.md`).

- Phases 1, 1.1, 2, 3 and 4 are complete. Phase 4 was closed in 0250f22 and is on `main`.
  The 2026-08-20 "cut a 41-commit PR" step in HANDOFF.json is obsolete because that work
  reached `main` through later PRs.
- **Phase 5 (admission truth)** is not formally planned or executed, but parts shipped
  without a plan:
  - label-based reclaim identification (9410656, `dgx.profile`)
  - overridden-footprint admission (8639fe0, done under 04-01)

  Still open: executor-budget metadata, consistent GiB units, a launch lock / in-flight
  reservation, and OOM surfacing. Per-criterion status is in ROADMAP.md.
  The draft plan is `phases/05-admission-truth/05-PLAN-DRAFT.md`.
- **Phase 6 (verifiable research capture)** has not started. It is independent of Phase 5.
- Suite: 540 passed, 3 skipped (`.venv/bin/python -m pytest -q`, about 2 s). The 3 skips
  are the host-dependent litellm config tests, which also skip in CI.
- CI (`.github/workflows/tests.yml`) runs pytest. A ruff correctness-lint job is proposed
  separately.

## Off-roadmap work since Phase 4 (on main)

Image drift checker (cb30594), unified background-job runner (b1ba906), dashboards/homelab
tabs (c08019b, 75090de), dashboard frontend extracted to `dashboard.html` (e8ab05e), pinned
vLLM model on boot + load-log tail (db32241), Qwen Flash host-headroom fix (6730286).

## Open items carried forward

- **litellm file-vs-ConfigMap drift is still unguarded.** `_sync_litellm` pushes the file
  into the ConfigMap, but nothing compares the two. (HANDOFF task 21)
- The running service is bound to a Tailscale IP with no API key and
  `MODEL_MANAGER_ALLOW_UNAUTH=1` (deploy unit, acknowledged tailnet trust model). Every
  mutating endpoint is therefore open to tailnet peers.

## Environment Notes

- **Live state, read-only check on 2026-10-18 (gb10):**
  - **Service:** `dgx-model-manager.service` (systemd --user) has been active since
    2026-10-13 20:55:53 UTC. `app.py` and `dashboard.html` were last modified at
    20:55:52, so the running code matches the checkout.
  - **Model:** `vllm_node` serves `qwen38-flash-dgx`, labelled
    `dgx.profile=hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid`, with `--max-model-len 262144`
    and `--gpu-memory-utilization 0.76`. The container was started 2026-10-16T03:18Z.
  - **Checkout:** gb10's checkout is one commit ahead of origin (e77077a, native 262k
    for that profile). That commit is proposed to origin as-is; it is already what's
    running.
  - **Memory:** MemTotal is 121.6 GiB. MemAvailable was about 16 GiB with this model up.

- **Page cache is charged against `--gpu-memory-utilization` on this box.** CUDA reports
  MemFree, not MemAvailable, so `usable = MemTotal * util - (MemTotal - MemFree)`. Drop
  caches before a large launch (`POST /api/vllm/reclaim-cache`).

- Editing `app.py` does NOT restart the running service. Deploying requires
  `systemctl --user restart dgx-model-manager.service` and Josh's approval. Engines run in
  their own `systemd-run --scope`, so a restart does not stop them.

- **Several Claude sessions may work in this repo concurrently.** Commit by explicit path
  and never use `git add -A`.
