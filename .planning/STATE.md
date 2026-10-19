---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-19T00:00:00.000Z"
last_activity: 2026-10-19 -- Phase 5 (admission truth) merged and deployed to gb10
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
**Current focus:** Phase 5 is deployed; remaining work is live util measurement and the open items below. Phase 6 is next on the roadmap but optional.

## Current Position

Updated 2026-10-19 against `origin/main` 2315f4f and a read-only check of gb10. Older
narrative is in git history (`git log -- .planning/STATE.md`).

- Phases 1, 1.1, 2, 3, 4 and 5 are complete.
- **Phase 5 (admission truth)** shipped in #23 and #24 (follow-ups #22, #25) and is
  deployed to gb10 at 1c2d785. Per-criterion status and the still-open items (util
  measurement for five profiles, refusal smoke test, Qwen3.6 label check) are in ROADMAP.md.
- **Phase 6 (verifiable research capture)** has not started. It is independent of the launch
  path and low value unless the Advisor's recommendations are relied on.
- Suite: 673 passed, 4 skipped (`.venv/bin/python -m pytest -q`, about 2.5 s). The skips
  are host-dependent tests (litellm config, which also skip in CI).
- CI (`.github/workflows/tests.yml`) runs pytest plus a ruff E9/F correctness lint (#19).

## Off-roadmap work since Phase 4 (on main)

Image drift checker (cb30594), unified background-job runner (b1ba906), dashboards/homelab
tabs (c08019b, 75090de), dashboard frontend extracted to `dashboard.html` (e8ab05e), pinned
vLLM model on boot + load-log tail (db32241), Qwen Flash host-headroom fix (6730286).

## Open items carried forward

- **litellm file-vs-ConfigMap drift is still unguarded.** `_sync_litellm` pushes the file
  into the ConfigMap, but nothing compares the two. (HANDOFF task 21)
- Next candidates, in order: measure utils for the five refused profiles; set an API key
  on gb10; litellm drift guard; start the `app.py` split (docs/app-py-modularization.md).
- The running service is bound to a Tailscale IP with no API key and
  `MODEL_MANAGER_ALLOW_UNAUTH=1` (deploy unit, acknowledged tailnet trust model). Every
  mutating endpoint is therefore open to tailnet peers.

## Environment Notes

- **Live state, 2026-10-19 (gb10):**
  - **Service:** `dgx-model-manager.service` (systemd --user) restarted after pulling
    1c2d785 (Phase 5 + the preflight fix). Engines run in their own `systemd-run --scope`
    and were unaffected.
  - **Model:** `vllm_node` serves `qwen38-flash-dgx`, labelled
    `dgx.profile=hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid`, `--max-model-len 262144`,
    `--gpu-memory-utilization 0.76`.
  - **Checkout:** gb10 is one merge behind `main` (docs, ruff config, `.bak` removal);
    no restart is needed for those.
  - **Memory:** MemTotal 121.6 GiB; MemAvailable about 16 GiB with this model up.

- **Page cache is charged against `--gpu-memory-utilization` on this box.** CUDA reports
  MemFree, not MemAvailable, so `usable = MemTotal * util - (MemTotal - MemFree)`. Drop
  caches before a large launch (`POST /api/vllm/reclaim-cache`).

- Editing `app.py` does NOT restart the running service. Deploying requires
  `systemctl --user restart dgx-model-manager.service` and Josh's approval. Engines run in
  their own `systemd-run --scope`, so a restart does not stop them.

- **Several Claude sessions may work in this repo concurrently.** Commit by explicit path
  and never use `git add -A`.
