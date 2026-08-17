---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: phase_complete
last_updated: "2026-08-17T00:00:00.000Z"
last_activity: 2026-08-17 -- Phase 03 closed out (5/5 criteria); Phase 06 added to roadmap
progress:
  total_phases: 7
  completed_phases: 4
  total_plans: 5
  completed_plans: 4
  percent: 57
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-02)

**Core value:** Switching the active vLLM model must be safe and correct.
**Current focus:** Phase 04 — parameterized scripts + UI settings (next)

## Current Position

Phase: 03 (curated-recipe-overrides) — COMPLETE, 5/5 criteria
Plan: executed without PLAN.md files; committed as 716ca98
Status: Phase 03 closed out. Next: `/gsd-plan-phase 4`.
Last activity: 2026-08-17 -- Phase 03 closed out (5/5 criteria); Phase 06 added to roadmap
Remaining: Phase 4 parameterizes generated scripts and surfaces the derived/recommended
values (and the so-far-unconsumed `warnings`) in the UI. Phase 6 (verifiable research
capture) is independent and can be planned at any point.

Progress: [██████░░░░] 57%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

## Environment Notes

- **vLLM is UP.** `vllm_node` serves Qwen3.6-35B-A3B-FP8 at util 0.55 / 262144 via the
  recipe-backed profile, aliased `vllm-active` for litellm. The earlier "intentionally
  down for Helix training" note is stale — that blocker is resolved.

- **Page cache is charged against `--gpu-memory-utilization` on this box.** CUDA reports
  MemFree, not MemAvailable, so `usable = MemTotal * util - (MemTotal - MemFree)`. Drop
  caches before a large launch (`POST /api/vllm/reclaim-cache`, or `sync && sudo sh -c
  'echo 3 > /proc/sys/vm/drop_caches'`). At an unchanged util 0.55, Qwen3.6 got 0.19 GiB
  of KV with ~25 GB cached and 25.97 GiB after reclaiming.

- Editing `app.py` does NOT restart the running service — `systemctl --user restart
  dgx-model-manager.service` is required to deploy.

- **~3 concurrent claude sessions run in this repo.** Commit by explicit path; never
  `git add -A`. `app.py` currently holds two sessions' work at once.
