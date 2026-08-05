# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-02)

**Core value:** Switching the active vLLM model must be safe and correct.
**Current focus:** Phase 1 — Unbreak the load path

## Current Position

Phase: 1.1 of 5 (Launch feedback — INSERTED, unplanned)
Plan: n/a (reactive work, no PLAN.md)
Status: Implemented, deployed, live-verified — **uncommitted by decision**
Last activity: 2026-08-04 — Phase 1.1 executed. Suite 100 → 130 tests, all passing.
Remaining: Josh decides what happens to a peer session's Site Scanner code in `app.py`
before anything is committed. Then `/gsd-plan-phase 2`.

Progress: [███░░░░░░░] 25%

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
