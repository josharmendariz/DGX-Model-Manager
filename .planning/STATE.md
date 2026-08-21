---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-08-21T02:49:06.122Z"
last_activity: 2026-08-21 -- Phase 04 closed (waves 1-3 shipped, live checkpoint run, gsd-verifier PASS)
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
**Current focus:** Phase 5 — admission truth

## Current Position

Phase: 04 (parameterized-scripts-ui-settings) — COMPLETE, 3/3 plans, gsd-verifier PASS
(04-VERIFICATION.md). Suite 464 passing at HEAD (rec-approve-actions, 3ccb35d). Live
human-verify checkpoint run 2026-08-21: override reaches the container (docker inspect),
concurrent-edit 409, HF-browse XSS payload renders as text, two profile scripts
parameterized for real via the browser dialog. Checkpoint caught three renderer defects
(classification outranking flag-parseability) and one missing affordance (Regenerate
metadata on generated-lineage scripts), all fixed and tested.
Next: Phase 5 (admission truth) — not yet planned. Phase 6 (verifiable research capture)
is independent and can be planned at any point.
Last activity: 2026-08-21 -- Phase 04 closed

See .planning/HANDOFF.json for the full situation: live engine/litellm state, the PR
intent and why a direct PR would be 41 commits, three open questions, and the one
root cause (litellm file-vs-ConfigMap drift) that is still unguarded.
Remaining: Phase 5 admits on executor budget and identifies the reclaim target by docker
label. Phase 6 (verifiable research capture) is independent and can be planned at any
point.

Progress: [██████░░░░] 63%

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
