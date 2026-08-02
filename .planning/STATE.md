# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-02)

**Core value:** Switching the active vLLM model must be safe and correct.
**Current focus:** Phase 1 — Unbreak the load path

## Current Position

Phase: 1 of 5 (Unbreak the load path)
Plan: 3 of 3 in current phase
Status: Phase 1 implemented — uncommitted, not yet deployed
Last activity: 2026-08-02 — Phase 1 executed. Suite 59 → 100 tests, all passing.
Remaining: `systemctl --user restart dgx-model-manager.service` to deploy, and a live
launch check once vLLM is back up after Helix training.

Progress: [██░░░░░░░░] 20%

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

- **vLLM is intentionally down** while the box runs local Helix training. All Phase 1 and
  Phase 2 work must be verifiable without launching a model: script generation, the HF
  download worker, and the derived-spec solver are all testable offline.
- Editing `app.py` does NOT restart the running service — `systemctl --user restart
  dgx-model-manager.service` is required to deploy.
- The installed systemd unit is a copy, not a symlink to `deploy/`, and carries an
  `EnvironmentFile` the repo unit lacks. Reinstalling from `deploy/` silently disables
  Discord alerting.

## Audit Provenance

The roadmap derives from a 2026-08-02 review of the vLLM surface. Findings confirmed
against the live box, not taken on trust:

| Finding | Verification |
|---|---|
| `_HF_XFER` undefined — every download fails | Reproduced: rc=1, `NameError`, after `starting` |
| Generated non-gpt-oss scripts can't launch | `eugr/spark-vllm:latest` ENTRYPOINT is `nvidia_entrypoint.sh`, CMD null |
| Unauthenticated shell injection | `api_key` unset + non-loopback host; `$(...)` reaches the generated script verbatim |
| Hybrid KV overestimated 4–11x | Computed from every model config on the box |
| Derived formula is sound | Independently produced 0.54 vs the hand-tuned 0.55 for qwen3-next-80b |
