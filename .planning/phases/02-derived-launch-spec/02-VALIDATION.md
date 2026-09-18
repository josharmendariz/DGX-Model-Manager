---
phase: 2
slug: derived-launch-spec
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-03
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

**Phase-specific constraint:** vLLM is intentionally DOWN (GB10 running Helix training).
Every verification in this phase is `pytest` against a pure function and a committed fixture
snapshot. No container, no GPU, no network, no live model. This is a hard requirement, not a
convenience — see `02-RESEARCH.md` §Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing — 100 tests green as of 2026-08-03) |
| **Config file** | none — `tests/conftest.py` supplies shared fixtures |
| **Quick run command** | `python3 -m pytest tests/test_launch_spec.py -q` |
| **Full suite command** | `python3 -m pytest tests/ -q` |
| **Estimated runtime** | ~1 second full suite (0.63s measured at 100 tests) |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/test_launch_spec.py -q`
- **After every plan wave:** Run `python3 -m pytest tests/ -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

The suite is sub-second, so there is no reason for any task in this phase to defer
verification. Run the full suite at every commit if in doubt.

---

## Per-Task Verification Map

Task IDs follow the ROADMAP's 2-plan split; the planner may subdivide, but every row below
must map to at least one task.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-----------|--------|
| V1 precedence order | 02-01 | 1 | REQ-04 | — | N/A | unit | `pytest tests/test_launch_spec.py -k precedence -q` | ❌ W0 | ⬜ pending |
| V2 hybrid counts (17-model table) | 02-01 | 1 | REQ-04 | — | N/A | table | `pytest tests/test_launch_spec.py -k topology_table -q` | ❌ W0 | ⬜ pending |
| V3 unknown layer type is loud | 02-01 | 1 | REQ-04 | T-02-01 | Unrecognized `layer_types` entry must not silently score zero KV | unit | `pytest tests/test_launch_spec.py -k unknown_layer -q` | ❌ W0 | ⬜ pending |
| V4 `use_sliding_window` guard | 02-01 | 1 | REQ-04 | T-02-02 | 5 dense models declaring `sliding_window` must stay dense | unit | `pytest tests/test_launch_spec.py -k sliding_guard -q` | ❌ W0 | ⬜ pending |
| V5 `text_config` nesting | 02-01 | 1 | REQ-04 | — | N/A | unit | `pytest tests/test_launch_spec.py -k text_config -q` | ❌ W0 | ⬜ pending |
| V6 `head_dim` fallback | 02-01 | 1 | REQ-04 | — | N/A | unit | `pytest tests/test_launch_spec.py -k head_dim -q` | ❌ W0 | ⬜ pending |
| V7 **calibration** qwen3-next ±0.02 of 0.55 | 02-02 | 2 | REQ-04 | — | N/A | unit | `pytest tests/test_launch_spec.py -k calibration -q` | ❌ W0 | ⬜ pending |
| V8 purity (no fs/net in function body) | 02-02 | 2 | REQ-04 | T-02-03 | Function must not read the filesystem it is meant to be tested independently of | unit | `pytest tests/test_launch_spec.py -k purity -q` | ❌ W0 | ⬜ pending |
| V9 degenerate configs | 02-02 | 2 | REQ-04 | — | N/A | unit | `pytest tests/test_launch_spec.py -k degenerate -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_launch_spec.py` — new module; stubs for V1–V9 against REQ-04
- [x] `tests/conftest.py` — exists; extend only if a shared fixture loader is needed
- [x] pytest — already installed and green (100 tests)
- [x] `.planning/phases/02-derived-launch-spec/02-MODEL-FIXTURES.json` — 17-model ground-truth
      snapshot, generated and verified 2026-08-03

**Fixture rule:** tests load the committed JSON snapshot. They must **not** glob
`~/.cache/huggingface` or `/mnt/models` — a live read makes the suite depend on which models
happen to be on the box and breaks the moment one is added or deleted.

---

## Threat Notes (informational — this phase ships no request-handling code)

| Ref | Concern | Why it is a validation target |
|---|---|---|
| T-02-01 | Silent zero-KV on an unrecognized layer type | Under-reserves memory → OOM at model load. Wrong-answer failure, not a crash, so only a test catches it. |
| T-02-02 | `sliding_window` read without `use_sliding_window` | Collapses a 17.2 GB dense KV estimate to near zero on 5 of 17 models here. |
| T-02-03 | Filesystem access inside the "pure" function | Makes the phase unverifiable while vLLM is down, which is the entire premise of planning it now. |

No new HTTP surface, no user input, no shell construction in this phase — the injection
surface was closed in Phase 1 and is untouched here.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Derived util vs. a real vLLM load | REQ-04 | Requires vLLM, which is intentionally down for Helix training | Deferred to Phase 4 wiring. The 0.54-vs-0.55 calibration against the *already hand-measured* recipe is the offline substitute and is sufficient for this phase. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
