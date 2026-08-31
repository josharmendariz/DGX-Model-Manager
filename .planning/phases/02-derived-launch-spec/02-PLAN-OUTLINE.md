# Phase 02 — Plan Outline (chunked)

| Plan ID | Objective | Wave | Depends On | Requirements |
|---------|-----------|------|------------|--------------|
| 02-01 | Attention-topology resolver + KV bytes/token solver (precedence chain, 3 KV classes, loud on unknown layer type) | 1 | — | REQ-04 |
| 02-02 | Context/utilization fitting, `_derive_launch_spec` public shape, and the 17-model calibration table | 2 | 02-01 | REQ-04 |

Shared files: `app.py`, `tests/test_launch_spec.py` (overlap forces sequential waves).
Fixture: `02-MODEL-FIXTURES.json` (`{generated, pool_gb, models, anomalies}`) — tests load
the committed snapshot only; never glob the filesystem (VALIDATION Wave 0 rule).

## 02-01 — notes (3 tasks, wave 1)

- **T1 (Wave 0):** create `tests/test_launch_spec.py` with the V1–V9 stub skeleton + a
  conftest fixture loader for `02-MODEL-FIXTURES.json`. Satisfies VALIDATION's MISSING refs.
- **T2 tdd:** `_resolve_attention_topology(config)` near `_infer_from_config` (app.py:677
  idiom). Precedence: `layer_types` → `hybrid_override_pattern` (`*`=full, rest stateless) →
  `full_attention_interval` → `sliding_window` **and** `use_sliding_window` → dense.
  `text_config` unwrap first. Explicit full/bounded/stateless classification; raise/flag on an
  unrecognized `layer_types` entry (T-02-01) instead of silently scoring zero.
- **T3 tdd:** `kv_bytes_per_token` + topology table test over all 17 fixtures. `head_dim`
  fallback `hidden_size // num_attention_heads`. Covers V1–V6.
- Assert resolved **counts** and the source field **as observed** — RESEARCH Correction A:
  Qwen3.6 and qwen3-next both select `layer_types`, not the handoff's attribution.

## 02-02 — notes (2 tasks, wave 2)

- **T1 tdd:** `_derive_launch_spec(config, weights_gb, pool_gb=121.0, kv_dtype_bytes=1, ...)`
  returning a dict: layer counts, kv_bytes_per_token, kv_gb at ctx, max fitting context,
  `recommended_util = min(0.95, (weights+kv+6.0)/pool + 0.04)`. Pure: pool and weights are
  parameters, no fs/net/vLLM/`/proc` (T-02-03).
- **T2 tdd:** calibration + robustness table — qwen3-next-80b util within 0.02 of 0.55 (V7),
  purity assertion via source/AST inspection (V8), degenerate configs (V9), and the
  `use_sliding_window` dense-regression rows (V4/T-02-02).
- Do **not** chase Qwen3.6's 0.42-vs-0.55 gap — that is Phase 3's recipe precedence.
- Do **not** touch `_build_vllm_profile_script` or the `0.75`/`32768`/`65536` literals.
