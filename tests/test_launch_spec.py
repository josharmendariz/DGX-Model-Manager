"""Phase 2 (REQ-04) — derived launch spec: attention topology and KV bytes per token.

vLLM is intentionally DOWN on this box (the GB10 is running local Helix training), so every
check here is a pure-function check: a parsed `config.json` dict goes in, a dict comes out.
Nothing in this module starts a container, touches a GPU, opens a socket, or reads a model
config off the filesystem.

Ground truth is the committed snapshot `.planning/phases/02-derived-launch-spec/
02-MODEL-FIXTURES.json` (17 models, generated and verified 2026-08-03), loaded via the
`model_fixtures` / `fixture_models` conftest fixtures. Tests must never scan the model cache
directories: a live read would make the suite depend on which models happen to be on the box,
and would break the moment one is added or deleted.

Validation rows (02-VALIDATION.md): V1 precedence · V2 topology table · V3 unknown layer type
is loud · V4 `use_sliding_window` guard · V5 `text_config` nesting · V6 `head_dim` fallback ·
V7 calibration · V8 purity · V9 degenerate configs. V7-V9 are owned by plan 02-02.
"""

import inspect
import json

import pytest

import app as appmod
from conftest import MODEL_FIXTURES_PATH


# Module-level load so the 17-row table test can be parameterized with the model name as the
# test id (pytest.mark.parametrize is evaluated at collection time and cannot consume a
# session fixture). Same committed file the conftest fixtures read.
with MODEL_FIXTURES_PATH.open(encoding="utf-8") as _fh:
    _SNAPSHOT = json.load(_fh)
MODEL_ROWS = _SNAPSHOT["models"]
MODEL_IDS = [row["name"] for row in MODEL_ROWS]

# The 5 models that declare a sliding_window while having use_sliding_window false. Reading
# the window without the guard reclassifies all of them as fully-sliding and collapses a
# 17.2 GB dense KV estimate to near zero — a silent wrong answer, not a crash (T-02-02).
TRAPPED_ROWS = [
    row for row in MODEL_ROWS
    if row.get("sliding_window") and row.get("use_sliding_window") is False
]
TRAPPED_IDS = [row["name"] for row in TRAPPED_ROWS]

NESTED_ROWS = [row for row in MODEL_ROWS if row.get("nested_text_config")]
NESTED_IDS = [row["name"] for row in NESTED_ROWS]

# Transformer fields copied verbatim from a fixture row into the rebuilt config.
_PLAIN_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "sliding_window",
    "use_sliding_window",
)


def config_from_fixture(row: dict) -> dict:
    """Rebuild a plausible parsed `config.json` from one committed fixture row.

    Caveat worth stating out loud: reconstructing `layer_types` from `topology_distribution`
    is partly self-fulfilling — a test that rebuilds the list from the counts it then asserts
    proves only that counting works. No snapshot row carries a verbatim `layer_types` list, so
    the independent signal in the table test is `precedence_field` (asserted as `source_field`):
    it says which branch of the chain the real config selected, which the reconstruction cannot
    fake. If a future snapshot regeneration adds verbatim `layer_types`, assert against that
    list instead of this reconstruction.

    A field absent from the row (JSON `null`) is left out of the rebuilt config entirely,
    matching real configs — several models simply do not declare `use_sliding_window`, and
    "absent" and "false" must not be conflated by the fixture builder.
    """
    inner: dict = {}
    for key in _PLAIN_FIELDS:
        if row.get(key) is not None:
            inner[key] = row[key]

    # head_dim is only declared by the hybrid Qwen/Nemotron families; the other 13 models
    # derive it. Only plant it when the snapshot says the real config declared it.
    if row.get("head_dim_source") == "explicit":
        inner["head_dim"] = row["head_dim"]

    field = row.get("precedence_field")
    dist = row.get("topology_distribution") or {}
    if field == "layer_types":
        layer_types: list[str] = []
        for layer_name, count in dist.items():
            layer_types.extend([layer_name] * count)
        inner["layer_types"] = layer_types
    elif field == "hybrid_override_pattern":
        inner["hybrid_override_pattern"] = "".join(ch * count for ch, count in dist.items())
    elif field == "full_attention_interval":
        inner["full_attention_interval"] = row["full_attention_interval"]
    # "dense" and the "sliding_window" branch add no topology field.

    if row.get("nested_text_config"):
        # VL / nested-text models carry none of the transformer fields at top level.
        return {"text_config": inner}
    return inner


# ── V1 precedence order ───────────────────────────────────────────────────────

def test_precedence_fixture_vocabulary_is_closed():
    """Every snapshot row names a branch the chain actually implements."""
    known = {
        "layer_types",
        "hybrid_override_pattern",
        "full_attention_interval",
        "sliding_window",
        "dense",
    }
    seen = {row["precedence_field"] for row in MODEL_ROWS}
    assert seen <= known, f"snapshot names an unimplemented branch: {seen - known}"


def test_precedence_first_hit_wins_as_fields_are_removed():
    """One config carrying every topology field resolves them in a fixed order.

    Peeling the fields off one at a time is the only way to prove the *order* rather than
    the individual branches — a refactor that reorders the chain still passes every
    single-field test.
    """
    config = {
        "num_hidden_layers": 48,
        "layer_types": ["full_attention"] * 12 + ["linear_attention"] * 36,
        "hybrid_override_pattern": "*" * 6 + "M" * 42,
        "full_attention_interval": 4,
        "sliding_window": 4096,
        "use_sliding_window": True,
    }
    assert appmod._resolve_attention_topology(config)["source_field"] == "layer_types"

    config.pop("layer_types")
    assert appmod._resolve_attention_topology(config)["source_field"] == "hybrid_override_pattern"

    config.pop("hybrid_override_pattern")
    assert appmod._resolve_attention_topology(config)["source_field"] == "full_attention_interval"

    config.pop("full_attention_interval")
    assert appmod._resolve_attention_topology(config)["source_field"] == "sliding_window"

    config.pop("sliding_window")
    assert appmod._resolve_attention_topology(config)["source_field"] == "dense"


def test_precedence_empty_topology_fields_do_not_claim_the_chain():
    """An empty list or empty string must fall through, not resolve to zero layers."""
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 10, "layer_types": [], "hybrid_override_pattern": ""}
    )
    assert topo["source_field"] == "dense"
    assert topo["full_attention_layers"] == 10


def test_precedence_full_attention_interval_is_synthetic_only():
    """V1b: no model on this box selects this branch, so it needs a synthetic config.

    Qwen3.6 and qwen3-next both declare full_attention_interval=4 *and* layer_types, and
    layer_types wins — the counts coincide (48//4 == 12) purely by arithmetic luck, which is
    exactly why the handoff's attribution was wrong. Without this test the branch ships
    untested.
    """
    assert not any(row["precedence_field"] == "full_attention_interval" for row in MODEL_ROWS)
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 48, "full_attention_interval": 4}
    )
    assert topo["source_field"] == "full_attention_interval"
    assert topo["full_attention_layers"] == 12
    assert topo["stateless_layers"] == 36
    assert topo["bounded_kv_layers"] == 0


def test_precedence_interval_larger_than_layer_count_does_not_crash():
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 2, "full_attention_interval": 8}
    )
    assert topo["full_attention_layers"] == 0
    assert topo["stateless_layers"] == 2


def test_precedence_returns_exactly_the_interface_keys():
    expected = {
        "num_hidden_layers",
        "full_attention_layers",
        "bounded_kv_layers",
        "stateless_layers",
        "sliding_window",
        "source_field",
        "warnings",
    }
    assert set(appmod._resolve_attention_topology({})) == expected


# ── V2 hybrid counts (17-model table) ─────────────────────────────────────────

@pytest.mark.parametrize("row", MODEL_ROWS, ids=MODEL_IDS)
def test_topology_table_fixture_row_is_self_consistent(row):
    """The snapshot's own layer counts must partition num_hidden_layers."""
    total = row["full_attention_layers"] + row["bounded_kv_layers"] + row["stateless_layers"]
    assert total == row["num_hidden_layers"]
    dist = row.get("topology_distribution") or {}
    if dist:
        assert sum(dist.values()) == row["num_hidden_layers"]


def test_topology_table_snapshot_has_all_seventeen_models():
    assert len(MODEL_ROWS) == 17
    assert len(set(MODEL_IDS)) == 17


@pytest.mark.parametrize("row", MODEL_ROWS, ids=MODEL_IDS)
def test_topology_table_reproduces_recorded_counts(row):
    """Every model on this box resolves to the layer counts the snapshot recorded.

    `source_field` is the independent signal here: the rebuilt config cannot fake which
    branch the real config selected, so this catches the handoff's wrong attribution
    (Qwen3.6 and qwen3-next resolve via layer_types, not hybrid_override_pattern /
    full_attention_interval) as well as any silent reordering of the chain.
    """
    topo = appmod._resolve_attention_topology(config_from_fixture(row))
    assert topo["source_field"] == row["precedence_field"]
    assert topo["num_hidden_layers"] == row["num_hidden_layers"]
    assert topo["full_attention_layers"] == row["full_attention_layers"]
    assert topo["bounded_kv_layers"] == row["bounded_kv_layers"]
    assert topo["stateless_layers"] == row["stateless_layers"]
    assert (topo["full_attention_layers"] + topo["bounded_kv_layers"]
            + topo["stateless_layers"]) == row["num_hidden_layers"]


@pytest.mark.parametrize(
    "name,expected_full_bytes,expected_bounded_total",
    [
        # per_layer = 2 * kv_heads * head_dim; hand-computed from the snapshot rows.
        ("openai/gpt-oss-120b", 18 * 2 * 8 * 64, 18 * 2 * 8 * 64 * 128),
        ("qwen3-next-80b-a3b-nvfp4", 12 * 2 * 2 * 256, 0),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 64 * 2 * 8 * 128, 0),
    ],
)
def test_topology_table_kv_byte_anchors(name, expected_full_bytes, expected_bounded_total):
    row = next(r for r in MODEL_ROWS if r["name"] == name)
    result = appmod._kv_bytes_per_token(config_from_fixture(row), kv_dtype_bytes=1)
    assert result["full_bytes_per_token"] == expected_full_bytes
    assert result["bounded_bytes_total"] == expected_bounded_total


def test_topology_table_bounded_bytes_are_context_independent():
    """bounded_bytes_total is already a TOTAL, not a per-token rate.

    gpt-oss's 18 windowed layers cost 128 tokens of KV each no matter how long the context
    is; only full_bytes_per_token gets multiplied by max_model_len downstream.
    """
    config = {
        "num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 64,
        "sliding_window": 128,
        "layer_types": ["full_attention"] * 18 + ["sliding_attention"] * 18,
    }
    result = appmod._kv_bytes_per_token(config)
    assert result["per_layer_bytes"] == 1024
    assert result["full_bytes_per_token"] == 18432
    assert result["bounded_bytes_total"] == 2359296
    assert result["topology"]["source_field"] == "layer_types"


def test_topology_table_stateless_layers_contribute_zero_bytes():
    """Stateless layers cost zero by construction, not by falling through a missed branch."""
    hybrid = appmod._kv_bytes_per_token(
        {"num_hidden_layers": 48, "num_key_value_heads": 2, "head_dim": 256,
         "layer_types": ["full_attention"] * 12 + ["linear_attention"] * 36}
    )
    dense = appmod._kv_bytes_per_token(
        {"num_hidden_layers": 12, "num_key_value_heads": 2, "head_dim": 256}
    )
    assert hybrid["full_bytes_per_token"] == dense["full_bytes_per_token"]
    assert hybrid["bounded_bytes_total"] == 0


def test_topology_table_kv_dtype_bytes_scales_linearly():
    config = {"num_hidden_layers": 4, "num_key_value_heads": 4, "head_dim": 64}
    one = appmod._kv_bytes_per_token(config, kv_dtype_bytes=1)
    two = appmod._kv_bytes_per_token(config, kv_dtype_bytes=2)
    assert two["per_layer_bytes"] == 2 * one["per_layer_bytes"] == 1024
    assert two["full_bytes_per_token"] == 2 * one["full_bytes_per_token"]


def test_topology_table_returns_exactly_the_interface_keys():
    expected = {
        "topology",
        "num_key_value_heads",
        "head_dim",
        "head_dim_source",
        "per_layer_bytes",
        "full_bytes_per_token",
        "bounded_bytes_total",
    }
    assert set(appmod._kv_bytes_per_token({})) == expected


# ── V3 unknown layer type is loud ─────────────────────────────────────────────

def test_unknown_layer_types_absent_from_snapshot():
    """Guard the flipside of V3: every layer type on this box is one the code classifies.

    If a snapshot regeneration introduces a new layer type, this fails here rather than
    silently scoring zero KV somewhere downstream.
    """
    known_layer_types = {"full_attention", "sliding_attention", "linear_attention"}
    known_pattern_chars = {"*", "M", "E"}
    for row in MODEL_ROWS:
        dist = row.get("topology_distribution") or {}
        if row["precedence_field"] == "layer_types":
            assert set(dist) <= known_layer_types, row["name"]
        elif row["precedence_field"] == "hybrid_override_pattern":
            assert set(dist) <= known_pattern_chars, row["name"]


def test_unknown_layer_type_counts_as_full_and_warns():
    """T-02-01: an unrecognized layer type must over-reserve loudly, never score zero.

    The prototype counted `x == "full_attention"` and `"sliding" in x`, so anything else fell
    through both buckets and silently contributed zero KV. That is accidentally correct for
    linear_attention today and actively dangerous for any future type that does carry KV: the
    recommender would confidently under-reserve and OOM at model load.
    """
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 1, "layer_types": ["quantum_attention"]}
    )
    assert topo["full_attention_layers"] == 1
    assert topo["stateless_layers"] == 0
    assert topo["bounded_kv_layers"] == 0
    assert len(topo["warnings"]) == 1
    assert topo["warnings"][0].startswith("unknown layer type: ")
    assert "quantum_attention" in topo["warnings"][0]


def test_unknown_layer_type_mixed_with_known_types_keeps_the_partition():
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 4, "layer_types": ["full_attention", "linear_attention",
                                                 "quantum_attention", "sliding_attention"]}
    )
    assert topo["full_attention_layers"] == 2  # the known full one plus the unknown one
    assert topo["bounded_kv_layers"] == 1
    assert topo["stateless_layers"] == 1
    assert len(topo["warnings"]) == 1


def test_unknown_layer_pattern_character_counts_as_full_and_warns():
    """Same rule for an unrecognized character in a Nemotron-style pattern."""
    topo = appmod._resolve_attention_topology({"hybrid_override_pattern": "*MEZ"})
    assert topo["full_attention_layers"] == 2  # '*' plus the unknown 'Z'
    assert topo["stateless_layers"] == 2
    # num_hidden_layers falls back to the pattern length when the config omits it, so the
    # three classes always partition the reported total.
    assert topo["num_hidden_layers"] == 4
    assert len(topo["warnings"]) == 1
    assert topo["warnings"][0].startswith("unknown layer type: ")


def test_unknown_layer_types_warn_once_per_distinct_type():
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 30, "layer_types": ["quantum_attention"] * 20 + ["warp_attention"] * 10}
    )
    assert topo["full_attention_layers"] == 30
    assert len(topo["warnings"]) == 2


def test_unknown_layer_types_absent_means_no_warnings_on_real_models():
    for row in MODEL_ROWS:
        topo = appmod._resolve_attention_topology(config_from_fixture(row))
        assert topo["warnings"] == [], f"{row['name']}: {topo['warnings']}"


# ── V4 use_sliding_window guard ───────────────────────────────────────────────

def test_sliding_guard_snapshot_anomalies_are_recorded_dense():
    """The 5 models declaring sliding_window with use_sliding_window false are dense."""
    trapped = [
        row for row in MODEL_ROWS
        if row.get("sliding_window") and row.get("use_sliding_window") is False
    ]
    assert len(trapped) == 5
    for row in trapped:
        assert row["precedence_field"] == "dense", row["name"]
        assert row["full_attention_layers"] == row["num_hidden_layers"], row["name"]
        assert row["bounded_kv_layers"] == 0, row["name"]


def test_sliding_guard_window_without_use_flag_stays_dense():
    """T-02-02: `sliding_window` alone must not claim the chain."""
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 64, "sliding_window": 131072, "use_sliding_window": False}
    )
    assert topo["source_field"] == "dense"
    assert topo["full_attention_layers"] == 64
    assert topo["bounded_kv_layers"] == 0
    # The vestigial window is not reported: no layer is bounded, so no window applies.
    assert topo["sliding_window"] is None


def test_sliding_guard_window_with_use_flag_bounds_every_layer():
    topo = appmod._resolve_attention_topology(
        {"num_hidden_layers": 32, "sliding_window": 4096, "use_sliding_window": True}
    )
    assert topo["source_field"] == "sliding_window"
    assert topo["bounded_kv_layers"] == 32
    assert topo["full_attention_layers"] == 0
    assert topo["sliding_window"] == 4096


def test_sliding_guard_missing_use_flag_stays_dense():
    """Absent is not truthy: a config that never declares the flag is dense, not sliding."""
    topo = appmod._resolve_attention_topology({"num_hidden_layers": 8, "sliding_window": 512})
    assert topo["source_field"] == "dense"
    assert topo["full_attention_layers"] == 8


@pytest.mark.parametrize("row", TRAPPED_ROWS, ids=TRAPPED_IDS)
def test_sliding_guard_real_trapped_model_keeps_full_kv(row):
    """The 5 real configs carrying the trap resolve dense against the shipped helper."""
    topo = appmod._resolve_attention_topology(config_from_fixture(row))
    assert topo["source_field"] == "dense"
    assert topo["full_attention_layers"] == row["num_hidden_layers"]
    assert topo["bounded_kv_layers"] == 0


# ── V5 text_config nesting ────────────────────────────────────────────────────

def test_text_config_nesting_and_hybridity_are_independent_axes():
    """Two rows nest; one is hybrid and one is dense, so the axes need separate cover."""
    nested = [row for row in MODEL_ROWS if row.get("nested_text_config")]
    assert len(nested) >= 2
    fields = {row["precedence_field"] for row in nested}
    assert "dense" in fields and fields - {"dense"}


def test_text_config_nested_resolves_identically_to_flat():
    """Nesting is a packaging detail; it must not change a single number."""
    flat = {
        "num_hidden_layers": 40,
        "layer_types": ["full_attention"] * 10 + ["linear_attention"] * 30,
    }
    nested = {"text_config": dict(flat)}
    assert appmod._resolve_attention_topology(nested) == appmod._resolve_attention_topology(flat)


def test_text_config_nested_fields_are_not_read_as_zero():
    """A VL-style config with nothing at top level must not resolve to an empty model."""
    topo = appmod._resolve_attention_topology(
        {"text_config": {"num_hidden_layers": 40,
                         "layer_types": ["full_attention"] * 10 + ["linear_attention"] * 30}}
    )
    assert topo["num_hidden_layers"] == 40
    assert topo["full_attention_layers"] == 10
    assert topo["stateless_layers"] == 30


@pytest.mark.parametrize("row", NESTED_ROWS, ids=NESTED_IDS)
def test_text_config_real_nested_model_resolves(row):
    topo = appmod._resolve_attention_topology(config_from_fixture(row))
    assert topo["num_hidden_layers"] == row["num_hidden_layers"]
    assert topo["full_attention_layers"] == row["full_attention_layers"]


# ── V6 head_dim fallback ──────────────────────────────────────────────────────

@pytest.mark.parametrize("row", MODEL_ROWS, ids=MODEL_IDS)
def test_head_dim_snapshot_source_is_consistent(row):
    """A derived head_dim must actually equal hidden_size // num_attention_heads."""
    assert row["head_dim_source"] in ("explicit", "hidden_size//num_attention_heads")
    if row["head_dim_source"] == "hidden_size//num_attention_heads":
        assert row["head_dim"] == row["hidden_size"] // row["num_attention_heads"], row["name"]


@pytest.mark.parametrize("row", MODEL_ROWS, ids=MODEL_IDS)
def test_head_dim_reproduced_from_rebuilt_config(row):
    """Each model's KV width and where it came from, reproduced by the shipped helper."""
    result = appmod._kv_bytes_per_token(config_from_fixture(row))
    assert result["head_dim"] == row["head_dim"], row["name"]
    assert result["head_dim_source"] == row["head_dim_source"], row["name"]
    assert result["num_key_value_heads"] == row["num_key_value_heads"], row["name"]


def test_head_dim_explicit_wins_over_the_derived_value():
    """Qwen3.6 declares head_dim 256 while hidden_size//heads would give 128 — a 2x error."""
    result = appmod._kv_bytes_per_token(
        {"num_hidden_layers": 1, "head_dim": 256, "hidden_size": 2048,
         "num_attention_heads": 16, "num_key_value_heads": 2}
    )
    assert result["head_dim"] == 256
    assert result["head_dim_source"] == "explicit"


def test_head_dim_falls_back_to_hidden_size_over_attention_heads():
    result = appmod._kv_bytes_per_token(
        {"num_hidden_layers": 64, "num_key_value_heads": 8, "hidden_size": 5120,
         "num_attention_heads": 40}
    )
    assert result["head_dim"] == 128
    assert result["head_dim_source"] == "hidden_size//num_attention_heads"
    assert result["full_bytes_per_token"] == 131072


def test_head_dim_zero_or_missing_attention_heads_does_not_divide():
    """T-02-04: a malformed snapshot must return zeros, not raise ZeroDivisionError."""
    for config in ({"num_hidden_layers": 4, "hidden_size": 4096, "num_attention_heads": 0},
                   {"num_hidden_layers": 4, "hidden_size": 4096},
                   {}):
        result = appmod._kv_bytes_per_token(config)
        assert result["head_dim"] == 0
        assert result["per_layer_bytes"] == 0
        assert result["full_bytes_per_token"] == 0


def test_head_dim_kv_heads_fall_back_to_attention_heads():
    """A config without grouped-query attention declares only num_attention_heads."""
    result = appmod._kv_bytes_per_token(
        {"num_hidden_layers": 2, "num_attention_heads": 8, "head_dim": 64}
    )
    assert result["num_key_value_heads"] == 8
    assert result["per_layer_bytes"] == 1024


# ── Budget: the page-cache-corrected KV headroom ──────────────────────────────

# Qwen3.6-35B-A3B-FP8's weights in decimal GB, the row the GB10 page-cache regime change was
# measured on. Kept as a constant so the budget tests and the calibration tests below cannot
# drift apart on the input.
QWEN36_WEIGHTS_GB = 37.46366216


def test_budget_idle_box_matches_the_hand_measured_recipe():
    """At the hand-validated 0.55 with nothing resident, Qwen3.6 has ~23.09 GB of KV headroom."""
    assert round(appmod._kv_budget_gb(0.55, 121.0, QWEN36_WEIGHTS_GB), 2) == 23.09


def test_budget_resident_page_cache_is_charged_against_the_pool():
    """The measured Phase 1.1 failure: CUDA reports MemFree, not MemAvailable.

    With ~25 GB of page cache resident, an unchanged 0.55 yielded 0.19 GiB of usable KV
    instead of 25.97 GiB and the engine refused to start, blaming max_model_len. The budget
    must reproduce that collapse rather than assume the whole pool is free.
    """
    assert appmod._kv_budget_gb(0.55, 121.0, QWEN36_WEIGHTS_GB, resident_gb=25.0) == 0.0


def test_budget_never_returns_a_negative_headroom():
    """A budget smaller than the model is 0.0, not a negative number a caller would add."""
    assert appmod._kv_budget_gb(0.95, 121.0, 10_000.0) == 0.0
    assert appmod._kv_budget_gb(0.0, 121.0, 0.0) == 0.0


def test_budget_is_the_pool_share_minus_resident_weights_and_overhead():
    assert appmod._kv_budget_gb(0.5, 100.0, 10.0, overhead_gb=5.0, resident_gb=5.0) == 30.0


def test_budget_rises_with_util_and_falls_with_resident():
    rising = [appmod._kv_budget_gb(u, 121.0, 30.0) for u in (0.5, 0.6, 0.7)]
    assert rising == sorted(rising) and len(set(rising)) == 3
    falling = [appmod._kv_budget_gb(0.9, 121.0, 30.0, resident_gb=r) for r in (0.0, 5.0, 10.0)]
    assert falling == sorted(falling, reverse=True) and len(set(falling)) == 3


# ── Fitting: context, max_model_len and the utilization solver ────────────────

QWEN3_NEXT_CONFIG = {
    "num_hidden_layers": 48,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "layer_types": ["full_attention"] * 12 + ["linear_attention"] * 36,
}
QWEN3_NEXT_WEIGHTS_GB = 50.75774832


def test_fitting_returns_exactly_the_sixteen_interface_keys():
    expected = {
        "topology",
        "num_hidden_layers",
        "full_attention_layers",
        "bounded_kv_layers",
        "stateless_layers",
        "source_field",
        "kv_bytes_per_token",
        "bounded_bytes_total",
        "declared_max_context",
        "max_fitting_context",
        "max_model_len",
        "kv_gb",
        "weights_gb",
        "overhead_gb",
        "recommended_util",
        "warnings",
    }
    spec = appmod._derive_launch_spec({})
    assert set(spec) == expected
    assert len(spec) == 16


def test_fitting_echoes_the_topology_and_its_source_field():
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG)
    assert spec["source_field"] == spec["topology"]["source_field"] == "layer_types"
    assert spec["full_attention_layers"] == 12
    assert spec["stateless_layers"] == 36
    assert spec["bounded_kv_layers"] == 0
    assert spec["num_hidden_layers"] == 48
    assert appmod._derive_launch_spec(
        {"num_hidden_layers": 4, "hybrid_override_pattern": "*MMM"}
    )["source_field"] == "hybrid_override_pattern"


def test_fitting_kv_gb_is_the_rate_times_context_plus_the_bounded_total():
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb=QWEN3_NEXT_WEIGHTS_GB)
    assert spec["kv_bytes_per_token"] == 12288
    assert spec["max_model_len"] == 262144
    assert round(spec["kv_gb"], 3) == 3.221
    expected = (spec["kv_bytes_per_token"] * spec["max_model_len"]
                + spec["bounded_bytes_total"]) / 1e9
    assert abs(spec["kv_gb"] - expected) < 1e-9


def test_fitting_bounded_layers_add_a_context_independent_total():
    """gpt-oss's 18 windowed layers cost the same at 8k as at 131k."""
    config = {
        "num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 64,
        "sliding_window": 128, "max_position_embeddings": 131072,
        "layer_types": ["full_attention"] * 18 + ["sliding_attention"] * 18,
    }
    spec = appmod._derive_launch_spec(config, weights_gb=65.248893184)
    assert spec["bounded_bytes_total"] == 2359296
    assert spec["kv_gb"] == (18432 * spec["max_model_len"] + 2359296) / 1e9


def test_fitting_max_model_len_never_exceeds_the_declared_maximum():
    """A huge budget must not invent context the model was never trained for."""
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb=1.0)
    assert spec["max_fitting_context"] > spec["declared_max_context"] == 262144
    assert spec["max_model_len"] == 262144


def test_fitting_budget_binds_below_the_declared_maximum_and_warns():
    """When memory is the binding limit, the warning must name both numbers."""
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb=107.0)
    assert 0 < spec["max_model_len"] == spec["max_fitting_context"] < 262144
    assert any("262144" in w and str(spec["max_fitting_context"]) in w
               for w in spec["warnings"]), spec["warnings"]


def test_fitting_all_stateless_model_is_not_kv_bounded():
    """With no full-attention layer, context costs nothing and the declared max applies."""
    spec = appmod._derive_launch_spec(
        {"num_hidden_layers": 40, "layer_types": ["linear_attention"] * 40,
         "max_position_embeddings": 262144}
    )
    assert spec["kv_bytes_per_token"] == 0
    assert spec["max_fitting_context"] == 262144
    assert spec["max_model_len"] == 262144
    assert any("not KV-bounded" in w for w in spec["warnings"])


def test_fitting_nested_text_config_resolves_context_and_layers_from_the_nest():
    spec = appmod._derive_launch_spec(
        {"text_config": {"num_hidden_layers": 36, "num_key_value_heads": 8,
                         "head_dim": 128, "max_position_embeddings": 262144}},
        weights_gb=6.021235456,
    )
    assert spec["declared_max_context"] == 262144
    assert spec["full_attention_layers"] == 36


def test_fitting_requested_context_caps_max_model_len():
    spec = appmod._derive_launch_spec(
        QWEN3_NEXT_CONFIG, weights_gb=QWEN3_NEXT_WEIGHTS_GB, requested_context=32768
    )
    assert spec["max_model_len"] == 32768
    assert spec["kv_gb"] < 1.0


def test_fitting_util_reproduces_the_prototype_on_an_idle_box():
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb=QWEN3_NEXT_WEIGHTS_GB)
    assert spec["recommended_util"] == 0.54


def test_fitting_util_rises_with_resident_memory():
    """Resident bytes are charged against the same budget, so the ask must grow."""
    utils = [
        appmod._derive_launch_spec(
            QWEN3_NEXT_CONFIG, weights_gb=QWEN3_NEXT_WEIGHTS_GB, resident_gb=r
        )["recommended_util"]
        for r in (0.0, 10.0, 20.0)
    ]
    assert utils == [0.54, 0.62, 0.7]
    assert utils == sorted(utils) and len(set(utils)) == 3


def test_fitting_util_clamps_up_to_the_floor_on_a_tiny_model():
    """Qwen2.5-0.5B computes ~0.091, which must not be handed to a launch as 0.09."""
    spec = appmod._derive_launch_spec(
        {"num_hidden_layers": 24, "num_key_value_heads": 2, "head_dim": 64,
         "max_position_embeddings": 32768},
        weights_gb=0.011488523,
    )
    assert spec["recommended_util"] == 0.10


def test_fitting_util_clamps_down_to_the_cap_on_an_absurd_model():
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb=10_000.0)
    assert spec["recommended_util"] == 0.95
    assert spec["max_fitting_context"] == 0


def test_fitting_invalid_pool_short_circuits_instead_of_dividing_by_zero():
    """T-02-07: pool_gb <= 0 is a caller bug, and must surface as a warning not a traceback."""
    spec = appmod._derive_launch_spec({"num_hidden_layers": 4}, pool_gb=0)
    assert spec["recommended_util"] == 0.95
    assert spec["max_fitting_context"] == 0
    assert spec["warnings"][0].startswith("invalid pool_gb")


def test_fitting_non_numeric_scalars_are_coerced_and_named_in_a_warning():
    """T-02-06: a bad input is visible in `warnings`, never silently zero."""
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb="banana")
    assert spec["weights_gb"] == 0.0
    assert any("weights_gb" in w for w in spec["warnings"]), spec["warnings"]


def test_fitting_negative_scalars_floor_at_zero_and_warn():
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb=-40.0)
    assert spec["weights_gb"] == 0.0
    assert any("weights_gb" in w for w in spec["warnings"]), spec["warnings"]


def test_fitting_topology_warnings_are_propagated_to_the_caller():
    """T-02-11: an unknown layer type must stay visible at the public entry point."""
    spec = appmod._derive_launch_spec(
        {"num_hidden_layers": 2, "num_key_value_heads": 2, "head_dim": 64,
         "layer_types": ["full_attention", "quantum_attention"]}
    )
    assert any("quantum_attention" in w for w in spec["warnings"]), spec["warnings"]


def test_fitting_echoes_its_scalar_inputs_back():
    spec = appmod._derive_launch_spec(QWEN3_NEXT_CONFIG, weights_gb=12.5, overhead_gb=7.5)
    assert spec["weights_gb"] == 12.5
    assert spec["overhead_gb"] == 7.5


# ── V7 calibration ────────────────────────────────────────────────────────────

# The hand-measured recipe this whole phase is calibrated against.
HAND_MEASURED_UTIL = 0.55
CALIBRATION_TOLERANCE = 0.02
CALIBRATION_MODEL = "qwen3-next-80b-a3b-nvfp4"


def _row(name: str) -> dict:
    """Look a fixture row up by name, never by index.

    A snapshot regeneration that reorders the rows must fail loudly here rather than quietly
    calibrating against a different model.
    """
    matches = [r for r in MODEL_ROWS if r["name"] == name]
    assert len(matches) == 1, f"{name!r} not uniquely present in the snapshot"
    return matches[0]


def _spec_for(row: dict, **kwargs) -> dict:
    """Derive a launch spec from a fixture row, weights in decimal GB as the prototype used."""
    kwargs.setdefault("weights_gb", row["weight_bytes"] / 1e9)
    return appmod._derive_launch_spec(config_from_fixture(row), **kwargs)


def test_calibration_qwen3_next_matches_hand_measured_util():
    """The phase's core evidence: derived arithmetic reproduces a hand-measured number.

    `qwen3-next-80b-a3b-nvfp4` was hand-tuned on this box to
    `--gpu-memory-utilization 0.55`. The prototype, working only from the parsed config and
    the weight size, independently derived 0.54. That independent agreement — not any
    internal consistency check — is the evidence that every other derived number in this
    project can be trusted.

    If this test fails after a refactor, the formula is wrong. The hand-measured recipe is
    the fixed point, not the thing to adjust.
    """
    row = _row(CALIBRATION_MODEL)
    util = _spec_for(row)["recommended_util"]
    assert util == 0.54
    assert abs(util - HAND_MEASURED_UTIL) <= CALIBRATION_TOLERANCE, (
        f"derived {util} drifted more than {CALIBRATION_TOLERANCE} from the hand-measured "
        f"{HAND_MEASURED_UTIL} for {CALIBRATION_MODEL}"
    )


def test_calibration_qwen3_next_supporting_anchors():
    """The intermediate numbers the 0.54 is built from, so a failure localises itself."""
    spec = _spec_for(_row(CALIBRATION_MODEL))
    assert spec["full_attention_layers"] == 12
    assert spec["kv_bytes_per_token"] == 12288
    assert spec["max_model_len"] == 262144
    assert round(spec["kv_gb"], 3) == 3.221


def test_calibration_naive_dense_count_falls_outside_the_tolerance_band():
    """Negative control: the calibration must actually discriminate.

    Scored with the naive `num_hidden_layers` count — 48 attention layers instead of 12, the
    4x KV overestimate this phase exists to remove — the same model derives 0.62, well
    outside the +/-0.02 band. Without this, the calibration test would still pass if hybrid
    classification regressed entirely.
    """
    row = _row(CALIBRATION_MODEL)
    naive_config = config_from_fixture(row)
    naive_config.pop("layer_types")  # fall through to the dense branch: all 48 layers count
    naive = appmod._derive_launch_spec(naive_config, weights_gb=row["weight_bytes"] / 1e9)
    assert naive["full_attention_layers"] == 48
    assert naive["kv_bytes_per_token"] == 4 * 12288
    assert naive["recommended_util"] == 0.62
    assert abs(naive["recommended_util"] - HAND_MEASURED_UTIL) > CALIBRATION_TOLERANCE


def test_calibration_qwen36_gap_is_left_for_phase_3():
    """Qwen3.6 derives ~0.42 against a 0.55 recipe — deliberately NOT closed here.

    The recipe reserves headroom the formula does not model. That is exactly why curated
    recipes must win over derived values, which is Phase 3's precedence work. Do not "fix"
    the formula to chase 0.55: doing so would break the qwen3-next calibration above, which
    is the only number here validated against a measurement.
    """
    spec = _spec_for(_row("Qwen/Qwen3.6-35B-A3B-FP8"))
    assert spec["recommended_util"] < HAND_MEASURED_UTIL
    assert round(spec["recommended_util"], 2) == 0.42


# ── V8 purity ─────────────────────────────────────────────────────────────────

# All four derived-spec helpers, not just the two newest. The first two were ported from a
# prototype whose documented impurity was scanning the model cache directories, so scanning
# only the new pair would leave "the function is pure" unproven for exactly the code most
# likely to have inherited a filesystem read.
DERIVED_SPEC_HELPERS = (
    appmod._resolve_attention_topology,
    appmod._kv_bytes_per_token,
    appmod._kv_budget_gb,
    appmod._derive_launch_spec,
)
HELPER_IDS = [fn.__name__ for fn in DERIVED_SPEC_HELPERS]

# Tokens that would make these helpers unverifiable while vLLM is down (T-02-03, T-02-10).
# `open(` is forbidden as a source token in its own right, not merely blocked at runtime by
# the patched-builtin test below.
FORBIDDEN_SOURCE_TOKENS = (
    "open(",
    "glob",
    "os.listdir",
    "os.scandir",
    "os.walk",
    "os.path",
    "subprocess",
    "requests",
    "httpx",
    "urllib",
    "/proc",
    "expanduser",
    "Path(",
    "_get_total_memory_gb",
    "_get_available_memory_gb",
    "import vllm",
    "docker",
)


@pytest.mark.parametrize("fn", DERIVED_SPEC_HELPERS, ids=HELPER_IDS)
def test_purity_no_filesystem_or_network_in_helper_bodies(fn):
    """No filesystem, network or vLLM dependency inside the derived-spec helper bodies."""
    source = inspect.getsource(fn)
    offenders = [token for token in FORBIDDEN_SOURCE_TOKENS if token in source]
    assert offenders == [], f"{fn.__name__} references {offenders}"


@pytest.mark.parametrize("fn", DERIVED_SPEC_HELPERS, ids=HELPER_IDS)
def test_purity_helpers_take_only_plain_arguments(fn):
    """A pure helper's inputs are a dict and scalars — never a path or a handle."""
    params = inspect.signature(fn).parameters
    assert "path" not in params and "model_dir" not in params


def test_purity_helpers_run_under_a_patched_builtin_open(monkeypatch):
    """Source inspection alone would miss a read made indirectly through another helper."""
    def _forbidden(*args, **kwargs):
        raise AssertionError("the derived-spec helpers must not read the filesystem")

    row = _row(CALIBRATION_MODEL)
    config = config_from_fixture(row)
    weights = row["weight_bytes"] / 1e9
    monkeypatch.setattr("builtins.open", _forbidden)

    assert isinstance(appmod._resolve_attention_topology(config), dict)
    assert isinstance(appmod._kv_bytes_per_token(config), dict)
    assert appmod._kv_budget_gb(0.55, 121.0, weights) >= 0.0
    assert appmod._derive_launch_spec(config, weights_gb=weights)["recommended_util"] == 0.54


def test_purity_pool_and_weights_are_parameters_not_lookups():
    """Changing the pool changes the answer, proving it is not read off the box."""
    config = config_from_fixture(_row(CALIBRATION_MODEL))
    small = appmod._derive_launch_spec(config, weights_gb=50.0, pool_gb=80.0)
    large = appmod._derive_launch_spec(config, weights_gb=50.0, pool_gb=200.0)
    assert small["recommended_util"] > large["recommended_util"]


# ── V9 degenerate configs ─────────────────────────────────────────────────────

SIXTEEN_KEYS = {
    "topology", "num_hidden_layers", "full_attention_layers", "bounded_kv_layers",
    "stateless_layers", "source_field", "kv_bytes_per_token", "bounded_bytes_total",
    "declared_max_context", "max_fitting_context", "max_model_len", "kv_gb",
    "weights_gb", "overhead_gb", "recommended_util", "warnings",
}

# Nine ways a vendor config or a Phase 4 caller can be wrong. None may raise.
DEGENERATE_CASES = (
    ("empty config", {}, {}),
    ("negative layer count", {"num_hidden_layers": -5}, {}),
    ("string layer count", {"num_hidden_layers": "48"}, {}),
    ("zero attention heads",
     {"num_hidden_layers": 32, "num_attention_heads": 0, "hidden_size": 4096}, {}),
    ("absurd layer count",
     {"num_hidden_layers": 10 ** 9, "num_key_value_heads": 8, "head_dim": 128,
      "max_position_embeddings": 262144}, {}),
    ("non-numeric weights", {"num_hidden_layers": 4}, {"weights_gb": "banana"}),
    ("negative weights", {"num_hidden_layers": 4}, {"weights_gb": -40.0}),
    ("zero pool", {"num_hidden_layers": 4}, {"pool_gb": 0}),
    ("absurd overhead", {"num_hidden_layers": 4}, {"overhead_gb": 1e12}),
)
DEGENERATE_IDS = [case[0] for case in DEGENERATE_CASES]


@pytest.mark.parametrize("label,config,kwargs", DEGENERATE_CASES, ids=DEGENERATE_IDS)
def test_degenerate_configs_return_defined_values(label, config, kwargs):
    """Empty dict, missing num_hidden_layers and zero heads yield zeros, never a traceback."""
    spec = appmod._derive_launch_spec(config, **kwargs)
    assert set(spec) == SIXTEEN_KEYS, label
    assert 0.10 <= spec["recommended_util"] <= 0.95, label
    assert spec["max_model_len"] >= 0 and spec["kv_gb"] >= 0.0, label
    assert isinstance(spec["warnings"], list), label


def test_degenerate_absurd_weights_saturate_at_the_cap():
    """T-02-08: a nonsense config must not request more memory than the box has."""
    spec = appmod._derive_launch_spec(
        {"num_hidden_layers": 48, "num_key_value_heads": 2, "head_dim": 256,
         "max_position_embeddings": 262144},
        weights_gb=10_000.0,
    )
    assert spec["recommended_util"] == 0.95
    assert spec["max_fitting_context"] == 0


def test_degenerate_bad_scalars_are_named_in_the_warnings():
    """T-02-06: a coerced input must leave a trail, not silently become zero."""
    spec = appmod._derive_launch_spec({"num_hidden_layers": 4}, weights_gb="banana",
                                      resident_gb=-3.0)
    assert any("weights_gb" in w for w in spec["warnings"])
    assert any("resident_gb" in w for w in spec["warnings"])


# ── 17-model derived spec table ───────────────────────────────────────────────

HYBRID_ROWS = [row for row in MODEL_ROWS if row["precedence_field"] != "dense"]
HYBRID_IDS = [row["name"] for row in HYBRID_ROWS]


@pytest.mark.parametrize("row", MODEL_ROWS, ids=MODEL_IDS)
def test_spec_table_every_model_derives_a_launchable_spec(row):
    """Every config on this box produces numbers a launch could actually use."""
    spec = _spec_for(row)
    name = row["name"]

    assert 0.10 <= spec["recommended_util"] <= 0.95, name
    assert spec["max_model_len"] <= row["max_position_embeddings"], name
    assert spec["max_model_len"] <= spec["max_fitting_context"], name
    assert spec["declared_max_context"] == row["max_position_embeddings"], name

    expected_kv = (spec["kv_bytes_per_token"] * spec["max_model_len"]
                   + spec["bounded_bytes_total"]) / 1e9
    assert abs(spec["kv_gb"] - expected_kv) < 1e-9, name

    assert (spec["full_attention_layers"] + spec["bounded_kv_layers"]
            + spec["stateless_layers"]) == spec["num_hidden_layers"] == row["num_hidden_layers"]
    assert isinstance(spec["warnings"], list), name

    # The independent signal: which branch of the precedence chain the REAL config selected.
    # The rebuilt fixture config cannot fake this, so it is what makes the table meaningful
    # rather than self-fulfilling.
    assert spec["source_field"] == row["precedence_field"], name

    verbatim = row.get("layer_types")
    if isinstance(verbatim, list) and verbatim:
        assert verbatim.count("full_attention") == spec["full_attention_layers"], name


@pytest.mark.parametrize("row", HYBRID_ROWS, ids=HYBRID_IDS)
def test_spec_table_hybrid_models_cost_less_than_the_naive_dense_estimate(row):
    """The 4-11x overestimate this phase exists to remove, asserted per hybrid model."""
    spec = _spec_for(row)
    assert spec["full_attention_layers"] < row["num_hidden_layers"], row["name"]

    per_layer = spec["kv_bytes_per_token"] // max(1, spec["full_attention_layers"])
    naive_kv_gb = per_layer * row["num_hidden_layers"] * spec["max_model_len"] / 1e9
    assert spec["kv_gb"] < naive_kv_gb, row["name"]


def test_spec_table_covers_both_hybrid_and_dense_families():
    """Guard the table's own coverage: a snapshot of only dense models would prove nothing."""
    assert len(HYBRID_ROWS) >= 6
    assert len(MODEL_ROWS) - len(HYBRID_ROWS) >= 6
