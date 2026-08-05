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


# ── V5 text_config nesting ────────────────────────────────────────────────────

def test_text_config_nesting_and_hybridity_are_independent_axes():
    """Two rows nest; one is hybrid and one is dense, so the axes need separate cover."""
    nested = [row for row in MODEL_ROWS if row.get("nested_text_config")]
    assert len(nested) >= 2
    fields = {row["precedence_field"] for row in nested}
    assert "dense" in fields and fields - {"dense"}


# ── V6 head_dim fallback ──────────────────────────────────────────────────────

@pytest.mark.parametrize("row", MODEL_ROWS, ids=MODEL_IDS)
def test_head_dim_snapshot_source_is_consistent(row):
    """A derived head_dim must actually equal hidden_size // num_attention_heads."""
    assert row["head_dim_source"] in ("explicit", "hidden_size//num_attention_heads")
    if row["head_dim_source"] == "hidden_size//num_attention_heads":
        assert row["head_dim"] == row["hidden_size"] // row["num_attention_heads"], row["name"]


# ── V7 calibration (plan 02-02) ───────────────────────────────────────────────

@pytest.mark.skip(reason="plan 02-02")
def test_calibration_qwen3_next_matches_hand_measured_util():
    """qwen3-next-80b derived utilization must land within 0.02 of the measured 0.55."""


# ── V8 purity (plan 02-02) ────────────────────────────────────────────────────

@pytest.mark.skip(reason="plan 02-02")
def test_purity_no_filesystem_or_network_in_helper_bodies():
    """No filesystem, network or vLLM dependency inside the derived-spec helper bodies."""


# ── V9 degenerate configs (plan 02-02) ────────────────────────────────────────

@pytest.mark.skip(reason="plan 02-02")
def test_degenerate_configs_return_defined_values():
    """Empty dict, missing num_hidden_layers and zero heads yield zeros, never a traceback."""
