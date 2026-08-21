"""Source-text tests over the profile-card render path (04-02, criterion 3).

The UI is a single Python-embedded HTML/JS blob with no test harness, so the render
template is asserted as source text. Every assertion below strips `#`-comment lines
first: prose in a comment must not be able to satisfy a gate.
"""

import inspect
import re

import app as appmod


def _code_only(source: str) -> str:
    """Drop Python comment lines so header prose cannot satisfy a gate."""
    return "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#"))


APP_SOURCE = _code_only(inspect.getsource(appmod))


def _fn(name: str) -> str:
    """The body of one JS function inside the embedded page source."""
    match = re.search(
        r"\n(?:async )?function " + name + r"\(.*?\n\}\n", APP_SOURCE, re.S)
    assert match, f"JS function {name} not found in app source"
    return match.group(0)


# ── editable state ────────────────────────────────────────────────────────────

def test_each_control_takes_its_placeholder_from_the_derived_object():
    body = _fn("renderProfileSettings")
    for field, derived_key in (
            ("max_model_len", "d.max_model_len"),
            ("gpu_memory_utilization", "d.util"),
            ("max_num_seqs", "d.max_num_seqs"),
    ):
        assert f'data-override="{field}"' in body, field
        # 04-03: every interpolation is escaped at the sink, so the guard
        # asserts the escaped form — a bare ${d.x} is now a regression.
        assert "${esc(" + derived_key + ")}" in body, derived_key
    assert "const d = p.derived" in body


def test_recommended_util_label_sits_next_to_the_utilization_input():
    body = _fn("renderProfileSettings")
    assert "rec.gpu_memory_utilization" in body
    assert "recommended ${esc(rec.gpu_memory_utilization)}" in body
    # The label markup must carry it, not a detached note elsewhere.
    assert "<label>GPU memory util ${recUtil}</label>" in body


def test_clamping_warning_is_surfaced_next_to_the_context_control():
    """04-01 carry-over: `_derive_launch_spec` CLAMPS an over-requested context — it
    warns and discards. A slider that hid that would silently under-serve the user."""
    body = _fn("renderProfileSettings")
    assert "clamped" in body
    assert "discarded, not applied" in body
    assert "d.max_fitting_context" in body and "d.declared_max_context" in body


# ── recipe-backed state ───────────────────────────────────────────────────────

def test_recipe_backed_branch_disables_the_controls_and_shows_a_reason():
    body = _fn("renderProfileSettings")
    recipe = body.split('data-state="recipe-backed"', 1)[1].split("data-state=", 1)[0]
    assert recipe.count("<input disabled") == 3
    assert "d.reason" in recipe
    assert "recipe YAML owns these flags" in recipe


# ── header-error state ────────────────────────────────────────────────────────

def test_meta_error_is_branched_on_before_classification():
    """A corrupt header is a defect, not a fallback: it must not be able to fall
    through to the legacy/read-only rendering."""
    body = _fn("renderProfileSettings")
    err = body.index("p.meta_error")
    assert err < body.index("p.derived"), "meta_error must be checked first"
    assert err < body.index('data-state="editable"')
    assert err < body.index("if (!d) return")
    assert 'data-state="header-error"' in body


def test_unparseable_header_selects_the_header_error_state(tmp_path):
    """End to end through the parser: a corrupt header yields meta_error, and the
    render path's first branch is the one that consumes it."""
    script = tmp_path / "start_corrupt.sh"
    script.write_text("#!/bin/bash\n# Name: Corrupt\n# Derived: {not json\ndocker run\n")
    meta = appmod._parse_script_meta(script)
    assert meta["meta_error"]
    assert meta["derived"] is None

    body = _fn("renderProfileSettings")
    first_state = re.search(r'data-state="([a-z-]+)"', body)
    assert first_state.group(1) == "header-error"


# ── warning text never reaches innerHTML ──────────────────────────────────────

def test_warning_and_error_text_are_written_with_textcontent():
    """T-04-07: warning strings embed {value!r} of vendor config fields."""
    assert ".p-set-warn" in APP_SOURCE
    assert "warnEl.textContent" in APP_SOURCE
    assert "errEl.textContent" in APP_SOURCE
    body = _fn("renderProfileSettings")
    assert "p.warnings" not in body, "warnings must not be interpolated into innerHTML"
    assert "p.meta_error ||" not in body


# ── request-body builder ──────────────────────────────────────────────────────

def test_blank_input_sends_no_override():
    assert appmod._collect_overrides({"max_model_len": ""}) == {}
    assert appmod._collect_overrides({"gpu_memory_utilization": "   "}) == {}
    assert appmod._collect_overrides({"max_num_seqs": None}) == {}
    assert appmod._collect_overrides({}) == {}
    assert appmod._collect_overrides(None) == {}


def test_filled_input_sends_the_typed_value():
    assert appmod._collect_overrides(
        {"max_model_len": 8192, "gpu_memory_utilization": 0.6}) == {
            "max_model_len": 8192, "gpu_memory_utilization": 0.6}
    # A string arrives trimmed, so the validator sees the same atom either way.
    assert appmod._collect_overrides({"max_num_seqs": " 4 "}) == {"max_num_seqs": "4"}


def test_mixed_blank_and_filled_keeps_only_the_filled_key():
    assert appmod._collect_overrides(
        {"max_model_len": "", "gpu_memory_utilization": 0.6}) == {
            "gpu_memory_utilization": 0.6}


def test_blank_overrides_do_not_reach_the_env():
    """The server half of the pair: a hand-rolled client sending "" must not 400 —
    and must not set the variable either."""
    assert appmod._resolve_overrides({"max_model_len": "", "max_num_seqs": " "}) == {}
    assert appmod._resolve_overrides({"max_model_len": " 4096 "}) == {
        "VLLM_MAX_MODEL_LEN": "4096"}


def test_js_builder_mirrors_the_python_rule():
    body = _fn("collectProfileOverrides")
    assert ".trim()" in body
    assert "if (!raw) return;" in body
    assert "Number.isFinite" in body
    assert 'data-state="editable"' in body, "only the editable panel may contribute"


def test_start_request_carries_the_overrides_object():
    assert "overrides: eng.key === 'vllm' ? collectProfileOverrides() : undefined" in APP_SOURCE


def test_overrides_reset_when_the_selection_changes():
    """Per-launch only: nothing persisted, and the inputs clear on re-selection."""
    body = _fn("selectEngineProfile")
    assert "input[data-override]" in body
    assert "i.value = ''" in body


# ── page-cache reclaim (Pitfall 4) ────────────────────────────────────────────

def test_reclaim_cache_is_offered_alongside_the_utilization_control():
    assert APP_SOURCE.count("reclaim-cache") >= 2
    assert "reclaimPageCache" in _fn("renderProfileSettings")
    assert "/api/vllm/reclaim-cache" in _fn("reclaimPageCache")
