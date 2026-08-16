"""Capability-map lookup and generated vLLM parser emission."""

import json
from pathlib import Path

import pytest

import app as appmod


SC4_VICTIMS = (
    "qwen2.5-14b-instruct-gptq-int8",
    "qwen3-vl-4b-fp8",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "lyf/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4",
)


def _fixture_model_dir(tmp_path: Path, row: dict) -> Path:
    config_path = Path(row["config_path"])
    if not config_path.exists():
        pytest.skip(f"survey model config is not mounted: {config_path}")
    model_dir = tmp_path / row["name"].replace("/", "--")
    model_dir.mkdir()
    (model_dir / "config.json").write_text(config_path.read_text())
    (model_dir / "model.safetensors").write_bytes(b"x")
    return model_dir


@pytest.mark.parametrize("model_name", SC4_VICTIMS)
def test_sc4_victims_use_only_proven_parser_flags(
        tmp_path, monkeypatch, fixture_models, model_name):
    row = next(row for row in fixture_models if row["name"] == model_name)
    model_dir = _fixture_model_dir(tmp_path, row)
    monkeypatch.setitem(appmod._app_config, "vllm", {})

    _, script, _ = appmod._build_vllm_profile_script(model_dir, model_name)

    assert "qwen3_coder" not in script
    if model_name.startswith("lyf/"):
        assert "--enable-auto-tool-choice" in script
        assert "--tool-call-parser qwen3_xml" in script
        assert "--reasoning-parser qwen3" in script
    else:
        assert "--tool-call-parser" not in script
        assert "--enable-auto-tool-choice" not in script


def test_unmapped_generated_model_gets_no_parser_flags(tmp_path, monkeypatch):
    model_dir = tmp_path / "flat"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "acme", "architectures": ["AcmeForCausalLM"],
    }))
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setitem(appmod._app_config, "vllm", {})

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Flat")

    for flag in ("--tool-call-parser", "--enable-auto-tool-choice",
                 "--reasoning-parser"):
        assert flag not in script


def test_whisper_preserves_positive_no_tool_support_finding():
    entry = appmod._capability_entry({
        "name": "Systran/faster-whisper-base", "architectures": [],
    })

    assert entry is not None
    assert entry["supports_tool_calling"] is False
    assert appmod._capability_emission(entry) == (None, None)


def test_exact_id_precedes_shared_architecture_and_lookup_is_casefolded():
    coder = appmod._capability_entry({
        "name": "QWEN3-CODER-NEXT-NVFP4",
        "architectures": ["Qwen3NextForCausalLM"],
    })
    inferred = appmod._capability_entry({
        "name": "qwen3-next-80b-a3b-nvfp4",
        "architectures": ["Qwen3NextForCausalLM"],
    })

    assert coder["tool_call_parser"] == "qwen3_coder"
    assert appmod._capability_emission(coder) == ("qwen3_coder", None)
    assert inferred["tool_call_parser"] == "hermes"
    assert appmod._capability_emission(inferred) == (None, None)


def test_loader_failure_is_empty_and_live_reload_is_per_call(
        tmp_path, monkeypatch):
    capabilities_path = tmp_path / "model_capabilities.json"
    capabilities_path.write_text("{broken")
    monkeypatch.setattr(appmod, "_MODEL_CAPABILITIES_FILE", capabilities_path)
    monkeypatch.setitem(appmod._app_config, "vllm", {})
    assert appmod._load_model_capabilities() == {"meta": {}, "models": []}

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
    }))
    (model_dir / "model.safetensors").write_bytes(b"x")
    _, broken_script, _ = appmod._build_vllm_profile_script(
        model_dir, "Qwen/Reload-Probe")
    assert "--tool-call-parser" not in broken_script

    capabilities_path.write_text(json.dumps({"meta": {}, "models": [{
        "matches": ["Qwen/Reload-Probe"],
        "tool_call_parser": "hermes",
        "reasoning_parser": None,
        "supports_tool_calling": True,
        "confidence": "recipe-proven",
        "evidence": "test fixture",
        "emit": True,
    }]}))
    _, fixed_script, _ = appmod._build_vllm_profile_script(
        model_dir, "Qwen/Reload-Probe")
    assert "--tool-call-parser hermes" in fixed_script


def test_invalid_parser_grammar_is_dropped_and_warned(tmp_path, monkeypatch):
    invalid = "hermes;touch-pwned"
    monkeypatch.setattr(appmod, "_load_model_capabilities", lambda: {
        "meta": {},
        "models": [{
            "matches": ["Acme/Grammar"],
            "tool_call_parser": invalid,
            "reasoning_parser": None,
            "supports_tool_calling": True,
            "confidence": "recipe-proven",
            "evidence": "test fixture",
            "emit": True,
        }],
    })
    monkeypatch.setitem(appmod._app_config, "vllm", {})
    model_dir = tmp_path / "grammar"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "acme", "architectures": ["AcmeForCausalLM"],
    }))
    (model_dir / "model.safetensors").write_bytes(b"x")

    _, script, info = appmod._build_vllm_profile_script(model_dir, "Acme/Grammar")

    assert invalid not in script
    assert "--tool-call-parser" not in script
    assert "--enable-auto-tool-choice" not in script
    assert any("tool_call_parser" in warning for warning in info["warnings"])


def test_ambiguous_architecture_families_emit_nothing_for_unknown_models():
    """A model unknown to the map must not inherit a first-row parser just by
    architecture, when that architecture maps to different parsers across rows.
    (NemotronHForCausalLM: nano vs super; Qwen2ForCausalLM: hermes vs none;
    Qwen3NextForCausalLM: qwen3_coder vs hermes.)"""
    import app as appmod
    for arch in ("NemotronHForCausalLM", "Qwen2ForCausalLM", "Qwen3NextForCausalLM"):
        info = {"name": "Unknown/Future-Model", "architectures": [arch]}
        assert appmod._capability_emission(appmod._capability_entry(info)) == (None, None), arch


def test_unanimous_architecture_family_still_resolves():
    """Qwen3_5MoeForConditionalGeneration is a family-key ONLY because every row
    carrying it agrees on the emitted values; unknown members of that family get
    the agreed parser, not silence."""
    import app as appmod
    info = {"name": "Unknown/Qwen3.6-Deriv", "architectures": ["Qwen3_5MoeForConditionalGeneration"]}
    tool, reason = appmod._capability_emission(appmod._capability_entry(info))
    assert (tool, reason) == ("qwen3_xml", "qwen3")
