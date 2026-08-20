"""Tests for profile script parsing and model metadata inference."""

import app as appmod


# ── _parse_script_meta ────────────────────────────────────────────────────────

def _write_script(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_parse_script_meta_full_header(tmp_path):
    script = _write_script(tmp_path, "start_mistral_small4.sh", (
        "#!/bin/bash\n"
        "# Name: Mistral Small 4\n"
        "# Description: 119B NVFP4 quantized\n"
        "# VRAM: 119\n"
        "docker run …\n"
    ))
    meta = appmod._parse_script_meta(script)
    assert meta["id"] == "start_mistral_small4"
    assert meta["name"] == "Mistral Small 4"
    assert meta["description"] == "119B NVFP4 quantized"
    assert meta["vram_gb"] == 119
    assert meta["script"] == str(script)


def test_parse_script_meta_vram_gb_suffix(tmp_path):
    script = _write_script(tmp_path, "start_x.sh", "# VRAM: 70GB\n")
    assert appmod._parse_script_meta(script)["vram_gb"] == 70


def test_parse_script_meta_invalid_vram_is_none(tmp_path):
    script = _write_script(tmp_path, "start_x.sh", "# VRAM: lots\n")
    assert appmod._parse_script_meta(script)["vram_gb"] is None


def test_parse_script_meta_fallback_name_from_filename(tmp_path):
    script = _write_script(tmp_path, "start_qwen3_coder.sh", "#!/bin/bash\n")
    meta = appmod._parse_script_meta(script)
    assert meta["name"] == "Qwen3 Coder"
    assert meta["description"] == "Script: start_qwen3_coder.sh"
    assert meta["vram_gb"] is None


def test_parse_script_meta_header_only_scanned_in_first_20_lines(tmp_path):
    script = _write_script(tmp_path, "start_x.sh", "#!/bin/bash\n" * 24 + "# VRAM: 50\n")
    assert appmod._parse_script_meta(script)["vram_gb"] is None


def test_parse_script_meta_missing_file_uses_fallbacks(tmp_path):
    meta = appmod._parse_script_meta(tmp_path / "start_ghost.sh")
    assert meta["name"] == "Ghost"
    assert meta["vram_gb"] is None


# ── _infer_from_name ──────────────────────────────────────────────────────────

def test_infer_from_name_dtype_and_params():
    info = appmod._infer_from_name("Qwen3-32B-FP8")
    assert info["dtype"] == "FP8"
    assert info["params_b"] == 32


def test_infer_from_name_no_signals():
    info = appmod._infer_from_name("Llama-3.3-70B-Instruct")
    assert info["dtype"] is None
    assert info["is_moe"] is None
    assert info["is_reasoning"] is None
    assert info["extra_modalities"] == []
    assert info["params_b"] == 70


def test_infer_from_name_moe_active_params_notation():
    assert appmod._infer_from_name("Qwen3-235B-A22B")["is_moe"] is True


def test_infer_from_name_reasoning_token():
    assert appmod._infer_from_name("DeepSeek-R1-Distill-Qwen-7B")["is_reasoning"] is True


def test_infer_from_name_vision_token():
    assert "Image" in appmod._infer_from_name("Qwen2.5-VL-7B-Instruct")["extra_modalities"]


def test_infer_from_name_gguf_quant_suffix():
    assert appmod._infer_from_name("Llama-3-8B-Q4_K_M")["dtype"] == "INT4"


def test_infer_from_name_embedding_token():
    assert "Embedding" in appmod._infer_from_name("bge-large-en-v1.5")["extra_modalities"]


# ── _infer_from_config ────────────────────────────────────────────────────────

def _hints(name=""):
    return appmod._infer_from_name(name)


def test_infer_from_config_torch_dtype():
    info = appmod._infer_from_config({"torch_dtype": "bfloat16"}, _hints("plain-model"))
    assert info["dtype"] == "BF16"
    assert info["is_moe"] is False
    assert info["modalities"] == ["Text"]


def test_infer_from_config_quantization_overrides_torch_dtype():
    cfg = {"torch_dtype": "bfloat16",
           "quantization_config": {"quant_method": "fp8"}}
    assert appmod._infer_from_config(cfg, _hints("m"))["dtype"] == "FP8"


def test_infer_from_config_name_quant_hint_beats_full_precision_config():
    # Config claims BF16 but the name says NVFP4 — quantized name hint wins.
    cfg = {"torch_dtype": "bfloat16"}
    assert appmod._infer_from_config(cfg, _hints("Model-NVFP4"))["dtype"] == "FP4"


def test_infer_from_config_moe_from_experts_key():
    cfg = {"torch_dtype": "bfloat16", "num_local_experts": 8}
    assert appmod._infer_from_config(cfg, _hints("m"))["is_moe"] is True


def test_infer_from_config_moe_from_known_model_type():
    cfg = {"model_type": "mixtral"}
    assert appmod._infer_from_config(cfg, _hints("m"))["is_moe"] is True


def test_infer_from_config_vision_modality():
    cfg = {"torch_dtype": "float16", "vision_config": {"hidden_size": 1024}}
    assert "Image" in appmod._infer_from_config(cfg, _hints("m"))["modalities"]


def test_infer_from_config_empty_config_falls_back_to_name_hints():
    info = appmod._infer_from_config({}, _hints("Qwen3-30B-A3B-FP8-Thinking"))
    assert info["dtype"] == "FP8"
    assert info["is_moe"] is True
    assert info["is_reasoning"] is True


# ── 04-02: derived / recommended / warnings header round-trip ─────────────────

def _generated_script(tmp_path, model_name, **config_extra):
    """Generate a real profile script into tmp_path and return its path + info."""
    import json as _json

    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"w" * 2048)
    config = {
        "model_type": "qwen3", "torch_dtype": "bfloat16",
        "max_position_embeddings": 40960, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
    }
    config.update(config_extra)
    (model_dir / "config.json").write_text(_json.dumps(config))
    script_name, script, info = appmod._build_vllm_profile_script(model_dir, model_name)
    path = tmp_path / script_name
    path.write_text(script)
    return path, script, info


def test_derived_roundtrip_through_the_script_header(tmp_path):
    """Generation → disk → parse must return exactly what generation produced."""
    import json as _json
    import re as _re

    path, script, info = _generated_script(tmp_path, "Acme/Roundtrip-7B")
    meta = appmod._parse_script_meta(path)

    assert meta["meta_error"] is None
    assert meta["derived"] == _json.loads(
        [l for l in script.splitlines() if l.startswith("# Derived:")][0][10:].strip())
    assert meta["recommended"] == _json.loads(
        [l for l in script.splitlines() if l.startswith("# Recommended:")][0][14:].strip())
    assert meta["warnings"] == list(info["warnings"])
    assert meta["generated_at"] and meta["generated_at"].endswith("Z")

    emitted = _re.search(r"\$\{VLLM_MAX_MODEL_LEN:-(\d+)\}", script)
    assert meta["derived"]["max_model_len"] == int(emitted.group(1))


def test_derived_roundtrip_carries_warnings_to_the_card(tmp_path):
    path, _, info = _generated_script(
        tmp_path, "Acme/RoundtripWarn-7B",
        layer_types=["full_attention"] * 35 + ["weird_type"])
    meta = appmod._parse_script_meta(path)
    assert meta["warnings"], "fixture should produce a warning"
    assert meta["warnings"] == list(info["warnings"])
    assert any("unknown layer type" in w for w in meta["warnings"])


def test_legacy_script_parses_with_derived_none(tmp_path):
    """A committed pre-04-02 profile has no meta headers. That is the legacy case,
    not an error."""
    import shutil
    from pathlib import Path

    legacy_dir = Path(__file__).resolve().parents[1] / "profiles" / "vLLM"
    sources = sorted(legacy_dir.glob("start_*.sh"))
    assert sources, "no committed vLLM profiles to use as a legacy sample"
    target = tmp_path / sources[0].name
    shutil.copy(sources[0], target)

    meta = appmod._parse_script_meta(target)
    assert meta["derived"] is None
    assert meta["recommended"] is None
    assert meta["warnings"] == []
    assert meta["generated_at"] is None
    assert meta["meta_error"] is None


def test_unparseable_derived_header_sets_meta_error(tmp_path):
    """Locked decision: a header the parser cannot read is a defect, not a fallback."""
    script = _write_script(tmp_path, "start_broken.sh", (
        "#!/bin/bash\n"
        "# Name: Broken\n"
        "# Derived: {not json\n"
        "docker run …\n"
    ))
    meta = appmod._parse_script_meta(script)
    assert meta["meta_error"]
    assert "# Derived:" in meta["meta_error"]
    assert meta["derived"] is None


def test_parse_script_meta_does_no_filesystem_reads_beyond_the_script(tmp_path):
    """T-04-08: `_scan_profiles` is the hot list path; Option A forbids config reads."""
    import inspect

    raw = inspect.getsource(appmod._parse_script_meta)
    # Strip comments and the docstring: prose about not reading config.json must not
    # trip the gate, and prose must not be able to satisfy it either.
    source = "\n".join(
        line.split("#", 1)[0] for line in raw.splitlines()
        if not line.strip().startswith(('"""', "#")))
    for forbidden in ("config.json", "_profile_source_dir", "glob(", "iterdir",
                      "_derive_launch_spec", "open("):
        assert forbidden not in source, f"{forbidden} must not appear in _parse_script_meta"
    assert source.count("read_text()") == 1
