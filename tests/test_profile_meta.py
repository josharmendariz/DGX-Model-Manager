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
