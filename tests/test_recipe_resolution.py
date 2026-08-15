"""Recipe matching, KV evidence, and launch-resolution ordering."""

import json
from pathlib import Path

import pytest

import app as appmod


RECIPES = Path(__file__).parent / "fixtures" / "recipes"
MODEL_FIXTURES = (Path(__file__).parents[1] / ".planning" / "phases" /
                  "02-derived-launch-spec" / "02-MODEL-FIXTURES.json")


def test_recipe_match_and_absent_or_empty_noop():
    cfg = {"recipe_dir": str(RECIPES), "recipes": {
        "Qwen/Qwen3.6-*": "qwen3.6-35b-a3b-fp8-solo",
    }}
    assert appmod._resolve_recipe_model("qWEN/qWEN3.6-35b", cfg) == (
        "qwen3.6-35b-a3b-fp8-solo", [])
    assert appmod._resolve_recipe_model("Acme/Other", cfg) == (None, [])
    assert appmod._resolve_recipe_model("Acme/Other", {}) == (None, [])
    assert appmod._resolve_recipe_model("Acme/Other", {"recipes": {}}) == (None, [])


@pytest.mark.parametrize("model_name", ["Acme/Dense", "Qwen/Qwen-MoE", "Acme/Vision"])
def test_absent_and_empty_recipe_maps_generate_identically(
        tmp_path, monkeypatch, model_name):
    model_dir = tmp_path / model_name.split("/")[-1]
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "torch_dtype": "bfloat16",
        "architectures": ["Qwen3ForCausalLM"],
    }))
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.delitem(appmod._app_config, "vllm", raising=False)
    absent = appmod._build_vllm_profile_script(model_dir, model_name)[1]
    monkeypatch.setitem(appmod._app_config, "vllm", {"recipes": {}})
    empty = appmod._build_vllm_profile_script(model_dir, model_name)[1]
    assert absent == empty


def test_recipe_collision_is_specific_deterministic_and_warned():
    pairs = [("Qwen/*", "broad"), ("Qwen/Qwen3.6-*", "specific")]
    for mapping in (dict(pairs), dict(reversed(pairs))):
        name, warnings = appmod._resolve_recipe_model(
            "Qwen/Qwen3.6-35B", {"recipes": mapping})
        assert name == "specific"
        assert len(warnings) == 1


def test_recipe_name_grammar_rejected():
    name, warnings = appmod._resolve_recipe_model(
        "Acme/Model", {"recipes": {"Acme/*": "../escape"}})
    assert name is None
    assert warnings


@pytest.mark.parametrize("config, expected", [
    ({}, None),
    ({"quantization_config": None}, None),
    ({"quantization_config": {"quant_method": "gptq"}}, None),
    ({"quantization_config": {"quant_method": "mxfp4"}}, None),
    ({"quantization_config": {"quant_method": "fp8"}}, None),
    ({"quantization_config": {"kv_cache_scheme": None, "quant_method": "fp8"}}, None),
    ({"quantization_config": {"kv_cache_dtype": "fp8"}}, "fp8"),
    ({"quantization_config": {"kv_cache_scheme": {"num_bits": 8, "type": "float"}}}, "fp8"),
    ({"quantization_config": {"kv_cache_scheme": {"num_bits": 8, "dtype": "float8_e4m3fn"}}}, "fp8"),
    ({"quantization_config": {"kv_cache_scheme": {"num_bits": 8, "type": "int"}}}, None),
])
def test_kv_dtype_requires_model_kv_evidence(config, expected):
    assert appmod._kv_dtype_from_config(config) == expected


def test_resolve_launch_recipe_and_warning_surfaces(tmp_path):
    recipe = (tmp_path / "recipes")
    recipe.mkdir()
    (recipe / "cluster.yaml").write_text(
        "cluster_only: true\ndefaults: {port: 9000}\n"
        "command: vllm serve model --port {port}\n")
    result = appmod._resolve_launch({}, {"name": "Acme/Model", "size_gb": 1}, {
        "recipe_dir": str(recipe), "recipes": {"Acme/*": "cluster"}})
    assert result["shape"] == "recipe"
    assert result["record"]["cluster_only"] is True
    assert any("cluster_only" in w for w in result["warnings"])
    assert any("port" in w for w in result["warnings"])


def test_kv_dtype_is_decided_before_derivation_and_script_matches(tmp_path, monkeypatch):
    rows = json.loads(MODEL_FIXTURES.read_text())["models"]
    row = next(r for r in rows if r["name"] == "Qwen/Qwen3.6-35B-A3B-FP8")
    config_path = Path(row["config_path"])
    if not config_path.exists():
        pytest.skip("survey model config is not mounted")
    config = json.loads(config_path.read_text())
    fp8 = appmod._derive_launch_spec(config, weights_gb=37.0, kv_dtype_bytes=1)
    full = appmod._derive_launch_spec(config, weights_gb=37.0, kv_dtype_bytes=2)
    assert (fp8["recommended_util"], full["recommended_util"]) == (0.42, 0.44)

    model_dir = tmp_path / "qwen36"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setitem(appmod._app_config, "vllm", {})
    monkeypatch.setattr(appmod, "_profile_model_info", lambda *_: {
        "name": row["name"], "served": row["name"].replace("/", "--"),
        "fmt": "safetensors", "dtype": "FP8", "is_moe": True,
        "modalities": ["Text"], "task_label": "Text Gen", "size_gb": 37.0,
        "vram_gb": 60, "architectures": row["architectures"],
    })
    _, script, _ = appmod._build_vllm_profile_script(model_dir, row["name"])
    assert "--kv-cache-dtype" not in script
    assert "--gpu-memory-utilization 0.44" in script


def test_gpt_oss_literal_branch_wins_even_over_wrong_mapping(tmp_path, monkeypatch):
    model_dir = tmp_path / "gptoss"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}}))
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setitem(appmod._app_config, "vllm", {
        "recipe_dir": str(RECIPES),
        "recipes": {"openai/gpt-oss-*": "openai-gpt-oss-120b"},
    })
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "openai/gpt-oss-120b")
    assert "run-recipe.sh" not in script
    assert "--max-model-len 65536" in script
    assert "--kv-cache-dtype" not in script
    for line in (
        "TIKTOKEN_ENCODINGS_BASE=/root/.cache/huggingface/harmony-encodings",
        "VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm",
        "VLLM_MARLIN_USE_ATOMIC_ADD=1",
    ):
        assert line in script


def test_degenerate_derivation_uses_defaults_not_floor_values(tmp_path, monkeypatch):
    model_dir = tmp_path / "minimal"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"model_type": "qwen3", "torch_dtype": "bfloat16"}))
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setitem(appmod._app_config, "vllm", {})
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Minimal")
    assert "--max-model-len 32768" in script
    assert "--gpu-memory-utilization 0.75" in script
    assert "--max-model-len 0" not in script
    assert "--gpu-memory-utilization 0.1 " not in script

    resolved = appmod._resolve_launch(
        {"model_type": "qwen3", "torch_dtype": "bfloat16"},
        {"name": "Acme/Minimal", "size_gb": 0.0}, {})
    assert resolved["max_model_len"] is None
    assert resolved["util"] is None


def test_resolver_preserves_usable_positive_derivations():
    config = {
        "model_type": "qwen3", "torch_dtype": "bfloat16",
        "num_hidden_layers": 2, "num_attention_heads": 8,
        "num_key_value_heads": 2, "hidden_size": 1024,
        "max_position_embeddings": 8192,
    }
    resolved = appmod._resolve_launch(
        config, {"name": "Acme/Usable", "size_gb": 37.0}, {})
    assert resolved["max_model_len"] > 0
    assert resolved["util"] > 0.10
