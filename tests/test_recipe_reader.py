"""Recipe YAML normalization and degradation behavior."""

from pathlib import Path

import pytest

import app as appmod


RECIPES = Path(__file__).parent / "fixtures" / "recipes"


def test_all_committed_recipe_fixtures_round_trip():
    for path in RECIPES.glob("*.yaml"):
        record, warnings = appmod._read_recipe(RECIPES, path.stem)
        assert record is not None, (path.name, warnings)
        assert warnings == []

    record, _ = appmod._read_recipe(RECIPES, "qwen3.6-35b-a3b-fp8-solo")
    assert record == {
        "name": "Qwen36-35B-A3B-solo",
        "description": "vLLM serving Qwen3.6-35B-A3B-FP8 (solo, LiteLLM vllm-active)",
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "gpu_memory_utilization": 0.55,
        "gpu_memory_utilization_gb": None,
        "max_model_len": 262144,
        "kv_cache_dtype": "fp8",
        "tool_call_parser": "qwen3_xml",
        "reasoning_parser": "qwen3",
        "port": 8000,
        "solo_only": True,
        "cluster_only": None,
        "mods": ["mods/fix-qwen3.6-chat-template"],
    }


def test_gpu_memory_gb_flag_is_not_a_fraction(tmp_path, monkeypatch):
    step, warnings = appmod._read_recipe(RECIPES, "step-3.7-flash-fp8")
    assert warnings == []
    assert step["gpu_memory_utilization"] is None
    assert step["gpu_memory_utilization_gb"] == 108.0

    (tmp_path / "qwen3.5-397b-int4-autoround.yaml").write_text(
        "defaults:\n  gpu_memory_utilization: 108\n"
        "command: vllm serve model --gpu-memory-utilization-gb {gpu_memory_utilization}\n"
    )
    other, warnings = appmod._read_recipe(tmp_path, "qwen3.5-397b-int4-autoround")
    assert warnings == []
    assert other["gpu_memory_utilization"] is None
    assert other["gpu_memory_utilization_gb"] == 108.0
    real_read = appmod._read_recipe
    monkeypatch.setattr(
        appmod, "_read_recipe",
        lambda _recipe_dir, name: real_read(
            RECIPES if name == "step-3.7-flash-fp8" else tmp_path, name))
    assert appmod._recipe_util("step-3.7-flash-fp8") is None
    assert appmod._recipe_util("qwen3.5-397b-int4-autoround") is None


def test_dflash_quoted_json_remains_one_shell_token(monkeypatch):
    real_split = appmod.shlex.split
    seen = []

    def recording_split(command):
        tokens = real_split(command)
        seen.extend(tokens)
        return tokens

    monkeypatch.setattr(appmod.shlex, "split", recording_split)
    record, warnings = appmod._read_recipe(RECIPES, "qwen3.6-35b-a3b-fp8-dflash")
    assert record is not None and warnings == []
    value = seen[seen.index("--speculative-config") + 1]
    assert value == '{"method": "dflash", "model": "z-lab/Qwen3.6-35B-A3B-DFlash", "num_speculative_tokens": 15}'


@pytest.mark.parametrize("failure", [
    "missing_dir", "missing_file", "malformed_yaml", "yaml_string",
    "missing_command", "format_error", "shlex_error",
])
def test_reader_failure_modes_degrade_to_warning(tmp_path, failure):
    recipe_dir = tmp_path / "recipes"
    if failure != "missing_dir":
        recipe_dir.mkdir()
    if failure not in ("missing_dir", "missing_file"):
        contents = {
            "malformed_yaml": "defaults: [\n",
            "yaml_string": "just text\n",
            "missing_command": "defaults: {}\n",
            "format_error": "command: 'vllm serve {missing}'\n",
            "shlex_error": "command: \"vllm serve 'unterminated\"\n",
        }[failure]
        (recipe_dir / "broken.yaml").write_text(contents)

    record, warnings = appmod._read_recipe(recipe_dir, "broken")
    assert record is None
    assert len(warnings) == 1


def test_reader_rejects_unsafe_name_before_filesystem_access():
    class ExplodingPath:
        def __fspath__(self):
            pytest.fail("filesystem touched")

    record, warnings = appmod._read_recipe(ExplodingPath(), "../escape/path")
    assert record is None
    assert warnings
