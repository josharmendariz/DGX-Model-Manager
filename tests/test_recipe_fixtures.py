import json
import warnings
from pathlib import Path

import pytest
import yaml

from conftest import MODEL_FIXTURES_PATH


RECIPE_NAMES = (
    "qwen3.6-35b-a3b-fp8-solo.yaml",
    "openai-gpt-oss-120b.yaml",
    "qwen3.6-35b-a3b-fp8-dflash.yaml",
    "step-3.7-flash-fp8.yaml",
    "deepseek-v4-flash.yaml",
)
FIXTURE_RECIPE_DIR = Path(__file__).parent / "fixtures" / "recipes"
LIVE_RECIPE_DIR = Path("/home/josh/spark-vllm-docker/recipes")
ARCH_VALUES_PATH = Path(__file__).parent / "fixtures" / "model-architectures.json"
DISK_ARCHITECTURE_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
)


def test_fixture_drift():
    if not LIVE_RECIPE_DIR.exists():
        pytest.skip("live recipe dir absent")

    for name in RECIPE_NAMES:
        fixture_bytes = (FIXTURE_RECIPE_DIR / name).read_bytes()
        live_bytes = (LIVE_RECIPE_DIR / name).read_bytes()
        assert fixture_bytes == live_bytes, f"recipe fixture drift: {name}"


def test_model_fixture_architectures():
    with MODEL_FIXTURES_PATH.open(encoding="utf-8") as fh:
        model_fixture = json.load(fh)
    with ARCH_VALUES_PATH.open(encoding="utf-8") as fh:
        expected_architectures = json.load(fh)

    rows = {row["name"]: row for row in model_fixture["models"]}
    assert all("architectures" in row for row in model_fixture["models"])

    for name, architectures in expected_architectures.items():
        assert rows[name]["architectures"] == architectures, name

    for name in DISK_ARCHITECTURE_MODELS:
        config_path = Path(rows[name]["config_path"])
        if not config_path.exists():
            warnings.warn(
                f"config path absent; skipping on-disk architecture assertion: {name}",
                stacklevel=1,
            )
            continue
        with config_path.open(encoding="utf-8") as fh:
            live_architectures = json.load(fh).get("architectures")
        assert rows[name]["architectures"] == live_architectures, name


def test_fixture_yaml_parses():
    for name in RECIPE_NAMES:
        with (FIXTURE_RECIPE_DIR / name).open(encoding="utf-8") as fh:
            recipe = yaml.safe_load(fh)
        assert isinstance(recipe, dict), name
        assert "name" in recipe, name
        assert "command" in recipe, name
