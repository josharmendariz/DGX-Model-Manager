"""Readiness regressions from the GB10 profile audit; no Docker or GPU required."""
import asyncio
import json
import os
import subprocess
import sys

import pytest

import app as appmod


@pytest.fixture
def script(tmp_path):
    return f'''#!/bin/bash
exec docker run --rm --name vllm_node \\
  -v "{tmp_path}:/models:ro" \\
  -e HF_HUB_OFFLINE=1 \\
  vllm/vllm-openai:test \\
  --model /models/model \\
  --gpu-memory-utilization ${{VLLM_GPU_MEMORY_UTILIZATION:-0.35}} \\
  --max-model-len ${{VLLM_MAX_MODEL_LEN:-32768}}
docker update --restart no vllm_node
'''


def docker_stub(monkeypatch, output="CONFIG_READABLE\n", stderr="", code=0, image=True):
    calls = []

    async def run(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("docker", "image", "inspect"):
            return subprocess.CompletedProcess(args, 0 if image else 1,
                                               stdout="sha256:local" if image else "", stderr="")
        if args[:2] == ("docker", "ps"):
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(args, code, stdout=output, stderr=stderr)

    monkeypatch.setattr(appmod, "_run", run)
    return calls


def check_named(checks, name):
    return next(c for c in checks if c["check"] == name)


@pytest.mark.parametrize("model,warns", [
    ("/models/model", False), ("/models", False), ("/models-other/model", True),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", False),
])
def test_mount_scope_handles_ro_boundary_and_hub_ids(script, model, warns):
    checks = appmod._preflight_static(script.replace("/models/model", model), {})
    assert bool([c for c in checks if c["check"] == "mount_scope"]) is warns


@pytest.mark.parametrize("explicit", [False, True])
def test_argument_probe_checks_both_image_entrypoint_styles(script, monkeypatch, explicit):
    if explicit:
        script = script.replace("vllm/vllm-openai:test", "eugr/spark-vllm:test \\\n  vllm serve")
    facts = appmod._parse_launch_script(script)
    calls = docker_stub(monkeypatch, output="ARGS_OK\n")
    checks = asyncio.run(appmod._preflight_smoke(facts, script))
    assert checks[0]["level"] == "ok"
    call = next(a for a in calls if a[:2] == ("docker", "run"))
    assert call[call.index(appmod._PF_ARGPARSE_PROBE) + 1:] == (
        "--model", "/models/model", "--gpu-memory-utilization", "0.35",
        "--max-model-len", "32768")
    assert "vllm" not in call and "docker update" not in " ".join(call)
    assert call[call.index("--network") + 1] == "none"
    assert call[call.index("--pull") + 1] == "never"
    assert "--gpus" not in call and "-v" not in call
    assert any(a.endswith("dst=/models,readonly") for a in call)


@pytest.mark.parametrize("explicit", [False, True])
def test_bad_server_flag_still_fails_for_both_styles(script, monkeypatch, explicit):
    if explicit:
        script = script.replace("vllm/vllm-openai:test", "eugr/spark-vllm:test \\\n  vllm serve")
    script = script.replace("--max-model-len", "--bad --max-model-len")
    calls = docker_stub(monkeypatch, output="", stderr="error: unrecognized arguments: --bad", code=2)
    result = asyncio.run(appmod._preflight_smoke(appmod._parse_launch_script(script), script))
    assert result[0]["level"] == "fail"
    assert "--bad" in next(a for a in calls if a[:2] == ("docker", "run"))


@pytest.mark.parametrize("failure", ["missing_mount", "dynamic_mount", "missing_image"])
def test_argument_probe_cannot_create_directories_or_pull_images(script, monkeypatch, tmp_path, failure):
    if failure == "missing_mount":
        script = script.replace(str(tmp_path), str(tmp_path / "absent"))
    elif failure == "dynamic_mount":
        script = script.replace(str(tmp_path), "$UNKNOWN")
    calls = docker_stub(monkeypatch, image=failure != "missing_image")
    checks = asyncio.run(appmod._preflight_smoke(appmod._parse_launch_script(script), script))
    assert checks[0]["level"] == "skip"
    assert not any(a[:2] == ("docker", "run") for a in calls)
    assert not (tmp_path / "absent").exists()


def test_unknown_shell_expressions_skip_without_execution(script, monkeypatch, tmp_path):
    sentinel = tmp_path / "should-not-exist"
    script = script.replace("${VLLM_MAX_MODEL_LEN:-32768}", f'$(touch {sentinel})')
    calls = docker_stub(monkeypatch)
    checks = asyncio.run(appmod._preflight_smoke(appmod._parse_launch_script(script), script))
    assert checks[0]["level"] == "skip"
    assert not calls and not sentinel.exists()


@pytest.mark.parametrize("model,offline,marker,expected", [
    ("org/model", True, "CONFIG_READABLE", "ok"),
    ("org/model", True, "CONFIG_MISSING", "fail"),
    ("org/model", False, "CONFIG_MISSING", "warn"),
    ("/models/model", True, "CONFIG_MISSING", "fail"),
    ("org/model", True, "CONFIG_PROBE_UNAVAILABLE", "skip"),
])
def test_model_probe_distinguishes_cache_absence_from_probe_failure(
        script, monkeypatch, model, offline, marker, expected):
    script = script.replace("/models/model", model)
    if not offline:
        script = script.replace("-e HF_HUB_OFFLINE=1", "-e HF_HUB_OFFLINE=0")
    calls = docker_stub(monkeypatch, output=marker + "\n")
    checks = asyncio.run(appmod._preflight_runtime(appmod._parse_launch_script(script), script))
    assert check_named(checks, "mount_readable")["level"] == expected
    call = next(a for a in calls if a[:2] == ("docker", "run"))
    assert call[-2:] == (model, "main")
    assert "--gpus" not in call and "-v" not in call
    assert "HF_HUB_OFFLINE=1" in call and "none" in call and "never" in call


def test_hub_revision_is_passed_to_offline_cache_lookup(script, monkeypatch):
    script = script.replace("/models/model", "org/model").replace(
        "--max-model-len", "--revision=abc123 --max-model-len")
    calls = docker_stub(monkeypatch)
    asyncio.run(appmod._preflight_runtime(appmod._parse_launch_script(script), script))
    assert next(a for a in calls if a[:2] == ("docker", "run"))[-1] == "abc123"


@pytest.mark.parametrize("extra", ["-e HF_HOME=/custom", "--download-dir /custom"])
def test_custom_hub_cache_is_unknown_rather_than_false_missing(script, monkeypatch, extra):
    script = script.replace("/models/model", "org/model")
    if extra.startswith("--"):
        script = script.replace("--model", extra + " --model")
    else:
        script = script.replace("-e HF_HUB_OFFLINE=1", "-e HF_HUB_OFFLINE=1 " + extra)
    calls = docker_stub(monkeypatch, output="CONFIG_MISSING\n")
    checks = asyncio.run(appmod._preflight_runtime(appmod._parse_launch_script(script), script))
    assert check_named(checks, "mount_readable")["level"] == "skip"
    assert not any(a[:2] == ("docker", "run") for a in calls)


@pytest.mark.parametrize("missing", [False, True])
def test_actual_config_probe_reads_hf_cache_and_detects_dangling_symlink(tmp_path, missing):
    cache = tmp_path / "hub"
    repo = cache / "models--org--model"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    (repo / "blobs").mkdir()
    if not missing:
        (repo / "blobs" / "config").write_text(json.dumps({"model_type": "test"}))
    (snapshot / "config.json").symlink_to("../../blobs/config")
    env = {**os.environ, "HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"}
    result = subprocess.run([sys.executable, "-c", appmod._PF_MODEL_CONFIG_PROBE,
                             "org/model", "main"], env=env, capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0
    assert result.stdout.strip() == ("CONFIG_MISSING" if missing else "CONFIG_READABLE")


@pytest.mark.parametrize("failure", ["missing_mount", "missing_image", "named_volume"])
def test_runtime_probe_does_not_create_sources_or_pull(script, tmp_path, monkeypatch, failure):
    if failure == "missing_mount":
        script = script.replace(str(tmp_path), str(tmp_path / "absent"))
    elif failure == "named_volume":
        script = script.replace(str(tmp_path), "model-volume")
    calls = docker_stub(monkeypatch, image=failure != "missing_image")
    checks = asyncio.run(appmod._preflight_runtime(appmod._parse_launch_script(script), script))
    name = "mount_source" if failure == "missing_mount" else "mount_readable"
    assert check_named(checks, name)["level"] == ("fail" if failure == "missing_mount" else "skip")
    assert not any(a[:2] == ("docker", "run") for a in calls)
    assert not (tmp_path / "absent").exists()


def test_comment_only_offline_setting_cannot_make_online_cache_miss_fail(script, monkeypatch):
    script = script.replace("/models/model", "org/model").replace("-e HF_HUB_OFFLINE=1", "")
    script = "# HF_HUB_OFFLINE=1 --revision=old HF_HOME=/old\n" + script
    docker_stub(monkeypatch, output="CONFIG_MISSING\n")
    checks = asyncio.run(appmod._preflight_runtime(appmod._parse_launch_script(script), script))
    assert check_named(checks, "mount_readable")["level"] == "warn"


def test_docker_failure_is_unknown_even_if_output_contains_success_marker(script, monkeypatch):
    docker_stub(monkeypatch, output="ARGS_OK\n", code=1)
    checks = asyncio.run(appmod._preflight_smoke(appmod._parse_launch_script(script), script))
    assert checks[0]["level"] == "skip"


def test_numeric_launch_defaults_do_not_expand_other_environment_defaults(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "private-value")
    assert appmod._expand_script_vars("${SECRET_TOKEN:-123}", "") == "${SECRET_TOKEN:-123}"
    assert appmod._expand_script_vars("${VLLM_MAX_NUM_SEQS:-2}", "") == "2"
