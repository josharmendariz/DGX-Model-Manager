"""Phase 04-01 criterion 2: per-launch overrides reach the launched process env,
invalid ones are rejected at the HTTP boundary, and the script on disk is never
rewritten to carry them.

No GPU, no docker, no model-cache scan: the launched thing is a `#!/bin/bash` stub.
"""

import asyncio
import hashlib
import shutil
import subprocess

import pytest
from fastapi import HTTPException

import app as appmod


def _stub_script(tmp_path, body="exit 0\n"):
    path = tmp_path / "stub.sh"
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)
    return path


# ── override_allowlist ────────────────────────────────────────────────────────

def test_override_allowlist_rejects_unknown_key():
    with pytest.raises(HTTPException) as exc:
        appmod._resolve_overrides({"tensor_parallel_size": 4})
    assert exc.value.status_code == 400
    assert "tensor_parallel_size" in exc.value.detail
    # the message must name what IS accepted, or the caller is left guessing
    assert "max_model_len" in exc.value.detail


@pytest.mark.parametrize("bad", [
    {"gpu_memory_utilization": 1.5},
    {"gpu_memory_utilization": 0.05},
    {"max_model_len": "abc"},
    {"max_model_len": 0},
    {"max_num_seqs": 0},
    {"max_num_seqs": 257},
])
def test_override_allowlist_rejects_out_of_range_values(bad):
    with pytest.raises(HTTPException) as exc:
        appmod._resolve_overrides(bad)
    assert exc.value.status_code == 400


def test_override_allowlist_bounds_match_derive_launch_spec_defaults():
    """Two sources of truth for the same clamp is how they drift apart."""
    import inspect
    sig = inspect.signature(appmod._derive_launch_spec)
    assert appmod._UTIL_FLOOR == sig.parameters["util_floor"].default
    assert appmod._UTIL_CAP == sig.parameters["util_cap"].default


def test_override_allowlist_accepts_and_stringifies_valid_values():
    resolved = appmod._resolve_overrides({
        "max_model_len": 65536,
        "gpu_memory_utilization": 0.55,
        "max_num_seqs": 4,
    })
    assert resolved == {
        "VLLM_MAX_MODEL_LEN": "65536",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.55",
        "VLLM_MAX_NUM_SEQS": "4",
    }
    assert appmod._resolve_overrides(None) == {}
    assert appmod._resolve_overrides({}) == {}


# ── env_transport ─────────────────────────────────────────────────────────────

_ECHO_BODY = 'echo "L=${VLLM_MAX_MODEL_LEN:-} U=${VLLM_GPU_MEMORY_UTILIZATION:-} S=${VLLM_MAX_NUM_SEQS:-}"\n'


def _run_with_env(argv, env):
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)


def test_env_transport_through_bare_bash_fallback(tmp_path, monkeypatch):
    """The fallback branch of `_launch_argv`, always exercised."""
    script = _stub_script(tmp_path, _ECHO_BODY)
    monkeypatch.setattr(appmod.shutil, "which", lambda name: None)
    argv = appmod._launch_argv(str(script), "t")
    assert argv[0] == "bash"

    env = {"PATH": "/usr/bin:/bin",
           **appmod._resolve_overrides({"max_model_len": 4096,
                                        "gpu_memory_utilization": 0.5,
                                        "max_num_seqs": 3})}
    out = _run_with_env(argv, env).stdout
    assert "L=4096" in out and "U=0.5" in out and "S=3" in out


@pytest.mark.skipif(not shutil.which("systemd-run"),
                    reason="systemd-run unavailable on this host")
def test_env_transport_through_systemd_run_scope(tmp_path):
    """`--scope` inherits Popen(env=) — the premise the whole phase rests on.
    Asserted against the real wrapper, not a mocked Popen call-args tuple."""
    script = _stub_script(tmp_path, _ECHO_BODY)
    argv = appmod._launch_argv(str(script), "t")
    if argv[0] != "systemd-run":
        pytest.skip("no systemd-run branch taken")

    import os
    env = {**os.environ,
           **appmod._resolve_overrides({"max_model_len": 8192,
                                        "gpu_memory_utilization": 0.6,
                                        "max_num_seqs": 5})}
    res = _run_with_env(argv, env)
    if res.returncode != 0:
        pytest.skip(f"no user session bus here: {res.stderr.strip()[:120]}")
    assert "L=8192" in res.stdout and "U=0.6" in res.stdout and "S=5" in res.stdout


# ── script_not_mutated ────────────────────────────────────────────────────────

def test_script_not_mutated_by_a_launch_with_overrides(tmp_path, monkeypatch):
    """Criterion 2: the knob must not rewrite the file it tunes."""
    script = _stub_script(tmp_path)
    before = hashlib.sha256(script.read_bytes()).hexdigest()

    profile = {"id": "p1", "name": "Stub", "script": str(script), "vram_gb": 10.0}

    async def _admit(*a, **kw):
        return None

    monkeypatch.setattr(appmod, "_vram_admission_check", _admit)
    monkeypatch.setattr(appmod.shutil, "which", lambda name: None)
    res = asyncio.run(appmod._engine_start(
        "p1", lambda: [profile], "vLLM", engine_key="vllm",
        overrides={"max_model_len": 4096, "gpu_memory_utilization": 0.5}))

    assert res["ok"] is True
    assert hashlib.sha256(script.read_bytes()).hexdigest() == before


def test_invalid_override_is_rejected_before_any_launch(tmp_path, monkeypatch):
    script = _stub_script(tmp_path)
    profile = {"id": "p1", "name": "Stub", "script": str(script), "vram_gb": 10.0}
    launched = []

    async def _admit(*a, **kw):
        launched.append("admit")

    monkeypatch.setattr(appmod, "_vram_admission_check", _admit)
    monkeypatch.setattr(appmod.subprocess, "Popen",
                        lambda *a, **kw: launched.append("popen"))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(appmod._engine_start(
            "p1", lambda: [profile], "vLLM", engine_key="vllm",
            overrides={"gpu_memory_utilization": 9.0}))
    assert exc.value.status_code == 400
    assert launched == []
