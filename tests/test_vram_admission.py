"""Tests for the GB10 unified-memory admission check and reclaim credit."""

import pytest
from fastapi import HTTPException

import app as appmod


def _profile(tmp_path, pid, vram_gb, script_body=""):
    script = tmp_path / f"{pid}.sh"
    script.write_text(script_body or "#!/bin/bash\n")
    return {"id": pid, "name": pid.removeprefix("start_").replace("_", " ").title(),
            "script": str(script), "description": "", "vram_gb": vram_gb}


def _mock_engine_status(status):
    async def stub(*args, **kwargs):
        return status
    return stub


def _mock_docker_label(value):
    async def stub(container_name, label):
        return value
    return stub


# ── _running_profile_vram_credit ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_credit_prefers_dgx_profile_label_over_name_match(tmp_path, monkeypatch):
    # Both profiles alias to the same generic served name ("vllm-active", as
    # every real profile does) — without the label, substring/order matching
    # could pick either one. The label must win regardless of script content.
    profiles = [
        _profile(tmp_path, "start_wrong", 20, "--served-model-name vllm-active\n"),
        _profile(tmp_path, "start_right", 60, "--served-model-name vllm-active\n"),
    ]
    monkeypatch.setattr(appmod, "_engine_status", _mock_engine_status({
        "running": True, "model": "vllm-active",
        "instances": [{"name": "vllm_node", "running": True}],
    }))
    monkeypatch.setattr(appmod, "_docker_label", _mock_docker_label("right"))
    credit, pid = await appmod._running_profile_vram_credit("vllm", profiles)
    assert credit == 60.0
    assert pid == "start_right"


@pytest.mark.asyncio
async def test_credit_falls_back_to_name_match_when_label_absent(tmp_path, monkeypatch):
    profiles = [_profile(tmp_path, "start_big", 60, "--served-model-name qwen3-coder\n")]
    monkeypatch.setattr(appmod, "_engine_status", _mock_engine_status({
        "running": True, "model": "qwen3-coder",
        "instances": [{"name": "vllm_node", "running": True}],
    }))
    monkeypatch.setattr(appmod, "_docker_label", _mock_docker_label(None))
    credit, pid = await appmod._running_profile_vram_credit("vllm", profiles)
    assert credit == 60.0
    assert pid == "start_big"

@pytest.mark.asyncio
async def test_credit_matches_served_name_in_script(tmp_path, monkeypatch):
    profiles = [
        _profile(tmp_path, "start_small", 20, "--served-model-name other-model\n"),
        _profile(tmp_path, "start_big", 60, "--served-model-name qwen3-coder\n"),
    ]
    monkeypatch.setattr(appmod, "_engine_status",
                        _mock_engine_status({"running": True, "model": "qwen3-coder"}))
    credit, pid = await appmod._running_profile_vram_credit("vllm", profiles)
    assert credit == 60.0
    assert pid == "start_big"


@pytest.mark.asyncio
async def test_credit_fuzzy_fallback_on_profile_id(tmp_path, monkeypatch):
    # Served name not verbatim in any script — falls back to token matching.
    profiles = [_profile(tmp_path, "start_qwen3_coder", 45)]
    monkeypatch.setattr(appmod, "_engine_status",
                        _mock_engine_status({"running": True, "model": "qwen3_coder-fp8"}))
    credit, pid = await appmod._running_profile_vram_credit("vllm", profiles)
    assert credit == 45.0
    assert pid == "start_qwen3_coder"


@pytest.mark.asyncio
async def test_credit_zero_when_engine_not_running(tmp_path, monkeypatch):
    profiles = [_profile(tmp_path, "start_big", 60)]
    monkeypatch.setattr(appmod, "_engine_status",
                        _mock_engine_status({"running": False}))
    assert await appmod._running_profile_vram_credit("vllm", profiles) == (0.0, "")


@pytest.mark.asyncio
async def test_credit_zero_when_running_profile_unidentifiable(tmp_path, monkeypatch):
    profiles = [_profile(tmp_path, "start_big", 60)]
    monkeypatch.setattr(appmod, "_engine_status",
                        _mock_engine_status({"running": True, "model": "mystery-model"}))
    assert await appmod._running_profile_vram_credit("vllm", profiles) == (0.0, "")


@pytest.mark.asyncio
async def test_credit_skips_profiles_without_vram_metadata(tmp_path, monkeypatch):
    profiles = [_profile(tmp_path, "start_big", None, "--served-model-name m1\n")]
    monkeypatch.setattr(appmod, "_engine_status",
                        _mock_engine_status({"running": True, "model": "m1"}))
    assert await appmod._running_profile_vram_credit("vllm", profiles) == (0.0, "")


@pytest.mark.asyncio
async def test_credit_unknown_engine_key():
    assert await appmod._running_profile_vram_credit("nope", []) == (0.0, "")


# ── _vram_admission_check ─────────────────────────────────────────────────────

def _patch_memory(monkeypatch, available_gb, credit=(0.0, ""), others=None):
    monkeypatch.setattr(appmod, "_get_available_memory_gib", lambda: available_gb)

    async def stub_credit(engine_key, profiles):
        return credit
    monkeypatch.setattr(appmod, "_running_profile_vram_credit", stub_credit)

    async def stub_others(exclude_key):
        return others or []
    monkeypatch.setattr(appmod, "_other_running_engines", stub_others)


@pytest.mark.asyncio
async def test_admission_allows_fitting_profile(monkeypatch):
    _patch_memory(monkeypatch, available_gb=50.0)
    profile = {"id": "start_fit", "vram_gb": 40}
    # 40 needed vs 50 available - 8 margin = 42 projected → admitted
    await appmod._vram_admission_check("vllm", profile, force=False, scan_fn=lambda: [])


@pytest.mark.asyncio
async def test_admission_rejects_overcommit_with_409(monkeypatch):
    _patch_memory(monkeypatch, available_gb=50.0)
    profile = {"id": "start_huge", "vram_gb": 60}
    with pytest.raises(HTTPException) as exc:
        await appmod._vram_admission_check("vllm", profile, force=False, scan_fn=lambda: [])
    assert exc.value.status_code == 409
    assert "force=true" in exc.value.detail


@pytest.mark.asyncio
async def test_admission_rejection_names_other_running_engines(monkeypatch):
    _patch_memory(monkeypatch, available_gb=50.0, others=["SGLang"])
    profile = {"id": "start_huge", "vram_gb": 60}
    with pytest.raises(HTTPException) as exc:
        await appmod._vram_admission_check("vllm", profile, force=False, scan_fn=lambda: [])
    assert "SGLang" in exc.value.detail
    assert "also running" in exc.value.detail


@pytest.mark.asyncio
async def test_admission_reclaim_credit_rescues_swap(monkeypatch):
    # Only 20 GB free, but the running 55 GB profile is torn down first.
    _patch_memory(monkeypatch, available_gb=20.0, credit=(55.0, "start_old"))
    profile = {"id": "start_new", "vram_gb": 60}
    await appmod._vram_admission_check("vllm", profile, force=False, scan_fn=lambda: [])


@pytest.mark.asyncio
async def test_admission_force_bypasses_check(monkeypatch):
    _patch_memory(monkeypatch, available_gb=1.0)
    profile = {"id": "start_huge", "vram_gb": 500}
    await appmod._vram_admission_check("vllm", profile, force=True, scan_fn=lambda: [])


@pytest.mark.asyncio
async def test_admission_unmetered_allowed_with_headroom(monkeypatch):
    # No vram_gb, but plenty of room: 100 free - 8 margin = 92 headroom >= floor.
    _patch_memory(monkeypatch, available_gb=100.0)
    profile = {"id": "start_unknown", "vram_gb": None}
    await appmod._vram_admission_check("vllm", profile, force=False, scan_fn=lambda: [])


@pytest.mark.asyncio
async def test_admission_unmetered_rejected_on_full_pool(monkeypatch):
    # No vram_gb and a nearly-full pool: 20 free - 8 margin = 12 headroom < floor.
    _patch_memory(monkeypatch, available_gb=20.0)
    profile = {"id": "start_unknown", "vram_gb": None}
    with pytest.raises(HTTPException) as exc:
        await appmod._vram_admission_check("vllm", profile, force=False, scan_fn=lambda: [])
    assert exc.value.status_code == 409
    assert "no VRAM footprint" in exc.value.detail


@pytest.mark.asyncio
async def test_admission_unmetered_force_bypasses(monkeypatch):
    _patch_memory(monkeypatch, available_gb=1.0)
    profile = {"id": "start_unknown", "vram_gb": None}
    await appmod._vram_admission_check("vllm", profile, force=True, scan_fn=lambda: [])


# ── Overrides resolve BEFORE admission (04-01), as util × MemTotal (05-01) ───

def _parameterized_profile(tmp_path, pid="start_hf_over", vram_gb=8, util="0.75"):
    """A generated-style script whose util is a `${VLLM_GPU_MEMORY_UTILIZATION:-N}`
    placeholder, with a stale `# VRAM: 8` header like the real generated scripts."""
    script = tmp_path / f"{pid}.sh"
    script.write_text("#!/bin/bash\n"
                      "# VRAM: 8\n"
                      "docker run -d --name vllm_node img \\\n"
                      f"  --gpu-memory-utilization ${{VLLM_GPU_MEMORY_UTILIZATION:-{util}}} \\\n"
                      "  --max-model-len ${VLLM_MAX_MODEL_LEN:-32768}\n")
    return {"id": pid, "name": "Over", "script": str(script),
            "description": "", "vram_gb": vram_gb}


def _capture_admission(monkeypatch):
    seen = {}

    async def _admit(engine_key, prof, force, scan_fn=None):
        seen.update(vram_gb=prof.get("vram_gb"), source=prof.get("footprint_source"))

    monkeypatch.setattr(appmod, "_vram_admission_check", _admit)
    monkeypatch.setattr(appmod.shutil, "which", lambda name: None)
    monkeypatch.setattr(appmod.subprocess, "Popen", lambda *a, **kw: None)
    monkeypatch.setattr(appmod, "_mem_total_gib", lambda: 121.6)
    return seen


def test_admission_sees_script_util_not_the_stale_header(tmp_path, monkeypatch):
    """The header says 8; the script launches at 0.75 × 121.6 = 91.2 GiB. Admitting
    on the header is exactly the overcommit Phase 5 exists to stop."""
    import asyncio
    profile = _parameterized_profile(tmp_path)
    seen = _capture_admission(monkeypatch)
    asyncio.run(appmod._engine_start(
        "start_hf_over", lambda: [profile], "vLLM", engine_key="vllm"))
    assert seen == {"vram_gb": 91.2, "source": "script-util"}


def test_util_override_sets_the_admitted_footprint(tmp_path, monkeypatch):
    import asyncio
    profile = _parameterized_profile(tmp_path)
    seen = _capture_admission(monkeypatch)
    asyncio.run(appmod._engine_start(
        "start_hf_over", lambda: [profile], "vLLM", engine_key="vllm",
        overrides={"gpu_memory_utilization": 0.5}))
    assert seen == {"vram_gb": 60.8, "source": "override-util"}


def test_context_only_override_does_not_change_the_footprint(tmp_path, monkeypatch):
    """vLLM reserves util × total whatever --max-model-len is."""
    import asyncio
    profile = _parameterized_profile(tmp_path)
    seen = _capture_admission(monkeypatch)
    asyncio.run(appmod._engine_start(
        "start_hf_over", lambda: [profile], "vLLM", engine_key="vllm",
        overrides={"max_model_len": 131072}))
    assert seen == {"vram_gb": 91.2, "source": "script-util"}


def test_util_override_ignored_when_the_script_does_not_read_it(tmp_path, monkeypatch):
    """A legacy script with a literal util launches at ITS util; the env var is inert,
    so the footprint must not shrink to the override."""
    import asyncio
    script = tmp_path / "start_legacy.sh"
    script.write_text("#!/bin/bash\n# VRAM: 8\nvllm serve m --gpu-memory-utilization 0.75\n")
    profile = {"id": "start_legacy", "name": "L", "script": str(script), "vram_gb": 8}
    seen = _capture_admission(monkeypatch)
    asyncio.run(appmod._engine_start(
        "start_legacy", lambda: [profile], "vLLM", engine_key="vllm",
        overrides={"gpu_memory_utilization": 0.2}))
    assert seen == {"vram_gb": 91.2, "source": "script-util"}


def test_context_override_on_header_only_script_admits_on_header(tmp_path, monkeypatch):
    """No util anywhere: the header is the honest fallback, and a context override no
    longer needs force (it cannot change the reservation)."""
    import asyncio
    profile = _profile(tmp_path, "start_legacy", 8, "#!/bin/bash\nexit 0\n")
    seen = _capture_admission(monkeypatch)
    res = asyncio.run(appmod._engine_start(
        "start_legacy", lambda: [profile], "vLLM", engine_key="vllm",
        overrides={"max_model_len": 131072}))
    assert res["ok"] is True
    assert seen == {"vram_gb": 8.0, "source": "header"}


def test_admission_runs_once_and_before_popen_for_an_overridden_launch(tmp_path, monkeypatch):
    import asyncio
    profile = _parameterized_profile(tmp_path)
    order = []

    async def _admit(*a, **kw):
        order.append("admit")

    def _popen(*a, **kw):
        order.append("popen")

    monkeypatch.setattr(appmod, "_vram_admission_check", _admit)
    monkeypatch.setattr(appmod.subprocess, "Popen", _popen)
    monkeypatch.setattr(appmod.shutil, "which", lambda name: None)

    asyncio.run(appmod._engine_start(
        "start_hf_over", lambda: [profile], "vLLM", engine_key="vllm",
        overrides={"max_model_len": 65536}))

    assert order == ["admit", "popen"]


def test_footprint_resolution_precedes_admission_in_source():
    """Source-level guard: the ordering is a comment away from silently regressing."""
    import inspect
    src = inspect.getsource(appmod._engine_start)
    assert src.index("_resolve_overrides(") < src.index("await _vram_admission_check")
    assert src.index("_profile_footprint_gib(") < src.index("await _vram_admission_check")


# ── _identify_active_profile ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_identify_prefers_dgx_profile_label_over_name_match(tmp_path, monkeypatch):
    profiles = [
        _profile(tmp_path, "start_wrong", 20, "--served-model-name vllm-active\n"),
        _profile(tmp_path, "start_right", 60, "--served-model-name vllm-active\n"),
    ]
    status = {
        "model": "vllm-active",
        "instances": [{"name": "vllm_node", "running": True}],
    }
    monkeypatch.setattr(appmod, "_docker_label", _mock_docker_label("right"))
    assert await appmod._identify_active_profile(status, profiles) == "start_right"


@pytest.mark.asyncio
async def test_identify_falls_back_to_name_match_when_label_absent(tmp_path, monkeypatch):
    profiles = [_profile(tmp_path, "start_big", 60, "--served-model-name qwen3-coder\n")]
    status = {
        "model": "qwen3-coder",
        "instances": [{"name": "vllm_node", "running": True}],
    }
    monkeypatch.setattr(appmod, "_docker_label", _mock_docker_label(None))
    assert await appmod._identify_active_profile(status, profiles) == "start_big"


@pytest.mark.asyncio
async def test_identify_returns_none_without_served_model_or_instances():
    assert await appmod._identify_active_profile({}, []) is None


# ── _docker_label ────────────────────────────────────────────────────────────

def _mock_run(stdout, returncode=0):
    async def stub(*cmd, timeout=30):
        import subprocess
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    return stub


@pytest.mark.asyncio
async def test_docker_label_returns_value(monkeypatch):
    monkeypatch.setattr(appmod, "_run", _mock_run("hf_qwen_qwen3.8-27b-fp8-mtp\n"))
    assert await appmod._docker_label("vllm_node", "dgx.profile") == "hf_qwen_qwen3.8-27b-fp8-mtp"


@pytest.mark.asyncio
async def test_docker_label_none_on_unset_sentinel(monkeypatch):
    # Docker's Go template prints the literal "<no value>" for a missing key.
    monkeypatch.setattr(appmod, "_run", _mock_run("<no value>\n"))
    assert await appmod._docker_label("vllm_node", "dgx.profile") is None


@pytest.mark.asyncio
async def test_docker_label_none_on_inspect_failure(monkeypatch):
    monkeypatch.setattr(appmod, "_run", _mock_run("", returncode=1))
    assert await appmod._docker_label("missing_container", "dgx.profile") is None
