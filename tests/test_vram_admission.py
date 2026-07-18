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


# ── _running_profile_vram_credit ──────────────────────────────────────────────

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

def _patch_memory(monkeypatch, available_gb, credit=(0.0, "")):
    monkeypatch.setattr(appmod, "_get_available_memory_gb", lambda: available_gb)

    async def stub_credit(engine_key, profiles):
        return credit
    monkeypatch.setattr(appmod, "_running_profile_vram_credit", stub_credit)


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
async def test_admission_skipped_without_vram_metadata(monkeypatch):
    _patch_memory(monkeypatch, available_gb=1.0)
    profile = {"id": "start_unknown", "vram_gb": None}
    await appmod._vram_admission_check("vllm", profile, force=False, scan_fn=lambda: [])
