"""Phase 05-01: admission compares GiB to GiB, and a vLLM footprint is util × MemTotal."""

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

import app as appmod

REPO = Path(__file__).resolve().parents[1]
TOTAL = 121.6  # gb10 MemTotal, GiB


def _script(tmp_path, body, name="start_x.sh", header=None):
    p = tmp_path / name
    head = f"# VRAM: {header}\n" if header is not None else ""
    p.write_text("#!/bin/bash\n" + head + body)
    return p


def _prof(path, vram=None):
    return {"id": path.stem, "name": path.stem, "script": str(path), "vram_gb": vram}


# ── unit helpers ──────────────────────────────────────────────────────────────

def _fake_meminfo(monkeypatch, text):
    import io

    def fake_open(path, *a, **kw):
        assert path == "/proc/meminfo"
        return io.StringIO(text)
    monkeypatch.setattr(appmod, "open", fake_open, raising=False)


def test_meminfo_is_read_as_binary_gib(monkeypatch):
    # 127535252 kB is gb10's MemTotal: 121.63 GiB, i.e. 130.6 decimal GB. The two
    # differ by ~7%, which is exactly the slack a mixed-unit admission hid.
    _fake_meminfo(monkeypatch, "MemTotal:       127535252 kB\n"
                               "MemFree:         1375420 kB\n"
                               "MemAvailable:   16702564 kB\n")
    assert appmod._mem_total_gib() == pytest.approx(127535252 / 2**20)
    assert appmod._get_available_memory_gib() == pytest.approx(16702564 / 2**20)
    assert appmod._mem_total_gib() == pytest.approx(121.63, abs=0.01)


def test_mem_total_falls_back_to_gb10_pool_when_unreadable(monkeypatch):
    def boom(*a, **kw):
        raise OSError("no procfs")
    monkeypatch.setattr(appmod, "open", boom, raising=False)
    assert appmod._mem_total_gib() == appmod._POOL_GIB_FALLBACK
    assert appmod._get_available_memory_gib() == 0.0


def test_gib_constant():
    assert appmod._GIB == 2 ** 30


# ── _profile_footprint_gib ────────────────────────────────────────────────────

def test_literal_util_beats_header(tmp_path):
    p = _script(tmp_path, "vllm serve m --gpu-memory-utilization 0.75\n", header=20)
    assert appmod._profile_footprint_gib(_prof(p, 20), mem_total_gib=TOTAL) == (91.2, "script-util")


def test_placeholder_default_util(tmp_path):
    p = _script(tmp_path, 'GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.76}" ./serve.sh\n', header=92)
    assert appmod._profile_footprint_gib(_prof(p, 92), mem_total_gib=TOTAL) == (92.4, "script-util")


def test_commented_util_is_ignored(tmp_path):
    p = _script(tmp_path, "# --gpu-memory-utilization 0.9 was too much\nexit 0\n", header=30)
    assert appmod._profile_footprint_gib(_prof(p, 30), mem_total_gib=TOTAL) == (30.0, "header")


@pytest.mark.parametrize("bad", ["1.5", "0", "$UTIL"])
def test_out_of_range_or_symbolic_util_falls_back_to_header(tmp_path, bad):
    p = _script(tmp_path, f"vllm serve m --gpu-memory-utilization {bad}\n", header=30)
    assert appmod._profile_footprint_gib(_prof(p, 30), mem_total_gib=TOTAL) == (30.0, "header")


def test_recipe_backed_uses_recipe_util(tmp_path, monkeypatch):
    p = _script(tmp_path, 'cd "$RECIPE_DIR/.."\nexec ./run-recipe.sh qwen-solo -d\n', header=10)
    monkeypatch.setattr(appmod, "_recipe_util", lambda name: 0.55 if name == "qwen-solo" else None)
    assert appmod._profile_footprint_gib(_prof(p, 10), mem_total_gib=TOTAL) == (66.9, "recipe-util")


def test_recipe_backed_without_readable_recipe_uses_header(tmp_path, monkeypatch):
    p = _script(tmp_path, "exec ./run-recipe.sh qwen-solo -d\n", header=67)
    monkeypatch.setattr(appmod, "_recipe_util", lambda name: None)
    assert appmod._profile_footprint_gib(_prof(p, 67), mem_total_gib=TOTAL) == (67.0, "header")


def test_unknown_when_nothing_declared(tmp_path):
    p = _script(tmp_path, "exit 0\n")
    assert appmod._profile_footprint_gib(_prof(p), mem_total_gib=TOTAL) == (None, "unknown")


def test_missing_script_falls_back_to_header():
    prof = {"id": "start_gone", "script": "/nonexistent/start_gone.sh", "vram_gb": 12}
    assert appmod._profile_footprint_gib(prof, mem_total_gib=TOTAL) == (12.0, "header")


def test_footprint_uses_live_mem_total_by_default(tmp_path, monkeypatch):
    p = _script(tmp_path, "vllm serve m --gpu-memory-utilization 0.5\n")
    monkeypatch.setattr(appmod, "_mem_total_gib", lambda: 100.0)
    assert appmod._profile_footprint_gib(_prof(p)) == (50.0, "script-util")


# ── every committed profile ───────────────────────────────────────────────────

def _committed_profiles():
    return sorted((REPO / "profiles").glob("*/start_*.sh"))


@pytest.mark.parametrize("path", _committed_profiles(), ids=lambda p: p.name)
def test_committed_profile_header_matches_its_reservation(path, monkeypatch):
    """The tracked `# VRAM:` headers are the profile card's number. With admission now
    reading util × MemTotal, a header that drifts from it only misleads the operator —
    and on 2026-10-18 several generated ones under-declared by ~70 GiB."""
    monkeypatch.setattr(appmod, "_recipe_util", lambda name: None)
    meta = appmod._parse_script_meta(path)
    gib, source = appmod._profile_footprint_gib(meta, mem_total_gib=TOTAL)
    assert gib is not None, f"{path.name} has no footprint at all"
    assert meta["vram_gb"] is not None, f"{path.name} has no # VRAM: header"
    if source == "script-util":
        assert abs(meta["vram_gb"] - gib) <= appmod._HEADER_DRIFT_WARN_GIB, (
            f"{path.name}: header {meta['vram_gb']} vs util × MemTotal {gib}")
    else:
        # Only recipe-backed vLLM and llama.cpp profiles may rely on the header.
        assert source == "header"
        text = path.read_text()
        assert "run-recipe.sh" in text or path.parent.name == "llama.cpp", path.name


# ── admission, end to end ─────────────────────────────────────────────────────

def _patch_pool(monkeypatch, available, credit=(0.0, "")):
    monkeypatch.setattr(appmod, "_get_available_memory_gib", lambda: available)
    monkeypatch.setattr(appmod, "_mem_total_gib", lambda: TOTAL)

    async def stub_credit(engine_key, profiles):
        return credit
    monkeypatch.setattr(appmod, "_running_profile_vram_credit", stub_credit)

    async def stub_others(exclude_key):
        return []
    monkeypatch.setattr(appmod, "_other_running_engines", stub_others)
    monkeypatch.setattr(appmod.shutil, "which", lambda name: None)
    launched = []
    monkeypatch.setattr(appmod.subprocess, "Popen", lambda *a, **kw: launched.append(a))
    return launched


def test_small_header_big_util_profile_is_refused(tmp_path, monkeypatch):
    """The regression Phase 5 exists for: qwen3-8b declared 20 GiB but launches at
    util 0.75. With 40 GiB free the old check admitted it (20 ≤ 40 − 8); vLLM would
    then try to reserve 91 GiB."""
    p = _script(tmp_path, "docker run -d --name vllm_node img \\\n"
                          "  --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.75}\n",
                name="start_hf_qwen_qwen3-8b.sh", header=20)
    prof = appmod._parse_script_meta(p)
    launched = _patch_pool(monkeypatch, available=40.0)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(appmod._engine_start(prof["id"], lambda: [prof], "vLLM",
                                         engine_key="vllm"))
    assert exc.value.status_code == 409
    assert "91.2 GiB" in exc.value.detail
    assert "[script-util]" in exc.value.detail
    assert launched == []


def test_lower_util_override_rescues_the_same_launch(tmp_path, monkeypatch):
    p = _script(tmp_path, "docker run -d --name vllm_node img \\\n"
                          "  --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.75}\n",
                name="start_hf_small.sh", header=91)
    prof = appmod._parse_script_meta(p)
    launched = _patch_pool(monkeypatch, available=40.0)
    res = asyncio.run(appmod._engine_start(
        prof["id"], lambda: [prof], "vLLM", engine_key="vllm",
        overrides={"gpu_memory_utilization": 0.25}))  # 30.4 GiB ≤ 40 − 8
    assert res["ok"] is True
    assert len(launched) == 1


def test_admission_messages_speak_gib(monkeypatch):
    _patch_pool(monkeypatch, available=50.0)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(appmod._vram_admission_check(
            "vllm", {"id": "start_huge", "vram_gb": 60.0, "footprint_source": "header"},
            force=False, scan_fn=lambda: []))
    assert "GiB" in exc.value.detail and " GB " not in exc.value.detail


def test_llamacpp_recipe_need_is_kept(tmp_path, monkeypatch):
    p = _script(tmp_path, 'RECIPE="${RECIPE:-balanced}"\n', name="start_gguf_x.sh", header=31)
    prof = appmod._parse_script_meta(p)
    seen = {}

    async def admit(engine_key, profile, force, scan_fn=None):
        seen.update(vram=profile["vram_gb"], source=profile["footprint_source"])
    monkeypatch.setattr(appmod, "_vram_admission_check", admit)
    monkeypatch.setattr(appmod, "_llamacpp_recipes", lambda: {"longctx": {"vram_gb": 60}})
    monkeypatch.setattr(appmod.shutil, "which", lambda name: None)
    monkeypatch.setattr(appmod.subprocess, "Popen", lambda *a, **kw: None)
    asyncio.run(appmod._engine_start(prof["id"], lambda: [prof], "llamacpp",
                                     engine_key="llamacpp", recipe="longctx"))
    assert seen == {"vram": 60, "source": "llamacpp-recipe"}


# ── reclaim credit uses the same quantity ─────────────────────────────────────

@pytest.mark.asyncio
async def test_credit_is_util_times_total_not_the_header(tmp_path, monkeypatch):
    p = _script(tmp_path, "vllm serve m --gpu-memory-utilization 0.76\n",
                name="start_flash.sh", header=20)
    prof = appmod._parse_script_meta(p)
    monkeypatch.setattr(appmod, "_mem_total_gib", lambda: TOTAL)

    async def status(*a, **kw):
        return {"running": True, "model": "vllm-active",
                "instances": [{"name": "vllm_node", "running": True}]}

    async def label(container, key):
        return "flash"
    monkeypatch.setattr(appmod, "_engine_status", status)
    monkeypatch.setattr(appmod, "_docker_label", label)
    assert await appmod._running_profile_vram_credit("vllm", [prof]) == (92.4, "start_flash")


# ── header parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("91", 91), ("91.2", 91.2), ("70GB", 70), ("70 GiB", 70), ("nan", None),
    ("inf", None), ("-5", None), ("1e3", None),
])
def test_vram_header_parse(tmp_path, raw, expected):
    p = _script(tmp_path, "exit 0\n", header=raw)
    assert appmod._parse_script_meta(p)["vram_gb"] == expected


# ── generator ─────────────────────────────────────────────────────────────────

def test_generated_header_is_util_times_total(monkeypatch):
    monkeypatch.setattr(appmod, "_mem_total_gib", lambda: TOTAL)
    info = {"vram_gb": 20}
    docker = {"shape": "docker", "util": 0.75, "max_model_len": 32768}
    assert appmod._generated_vram_header(info, docker) == 91
    recipe = {"shape": "recipe", "record": {"gpu_memory_utilization": 0.55}}
    assert appmod._generated_vram_header(info, recipe) == 67
    # No util knowable: the weights estimate is the honest last resort.
    assert appmod._generated_vram_header(info, {"shape": "recipe", "record": {}}) == 20
    assert appmod._generated_vram_header(info, None) == 20


def test_generated_header_uses_fallback_util_when_underived(monkeypatch):
    monkeypatch.setattr(appmod, "_mem_total_gib", lambda: TOTAL)
    docker = {"shape": "docker", "util": None, "max_model_len": None}
    assert appmod._generated_vram_header({"vram_gb": 20}, docker) == round(
        appmod._FALLBACK_UTIL * TOTAL)


# ── preflight drift check ─────────────────────────────────────────────────────

def _levels(checks):
    return [(c["level"], c["check"]) for c in checks]


def test_preflight_warns_on_header_drift(tmp_path):
    p = _script(tmp_path, "vllm serve m --gpu-memory-utilization 0.75\n", header=20)
    out = appmod._preflight_footprint(appmod._parse_script_meta(p), mem_total_gib=TOTAL)
    assert _levels(out) == [("warn", "footprint")]
    assert "# VRAM: 91" in out[0]["fix"]


def test_preflight_ok_within_drift(tmp_path):
    p = _script(tmp_path, "vllm serve m --gpu-memory-utilization 0.75\n", header=91)
    out = appmod._preflight_footprint(appmod._parse_script_meta(p), mem_total_gib=TOTAL)
    assert _levels(out) == [("ok", "footprint")]


def test_preflight_warns_on_missing_header(tmp_path):
    p = _script(tmp_path, "vllm serve m --gpu-memory-utilization 0.5\n")
    out = appmod._preflight_footprint(appmod._parse_script_meta(p), mem_total_gib=TOTAL)
    assert _levels(out) == [("warn", "footprint")]
    assert "# VRAM: 61" in out[0]["fix"]


def test_preflight_warns_when_nothing_declared(tmp_path):
    p = _script(tmp_path, "exit 0\n")
    out = appmod._preflight_footprint(appmod._parse_script_meta(p), mem_total_gib=TOTAL)
    assert _levels(out) == [("warn", "footprint")]


def test_preflight_header_only_is_ok(tmp_path):
    p = _script(tmp_path, "exit 0\n", header=31)
    out = appmod._preflight_footprint(appmod._parse_script_meta(p), mem_total_gib=TOTAL)
    assert _levels(out) == [("ok", "footprint")]
