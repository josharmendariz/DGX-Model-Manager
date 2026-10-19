"""Phase 05-02: launch lock, in-flight reservations, failure surfacing, preflight gate."""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from fastapi import HTTPException

import app as appmod
from conftest import _REAL_REQUIRE_LAUNCH_PREFLIGHT

REPO = Path(__file__).resolve().parents[1]
TOTAL = 121.6


def _util_script(tmp_path, pid, util, header=None):
    p = tmp_path / f"{pid}.sh"
    head = f"# VRAM: {header}\n" if header is not None else ""
    p.write_text("#!/bin/bash\n" + head +
                 f"docker run -d --name vllm_node img --gpu-memory-utilization {util}\n")
    return {"id": pid, "name": pid, "script": str(p), "description": "", "vram_gb": header}


def _header_script(tmp_path, pid, header):
    p = tmp_path / f"{pid}.sh"
    p.write_text(f"#!/bin/bash\n# VRAM: {header}\nexit 0\n")
    return {"id": pid, "name": pid, "script": str(p), "description": "", "vram_gb": header}


class _Pool:
    """A fake unified pool: admission reads it; the credit/other-engine probes are
    stubbed; every docker/health probe says "still loading" unless told otherwise."""

    def __init__(self, monkeypatch, available, admission_delay=0.01):
        self.available = available
        self.popens = []
        self.admissions = []          # (engine, enter, exit) in loop time
        self.container = {}           # container -> state dict
        self.healthy = set()          # engine bases answering /health

        monkeypatch.setattr(appmod, "_get_available_memory_gib", lambda: self.available)
        monkeypatch.setattr(appmod, "_mem_total_gib", lambda: TOTAL)

        async def credit(engine_key, profiles):
            # Yield to the loop mid-admission: without the lock, a second start would
            # interleave right here and read the same MemAvailable.
            await asyncio.sleep(admission_delay)
            return 0.0, ""
        monkeypatch.setattr(appmod, "_running_profile_vram_credit", credit)

        async def others(exclude_key):
            return []
        monkeypatch.setattr(appmod, "_other_running_engines", others)

        real_admit = appmod._vram_admission_check

        async def admit(engine_key, profile, force, scan_fn=None):
            loop = asyncio.get_running_loop()
            enter = loop.time()
            try:
                await real_admit(engine_key, profile, force, scan_fn)
            finally:
                self.admissions.append((engine_key, enter, loop.time()))
        monkeypatch.setattr(appmod, "_vram_admission_check", admit)

        async def prior(container):
            # Real code awaits `docker inspect` here, between admission and Popen —
            # the window in which an unlocked second start would read a stale pool.
            await asyncio.sleep(admission_delay)
            return ""
        monkeypatch.setattr(appmod, "_prior_container_id", prior)

        def popen(argv, **kw):
            self.popens.append(argv)
            return _FakeProc(None)
        monkeypatch.setattr(appmod.subprocess, "Popen", popen)
        monkeypatch.setattr(appmod.shutil, "which", lambda name: None)

        async def state(name):
            # docker inspect accepts a name or an id; the progress stream uses both.
            hit = self.container.get(name) or next(
                (s for s in self.container.values() if s.get("id") == name), None)
            return hit or {"exists": False, "status": "absent", "exit_code": None,
                           "id": "", "oom_killed": None}
        monkeypatch.setattr(appmod, "_container_state", state)

        async def ok(base, path="/health"):
            return base in self.healthy
        monkeypatch.setattr(appmod, "service_ok", ok)


class _FakeProc:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc


def _start(profile, key, force=False, recipe=None):
    return appmod._engine_start(profile["id"], lambda: [profile], key,
                                engine_key=key, force=force, recipe=recipe)


# ── the race ──────────────────────────────────────────────────────────────────

def test_two_concurrent_starts_on_one_engine_cannot_both_launch(tmp_path, monkeypatch):
    """Both would fit alone (60 ≤ 100 − 8). Without the lock both read the same
    MemAvailable and both Popen; two scripts then race on `docker rm -f vllm_node`."""
    pool = _Pool(monkeypatch, available=100.0)
    a = _util_script(tmp_path, "start_a", 0.5)   # 60.8 GiB
    b = _util_script(tmp_path, "start_b", 0.5)

    async def both():
        return await asyncio.gather(_start(a, "vllm"), _start(b, "vllm"),
                                    return_exceptions=True)
    results = asyncio.run(both())

    oks = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if isinstance(r, HTTPException)]
    assert len(oks) == 1 and len(errs) == 1
    assert errs[0].status_code == 409
    assert "already in progress" in errs[0].detail
    assert len(pool.popens) == 1
    assert appmod._INFLIGHT["vllm"]["profile"] in ("start_a", "start_b")


def test_two_concurrent_starts_on_different_engines_share_one_pool(tmp_path, monkeypatch):
    """vLLM (60.8) and llama.cpp (31) each fit 80 − 8 alone; together they do not.
    The second admission must subtract the first's reservation."""
    pool = _Pool(monkeypatch, available=80.0)
    v = _util_script(tmp_path, "start_v", 0.5)
    llama = _header_script(tmp_path, "start_gguf_x", 31)

    async def both():
        return await asyncio.gather(_start(v, "vllm"), _start(llama, "llamacpp"),
                                    return_exceptions=True)
    results = asyncio.run(both())

    oks = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if isinstance(r, HTTPException)]
    assert len(oks) == 1 and len(errs) == 1, results
    assert errs[0].status_code == 409
    assert "reserved for launches still loading" in errs[0].detail
    assert len(pool.popens) == 1


def test_admissions_are_serialized_by_the_lock(tmp_path, monkeypatch):
    """Plenty of memory, two engines: both launch, but never inside each other's
    admission window."""
    pool = _Pool(monkeypatch, available=500.0, admission_delay=0.05)
    v = _util_script(tmp_path, "start_v", 0.5)
    llama = _header_script(tmp_path, "start_gguf_x", 31)

    async def both():
        return await asyncio.gather(_start(v, "vllm"), _start(llama, "llamacpp"))
    asyncio.run(both())

    assert len(pool.popens) == 2
    (_, a0, a1), (_, b0, b1) = sorted(pool.admissions, key=lambda r: r[1])
    assert a1 <= b0, f"admissions overlapped: {pool.admissions}"


def test_sequential_start_on_other_engine_sees_the_reservation(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=80.0)
    asyncio.run(_start(_util_script(tmp_path, "start_v", 0.5), "vllm"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_start(_header_script(tmp_path, "start_gguf_x", 31), "llamacpp"))
    assert exc.value.status_code == 409
    assert len(pool.popens) == 1


def test_same_engine_restart_after_settle_is_allowed(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=100.0)
    a = _util_script(tmp_path, "start_a", 0.5)
    asyncio.run(_start(a, "vllm"))
    # The new container comes up healthy: the launch settles as ready.
    pool.container["vllm_node"] = {"exists": True, "id": "new", "status": "running",
                                   "exit_code": 0, "oom_killed": False}
    pool.healthy.add(appmod._engine_bases["vllm"])
    asyncio.run(_start(_util_script(tmp_path, "start_b", 0.5), "vllm"))
    assert len(pool.popens) == 2
    assert appmod._LAST_LAUNCH["vllm"]["result"] == "ready"
    assert appmod._INFLIGHT["vllm"]["profile"] == "start_b"


def test_force_overrides_an_in_flight_launch(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=100.0)
    asyncio.run(_start(_util_script(tmp_path, "start_a", 0.5), "vllm"))
    asyncio.run(_start(_util_script(tmp_path, "start_b", 0.5), "vllm", force=True))
    assert len(pool.popens) == 2
    assert appmod._LAST_LAUNCH["vllm"]["result"] == "superseded"
    assert appmod._INFLIGHT["vllm"]["profile"] == "start_b"


def test_refused_launch_leaves_no_reservation(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=20.0)
    with pytest.raises(HTTPException):
        asyncio.run(_start(_util_script(tmp_path, "start_big", 0.75), "vllm"))
    assert appmod._INFLIGHT == {}
    assert pool.popens == []


def test_unmetered_launch_reserves_the_headroom_floor(tmp_path, monkeypatch):
    _Pool(monkeypatch, available=100.0)
    p = tmp_path / "start_unknown.sh"
    p.write_text("#!/bin/bash\nexit 0\n")
    prof = {"id": "start_unknown", "name": "u", "script": str(p), "vram_gb": None}
    asyncio.run(_start(prof, "vllm"))
    assert appmod._INFLIGHT["vllm"]["gib"] == appmod._VRAM_UNKNOWN_MIN_HEADROOM_GIB


# ── settling ──────────────────────────────────────────────────────────────────

def _pending(monkeypatch, tmp_path, available=100.0):
    pool = _Pool(monkeypatch, available=available)
    asyncio.run(_start(_util_script(tmp_path, "start_a", 0.5), "vllm"))
    return pool


def test_pending_launch_stays_reserved_while_loading(tmp_path, monkeypatch):
    _pending(monkeypatch, tmp_path)
    asyncio.run(appmod._prune_inflight())
    assert "vllm" in appmod._INFLIGHT


def test_old_container_dying_is_not_the_new_launch_failing(tmp_path, monkeypatch):
    """The outgoing container (prior id) exits when the script `rm -f`s it; that must
    not be read as the new launch's verdict."""
    pool = _pending(monkeypatch, tmp_path)
    appmod._INFLIGHT["vllm"]["prior_id"] = "old"
    pool.container["vllm_node"] = {"exists": True, "id": "old", "status": "exited",
                                   "exit_code": 137, "oom_killed": False}
    asyncio.run(appmod._prune_inflight())
    assert "vllm" in appmod._INFLIGHT


def test_oom_killed_container_releases_with_cause(tmp_path, monkeypatch):
    pool = _pending(monkeypatch, tmp_path)
    pool.container["vllm_node"] = {"exists": True, "id": "new", "status": "exited",
                                   "exit_code": 137, "oom_killed": True}
    asyncio.run(appmod._prune_inflight())
    assert "vllm" not in appmod._INFLIGHT
    last = appmod._LAST_LAUNCH["vllm"]
    assert last["result"] == "failed" and last["exit_code"] == 137
    assert last["oom_killed"] is True and "OOM killer" in last["cause"]
    assert last["profile"] == "start_a"


def test_exit_137_without_oom_is_explained_as_sigkill(tmp_path, monkeypatch):
    pool = _pending(monkeypatch, tmp_path)
    pool.container["vllm_node"] = {"exists": True, "id": "new", "status": "exited",
                                   "exit_code": 137, "oom_killed": False}
    asyncio.run(appmod._prune_inflight())
    assert "SIGKILL" in appmod._LAST_LAUNCH["vllm"]["cause"]


def test_ttl_releases_a_launch_that_never_settles(tmp_path, monkeypatch):
    _pending(monkeypatch, tmp_path)
    appmod._INFLIGHT["vllm"]["started"] -= appmod._INFLIGHT_TTL_S + 1
    asyncio.run(appmod._prune_inflight())
    assert "vllm" not in appmod._INFLIGHT
    assert appmod._LAST_LAUNCH["vllm"]["result"] == "timeout"


def test_launch_script_failing_before_any_container_releases(tmp_path, monkeypatch):
    _pending(monkeypatch, tmp_path)
    appmod._INFLIGHT["vllm"]["proc"] = _FakeProc(2)
    asyncio.run(appmod._prune_inflight())
    last = appmod._LAST_LAUNCH["vllm"]
    assert last["result"] == "failed" and "exited 2" in last["cause"]


def test_status_route_reports_in_flight_then_last_verdict(tmp_path, monkeypatch):
    pool = _pending(monkeypatch, tmp_path)

    async def engine_status(*a, **kw):
        return {"running": False, "model": None, "state": "loading", "instances": []}
    monkeypatch.setattr(appmod, "_engine_status", engine_status)
    route = next(r for r in appmod.app.routes if getattr(r, "name", "") == "vllm_status")

    out = asyncio.run(route.endpoint())
    assert out["launch"]["in_flight"]["profile"] == "start_a"
    assert out["launch"]["in_flight"]["gib"] == 60.8
    assert out["launch"]["last"] is None
    json.dumps(out)  # no Popen object or other non-JSON leaks into the response

    pool.container["vllm_node"] = {"exists": True, "id": "new", "status": "exited",
                                   "exit_code": 1, "oom_killed": False}
    out = asyncio.run(route.endpoint())
    assert out["launch"]["in_flight"] is None
    assert out["launch"]["last"]["result"] == "failed"
    assert out["launch"]["last"]["exit_code"] == 1


# ── container state + causes ──────────────────────────────────────────────────

def _run_stub(stdout, rc=0):
    async def stub(*cmd, timeout=30):
        return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
    return stub


@pytest.mark.parametrize("stdout,expected", [
    ("abc|exited|137|true\n", {"id": "abc", "status": "exited", "exit_code": 137,
                               "oom_killed": True}),
    ("abc|running|0|false\n", {"id": "abc", "status": "running", "exit_code": 0,
                               "oom_killed": False}),
    ("abc|exited|1|<no value>\n", {"id": "abc", "status": "exited", "exit_code": 1,
                                   "oom_killed": None}),
    ("abc|exited\n", {"id": "abc", "status": "exited", "exit_code": None,
                      "oom_killed": None}),
])
def test_container_state_reads_oomkilled(monkeypatch, stdout, expected):
    monkeypatch.setattr(appmod, "_run", _run_stub(stdout))
    state = asyncio.run(appmod._container_state("vllm_node"))
    assert state == {"exists": True, **expected}


def test_container_state_absent(monkeypatch):
    monkeypatch.setattr(appmod, "_run", _run_stub("", rc=1))
    state = asyncio.run(appmod._container_state("vllm_node"))
    assert state["exists"] is False and state["oom_killed"] is None


@pytest.mark.parametrize("code,oom,needle", [
    (137, True, "OOM killer"), (None, True, "OOM killer"), (137, False, "SIGKILL"),
    (139, False, "segmentation"), (143, None, "SIGTERM"), (1, False, "code 1"),
    (0, False, "cleanly"), (None, None, "stopped"),
])
def test_exit_cause(code, oom, needle):
    assert needle in appmod._exit_cause(code, oom)


# ── progress stream surfaces OOM / 137 ───────────────────────────────────────

class _EOFLogs:
    returncode = 0

    class stdout:
        @staticmethod
        async def readline():
            return b""

    def kill(self):
        pass

    async def wait(self):
        return 0


def test_progress_stream_reports_oom_and_releases_the_reservation(tmp_path, monkeypatch):
    pool = _pending(monkeypatch, tmp_path)
    pool.container["vllm_node"] = {"exists": True, "id": "new", "status": "exited",
                                   "exit_code": 137, "oom_killed": True}

    async def not_healthy():
        return False
    monkeypatch.setattr(appmod, "_vllm_health_ok", not_healthy)

    async def fake_exec(*a, **kw):
        return _EOFLogs()
    monkeypatch.setattr(appmod.asyncio, "create_subprocess_exec", fake_exec)

    async def collect():
        return [f async for f in appmod._load_progress_events("vllm_node")]
    frames = [json.loads(f[len("data: "):]) for f in asyncio.run(collect())]
    terminal = [f for f in frames if f["status"] in ("failed", "ready")]
    assert len(terminal) == 1
    t = terminal[0]
    assert t["status"] == "failed" and t["exit_code"] == 137
    assert t["oom_killed"] is True and "OOM killer" in t["cause"]
    assert "vllm" not in appmod._INFLIGHT
    assert appmod._LAST_LAUNCH["vllm"]["oom_killed"] is True


# ── mandatory preflight gate (absorbs b6d1f00) ───────────────────────────────

def _preflight_result(verdict, checks=()):
    async def fake(profile):
        return {"verdict": verdict, "checks": list(checks)}
    return fake


def test_gate_blocks_a_failing_profile_before_admission_and_popen(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=100.0)
    monkeypatch.setattr(appmod, "_require_launch_preflight", _REAL_REQUIRE_LAUNCH_PREFLIGHT)
    monkeypatch.setattr(appmod, "_preflight_profile", _preflight_result("fail", [
        {"level": "fail", "title": "Model mount missing",
         "detail": "/mnt/models/x does not exist", "fix": "Regenerate the profile"}]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_start(_util_script(tmp_path, "start_broken", 0.5), "vllm"))
    assert exc.value.status_code == 409
    assert "preflight blocked" in exc.value.detail
    assert "Model mount missing" in exc.value.detail and "Regenerate" in exc.value.detail
    assert pool.popens == [] and pool.admissions == []
    assert appmod._INFLIGHT == {}


def test_gate_passes_a_warning_profile(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=100.0)
    monkeypatch.setattr(appmod, "_require_launch_preflight", _REAL_REQUIRE_LAUNCH_PREFLIGHT)
    monkeypatch.setattr(appmod, "_preflight_profile", _preflight_result("warn"))
    asyncio.run(_start(_util_script(tmp_path, "start_ok", 0.5), "vllm"))
    assert len(pool.popens) == 1


def test_force_is_the_explicit_preflight_bypass(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=100.0)
    called = []

    async def fake(profile):
        called.append(profile["id"])
        return {"verdict": "fail", "checks": []}
    monkeypatch.setattr(appmod, "_require_launch_preflight", _REAL_REQUIRE_LAUNCH_PREFLIGHT)
    monkeypatch.setattr(appmod, "_preflight_profile", fake)
    asyncio.run(_start(_util_script(tmp_path, "start_f", 0.5), "vllm", force=True))
    assert called == [] and len(pool.popens) == 1


def test_gate_is_vllm_only(tmp_path, monkeypatch):
    pool = _Pool(monkeypatch, available=100.0)
    monkeypatch.setattr(appmod, "_require_launch_preflight", _REAL_REQUIRE_LAUNCH_PREFLIGHT)
    monkeypatch.setattr(appmod, "_preflight_profile", _preflight_result("fail"))
    asyncio.run(_start(_header_script(tmp_path, "start_gguf_x", 31), "llamacpp"))
    assert len(pool.popens) == 1


def test_gate_runs_inside_the_lock_before_admission():
    import inspect
    src = inspect.getsource(appmod._engine_start)
    lock = src.index("async with _LAUNCH_LOCK")
    assert lock < src.index("_prune_inflight()") < src.index("_require_launch_preflight(")
    assert src.index("_require_launch_preflight(") < src.index("await _vram_admission_check")
    assert src.index("await _vram_admission_check") < src.index("_spawn_launch(")


def test_missing_mount_source_fails_without_asking_docker_to_create_it(tmp_path, monkeypatch):
    missing = tmp_path / "removed-model"
    docker_calls = []

    async def run(*args, **kwargs):
        docker_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
    monkeypatch.setattr(appmod, "_run", run)
    facts = {"image": "vllm/vllm-openai:test", "mounts": [f"{missing}:/models/x:ro"],
             "model": "/models/x", "recipe_backed": False}
    checks = asyncio.run(appmod._preflight_runtime(facts, ""))
    assert [c["level"] for c in checks if c["check"] == "mount_source"] == ["fail"]
    assert not any(a[:2] == ("docker", "run") for a in docker_calls)
    assert not missing.exists()


# ── reclaim-label preflight check ─────────────────────────────────────────────

def _label_env(monkeypatch, state, label):
    async def cs(name):
        return state
    monkeypatch.setattr(appmod, "_container_state", cs)

    async def lbl(container, key):
        return label
    monkeypatch.setattr(appmod, "_docker_label", lbl)


def test_preflight_warns_on_unlabeled_running_container(monkeypatch):
    _label_env(monkeypatch, {"exists": True, "status": "running", "id": "x"}, None)
    out = asyncio.run(appmod._preflight_reclaim_label())
    assert [(c["level"], c["check"]) for c in out] == [("warn", "reclaim_label")]
    assert "credits NOTHING" in out[0]["detail"]


def test_preflight_ok_on_labelled_container(monkeypatch):
    _label_env(monkeypatch, {"exists": True, "status": "running", "id": "x"}, "hf_x")
    out = asyncio.run(appmod._preflight_reclaim_label())
    assert [(c["level"], c["check"]) for c in out] == [("ok", "reclaim_label")]


def test_preflight_silent_when_nothing_running(monkeypatch):
    _label_env(monkeypatch, {"exists": False, "status": "absent"}, None)
    assert asyncio.run(appmod._preflight_reclaim_label()) == []


# ── every committed profile labels its container ─────────────────────────────

# Labelled outside this repo: flash-next's wrapper delegates `docker run` to
# ~/qwen3.8-Flash-DGX/scripts/serve.sh, which sets dgx.profile itself (verified live
# on gb10 2026-10-18: dgx.profile=hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid).
_LABELLED_EXTERNALLY = {"start_hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid.sh"}


@pytest.mark.parametrize("path", sorted((REPO / "profiles").glob("*/start_*.sh")),
                         ids=lambda p: p.name)
def test_committed_profile_labels_its_container(path):
    if path.name in _LABELLED_EXTERNALLY:
        pytest.skip("label set by an external serve script")
    code = appmod._script_code(path.read_text())
    expected = "dgx.profile=" + path.stem.removeprefix("start_")
    assert expected in code, f"{path.name} launches an unlabeled container"
