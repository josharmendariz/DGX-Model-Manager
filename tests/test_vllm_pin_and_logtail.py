"""vLLM pinned model (load on boot) and load-log tail.

No docker, no GPU: container state, health and the launch itself are stubbed. The
properties under test are the ones that make an unattended boot safe — once per
boot, never over a running container, never a surprise download.
"""

import os
import subprocess

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app as appmod


# ── log tail ──────────────────────────────────────────────────────────────────

def test_clean_log_lines_collapses_tqdm_and_strips_ansi():
    text = ("first\n"
            "Loading shards:  10%|#   | 1/10\rLoading shards:  50%|#####| 5/10\r\n"
            "\x1b[32mINFO\x1b[0m ready\n"
            "\n")
    assert appmod._clean_log_lines(text, 50) == [
        "first", "Loading shards:  50%|#####| 5/10", "INFO ready"]


def test_clean_log_lines_keeps_only_last_n():
    text = "\n".join(f"line {i}" for i in range(100))
    out = appmod._clean_log_lines(text, 50)
    assert len(out) == 50 and out[0] == "line 50" and out[-1] == "line 99"


def test_tail_file_reads_only_the_end_and_drops_partial_line(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "_LOG_TAIL_READ_BYTES", 64)
    p = tmp_path / "big.log"
    p.write_text("".join(f"line-{i:04d}-xxxxxxxxxx\n" for i in range(1000)))
    out = appmod._tail_file(str(p), 50)
    assert out[-1] == "line-0999-xxxxxxxxxx"
    assert all(l.startswith("line-") and l.endswith("x") for l in out)  # no torn first line


def _fake_proc(root, pid, argv, stdout_target):
    d = root / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    os.symlink(stdout_target, d / "fd" / "1")


def test_docker_exec_log_file_finds_exec_stdout(tmp_path):
    """The host `docker exec ... <container>` process's stdout file is found via /proc."""
    proc = tmp_path / "proc"
    log = tmp_path / "vllm-active.log"
    log.write_text("x\n")
    _fake_proc(proc, 8837, ["docker", "exec", "-i", "vllm_node", "bash", "-c", "/w/x.sh"], str(log))
    _fake_proc(proc, 9000, ["docker", "logs", "-f", "vllm_node"], str(tmp_path / "other.log"))
    _fake_proc(proc, 9001, ["/usr/bin/docker", "exec", "llamacpp_node", "sh"], str(log))
    _fake_proc(proc, 9002, ["docker", "exec", "-i", "vllm_node", "bash"], "pipe:[123]")
    (proc / "self").mkdir()
    assert appmod._docker_exec_log_file("vllm_node", str(proc)) == str(log)
    assert appmod._docker_exec_log_file("sglang_node", str(proc)) is None


@pytest.mark.asyncio
async def test_log_tail_falls_back_from_empty_docker_logs_to_exec_file(tmp_path, monkeypatch):
    log = tmp_path / "vllm-active.log"
    log.write_text("a\nb\nStarting to load model\n")

    async def state(_):
        return {"exists": True, "status": "running", "id": "x", "exit_code": None}

    async def empty_docker_logs(_c, _n):
        return []   # recipe path: PID 1 is `sleep infinity`

    monkeypatch.setattr(appmod, "_container_state", state)
    monkeypatch.setattr(appmod, "_docker_logs_tail", empty_docker_logs)
    monkeypatch.setattr(appmod, "_docker_exec_log_file", lambda c: str(log))
    out = await appmod._vllm_log_tail("vllm_node", 50)
    assert out["source"] == str(log)
    assert out["lines"][-1] == "Starting to load model"


def test_log_tail_endpoint_rejects_bad_container_name():
    client = TestClient(appmod.app)
    r = client.get("/api/vllm/log-tail", params={"container": "x; rm -rf /"})
    assert r.status_code == 400


# ── pinned model ──────────────────────────────────────────────────────────────

PROFILES = [
    {"id": "good", "name": "Good Model", "script": "/x/good.sh",
     "model_dir": None, "model_missing": False},
    {"id": "gone", "name": "Gone Model", "script": "/x/gone.sh",
     "model_dir": None, "model_missing": True},
]


@pytest.fixture
def pin_env(monkeypatch, tmp_path):
    """Stubbed world: docker up, nothing serving, no container, launches recorded."""
    env = {"launched": [], "healthy": False,
           "container": {"exists": False, "status": "absent", "id": "", "exit_code": None}}

    async def run(*cmd, timeout=30):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    async def health():
        return env["healthy"]

    async def state(_):
        return env["container"]

    async def label(_c, _l):
        return None

    async def start(profile_id, scan_fn, engine_name, **kw):
        env["launched"].append(profile_id)
        return {"ok": True, "message": f"Launched {profile_id}"}

    monkeypatch.setattr(appmod, "_run", run)
    monkeypatch.setattr(appmod, "_vllm_health_ok", health)
    monkeypatch.setattr(appmod, "_container_state", state)
    monkeypatch.setattr(appmod, "_docker_label", label)
    monkeypatch.setattr(appmod, "_engine_start", start)
    monkeypatch.setattr(appmod, "_profiles_with_models", lambda k: [dict(p) for p in PROFILES])
    boot = tmp_path / "boot_id"
    boot.write_text("boot-A\n")
    monkeypatch.setattr(appmod, "_BOOT_ID_FILE", boot)
    env["boot"] = boot
    return env


@pytest.mark.asyncio
async def test_autostart_launches_pinned_once_per_boot(pin_env):
    appmod._write_pin_state({"profile": "good"})
    await appmod._pinned_autostart()
    await appmod._pinned_autostart()   # manager restart, same boot
    assert pin_env["launched"] == ["good"]
    st = appmod._read_pin_state()
    assert st["last_autostart"]["action"] == "launched"
    assert st["handled_boot_id"] == "boot-A"

    pin_env["boot"].write_text("boot-B\n")   # reboot
    await appmod._pinned_autostart()
    assert pin_env["launched"] == ["good", "good"]


@pytest.mark.asyncio
async def test_autostart_never_touches_a_running_container(pin_env):
    pin_env["container"] = {"exists": True, "status": "running", "id": "c1", "exit_code": None}
    appmod._write_pin_state({"profile": "good"})
    await appmod._pinned_autostart()
    assert pin_env["launched"] == []
    assert appmod._read_pin_state()["last_autostart"]["action"] == "skipped"


@pytest.mark.asyncio
async def test_autostart_skips_when_already_serving(pin_env):
    pin_env["healthy"] = True
    appmod._write_pin_state({"profile": "good"})
    await appmod._pinned_autostart()
    assert pin_env["launched"] == []


@pytest.mark.asyncio
async def test_autostart_refuses_missing_weights(pin_env):
    appmod._write_pin_state({"profile": "gone"})
    await appmod._pinned_autostart()
    assert pin_env["launched"] == []
    la = appmod._read_pin_state()["last_autostart"]
    assert la["action"] == "failed" and "not on disk" in la["detail"]


@pytest.mark.asyncio
async def test_autostart_records_admission_refusal(pin_env, monkeypatch):
    async def refuse(*a, **kw):
        raise HTTPException(409, "not enough memory")
    monkeypatch.setattr(appmod, "_engine_start", refuse)
    appmod._write_pin_state({"profile": "good"})
    await appmod._pinned_autostart()
    la = appmod._read_pin_state()["last_autostart"]
    assert la["action"] == "failed" and "not enough memory" in la["detail"]


@pytest.mark.asyncio
async def test_first_run_seeds_boot_without_launching(pin_env):
    """Deploying the feature must not launch the default pin in the current boot."""
    assert not appmod._PIN_FILE.exists()
    await appmod._pinned_autostart()
    assert pin_env["launched"] == []
    assert appmod._read_pin_state() == {"handled_boot_id": "boot-A"}


@pytest.mark.asyncio
async def test_default_pin_launches_on_next_boot(pin_env, monkeypatch):
    monkeypatch.setattr(appmod, "_DEFAULT_PIN_PROFILE", "good")
    await appmod._pinned_autostart()          # first run: seed only
    pin_env["boot"].write_text("boot-B\n")
    await appmod._pinned_autostart()
    assert pin_env["launched"] == ["good"]


@pytest.mark.asyncio
async def test_explicit_unpin_overrides_default(pin_env, monkeypatch):
    monkeypatch.setattr(appmod, "_DEFAULT_PIN_PROFILE", "good")
    appmod._write_pin_state({"profile": None})
    await appmod._pinned_autostart()
    assert pin_env["launched"] == []


def test_default_pin_is_qwen38_flash_next():
    assert appmod._effective_pin({}) == ("start_hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid", True)
    assert appmod._effective_pin({"profile": None}) == (None, False)


def _hf_dir(tmp_path, shards_present, stale_incomplete=False):
    root = tmp_path / "models--x--y"
    snap = root / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (root / "blobs").mkdir()
    import json
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    for name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"][:shards_present]:
        (snap / name).write_text("x")
    if stale_incomplete:
        (root / "blobs" / "deadbeef.1234.incomplete").write_text("x")
    return str(root)


def test_weights_complete_despite_stale_incomplete_blob(tmp_path):
    """Qwen3.8-Flash-Next on gb10: full snapshot plus an abandoned .incomplete blob."""
    assert not appmod._weights_incomplete(_hf_dir(tmp_path, 2, stale_incomplete=True))


def test_weights_incomplete_when_index_shard_missing(tmp_path):
    """Qwen3.6-35B-A3B-FP8 on 2026-10-13: dir exists, shards still downloading."""
    assert appmod._weights_incomplete(_hf_dir(tmp_path, 1))


def test_non_safetensors_dir_is_not_judged(tmp_path):
    (tmp_path / "m.gguf").write_text("x")
    assert not appmod._weights_incomplete(str(tmp_path))


def test_pin_endpoints_roundtrip(pin_env, monkeypatch):
    monkeypatch.setenv("MODEL_MANAGER_ALLOW_UNAUTH", "1")
    client = TestClient(appmod.app)
    r = client.put("/api/vllm/pin", json={"profile": "good"})
    assert r.status_code == 200, r.text
    st = appmod._read_pin_state()
    # Pinning must not fire on the next manager restart within this boot.
    assert st["profile"] == "good" and st["handled_boot_id"] == "boot-A"

    d = client.get("/api/vllm/pin").json()
    assert d["profile"] == "good" and d["name"] == "Good Model" and d["problem"] is None

    assert client.put("/api/vllm/pin", json={"profile": "gone"}).status_code == 400
    assert client.put("/api/vllm/pin", json={"profile": "nope"}).status_code == 404
    assert appmod._read_pin_state()["profile"] == "good"   # failed pins change nothing

    assert client.delete("/api/vllm/pin").status_code == 200
    d = client.get("/api/vllm/pin").json()
    assert d["profile"] is None and d["is_default"] is False
