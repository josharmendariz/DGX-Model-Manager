"""Regression cover for the HF download worker and its SSE stream.

Two independent guarantees are tested here:

1. The embedded worker script starts cleanly. It previously referenced an undefined
   `_HF_XFER` and died with a NameError *before* entering its own try block, so every
   download failed while the UI just spun. These tests run the real worker as a
   subprocess under HF_HUB_OFFLINE=1 — no network, no vLLM.
2. `_hf_download_events` emits exactly one terminal event in every case, including when
   the worker says nothing at all. That is the guarantee that makes a failure like the
   one above impossible to hide again.
"""

import asyncio
import json
import os
import subprocess
import sys

import pytest

import app as appmod


# ── worker subprocess ─────────────────────────────────────────────────────────

def run_worker(env_overrides: dict, timeout: int = 60) -> tuple[int, list[dict], str]:
    """Execute the real worker script. Returns (returncode, parsed events, stderr)."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    # Drop any inherited HF_REPO_ID so the "no repo id" case tests what it claims to.
    env.pop("HF_REPO_ID", None)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", appmod._HF_DOWNLOAD_SCRIPT],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    events = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return proc.returncode, events, proc.stderr


def terminal_events(events: list[dict]) -> list[dict]:
    return [e for e in events if appmod._is_terminal_hf_event(e)]


def test_worker_script_has_no_orphaned_transfer_flag():
    """The undefined name that broke every download must not come back."""
    assert "_HF_XFER" not in appmod._HF_DOWNLOAD_SCRIPT


def test_worker_script_compiles():
    """Parse-level guard against another orphaned-name regression."""
    compile(appmod._HF_DOWNLOAD_SCRIPT, "<hf>", "exec")


def test_worker_starts_and_reports_one_terminal_event():
    rc, events, stderr = run_worker({"HF_REPO_ID": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert rc == 0, f"worker exited {rc}; stderr: {stderr[-500:]}"
    assert events and events[0].get("status") == "starting", events
    assert len(terminal_events(events)) == 1, events
    # Offline mode is what makes this network-free: the repo listing fails cleanly.
    assert terminal_events(events)[0]["status"] == "error"
    assert "NameError" not in stderr


def test_worker_without_repo_id_reports_error_not_traceback():
    rc, events, stderr = run_worker({})
    assert rc == 0, f"worker exited {rc}; stderr: {stderr[-500:]}"
    assert len(terminal_events(events)) == 1, events
    assert "HF_REPO_ID" in terminal_events(events)[0]["error"]
    assert "Traceback" not in stderr


# ── stream terminal-event contract ────────────────────────────────────────────

class _FakeStdout:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __aiter__(self):
        async def gen():
            for ln in self._lines:
                yield ln
        return gen()


class _FakeStderr:
    def __init__(self, data: bytes = b""):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeProc:
    def __init__(self, lines: list[bytes], returncode: int = 0, stderr: bytes = b""):
        self.stdout = _FakeStdout(lines)
        self.stderr = _FakeStderr(stderr)
        self._rc = returncode
        self.returncode = None
        self.terminated = False

    async def wait(self) -> int:
        self.returncode = self._rc
        return self._rc

    def terminate(self):
        self.terminated = True


def drive(monkeypatch, lines: list[bytes], returncode: int = 0,
          stderr: bytes = b"", dl_key=("owner/model", "")) -> list[dict]:
    """Run _hf_download_events to exhaustion against a fake worker process."""
    proc = _FakeProc(lines, returncode, stderr)

    async def fake_exec(*a, **kw):
        return proc

    monkeypatch.setattr(appmod.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(appmod, "_create_vllm_profile_from_path",
                        lambda path, name=None: {"ok": True, "profile": {"id": "stub"}})
    appmod._active_downloads.add(dl_key)

    async def collect():
        out = []
        async for frame in appmod._hf_download_events({}, "owner/model", dl_key):
            assert frame.startswith("data: ")
            out.append(frame[len("data: "):].strip())
        return out

    frames = asyncio.run(collect())
    parsed = []
    for f in frames:
        try:
            parsed.append(json.loads(f))
        except json.JSONDecodeError:
            parsed.append({"_raw": f})
    return parsed


def test_complete_yields_exactly_one_terminal(monkeypatch):
    events = drive(monkeypatch, [
        b'{"status": "starting", "repo": "owner/model"}\n',
        b'{"status": "complete", "path": "/tmp/x", "errors": 0}\n',
    ])
    assert len(terminal_events(events)) == 1, events
    assert any("auto_profile" in e for e in events)


def test_silent_worker_nonzero_exit_synthesizes_one_error(monkeypatch):
    events = drive(monkeypatch, [], returncode=1,
                   stderr=b"Traceback...\nNameError: name '_x' is not defined\n")
    term = terminal_events(events)
    assert len(term) == 1, events
    assert term[0]["status"] == "error"
    assert term[0]["returncode"] == 1
    assert "NameError" in term[0]["error"]


def test_silent_worker_zero_exit_still_errors(monkeypatch):
    """A worker that says nothing is a failed worker even if it exits 0."""
    events = drive(monkeypatch, [b'{"status": "starting", "repo": "owner/model"}\n'],
                   returncode=0)
    term = terminal_events(events)
    assert len(term) == 1, events
    assert term[0]["status"] == "error"


def test_worker_error_and_nonzero_exit_is_not_double_terminal(monkeypatch):
    events = drive(monkeypatch, [
        b'{"status": "starting", "repo": "owner/model"}\n',
        b'{"status": "error", "error": "boom"}\n',
    ], returncode=1)
    assert len(terminal_events(events)) == 1, events


def test_second_terminal_event_is_demoted_to_log(monkeypatch):
    events = drive(monkeypatch, [
        b'{"status": "complete", "path": "/tmp/x", "errors": 0}\n',
        b'{"status": "error", "error": "late"}\n',
    ])
    assert len(terminal_events(events)) == 1, events
    assert any("log" in e for e in events)


def test_malformed_line_is_forwarded_not_auto_profile_error(monkeypatch):
    events = drive(monkeypatch, [
        b'this is not json\n',
        b'{"status": "complete", "path": "/tmp/x", "errors": 0}\n',
    ])
    assert not any("auto_profile_error" in e for e in events), events
    assert any(e.get("_raw") == "this is not json" for e in events), events


@pytest.mark.parametrize("lines,rc", [
    ([b'{"status": "complete", "path": "/tmp/x", "errors": 0}\n'], 0),
    ([], 1),
    ([b'{"status": "error", "error": "boom"}\n'], 1),
])
def test_active_downloads_always_released(monkeypatch, lines, rc):
    key = ("owner/model", "")
    drive(monkeypatch, lines, returncode=rc, dl_key=key)
    assert key not in appmod._active_downloads
