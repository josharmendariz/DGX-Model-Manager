"""Tests for the unified Agents backend: run store, runner, progress, endpoints."""

import json

import pytest
from fastapi.testclient import TestClient

import app as appmod


def _client(monkeypatch):
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    # Keyless mutation access is loopback-only (see tests/test_auth.py); pin the host.
    monkeypatch.setattr(appmod, "APP_HOST", "127.0.0.1")
    return TestClient(appmod.app)


# ── GET /api/agents ───────────────────────────────────────────────────────────

def test_list_agents_returns_all_three_idle(monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/api/agents")
    assert r.status_code == 200
    d = r.json()
    assert [a["id"] for a in d["agents"]] == ["alert-check", "image-drift", "research-refresh"]
    for a in d["agents"]:
        assert a["running"] is False
        assert a["status"] == "idle"
        assert a["progress"] == 0
        assert a["last"] is None
        assert a["history"] == []
        assert a["name"] and a["desc"]


# ── _run_agent: progress, state, summary, history ─────────────────────────────

@pytest.mark.asyncio
async def test_run_agent_ok_propagates_progress_and_records_history(monkeypatch):
    seen = []

    async def fake(agent_id, cb):
        for frac, label in ((0.2, "step-1"), (0.6, "step-2"), (1.0, "step-3")):
            seen.append((frac, label))
            cb(frac, label)
        return {"alerts": []}

    monkeypatch.setattr(appmod, "_agent_core", fake)
    await appmod._run_agent("alert-check")

    assert seen == [(0.2, "step-1"), (0.6, "step-2"), (1.0, "step-3")]
    st = appmod._AGENTS["alert-check"]["state"]
    assert st["status"] == "ok"
    assert st["progress"] == 100
    assert st["label"] == "Done"
    assert st["summary"].startswith("No alerts")
    assert st["duration_s"] is not None

    assert len(appmod._agent_history) == 1
    rec = appmod._agent_history[0]
    assert rec["agent"] == "alert-check"
    assert rec["status"] == "ok"
    assert rec["summary"].startswith("No alerts")
    data = json.loads(appmod._AGENT_RUNS_FILE.read_text())
    assert data["runs"][0]["agent"] == "alert-check"


@pytest.mark.asyncio
async def test_run_agent_alert_summary_lists_types(monkeypatch):
    async def fake(agent_id, cb):
        return {"alerts": [{"type": "memory_high_usage"}, {"type": "endpoint_failures"}]}

    monkeypatch.setattr(appmod, "_agent_core", fake)
    await appmod._run_agent("alert-check")
    assert appmod._AGENTS["alert-check"]["state"]["summary"] == \
        "2 alert(s): memory_high_usage; endpoint_failures"


@pytest.mark.asyncio
async def test_run_agent_research_summary_counts_proposals(monkeypatch):
    async def fake(agent_id, cb):
        return {"proposed": [{"x": 1}, {"x": 2}, {"x": 3}], "summary": "found 3"}

    monkeypatch.setattr(appmod, "_agent_core", fake)
    await appmod._run_agent("research-refresh")
    assert appmod._AGENTS["research-refresh"]["state"]["summary"] == "3 proposal(s). found 3"


@pytest.mark.asyncio
async def test_run_agent_image_drift_summaries(monkeypatch):
    async def ok(agent_id, cb):
        return {"ok": True, "rows": [
            {"status": "outdated"}, {"status": "current"}, {"status": "current"}]}

    async def fail(agent_id, cb):
        return {"ok": False, "errors": ["kubectl failed"]}

    monkeypatch.setattr(appmod, "_agent_core", ok)
    await appmod._run_agent("image-drift")
    assert appmod._AGENTS["image-drift"]["state"]["summary"] == "1 outdated of 3 image(s) checked."

    monkeypatch.setattr(appmod, "_agent_core", fail)
    await appmod._run_agent("image-drift")
    assert appmod._AGENTS["image-drift"]["state"]["summary"] == "Check failed: kubectl failed"


@pytest.mark.asyncio
async def test_run_agent_error_recorded(monkeypatch):
    async def boom(agent_id, cb):
        raise RuntimeError("boom")

    monkeypatch.setattr(appmod, "_agent_core", boom)
    await appmod._run_agent("image-drift")
    st = appmod._AGENTS["image-drift"]["state"]
    assert st["status"] == "error"
    assert "boom" in st["summary"]
    assert appmod._agent_history[0]["status"] == "error"


@pytest.mark.asyncio
async def test_history_is_capped(monkeypatch):
    monkeypatch.setattr(appmod, "_AGENT_HISTORY_CAP", 3)

    async def fast(agent_id, cb):
        return {"alerts": []}

    monkeypatch.setattr(appmod, "_agent_core", fast)
    for _ in range(5):
        await appmod._run_agent("alert-check")
    assert len(appmod._agent_history) == 3


# ── POST /api/agents/{id}/run ─────────────────────────────────────────────────

def test_post_run_unknown_agent_404(monkeypatch):
    client = _client(monkeypatch)
    r = client.post("/api/agents/nope/run")
    assert r.status_code == 404


def test_post_run_409_when_already_running(monkeypatch):
    client = _client(monkeypatch)
    appmod._AGENTS["alert-check"]["state"]["status"] = "running"
    r = client.post("/api/agents/alert-check/run")
    assert r.status_code == 409


def test_post_run_returns_success_message(monkeypatch):
    client = _client(monkeypatch)

    async def fast(agent_id, cb):
        return {"alerts": []}

    monkeypatch.setattr(appmod, "_agent_core", fast)
    r = client.post("/api/agents/alert-check/run")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["agent"] == "alert-check"
    assert d["message"] == "Alert check started"
    assert d["run_id"]