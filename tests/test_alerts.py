"""Tests for alert collection, notifier routing, and cooldown persistence."""

import json

import pytest
from fastapi.testclient import TestClient

import app as appmod


def _mem(used_pct, used_gb=0, total_gb=121):
    return {"used_pct": used_pct, "used_gb": used_gb, "total_gb": total_gb}


def _patch_health(monkeypatch, health):
    async def stub():
        return health
    monkeypatch.setattr(appmod, "_alert_endpoint_health", stub)


# ── _collect_alerts ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collect_alerts_all_clear(monkeypatch):
    monkeypatch.setattr(appmod, "_meminfo_snapshot", lambda: _mem(50))
    _patch_health(monkeypatch, {"vllm": True, "litellm": True})
    assert await appmod._collect_alerts() == []


@pytest.mark.asyncio
async def test_collect_alerts_memory_high(monkeypatch):
    monkeypatch.setitem(appmod._ALERT_THRESHOLDS, "memory_percent", 90)
    monkeypatch.setattr(appmod, "_meminfo_snapshot", lambda: _mem(95.2, 115.1, 121.0))
    _patch_health(monkeypatch, {"vllm": True, "litellm": True})
    alerts = await appmod._collect_alerts()
    assert [a["type"] for a in alerts] == ["memory_high_usage"]
    assert alerts[0]["severity"] == "critical"
    assert "95.2%" in alerts[0]["message"]


@pytest.mark.asyncio
async def test_collect_alerts_endpoint_failures_at_threshold(monkeypatch):
    monkeypatch.setitem(appmod._ALERT_THRESHOLDS, "endpoint_failures", 2)
    monkeypatch.setattr(appmod, "_meminfo_snapshot", lambda: _mem(50))
    _patch_health(monkeypatch, {"vllm": False, "litellm": False})
    alerts = await appmod._collect_alerts()
    assert [a["type"] for a in alerts] == ["endpoint_failures"]
    assert "vllm" in alerts[0]["message"] and "litellm" in alerts[0]["message"]


@pytest.mark.asyncio
async def test_collect_alerts_single_failure_below_threshold(monkeypatch):
    monkeypatch.setitem(appmod._ALERT_THRESHOLDS, "endpoint_failures", 2)
    monkeypatch.setattr(appmod, "_meminfo_snapshot", lambda: _mem(50))
    _patch_health(monkeypatch, {"vllm": False, "litellm": True})
    assert await appmod._collect_alerts() == []


# ── _send_alerts / cooldown ───────────────────────────────────────────────────

ALERT = {"type": "memory_high_usage", "severity": "critical", "message": "test"}


@pytest.fixture
def mock_channel(monkeypatch):
    """Register a recording notifier as the only enabled channel."""
    calls = []

    def notifier(alert, cfg):
        calls.append(alert["type"])
        return True

    monkeypatch.setitem(appmod._ALERT_NOTIFIERS, "mock", notifier)
    monkeypatch.setattr(appmod, "_ALERT_CHANNELS", {"mock": {"enabled": True}})
    return calls


def test_send_alerts_delivers_and_records(mock_channel):
    assert appmod._send_alerts([ALERT]) == ["memory_high_usage"]
    assert mock_channel == ["memory_high_usage"]
    assert "memory_high_usage" in appmod._last_alert_sent


def test_send_alerts_cooldown_suppresses_repeat(mock_channel):
    appmod._send_alerts([ALERT])
    assert appmod._send_alerts([ALERT]) == []
    assert mock_channel == ["memory_high_usage"]  # only the first delivery


def test_send_alerts_force_bypasses_cooldown(mock_channel):
    appmod._send_alerts([ALERT])
    assert appmod._send_alerts([ALERT], force=True) == ["memory_high_usage"]
    assert mock_channel == ["memory_high_usage", "memory_high_usage"]


def test_send_alerts_failed_delivery_retries_next_time(monkeypatch):
    monkeypatch.setitem(appmod._ALERT_NOTIFIERS, "mock", lambda alert, cfg: False)
    monkeypatch.setattr(appmod, "_ALERT_CHANNELS", {"mock": {"enabled": True}})
    assert appmod._send_alerts([ALERT]) == []
    assert "memory_high_usage" not in appmod._last_alert_sent  # no cooldown recorded


def test_send_alerts_no_channels_is_noop(monkeypatch):
    monkeypatch.setattr(appmod, "_ALERT_CHANNELS", {})
    assert appmod._send_alerts([ALERT]) == []


def test_send_alerts_persists_state_file(mock_channel):
    appmod._send_alerts([ALERT])
    saved = json.loads(appmod._ALERT_STATE_FILE.read_text())
    assert "memory_high_usage" in saved
    assert appmod._load_alert_state() == appmod._last_alert_sent


def test_load_alert_state_survives_corrupt_file():
    appmod._ALERT_STATE_FILE.write_text("{not json")
    assert appmod._load_alert_state() == {}


def test_discord_channel_not_ready_without_webhook(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(appmod, "_ALERT_CHANNELS", {"discord": {"enabled": True}})
    assert appmod._configured_alert_channels() == {}


def test_discord_webhook_config_beats_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://env.example/hook")
    assert appmod._resolve_discord_webhook({"webhook_url": "https://cfg.example/hook"}) \
        == "https://cfg.example/hook"
    assert appmod._resolve_discord_webhook({}) == "https://env.example/hook"


# ── POST /api/alerts/check ────────────────────────────────────────────────────

def test_alerts_check_endpoint(mock_channel, monkeypatch):
    async def stub_collect():
        return [ALERT]

    monkeypatch.setattr(appmod, "_collect_alerts", stub_collect)
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    client = TestClient(appmod.app)  # no lifespan needed for this endpoint
    r = client.post("/api/alerts/check")
    assert r.status_code == 200
    d = r.json()
    assert d["alerts"] == [ALERT]
    assert d["sent"] == ["memory_high_usage"]
    assert d["channels"] == ["mock"]
    assert d["webhook_configured"] is False
