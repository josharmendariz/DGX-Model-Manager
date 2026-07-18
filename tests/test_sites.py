"""Tests for the /api/sites dashboards endpoint — URL resolution + empty state."""

from fastapi.testclient import TestClient

import app as appmod


def _client():
    return TestClient(appmod.app)  # no lifespan needed — reachability is stubbed


def _stub_reachable(monkeypatch, value=True):
    async def stub(url):
        return value
    monkeypatch.setattr(appmod, "_site_reachable", stub)


def test_sites_empty_config(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES", [])
    r = _client().get("/api/sites")
    assert r.status_code == 200
    assert r.json() == {"sites": []}


def test_sites_port_resolved_against_app_host(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES",
                        [{"name": "Grafana", "port": 30000, "group": "Metrics", "desc": "d"}])
    monkeypatch.setattr(appmod, "_SITES_BASE", "")
    monkeypatch.setattr(appmod, "APP_HOST", "100.115.54.83")
    _stub_reachable(monkeypatch, True)
    d = _client().get("/api/sites").json()
    assert d["sites"] == [{"name": "Grafana", "desc": "d", "group": "Metrics",
                           "url": "http://100.115.54.83:30000", "reachable": True}]


def test_sites_verbatim_url_and_wildcard_host_fallback(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES", [
        {"name": "Ext", "url": "https://example.com/x"},
        {"name": "Local", "port": 3000},
    ])
    monkeypatch.setattr(appmod, "_SITES_BASE", "")
    monkeypatch.setattr(appmod, "APP_HOST", "0.0.0.0")
    _stub_reachable(monkeypatch, False)
    d = _client().get("/api/sites").json()
    assert d["sites"][0]["url"] == "https://example.com/x"
    # 0.0.0.0 bind falls back to the request host (TestClient's "testserver")
    assert d["sites"][1]["url"] == "http://testserver:3000"
    assert d["sites"][1]["reachable"] is False


def test_sites_base_override_and_scheme(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES", [{"name": "A", "port": 8443, "scheme": "https"}])
    monkeypatch.setattr(appmod, "_SITES_BASE", "myhost.example")
    _stub_reachable(monkeypatch, True)
    d = _client().get("/api/sites").json()
    assert d["sites"][0]["url"] == "https://myhost.example:8443"


def test_sites_skips_malformed_entries(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES", [
        {"_comment": "doc entry from config.example.json"},
        {"name": "no port or url"},
        {"name": "OK", "port": 81},
    ])
    monkeypatch.setattr(appmod, "_SITES_BASE", "h")
    _stub_reachable(monkeypatch, True)
    d = _client().get("/api/sites").json()
    assert [s["name"] for s in d["sites"]] == ["OK"]
