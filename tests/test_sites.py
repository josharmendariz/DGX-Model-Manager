"""Tests for the /api/sites dashboards endpoint.

Two halves: the config-overlay path (URL resolution, malformed entries, entries
discovery can't see) and discovery itself — parsing, candidate filtering, probe
classification and naming. Discovery is disabled by default here so tests never
shell out to `ss`/`kubectl` or probe the machine they run on; the tests that
exercise it feed the parsers fixed text instead.
"""

import pytest
from fastapi.testclient import TestClient

import app as appmod


@pytest.fixture(autouse=True)
def _isolated_sites(monkeypatch):
    """No discovery, no cache bleed between tests — /api/sites memoises."""
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {"enabled": False})
    appmod._SITES_CACHE.update({"at": 0.0, "payload": None})
    yield
    appmod._SITES_CACHE.update({"at": 0.0, "payload": None})


def _client():
    return TestClient(appmod.app)  # no lifespan needed — reachability is stubbed


def _stub_reachable(monkeypatch, value=True):
    async def stub(url):
        return value
    monkeypatch.setattr(appmod, "_site_reachable", stub)


def _stub_probes(monkeypatch, by_port):
    """Stand in for the HTTP sweep: {port: (kind, title)}, default down."""
    async def stub(url, timeout):
        port = int(url.rstrip("/").rsplit(":", 1)[1])
        kind, title = by_port.get(port, ("down", ""))
        return {"kind": kind, "status": 200 if kind != "down" else 0,
                "title": title, "url": url}
    monkeypatch.setattr(appmod, "_probe_site", stub)


# ── Config overlay ───────────────────────────────────────────────────────────

def test_sites_empty_config(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES", [])
    r = _client().get("/api/sites")
    assert r.status_code == 200
    assert r.json()["sites"] == []
    assert r.json()["discovery"]["enabled"] is False


def test_sites_port_resolved_against_app_host(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES",
                        [{"name": "Grafana", "port": 30000, "group": "Metrics", "desc": "d"}])
    monkeypatch.setattr(appmod, "_SITES_BASE", "")
    monkeypatch.setattr(appmod, "APP_HOST", "192.0.2.10")
    _stub_reachable(monkeypatch, True)
    site = _client().get("/api/sites").json()["sites"][0]
    assert site["name"] == "Grafana"
    assert site["url"] == "http://192.0.2.10:30000"
    assert site["reachable"] is True
    assert site["source"] == "config"


def test_sites_verbatim_url_and_wildcard_host_fallback(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES", [
        {"name": "Ext", "url": "https://example.com/x"},
        {"name": "Local", "port": 3000},
    ])
    monkeypatch.setattr(appmod, "_SITES_BASE", "")
    monkeypatch.setattr(appmod, "APP_HOST", "0.0.0.0")
    _stub_reachable(monkeypatch, False)
    sites = {s["name"]: s for s in _client().get("/api/sites").json()["sites"]}
    assert sites["Ext"]["url"] == "https://example.com/x"
    # 0.0.0.0 bind falls back to the request host (TestClient's "testserver")
    assert sites["Local"]["url"] == "http://testserver:3000"
    assert sites["Local"]["reachable"] is False


def test_sites_base_override_and_scheme(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES", [{"name": "A", "port": 8443, "scheme": "https"}])
    monkeypatch.setattr(appmod, "_SITES_BASE", "myhost.example")
    _stub_reachable(monkeypatch, True)
    assert _client().get("/api/sites").json()["sites"][0]["url"] == "https://myhost.example:8443"


def test_sites_named_host_resolves_through_hosts_map(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES",
                        [{"name": "Console", "host": "controlplane", "port": 8787}])
    monkeypatch.setattr(appmod, "_SITES_HOSTS", {"controlplane": "192.0.2.50"})
    monkeypatch.setattr(appmod, "_SITES_BASE", "gb10")
    _stub_reachable(monkeypatch, True)
    site = _client().get("/api/sites").json()["sites"][0]
    # the entry's host beats sites_base, and the alias resolves to its IP
    assert site["url"] == "http://192.0.2.50:8787"


def test_sites_unmapped_host_used_literally_but_warns(monkeypatch, caplog):
    monkeypatch.setattr(appmod, "_SITES",
                        [{"name": "Console", "host": "otherbox.lan", "port": 8787}])
    monkeypatch.setattr(appmod, "_SITES_HOSTS", {"controlplane": "192.0.2.50"})
    _stub_reachable(monkeypatch, True)
    with caplog.at_level("WARNING"):
        site = _client().get("/api/sites").json()["sites"][0]
    assert site["url"] == "http://otherbox.lan:8787"
    assert "otherbox.lan" in caplog.text


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


def test_sites_response_is_cached_until_refresh(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {"enabled": False, "ttl_s": 900})
    monkeypatch.setattr(appmod, "_SITES", [{"name": "A", "port": 81}])
    monkeypatch.setattr(appmod, "_SITES_BASE", "h")
    _stub_reachable(monkeypatch, True)
    c = _client()
    assert [s["name"] for s in c.get("/api/sites").json()["sites"]] == ["A"]
    monkeypatch.setattr(appmod, "_SITES", [{"name": "B", "port": 82}])
    assert [s["name"] for s in c.get("/api/sites").json()["sites"]] == ["A"]  # served from cache
    assert [s["name"] for s in c.get("/api/sites?refresh=1").json()["sites"]] == ["B"]


# ── Discovery: parsing ───────────────────────────────────────────────────────

SS_SAMPLE = """\
LISTEN 0      4096                     127.0.0.1:631   0.0.0.0:*
LISTEN 0      511                        0.0.0.0:3000  0.0.0.0:* users:(("python3",pid=2200,fd=3))
LISTEN 0      4096               100.115.54.83:8090  0.0.0.0:* users:(("python3",pid=113036,fd=13))
LISTEN 0      4096                          *:9090  *:*
LISTEN 0      4096                       [::]:8123  [::]:*
LISTEN 0      128                127.0.0.53%lo:53    0.0.0.0:*
garbage line
"""


def test_parse_ss_listeners_normalises_addresses():
    rows = {r["port"]: r for r in appmod._parse_ss_listeners(SS_SAMPLE)}
    assert rows[631]["addr"] == "127.0.0.1"
    assert rows[3000]["addr"] == "0.0.0.0" and rows[3000]["proc"] == "python3"
    assert rows[8090]["addr"] == "100.115.54.83"
    assert rows[9090]["addr"] == "0.0.0.0"     # *:9090 is a wildcard bind
    assert rows[8123]["addr"] == "0.0.0.0"     # [::]:8123 too
    assert rows[53]["addr"] == "127.0.0.53%lo"
    assert len(rows) == 6                      # the garbage line is dropped


def test_parse_nodeport_services_keeps_only_nodeports():
    payload = {"items": [
        {"metadata": {"name": "grafana", "namespace": "grafana"},
         "spec": {"ports": [{"nodePort": 30000, "port": 80}]}},
        {"metadata": {"name": "clusterip-only", "namespace": "x"},
         "spec": {"ports": [{"port": 8000}]}},
    ]}
    assert appmod._parse_nodeport_services(payload) == [
        {"port": 30000, "svc": "grafana", "namespace": "grafana"}]


# ── Discovery: candidate filtering ───────────────────────────────────────────

def test_build_candidates_drops_loopback_and_infra_ports(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {})
    cands = appmod._build_candidates(appmod._parse_ss_listeners(SS_SAMPLE), [])
    assert set(cands) == {3000, 8090, 9090, 8123}   # 631 loopback, 53 loopback+infra
    assert cands[8090]["bind_addr"] == "100.115.54.83"
    assert cands[3000]["bind_addr"] == ""            # wildcard: reachable anywhere
    assert cands[8090]["probe_host"] == "100.115.54.83"
    assert cands[3000]["probe_host"] == "127.0.0.1"


def test_build_candidates_honours_exclude_and_include(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY",
                        {"exclude_ports": [3000], "include_ports": [631]})
    cands = appmod._build_candidates(appmod._parse_ss_listeners(SS_SAMPLE), [])
    assert 3000 not in cands
    assert 631 in cands


def test_build_candidates_prefers_wildcard_bind(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {})
    rows = [{"addr": "192.0.2.5", "port": 8080, "proc": "a"},
            {"addr": "0.0.0.0", "port": 8080, "proc": ""}]
    assert appmod._build_candidates(rows, [])[8080]["bind_addr"] == ""


def test_build_candidates_exclude_services_drops_the_listener_too(monkeypatch):
    """A NodePort on a k3s node is also a real listener, so excluding the
    service has to remove the candidate, not just the k8s-sourced half of it."""
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY",
                        {"exclude_services": ["llm-inference/openwebui"]})
    cands = appmod._build_candidates(
        [{"addr": "0.0.0.0", "port": 30080, "proc": "k3s-server"}],
        [{"port": 30080, "svc": "openwebui", "namespace": "llm-inference"}])
    assert 30080 not in cands
    # a bare service name matches in any namespace
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {"exclude_services": ["openwebui"]})
    assert appmod._build_candidates(
        [{"addr": "0.0.0.0", "port": 30080, "proc": "k3s-server"}],
        [{"port": 30080, "svc": "openwebui", "namespace": "llm-inference"}]) == {}


def test_build_candidates_merges_k8s_metadata(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {})
    cands = appmod._build_candidates(
        [], [{"port": 30080, "svc": "openwebui", "namespace": "llm-inference"}])
    assert cands[30080]["source"] == "k8s"
    assert cands[30080]["svc"] == "openwebui"


# ── Discovery: naming ────────────────────────────────────────────────────────

@pytest.mark.parametrize("title,cand,expected", [
    ("Open WebUI", {"port": 30080, "svc": "openwebui", "proc": ""}, "Open WebUI"),
    # A framework's default title names nothing — fall through to the service.
    ("Streamlit", {"port": 30002, "svc": "spark-3d-dashboard-service", "proc": ""},
     "spark-3d-dashboard"),
    ("Directory listing for /", {"port": 8088, "svc": "", "proc": "python3"}, "python3"),
    ("", {"port": 8501, "svc": "", "proc": ""}, "Port 8501"),
])
def test_discovered_name_falls_back_through_evidence(title, cand, expected):
    assert appmod._discovered_name(cand, title) == expected


# ── Discovery: probe classification ──────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, ctype="text/html", text=""):
        self.status_code, self.headers, self.text = status, {"content-type": ctype}, text


def _stub_http(monkeypatch, handler):
    class _Client:
        async def get(self, url, timeout=None, follow_redirects=False):
            return handler(url)
    monkeypatch.setattr(appmod, "_http", _Client())


@pytest.mark.asyncio
async def test_probe_classifies_html_as_ui_and_extracts_title(monkeypatch):
    _stub_http(monkeypatch, lambda url: _Resp(text="<html><title> Home\n Assistant </title>"))
    r = await appmod._probe_site("http://127.0.0.1:8123/", 1.0)
    assert r["kind"] == "ui" and r["title"] == "Home Assistant"


@pytest.mark.asyncio
async def test_probe_classifies_json_as_api(monkeypatch):
    _stub_http(monkeypatch, lambda url: _Resp(404, "application/json"))
    # 404 with no page is nothing to link to
    assert (await appmod._probe_site("http://127.0.0.1:8000/", 1.0))["kind"] == "down"
    _stub_http(monkeypatch, lambda url: _Resp(200, "application/json"))
    assert (await appmod._probe_site("http://127.0.0.1:8000/", 1.0))["kind"] == "api"


@pytest.mark.asyncio
async def test_probe_treats_auth_wall_as_ui(monkeypatch):
    _stub_http(monkeypatch, lambda url: _Resp(403, "text/html", "<title>Sign in</title>"))
    assert (await appmod._probe_site("http://127.0.0.1:30007/", 1.0))["kind"] == "ui"


@pytest.mark.asyncio
async def test_probe_retries_https_when_plain_http_fails(monkeypatch):
    def handler(url):
        if url.startswith("http://"):
            raise RuntimeError("plain HTTP sent to an HTTPS port")
        return _Resp(text="<title>Secure</title>")
    _stub_http(monkeypatch, handler)
    r = await appmod._probe_site("http://127.0.0.1:8443/", 1.0)
    assert r["kind"] == "ui" and r["url"] == "https://127.0.0.1:8443/"


@pytest.mark.asyncio
async def test_probe_down_when_nothing_answers(monkeypatch):
    def handler(url):
        raise ConnectionError("refused")
    _stub_http(monkeypatch, handler)
    assert (await appmod._probe_site("http://127.0.0.1:9999/", 1.0))["kind"] == "down"


# ── Discovery: end-to-end merge ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_sites_merges_overlay_and_drops_dead_ports(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {"enabled": True, "kubernetes": True})
    monkeypatch.setattr(appmod, "_SITES_BASE", "gb10")
    monkeypatch.setattr(appmod, "_SITES", [
        {"name": "Grafana", "port": 30000, "group": "Metrics", "desc": "curated"},
        {"name": "Remote", "url": "https://example.com/x", "group": "Ops"},
        {"name": "Gone", "port": 4444, "group": "Ops"},
    ])

    async def listeners():
        return {"ok": True, "error": "", "rows": [
            {"addr": "0.0.0.0", "port": 8123, "proc": "hass"},
            {"addr": "100.115.54.83", "port": 8090, "proc": "python3"},
            {"addr": "0.0.0.0", "port": 11434, "proc": "ollama"},
            {"addr": "0.0.0.0", "port": 7777, "proc": "dead"},
        ]}

    async def nodeports():
        return {"ok": True, "error": "", "rows": [
            {"port": 30000, "svc": "grafana", "namespace": "grafana"}]}

    monkeypatch.setattr(appmod, "_discover_listeners", listeners)
    monkeypatch.setattr(appmod, "_discover_nodeports", nodeports)
    _stub_probes(monkeypatch, {8123: ("ui", "Home Assistant"), 8090: ("ui", "DGX"),
                               30000: ("ui", "Grafana"), 11434: ("api", "")})
    _stub_reachable(monkeypatch, False)

    d = await appmod._discover_sites("req-host")
    sites = {s["name"]: s for s in d["sites"]}

    # curated name/desc/group beat the discovered <title>
    assert sites["Grafana"]["group"] == "Metrics" and sites["Grafana"]["desc"] == "curated"
    assert sites["Grafana"]["url"] == "http://gb10:30000"
    # discovered-only entries get auto groups and a link on their own bind address
    assert sites["Home Assistant"]["group"] == "Host"
    assert sites["DGX"]["url"] == "http://100.115.54.83:8090"
    # non-HTML stays, tagged api; a dead port produces no card at all
    assert sites["ollama"]["kind"] == "api"
    assert "dead" not in sites
    # configured entries discovery can't confirm are still listed, honestly dotted
    assert sites["Remote"]["url"] == "https://example.com/x"
    assert sites["Gone"]["reachable"] is False
    assert d["discovery"] == {"enabled": True, "host": {"ok": True, "error": ""},
                              "kubernetes": {"ok": True, "error": ""},
                              "candidates": 5, "ui": 3, "api": 1}


@pytest.mark.asyncio
async def test_discover_sites_host_entry_never_overlays_a_local_port(monkeypatch):
    """A remote card and a local listener can share a port number without the
    remote one hijacking the local one's name and link."""
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {"enabled": True, "kubernetes": False})
    monkeypatch.setattr(appmod, "_SITES_BASE", "gb10")
    monkeypatch.setattr(appmod, "_SITES_HOSTS", {"controlplane": "192.0.2.50"})
    monkeypatch.setattr(appmod, "_SITES", [
        {"name": "Cluster Console", "host": "controlplane", "port": 8787, "group": "Ops"}])

    async def listeners():
        return {"ok": True, "error": "", "rows": [
            {"addr": "0.0.0.0", "port": 8787, "proc": "someapp"}]}

    monkeypatch.setattr(appmod, "_discover_listeners", listeners)
    _stub_probes(monkeypatch, {8787: ("ui", "Local Thing")})
    _stub_reachable(monkeypatch, True)

    sites = {s["name"]: s for s in (await appmod._discover_sites("req-host"))["sites"]}
    assert set(sites) == {"Local Thing", "Cluster Console"}
    assert sites["Local Thing"]["url"] == "http://gb10:8787"
    assert sites["Cluster Console"]["url"] == "http://192.0.2.50:8787"


@pytest.mark.asyncio
async def test_discover_sites_survives_missing_kubectl(monkeypatch):
    monkeypatch.setattr(appmod, "_SITES_DISCOVERY", {"enabled": True})
    monkeypatch.setattr(appmod, "_SITES", [])

    async def listeners():
        return {"ok": True, "error": "", "rows": [{"addr": "0.0.0.0", "port": 3000, "proc": "n"}]}

    async def nodeports():
        return {"ok": False, "error": "kubectl: not found", "rows": []}

    monkeypatch.setattr(appmod, "_discover_listeners", listeners)
    monkeypatch.setattr(appmod, "_discover_nodeports", nodeports)
    _stub_probes(monkeypatch, {3000: ("ui", "Spark Dashboard")})
    d = await appmod._discover_sites("h")
    assert [s["name"] for s in d["sites"]] == ["Spark Dashboard"]
    assert d["discovery"]["kubernetes"] == {"ok": False, "error": "kubectl: not found"}


# ── Homelab tab ──────────────────────────────────────────────────────────────

def test_homelab_default_nodes_listed_with_both_ips():
    nodes = {n["name"]: n for n in _client().get("/api/homelab").json()["nodes"]}
    assert set(nodes) == {"controlplane", "gb10", "dailydriver", "raspberrypi"}
    assert all(n["local_ip"] and n["tailscale_ip"] for n in nodes.values())


def test_homelab_config_override_and_optional_tailscale_ip(monkeypatch):
    monkeypatch.setattr(appmod, "_HOMELAB_NODES", [
        {"name": "nas", "local_ip": "192.0.2.9"}, {"role": "nameless, dropped"}])
    assert _client().get("/api/homelab").json()["nodes"] == [
        {"name": "nas", "role": "", "local_ip": "192.0.2.9", "tailscale_ip": ""}]
