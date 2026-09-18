"""Image-drift checker tests.

All offline. The resolvers hit GitHub/Docker Hub/GHCR, so every test here feeds
fixed tag lists instead -- a suite that depends on what upstream released today
fails for reasons that have nothing to do with this code.
"""

import json

import pytest

import image_updates as iu


# ── Parsing ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ref,registry,repo,tag,digest", [
    ("ghcr.io/open-webui/open-webui:v0.11.0", "ghcr.io", "open-webui/open-webui", "v0.11.0", None),
    ("postgres:15-alpine", "docker.io", "library/postgres", "15-alpine", None),
    ("prom/prometheus:v2.54.1", "docker.io", "prom/prometheus", "v2.54.1", None),
    ("192.168.68.67:5000/kalshi-bot@sha256:5a2b73", "192.168.68.67:5000", "kalshi-bot", None, "sha256:5a2b73"),
    ("codernext-activator:latest", "docker.io", "library/codernext-activator", "latest", None),
])
def test_parse_image(ref, registry, repo, tag, digest):
    got = iu.parse_image(ref)
    assert (got["registry"], got["repo"], got["tag"], got["digest"]) == (registry, repo, tag, digest)


# ── Classification ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ref,expected", [
    ("ghcr.io/open-webui/open-webui:v0.11.0", "pinned"),
    ("ghcr.io/berriai/litellm:main-v1.51.0", "pinned"),
    ("ghcr.io/timothystewart6/vllm-gb10:v0.21.0-gb10.0", "pinned"),
    ("prom/prometheus:v2.54.1", "pinned"),
    # 15-alpine floats: the registry repoints it at each 15.x patch, so it
    # already self-updates and a "15 -> 15.14" row would be pure noise.
    ("postgres:15-alpine", "partial"),
    ("codernext-activator:latest", "rolling"),
    ("reg:5000/kalshi-bot@sha256:abc", "digest"),
])
def test_classify(ref, expected):
    assert iu.classify(iu.parse_image(ref)) == expected


# ── Version comparison ────────────────────────────────────────────────────────

def test_is_upgrade_basic():
    assert iu.is_upgrade("v0.11.0", "v0.11.1")
    assert not iu.is_upgrade("v0.11.1", "v0.11.0")
    assert not iu.is_upgrade("v0.11.0", "v0.11.0")


def test_is_upgrade_refuses_major_jump():
    """Prometheus v2 -> v3 and Postgres 15 -> 16 are migrations, not bumps."""
    assert not iu.is_upgrade("v2.54.1", "v3.0.0")
    assert not iu.is_upgrade("15.4", "16.1")


def test_track_separates_variants():
    """`alpine` must never be compared against `trixie`."""
    track = iu.derive_track("15.4-alpine")
    assert track.match("15.9-alpine")
    assert not track.match("15.9-trixie")
    assert not track.match("15.9")


def test_track_handles_compound_suffix():
    track = iu.derive_track("v0.21.0-gb10.0")
    assert track.match("v0.28.0-gb10.2")
    assert not track.match("v0.28.0")


def test_rolling_tags_have_no_track():
    assert iu.derive_track("latest") is None
    assert iu.derive_track("main") is None


# ── The regressions that actually bit ─────────────────────────────────────────

def test_dockerhub_downgrade_trap():
    """Docker Hub's last_updated ordering lists 14.24 first for postgres.

    Following "newest tag" would propose 15 -> 14: both a downgrade and a major
    version change. The track regex plus the major-line rule must reject it.
    """
    candidates = ["14.24-trixie", "14.24", "14-trixie", "16.15-trixie", "15.14-alpine"]
    got = iu.pick_latest("15.4-alpine", candidates, iu.derive_track("15.4-alpine"),
                         pullable=set(candidates))
    assert got["latest"] == "15.14-alpine"


def test_git_sha_tags_are_not_versions():
    """`git-0d924a1` parses as (0, 924, 1) and outranks a real v0.11.0.

    Registry listings are full of these. Without the version-shape filter the
    checker recommends upgrading to a commit SHA.
    """
    assert not iu.is_version_like("git-0d924a1")
    assert not iu.is_version_like("sha-6ab6d5c")
    assert not iu.is_version_like("buildcache")
    assert iu.is_version_like("v0.11.1")
    assert iu.is_version_like("main-v1.98.0")


def test_prefers_same_tag_scheme():
    """Most projects keep their form: main-v1.51.0 -> main-v1.82.3, not v1.82.3."""
    candidates = ["main-v1.82.3", "v1.82.3", "main-v1.60.0"]
    got = iu.pick_latest("main-v1.51.0", candidates, iu.derive_track("main-v1.51.0"),
                         pullable=set(candidates))
    assert got["latest"] == "main-v1.82.3"
    assert got["inferred"] is False


def test_cross_scheme_fallback_is_flagged():
    """litellm publishes 1.98.0 as plain v1.98.0, not main-v1.98.0.

    A cross-scheme jump is reportable but must carry a note -- it is the one
    case where the tag was not confirmed to follow the running convention.
    """
    got = iu.pick_latest("main-v1.51.0", ["v1.98.0"], iu.derive_track("main-v1.51.0"),
                         pullable={"v1.98.0"})
    assert got["latest"] == "v1.98.0"
    assert got["inferred"] is True
    assert "confirm" in got["note"]


def test_unknown_pullability_is_not_absent():
    """pullable=None means "could not check" (GHCR), never "not there".

    Recording unknown as absent is how a checker produces a false clean.
    """
    got = iu.pick_latest("v0.11.0", ["v0.11.1"], iu.derive_track("v0.11.0"), pullable=None)
    assert got["latest"] == "v0.11.1"
    assert got["verified"] is not False


def test_no_upgrade_when_current():
    got = iu.pick_latest("v0.11.1", ["v0.11.0", "v0.11.1"], iu.derive_track("v0.11.1"),
                         pullable={"v0.11.0", "v0.11.1"})
    assert got["latest"] is None


# ── Alert contract ────────────────────────────────────────────────────────────

def _report(**over):
    row = {"deployment": "openwebui", "tag": "v0.11.0", "latest": "v0.11.1",
           "status": "outdated", "inferred": False, "verified": True, "note": None}
    row.update(over)
    return {"rows": [row]}


def test_alert_shape_matches_app_contract():
    """app.py's _send_alerts requires type/severity/message and keys cooldown on type."""
    alerts = iu.to_alerts(_report())
    assert len(alerts) == 1
    assert set(alerts[0]) >= {"type", "severity", "message"}
    assert alerts[0]["severity"] == "warning"   # red is reserved for outages


def test_alert_type_is_stable_and_version_scoped():
    """The type doubles as the dedupe key, so it must be identical run to run
    for the same target, and different once a newer release appears."""
    a1 = iu.to_alerts(_report())[0]["type"]
    a2 = iu.to_alerts(_report())[0]["type"]
    assert a1 == a2
    a3 = iu.to_alerts(_report(latest="v0.12.0"))[0]["type"]
    assert a3 != a1


def test_only_outdated_rows_alert():
    for status in ("current", "partial", "rolling", "digest", "unknown"):
        assert iu.to_alerts(_report(status=status)) == []


# ── Report-only guarantee ─────────────────────────────────────────────────────

def test_module_has_no_cluster_write_path():
    """The checker must be incapable of mutating the cluster.

    Asserted against the source rather than trusted by convention, because the
    whole safety argument for running this unattended on a timer rests on it.
    """
    src = (iu.APP_DIR / "image_updates.py").read_text()
    for verb in ("apply", "set", "scale", "patch", "delete", "rollout", "edit", "create"):
        assert f'"{verb}"' not in src.split("subprocess.run")[1].split("]")[0], verb
    assert src.count("subprocess.run") == 1


def test_collect_images_is_read_only(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        class R:
            returncode = 0
            stdout = json.dumps({"items": []})
            stderr = ""
        return R()

    monkeypatch.setattr(iu.subprocess, "run", fake_run)
    iu.collect_images("llm-inference")
    assert seen["cmd"][:3] == ["kubectl", "get", "deploy"]
