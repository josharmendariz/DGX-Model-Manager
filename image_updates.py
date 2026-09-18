#!/usr/bin/env python3
"""Container image drift checker for the inference namespace.

Reports how far each running image is behind its upstream release. Report-only
by construction: this module never mutates the cluster, and there is a test
asserting it contains no kubectl write verbs. Applying an upgrade is a manual
procedure (ai-infra/k8s/inference/UPGRADING.md) -- the value here is *knowing*,
which was the actual gap. litellm sat 47 minor versions behind unnoticed.

    python3 image_updates.py           # print the drift table
    python3 image_updates.py --json    # machine-readable report

Mirrors research_refresh.py: importable by app.py, runnable as a CLI, and
propose-only with no write path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CONFIG_FILE = APP_DIR / "config.json"
REPORT_FILE = APP_DIR / "image_drift.json"

USER_AGENT = "dgx-model-manager-image-updates/1.0"
HTTP_TIMEOUT = 15

DEFAULTS = {
    "enabled": True,
    "namespace": "llm-inference",
    "interval_s": 86400,
    "tracks": {},        # image repo -> explicit track regex, overriding derivation
    "github_repos": {},  # image repo -> owner/name, when it isn't inferable
}

# GHCR paths usually match their GitHub repo, but not always -- the registry
# path is lowercased while the GitHub org may not be.
GITHUB_REPO_FIXUPS = {
    "berriai/litellm": "BerriAI/litellm",
}


def load_config() -> dict:
    try:
        cfg = json.loads(CONFIG_FILE.read_text()).get("image_updates") or {}
    except Exception:
        cfg = {}
    return {**DEFAULTS, **cfg}


# ── Image reference parsing ───────────────────────────────────────────────────

def parse_image(ref: str) -> dict:
    """Split an image reference into registry / repo / tag or digest.

    Digest-pinned images (kalshi-*, hermes-webui on this box resolve to
    @sha256:...) carry no upstream tag to compare against, so they are reported
    as `digest` rather than silently dropped -- an unnoticed skip is how a stale
    image hides.
    """
    digest = None
    tag = None
    rest = ref

    if "@" in rest:
        rest, digest = rest.split("@", 1)
    if ":" in rest.rsplit("/", 1)[-1]:
        rest, tag = rest.rsplit(":", 1)

    parts = rest.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry, repo = parts[0], "/".join(parts[1:])
    else:
        registry, repo = "docker.io", rest
        if "/" not in repo:
            repo = f"library/{repo}"

    return {"ref": ref, "registry": registry, "repo": repo, "tag": tag, "digest": digest}


# ── Version tracks ────────────────────────────────────────────────────────────
#
# The single most important rule in this module: compare a tag only against
# others of the SAME SHAPE. Docker Hub's `ordering=last_updated` returns
# `14.24-trixie` for postgres and `main` for prometheus -- following "newest
# tag" would have proposed a 15 -> 14 downgrade and a major Postgres jump as
# if they were routine updates.

_NUM = re.compile(r"\d+")


def derive_track(tag: str) -> re.Pattern | None:
    """Build a regex matching tags of the same shape as `tag`.

    Every digit run becomes \\d+, everything else is literal. So:
        v0.21.0-gb10.0  -> ^v\\d+\\.\\d+\\.\\d+-gb\\d+\\.\\d+$
        15-alpine       -> ^\\d+-alpine$      (matches 16-alpine too; see below)
        main-v1.51.0    -> ^main-v\\d+\\.\\d+\\.\\d+$

    Shape alone does not stop a major-version jump, so `is_upgrade` additionally
    refuses any move that changes the leading version component. Shape keeps
    `alpine` from being compared to `trixie`; the leading-component rule keeps
    15 from being compared to 16.
    """
    if not tag:
        return None
    if not _NUM.search(tag):
        return None  # `latest`, `main`, `stable` -- no version to track
    out, pos = [], 0
    for m in _NUM.finditer(tag):
        out.append(re.escape(tag[pos:m.start()]))
        out.append(r"\d+")
        pos = m.end()
    out.append(re.escape(tag[pos:]))
    try:
        return re.compile("^" + "".join(out) + "$")
    except re.error:
        return None


def version_key(tag: str) -> tuple:
    """Ordered numeric key for a tag: every digit run, in order."""
    return tuple(int(n) for n in _NUM.findall(tag))


def is_upgrade(current: str, candidate: str) -> bool:
    """True if `candidate` is a strictly newer release on the same major line.

    Refuses cross-major moves outright. Postgres 15 -> 16 and Prometheus v2 -> v3
    are data migrations with their own procedures, not image bumps, and surfacing
    them as "an update" would be actively misleading.
    """
    cur, cand = version_key(current), version_key(candidate)
    if not cur or not cand or len(cur) != len(cand):
        return False
    if cur[0] != cand[0]:
        return False
    return cand > cur


# ── Cluster collection ────────────────────────────────────────────────────────

def collect_images(namespace: str) -> list[dict]:
    """Every container image running in `namespace`, one row per container.

    Deliberately unfiltered. app.py's _k8s_vllm_deployments() keeps only rows
    with "vllm" in the name for the Warm tab; that filter is exactly why
    openwebui and litellm drift went unseen for months.
    """
    r = subprocess.run(
        ["kubectl", "get", "deploy", "-n", namespace, "-o", "json"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or "kubectl failed")

    rows = []
    for item in json.loads(r.stdout).get("items", []):
        name = item.get("metadata", {}).get("name", "")
        spec = item.get("spec", {})
        containers = spec.get("template", {}).get("spec", {}).get("containers", [])
        for c in containers:
            image = c.get("image", "")
            if not image:
                continue
            rows.append({
                "deployment": name,
                "container": c.get("name", ""),
                "replicas": spec.get("replicas", 0),
                **parse_image(image),
            })
    return rows


# ── Upstream resolvers ────────────────────────────────────────────────────────

def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def github_repo_for(img: dict, cfg: dict) -> str | None:
    """Map a GHCR image path to its GitHub repo."""
    repo = img["repo"]
    override = (cfg.get("github_repos") or {}).get(repo)
    if override:
        return override
    if img["registry"] != "ghcr.io":
        return None
    if repo.lower() in GITHUB_REPO_FIXUPS:
        return GITHUB_REPO_FIXUPS[repo.lower()]
    return repo if repo.count("/") == 1 else None


def github_tags(repo: str) -> list[str]:
    """Release tags from the GitHub API.

    Preferred over the GHCR registry API for ghcr.io images: ghcr.io/v2/.../tags/list
    returns an unordered 100-entry page dominated by git-<sha> build tags, so
    finding the newest semver there means paginating through noise.
    """
    tags = []
    try:
        latest = _get_json(f"https://api.github.com/repos/{repo}/releases/latest")
        if isinstance(latest, dict) and latest.get("tag_name"):
            tags.append(latest["tag_name"])
    except Exception:
        pass
    try:
        rel = _get_json(f"https://api.github.com/repos/{repo}/releases?per_page=100")
        if isinstance(rel, list):
            tags += [r["tag_name"] for r in rel
                     if isinstance(r, dict) and r.get("tag_name") and not r.get("prerelease")]
    except Exception:
        pass
    return list(dict.fromkeys(tags))


def dockerhub_tags(repo: str, pages: int = 3) -> list[str]:
    """Tag names from Docker Hub, newest-updated first.

    Order here is NOT trustworthy for picking a version -- it is only a way to
    harvest candidate names. Selection happens in pick_latest via track + semver.
    """
    tags, url = [], (f"https://hub.docker.com/v2/repositories/{repo}/tags"
                     f"?page_size=100&ordering=last_updated")
    for _ in range(pages):
        if not url:
            break
        try:
            data = _get_json(url)
        except Exception:
            break
        if not isinstance(data, dict):
            break
        tags += [t["name"] for t in data.get("results", []) if t.get("name")]
        url = data.get("next")
    return list(dict.fromkeys(tags))


# A version tag must contain a dotted numeric run. Without this guard, GHCR's
# build tags parse as enormous versions -- `git-0d924a1` reads as (0, 924, 1)
# and beats openwebui's real v0.11.0, so the checker confidently recommends
# upgrading to a commit SHA. Registry listings are full of these (`sha-6ab6d5c`,
# `buildcache`), which is why "newest tag" is never the right question.
_VERSIONISH = re.compile(r"\d+\.\d+")


def is_version_like(tag: str) -> bool:
    return bool(_VERSIONISH.search(tag))


def candidate_tags(img: dict, cfg: dict) -> tuple[list[str], set[str], str]:
    """Harvest candidate tags, the subset known pullable, and the source label.

    Both sources are used, because each is incomplete on its own. The registry
    is authoritative for what can actually be pulled -- timothystewart6 tagged
    v0.28.0-gb10.0 on GitHub but never pushed that image. GitHub releases fill
    gaps where a registry listing is truncated or buried in build tags.

    Returns the union as candidates and the registry set separately, so a
    proposal sourced only from GitHub can be reported as unverified rather than
    presented as pullable.
    """
    gh = github_repo_for(img, cfg)
    if gh:
        tags = [t for t in github_tags(gh) if is_version_like(t)]
        if tags:
            # pullable=None, not an empty set: GHCR cannot be checked anonymously
            # (see ghcr_tags), and "unknown" must never be recorded as "absent".
            return tags, None, f"github:{gh}"

    if img["registry"] == "docker.io":
        tags = [t for t in dockerhub_tags(img["repo"]) if is_version_like(t)]
        return tags, set(tags), f"dockerhub:{img['repo']}"

    tags = [t for t in ghcr_tags(img["repo"], pages=3) if is_version_like(t)]
    return tags, None, f"ghcr:{img['repo']}"


# ── Classification ────────────────────────────────────────────────────────────
#
# Not every image is comparable, and pretending otherwise is how a checker earns
# distrust. Four classes, only one of which can produce an "update available":
#
#   digest   pinned by @sha256 -- no upstream tag exists to compare against
#   rolling  no digits at all (`latest`, `main`) -- already floats, nothing to do
#   partial  fewer than 3 version components (`15-alpine`) -- ALSO floats: the
#            registry repoints 15-alpine at each 15.x patch, so it self-updates
#            on pull and a "15 -> 15.14" suggestion would be noise
#   pinned   3+ components -- the only class we resolve upstream

MIN_PINNED_COMPONENTS = 3


def classify(img: dict) -> str:
    if img.get("digest"):
        return "digest"
    tag = img.get("tag") or ""
    n = len(version_key(tag))
    if n == 0:
        return "rolling"
    if n < MIN_PINNED_COMPONENTS:
        return "partial"
    return "pinned"


def render_like(template_tag: str, numbers: tuple) -> str | None:
    """Rewrite `template_tag` with a new set of numbers, preserving its shape.

    Lets a GitHub release (`v1.98.0`) be expressed in the registry's tag form
    (`main-v1.98.0`) when litellm's GHCR path prefixes its builds.
    """
    runs = list(_NUM.finditer(template_tag))
    if len(runs) != len(numbers):
        return None
    out, pos = [], 0
    for m, num in zip(runs, numbers):
        out.append(template_tag[pos:m.start()])
        out.append(str(num))
        pos = m.end()
    out.append(template_tag[pos:])
    return "".join(out)


def ghcr_tags(repo: str, pages: int = 3) -> list[str]:
    """Tags from GHCR's /tags/list API. Last-resort candidate source only.

    GHCR anonymous access is not trustworthy enough to verify anything against.
    Measured on 2026-08-28: `open-webui/open-webui:v0.11.0` is the image running
    in this cluster right now, yet it appears in neither a 12,000-tag harvest nor
    a direct manifest probe -- both report it absent. `timothystewart6/vllm-gb10`
    behaves the opposite way, listing `v0.28.0-gb10.2` while returning 404 for
    its manifest.

    So this module does not claim to verify GHCR tags at all. GitHub Releases is
    the primary source for ghcr.io images, and the `kubectl diff` + `rollout
    status` steps in UPGRADING.md are the real safety net -- a tag that does not
    exist fails visibly at rollout, which is a far better failure than a checker
    that quietly under-reports drift.
    """
    tok = ""
    try:
        tok = _get_json(f"https://ghcr.io/token?scope=repository:{repo}:pull").get("token", "")
    except Exception:
        return []
    if not tok:
        return []

    tags, last = [], None
    for _ in range(pages):
        url = f"https://ghcr.io/v2/{repo}/tags/list?n=1000"
        if last:
            url += f"&last={urllib.parse.quote(last)}"
        try:
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {tok}", "User-Agent": USER_AGENT,
                "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                page = json.loads(resp.read().decode()).get("tags") or []
        except Exception:
            break
        if not page:
            break
        tags += page
        last = page[-1]
        if len(page) < 1000:
            break
    return list(dict.fromkeys(tags))


def pick_latest(current_tag: str, candidates: list[str], track: re.Pattern | None,
                pullable: set[str] | None = None) -> dict:
    """Best in-track upgrade for `current_tag`.

    Two passes. The in-track pass takes the newest candidate of the same shape;
    when candidates come from the registry that tag is pullable by construction,
    which is the normal case and needs no further checking.

    The cross-form pass exists for projects that changed their tag scheme:
    litellm runs `main-v1.51.0` but publishes 1.98.0 as plain `v1.98.0`. It
    compares numeric cores, then prefers a candidate that genuinely exists in
    the harvested tag list over a re-rendered guess -- `inferred` is set only
    when we had to synthesize a name nobody confirmed.
    """
    known = pullable if pullable is not None else set(candidates)
    upstream, upstream_key = None, version_key(current_tag)
    if track is not None:
        for c in candidates:
            if track.match(c) and is_upgrade(current_tag, c):
                k = version_key(c)
                if upstream is None or k > upstream_key:
                    upstream, upstream_key = c, k
    if upstream:
        return {"latest": upstream, "inferred": False, "verified": upstream in known}

    cur_core = version_key(current_tag)[:MIN_PINNED_COMPONENTS]
    best_core, cross = cur_core, None
    for c in candidates:
        core = version_key(c)[:MIN_PINNED_COMPONENTS]
        if len(core) != len(cur_core) or not core or core[0] != cur_core[0]:
            continue
        if core > best_core:
            cross, best_core = c, core
    if not cross:
        return {"latest": None, "inferred": False, "verified": None}

    rendered = render_like(current_tag,
                           best_core + version_key(current_tag)[MIN_PINNED_COMPONENTS:])
    if rendered and rendered in known:
        return {"latest": rendered, "inferred": False, "verified": True}
    return {"latest": cross, "inferred": True, "verified": cross in known,
            "note": (f"tag scheme differs from the running {current_tag} -- "
                     f"confirm {cross} is the intended build")}


# ── Report ────────────────────────────────────────────────────────────────────

def check(cfg: dict | None = None, progress_cb=None) -> dict:
    """Build the drift report. Network reads + one kubectl get; no writes.

    `progress_cb(fraction, label)` is optional; it is invoked per row so a
    caller (the app's Agents page) can surface percentage completion.
    """
    cfg = cfg or load_config()
    ns = cfg.get("namespace", "llm-inference")
    rows, errors = [], []

    if progress_cb:
        progress_cb(0.0, "Collecting cluster images")
    try:
        images = collect_images(ns)
    except Exception as e:
        return {"ok": False, "namespace": ns, "checked_at": time.time(),
                "rows": [], "errors": [str(e)]}

    n = len(images)
    cache: dict[str, tuple[list[str], set[str], str]] = {}
    for idx, img in enumerate(images):
        if progress_cb and n:
            progress_cb(0.05 + 0.95 * idx / n,
                        f"Resolving upstream · {img['deployment']} ({idx + 1}/{n})")
        kind = classify(img)
        row = {
            "deployment": img["deployment"], "container": img["container"],
            "replicas": img["replicas"], "image": img["ref"], "repo": img["repo"],
            "registry": img["registry"], "tag": img["tag"], "status": kind,
            "latest": None, "inferred": False, "verified": None,
            "source": None, "behind": None, "note": None,
        }
        if kind != "pinned":
            rows.append(row)
            continue

        key = f"{img['registry']}/{img['repo']}"
        if key not in cache:
            try:
                cache[key] = candidate_tags(img, cfg)
            except Exception as e:
                cache[key] = ([], set(), "error")
                errors.append(f"{key}: {e}")
        candidates, pullable, source = cache[key]
        row["source"] = source

        override = (cfg.get("tracks") or {}).get(img["repo"])
        track = re.compile(override) if override else derive_track(img["tag"])
        found = pick_latest(img["tag"], candidates, track, pullable)
        row["latest"] = found["latest"]
        row["inferred"] = found["inferred"]
        row["verified"] = found.get("verified")
        row["note"] = found.get("note")

        if not candidates:
            row["status"] = "unknown"
        elif row["latest"]:
            row["status"] = "outdated"
            cur, new = version_key(img["tag"]), version_key(row["latest"])
            row["behind"] = next((n - c for c, n in zip(cur, new) if n != c), 0)
        elif row["note"]:
            # Upstream moved ahead but under a tag form we could not resolve.
            # Reporting "current" here would be a false clean.
            row["status"] = "unknown"
        else:
            row["status"] = "current"
        rows.append(row)

    if progress_cb:
        progress_cb(1.0, "Writing report")
    report = {"ok": True, "namespace": ns, "checked_at": time.time(),
              "rows": rows, "errors": errors}
    try:
        REPORT_FILE.write_text(json.dumps(report, indent=2))
    except Exception:
        pass
    if progress_cb:
        progress_cb(1.0, "Done")
    return report


def load_report() -> dict:
    """Last written report, for the cached GET endpoint. No network."""
    try:
        return {**json.loads(REPORT_FILE.read_text()), "exists": True}
    except Exception:
        return {"ok": False, "rows": [], "errors": [], "exists": False}


def to_alerts(report: dict) -> list[dict]:
    """Outdated rows as alert dicts for app.py's _send_alerts().

    The `type` doubles as the cooldown key (app.py keys _last_alert_sent on it),
    so embedding the target version makes each release notify exactly once and
    then stay quiet -- no new dedupe code, and it survives restarts. Severity is
    `warning`: red is reserved for things that are actually down.
    """
    alerts = []
    for r in report.get("rows", []):
        if r.get("status") != "outdated" or not r.get("latest"):
            continue
        note = " (tag form inferred -- verify it exists before applying)" if r.get("inferred") else ""
        alerts.append({
            "type": f"image_update:{r['deployment']}:{r['latest']}",
            "severity": "warning",
            "message": (f"{r['deployment']} is running {r['tag']}, "
                        f"upstream has {r['latest']}{note}. "
                        f"See ai-infra/k8s/inference/UPGRADING.md"),
        })
    return alerts


# ── CLI ───────────────────────────────────────────────────────────────────────

_ORDER = {"outdated": 0, "unknown": 1, "current": 2, "partial": 3, "rolling": 4, "digest": 5}


def print_table(report: dict) -> None:
    rows = sorted(report.get("rows", []), key=lambda r: (_ORDER.get(r["status"], 9), r["deployment"]))
    if not report.get("ok"):
        print("check failed:", "; ".join(report.get("errors") or ["unknown error"]))
        return
    w = max([len(r["deployment"]) for r in rows] + [10])
    print(f"{'DEPLOYMENT':<{w}}  {'CURRENT':<22} {'LATEST':<22} STATUS")
    for r in rows:
        latest = r["latest"] or "-"
        if r.get("latest") and r.get("verified") is False:
            latest += " *"
        cur = r["tag"] or (r["image"].split("@")[-1][:19] if r.get("image") else "-")
        print(f"{r['deployment']:<{w}}  {cur:<22} {latest:<22} {r['status']}")
    if any(r.get("latest") and r.get("verified") is False for r in rows):
        print("\n* not present in the registry tag list; confirm before applying.")
    for r in rows:
        if r.get("note"):
            print(f"note: {r['deployment']}: {r['note']}")
    for e in report.get("errors", []):
        print("warn:", e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the raw report")
    ap.add_argument("--cached", action="store_true", help="read the last report, no network")
    args = ap.parse_args()
    report = load_report() if args.cached else check()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report)
    sys.exit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
