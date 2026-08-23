"""Guards on the litellm router config DMM syncs into the k8s ConfigMap.

Background: on 2026-08-20 the on-disk source carried 24 model aliases while the live
ConfigMap carried 1. They had drifted since July, and nobody noticed because nothing
compares them. Worse than the drift was its content — 19 of the 24 aliases lied about
what they routed to:

  - six impersonated Anthropic models (claude-opus-4-6, claude-sonnet-4-6, ...) while
    resolving to `openai/vllm-active`, a local uncensored model. A caller asking for
    Claude could not tell.
  - thirteen named specific checkpoints (deepseek-32b, qwen-14b, nemotron, ...) while
    also resolving to `openai/vllm-active` — i.e. whatever DMM last loaded. That is how
    a measurement ends up attributed to the wrong model.

These tests encode the rule that survived the cleanup: an alias may abbreviate what it
routes to, but it may not contradict it.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

import app as appmod


# Vendor prefixes that make a claim about *who made* the model. An alias may only carry
# one if it actually routes to that vendor.
_VENDOR_MARKERS = {
    "claude": ("anthropic",),
    "gpt-": ("openai.com", "api.openai"),
    "gemini": ("googleapis", "google"),
    "grok": ("x.ai",),
    "mistral": ("mistral.ai",),
}


def _load(text):
    return yaml.safe_load(text)["model_list"]


def _routes_to_vendor(entry, needles):
    params = entry.get("litellm_params", {})
    hay = f"{params.get('api_base', '')} {params.get('model', '')}".lower()
    return any(n in hay for n in needles)


def _impersonations(model_list):
    bad = []
    for entry in model_list:
        name = entry.get("model_name", "").lower()
        for marker, needles in _VENDOR_MARKERS.items():
            if marker in name and not _routes_to_vendor(entry, needles):
                bad.append(entry["model_name"])
    return bad


def test_impersonation_is_detected():
    """The exact shape that shipped for months must be caught."""
    bad = _impersonations(_load("""
model_list:
- model_name: claude-opus-4-6
  litellm_params:
    model: openai/vllm-active
    api_base: http://192.168.68.67:8000/v1
"""))
    assert bad == ["claude-opus-4-6"]


def test_genuine_vendor_route_is_allowed():
    """A claude-* alias is fine when it really does reach Anthropic."""
    assert _impersonations(_load("""
model_list:
- model_name: claude-sonnet-4-6
  litellm_params:
    model: anthropic/claude-sonnet-4-6
    api_base: https://api.anthropic.com
""")) == []


def test_local_aliases_are_not_flagged():
    """Role aliases and local engines carry no vendor claim, so they are unaffected."""
    assert _impersonations(_load("""
model_list:
- model_name: vllm-active
  litellm_params: {model: openai/vllm-active, api_base: 'http://h:8000/v1'}
- model_name: uncenaggro
  litellm_params: {model: openai/Qwen3.8-27B, api_base: 'http://h:8081/v1'}
- model_name: fast
  litellm_params: {model: openai/vllm-active, api_base: 'http://h:8000/v1'}
""")) == []


def test_duplicate_alias_is_rejected():
    ml = _load("""
model_list:
- model_name: dup
  litellm_params: {model: openai/a, api_base: 'http://h:8000/v1'}
- model_name: dup
  litellm_params: {model: openai/b, api_base: 'http://h:8001/v1'}
""")
    names = [e["model_name"] for e in ml]
    assert len(names) != len(set(names)), "fixture should contain a duplicate"


def test_live_config_has_no_impersonations():
    """The real file DMM syncs, when present. Skips on a host without it."""
    path = appmod.LITELLM_CONFIG
    if not path.exists():
        pytest.skip(f"no litellm config at {path}")
    model_list = _load(path.read_text())

    bad = _impersonations(model_list)
    assert not bad, f"aliases impersonate a vendor they do not route to: {bad}"

    names = [e["model_name"] for e in model_list]
    assert len(names) == len(set(names)), f"duplicate model_name in {path}"


def _model_lists_in(path):
    """Yield every model_list in a file: a bare router config, or a ConfigMap wrapping one."""
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception:
        return
    if not isinstance(doc, dict):
        return
    if isinstance(doc.get("model_list"), list):
        yield doc["model_list"]
    for value in (doc.get("data") or {}).values():
        if not isinstance(value, str) or "model_list" not in value:
            continue
        try:
            inner = yaml.safe_load(value)
        except Exception:
            continue
        if isinstance(inner, dict) and isinstance(inner.get("model_list"), list):
            yield inner["model_list"]


# Every router config on this host, not just the one DMM syncs. On 2026-08-23 six
# impersonating aliases were found in k8s/codernext-activator/litellm-config.repoint.yaml
# and four more in an uncommitted ConfigMap — both invisible to the single-file check
# above, which is why the 2026-08-20 cleanup did not stay clean.
_ROUTER_CONFIG_ROOTS = [
    Path.home() / "dgx-stack",
    Path.home() / "wt-phase2",
    Path.home() / "ai-infra",
]


def test_every_router_config_on_host_is_clean():
    checked, bad = [], {}
    for root in _ROUTER_CONFIG_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*litellm*.y*ml"):
            if ".git" in path.parts:
                continue
            for model_list in _model_lists_in(path):
                checked.append(path)
                found = _impersonations(model_list)
                if found:
                    bad.setdefault(str(path), []).extend(found)

    if not checked:
        pytest.skip("no router configs found on this host")
    assert not bad, f"impersonating aliases: {bad}"


def test_helix_mandatory_aliases_present():
    """helix/server/intent_engine.py:38-39 hardcodes these and is to stay untouched.

    Removing them from the router breaks helix's intent engine silently — it was already
    broken this way once, between the July drift and 2026-08-20.
    """
    path = appmod.LITELLM_CONFIG
    if not path.exists():
        pytest.skip(f"no litellm config at {path}")
    names = {e["model_name"] for e in _load(path.read_text())}
    assert {"fast", "coding"} <= names, "helix requires the 'fast' and 'coding' aliases"
