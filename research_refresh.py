#!/usr/bin/env python3
"""KB research-refresh job (layer 3 of the recommender).

Fetches curated Spark/GB10 sources, feeds them + the current KB to a synthesis
engine, and writes PROPOSED recommendation updates for human review. It never
touches the live KB — approval is a separate, deliberate step:

    python3 research_refresh.py            # research -> recommendations.proposed.json
    python3 research_refresh.py --show      # print the proposed diff again
    python3 research_refresh.py --apply     # merge approved proposals into the KB

Engine is pluggable via RESEARCH_ENGINE (default "codex"; "litellm" to migrate
to the local :30400 router later). Codex is the strongest option today; the
litellm path keeps migration to a config flip.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
KB_FILE = APP_DIR / "recommendations.json"
SOURCES_FILE = APP_DIR / "research_sources.json"
PROPOSED_FILE = APP_DIR / "recommendations.proposed.json"
PROMPT_FILE = APP_DIR / ".research_prompt.txt"
SCHEMA_FILE = APP_DIR / ".research_schema.json"

ENGINE = os.environ.get("RESEARCH_ENGINE", "codex")
LITELLM_BASE = os.environ.get("LITELLM_BASE", "http://127.0.0.1:30400")
LITELLM_MODEL = os.environ.get("RESEARCH_MODEL", "vllm-active")

# Structured-output contract for a proposal. `match` is intentionally a free
# `suggested_match` string — the synthesis engine proposes intent, a human
# formalizes the real match rule at approval time (avoids inventing broken
# evaluator logic).
PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposed", "summary"],
    "properties": {
        "summary": {"type": "string"},
        "proposed": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "kind", "severity", "title", "summary",
                             "action", "suggested_match", "sources", "confidence"],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["config", "tuning", "model"]},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "action": {"type": "string"},
                    "suggested_match": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["url", "note"],
                            "properties": {
                                "url": {"type": "string"},
                                "note": {"type": "string"},
                                "date": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _html_to_text(html: str, limit: int) -> str:
    html = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&#39;", "'"), ("&quot;", '"'), ("&nbsp;", " ")):
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _fetch(url: str, limit: int) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dmm-research/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return _html_to_text(r.read().decode("utf-8", "replace"), limit)
    except Exception as e:
        return f"[fetch failed: {e}]"


def _build_prompt(kb: dict, sources_cfg: dict) -> str:
    limit = sources_cfg.get("meta", {}).get("max_chars_per_source", 6000)
    existing = [{"id": r["id"], "title": r["title"], "summary": r["summary"]}
                for r in kb.get("recommendations", [])]
    blocks = []
    for s in sources_cfg.get("sources", []):
        blocks.append(f"### SOURCE: {s.get('note','')} ({s['url']})\n"
                      + _fetch(s["url"], limit))
    topics = "\n".join(f"- {t}" for t in sources_cfg.get("topics", []))
    return (
        "You are a systems engineer curating a knowledge base of NVIDIA DGX Spark "
        "(GB10 Grace Blackwell, 128GB unified LPDDR5X ~273 GB/s, SM121) local-LLM "
        "recommendations for a vLLM/SGLang model manager.\n\n"
        "Below are (A) the recommendations ALREADY in the KB and (B) freshly fetched "
        "sources. Propose NEW recommendations, or UPDATES to existing ones, that are "
        "specific to GB10/Spark and actionable in a start-script or model choice. "
        "Only include items well-supported by the sources. For each, cite the source "
        "URL(s). Reuse an existing `id` when you mean it as an update; use a new "
        "kebab-case `id` for a new item. `suggested_match` should describe, in words, "
        "when the recommendation should fire against a profile (a human will formalize "
        "the rule). Do NOT restate items already covered unless the sources add "
        "something material.\n\n"
        "Focus topics:\n" + topics + "\n\n"
        "=== (A) EXISTING KB ITEMS ===\n" + json.dumps(existing, indent=1) + "\n\n"
        "=== (B) FETCHED SOURCES ===\n" + "\n\n".join(blocks) + "\n\n"
        "Return JSON matching the provided schema. If nothing new is warranted, "
        "return an empty `proposed` array with a short `summary` saying so."
    )


def _synthesize_codex(prompt: str) -> dict:
    PROMPT_FILE.write_text(prompt)
    SCHEMA_FILE.write_text(json.dumps(PROPOSAL_SCHEMA))
    out = APP_DIR / ".research_out.json"
    inner = (f"codex exec --sandbox read-only --skip-git-repo-check --ephemeral "
             f"--output-schema {SCHEMA_FILE} -o {out} "
             f"'Read the file {PROMPT_FILE} in full and produce the JSON it asks for.'")
    proc = subprocess.run(["script", "-qec", inner, "/dev/null"],
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=300)
    if not out.exists():
        raise RuntimeError(f"codex produced no output (exit {proc.returncode}):\n"
                           + (proc.stdout or "")[-800:])
    return json.loads(out.read_text())


def _synthesize_litellm(prompt: str) -> dict:
    import urllib.request
    body = json.dumps({
        "model": LITELLM_MODEL,
        "messages": [{"role": "user", "content": prompt
                      + "\n\nReturn ONLY a JSON object with keys 'proposed' (array) "
                        "and 'summary' (string). No prose."}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(LITELLM_BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0) if m else content)


def research() -> dict:
    kb = _load(KB_FILE, {"recommendations": []})
    sources_cfg = _load(SOURCES_FILE, {"sources": [], "topics": []})
    if not sources_cfg.get("sources"):
        sys.exit("No sources configured in research_sources.json")
    prompt = _build_prompt(kb, sources_cfg)
    print(f"Engine: {ENGINE} · {len(sources_cfg['sources'])} sources · "
          f"prompt {len(prompt):,} chars", file=sys.stderr)
    synth = {"codex": _synthesize_codex, "litellm": _synthesize_litellm}.get(ENGINE)
    if not synth:
        sys.exit(f"Unknown RESEARCH_ENGINE '{ENGINE}' (use codex|litellm)")
    result = synth(prompt)
    existing_ids = {r["id"] for r in kb.get("recommendations", [])}
    for p in result.get("proposed", []):
        p["status"] = "update" if p.get("id") in existing_ids else "new"
    PROPOSED_FILE.write_text(json.dumps(result, indent=2) + "\n")
    return result


def show(result: dict) -> None:
    props = result.get("proposed", [])
    print(f"\n{result.get('summary','').strip()}\n")
    if not props:
        print("No proposals.")
        return
    for p in props:
        tag = "UPDATE" if p.get("status") == "update" else "NEW"
        print(f"[{tag} · {p.get('severity','?'):6}] {p['id']}")
        print(f"    {p['title']}")
        print(f"    match: {p.get('suggested_match','')}")
        print(f"    sources: {', '.join(s['url'] for s in p.get('sources', []))}\n")
    print(f"{len(props)} proposal(s) in {PROPOSED_FILE.name}. "
          f"Review, then --apply to merge.")


def apply() -> None:
    proposed = _load(PROPOSED_FILE, None)
    if not proposed:
        sys.exit(f"No {PROPOSED_FILE.name} to apply — run a refresh first.")
    kb = _load(KB_FILE, {"recommendations": []})
    by_id = {r["id"]: r for r in kb["recommendations"]}
    applied = 0
    for p in proposed.get("proposed", []):
        entry = {k: p[k] for k in ("id", "kind", "severity", "title", "summary",
                                   "action", "sources", "confidence") if k in p}
        # suggested_match -> a placeholder match a human must formalize before it fires.
        entry["match"] = {"type": "manual", "suggested": p.get("suggested_match", "")}
        by_id[p["id"]] = {**by_id.get(p["id"], {}), **entry}
        applied += 1
    kb["recommendations"] = list(by_id.values())
    kb.setdefault("meta", {})["last_updated"] = __import__("datetime").date.today().isoformat()
    KB_FILE.write_text(json.dumps(kb, indent=2) + "\n")
    print(f"Merged {applied} proposal(s) into {KB_FILE.name}. "
          f"Formalize each match.type == 'manual' rule, then commit.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true", help="reprint the last proposals")
    ap.add_argument("--apply", action="store_true", help="merge proposals into the KB")
    args = ap.parse_args()
    if args.apply:
        apply()
    elif args.show:
        show(_load(PROPOSED_FILE, {"proposed": [], "summary": "(none)"}))
    else:
        show(research())


if __name__ == "__main__":
    main()
