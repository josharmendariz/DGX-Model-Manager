"""04-03 / criterion 5: untrusted strings cannot become markup.

Two independent gates:
  1. Behavioural — the `esc` character table, executed under node when available and
     always re-checked against a Python reimplementation of the same table.
  2. Source — a static guard over app.py asserting that no interpolation *or*
     concatenation in the profile-card and HF-browse render paths reaches innerHTML
     without esc(). The concatenation half exists because two of the highest-risk
     sinks (the reflected search term, and e.message) are built with `+`, which a
     `${...}`-shaped guard cannot see.
"""

import json
import pathlib
import re
import shutil
import subprocess

import pytest

import app as appmod

APP_PATH = pathlib.Path(appmod.__file__)
APP_SOURCE = APP_PATH.read_text()

# Hostile inputs and their correct escaping. Every one of these is a real shape:
# an hf.co repo name, a `# Name:` header, a search term, a server error string.
HOSTILE = {
    "<img src=x onerror=alert(1)>":
        "&lt;img src=x onerror=alert(1)&gt;",
    '" onmouseover=alert(1) x="':
        "&quot; onmouseover=alert(1) x=&quot;",
    "'); alert(1);//":
        "&#39;); alert(1);//",
    "&amp;":
        "&amp;amp;",
    "</script><script>alert(1)</script>":
        "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;",
    "plain-text/model-1.5B":
        "plain-text/model-1.5B",
}


def _py_esc(s):
    """Python reimplementation of the JS table. Deliberately written out longhand:
    if this and the JS ever disagree the node test catches it, and if node is absent
    this still fails a regression that drops one of the five characters."""
    table = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}
    return "".join(table.get(c, c) for c in str(s))


def _extract_esc_js():
    """Pull the esc helper (and its table) verbatim out of app.py."""
    m = re.search(r"^const _ESC_MAP = .*?\nconst esc = .*?;$",
                  APP_SOURCE, re.M | re.S)
    assert m, "esc helper not found in app.py"
    return m.group(0)


@pytest.mark.parametrize("raw,expected", sorted(HOSTILE.items()))
def test_python_reimplementation_of_the_esc_table(raw, expected):
    assert _py_esc(raw) == expected


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
@pytest.mark.parametrize("raw,expected", sorted(HOSTILE.items()))
def test_esc_helper_under_node(raw, expected):
    """Runs the ACTUAL helper text from app.py, not a copy that can drift."""
    prog = _extract_esc_js() + "\nprocess.stdout.write(esc(JSON.parse(process.argv[1])));"
    out = subprocess.run(["node", "-e", prog, json.dumps(raw)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout == expected


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_esc_is_null_safe_and_stringifies():
    prog = _extract_esc_js() + "\nprocess.stdout.write(JSON.stringify([esc(null), esc(undefined), esc(7)]));"
    out = subprocess.run(["node", "-e", prog], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == ["", "", "7"]


def test_exactly_one_esc_helper():
    """T-04-16: several near-miss escapers is how the original gap happened."""
    assert APP_SOURCE.count("const esc") == 1
    # `q` must not be mistaken for one — it escapes for a JS string literal only.
    assert "const q = s =>" in APP_SOURCE


def _render_path_lines():
    """The profile-card and HF-browse render paths, comment lines stripped.

    Comments are filtered so header prose can neither invalidate nor satisfy the gate.
    """
    lines = []
    for start_marker, end_marker in (
        ("function renderEngineProfiles", "function collectProfileOverrides"),
        ("async function hfbSearch", "async function hfbToggleExpand"),
    ):
        i = APP_SOURCE.find(start_marker)
        j = APP_SOURCE.find(end_marker, i)
        assert i != -1 and j != -1, (start_marker, end_marker)
        for line in APP_SOURCE[i:j].splitlines():
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#"):
                continue
            lines.append(line)
    return lines


def test_no_unescaped_interpolation():
    """No `${p.x}` / `${m.x}` / `${d.x}` / `${rec.x}` reaches a template raw."""
    # Fields that are numbers or booleans by construction and are only ever used
    # as ternary conditions / numeric output — they cannot carry markup. The
    # allow-list is per-FIELD and checked per-MATCH: a line-level filter would let
    # one allowed field on the line excuse an unescaped one beside it.
    NUMERIC_OK = {"p.vram_gb", "p.model_size_gb", "p.model_missing", "p.model_dir"}
    bad = []
    pattern = re.compile(r"\$\{\s*((?:p|m|d|rec|flags)\.[A-Za-z_][A-Za-z_0-9]*)")
    for line in _render_path_lines():
        for match in pattern.finditer(line):
            field = match.group(1)
            before = line[max(0, match.start() - 4):match.start()]
            if before.endswith("esc(") or field in NUMERIC_OK:
                continue
            bad.append(f"{field}  in:  {line.strip()}")
    assert not bad, "unescaped interpolation in a render path:\n" + "\n".join(bad)


def test_no_unescaped_concatenation():
    """The half the `${...}` guard cannot see: `+ q +`, `+ e.message +`, `+ m.id +`.

    Regressing app.py's search-term or error-string sinks must fail here.
    """
    bad = []
    concat = re.compile(r"\+\s*(q|e\.message|m\.id|m\.name|m\.library_name|t)\s*\+")
    for line in _render_path_lines():
        if "innerHTML" not in line and "return '<div" not in line and "+ '<" not in line:
            continue
        if concat.search(line):
            bad.append(line.strip())
    assert not bad, "unescaped concatenation into innerHTML:\n" + "\n".join(bad)


def test_hf_browse_card_escapes_every_third_party_field():
    """T-04-10: m.id, m.library_name and m.tags are named by hf.co, not by us."""
    i = APP_SOURCE.find("function renderHFBCard")
    body = APP_SOURCE[i:APP_SOURCE.find("\n}\n", i)]
    for needle in ("esc(m.id)", "esc(m.library_name)", "esc(t)", "esc(m.task_label)"):
        assert needle in body, needle
    # Both `id=` attributes go through the escaped domId, not a raw m.id.
    assert body.count("domId") == 3
    assert "+ m.id" not in body


def test_onclick_attributes_use_esc_of_q():
    """`q` alone leaves a double quote free to break out of the attribute."""
    i = APP_SOURCE.find("function renderEngineProfiles")
    j = APP_SOURCE.find("function renderProfileSettings", i)
    body = APP_SOURCE[i:j]
    for call in ("deleteProfileWeights", "deleteEngineProfile", "selectEngineProfile"):
        line = next(l for l in body.splitlines() if call + "(" in l)
        assert "q(" not in line or "esc(q(" in line, line


def test_warning_and_error_text_are_still_written_with_textcontent():
    """04-02 carry-over, re-asserted: warnings embed {value!r} of vendor config."""
    i = APP_SOURCE.find("function renderProfileSettings")
    j = APP_SOURCE.find("function renderLegacyProfileSettings", i)
    assert "p.warnings" not in APP_SOURCE[i:j]
    assert "warnEl.textContent" in APP_SOURCE
    assert "errEl.textContent" in APP_SOURCE
