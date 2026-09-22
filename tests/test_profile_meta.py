"""Tests for profile script parsing and model metadata inference."""

import inspect
from pathlib import Path

import app as appmod


# ── _parse_script_meta ────────────────────────────────────────────────────────

def _write_script(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_parse_script_meta_full_header(tmp_path):
    script = _write_script(tmp_path, "start_mistral_small4.sh", (
        "#!/bin/bash\n"
        "# Name: Mistral Small 4\n"
        "# Description: 119B NVFP4 quantized\n"
        "# VRAM: 119\n"
        "docker run …\n"
    ))
    meta = appmod._parse_script_meta(script)
    assert meta["id"] == "start_mistral_small4"
    assert meta["name"] == "Mistral Small 4"
    assert meta["description"] == "119B NVFP4 quantized"
    assert meta["vram_gb"] == 119
    assert meta["script"] == str(script)


def test_parse_script_meta_vram_gb_suffix(tmp_path):
    script = _write_script(tmp_path, "start_x.sh", "# VRAM: 70GB\n")
    assert appmod._parse_script_meta(script)["vram_gb"] == 70


def test_parse_script_meta_invalid_vram_is_none(tmp_path):
    script = _write_script(tmp_path, "start_x.sh", "# VRAM: lots\n")
    assert appmod._parse_script_meta(script)["vram_gb"] is None


def test_parse_script_meta_fallback_name_from_filename(tmp_path):
    script = _write_script(tmp_path, "start_qwen3_coder.sh", "#!/bin/bash\n")
    meta = appmod._parse_script_meta(script)
    assert meta["name"] == "Qwen3 Coder"
    assert meta["description"] == "Script: start_qwen3_coder.sh"
    assert meta["vram_gb"] is None


def test_parse_script_meta_header_only_scanned_in_first_20_lines(tmp_path):
    script = _write_script(tmp_path, "start_x.sh", "#!/bin/bash\n" * 24 + "# VRAM: 50\n")
    assert appmod._parse_script_meta(script)["vram_gb"] is None


def test_parse_script_meta_missing_file_uses_fallbacks(tmp_path):
    meta = appmod._parse_script_meta(tmp_path / "start_ghost.sh")
    assert meta["name"] == "Ghost"
    assert meta["vram_gb"] is None


# ── _infer_from_name ──────────────────────────────────────────────────────────

def test_infer_from_name_dtype_and_params():
    info = appmod._infer_from_name("Qwen3-32B-FP8")
    assert info["dtype"] == "FP8"
    assert info["params_b"] == 32


def test_infer_from_name_no_signals():
    info = appmod._infer_from_name("Llama-3.3-70B-Instruct")
    assert info["dtype"] is None
    assert info["is_moe"] is None
    assert info["is_reasoning"] is None
    assert info["extra_modalities"] == []
    assert info["params_b"] == 70


def test_infer_from_name_moe_active_params_notation():
    assert appmod._infer_from_name("Qwen3-235B-A22B")["is_moe"] is True


def test_infer_from_name_reasoning_token():
    assert appmod._infer_from_name("DeepSeek-R1-Distill-Qwen-7B")["is_reasoning"] is True


def test_infer_from_name_vision_token():
    assert "Image" in appmod._infer_from_name("Qwen2.5-VL-7B-Instruct")["extra_modalities"]


def test_infer_from_name_gguf_quant_suffix():
    assert appmod._infer_from_name("Llama-3-8B-Q4_K_M")["dtype"] == "INT4"


def test_infer_from_name_embedding_token():
    assert "Embedding" in appmod._infer_from_name("bge-large-en-v1.5")["extra_modalities"]


# ── _infer_from_config ────────────────────────────────────────────────────────

def _hints(name=""):
    return appmod._infer_from_name(name)


def test_infer_from_config_torch_dtype():
    info = appmod._infer_from_config({"torch_dtype": "bfloat16"}, _hints("plain-model"))
    assert info["dtype"] == "BF16"
    assert info["is_moe"] is False
    assert info["modalities"] == ["Text"]


def test_infer_from_config_quantization_overrides_torch_dtype():
    cfg = {"torch_dtype": "bfloat16",
           "quantization_config": {"quant_method": "fp8"}}
    assert appmod._infer_from_config(cfg, _hints("m"))["dtype"] == "FP8"


def test_infer_from_config_name_quant_hint_beats_full_precision_config():
    # Config claims BF16 but the name says NVFP4 — quantized name hint wins.
    cfg = {"torch_dtype": "bfloat16"}
    assert appmod._infer_from_config(cfg, _hints("Model-NVFP4"))["dtype"] == "FP4"


def test_infer_from_config_moe_from_experts_key():
    cfg = {"torch_dtype": "bfloat16", "num_local_experts": 8}
    assert appmod._infer_from_config(cfg, _hints("m"))["is_moe"] is True


def test_infer_from_config_moe_from_known_model_type():
    cfg = {"model_type": "mixtral"}
    assert appmod._infer_from_config(cfg, _hints("m"))["is_moe"] is True


def test_infer_from_config_vision_modality():
    cfg = {"torch_dtype": "float16", "vision_config": {"hidden_size": 1024}}
    assert "Image" in appmod._infer_from_config(cfg, _hints("m"))["modalities"]


def test_infer_from_config_empty_config_falls_back_to_name_hints():
    info = appmod._infer_from_config({}, _hints("Qwen3-30B-A3B-FP8-Thinking"))
    assert info["dtype"] == "FP8"
    assert info["is_moe"] is True
    assert info["is_reasoning"] is True


# ── 04-02: derived / recommended / warnings header round-trip ─────────────────

def _generated_script(tmp_path, model_name, **config_extra):
    """Generate a real profile script into tmp_path and return its path + info."""
    import json as _json

    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"w" * 2048)
    config = {
        "model_type": "qwen3", "torch_dtype": "bfloat16",
        "max_position_embeddings": 40960, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
    }
    config.update(config_extra)
    (model_dir / "config.json").write_text(_json.dumps(config))
    script_name, script, info = appmod._build_vllm_profile_script(model_dir, model_name)
    path = tmp_path / script_name
    path.write_text(script)
    return path, script, info


def test_derived_roundtrip_through_the_script_header(tmp_path):
    """Generation → disk → parse must return exactly what generation produced."""
    import json as _json
    import re as _re

    path, script, info = _generated_script(tmp_path, "Acme/Roundtrip-7B")
    meta = appmod._parse_script_meta(path)

    assert meta["meta_error"] is None
    assert meta["derived"] == _json.loads(
        [l for l in script.splitlines() if l.startswith("# Derived:")][0][10:].strip())
    assert meta["recommended"] == _json.loads(
        [l for l in script.splitlines() if l.startswith("# Recommended:")][0][14:].strip())
    assert meta["warnings"] == list(info["warnings"])
    assert meta["generated_at"] and meta["generated_at"].endswith("Z")

    emitted = _re.search(r"\$\{VLLM_MAX_MODEL_LEN:-(\d+)\}", script)
    assert meta["derived"]["max_model_len"] == int(emitted.group(1))


def test_derived_roundtrip_carries_warnings_to_the_card(tmp_path):
    path, _, info = _generated_script(
        tmp_path, "Acme/RoundtripWarn-7B",
        layer_types=["full_attention"] * 35 + ["weird_type"])
    meta = appmod._parse_script_meta(path)
    assert meta["warnings"], "fixture should produce a warning"
    assert meta["warnings"] == list(info["warnings"])
    assert any("unknown layer type" in w for w in meta["warnings"])


def test_legacy_script_parses_with_derived_none(tmp_path):
    """A committed pre-04-02 profile has no meta headers. That is the legacy case,
    not an error."""
    import shutil
    from pathlib import Path

    legacy_dir = Path(__file__).resolve().parents[1] / "profiles" / "vLLM"
    sources = sorted(legacy_dir.glob("start_*.sh"))
    assert sources, "no committed vLLM profiles to use as a legacy sample"
    target = tmp_path / sources[0].name
    shutil.copy(sources[0], target)

    meta = appmod._parse_script_meta(target)
    assert meta["derived"] is None
    assert meta["recommended"] is None
    assert meta["warnings"] == []
    assert meta["generated_at"] is None
    assert meta["meta_error"] is None


def test_unparseable_derived_header_sets_meta_error(tmp_path):
    """Locked decision: a header the parser cannot read is a defect, not a fallback."""
    script = _write_script(tmp_path, "start_broken.sh", (
        "#!/bin/bash\n"
        "# Name: Broken\n"
        "# Derived: {not json\n"
        "docker run …\n"
    ))
    meta = appmod._parse_script_meta(script)
    assert meta["meta_error"]
    assert "# Derived:" in meta["meta_error"]
    assert meta["derived"] is None


def test_parse_script_meta_does_no_filesystem_reads_beyond_the_script(tmp_path):
    """T-04-08: `_scan_profiles` is the hot list path; Option A forbids config reads."""
    import inspect

    raw = inspect.getsource(appmod._parse_script_meta)
    # Strip comments and the docstring: prose about not reading config.json must not
    # trip the gate, and prose must not be able to satisfy it either.
    source = "\n".join(
        line.split("#", 1)[0] for line in raw.splitlines()
        if not line.strip().startswith(('"""', "#")))
    for forbidden in ("config.json", "_profile_source_dir", "glob(", "iterdir",
                      "_derive_launch_spec", "open("):
        assert forbidden not in source, f"{forbidden} must not appear in _parse_script_meta"
    assert source.count("read_text()") == 1


# ── _parse_script_flags / _classify_script (04-03) ────────────────────────────

import pathlib

import pytest

UNPARSEABLE = appmod.UNPARSEABLE
_REPO_PROFILES = pathlib.Path(appmod.__file__).parent / "profiles" / "vLLM"

# Expected class per committed script. Derived by inspection, not by running the
# classifier — this table is the oracle, so it must not be generated from the code.
_EXPECTED_CLASSES = {
    "start_hf_deepseek-ai_deepseek-r1-distill-qwen-14b.sh": "generated",
    "start_hf_deepseek-ai_deepseek-r1-distill-qwen-32b.sh": "generated",
    # Hand-written wrapper around a third-party serve.sh; no generated-by marker,
    # no placeholder, no recipe call.
    "start_hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid.sh": "legacy",
    "start_hf_openai_gpt-oss-120b.sh": "legacy",
    "start_hf_qwen2.5-14b-instruct-gptq-int8.sh": "generated",
    # Rewritten by parameterize/apply during the 04-03 live checkpoint (2026-08-21,
    # commit 43fee2e) — both were `generated`/`legacy` when this oracle was first
    # written; the .sh.bak beside each still holds the pre-rewrite text.
    "start_hf_qwen3-vl-4b-fp8.sh": "parameterized",
    "start_hf_qwen_qwen3-14b.sh": "legacy",
    "start_hf_qwen_qwen3-8b.sh": "parameterized",
    "start_hf_qwen_qwen3.6-35b-a3b-fp8.sh": "recipe",
    # Hand-written, but uses ${VLLM_MAX_MODEL_LEN...} env-var-default syntax — the
    # placeholder marker alone is the evidence, so this is `parameterized` by the
    # same design as start_hf_qwen_qwen3-8b.sh above, not `legacy`.
    "start_hf_qwen_qwen3.8-27b-fp8-mtp.sh": "parameterized",
    "start_hf_qwen_qwen3.8-27b-fp8.sh": "parameterized",
    "start_hf_qwen_qwen3.8-27b.sh": "parameterized",
    "start_hf_unsloth_qwen3.8-27b-nvfp4-mtp.sh": "parameterized",
    "start_hf_unsloth_qwen3.8-27b-nvfp4.sh": "parameterized",
    "start_nemotron_nano.sh": "legacy",
    "start_nemotron_super.sh": "legacy",
    "start_qwen3_coder_next.sh": "legacy",
    "start_qwen3_next_80b.sh": "legacy",
}


def _committed_text(name):
    """Read the COMMITTED script, not the working-tree copy. ~6 concurrent sessions
    share this checkout and the app itself rewrites profiles/vLLM/ at runtime, so a
    working-tree read makes this oracle flap for reasons unrelated to the classifier."""
    import subprocess
    out = subprocess.run(["git", "show", f"HEAD:profiles/vLLM/{name}"],
                         cwd=_REPO_PROFILES.parents[1],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _committed_profile_names():
    """Return profile scripts tracked by HEAD, ignoring shared-checkout work."""
    import subprocess
    out = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD", "profiles/vLLM/"],
        cwd=_REPO_PROFILES.parents[1], capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    return sorted(
        Path(line).name for line in out.stdout.splitlines()
        if Path(line).name.startswith("start_") and line.endswith(".sh")
    )


def test_qwen_flash_profile_preserves_500k_with_bounded_memory_default():
    text = _committed_text(
        "start_hf_nvidia_qwen3.8-flash-next-nvfp4-hybrid.sh"
    )
    assert "CTX=500000" in text
    assert 'GPU_MEM="${VLLM_GPU_MEMORY_UTILIZATION:-0.76}"' in text
    assert "GPU_MEM=0.80" not in text


def test_classify_covers_every_committed_profile(tmp_path):
    """Copied into tmp_path so the classifier is proved text-only — no live reads."""
    names = _committed_profile_names()
    assert names == sorted(_EXPECTED_CLASSES), "profile set changed; update the oracle"
    for name, expected in _EXPECTED_CLASSES.items():
        copy = tmp_path / name
        copy.write_text(_committed_text(name))
        got = appmod._classify_script(copy.read_text())
        # A committed script that has since been regenerated is legitimately
        # `parameterized` rather than `generated` — both carry the marker, and the
        # distinction the oracle is pinning is marker vs recipe vs neither.
        allowed = {"generated", "parameterized"} if expected == "generated" else {expected}
        assert got in allowed, f"{name}: {got} not in {allowed}"


def test_classify_parameterized_needs_only_the_placeholder():
    """The placeholder is the evidence; the generated-by marker is not required.

    Originally this asserted the opposite — placeholder-without-marker was read as "a
    hand-edit, not our output". `parameterize/apply` invalidated that premise: it
    rewrites HAND-WRITTEN scripts, which have no marker, so the old rule left its own
    output classified `legacy` with three `unparseable` flags and no way back.
    """
    marker = appmod._GENERATED_FROM_MARKER
    body = "docker run --max-model-len ${VLLM_MAX_MODEL_LEN:-32768}\n"
    assert appmod._classify_script(marker + "\n" + body) == "parameterized"
    assert appmod._classify_script(body) == "parameterized"


def test_parameterizing_a_hand_written_script_leaves_it_classified_parameterized(tmp_path):
    """End-to-end on the real defect: apply, then re-classify the bytes written."""
    script = tmp_path / "start_handwritten.sh"
    script.write_text("#!/bin/bash\ndocker run -d --name vllm_node \\\n"
                      "  --gpu-memory-utilization 0.75 \\\n"
                      "  --max-model-len 32768 --max-num-seqs 8 \\\n")
    assert appmod._classify_script(script.read_text()) == "legacy"
    rewritten, notes = appmod._parameterize_script_text(script.read_text())
    assert notes
    assert appmod._classify_script(rewritten) == "parameterized"


@pytest.mark.parametrize("token,expected", [
    ("32768", 32768),
    ("$VLLM_MAX_MODEL_LEN", UNPARSEABLE),
    ("${VLLM_MAX_MODEL_LEN:-32768}", UNPARSEABLE),
    ("$(cat x)", UNPARSEABLE),
    ('"$CTX"', UNPARSEABLE),
])
def test_parse_flags_literal_vs_expansion(token, expected):
    text = f"docker run --max-model-len {token} --served-model-name m\n"
    assert appmod._parse_script_flags(text)["max_model_len"] == expected


def test_parse_flags_absent_is_unparseable():
    flags = appmod._parse_script_flags("docker run --model foo\n")
    assert flags == {"max_model_len": UNPARSEABLE, "util": UNPARSEABLE,
                     "max_num_seqs": UNPARSEABLE}


def test_parse_flags_conflicting_duplicates_are_unparseable():
    text = "docker run --max-model-len 32768\ndocker run --max-model-len 65536\n"
    assert appmod._parse_script_flags(text)["max_model_len"] == UNPARSEABLE


def test_parse_flags_identical_duplicates_resolve():
    text = "docker run --max-model-len 32768\necho --max-model-len 32768\n"
    assert appmod._parse_script_flags(text)["max_model_len"] == 32768


def test_parse_flags_llamacpp_array_shape_is_unparseable():
    """The `ARGS=( --flag "$VAR" )` shape must fail safe, not report a value."""
    text = 'ARGS=( --max-model-len "$CTX" --max-num-seqs "$SEQS" )\n"${ARGS[@]}"\n'
    flags = appmod._parse_script_flags(text)
    assert flags["max_model_len"] == UNPARSEABLE
    assert flags["max_num_seqs"] == UNPARSEABLE


def test_parse_flags_reads_all_three_and_a_float_util():
    text = ("docker run --gpu-memory-utilization 0.55 \\\n"
            "  --max-model-len 32768 --max-num-seqs 4\n")
    assert appmod._parse_script_flags(text) == {
        "max_model_len": 32768, "util": 0.55, "max_num_seqs": 4}


def test_parse_flags_ignores_comment_lines():
    text = "# --max-model-len 999999 in an earlier revision\ndocker run --max-model-len 8192\n"
    assert appmod._parse_script_flags(text)["max_model_len"] == 8192


def test_parse_script_meta_surfaces_classification_and_flags(tmp_path):
    script = _write_script(tmp_path, "start_legacy.sh", (
        "#!/bin/bash\n# Name: Legacy\n"
        "docker run --gpu-memory-utilization 0.55 --max-model-len 32768 --max-num-seqs 4\n"))
    meta = appmod._parse_script_meta(script)
    assert meta["classification"] == "legacy"
    assert meta["flags"]["max_model_len"] == 32768


def test_parse_script_flags_uses_no_third_party_import():
    import inspect
    src = inspect.getsource(appmod._parse_script_flags)
    assert "import " not in src


# ── parameterize preview / apply (04-03) ──────────────────────────────────────

import hashlib
import shutil
import subprocess

_LEGACY = ("#!/bin/bash\n"
           "# Name: Legacy\n"
           "set -euo pipefail\n"
           "docker run -d --name vllm_node \\\n"
           "  --gpu-memory-utilization 0.55 \\\n"
           "  --max-model-len 32768 --max-num-seqs 4 \\\n"
           "  vllm/vllm-openai\n")


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    """Point the app's vLLM profile dir at tmp_path. HARD requirement: no test in
    this file may touch the live profiles/vLLM/ directory."""
    d = tmp_path / "vLLM"
    d.mkdir()
    monkeypatch.setitem(appmod._engine_dirs, "vllm", d)
    return d


def test_parameterize_rewrites_literals_preserving_defaults():
    out, notes = appmod._parameterize_script_text(_LEGACY)
    assert "--max-model-len ${VLLM_MAX_MODEL_LEN:-32768}" in out
    assert "--gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.55}" in out
    assert "--max-num-seqs ${VLLM_MAX_NUM_SEQS:-4}" in out
    # The rewrite changes no behaviour: every default is the original literal.
    assert len(notes) == 3


def test_parameterize_idempotent_and_bash_clean(tmp_path):
    once, _ = appmod._parameterize_script_text(_LEGACY)
    twice, _ = appmod._parameterize_script_text(once)
    assert twice == once
    script = tmp_path / "p.sh"
    script.write_text(once)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_parameterize_refuses_unparseable_flag():
    text = _LEGACY.replace("--max-model-len 32768", '--max-model-len "$CTX"')
    with pytest.raises(appmod.ParameterizeRefused):
        appmod._parameterize_script_text(text)


def test_parameterize_refuses_when_a_target_flag_is_absent():
    text = _LEGACY.replace(" --max-num-seqs 4", "")
    with pytest.raises(appmod.ParameterizeRefused):
        appmod._parameterize_script_text(text)


def test_preview_writes_nothing_and_returns_a_diff(profile_dir):
    script = profile_dir / "start_legacy.sh"
    script.write_text(_LEGACY)
    before = hashlib.sha256(script.read_bytes()).hexdigest()
    res = appmod._parameterize_preview("start_legacy")
    assert res["changed"] and "VLLM_MAX_MODEL_LEN" in res["diff"]
    assert res["diff"].startswith("---")
    assert res["sha256"] == before
    assert hashlib.sha256(script.read_bytes()).hexdigest() == before


def test_apply_with_stale_hash_is_409_and_file_untouched(profile_dir):
    script = profile_dir / "start_legacy.sh"
    script.write_text(_LEGACY)
    stale = appmod._parameterize_preview("start_legacy")["sha256"]
    script.write_text(_LEGACY + "\n# a concurrent session edited this\n")
    mutated = hashlib.sha256(script.read_bytes()).hexdigest()

    with pytest.raises(appmod.HTTPException) as exc:
        appmod._parameterize_apply("start_legacy", stale)
    assert exc.value.status_code == 409
    assert hashlib.sha256(script.read_bytes()).hexdigest() == mutated
    assert not (profile_dir / "start_legacy.sh.bak").exists()


def test_apply_writes_backup_and_parameterizes(profile_dir):
    script = profile_dir / "start_legacy.sh"
    script.write_text(_LEGACY)
    prev = appmod._parameterize_preview("start_legacy")
    res = appmod._parameterize_apply("start_legacy", prev["sha256"])
    assert res["ok"]
    bak = profile_dir / "start_legacy.sh.bak"
    assert bak.exists() and bak.read_text() == _LEGACY
    assert script.read_text() == prev["proposed"]
    assert res["profile"]["classification"] in ("legacy", "parameterized")
    assert not (profile_dir / "start_legacy.sh.tmp").exists()
    assert script.stat().st_mode & 0o111


def test_apply_refuses_unparseable_script_and_leaves_it_untouched(profile_dir):
    text = _LEGACY.replace("--max-model-len 32768", '--max-model-len "$CTX"')
    script = profile_dir / "start_hand.sh"
    script.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    with pytest.raises(appmod.HTTPException) as exc:
        appmod._parameterize_apply("start_hand", sha)
    assert exc.value.status_code == 422
    assert script.read_text() == text
    assert not (profile_dir / "start_hand.sh.bak").exists()


def test_profile_id_traversal_is_rejected(profile_dir):
    for bad in ("../../etc/passwd", "start_../x", "notstart_foo", ""):
        with pytest.raises(appmod.HTTPException) as exc:
            appmod._resolve_profile_script(bad)
        assert exc.value.status_code in (400, 404)


def test_parameterize_endpoints_require_auth():
    routes = {r.path: r for r in appmod.app.routes if hasattr(r, "dependencies")}
    for path in ("/api/vllm/profiles/{profile_id}/parameterize/preview",
                 "/api/vllm/profiles/{profile_id}/parameterize/apply"):
        deps = routes[path].dependencies
        assert any(getattr(d, "dependency", None) is appmod.verify_auth for d in deps), path


def test_apply_uses_the_atomic_write_idiom():
    import inspect
    src = inspect.getsource(appmod._parameterize_apply)
    assert "os.replace" in src and "os.chmod" in src
    assert "target.write_text" not in src


# ── 04-03 follow-up: regenerate is offered only where it is safe ──────────────

def test_regen_path_is_the_dir_the_script_names(tmp_path):
    script = tmp_path / "start_x.sh"
    script.write_text(f"#!/bin/bash\n{appmod._GENERATED_FROM_MARKER}\n"
                      "# /mnt/models/thing\nset -e\n")
    assert appmod._parse_script_meta(script)["regen_path"] == "/mnt/models/thing"


def test_hand_written_script_offers_no_regen_path(tmp_path):
    """No marker means from-hf would 409 — or replace tuning it cannot reproduce."""
    script = tmp_path / "start_handwritten.sh"
    script.write_text("#!/bin/bash\n# Name: hand tuned\ndocker run -d --gpus all\n")
    assert appmod._parse_script_meta(script)["regen_path"] is None


def test_regen_source_dir_does_not_stat_the_filesystem():
    """T-04-08: this sits on the hot list path, so it must stay string-only."""
    src = inspect.getsource(appmod._regen_source_dir)
    # Strip the docstring: prose ABOUT not statting must not satisfy — or fail — the gate.
    body = src.split('"""')[-1]
    for banned in ("is_dir", "exists", "Path("):
        assert banned not in body, banned


# ── 04-03 follow-up: templated defaults feed the placeholders ─────────────────

def test_templated_defaults_are_read_from_the_placeholder():
    text = ("docker run --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.75} \\\n"
            "  --max-model-len ${VLLM_MAX_MODEL_LEN:-32768} "
            "--max-num-seqs ${VLLM_MAX_NUM_SEQS:-8}\n")
    assert appmod._parse_templated_defaults(text) == {
        "max_model_len": 32768, "gpu_memory_utilization": 0.75, "max_num_seqs": 8}


def test_templated_defaults_stay_out_of_the_flags_verdict():
    """`flags` still says unparseable — that verdict drives classification."""
    text = "docker run --max-model-len ${VLLM_MAX_MODEL_LEN:-32768}\n"
    assert appmod._parse_script_flags(text)["max_model_len"] == UNPARSEABLE
    assert appmod._parse_templated_defaults(text)["max_model_len"] == 32768


def test_non_literal_default_is_omitted_not_guessed():
    text = "--max-model-len ${VLLM_MAX_MODEL_LEN:-$CTX} --max-num-seqs ${VLLM_MAX_NUM_SEQS:-4}\n"
    out = appmod._parse_templated_defaults(text)
    assert "max_model_len" not in out
    assert out["max_num_seqs"] == 4


def test_conflicting_templated_defaults_show_neither():
    text = ("--max-model-len ${VLLM_MAX_MODEL_LEN:-32768}\n"
            "--max-model-len ${VLLM_MAX_MODEL_LEN:-8192}\n")
    assert appmod._parse_templated_defaults(text) == {}


def test_commented_out_placeholder_is_ignored():
    assert appmod._parse_templated_defaults(
        "# --max-model-len ${VLLM_MAX_MODEL_LEN:-999}\n") == {}


def test_env_to_override_key_map_cannot_drift_from_the_allow_list():
    assert set(appmod._ENV_TO_OVERRIDE_KEY.values()) == set(appmod._OVERRIDE_ENV)


def test_parameterized_profile_exposes_script_defaults(tmp_path):
    script = tmp_path / "start_x.sh"
    script.write_text("#!/bin/bash\ndocker run "
                      "--max-model-len ${VLLM_MAX_MODEL_LEN:-16384}\n")
    meta = appmod._parse_script_meta(script)
    assert meta["classification"] == "parameterized"
    assert meta["script_defaults"]["max_model_len"] == 16384


# ── _scan_directory: a dir that will not parse must not vanish silently ────────

def _hf_model_dir(root, name, config_text):
    snap = root / name / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(config_text)
    return root / name


def _flat_model_dir(root, name, config_text):
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(config_text)
    return d


def test_scan_directory_clean_dirs_report_no_scan_error(tmp_path):
    _hf_model_dir(tmp_path, "models--acme--good", '{"torch_dtype": "bfloat16"}')
    _flat_model_dir(tmp_path, "good-flat", '{"torch_dtype": "bfloat16"}')
    res = appmod._scan_directory(tmp_path)
    assert {m["name"] for m in res["models"]} == {"good", "good-flat"}
    assert res["scan_error"] is None


def test_scan_directory_surfaces_unparseable_hf_dir(tmp_path, caplog):
    _hf_model_dir(tmp_path, "models--acme--good", '{"torch_dtype": "bfloat16"}')
    _hf_model_dir(tmp_path, "models--acme--broken", '["not a config object"]')
    with caplog.at_level("WARNING"):
        res = appmod._scan_directory(tmp_path)
    assert [m["name"] for m in res["models"]] == ["good"]
    assert "1 model dir(s) failed to parse" in res["scan_error"]
    assert "models--acme--broken" in res["scan_error"]
    assert "models--acme--broken" in caplog.text


def test_scan_directory_surfaces_unparseable_flat_dir(tmp_path, caplog):
    _flat_model_dir(tmp_path, "broken-flat", '["not a config object"]')
    with caplog.at_level("WARNING"):
        res = appmod._scan_directory(tmp_path)
    assert res["models"] == []
    assert "broken-flat" in res["scan_error"]
    assert "broken-flat" in caplog.text


def test_scan_directory_counts_every_failed_dir(tmp_path):
    _hf_model_dir(tmp_path, "models--acme--broken", '["nope"]')
    _flat_model_dir(tmp_path, "broken-flat", '["nope"]')
    res = appmod._scan_directory(tmp_path)
    assert res["scan_error"].startswith("2 model dir(s) failed to parse")
