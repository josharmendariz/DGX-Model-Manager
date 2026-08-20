"""Tests for auto-generated vLLM launch profiles.

Regression cover for the mount-scope bug: HF snapshot dirs are full of relative
symlinks into ../../blobs, so mounting snapshots/<rev> alone leaves every model
file dangling inside the container.
"""

import json

import app as appmod


def _write_model_config(path, *, moe=False):
    config = {"model_type": "qwen3", "torch_dtype": "bfloat16"}
    if moe:
        config["num_local_experts"] = 8
    path.write_text(json.dumps(config))


def _make_hf_repo(tmp_path, name="models--Acme--Widget", rev="revision-123"):
    repo_root = tmp_path / name
    blobs = repo_root / "blobs"
    snapshot = repo_root / "snapshots" / rev
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    _write_model_config(blobs / "config-blob")
    (blobs / "weights-blob").write_bytes(b"weights")
    (snapshot / "config.json").symlink_to("../../blobs/config-blob")
    (snapshot / "model.safetensors").symlink_to("../../blobs/weights-blob")
    return repo_root, snapshot


def test_custom_dir_snapshot_mounts_repo_root_not_snapshot(tmp_path):
    """The bug: /opt/models-style HF repos fell through to a snapshot-only mount."""
    repo_root, snapshot = _make_hf_repo(tmp_path)

    _, script, _ = appmod._build_vllm_profile_script(snapshot)

    # shlex.quote leaves metacharacter-free paths bare, so tmp_path values appear unquoted.
    assert f'-v {repo_root}:/models/acme_widget:ro' in script
    assert '--model /models/acme_widget/snapshots/revision-123' in script
    # the broken form must not reappear
    assert f'-v {snapshot}:' not in script


def test_symlinks_resolve_under_the_mounted_root(tmp_path):
    """blobs/ must be inside the mount for ../../blobs/<hash> to resolve."""
    repo_root, snapshot = _make_hf_repo(tmp_path)

    assert (repo_root / "blobs").is_dir()
    assert (snapshot / "config.json").resolve().is_relative_to(repo_root)
    assert json.loads((snapshot / "config.json").read_text())["model_type"] == "qwen3"


def test_flat_model_directory_remains_directly_mounted(tmp_path):
    model_dir = tmp_path / "flat-model"
    model_dir.mkdir()
    _write_model_config(model_dir / "config.json")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Flat")

    assert f'-v {model_dir}:/models/acme_flat:ro' in script
    assert '--model /models/acme_flat' in script


def test_image_and_moe_backend_are_configurable(tmp_path, monkeypatch):
    model_dir = tmp_path / "qwen-moe"
    model_dir.mkdir()
    _write_model_config(model_dir / "config.json", moe=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setitem(appmod._app_config, "vllm", {
        "image": "registry.example/vllm:blackwell",
        "moe_backend": "flashinfer_cutlass",
    })

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen-MoE")

    assert "registry.example/vllm:blackwell" in script
    assert "--moe-backend flashinfer_cutlass" in script
    # the vllm-active alias is load-bearing for litellm routing
    assert '--served-model-name Qwen/Qwen-MoE Qwen--Qwen-MoE vllm-active' in script


def test_empty_moe_backend_omits_the_flag(tmp_path, monkeypatch):
    model_dir = tmp_path / "qwen-moe2"
    model_dir.mkdir()
    _write_model_config(model_dir / "config.json", moe=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setitem(appmod._app_config, "vllm", {"moe_backend": ""})

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen-MoE2")

    assert "--moe-backend" not in script


def test_defaults_unchanged_when_no_vllm_config(tmp_path, monkeypatch):
    """Existing users with no vllm section must get the previous behaviour."""
    model_dir = tmp_path / "qwen-moe3"
    model_dir.mkdir()
    _write_model_config(model_dir / "config.json", moe=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.delitem(appmod._app_config, "vllm", raising=False)

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen-MoE3")

    assert "vllm/vllm-openai:v0.20.0" in script
    assert "--moe-backend marlin" in script
    # vllm-openai's entrypoint already starts the server — adding `vllm serve` breaks it.
    assert "vllm serve" not in script


# ── serve subcommand is an image property, not a model-family property ────────


def _bash_syntax_ok(tmp_path, script: str) -> bool:
    """Parse the generated script without executing it (safe with vLLM down)."""
    import subprocess
    path = tmp_path / "generated.sh"
    path.write_text(script)
    return subprocess.run(["bash", "-n", str(path)]).returncode == 0


def _plain_model(tmp_path, name="plain"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    _write_model_config(model_dir / "config.json")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    return model_dir


def test_exec_args_image_gets_explicit_serve_for_non_gpt_oss(tmp_path, monkeypatch):
    """The bug: `vllm serve` was gated on gpt-oss, so every other model was unlaunchable
    whenever vllm.image pointed at an image whose entrypoint execs its arguments."""
    model_dir = _plain_model(tmp_path)
    monkeypatch.setitem(appmod._app_config, "vllm", {"image": "eugr/spark-vllm:latest"})

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen3-8B")

    assert "vllm serve" in script
    # ordering: image line, then serve, then the flags it applies to
    assert script.index("eugr/spark-vllm:latest") < script.index("vllm serve")
    assert script.index("vllm serve") < script.index("--model")
    assert _bash_syntax_ok(tmp_path, script)


def test_gpt_oss_still_gets_serve(tmp_path, monkeypatch):
    model_dir = _plain_model(tmp_path, "gptoss")
    monkeypatch.setitem(appmod._app_config, "vllm", {})

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "openai/gpt-oss-120b")

    assert "vllm serve" in script
    assert _bash_syntax_ok(tmp_path, script)


def test_explicit_empty_serve_command_omits_it(tmp_path, monkeypatch):
    model_dir = _plain_model(tmp_path, "explicit-empty")
    monkeypatch.setitem(appmod._app_config, "vllm", {
        "image": "eugr/spark-vllm:latest", "serve_command": "",
    })

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen3-8B")

    assert "vllm serve" not in script
    assert _bash_syntax_ok(tmp_path, script)


def test_explicit_serve_command_forces_it_on_vllm_openai(tmp_path, monkeypatch):
    model_dir = _plain_model(tmp_path, "explicit-forced")
    monkeypatch.setitem(appmod._app_config, "vllm", {
        "image": "vllm/vllm-openai:v0.20.0", "serve_command": "vllm serve",
    })

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen3-8B")

    assert "vllm serve" in script


def test_serve_command_helper_resolution():
    assert appmod._vllm_serve_command("vllm/vllm-openai:v0.20.0", {}) == ""
    assert appmod._vllm_serve_command("eugr/spark-vllm:latest", {}) == "vllm serve"
    # a digest-pinned vllm-openai must still be recognised
    assert appmod._vllm_serve_command("vllm/vllm-openai@sha256:abc123", {}) == ""
    # unknown images default to the explicit form (safe direction)
    assert appmod._vllm_serve_command("some/unknown-image:1", {}) == "vllm serve"
    # explicit config wins in both directions
    assert appmod._vllm_serve_command("eugr/spark-vllm:latest", {"serve_command": ""}) == ""
    assert appmod._vllm_serve_command(
        "vllm/vllm-openai:v0.20.0", {"serve_command": "vllm serve"}) == "vllm serve"


# ── shell injection ──────────────────────────────────────────────────────────


import pytest
from fastapi import HTTPException


@pytest.mark.parametrize("evil", [
    # the exploit string recorded during the audit
    "evil$(id > /tmp/dgx-pwn-proof)",
    "back`id`tick",
    'has"doublequote',
    "has'singlequote",
    "has;semicolon",
    "has\nnewline",
    "has\rcarriagereturn",
    "has\x00nul",
    "has spaces",
    "a" * 129,
])
def test_metacharacter_model_names_are_rejected(tmp_path, evil):
    model_dir = _plain_model(tmp_path, "evilsrc")

    with pytest.raises(HTTPException) as exc:
        appmod._build_vllm_profile_script(model_dir, evil)
    assert exc.value.status_code == 400


def test_exploit_string_never_reaches_script_text(tmp_path):
    model_dir = _plain_model(tmp_path, "evilsrc2")
    try:
        _, script, _ = appmod._build_vllm_profile_script(
            model_dir, "evil$(id > /tmp/dgx-pwn-proof)")
    except HTTPException:
        return  # rejected outright — the intended outcome
    assert "$(id" not in script  # pragma: no cover — reached only on regression


def test_legitimate_name_is_accepted_unchanged(tmp_path):
    # Deliberately a name that matches NO configured recipe: a recipe-backed
    # model delegates to run-recipe.sh and never emits --served-model-name, which
    # would mask the passthrough this test exists to guard (REQ-03).
    model_dir = _plain_model(tmp_path, "legit")
    _, script, info = appmod._build_vllm_profile_script(
        model_dir, "Acme/Legit-Model-7B")
    assert info["name"] == "Acme/Legit-Model-7B"
    assert "--served-model-name Acme/Legit-Model-7B" in script
    assert _bash_syntax_ok(tmp_path, script)


def test_recipe_backed_name_is_accepted_unchanged(tmp_path):
    """The recipe branch must pass a legitimate name through just as safely."""
    model_dir = _plain_model(tmp_path, "legit-recipe")
    _, script, info = appmod._build_vllm_profile_script(
        model_dir, "Qwen/Qwen3.6-35B-A3B-FP8")
    assert info["name"] == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert _bash_syntax_ok(tmp_path, script)


def test_awkward_host_path_still_produces_parsable_script(tmp_path):
    """A model dir containing a space and a quote must not break the script."""
    model_dir = tmp_path / "we ird's dir"
    model_dir.mkdir()
    _write_model_config(model_dir / "config.json")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Awkward")

    assert _bash_syntax_ok(tmp_path, script)


def test_metacharacter_moe_backend_is_rejected(tmp_path, monkeypatch):
    model_dir = tmp_path / "moe-evil"
    model_dir.mkdir()
    _write_model_config(model_dir / "config.json", moe=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setitem(appmod._app_config, "vllm", {"moe_backend": "marlin; id"})

    with pytest.raises(HTTPException) as exc:
        appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen-MoE")
    assert exc.value.status_code == 400


def test_metacharacter_image_is_quoted_not_raw(tmp_path, monkeypatch):
    model_dir = _plain_model(tmp_path, "img-evil")
    monkeypatch.setitem(appmod._app_config, "vllm", {"image": "evil:$(id)"})

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Img")

    assert "evil:$(id)" not in script or "'evil:$(id)'" in script
    assert _bash_syntax_ok(tmp_path, script)


def test_regenerated_deepseek_profiles_mount_repo_root():
    """The two committed profiles predated the mount fix; they must stay migrated."""
    from pathlib import Path
    for slug in ("14b", "32b"):
        p = Path(__file__).resolve().parent.parent / "profiles" / "vLLM" / (
            f"start_hf_deepseek-ai_deepseek-r1-distill-qwen-{slug}.sh")
        text = p.read_text()
        assert "snapshots" in text
        # the mount must be the repo root (blobs/ + snapshots/), never snapshots/<rev>
        mount = [l for l in text.splitlines() if l.strip().startswith("-v ")][0]
        assert "/snapshots/" not in mount, f"{p.name} still mounts a snapshot dir"
        assert "vllm serve" in text


def test_recipe_dir_precedence_is_config_env_default(monkeypatch):
    """`vllm.recipe_dir` follows the `alerts` idiom: config -> env -> default.

    Config must win over the environment (an operator's committed choice is not
    overridden by an inherited env var), and the default only applies when
    neither is set.
    """
    monkeypatch.setattr(appmod, "_app_config",
                        {"vllm": {"recipe_dir": "/from/config"}}, raising=False)
    monkeypatch.setenv("MODEL_MANAGER_VLLM_RECIPE_DIR", "/from/env")
    assert appmod._vllm_recipe_dir() == "/from/config"

    monkeypatch.setattr(appmod, "_app_config", {"vllm": {}}, raising=False)
    assert appmod._vllm_recipe_dir() == "/from/env"

    monkeypatch.delenv("MODEL_MANAGER_VLLM_RECIPE_DIR")
    assert appmod._vllm_recipe_dir() == "~/spark-vllm-docker/recipes"


def test_recipe_dir_env_override_reaches_the_generator(monkeypatch):
    """The env layer must reach `_build_vllm_profile_script`'s vllm_cfg too —
    otherwise the generator and the preflight helpers resolve different dirs."""
    monkeypatch.setattr(appmod, "_app_config", {"vllm": {}}, raising=False)
    monkeypatch.setenv("MODEL_MANAGER_VLLM_RECIPE_DIR", "/from/env")
    assert appmod._live_vllm_cfg().get("recipe_dir") == "/from/env"


# ── Phase 04-01: env-override placeholders ────────────────────────────────────

def test_generated_script_carries_the_three_placeholders(tmp_path):
    """Criterion 1: a launch is tunable without rewriting the script on disk."""
    model_dir = _plain_model(tmp_path, "placeholders")
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Placeholder-7B")

    assert "${VLLM_MAX_MODEL_LEN:-" in script
    assert "${VLLM_GPU_MEMORY_UTILIZATION:-" in script
    assert "${VLLM_MAX_NUM_SEQS:-2}" in script


def test_placeholder_default_is_the_derived_value_not_a_constant(tmp_path):
    """The default must track `_derive_launch_spec`, or the script silently
    downgrades a model the moment the derivation improves."""
    import json as _json
    import re as _re

    model_dir = _plain_model(tmp_path, "derived")
    # A real-shaped config: the minimal fixture derives nothing, so it would prove
    # only that the fallback constant is the fallback constant.
    config = {
        "model_type": "qwen3", "torch_dtype": "bfloat16",
        "max_position_embeddings": 40960, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
    }
    (model_dir / "config.json").write_text(_json.dumps(config))
    _, script, info = appmod._build_vllm_profile_script(model_dir, "Acme/Derived-7B")

    kv_dtype = appmod._kv_dtype_from_config(config)
    spec = appmod._derive_launch_spec(
        config, weights_gb=info.get("size_gb", 0.0), pool_gb=121.0,
        kv_dtype_bytes=1 if kv_dtype == "fp8" else 2)
    expected = spec["max_model_len"]

    emitted = _re.search(r"\$\{VLLM_MAX_MODEL_LEN:-(\d+)\}", script)
    assert emitted, script
    assert int(emitted.group(1)) == int(expected)


def test_gpt_oss_branch_keeps_65536_as_its_default(tmp_path, monkeypatch):
    model_dir = _plain_model(tmp_path, "gptoss-ph")
    monkeypatch.setitem(appmod._app_config, "vllm", {})

    _, script, _ = appmod._build_vllm_profile_script(model_dir, "openai/gpt-oss-120b")

    assert "--max-model-len ${VLLM_MAX_MODEL_LEN:-65536}" in script
    assert "--max-num-seqs ${VLLM_MAX_NUM_SEQS:-2}" in script
    assert _bash_syntax_ok(tmp_path, script)


def test_parameterized_script_still_parses_as_bash(tmp_path):
    """The guard block uses [[ ]] and ${!var} — bash -n is the cheap proof."""
    model_dir = _plain_model(tmp_path, "syntax")
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Syntax-7B")
    assert _bash_syntax_ok(tmp_path, script)


def test_recipe_shape_is_left_unparameterized(tmp_path):
    """`run-recipe.sh` lives in another repo; emitting placeholders it never reads
    would advertise a knob that does nothing."""
    model_dir = _plain_model(tmp_path, "recipe-shape")
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen3.6-35B-A3B-FP8")

    assert "run-recipe.sh" in script
    assert "${VLLM_" not in script


def test_generator_emits_detached_docker_run(tmp_path):
    """079917b fixed the 12 on-disk profiles but not the generator, so any
    regeneration silently reverted half the cgroup fix."""
    model_dir = _plain_model(tmp_path, "detached")
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/Detached-7B")

    assert "docker run -d --name vllm_node" in script
    assert "exec docker run" not in script


# ─── 04-02: machine-readable header comments ──────────────────────────────────

_META_KEYS = ("# Derived:", "# Recommended:", "# Warnings:", "# Generated:")


def _header_values(script: str) -> dict:
    """Return {header-prefix: raw text after the colon} for the four meta headers."""
    out = {}
    for line in script.splitlines():
        for key in _META_KEYS:
            if line.startswith(key):
                out[key] = line[len(key):].strip()
    return out


def _real_config(model_dir, **extra):
    import json as _json
    config = {
        "model_type": "qwen3", "torch_dtype": "bfloat16",
        "max_position_embeddings": 40960, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
    }
    config.update(extra)
    (model_dir / "config.json").write_text(_json.dumps(config))
    return config


def test_meta_headers_are_valid_json(tmp_path):
    """Option A makes the header a data format; every value must round-trip."""
    import json as _json

    model_dir = _plain_model(tmp_path, "meta-json")
    _real_config(model_dir)
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/MetaJson-7B")

    values = _header_values(script)
    for key in ("# Derived:", "# Recommended:", "# Warnings:"):
        assert key in values, f"{key} missing from script"
        _json.loads(values[key])  # raises if the contract is broken
    assert values["# Generated:"].endswith("Z")


def test_meta_headers_land_within_the_first_20_lines(tmp_path):
    """`_parse_script_meta` only scans 20 lines — a header past that is invisible."""
    model_dir = _plain_model(tmp_path, "meta-window")
    _real_config(model_dir)
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/MetaWindow-7B")

    head = script.splitlines()[:20]
    for key in _META_KEYS:
        assert any(line.startswith(key) for line in head), f"{key} outside first 20 lines"


def test_derived_max_model_len_equals_the_emitted_placeholder(tmp_path):
    """One source of truth, two renderings: header and argv cannot disagree."""
    import json as _json
    import re as _re

    model_dir = _plain_model(tmp_path, "meta-agree")
    _real_config(model_dir)
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/MetaAgree-7B")

    derived = _json.loads(_header_values(script)["# Derived:"])
    emitted = _re.search(r"\$\{VLLM_MAX_MODEL_LEN:-(\d+)\}", script)
    assert emitted, script
    assert derived["max_model_len"] == int(emitted.group(1))
    assert derived["v"] == appmod._SCRIPT_META_VERSION

    util = _re.search(r"\$\{VLLM_GPU_MEMORY_UTILIZATION:-([0-9.]+)\}", script)
    assert util and float(util.group(1)) == derived["util"]
    seqs = _re.search(r"\$\{VLLM_MAX_NUM_SEQS:-(\d+)\}", script)
    assert seqs and int(seqs.group(1)) == derived["max_num_seqs"]


def test_warnings_round_trip_without_a_literal_newline(tmp_path):
    """Warning strings embed {value!r} of vendor config fields; a raw newline in one
    would split the header line and silently truncate the contract."""
    import json as _json

    model_dir = _plain_model(tmp_path, "meta-warn")
    _real_config(model_dir, layer_types=["full_attention"] * 35 + ["we\nird_type"])
    _, script, info = appmod._build_vllm_profile_script(model_dir, "Acme/MetaWarn-7B")

    raw = _header_values(script)["# Warnings:"]
    assert "\n" not in raw
    warnings = _json.loads(raw)
    assert warnings, "fixture was expected to produce at least one warning"
    assert any("unknown layer type" in w for w in warnings)
    assert len(warnings) == len(info["warnings"])


def test_recipe_backed_header_declares_itself_non_editable(tmp_path):
    import json as _json

    model_dir = _plain_model(tmp_path, "meta-recipe")
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Qwen/Qwen3.6-35B-A3B-FP8")

    derived = _json.loads(_header_values(script)["# Derived:"])
    assert derived["editable"] is False
    assert derived["reason"]


def test_recommended_header_carries_the_spec_values(tmp_path):
    import json as _json

    model_dir = _plain_model(tmp_path, "meta-rec")
    config = _real_config(model_dir)
    _, script, info = appmod._build_vllm_profile_script(model_dir, "Acme/MetaRec-7B")

    kv_dtype = appmod._kv_dtype_from_config(config)
    spec = appmod._derive_launch_spec(
        config, weights_gb=info.get("size_gb", 0.0), pool_gb=121.0,
        kv_dtype_bytes=1 if kv_dtype == "fp8" else 2)

    rec = _json.loads(_header_values(script)["# Recommended:"])
    assert rec["gpu_memory_utilization"] == spec["recommended_util"]
    assert rec["max_model_len"] == spec["max_model_len"]

    derived = _json.loads(_header_values(script)["# Derived:"])
    assert derived["declared_max_context"] == spec["declared_max_context"]
    assert derived["max_fitting_context"] == spec["max_fitting_context"]


def test_headers_do_not_break_bash_syntax(tmp_path):
    model_dir = _plain_model(tmp_path, "meta-bash")
    _real_config(model_dir, layer_types=["full_attention"] * 35 + ["od'd\"type"])
    _, script, _ = appmod._build_vllm_profile_script(model_dir, "Acme/MetaBash-7B")
    assert _bash_syntax_ok(tmp_path, script)
