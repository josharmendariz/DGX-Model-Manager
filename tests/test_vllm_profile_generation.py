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
