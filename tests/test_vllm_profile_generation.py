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

    assert f'-v "{repo_root}:/models/acme_widget:ro"' in script
    assert '--model "/models/acme_widget/snapshots/revision-123"' in script
    # the broken form must not reappear
    assert f'-v "{snapshot}:' not in script


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

    assert f'-v "{model_dir}:/models/acme_flat:ro"' in script
    assert '--model "/models/acme_flat"' in script


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
    assert '--served-model-name "Qwen/Qwen-MoE" "Qwen--Qwen-MoE" vllm-active' in script


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
