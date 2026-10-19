"""Startup comparison is read-only, bounded, and does not leak router credentials."""

import asyncio
import json
import subprocess

import pytest

import app as appmod


@pytest.fixture
def router(tmp_path, monkeypatch):
    source = tmp_path / "router.yaml"
    source.write_text("model_list: []\ngeneral_settings:\n  master_key: source-secret\n")
    monkeypatch.setattr(appmod, "LITELLM_CONFIG", source)
    monkeypatch.setattr(appmod, "_litellm_k8s", {
        "enabled": True, "namespace": "router-ns", "configmap": "router-cm",
        "configmap_key": "router.yaml",
    })
    calls = []

    def install(stdout, code=0, error=None):
        async def run(*args, **kwargs):
            calls.append((args, kwargs))
            if error:
                raise error
            return subprocess.CompletedProcess(args, code, stdout, "stderr-secret")
        monkeypatch.setattr(appmod, "_run", run)

    return source, calls, install


def check():
    return asyncio.run(appmod._check_litellm_config_drift())


def configmap(text):
    return json.dumps({"data": {"router.yaml": text}})


def test_disabled_does_not_read_files_or_run_commands(router, monkeypatch):
    source, calls, install = router
    source.unlink()
    install("")
    monkeypatch.setattr(appmod, "_litellm_k8s", {})
    assert check() == "disabled"
    assert not calls


def test_equivalent_yaml_matches_and_only_reads_configmap(router, caplog):
    source, calls, install = router
    before = source.read_bytes()
    install(configmap("# formatting is irrelevant\ngeneral_settings: {master_key: source-secret}\nmodel_list: []\n"))
    assert check() == "match"
    assert calls == [(("kubectl", "get", "configmap", "router-cm", "-n", "router-ns", "-o", "json"), {"timeout": 5})]
    assert source.read_bytes() == before
    assert "matches ConfigMap" in caplog.text
    assert "source-secret" not in caplog.text


def test_drift_warns_without_writing_or_logging_values(router, caplog):
    source, calls, install = router
    before = source.read_bytes()
    install(configmap("model_list: []\ngeneral_settings: {master_key: deployed-secret}"))
    assert check() == "drift"
    assert len(calls) == 1
    assert source.read_bytes() == before
    assert "drift detected" in caplog.text
    assert "source-secret" not in caplog.text
    assert "deployed-secret" not in caplog.text


@pytest.mark.parametrize("response,code", [
    ("stdout-secret", 1),  # failed kubectl, including timeout
    ("stdout-secret", 0),  # malformed JSON
    ("{}", 0),  # missing data/key
    ('{"data": null}', 0),
    (configmap("master_key: [deployed-secret"), 0),  # malformed YAML
    (configmap(""), 0),
    (configmap("[]"), 0),
    (configmap("{}"), 0),
    (configmap(None), 0),
])
def test_failed_comparison_never_claims_match_or_leaks(router, caplog, response, code):
    _, _, install = router
    install(response, code)
    assert check() == "unavailable"
    assert "unavailable" in caplog.text
    for secret in ("stdout-secret", "stderr-secret", "deployed-secret", "source-secret"):
        assert secret not in caplog.text


@pytest.mark.parametrize("text", [None, "master_key: [source-secret", "", "[]", "{}"])
def test_bad_source_skips_kubernetes(router, caplog, text):
    source, calls, install = router
    install("")
    if text is None:
        source.unlink()
    else:
        source.write_text(text)
    assert check() == "unavailable"
    assert not calls
    assert "source-secret" not in caplog.text


def test_missing_kubectl_is_nonfatal_and_redacted(router, caplog):
    _, _, install = router
    install("", error=FileNotFoundError("exception-secret"))
    assert check() == "unavailable"
    assert "exception-secret" not in caplog.text


def test_startup_runs_check_once_even_when_unavailable(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(appmod, "APP_HOST", "127.0.0.1")
    monkeypatch.setattr(appmod, "_http", appmod._http)

    async def drift():
        calls.append("drift")
        return "unavailable"

    async def noop():
        return None

    monkeypatch.setattr(appmod, "_check_litellm_config_drift", drift)
    monkeypatch.setattr(appmod, "_alert_loop", noop)
    monkeypatch.setattr(appmod, "_pinned_autostart", noop)

    async def startup():
        async with appmod._lifespan(appmod.app):
            assert calls == ["drift"]

    asyncio.run(startup())
    assert calls == ["drift"]
    assert "LiteLLM config drift startup result: unavailable" in capsys.readouterr().out
