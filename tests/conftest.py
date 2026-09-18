"""Shared fixtures — import the app module and isolate mutable alert state."""

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

import app as appmod


# Committed ground-truth snapshot of every model config on this box as of 2026-08-03.
# Deliberately a *file*, not a live glob of ~/.cache/huggingface or /mnt/models: a live
# read would make the suite depend on which models happen to be present, so adding or
# deleting one model would break unrelated tests. Regenerating this snapshot is a
# deliberate act (02-VALIDATION.md, Wave 0 fixture rule).
# Lives under tests/ rather than .planning/ because it is a test input: PR branches strip
# transient .planning/ paths, and a fixture parked there makes the suite uncollectable there.
MODEL_FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "model-fixtures.json"


@pytest.fixture(autouse=True)
def isolate_alert_state(tmp_path, monkeypatch):
    """Keep alert cooldown state out of the repo and reset it per test."""
    monkeypatch.setattr(appmod, "_ALERT_STATE_FILE", tmp_path / "alert_state.json")
    monkeypatch.setattr(appmod, "_last_alert_sent", {})
    yield


@pytest.fixture(autouse=True)
def isolate_agent_history(tmp_path, monkeypatch):
    """Keep the agent run-history store out of the repo; reset state + locks per test.

    Locks are re-created so an `asyncio.Lock` never outlives the event loop it was
    bound to (each asyncio test runs on a fresh loop).
    """
    monkeypatch.setattr(appmod, "_AGENT_RUNS_FILE", tmp_path / "agent_runs.json")
    monkeypatch.setattr(appmod, "_agent_history", [])
    appmod._agent_tasks.clear()
    for agent in appmod._AGENTS.values():
        agent["lock"] = asyncio.Lock()
        agent["state"] = appmod._new_agent_state()
    yield


@pytest.fixture(scope="session")
def model_fixtures():
    """The whole parsed model-fixtures.json snapshot: {generated, pool_gb, models, anomalies}."""
    with MODEL_FIXTURES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def fixture_models(model_fixtures):
    """Just the 17 model rows from the committed snapshot."""
    return model_fixtures["models"]
