"""Shared fixtures — import the app module and isolate mutable alert state."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app as appmod


@pytest.fixture(autouse=True)
def isolate_alert_state(tmp_path, monkeypatch):
    """Keep alert cooldown state out of the repo and reset it per test."""
    monkeypatch.setattr(appmod, "_ALERT_STATE_FILE", tmp_path / "alert_state.json")
    monkeypatch.setattr(appmod, "_last_alert_sent", {})
    yield
